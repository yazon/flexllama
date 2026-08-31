"""
KV cache persistence for FlexLLama runners.

This module adds disk-backed persistence of llama.cpp slot state (KV cache +
prompt tokens) on top of the vanilla llama-server slot save/restore API
(``POST /slots/{id}?action=save|restore`` with ``--slot-save-path`` enabled).

Design goals (see HANDOFF / feature spec):

* **Stable per-chat filenames.** Each chat is identified by a hash of the
  first words of its message list (the chat is append-only, so the identity
  is stable as the conversation grows). Snapshots are written in place to
  ``{model}__{identity}.llama`` — one file per chat, overwritten on refresh,
  never one fresh file per request step (the "disk bomb" anti-pattern).
* **Lifecycle-aware saves.** The outgoing chat is persisted right before its
  slot state can be overwritten (chat switch), before the runner is stopped
  (manual stop, model switch, auto-unload, FlexLLama shutdown), and on a
  periodic refresh while the runner is idle.
* **Bounded disk usage.** An LRU cap (``max_snapshots``) per model removes
  the oldest snapshots, and a free-space check skips saves that would not fit.
* **Fail-soft.** Every persistence operation logs and continues on error;
  persistence problems never break request forwarding.

v1 scope: one slot per runner (``-np 1``), the chat-completions endpoint.
Multi-slot runners (``-np > 1``) need an identity→slot mapping and are
intentionally out of scope.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import aiohttp

# Get logger for this module
logger = logging.getLogger(__name__)

# Default values for the kv_cache runner configuration block.
DEFAULT_MAX_SNAPSHOTS = 4
DEFAULT_REFRESH_INTERVAL_SECONDS = 300
DEFAULT_AUTO_RESTORE_ON_START = False

# Name of the persisted last-chat state file (per runner, inside snapshot dir).
_STATE_FILE_TEMPLATE = "flexllama__{runner}.json"

# Snapshot files are named {model}__{identity}.llama; identity is 16 hex chars
# and never contains "__", so the model part is everything before the LAST "__".
_SNAPSHOT_SUFFIX = ".llama"


def sanitize_name(name) -> str:
    """Make an arbitrary string safe to use as a filename component.

    Args:
        name: The raw name (model alias, runner name, ...).

    Returns:
        A sanitized, non-empty string containing only ``[A-Za-z0-9._-]``.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return cleaned or "unnamed"


def _content_to_text(content) -> str:
    """Flatten a chat message ``content`` value to a string for hashing.

    Handles plain strings and the OpenAI multimodal list form; non-text
    parts are represented by their type so that chats with/without media
    produce different identities.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    media_type = item.get("type", "")
                    if media_type:
                        parts.append(str(media_type))
        return " ".join(parts)
    return str(content)


def compute_chat_identity(messages, max_words: int = 100) -> str:
    """Compute a stable identity for a chat conversation.

    The identity is the first 16 hex chars of SHA-256 over the first
    ``max_words`` words of the serialized ``role::content`` lines. Chat
    histories are append-only, so the identity is stable as the conversation
    grows and changes only when the leading messages change (a new chat).

    Args:
        messages: The OpenAI chat-completions ``messages`` list.
        max_words: Number of leading words to include in the hash.

    Returns:
        A 16-character hex string.
    """
    if not isinstance(messages, (list, tuple)):
        messages = []

    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        parts.append(f"{role}::{_content_to_text(msg.get('content'))}")

    text = "\n".join(parts)
    words = text.split()[:max_words]
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()[:16]


class KVSnapshotStore:
    """Per-runner KV snapshot state manager.

    Tracks which chat is currently resident in the runner's slot, persists
    it to disk at the right lifecycle moments, and restores snapshots for
    returning chats.

    The store is created by :class:`RunnerManager` for each runner with
    ``kv_cache.enabled: true`` and is attached to the
    :class:`RunnerProcess` as ``kv_store``.
    """

    def __init__(
        self,
        runner_name: str,
        host: str,
        port: int,
        snapshot_dir,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
        auto_restore_on_start: bool = DEFAULT_AUTO_RESTORE_ON_START,
    ):
        """Initialize the snapshot store.

        Args:
            runner_name: The FlexLLama runner name.
            host: Host of the llama-server process.
            port: Port of the llama-server process.
            snapshot_dir: Directory for snapshot files (created if missing).
            max_snapshots: LRU cap of snapshot files kept per model.
            refresh_interval_seconds: Periodic idle-save interval (0 disables).
            auto_restore_on_start: Restore the latest snapshot after start.

        Raises:
            ValueError: If the snapshot directory cannot be created or is
                not writable (the caller disables persistence for the runner).
        """
        self.runner_name = runner_name
        self.base_url = f"http://{host}:{port}"
        self.snapshot_dir = Path(snapshot_dir)
        self.max_snapshots = int(max_snapshots)
        self.refresh_interval_seconds = int(refresh_interval_seconds)
        # Stored under a different name than the method of the same name:
        # an instance attribute would shadow it.
        self.auto_restore = bool(auto_restore_on_start)

        # Chat currently tracked for this runner: (model_alias, identity).
        self.last_pair = None
        # True while the runner's slot is known to hold last_pair's state.
        self.slot_has_state = False
        self.last_save_ts = 0.0

        self._lock = asyncio.Lock()
        self._state_file = self.snapshot_dir / _STATE_FILE_TEMPLATE.format(
            runner=sanitize_name(runner_name)
        )

        self._prepare_dir()
        self._load_state_file()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _prepare_dir(self):
        """Create the snapshot directory and verify it is writable."""
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            probe = self.snapshot_dir / ".flexllama_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            raise ValueError(f"KV snapshot dir not writable: {e}") from e

    def _load_state_file(self):
        """Load the last-tracked chat from disk (survives FlexLLama restarts)."""
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            model = state.get("model")
            identity = state.get("identity")
            if isinstance(model, str) and isinstance(identity, str):
                self.last_pair = (model, identity)
                logger.info(
                    f"KV cache: restored last-chat state for runner "
                    f"{self.runner_name} (model={model}, identity={identity})"
                )
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"KV cache: failed to load state file: {e}")

    def _save_state_file(self):
        """Persist the last-tracked chat atomically (tmp file + rename)."""
        if self.last_pair is None:
            return
        try:
            payload = {
                "model": self.last_pair[0],
                "identity": self.last_pair[1],
                "ts": time.time(),
            }
            tmp = self._state_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self._state_file)
        except OSError as e:
            logger.warning(f"KV cache: failed to save state file: {e}")

    def mark_runner_stopped(self):
        """Call after the runner process was stopped: the slot is empty now."""
        self.slot_has_state = False

    # ------------------------------------------------------------------
    # Snapshot files
    # ------------------------------------------------------------------

    @staticmethod
    def _model_key(model_alias: str) -> str:
        """Sanitized model name used in filenames and comparisons."""
        return sanitize_name(model_alias)

    def snapshot_filename(self, model_alias: str, identity: str) -> str:
        """Stable per-chat snapshot filename (overwritten in place)."""
        return f"{self._model_key(model_alias)}__{identity}{_SNAPSHOT_SUFFIX}"

    def _snapshot_path(self, model_alias: str, identity: str) -> Path:
        return self.snapshot_dir / self.snapshot_filename(model_alias, identity)

    def has_snapshot(self, model_alias: str, identity: str) -> bool:
        """Whether a snapshot file exists for this chat."""
        return self._snapshot_path(model_alias, identity).is_file()

    def list_snapshots(self) -> list:
        """List snapshot files, newest first.

        Returns:
            A list of dicts: filename, model (sanitized), identity,
            size_bytes, modified_ts.
        """
        entries = []
        try:
            files = list(self.snapshot_dir.glob(f"*{_SNAPSHOT_SUFFIX}"))
        except OSError:
            return entries
        for path in files:
            stem = path.name[: -len(_SNAPSHOT_SUFFIX)]
            if "__" not in stem:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            model, identity = stem.rsplit("__", 1)
            entries.append(
                {
                    "filename": path.name,
                    "model": model,
                    "identity": identity,
                    "size_bytes": st.st_size,
                    "modified_ts": st.st_mtime,
                }
            )
        entries.sort(key=lambda e: e["modified_ts"], reverse=True)
        return entries

    def newest_snapshot_for_model(self, model_alias: str):
        """Newest snapshot entry for a model (by mtime), or None."""
        model = self._model_key(model_alias)
        newest = None
        for entry in self.list_snapshots():
            if entry["model"] == model:
                if newest is None or entry["modified_ts"] > newest["modified_ts"]:
                    newest = entry
        return newest

    def enforce_lru(self, model_alias: str, protect_filename: str = None):
        """Remove oldest snapshots for this model beyond the LRU cap.

        Args:
            model_alias: Model to enforce the cap for.
            protect_filename: Snapshot just written; never removed.
        """
        model = self._model_key(model_alias)
        entries = [e for e in self.list_snapshots() if e["model"] == model]
        if len(entries) <= self.max_snapshots:
            return
        entries.sort(key=lambda e: e["modified_ts"])  # oldest first
        to_remove = len(entries) - self.max_snapshots
        for entry in entries[:to_remove]:
            if entry["filename"] == protect_filename:
                continue
            try:
                (self.snapshot_dir / entry["filename"]).unlink()
                logger.info(
                    f"KV cache: LRU eviction removed {entry['filename']} "
                    f"({entry['size_bytes']} bytes)"
                )
            except OSError as e:
                logger.warning(f"KV cache: failed to remove {entry['filename']}: {e}")

    def has_enough_free_space(self, model_alias: str) -> bool:
        """Cheap guard: skip the save if the volume is nearly full.

        The required size is estimated from the largest existing snapshot of
        this model (1 GiB floor). When free space is below that estimate the
        save is skipped so a half-written file cannot clobber a good snapshot.
        """
        try:
            usage = shutil.disk_usage(self.snapshot_dir)
        except OSError:
            return True  # Cannot determine — allow the save.
        model = self._model_key(model_alias)
        sizes = [e["size_bytes"] for e in self.list_snapshots() if e["model"] == model]
        reference = max(sizes) if sizes else 1 * 1024 * 1024 * 1024
        return usage.free >= int(reference * 1.1)

    # ------------------------------------------------------------------
    # llama-server HTTP API
    # ------------------------------------------------------------------

    async def _request(self, method: str, path: str, payload=None):
        """Perform an HTTP request against the llama-server.

        Returns:
            Tuple (status_code, json_data). Errors are logged and returned
            as (0, {"error": ...}) — callers treat non-200 as failure.
        """
        url = f"{self.base_url}{path}"
        timeout = aiohttp.ClientTimeout(total=600, sock_connect=10, sock_read=120)
        try:
            async with aiohttp.ClientSession() as session:
                kwargs = {"timeout": timeout}
                if payload is not None:
                    kwargs["json"] = payload
                async with session.request(method, url, **kwargs) as resp:
                    try:
                        data = await resp.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        data = {"raw": await resp.text()}
                    return resp.status, data
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.warning(f"KV cache: HTTP {method} {path} failed: {e}")
            return 0, {"error": {"message": str(e)}}

    async def _pick_slot_id(self) -> int:
        """Pick the slot to save/restore.

        v1 targets ``-np 1`` runners (a single slot). The first idle slot is
        preferred; when the server defers busy slots the request simply
        waits, so any valid slot id works. Falls back to 0.
        """
        status, data = await self._request("GET", "/slots")
        if status != 200:
            return 0
        if isinstance(data, dict):
            slots = data.get("slots", data)
        else:
            slots = data
        if not isinstance(slots, list):
            return 0
        fallback = None
        for slot in slots:
            if not isinstance(slot, dict) or slot.get("id") is None:
                continue
            if fallback is None:
                fallback = slot["id"]
            if "is_processing" in slot:
                idle = not slot.get("is_processing")
            else:
                idle = slot.get("state") == "idle"
            if idle:
                return slot["id"]
        return fallback if fallback is not None else 0

    async def _save_file(self, model_alias: str, filename: str) -> bool:
        """Call llama-server action=save for a given filename."""
        if not self.has_enough_free_space(model_alias):
            logger.warning(
                f"KV cache: insufficient free space in {self.snapshot_dir}, "
                f"skipping save {filename}"
            )
            return False
        slot_id = await self._pick_slot_id()
        start = time.monotonic()
        status, data = await self._request(
            "POST", f"/slots/{slot_id}?action=save", {"filename": filename}
        )
        if status != 200:
            logger.warning(f"KV cache: save {filename} failed ({status}): {data}")
            return False
        self.last_save_ts = time.time()
        size_mb = data.get("n_written", 0) / 1e6
        logger.info(
            f"KV cache: saved {filename} "
            f"({data.get('n_saved', '?')} tokens, {size_mb:.0f} MB) "
            f"in {time.monotonic() - start:.1f}s"
        )
        return True

    async def save_slot(self, model_alias: str, identity: str) -> bool:
        """Persist the current slot state under the chat's stable filename."""
        if identity is None:
            return False
        filename = self.snapshot_filename(model_alias, identity)
        if await self._save_file(model_alias, filename):
            self.enforce_lru(model_alias, protect_filename=filename)
            return True
        return False

    async def _restore_file(self, filename: str) -> bool:
        """Call llama-server action=restore for a given filename."""
        if not (self.snapshot_dir / filename).is_file():
            logger.warning(f"KV cache: snapshot {filename} not found on disk")
            return False
        slot_id = await self._pick_slot_id()
        start = time.monotonic()
        status, data = await self._request(
            "POST", f"/slots/{slot_id}?action=restore", {"filename": filename}
        )
        if status != 200:
            logger.warning(f"KV cache: restore {filename} failed ({status}): {data}")
            return False
        # The snapshot on disk is now the freshest copy of this state;
        # defer the periodic refresh by a full interval.
        self.last_save_ts = time.time()
        logger.info(
            f"KV cache: restored {filename} "
            f"({data.get('n_restored', '?')} tokens) "
            f"in {time.monotonic() - start:.1f}s"
        )
        return True

    async def restore_slot(self, model_alias: str, identity: str) -> bool:
        """Restore the chat's snapshot into the runner's slot."""
        if identity is None:
            return False
        return await self._restore_file(self.snapshot_filename(model_alias, identity))

    # ------------------------------------------------------------------
    # Request-path hook (called from the API layer, in-band)
    # ------------------------------------------------------------------

    async def on_request(self, model_alias: str, messages) -> None:
        """Handle an incoming chat request: persist the outgoing chat,
        warm the incoming one.

        Must be awaited before the request is forwarded to the runner.
        Never raises — persistence problems never break forwarding.
        """
        try:
            identity = compute_chat_identity(messages)
            async with self._lock:
                last = self.last_pair
                same_model = last is not None and self._model_key(last[0]) == (
                    self._model_key(model_alias)
                )
                # The slot holds the outgoing chat's state only while the
                # runner kept running since that chat was processed.
                need_save = self.slot_has_state and same_model and last[1] != identity
                slot_holds_incoming = (
                    self.slot_has_state and same_model and last[1] == identity
                )
                need_restore = not slot_holds_incoming and self.has_snapshot(
                    model_alias, identity
                )
                self.last_pair = (model_alias, identity)
                self._save_state_file()

            if need_save:
                await self.save_slot(last[0], last[1])
            if need_restore:
                if await self.restore_slot(model_alias, identity):
                    self.slot_has_state = True
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.warning(f"KV cache: on_request hook failed for {model_alias}: {e}")

    def mark_processed(self, model_alias: str, identity: str, ok: bool) -> None:
        """Record the outcome of a forwarded request.

        On success the slot is known to hold this chat's state; on failure
        it may hold something else (or be empty), so tracking is dropped.
        """
        try:
            if ok:
                self.last_pair = (model_alias, identity)
                self.slot_has_state = True
                self._save_state_file()
            else:
                self.slot_has_state = False
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.warning(f"KV cache: mark_processed failed: {e}")

    # ------------------------------------------------------------------
    # Lifecycle hooks (called from RunnerProcess / RunnerManager)
    # ------------------------------------------------------------------

    async def save_before_stop(self) -> bool:
        """Persist the tracked chat before the runner process is killed."""
        if not self.slot_has_state or self.last_pair is None:
            logger.debug(
                f"KV cache: no tracked state to save before stop "
                f"(runner {self.runner_name})"
            )
            return False
        return await self.save_slot(*self.last_pair)

    async def auto_restore_on_start(self, model_alias: str) -> bool:
        """After (re)start: restore the latest snapshot for this model.

        Prefers the chat recorded in the state file (when it belongs to this
        model), otherwise falls back to the newest snapshot of the model.
        """
        if not self.auto_restore:
            return False
        try:
            restored = False
            if (
                self.last_pair is not None
                and self._model_key(self.last_pair[0]) == self._model_key(model_alias)
                and self.has_snapshot(*self.last_pair)
            ):
                restored = await self.restore_slot(*self.last_pair)
            else:
                newest = self.newest_snapshot_for_model(model_alias)
                if newest is not None:
                    restored = await self._restore_file(newest["filename"])
                    if restored:
                        self.last_pair = (newest["model"], newest["identity"])
                        self._save_state_file()
            if restored:
                self.slot_has_state = True
            return restored
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.warning(f"KV cache: auto-restore failed: {e}")
            return False

    async def refresh_tick(self) -> None:
        """Periodic idle save (called once per second by the refresh loop)."""
        if self.refresh_interval_seconds <= 0:
            return
        if not self.slot_has_state or self.last_pair is None:
            return
        if time.time() - self.last_save_ts < self.refresh_interval_seconds:
            return
        await self.save_slot(*self.last_pair)

    # ------------------------------------------------------------------
    # Manual operations (used by the /v1/runners/{name}/kv endpoints)
    # ------------------------------------------------------------------

    async def manual_save(self):
        """Force-save the currently tracked chat. Returns (ok, message)."""
        if self.last_pair is None or not self.slot_has_state:
            return False, "no tracked chat state to save"
        ok = await self.save_slot(*self.last_pair)
        return ok, "saved" if ok else "save failed (see server log)"

    async def manual_restore(self, filename: str = None):
        """Restore a snapshot by filename, or the latest known one.

        Returns (ok, message).
        """
        try:
            if filename:
                # Constrain to the snapshot directory (basename only).
                safe = os.path.basename(str(filename))
                if not safe or safe != str(filename):
                    return False, "invalid filename"
                ok = await self._restore_file(safe)
                if ok:
                    self._track_restored_file(safe)
                    return True, f"restored {safe}"
                return False, f"restore of {safe} failed (see server log)"

            if self.last_pair is not None and self.has_snapshot(*self.last_pair):
                ok = await self.restore_slot(*self.last_pair)
                target = self.snapshot_filename(*self.last_pair)
            else:
                newest = (
                    self.newest_snapshot_for_model(self.last_pair[0])
                    if self.last_pair is not None
                    else self.list_snapshots()[0]
                    if self.list_snapshots()
                    else None
                )
                if newest is None:
                    return False, "no snapshots available"
                ok = await self._restore_file(newest["filename"])
                target = newest["filename"]
            if ok:
                self._track_restored_file(target)
                return True, f"restored {target}"
            return False, f"restore of {target} failed (see server log)"
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.warning(f"KV cache: manual restore failed: {e}")
            return False, str(e)

    def _track_restored_file(self, filename: str):
        """Update tracking after a successful restore.

        The slot is known to hold the restored state. When the filename
        follows the {model}__{identity} convention the tracked chat is
        updated as well.
        """
        self.slot_has_state = True
        stem = (
            filename[: -len(_SNAPSHOT_SUFFIX)]
            if filename.endswith(_SNAPSHOT_SUFFIX)
            else os.path.basename(filename)
        )
        if "__" in stem:
            model, identity = stem.rsplit("__", 1)
            self.last_pair = (model, identity)
            self._save_state_file()

    def manual_erase(self, filename: str):
        """Delete a snapshot file (does not touch the slot). (ok, message)."""
        try:
            safe = os.path.basename(str(filename))
            if not safe or safe != str(filename):
                return False, "invalid filename"
            path = self.snapshot_dir / safe
            if not path.is_file():
                return False, "snapshot not found"
            path.unlink()
            return True, f"erased {safe}"
        except Exception as e:  # noqa: BLE001 - fail-soft by design
            logger.warning(f"KV cache: manual erase failed: {e}")
            return False, str(e)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """JSON-serializable status for the dashboard / status endpoint."""
        return {
            "runner": self.runner_name,
            "enabled": True,
            "dir": str(self.snapshot_dir),
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "max_snapshots": self.max_snapshots,
            "auto_restore_on_start": self.auto_restore,
            "last_model": self.last_pair[0] if self.last_pair else None,
            "last_identity": self.last_pair[1] if self.last_pair else None,
            "slot_has_state": self.slot_has_state,
            "last_save_ts": self.last_save_ts,
            "snapshots": self.list_snapshots(),
        }
