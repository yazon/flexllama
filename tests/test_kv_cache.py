"""
KV cache persistence verification script for FlexLLama.

This script validates the KV cache persistence feature offline (no live
llama-server required):

* chat identity computation (stability for append-only chats)
* config validation of the per-runner ``kv_cache`` block
* snapshot store file operations (naming, LRU, free space, state file)
* request-path and lifecycle decision matrix (with a faked llama-server
  slot API)
* ``--slot-save-path`` command-line injection in RunnerProcess
* API wiring (helpers, KV endpoints, route registration)

Usage:
    python tests/test_kv_cache.py

Exit code is 0 when all tests pass, 1 otherwise.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace
import uuid

# Allow running from the repository root or from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import ConfigManager  # noqa: E402
from backend.kv_cache import (  # noqa: E402
    KVSnapshotStore,
    compute_chat_identity,
    sanitize_name,
)
from backend.runner import RunnerProcess  # noqa: E402


def setup_test_logging(debug=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    session_id = f"test_kv_cache_{timestamp}_{session_uuid}"

    test_log_dir = os.path.join("tests", "logs", session_id)
    os.makedirs(test_log_dir, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        os.path.join(test_log_dir, "test_kv_cache.log"), mode="w", encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        )
    )
    root_logger.addHandler(file_handler)

    return session_id, test_log_dir


logger = logging.getLogger(__name__)

PASSED = 0
FAILED = 0


def record_pass(name):
    global PASSED
    PASSED += 1
    logger.info(f"  PASS: {name}")


def record_fail(name, detail=""):
    global FAILED
    FAILED += 1
    logger.error(f"  FAIL: {name}" + (f" - {detail}" if detail else ""))


def check(name, condition, detail=""):
    if condition:
        record_pass(name)
    else:
        record_fail(name, detail)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class FakeLlamaServer:
    """Stands in for the llama-server slot save/restore API.

    Records every call; returns canned 200 responses by default. Set
    ``status`` to force every call to fail.
    """

    def __init__(self):
        self.calls = []
        self.status = 200
        self.slot_list = [{"id": 0, "is_processing": False, "state": "idle"}]

    def attach(self, store):
        store._request = self.request_for(store)

    def request_for(self, store):
        async def request(method, path, payload=None):
            return await self.request(store, method, path, payload)

        return request

    async def request(self, store, method, path, payload=None):
        self.calls.append({"method": method, "path": path, "payload": payload})
        if self.status != 200:
            return self.status, {"error": "forced failure"}
        if path == "/slots":
            return 200, self.slot_list
        if path.startswith("/slots/") and "action=save" in path:
            fn = (payload or {}).get("filename")
            return (
                200,
                {
                    "id_slot": 0,
                    "filename": fn,
                    "n_saved": 10,
                    "n_written": 10,
                    "timings": {"save_ms": 1.0},
                },
            )
        if path.startswith("/slots/") and "action=restore" in path:
            fn = (payload or {}).get("filename")
            return (
                200,
                {
                    "id_slot": 0,
                    "filename": fn,
                    "n_restored": 10,
                    "n_read": 10,
                    "timings": {"restore_ms": 1.0},
                },
            )
        if path.startswith("/slots/") and "action=erase" in path:
            return 200, {"id_slot": 0, "n_erased": 5}
        return 404, {}

    def save_calls(self):
        return [c for c in self.calls if "action=save" in c["path"]]

    def restore_calls(self):
        return [c for c in self.calls if "action=restore" in c["path"]]


def make_messages(words):
    """Build a chat with one user message made of the given words."""
    return [{"role": "user", "content": " ".join(words)}]


def make_long_chat(n_words):
    """Build a chat whose serialized text has at least n_words words."""
    return [{"role": "user", "content": " ".join(f"w{i}" for i in range(n_words))}]


def make_store(tmp_dir, **kwargs):
    """Create a KVSnapshotStore in a temp dir (defaults: runner1)."""
    params = dict(
        runner_name="runner1",
        host="127.0.0.1",
        port=8095,
        snapshot_dir=os.path.join(tmp_dir, "snapshots"),
    )
    params.update(kwargs)
    return KVSnapshotStore(**params)


def write_snapshot(store, model, identity, mtime=None):
    """Create a real snapshot file on disk (the fake server does not write)."""
    path = store.snapshot_dir / store.snapshot_filename(model, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def make_config(tmp_dir, kv_cache=None, runner_name="runner1", port=8095):
    """Write a minimal valid FlexLLama config and return its path."""
    runner = {"type": "server", "path": "llama-server", "port": port}
    if kv_cache is not None:
        runner["kv_cache"] = kv_cache
    config = {
        "models": [
            {
                "model": "models\\model1.gguf",
                "runner": runner_name,
                "model_alias": "model1",
            }
        ],
        "api": {"host": "127.0.0.1", "port": 8080},
        runner_name: runner,
    }
    path = os.path.join(tmp_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return path


# ---------------------------------------------------------------------------
# Identity / sanitize tests
# ---------------------------------------------------------------------------


def test_identity():
    logger.info("Test group: identity and sanitize")

    a = make_messages(["hello", "world"])
    b = a + [
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "second turn"},
    ]

    id_a = compute_chat_identity(a)
    id_b = compute_chat_identity(b)

    check(
        "identity_is_16_hex",
        len(id_a) == 16 and all(c in "0123456789abcdef" for c in id_a),
        id_a,
    )
    check("identity_stable_for_same_input", compute_chat_identity(a) == id_a)
    check("identity_changes_when_chat_grows_short", id_a != id_b)

    # Append-only stability once the chat exceeds max_words: the identity
    # must freeze because the first 100 words never change.
    long_chat = make_long_chat(150)
    long_chat2 = long_chat + [{"role": "assistant", "content": "extra reply"}]
    check(
        "identity_freezes_past_max_words",
        compute_chat_identity(long_chat) == compute_chat_identity(long_chat2),
    )

    # Edge cases
    check(
        "identity_none_and_empty_constant",
        compute_chat_identity(None)
        == compute_chat_identity([])
        == compute_chat_identity("not a list"),
    )
    check("identity_none_not_16_hex_or_is", len(compute_chat_identity(None)) == 16)

    # Multimodal content is handled
    mm = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        }
    ]
    check("identity_multimodal_ok", len(compute_chat_identity(mm)) == 16)
    check(
        "identity_multimodal_differs_from_text",
        compute_chat_identity(mm)
        != compute_chat_identity([{"role": "user", "content": "describe"}]),
    )

    # sanitize
    check("sanitize_basic", sanitize_name("model one") == "model_one")
    check("sanitize_specials", sanitize_name("a/b\\c:d") == "a_b_c_d")
    check("sanitize_empty", sanitize_name("...") == "unnamed")
    check(
        "sanitize_keeps_dots_dashes", sanitize_name("qwen3.8_27b-x") == "qwen3.8_27b-x"
    )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


def test_config_validation(tmp_dir):
    logger.info("Test group: config validation")

    # Absent block -> disabled getter
    cm = ConfigManager(make_config(tmp_dir, kv_cache=None))
    kv = cm.get_kv_cache_config("runner1")
    check("config_absent_disabled", kv == {"enabled": False}, str(kv))

    # Explicitly disabled with junk in it -> no validation errors
    cm = ConfigManager(
        make_config(
            tmp_dir,
            kv_cache={
                "enabled": False,
                "dir": "",
                "max_snapshots": -5,
                "refresh_interval_seconds": "x",
                "auto_restore_on_start": "yes",
            },
        )
    )
    check(
        "config_disabled_not_validated",
        cm.get_kv_cache_config("runner1")["enabled"] is False,
    )

    # Enabled with defaults filled in
    cm = ConfigManager(make_config(tmp_dir, kv_cache={"enabled": True}))
    kv = cm.get_kv_cache_config("runner1")
    check(
        "config_default_dir",
        kv["dir"] == os.path.join("kv_snapshots", "runner1"),
        str(kv),
    )
    check("config_default_refresh", kv["refresh_interval_seconds"] == 300)
    check("config_default_max_snapshots", kv["max_snapshots"] == 4)
    check("config_default_auto_restore", kv["auto_restore_on_start"] is False)

    # Enabled with explicit values
    cm = ConfigManager(
        make_config(
            tmp_dir,
            kv_cache={
                "enabled": True,
                "dir": "my_cache",
                "refresh_interval_seconds": 60,
                "max_snapshots": 2,
                "auto_restore_on_start": True,
            },
        )
    )
    kv = cm.get_kv_cache_config("runner1")
    check(
        "config_explicit_values", kv["dir"] == "my_cache" and kv["max_snapshots"] == 2
    )

    # Invalid values -> ValueError
    bad_cases = [
        ("block_not_dict", "not-a-dict"),
        ("dir_empty", {"enabled": True, "dir": "   "}),
        ("dir_not_str", {"enabled": True, "dir": 42}),
        ("refresh_str", {"enabled": True, "refresh_interval_seconds": "x"}),
        ("refresh_bool", {"enabled": True, "refresh_interval_seconds": True}),
        ("refresh_negative", {"enabled": True, "refresh_interval_seconds": -1}),
        ("max_zero", {"enabled": True, "max_snapshots": 0}),
        ("max_bool", {"enabled": True, "max_snapshots": True}),
        ("auto_restore_str", {"enabled": True, "auto_restore_on_start": "yes"}),
        ("enabled_not_bool", {"enabled": "true"}),
    ]
    for name, block in bad_cases:
        try:
            ConfigManager(make_config(tmp_dir, kv_cache=block))
            check(f"config_rejects_{name}", False, "no error raised")
        except ValueError:
            check(f"config_rejects_{name}", True)
        except Exception as e:  # noqa: BLE001
            check(f"config_rejects_{name}", False, f"wrong error type: {e!r}")


# ---------------------------------------------------------------------------
# Store file-operation tests
# ---------------------------------------------------------------------------


def test_store_file_ops(tmp_dir):
    logger.info("Test group: store file operations")

    store = make_store(tmp_dir, max_snapshots=3)
    check("store_dir_created", store.snapshot_dir.is_dir())
    check(
        "store_filename_format",
        store.snapshot_filename("my model", "abcd1234ef567890")
        == "my_model__abcd1234ef567890.llama",
    )

    # has_snapshot
    check("store_has_snapshot_false", not store.has_snapshot("m1", "id1"))
    write_snapshot(store, "m1", "id1")
    check("store_has_snapshot_true", store.has_snapshot("m1", "id1"))
    store._snapshot_path("m1", "id1").unlink()  # keep the dir clean for the list tests

    # list_snapshots ordering (newest first by mtime)
    base = 1_700_000_000.0
    write_snapshot(store, "m1", "old", mtime=base)
    write_snapshot(store, "m1", "mid", mtime=base + 50)
    write_snapshot(store, "m1", "new", mtime=base + 100)
    write_snapshot(store, "m2", "other", mtime=base + 50)
    snaps = store.list_snapshots()
    check("store_list_count", len(snaps) == 4, str(snaps))
    check(
        "store_list_newest_first",
        snaps[0]["filename"] == "m1__new.llama",
        str([s["filename"] for s in snaps]),
    )
    check(
        "store_list_fields",
        snaps[0]["model"] == "m1"
        and snaps[0]["identity"] == "new"
        and snaps[0]["size_bytes"] == 64,
    )

    # newest_snapshot_for_model
    newest = store.newest_snapshot_for_model("m1")
    check("store_newest_for_model", newest and newest["identity"] == "new")
    check(
        "store_newest_for_model_missing",
        store.newest_snapshot_for_model("m9") is None,
    )

    # enforce_lru: 3 snapshots for m1, cap 3 -> nothing removed
    store.enforce_lru("m1")
    check("store_lru_no_evict_at_cap", store.has_snapshot("m1", "old"))

    # add a 4th -> oldest (old) removed, protected file survives
    write_snapshot(store, "m1", "newer", mtime=base + 200)
    store.enforce_lru("m1", protect_filename="m1__new.llama")
    check("store_lru_evicts_oldest", not store.has_snapshot("m1", "old"))
    check("store_lru_keeps_others", store.has_snapshot("m1", "new"))
    check("store_lru_other_model_untouched", store.has_snapshot("m2", "other"))

    # state file round-trip
    store.last_pair = ("m1", "newer")
    store._save_state_file()
    store2 = make_store(tmp_dir, max_snapshots=3)
    check(
        "store_state_file_roundtrip",
        store2.last_pair == ("m1", "newer"),
        str(store2.last_pair),
    )

    # corrupted state file is tolerated
    state_file = store2._state_file
    state_file.write_text("{not json", encoding="utf-8")
    store3 = make_store(tmp_dir, max_snapshots=3)
    check("store_corrupt_state_ignored", store3.last_pair is None)


# ---------------------------------------------------------------------------
# HTTP flow tests (faked llama-server)
# ---------------------------------------------------------------------------


async def test_store_http_flows(tmp_dir):
    logger.info("Test group: store HTTP flows")

    store = make_store(tmp_dir)
    fake = FakeLlamaServer()
    fake.attach(store)

    # save_slot
    ok = await store.save_slot("m1", "idA")
    check("http_save_ok", ok)
    save = fake.save_calls()
    check("http_save_called_once", len(save) == 1, str(save))
    check(
        "http_save_payload",
        save and save[0]["payload"] == {"filename": "m1__idA.llama"},
        str(save),
    )
    check(
        "http_save_path",
        save
        and save[0]["path"].startswith("/slots/")
        and "action=save" in save[0]["path"],
    )

    # failed save (forced 500)
    fake.status = 500
    ok = await store.save_slot("m1", "idA")
    check("http_save_failure", not ok)
    fake.status = 200

    # restore without a local file -> refused without HTTP
    fake.calls.clear()
    ok = await store.restore_slot("m1", "nope")
    check("http_restore_missing_file", not ok)
    check("http_restore_missing_file_no_http", len(fake.calls) == 0, str(fake.calls))

    # restore with a local file
    write_snapshot(store, "m1", "idB")
    ok = await store.restore_slot("m1", "idB")
    check("http_restore_ok", ok)
    check(
        "http_restore_payload",
        fake.restore_calls()
        and fake.restore_calls()[-1]["payload"] == {"filename": "m1__idB.llama"},
    )
    check("http_restore_sets_last_save_ts", store.last_save_ts > 0)

    # free-space guard blocks the save before any HTTP.
    # Required size = 1.1 * max(existing snapshot size, floor); here the only
    # m1 snapshot is 64 bytes, so anything under ~70 free bytes blocks it.
    fake.calls.clear()
    real_disk_usage = shutil.disk_usage

    def tiny_disk_usage(path):
        return SimpleNamespace(total=10 * 1024**3, used=10 * 1024**3 - 10, free=10)

    shutil.disk_usage = tiny_disk_usage
    try:
        ok = await store.save_slot("m1", "idA")
    finally:
        shutil.disk_usage = real_disk_usage
    check("http_free_space_guard", not ok)
    check(
        "http_free_space_guard_no_http",
        len(fake.calls) == 0,
        str(fake.calls),
    )

    # 1 GiB floor: a fresh model with no snapshots needs ~1.1 GiB free
    fresh = make_store(os.path.join(tmp_dir, "fresh"))
    shutil.disk_usage = lambda path: SimpleNamespace(
        total=10 * 1024**3, used=9 * 1024**3, free=100 * 1024**2
    )
    try:
        check("http_free_space_floor_blocks", not fresh.has_enough_free_space("newm"))
        shutil.disk_usage = lambda path: SimpleNamespace(
            total=20 * 1024**3, used=5 * 1024**3, free=15 * 1024**3
        )
        check("http_free_space_floor_allows", fresh.has_enough_free_space("newm"))
    finally:
        shutil.disk_usage = real_disk_usage

    # manual operations
    store.last_pair = ("m1", "idB")
    store.slot_has_state = True
    ok, msg = await store.manual_save()
    check("http_manual_save", ok and msg == "saved", msg)

    ok, msg = await store.manual_restore("m1__idB.llama")
    check("http_manual_restore_named", ok and "idB" in msg, msg)
    check(
        "http_manual_restore_tracks",
        store.last_pair == ("m1", "idB") and store.slot_has_state,
    )

    ok, msg = await store.manual_restore("..\\..\\evil.llama")
    check("http_manual_restore_path_traversal", not ok, msg)
    ok, msg = await store.manual_restore("missing__x.llama")
    check("http_manual_restore_missing", not ok, msg)

    ok, msg = store.manual_erase("m1__idB.llama")
    check("http_manual_erase", ok and not store.has_snapshot("m1", "idB"), msg)
    ok, msg = store.manual_erase("m1__idB.llama")
    check("http_manual_erase_missing", not ok, msg)

    # status shape
    st = store.status()
    check(
        "http_status_fields",
        all(
            k in st
            for k in (
                "runner",
                "enabled",
                "dir",
                "last_model",
                "slot_has_state",
                "snapshots",
            )
        ),
        str(st.keys()),
    )
    json.dumps(st)  # must be JSON-serializable


# ---------------------------------------------------------------------------
# Request-path / lifecycle decision matrix
# ---------------------------------------------------------------------------


async def test_on_request_matrix(tmp_dir):
    logger.info("Test group: on_request decision matrix")

    a = make_messages(["hello", "world"])
    b = a + [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "second turn"},
    ]
    id_a = compute_chat_identity(a)
    id_b = compute_chat_identity(b)

    store = make_store(tmp_dir)
    fake = FakeLlamaServer()
    fake.attach(store)

    # 1. Cold start: no tracking, no snapshots -> no HTTP at all
    await store.on_request("m1", a)
    check("matrix_cold_no_http", len(fake.calls) == 0, str(fake.calls))
    check("matrix_cold_last_pair", store.last_pair == ("m1", id_a))
    check("matrix_cold_no_state", not store.slot_has_state)

    # 2. Same chat processed successfully -> slot holds state
    store.mark_processed("m1", id_a, ok=True)
    check("matrix_marked_state", store.slot_has_state)
    fake.calls.clear()
    await store.on_request("m1", a)
    check("matrix_same_chat_no_http", len(fake.calls) == 0, str(fake.calls))

    # 3. Chat grows (new identity): outgoing saved, nothing to restore
    fake.calls.clear()
    await store.on_request("m1", b)
    saves = fake.save_calls()
    check(
        "matrix_grow_saves_outgoing",
        len(saves) == 1 and saves[0]["payload"]["filename"] == f"m1__{id_a}.llama",
        str(fake.calls),
    )
    check("matrix_grow_no_restore", len(fake.restore_calls()) == 0)
    store.mark_processed("m1", id_b, ok=True)

    # 4. Failed forward -> tracking dropped, no state
    store.mark_processed("m1", id_b, ok=False)
    check("matrix_failed_drops_state", not store.slot_has_state)

    # 5. Snapshot exists, slot empty -> restore on return
    write_snapshot(store, "m1", id_b)
    fake.calls.clear()
    await store.on_request("m1", b)
    check(
        "matrix_return_restores",
        len(fake.restore_calls()) == 1
        and fake.restore_calls()[0]["payload"]["filename"] == f"m1__{id_b}.llama",
        str(fake.calls),
    )
    check("matrix_return_state_after_restore", store.slot_has_state)

    # 6. Model switch while holding state: in-band hook does NOT save the
    # outgoing chat (the runner stop hook does that on a real model switch)
    write_snapshot(store, "m2", id_a)
    store.last_pair = ("m1", id_b)
    store.slot_has_state = True
    fake.calls.clear()
    await store.on_request("m2", a)
    check("matrix_model_switch_no_save", len(fake.save_calls()) == 0, str(fake.calls))
    check("matrix_model_switch_restores", len(fake.restore_calls()) == 1)
    check("matrix_model_switch_last_pair", store.last_pair == ("m2", id_a))

    # 7. save_before_stop: with state -> save; without -> no HTTP
    store.last_pair = ("m2", id_a)
    store.slot_has_state = True
    fake.calls.clear()
    ok = await store.save_before_stop()
    check("matrix_save_before_stop", ok and len(fake.save_calls()) == 1)
    store.slot_has_state = False
    fake.calls.clear()
    ok = await store.save_before_stop()
    check(
        "matrix_save_before_stop_no_state",
        not ok and len(fake.calls) == 0,
        str(fake.calls),
    )

    # 8. mark_runner_stopped drops state
    store.slot_has_state = True
    store.mark_runner_stopped()
    check("matrix_runner_stopped", not store.slot_has_state)

    # 9. refresh_tick: interval gating
    store.last_pair = ("m2", id_a)
    store.slot_has_state = True
    store.last_save_ts = 0.0
    fake.calls.clear()
    await store.refresh_tick()
    check("matrix_refresh_due", len(fake.save_calls()) == 1, str(fake.calls))
    store.last_save_ts = datetime.now().timestamp()
    fake.calls.clear()
    await store.refresh_tick()
    check("matrix_refresh_not_due", len(fake.calls) == 0, str(fake.calls))

    # 10. on_request never raises even when the HTTP layer blows up
    # (a snapshot exists, so a restore is attempted and the fake raises)
    write_snapshot(store, "m1", id_a)
    store.slot_has_state = False

    async def boom(method, path, payload=None):
        raise RuntimeError("boom")

    store._request = boom
    try:
        await store.on_request("m1", a)
        check("matrix_never_raises", True)
    except Exception as e:  # noqa: BLE001
        check("matrix_never_raises", False, repr(e))


async def test_lifecycle_hooks(tmp_dir):
    logger.info("Test group: lifecycle hooks")

    id_a = compute_chat_identity(make_messages(["hello", "world"]))

    # Separate subdirs per subtest: state files persist across store
    # instances inside one snapshot dir.
    dir_a = os.path.join(tmp_dir, "a")
    dir_b = os.path.join(tmp_dir, "b")
    dir_c = os.path.join(tmp_dir, "c")

    # auto_restore disabled -> no-op
    store = make_store(dir_a, auto_restore_on_start=False)
    fake = FakeLlamaServer()
    fake.attach(store)
    write_snapshot(store, "m1", id_a)
    store.last_pair = ("m1", id_a)
    ok = await store.auto_restore_on_start("m1")
    check("lifecycle_autorestore_disabled", not ok and len(fake.calls) == 0)

    # enabled + state file chat matches the started model
    store = make_store(dir_b, auto_restore_on_start=True)
    fake = FakeLlamaServer()
    fake.attach(store)
    write_snapshot(store, "m1", id_a)
    store.last_pair = ("m1", id_a)
    ok = await store.auto_restore_on_start("m1")
    check("lifecycle_autorestate_state_chat", ok)
    check(
        "lifecycle_autorestate_file",
        fake.restore_calls()
        and fake.restore_calls()[0]["payload"]["filename"] == f"m1__{id_a}.llama",
        str(fake.calls),
    )
    check("lifecycle_autorestate_sets_state", store.slot_has_state)

    # enabled, state file chat is another model -> newest of started model
    id_b = compute_chat_identity(make_messages(["another", "chat"]))
    write_snapshot(store, "m1", id_b, mtime=datetime.now().timestamp() + 10)
    store.last_pair = ("m2", "deadbeefdeadbeef")
    fake = FakeLlamaServer()
    fake.attach(store)
    ok = await store.auto_restore_on_start("m1")
    check("lifecycle_autorestate_newest", ok)
    check(
        "lifecycle_autorestate_newest_file",
        fake.restore_calls()
        and fake.restore_calls()[0]["payload"]["filename"] == f"m1__{id_b}.llama",
        str(fake.calls),
    )
    check("lifecycle_autorestate_updates_pair", store.last_pair == ("m1", id_b))

    # enabled, nothing on disk -> no-op
    store = make_store(dir_c, auto_restore_on_start=True)
    fake = FakeLlamaServer()
    fake.attach(store)
    ok = await store.auto_restore_on_start("m1")
    check("lifecycle_autorestate_empty", not ok and len(fake.calls) == 0)


# ---------------------------------------------------------------------------
# Command-line injection tests
# ---------------------------------------------------------------------------


def test_command_building(tmp_dir):
    logger.info("Test group: --slot-save-path command injection")

    model_cfg = {"model": "models\\model1.gguf", "model_alias": "model1"}

    # No store -> no flag
    rp = RunnerProcess(
        "runner1", {"type": "server", "path": "llama-server"}, "127.0.0.1", 8095
    )
    cmd, _ = rp._build_command_and_env(model_cfg)
    check("cmd_no_store_no_flag", "--slot-save-path" not in cmd, str(cmd))

    # With store -> flag present exactly once
    rp = RunnerProcess(
        "runner1", {"type": "server", "path": "llama-server"}, "127.0.0.1", 8095
    )
    store = make_store(tmp_dir)
    rp.kv_store = store
    cmd, _ = rp._build_command_and_env(model_cfg)
    check(
        "cmd_store_flag_once",
        cmd.count("--slot-save-path") == 1
        and cmd[cmd.index("--slot-save-path") + 1] == str(store.snapshot_dir),
        str(cmd),
    )

    # User already provided it via extra_args -> not duplicated
    rp = RunnerProcess(
        "runner1",
        {
            "type": "server",
            "path": "llama-server",
            "extra_args": ["--slot-save-path", "custom_dir"],
        },
        "127.0.0.1",
        8095,
    )
    rp.kv_store = store
    cmd, _ = rp._build_command_and_env(model_cfg)
    check(
        "cmd_user_flag_not_duplicated",
        cmd.count("--slot-save-path") == 1
        and cmd[cmd.index("--slot-save-path") + 1] == "custom_dir",
        str(cmd),
    )


# ---------------------------------------------------------------------------
# API wiring tests
# ---------------------------------------------------------------------------


class StubRunnerManager:
    """Minimal stand-in for RunnerManager in API wiring tests."""

    def __init__(self, runners):
        self.runners = runners
        self.model_runner_map = {}
        for runner_name, runner in runners.items():
            for model in runner.models:
                alias = model.get("model_alias", os.path.basename(model["model"]))
                self.model_runner_map[alias] = runner_name

    def get_runner_names(self):
        return list(self.runners)

    def get_runner_for_model(self, model_alias):
        runner_name = self.model_runner_map.get(model_alias)
        return self.runners.get(runner_name)

    def get_kv_store_for_model(self, model_alias):
        runner = self.get_runner_for_model(model_alias)
        return runner.kv_store if runner is not None else None


def make_api(tmp_dir, kv_cache=None):
    """Build a real APIServer with a stub runner manager (no sockets)."""
    from backend.api import APIServer

    cfg_path = make_config(tmp_dir, kv_cache=kv_cache)
    config_manager = ConfigManager(cfg_path)

    runner = RunnerProcess(
        "runner1",
        {"type": "server", "path": "llama-server"},
        "127.0.0.1",
        8095,
    )
    runner.add_model({"model": "models\\model1.gguf", "model_alias": "model1"})
    stub_rm = StubRunnerManager({"runner1": runner})
    api = APIServer(config_manager, stub_rm)
    return api, runner


async def test_api_wiring(tmp_dir):
    logger.info("Test group: API wiring")

    # KV disabled: helpers are no-ops, endpoints answer 409
    api, runner = make_api(tmp_dir, kv_cache=None)
    msgs = make_messages(["hello", "world"])

    check(
        "api_handle_incoming_disabled",
        await api._kv_handle_incoming("model1", {"messages": msgs}) is None,
    )
    api._kv_mark_processed("model1", "whatever", ok=True)  # must not raise
    check("api_mark_processed_none_identity", True)

    err = api._get_runner_kv_store(
        SimpleNamespace(match_info={"runner_name": "runner1"})
    )[1]
    check(
        "api_kv_disabled_409",
        err is not None and err.status == 409,
        str(getattr(err, "status", None)),
    )

    err = api._get_runner_kv_store(
        SimpleNamespace(match_info={"runner_name": "ghost"})
    )[1]
    check("api_kv_unknown_runner_404", err is not None and err.status == 404)

    # KV enabled: helpers drive the store
    api, runner = make_api(
        tmp_dir, kv_cache={"enabled": True, "dir": os.path.join(tmp_dir, "snap")}
    )
    store = make_store(tmp_dir, runner_name="runner1")
    fake = FakeLlamaServer()
    fake.attach(store)
    runner.kv_store = store

    identity = await api._kv_handle_incoming("model1", {"messages": msgs})
    check(
        "api_handle_incoming_identity",
        isinstance(identity, str) and len(identity) == 16,
    )
    check("api_handle_incoming_tracks", store.last_pair == ("model1", identity))

    api._kv_mark_processed("model1", identity, ok=True)
    check("api_mark_processed_ok", store.slot_has_state)
    api._kv_mark_processed("model1", identity, ok=False)
    check("api_mark_processed_fail", not store.slot_has_state)

    # status endpoint
    resp = await api.handle_runner_kv_status(
        SimpleNamespace(match_info={"runner_name": "runner1"})
    )
    body = json.loads(resp.body)
    check("api_kv_status_200", resp.status == 200 and body.get("success") is True)
    check("api_kv_status_runner", body["kv_cache"]["runner"] == "runner1")

    # erase endpoint (real file)
    write_snapshot(store, "model1", "abc123abc123abc1")
    fname = store.snapshot_filename("model1", "abc123abc123abc1")

    async def fake_json(payload):
        return payload

    req = SimpleNamespace(
        match_info={"runner_name": "runner1"},
        json=lambda: fake_json({"filename": fname}),
    )
    resp = await api.handle_runner_kv_erase(req)
    body = json.loads(resp.body)
    check("api_kv_erase", resp.status == 200 and body.get("success") is True)
    check("api_kv_erase_removed", not store.has_snapshot("model1", "abc123abc123abc1"))

    # erase without filename -> 400
    req = SimpleNamespace(
        match_info={"runner_name": "runner1"},
        json=lambda: fake_json({}),
    )
    resp = await api.handle_runner_kv_erase(req)
    check("api_kv_erase_no_filename_400", resp.status == 400)

    # route registration
    paths = {route.resource.canonical for route in api.app.router.routes()}
    for expected in (
        "/v1/runners/{runner_name}/kv",
        "/v1/runners/{runner_name}/kv/save",
        "/v1/runners/{runner_name}/kv/restore",
        "/v1/runners/{runner_name}/kv/erase",
    ):
        check(f"api_route_{expected}", expected in paths, str(paths))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_tests():
    logger.info("KV cache persistence offline verification")
    logger.info("=" * 60)

    test_identity()

    # Temporary dirs live inside the workspace so the test also runs under
    # file-sandboxed environments where the system temp dir is read-only.
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
    os.makedirs(base_dir, exist_ok=True)

    for group in (
        "config",
        "store",
        "http",
        "matrix",
        "lifecycle",
        "cmd",
        "api",
    ):
        # Plain makedirs (not tempfile.mkdtemp): some file sandboxes do not
        # trust directories created via the tempfile module.
        tmp_dir = os.path.join(base_dir, f"kvtest_{group}_{uuid.uuid4().hex[:8]}")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            if group == "config":
                test_config_validation(tmp_dir)
            elif group == "store":
                test_store_file_ops(tmp_dir)
            elif group == "http":
                await test_store_http_flows(tmp_dir)
            elif group == "matrix":
                await test_on_request_matrix(tmp_dir)
            elif group == "lifecycle":
                await test_lifecycle_hooks(tmp_dir)
            elif group == "cmd":
                test_command_building(tmp_dir)
            elif group == "api":
                await test_api_wiring(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    shutil.rmtree(base_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED > 0:
        logger.error("Some tests FAILED")
        sys.exit(1)
    logger.info("All tests PASSED")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KV cache persistence verification")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_test_logging(args.debug)
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
