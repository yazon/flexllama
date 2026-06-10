"""
GPU metrics collector for FlexLLama.

Collects per-GPU telemetry via nvidia-smi and amd-smi (when available),
normalizes the payload, and keeps bounded history for dashboard sparklines.
The collector degrades gracefully on unsupported platforms or hosts without
vendor tools installed.
"""

import asyncio
import csv
import json
import logging
import platform
import re
import shlex
import shutil
import subprocess
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REASON_UNSUPPORTED_PLATFORM = "unsupported_platform"
REASON_TOOL_NOT_FOUND = "tool_not_found"
REASON_COMMAND_FAILED = "command_failed"
REASON_COMMAND_TIMEOUT = "command_timeout"
REASON_PARSE_ERROR = "parse_error"
REASON_NO_VISIBLE_GPUS = "no_visible_gpus"

STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DISABLED = "disabled"

DEFAULT_POLL_INTERVAL = 2
DEFAULT_HISTORY_POINTS = 60
DEFAULT_COMMAND_TIMEOUT = 3
DEFAULT_RATE_LIMIT_RPM = 120
DEFAULT_VENDORS = ["nvidia", "amd"]

HISTORY_FIELDS = (
    "memory_used_mb",
    "utilization_gpu_percent",
    "temperature_c",
)


class GPUMetricsCollector:
    """Background GPU telemetry collector with vendor adapters."""

    def __init__(self, config: Dict[str, Any]):
        self._enabled: bool = config.get("enabled", True)
        self._poll_interval: int = config.get(
            "poll_interval_seconds", DEFAULT_POLL_INTERVAL
        )
        self._history_points: int = config.get("history_points", DEFAULT_HISTORY_POINTS)
        self._command_timeout: int = config.get(
            "command_timeout_seconds", DEFAULT_COMMAND_TIMEOUT
        )
        self._rate_limit_rpm: int = config.get(
            "rate_limit_requests_per_minute", DEFAULT_RATE_LIMIT_RPM
        )
        self._requested_vendors: List[str] = config.get("vendors", DEFAULT_VENDORS)

        self._status: str = STATUS_UNAVAILABLE
        self._reason: Optional[str] = None
        self._platform_info: Dict[str, Any] = {}
        self._gpus: List[Dict[str, Any]] = []
        self._history: Dict[str, Dict[str, deque]] = {}
        self._collected_at: Optional[float] = None
        self._collection_error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

        self._vendor_tools: Dict[str, str] = {}
        self._amd_gpu_names: Dict[int, str] = {}

        self._detect_environment()

    def _detect_environment(self):
        os_name = platform.system()
        requested = [v for v in self._requested_vendors if v in ("nvidia", "amd")]
        if not requested:
            requested = list(DEFAULT_VENDORS)

        compatible = [v for v in requested if _vendor_supported_on_os(v, os_name)]

        self._platform_info = {
            "os": os_name,
            "platform": platform.platform(),
            "vendors_requested": requested,
            "vendors_compatible": compatible,
            "vendors_available": [],
        }

        if not self._enabled:
            self._status = STATUS_DISABLED
            self._reason = "disabled_in_config"
            logger.info("GPU metrics collection is disabled in configuration")
            return

        if not compatible:
            self._status = STATUS_UNAVAILABLE
            self._reason = REASON_UNSUPPORTED_PLATFORM
            logger.info(
                "GPU metrics unavailable: no compatible vendors for OS %s",
                os_name,
            )
            return

        tool_names = {
            "nvidia": "nvidia-smi",
            "amd": "amd-smi",
        }
        for vendor in compatible:
            tool = tool_names[vendor]
            path = shutil.which(tool)
            if path:
                self._vendor_tools[vendor] = path

        self._platform_info["vendors_available"] = sorted(self._vendor_tools.keys())

        if not self._vendor_tools:
            self._status = STATUS_UNAVAILABLE
            self._reason = REASON_TOOL_NOT_FOUND
            logger.info(
                "GPU metrics unavailable: no requested vendor tools found (%s)",
                ", ".join(tool_names[v] for v in compatible),
            )
            return

        self._status = STATUS_UNAVAILABLE
        self._reason = None
        self._collection_error = None
        self._platform_info["tool_paths"] = dict(self._vendor_tools)
        logger.info(
            "GPU metrics collectors available: %s",
            ", ".join(self._vendor_tools.keys()),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def rate_limit_rpm(self) -> int:
        return self._rate_limit_rpm

    async def start(self):
        if not self._enabled or not self._vendor_tools:
            return
        if self._task is not None:
            return

        if "amd" in self._vendor_tools:
            await asyncio.get_running_loop().run_in_executor(
                None, self._fetch_amd_gpu_names
            )

        try:
            await self._collect_once()
        except Exception as exc:
            logger.error("Initial GPU metrics collection error: %s", exc)
            self._collection_error = str(exc)

        self._task = asyncio.create_task(self._poll_loop())
        logger.info("GPU metrics collector started")

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("GPU metrics collector stopped")

    def get_snapshot(
        self, runner_associations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": self._status,
            "reason": self._reason,
            "platform": self._platform_info,
            "collected_at": self._collected_at,
            "poll_interval_seconds": self._poll_interval if self._enabled else 0,
            "collection_error": self._collection_error,
            "gpus": self._gpus,
            "gpu_history": self._serialize_history(),
        }
        if runner_associations is not None:
            result["runner_associations"] = runner_associations
        return result

    def _serialize_history(self) -> Dict[str, Dict[str, List[Optional[float]]]]:
        out: Dict[str, Dict[str, List[Optional[float]]]] = {}
        for gpu_id, fields in self._history.items():
            out[gpu_id] = {fname: list(buf) for fname, buf in fields.items()}
        return out

    async def _poll_loop(self):
        while True:
            try:
                await self._collect_once()
            except Exception as exc:
                logger.error("GPU metrics collection error: %s", exc)
                self._collection_error = str(exc)
            await asyncio.sleep(self._poll_interval)

    async def _collect_once(self):
        loop = asyncio.get_running_loop()

        all_gpus: List[Dict[str, Any]] = []
        errors: List[Tuple[str, str]] = []

        for vendor in sorted(self._vendor_tools.keys()):
            try:
                vendor_gpus, reason, error = await asyncio.wait_for(
                    loop.run_in_executor(None, self._collect_vendor_blocking, vendor),
                    timeout=self._command_timeout + 2,
                )
            except asyncio.TimeoutError:
                errors.append(
                    (REASON_COMMAND_TIMEOUT, f"{vendor} collection timed out")
                )
                continue

            if vendor_gpus:
                all_gpus.extend(vendor_gpus)
            elif reason:
                errors.append((reason, error or f"{vendor} collection failed"))

        if not all_gpus:
            self._status = STATUS_UNAVAILABLE
            if errors:
                self._reason = errors[0][0]
                self._collection_error = "; ".join(err for _, err in errors)
            else:
                self._reason = REASON_NO_VISIBLE_GPUS
                self._collection_error = None
            self._gpus = []
            self._collected_at = None
            self._history = {}
            return

        self._gpus = all_gpus
        self._status = STATUS_AVAILABLE
        self._reason = None
        self._collection_error = "; ".join(err for _, err in errors) if errors else None
        self._collected_at = time.time()

        for gpu in all_gpus:
            gpu_id = str(
                gpu.get("id", f"{gpu.get('vendor', 'gpu')}:{gpu.get('index', 0)}")
            )
            if gpu_id not in self._history:
                self._history[gpu_id] = {
                    fname: deque(maxlen=self._history_points)
                    for fname in HISTORY_FIELDS
                }
            for fname in HISTORY_FIELDS:
                self._history[gpu_id][fname].append(gpu.get(fname))

    def _collect_vendor_blocking(
        self, vendor: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        if vendor == "amd":
            return self._collect_amd_blocking()
        if vendor == "nvidia":
            return self._collect_nvidia_blocking()
        return [], REASON_COMMAND_FAILED, f"Unknown vendor collector: {vendor}"

    def _fetch_amd_gpu_names(self):
        try:
            result = subprocess.run(
                ["amd-smi", "static", "--json"],
                capture_output=True,
                text=True,
                timeout=self._command_timeout,
            )
            if result.returncode != 0:
                logger.debug("amd-smi static returned non-zero: %s", result.returncode)
                return
            data = json.loads(result.stdout)
            if not isinstance(data, list):
                data = [data]
            for gpu in data:
                idx = gpu.get("gpu", 0)
                market_name = _deep_get(gpu, "asic.market_name")
                if market_name and market_name != "N/A":
                    self._amd_gpu_names[idx] = market_name
        except Exception as exc:
            logger.debug("Could not fetch GPU names from amd-smi static: %s", exc)

    def _collect_amd_blocking(
        self,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        try:
            result = subprocess.run(
                ["amd-smi", "metric", "--json"],
                capture_output=True,
                text=True,
                timeout=self._command_timeout,
            )
        except FileNotFoundError:
            return [], REASON_TOOL_NOT_FOUND, "amd-smi command not found"
        except subprocess.TimeoutExpired:
            return [], REASON_COMMAND_TIMEOUT, "amd-smi command timed out"

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:200]
            return (
                [],
                REASON_COMMAND_FAILED,
                f"amd-smi exited with code {result.returncode}: {stderr}",
            )

        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            return [], REASON_PARSE_ERROR, f"Failed to parse amd-smi JSON: {exc}"

        if not isinstance(data, list):
            data = [data]

        return self._normalize_amd(data), None, None

    def _collect_nvidia_blocking(
        self,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        query_fields = [
            "index",
            "name",
            "pci.bus_id",
            "memory.used",
            "memory.total",
            "utilization.gpu",
            "temperature.gpu",
            "power.draw",
        ]
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._command_timeout,
            )
        except FileNotFoundError:
            return [], REASON_TOOL_NOT_FOUND, "nvidia-smi command not found"
        except subprocess.TimeoutExpired:
            return [], REASON_COMMAND_TIMEOUT, "nvidia-smi command timed out"

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:200]
            return (
                [],
                REASON_COMMAND_FAILED,
                f"nvidia-smi exited with code {result.returncode}: {stderr}",
            )

        rows = []
        try:
            for row in csv.reader(result.stdout.splitlines()):
                if row:
                    rows.append([col.strip() for col in row])
        except Exception as exc:
            return [], REASON_PARSE_ERROR, f"Failed to parse nvidia-smi CSV: {exc}"

        return self._normalize_nvidia(rows), None, None

    def _normalize_amd(self, raw_gpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for i, raw in enumerate(raw_gpus):
            gpu_index = int(raw.get("gpu", i))
            name = self._amd_gpu_names.get(gpu_index, f"AMD GPU {gpu_index}")

            mem_usage = raw.get("mem_usage", {})
            total_vram = _deep_get(mem_usage, "total_vram.value")
            used_vram = _deep_get(mem_usage, "used_vram.value")
            temperature = _deep_get(raw, "temperature.edge.value")

            usage = raw.get("usage")
            if isinstance(usage, dict):
                util_val = _deep_get(usage, "gfx")
            else:
                util_val = usage

            socket_power = _deep_get(raw, "power.socket_power")

            normalized.append(
                {
                    "id": f"amd:{gpu_index}",
                    "vendor": "amd",
                    "index": gpu_index,
                    "name": name,
                    "pci_bus_id": _deep_get(raw, "bus.bdf"),
                    "memory_used_mb": _safe_float(used_vram),
                    "memory_total_mb": _safe_float(total_vram),
                    "utilization_gpu_percent": _safe_float(util_val),
                    "temperature_c": _safe_float(temperature),
                    "power_w": _safe_float(socket_power),
                    "raw_source": {
                        k: v
                        for k, v in raw.items()
                        if k
                        in (
                            "gpu",
                            "usage",
                            "power",
                            "temperature",
                            "mem_usage",
                            "pcie",
                            "bus",
                            "ecc",
                            "fan",
                        )
                    },
                }
            )

        return normalized

    @staticmethod
    def _normalize_nvidia(rows: List[List[str]]) -> List[Dict[str, Any]]:
        normalized = []
        for row in rows:
            if len(row) < 8:
                continue

            index = int(_safe_int(row[0], default=0))
            normalized.append(
                {
                    "id": f"nvidia:{index}",
                    "vendor": "nvidia",
                    "index": index,
                    "name": row[1] or f"NVIDIA GPU {index}",
                    "pci_bus_id": row[2] or None,
                    "memory_used_mb": _safe_float(row[3]),
                    "memory_total_mb": _safe_float(row[4]),
                    "utilization_gpu_percent": _safe_float(row[5]),
                    "temperature_c": _safe_float(row[6]),
                    "power_w": _safe_float(row[7]),
                    "raw_source": {
                        "csv_row": row,
                    },
                }
            )

        return normalized


def _vendor_supported_on_os(vendor: str, os_name: str) -> bool:
    if vendor == "nvidia":
        return os_name in ("Linux", "Windows")
    if vendor == "amd":
        return os_name == "Linux"
    return False


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        f = float(value)
        return f if f >= 0 else None

    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        f = float(match.group(0))
        return f if f >= 0 else None
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _deep_get(data: Any, path: str) -> Any:
    """Traverse a nested dict using a dot-separated path."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    if current == "N/A":
        return None
    return current


class RateLimiter:
    """Simple in-memory per-IP rate limiter."""

    def __init__(self, requests_per_minute: int):
        self._max = requests_per_minute
        self._windows: Dict[str, List[float]] = {}

    def check(self, key: str) -> Tuple[bool, int]:
        """Check whether key is allowed and return retry-after seconds.

        Returns:
            Tuple of (allowed, retry_after_seconds).
        """
        now = time.time()
        window = self._windows.get(key, [])
        cutoff = now - 60.0
        window = [ts for ts in window if ts > cutoff]
        if len(window) >= self._max:
            self._windows[key] = window
            oldest = min(window)
            retry_after = max(1, int((oldest + 60.0) - now) + 1)
            return False, retry_after

        window.append(now)
        self._windows[key] = window
        return True, 0

    def is_allowed(self, key: str) -> bool:
        allowed, _ = self.check(key)
        return allowed


def build_runner_gpu_associations(
    config_manager, observed_gpus: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Build advisory runner-to-GPU associations from model config fields.

    Uses ``main_gpu``, non-zero ``tensor_split`` indices, and device selector
    args (``--device`` / ``--device=``) to build advisory runner-to-GPU
    associations. When live GPU telemetry is available, selector indices are
    resolved against observed devices, with a one-based fallback only when an
    exact index does not exist for that selector's vendor scope.
    """
    associations: Dict[str, Any] = {}
    models = config_manager.get_config().get("models", [])

    for model in models:
        runner_name = model.get("runner")
        if not runner_name:
            continue
        model_alias = model.get("model_alias", "")

        selectors: List[Dict[str, Any]] = []

        model_arg_selectors = _extract_device_selectors(model.get("args"))

        runner_arg_selectors: List[Dict[str, Any]] = []

        try:
            runner_config = config_manager.get_runner_config(runner_name)
        except Exception:
            runner_config = {}

        runner_arg_selectors.extend(
            _extract_device_selectors(runner_config.get("args"))
        )
        runner_arg_selectors.extend(
            _extract_device_selectors(runner_config.get("extra_args"))
        )

        explicit_selectors = model_arg_selectors + runner_arg_selectors
        selectors.extend(explicit_selectors)

        default_vendor = _infer_association_vendor_hint(
            runner_name, runner_config, explicit_selectors, observed_gpus
        )

        if default_vendor is not None:
            selectors = [
                {
                    "vendor": selector.get("vendor") or default_vendor,
                    "index": selector.get("index"),
                }
                for selector in selectors
            ]

        if "main_gpu" in model and isinstance(model["main_gpu"], int):
            selectors.append({"vendor": default_vendor, "index": model["main_gpu"]})

        if "tensor_split" in model and isinstance(model["tensor_split"], list):
            for idx, val in enumerate(model["tensor_split"]):
                if isinstance(val, (int, float)) and val > 0:
                    selectors.append({"vendor": default_vendor, "index": idx})

        gpu_ids = _resolve_gpu_ids(selectors, observed_gpus)

        if runner_name not in associations:
            associations[runner_name] = {
                "gpu_ids": set(),
                "models": [],
            }

        associations[runner_name]["gpu_ids"].update(gpu_ids)
        associations[runner_name]["models"].append(model_alias)

    serializable: Dict[str, Any] = {}
    for runner_name, info in associations.items():
        serializable[runner_name] = {
            "gpu_ids": sorted(info["gpu_ids"]),
            "models": info["models"],
            "attribution": "advisory",
        }

    return serializable


def _extract_device_selectors(args_value: Any) -> List[Dict[str, Any]]:
    if not args_value:
        return []

    tokens: List[str] = []
    if isinstance(args_value, str):
        try:
            tokens = shlex.split(args_value)
        except ValueError:
            tokens = args_value.split()
    elif isinstance(args_value, list):
        for item in args_value:
            if isinstance(item, str):
                tokens.append(item)

    if not tokens:
        return []

    out: List[Dict[str, Any]] = []
    for i, token in enumerate(tokens):
        device_spec = None

        if token == "--device" and i + 1 < len(tokens):
            device_spec = tokens[i + 1]
        elif token.startswith("--device="):
            device_spec = token.split("=", 1)[1]

        if not device_spec:
            continue

        for raw in [part.strip() for part in device_spec.split(",") if part.strip()]:
            parsed = _parse_device_selector(raw)
            if parsed is not None:
                out.append(parsed)

    return out


def _parse_device_selector(value: str) -> Optional[Dict[str, Any]]:
    token = value.strip()
    if not token:
        return None

    vendor_aliases = {
        # Vulkan is a backend selector, not a vendor selector.
        "vulkan": None,
        "rocm": "amd",
        "hip": "amd",
        "cuda": "nvidia",
        "nvidia": "nvidia",
        "amd": "amd",
        "intel": "intel",
        "metal": "apple",
        "apple": "apple",
    }

    match = re.match(r"(?i)^([a-z_\-]+)?\s*(\d+)$", token)
    if match:
        vendor_raw = (match.group(1) or "").lower()
        vendor = vendor_aliases.get(vendor_raw) if vendor_raw else None
        return {"vendor": vendor, "index": int(match.group(2))}

    if token.isdigit():
        return {"vendor": None, "index": int(token)}

    return None


def _resolve_gpu_ids(
    selectors: List[Dict[str, Any]], observed_gpus: Optional[List[Dict[str, Any]]]
) -> set:
    if not selectors:
        return set()

    if not observed_gpus:
        out = set()
        for selector in selectors:
            index = selector.get("index")
            vendor = selector.get("vendor")
            if not isinstance(index, int):
                continue
            if vendor:
                out.add(f"{vendor}:{index}")
        return out

    observed = []
    for gpu in observed_gpus:
        gpu_id = gpu.get("id")
        vendor = gpu.get("vendor")
        index = gpu.get("index")
        if isinstance(gpu_id, str) and isinstance(index, int):
            observed.append({"id": gpu_id, "vendor": vendor, "index": index})

    out = set()
    for selector in selectors:
        idx = selector.get("index")
        vendor = selector.get("vendor")
        if not isinstance(idx, int):
            continue

        scope = [g for g in observed if (vendor is None or g["vendor"] == vendor)]
        exact = [g for g in scope if g["index"] == idx]
        if vendor is None and len(exact) > 1:
            continue
        if exact:
            for gpu in exact:
                out.add(gpu["id"])
            continue

        if idx > 0:
            fallback = [g for g in scope if g["index"] == idx - 1]
            if vendor is None and len(fallback) > 1:
                continue
            if fallback:
                for gpu in fallback:
                    out.add(gpu["id"])

    return out


def _infer_default_vendor_hint(
    runner_name: str, runner_config: Dict[str, Any], selectors: List[Dict[str, Any]]
) -> Optional[str]:
    explicit_vendors = {
        s.get("vendor") for s in selectors if isinstance(s.get("vendor"), str)
    }
    if len(explicit_vendors) == 1:
        return next(iter(explicit_vendors))

    text_parts = [runner_name]
    for key in ("type", "path", "args"):
        value = runner_config.get(key)
        if isinstance(value, str):
            text_parts.append(value)

    combined = " ".join(text_parts).lower()
    if any(tok in combined for tok in ("cuda", "nvidia")):
        return "nvidia"
    if any(tok in combined for tok in ("rocm", "hip", "amd")):
        return "amd"

    return None


def _infer_association_vendor_hint(
    runner_name: str,
    runner_config: Dict[str, Any],
    selectors: List[Dict[str, Any]],
    observed_gpus: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """Infer a vendor hint for advisory GPU association only.

    Vulkan device ordinals are backend-global rather than vendor-local. On mixed
    CUDA + Vulkan hosts this can make an otherwise valid Vulkan selector look
    ambiguous against vendor-local telemetry indexes. Keep the default parser
    vendor-neutral, but for advisory mapping prefer AMD when a Vulkan runner is
    present and the host exposes exactly one AMD GPU.
    """
    default_vendor = _infer_default_vendor_hint(runner_name, runner_config, selectors)
    if default_vendor is not None:
        return default_vendor

    if not observed_gpus:
        return None

    text_parts = [runner_name]
    for key in ("type", "path", "args", "extra_args"):
        value = runner_config.get(key)
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, list):
            text_parts.extend(str(item) for item in value)

    combined = " ".join(text_parts).lower()
    if "vulkan" not in combined:
        return None

    vendors = [gpu.get("vendor") for gpu in observed_gpus if isinstance(gpu, dict)]
    amd_count = vendors.count("amd")
    nvidia_count = vendors.count("nvidia")

    if amd_count == 1:
        return "amd"
    if amd_count == 0 and nvidia_count == 1:
        return "nvidia"

    return None
