"""
Token throughput metrics collector for FlexLLama.

Tracks tokens-per-second (prompt + generation) per model over time by reading
the ``usage`` and ``timings`` objects that llama-server includes in its
responses. Recording is event-driven (one bounded sample per completed
request) and intentionally cheap: a few float operations plus deque appends,
with no I/O, no awaits and no locks, so it can be called synchronously on the
request hot path without blocking real completions. All recording is wrapped
defensively and never raises into the request path.
"""

import math
import time
from collections import deque

DEFAULT_ENABLED = True
DEFAULT_HISTORY_POINTS = 60
DEFAULT_RATE_LIMIT_RPM = 120

STATUS_AVAILABLE = "available"
STATUS_DISABLED = "disabled"

# Window (seconds) used for the rolling 1-minute average and peak.
ROLLING_WINDOW_SECONDS = 60.0

# Per-model history fields retained as bounded deques. "timestamp" is the
# wall-clock time the sample was recorded, used to compute the rolling window.
HISTORY_FIELDS = (
    "generation_tokens_per_second",
    "prompt_tokens_per_second",
    "completion_tokens",
    "prompt_tokens",
    "timestamp",
)


class ThroughputMetricsCollector:
    """Event-driven per-model token throughput collector with bounded history.

    Holds one bounded deque per history field per model alias. ``record`` is
    called once per completed request from the API request coroutine and is
    pure-CPU and exception-safe. ``get_snapshot`` returns a JSON-serializable
    view with per-model rolling-1-minute averages and peaks.
    """

    def __init__(self, config: dict[str, object]) -> None:
        """Initialize the collector from a throughput metrics config dict.

        Args:
            config: The metrics.throughput configuration dictionary. Recognized
                keys are "enabled" (bool), "history_points" (positive int) and
                "rate_limit_requests_per_minute" (positive int). Missing keys
                fall back to module defaults.
        """
        self._enabled: bool = bool(config.get("enabled", DEFAULT_ENABLED))
        history_points = config.get("history_points", DEFAULT_HISTORY_POINTS)
        try:
            self._history_points: int = int(history_points)
        except (TypeError, ValueError):
            self._history_points = DEFAULT_HISTORY_POINTS
        if self._history_points < 1:
            self._history_points = DEFAULT_HISTORY_POINTS
        self._rate_limit_rpm: int = int(
            config.get("rate_limit_requests_per_minute", DEFAULT_RATE_LIMIT_RPM)
        )

        # Per model_alias -> {field_name: deque(maxlen=history_points)}.
        self._history: dict[str, dict[str, deque]] = {}
        # Per model_alias -> total completed requests recorded.
        self._request_counts: dict[str, int] = {}
        self._last_recorded_at: float | None = None

    @property
    def enabled(self) -> bool:
        """Whether throughput tracking is enabled."""
        return self._enabled

    @property
    def rate_limit_rpm(self) -> int:
        """The configured per-IP rate limit for the throughput endpoint."""
        return self._rate_limit_rpm

    def record(
        self,
        model_alias: str,
        usage: dict | None,
        timings: dict | None,
        wall_ms: float,
    ) -> None:
        """Record one throughput sample for a completed request.

        Derives generation tokens/sec from ``timings.predicted_per_second``
        when present, otherwise from ``usage.completion_tokens`` divided by the
        measured wall-clock seconds. Prompt tokens/sec is taken from
        ``timings.prompt_per_second`` when present. A sample is only recorded
        when generation-side data is available (predicted_per_second or
        completion_tokens > 0), so prompt-only responses (embeddings/rerank)
        do not pollute the generation throughput charts.

        This method is defensive: it no-ops when disabled or when both usage
        and timings are absent, and it swallows every exception so it can never
        raise into the request path.

        Args:
            model_alias: The model alias that handled the request.
            usage: The upstream ``usage`` object, or None.
            timings: The upstream ``timings`` object, or None.
            wall_ms: The measured wall-clock duration of the request, in ms.
        """
        if not self._enabled:
            return
        if usage is None and timings is None:
            return

        try:
            usage = usage if isinstance(usage, dict) else {}
            timings = timings if isinstance(timings, dict) else {}

            completion_tokens = _safe_float(usage.get("completion_tokens"))
            prompt_tokens = _safe_float(usage.get("prompt_tokens"))

            gen_tps = _safe_float(timings.get("predicted_per_second"))
            if gen_tps is None and completion_tokens and wall_ms > 0:
                gen_tps = completion_tokens / (wall_ms / 1000.0)

            prompt_tps = _safe_float(timings.get("prompt_per_second"))

            # Only record when generation-side data is present; prompt-only
            # responses (e.g. embeddings/rerank) are skipped so the generation
            # tok/s chart is not polluted with misleading zeros.
            has_generation = gen_tps is not None or (
                completion_tokens is not None and completion_tokens > 0
            )
            if not has_generation:
                return

            now = time.monotonic()
            buffers = self._history.get(model_alias)
            if buffers is None:
                buffers = {
                    field: deque(maxlen=self._history_points)
                    for field in HISTORY_FIELDS
                }
                self._history[model_alias] = buffers

            buffers["generation_tokens_per_second"].append(gen_tps)
            buffers["prompt_tokens_per_second"].append(prompt_tps)
            buffers["completion_tokens"].append(completion_tokens)
            buffers["prompt_tokens"].append(prompt_tokens)
            buffers["timestamp"].append(now)

            self._request_counts[model_alias] = (
                self._request_counts.get(model_alias, 0) + 1
            )
            self._last_recorded_at = time.time()
        except Exception:
            # Recording must never break or delay a real completion.
            return

    def get_snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of throughput history.

        When disabled, returns a stable "disabled" envelope with empty model
        and history collections. When enabled, returns per-model summary stats
        (last/avg-1m/peak generation tok/s and last prompt tok/s) plus the full
        bounded history with deques serialized to lists.

        Returns:
            The throughput metrics snapshot dictionary.
        """
        if not self._enabled:
            return {
                "status": STATUS_DISABLED,
                "enabled": False,
                "collected_at": None,
                "poll_interval_seconds": 0,
                "models": [],
                "throughput_history": {},
            }

        models: list[dict[str, object]] = []
        for alias, buffers in self._history.items():
            gen_buf = buffers.get("generation_tokens_per_second")
            prompt_buf = buffers.get("prompt_tokens_per_second")
            ts_buf = buffers.get("timestamp")

            gen_values = list(gen_buf) if gen_buf is not None else []
            prompt_values = list(prompt_buf) if prompt_buf is not None else []
            ts_values = list(ts_buf) if ts_buf is not None else []

            last_gen = _last_non_null(gen_values)
            last_prompt = _last_non_null(prompt_values)
            avg_gen, peak_gen = _rolling_avg_and_peak(gen_values, ts_values)

            models.append(
                {
                    "alias": alias,
                    "requests": self._request_counts.get(alias, 0),
                    "last_gen_tps": last_gen,
                    "avg_gen_tps_1m": avg_gen,
                    "peak_gen_tps": peak_gen,
                    "last_prompt_tps": last_prompt,
                }
            )

        return {
            "status": STATUS_AVAILABLE,
            "enabled": True,
            "collected_at": self._last_recorded_at,
            "poll_interval_seconds": 0,
            "models": models,
            "throughput_history": self._serialize_history(),
        }

    def _serialize_history(self) -> dict[str, dict[str, list]]:
        """Serialize bounded history deques to JSON-safe lists per model.

        Returns:
            A mapping of model_alias -> field_name -> list of values.
        """
        out: dict[str, dict[str, list]] = {}
        for alias, buffers in self._history.items():
            out[alias] = {field: list(buf) for field, buf in buffers.items()}
        return out


def _safe_float(value: object) -> float | None:
    """Coerce a value to a non-negative float, or None when not possible.

    Args:
        value: The value to coerce.

    Returns:
        The float value when it is a finite, non-negative number, else None.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if (math.isfinite(f) and f >= 0) else None
    try:
        f = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return f if (math.isfinite(f) and f >= 0) else None


def _last_non_null(values: list) -> float | None:
    """Return the last non-null entry in a list, or None.

    Args:
        values: The list of values (may contain None).

    Returns:
        The last non-null value, or None when there is none.
    """
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _rolling_avg_and_peak(
    values: list, timestamps: list
) -> tuple[float | None, float | None]:
    """Compute the rolling 1-minute average and peak of a value series.

    Points whose timestamp is within the last ROLLING_WINDOW_SECONDS are
    preferred; when none fall in that window, all retained points are used as
    a fallback so the dashboard still shows a meaningful trend.

    Args:
        values: The value series (may contain None entries).
        timestamps: The matching monotonic timestamps for each value.

    Returns:
        A tuple of (average, peak), each a float or None when no data exists.
    """
    now = time.monotonic()
    cutoff = now - ROLLING_WINDOW_SECONDS

    windowed: list[float] = []
    fallback: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        fallback.append(value)
        ts = timestamps[index] if index < len(timestamps) else None
        if ts is not None and ts >= cutoff:
            windowed.append(value)

    selected = windowed if windowed else fallback
    if not selected:
        return None, None

    avg = sum(selected) / len(selected)
    peak = max(selected)
    return avg, peak
