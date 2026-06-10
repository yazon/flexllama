"""
GPU metrics endpoint verification script for FlexLLama.

This script validates that the /v1/metrics/gpus endpoint returns a stable
JSON schema and degrades gracefully when GPU tooling is unavailable.
It also verifies rate-limiting behavior.

Usage:
    python tests/test_gpu_metrics.py [base_url]

    base_url defaults to http://localhost:8080
"""

import os
import sys
import logging
import argparse
import asyncio
import aiohttp
import uuid
from datetime import datetime

from backend.gpu_metrics import (
    _infer_default_vendor_hint,
    _parse_device_selector,
    build_runner_gpu_associations,
)


def setup_test_logging(debug=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    session_id = f"test_gpu_metrics_{timestamp}_{session_uuid}"

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
        os.path.join(test_log_dir, "test_gpu_metrics.log"), mode="w", encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
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


async def test_metrics_endpoint_schema(base_url):
    logger.info("Test: GET /v1/metrics/gpus returns valid schema")
    url = f"{base_url}/v1/metrics/gpus"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    record_fail(
                        "metrics_endpoint_status", f"Expected 200, got {resp.status}"
                    )
                    return

                data = await resp.json()

                required_top = ["status", "gpus", "gpu_history"]
                for field in required_top:
                    if field not in data:
                        record_fail(
                            f"metrics_schema_field_{field}",
                            f"Missing top-level field: {field}",
                        )
                    else:
                        record_pass(f"metrics_schema_field_{field}")

                if data["status"] not in ("available", "unavailable", "disabled"):
                    record_fail(
                        "metrics_status_value", f"Unexpected status: {data['status']}"
                    )
                else:
                    record_pass("metrics_status_value")

                if not isinstance(data["gpus"], list):
                    record_fail("metrics_gpus_type", "gpus must be a list")
                else:
                    record_pass("metrics_gpus_type")

                if not isinstance(data["gpu_history"], dict):
                    record_fail(
                        "metrics_gpu_history_type", "gpu_history must be a dict"
                    )
                else:
                    record_pass("metrics_gpu_history_type")

    except Exception as e:
        record_fail("metrics_endpoint_fetch", str(e))


async def test_metrics_available_state(base_url):
    logger.info("Test: When metrics are available, GPU objects have expected fields")
    url = f"{base_url}/v1/metrics/gpus"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()

                if data["status"] != "available":
                    logger.info(
                        "  SKIP: GPU metrics not available on this system (expected on non-Linux or without amd-smi)"
                    )
                    return

                if len(data["gpus"]) == 0:
                    record_fail(
                        "metrics_available_no_gpus",
                        "Status is available but gpus list is empty",
                    )
                    return

                gpu = data["gpus"][0]
                expected_fields = [
                    "index",
                    "name",
                    "memory_used_mb",
                    "memory_total_mb",
                    "utilization_gpu_percent",
                    "temperature_c",
                ]
                for field in expected_fields:
                    if field not in gpu:
                        record_fail(
                            f"gpu_object_field_{field}",
                            f"Missing field in GPU object: {field}",
                        )
                    else:
                        record_pass(f"gpu_object_field_{field}")

    except Exception as e:
        record_fail("metrics_available_test", str(e))


async def test_metrics_unavailable_state(base_url):
    logger.info("Test: When metrics are unavailable, response includes reason")
    url = f"{base_url}/v1/metrics/gpus"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                data = await resp.json()

                if data["status"] == "available":
                    logger.info("  SKIP: GPU metrics are available on this system")
                    return

                if "reason" not in data:
                    record_fail(
                        "unavailable_reason_field",
                        "Missing 'reason' field in unavailable response",
                    )
                else:
                    record_pass("unavailable_reason_field")

                valid_reasons = [
                    "unsupported_platform",
                    "tool_not_found",
                    "command_failed",
                    "command_timeout",
                    "parse_error",
                    "no_visible_gpus",
                    "disabled_in_config",
                ]
                if data["reason"] in valid_reasons:
                    record_pass("unavailable_reason_value")
                else:
                    record_fail(
                        "unavailable_reason_value",
                        f"Unexpected reason: {data['reason']}",
                    )

                if data["reason"] == "command_failed" and not data.get(
                    "collection_error"
                ):
                    record_fail(
                        "unavailable_command_failed_has_error",
                        "command_failed reason must include collection_error",
                    )
                else:
                    record_pass("unavailable_command_failed_has_error")

                if data.get("gpus") != []:
                    record_fail(
                        "unavailable_gpus_empty",
                        "Unavailable state should return an empty gpus list",
                    )
                else:
                    record_pass("unavailable_gpus_empty")

                if data.get("gpu_history") != {}:
                    record_fail(
                        "unavailable_gpu_history_empty",
                        "Unavailable state should clear gpu_history",
                    )
                else:
                    record_pass("unavailable_gpu_history_empty")

                if data.get("collected_at") is not None:
                    record_fail(
                        "unavailable_collected_at_none",
                        "Unavailable state should set collected_at to null",
                    )
                else:
                    record_pass("unavailable_collected_at_none")

    except Exception as e:
        record_fail("metrics_unavailable_test", str(e))


def test_vendor_selector_and_associations():
    logger.info("Test: Vulkan selectors stay vendor-neutral")

    vulkan_selector = _parse_device_selector("vulkan0")
    if vulkan_selector == {"vendor": None, "index": 0}:
        record_pass("selector_vulkan_vendor_neutral")
    else:
        record_fail(
            "selector_vulkan_vendor_neutral",
            f"Unexpected selector parse result: {vulkan_selector}",
        )

    cuda_selector = _parse_device_selector("cuda1")
    if cuda_selector == {"vendor": "nvidia", "index": 1}:
        record_pass("selector_cuda_vendor")
    else:
        record_fail(
            "selector_cuda_vendor",
            f"Unexpected selector parse result: {cuda_selector}",
        )

    vendor_hint = _infer_default_vendor_hint(
        "runner_vulkan",
        {"args": "--device vulkan0"},
        [{"vendor": None, "index": 0}],
    )
    if vendor_hint is None:
        record_pass("vendor_hint_vulkan_neutral")
    else:
        record_fail("vendor_hint_vulkan_neutral", f"Unexpected hint: {vendor_hint}")

    class StubConfigManager:
        def __init__(self):
            self._config = {
                "models": [
                    {
                        "runner": "runner_vulkan",
                        "model_alias": "test-model",
                        "main_gpu": 0,
                    }
                ]
            }
            self._runners = {
                "runner_vulkan": {
                    "args": "--device vulkan0",
                }
            }

        def get_config(self):
            return self._config

        def get_runner_config(self, runner_name):
            return self._runners[runner_name]

    associations = build_runner_gpu_associations(
        StubConfigManager(),
        observed_gpus=[{"id": "nvidia:0", "vendor": "nvidia", "index": 0}],
    )
    runner_assoc = associations.get("runner_vulkan", {})
    gpu_ids = runner_assoc.get("gpu_ids", [])
    if "nvidia:0" in gpu_ids and "amd:0" not in gpu_ids:
        record_pass("association_vulkan_without_forced_amd")
    else:
        record_fail(
            "association_vulkan_without_forced_amd",
            f"Unexpected associations: {gpu_ids}",
        )

    mixed_associations = build_runner_gpu_associations(
        StubConfigManager(),
        observed_gpus=[
            {"id": "amd:0", "vendor": "amd", "index": 0},
            {"id": "nvidia:0", "vendor": "nvidia", "index": 0},
        ],
    )
    mixed_gpu_ids = mixed_associations.get("runner_vulkan", {}).get("gpu_ids", [])
    if "amd:0" in mixed_gpu_ids and "nvidia:0" not in mixed_gpu_ids:
        record_pass("association_vulkan_prefers_single_amd_gpu")
    else:
        record_fail(
            "association_vulkan_prefers_single_amd_gpu",
            f"Unexpected mixed-vendor associations: {mixed_gpu_ids}",
        )


async def test_rate_limiting(base_url):
    logger.info("Test: Rate limiting returns 429 after burst")
    url = f"{base_url}/v1/metrics/gpus"

    got_429 = False
    try:
        async with aiohttp.ClientSession() as session:
            for i in range(150):
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 429:
                        got_429 = True
                        body = await resp.json()
                        if "error" in body:
                            record_pass("rate_limit_429_body")
                        else:
                            record_fail(
                                "rate_limit_429_body",
                                "429 response missing error object",
                            )

                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            record_pass("rate_limit_retry_after_header")
                        else:
                            record_fail(
                                "rate_limit_retry_after_header",
                                "Missing Retry-After header",
                            )
                        break

            if got_429:
                record_pass("rate_limit_triggered")
            else:
                logger.info(
                    "  SKIP: Rate limit not triggered (may have high default limit)"
                )

    except Exception as e:
        record_fail("rate_limit_test", str(e))


async def test_health_unchanged(base_url):
    logger.info("Test: /health endpoint still works with GPU metrics enabled")
    url = f"{base_url}/health"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    record_fail(
                        "health_endpoint_status", f"Expected 200, got {resp.status}"
                    )
                    return

                data = await resp.json()
                required = ["status", "active_runners", "model_health"]
                for field in required:
                    if field not in data:
                        record_fail(f"health_field_{field}", f"Missing field: {field}")
                    else:
                        record_pass(f"health_field_{field}")

    except Exception as e:
        record_fail("health_endpoint_test", str(e))


async def run_tests(base_url, skip_rate_limit=False):
    logger.info(f"Testing GPU metrics endpoint at {base_url}")
    logger.info("=" * 60)

    test_vendor_selector_and_associations()
    await test_metrics_endpoint_schema(base_url)
    await test_metrics_available_state(base_url)
    await test_metrics_unavailable_state(base_url)
    if not skip_rate_limit:
        await test_rate_limiting(base_url)
    else:
        logger.info("SKIP: Rate limit test (disabled by flag)")
    await test_health_unchanged(base_url)

    logger.info("=" * 60)
    logger.info(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED > 0:
        logger.error("Some tests FAILED")
        sys.exit(1)
    else:
        logger.info("All tests PASSED")


def main():
    parser = argparse.ArgumentParser(description="GPU metrics endpoint verification")
    parser.add_argument(
        "base_url",
        nargs="?",
        default="http://localhost:8080",
        help="Base URL of the FlexLLama server",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--skip-rate-limit", action="store_true", help="Skip rate-limit burst test"
    )
    args = parser.parse_args()

    setup_test_logging(args.debug)
    asyncio.run(run_tests(args.base_url, args.skip_rate_limit))


if __name__ == "__main__":
    main()
