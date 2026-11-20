"""
Offline checks for llama.cpp compatibility handling in FlexLLama.

Covers how model configuration is translated into llama-server arguments for
current and older llama.cpp builds (load mode / deprecated mlock, Jinja and
built-in chat templates, unknown-flag warnings) and the related config
validation rules, plus how load failures and still-loading runners are
handled (using a small fake llama-server). No real server, model or
llama-server binary is needed.

Usage:
    python tests/test_llama_compat.py
"""

import asyncio
import json
import logging
import os
import socket
import stat
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import runner as runner_module  # noqa: E402
from backend.config import ConfigManager  # noqa: E402
from backend.runner import (  # noqa: E402
    RunnerManager,
    RunnerProcess,
    is_out_of_memory_error,
    legacy_flags_to_load_mode,
    probe_supported_flags,
    summarize_load_error,
)

# Flag sets standing in for `llama-server --help` of a current build and of a
# build from before --load-mode existed (mid-2025).
CURRENT_FLAGS = frozenset(
    {
        "--model",
        "--port",
        "--host",
        "--alias",
        "--ctx-size",
        "--load-mode",
        "--jinja",
        "--no-jinja",
        "--chat-template",
        "--flash-attn",
        "--n-gpu-layers",
        "--device",
    }
)
LEGACY_FLAGS = frozenset(
    {
        "--model",
        "--port",
        "--host",
        "--alias",
        "--ctx-size",
        "--mlock",
        "--no-mmap",
        "--jinja",
        "--chat-template",
        "--flash-attn",
        "--n-gpu-layers",
        "--device",
    }
)


def make_runner(supported_flags=None, extra_args=None):
    runner = RunnerProcess(
        "runner1",
        {
            "type": "llama-server",
            "path": "llama-server",
            "extra_args": extra_args or [],
        },
        "127.0.0.1",
        8085,
    )
    runner.supported_flags = supported_flags
    return runner


def build(model, supported_flags=None, extra_args=None):
    model = {"runner": "runner1", "model": "/m.gguf", **model}
    return make_runner(supported_flags, extra_args)._build_command(model)


def value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class LoadModeTests(unittest.TestCase):
    def test_use_mlock_becomes_mmap_mlock_on_current_build(self):
        # The old --mlock locked the default mmapped model: mmap+mlock
        cmd = build({"use_mlock": True}, CURRENT_FLAGS)
        self.assertNotIn("--mlock", cmd)
        self.assertEqual(value_after(cmd, "--load-mode"), "mmap+mlock")

    def test_use_mlock_defaults_to_load_mode_when_build_unknown(self):
        cmd = build({"use_mlock": True}, None)
        self.assertEqual(value_after(cmd, "--load-mode"), "mmap+mlock")

    def test_use_mlock_stays_mlock_on_legacy_build(self):
        cmd = build({"use_mlock": True}, LEGACY_FLAGS)
        self.assertIn("--mlock", cmd)
        self.assertNotIn("--no-mmap", cmd)
        self.assertNotIn("--load-mode", cmd)

    def test_use_mlock_false_emits_nothing(self):
        cmd = build({"use_mlock": False}, CURRENT_FLAGS)
        self.assertNotIn("--mlock", cmd)
        self.assertNotIn("--load-mode", cmd)

    def test_load_mode_wins_over_use_mlock(self):
        cmd = build({"use_mlock": True, "load_mode": "mmap+mlock"}, CURRENT_FLAGS)
        self.assertEqual(value_after(cmd, "--load-mode"), "mmap+mlock")

    def test_load_mode_legacy_mapping(self):
        self.assertIn("--no-mmap", build({"load_mode": "none"}, LEGACY_FLAGS))
        cmd = build({"load_mode": "mmap+mlock"}, LEGACY_FLAGS)
        self.assertIn("--mlock", cmd)
        self.assertNotIn("--no-mmap", cmd)
        cmd = build({"load_mode": "mlock"}, LEGACY_FLAGS)
        self.assertIn("--mlock", cmd)
        self.assertIn("--no-mmap", cmd)
        cmd = build({"load_mode": "auto"}, LEGACY_FLAGS)
        self.assertNotIn("--load-mode", cmd)

    def test_legacy_flag_in_args_is_translated(self):
        with self.assertLogs("backend.runner", level="WARNING"):
            cmd = build({"args": "--flash-attn on --mlock"}, CURRENT_FLAGS)
        self.assertNotIn("--mlock", cmd)
        self.assertEqual(value_after(cmd, "--load-mode"), "mmap+mlock")
        self.assertEqual(value_after(cmd, "--flash-attn"), "on")

    def test_legacy_flags_in_extra_args_are_combined(self):
        cmd = build({}, CURRENT_FLAGS, extra_args=["--mlock", "--no-mmap"])
        self.assertEqual(value_after(cmd, "--load-mode"), "mlock")
        self.assertEqual(cmd.count("--load-mode"), 1)

    def test_legacy_flags_to_load_mode(self):
        cases = {
            ("--mlock",): "mmap+mlock",
            ("--no-mmap", "--mlock"): "mlock",
            ("--no-mmap",): "none",
            ("--mmap",): "mmap",
            ("--mlock", "--direct-io"): "dio",
            ("-dio", "-ndio"): None,
        }
        for flags, expected in cases.items():
            with self.subTest(flags=flags):
                self.assertEqual(legacy_flags_to_load_mode(list(flags)), expected)

    def test_legacy_flag_dropped_when_load_mode_configured(self):
        cmd = build({"load_mode": "mmap", "args": "--mlock"}, CURRENT_FLAGS)
        self.assertNotIn("--mlock", cmd)
        self.assertEqual(cmd.count("--load-mode"), 1)
        self.assertEqual(value_after(cmd, "--load-mode"), "mmap")

    def test_legacy_flag_in_args_untouched_on_legacy_build(self):
        cmd = build({"args": "--mlock"}, LEGACY_FLAGS)
        self.assertIn("--mlock", cmd)
        self.assertNotIn("--load-mode", cmd)


class JinjaTests(unittest.TestCase):
    def test_builtin_template_name_disables_jinja_before_template(self):
        cmd = build({"chat_template": "gemma", "jinja": True}, CURRENT_FLAGS)
        self.assertNotIn("--jinja", cmd)
        self.assertLess(cmd.index("--no-jinja"), cmd.index("--chat-template"))

    def test_inline_jinja_template_keeps_jinja(self):
        template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        cmd = build({"chat_template": template, "jinja": True}, CURRENT_FLAGS)
        self.assertIn("--jinja", cmd)
        self.assertNotIn("--no-jinja", cmd)

    def test_jinja_false_emits_no_jinja(self):
        self.assertIn("--no-jinja", build({"jinja": False}, CURRENT_FLAGS))

    def test_jinja_unset_emits_nothing(self):
        cmd = build({}, CURRENT_FLAGS)
        self.assertNotIn("--jinja", cmd)
        self.assertNotIn("--no-jinja", cmd)

    def test_no_jinja_omitted_on_builds_without_it(self):
        # Older builds had Jinja off by default and no --no-jinja flag
        cmd = build({"chat_template": "gemma"}, LEGACY_FLAGS)
        self.assertNotIn("--no-jinja", cmd)
        self.assertEqual(value_after(cmd, "--chat-template"), "gemma")


class ModelOptionTests(unittest.TestCase):
    def test_n_parallel_becomes_parallel(self):
        cmd = build({"n_parallel": 1}, CURRENT_FLAGS)
        self.assertEqual(value_after(cmd, "--parallel"), "1")

    def test_n_parallel_unset_emits_nothing(self):
        self.assertNotIn("--parallel", build({}, CURRENT_FLAGS))


class LoadErrorTests(unittest.TestCase):
    def test_picks_first_error_level_line(self):
        lines = [
            "0.00.020.767 I srv    load_model: loading model '/m.gguf'",
            "0.34.837.105 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating "
            "20155.56 MiB on device 0: cudaMalloc failed: out of memory",
            "0.36.473.746 E srv  llama_server: exiting due to model loading error",
        ]
        message = summarize_load_error(lines, 1)
        self.assertTrue(
            message.startswith("llama-server failed to load the model (exit code 1)")
        )
        self.assertTrue(message.endswith("cudaMalloc failed: out of memory"))
        self.assertNotIn("0.34.837.105", message)

    def test_rejected_argument_before_usage_text(self):
        lines = [
            'error while handling argument "--chat-template-file": error: '
            "failed to open file '/t.jinja'",
            "usage:",
            "--chat-template-file JINJA_TEMPLATE_FILE",
        ]
        self.assertIn("--chat-template-file", summarize_load_error(lines, 1))

    def test_no_reason_found(self):
        self.assertEqual(
            summarize_load_error(["loading..."], 2),
            "llama-server failed to load the model (exit code 2)",
        )

    def test_out_of_memory_detection(self):
        for message in (
            "cudaMalloc failed: out of memory",
            "failed to allocate buffer for kv cache",
            "vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
            "error loading model: vk::Queue::submit: ErrorDeviceLost",
            "ggml_vulkan: Device memory allocation of size 798040448 failed.",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_out_of_memory_error(message))
        self.assertFalse(
            is_out_of_memory_error("device CUDA0 does not support split buffers")
        )
        self.assertFalse(is_out_of_memory_error(None))


class UnsupportedFlagTests(unittest.TestCase):
    def test_warns_about_flags_the_binary_rejects(self):
        with self.assertLogs("backend.runner", level="WARNING") as logs:
            build({"talker_model": "/talker.gguf"}, CURRENT_FLAGS)
        self.assertTrue(any("--talker-model" in line for line in logs.output))

    def test_no_warning_for_values_or_known_flags(self):
        runner = make_runner(CURRENT_FLAGS)
        with self.assertNoLogs("backend.runner", level="WARNING"):
            runner._warn_unsupported_flags(
                ["llama-server", "--n-gpu-layers", "-1", "--device", "Vulkan0"]
            )

    def test_no_check_when_flags_unknown(self):
        runner = make_runner(None)
        with self.assertNoLogs("backend.runner", level="WARNING"):
            runner._warn_unsupported_flags(["tts-server", "--talker-model", "x"])


class ProbeTests(unittest.TestCase):
    def setUp(self):
        runner_module._SUPPORTED_FLAGS_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        runner_module._SUPPORTED_FLAGS_CACHE.clear()

    def _script(self, body):
        path = os.path.join(self.tmp.name, "fake-server")
        with open(path, "w") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path

    @unittest.skipIf(os.name == "nt", "shell script fake binary")
    def test_parses_llama_server_help(self):
        path = self._script(
            "echo '-m,    --model FNAME   model path'\n"
            "echo '--port PORT            port to listen'\n"
            "echo '-lm,   --load-mode MODE  loading mode (non-Metal devices)'\n"
            "echo '--jinja, --no-jinja    use jinja'\n"
        )
        flags = probe_supported_flags(path, [], dict(os.environ))
        self.assertIn("--load-mode", flags)
        self.assertIn("-lm", flags)
        self.assertIn("--no-jinja", flags)
        self.assertNotIn("-Metal", flags)

    @unittest.skipIf(os.name == "nt", "shell script fake binary")
    def test_non_llama_server_output_gives_none(self):
        path = self._script("echo 'usage: tts-server [options]'\n")
        self.assertIsNone(probe_supported_flags(path, [], dict(os.environ)))

    def test_missing_binary_gives_none(self):
        missing = os.path.join(self.tmp.name, "does-not-exist")
        self.assertIsNone(probe_supported_flags(missing, [], dict(os.environ)))


@unittest.skipIf(os.name == "nt", "shell script fake binary")
class StartFailureTests(unittest.TestCase):
    def setUp(self):
        runner_module._SUPPORTED_FLAGS_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        runner_module._SUPPORTED_FLAGS_CACHE.clear()

    def test_exit_during_start_is_reported_with_log_tail(self):
        path = os.path.join(self.tmp.name, "llama-server")
        with open(path, "w") as f:
            f.write(
                "#!/bin/sh\n"
                "echo 'llama_model_load: error loading model: out of memory'\n"
                "exit 1\n"
            )
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        runner = RunnerProcess(
            "runner1",
            {"type": "llama-server", "path": path},
            "127.0.0.1",
            18797,
            session_log_dir=self.tmp.name,
        )
        model = {"runner": "runner1", "model": "/m.gguf", "model_alias": "m"}
        runner.add_model(model)

        async def start():
            # Simulate the auto-unload watchdog noticing the exit first
            async def watchdog():
                while True:
                    await runner.is_running()
                    await asyncio.sleep(0.2)

            task = asyncio.create_task(watchdog())
            try:
                return await runner.start_with_model("m")
            finally:
                task.cancel()

        started = time.monotonic()
        with self.assertLogs("backend.runner", level="ERROR") as logs:
            ok = asyncio.run(start())
        self.assertFalse(ok)
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(any("exited during startup" in line for line in logs.output))
        self.assertTrue(any("out of memory" in line for line in logs.output))
        self.assertFalse(runner.is_starting)
        self.assertEqual(runner.load_error_alias, "m")
        self.assertIn("out of memory", runner.load_error)


# Stand-in for llama-server, driven by the model path:
#   *oom*       prints an out-of-memory error and exits 1 before listening
#   *late-oom*  answers /health with 503 (loading), then fails the same way
#   *loading*   answers 503 until "<model>.ready" exists, then 200
#   otherwise   answers 200 (loaded)
FAKE_SERVER = """
import os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
if "--help" in args:
    print("--model FNAME --port PORT --parallel N")
    sys.exit(0)
port = int(args[args.index("--port") + 1])
model = args[args.index("--model") + 1]

def fail():
    print("0.01.000.000 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating "
          "20155.56 MiB on device 0: cudaMalloc failed: out of memory", flush=True)
    print("0.01.000.001 E srv  llama_server: exiting due to model loading error", flush=True)
    os._exit(1)

name = os.path.basename(model)
if name.startswith("oom"):
    fail()
if name.startswith("late-oom"):
    threading.Timer(3, fail).start()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        loading = name.startswith("late-oom") or (
            name.startswith("loading") and not os.path.exists(model + ".ready")
        )
        self.send_response(503 if loading else 200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "loading"}' if loading else b'{"status": "ok"}')

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipIf(os.name == "nt", "fake server script needs a shebang")
class RunnerLifecycleTests(unittest.TestCase):
    """Load failures and still-loading runners, against a fake llama-server."""

    def setUp(self):
        runner_module._SUPPORTED_FLAGS_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.server = os.path.join(self.tmp.name, "llama-server")
        with open(self.server, "w") as f:
            f.write(f"#!{sys.executable}\n{FAKE_SERVER}")
        os.chmod(self.server, os.stat(self.server).st_mode | stat.S_IEXEC)

    def tearDown(self):
        self.tmp.cleanup()
        runner_module._SUPPORTED_FLAGS_CACHE.clear()

    def _manager(self, models, auto_unload=0, max_retries=3):
        """Build a RunnerManager with one runner per entry of models (alias -> file)."""
        config = {
            "api": {"host": "127.0.0.1", "port": free_port()},
            "retry_config": {"max_retries": max_retries, "base_delay_seconds": 4},
            "models": [],
        }
        for i, (alias, filename) in enumerate(models.items()):
            runner = f"runner{i}"
            config[runner] = {
                "type": "llama-server",
                "path": self.server,
                "host": "127.0.0.1",
                "port": free_port(),
                "auto_unload_timeout_seconds": auto_unload,
            }
            config["models"].append(
                {
                    "runner": runner,
                    "model": os.path.join(self.tmp.name, filename),
                    "model_alias": alias,
                }
            )
        path = os.path.join(self.tmp.name, "config.json")
        with open(path, "w") as f:
            json.dump(config, f)
        with self.assertLogs("backend.config", level="WARNING"):  # files missing
            config_manager = ConfigManager(path)
        return RunnerManager(config_manager, session_log_dir=self.tmp.name)

    def _starts(self, runner_name):
        with open(os.path.join(self.tmp.name, f"{runner_name}.log")) as f:
            return f.read().count("=== Starting with model")

    def test_out_of_memory_evicts_idle_runner_then_fails_fast(self):
        manager = self._manager({"big": "oom.gguf", "idle": "idle.gguf"})

        async def scenario():
            try:
                ok, _ = await manager.ensure_model_ready_with_retry("idle")
                self.assertTrue(ok)
                started = time.monotonic()
                ok, error = await manager.ensure_model_ready_with_retry("big")
                return ok, error, time.monotonic() - started
            finally:
                await manager.stop_all_runners()

        with self.assertLogs("backend.runner", level="INFO") as logs:
            ok, error, elapsed = asyncio.run(scenario())
        self.assertFalse(ok)
        self.assertIn("cudaMalloc failed: out of memory", error)
        # Without fail-fast the retry schedule alone takes 4+8+16 s
        self.assertLess(elapsed, 15)
        self.assertTrue(
            any("Stopping idle runner runner1" in line for line in logs.output)
        )
        self.assertEqual(self._starts("runner0"), 2)  # first try + after eviction

    def test_exit_while_loading_is_not_restarted(self):
        manager = self._manager({"big": "late-oom.gguf"})

        async def scenario():
            try:
                started = time.monotonic()
                ok, error = await manager.ensure_model_ready_with_retry("big")
                return ok, error, time.monotonic() - started
            finally:
                await manager.stop_all_runners()

        with self.assertLogs("backend.runner", level="INFO"):
            ok, error, elapsed = asyncio.run(scenario())
        self.assertFalse(ok)
        self.assertIn("out of memory", error)
        self.assertLess(elapsed, 20)  # no 30 s wait for a dead process
        self.assertEqual(self._starts("runner0"), 1)

    def test_eviction_when_failure_is_seen_after_last_retry(self):
        # With no retries left, a load that fails late must still get its
        # one retry after freeing memory
        manager = self._manager(
            {"big": "late-oom.gguf", "idle": "idle.gguf"}, max_retries=0
        )

        async def scenario():
            try:
                ok, _ = await manager.ensure_model_ready_with_retry("idle")
                self.assertTrue(ok)
                return await manager.ensure_model_ready_with_retry("big")
            finally:
                await manager.stop_all_runners()

        with self.assertLogs("backend.runner", level="INFO") as logs:
            ok, error = asyncio.run(scenario())
        self.assertFalse(ok)
        self.assertIn("out of memory", error)
        self.assertTrue(
            any("Stopping idle runner runner1" in line for line in logs.output)
        )
        self.assertEqual(self._starts("runner0"), 2)

    def test_auto_unload_waits_for_load_to_finish(self):
        manager = self._manager({"slow": "loading.gguf"}, auto_unload=1)
        runner = manager.runners["runner0"]
        ready_flag = os.path.join(self.tmp.name, "loading.gguf.ready")

        async def scenario():
            try:
                self.assertTrue(await manager.start_runner_for_model("slow"))
                await manager.start_auto_unload_watchdog()
                await asyncio.sleep(3.5)  # well past the 1 s idle timeout
                still_loading = await runner.is_running()
                open(ready_flag, "w").close()
                await asyncio.sleep(4)
                return still_loading, await runner.is_running()
            finally:
                await manager.stop_auto_unload_watchdog()
                await manager.stop_all_runners()

        with self.assertLogs("backend.runner", level="INFO"):
            still_loading, after_load = asyncio.run(scenario())
        self.assertTrue(still_loading)
        self.assertFalse(after_load)


class ConfigValidationTests(unittest.TestCase):
    def _load(self, **model_overrides):
        config = {
            "api": {"host": "127.0.0.1", "port": 8080},
            "runner1": {"type": "llama-server", "path": "llama-server", "port": 8085},
            "models": [{"runner": "runner1", "model": "/m.gguf", **model_overrides}],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(config, handle)
            path = handle.name
        try:
            return ConfigManager(path)
        finally:
            os.unlink(path)

    def test_rope_scale_accepts_float(self):
        self._load(**{"rope-scale": 2.5})

    def test_n_gpu_layers_accepts_auto_and_all(self):
        self._load(n_gpu_layers="auto")
        self._load(n_gpu_layers="all")
        with self.assertRaises(ValueError):
            self._load(n_gpu_layers="max")
        with self.assertRaises(ValueError):
            self._load(n_gpu_layers=True)

    def test_load_mode_values(self):
        self._load(load_mode="mmap+mlock")
        with self.assertRaises(ValueError):
            self._load(load_mode="lock")

    def test_use_mlock_is_deprecated(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(use_mlock=True)
        self.assertTrue(any("deprecated" in line for line in logs.output))

    def test_unknown_model_key_warns(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(flash_attention="on")
        self.assertTrue(any("flash_attention" in line for line in logs.output))

    def test_jinja_with_builtin_template_warns(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(chat_template="mistral-instruct", jinja=True)
        self.assertTrue(any("built-in template" in line for line in logs.output))

    def test_n_parallel_must_be_integer(self):
        self._load(n_parallel=1)
        with self.assertRaises(ValueError):
            self._load(n_parallel="1")

    def test_missing_model_file_warns(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(model="/does/not/exist.gguf")
        self.assertTrue(any("file not found" in line for line in logs.output))

    def test_split_mode_row_warns(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(split_mode="row")
        self.assertTrue(any("split buffers" in line for line in logs.output))

    def test_tts_keys_on_llama_server_runner_warn(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(talker_model="/talker.gguf")
        self.assertTrue(any("custom TTS server" in line for line in logs.output))

    def test_example_configs_validate(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in (
            "config_example.json",
            "docker/config.json",
            "docker/config.qwen3.unified.json",
        ):
            with self.subTest(config=name):
                ConfigManager(os.path.join(root, name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    unittest.main(verbosity=2)
