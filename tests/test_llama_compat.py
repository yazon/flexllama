"""
Offline checks for llama.cpp compatibility handling in FlexLLama.

Covers how model configuration is translated into llama-server arguments for
current and older llama.cpp builds (load mode / deprecated mlock, Jinja and
built-in chat templates, unknown-flag warnings) and the related config
validation rules. No running server, model or llama-server binary is needed.

Usage:
    python tests/test_llama_compat.py
"""

import asyncio
import json
import logging
import os
import stat
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import runner as runner_module  # noqa: E402
from backend.config import ConfigManager  # noqa: E402
from backend.runner import (  # noqa: E402
    RunnerProcess,
    legacy_flags_to_load_mode,
    probe_supported_flags,
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

    def test_split_mode_row_warns(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(split_mode="row")
        self.assertTrue(any("split buffers" in line for line in logs.output))

    def test_tts_keys_on_llama_server_runner_warn(self):
        with self.assertLogs("backend.config", level="WARNING") as logs:
            self._load(talker_model="/talker.gguf")
        self.assertTrue(any("TTS-capable" in line for line in logs.output))

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
