"""
Runner manager for FlexLLama.

This module handles the lifecycle of llama.cpp server processes, including
starting, stopping, and monitoring processes based on configuration.
It supports multiple concurrent runners for different models.
"""

import os
import subprocess
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import socket
import aiohttp
import json
import psutil
import shlex

# Get logger for this module
logger = logging.getLogger(__name__)


# Health Status Constants
class HealthStatus:
    """Constants for health check status values."""

    OK = "ok"
    LOADING = "loading"
    ERROR = "error"
    NOT_RUNNING = "not_running"
    NOT_LOADED = "not_loaded"


# Health Messages
class HealthMessages:
    """Constants for health check messages."""

    READY = "Ready"
    MODEL_LOADING = "Model is still loading"
    RUNNER_NOT_RUNNING = "Runner not running"
    MODEL_NOT_LOADED = "Model not loaded in runner"
    NO_RUNNER_AVAILABLE = "No runner available"
    HEALTH_CHECK_TIMEOUT = "Health check timeout"
    CONNECTION_ERROR = "Connection error"
    HEALTH_CHECK_FAILED = "Health check failed"


class RunnerProcess:
    """Class representing a single FlexLLama process."""

    def __init__(self, runner_name, runner_config, host, port, session_log_dir=None):
        """Initialize a runner process.

        Args:
            runner_name: Name of the runner.
            runner_config: Configuration for the runner.
            host: Host to bind to.
            port: Port to bind to.
            session_log_dir: Session-specific log directory (optional).
        """
        self.runner_name = runner_name
        self.runner_config = runner_config
        self.host = host
        self.port = port
        self.session_log_dir = session_log_dir or "logs"
        self.process = None
        self.output_file = None
        self.models = []  # List of models this runner is responsible for
        self.current_model = None  # Track which model is currently loaded
        self.is_starting = False
        self.start_time = None

    def _kill_process_tree(self, pid: int):
        """Terminate a process and all of its children, using an OS-specific method."""
        if os.name == "nt":
            try:
                # On Windows, taskkill is more reliable for killing process trees.
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0 and result.returncode != 128:
                    logger.warning(
                        f"taskkill for PID {pid} returned exit code {result.returncode}."
                        f"\n  stdout: {result.stdout.strip()}"
                        f"\n  stderr: {result.stderr.strip()}"
                    )
            except FileNotFoundError:
                logger.warning("`taskkill` command not found. Falling back to psutil.")
                self._kill_with_psutil(pid)
        else:
            self._kill_with_psutil(pid)

    def _kill_with_psutil(self, pid: int):
        """Terminate a process and all of its children using psutil."""
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                continue

        _, alive = psutil.wait_procs(children, timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            return

        try:
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

    def add_model(self, model_config):
        """Add a model to this runner.

        Args:
            model_config: Configuration for the model.
        """
        self.models.append(model_config)

    def get_model_by_alias(self, model_alias):
        """Get a model configuration by alias.

        Args:
            model_alias: The alias of the model to get.

        Returns:
            The model configuration, or None if not found.
        """
        for model in self.models:
            if (
                model.get("model_alias", os.path.basename(model["model"]))
                == model_alias
            ):
                return model
        return None

    def is_model_loaded(self, model_alias):
        """Check if a specific model is currently loaded.

        Args:
            model_alias: The alias of the model to check.

        Returns:
            True if the model is currently loaded, False otherwise.
        """
        if self.current_model is None:
            return False

        current_alias = self.current_model.get(
            "model_alias", os.path.basename(self.current_model["model"])
        )
        return current_alias == model_alias

    async def start_with_model(self, model_alias):
        """Start the runner with a specific model, handling model switching.

        If the runner is already running with a different model, it will be
        stopped and restarted with the new model.

        Args:
            model_alias: The alias of the model to start with.

        Returns:
            True if the process was started or switched successfully, False otherwise.
        """
        # Find the model configuration
        model_config = self.get_model_by_alias(model_alias)
        if model_config is None:
            logger.error(f"Model {model_alias} not found in runner {self.runner_name}")
            return False

        # Check if this model is already loaded
        if await self.is_running() and self.is_model_loaded(model_alias):
            logger.info(
                f"Model {model_alias} is already loaded in runner {self.runner_name}"
            )
            return True

        # If a different model is loaded, stop the runner first
        if await self.is_running() and not self.is_model_loaded(model_alias):
            current_alias = (
                self.current_model.get(
                    "model_alias", os.path.basename(self.current_model["model"])
                )
                if self.current_model
                else "unknown"
            )
            logger.info(
                f"Switching runner {self.runner_name} from model {current_alias} to {model_alias}"
            )
            await self.stop()

        # Start with the specified model
        return await self._start_with_specific_model(model_config)

    async def start(self):
        """Start the runner process with the first available model.

        Returns:
            True if the process was started successfully, False otherwise.
        """
        if not self.models:
            logger.error(f"Runner {self.runner_name} has no models")
            return False

        # Use the first model if no specific model is requested
        return await self._start_with_specific_model(self.models[0])

    async def _start_with_specific_model(self, model_config):
        """Internal method to start the runner process with a specific model.

        Args:
            model_config: Configuration for the model to start with.

        Returns:
            True if the process was started successfully, False otherwise.
        """
        if self.process is not None and self.process.poll() is None:
            logger.info(f"Runner {self.runner_name} is already running")
            return True

        if self.is_starting:
            logger.info(f"Runner {self.runner_name} is already starting")
            while self.is_starting:
                await asyncio.sleep(0.5)
            return self.process is not None and self.process.poll() is None

        self.is_starting = True
        self.start_time = time.time()

        try:
            # Build command and parse environment variables
            cmd, env_from_path = self._build_command_and_env(model_config)

            # Compose the environment for the subprocess
            env_for_child = self._compose_environment(model_config, env_from_path)

            # Log deprecation warning if inline env vars were found in path
            if env_from_path:
                logger.warning(
                    f"Runner {self.runner_name}: inline env assignments in 'path' are deprecated; "
                    f"please use runner.env/model.env. Parsed vars: {', '.join(sorted(env_from_path.keys()))}"
                )

            # Log applied environment variables (names only, not values for security)
            runner_env_vars = list(self.runner_config.get("env", {}).keys())
            model_env_vars = list(model_config.get("env", {}).keys())
            all_env_vars = runner_env_vars + model_env_vars + list(env_from_path.keys())
            if all_env_vars:
                logger.info(
                    f"Runner {self.runner_name}: applying env vars {', '.join(sorted(set(all_env_vars)))}"
                )

            # Use session log directory
            log_dir = self.session_log_dir
            os.makedirs(log_dir, exist_ok=True)

            # Create log file in session directory
            log_file = os.path.join(log_dir, f"{self.runner_name}.log")

            model_alias = model_config.get(
                "model_alias", os.path.basename(model_config["model"])
            )
            logger.info(f"Starting runner {self.runner_name} with model {model_alias}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Log file: {log_file}")

            # Open log file
            self.output_file = open(
                log_file, "a"
            )  # Use append mode to preserve logs across restarts

            # Add a separator in the log file for model switches
            self.output_file.write(
                f"\n=== Starting with model {model_alias} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            self.output_file.flush()

            # Start process
            try:
                popen_kwargs = {
                    "stdout": self.output_file,
                    "stderr": self.output_file,
                    "text": True,
                    "bufsize": 1,
                    "env": env_for_child,
                }
                if os.name == "posix":
                    popen_kwargs["start_new_session"] = True
                elif os.name == "nt":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

                self.process = subprocess.Popen(cmd, **popen_kwargs)
            except Exception as e:
                logger.error(
                    f"Failed to create subprocess for runner {self.runner_name}: {e}"
                )
                self.process = None
                self.output_file.close()
                self.output_file = None
                self.current_model = None
                self.is_starting = False
                return False

            # Wait for server to start
            await asyncio.sleep(2)  # Initial wait

            # Check if process is still running
            if self.process is not None and self.process.poll() is not None:
                logger.error(
                    f"Runner {self.runner_name} exited with code: {self.process.returncode}"
                )
                self.process = None
                self.output_file.close()
                self.output_file = None
                self.current_model = None
                self.is_starting = False
                return False

            # Wait for server to be ready
            max_retries = 30
            retry_interval = 1
            for _ in range(max_retries):
                if await self._is_server_ready():
                    logger.info(
                        f"Runner {self.runner_name} started successfully with model {model_alias}"
                    )
                    self.current_model = model_config
                    self.is_starting = False
                    return True

                # Check if process is still running
                if self.process is not None and self.process.poll() is not None:
                    logger.error(
                        f"Runner {self.runner_name} exited with code: {self.process.returncode}"
                    )
                    self.process = None
                    self.output_file.close()
                    self.output_file = None
                    self.current_model = None
                    self.is_starting = False
                    return False

                await asyncio.sleep(retry_interval)

            # Server did not start in time
            logger.error(f"Runner {self.runner_name} did not start in time")
            await self.stop()
            self.is_starting = False
            return False

        except Exception as e:
            logger.error(f"Failed to start runner {self.runner_name}: {e}")
            if self.process is not None:
                await self.stop()
            self.is_starting = False
            return False

    async def stop(self):
        """Stop the runner process.

        Returns:
            True if the process was stopped successfully, False otherwise.
        """
        if self.process is None:
            logger.info(f"Runner {self.runner_name} is not running")
            return True

        try:
            current_alias = (
                self.current_model.get(
                    "model_alias", os.path.basename(self.current_model["model"])
                )
                if self.current_model
                else "unknown"
            )
            logger.info(
                f"Stopping runner {self.runner_name} (current model: {current_alias})"
            )

            pid = self.process.pid
            loop = asyncio.get_event_loop()

            # Run synchronous process killing in a thread to avoid blocking
            await loop.run_in_executor(None, self._kill_process_tree, pid)

            # Poll to update the return code
            if self.process:
                self.process.poll()

            # Get exit code before nullifying process
            exit_code = (
                self.process.returncode if self.process is not None else "unknown"
            )
            logger.info(
                f"Runner {self.runner_name} stopped with exit code: {exit_code}"
            )

            # Close log file
            if self.output_file is not None:
                try:
                    self.output_file.close()
                except Exception:
                    pass
                self.output_file = None

            self.process = None
            self.current_model = None  # Reset current model

            # Wait to offload GPU memory
            await asyncio.sleep(0.5)

            return True

        except Exception as e:
            logger.error(f"Failed to stop runner {self.runner_name}: {e}")
            # Ensure cleanup even if there was an error
            if self.output_file is not None:
                try:
                    self.output_file.close()
                except Exception:
                    pass
                self.output_file = None
            self.process = None
            self.current_model = None
            return False

    async def is_running(self):
        """Check if the runner process is running.

        Returns:
            True if the process is running, False otherwise.
        """
        if self.process is None:
            return False

        # Check if process is still running
        if self.process.poll() is not None:
            current_alias = (
                self.current_model.get(
                    "model_alias", os.path.basename(self.current_model["model"])
                )
                if self.current_model
                else "unknown"
            )
            logger.warning(
                f"Runner {self.runner_name} has exited with code: {self.process.returncode} (was running model: {current_alias})"
            )
            self.process = None
            self.current_model = None  # Reset current model

            # Close log file
            if self.output_file is not None:
                self.output_file.close()
                self.output_file = None

            return False

        return True

    def _parse_runner_path_with_env(
        self, raw_path: str
    ) -> tuple[str, list[str], dict[str, str]]:
        """Parse runner 'path' that may contain leading environment assignments or an 'env' wrapper.

        Args:
            raw_path: The raw path string from runner configuration.

        Returns:
            A tuple of (executable_path, additional_args, env_from_path).
        """
        try:
            tokens = shlex.split(raw_path)
        except ValueError as e:
            logger.warning(
                f"Runner {self.runner_name}: Failed to parse path '{raw_path}': {e}. "
                "Using simple split as fallback."
            )
            tokens = raw_path.split()

        env_from_path: dict[str, str] = {}
        if not tokens:
            return raw_path, [], env_from_path

        index = 0
        # Handle 'env' command prefix
        if tokens[0] == "env":
            index = 1

        # Collect NAME=VALUE assignments
        while (
            index < len(tokens)
            and "=" in tokens[index]
            and not tokens[index].startswith("--")
        ):
            try:
                name, value = tokens[index].split("=", 1)
                env_from_path[name] = value
                index += 1
            except ValueError:
                # Malformed assignment, stop parsing env vars
                break

        if index >= len(tokens):
            # No executable token found; treat the entire string as a path
            return raw_path, [], env_from_path

        executable = tokens[index]
        index += 1
        initial_args = tokens[index:] if index < len(tokens) else []
        return executable, initial_args, env_from_path

    def _compose_environment(
        self, model_config: dict, env_from_path: dict[str, str]
    ) -> dict[str, str]:
        """Build the environment for the subprocess, honoring inherit_env and overrides.

        Precedence (last wins):
          1) base env (os.environ) if inherit_env is true, else {}
          2) runner_config["env"]
          3) model_config["env"]
          4) env_from_path (parsed from runner.path; kept for backward-compat)

        Args:
            model_config: The model configuration.
            env_from_path: Environment variables parsed from runner path.

        Returns:
            The composed environment dictionary.
        """
        # Resolve inherit_env with model override
        inherit = self.runner_config.get("inherit_env", True)
        if "inherit_env" in model_config:
            try:
                inherit = bool(model_config["inherit_env"])
            except Exception:
                inherit = True

        base_env = os.environ.copy() if inherit else {}
        merged: dict[str, str] = dict(base_env)

        # Apply environment variables in precedence order
        for mapping in (
            self.runner_config.get("env", {}),
            model_config.get("env", {}),
            env_from_path,
        ):
            if not isinstance(mapping, dict):
                continue
            for key, value in mapping.items():
                merged[str(key)] = str(value)

        return merged

    def _build_command_and_env(
        self, model_config: dict
    ) -> tuple[list[str], dict[str, str]]:
        """Build the command to start the runner process and collect any env from runner.path.

        Args:
            model_config: Configuration for the model.

        Returns:
            A tuple of (command_list, env_from_path).
        """
        # Parse runner path for environment variables and executable
        executable, initial_args, env_from_path = self._parse_runner_path_with_env(
            self.runner_config["path"]
        )

        # Start with executable and any initial args from path
        cmd: list[str] = [executable]
        if initial_args:
            cmd.extend(initial_args)

        logger.debug(
            f"Command building for {self.runner_name}: Starting with {len(cmd)} items"
        )

        # Add model path
        cmd.extend(["--model", model_config["model"]])
        logger.debug(f"Command building: After model, {len(cmd)} items: {cmd[-2:]}")

        # Add host and port
        cmd.extend(["--host", self.host, "--port", str(self.port)])
        logger.debug(f"Command building: After host/port, {len(cmd)} items: {cmd[-4:]}")

        # Add model parameters
        if "mmproj" in model_config:
            cmd.extend(["--mmproj", model_config["mmproj"]])
            logger.debug(
                f"Command building: After mmproj, {len(cmd)} items: {cmd[-2:]}"
            )

        if "model_alias" in model_config:
            cmd.extend(["--alias", model_config["model_alias"]])
            logger.debug(f"Command building: After alias, {len(cmd)} items: {cmd[-2:]}")

        if "n_ctx" in model_config:
            cmd.extend(["--ctx-size", str(model_config["n_ctx"])])
            logger.debug(f"Command building: After n_ctx, {len(cmd)} items: {cmd[-2:]}")

        if "n_batch" in model_config:
            cmd.extend(["--batch-size", str(model_config["n_batch"])])
            logger.debug(
                f"Command building: After n_batch, {len(cmd)} items: {cmd[-2:]}"
            )

        if "n_threads" in model_config:
            cmd.extend(["--threads", str(model_config["n_threads"])])
            logger.debug(
                f"Command building: After n_threads, {len(cmd)} items: {cmd[-2:]}"
            )

        if "chat_template" in model_config:
            cmd.extend(["--chat-template", model_config["chat_template"]])
            logger.debug(
                f"Command building: After chat_template, {len(cmd)} items: {cmd[-2:]}"
            )

        if "split_mode" in model_config:
            split_mode_val = model_config["split_mode"]
            logger.debug(
                f"Command building: split_mode value is {split_mode_val} (type: {type(split_mode_val)})"
            )
            cmd.extend(["--split-mode", split_mode_val])
            logger.debug(
                f"Command building: After split_mode, {len(cmd)} items: {cmd[-2:]}"
            )

        if model_config.get("embedding", False):
            cmd.extend(["--embedding"])
            logger.debug(
                f"Command building: After embedding, {len(cmd)} items: {cmd[-1:]}"
            )

        if model_config.get("reranking", False):
            cmd.extend(["--reranking"])
            logger.debug(
                f"Command building: After reranking, {len(cmd)} items: {cmd[-1:]}"
            )

        if not model_config.get("offload_kqv", True):
            cmd.append("--no-kv-offload")
            logger.debug(
                f"Command building: After no-kv-offload, {len(cmd)} items: {cmd[-1:]}"
            )

        if model_config.get("jinja", False):
            cmd.append("--jinja")
            logger.debug(f"Command building: After jinja, {len(cmd)} items: {cmd[-1:]}")

        if "pooling" in model_config:
            cmd.extend(["--pooling", model_config["pooling"]])
            logger.debug(
                f"Command building: After pooling, {len(cmd)} items: {cmd[-2:]}"
            )

        if "flash_attn" in model_config:
            cmd.extend(["--flash-attn", model_config["flash_attn"]])
            logger.debug(
                f"Command building: After flash_attn, {len(cmd)} items: {cmd[-2:]}"
            )

        if model_config.get("use_mlock", False):
            cmd.append("--mlock")
            logger.debug(
                f"Command building: After use_mlock, {len(cmd)} items: {cmd[-1:]}"
            )

        if "main_gpu" in model_config:
            cmd.extend(["--main-gpu", str(model_config["main_gpu"])])
            logger.debug(
                f"Command building: After main_gpu, {len(cmd)} items: {cmd[-2:]}"
            )

        if "tensor_split" in model_config:
            cmd.extend(
                ["--tensor-split", ",".join(map(str, model_config["tensor_split"]))]
            )
            logger.debug(
                f"Command building: After tensor_split, {len(cmd)} items: {cmd[-2:]}"
            )

        if "n_gpu_layers" in model_config:
            cmd.extend(["--n-gpu-layers", str(model_config["n_gpu_layers"])])
            logger.debug(
                f"Command building: After n_gpu_layers, {len(cmd)} items: {cmd[-2:]}"
            )

        # Handle cache type parameters (config uses hyphens)
        if "cache-type-k" in model_config:
            cache_k_val = model_config["cache-type-k"]
            logger.debug(
                f"Command building: cache-type-k value is {cache_k_val} (type: {type(cache_k_val)})"
            )
            cmd.extend(["--cache-type-k", cache_k_val])
            logger.debug(
                f"Command building: After cache-type-k, {len(cmd)} items: {cmd[-2:]}"
            )

        if "cache-type-v" in model_config:
            cache_v_val = model_config["cache-type-v"]
            logger.debug(
                f"Command building: cache-type-v value is {cache_v_val} (type: {type(cache_v_val)})"
            )
            cmd.extend(["--cache-type-v", cache_v_val])
            logger.debug(
                f"Command building: After cache-type-v, {len(cmd)} items: {cmd[-2:]}"
            )

        # Handle rope scaling parameters
        if "rope-scaling" in model_config:
            cmd.extend(["--rope-scaling", str(model_config["rope-scaling"])])
            logger.debug(
                f"Command building: After rope-scaling, {len(cmd)} items: {cmd[-2:]}"
            )

        if "rope-scale" in model_config:
            cmd.extend(["--rope-scale", str(model_config["rope-scale"])])
            logger.debug(
                f"Command building: After rope-scale, {len(cmd)} items: {cmd[-2:]}"
            )

        if "yarn-orig-ctx" in model_config:
            cmd.extend(["--yarn-orig-ctx", str(model_config["yarn-orig-ctx"])])
            logger.debug(
                f"Command building: After yarn-orig-ctx, {len(cmd)} items: {cmd[-2:]}"
            )

        # Add model-specific arguments
        if "args" in model_config and model_config["args"].strip():
            try:
                model_args = shlex.split(model_config["args"].strip())
                cmd.extend(model_args)
                logger.debug(
                    f"Command building: After model args, {len(cmd)} items: added {len(model_args)} args"
                )
            except ValueError as e:
                logger.error(
                    f"Failed to parse model args '{model_config['args']}': {e}. "
                    "Please check for unmatched quotes or invalid shell syntax."
                )
                # Fallback to simple split for malformed arguments
                model_args = model_config["args"].strip().split()
                cmd.extend(model_args)
                logger.debug(
                    f"Command building: Using fallback split, {len(cmd)} items: added {len(model_args)} args"
                )

        # Add extra arguments
        cmd.extend(self.runner_config.get("extra_args", []))

        return cmd, env_from_path

    def _build_command(self, model_config):
        """Build the command to start the runner process.

        Args:
            model_config: Configuration for the model.

        Returns:
            The command as a list of strings.
        """
        cmd, _ = self._build_command_and_env(model_config)
        return cmd

    async def _is_server_ready(self):
        """Check if the server is ready to accept connections.

        Returns:
            True if the server is ready, False otherwise.
        """
        loop = asyncio.get_event_loop()

        def check_socket():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect((self.host, self.port))
                    return True
            except (socket.timeout, ConnectionRefusedError):
                return False
            except Exception as e:
                logger.error(f"Error checking server readiness: {e}")
                return False

        # Run socket check in a thread to avoid blocking
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, check_socket)


class RunnerManager:
    """Manager for FlexLLama processes."""

    def __init__(self, config_manager, session_log_dir=None):
        """Initialize the runner manager.

        Args:
            config_manager: The configuration manager.
            session_log_dir: Session-specific log directory (optional).
        """
        self.config_manager = config_manager
        self.session_log_dir = session_log_dir or "logs"
        self.runners = {}  # Map of runner name to RunnerProcess
        self.model_runner_map = {}  # Map of model alias to runner name
        self.timeout = 300  # 5 minutes
        self._initialize_runners()

    def _initialize_runners(self):
        """Initialize runner processes based on configuration."""
        # Create runner processes with their specific host and port
        for runner_name in self.config_manager.get_runner_names():
            runner_config = self.config_manager.get_runner_config(runner_name)
            host = self.config_manager.get_runner_host(runner_name)
            port = self.config_manager.get_runner_port(runner_name)
            self.runners[runner_name] = RunnerProcess(
                runner_name, runner_config, host, port, self.session_log_dir
            )

        # Assign models to runners
        for model in self.config_manager.get_config()["models"]:
            model_alias = model.get("model_alias", os.path.basename(model["model"]))
            runner_name = model["runner"]

            if runner_name in self.runners:
                self.runners[runner_name].add_model(model)
                self.model_runner_map[model_alias] = runner_name
            else:
                logger.error(
                    f"Model {model_alias} references unknown runner {runner_name}"
                )

    async def start_runner(self, runner_name):
        """Start a runner process.

        Args:
            runner_name: Name of the runner to start.

        Returns:
            True if the runner was started successfully, False otherwise.
        """
        if runner_name not in self.runners:
            logger.error(f"Unknown runner: {runner_name}")
            return False

        return await self.runners[runner_name].start()

    async def start_runner_for_model(self, model_alias):
        """Start the runner for a specific model, loading or switching to it.

        This method ensures the correct runner is active and has the specified
        model loaded.

        Args:
            model_alias: Alias of the model.

        Returns:
            True if the runner was started/switched successfully, False otherwise.
        """
        if model_alias not in self.model_runner_map:
            logger.error(f"Unknown model: {model_alias}")
            return False

        runner_name = self.model_runner_map[model_alias]
        runner = self.runners[runner_name]

        return await runner.start_with_model(model_alias)

    async def stop_runner(self, runner_name):
        """Stop a runner process.

        Args:
            runner_name: Name of the runner to stop.

        Returns:
            True if the runner was stopped successfully, False otherwise.
        """
        if runner_name not in self.runners:
            logger.error(f"Unknown runner: {runner_name}")
            return False

        return await self.runners[runner_name].stop()

    async def stop_all_runners(self):
        """Stop all runner processes.

        Returns:
            True if all runners were stopped successfully, False otherwise.
        """
        success = True
        for runner_name in self.runners:
            if not await self.stop_runner(runner_name):
                success = False
        return success

    async def auto_start_default_runners(self):
        """Auto-start runners with their first model if enabled in configuration.

        Returns:
            True if all auto-starts were successful, False otherwise.
        """
        if not self.config_manager.get_auto_start_runners():
            logger.info("Auto-start is disabled, skipping runner auto-start")
            return True

        logger.info("Auto-starting default runners...")
        success = True
        started_count = 0

        for runner_name in self.get_runner_names():
            runner = self.runners[runner_name]
            if runner.models:  # If runner has models assigned
                logger.info(
                    f"Auto-starting runner {runner_name} with model {runner.models[0].get('model_alias', 'unknown')}"
                )
                if await self.start_runner(runner_name):
                    started_count += 1
                    logger.info(f"Successfully auto-started runner {runner_name}")
                else:
                    logger.error(f"Failed to auto-start runner {runner_name}")
                    success = False
            else:
                logger.warning(
                    f"Runner {runner_name} has no models assigned, skipping auto-start"
                )

        if success and started_count > 0:
            logger.info(f"Successfully auto-started {started_count} runners")
        elif started_count == 0:
            logger.info("No runners were auto-started (no models assigned)")

        return success

    async def is_runner_running(self, runner_name):
        """Check if a runner process is running.

        Args:
            runner_name: Name of the runner to check.

        Returns:
            True if the runner is running, False otherwise.
        """
        if runner_name not in self.runners:
            logger.error(f"Unknown runner: {runner_name}")
            return False

        return await self.runners[runner_name].is_running()

    async def is_model_available(self, model_alias):
        """Check if a model is available (loaded and running).

        Args:
            model_alias: Alias of the model to check.

        Returns:
            True if the model is loaded and running, False otherwise.
        """
        if model_alias not in self.model_runner_map:
            logger.error(f"Unknown model: {model_alias}")
            return False

        runner_name = self.model_runner_map[model_alias]
        runner = self.runners[runner_name]

        # Check if runner is running and has the specific model loaded
        if not await self.is_runner_running(runner_name):
            return False

        return runner.is_model_loaded(model_alias)

    def get_runner_for_model(self, model_alias):
        """Get the runner process for a model.

        Args:
            model_alias: Alias of the model.

        Returns:
            The runner process, or None if not found.
        """
        if model_alias not in self.model_runner_map:
            logger.error(f"Unknown model: {model_alias}")
            return None

        runner_name = self.model_runner_map[model_alias]
        return self.runners.get(runner_name)

    def get_port_for_model(self, model_alias):
        """Get the port for a model.

        Args:
            model_alias: Alias of the model.

        Returns:
            The port, or None if not found.
        """
        runner = self.get_runner_for_model(model_alias)
        if runner is None:
            return None

        return runner.port

    def get_model_aliases(self):
        """Get all model aliases.

        Returns:
            A list of all model aliases.
        """
        return list(self.model_runner_map.keys())

    def get_runner_names(self):
        """Get all runner names.

        Returns:
            A list of all runner names.
        """
        return list(self.runners.keys())

    def get_model_runner_map(self):
        """Get the model-to-runner mapping.

        Returns:
            A dictionary mapping model aliases to runner names.
        """
        return self.model_runner_map.copy()

    async def get_current_model_for_runner(self, runner_name):
        """Get the currently loaded model for a runner.

        Args:
            runner_name: Name of the runner.

        Returns:
            The model alias currently loaded, or None if no model is loaded or runner not found.
        """
        if runner_name not in self.runners:
            logger.error(f"Unknown runner: {runner_name}")
            return None

        runner = self.runners[runner_name]
        if runner.current_model is None:
            return None

        return runner.current_model.get(
            "model_alias", os.path.basename(runner.current_model["model"])
        )

    async def switch_model(self, from_model_alias, to_model_alias):
        """Switch from one model to another on the same runner.

        Args:
            from_model_alias: Alias of the current model, used to ensure the
                switch is on the same runner.
            to_model_alias: Alias of the target model.

        Returns:
            True if the switch was successful, False otherwise.
        """
        # Check if both models exist and are on the same runner
        if from_model_alias not in self.model_runner_map:
            logger.error(f"Unknown source model: {from_model_alias}")
            return False

        if to_model_alias not in self.model_runner_map:
            logger.error(f"Unknown target model: {to_model_alias}")
            return False

        from_runner = self.model_runner_map[from_model_alias]
        to_runner = self.model_runner_map[to_model_alias]

        if from_runner != to_runner:
            logger.error(
                f"Models {from_model_alias} and {to_model_alias} are on different runners ({from_runner} vs {to_runner})"
            )
            return False

        # Use start_runner_for_model which will handle the switching
        return await self.start_runner_for_model(to_model_alias)

    async def get_runner_status(self):
        """Get the status of all runners and their loaded models.

        Returns:
            A dictionary with runner status information.
        """
        status = {}

        for runner_name, runner in self.runners.items():
            runner_status = {
                "is_running": await runner.is_running(),
                "current_model": await self.get_current_model_for_runner(runner_name),
                "available_models": [
                    model.get("model_alias", os.path.basename(model["model"]))
                    for model in runner.models
                ],
                "host": runner.host,
                "port": runner.port,
            }
            status[runner_name] = runner_status

        return status

    async def ensure_model_ready_with_retry(self, model_alias):
        """Ensure a model is ready, performing readiness checks with retry logic.

        Args:
            model_alias: Alias of the model to ensure is ready.

        Returns:
            Tuple of (success: bool, error_message: str or None)
        """
        if not self.config_manager.get_retry_on_model_loading():
            # Retry disabled, use single attempt to ensure model is ready
            if not await self.is_model_available(model_alias):
                logger.info(f"Starting runner for model {model_alias}")
                if not await self.start_runner_for_model(model_alias):
                    return False, f"Failed to start model: {model_alias}"

            # Wait a bit for model to be fully ready
            await self._wait_for_model_readiness(model_alias, max_wait_seconds=30)
            is_ready, error = await self._check_model_readiness(model_alias)
            return is_ready, error

        # Retry enabled - do pre-flight readiness checks with exponential backoff
        max_retries = self.config_manager.get_max_retries()
        base_delay = self.config_manager.get_base_delay_seconds()
        max_delay = self.config_manager.get_max_delay_seconds()

        last_error = None

        # Initial check before starting retry loop
        is_ready, last_error = await self._perform_readiness_check(model_alias)
        if is_ready:
            return True, None

        for attempt in range(max_retries):
            delay = min(base_delay * (2**attempt), max_delay)
            logger.info(
                f"Retrying model readiness check for {model_alias} (attempt {attempt + 2}/{max_retries + 1}) after {delay}s delay"
            )
            await asyncio.sleep(delay)

            is_ready, last_error = await self._perform_readiness_check(model_alias)
            if is_ready:
                return True, None

        # All retries exhausted
        logger.error(
            f"Model readiness check for {model_alias} failed after {max_retries + 1} attempts. Last error: {last_error}"
        )
        return False, last_error

    async def _perform_readiness_check(self, model_alias):
        """Helper to perform a single readiness check, including starting the model if needed."""
        try:
            # Ensure model is started
            if not await self.is_model_available(model_alias):
                logger.info(f"Starting runner for model {model_alias}")
                if not await self.start_runner_for_model(model_alias):
                    return False, f"Failed to start model: {model_alias}"

            # Wait for the newly started model to become ready
            logger.debug(
                f"Waiting for newly started model {model_alias} to become ready"
            )
            await self._wait_for_model_readiness(model_alias, max_wait_seconds=30)

            # Do a pre-flight readiness check
            is_ready, readiness_error = await self._check_model_readiness(model_alias)
            if not is_ready:
                logger.info(f"Model {model_alias} not ready: {readiness_error}")
                return False, f"Model not ready: {readiness_error}"

            # Model is ready
            logger.debug(f"Model {model_alias} is ready")
            return True, None

        except Exception as e:
            logger.error(f"Error checking model readiness for {model_alias}: {e}")
            return False, f"Readiness check error: {str(e)}"

    async def _check_model_readiness(self, model_alias):
        """Check if a model is ready to handle requests by making a simple health check.

        Args:
            model_alias: The model alias to check.

        Returns:
            Tuple of (is_ready: bool, error_message: str or None)
        """
        try:
            # Get runner for model
            runner = self.get_runner_for_model(model_alias)
            if runner is None:
                return False, HealthMessages.NO_RUNNER_AVAILABLE

            # Build health check URL
            health_url = f"http://{runner.host}:{runner.port}/health"

            # Make a quick health check request
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        health_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            return True, None
                        elif response.status == 503:
                            # Parse the error response for loading status
                            try:
                                error_data = await response.json()
                                error_message = error_data.get("error", {}).get(
                                    "message", "Unknown error"
                                )
                                if HealthStatus.LOADING in error_message.lower():
                                    return False, HealthMessages.MODEL_LOADING
                                else:
                                    return False, error_message
                            except (json.JSONDecodeError, aiohttp.ContentTypeError):
                                # Fallback if JSON parsing fails
                                response_text = await response.text()
                                if HealthStatus.LOADING in response_text.lower():
                                    return False, HealthMessages.MODEL_LOADING
                                return (
                                    False,
                                    f"HTTP {response.status}: {response_text[:100]}",
                                )
                        else:
                            response_text = (
                                await response.text()
                                if response.content_type != "application/json"
                                else str(await response.json())
                            )
                            return (
                                False,
                                f"Health check failed with status {response.status}: {response_text[:100]}",
                            )
                except asyncio.TimeoutError:
                    return False, HealthMessages.HEALTH_CHECK_TIMEOUT
                except aiohttp.ClientError as e:
                    return False, f"{HealthMessages.CONNECTION_ERROR}: {str(e)}"
        except Exception as e:
            return False, f"{HealthMessages.HEALTH_CHECK_FAILED}: {str(e)}"

    async def _wait_for_model_readiness(self, model_alias, max_wait_seconds=10):
        """Wait for a model to become ready, with a simple polling approach.

        Args:
            model_alias: The model alias to wait for.
            max_wait_seconds: Maximum time to wait in seconds.
        """
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < max_wait_seconds:
            is_ready, _ = await self._check_model_readiness(model_alias)
            if is_ready:
                logger.debug(
                    f"Model {model_alias} became ready after {asyncio.get_event_loop().time() - start_time:.1f}s"
                )
                return
            await asyncio.sleep(0.5)
        logger.warning(
            f"Model {model_alias} did not become ready within {max_wait_seconds}s"
        )

    async def forward_request(self, model_alias, endpoint, request_data):
        """Forward a request to a model's runner (assumes model is already ready).

        Args:
            model_alias: Alias of the model to forward to.
            endpoint: API endpoint to forward to (e.g., "/v1/chat/completions").
            request_data: The request data to forward.

        Returns:
            Tuple of (success: bool, response_data: dict, status_code: int)
        """
        # Get runner for model
        runner = self.get_runner_for_model(model_alias)
        if runner is None:
            return (
                False,
                {"error": {"message": f"Model not available: {model_alias}"}},
                500,
            )

        # Build URL
        url = f"http://{runner.host}:{runner.port}{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    try:
                        response_data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        # If JSON parsing fails, create error response
                        response_text = await response.text()
                        response_data = {
                            "error": {"message": f"Invalid response: {response_text}"}
                        }

                    return response.status == 200, response_data, response.status

        except aiohttp.ClientError as e:
            logger.error(f"Client error forwarding to {url}: {e}")
            return False, {"error": {"message": f"Connection error: {str(e)}"}}, 503
        except asyncio.TimeoutError:
            logger.error(f"Timeout forwarding to {url}")
            return False, {"error": {"message": "Request timeout"}}, 408
        except Exception as e:
            logger.error(f"Unexpected error forwarding to {url}: {e}")
            return False, {"error": {"message": f"Unexpected error: {str(e)}"}}, 500

    async def check_model_health(self, model_alias):
        """Check the health of a specific model by making a health check request.

        Args:
            model_alias: The alias of the model to check.

        Returns:
            Dict with health status information.
        """
        is_ready, error_message = await self._check_model_readiness(model_alias)
        return {
            "status": HealthStatus.OK if is_ready else HealthStatus.ERROR,
            "message": HealthMessages.READY if is_ready else error_message,
            "model_alias": model_alias,
        }


async def main():
    """Example usage of RunnerManager."""
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_path>")
        sys.exit(1)

    try:
        from config import ConfigManager

        config_manager = ConfigManager(sys.argv[1])
        runner_manager = RunnerManager(config_manager)

        # Start all runners
        for runner_name in runner_manager.get_runner_names():
            print(f"Starting runner {runner_name}...")
            if await runner_manager.start_runner(runner_name):
                print(f"Runner {runner_name} started successfully")
            else:
                print(f"Failed to start runner {runner_name}")

        # Wait for user input
        print("Press Enter to stop all runners...")
        await asyncio.get_event_loop().run_in_executor(None, input)

        # Stop all runners
        if await runner_manager.stop_all_runners():
            print("All runners stopped successfully")
        else:
            print("Failed to stop all runners")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
