"""
Configuration parser for FlexLLama.

This module handles loading, parsing, and validating configuration files for the FlexLLama.
It ensures that all required fields are present and that values are of the correct type.
"""

import json
import os
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for FlexLLama configuration."""

    def __init__(self, config_path: str):
        """Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self):
        """Load the configuration from the file.

        Returns:
            The loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        logger.info(f"Loading configuration from {self.config_path}")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            return config

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse configuration file: {e}")
            raise

    def _validate_config(self):
        """Validate the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        required_fields = ["models"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

        # Validate models
        if (
            not isinstance(self.config["models"], list)
            or len(self.config["models"]) == 0
        ):
            raise ValueError("Configuration must contain at least one model")

        # Collect all runner names
        runner_names = set()
        for key, value in self.config.items():
            if key not in [
                "models",
                "host",
                "port",
                "api",
                "auto_start_runners",
                "retry_config",
            ] and isinstance(value, dict):
                runner_names.add(key)

        # Validate models and their runner references
        for i, model in enumerate(self.config["models"]):
            self._validate_model_config(model, i, runner_names)

        # Validate API configuration
        self._validate_api_config()

        # Validate retry configuration
        self._validate_retry_config()

        # Validate auto_start_runners if present
        if "auto_start_runners" in self.config:
            if not isinstance(self.config["auto_start_runners"], bool):
                raise ValueError("auto_start_runners must be a boolean")

        # Validate runner configurations
        used_ports = set()
        for runner_name in runner_names:
            self._validate_runner_config(
                self.config[runner_name], runner_name, used_ports
            )

        logger.info("Configuration validation successful")

    def _validate_api_config(self):
        """Validate API configuration."""
        # Check if new API section exists
        if "api" in self.config:
            api_config = self.config["api"]
            if not isinstance(api_config, dict):
                raise ValueError("API configuration must be a dictionary")

            if "host" not in api_config:
                raise ValueError("API configuration missing required field: host")
            if not isinstance(api_config["host"], str):
                raise ValueError("API host must be a string")

            if "port" not in api_config:
                raise ValueError("API configuration missing required field: port")
            if not isinstance(api_config["port"], int):
                raise ValueError("API port must be an integer")
        else:
            raise ValueError("API configuration missing required field: api")

    def _validate_retry_config(self):
        """Validate retry configuration with sensible defaults."""
        if "retry_config" not in self.config:
            # Set default retry configuration
            self.config["retry_config"] = {
                "max_retries": 5,
                "base_delay_seconds": 2,
                "max_delay_seconds": 30,
                "retry_on_model_loading": True,
            }
            return

        retry_config = self.config["retry_config"]
        if not isinstance(retry_config, dict):
            raise ValueError("retry_config must be a dictionary")

        # Validate individual fields with defaults
        if "max_retries" not in retry_config:
            retry_config["max_retries"] = 5
        elif (
            not isinstance(retry_config["max_retries"], int)
            or retry_config["max_retries"] < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")

        if "base_delay_seconds" not in retry_config:
            retry_config["base_delay_seconds"] = 2
        elif (
            not isinstance(retry_config["base_delay_seconds"], (int, float))
            or retry_config["base_delay_seconds"] < 0
        ):
            raise ValueError("base_delay_seconds must be a non-negative number")

        if "max_delay_seconds" not in retry_config:
            retry_config["max_delay_seconds"] = 30
        elif (
            not isinstance(retry_config["max_delay_seconds"], (int, float))
            or retry_config["max_delay_seconds"] < 0
        ):
            raise ValueError("max_delay_seconds must be a non-negative number")

        if "retry_on_model_loading" not in retry_config:
            retry_config["retry_on_model_loading"] = True
        elif not isinstance(retry_config["retry_on_model_loading"], bool):
            raise ValueError("retry_on_model_loading must be a boolean")

        # Validate logical constraints
        if retry_config["max_delay_seconds"] < retry_config["base_delay_seconds"]:
            raise ValueError(
                "max_delay_seconds must be greater than or equal to base_delay_seconds"
            )

    def _validate_model_config(self, model, index: int, runner_names: set):
        """Validate a model configuration.

        Args:
            model: The model configuration to validate.
            index: The index of the model in the configuration.
            runner_names: Set of valid runner names.

        Raises:
            ValueError: If the model configuration is invalid.
        """
        # Check required fields
        required_fields = ["model", "runner"]
        for field in required_fields:
            if field not in model:
                raise ValueError(f"Model {index}: Missing required field: {field}")

        # Validate model path
        if not isinstance(model["model"], str):
            raise ValueError(f"Model {index}: Model path must be a string")

        # Validate runner reference
        if not isinstance(model["runner"], str):
            raise ValueError(f"Model {index}: Runner reference must be a string")

        if model["runner"] not in runner_names:
            raise ValueError(
                f"Model {index}: Referenced runner '{model['runner']}' not found in configuration"
            )

        # Validate optional fields
        if "model_alias" in model and not isinstance(model["model_alias"], str):
            raise ValueError(f"Model {index}: Model alias must be a string")

        int_fields = [
            "n_ctx",
            "n_batch",
            "n_threads",
            "main_gpu",
            "n_gpu_layers",
            "type_k",
            "type_v",
        ]
        for field in int_fields:
            if field in model and not isinstance(model[field], int):
                raise ValueError(f"Model {index}: {field} must be an integer")

        bool_fields = [
            "offload_kqv",
            "flash_attn",
            "use_mlock",
            "jinja",
            "embedding",
            "reranking",
        ]
        for field in bool_fields:
            if field in model and not isinstance(model[field], bool):
                raise ValueError(f"Model {index}: {field} must be a boolean")

        if "chat_format" in model and not isinstance(model["chat_format"], str):
            raise ValueError(f"Model {index}: chat_format must be a string")

        if "tensor_split" in model:
            if not isinstance(model["tensor_split"], list):
                raise ValueError(f"Model {index}: tensor_split must be a list")
            for value in model["tensor_split"]:
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Model {index}: tensor_split values must be numbers"
                    )

    def _validate_runner_config(self, runner, runner_name: str, used_ports: set):
        """Validate the runner configuration.

        Args:
            runner: The runner configuration to validate.
            runner_name: The name of the runner.
            used_ports: Set of already used ports for conflict detection.

        Raises:
            ValueError: If the runner configuration is invalid.
        """
        # Check type
        if "type" not in runner:
            raise ValueError(f"Runner {runner_name}: Missing required field: type")

        if not isinstance(runner["type"], str):
            raise ValueError(f"Runner {runner_name}: Type must be a string")

        # Check path
        if "path" not in runner:
            # Default to type if path not specified
            runner["path"] = runner["type"]
        elif not isinstance(runner["path"], str):
            raise ValueError(f"Runner {runner_name}: Path must be a string")

        # Check host (optional, will default if not provided)
        if "host" in runner and not isinstance(runner["host"], str):
            raise ValueError(f"Runner {runner_name}: Host must be a string")

        # Check port (optional, will auto-assign if not provided)
        if "port" in runner:
            if not isinstance(runner["port"], int):
                raise ValueError(f"Runner {runner_name}: Port must be an integer")

            # Check for port conflicts
            port = runner["port"]
            if port in used_ports:
                raise ValueError(f"Runner {runner_name}: Port {port} already in use")
            used_ports.add(port)

        # Check extra_args
        if "extra_args" not in runner:
            runner["extra_args"] = []
        elif not isinstance(runner["extra_args"], list):
            raise ValueError(f"Runner {runner_name}: extra_args must be a list")
        else:
            for arg in runner["extra_args"]:
                if not isinstance(arg, str):
                    raise ValueError(
                        f"Runner {runner_name}: extra_args must contain only strings"
                    )

    def get_config(self):
        """Get the full configuration.

        Returns:
            The full configuration.
        """
        return self.config

    def get_model_config(self, model_alias=None):
        """Get a model configuration by alias.

        Args:
            model_alias: The alias of the model to get. If None, returns the first model.

        Returns:
            The model configuration.

        Raises:
            ValueError: If the model alias is not found.
        """
        if model_alias is None:
            return self.config["models"][0]

        for model in self.config["models"]:
            if model.get("model_alias") == model_alias:
                return model

        raise ValueError(f"Model alias not found: {model_alias}")

    def get_runner_config(self, runner_name: str):
        """Get a runner configuration by name.

        Args:
            runner_name: The name of the runner.

        Returns:
            The runner configuration.

        Raises:
            ValueError: If the runner name is not found.
        """
        if runner_name not in self.config:
            raise ValueError(f"Runner not found: {runner_name}")

        return self.config[runner_name]

    def get_runner_for_model(self, model_alias=None):
        """Get the runner configuration for a model.

        Args:
            model_alias: The alias of the model. If None, uses the first model.

        Returns:
            Tuple of (runner_name, runner_config).

        Raises:
            ValueError: If the model alias or referenced runner is not found.
        """
        model = self.get_model_config(model_alias)
        runner_name = model["runner"]
        runner_config = self.get_runner_config(runner_name)
        return runner_name, runner_config

    def get_host(self):
        """Get the API server host.

        This is an alias for get_api_host().

        Returns:
            The API server host.
        """
        return self.get_api_host()

    def get_port(self):
        """Get the API server port.

        This is an alias for get_api_port().

        Returns:
            The API server port.
        """
        return self.get_api_port()

    def get_model_aliases(self):
        """Get all model aliases.

        Returns:
            A list of all model aliases.
        """
        return [
            model.get("model_alias", os.path.basename(model["model"]))
            for model in self.config["models"]
        ]

    def get_runner_names(self):
        """Get all runner names.

        Returns:
            A list of all runner names.
        """
        return [
            key
            for key in self.config.keys()
            if key
            not in [
                "models",
                "host",
                "port",
                "api",
                "auto_start_runners",
                "retry_config",
            ]
            and isinstance(self.config[key], dict)
        ]

    def get_model_runner_map(self):
        """Get a mapping of model aliases to runner names.

        Returns:
            A dictionary mapping model aliases to runner names.
        """
        model_runner_map = {}
        for model in self.config["models"]:
            alias = model.get("model_alias", os.path.basename(model["model"]))
            model_runner_map[alias] = model["runner"]
        return model_runner_map

    def get_auto_start_runners(self):
        """Get the auto-start runners setting.

        Returns:
            True if runners should be auto-started, False otherwise. Defaults to True.
        """
        return self.config.get("auto_start_runners", True)

    def get_api_host(self):
        """Get the API server host.

        Returns:
            The API server host.
        """
        if "api" in self.config:
            return self.config["api"]["host"]
        else:
            raise ValueError("API configuration missing required field: api")

    def get_api_port(self):
        """Get the API server port.

        Returns:
            The API server port.
        """
        if "api" in self.config:
            return self.config["api"]["port"]
        else:
            raise ValueError("API configuration missing required field: api")

    def get_runner_host(self, runner_name: str):
        """Get the host for a specific runner.

        Args:
            runner_name: The name of the runner.

        Returns:
            The host for the runner, or API host if not specified.

        Raises:
            ValueError: If the runner name is not found.
        """
        if runner_name not in self.config:
            raise ValueError(f"Runner not found: {runner_name}")

        runner_config = self.config[runner_name]
        return runner_config.get("host", self.get_api_host())

    def get_runner_port(self, runner_name: str):
        """Get the port for a specific runner.

        Args:
            runner_name: The name of the runner.

        Returns:
            The configured port for the runner.

        Raises:
            ValueError: If the runner name is not found or the port is not configured.
        """
        if runner_name not in self.config:
            raise ValueError(f"Runner not found: {runner_name}")

        runner_config = self.config[runner_name]
        if "port" in runner_config:
            return runner_config["port"]
        else:
            raise ValueError(f"Runner {runner_name}: Port not configured")

    def get_retry_config(self):
        """Get the retry configuration.

        Returns:
            The retry configuration dictionary.
        """
        return self.config.get(
            "retry_config",
            {
                "max_retries": 5,
                "base_delay_seconds": 2,
                "max_delay_seconds": 30,
                "retry_on_model_loading": True,
            },
        )

    def get_max_retries(self):
        """Get the maximum number of retries for model loading.

        Returns:
            The maximum number of retries.
        """
        return self.get_retry_config().get("max_retries", 5)

    def get_base_delay_seconds(self):
        """Get the base delay in seconds between retries.

        Returns:
            The base delay in seconds.
        """
        return self.get_retry_config().get("base_delay_seconds", 2)

    def get_max_delay_seconds(self):
        """Get the maximum delay in seconds between retries.

        Returns:
            The maximum delay in seconds.
        """
        return self.get_retry_config().get("max_delay_seconds", 30)

    def get_retry_on_model_loading(self):
        """Get whether to retry on model loading errors.

        Returns:
            True if retries should be performed on model loading errors, False otherwise.
        """
        return self.get_retry_config().get("retry_on_model_loading", True)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_path>")
        sys.exit(1)

    try:
        config_manager = ConfigManager(sys.argv[1])
        print("Configuration loaded successfully")
        print(f"Models: {config_manager.get_model_aliases()}")
        print(f"Runners: {config_manager.get_runner_names()}")
        print(f"Model-Runner Map: {config_manager.get_model_runner_map()}")
        print(f"Host: {config_manager.get_host()}")
        print(f"Port: {config_manager.get_port()}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
