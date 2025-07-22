"""
Main module for FlexLLama.

This module provides the main entry point for the FlexLLama application.
"""

import os
import sys
import signal
import argparse
import logging
import asyncio
import platform
import uuid
import tempfile
from datetime import datetime
import json

from backend.config import ConfigManager
from backend.runner import RunnerManager
from backend.api import APIServer

# Global session ID for organizing logs
SESSION_ID = None
SESSION_LOG_DIR = None


def _get_writable_log_dir() -> str:
    """
    Determines and prepares a writable log directory.

    It first checks the 'FLEXLLAMA_LOG_DIR' environment variable,
    then defaults to 'logs'. If the target directory is not writable,
    it falls back to a temporary directory.

    Returns:
        The path to the writable log directory.
    """
    preferred_dir = os.getenv("FLEXLLAMA_LOG_DIR", "logs")

    try:
        os.makedirs(preferred_dir, mode=0o777, exist_ok=True)
        if os.access(preferred_dir, os.W_OK):
            return preferred_dir
    except OSError:
        # Fallback will be handled below if creation or access fails
        pass

    # If we are here, the preferred_dir is not writable.
    # Use a cross-platform approach for fallback directory
    temp_base = tempfile.gettempdir()

    # Create a unique identifier that works on all platforms
    try:
        # Try to use user ID on Unix systems
        user_id = str(os.getuid())
    except AttributeError:
        # On Windows, use username instead
        user_id = os.getenv("USERNAME", "user")

    fallback_dir = os.path.join(temp_base, f"flexllama_logs_{user_id}")
    print(
        f"Warning: Log directory '{preferred_dir}' not writable. "
        f"Falling back to '{fallback_dir}'."
    )
    os.makedirs(fallback_dir, exist_ok=True)
    return fallback_dir


def setup_logging(debug: bool = False):
    """Set up logging configuration for both console and file output.

    Args:
        debug: Whether to enable debug logging.
    """
    global SESSION_ID, SESSION_LOG_DIR

    # Generate a unique session ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    SESSION_ID = f"{timestamp}_{session_uuid}"

    # Determine and prepare log directory
    base_log_dir = _get_writable_log_dir()

    # Create session-specific log directory and store it globally
    SESSION_LOG_DIR = os.path.join(base_log_dir, SESSION_ID)
    os.makedirs(SESSION_LOG_DIR, exist_ok=True)

    # Set logging level
    log_level = logging.DEBUG if debug else logging.INFO

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler for main application logs
    main_log_file = os.path.join(SESSION_LOG_DIR, "main.log")
    file_handler = logging.FileHandler(main_log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create a file handler for errors only
    error_log_file = os.path.join(SESSION_LOG_DIR, "errors.log")
    error_handler = logging.FileHandler(error_log_file, mode="w", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Session ID: {SESSION_ID}")
    logger.info(f"Session log directory: {SESSION_LOG_DIR}")
    logger.info(
        f"Logging initialized - Console: {log_level}, Main log: {main_log_file}"
    )
    logger.info(f"Error log: {error_log_file}")


def get_session_id():
    """Get the current session ID.

    Returns:
        The current session ID or None if not initialized.
    """
    return SESSION_ID


def get_session_log_dir():
    """Get the current session log directory.

    Returns:
        The current session log directory path or None if not initialized.
    """
    return SESSION_LOG_DIR


def create_session_info(config_file_path):
    """Create a session info file with metadata about the current session.

    Args:
        config_file_path: Path to the configuration file used for this session.
    """
    if not SESSION_ID:
        return

    session_log_dir = get_session_log_dir()
    if not session_log_dir:
        return

    session_info = {
        "session_id": SESSION_ID,
        "start_time": datetime.now().isoformat(),
        "config_file": config_file_path,
        "platform": platform.system(),
        "python_version": sys.version,
        "log_files": {
            "main_log": "main.log",
            "error_log": "errors.log",
            "runner_logs": "Runner logs (<runner_name>.log) are created when runners start successfully",
        },
    }

    try:
        session_info_file = os.path.join(session_log_dir, "session_info.json")
        with open(session_info_file, "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)

        logger = logging.getLogger(__name__)
        logger.info(f"Session info saved to: {session_info_file}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create session info file: {e}")


async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="FlexLLama")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging first
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_manager = ConfigManager(args.config)

        # Create session info file
        create_session_info(args.config)

        # Create runner manager
        logger.info("Creating runner manager")
        runner_manager = RunnerManager(config_manager, get_session_log_dir())

        # Create API server
        logger.info("Creating API server")
        api_server = APIServer(config_manager, runner_manager)

        # Set up shutdown handler based on platform
        loop = asyncio.get_running_loop()

        # Create a shutdown event
        shutdown_event = asyncio.Event()

        # Define shutdown handler
        async def shutdown():
            logger.info("Shutting down...")
            await api_server.stop()
            await runner_manager.stop_all_runners()
            shutdown_event.set()

        # Set up platform-specific signal handling
        if platform.system() == "Windows":
            # Windows doesn't support asyncio signal handlers
            logger.info("Running on Windows, using Windows-specific signal handling")

            # Define a sync signal handler that schedules the async shutdown
            def win_signal_handler(sig, frame):
                logger.info(f"Received signal {sig}")
                if not shutdown_event.is_set():
                    asyncio.create_task(shutdown())

            # Register the sync handler
            signal.signal(signal.SIGINT, win_signal_handler)
            signal.signal(signal.SIGTERM, win_signal_handler)

        else:
            logger.info("Running on Unix-like system, using asyncio signal handlers")

            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

        # Start API server
        success = await api_server.start()
        if not success:
            logger.error("Failed to start API server")
            sys.exit(1)

        logger.info(f"API server running at {api_server.get_url()}")

        # Auto-start default runners if enabled
        logger.info("Checking for auto-start configuration...")
        await runner_manager.auto_start_default_runners()

        logger.info("Press Ctrl+C to stop")

        # Keep main thread alive until shutdown is requested
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


def main_entry():
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
