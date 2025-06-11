#!/usr/bin/env python3
"""
Test script for model switching functionality.

This script demonstrates the dynamic model reloading capability of the FlexLLama.
"""

import asyncio
import json
import os
import sys
import logging
import uuid
from datetime import datetime

from backend.config import ConfigManager
from backend.runner import RunnerManager


def setup_test_logging(debug: bool = False):
    """Set up logging configuration for test with session-based logs.

    Args:
        debug: Whether to enable debug logging.

    Returns:
        tuple: (session_id, test_log_dir)
    """
    # Generate a unique session ID for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    session_id = f"test_model_switching_{timestamp}_{session_uuid}"

    # Create test-specific log directory
    test_log_dir = os.path.join("tests", "logs", session_id)
    os.makedirs(test_log_dir, exist_ok=True)

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

    # Create file handler
    test_log_file = os.path.join(test_log_dir, "test_model_switching.log")
    file_handler = logging.FileHandler(test_log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create error log handler
    error_log_file = os.path.join(test_log_dir, "errors.log")
    error_handler = logging.FileHandler(error_log_file, mode="w", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Test session ID: {session_id}")
    logger.info(f"Test log directory: {test_log_dir}")
    logger.info(f"Test log file: {test_log_file}")
    logger.info(f"Error log file: {error_log_file}")

    return session_id, test_log_dir


def create_test_session_info(session_id, test_log_dir, config_path):
    """Create a session info file for this test run.

    Args:
        session_id: The test session ID.
        test_log_dir: The test log directory.
        config_path: Path to the configuration file used.
    """
    session_info = {
        "test_type": "model_switching",
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "config_file": config_path,
        "log_files": {
            "main_log": "test_model_switching.log",
            "error_log": "errors.log",
        },
    }

    try:
        session_info_file = os.path.join(test_log_dir, "session_info.json")
        with open(session_info_file, "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)

        logger = logging.getLogger(__name__)
        logger.info(f"Test session info saved to: {session_info_file}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create test session info file: {e}")


async def test_model_switching(config_path, debug=False):
    """Test the model switching functionality.

    Args:
        config_path: Path to the configuration file.
        debug: Whether to enable debug logging.
    """
    # Set up test logging
    session_id, test_log_dir = setup_test_logging(debug)
    logger = logging.getLogger(__name__)

    # Track test results
    test_results = {
        "Test 1: Start with first model": {"status": "PENDING", "details": ""},
        "Test 2: Switch to second model": {"status": "PENDING", "details": ""},
        "Test 3: Switch back to first model": {"status": "PENDING", "details": ""},
        "Test 4: Check model configuration and switchability": {
            "status": "PENDING",
            "details": "",
        },
        "Test 5: Runner health and status": {"status": "PENDING", "details": ""},
        "Cleanup": {"status": "PENDING", "details": ""},
    }

    try:
        # Create session info
        create_test_session_info(session_id, test_log_dir, config_path)

        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config_manager = ConfigManager(config_path)

        # Create runner manager with test log directory
        logger.info("Creating runner manager")
        runner_manager = RunnerManager(config_manager, test_log_dir)

        # Get status before starting
        logger.info("Initial runner status:")
        status = await runner_manager.get_runner_status()
        logger.info(json.dumps(status, indent=2))

        # Find a runner with multiple models for testing
        test_runner = None
        test_models = []

        for runner_name, runner in runner_manager.runners.items():
            if len(runner.models) >= 2:
                test_runner = runner_name
                test_models = [
                    model.get("model_alias", os.path.basename(model["model"]))
                    for model in runner.models[:2]  # Take first 2 models
                ]
                break

        if test_runner is None:
            logger.error("No runner found with multiple models for testing")
            logger.info("Available runners and their models:")
            for runner_name, runner in runner_manager.runners.items():
                models = [
                    model.get("model_alias", os.path.basename(model["model"]))
                    for model in runner.models
                ]
                logger.info(f"  {runner_name}: {models}")
            # Mark all tests as skipped
            for test_name in test_results:
                test_results[test_name] = {
                    "status": "SKIPPED",
                    "details": "No runner with multiple models found",
                }
            return False

        logger.info(
            f"Testing model switching on runner '{test_runner}' with models: {test_models}"
        )

        # Test 1: Start with first model
        logger.info(f"\n=== Test 1: Starting runner with model '{test_models[0]}' ===")
        try:
            success = await runner_manager.start_runner_for_model(test_models[0])
            if success:
                logger.info(
                    f"✓ Successfully started runner with model '{test_models[0]}'"
                )
                current_model = await runner_manager.get_current_model_for_runner(
                    test_runner
                )
                logger.info(f"Current model: {current_model}")
                test_results["Test 1: Start with first model"] = {
                    "status": "PASSED",
                    "details": f"Started with model '{test_models[0]}', current: {current_model}",
                }
            else:
                logger.error(f"✗ Failed to start runner with model '{test_models[0]}'")
                test_results["Test 1: Start with first model"] = {
                    "status": "FAILED",
                    "details": f"Failed to start runner with model '{test_models[0]}'",
                }
        except Exception as e:
            logger.error(f"✗ Test 1 failed with exception: {e}")
            test_results["Test 1: Start with first model"] = {
                "status": "FAILED",
                "details": f"Exception: {str(e)}",
            }

        # Wait a bit for model to fully load
        await asyncio.sleep(5)

        # Test 2: Switch to second model
        logger.info(f"\n=== Test 2: Switching to model '{test_models[1]}' ===")
        try:
            success = await runner_manager.start_runner_for_model(test_models[1])
            if success:
                logger.info(f"✓ Successfully switched to model '{test_models[1]}'")
                current_model = await runner_manager.get_current_model_for_runner(
                    test_runner
                )
                logger.info(f"Current model: {current_model}")
                test_results["Test 2: Switch to second model"] = {
                    "status": "PASSED",
                    "details": f"Switched to model '{test_models[1]}', current: {current_model}",
                }
            else:
                logger.error(f"✗ Failed to switch to model '{test_models[1]}'")
                test_results["Test 2: Switch to second model"] = {
                    "status": "FAILED",
                    "details": f"Failed to switch to model '{test_models[1]}'",
                }
        except Exception as e:
            logger.error(f"✗ Test 2 failed with exception: {e}")
            test_results["Test 2: Switch to second model"] = {
                "status": "FAILED",
                "details": f"Exception: {str(e)}",
            }

        # Wait a bit for model to fully load
        await asyncio.sleep(5)

        # Test 3: Switch back to first model
        logger.info(f"\n=== Test 3: Switching back to model '{test_models[0]}' ===")
        try:
            success = await runner_manager.start_runner_for_model(test_models[0])
            if success:
                logger.info(f"✓ Successfully switched back to model '{test_models[0]}'")
                current_model = await runner_manager.get_current_model_for_runner(
                    test_runner
                )
                logger.info(f"Current model: {current_model}")
                test_results["Test 3: Switch back to first model"] = {
                    "status": "PASSED",
                    "details": f"Switched back to model '{test_models[0]}', current: {current_model}",
                }
            else:
                logger.error(f"✗ Failed to switch back to model '{test_models[0]}'")
                test_results["Test 3: Switch back to first model"] = {
                    "status": "FAILED",
                    "details": f"Failed to switch back to model '{test_models[0]}'",
                }
        except Exception as e:
            logger.error(f"✗ Test 3 failed with exception: {e}")
            test_results["Test 3: Switch back to first model"] = {
                "status": "FAILED",
                "details": f"Exception: {str(e)}",
            }

        # Wait a bit longer for model switching to fully complete
        await asyncio.sleep(5)

        # Test 4: Check model configuration and switchability
        logger.info("\n=== Test 4: Checking model configuration and switchability ===")
        try:
            # Check current state
            current_model = await runner_manager.get_current_model_for_runner(
                test_runner
            )
            logger.info(f"Currently loaded model: {current_model}")

            # Check that models are properly configured in the runner
            runner = runner_manager.runners[test_runner]
            configured_models = [
                model.get("model_alias", os.path.basename(model["model"]))
                for model in runner.models
            ]
            logger.info(f"Configured models in runner: {configured_models}")

            # Verify both test models are configured
            missing_models = [
                model for model in test_models if model not in configured_models
            ]
            extra_config_check = []

            for model_alias in test_models:
                # Check if model is in the model-runner mapping
                is_mapped = model_alias in runner_manager.model_runner_map
                # Check if runner knows about this model
                model_config = runner.get_model_by_alias(model_alias)
                is_configured = model_config is not None

                is_loaded = model_alias == current_model
                status_emoji = "✓" if (is_mapped and is_configured) else "✗"
                loaded_indicator = " (CURRENTLY LOADED)" if is_loaded else ""

                logger.info(
                    f"{status_emoji} Model '{model_alias}' - Mapped: {is_mapped}, Configured: {is_configured}{loaded_indicator}"
                )
                extra_config_check.append(
                    f"{model_alias}: mapped={is_mapped}, configured={is_configured}"
                )

            if len(missing_models) == 0:
                test_results["Test 4: Check model configuration and switchability"] = {
                    "status": "PASSED",
                    "details": f"All models properly configured: {', '.join(extra_config_check)}",
                }
            else:
                test_results["Test 4: Check model configuration and switchability"] = {
                    "status": "FAILED",
                    "details": f"Missing models: {missing_models}, Config status: {', '.join(extra_config_check)}",
                }
        except Exception as e:
            logger.error(f"✗ Test 4 failed with exception: {e}")
            test_results["Test 4: Check model configuration and switchability"] = {
                "status": "FAILED",
                "details": f"Exception: {str(e)}",
            }

        # Test 5: Check runner status
        logger.info("\n=== Test 5: Runner health and status ===")
        try:
            is_running = await runner_manager.is_runner_running(test_runner)
            logger.info(f"Runner '{test_runner}' is running: {is_running}")

            health_details = []
            # Check health of current model
            current_model = await runner_manager.get_current_model_for_runner(
                test_runner
            )
            if current_model:
                health_result = await runner_manager.check_model_health(current_model)
                logger.info(f"Health check for '{current_model}': {health_result}")
                health_details.append(f"{current_model}: {health_result}")

            if is_running:
                test_results["Test 5: Runner health and status"] = {
                    "status": "PASSED",
                    "details": f"Runner running: {is_running}, Health: {', '.join(health_details) if health_details else 'No health data'}",
                }
            else:
                test_results["Test 5: Runner health and status"] = {
                    "status": "FAILED",
                    "details": f"Runner not running, Health: {', '.join(health_details) if health_details else 'No health data'}",
                }
        except Exception as e:
            logger.error(f"✗ Test 5 failed with exception: {e}")
            test_results["Test 5: Runner health and status"] = {
                "status": "FAILED",
                "details": f"Exception: {str(e)}",
            }

        # Final status
        logger.info("\n=== Final Status ===")
        final_status = await runner_manager.get_runner_status()
        logger.info(json.dumps(final_status, indent=2))

        # Cleanup
        logger.info("\n=== Cleanup ===")
        try:
            success = await runner_manager.stop_all_runners()
            if success:
                logger.info("✓ All runners stopped successfully")
                test_results["Cleanup"] = {
                    "status": "PASSED",
                    "details": "All runners stopped successfully",
                }
            else:
                logger.error("✗ Failed to stop all runners")
                test_results["Cleanup"] = {
                    "status": "FAILED",
                    "details": "Failed to stop all runners",
                }
        except Exception as e:
            logger.error(f"✗ Cleanup failed with exception: {e}")
            test_results["Cleanup"] = {
                "status": "FAILED",
                "details": f"Exception: {str(e)}",
            }

        # Print test summary
        print_test_summary(test_results, logger)

        logger.info(f"\nTest logs saved to: {test_log_dir}")

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        # Mark all pending tests as failed
        for test_name, result in test_results.items():
            if result["status"] == "PENDING":
                test_results[test_name] = {
                    "status": "FAILED",
                    "details": f"Test suite exception: {str(e)}",
                }
        print_test_summary(test_results, logger)
        return False

    # Determine overall success
    failed_tests = [
        name for name, result in test_results.items() if result["status"] == "FAILED"
    ]
    return len(failed_tests) == 0


def print_test_summary(test_results, logger):
    """Print a formatted test summary.

    Args:
        test_results: Dictionary of test results.
        logger: Logger instance.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 60}")

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for test_name, result in test_results.items():
        status = result["status"]
        details = result["details"]

        if status == "PASSED":
            status_emoji = "✅"
            passed_count += 1
        elif status == "FAILED":
            status_emoji = "❌"
            failed_count += 1
        elif status == "SKIPPED":
            status_emoji = "⏭️"
            skipped_count += 1
        else:
            status_emoji = "❓"

        logger.info(f"{status_emoji} {test_name}: {status}")
        if details:
            logger.info(f"   Details: {details}")

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"RESULTS: {passed_count} PASSED, {failed_count} FAILED, {skipped_count} SKIPPED"
    )
    logger.info(f"{'=' * 60}")

    # Also print to console for immediate visibility
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    for test_name, result in test_results.items():
        status = result["status"]
        details = result["details"]

        if status == "PASSED":
            status_emoji = "✅"
        elif status == "FAILED":
            status_emoji = "❌"
        elif status == "SKIPPED":
            status_emoji = "⏭️"
        else:
            status_emoji = "❓"

        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")

    print(f"\n{'=' * 60}")
    print(
        f"RESULTS: {passed_count} PASSED, {failed_count} FAILED, {skipped_count} SKIPPED"
    )
    print(f"{'=' * 60}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test model switching functionality")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    success = await test_model_switching(args.config, args.debug)

    if success:
        print("\nModel switching test completed successfully!")
        sys.exit(0)
    else:
        print("\nModel switching test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
