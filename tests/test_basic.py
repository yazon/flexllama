"""
Basic test script for FlexLLama.

This script tests the FlexLLama with basic functionality,
focusing on API endpoints and basic request handling.
"""

import os
import sys
import json
import time
import logging
import argparse
import asyncio
import aiohttp
import uuid
from datetime import datetime


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
    session_id = f"test_basic_{timestamp}_{session_uuid}"

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
    test_log_file = os.path.join(test_log_dir, "test_basic.log")
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


def create_test_session_info(session_id, test_log_dir):
    """Create a session info file for this test run.

    Args:
        session_id: The test session ID.
        test_log_dir: The test log directory.
    """
    session_info = {
        "test_type": "basic",
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "log_files": {"main_log": "test_basic.log", "error_log": "errors.log"},
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


# Configure logging
logger = logging.getLogger(__name__)


async def test_models_endpoint(base_url):
    """Test the /v1/models endpoint.

    Args:
        base_url: The base URL of the API server.

    Returns:
        A list of model aliases.
    """
    logger.info("Testing /v1/models endpoint")

    url = f"{base_url}/v1/models"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to get models: {response.status} {await response.text()}"
                    )
                    return []

                data = await response.json()
                models = [model["id"] for model in data["data"]]

                logger.info(f"Available models: {models}")
                return models
        except Exception as e:
            logger.error(f"Error testing models endpoint: {e}")
            return []


async def test_health_endpoint(base_url):
    """Test the /health endpoint.

    Args:
        base_url: The base URL of the API server.

    Returns:
        The health status data.
    """
    logger.info("Testing /health endpoint")

    url = f"{base_url}/health"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to get health: {response.status} {await response.text()}"
                    )
                    return None

                data = await response.json()
                logger.info(f"Health status: {json.dumps(data, indent=2)}")
                return data
        except Exception as e:
            logger.error(f"Error testing health endpoint: {e}")
            return None


async def test_chat_completions(base_url, model):
    """Test the /v1/chat/completions endpoint.

    Args:
        base_url: The base URL of the API server.
        model: The model alias to use.

    Returns:
        True if the test passed, False otherwise.
    """
    logger.info(f"Testing /v1/chat/completions endpoint with model: {model}")

    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
        ],
        "temperature": 0.7,
        "max_tokens": 50,
    }

    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()
            async with session.post(
                url, json=data, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                end_time = time.time()
                response_time = int((end_time - start_time) * 1000)

                if response.status != 200:
                    logger.error(
                        f"Failed to get chat completion: {response.status} {await response.text()}"
                    )
                    return False

                result = await response.json()
                if "choices" not in result or len(result["choices"]) == 0:
                    logger.error(f"Invalid response: {result}")
                    return False

                content = result["choices"][0].get("message", {}).get("content", "")
                logger.info(
                    f"Response from {model} in {response_time}ms: {content[:50]}..."
                )
                return True
        except Exception as e:
            logger.error(f"Error testing chat completions: {e}")
            return False


async def test_streaming_chat_completions(base_url, model):
    """Test the /v1/chat/completions endpoint with streaming.

    Args:
        base_url: The base URL of the API server.
        model: The model alias to use.

    Returns:
        True if the test passed, False otherwise.
    """
    logger.info(f"Testing streaming /v1/chat/completions endpoint with model: {model}")

    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 5."},
        ],
        "temperature": 0.7,
        "max_tokens": 30,
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()
            async with session.post(
                url, json=data, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to get streaming chat completion: {response.status} {await response.text()}"
                    )
                    return False

                chunks_received = 0
                content_parts = []

                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        chunks_received += 1
                        chunk_text = chunk.decode("utf-8", errors="ignore")

                        # Parse SSE format
                        for line in chunk_text.strip().split("\n"):
                            if line.startswith("data: "):
                                data_part = line[6:]  # Remove 'data: ' prefix
                                if data_part == "[DONE]":
                                    continue
                                try:
                                    json_data = json.loads(data_part)
                                    if (
                                        "choices" in json_data
                                        and len(json_data["choices"]) > 0
                                    ):
                                        delta = json_data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content_parts.append(delta["content"])
                                except json.JSONDecodeError:
                                    continue

                end_time = time.time()
                response_time = int((end_time - start_time) * 1000)

                full_content = "".join(content_parts)
                logger.info(
                    f"Streaming response from {model} in {response_time}ms ({chunks_received} chunks): {full_content[:50]}..."
                )
                return chunks_received > 0

        except Exception as e:
            logger.error(f"Error testing streaming chat completions: {e}")
            return False


async def test_concurrent_requests(base_url, health_data):
    """Test concurrent requests to different runners.

    Args:
        base_url: The base URL of the API server.
        health_data: The health status data containing runner information.

    Returns:
        True if the test passed, False otherwise.
    """
    # Extract active runners from health data
    active_runners = []
    if health_data and "runner_info" in health_data:
        for runner_id, runner_info in health_data["runner_info"].items():
            if runner_info.get("is_active") is True and "current_model" in runner_info:
                active_runners.append(
                    {
                        "runner_id": runner_id,
                        "model_alias": runner_info["current_model"],
                    }
                )

    if len(active_runners) < 2:
        logger.warning(
            f"Need at least 2 active runners to test concurrent requests, found {len(active_runners)}"
        )
        return False

    # Use only the first 2 active runners for concurrent testing
    test_runners = active_runners[:2]
    logger.info("Testing concurrent requests across 2 active runners:")
    for i, runner in enumerate(test_runners, 1):
        logger.info(
            f"  Runner{i}: {runner['runner_id']} -> model: {runner['model_alias']}"
        )

    # Create tasks for concurrent requests - one request per runner
    tasks = []
    for runner in test_runners:
        tasks.append(test_chat_completions(base_url, runner["model_alias"]))

    # Run tasks concurrently
    try:
        results = await asyncio.gather(*tasks)

        # Check results
        if all(results):
            logger.info("All concurrent requests across runners completed successfully")
            return True
        else:
            logger.error("Some concurrent requests to runners failed")
            return False
    except Exception as e:
        logger.error(f"Error in concurrent requests across runners: {e}")
        return False


async def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Basic test for FlexLLama")
    parser.add_argument(
        "--url", default="http://localhost:8080", help="Base URL of the API server"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up test logging
    session_id, test_log_dir = setup_test_logging(args.debug)

    try:
        # Create session info
        create_test_session_info(session_id, test_log_dir)

        logger.info("Starting basic tests for FlexLLama")
        logger.info(f"Testing API server at: {args.url}")

        # Test models endpoint
        logger.info("=" * 60)
        logger.info("Testing /v1/models endpoint")
        logger.info("=" * 60)
        models = await test_models_endpoint(args.url)
        if not models:
            logger.error("Failed to get models")
            sys.exit(1)

        # Test health endpoint
        logger.info("=" * 60)
        logger.info("Testing /health endpoint")
        logger.info("=" * 60)
        health = await test_health_endpoint(args.url)
        if health is None:
            logger.error("Failed to get health")
            sys.exit(1)

        # Test individual models
        logger.info("=" * 60)
        logger.info("Testing single model (first available)")
        logger.info("=" * 60)

        # Use only the first model for testing
        test_model = models[0]
        logger.info(f"Testing model: {test_model}")
        if not await test_chat_completions(args.url, test_model):
            logger.error(f"Failed to test chat completions for model {test_model}")
            sys.exit(1)

        # Test streaming for the selected model
        logger.info("=" * 60)
        logger.info("Testing streaming functionality")
        logger.info("=" * 60)
        if not await test_streaming_chat_completions(args.url, test_model):
            logger.warning(f"Streaming test failed for model {test_model}")

        # Test concurrent requests across different runners (if multiple runners are active)
        logger.info("=" * 60)
        logger.info("Testing concurrent requests across runners")
        logger.info("=" * 60)

        # Check if we have at least 2 active runners from health data
        active_runner_count = 0
        if health and "runner_info" in health:
            for runner_info in health["runner_info"].values():
                if runner_info.get("is_active"):
                    active_runner_count += 1

        if active_runner_count >= 2:
            if not await test_concurrent_requests(args.url, health):
                logger.error("Failed to test concurrent requests across runners")
                sys.exit(1)
        else:
            logger.info(
                f"Skipping concurrent requests test (only {active_runner_count} active runners found, need 2)"
            )

        logger.info("=" * 60)
        logger.info("All basic tests passed!")
        logger.info(f"Test logs saved to: {test_log_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
