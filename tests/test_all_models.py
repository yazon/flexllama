"""
Comprehensive test script for all models in config.json.

This script tests all models defined in the configuration file by sending requests
to the main API server, which handles routing to the appropriate runners.
The main API server should already be running before executing this script.
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
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional


# Test configuration
SYSTEM_PROMPT = "You are a helpful assistant. Please respond concisely."
USER_PROMPT = "Hello! What is 2+2? Just give me the number."
TEMPERATURE = 0.1
MAX_TOKENS = 300
TIMEOUT = 120


def setup_test_logging(debug: bool = False):
    """Set up logging configuration for test with session-based logs.

    Args:
        debug: Whether to enable debug logging.

    Returns:
        Tuple of (session_id, test_log_dir)
    """
    # Generate a unique session ID for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    session_id = f"test_all_models_{timestamp}_{session_uuid}"

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
    test_log_file = os.path.join(test_log_dir, "test_all_models.log")
    file_handler = logging.FileHandler(test_log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Test session ID: {session_id}")
    logger.info(f"Test log directory: {test_log_dir}")
    logger.info(f"Test log file: {test_log_file}")

    return session_id, test_log_dir


# Configure logging
logger = logging.getLogger(__name__)


class ModelTestResult:
    """Container for individual model test results."""

    def __init__(self, model_alias: str, runner_name: str):
        self.model_alias = model_alias
        self.runner_name = runner_name
        self.loaded_successfully = False
        self.health_check_passed = False
        self.appears_in_models_list = False
        self.chat_completion_works = False
        self.streaming_works = False
        self.response_time_ms = None
        self.streaming_time_ms = None
        self.error_messages = []
        self.sample_response = None
        self.sample_streaming_response = None
        self.model_config = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_alias": self.model_alias,
            "runner_name": self.runner_name,
            "loaded_successfully": self.loaded_successfully,
            "health_check_passed": self.health_check_passed,
            "appears_in_models_list": self.appears_in_models_list,
            "chat_completion_works": self.chat_completion_works,
            "streaming_works": self.streaming_works,
            "response_time_ms": self.response_time_ms,
            "streaming_time_ms": self.streaming_time_ms,
            "error_messages": self.error_messages,
            "sample_response": self.sample_response,
            "sample_streaming_response": self.sample_streaming_response,
            "model_file": self.model_config.get("model", "")
            if self.model_config
            else "",
        }


async def test_api_server_availability(base_url: str) -> bool:
    """Test if the main API server is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    logger.info("Main API server is available")
                    return True
                else:
                    logger.error(f"Main API server returned status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Failed to connect to main API server at {base_url}: {e}")
        return False


async def test_individual_model(
    base_url: str, model_config: Dict, max_retries: int = 5, retry_delay: int = 2
) -> ModelTestResult:
    """Test a specific model via the main API server."""
    model_alias = model_config.get(
        "model_alias", os.path.basename(model_config["model"])
    )
    runner_name = model_config.get("runner", "unknown")
    result = ModelTestResult(model_alias, runner_name)
    result.model_config = model_config

    logger.info(f"Testing model: {model_alias} (runner: {runner_name})")

    try:
        # Test 1: Health check - see if model is available
        if await test_health_endpoint(base_url, result):
            result.health_check_passed = True

        # Test 2: Check if model appears in models list
        if await test_models_endpoint(base_url, result):
            result.appears_in_models_list = True

        # Test 3: Chat completion - the main test
        if await test_chat_completion(base_url, result, max_retries, retry_delay):
            result.chat_completion_works = True
            result.loaded_successfully = True

        # Test 4: Streaming chat completion (only if regular completion works)
        if result.chat_completion_works:
            if await test_streaming_chat_completion(base_url, result):
                result.streaming_works = True

    except Exception as e:
        error_msg = f"Unexpected error testing {model_alias}: {e}"
        logger.error(error_msg)
        result.error_messages.append(error_msg)

    return result


async def test_health_endpoint(base_url: str, result: ModelTestResult) -> bool:
    """Test the health endpoint to see if model is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check if our model shows as available
                    model_health = data.get("model_health", {})
                    if result.model_alias in model_health:
                        status = model_health[result.model_alias].get("status", "error")
                        is_ready = status == "ok"
                        logger.info(
                            f"Model {result.model_alias} health status: {status}"
                        )
                        return is_ready
                    else:
                        logger.warning(
                            f"Model {result.model_alias} not found in health endpoint"
                        )
                        return False
                else:
                    error_text = await response.text()
                    result.error_messages.append(
                        f"Health check failed ({response.status}): {error_text}"
                    )
                    return False
    except Exception as e:
        result.error_messages.append(f"Health check failed: {e}")
        return False


async def test_models_endpoint(base_url: str, result: ModelTestResult) -> bool:
    """Test if model appears in models list."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    model_ids = [model["id"] for model in data.get("data", [])]
                    logger.info(f"Available models from /v1/models: {model_ids}")
                    is_listed = result.model_alias in model_ids
                    if not is_listed:
                        result.error_messages.append(
                            f"Model {result.model_alias} not found in models list"
                        )
                    return is_listed
                else:
                    error_text = await response.text()
                    result.error_messages.append(
                        f"Models endpoint failed ({response.status}): {error_text}"
                    )
                    return False
    except Exception as e:
        result.error_messages.append(f"Models endpoint failed: {e}")
        return False


async def test_chat_completion(
    base_url: str, result: ModelTestResult, max_retries: int = 5, base_delay: int = 2
) -> bool:
    """Test chat completion with the model."""
    for attempt in range(max_retries + 1):
        try:
            test_data = {
                "model": result.model_alias,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            }

            if attempt > 0:
                logger.info(
                    f"Retrying chat completion for model {result.model_alias} (attempt {attempt + 1}/{max_retries + 1})"
                )
            else:
                logger.info(
                    f"Sending chat completion request for model: {result.model_alias}"
                )

            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT),
                ) as response:
                    end_time = time.time()
                    result.response_time_ms = int((end_time - start_time) * 1000)

                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = (
                                data["choices"][0].get("message", {}).get("content", "")
                            )
                            result.sample_response = content.strip()
                            logger.info(
                                f"Model {result.model_alias} responded in {result.response_time_ms}ms: {content[:50]}..."
                            )
                            return True
                        else:
                            result.error_messages.append(
                                f"Invalid response format: {data}"
                            )
                            return False
                    elif response.status == 503:
                        # Check if this is a "Loading model" error that we can retry
                        try:
                            error_data = await response.json()
                            error_message = error_data.get("error", {}).get(
                                "message", ""
                            )
                            if (
                                "loading" in error_message.lower()
                                and attempt < max_retries
                            ):
                                delay = base_delay * (2**attempt)  # Exponential backoff
                                logger.info(
                                    f"Model {result.model_alias} is loading, waiting {delay}s before retry..."
                                )
                                await asyncio.sleep(delay)
                                continue  # Retry
                            else:
                                error_text = await response.text()
                                result.error_messages.append(
                                    f"Chat completion failed ({response.status}): {error_text}"
                                )
                                return False
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                            # If we can't parse the error, treat as non-retryable
                            error_text = await response.text()
                            result.error_messages.append(
                                f"Chat completion failed ({response.status}): {error_text}"
                            )
                            return False
                    else:
                        error_text = await response.text()
                        result.error_messages.append(
                            f"Chat completion failed ({response.status}): {error_text}"
                        )
                        return False
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Chat completion error for {result.model_alias}: {e}, retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                continue
            else:
                result.error_messages.append(f"Chat completion error: {e}")
                return False

    # If we get here, all retries failed
    result.error_messages.append(
        f"Chat completion failed after {max_retries + 1} attempts"
    )
    return False


def _extract_content_from_json(json_data: dict) -> Optional[str]:
    """Extract content from various JSON streaming formats.

    Args:
        json_data: The parsed JSON data from a streaming chunk.

    Returns:
        The content string if found, None otherwise.
    """
    # OpenAI ChatCompletions format: choices[0].delta.content or choices[0].delta.reasoning_content
    if "choices" in json_data and len(json_data["choices"]) > 0:
        choice = json_data["choices"][0]
        if "delta" in choice and isinstance(choice["delta"], dict):
            delta = choice["delta"]
            # Check for reasoning_content first (for reasoning models)
            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                return delta["reasoning_content"]
            # Then check for regular content
            if "content" in delta and delta["content"] is not None:
                return delta["content"]

        # Alternative: choices[0].message.content (for non-streaming responses mixed in)
        if "message" in choice and isinstance(choice["message"], dict):
            message = choice["message"]
            if "content" in message and message["content"] is not None:
                return message["content"]

    # Alternative format: direct content field
    if "content" in json_data and json_data["content"] is not None:
        return json_data["content"]

    # Alternative format: text field
    if "text" in json_data and json_data["text"] is not None:
        return json_data["text"]

    return None


async def parse_streaming_response(
    response, model_alias: str, timeout: int = TIMEOUT
) -> tuple[bool, list[str], int, list[str]]:
    """Parse streaming response and extract content.

    Args:
        response: The aiohttp response object
        model_alias: Name of the model being tested
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, content_parts, chunks_received, debug_info)
    """
    logger.debug(f"Starting to parse streaming response for {model_alias}")

    if response.status != 200:
        error_text = await response.text()
        logger.debug(
            f"Streaming response failed with status {response.status}: {error_text}"
        )
        return False, [], 0, [f"HTTP {response.status}: {error_text}"]

    chunks_received = 0
    content_parts = []
    incomplete_line = ""  # Handle lines split across chunks
    raw_chunks_for_debug = []  # Store first few chunks for debugging
    debug_info = []

    try:
        async for chunk in response.content.iter_chunked(8192):
            if chunk:
                chunks_received += 1
                chunk_text = chunk.decode("utf-8", errors="ignore")

                # Store raw chunks for debugging (first 3 chunks only)
                if chunks_received <= 3:
                    raw_chunks_for_debug.append(chunk_text.replace("\n", "\\n"))
                    logger.debug(
                        f"Chunk {chunks_received} for {model_alias}: {repr(chunk_text)}"
                    )

                # Combine with any incomplete line from previous chunk
                full_text = incomplete_line + chunk_text
                lines = full_text.split("\n")

                # Last line might be incomplete if chunk ended mid-line
                incomplete_line = lines[-1] if not chunk_text.endswith("\n") else ""

                # Process all complete lines (exclude the last potentially incomplete one)
                complete_lines = lines[:-1] if not chunk_text.endswith("\n") else lines

                # Parse different streaming formats
                for line in complete_lines:
                    line = line.strip()
                    if not line:  # Skip empty lines (common in SSE)
                        continue

                    logger.debug(f"Processing line for {model_alias}: {repr(line)}")

                    # Try different parsing strategies
                    parsed_content = None

                    # Strategy 1: SSE format with 'data: ' prefix
                    if line.startswith("data: "):
                        data_part = line[6:]  # Remove 'data: ' prefix
                        if data_part == "[DONE]":
                            logger.debug(f"Received [DONE] marker for {model_alias}")
                            continue
                        try:
                            json_data = json.loads(data_part)
                            logger.debug(
                                f"Parsed SSE JSON for {model_alias}: {json_data}"
                            )
                            parsed_content = _extract_content_from_json(json_data)
                            if parsed_content is not None:
                                logger.debug(
                                    f"Extracted content from SSE for {model_alias}: {repr(parsed_content)}"
                                )
                        except json.JSONDecodeError as e:
                            logger.debug(
                                f"JSON decode error in SSE format for {model_alias}: {e}, data: {data_part[:100]}"
                            )
                            continue

                    # Strategy 2: Raw JSON line
                    elif line.startswith("{"):
                        try:
                            json_data = json.loads(line)
                            logger.debug(
                                f"Parsed raw JSON for {model_alias}: {json_data}"
                            )
                            parsed_content = _extract_content_from_json(json_data)
                            if parsed_content is not None:
                                logger.debug(
                                    f"Extracted content from raw JSON for {model_alias}: {repr(parsed_content)}"
                                )
                        except json.JSONDecodeError:
                            logger.debug(
                                f"Failed to parse raw JSON for {model_alias}: {repr(line)}"
                            )
                            continue

                    # Strategy 3: Event/ID lines in SSE (skip, but don't treat as error)
                    elif line.startswith("event:") or line.startswith("id:"):
                        logger.debug(f"SSE metadata for {model_alias}: {line}")
                        continue

                    else:
                        logger.debug(
                            f"Unrecognized line format for {model_alias}: {repr(line)}"
                        )

                    # Add parsed content if found
                    if parsed_content is not None:
                        content_parts.append(parsed_content)

        # Debug logging for failed content extraction
        if chunks_received > 0 and len(content_parts) == 0:
            debug_msg = f"Received {chunks_received} chunks but extracted 0 content parts for {model_alias}"
            logger.debug(debug_msg)
            debug_info.append(debug_msg)
            if raw_chunks_for_debug:
                debug_msg = (
                    f"First raw chunks for {model_alias}: {raw_chunks_for_debug[:3]}"
                )
                logger.debug(debug_msg)
                debug_info.append(debug_msg)

        success = chunks_received > 0 and len(content_parts) > 0
        if not success:
            if chunks_received == 0:
                debug_info.append("No streaming chunks received")
            else:
                debug_info.append("No content received in streaming chunks")

        logger.debug(
            f"Streaming parsing complete for {model_alias}: {chunks_received} chunks, {len(content_parts)} content parts"
        )
        return success, content_parts, chunks_received, debug_info

    except Exception as e:
        error_msg = f"Error parsing streaming response for {model_alias}: {e}"
        logger.debug(error_msg)
        return False, [], chunks_received, [error_msg]


async def test_streaming_chat_completion(
    base_url: str, result: ModelTestResult
) -> bool:
    """Test streaming chat completion with the model."""
    try:
        test_data = {
            "model": result.model_alias,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }

        logger.info(f"Testing streaming for model: {result.model_alias}")
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            ) as response:
                (
                    success,
                    content_parts,
                    chunks_received,
                    debug_info,
                ) = await parse_streaming_response(
                    response, result.model_alias, TIMEOUT
                )

                end_time = time.time()
                result.streaming_time_ms = int((end_time - start_time) * 1000)

                if success:
                    result.sample_streaming_response = "".join(content_parts)
                    logger.info(
                        f"Model {result.model_alias} streaming completed in {result.streaming_time_ms}ms ({chunks_received} chunks, {len(content_parts)} content parts)"
                    )
                    return True
                else:
                    # Add debug info to error messages
                    for debug_msg in debug_info:
                        result.error_messages.append(debug_msg)
                    return False

    except Exception as e:
        result.error_messages.append(f"Streaming completion error: {e}")
        return False


async def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def group_models_by_runner(config: Dict) -> Dict[str, List[Dict]]:
    """Group models by their assigned runner."""
    runner_models = defaultdict(list)

    for model_config in config.get("models", []):
        runner_name = model_config.get("runner")
        if runner_name:
            runner_models[runner_name].append(model_config)
        else:
            logger.warning(
                f"Model {model_config.get('model_alias', 'unknown')} has no runner assigned"
            )

    return dict(runner_models)


def print_summary(all_results: List[ModelTestResult]):
    """Print a comprehensive summary of test results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL TEST SUMMARY")
    print("=" * 80)

    total_models = len(all_results)
    successful_models = sum(1 for r in all_results if r.loaded_successfully)
    streaming_working = sum(1 for r in all_results if r.streaming_works)

    print("\nOverall Results:")
    print(f"  Total models tested: {total_models}")
    print(f"  Successfully loaded: {successful_models}")
    print(f"  Streaming working: {streaming_working}")
    print(f"  Failed to load: {total_models - successful_models}")
    print(
        f"  Success rate: {(successful_models / total_models * 100):.1f}%"
        if total_models > 0
        else "  Success rate: 0%"
    )

    # Group by runner
    by_runner = defaultdict(list)
    for result in all_results:
        by_runner[result.runner_name].append(result)

    for runner_name, results in by_runner.items():
        print(f"\n{'-' * 60}")
        print(f"Runner: {runner_name}")
        print(f"{'-' * 60}")

        for result in results:
            status = "✅ SUCCESS" if result.loaded_successfully else "❌ FAILED"
            streaming_status = (
                " [STREAMING ✅]"
                if result.streaming_works
                else " [STREAMING ❌]"
                if result.loaded_successfully
                else ""
            )
            response_time = (
                f" ({result.response_time_ms}ms)" if result.response_time_ms else ""
            )
            print(f"  {result.model_alias}: {status}{streaming_status}{response_time}")

            if result.sample_response and len(result.sample_response) > 0:
                print(f"    Response: {result.sample_response[:200]}...")

            if result.error_messages:
                for error in result.error_messages[:3]:  # Show first 3 errors
                    print(f"    Error: {error}")

    # Performance summary for successful models
    successful_results = [
        r for r in all_results if r.loaded_successfully and r.response_time_ms
    ]
    if successful_results:
        avg_time = sum(r.response_time_ms for r in successful_results) / len(
            successful_results
        )
        min_time = min(r.response_time_ms for r in successful_results)
        max_time = max(r.response_time_ms for r in successful_results)

        print("\nPerformance Summary (successful models):")
        print(f"  Average response time: {avg_time:.0f}ms")
        print(f"  Fastest response: {min_time}ms")
        print(f"  Slowest response: {max_time}ms")

    print("\n" + "=" * 80)


async def save_detailed_results(all_results: List[ModelTestResult], test_log_dir: str):
    """Save detailed results to JSON file."""
    try:
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "test_session_dir": test_log_dir,
            "summary": {
                "total_models": len(all_results),
                "successful_models": sum(
                    1 for r in all_results if r.loaded_successfully
                ),
                "failed_models": sum(
                    1 for r in all_results if not r.loaded_successfully
                ),
                "streaming_working": sum(1 for r in all_results if r.streaming_works),
            },
            "results": [result.to_dict() for result in all_results],
        }

        # Save to test log directory
        output_path = os.path.join(test_log_dir, "model_test_results.json")
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Detailed results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


async def debug_streaming_format(
    base_url: str, model_alias: str, max_chunks: int = 10
) -> None:
    """Debug function to inspect raw streaming format from llama.cpp."""
    test_data = {
        "model": model_alias,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    logger.debug(f"=== Debugging streaming format for {model_alias} ===")
    logger.debug(f"Request URL: {base_url}/v1/chat/completions")
    logger.debug(f"Request payload: {json.dumps(test_data, indent=2)}")

    try:
        async with aiohttp.ClientSession() as session:
            logger.debug("Sending request...")
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            ) as response:
                logger.debug(f"Response status: {response.status}")
                logger.debug(f"Response headers: {dict(response.headers)}")

                if response.status != 200:
                    logger.debug(f"Error: HTTP {response.status}")
                    error_text = await response.text()
                    logger.debug(f"Error response: {error_text}")
                    return

                logger.debug("Starting to read streaming chunks...")
                chunk_count = 0
                total_bytes = 0

                async for chunk in response.content.iter_chunked(8192):
                    if chunk and chunk_count < max_chunks:
                        chunk_count += 1
                        chunk_size = len(chunk)
                        total_bytes += chunk_size

                        try:
                            chunk_text = chunk.decode("utf-8", errors="replace")
                        except Exception as decode_error:
                            logger.debug(
                                f"Chunk {chunk_count} decode error: {decode_error}"
                            )
                            logger.debug(f"Raw chunk bytes: {chunk[:100]}...")
                            continue

                        logger.debug(f"Chunk {chunk_count} (size: {chunk_size} bytes):")
                        logger.debug(f"Raw: {repr(chunk_text)}")

                        # Try to parse as lines
                        try:
                            lines = chunk_text.strip().split("\n")
                            for i, line in enumerate(lines):
                                if line.strip():
                                    logger.debug(f"  Line {i + 1}: {repr(line)}")

                                    # Try to parse as JSON
                                    line_clean = line.strip()
                                    if line_clean.startswith("data: "):
                                        data_part = line_clean[6:]
                                        if data_part != "[DONE]":
                                            try:
                                                json_data = json.loads(data_part)
                                                logger.debug(
                                                    f"    Parsed SSE JSON: {json_data}"
                                                )
                                                # Try to extract content
                                                content = _extract_content_from_json(
                                                    json_data
                                                )
                                                if content is not None:
                                                    logger.debug(
                                                        f"    Extracted content: {repr(content)}"
                                                    )
                                            except json.JSONDecodeError as json_err:
                                                logger.debug(
                                                    f"    Failed to parse SSE JSON: {json_err}"
                                                )
                                                logger.debug(
                                                    f"    Data was: {repr(data_part)}"
                                                )
                                    elif line_clean.startswith("{"):
                                        try:
                                            json_data = json.loads(line_clean)
                                            logger.debug(
                                                f"    Parsed raw JSON: {json_data}"
                                            )
                                            # Try to extract content
                                            content = _extract_content_from_json(
                                                json_data
                                            )
                                            if content is not None:
                                                logger.debug(
                                                    f"    Extracted content: {repr(content)}"
                                                )
                                        except json.JSONDecodeError as json_err:
                                            logger.debug(
                                                f"    Failed to parse raw JSON: {json_err}"
                                            )
                                            logger.debug(
                                                f"    Line was: {repr(line_clean)}"
                                            )
                                    elif line_clean.startswith(
                                        "event:"
                                    ) or line_clean.startswith("id:"):
                                        logger.debug(f"    SSE metadata: {line_clean}")
                                    else:
                                        logger.debug(
                                            f"    Unrecognized format: {repr(line_clean)}"
                                        )
                        except Exception as parse_error:
                            logger.debug(f"Error parsing chunk lines: {parse_error}")
                            logger.debug(f"Chunk text was: {repr(chunk_text)}")

                    if chunk_count >= max_chunks:
                        logger.debug(f"Reached max chunks limit ({max_chunks})")
                        break

                logger.debug("Debug summary:")
                logger.debug(f"  Total chunks processed: {chunk_count}")
                logger.debug(f"  Total bytes received: {total_bytes}")

    except aiohttp.ClientError as client_error:
        logger.debug(f"HTTP client error: {client_error}")
        logger.debug(f"Error type: {type(client_error).__name__}")
    except asyncio.TimeoutError:
        logger.debug(f"Request timed out after {TIMEOUT} seconds")
    except Exception as e:
        logger.debug(f"Unexpected debug error: {e}")
        logger.debug(f"Error type: {type(e).__name__}")
        import traceback

        logger.debug(f"Traceback: {traceback.format_exc()}")

    logger.debug("=== End debug ===")


async def main():
    """Main entry point for the comprehensive model test."""
    parser = argparse.ArgumentParser(
        description="Test all models from config.json via the main API server"
    )
    parser.add_argument(
        "--config", default="config.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="Base URL of the main API server",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for loading models (default: 5)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Base delay in seconds between retries (default: 2)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--debug-streaming",
        help="Debug streaming format for specific model (provide model alias)",
    )
    args = parser.parse_args()

    # Set up test logging - enable debug if debug-streaming is specified
    debug_mode = args.debug or bool(args.debug_streaming)
    session_id, test_log_dir = setup_test_logging(debug_mode)

    # Ensure API URL has protocol
    if not args.api_url.startswith(("http://", "https://")):
        args.api_url = f"http://{args.api_url}"

    try:
        logger.info("Starting comprehensive model testing")
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"API server URL: {args.api_url}")

        # Test API server availability first
        logger.info(f"Testing connection to main API server at {args.api_url}")
        if not await test_api_server_availability(args.api_url):
            logger.error(
                "Main API server is not available. Please start it first with:"
            )
            logger.error("  python main.py config.json")
            sys.exit(1)

        # If debug streaming mode, run debug and exit
        if args.debug_streaming:
            await debug_streaming_format(args.api_url, args.debug_streaming)
            return

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = await load_config(args.config)

        # Get all models from config
        models = config.get("models", [])
        if not models:
            logger.error("No models found in configuration")
            sys.exit(1)

        # Group models by runner for reporting
        runner_models = group_models_by_runner(config)
        logger.info(f"Found {len(models)} models across {len(runner_models)} runners:")
        for runner_name, runner_models_list in runner_models.items():
            logger.info(f"  {runner_name}: {len(runner_models_list)} models")

        # Test each model individually
        logger.info("Starting individual model tests...")
        all_results = []

        for model_config in models:
            result = await test_individual_model(
                args.api_url, model_config, args.max_retries, args.retry_delay
            )
            all_results.append(result)

            # Small delay between tests to avoid overwhelming the server
            await asyncio.sleep(1)

        # Print summary
        print_summary(all_results)

        # Save detailed results
        await save_detailed_results(all_results, test_log_dir)

        # Exit with appropriate code
        failed_models = sum(1 for r in all_results if not r.loaded_successfully)
        if failed_models > 0:
            logger.warning(f"{failed_models} models failed to load")
            sys.exit(1)
        else:
            logger.info("All models tested successfully!")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
