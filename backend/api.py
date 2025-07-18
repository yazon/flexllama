"""
API server for FlexLLama.

This module implements OpenAI-compatible API endpoints that route requests to the
appropriate llama.cpp server instance based on the requested model.
"""

import os
import time
import logging
import json
import asyncio
import aiohttp
from aiohttp import web
from .runner import HealthStatus, HealthMessages

# Get logger for this module
logger = logging.getLogger(__name__)


class APIServer:
    """OpenAI-compatible API server with async support."""

    def __init__(self, config_manager, runner_manager):
        """Initialize the API server.

        Args:
            config_manager: The configuration manager.
            runner_manager: The runner manager.
        """
        self.config_manager = config_manager
        self.runner_manager = runner_manager

        # Get host and port for API server from API-specific configuration
        self.host = config_manager.get_api_host()
        self.port = config_manager.get_api_port()
        self.health_endpoint = config_manager.get_health_endpoint()

        # Set a larger client_max_size to handle image uploads (10MB should be enough)
        self.app = web.Application(client_max_size=10 * 1024 * 1024)
        self._setup_routes()
        self.runner = None
        self.site = None

    def _setup_routes(self):
        """Set up API routes."""
        self.app.add_routes(
            [
                web.get("/v1/models", self.handle_models),
                web.get(self.health_endpoint, self.handle_health),
                web.post("/v1/chat/completions", self.handle_chat_completions),
                web.post("/v1/completions", self.handle_completions),
                web.post("/v1/embeddings", self.handle_embeddings),
                web.post("/v1/rerank", self.handle_rerank),
                web.options("/{tail:.*}", self.handle_options),
                # Runner control routes
                web.post("/v1/runners/{runner_name}/start", self.handle_runner_start),
                web.post("/v1/runners/{runner_name}/stop", self.handle_runner_stop),
                web.post(
                    "/v1/runners/{runner_name}/restart", self.handle_runner_restart
                ),
                web.get("/v1/runners/status", self.handle_runners_status),
                # Dashboard routes
                web.get("/", self.handle_dashboard),
                web.get("/dashboard", self.handle_dashboard),
                web.get("/health", self.handle_health),
                web.static("/frontend", "frontend", show_index=True),
            ]
        )

    async def start(self):
        """Start the API server asynchronously.

        Returns:
            An awaitable that resolves to True if the server was started successfully, False otherwise.
        """
        try:
            logger.info(f"Starting API server on {self.host}:{self.port}")

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            logger.info("API server started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False

    async def stop(self):
        """Stop the API server asynchronously.

        Returns:
            An awaitable that resolves to True if the server was stopped successfully, False otherwise.
        """
        try:
            if self.runner:
                logger.info("Stopping API server")
                await self.runner.cleanup()
                logger.info("API server stopped successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to stop API server: {e}")
            return False

    def get_url(self):
        """Get the URL of the API server.

        Returns:
            The URL of the API server.
        """
        return f"http://{self.host}:{self.port}"

    async def handle_options(self, request):
        """Handle OPTIONS requests for CORS.

        Args:
            request: The request.

        Returns:
            The response.
        """
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return web.Response(status=200, headers=headers)

    async def handle_dashboard(self, request):
        """Handle GET / and /dashboard requests to serve the dashboard.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            dashboard_path = os.path.join("frontend", "index.html")
            if os.path.exists(dashboard_path):
                with open(dashboard_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Inject the health endpoint into the dashboard
                content = content.replace("__HEALTH_ENDPOINT__", self.health_endpoint)
                return web.Response(text=content, content_type="text/html")
            else:
                return web.Response(
                    text="Dashboard not found. Please ensure the frontend folder exists with index.html.",
                    status=404,
                )
        except Exception as e:
            logger.error(f"Error serving dashboard: {e}")
            return web.Response(text=f"Error loading dashboard: {str(e)}", status=500)

    async def handle_models(self, request):
        """Handle GET /v1/models requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        models = []
        for alias in self.runner_manager.get_model_aliases():
            models.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "user",
                }
            )

        response = {"object": "list", "data": models}

        return web.json_response(response)

    async def handle_health(self, request):
        """Handle GET /health requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        # Check which runners are active
        active_runners = {}
        for runner_name in self.runner_manager.get_runner_names():
            active_runners[runner_name] = await self.runner_manager.is_runner_running(
                runner_name
            )

        # Check actual model status from llama.cpp health endpoints
        model_health = {}
        for model_alias in self.runner_manager.get_model_aliases():
            try:
                # Get runner for this model
                runner = self.runner_manager.get_runner_for_model(model_alias)
                if runner is None:
                    model_health[model_alias] = {
                        "status": HealthStatus.ERROR,
                        "message": HealthMessages.NO_RUNNER_AVAILABLE,
                    }
                    continue

                # Check if runner process is even running
                if not await self.runner_manager.is_runner_running(runner.runner_name):
                    model_health[model_alias] = {
                        "status": HealthStatus.NOT_RUNNING,
                        "message": HealthMessages.RUNNER_NOT_RUNNING,
                    }
                    continue

                # Check if this specific model is loaded in the runner
                if not runner.is_model_loaded(model_alias):
                    model_health[model_alias] = {
                        "status": HealthStatus.NOT_LOADED,
                        "message": HealthMessages.MODEL_NOT_LOADED,
                    }
                    continue

                # Call llama.cpp health endpoint
                health_url = f"http://{runner.host}:{runner.port}/health"

                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(
                            health_url, timeout=aiohttp.ClientTimeout(total=3)
                        ) as response:
                            if response.status == 200:
                                model_health[model_alias] = {
                                    "status": HealthStatus.OK,
                                    "message": HealthMessages.READY,
                                }
                            elif response.status == 503:
                                # Parse the error response
                                try:
                                    error_data = await response.json()
                                    error_message = error_data.get("error", {}).get(
                                        "message", "Unknown error"
                                    )
                                    if "loading" in error_message.lower():
                                        model_health[model_alias] = {
                                            "status": HealthStatus.LOADING,
                                            "message": error_message,
                                        }
                                    else:
                                        model_health[model_alias] = {
                                            "status": HealthStatus.ERROR,
                                            "message": error_message,
                                        }
                                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                                    # Fallback if JSON parsing fails
                                    error_text = await response.text()
                                    if "loading" in error_text.lower():
                                        model_health[model_alias] = {
                                            "status": HealthStatus.LOADING,
                                            "message": HealthMessages.MODEL_LOADING,
                                        }
                                    else:
                                        model_health[model_alias] = {
                                            "status": HealthStatus.ERROR,
                                            "message": f"HTTP {response.status}: {error_text[:100]}",
                                        }
                                except UnicodeDecodeError:
                                    model_health[model_alias] = {
                                        "status": HealthStatus.ERROR,
                                        "message": f"HTTP {response.status}",
                                    }
                            else:
                                # Other HTTP errors
                                try:
                                    error_text = await response.text()
                                    model_health[model_alias] = {
                                        "status": HealthStatus.ERROR,
                                        "message": f"HTTP {response.status}: {error_text[:100]}",
                                    }
                                except UnicodeDecodeError:
                                    model_health[model_alias] = {
                                        "status": HealthStatus.ERROR,
                                        "message": f"HTTP {response.status}",
                                    }

                    except asyncio.TimeoutError:
                        model_health[model_alias] = {
                            "status": HealthStatus.ERROR,
                            "message": HealthMessages.HEALTH_CHECK_TIMEOUT,
                        }
                    except aiohttp.ClientError as e:
                        model_health[model_alias] = {
                            "status": HealthStatus.ERROR,
                            "message": f"{HealthMessages.CONNECTION_ERROR}: {str(e)}",
                        }

            except Exception as e:
                model_health[model_alias] = {
                    "status": HealthStatus.ERROR,
                    "message": f"{HealthMessages.HEALTH_CHECK_FAILED}: {str(e)}",
                }

        # Get current model assignments for each runner
        runner_models = {}
        runner_info = {}
        for runner_name in self.runner_manager.get_runner_names():
            current_model = await self.runner_manager.get_current_model_for_runner(
                runner_name
            )
            runner_models[runner_name] = current_model

            # Get runner info including host and port
            runner = self.runner_manager.runners.get(runner_name)
            if runner:
                runner_info[runner_name] = {
                    "host": runner.host,
                    "port": runner.port,
                    "current_model": current_model,
                    "is_active": active_runners.get(runner_name, False),
                }

        response = {
            "status": "ok",
            "active_runners": active_runners,
            "runner_current_models": runner_models,
            "runner_info": runner_info,
            "model_health": model_health,
        }

        return web.json_response(response)

    async def handle_runner_start(self, request):
        """Handle POST /v1/runners/{runner_name}/start requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        runner_name = request.match_info.get("runner_name")
        if not runner_name:
            return web.json_response(
                {"success": False, "error": {"message": "Runner name not provided"}},
                status=400,
            )

        # Check if runner exists
        if runner_name not in self.runner_manager.get_runner_names():
            return web.json_response(
                {
                    "success": False,
                    "error": {"message": f"Unknown runner: {runner_name}"},
                },
                status=404,
            )

        try:
            # Start the runner
            success = await self.runner_manager.start_runner(runner_name)

            if success:
                return web.json_response(
                    {
                        "success": True,
                        "message": f"Runner {runner_name} started successfully",
                        "runner_name": runner_name,
                        "action": "start",
                        "status": "starting",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            else:
                return web.json_response(
                    {
                        "success": False,
                        "error": {
                            "message": f"Failed to start runner: {runner_name}",
                            "type": "runner_error",
                            "runner_name": runner_name,
                        },
                    },
                    status=500,
                )

        except Exception as e:
            logger.error(f"Error starting runner {runner_name}: {e}")
            return web.json_response(
                {
                    "success": False,
                    "error": {
                        "message": f"Failed to start runner: {str(e)}",
                        "type": "runner_error",
                        "runner_name": runner_name,
                    },
                },
                status=500,
            )

    async def handle_runner_stop(self, request):
        """Handle POST /v1/runners/{runner_name}/stop requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        runner_name = request.match_info.get("runner_name")
        if not runner_name:
            return web.json_response(
                {"success": False, "error": {"message": "Runner name not provided"}},
                status=400,
            )

        # Check if runner exists
        if runner_name not in self.runner_manager.get_runner_names():
            return web.json_response(
                {
                    "success": False,
                    "error": {"message": f"Unknown runner: {runner_name}"},
                },
                status=404,
            )

        try:
            # Stop the runner
            success = await self.runner_manager.stop_runner(runner_name)

            if success:
                return web.json_response(
                    {
                        "success": True,
                        "message": f"Runner {runner_name} stopped successfully",
                        "runner_name": runner_name,
                        "action": "stop",
                        "status": "stopping",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            else:
                return web.json_response(
                    {
                        "success": False,
                        "error": {
                            "message": f"Failed to stop runner: {runner_name}",
                            "type": "runner_error",
                            "runner_name": runner_name,
                        },
                    },
                    status=500,
                )

        except Exception as e:
            logger.error(f"Error stopping runner {runner_name}: {e}")
            return web.json_response(
                {
                    "success": False,
                    "error": {
                        "message": f"Failed to stop runner: {str(e)}",
                        "type": "runner_error",
                        "runner_name": runner_name,
                    },
                },
                status=500,
            )

    async def handle_runner_restart(self, request):
        """Handle POST /v1/runners/{runner_name}/restart requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        runner_name = request.match_info.get("runner_name")
        if not runner_name:
            return web.json_response(
                {"success": False, "error": {"message": "Runner name not provided"}},
                status=400,
            )

        # Check if runner exists
        if runner_name not in self.runner_manager.get_runner_names():
            return web.json_response(
                {
                    "success": False,
                    "error": {"message": f"Unknown runner: {runner_name}"},
                },
                status=404,
            )

        try:
            # Restart the runner (stop then start)
            logger.info(f"Restarting runner {runner_name}")

            # Stop first
            stop_success = await self.runner_manager.stop_runner(runner_name)
            if not stop_success:
                logger.warning(
                    f"Failed to stop runner {runner_name} during restart, continuing anyway"
                )

            # Wait a bit for cleanup
            await asyncio.sleep(1)

            # Start again
            start_success = await self.runner_manager.start_runner(runner_name)

            if start_success:
                return web.json_response(
                    {
                        "success": True,
                        "message": f"Runner {runner_name} restarted successfully",
                        "runner_name": runner_name,
                        "action": "restart",
                        "status": "restarting",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            else:
                return web.json_response(
                    {
                        "success": False,
                        "error": {
                            "message": f"Failed to restart runner: {runner_name}",
                            "type": "runner_error",
                            "runner_name": runner_name,
                        },
                    },
                    status=500,
                )

        except Exception as e:
            logger.error(f"Error restarting runner {runner_name}: {e}")
            return web.json_response(
                {
                    "success": False,
                    "error": {
                        "message": f"Failed to restart runner: {str(e)}",
                        "type": "runner_error",
                        "runner_name": runner_name,
                    },
                },
                status=500,
            )

    async def handle_runners_status(self, request):
        """Handle GET /v1/runners/status requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            status = await self.runner_manager.get_runner_status()
            return web.json_response(
                {
                    "success": True,
                    "runners": status,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )

        except Exception as e:
            logger.error(f"Error getting runner status: {e}")
            return web.json_response(
                {
                    "success": False,
                    "error": {"message": f"Failed to get runner status: {str(e)}"},
                },
                status=500,
            )

    async def handle_chat_completions(self, request):
        """Handle POST /v1/chat/completions requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        model_alias = self._extract_model_alias(data)

        if model_alias is None:
            return web.json_response(
                {"error": {"message": "Model not specified"}}, status=400
            )

        try:
            self.config_manager.get_model_config(model_alias)
        except ValueError:
            return web.json_response(
                {"error": {"message": f"Model not found: {model_alias}"}}, status=404
            )

        # Forward request with unified pre-flight approach
        return await self._forward_request_unified(
            request, model_alias, "/v1/chat/completions", data
        )

    async def handle_completions(self, request):
        """Handle POST /v1/completions requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        model_alias = self._extract_model_alias(data)

        if model_alias is None:
            return web.json_response(
                {"error": {"message": "Model not specified"}}, status=400
            )

        try:
            self.config_manager.get_model_config(model_alias)
        except ValueError:
            return web.json_response(
                {"error": {"message": f"Model not found: {model_alias}"}}, status=404
            )

        # Forward request with unified pre-flight approach
        return await self._forward_request_unified(
            request, model_alias, "/v1/completions", data
        )

    async def handle_embeddings(self, request):
        """Handle POST /v1/embeddings requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        model_alias = self._extract_model_alias(data)

        if model_alias is None:
            return web.json_response(
                {"error": {"message": "Model not specified"}}, status=400
            )

        try:
            self.config_manager.get_model_config(model_alias)
        except ValueError:
            return web.json_response(
                {"error": {"message": f"Model not found: {model_alias}"}}, status=404
            )

        # Forward request with unified pre-flight approach
        return await self._forward_request_unified(
            request, model_alias, "/v1/embeddings", data
        )

    async def handle_rerank(self, request):
        """Handle POST /v1/rerank requests.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        model_alias = self._extract_model_alias(data)

        if model_alias is None:
            return web.json_response(
                {"error": {"message": "Model not specified"}}, status=400
            )

        try:
            self.config_manager.get_model_config(model_alias)
        except ValueError:
            return web.json_response(
                {"error": {"message": f"Model not found: {model_alias}"}}, status=404
            )

        # Forward request with unified pre-flight approach
        return await self._forward_request_unified(
            request, model_alias, "/v1/rerank", data
        )

    def _extract_model_alias(self, data):
        """Extract the model alias from the request data.

        If 'model' is not specified in the request data, this function defaults
        to the first available model alias from the runner manager.

        Args:
            data: The request data.

        Returns:
            The model alias, or None if not found.
        """
        if "model" in data:
            return data["model"]

        # Default to first model
        try:
            return self.runner_manager.get_model_aliases()[0]
        except IndexError:
            return None

    async def _forward_request_unified(self, request, model_alias, endpoint, data):
        """Unified request forwarding with pre-flight readiness check for both streaming and non-streaming.

        Args:
            request: The original request object.
            model_alias: The model alias.
            endpoint: The API endpoint.
            data: The request data.

        Returns:
            The response.
        """
        # Step 1: Pre-flight readiness check with retry
        logger.debug(f"Ensuring model {model_alias} is ready for request to {endpoint}")
        (
            is_ready,
            error_message,
        ) = await self.runner_manager.ensure_model_ready_with_retry(model_alias)

        if not is_ready:
            logger.error(f"Model {model_alias} not ready: {error_message}")
            return web.json_response(
                {
                    "error": {
                        "message": f"Model not ready: {error_message}",
                        "type": "model_not_ready",
                    }
                },
                status=503,
            )

        # Step 2: Forward request (streaming or non-streaming)
        is_streaming = data.get("stream", False)

        if is_streaming:
            logger.debug(
                f"Forwarding streaming request to model {model_alias} at {endpoint}"
            )
            return await self._forward_streaming_request(
                request, model_alias, endpoint, data
            )
        else:
            logger.debug(
                f"Forwarding non-streaming request to model {model_alias} at {endpoint}"
            )
            (
                success,
                response_data,
                status_code,
            ) = await self.runner_manager.forward_request(model_alias, endpoint, data)
            return web.json_response(response_data, status=status_code)

    async def _forward_streaming_request(self, request, model_alias, endpoint, data):
        """Forward a streaming request to the appropriate runner.

        Args:
            request: The original request object.
            model_alias: The model alias.
            endpoint: The API endpoint.
            data: The request data.

        Returns:
            The streaming response.
        """
        # Get runner for model
        runner = self.runner_manager.get_runner_for_model(model_alias)
        if runner is None:
            return web.json_response(
                {"error": {"message": f"Model not available: {model_alias}"}},
                status=500,
            )

        # Double-check model is still loaded (defensive programming)
        if not runner.is_model_loaded(model_alias):
            logger.warning(
                f"Model {model_alias} not loaded during streaming request, attempting to ensure readiness"
            )
            (
                is_ready,
                error_message,
            ) = await self.runner_manager.ensure_model_ready_with_retry(model_alias)
            if not is_ready:
                return web.json_response(
                    {
                        "error": {
                            "message": f"Model not ready for streaming: {error_message}",
                            "type": "model_not_ready",
                        }
                    },
                    status=503,
                )

        # Build URL using runner's host and port
        url = f"http://{runner.host}:{runner.port}{endpoint}"

        # Forward streaming request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    # Check if this is an error response
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            return web.json_response(error_data, status=response.status)
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                            error_text = await response.text()
                            return web.json_response(
                                {"error": {"message": error_text}},
                                status=response.status,
                            )

                    # Create a streaming response with the same headers
                    headers = {
                        "Content-Type": response.headers.get(
                            "Content-Type", "text/event-stream"
                        ),
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }

                    # Create streaming response
                    streaming_response = web.StreamResponse(
                        status=response.status, headers=headers
                    )
                    await streaming_response.prepare(request)

                    # Stream the response data
                    async for chunk in response.content.iter_chunked(8192):
                        await streaming_response.write(chunk)

                    await streaming_response.write_eof()
                    return streaming_response

        except aiohttp.ClientError as e:
            logger.error(
                f"Error forwarding streaming request to {url}: message='{str(e)}', url='{url}'"
            )
            return web.json_response(
                {"error": {"message": f"Error forwarding streaming request: {str(e)}"}},
                status=503,
            )
        except Exception as e:
            logger.error(f"Error forwarding streaming request to {url}: {e}")
            return web.json_response(
                {"error": {"message": f"Error forwarding streaming request: {str(e)}"}},
                status=500,
            )


async def main():
    """Example usage of APIServer."""
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_path>")
        sys.exit(1)

    try:
        from config import ConfigManager
        from runner import RunnerManager

        config_manager = ConfigManager(sys.argv[1])
        runner_manager = RunnerManager(config_manager)
        api_server = APIServer(config_manager, runner_manager)

        # Start API server
        if await api_server.start():
            print(f"API server running at {api_server.get_url()}")
            print("Press Ctrl+C to stop")

            # Keep main thread alive
            while True:
                await asyncio.sleep(1)
        else:
            print("Failed to start API server")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Stop all runners
        if "runner_manager" in locals():
            await runner_manager.stop_all_runners()

        # Stop API server
        if "api_server" in locals():
            await api_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
