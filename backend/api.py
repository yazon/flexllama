"""
API server for FlexLLama.

This module implements OpenAI-compatible API endpoints that route requests to the
appropriate llama.cpp server instance based on the requested model. Supported
endpoints include Chat Completions, Completions, Embeddings, Rerank, the
OpenAI Responses API, and the OpenAI Audio API (transcriptions and speech).
"""

import time
import logging
import json
import asyncio
import aiohttp
from aiohttp import web
from pathlib import Path
import importlib.resources
from .runner import HealthStatus, HealthMessages
from .gpu_metrics import GPUMetricsCollector, RateLimiter, build_runner_gpu_associations
from .throughput_metrics import ThroughputMetricsCollector

# Get logger for this module
logger = logging.getLogger(__name__)


class APIServer:
    """OpenAI-compatible API server with async support.

    Proxies requests to llama.cpp server instances for the following endpoints:
    /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/rerank,
    /v1/responses (OpenAI Responses API), and the OpenAI Audio API
    (/v1/audio/transcriptions and /v1/audio/speech).
    """

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

        # Get frontend directory path - try package resources first, fallback to relative path
        self.frontend_path = self._get_frontend_path()

        # Load CORS configuration before building the application so the middleware
        # can be registered with the right allowlist.
        self.cors_allow_origins = config_manager.get_cors_allow_origins()

        # Set a larger client_max_size to handle image uploads (10MB should be enough)
        self.app = web.Application(
            client_max_size=10 * 1024 * 1024,
            middlewares=[self._build_cors_middleware()],
        )
        # CORS responsibilities are split: the middleware answers preflight
        # (OPTIONS) directly, and on_response_prepare attaches the
        # Access-Control-Allow-Origin header to every other response. The
        # signal is used instead of mutating the response after the handler
        # returns because StreamResponse.prepare() flushes headers immediately,
        # so post-handler mutation never reaches streaming clients.
        self.app.on_response_prepare.append(self._apply_cors_headers)

        # Set keepalive timeout to 60 minutes for long-running requests
        self.keepalive_timeout = 3600  # 60 minutes

        gpu_metrics_config = config_manager.get_gpu_metrics_config()
        self.gpu_metrics_collector = GPUMetricsCollector(gpu_metrics_config)
        self.gpu_metrics_limiter = RateLimiter(
            gpu_metrics_config.get("rate_limit_requests_per_minute", 120)
        )
        self.gpu_metrics_endpoint = "/v1/metrics/gpus"

        # Token throughput metrics. Event-driven (no poll loop / background
        # task): the collector self-disables via config.enabled and returns a
        # disabled snapshot, so the route is always registered and absence of
        # the metrics.throughput block keeps today's behavior unchanged.
        throughput_config = config_manager.get_throughput_metrics_config()
        self.throughput_metrics_collector = ThroughputMetricsCollector(
            throughput_config
        )
        self.throughput_metrics_limiter = RateLimiter(
            throughput_config.get("rate_limit_requests_per_minute", 120)
        )
        self.throughput_metrics_endpoint = "/v1/metrics/throughput"

        # MCP proxy configuration. The endpoint is only registered when
        # mcp.enabled is true, so when the config block is absent or disabled
        # the route simply does not exist and behavior is identical to before.
        self.mcp_config = config_manager.get_mcp_config()
        self.mcp_enabled = bool(self.mcp_config.get("enabled", False))
        self.mcp_endpoint = self.mcp_config.get("endpoint", "/v1/mcp")
        self.mcp_upstream_path = self.mcp_config.get("upstream_path", "/mcp")
        self.mcp_limiter = RateLimiter(
            self.mcp_config.get("rate_limit_requests_per_minute", 120)
        )

        self._http_session: aiohttp.ClientSession | None = None

        self._setup_routes()
        self.runner = None
        self.site = None

    def _get_frontend_path(self):
        """Get the path to the frontend directory.

        Returns:
            Path object pointing to the frontend directory.
        """
        try:
            # Get frontend path from package resources (Python 3.12+)
            frontend_ref = importlib.resources.files("frontend")
            # Convert Traversable to actual path
            if hasattr(frontend_ref, "__fspath__"):
                return Path(frontend_ref)
            else:
                # For MultiplexedPath and other Traversable objects
                # We need to use str() to convert properly
                return Path(str(frontend_ref))
        except (ImportError, FileNotFoundError, ModuleNotFoundError, TypeError):
            # Fallback to relative path (development mode)
            logger.warning("Frontend package not found, falling back to relative path")
            return Path("frontend")

    def _setup_routes(self):
        """Set up API routes."""
        # Only add static route if frontend path exists
        routes = [
            web.get("/v1/models", self.handle_models),
            web.post("/v1/chat/completions", self.handle_chat_completions),
            web.post("/v1/completions", self.handle_completions),
            web.post("/v1/embeddings", self.handle_embeddings),
            web.post("/v1/rerank", self.handle_rerank),
            web.post("/v1/responses", self.handle_responses),
            web.post("/v1/audio/transcriptions", self.handle_audio_transcriptions),
            web.post("/v1/audio/speech", self.handle_audio_speech),
            # Runner control routes
            web.post("/v1/runners/{runner_name}/start", self.handle_runner_start),
            web.post("/v1/runners/{runner_name}/stop", self.handle_runner_stop),
            web.post("/v1/runners/{runner_name}/restart", self.handle_runner_restart),
            web.get("/v1/runners/status", self.handle_runners_status),
            web.get(self.gpu_metrics_endpoint, self.handle_gpu_metrics),
            web.get(self.throughput_metrics_endpoint, self.handle_throughput_metrics),
            # Dashboard routes
            web.get("/", self.handle_dashboard),
            web.get("/dashboard", self.handle_dashboard),
            web.get(self.health_endpoint, self.handle_health),
        ]

        # Add static route if frontend path exists
        if self.frontend_path.exists() and self.frontend_path.is_dir():
            routes.append(
                web.static("/frontend", str(self.frontend_path), show_index=True)
            )
            logger.info(f"Serving static files from: {self.frontend_path}")
        else:
            logger.warning(f"Frontend directory not found at: {self.frontend_path}")

        # Register the MCP proxy endpoint only when explicitly enabled. When
        # disabled (or the config block is absent) the route never exists, so
        # the server's behavior is byte-identical to the pre-MCP version.
        if self.mcp_enabled:
            routes.append(web.post(self.mcp_endpoint, self.handle_mcp))
            logger.info(
                f"MCP proxy enabled: {self.mcp_endpoint} -> runner{self.mcp_upstream_path}"
            )

        self.app.add_routes(routes)

    async def start(self):
        """Start the API server asynchronously.

        Returns:
            An awaitable that resolves to True if the server was started successfully, False otherwise.
        """
        try:
            logger.info(f"Starting API server on {self.host}:{self.port}")
            logger.info(f"Keepalive timeout set to {self.keepalive_timeout} seconds")

            self.runner = web.AppRunner(
                self.app, keepalive_timeout=self.keepalive_timeout
            )
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            self._http_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=20)
            )

            # Start auto-unload watchdog
            await self.runner_manager.start_auto_unload_watchdog()

            # Start GPU metrics collector
            await self.gpu_metrics_collector.start()

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
            # Stop auto-unload watchdog first
            await self.runner_manager.stop_auto_unload_watchdog()

            # Stop GPU metrics collector
            await self.gpu_metrics_collector.stop()

            if self._http_session and not self._http_session.closed:
                await self._http_session.close()

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

    async def _notify_request_start(self, model_alias: str) -> None:
        """Notify that a request has started for a model.

        Args:
            model_alias: The alias of the model handling the request.
        """
        runner = self.runner_manager.get_runner_for_model(model_alias)
        if runner:
            async with runner._request_lock:
                runner.active_requests += 1
                runner.last_activity_ts = time.time()
                logger.debug(
                    f"Request started for model {model_alias}, active requests: {runner.active_requests}"
                )

    async def _notify_request_end(self, model_alias: str) -> None:
        """Notify that a request has ended for a model.

        Args:
            model_alias: The alias of the model that handled the request.
        """
        runner = self.runner_manager.get_runner_for_model(model_alias)
        if runner:
            async with runner._request_lock:
                runner.active_requests = max(0, runner.active_requests - 1)
                runner.last_activity_ts = time.time()
                logger.debug(
                    f"Request ended for model {model_alias}, active requests: {runner.active_requests}"
                )

    def _build_cors_middleware(self):
        """Build an aiohttp middleware that answers CORS preflight requests.

        Non-preflight responses get their CORS headers from
        _apply_cors_headers via the on_response_prepare signal; this
        middleware only short-circuits OPTIONS so preflight never reaches
        the regular handlers.

        Access-Control-Allow-Credentials is intentionally NOT set: combining
        it with "*" is invalid, and enabling it by default would let any
        allowed origin read authenticated responses. Operators who need
        credentialed CORS can extend this later with an explicit config flag.
        """

        @web.middleware
        async def cors_middleware(request, handler):
            if request.method == "OPTIONS":
                origin = self._resolve_cors_origin(request)
                headers = {}
                if origin:
                    headers["Access-Control-Allow-Origin"] = origin
                    if origin != "*":
                        headers["Vary"] = "Origin"
                    # Echo the headers the client asked to send so Authorization
                    # (used by OpenAI-compatible clients) passes preflight.
                    requested_headers = request.headers.get(
                        "Access-Control-Request-Headers", "Content-Type, Authorization"
                    )
                    headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                    headers["Access-Control-Allow-Headers"] = requested_headers
                    headers["Access-Control-Max-Age"] = "600"
                return web.Response(status=200, headers=headers)

            # Non-preflight: actual CORS header is set by on_response_prepare
            # so it also reaches the client for streaming responses.
            return await handler(request)

        return cors_middleware

    def _resolve_cors_origin(self, request):
        """Resolve the Access-Control-Allow-Origin value for a request, or None."""
        allow_origins = self.cors_allow_origins
        if not allow_origins:
            return None
        if allow_origins == ["*"]:
            return "*"
        request_origin = request.headers.get("Origin")
        if request_origin and request_origin in allow_origins:
            return request_origin
        return None

    async def _apply_cors_headers(self, request, response):
        """Attach CORS headers just before response headers are flushed.

        Runs on the on_response_prepare signal so that streaming endpoints
        (which call StreamResponse.prepare() and flush headers immediately)
        also emit Access-Control-Allow-Origin.
        """
        origin = self._resolve_cors_origin(request)
        if not origin:
            return
        response.headers["Access-Control-Allow-Origin"] = origin
        if origin != "*":
            existing_vary = response.headers.get("Vary")
            response.headers["Vary"] = (
                f"{existing_vary}, Origin" if existing_vary else "Origin"
            )

    async def handle_dashboard(self, request):
        """Handle GET / and /dashboard requests to serve the dashboard.

        Args:
            request: The request.

        Returns:
            The response.
        """
        try:
            dashboard_path = self.frontend_path / "index.html"
            if dashboard_path.exists():
                with open(dashboard_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Inject the health endpoint into the dashboard
                content = content.replace("__HEALTH_ENDPOINT__", self.health_endpoint)
                content = content.replace(
                    "__GPU_METRICS_ENDPOINT__", self.gpu_metrics_endpoint
                )
                content = content.replace(
                    "__THROUGHPUT_METRICS_ENDPOINT__",
                    self.throughput_metrics_endpoint,
                )
                return web.Response(text=content, content_type="text/html")
            else:
                return web.Response(
                    text=f"Dashboard not found at {dashboard_path}. Please ensure the frontend folder exists with index.html.",
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
                # Calculate auto-unload countdown
                auto_unload_countdown = None
                if (
                    runner.auto_unload_timeout_seconds > 0
                    and runner.current_model is not None
                    and runner.last_activity_ts is not None
                    and runner.active_requests == 0
                ):
                    elapsed = time.time() - runner.last_activity_ts
                    remaining = runner.auto_unload_timeout_seconds - elapsed
                    auto_unload_countdown = max(0, int(remaining))

                runner_info[runner_name] = {
                    "host": runner.host,
                    "port": runner.port,
                    "current_model": current_model,
                    "is_active": active_runners.get(runner_name, False),
                    "auto_unload_timeout_seconds": runner.auto_unload_timeout_seconds,
                    "auto_unload_countdown_seconds": auto_unload_countdown,
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

    async def handle_gpu_metrics(self, request):
        """Handle GET /v1/metrics/gpus requests.

        Returns normalized GPU telemetry with bounded history and advisory
        runner-to-GPU associations.  The endpoint is rate-limited per remote
        IP to prevent abuse.

        Args:
            request: The request.

        Returns:
            The response.
        """
        remote = request.remote or "unknown"
        allowed, retry_after = self.gpu_metrics_limiter.check(remote)
        if not allowed:
            return web.json_response(
                {
                    "error": {
                        "message": "Rate limit exceeded for GPU metrics endpoint",
                        "type": "rate_limit_error",
                    }
                },
                status=429,
                headers={"Retry-After": str(retry_after)},
            )

        try:
            snapshot = self.gpu_metrics_collector.get_snapshot()
            associations = build_runner_gpu_associations(
                self.config_manager, snapshot.get("gpus", [])
            )
            snapshot["runner_associations"] = associations
            return web.json_response(snapshot)

        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return web.json_response(
                {
                    "status": "unavailable",
                    "reason": "internal_error",
                    "error": {"message": f"Failed to get GPU metrics: {str(e)}"},
                    "gpus": [],
                    "gpu_history": {},
                },
                status=500,
            )

    async def handle_throughput_metrics(self, request):
        """Handle GET /v1/metrics/throughput requests.

        Returns per-model token throughput history (rolling tokens/sec for
        prompt and generation) with bounded per-model history captured from
        upstream usage/timings. The endpoint is rate-limited per remote IP to
        prevent abuse and mirrors the GPU metrics endpoint contract. When the
        feature is disabled in config the collector returns a stable "disabled"
        snapshot rather than an error.

        Args:
            request: The request.

        Returns:
            The response.
        """
        remote = request.remote or "unknown"
        allowed, retry_after = self.throughput_metrics_limiter.check(remote)
        if not allowed:
            return web.json_response(
                {
                    "error": {
                        "message": "Rate limit exceeded for throughput metrics endpoint",
                        "type": "rate_limit_error",
                    }
                },
                status=429,
                headers={"Retry-After": str(retry_after)},
            )

        try:
            snapshot = self.throughput_metrics_collector.get_snapshot()
            return web.json_response(snapshot)

        except Exception as e:
            logger.error(f"Error getting throughput metrics: {e}")
            return web.json_response(
                {
                    "status": "unavailable",
                    "reason": "internal_error",
                    "error": {"message": f"Failed to get throughput metrics: {str(e)}"},
                    "models": [],
                    "throughput_history": {},
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

    async def handle_responses(self, request):
        """Handle POST /v1/responses requests (OpenAI Responses API).

        Forwards the request as-is to the underlying llama.cpp server's
        /v1/responses endpoint. Requires a llama-server build that includes
        Responses API support (llama.cpp PR #18227+).

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
            request, model_alias, "/v1/responses", data
        )

    @staticmethod
    def _mcp_error_response(
        request_id: object, code: int, message: str, status: int
    ) -> web.Response:
        """Build a JSON-RPC 2.0 error response with the given HTTP status.

        Args:
            request_id: The JSON-RPC request id to echo back (may be None).
            code: The JSON-RPC error code (e.g. -32700 for a parse error).
            message: A human-readable error message.
            status: The HTTP status code for the response.

        Returns:
            A JSON response carrying a JSON-RPC 2.0 error envelope.
        """
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            },
            status=status,
        )

    async def handle_mcp(self, request: web.Request) -> web.StreamResponse:
        """Handle POST requests to the unified MCP proxy endpoint.

        Reads a JSON-RPC 2.0 envelope, selects an mcp-tagged model (an explicit
        top-level "model" or "params.model" hint when present and mcp-tagged,
        otherwise the first mcp-tagged model in config order), ensures that
        model's runner is ready, then proxies the request to the runner's MCP
        upstream path (default /mcp). Non-streaming requests reuse
        runner_manager.forward_request and return the upstream JSON-RPC envelope;
        when the client sends Accept: text/event-stream the SSE streaming path is
        used instead. Pre-flight errors (parse, routing, readiness) and
        non-streaming upstream failures are returned as JSON-RPC 2.0 error
        envelopes echoing the request id. On the SSE streaming path, a
        transport-level error after the readiness check surfaces in the generic
        streaming error shape (MCP-over-SSE error framing is upstream-dependent).

        This endpoint is only registered when mcp.enabled is true, so it does not
        affect any existing behavior when disabled.

        Args:
            request: The request.

        Returns:
            The response (upstream JSON-RPC envelope, an SSE stream, or a
            JSON-RPC error envelope).
        """
        remote = request.remote or "unknown"
        allowed, retry_after = self.mcp_limiter.check(remote)
        if not allowed:
            resp = self._mcp_error_response(None, -32000, "Rate limit exceeded", 429)
            resp.headers["Retry-After"] = str(retry_after)
            return resp

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return self._mcp_error_response(None, -32700, "Parse error", 400)

        request_id = data.get("id") if isinstance(data, dict) else None

        # Extract an optional routing hint. Prefer a top-level "model", then
        # fall back to params.model (some MCP clients nest it).
        requested_alias = None
        if isinstance(data, dict):
            requested_alias = data.get("model")
            if requested_alias is None and isinstance(data.get("params"), dict):
                requested_alias = data["params"].get("model")

        model_alias = self.runner_manager.select_mcp_model(requested_alias)

        if model_alias is None:
            if requested_alias is not None:
                return self._mcp_error_response(
                    request_id,
                    -32602,
                    f"Model not found or not mcp-tagged: {requested_alias}",
                    404,
                )
            return self._mcp_error_response(
                request_id, -32601, "No mcp-tagged model configured", 400
            )

        # Pre-flight readiness check with retry (same shape as other endpoints).
        (
            is_ready,
            error_message,
        ) = await self.runner_manager.ensure_model_ready_with_retry(model_alias)
        if not is_ready:
            logger.error(f"MCP model {model_alias} not ready: {error_message}")
            return self._mcp_error_response(
                request_id, -32000, f"Model not ready: {error_message}", 503
            )

        # MCP Streamable HTTP may return SSE when the client advertises it.
        accept_header = request.headers.get("Accept", "")
        if "text/event-stream" in accept_header:
            logger.debug(
                f"Forwarding streaming MCP request to model {model_alias} "
                f"at {self.mcp_upstream_path}"
            )
            return await self._forward_streaming_request(
                request, model_alias, self.mcp_upstream_path, data
            )

        logger.debug(
            f"Forwarding MCP request to model {model_alias} at {self.mcp_upstream_path}"
        )
        request_start_notified = False
        try:
            await self._notify_request_start(model_alias)
            request_start_notified = True
            (
                success,
                response_data,
                status_code,
            ) = await self.runner_manager.forward_request(
                model_alias, self.mcp_upstream_path, data
            )
            if not success:
                # forward_request synthesizes OpenAI-shaped errors on
                # connection/timeout/non-JSON upstream failures. Re-wrap them as
                # a JSON-RPC 2.0 error envelope (echoing the request id) so the
                # MCP contract holds even when the upstream is unavailable.
                message = "Upstream MCP request failed"
                return self._mcp_error_response(
                    request_id, -32000, message, status_code
                )
            return web.json_response(response_data, status=status_code)
        finally:
            if request_start_notified:
                await self._notify_request_end(model_alias)

    async def handle_audio_speech(self, request):
        """Handle POST /v1/audio/speech requests (OpenAI Speech API).

        Accepts a JSON body (OpenAI Speech shape: model, input, voice,
        response_format, ...), resolves the target audio model, ensures its
        runner is ready, then transparently proxies the request to the
        underlying llama.cpp server and returns the raw binary audio body.

        Unlike the JSON endpoints, the upstream response is binary audio, so it
        cannot use runner.forward_request (which JSON-decodes the body); this
        handler opens its own aiohttp.ClientSession. If the running llama-server
        build lacks the endpoint, the upstream error/status is surfaced verbatim.

        Args:
            request: The request.

        Returns:
            The response (binary audio on success, JSON error otherwise).
        """
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"message": "Invalid JSON"}}, status=400)

        model_alias = self._extract_audio_model_alias(data)

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

        return await self._forward_audio_binary_request(
            model_alias, "/v1/audio/speech", data
        )

    async def handle_audio_transcriptions(self, request):
        """Handle POST /v1/audio/transcriptions requests (OpenAI Transcription API).

        Accepts multipart/form-data (a 'file' audio part plus a 'model' field and
        optional OpenAI fields), resolves the target audio model, ensures its
        runner is ready, then transparently proxies the upload to the underlying
        llama.cpp server and returns the upstream JSON.

        Unlike the JSON endpoints, the request body is multipart/form-data, so
        this handler reads request.post() and re-forwards an aiohttp.FormData; it
        cannot use runner.forward_request (which sends json=). If the running
        llama-server build lacks the endpoint, the upstream error/status is
        surfaced verbatim.

        Args:
            request: The request.

        Returns:
            The response (upstream JSON passed through with upstream status).
        """
        try:
            data = await request.post()
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"Invalid multipart form data: {str(e)}"}},
                status=400,
            )

        model_value = data.get("model")
        model_alias = str(model_value) if model_value else None
        if model_alias is None:
            model_alias = self.runner_manager.get_default_audio_model_alias()

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

        # Pre-flight readiness check with retry (same shape as the JSON endpoints).
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

        runner = self.runner_manager.get_runner_for_model(model_alias)
        if runner is None:
            return web.json_response(
                {"error": {"message": f"Model not available: {model_alias}"}},
                status=500,
            )

        # Rebuild the multipart body for the upstream request. The 'file' part is
        # a FileField; scalar fields (model/response_format/language/...) are
        # forwarded verbatim as strings.
        form = aiohttp.FormData()
        # Iterate items() (not keys()): request.post() returns a MultiDict, so a
        # repeated field (e.g. OpenAI's "timestamp_granularities[]") appears
        # multiple times. keys()+data[key] would forward the first value N times
        # and drop the rest; items() forwards each (key, value) pair once.
        for key, field in data.items():
            if isinstance(field, web.FileField):
                form.add_field(
                    key,
                    field.file,
                    filename=field.filename,
                    content_type=field.content_type,
                )
            else:
                form.add_field(key, str(field))

        url = f"http://{runner.host}:{runner.port}/v1/audio/transcriptions"

        request_start_notified = False
        try:
            await self._notify_request_start(model_alias)
            request_start_notified = True

            if self.runner_manager.timeout == 0:
                timeout_config = aiohttp.ClientTimeout(
                    total=None, sock_connect=None, sock_read=None
                )
            else:
                timeout_config = aiohttp.ClientTimeout(
                    total=self.runner_manager.timeout,
                    sock_connect=self.runner_manager.timeout,
                    sock_read=self.runner_manager.timeout,
                )

            async with self._http_session.post(
                url, data=form, timeout=timeout_config
            ) as response:
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    response_text = await response.text()
                    response_data = {"error": {"message": response_text}}
                return web.json_response(response_data, status=response.status)

        except asyncio.TimeoutError:
            logger.error(f"Timeout forwarding audio transcription to {url}")
            return web.json_response(
                {"error": {"message": "Request timeout"}}, status=408
            )
        except aiohttp.ClientError as e:
            logger.error(f"Client error forwarding audio transcription to {url}: {e}")
            return web.json_response(
                {"error": {"message": "Backend connection failed"}}, status=503
            )
        except Exception as e:
            logger.error(
                f"Unexpected error forwarding audio transcription to {url}: {e}"
            )
            return web.json_response(
                {"error": {"message": "Internal server error"}}, status=500
            )
        finally:
            if request_start_notified:
                await self._notify_request_end(model_alias)

    async def _forward_audio_binary_request(self, model_alias, endpoint, data):
        """Forward a JSON audio request and pass through a binary response.

        Ensures the model is ready, then POSTs the JSON body to the upstream
        runner and returns the raw response body (binary audio) on success, or
        the upstream JSON/text error on a non-200 status. Used by
        /v1/audio/speech, whose upstream response is binary audio and therefore
        cannot use the JSON-decoding forward helpers.

        Args:
            model_alias: The audio model alias.
            endpoint: The upstream endpoint (e.g. "/v1/audio/speech").
            data: The JSON request body.

        Returns:
            The response (binary audio on success, JSON error otherwise).
        """
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

        runner = self.runner_manager.get_runner_for_model(model_alias)
        if runner is None:
            return web.json_response(
                {"error": {"message": f"Model not available: {model_alias}"}},
                status=500,
            )

        url = f"http://{runner.host}:{runner.port}{endpoint}"

        request_start_notified = False
        try:
            await self._notify_request_start(model_alias)
            request_start_notified = True

            if self.runner_manager.timeout == 0:
                timeout_config = aiohttp.ClientTimeout(
                    total=None, sock_connect=None, sock_read=None
                )
            else:
                timeout_config = aiohttp.ClientTimeout(
                    total=self.runner_manager.timeout,
                    sock_connect=self.runner_manager.timeout,
                    sock_read=self.runner_manager.timeout,
                )

            async with self._http_session.post(
                url, json=data, timeout=timeout_config
            ) as response:
                if response.status == 200:
                    body = await response.read()
                    content_type = response.headers.get("Content-Type", "audio/mpeg")
                    return web.Response(
                        body=body,
                        status=response.status,
                        content_type=content_type,
                    )

                # Non-200: surface the upstream error verbatim.
                try:
                    error_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    error_text = await response.text()
                    error_data = {"error": {"message": error_text}}
                return web.json_response(error_data, status=response.status)

        except asyncio.TimeoutError:
            logger.error(f"Timeout forwarding audio request to {url}")
            return web.json_response(
                {"error": {"message": "Request timeout"}}, status=408
            )
        except aiohttp.ClientError as e:
            logger.error(f"Client error forwarding audio request to {url}: {e}")
            return web.json_response(
                {"error": {"message": "Backend connection failed"}}, status=503
            )
        except Exception as e:
            logger.error(f"Unexpected error forwarding audio request to {url}: {e}")
            return web.json_response(
                {"error": {"message": "Internal server error"}}, status=500
            )
        finally:
            if request_start_notified:
                await self._notify_request_end(model_alias)

    def _extract_audio_model_alias(self, data):
        """Extract the audio model alias from a JSON audio request body.

        Prefers an explicit 'model' field. When absent, falls back to the first
        configured audio model (type == "audio") rather than the generic first
        model, so audio requests never silently route to a text model. Returns
        None when neither is available.

        Args:
            data: The JSON request body.

        Returns:
            The audio model alias, or None if not resolvable.
        """
        if "model" in data:
            return data["model"]
        return self.runner_manager.get_default_audio_model_alias()

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
            request_start_notified = False
            try:
                await self._notify_request_start(model_alias)
                request_start_notified = True
                forward_start = time.monotonic()
                (
                    success,
                    response_data,
                    status_code,
                ) = await self.runner_manager.forward_request(
                    model_alias, endpoint, data
                )
                if success and isinstance(response_data, dict):
                    # Cheap, non-blocking throughput capture: usage/timings are
                    # already in scope. Never let this raise into the request.
                    try:
                        self.throughput_metrics_collector.record(
                            model_alias,
                            response_data.get("usage"),
                            response_data.get("timings"),
                            (time.monotonic() - forward_start) * 1000.0,
                        )
                    except Exception:
                        pass
                return web.json_response(response_data, status=status_code)
            finally:
                if request_start_notified:
                    await self._notify_request_end(model_alias)

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
        request_start_notified = False

        # keepalive check timeout
        keepalive_check_timeout = 10

        try:
            await self._notify_request_start(model_alias)
            request_start_notified = True

            # Get streaming timeout configuration
            streaming_timeout = self.config_manager.get_streaming_timeout_seconds()
            if streaming_timeout == 0:
                timeout_config = aiohttp.ClientTimeout(
                    total=None, sock_connect=None, sock_read=None
                )
            else:
                # Use the same value for total and sock_read to avoid premature read timeouts
                timeout_config = aiohttp.ClientTimeout(
                    total=streaming_timeout,
                    sock_connect=streaming_timeout,
                    sock_read=streaming_timeout,
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=data, timeout=timeout_config
                ) as response:
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

                    # Track last activity to avoid sending keepalive during active streaming
                    last_activity = time.time()
                    keepalive_stop = asyncio.Event()

                    # Start SSE keepalive pings to avoid client inactivity timeouts
                    async def _keepalive():
                        try:
                            while not keepalive_stop.is_set():
                                await asyncio.sleep(
                                    keepalive_check_timeout
                                )  # Check every keepalive_check_timeout seconds
                                if keepalive_stop.is_set():
                                    break

                                # Only send keepalive if no data sent in last keepalive_check_timeout seconds
                                if (
                                    time.time() - last_activity
                                    > keepalive_check_timeout
                                ):
                                    try:
                                        # SSE comment - standard keepalive that clients ignore
                                        await streaming_response.write(b":\n\n")
                                    except Exception:
                                        break
                        except asyncio.CancelledError:
                            pass

                    keepalive_task = asyncio.create_task(_keepalive())

                    # Read-only throughput tap. We retain only the most recent
                    # non-empty "data:" SSE line (whose payload is not [DONE])
                    # so the final usage/timings chunk can be parsed once after
                    # the loop. iter_chunked(8192) can split a single SSE line
                    # across reads, so an O(1) trailing-partial-line carry
                    # buffer is kept between chunks. The original chunk is always
                    # written verbatim; this never mutates the client stream.
                    stream_start = time.monotonic()
                    last_data_line = None
                    carry = ""

                    # Stream the response data
                    try:
                        async for chunk in response.content.iter_chunked(8192):
                            await streaming_response.write(chunk)
                            last_activity = time.time()
                            try:
                                text = carry + chunk.decode("utf-8", errors="ignore")
                                lines = text.split("\n")
                                carry = lines.pop()
                                for line in lines:
                                    stripped = line.strip()
                                    if not stripped.startswith("data:"):
                                        continue
                                    payload = stripped[len("data:") :].strip()
                                    if payload and payload != "[DONE]":
                                        last_data_line = payload
                            except Exception:
                                # Never let the throughput tap disturb streaming.
                                carry = ""
                    finally:
                        keepalive_stop.set()
                        try:
                            keepalive_task.cancel()
                        except Exception:
                            pass

                    # Parse the retained final data line once and record a
                    # throughput sample. Fully defensive: missing usage/timings
                    # (older llama-server builds or include_usage off) simply
                    # records nothing.
                    if last_data_line is not None:
                        try:
                            parsed = json.loads(last_data_line)
                            if isinstance(parsed, dict):
                                self.throughput_metrics_collector.record(
                                    model_alias,
                                    parsed.get("usage"),
                                    parsed.get("timings"),
                                    (time.monotonic() - stream_start) * 1000.0,
                                )
                        except Exception:
                            pass

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
        finally:
            if request_start_notified:
                await self._notify_request_end(model_alias)


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
