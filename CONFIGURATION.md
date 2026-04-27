# FlexLLama Configuration Guide

This guide covers all configuration options available in FlexLLama's `config.json` file.

## Basic Structure

Edit `config.json` to configure your runners and models:

```json
{
    "auto_start_runners": true,
    "api": {
        "host": "0.0.0.0",
        "port": 8080,
        "health_endpoint": "/health"
    },
    "runner1": {
        "type": "llama-server",
        "path": "/path/to/llama-server",
        "host": "127.0.0.1",
        "port": 8085,
        "inherit_env": true,
        "env": {}
    },
    "models": [
        {
            "runner": "runner1",
            "model": "/path/to/model.gguf",
            "model_alias": "my-model",
            "n_ctx": 4096,
            "n_gpu_layers": 99,
            "main_gpu": 0
        }
    ]
}
```

## Multi-GPU Setup

```json
{
    "runner_gpu0": {
        "path": "/path/to/llama-server",
        "port": 8085,
        "inherit_env": true,
        "env": {}
    },
    "runner_gpu1": {
        "path": "/path/to/llama-server", 
        "port": 8086,
        "inherit_env": true,
        "env": {}
    },
    "models": [
        {
            "runner": "runner_gpu0",
            "model": "/path/to/chat-model.gguf",
            "model_alias": "chat-model",
            "main_gpu": 0,
            "n_gpu_layers": 99
        },
        {
            "runner": "runner_gpu1",
            "model": "/path/to/embedding-model.gguf",
            "model_alias": "embedding-model",
            "embedding": true,
            "main_gpu": 1,
            "n_gpu_layers": 99
        }
    ]
}
```

## Auto-unload Configuration

FlexLLama supports automatic model unloading to free up RAM when models are idle. This is useful for managing memory usage when running multiple models.

```json
{
    "runner_memory_saver": {
        "path": "/path/to/llama-server",
        "port": 8085,
        "auto_unload_timeout_seconds": 300
    },
    "runner_always_on": {
        "path": "/path/to/llama-server",
        "port": 8086,
        "auto_unload_timeout_seconds": 0
    },
    "models": [
        {
            "runner": "runner_memory_saver",
            "model": "/path/to/large-model.gguf",
            "model_alias": "large-model"
        },
        {
            "runner": "runner_always_on",
            "model": "/path/to/small-model.gguf",
            "model_alias": "small-model"
        }
    ]
}
```

**Auto-unload Behavior:**
- `auto_unload_timeout_seconds: 0` - Disables auto-unload (default)
- `auto_unload_timeout_seconds: 300` - Unloads model after 5 minutes of inactivity
- Models are considered "active" while processing requests (including streaming)
- The timeout is measured from the last request completion
- Auto-unload frees RAM by stopping the runner process entirely
- Models will be automatically reloaded when the next request arrives

## Environment Variables

FlexLLama supports setting environment variables for runners and individual models. This is useful for configuring GPU devices, library paths, or other runtime settings.

```json
{
    "runner_vulkan": {
        "type": "llama-server",
        "path": "/path/to/llama-server",
        "port": 8085,
        "inherit_env": true,
        "env": {
            "GGML_VULKAN_DEVICE": "1",
            "RUNNER_SPECIFIC_VAR": "value"
        }
    },
    "models": [
        {
            "runner": "runner_vulkan",
            "model": "/path/to/model.gguf",
            "model_alias": "my-model",
            "env": {
                "MODEL_SPECIFIC_VAR": "override"
            }
        }
    ]
}
```

## Timeout Configuration

FlexLLama supports configurable timeouts for long-running requests:

```json
{
    "request_timeout_seconds": 1800,
    "streaming_timeout_seconds": 0
}
```

- `request_timeout_seconds`: Timeout for non-streaming requests (default: 1800 = 30 minutes)
- `streaming_timeout_seconds`: Timeout for streaming requests (default: 0 = no timeout)

**Note:** The server keepalive timeout is set to 60 minutes by default to support long-running model inference.

## CORS Configuration

By default FlexLLama does **not** emit CORS headers, so browser-based clients
loaded from a different origin cannot call the API. This is the safe default:
allowing cross-origin access means any website a user visits can script
requests against their FlexLLama instance (which has no authentication).

To opt in, set `api.cors_allow_origins` to a list of origins:

```json
{
    "api": {
        "host": "0.0.0.0",
        "port": 8080,
        "cors_allow_origins": ["http://localhost:5173", "https://my-chat-ui.example"]
    }
}
```

- `[]` (default): CORS disabled. Recommended when FlexLLama is only called
  from same-origin pages or from non-browser clients.
- `["*"]`: Allow any origin (non-credentialed). Convenient for local dev;
  understand that any page in any browser on your network can then call
  the API.
- Explicit list: The request `Origin` header must match exactly; the server
  reflects it and sets `Vary: Origin`. Preferred for production.

`Access-Control-Allow-Credentials` is never set, so browser cookies are not
forwarded cross-origin regardless of this setting.

## GPU Metrics Configuration

FlexLLama can collect real-time GPU telemetry and display it in the dashboard.
The feature supports both NVIDIA and AMD tools when present:

- `nvidia-smi` for NVIDIA GPUs (Linux and Windows)
- `amd-smi` for AMD GPUs (Linux)

```json
{
    "metrics": {
        "gpu": {
            "enabled": true,
            "vendors": ["nvidia", "amd"],
            "poll_interval_seconds": 2,
            "history_points": 60,
            "command_timeout_seconds": 3,
            "rate_limit_requests_per_minute": 120
        }
    }
}
```

**GPU Metrics Options:**

- `enabled`: Enable or disable GPU metrics collection (default: `true`). When disabled, the dashboard shows "GPU metrics unavailable".
- `vendors`: Which collectors to try (default: `["nvidia", "amd"]`). Supported values: `"nvidia"`, `"amd"`.
- `poll_interval_seconds`: How often to poll vendor tools for new metrics (default: `2`).
- `history_points`: Number of historical data points to retain per metric for sparkline rendering (default: `60`).
- `command_timeout_seconds`: Subprocess timeout for telemetry commands in seconds (default: `3`).
- `rate_limit_requests_per_minute`: Per-IP rate limit for the `/v1/metrics/gpus` endpoint (default: `120`).

**Platform Support:**

| Platform | GPU Metrics Support |
|----------|-------------------|
| Linux with `nvidia-smi` and/or `amd-smi` | Fully supported |
| Windows with `nvidia-smi` | Supported (NVIDIA) |
| Windows without `nvidia-smi` | Unavailable (tool not found) |
| macOS | Unavailable (unsupported platform) |
| Docker without GPU visibility | Unavailable (no visible GPUs) |

**Requirements:**

- For NVIDIA metrics: `nvidia-smi` installed and available in `PATH`
- For AMD metrics: `amd-smi` installed and available in `PATH` (part of ROCm)
- GPU devices accessible to the FlexLLama process

**Important Notes:**

- GPU metrics collection never blocks server startup or request handling.
- If no supported vendor tool is installed or visible, the dashboard displays a clear "GPU metrics unavailable" message.
- Runner-to-GPU associations are based on model configuration (`main_gpu`, `tensor_split`) and are labeled as advisory.
- The `/v1/metrics/gpus` endpoint is rate-limited to prevent abuse.

## Configuration Options Reference

### Runner Options

- `path`: Path to llama-server binary
- `host`/`port`: Where to run this instance
- `inherit_env`: Whether to inherit parent environment variables (default: `true`)
- `env`: Dictionary of environment variables to set for all models on this runner
- `extra_args`: Additional arguments for llama-server (applied to all models using this runner)
- `auto_unload_timeout_seconds`: Automatically unload model after this many seconds of inactivity (0 disables, default: 0)

### Model Options

#### Core Settings

- `runner`: Which runner to use for this model
- `model`: Path to .gguf model file
- `model_alias`: Name to use in API calls
- `inherit_env`: Override runner's inherit_env setting for this model (optional)
- `env`: Dictionary of environment variables specific to this model (overrides runner env)

#### Model Types

- `embedding`: Set to `true` for embedding models
- `reranking`: Set to `true` for reranking models
- `mmproj`: Path to multimodal projection file (for vision models)

#### Performance & Memory

- `n_ctx`: Context window size (e.g., 4096, 8192, 32768)
- `n_batch`: Batch size for processing (e.g., 256, 512)
- `u_batch`: Physical batch size for prompt processing (e.g., 64, 128)
- `n_threads`: Number of CPU threads to use
- `main_gpu`: Which GPU to use (0, 1, 2...)
- `n_gpu_layers`: How many layers to offload to GPU (99 for all layers)
- `tensor_split`: Array defining how to split model across GPUs (e.g., [1.0, 0.0])
- `offload_kqv`: Whether to offload key-value cache to GPU (`true`/`false`)
- `use_mlock`: Lock model in RAM to prevent swapping (`true`/`false`)

#### Optimization

- `flash_attn`: Flash attention mode - `"on"`, `"off"`, or `"auto"` (case-sensitive). Boolean values (`true`/`false`) are deprecated but still supported for backwards compatibility.
- `split_mode`: How to split model layers ("row" or other modes)
- `cache-type-k`: Key cache quantization type (e.g., "q8_0")
- `cache-type-v`: Value cache quantization type (e.g., "q8_0")

#### Chat & Templates

- `chat_template`: Chat template format (e.g., "mistral-instruct", "gemma")
- `jinja`: Enable Jinja templating (`true`/`false`)

#### Advanced Options

- `rope-scaling`: RoPE scaling method (e.g., "linear")
- `rope-scale`: RoPE scaling factor (e.g., 2)
- `yarn-orig-ctx`: Original context size for YaRN scaling
- `pooling`: Pooling method for embeddings (e.g., "cls")
- `args`: Additional custom arguments to pass directly to llama-server for this specific model (string, e.g., "--custom-flag --param value"). These are applied after all other model parameters and before runner `extra_args`.

## Validating Configuration

To validate your `config.json` file, run:

```bash
python backend/config.py config.json
```

A successful validation will print a confirmation message. If there are errors, they will be displayed with details on how to fix them.
