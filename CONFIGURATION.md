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

