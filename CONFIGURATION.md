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

## MCP Proxy Configuration

FlexLLama can expose a single Model Context Protocol (MCP) endpoint that forwards requests to one of your models. It is **off by default** — when the `mcp` block is absent or `enabled` is `false`, nothing changes.

To turn it on, add an `mcp` block and tag the model(s) it should use with `"mcp": true`:

```json
{
    "mcp": {
        "enabled": true
    },
    "models": [
        {
            "runner": "runner1",
            "model": "/path/to/model.gguf",
            "model_alias": "model1",
            "mcp": true
        }
    ]
}
```

**Options:**

- `enabled`: Turn the MCP endpoint on or off (default: `false`).
- `endpoint`: The path FlexLLama listens on (default: `/v1/mcp`).
- `upstream_path`: The path on the model's llama-server that requests are forwarded to (default: `/mcp`).

Once enabled, send your MCP (JSON-RPC) requests to `POST http://localhost:8080/v1/mcp`. If you tag more than one model, include a `"model"` field in the request to pick one; otherwise the first `mcp`-tagged model is used.

> **Tip:** give the MCP model its own runner/port so an MCP request doesn't unload another model sharing that runner.
>
> **Note:** MCP support in llama.cpp is still evolving. If your `llama-server` build doesn't serve the MCP path, FlexLLama simply passes the upstream response back (it won't crash), and you can point `upstream_path` at wherever a future build serves it.

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
- `type`: Model type, `"text"` (default) or `"audio"`. Optional; when omitted the model is treated as text, so existing configs are unaffected.
- `audio_encoder`: Path to the audio multimodal projector gguf (passed to llama-server as `--mmproj`). Use for speech-input models like Voxtral, Ultravox, Qwen2-Audio, and Qwen3-ASR. If both `mmproj` and `audio_encoder` are set, `mmproj` wins and `audio_encoder` is ignored (llama-server rejects a duplicate `--mmproj`).
- `talker_model`: Path to the qwen3-omni talker gguf (passed as `--talker-model`). Enables the upstream `/v1/audio/speech` endpoint. Requires a llama-server build with TTS support.
- `code2wav_model`: Path to the qwen3-omni code2wav (talker code detokenizer) gguf (passed as `--code2wav-model`).
- `model_vocoder`: Path to a vocoder gguf (passed as `--model-vocoder`).
- `mcp`: Set to `true` to tag a model as a target for the unified `/v1/mcp` endpoint. Pure metadata (it is never passed to llama-server); see [MCP Proxy Configuration](#mcp-proxy-configuration).

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

## Audio Endpoints

FlexLLama can proxy OpenAI-style audio requests to a model that handles audio:

- `POST /v1/audio/transcriptions` — speech-to-text (send an audio file, get text back).
- `POST /v1/audio/speech` — text-to-speech (send text, get audio back; needs a TTS model).

To enable them, configure an audio model with `"type": "audio"` and point `audio_encoder` at the model's audio projector file. Give it its own runner so it doesn't compete with your chat models:

```json
{
    "runner": "audio-runner",
    "type": "audio",
    "model": "/path/to/voxtral.gguf",
    "audio_encoder": "/path/to/voxtral-mmproj.gguf",
    "model_alias": "voxtral-audio",
    "n_ctx": 8192,
    "n_gpu_layers": 99
}
```

For text-to-speech, also set `talker_model` / `code2wav_model` / `model_vocoder` (see the model options above) and use a TTS-capable model and `llama-server` build.

### Usage Examples

All examples call FlexLLama on `http://localhost:8080`; it routes each request to the runner hosting your audio model. Replace the aliases with your own.

**Speech-to-text (file upload):** the `/v1/audio/transcriptions` endpoint follows the OpenAI standard — it takes a `multipart/form-data` file upload, not JSON:

```bash
curl -sS http://localhost:8080/v1/audio/transcriptions \
  -F model=voxtral-audio \
  -F file=@/path/to/audio.wav \
  -F temperature=0 \
  -F 'prompt=Transcribe the audio verbatim. Output only the transcript, no commentary.'
```

**Speech-to-text (JSON):** to send audio in a JSON body (like the speech endpoint), use `/v1/chat/completions` with a base64-encoded `input_audio` part:

```bash
AUDIO_B64=$(base64 -w0 /path/to/audio.wav)
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxtral-audio",
    "temperature": 0,
    "messages": [
      {"role": "system", "content": "Transcribe the audio verbatim. Output only the transcript, no commentary."},
      {"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "'"$AUDIO_B64"'", "format": "wav"}}
      ]}
    ]
  }'
```

**Text-to-speech:**

```bash
curl -sS http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-omni-tts", "input": "Hello from FlexLLama.", "voice": "default", "response_format": "wav"}' \
  --output speech.wav
```

To check audio works end-to-end, run `python tests/test_audio.py` while FlexLLama is running.

## Validating Configuration

To validate your `config.json` file, run:

```bash
python backend/config.py config.json
```

A successful validation will print a confirmation message. If there are errors, they will be displayed with details on how to fix them.
