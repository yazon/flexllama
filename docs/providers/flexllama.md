---
summary: "Run OpenClaw with FlexLLama (multi-model llama.cpp manager with OpenAI-compatible API)"
read_when:
  - You want to run multiple local models simultaneously (chat + embeddings)
  - You want multi-GPU model distribution with a single API endpoint
  - You need dynamic model switching with auto-unload
title: "FlexLLama"
---

# FlexLLama

FlexLLama is a lightweight self-hosted tool that manages multiple llama.cpp server instances behind a single **OpenAI-compatible `/v1` endpoint**. It handles multi-model orchestration, multi-GPU distribution, and dynamic model loading — all with a single config file.

- Provider: `flexllama`
- Auth: None required (local server)
- Default base URL: `http://127.0.0.1:8080/v1`
- GitHub: [github.com/yazon/flexllama](https://github.com/yazon/flexllama)

**When to pick FlexLLama over Ollama:**

- You need **multiple models running simultaneously** (e.g. chat on GPU 0, embeddings on GPU 1)
- You want explicit control over GPU assignment per model
- You need embedding or reranking models alongside chat models
- You want auto-unload of idle models to free GPU memory

## Quick start

1. Install and start FlexLLama:

```bash
pip install git+https://github.com/yazon/flexllama.git
```

2. Create a `config.json` (adjust paths to your llama-server binary and model files):

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
        "auto_unload_timeout_seconds": 300
    },
    "models": [
        {
            "runner": "runner1",
            "model": "/path/to/your-model.gguf",
            "model_alias": "my-local-model",
            "n_ctx": 8192,
            "n_gpu_layers": 99,
            "flash_attn": "on",
            "jinja": true
        }
    ]
}
```

3. Start FlexLLama:

```bash
flexllama config.json
```

4. Verify it's running:

```bash
curl http://127.0.0.1:8080/v1/models
```

5. Configure OpenClaw:

```json5
{
  agents: {
    defaults: {
      model: { primary: "flexllama/my-local-model" },
    },
  },
  models: {
    mode: "merge",
    providers: {
      flexllama: {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "flexllama",
        api: "openai-completions",
        models: [
          {
            id: "my-local-model",
            name: "My Local Model",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 8192,
            maxTokens: 4096,
          },
        ],
      },
    },
  },
}
```

The `id` must match the `model_alias` in FlexLLama's config.

## Docker quick start

FlexLLama also provides Docker images (CPU, CUDA, Vulkan):

```bash
git clone https://github.com/yazon/flexllama.git
cd flexllama
mkdir -p models
# Download a model into models/
./docker-start.sh              # CPU
./docker-start.sh --gpu=cuda   # NVIDIA GPU
./docker-start.sh --gpu=vulkan # AMD/Intel GPU
```

The Docker setup runs on port `8090` by default. Adjust the `baseUrl` in your OpenClaw config accordingly:

```json5
{
  models: {
    providers: {
      flexllama: {
        baseUrl: "http://127.0.0.1:8090/v1",
        // ...
      },
    },
  },
}
```

## Multi-model setup

FlexLLama can serve multiple models simultaneously on different GPUs. This is useful for running a chat model and an embedding model side by side:

```json5
// FlexLLama config.json
{
  "auto_start_runners": true,
  "api": { "host": "0.0.0.0", "port": 8080, "health_endpoint": "/health" },
  "runner_chat": {
    "type": "llama-server",
    "path": "/path/to/llama-server",
    "host": "127.0.0.1",
    "port": 8085,
    "auto_unload_timeout_seconds": 300
  },
  "runner_embed": {
    "type": "llama-server",
    "path": "/path/to/llama-server",
    "host": "127.0.0.1",
    "port": 8086
  },
  "models": [
    {
      "runner": "runner_chat",
      "model": "/models/Qwen3-8B-Q4_K_M.gguf",
      "model_alias": "qwen3-8b",
      "n_ctx": 32768,
      "n_gpu_layers": 99,
      "main_gpu": 0,
      "flash_attn": "on",
      "jinja": true
    },
    {
      "runner": "runner_embed",
      "model": "/models/nomic-embed-text-v1.5.Q8_0.gguf",
      "model_alias": "nomic-embed",
      "embedding": true,
      "pooling": "cls",
      "n_ctx": 8192,
      "n_gpu_layers": 99,
      "main_gpu": 1
    }
  ]
}
```

Then in OpenClaw, register both models:

```json5
{
  agents: {
    defaults: {
      model: { primary: "flexllama/qwen3-8b" },
    },
  },
  models: {
    mode: "merge",
    providers: {
      flexllama: {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "flexllama",
        api: "openai-completions",
        models: [
          {
            id: "qwen3-8b",
            name: "Qwen3 8B (GPU 0)",
            contextWindow: 32768,
            maxTokens: 8192,
          },
          {
            id: "nomic-embed",
            name: "Nomic Embed (GPU 1)",
            input: ["text"],
          },
        ],
      },
    },
  },
}
```

## Hybrid config (local primary, cloud fallback)

Keep hosted models as fallbacks when the local box is under load or down:

```json5
{
  agents: {
    defaults: {
      model: {
        primary: "flexllama/qwen3-8b",
        fallbacks: ["anthropic/claude-sonnet-4-5"],
      },
    },
  },
  models: {
    mode: "merge",
    providers: {
      flexllama: {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "flexllama",
        api: "openai-completions",
        models: [
          {
            id: "qwen3-8b",
            name: "Qwen3 8B Local",
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

## Troubleshooting

- **Gateway can't reach FlexLLama?** Check `curl http://127.0.0.1:8080/v1/models`.
- **Model not loading?** Check FlexLLama logs. First model load takes time. FlexLLama returns `503` while loading, and OpenClaw will retry automatically.
- **Wrong model alias?** The `id` in OpenClaw's provider config must exactly match the `model_alias` in FlexLLama's `config.json`.
- **Docker networking?** If both run in Docker, use the container/service name instead of `127.0.0.1` (e.g. `http://flexllama:8090/v1`).
- **Dashboard**: FlexLLama serves a monitoring dashboard at `http://127.0.0.1:8080/` to check model status.

## See also

- [Local Models](/gateway/local-models) — General local model guidance
- [Model Providers](/concepts/model-providers) — All provider overview
- [vLLM](/providers/vllm) — Alternative OpenAI-compatible local server
- [Ollama](/providers/ollama) — Alternative local LLM runtime
