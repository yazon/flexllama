# FlexLLama + OpenClaw Integration Analysis

## 1. What is FlexLLama?

**FlexLLama** (v0.1.8) is a lightweight, self-hosted Python tool that manages multiple [llama.cpp](https://github.com/ggerganov/llama.cpp) server instances behind a single **OpenAI v1-compatible API**. It acts as a smart reverse proxy and process orchestrator for local LLM inference.

### Core Architecture

```
Client (OpenAI API calls)
        │
        ▼
┌──────────────────────────┐
│  FlexLLama API Server    │   ← aiohttp, port 8080/8090
│  (OpenAI v1 endpoints)   │
│  /v1/chat/completions    │
│  /v1/completions         │
│  /v1/embeddings          │
│  /v1/rerank              │
│  /v1/models              │
│  /health                 │
└──────────┬───────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Runner1 │ │ Runner2 │   ← llama-server processes
│ :8085   │ │ :8086   │
│ GPU 0   │ │ GPU 1   │
└─────────┘ └─────────┘
```

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-model management** | Run multiple GGUF models simultaneously on different runners |
| **Multi-GPU support** | Assign models to specific GPUs (CUDA, Vulkan, CPU) |
| **OpenAI v1 API** | Drop-in compatible `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/models` |
| **Dynamic model switching** | Load/unload models on the fly per runner |
| **Auto model unload** | Automatically free RAM after configurable idle timeout |
| **Streaming (SSE)** | Full Server-Sent Events streaming support |
| **Auto-start** | Automatically start default runners on launch |
| **Retry with backoff** | Configurable retry logic for model loading |
| **Web dashboard** | Real-time monitoring UI |
| **Docker support** | CPU, CUDA, and Vulkan Docker profiles |
| **CORS support** | Built-in CORS headers for web clients |

### Technical Details

- **Language**: Python 3.10+
- **Dependencies**: `aiohttp`, `psutil` (minimal footprint)
- **License**: BSD-3-Clause
- **Backend**: Wraps `llama-server` (llama.cpp's built-in HTTP server)
- **Config**: JSON-based (`config.json`)
- **Author**: Wojciech Czaplejewicz

---

## 2. What is OpenClaw?

**OpenClaw** is a personal AI assistant platform that runs on your own devices and connects to messaging channels you already use (WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, Microsoft Teams, WebChat, and more).

### Core Architecture

```
WhatsApp / Telegram / Slack / Discord / Signal / iMessage / Teams / WebChat
               │
               ▼
┌───────────────────────────────┐
│         OpenClaw Gateway      │
│       (control plane)         │
│     ws://127.0.0.1:18789      │
└──────────────┬────────────────┘
               │
               ├─ Pi agent (RPC) → LLM Providers
               ├─ CLI (openclaw …)
               ├─ WebChat UI
               ├─ macOS app
               └─ iOS / Android nodes
```

### Key Characteristics

- **TypeScript/Node.js** (v22+) with companion Swift (macOS/iOS) and Kotlin (Android) apps
- **License**: MIT
- **Model providers**: Anthropic, OpenAI, Google, Ollama, vLLM, LM Studio, OpenRouter, and many more
- **Local model support**: Via OpenAI-compatible endpoints (`/v1/chat/completions`)
- **Provider config**: JSON5-based `openclaw.json` with `models.providers` section
- **Auth**: API keys, OAuth, bearer tokens
- **Multi-agent**: Session routing, per-agent model selection, fallback chains

---

## 3. How OpenClaw Connects to Local Models

OpenClaw supports local/self-hosted LLMs through its **custom provider** system. Any server exposing OpenAI-compatible `/v1` endpoints can be registered as a provider.

### OpenClaw's Provider Configuration Pattern

```json5
{
  models: {
    mode: "merge",          // Keep hosted models as fallbacks
    providers: {
      "<provider-name>": {
        baseUrl: "http://<host>:<port>/v1",
        apiKey: "<any-string>",
        api: "openai-completions",     // or "openai-responses"
        models: [
          {
            id: "<model-id>",
            name: "<display-name>",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

### Endpoints OpenClaw Consumes

From the local-models and provider documentation, OpenClaw primarily uses:

1. **`GET /v1/models`** — Model discovery and availability checks
2. **`POST /v1/chat/completions`** — Primary inference endpoint (streaming + non-streaming)
3. Health/availability probes via the models endpoint

---

## 4. Compatibility Analysis: FlexLLama ↔ OpenClaw

### Already Compatible (No Changes Needed)

FlexLLama **already provides** exactly the endpoints OpenClaw expects for a local model provider:

| OpenClaw Expects | FlexLLama Provides | Status |
|---|---|---|
| `GET /v1/models` | `handle_models` — returns `{ object: "list", data: [...] }` | **Compatible** |
| `POST /v1/chat/completions` (non-streaming) | `handle_chat_completions` — forwards to llama.cpp and returns OpenAI-format JSON | **Compatible** |
| `POST /v1/chat/completions` (streaming SSE) | `_forward_streaming_request` — full SSE with `data: [DONE]` termination and keepalive pings | **Compatible** |
| `POST /v1/completions` | `handle_completions` — text completions | **Compatible** |
| `POST /v1/embeddings` | `handle_embeddings` — embedding models | **Compatible** |
| CORS headers | `handle_options` — returns `Access-Control-Allow-Origin: *` | **Compatible** |
| Health checks | `GET /health` (configurable endpoint) | **Compatible** |

### OpenClaw Configuration for FlexLLama

A user can connect OpenClaw to FlexLLama **today, with zero code changes**, using this config:

```json5
{
  agents: {
    defaults: {
      model: {
        primary: "flexllama/my-chat-model",
        fallbacks: ["anthropic/claude-sonnet-4-5"],  // hosted fallback
      },
      models: {
        "flexllama/my-chat-model": { alias: "Local Chat" },
        "flexllama/my-embed-model": { alias: "Local Embeddings" },
        "anthropic/claude-sonnet-4-5": { alias: "Sonnet (cloud)" },
      },
    },
  },
  models: {
    mode: "merge",
    providers: {
      flexllama: {
        baseUrl: "http://127.0.0.1:8080/v1",   // FlexLLama API port
        apiKey: "flexllama-local",               // Any string (FlexLLama has no auth)
        api: "openai-completions",
        models: [
          {
            id: "my-chat-model",                 // Must match FlexLLama model_alias
            name: "Qwen3-4B Local",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 32768,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
}
```

### What Works Out of the Box

1. **Chat completions** (streaming and non-streaming) — the primary use case
2. **Embeddings** — FlexLLama supports embedding models with `"embedding": true`
3. **Multi-model via aliases** — OpenClaw can reference different FlexLLama model aliases
4. **Health monitoring** — OpenClaw can probe FlexLLama's `/health` endpoint
5. **Model listing** — `GET /v1/models` returns the expected format
6. **Hybrid setup** — `models.mode: "merge"` keeps cloud providers as fallbacks

---

## 5. Potential Improvements to Enhance OpenClaw Support

While FlexLLama works with OpenClaw today, several enhancements could improve the integration experience:

### 5.1 Authentication Support (Medium Priority)

**Current state**: FlexLLama has no authentication. All endpoints are open.

**Why it matters**: OpenClaw sends `Authorization: Bearer <token>` headers to all providers. FlexLLama currently ignores these, which works fine. However, if FlexLLama is exposed over a network (e.g., running on a GPU server while OpenClaw runs on a different machine), authentication becomes important for security.

**Recommendation**: Add optional bearer token authentication via config:

```json
{
  "api": {
    "auth_token": "my-secret-token"
  }
}
```

### 5.2 Model Metadata in `/v1/models` Response (Low Priority)

**Current state**: FlexLLama's `/v1/models` returns minimal metadata:

```json
{
  "id": "model-alias",
  "object": "model",
  "created": 1234567890,
  "owned_by": "user"
}
```

**Why it matters**: OpenClaw can auto-discover models from local providers. Richer metadata (context window, capabilities) would allow OpenClaw to configure itself automatically rather than requiring manual model entries.

**Recommendation**: Extend model metadata from config to the `/v1/models` response (this is what Ollama and vLLM do for auto-discovery). Fields like `context_length`, `capabilities` (embedding, reranking), and model family info would be useful.

### 5.3 API Key Validation Passthrough (Low Priority)

**Current state**: FlexLLama does not validate or check any API key.

**Recommendation**: When auth is enabled, return proper `401 Unauthorized` responses so OpenClaw's failover logic knows to retry with different credentials or fall to the next provider.

### 5.4 Graceful Model Loading Feedback (Already Good)

**Current state**: FlexLLama already returns `503` when models are loading, which is the standard signal OpenClaw uses to detect loading states and retry. This is well-implemented.

### 5.5 Docker Networking for OpenClaw + FlexLLama (Documentation)

When both run in Docker, they need to share a network or use host networking. This is a documentation/example concern, not a code change.

---

## 6. How to Promote FlexLLama for OpenClaw Usage

### 6.1 Unique Value Proposition

FlexLLama fills a specific gap that other local inference servers don't cover well:

| Feature | FlexLLama | Ollama | LM Studio | vLLM |
|---------|-----------|--------|-----------|------|
| Multiple models simultaneously | **Yes** (multi-runner) | Limited (auto-swap) | Single model | Single model |
| Multi-GPU distribution | **Yes** (per-runner GPU assignment) | Limited | No | Yes (tensor parallel) |
| Dynamic model switching | **Yes** (auto load/unload) | Yes (auto) | Manual | No |
| Auto-unload idle models | **Yes** (configurable timeout) | Yes | No | No |
| OpenAI v1 API | **Yes** | Yes (partial) | Yes | Yes |
| Embedding models | **Yes** | Yes | No | Yes |
| Reranking models | **Yes** | No | No | No |
| Lightweight (2 deps) | **Yes** | Heavy (Go binary) | Desktop app | Heavy (Python + CUDA) |
| Docker multi-profile | **Yes** (CPU/CUDA/Vulkan) | Yes | No | Yes |
| Dashboard | **Yes** | No | Yes | No |

**Key differentiator for OpenClaw users**: FlexLLama allows running **multiple specialized models** (chat + embedding + reranking) **simultaneously across different GPUs**, all behind a single OpenAI-compatible endpoint. This is ideal for OpenClaw's multi-model architecture where you might want:

- A large chat model on GPU 0
- An embedding model on GPU 1
- A fallback smaller chat model ready to swap in

### 6.2 Promotion Strategies

#### A. OpenClaw Provider Documentation

Create a pull request to OpenClaw's `docs/providers/` directory adding a `flexllama.md` provider page (similar to their existing `ollama.md` and `vllm.md` pages). Content:

- What FlexLLama is and why it's different (multi-model, multi-GPU)
- Quick setup with Docker
- Example OpenClaw config for FlexLLama
- Multi-model setup (chat + embeddings on different runners)
- Hybrid config with cloud fallbacks

#### B. FlexLLama Documentation

Add an "OpenClaw Integration" section to FlexLLama's README or a dedicated doc page showing:

1. How to configure FlexLLama as an OpenClaw provider
2. Recommended model configurations for OpenClaw use cases
3. Docker Compose setup for running both together
4. Example configs for common scenarios

#### C. Example Configurations

Create ready-to-use configs showing FlexLLama serving OpenClaw:

**Single model (simplest)**:
```json5
// FlexLLama config.json
{
  "auto_start_runners": true,
  "api": { "host": "0.0.0.0", "port": 8080, "health_endpoint": "/health" },
  "runner1": {
    "type": "llama-server",
    "path": "/path/to/llama-server",
    "host": "127.0.0.1",
    "port": 8085
  },
  "models": [
    {
      "runner": "runner1",
      "model": "/models/Qwen3-8B-Q4_K_M.gguf",
      "model_alias": "qwen3-8b",
      "n_ctx": 32768,
      "n_gpu_layers": 99,
      "flash_attn": "on",
      "jinja": true
    }
  ]
}
```

**Multi-model for OpenClaw power users**:
```json5
// FlexLLama config.json — chat + embeddings + fallback
{
  "auto_start_runners": true,
  "api": { "host": "0.0.0.0", "port": 8080, "health_endpoint": "/health" },
  "runner_chat": {
    "type": "llama-server",
    "path": "/path/to/llama-server",
    "host": "127.0.0.1",
    "port": 8085,
    "auto_unload_timeout_seconds": 600
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
      "runner": "runner_chat",
      "model": "/models/Mistral-7B-Q4_K_M.gguf",
      "model_alias": "mistral-7b",
      "n_ctx": 8192,
      "n_gpu_layers": 99,
      "main_gpu": 0,
      "flash_attn": "on"
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

#### D. Community Outreach

1. **OpenClaw Discord**: Post in their Discord (https://discord.gg/clawd) about FlexLLama as a local model backend, highlighting multi-GPU and multi-model capabilities
2. **GitHub Discussions/Issues**: Open a discussion on OpenClaw's repo suggesting FlexLLama as a recommended local backend for multi-model setups
3. **Blog post / README mention**: Write a focused comparison showing FlexLLama's advantages over Ollama for OpenClaw power users who need multiple simultaneous models

#### E. Docker Compose "All-in-One" Template

Create a Docker Compose file that runs both FlexLLama and OpenClaw together:

```yaml
services:
  flexllama:
    image: flexllama-gpu:latest
    build:
      context: ./flexllama
      dockerfile: Dockerfile.cuda
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro
      - ./flexllama-config.json:/app/config.json:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: all

  openclaw:
    image: ghcr.io/openclaw/openclaw:latest
    ports:
      - "18789:18789"
    volumes:
      - ./openclaw-config:/root/.openclaw
    environment:
      - OPENCLAW_GATEWAY_TOKEN=your-token
    depends_on:
      flexllama:
        condition: service_healthy
```

---

## 7. Summary

### Current State

FlexLLama is **already fully compatible** with OpenClaw as a local model provider. No code changes are strictly required. OpenClaw users can configure FlexLLama as a custom provider using `api: "openai-completions"` and point to FlexLLama's `/v1` endpoints.

### FlexLLama's Unique Advantages for OpenClaw

1. **Multi-model, multi-GPU**: Run chat, embedding, and reranking models simultaneously — ideal for OpenClaw's multi-model architecture
2. **Dynamic model switching**: Swap models on a runner without restarting — supports OpenClaw's model selection flexibility
3. **Auto-unload**: Free GPU memory when idle — important for resource-constrained local setups
4. **Lightweight**: Only 2 Python dependencies vs. heavier alternatives
5. **Dashboard**: Visual monitoring of model status complements OpenClaw's Gateway UI

### Recommended Next Steps

| Priority | Action | Effort |
|----------|--------|--------|
| **High** | Add FlexLLama provider docs to OpenClaw (`docs/providers/flexllama.md`) | Low |
| **High** | Add OpenClaw integration guide to FlexLLama docs | Low |
| **Medium** | Create example Docker Compose for FlexLLama + OpenClaw | Low |
| **Medium** | Post in OpenClaw Discord about FlexLLama | Minimal |
| **Low** | Add optional bearer token auth to FlexLLama | Medium |
| **Low** | Enrich `/v1/models` metadata for auto-discovery | Medium |
| **Low** | Add `/v1/models` auto-discovery support (like Ollama/vLLM) | Medium |

### Bottom Line

FlexLLama is the **best local inference manager for OpenClaw users who need multi-model or multi-GPU setups**. While Ollama is simpler for single-model use, FlexLLama excels when you need specialized models (chat + embeddings + reranking) running simultaneously across GPUs — which is exactly what power users of a personal AI assistant like OpenClaw want.
