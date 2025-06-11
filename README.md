<p align="center">
  <img src="static/logo.png" alt="FlexLlama Logo" width="600"/>
</p>

<h1 align="center">FlexLLama - "One to rule them all"</h1>

![FlexLLama](https://img.shields.io/badge/FlexLLama-blue)
![GitHub stars](https://img.shields.io/github/stars/yazon/flexllama?style=social)
![GitHub top language](https://img.shields.io/github/languages/top/yazon/flexllama)
![GitHub repo size](https://img.shields.io/github/repo-size/yazon/flexllama)
![GitHub last commit](https://img.shields.io/github/last-commit/yazon/flexllama?color=red)
![GitHub License](https://img.shields.io/github/license/yazon/flexllama)


**FlexLLama** is a lightweight, extensible, and user-friendly self-hosted tool that easily runs multiple llama.cpp server instances with **OpenAI v1 API compatibility**. It's designed to manage multiple models across different GPUs, making it a powerful solution for local AI development and deployment.

## Key Features of FlexLLama

- 🚀 **Multiple llama.cpp instances** - Run different models simultaneously
- 🎯 **Multi-GPU support** - Distribute models across different GPUs
- 🔌 **OpenAI v1 API compatible** - Drop-in replacement for OpenAI endpoints
- 📊 **Real-time dashboard** - Monitor model status with a web interface
- 🤖 **Chat & Completions** - Full chat and text completion support
- 🔍 **Embeddings & Reranking** - Supports models for embeddings and reranking
- ⚡ **Auto-start** - Automatically start default runners on launch
- 🔄 **Model switching** - Dynamically load/unload models as needed

![FlexLLama Dashboard](static/dashboard.gif)

## Quickstart

1.  **Install FlexLLama:**

    *From GitHub:*
    ```bash
    pip install git+https://github.com/yazon/flexllama.git
    ```

    *From local source (after cloning):*
    ```bash
    # git clone https://github.com/yazon/flexllama.git
    # cd flexllama
    pip install .
    ```

2.  **Create your configuration:**
    Copy the example configuration file to create your own. If you installed from a local clone, you can run:
    ```bash
    cp backend/config_example.json config.json
    ```
    If you installed from git, you may need to download it from the repository.

3.  **Edit `config.json`:**
    Update `config.json` with the correct paths for your `llama-server` binary and your model files (`.gguf`).

4.  **Run FlexLLama:**
    ```bash
    python main.py config.json
    ```
    or
    ```bash
    flexllama config.json
    ```

5.  **Open dashboard:**
    ```
    http://localhost:8080
    ```

## Configuration

Edit `config.json` to configure your runners and models:

### Basic Structure
```json
{
    "auto_start_runners": true,
    "api": {
        "host": "0.0.0.0",
        "port": 8080
    },
    "runner1": {
        "type": "llama-server",
        "path": "/path/to/llama-server",
        "host": "127.0.0.1",
        "port": 8085
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

### Multi-GPU Setup
```json
{
    "runner_gpu0": {
        "path": "/path/to/llama-server",
        "port": 8085
    },
    "runner_gpu1": {
        "path": "/path/to/llama-server", 
        "port": 8086
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

### Key Configuration Options

**Runner Options:**
- `path`: Path to llama-server binary
- `host`/`port`: Where to run this instance
- `extra_args`: Additional arguments for llama-server

**Model Options:**
- `model`: Path to .gguf model file
- `model_alias`: Name to use in API calls
- `embedding`: Set to `true` for embedding models
- `reranking`: Set to `true` for reranking models
- `main_gpu`: Which GPU to use (0, 1, 2...)
- `n_gpu_layers`: How many layers to offload to GPU
- `n_ctx`: Context window size
- `mmproj`: Path to multimodal projection file (for vision models)

## Testing

You can validate your configuration file and run a suite of tests to ensure the application is working correctly.

### Validating `config.json`

To validate your `config.json` file, run `config.py` and provide the path to your configuration file. This will check for correct formatting and required fields.

```bash
python backend/config.py config.json
```
A successful validation will print a confirmation message. If there are errors, they will be displayed with details on how to fix them.

### Running Tests

The `tests/` directory contains scripts for different testing purposes. All test scripts generate detailed logs in the `tests/logs/{session_id}/` directory.

**Prerequisites:**
- For `test_basic.py` and `test_all_models.py`, the main application must be running (`flexllama config.json`).
- For `test_model_switching.py`, the main application should **not** be running.

#### Basic API Tests

`test_basic.py` performs basic checks on the API endpoints to ensure they are responsive.

```bash
# Run basic tests against the default URL (http://localhost:8080)
python tests/test_basic.py
```

**What it tests:**
- `/v1/models` and `/health` endpoints
- `/v1/chat/completions` with both regular and streaming responses
- Concurrent request handling

#### All Models Test

`test_all_models.py` runs a comprehensive test suite against every model defined in your `config.json`.

```bash
# Test all configured models
python tests/test_all_models.py config.json
```

**What it tests:**
- Model loading and health checks
- Chat completions (regular and streaming) for each model
- Response time and error handling

#### Model Switching Test

`test_model_switching.py` verifies the dynamic loading and unloading of models.

```bash
# Run model switching tests
python tests/test_model_switching.py config.json
```

**What it tests:**
- Dynamic model loading and switching
- Runner state management and health monitoring
- Proper cleanup of resources

## License

This project is licensed under the BSD-3-Clause License. See the `LICENSE` file for details.

---

**🚀 Ready to run multiple LLMs like a pro? Edit your `config.json` and start FlexLLama!**
