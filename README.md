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

### 📦 Local Installation

1. **Install FlexLLama:**

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

1. **Create your configuration:**
   Copy the example configuration file to create your own. If you installed from a local clone, you can run:

   ```bash
   cp backend/config_example.json config.json
   ```

   If you installed from git, you may need to download it from the repository.

1. **Edit `config.json`:**
   Update `config.json` with the correct paths for your `llama-server` binary and your model files (`.gguf`).

1. **Run FlexLLama:**

   ```bash
   python main.py config.json
   ```

   or

   ```bash
   flexllama config.json
   ```

1. **Open dashboard:**

   ```
   http://localhost:8080
   ```

### 🐳 Docker

FlexLLama can be run using Docker and Docker Compose. We provide profiles for both CPU-only and GPU-accelerated (NVIDIA CUDA) environments.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yazon/flexllama.git
   cd flexllama
   ```

After cloning, you can proceed with the quick start script or a manual setup.

______________________________________________________________________

#### Using the Quick Start Script (`docker-start.sh`)

For an easier start, the `docker-start.sh` helper script automates several setup steps. It checks your Docker environment, builds the correct image (CPU or GPU) and provides the commands to launch FlexLLama.

1. **Make the script executable (Linux/Unix):**

   ```bash
   chmod +x docker-start.sh
   ```

1. **Run the script:**
   Use the `--gpu` flag for NVIDIA GPU support.

   *For CPU-only setup:*

   ```bash
   ./docker-start.sh
   ```

   or

   ```bash
   ./docker-start.ps1
   ```

   *For GPU-accelerated setup:*

   ```bash
   ./docker-start.sh --gpu
   ```

   or

   ```bash
   ./docker-start.ps1 -gpu
   ```

1. **Follow the on-screen instructions:**
   The script will guide you.

______________________________________________________________________

**Manual Docker and Docker Compose Setup**

If you prefer to run the steps manually, follow this guide:

1. **Place your models:**

   ```bash
   # Create the models directory if it doesn't exist
   mkdir -p models
   # Copy your .gguf model files into it
   cp /path/to/your/model.gguf models/
   ```

1. **Configure your models:**

   ```bash
   # Edit the Docker configuration to point to your models
   #   • CPU-only: keep "n_gpu_layers": 0
   #   • GPU: set "n_gpu_layers" to e.g. 99 and specify "main_gpu": 0
   ```

1. **Build and Start FlexLLama with Docker Compose (Recommended):**
   Use the `--profile` flag to select your environment. The service will be available at `http://localhost:8080`.

   *For CPU-only:*

   ```bash
   docker compose --profile cpu up --build -d
   ```

   *For GPU support (NVIDIA CUDA):*

   ```bash
   docker compose --profile gpu up --build -d
   ```

1. **View Logs**
   To monitor the output of your services, you can view their logs in real-time.

   *For the CPU service:*

   ```bash
   docker compose --profile cpu logs -f
   ```

   *For the GPU service:*

   ```bash
   docker compose --profile gpu logs -f
   ```

   *(Press `Ctrl+C` to stop viewing the logs.)*

1. **(Alternative) Using `docker run`:**
   You can also build and run the containers manually.

   *For CPU-only:*

   ```bash
   # Build the image
   docker build -t flexllama:latest .
   # Run the container
   docker run -d -p 8080:8080 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/docker/config.json:/app/config.json:ro \
     flexllama:latest
   ```

   *For GPU support (NVIDIA CUDA):*

   ```bash
   # Build the image
   docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
   # Run the container
   docker run -d --gpus all -p 8080:8080 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/docker/config.json:/app/config.json:ro \
     flexllama-gpu:latest
   ```

1. **Open the dashboard:**
   Access the FlexLLama dashboard in your browser: `http://localhost:8080`

## Configuration

Edit `config.json` to configure your runners and models:

### Basic Structure

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
- `extra_args`: Additional arguments for llama-server (applied to all models using this runner)

**Model Options:**

*Core Settings:*

- `runner`: Which runner to use for this model
- `model`: Path to .gguf model file
- `model_alias`: Name to use in API calls

*Model Types:*

- `embedding`: Set to `true` for embedding models
- `reranking`: Set to `true` for reranking models
- `mmproj`: Path to multimodal projection file (for vision models)

*Performance & Memory:*

- `n_ctx`: Context window size (e.g., 4096, 8192, 32768)
- `n_batch`: Batch size for processing (e.g., 256, 512)
- `n_threads`: Number of CPU threads to use
- `main_gpu`: Which GPU to use (0, 1, 2...)
- `n_gpu_layers`: How many layers to offload to GPU (99 for all layers)
- `tensor_split`: Array defining how to split model across GPUs (e.g., [1.0, 0.0])
- `offload_kqv`: Whether to offload key-value cache to GPU (`true`/`false`)
- `use_mlock`: Lock model in RAM to prevent swapping (`true`/`false`)

*Optimization:*

- `flash_attn`: Enable flash attention for faster processing (`true`/`false`)
- `split_mode`: How to split model layers ("row" or other modes)
- `cache-type-k`: Key cache quantization type (e.g., "q8_0")
- `cache-type-v`: Value cache quantization type (e.g., "q8_0")

*Chat & Templates:*

- `chat_template`: Chat template format (e.g., "mistral-instruct", "gemma")
- `jinja`: Enable Jinja templating (`true`/`false`)

*Advanced Options:*

- `rope-scaling`: RoPE scaling method (e.g., "linear")
- `rope-scale`: RoPE scaling factor (e.g., 2)
- `yarn-orig-ctx`: Original context size for YaRN scaling
- `pooling`: Pooling method for embeddings (e.g., "cls")
- `args`: Additional custom arguments to pass directly to llama-server for this specific model (string, e.g., "--custom-flag --param value"). These are applied after all other model parameters and before runner `extra_args`.

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

______________________________________________________________________

**🚀 Ready to run multiple LLMs like a pro? Edit your `config.json` and start FlexLLama!**
