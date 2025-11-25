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
- ⏱️ **Auto model unload** - Automatically unload models after a configurable idle timeout

![FlexLLama Dashboard](static/dashboard.gif)

## Quickstart

> **🚀 Want to get started in 5 minutes?** Check out our [**QUICKSTART.md**](QUICKSTART.md) for a simple Docker setup with the Qwen3-4B model!

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
   cp config_example.json config.json
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

FlexLLama can be run using Docker and Docker Compose. We provide profiles for CPU-only, GPU-accelerated (NVIDIA CUDA), and Vulkan GPU environments.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yazon/flexllama.git
   cd flexllama
   ```

After cloning, you can proceed with the quick start script or a manual setup.

______________________________________________________________________

#### Using the Quick Start Script (`docker-start.sh`) - ONE COMMAND SETUP! ✨

The `docker-start.sh` script provides a **fully automated, plug-and-play setup**. Just run ONE command and everything is configured automatically:
- Auto-detects your GPU(s)
- Auto-configures NVIDIA runtime (if needed)
- Builds the correct Docker image
- Starts the container automatically
- **NVIDIA + AMD multi-GPU systems work automatically!**

1. **Make the script executable (Linux/Unix):**

   ```bash
   chmod +x docker-start.sh
   ```

2. **Run ONE command - that's it!**

   *For CPU-only:*
   ```bash
   ./docker-start.sh
   # Windows:
   .\docker-start.ps1
   ```

   *For NVIDIA CUDA GPUs:*
   ```bash
   ./docker-start.sh --gpu=cuda
   # Windows:
   .\docker-start.ps1 -gpu cuda
   ```

   *For Vulkan (AMD/Intel - works with multiple GPUs!):*
   ```bash
   ./docker-start.sh --gpu=vulkan
   # Windows:
   .\docker-start.ps1 -gpu vulkan
   ```

3. **Done! FlexLLama is running automatically!**

   The script automatically:
   - Detects your GPUs
   - Configures NVIDIA runtime (if needed)
   - Builds and starts everything
   
   **Just open:** http://localhost:8090

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
   Use the `--profile` flag to select your environment. The service will be available at `http://localhost:8090`.

   *For CPU-only:*

   ```bash
   docker compose --profile cpu up --build -d
   ```

   *For GPU support (NVIDIA CUDA):*

   ```bash
   docker compose --profile gpu up --build -d
   ```

   *For Vulkan GPU support (AMD/Intel):*

   ```bash
   docker compose --profile vulkan up --build -d
   ```

1. **View Logs**
   To monitor the output of your services, you can view their logs in real-time.

   *For the CPU service:*

   ```bash
   docker compose --profile cpu logs -f
   ```

   *For the GPU service (CUDA):*

   ```bash
   docker compose --profile gpu logs -f
   ```

   *For the Vulkan service:*

   ```bash
   docker compose --profile vulkan logs -f
   ```

   *(Press `Ctrl+C` to stop viewing the logs.)*

1. **(Alternative) Using `docker run`:**
   You can also build and run the containers manually.

   *For CPU-only:*

   ```bash
   # Build the image
   docker build -t flexllama:latest .
   # Run the container
   docker run -d -p 8090:8090 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/docker/config.json:/app/config.json:ro \
     flexllama:latest
   ```

   *For GPU support (NVIDIA CUDA):*

   ```bash
   # Build the image
   docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
   # Run the container
   docker run -d --gpus all -p 8090:8090 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/docker/config.json:/app/config.json:ro \
     flexllama-gpu:latest
   ```

   *For Vulkan GPU support:*

   ```bash
   # Build the image
   docker build -f Dockerfile.vulkan -t flexllama-vulkan:latest .
   # Run the container (AMD/Intel GPUs)
   docker run -d --device /dev/dri:/dev/dri -p 8090:8090 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/docker/config.json:/app/config.json:ro \
     flexllama-vulkan:latest
   ```

1. **Open the dashboard:**
   Access the FlexLLama dashboard in your browser: `http://localhost:8090`

______________________________________________________________________

### Vulkan GPU Support

FlexLLama supports Vulkan-based GPU acceleration for AMD and Intel GPUs on Linux.

**Prerequisites:**
- Linux host with Vulkan drivers installed
- **AMD GPUs**: Mesa RADV drivers (`mesa-vulkan-drivers`) - works immediately
- **Intel GPUs**: Mesa ANV drivers (`mesa-vulkan-drivers`) - works immediately

**Note:** For NVIDIA GPUs, please use the CUDA backend (`--gpu=cuda`).

**Configuration:**

Edit your `docker/config.json` to enable Vulkan:

```json
{
  "runner": "runner1",
  "model": "/app/models/your-model.gguf",
  "model_alias": "vulkan-model",
  "n_gpu_layers": 99,
  "args": "--device Vulkan0"
}
```

You can also use the example configuration:
```bash
cp docker/config.vulkan.json docker/config.json
```

**Troubleshooting:**

Check if Vulkan is working inside the container:
```bash
docker exec -it <container-id> vulkaninfo --summary
```

For AMD ROCm systems, you may need to add `/dev/kfd` device mapping in `docker-compose.yml`.

**Note**: Vulkan support on Windows with Docker is limited. For best results on Windows, use WSL2 with the Linux instructions or use the CUDA backend instead.

## Configuration

FlexLLama is highly configurable through the `config.json` file. You can set up multiple runners, distribute models across GPUs, configure auto-unload timeouts, set environment variables, and much more.

📖 **For detailed configuration options, examples, and advanced setups, see [CONFIGURATION.md](CONFIGURATION.md)**

### Quick Configuration Tips

- Edit `config.json` to add your models and runners
- Use `config_example.json` as a reference
- Validate your configuration: `python backend/config.py config.json`
- Set `auto_start_runners: true` to automatically load models on startup

## Testing

FlexLLama includes a comprehensive test suite to validate your setup and ensure everything is working correctly.

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
