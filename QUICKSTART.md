# FlexLLama Quickstart - Run Qwen3-4B in 5 Minutes

Get FlexLLama running with the Qwen3-4B-Instruct model in just a few commands using Docker.

## Prerequisites

- Docker and Docker Compose installed
- ~3GB free disk space (2.5GB for model + build cache)
- Internet connection for model download
- For GPU acceleration: CUDA-capable GPU (NVIDIA) or Vulkan-capable GPU (AMD/Intel)

## Quick Setup

**Linux/macOS:**

```bash
# 1. Clone the repository
git clone https://github.com/yazon/flexllama.git
cd flexllama

# 2. Download the Qwen3-4B model (2.5GB)
mkdir -p models
wget -O models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true"

# 3. Use the unified configuration
cp docker/config.qwen3.unified.json docker/config.json

# 4. Start FlexLLama - choose your backend:
./docker-start.sh              # CPU (default)
./docker-start.sh --gpu=cuda   # CUDA GPU (NVIDIA)
./docker-start.sh --gpu=vulkan # Vulkan GPU (AMD/Intel)

# 5. Verify it's working
curl http://localhost:8090/health
curl http://localhost:8090/v1/models
```

**Windows (PowerShell):**

```powershell
# 1. Clone the repository
git clone https://github.com/yazon/flexllama.git
Set-Location flexllama

# 2. Download the Qwen3-4B model (2.5GB)
New-Item -ItemType Directory -Force -Path models | Out-Null
curl.exe -L "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true" -o "models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# 3. Use the unified configuration
Copy-Item docker\config.qwen3.unified.json docker\config.json -Force

# 4. Start FlexLLama - choose your backend:
.\docker-start.ps1             # CPU (default)
.\docker-start.ps1 -gpu cuda   # CUDA GPU (NVIDIA)
.\docker-start.ps1 -gpu vulkan # Vulkan GPU (AMD/Intel)

# 5. Verify it's working
curl http://localhost:8090/health
curl http://localhost:8090/v1/models
```

## Test Chat Completion

Once FlexLLama is running, test it with a simple chat request:

```bash
curl -s http://localhost:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b-instruct-q4_k_m-cpu",
    "messages": [{"role": "user", "content": "Hello! Please introduce yourself in one sentence."}],
    "stream": false
  }'
```

**Model Aliases:** Use the alias matching your backend:
- `qwen3-4b-instruct-q4_k_m-cpu` (CPU)
- `qwen3-4b-instruct-q4_k_m-cuda` (CUDA GPU)
- `qwen3-4b-instruct-q4_k_m-vulkan` (Vulkan GPU)

## Access the Dashboard

Open your browser and go to: **http://localhost:8090**

The dashboard provides a web interface to interact with your models and monitor system status.

## View Logs

To see what's happening:

```bash
# View logs for all running services
docker compose logs -f
```

## Stop FlexLLama

```bash
docker compose down
```

## Troubleshooting

**Port already in use**: If port 8090 is busy, edit `docker-compose.yml` and change `8090:8090` to another port like `8091:8090`.

**Slow startup**: First-time model loading can take 1-2 minutes.

**Model not found**: Ensure the model file exists at `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

**Out of memory**: The Qwen3-4B Q4_K_M model requires ~3-4GB RAM. Reduce `n_ctx` in the config if needed.

## Backend Options

FlexLLama supports three backends with the **same setup** - just change the startup parameter:

| Backend | Command (Linux/macOS) | Command (Windows) | Requirements |
|---------|----------------------|-------------------|--------------|
| **CPU** | `./docker-start.sh` | `.\docker-start.ps1` | None |
| **CUDA** | `./docker-start.sh --gpu=cuda` | `.\docker-start.ps1 -gpu cuda` | NVIDIA GPU, drivers, nvidia-container-toolkit* |
| **Vulkan** | `./docker-start.sh --gpu=vulkan` | `.\docker-start.ps1 -gpu vulkan` | AMD: `mesa-vulkan-drivers`<br>Intel: `mesa-vulkan-drivers` |

*The setup script will help configure nvidia-container-toolkit automatically.

## Switching Backends

To switch between CPU and GPU, just restart with a different parameter:

```bash
# Stop current backend
docker compose down

# Start with different backend (Linux/macOS)
./docker-start.sh --gpu=cuda  # or --gpu=vulkan, or no flag for CPU

# Or use Docker Compose directly
docker compose --profile cpu up -d     # CPU
docker compose --profile gpu up -d     # CUDA GPU
docker compose --profile vulkan up -d  # Vulkan GPU
```

## Next Steps

- Explore the [full documentation](README.md) for advanced configuration
- Add more models to your `docker/config.json`
- All three backends are pre-configured in `docker/config.qwen3.unified.json`
- Try the dashboard at http://localhost:8090 for a web interface

---

**Model Reference**: [Qwen3-4B-Instruct-2507-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/blob/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) by Unsloth
