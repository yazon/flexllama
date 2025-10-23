# FlexLLama Quickstart - Run Qwen3-4B in 5 Minutes

Get FlexLLama running with the Qwen3-4B-Instruct model in just a few commands using Docker.

## Prerequisites

- Docker and Docker Compose installed
- ~3GB free disk space (2.5GB for model + build cache)
- Internet connection for model download

## Linux/macOS

```bash
# 1. Clone the repository
git clone https://github.com/yazon/flexllama.git
cd flexllama

# 2. Download the Qwen3-4B model (2.5GB)
mkdir -p models
wget -O models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true"

# 3. Use the CPU-optimized configuration
cp docker/config.qwen3.cpu.json docker/config.json

# 4. Start FlexLLama with Docker Compose
docker compose --profile cpu up --build -d

# 5. Verify it's working
curl http://localhost:8090/health
curl http://localhost:8090/v1/models
```

## Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/yazon/flexllama.git
Set-Location flexllama

# 2. Download the Qwen3-4B model (2.5GB)
New-Item -ItemType Directory -Force -Path models | Out-Null
curl.exe -L "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf?download=true" -o "models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# 3. Use the CPU-optimized configuration
Copy-Item docker\config.qwen3.cpu.json docker\config.json -Force

# 4. Start FlexLLama with Docker Compose
docker compose --profile cpu up --build -d

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
    "model": "qwen3-4b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hello! Please introduce yourself in one sentence."}],
    "stream": false
  }'
```

## Access the Dashboard

Open your browser and go to: **http://localhost:8090**

The dashboard provides a web interface to interact with your models and monitor system status.

## View Logs

To see what's happening:

```bash
# View logs
docker compose --profile cpu logs -f
```

## Stop FlexLLama

```bash
docker compose down
```

## Troubleshooting

**Port already in use**: If port 8090 is busy, edit `docker-compose.yml` and change `8090:8080` to another port like `8091:8080`.

**Slow startup**: First-time model loading can take 1-2 minutes.

**Model not found**: Ensure the model file exists at `models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

**Out of memory**: The Qwen3-4B Q4_K_M model requires ~3-4GB RAM. Reduce `n_ctx` in the config if needed.

## Next Steps

- Explore the [full documentation](README.md) for advanced configuration
- Add more models to your `docker/config.json`
- Try GPU acceleration with `--profile gpu`

---

**Model Reference**: [Qwen3-4B-Instruct-2507-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/blob/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) by Unsloth
