# FlexLLama Docker Guide

This guide covers how to run FlexLLama using Docker, providing an easy way to deploy and manage multiple llama.cpp server instances.

## Quick Start

1. **Build the Docker image:**
   ```bash
   docker build -t flexllama .
   ```

2. **Create necessary directories:**
   ```bash
   mkdir -p models logs
   ```

3. **Copy your models to the models directory:**
   ```bash
   cp /path/to/your/model.gguf models/
   ```

4. **Edit the configuration:**
   ```bash
   cp docker/config.json docker/config.json.local
   # Edit docker/config.json.local to point to your actual models
   ```

5. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

6. **Access FlexLLama:**
   - Dashboard: http://localhost:8080
   - API: http://localhost:8080/v1/models

## Configuration

### Basic Configuration

The Docker setup uses `/app/config.json` as the configuration file. Key differences from standalone setup:

- **llama-server path**: `/usr/local/bin/llama-server` (pre-installed in container)
- **Model paths**: Must be under `/app/models/` (mounted volume)
- **API host**: `0.0.0.0` (to accept external connections)
- **GPU layers**: Set to `0` by default (CPU-only), increase for GPU acceleration

### Example Configuration

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
        "path": "/usr/local/bin/llama-server",
        "host": "127.0.0.1",
        "port": 8085
    },
    "models": [
        {
            "runner": "runner1",
            "model": "/app/models/your-model.gguf",
            "model_alias": "my-model",
            "n_ctx": 4096,
            "n_gpu_layers": 0
        }
    ]
}
```

## Docker Compose Setup

### Standard Setup

```yaml
services:
  flexllama:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./docker/config.json:/app/config.json:ro
      - ./models:/app/models:ro
      - ./logs:/app/logs
```

### GPU-Enabled Setup

For GPU acceleration, you need:

1. **NVIDIA Docker runtime installed**
2. **CUDA-compatible GPU**
3. **Modified configuration with GPU layers**

```yaml
services:
  flexllama:
    build: .
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8080:8080"
    volumes:
      - ./docker/config.json:/app/config.json:ro
      - ./models:/app/models:ro
      - ./logs:/app/logs
```

Update your config.json to use GPU:
```json
{
    "models": [
        {
            "runner": "runner1",
            "model": "/app/models/your-model.gguf",
            "model_alias": "my-model",
            "n_gpu_layers": 99,
            "main_gpu": 0
        }
    ]
}
```

## Volume Mounts

| Host Path | Container Path | Purpose | Required |
|-----------|----------------|---------|----------|
| `./docker/config.json` | `/app/config.json` | Configuration file | Yes |
| `./models` | `/app/models` | Model files (.gguf) | Yes |
| `./logs` | `/app/logs` | Application logs | No |

## Port Mapping

| Port | Purpose |
|------|---------|
| 8080 | FlexLLama API and Dashboard |
| 8085 | Default Runner 1 |
| 8086 | Default Runner 2 |
| 8087-8090 | Additional Runners (optional) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLEXLLAMA_CONFIG` | `/app/config.json` | Path to configuration file |
| `FLEXLLAMA_HOST` | `0.0.0.0` | API server host |
| `FLEXLLAMA_PORT` | `8080` | API server port |
| `CUDA_VISIBLE_DEVICES` | (unset) | GPU selection for CUDA |

## Commands

### Build and Run

```bash
# Build image
docker build -t flexllama .

# Run container
docker run -d \
  --name flexllama \
  -p 8080:8080 \
  -v $(pwd)/docker/config.json:/app/config.json:ro \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  flexllama

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f flexllama

# Stop and remove
docker-compose down
```

### Health Check

```bash
# Check if container is healthy
docker ps

# Test API endpoint
curl http://localhost:8080/health

# Check models
curl http://localhost:8080/v1/models
```

## Troubleshooting

### Common Issues

**1. Container starts but models don't load**
- Check model file paths in config.json
- Ensure models directory is properly mounted
- Verify model files are readable (permissions)

**2. Port already in use**
```bash
# Check what's using the port
lsof -i :8080
# Change port mapping in docker-compose.yml
```

**3. Out of memory errors**
- Reduce model context size (`n_ctx`)
- Decrease GPU layers (`n_gpu_layers`)
- Monitor container memory usage: `docker stats flexllama`

**4. GPU not detected**
- Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Check CUDA_VISIBLE_DEVICES setting
- Ensure GPU support in configuration

### Debugging

```bash
# Enter container shell
docker exec -it flexllama bash

# Check llama-server installation
docker exec flexllama which llama-server

# View container logs
docker logs flexllama

# Check configuration
docker exec flexllama cat /app/config.json

# Test llama-server directly
docker exec flexllama /usr/local/bin/llama-server --help
```

### Performance Optimization

**Memory Management:**
- Set appropriate `n_ctx` based on your needs
- Use `use_mlock: false` in containers
- Monitor memory usage with `docker stats`

**CPU Optimization:**
- Set `n_threads` to match available cores
- Use `--cpus` Docker flag to limit CPU usage

**GPU Optimization:**
- Set `n_gpu_layers` appropriately for your GPU memory
- Use `tensor_split` for multi-GPU setups
- Monitor GPU usage with `nvidia-smi`

## Development

### Local Development

```bash
# Mount source code for development
docker run -it --rm \
  -p 8080:8080 \
  -v $(pwd):/app \
  -v $(pwd)/models:/app/models:ro \
  python:3.12-slim bash

# Inside container
cd /app
pip install -e .
python main.py docker/config.json
```

### Custom Builds

```bash
# CPU-only build (smaller image)
docker build -t flexllama:cpu .

# Build with specific llama.cpp version
docker build --build-arg LLAMA_CPP_TAG=b3259 -t flexllama:custom .
```

## Security Considerations

- Container runs as non-root user `flexllama`
- Model files mounted read-only
- No sensitive data in image layers
- Use secrets management for API keys if needed
- Network isolation with Docker networks

## Monitoring

### Health Checks

The container includes built-in health checks:
```bash
# Check health status
docker inspect flexllama | grep Health -A 10
```

### Metrics Collection

```bash
# Resource usage
docker stats flexllama

# Application metrics (if enabled)
curl http://localhost:8080/metrics
```