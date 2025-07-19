# Docker Implementation Plan for FlexLLama

## Project Analysis

**FlexLLama** is a Python application that:
- Runs multiple llama.cpp server instances
- Provides OpenAI v1 API compatibility
- Has a web dashboard for monitoring
- Supports multi-GPU setups
- Requires external `llama-server` binary from llama.cpp
- Uses port 8080 by default (configurable)
- Depends on Python 3.12+ with aiohttp and psutil

## Docker Implementation Strategy

### 1. Docker Image Architecture

**Base Image**: Use `python:3.12-slim` for lightweight Python runtime

**Key Components to Include**:
- Python application code
- llama.cpp binary (compiled for the container architecture)
- Frontend static files
- Default configuration template

### 2. Multi-Stage Build

**Stage 1: Build llama.cpp**
- Use Ubuntu/Debian base with build tools
- Clone and compile llama.cpp with CUDA support (optional)
- Build llama-server binary

**Stage 2: Runtime**
- Copy compiled llama-server binary
- Install Python dependencies
- Copy FlexLLama application code
- Set up runtime environment

### 3. Directory Structure in Container

```
/app/
├── main.py
├── backend/
├── frontend/
├── static/
├── config.json (mounted volume)
├── models/ (mounted volume)
└── logs/ (mounted volume)
/usr/local/bin/
└── llama-server (compiled binary)
```

### 4. Port Exposure

- **Primary Port**: 8080 (FlexLLama API and dashboard)
- **Runner Ports**: 8085-8090 (configurable, for llama-server instances)

### 5. Volume Mounts

**Required Volumes**:
- `/app/config.json` - Configuration file
- `/app/models/` - Model files (.gguf)
- `/app/logs/` - Application logs

**Optional Volumes**:
- `/tmp` - Temporary files and caching

### 6. Environment Variables

- `FLEXLLAMA_CONFIG` - Path to config file (default: `/app/config.json`)
- `FLEXLLAMA_HOST` - API host (default: `0.0.0.0`)
- `FLEXLLAMA_PORT` - API port (default: `8080`)
- `CUDA_VISIBLE_DEVICES` - GPU selection for CUDA builds

### 7. Docker Compose Setup

**Services**:
- `flexllama` - Main application container
- Optional: separate containers for different GPU assignments

### 8. Implementation Files

**Files to Create**:
1. `Dockerfile` - Multi-stage build
2. `docker-compose.yml` - Complete setup with volumes
3. `docker/config.json` - Docker-optimized configuration
4. `docker/README.md` - Docker usage instructions
5. `.dockerignore` - Exclude unnecessary files

### 9. Build Variants

**CPU-only Build**:
- Smaller image size
- No CUDA dependencies
- Suitable for development/testing

**CUDA Build**:
- Include NVIDIA runtime
- CUDA-compiled llama-server
- GPU acceleration support

### 10. Testing Strategy

1. Build and run container with sample config
2. Test API endpoints (`/health`, `/v1/models`)
3. Test model loading with small test model
4. Verify dashboard accessibility
5. Test volume mounting and persistence

### 11. Documentation Updates

- Add Docker section to main README.md
- Create docker-specific configuration examples
- Add troubleshooting guide for common Docker issues

## Implementation Steps

1. Create multi-stage Dockerfile
2. Create Docker Compose configuration
3. Create Docker-optimized config template
4. Add .dockerignore file
5. Test build and functionality
6. Update documentation
7. Create usage examples

## Implementation Results

### ✅ Completed Tasks

1. **Docker Image Creation**:
   - Multi-stage build with llama.cpp compilation ✓
   - Optimized image size: ~181MB ✓
   - Non-root user security ✓

2. **Configuration Management**:
   - Environment variable support ✓
   - Multiple config templates (CPU/GPU) ✓
   - Enhanced entrypoint script with diagnostics ✓

3. **Docker Compose Setup**:
   - Multi-port exposure for runners ✓
   - Volume mounts for models and logs ✓
   - Health checks and restart policies ✓

4. **Testing & Validation**:
   - Build verification ✓
   - Startup testing ✓
   - Configuration override testing ✓
   - Port exposure validation ✓

### 📋 Usage Summary

**Quick Start**:
```bash
# Build and run
docker build -t flexllama .
docker run -p 8080:8080 flexllama

# Using Docker Compose
docker-compose up -d
```

**Configuration Options**:
- CPU-only: Use default `docker/config.json`
- GPU-enabled: Use `docker/config-gpu.json` with `Dockerfile.cuda`
- Custom ports: Set `FLEXLLAMA_PORT` environment variable

## Considerations

- **Model Size**: Large models require significant disk space and memory
- **GPU Support**: NVIDIA Docker runtime required for GPU acceleration
- **Performance**: Container overhead minimal for this use case
- **Security**: Run as non-root user, limit container privileges
- **Networking**: Ensure proper port mapping and internal communication