# Multi-stage build for FlexLLama with llama.cpp support

# Stage 1: Build llama.cpp
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
WORKDIR /build

RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llama.cpp \
        -DLLAMA_BUILD_SERVER=ON && \
    cmake --build build --config Release -j$(nproc) && \
    cmake --install build && \
     # Copy all built libraries to a persistent location before the layer ends
     mkdir -p /opt/llama.cpp/lib && \
     find build -name "*.so*" -type f -exec cp -a {} /opt/llama.cpp/lib/ \; && \
     # Specifically ensure libmtmd.so is copied
     find build/tools/mtmd -name "*.so*" -type f -exec cp -a {} /opt/llama.cpp/lib/ \; || true && \
     # List what we have for debugging
     echo "=== Libraries in /opt/llama.cpp/lib ===" && \
     ls -la /opt/llama.cpp/lib/

# Stage 2: Runtime environment
FROM python:3.12.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from the installed location
COPY --from=builder /opt/llama.cpp/ /usr/local/

# Ensure all libraries are properly linked
RUN ldconfig && \
    # Verify libmtmd.so is available
    ldconfig -p | grep mtmd || echo "Warning: libmtmd.so not found in ldconfig" && \
    # Check if llama-server can find all its dependencies
    ldd /usr/local/bin/llama-server || echo "Warning: Some dependencies might be missing"

# Create non-root user
RUN useradd -r -m -s /bin/bash flexllama

# Set working directory
WORKDIR /app

# Copy Python dependencies and install them
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application code
COPY main.py ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY static/ ./static/
COPY docker/ ./docker/

# Copy and setup entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/models /app/config && \
    chown -R flexllama:flexllama /app

# Create default config from template
COPY docker/config.qwen3.unified.json /app/config.json

# Switch to non-root user
USER flexllama

# Expose ports
# 8090: FlexLLama API and dashboard
# 8095-8100: Default ports for llama-server runners
EXPOSE 8090 8095 8096 8097 8098 8099 8100

# Environment variables
ENV FLEXLLAMA_CONFIG=/app/config.json
ENV FLEXLLAMA_HOST=0.0.0.0
ENV FLEXLLAMA_PORT=8090
ENV PYTHONPATH=/app
ENV PATH=/usr/local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD []