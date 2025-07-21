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
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DLLAMA_SERVER=ON -DLLAMA_CURL=OFF && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-server /usr/local/bin/

# Stage 2: Runtime environment
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy llama-server binary from builder stage
COPY --from=builder /usr/local/bin/llama-server /usr/local/bin/

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
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/config && \
    chown -R flexllama:flexllama /app

# Create default config template
COPY docker/config.json /app/config/config.json.template

# Switch to non-root user
USER flexllama

# Expose ports
# 8080: FlexLLama API and dashboard
# 8085-8090: Default ports for llama-server runners
EXPOSE 8080 8085 8086 8087 8088 8089 8090

# Environment variables
ENV FLEXLLAMA_CONFIG=/app/config.json
ENV FLEXLLAMA_HOST=0.0.0.0
ENV FLEXLLAMA_PORT=8080
ENV PYTHONPATH=/app
ENV PATH=/usr/local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD []