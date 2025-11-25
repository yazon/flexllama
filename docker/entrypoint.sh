#!/bin/bash

set -e

# Handle Vulkan GPU permissions if running as root
if [[ "$(id -u)" == "0" ]]; then
    echo "Running as root, setting up GPU access permissions..."
    
    # Get GIDs from environment or use defaults
    VIDEO_GID="${HOST_VIDEO_GID:-44}"
    RENDER_GID="${HOST_RENDER_GID:-109}"
    
    # Create groups if they don't exist and add flexllama user to them
    if ! getent group "$VIDEO_GID" >/dev/null 2>&1; then
        groupadd -g "$VIDEO_GID" video_host || true
    fi
    
    if ! getent group "$RENDER_GID" >/dev/null 2>&1 && [ "$RENDER_GID" != "$VIDEO_GID" ]; then
        groupadd -g "$RENDER_GID" render_host || true
    fi
    
    # Add flexllama user to the groups
    usermod -aG "$VIDEO_GID" flexllama 2>/dev/null || true
    if [ "$RENDER_GID" != "$VIDEO_GID" ]; then
        usermod -aG "$RENDER_GID" flexllama 2>/dev/null || true
    fi
    
    # Ensure /dev/dri is accessible
    if [ -d "/dev/dri" ]; then
        chmod -R g+rw /dev/dri 2>/dev/null || true
        echo "✅ GPU device permissions configured"
        ls -la /dev/dri
    else
        echo "⚠️  /dev/dri not found - GPU may not be available"
    fi
    
    # Create logs directory with correct permissions
    mkdir -p /app/logs
    chown -R flexllama:flexllama /app/logs
    
    # Switch to flexllama user and re-execute this script
    echo "Switching to flexllama user..."
    exec gosu flexllama "$0" "$@"
fi

# Default configuration file
DEFAULT_CONFIG="/app/docker/config.json"

# Use environment variable if set, otherwise use default
CONFIG_FILE="${FLEXLLAMA_CONFIG:-$DEFAULT_CONFIG}"

# Validate configuration file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Available configuration files:"
    find /app -name "*.json" -type f | head -10
    exit 1
fi

# Override config values with environment variables if provided
if [[ -n "$FLEXLLAMA_HOST" ]] || [[ -n "$FLEXLLAMA_PORT" ]]; then
    echo "Applying environment variable overrides to configuration..."
    TEMP_CONFIG="/tmp/config.json"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"
    
    if [[ -n "$FLEXLLAMA_HOST" ]]; then
        echo "Setting API host to: $FLEXLLAMA_HOST"
        jq --arg host "$FLEXLLAMA_HOST" '.api.host = $host' "$TEMP_CONFIG" > "$TEMP_CONFIG.tmp" && mv "$TEMP_CONFIG.tmp" "$TEMP_CONFIG"
    fi
    
    if [[ -n "$FLEXLLAMA_PORT" ]]; then
        echo "Setting API port to: $FLEXLLAMA_PORT"
        jq --arg port "$FLEXLLAMA_PORT" '.api.port = ($port | tonumber)' "$TEMP_CONFIG" > "$TEMP_CONFIG.tmp" && mv "$TEMP_CONFIG.tmp" "$TEMP_CONFIG"
    fi
    
    CONFIG_FILE="$TEMP_CONFIG"
fi

# Print startup information
echo "================================================"
echo "FlexLLama Docker Container"
echo "================================================"
echo "Configuration file: $CONFIG_FILE"
echo "llama-server binary: $(which llama-server)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "================================================"

# Check if llama-server is available
if ! command -v llama-server &> /dev/null; then
    echo "Warning: llama-server binary not found in PATH"
    echo "Available binaries:"
    find /usr -name "*llama*" 2>/dev/null | head -5
fi

# Check Vulkan availability if vulkaninfo is present
if command -v vulkaninfo &> /dev/null; then
    echo "Vulkan information:"
    vulkaninfo --summary 2>/dev/null || echo "  (Unable to query Vulkan devices)"
fi

# Note: Running as user flexllama (set by Dockerfile USER directive)

# Start FlexLLama with the appropriate configuration
echo "Starting FlexLLama..."
exec python main.py "$CONFIG_FILE" "$@"