#!/bin/bash

set -e

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

# Note: Running as user flexllama (set by Dockerfile USER directive)

# Start FlexLLama with the appropriate configuration
echo "Starting FlexLLama..."
exec python main.py "$CONFIG_FILE" "$@"