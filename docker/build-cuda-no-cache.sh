#!/bin/bash
# Script to build the CUDA Docker image with options to bypass cache

# Default values
FORCE_FULL_REBUILD=false
FORCE_LLAMA_REBUILD=false
IMAGE_NAME="flexllama-cuda"
DOCKERFILE="Dockerfile.cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-all)
            FORCE_FULL_REBUILD=true
            shift
            ;;
        --force-llama)
            FORCE_LLAMA_REBUILD=true
            shift
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --force-all       Force rebuild of entire image (no cache)"
            echo "  --force-llama     Force rebuild of llama.cpp only"
            echo "  --image-name      Set custom image name (default: flexllama-cuda)"
            echo "  --dockerfile      Set custom Dockerfile (default: Dockerfile.cuda)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
BUILD_CMD="docker build -f $DOCKERFILE"

if [ "$FORCE_FULL_REBUILD" = true ]; then
    echo "Forcing full rebuild (no cache)..."
    BUILD_CMD="$BUILD_CMD --no-cache"
elif [ "$FORCE_LLAMA_REBUILD" = true ]; then
    echo "Forcing llama.cpp rebuild..."
    # Use build argument to invalidate cache for llama.cpp build
    BUILD_CMD="$BUILD_CMD --build-arg FORCE_REBUILD=$(date +%s)"
fi

BUILD_CMD="$BUILD_CMD -t $IMAGE_NAME ."

echo "Running: $BUILD_CMD"
$BUILD_CMD

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "To verify libmtmd.so is included:"
    echo "  docker run --rm $IMAGE_NAME ls -la /usr/local/lib/libmtmd.so"
    echo ""
    echo "To check all dependencies:"
    echo "  docker run --rm $IMAGE_NAME ldd /usr/local/bin/llama-server"
else
    echo "Build failed!"
    exit 1
fi