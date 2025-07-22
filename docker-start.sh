#!/bin/bash

set -e

# FlexLLama Docker Quick Start Script
echo "================================================"
echo "FlexLLama Docker Setup"
echo "================================================"

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    
    echo "✅ Docker is available and running"
}

# Function to build the image
build_image() {
    echo "🔨 Building FlexLLama Docker image..."
    if [[ "$1" == "--gpu" ]]; then
        echo "   Building with GPU support..."
        docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
        IMAGE_TAG="flexllama-gpu:latest"
    else
        echo "   Building CPU-only version..."
        docker build -t flexllama:latest .
        IMAGE_TAG="flexllama:latest"
    fi
    echo "✅ Image built successfully: $IMAGE_TAG"
}

# Function to create directories
setup_directories() {
    echo "📁 Setting up directories..."
    mkdir -p models logs
    echo "✅ Created models/ and logs/ directories"
}

# Function to show usage examples
show_examples() {
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Place your .gguf model files in the models/ directory."
    echo "   2. Edit docker/config.json to point to your models (set 'n_gpu_layers' > 0 for GPU)."
    echo "   3. Run one of the commands below to start the service."
    echo "   4. Access the dashboard at http://localhost:8080"
    echo ""
    echo "🚀 Usage Examples:"
    echo ""
    echo "1. Run with Docker Compose (Recommended):"
    if [[ "$GPU_OPTION" == "--gpu" ]]; then
        echo "   # Start the GPU service:"
        echo "   docker compose --profile gpu up -d"
    else
        echo "   # Start the CPU service:"
        echo "   docker compose --profile cpu up -d"
    fi
    echo ""
    echo "2. Run with a direct 'docker run' command:"
    if [[ "$GPU_OPTION" == "--gpu" ]]; then
        echo "   docker run -d --gpus all -p 8080:8080 \\"
    else
        echo "   docker run -d -p 8080:8080 \\"
    fi
    echo "     -v \$(pwd)/models:/app/models:ro \\"
    echo "     -v \$(pwd)/docker/config.json:/app/config.json:ro \\"
    echo "     $IMAGE_TAG"
    echo ""
}

# Main script
echo "Checking Docker..."
check_docker

echo ""
echo "Setting up directories..."
setup_directories

echo ""
GPU_OPTION=""
if [[ "$1" == "--gpu" ]] || [[ "$1" == "-g" ]]; then
    GPU_OPTION="--gpu"
    echo "🎮 GPU support requested"
fi

build_image "$GPU_OPTION"

show_examples

echo "================================================"
echo "✅ FlexLLama Docker setup complete!"
echo "================================================"