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

# Global flag to track if we need to restart containers
NVIDIA_JUST_CONFIGURED=false

# Function to check and configure NVIDIA Docker runtime (FULLY AUTOMATIC)
check_nvidia_runtime() {
    # Only check if nvidia-container-toolkit is installed
    if ! command -v nvidia-ctk &> /dev/null && ! command -v nvidia-container-cli &> /dev/null; then
        return 0  # Not installed, skip check
    fi

    echo ""
    echo "🔍 Checking NVIDIA Docker runtime configuration..."
    
    # Check if Docker daemon.json has NVIDIA runtime configured
    if docker info 2>/dev/null | grep -qi "nvidia"; then
        echo "✅ NVIDIA runtime already configured in Docker"
        return 0
    fi
    
    echo "⚠️  NVIDIA Container Toolkit detected - configuring Docker runtime..."
    echo "   (This is needed for NVIDIA GPUs to work in containers)"
    echo ""
    
    # AUTO-CONFIGURE (requires sudo)
    echo "🔧 Auto-configuring NVIDIA runtime (requires sudo)..."
    if sudo nvidia-ctk runtime configure --runtime=docker; then
        echo "✅ NVIDIA runtime configured"
        echo ""
        echo "🔄 Restarting Docker service..."
        if sudo systemctl restart docker; then
            echo "✅ Docker restarted successfully"
            sleep 3  # Give Docker time to fully start
            
            # Verify configuration worked
            if docker info 2>/dev/null | grep -qi "nvidia"; then
                echo "✅ NVIDIA runtime verified and working"
                NVIDIA_JUST_CONFIGURED=true
                return 0
            fi
        fi
    fi
    
    # If we get here, auto-config might have failed but let's try to continue anyway
    echo "⚠️  Auto-configuration completed, continuing with setup..."
    echo "   If NVIDIA GPU doesn't work, you may need to restart Docker manually"
    return 0
}

# Function to parse GPU backend argument
parse_gpu_backend() {
    local GPU_BACKEND=""

    # Check for --gpu=<backend> syntax
    if [[ "$1" =~ ^--gpu= ]]; then
        GPU_BACKEND="${1#*=}"
        shift
    # Check for --gpu <backend> syntax
    elif [[ "$1" == "--gpu" ]]; then
        if [[ -n "$2" && "$2" != -* ]]; then
            GPU_BACKEND="$2"
            shift 2
        else
            GPU_BACKEND="cuda"  # Default to CUDA if no value
            shift
        fi
    # Check for legacy -g flag
    elif [[ "$1" == "-g" ]]; then
        GPU_BACKEND="cuda"
        shift
    fi

    # Normalize backend
    case "$GPU_BACKEND" in
        cuda|CUDA)
            GPU_BACKEND="cuda"
            ;;
        vulkan|VULKAN)
            GPU_BACKEND="vulkan"
            ;;
        cpu|CPU|"")
            GPU_BACKEND="cpu"  # CPU mode
            ;;
        *)
            echo "❌ Unknown GPU backend: $GPU_BACKEND"
            echo "   Valid options: cpu, cuda, vulkan"
            exit 1
            ;;
    esac

    echo "$GPU_BACKEND"
}

# Function to build the image
build_image() {
    local backend="$1"
    echo "🔨 Building FlexLLama Docker image..."

    case "$backend" in
        cuda)
            echo "   Building with CUDA support..."
            docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
            IMAGE_TAG="flexllama-gpu:latest"
            ;;
        vulkan)
            echo "   Building with Vulkan support..."
            docker build -f Dockerfile.vulkan -t flexllama-vulkan:latest .
            IMAGE_TAG="flexllama-vulkan:latest"
            ;;
        cpu)
            echo "   Building CPU-only version..."
            docker build -t flexllama:latest .
            IMAGE_TAG="flexllama:latest"
            ;;
    esac

    echo "✅ Image built successfully: $IMAGE_TAG"
}

# Function to create directories
setup_directories() {
    echo "📁 Setting up directories..."
    mkdir -p models logs
    echo "✅ Created models/ and logs/ directories"
}

# Function to setup environment for Vulkan
setup_vulkan_env() {
    echo "🌋 Setting up Vulkan environment..."
    
    # Detect GPU vendors
    local HAS_AMD=false
    local HAS_INTEL=false
    
    if command -v lspci &> /dev/null; then
        if lspci | grep -iE "amd.*(vga|3d|display)" || lspci | grep -qi "radeon\|advanced micro devices"; then
            HAS_AMD=true
            echo "   Detected: AMD GPU"
        fi
        if lspci | grep -iE "intel.*(vga|3d|display|graphics)"; then
            HAS_INTEL=true
            echo "   Detected: Intel GPU"
        fi
    fi
    
    # Detect video group GID
    VIDEO_GID=$(getent group video | cut -d: -f3 2>/dev/null || echo "44")
    
    # Detect render group GID (from /dev/dri/renderD128 if it exists)
    if [ -e "/dev/dri/renderD128" ]; then
        RENDER_GID=$(stat -c '%g' /dev/dri/renderD128 2>/dev/null || echo "")
    fi
    
    # Fallback: try to get render group from system
    if [ -z "$RENDER_GID" ]; then
        RENDER_GID=$(getent group render | cut -d: -f3 2>/dev/null || echo "")
    fi
    
    # If render GID still not found, use video GID
    if [ -z "$RENDER_GID" ]; then
        RENDER_GID="$VIDEO_GID"
        echo "⚠️  Render group not found, using video GID: $VIDEO_GID"
    fi
    
    # Create or update .env file
    cat > .env << EOF
# Auto-generated by docker-start.sh for Vulkan support
# These GIDs are detected from your host system
VIDEO_GID=$VIDEO_GID
RENDER_GID=$RENDER_GID
EOF
    
    echo "✅ Created .env file with VIDEO_GID=$VIDEO_GID, RENDER_GID=$RENDER_GID"
    
    # Verify /dev/dri exists
    if [ ! -d "/dev/dri" ]; then
        echo "⚠️  WARNING: /dev/dri directory not found!"
        echo "   Your system may not have GPU devices or drivers installed."
        echo "   Vulkan support may not work correctly."
    else
        echo "✅ /dev/dri directory found"
        ls -la /dev/dri 2>/dev/null | grep -E "render|card" || echo "   No GPU devices detected"
    fi
}

# Function to show usage examples
show_examples() {
    local backend="$1"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Place your .gguf model files in the models/ directory."
    echo "   2. Edit docker/config.json:"

    case "$backend" in
        cpu)
            echo "      - For CPU: set 'n_gpu_layers': 0"
            ;;
        cuda)
            echo "      - For CUDA: set 'n_gpu_layers': 99, ensure CUDA is available"
            ;;
        vulkan)
            echo "      - For Vulkan: set 'n_gpu_layers': 99, add 'args': '--device Vulkan0'"
            echo "      - Note: .env file with GPU group IDs has been auto-generated"
            ;;
    esac

    echo "   3. Run one of the commands below to start the service."
    echo "   4. Access the dashboard at http://localhost:8090"
    echo ""
    echo "🚀 Usage Examples:"
    echo ""

    case "$backend" in
        cpu)
            echo "1. Docker Compose (Recommended):"
            echo "   docker compose --profile cpu up -d"
            echo ""
            echo "2. Direct docker run:"
            echo "   docker run -d -p 8090:8090 \\"
            echo "     -v \$(pwd)/models:/app/models:ro \\"
            echo "     -v \$(pwd)/docker/config.json:/app/config.json:ro \\"
            echo "     $IMAGE_TAG"
            ;;
        cuda)
            echo "1. Docker Compose (Recommended):"
            echo "   docker compose --profile gpu up -d"
            echo ""
            echo "2. Direct docker run:"
            echo "   docker run -d --gpus all -p 8090:8090 \\"
            echo "     -v \$(pwd)/models:/app/models:ro \\"
            echo "     -v \$(pwd)/docker/config.json:/app/config.json:ro \\"
            echo "     $IMAGE_TAG"
            ;;
        vulkan)
            echo "1. Docker Compose (Recommended - works for all GPUs):"
            echo "   docker compose --profile vulkan up -d"
            echo ""
            echo "   Note: The Vulkan profile now supports:"
            echo "   ✅ Intel GPUs (integrated/Arc) - works out of the box"
            echo "   ✅ AMD GPUs (all models) - works out of the box"
            echo ""
            echo "2. Direct docker run:"
            echo ""
            echo "   # For AMD/Intel GPUs:"
            echo "   docker run -d --device /dev/dri:/dev/dri -p 8090:8090 \\"
            echo "     -v \$(pwd)/models:/app/models:ro \\"
            echo "     -v \$(pwd)/docker/config.json:/app/config.json:ro \\"
            echo "     $IMAGE_TAG"
            ;;
    esac

    echo ""
}

# Main script
echo "Checking Docker..."
check_docker

echo ""
echo "Setting up directories..."
setup_directories

echo ""
# Parse GPU backend from arguments
GPU_BACKEND=$(parse_gpu_backend "$@")

case "$GPU_BACKEND" in
    cuda)
        echo "🎮 CUDA GPU support requested"
        check_nvidia_runtime
        ;;
    vulkan)
        echo "🌋 Vulkan GPU support requested"
        setup_vulkan_env
        ;;
    cpu)
        echo "💻 CPU-only mode"
        ;;
esac

build_image "$GPU_BACKEND"

echo ""
echo "🚀 Starting FlexLLama container..."

# Determine which profile to use
case "$GPU_BACKEND" in
    cuda)
        PROFILE="gpu"
        ;;
    vulkan)
        PROFILE="vulkan"
        ;;
    cpu)
        PROFILE="cpu"
        ;;
esac

# Start the container
if docker compose --profile "$PROFILE" up -d; then
    echo "✅ Container started successfully!"
    echo ""
    
    # If NVIDIA was just configured, show a note about GPU detection
    if [ "$NVIDIA_JUST_CONFIGURED" = true ]; then
        echo "🎮 NVIDIA GPU support has been enabled!"
        echo "   Your NVIDIA GPU should now be visible in the container."
        echo ""
    fi
    
    # Wait a moment for container to initialize
    sleep 3
    
    # Show container status
    echo "📊 Container Status:"
    docker ps --filter "name=flexllama" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    echo "================================================"
    echo "✅ FlexLLama is ready!"
    echo "================================================"
    echo ""
    echo "🌐 Access the dashboard: http://localhost:8090"
    echo ""
    
    # Offer to verify setup for Vulkan
    if [ "$GPU_BACKEND" = "vulkan" ]; then
        echo "💡 Tip: Run './verify-vulkan-setup.sh' to verify GPU detection"
        echo ""
    fi
else
    echo "⚠️  Failed to start container"
    echo "   Try manually: docker compose --profile $PROFILE up -d"
fi

echo ""
echo "================================================"
echo "✅ FlexLLama Docker setup complete!"
echo "================================================"
