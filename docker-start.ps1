# FlexLLama Docker Quick Start Script for Windows PowerShell
# ================================================

param(
    [string]$gpu = ""
)

# Error configuration
$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "FlexLLama Docker Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Function to check if Docker is running
function Test-Docker {
    try {
        # Check if Docker is installed
        $dockerVersion = docker --version 2>$null
        if (-not $dockerVersion) {
            Write-Host "Docker is not installed. Please install Docker first." -ForegroundColor Red
            exit 1
        }

        # Check if Docker is running using a simpler command
        $null = docker ps --quiet 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Docker is not running. Please start Docker first." -ForegroundColor Red
            exit 1
        }

        Write-Host "Docker is available and running" -ForegroundColor Green
    }
    catch {
        Write-Host "Error checking Docker: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Function to check NVIDIA Docker runtime (Windows)
function Test-NvidiaRuntime {
    # Check if nvidia-container-toolkit is available
    $nvidiaCtk = Get-Command nvidia-ctk -ErrorAction SilentlyContinue
    
    if (-not $nvidiaCtk) {
        return  # Not installed, skip check
    }

    Write-Host ""
    Write-Host "Checking NVIDIA Docker runtime configuration..." -ForegroundColor Yellow
    
    # Check if Docker has NVIDIA runtime
    $dockerInfo = docker info 2>$null | Out-String
    if ($dockerInfo -match "nvidia") {
        Write-Host "NVIDIA runtime already configured in Docker" -ForegroundColor Green
        return
    }
    
    Write-Host "NVIDIA Container Toolkit is installed but Docker runtime not configured" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "For NVIDIA GPU support, you may need to configure Docker Desktop:" -ForegroundColor Yellow
    Write-Host "  1. Open Docker Desktop Settings" -ForegroundColor White
    Write-Host "  2. Go to 'Resources' -> 'WSL Integration'" -ForegroundColor White
    Write-Host "  3. Enable WSL2 integration for your distros" -ForegroundColor White
    Write-Host "  4. Or use Docker in WSL2 directly for better GPU support" -ForegroundColor White
    Write-Host ""
}

# Function to parse and normalize GPU backend
function Get-GPUBackend {
    param([string]$backend)

    # Normalize backend
    $normalizedBackend = switch ($backend.ToLower()) {
        "cuda"   { "cuda" }
        "vulkan" { "vulkan" }
        ""       { "cpu" }
        default {
            Write-Host "Unknown GPU backend: $backend" -ForegroundColor Red
            Write-Host "Valid options: cuda, vulkan" -ForegroundColor Yellow
            exit 1
        }
    }

    return $normalizedBackend
}

# Function to build the image
function Build-DockerImage {
    param([string]$backend)

    Write-Host "Building FlexLLama Docker image..." -ForegroundColor Yellow

    switch ($backend) {
        "cuda" {
            Write-Host "   Building with CUDA support..." -ForegroundColor Yellow
            docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
            $script:IMAGE_TAG = "flexllama-gpu:latest"
        }
        "vulkan" {
            Write-Host "   Building with Vulkan support..." -ForegroundColor Yellow
            docker build -f Dockerfile.vulkan -t flexllama-vulkan:latest .
            $script:IMAGE_TAG = "flexllama-vulkan:latest"
        }
        "cpu" {
            Write-Host "   Building CPU-only version..." -ForegroundColor Yellow
            docker build -t flexllama:latest .
            $script:IMAGE_TAG = "flexllama:latest"
        }
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Image built successfully: $IMAGE_TAG" -ForegroundColor Green
    }
    else {
        Write-Host "Error building Docker image" -ForegroundColor Red
        exit 1
    }
}

# Function to create directories
function New-Directories {
    Write-Host "Setting up directories..." -ForegroundColor Yellow
    
    try {
        if (-not (Test-Path "models")) {
            New-Item -ItemType Directory -Path "models" -Force | Out-Null
        }
        if (-not (Test-Path "logs")) {
            New-Item -ItemType Directory -Path "logs" -Force | Out-Null
        }
        Write-Host "Created models/ and logs/ directories" -ForegroundColor Green
    }
    catch {
        Write-Host "Error creating directories: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Function to setup Vulkan environment
function Setup-VulkanEnv {
    Write-Host "Setting up Vulkan environment..." -ForegroundColor Yellow
    
    # On Windows, we can't easily detect the Linux GIDs for the Docker VM
    # So we will use the defaults that are common in many Linux distros
    $VIDEO_GID = "44"
    $RENDER_GID = "109"
    
    $envContent = @"
# Auto-generated by docker-start.ps1 for Vulkan support
# On Windows/WSL, these are set to defaults as host detection is limited from PowerShell
VIDEO_GID=$VIDEO_GID
RENDER_GID=$RENDER_GID
"@
    
    try {
        $envContent | Out-File -FilePath ".env" -Encoding ASCII
        Write-Host "Created .env file with defaults (VIDEO_GID=$VIDEO_GID, RENDER_GID=$RENDER_GID)" -ForegroundColor Green
    }
    catch {
        Write-Host "Warning: Could not create .env file. Docker Compose might use defaults." -ForegroundColor Yellow
    }
}

# Function to show usage examples
function Show-UsageExamples {
    param([string]$backend)

    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Place your .gguf model files in the models/ directory." -ForegroundColor White
    Write-Host "   2. Edit docker/config.json:" -ForegroundColor White

    switch ($backend) {
        "cpu" {
            Write-Host "      - For CPU: set 'n_gpu_layers': 0" -ForegroundColor White
        }
        "cuda" {
            Write-Host "      - For CUDA: set 'n_gpu_layers': 99, ensure CUDA is available" -ForegroundColor White
        }
        "vulkan" {
            Write-Host "      - For Vulkan: set 'n_gpu_layers': 99, add 'args': '--device Vulkan0'" -ForegroundColor White
            Write-Host "      - Note: .env file with GPU group IDs has been auto-generated" -ForegroundColor White
        }
    }

    Write-Host "   3. Run one of the commands below to start the service." -ForegroundColor White
    Write-Host "   4. Access the dashboard at http://localhost:8090" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage Examples:" -ForegroundColor Cyan
    Write-Host ""

    switch ($backend) {
        "cpu" {
            Write-Host "1. Run with Docker Compose (Recommended):" -ForegroundColor Yellow
            Write-Host "   docker compose --profile cpu up -d" -ForegroundColor Green
            Write-Host ""
            Write-Host "2. Run with a direct 'docker run' command:" -ForegroundColor Yellow
            Write-Host "   docker run -d -p 8090:8090 \" -ForegroundColor Green
            Write-Host "     -v ${PWD}/models:/app/models:ro \" -ForegroundColor Green
            Write-Host "     -v ${PWD}/docker/config.json:/app/config.json:ro \" -ForegroundColor Green
            Write-Host "     $IMAGE_TAG" -ForegroundColor Green
        }
        "cuda" {
            Write-Host "1. Run with Docker Compose (Recommended):" -ForegroundColor Yellow
            Write-Host "   docker compose --profile gpu up -d" -ForegroundColor Green
            Write-Host ""
            Write-Host "2. Run with a direct 'docker run' command:" -ForegroundColor Yellow
            Write-Host "   docker run -d --gpus all -p 8090:8090 \" -ForegroundColor Green
            Write-Host "     -v ${PWD}/models:/app/models:ro \" -ForegroundColor Green
            Write-Host "     -v ${PWD}/docker/config.json:/app/config.json:ro \" -ForegroundColor Green
            Write-Host "     $IMAGE_TAG" -ForegroundColor Green
        }
        "vulkan" {
            Write-Host "1. Run with Docker Compose (Recommended):" -ForegroundColor Yellow
            Write-Host "   docker compose --profile vulkan up -d" -ForegroundColor Green
            Write-Host ""
            Write-Host "2. Run with a direct 'docker run' command:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "   Note: Vulkan support on Windows with Docker is limited." -ForegroundColor Yellow
            Write-Host "   Consider using WSL2 with Linux instructions for better Vulkan support." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "   # Basic Vulkan setup (may require additional configuration):" -ForegroundColor White
            Write-Host "   docker run -d -p 8090:8090 \" -ForegroundColor Green
            Write-Host "     -v ${PWD}/models:/app/models:ro \" -ForegroundColor Green
            Write-Host "     -v ${PWD}/docker/config.json:/app/config.json:ro \" -ForegroundColor Green
            Write-Host "     $IMAGE_TAG" -ForegroundColor Green
        }
    }

    Write-Host ""
}

# Main script
Write-Host "Checking Docker..." -ForegroundColor Yellow
Test-Docker

Write-Host ""
Write-Host "Setting up directories..." -ForegroundColor Yellow
New-Directories

Write-Host ""
# Parse and normalize GPU backend
$GPU_BACKEND = Get-GPUBackend -backend $gpu

switch ($GPU_BACKEND) {
    "cuda" {
        Write-Host "CUDA GPU support requested" -ForegroundColor Magenta
        Test-NvidiaRuntime
    }
    "vulkan" {
        Write-Host "Vulkan GPU support requested" -ForegroundColor Magenta
        Setup-VulkanEnv
    }
    "cpu" {
        Write-Host "CPU-only mode" -ForegroundColor Cyan
    }
}

Build-DockerImage -backend $GPU_BACKEND

Write-Host ""
Write-Host "Starting FlexLLama container..." -ForegroundColor Yellow

# Determine which profile to use
$PROFILE = switch ($GPU_BACKEND) {
    "cuda"   { "gpu" }
    "vulkan" { "vulkan" }
    "cpu"    { "cpu" }
}

# Start the container
try {
    docker compose --profile $PROFILE up -d 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Container started successfully!" -ForegroundColor Green
        Write-Host ""
        
        # Wait a moment for container to initialize
        Start-Sleep -Seconds 3
        
        # Show container status
        Write-Host "Container Status:" -ForegroundColor Cyan
        docker ps --filter "name=flexllama" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
        Write-Host ""
        
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host "FlexLLama is ready!" -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Access the dashboard: http://localhost:8090" -ForegroundColor Cyan
        Write-Host ""
    }
    else {
        Write-Host "Failed to start container" -ForegroundColor Red
        Write-Host "Try manually: docker compose --profile $PROFILE up -d" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Error starting container: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Try manually: docker compose --profile $PROFILE up -d" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "FlexLLama Docker setup complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
