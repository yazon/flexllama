# FlexLLama Docker Quick Start Script for Windows PowerShell
# ================================================

param(
    [switch]$GPU
)

# Error configuration
$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "FlexLLama Docker Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Function to check if Docker is running
function Test-Docker {
    try {
        # Check if Docker is installed xd
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

# Function to build the image
function Build-DockerImage {
    param([switch]$GPU)
    
    Write-Host "Building FlexLLama Docker image..." -ForegroundColor Yellow
    
    if ($GPU) {
        Write-Host "   Building with GPU support..." -ForegroundColor Yellow
        docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
        $script:IMAGE_TAG = "flexllama-gpu:latest"
    }
    else {
        Write-Host "   Building CPU-only version..." -ForegroundColor Yellow
        docker build -t flexllama:latest .
        $script:IMAGE_TAG = "flexllama:latest"
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

# Function to show usage examples
function Show-UsageExamples {
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Place your .gguf model files in the models/ directory." -ForegroundColor White
    Write-Host "   2. Edit docker/config.json to point to your models (set 'n_gpu_layers' > 0 for GPU)." -ForegroundColor White
    Write-Host "   3. Run one of the commands below to start the service." -ForegroundColor White
    Write-Host "   4. Access the dashboard at http://localhost:8080" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage Examples:" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "1. Run with Docker Compose (Recommended):" -ForegroundColor Yellow
    if ($GPU) {
        Write-Host "   # Start the GPU service:" -ForegroundColor White
        Write-Host "   docker compose --profile gpu up -d" -ForegroundColor Green
    }
    else {
        Write-Host "   # Start the CPU service:" -ForegroundColor White
        Write-Host "   docker compose --profile cpu up -d" -ForegroundColor Green
    }
    Write-Host ""
    
    Write-Host "2. Run with a direct 'docker run' command:" -ForegroundColor Yellow
    if ($GPU) {
        Write-Host "   docker run -d --gpus all -p 8080:8080 \" -ForegroundColor Green
    }
    else {
        Write-Host "   docker run -d -p 8080:8080 \" -ForegroundColor Green
    }
    Write-Host "     -v ${PWD}/models:/app/models:ro \" -ForegroundColor Green
    Write-Host "     -v ${PWD}/docker/config.json:/app/config.json:ro \" -ForegroundColor Green
    Write-Host "     $IMAGE_TAG" -ForegroundColor Green
    Write-Host ""
}

# Main script
Write-Host "Checking Docker..." -ForegroundColor Yellow
Test-Docker

Write-Host ""
Write-Host "Setting up directories..." -ForegroundColor Yellow
New-Directories

Write-Host ""
if ($GPU) {
    Write-Host "GPU support requested" -ForegroundColor Magenta
}

Build-DockerImage -GPU:$GPU

Show-UsageExamples

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "FlexLLama Docker setup complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
