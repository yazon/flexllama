Param(
    [switch]$gpu,
    [switch]$g
)

Write-Host "================================================"
Write-Host "FlexLLama Docker Setup (Windows PowerShell)"
Write-Host "================================================"

function Test-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop."; exit 1
    }
    try {
        docker info | Out-Null
    } catch {
        Write-Error "Docker is not running. Please start Docker Desktop."; exit 1
    }
    Write-Host "✅ Docker is available and running"
}

function Setup-Directories {
    Write-Host "📁 Setting up directories..."
    foreach ($dir in @('models','logs')) {
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    }
    Write-Host "✅ Ensured models/ and logs/ directories exist"
}

function Build-Image([bool]$UseGpu) {
    Write-Host "🔨 Building FlexLLama Docker image..."
    if ($UseGpu) {
        Write-Host "   Building with GPU support..."
        docker build -f Dockerfile.cuda -t flexllama-gpu:latest .
        return "flexllama-gpu:latest"
    } else {
        Write-Host "   Building CPU-only version..."
        docker build -t flexllama:latest .
        return "flexllama:latest"
    }
}

function Show-Examples([bool]$UseGpu, [string]$ImageTag) {
    Write-Host ""
    Write-Host "📝 Next Steps:"
    Write-Host "   1. Place your .gguf model files in the models/ directory."
    Write-Host "   2. Edit docker/config.json to point to your models (set 'n_gpu_layers' > 0 for GPU)."
    Write-Host "   3. Run one of the commands below to start the service."
    Write-Host "   4. Access the dashboard at http://localhost:8080"
    Write-Host ""
    Write-Host "🚀 Usage Examples:"
    Write-Host ""
    Write-Host "1. Run with Docker Compose (Recommended):"
    if ($UseGpu) {
        Write-Host "   # Start the GPU service:"
        Write-Host "   docker compose --profile gpu up -d"
    } else {
        Write-Host "   # Start the CPU service:"
        Write-Host "   docker compose --profile cpu up -d"
    }
    Write-Host ""
    Write-Host "2. Run with a direct 'docker run' command:"
    if ($UseGpu) {
        Write-Host "   docker run -d --gpus all -p 8080:8080 \"
    } else {
        Write-Host "   docker run -d -p 8080:8080 \"
    }
    # On Windows PowerShell, ${PWD} expands to a Windows path which Docker Desktop accepts.
    Write-Host "     -v \"\${PWD}/models:/app/models:ro\" \"
    Write-Host "     -v \"\${PWD}/docker/config.json:/app/config.json:ro\" \"
    Write-Host "     $ImageTag"
    Write-Host ""
}

# Main
Write-Host "Checking Docker..."
Test-Docker

Write-Host ""
Write-Host "Setting up directories..."
Setup-Directories

$useGpu = ($gpu.IsPresent -or $g.IsPresent -or ($args -contains "--gpu") -or ($args -contains "-g"))
if ($useGpu) { Write-Host "🎮 GPU support requested" }

$imageTag = Build-Image -UseGpu:$useGpu
Write-Host "✅ Image built successfully: $imageTag"

Show-Examples -UseGpu:$useGpu -ImageTag:$imageTag

Write-Host "================================================"
Write-Host "✅ FlexLLama Docker setup complete!"
Write-Host "================================================"

