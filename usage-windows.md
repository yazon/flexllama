# Usage Example - FlexLLama on Windows

## How to run the PowerShell script

### 1. Open PowerShell as Administrator
- Press `Win + X` and select "Windows PowerShell (Admin)" or "Terminal (Admin)"

### 2. Navigate to the project directory
```powershell
cd C:\Users\cegux\Documents\GitHub\flexllama
```

### 3. Run the script

**For CPU version (default):**
```powershell
.\docker-start.ps1
```

**For GPU version:**
```powershell
.\docker-start.ps1 -GPU
```

### 4. What happens during execution

The script will:
1. ✅ Check if Docker Desktop is running
2. 📁 Create `models/` and `logs/` directories
3. 🔨 Build the Docker image (CPU or GPU)
4. 📝 Show usage instructions

### 5. After execution

The script will show commands to:
- Start the service with Docker Compose
- Start the service with direct Docker Run
- Access the dashboard at http://localhost:8080

## Example script output

```
================================================
FlexLLama Docker Setup
================================================
Checking Docker...
Docker is available and running

Setting up directories...
Created models/ and logs/ directories

Building FlexLLama Docker image...
   Building CPU-only version...
Image built successfully: flexllama:latest

Next Steps:
   1. Place your .gguf model files in the models/ directory.
   2. Edit docker/config.json to point to your models.
   3. Run one of the commands below to start the service.
   4. Access the dashboard at http://localhost:8080

Usage Examples:

1. Run with Docker Compose (Recommended):
   # Start the CPU service:
   docker compose --profile cpu up -d

2. Run with a direct 'docker run' command:
   docker run -d -p 8080:8080 \
     -v C:\Users\cegux\Documents\GitHub\flexllama/models:/app/models:ro \
     -v C:\Users\cegux\Documents\GitHub\flexllama/docker/config.json:/app/config.json:ro \
     flexllama:latest

================================================
FlexLLama Docker setup complete!
================================================
```

## Next steps

1. **Add your .gguf models** to the `models/` directory
2. **Configure the file** `docker/config.json`
3. **Run the command** shown by the script
4. **Access the dashboard** at http://localhost:8080

## Troubleshooting

If you encounter problems:

1. **Script execution error:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Docker not running:**
   - Start Docker Desktop
   - Wait until the Docker icon turns green

3. **Permission error:**
   - Run PowerShell as administrator

4. **GPU issues:**
   - Make sure Docker Desktop has access to GPU
   - Check if NVIDIA drivers are up to date
