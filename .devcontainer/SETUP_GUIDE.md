# 🚀 DevContainer Setup Guide

## Quick Start

### For VS Code Users (Recommended)
1. **Clone the repository**
2. **Open in VS Code**
3. **Run:** `Ctrl/⌘ + Shift + P` → `Dev Containers: Reopen in Container`
4. **Wait for setup to complete** (first time may take 5-10 minutes)

### For Docker Compose Users
```bash
# Clone the repository
git clone https://github.com/ArtemisAI/raglite.git
cd raglite

# Start the development environment
docker compose up -d

# Access the container
docker compose exec devcontainer bash
```

### For Direct Docker Users
```bash
# Build the development image
docker build --target dev -t raglite:dev .

# Run with GPU support
docker run -it --rm --gpus all \
  -v "$(pwd)":/workspaces/raglite \
  raglite:dev bash
```

## 🔧 System Requirements

### Minimum Requirements
- **Docker**: 20.10+
- **RAM**: 8GB+
- **Storage**: 10GB+ free space

### For GPU Development
- **NVIDIA GPU** with compute capability 7.0+
- **NVIDIA Driver**: 470.57.02+
- **CUDA**: 12.4 (installed in container)

### Supported Operating Systems
- **Linux**: Ubuntu 20.04+, Debian 11+
- **Windows**: Windows 10/11 with WSL2
- **macOS**: Intel or Apple Silicon (CPU-only)

## 📋 Setup Validation

Run the validation script to check your setup:
```bash
./.devcontainer/validate-setup.sh
```

This will verify:
- Docker configuration syntax
- DevContainer JSON validity
- Required files presence
- GPU detection (if available)
- Python dependencies
- Docker build capability

## 🐛 Troubleshooting

### Common Issues

1. **"unknown or invalid runtime name: nvidia"**
   - Remove `--runtime=nvidia` from runArgs (already fixed in this repo)
   - The configuration now uses modern `--gpus all` approach

2. **"Package 'libnvidia-compute-550' has no installation candidate"**
   - Clear VS Code devcontainer cache
   - Use `Dev Containers: Rebuild Container (No Cache)`

3. **"ModuleNotFoundError: No module named 'raglite'"**
   - The setup script automatically installs raglite in development mode
   - If issues persist, manually run: `pip install -e .`

4. **GPU not detected in container**
   - Ensure NVIDIA drivers are installed on host
   - Install NVIDIA Container Toolkit
   - Restart Docker daemon

For comprehensive troubleshooting, see: `.devcontainer/DEVCONTAINER_TROUBLESHOOTING.md`

## 🎯 Features

### ✅ What Works
- **Modern GPU Support**: Uses `--gpus all` instead of deprecated nvidia runtime
- **Automatic Setup**: Python environment and dependencies installed automatically
- **Persistent Volumes**: Development state preserved across container rebuilds
- **Graceful Fallbacks**: Works on systems without GPU
- **VS Code Integration**: Full VS Code devcontainer support with extensions
- **Multiple Access Methods**: VS Code, docker-compose, or direct Docker

### 🔄 Development Workflow
1. **Edit code** in VS Code or your preferred editor
2. **Test changes** using the integrated terminal
3. **Run tests** with `python -m pytest tests/`
4. **Debug** with VS Code's integrated debugger
5. **Commit changes** with pre-commit hooks enabled

## 🚀 Advanced Configuration

### CPU-Only Mode
Add to your environment:
```bash
export RAGLITE_DISABLE_GPU=1
```

### Custom Docker Compose
For specific GPU configurations:
```bash
docker-compose -f docker-compose.yml -f docker-compose.fallback.yml up
```

### Alternative CUDA Versions
Modify `Dockerfile` CUDA version if needed:
```dockerfile
ENV CUDA_HOME=/usr/local/cuda-12.4
```

## 📞 Getting Help

1. **Check validation**: `./.devcontainer/validate-setup.sh`
2. **Read troubleshooting**: `.devcontainer/DEVCONTAINER_TROUBLESHOOTING.md`
3. **Test end-to-end**: `./.devcontainer/test-e2e.sh` (optional)
4. **Open an issue** on GitHub with your configuration details

## 🎉 Success Criteria

Your setup is working correctly when:
- ✅ Container builds without errors
- ✅ `python -c "import raglite"` works
- ✅ `nvidia-smi` shows your GPU (if applicable)
- ✅ VS Code attaches and extensions load
- ✅ Tests run successfully: `python -m pytest tests/test_import.py`