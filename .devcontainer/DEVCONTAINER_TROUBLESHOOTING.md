# DevContainer Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. **VS Code Build Failures**

#### Issue: "Package 'libnvidia-compute-550' has no installation candidate"
**Cause**: VS Code Dev Containers extension is caching old `Dockerfile-with-features` containing outdated NVIDIA driver packages.

**Solution**:
```bash
# Clear VS Code devcontainer cache
rm -rf ~/.local/share/containers/
# Or in VS Code: Dev Containers: Rebuild Container (No Cache)
```

#### Issue: "unknown or invalid runtime name: nvidia"
**Cause**: Docker daemon not configured for NVIDIA runtime or missing NVIDIA Container Toolkit.

**Solution**:
```bash
# Install NVIDIA Container Toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. **GPU Access Issues**

#### Issue: "could not select device driver with capabilities: [[gpu]]"
**Cause**: Docker GPU runtime not properly configured.

**Solution**:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# If this fails, restart Docker daemon
sudo systemctl restart docker
```

#### Issue: "nvidia-smi not found" in container
**Cause**: NVIDIA drivers not accessible from container.

**Solution**: Ensure NVIDIA drivers are installed on host:
```bash
# Check host GPU drivers
nvidia-smi
# Install if missing:
sudo apt install nvidia-driver-470  # or latest version
```

### 3. **Python Environment Issues**

#### Issue: "ModuleNotFoundError: No module named 'raglite'"
**Cause**: RAGLite not installed in development mode.

**Solution**:
```bash
# Inside container:
cd /workspaces/raglite
pip install -e .
```

#### Issue: Virtual environment not activated
**Cause**: Python path not properly configured.

**Solution**: Verify VS Code settings:
```json
{
    "python.defaultInterpreterPath": "/opt/venv/bin/python",
    "python.terminal.activateEnvironment": false
}
```

### 4. **Container Build Optimization**

#### Slow builds
**Solution**: Use build cache effectively:
```bash
# Clear build cache if needed
docker system prune -f
docker builder prune -f

# Build with cache
docker build --target dev -t raglite:dev .
```

#### Permission issues
**Solution**: Ensure correct ownership:
```bash
# Inside container:
sudo chown -R user:user /opt/venv /home/user/.cache
```

## 🧪 Validation Commands

### Test GPU Access
```bash
# In container:
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test Python Environment
```bash
# In container:
python -c "import raglite; print('RAGLite imported successfully')"
python -m pytest tests/test_import.py -v
```

### Test Development Workflow
```bash
# In container:
cd /workspaces/raglite
python -m pytest tests/ -x --tb=short
```

## 🚀 Alternative Setup Methods

### Method 1: Direct Docker (Bypass VS Code issues)
```bash
docker build --target dev -t raglite:dev .
docker run -it --rm --gpus all \
    -v "$(pwd)":/workspaces/raglite \
    -v raglite_venv:/opt/venv \
    raglite:dev bash
```

### Method 2: Docker Compose (For persistent development)
```bash
docker compose up -d
docker compose exec devcontainer bash
```

### Method 3: CPU-Only Development
Set environment variable to disable GPU:
```bash
export RAGLITE_DISABLE_GPU=1
# Then build devcontainer normally
```

## 📋 Debugging Checklist

- [ ] NVIDIA drivers installed on host (`nvidia-smi` works)
- [ ] Docker has GPU support (`docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi`)
- [ ] VS Code devcontainer cache cleared
- [ ] Python virtual environment has raglite installed (`pip list | grep raglite`)
- [ ] Container user has proper permissions (`whoami` shows `user`)
- [ ] CUDA environment variables set correctly

## 🔧 Reset Instructions

### Complete DevContainer Reset
```bash
# 1. Stop all containers
docker stop $(docker ps -aq)

# 2. Remove volumes
docker volume rm raglite_command-history-volume raglite_python-venv-volume raglite_cache-volume raglite_local-volume raglite_models-volume

# 3. Clear VS Code cache
rm -rf ~/.local/share/containers/

# 4. Rebuild
# VS Code: Dev Containers: Rebuild Container (No Cache)
```

### Emergency CPU-Only Mode
If GPU issues persist, disable GPU acceleration:
```bash
# Add to devcontainer.json containerEnv:
"RAGLITE_DISABLE_GPU": "1"
```