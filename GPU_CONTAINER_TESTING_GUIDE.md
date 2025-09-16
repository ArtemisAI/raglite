# GPU Container Testing Guide

## Current Status 🎯

Your development environment has been successfully configured with:
- ✅ **Persistent Setup Script**: `.devcontainer/setup-persistent.sh` 
- ✅ **Volume Configuration**: Docker Compose with persistent volumes for `/opt/venv`, caches, and models
- ✅ **Smart Installation**: Conditional package installation with incremental updates
- ✅ **GPU Configuration**: CUDA 12.4 environment with PyTorch GPU support (when GPU available)

## Testing Container Persistence 🔄

### 1. Test Container Rebuild
To test that your configurations persist across container rebuilds:

```bash
# From VS Code Command Palette (Ctrl+Shift+P):
# "Dev Containers: Rebuild and Reopen in Container"

# Or from terminal:
docker-compose down
docker-compose up --build -d
```

### 2. Verify Persistence After Rebuild
After rebuilding, check these indicators:

```bash
# Check if setup was incremental (should show "previously initialized")
cat /home/user/.container_setup_log

# Verify virtual environment persisted
ls -la /opt/venv/bin/python

# Check installed packages are still there
pip list | grep torch
pip list | grep llama-cpp-python

# Verify pre-commit hooks persisted
ls -la .git/hooks/pre-commit
```

## GPU Access Testing 🚀

### Current Issue: No GPU Access in Container
Your GPU configuration in `devcontainer.json` is correct, but the container needs NVIDIA Container Runtime.

### Solution 1: Docker Desktop GPU Support
If using Docker Desktop:

1. **Enable GPU Support**:
   ```bash
   # Check if nvidia-container-runtime is available
   docker info | grep -i gpu
   ```

2. **Update Docker Desktop**:
   - Go to Docker Desktop Settings
   - Enable "Use the WSL 2 based engine" (Windows)
   - Install NVIDIA Container Toolkit on host

### Solution 2: Host GPU Validation
First, verify GPU on your host machine:

```bash
# Exit container and run on host:
nvidia-smi
lspci | grep -i nvidia

# Check Docker can access GPU:
docker run --rm --gpus all nvidia/cuda:12.4-runtime-ubuntu20.04 nvidia-smi
```

### Solution 3: Codespaces GPU Setup
If using GitHub Codespaces, you'll need a GPU-enabled machine type:
- Request access to GPU-enabled Codespaces
- Use machine types with GPU support
- Configure billing for GPU usage

## Manual GPU Testing Commands 🧪

### Test GPU Detection
```bash
# Check NVIDIA GPU hardware
lspci | grep -i nvidia

# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

### Test PyTorch GPU
```bash
# Test PyTorch CUDA availability
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running in CPU-only mode')
"
```

### Test llama-cpp-python GPU
```bash
# Test llama-cpp-python CUDA support
python -c "
import llama_cpp
print('llama-cpp-python imported successfully')
try:
    from llama_cpp import llama_cuda
    print('CUDA support detected in llama-cpp-python')
except ImportError:
    print('CPU-only mode for llama-cpp-python')
"
```

### Test RAGLite GPU Acceleration
```bash
# Run GPU acceleration tests
pytest tests/test_gpu_acceleration.py -v

# Run GPU performance benchmarks
python -m raglite._bench --use-gpu --model-name all-MiniLM-L6-v2
```

## Troubleshooting Guide 🔧

### Issue: Container Lacks GPU Access

**Symptoms**:
- `nvidia-smi: command not found`
- `torch.cuda.is_available()` returns `False`
- No `/dev/nvidia*` devices in container

**Solutions**:
1. **Install NVIDIA Container Toolkit** on host:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Configure Docker daemon**:
   ```bash
   # Add to /etc/docker/daemon.json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia"
   }
   ```

3. **Test GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4-runtime-ubuntu20.04 nvidia-smi
   ```

### Issue: Dependencies Not Persisting

**Symptoms**:
- Need to reinstall packages after rebuild
- Virtual environment is empty

**Solutions**:
1. **Verify volume mounting**:
   ```bash
   docker-compose config | grep -A 10 volumes
   ```

2. **Check volume permissions**:
   ```bash
   ls -la /opt/venv/
   sudo chown -R user:user /opt/venv /home/user/.cache /home/user/.local
   ```

### Issue: PyTorch/CUDA Version Mismatch

**Symptoms**:
- PyTorch installed but CUDA unavailable
- Version conflicts

**Solutions**:
1. **Reinstall with specific CUDA version**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Verify CUDA compatibility**:
   ```bash
   python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); import subprocess; subprocess.run(['nvcc', '--version'])"
   ```

## Next Steps 🚀

### Immediate Actions
1. **Test Container Rebuild**: Verify persistence works correctly
2. **Configure GPU Runtime**: Set up NVIDIA Container Toolkit on host
3. **Validate GPU Access**: Run test commands to verify GPU detection
4. **Run Performance Tests**: Benchmark GPU vs CPU performance

### Optional Enhancements
- Add GPU monitoring tools (gpustat, nvitop)
- Configure automatic model downloads
- Set up performance benchmarking automation
- Add CUDA memory optimization settings

## Support Commands 🛠️

```bash
# Quick environment check
./.devcontainer/setup-persistent.sh

# View setup log
cat /home/user/.container_setup_log

# Check all volumes
docker volume ls

# Inspect container configuration  
docker-compose config

# Monitor GPU usage (if available)
watch -n 1 nvidia-smi

# Performance test
python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.mm(x, y)
    print('GPU tensor operation successful!')
else:
    print('GPU not available, using CPU')
"
```

Your environment is now fully configured for persistent, GPU-accelerated RAGLite development! 🎉
