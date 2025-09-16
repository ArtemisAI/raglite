# Dev Container and GitHub Codespaces Setup for RAGLite with NVIDIA GPU Support

## 🎯 **Overview**

This document provides comprehensive instructions for setting up a development environment for the RAGLite project that supports NVIDIA GPU acceleration, both locally with Dev Containers and in the cloud with GitHub Codespaces.

## 📋 **Current Project Context**

### **Existing GPU Infrastructure**
- ✅ GPU acceleration implemented in `src/raglite/_embedding_gpu.py` and `src/raglite/_gpu_utils.py`
- ✅ CUDA detection and graceful CPU fallback
- ✅ Configuration management in `src/raglite/_config.py`
- ✅ GPU-aware test suites and benchmarking
- 🚧 Smart installation script system (in development)

### **Hardware Requirements**
- **NVIDIA GPU**: Required for optimal performance
- **CUDA Toolkit**: Version 11.8+ or 12.x
- **VRAM**: Minimum 4GB, recommended 8GB+
- **Python**: Version 3.10+

## 🔧 **1. Local Environment Assessment**

Before setting up the dev container, assess your current environment:

### **System Information**
```bash
# Operating System
# Windows:
ver

# macOS:
sw_vers

# Linux:
lsb_release -a || cat /etc/os-release
```

### **GPU and CUDA Detection**
```bash
# Check NVIDIA GPU presence
nvidia-smi

# Check CUDA version
nvcc --version

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### **Required Runtimes**
```bash
# Python (3.10+)
python3 --version

# Docker (with GPU support)
docker --version
docker compose version

# Node.js (for development tools)
node --version
```

## 🐳 **2. Enhanced Dev Container Configuration**

### **Prerequisites ✔️**

1. **Visual Studio Code** with extensions:
   - Dev Containers (`ms-vscode-remote.remote-containers`)
   - GitHub Copilot (`GitHub.copilot`)
   - Python (`ms-python.python`)

2. **Docker Desktop** with:
   - WSL 2 backend (Windows)
   - GPU support enabled
   - Resource limits: 8GB+ RAM, 4+ CPUs

3. **NVIDIA Container Toolkit** (Linux):
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### **Enhanced devcontainer.json Configuration**

Create `.devcontainer/devcontainer.json` with GPU support:

```json
{
  "name": "RAGLite Development Environment",
  
  // Base image with CUDA support
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  
  // Docker Compose for GPU support
  "dockerComposeFile": "../docker-compose.dev.yml",
  "service": "raglite-dev",
  "workspaceFolder": "/workspace",
  
  // GPU runtime configuration
  "runArgs": [
    "--gpus=all",
    "--shm-size=2g"
  ],
  
  // Development features
  "features": {
    "ghcr.io/devcontainers/features/git-lfs:1": {},
    "ghcr.io/devcontainers/features/nvidia-cuda:1": {
      "installCUDNN": true,
      "version": "12.2"
    },
    "ghcr.io/devcontainers/features/node:1": {
      "version": "18"
    }
  },
  
  // Container lifecycle commands
  "postCreateCommand": "bash .devcontainer/post-create.sh",
  "postStartCommand": "bash .devcontainer/post-start.sh",
  
  // Port forwarding for web interfaces
  "forwardPorts": [
    8000,  // Chainlit interface
    8888,  // Jupyter notebooks
    6006   // TensorBoard
  ],
  
  // VS Code customizations
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/opt/conda/bin/python",
        "python.terminal.activateEnvironment": true,
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": ["tests/"],
        "python.linting.enabled": true,
        "python.linting.ruffEnabled": true,
        "jupyter.kernels.excludePythonEnvironments": ["/usr/bin/python3"],
        "github.copilot.enable": {
          "*": true,
          "plaintext": false,
          "markdown": true,
          "python": true
        }
      },
      
      // Essential extensions for RAGLite development
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "GitHub.copilot",
        "GitHub.copilot-chat",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "ms-vscode.test-adapter-converter",
        "tamasfe.even-better-toml"
      ]
    }
  },
  
  // Development user configuration
  "remoteUser": "vscode",
  "containerUser": "vscode",
  
  // Environment variables
  "containerEnv": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONPATH": "/workspace/src",
    "RAGLITE_DEV_MODE": "true"
  }
}
```

### **Docker Compose Configuration**

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  raglite-dev:
    build:
      context: .
      dockerfile: .devcontainer/Dockerfile
    
    runtime: nvidia
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/workspace/src
    
    volumes:
      - ../:/workspace:cached
      - ~/.gitconfig:/home/vscode/.gitconfig:ro
      - ~/.ssh:/home/vscode/.ssh:ro
      - raglite-cache:/workspace/.cache
      - raglite-models:/workspace/models
    
    working_dir: /workspace
    
    command: /bin/sh -c "while sleep 1000; do :; done"
    
    ports:
      - "8000:8000"
      - "8888:8888"
      - "6006:6006"

volumes:
  raglite-cache:
  raglite-models:
```

### **Enhanced Dockerfile**

Create `.devcontainer/Dockerfile`:

```dockerfile
# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create development user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\\(root\\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Python packages optimized for GPU
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers accelerate \
    jupyter jupyterlab \
    tensorboard

USER $USERNAME
WORKDIR /workspace
```

### **Lifecycle Scripts**

Create `.devcontainer/post-create.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Setting up RAGLite development environment..."

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e ".[gpu,dev,bench]"

# Set up pre-commit hooks
pre-commit install

# Create necessary directories
mkdir -p .cache models logs

# Download and cache common models
python -c "
from transformers import AutoTokenizer, AutoModel
models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'microsoft/DialoGPT-small'
]
for model in models:
    try:
        AutoTokenizer.from_pretrained(model)
        AutoModel.from_pretrained(model)
        print(f'✅ Cached {model}')
    except Exception as e:
        print(f'⚠️  Failed to cache {model}: {e}')
"

# Test GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "✅ Development environment setup complete!"
```

Create `.devcontainer/post-start.sh`:

```bash
#!/bin/bash
set -e

echo "🔄 Starting development services..."

# Activate conda environment if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

# Start Jupyter Lab in background (optional)
# nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root > logs/jupyter.log 2>&1 &

echo "✅ Development services started!"
```

## ☁️ **3. GitHub Codespaces Configuration**

### **Codespaces-Specific Considerations**

GitHub Codespaces has limitations with GPU access, but we can configure it for CPU development:

### **Enhanced codespaces.json**

Create `.devcontainer/codespaces.json`:

```json
{
  "name": "RAGLite Codespaces",
  
  // Use CPU-optimized image for Codespaces
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  
  // Codespaces-specific features
  "features": {
    "ghcr.io/devcontainers/features/git-lfs:1": {},
    "ghcr.io/devcontainers/features/node:1": {
      "version": "18"
    }
  },
  
  // Lifecycle commands for Codespaces
  "postCreateCommand": "bash .devcontainer/codespaces-setup.sh",
  
  // Codespaces-optimized settings
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python3",
        "raglite.gpu.enabled": false,
        "raglite.fallback.cpu": true
      },
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "GitHub.copilot",
        "GitHub.copilot-chat",
        "charliermarsh.ruff"
      ]
    }
  },
  
  // Resource configuration
  "hostRequirements": {
    "cpus": 4,
    "memory": "8gb",
    "storage": "32gb"
  }
}
```

### **Codespaces Setup Script**

Create `.devcontainer/codespaces-setup.sh`:

```bash
#!/bin/bash
set -e

echo "☁️ Setting up RAGLite for GitHub Codespaces..."

# Install CPU-optimized dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install RAGLite with CPU optimizations
pip install -e ".[dev]"

# Configure for CPU-only mode
echo "RAGLITE_GPU_ENABLED=false" >> .env
echo "RAGLITE_FALLBACK_CPU=true" >> .env

# Test CPU setup
python -c "
import raglite
from raglite._config import RAGLiteConfig
config = RAGLiteConfig()
print('✅ RAGLite configured for CPU mode')
print(f'Database: {config.db_url}')
"

echo "✅ Codespaces setup complete!"
```

## 🚀 **4. GitHub Copilot Optimization**

### **Copilot Configuration**

Create `.vscode/settings.json`:

```json
{
  "github.copilot.enable": {
    "*": true,
    "plaintext": false,
    "markdown": true,
    "python": true,
    "json": true,
    "yaml": true,
    "dockerfile": true,
    "shellscript": true
  },
  
  "github.copilot.editor.enableAutoCompletions": true,
  "github.copilot.conversation.localeOverride": "en",
  
  // RAGLite-specific Copilot context
  "github.copilot.chat.experimental.codeCitation": true,
  "github.copilot.chat.experimental.codeGeneration": true
}
```

### **Copilot Instructions**

Create `.github/copilot-instructions.md` (if not exists):

```markdown
# RAGLite Development Context for GitHub Copilot

## Project Overview
RAGLite is a high-performance Python library for Retrieval-Augmented Generation (RAG) systems with GPU acceleration support.

## Key Components
- GPU acceleration: `src/raglite/_embedding_gpu.py`, `src/raglite/_gpu_utils.py`
- Configuration: `src/raglite/_config.py`
- Database: SQLite with sqlite-vec extension
- Testing: `tests/` with GPU acceleration tests

## Development Guidelines
1. Always include type hints
2. Handle GPU/CPU fallbacks gracefully
3. Write comprehensive tests
4. Use ruff for formatting
5. Follow existing patterns in the codebase

## GPU Considerations
- Detect CUDA availability before GPU operations
- Provide CPU fallbacks for all GPU functionality
- Test both GPU and CPU code paths
- Handle VRAM limitations gracefully
```

## 📋 **5. Setup Checklist**

### **Local Development Setup**
- [ ] Docker Desktop installed with GPU support
- [ ] NVIDIA Container Toolkit configured (Linux)
- [ ] VS Code with Dev Containers extension
- [ ] `.devcontainer/` configuration files created
- [ ] Dev container builds and starts successfully
- [ ] GPU detection working in container
- [ ] All tests pass

### **GitHub Codespaces Setup**
- [ ] Repository has `.devcontainer/` configuration
- [ ] Codespaces prebuild configured (optional)
- [ ] CPU fallback mode works correctly
- [ ] All essential features available without GPU

### **GitHub Copilot Setup**
- [ ] Copilot enabled for account/organization
- [ ] VS Code Copilot extensions installed
- [ ] Copilot instructions file created
- [ ] Context-aware suggestions working

## 🔧 **6. Troubleshooting**

### **Common Issues**

1. **GPU Not Available in Container**
   ```bash
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check CUDA compatibility
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Permission Issues**
   ```bash
   # Fix container permissions
   sudo chown -R vscode:vscode /workspace
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limits
   # Docker Desktop: Settings > Resources > Memory > 8GB+
   ```

### **Validation Commands**

```bash
# Test complete setup
python -c "
import torch
import raglite
from raglite._gpu_utils import detect_gpu_info

print('🔍 System Check:')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
print(f'RAGLite GPU utils: {detect_gpu_info()}')
print('✅ Setup validation complete!')
"

# Run basic tests
pytest tests/test_gpu_acceleration.py -v
```

This comprehensive setup ensures optimal development experience for RAGLite with full GPU acceleration support locally and graceful CPU fallback in cloud environments.
