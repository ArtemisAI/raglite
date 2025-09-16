# Dev Container Configuration Persistence Guide
## Avoiding Loss of Setup When Container Rebuilds

This guide shows how to ensure your dev container configurations, installations, and customizations persist across rebuilds and reboots.

## 🎯 **Current State Analysis**

**Your Current Setup** (from devcontainer.json):
- ✅ GPU support with CUDA 12.4
- ✅ PyTorch GPU installation 
- ✅ llama-cpp-python with CUDA compilation
- ✅ Comprehensive development tools

**Persistence Strategies Needed**:
1. **Volume Mounting** for persistent data
2. **Dockerfile Optimization** to minimize rebuilds
3. **Startup Scripts** for dynamic configuration
4. **Caching Strategies** for package installations
5. **Environment Persistence** for runtime settings

---

## 🏗️ **Implementation Strategy**

### **1. Volume Persistence Configuration**

Update your `docker-compose.yml` to add persistent volumes:

```yaml
services:
  devcontainer:
    build:
      target: dev
    environment:
      - CI
      - OPENAI_API_KEY
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    depends_on:
      - postgres
    volumes:
      - ..:/workspaces
      - command-history-volume:/home/user/.history/
      # Persistent volumes for configuration
      - venv-volume:/opt/venv                    # Python virtual environment
      - cache-volume:/home/user/.cache          # Package caches (pip, uv, etc.)
      - conda-volume:/home/user/.conda          # Conda environments (if used)
      - local-bin-volume:/home/user/.local      # User-installed packages
      - git-config-volume:/home/user/.gitconfig # Git configuration
      - ssh-volume:/home/user/.ssh              # SSH keys
      - models-cache:/home/user/.cache/huggingface  # ML model cache
    # GPU runtime configuration  
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: raglite_user
      POSTGRES_PASSWORD: raglite_password
    tmpfs:
      - /var/lib/postgresql/data

volumes:
  command-history-volume:
  venv-volume:              # Persistent Python environment
  cache-volume:             # Package caches
  conda-volume:             # Conda environments
  local-bin-volume:         # User binaries
  git-config-volume:        # Git settings
  ssh-volume:               # SSH configuration
  models-cache:             # ML models cache
```

### **2. Enhanced Dockerfile for Better Persistence**

Create a more optimized Dockerfile that minimizes rebuilds:

```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS gpu-base

# Install system dependencies (rarely change)
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    python3-pip git curl wget sudo clang libomp-dev \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv (package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

FROM gpu-base AS dev

# Create virtual environment path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compiler configuration
ENV CC=clang
ENV CXX=clang++

# Create non-root user with GPU access
RUN groupadd --gid 1000 user && \
    useradd --create-home --no-log-init --gid 1000 --uid 1000 --shell /usr/bin/bash user && \
    usermod -aG video user && \
    chown user:user /opt/ && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && \
    chmod 0440 /etc/sudoers.d/user

USER user

# Configure persistent directories
RUN mkdir -p ~/.history/ ~/.cache ~/.local ~/.ssh && \
    chown -R user:user /home/user

# Configure shell with persistent history and GPU environment
RUN echo 'HISTFILE=~/.history/.bash_history' >> ~/.bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc && \
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && \
    echo 'eval "$(starship init bash)"' >> ~/.bashrc
```

### **3. Smart Startup Script for Conditional Installs**

Create `.devcontainer/setup-persistent.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Setting up persistent RAGLite GPU development environment..."

# Function to check if package is already installed
package_installed() {
    pip list | grep -q "^$1 " 2>/dev/null
}

# Function to check if virtual environment needs setup
venv_needs_setup() {
    [ ! -f "/opt/venv/pyvenv.cfg" ] || [ ! -f "/opt/venv/bin/python" ]
}

# Fix ownership (always needed)
sudo chown -R user:user /opt/ /home/user/.cache /home/user/.local

echo "🔍 Checking Python environment..."
if venv_needs_setup; then
    echo "📦 Setting up Python virtual environment..."
    uv sync --python ${PYTHON_VERSION:-3.11} --resolution ${RESOLUTION_STRATEGY:-highest} --all-extras
    pip install -r requirements.txt -r requirements-dev.txt
else
    echo "✅ Virtual environment already exists, checking for updates..."
    uv sync --no-install-package raglite  # Update dependencies without reinstalling main package
fi

# GPU-specific package installation (check if already installed)
echo "🎮 Checking GPU packages..."
if ! package_installed "torch" || ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "📦 Installing/updating GPU-accelerated PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "✅ GPU PyTorch already installed and working"
fi

if ! package_installed "llama-cpp-python" || ! python -c "import llama_cpp" 2>/dev/null; then
    echo "📦 Installing/updating llama-cpp-python with CUDA..."
    CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall --no-cache-dir
else
    echo "✅ llama-cpp-python already installed"
fi

# Development tools setup (idempotent)
echo "🔧 Setting up development tools..."
if [ ! -f ".git/hooks/pre-commit" ]; then
    pre-commit install --install-hooks
else
    echo "✅ Pre-commit hooks already installed"
fi

# Validate GPU setup
echo "🧪 Validating GPU setup..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import llama_cpp
    print('✅ llama-cpp-python imported successfully')
except ImportError as e:
    print(f'⚠️ llama-cpp-python import failed: {e}')
"

# Install additional tools that might be useful
echo "🛠️ Checking additional development tools..."
tools=("nvidia-ml-py" "gpustat" "nvitop")
for tool in "${tools[@]}"; do
    if ! package_installed "$tool"; then
        echo "📦 Installing $tool..."
        pip install "$tool" || echo "⚠️ Failed to install $tool (optional)"
    fi
done

echo "✅ Persistent RAGLite GPU development environment ready!"
```

### **4. Update DevContainer Configuration**

Update your `.devcontainer/devcontainer.json` to use the persistent setup:

```jsonc
{
    "name": "raglite-gpu",
    "dockerComposeFile": "../docker-compose.yml", 
    "service": "devcontainer",
    "workspaceFolder": "/workspaces/${localWorkspaceFolderBasename}/",
    
    "features": {
        "ghcr.io/devcontainers-extra/features/starship:1": {}
    },
    
    "overrideCommand": true,
    "remoteUser": "user",
    
    // Use persistent setup script
    "postCreateCommand": "bash .devcontainer/setup-persistent.sh",
    "postStartCommand": "echo '🎉 RAGLite GPU Dev Container Ready!'",
    
    // Lifecycle commands for maintenance
    "updateContentCommand": "bash .devcontainer/setup-persistent.sh",
    
    "customizations": {
        "vscode": {
            "extensions": [
                "charliermarsh.ruff",
                "GitHub.copilot", 
                "GitHub.copilot-chat",
                "GitHub.vscode-github-actions",
                "GitHub.vscode-pull-request-github",
                "ms-azuretools.vscode-docker",
                "ms-python.mypy-type-checker",
                "ms-python.python",
                "ms-toolsai.jupyter",
                "ryanluker.vscode-coverage-gutters",
                "tamasfe.even-better-toml",
                "visualstudioexptteam.vscodeintellicode"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/opt/venv/bin/python",
                "python.testing.pytestEnabled": true,
                "github.copilot.chat.agent.enabled": true,
                "github.copilot.chat.codesearch.enabled": true,
                "github.copilot.chat.edits.enabled": true,
                "github.copilot.nextEditSuggestions.enabled": true,
                "terminal.integrated.env.linux": {
                    "GIT_EDITOR": "code --wait",
                    "CUDA_HOME": "/usr/local/cuda-12.4",
                    "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}",
                    "NVIDIA_VISIBLE_DEVICES": "all",
                    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"
                }
            }
        }
    }
}
```

### **5. Environment Variables Persistence**

Create `.devcontainer/.env` for persistent environment variables:

```bash
# RAGLite GPU Configuration
CUDA_HOME=/usr/local/cuda-12.4
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python Environment
PYTHON_VERSION=3.11
UV_PROJECT_ENVIRONMENT=/opt/venv
VIRTUAL_ENV=/opt/venv

# Development Settings  
RAGLITE_DEBUG=true
RAGLITE_GPU_ENABLED=true
RAGLITE_CACHE_DIR=/home/user/.cache/raglite

# Model Caching
HF_HOME=/home/user/.cache/huggingface
TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
```

### **6. Quick Validation Script**

Create `.devcontainer/validate-setup.sh` for quick checks:

```bash
#!/bin/bash
echo "🔍 Validating RAGLite GPU Development Environment..."

echo "📊 System Info:"
echo "  OS: $(lsb_release -d | cut -f2-)"
echo "  CUDA: $CUDA_HOME"
echo "  Python: $(python --version)"

echo "🎮 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "  ⚠️ nvidia-smi not available"
fi

echo "🐍 Python Packages:"
python -c "
import sys
packages = ['torch', 'llama_cpp', 'raglite']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'  ✅ {pkg}: {version}')
    except ImportError:
        print(f'  ❌ {pkg}: not installed')
"

echo "🧪 GPU Tests:"
python -c "
import torch
print(f'  PyTorch CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU Count: {torch.cuda.device_count()}')
    print(f'  Current Device: {torch.cuda.get_device_name()}')
"

echo "✅ Validation complete!"
```

---

## 🚀 **Best Practices Summary**

### **For Persistent Configurations**:
1. **Use Named Volumes** for `/opt/venv`, `~/.cache`, `~/.local`
2. **Conditional Installation** in startup scripts
3. **Layer Caching** in Dockerfile optimization
4. **Environment Files** for consistent variables

### **For Development Workflow**:
1. **Incremental Updates** instead of full rebuilds
2. **Package Version Pinning** in requirements files
3. **Health Check Scripts** for validation
4. **Backup Strategies** for important configurations

### **For GPU Development**:
1. **GPU State Validation** on container start
2. **Model Cache Persistence** to avoid re-downloads
3. **Memory Management** for long-running sessions
4. **Fallback Mechanisms** for CPU-only environments

This setup ensures your GPU configurations, Python packages, and development tools persist across container rebuilds while maintaining optimal performance! 🎉
