# RAGLite Dev Container & GitHub Codespaces Setup Guide
## NVIDIA GPU Support for GitHub Copilot Development

*Based on official documentation from [containers.dev](https://containers.dev/), [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers), [GitHub Codespaces](https://docs.github.com/en/codespaces), and [GitHub Copilot](https://docs.github.com/en/copilot) with GPU acceleration requirements.*

---

## 🎯 **Overview**

This guide provides comprehensive setup instructions for developing RAGLite with full NVIDIA GPU acceleration support in both local Dev Containers and GitHub Codespaces, optimized for GitHub Copilot integration.

### **Current Context Analysis**

**Existing Infrastructure** (already implemented):
- ✅ Basic Dev Container setup with Docker Compose
- ✅ Python 3.10+ with uv package manager
- ✅ GitHub Copilot extensions configured
- ✅ PostgreSQL with pgvector for database testing
- ✅ Requirements files for consistent dependencies

**Missing GPU Support** (needs implementation):
- ❌ NVIDIA GPU runtime configuration
- ❌ CUDA toolkit installation
- ❌ GPU-accelerated PyTorch and llama-cpp-python
- ❌ GPU detection and validation
- ❌ Codespaces GPU machine type configuration

---

## 📋 **Prerequisites**

### **For Local Development**
1. **Docker Desktop** with GPU support:
   - Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: Docker Engine + NVIDIA Container Toolkit
2. **NVIDIA GPU** with compute capability 3.5+
3. **Visual Studio Code** with extensions:
   - `ms-vscode-remote.remote-containers` (Dev Containers)
   - `GitHub.copilot` (GitHub Copilot)
   - `GitHub.copilot-chat` (Copilot Chat)

### **For GitHub Codespaces**
1. **GitHub Copilot subscription** (Personal/Organization)
2. **Codespaces access** with GPU-enabled machines
3. **Repository access** to ArtemisAI/raglite

---

## 🏗️ **Implementation Plan**

### **Phase 1: Local Dev Container GPU Enhancement**

#### **1.1 Docker Compose GPU Configuration**

**Update `docker-compose.yml`** to support NVIDIA GPU:

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
```

#### **1.2 Dockerfile GPU Enhancement**

**Update `Dockerfile`** with CUDA support:

```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS gpu-base

# Install uv and Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install system dependencies including CUDA development tools
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    git curl wget sudo clang libomp-dev \
    nvidia-cuda-toolkit nvidia-cuda-dev \
    && rm -rf /var/lib/apt/lists/*

FROM gpu-base AS dev

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compiler configuration for GPU compilation
ENV CC=clang
ENV CXX=clang++
ENV NVCC=/usr/local/cuda/bin/nvcc

# Create non-root user with GPU access
RUN groupadd --gid 1000 user && \
    useradd --create-home --no-log-init --gid 1000 --uid 1000 --shell /usr/bin/bash user && \
    usermod -aG video user && \
    chown user:user /opt/ && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && \
    chmod 0440 /etc/sudoers.d/user

USER user

# Configure shell with GPU detection
RUN mkdir ~/.history/ && \
    echo 'HISTFILE=~/.history/.bash_history' >> ~/.bashrc && \
    echo 'bind "\"\e[A\": history-search-backward"' >> ~/.bashrc && \
    echo 'bind "\"\e[B\": history-search-forward"' >> ~/.bashrc && \
    echo 'eval "$(starship init bash)"' >> ~/.bashrc && \
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc && \
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

#### **1.3 Dev Container Configuration Enhancement**

**Update `.devcontainer/devcontainer.json`** with GPU support:

```jsonc
{
    "name": "raglite-gpu",
    "dockerComposeFile": "../docker-compose.yml",
    "service": "devcontainer",
    "workspaceFolder": "/workspaces/${localWorkspaceFolderBasename}/",
    
    // GPU-specific features
    "features": {
        "ghcr.io/devcontainers-extra/features/starship:1": {},
        "ghcr.io/devcontainers/features/nvidia-cuda:1": {
            "installToolkit": true,
            "cudaVersion": "12.1"
        }
    },
    
    "overrideCommand": true,
    "remoteUser": "user",
    
    // Enhanced post-create setup with GPU validation
    "postCreateCommand": "bash .devcontainer/post-create.sh",
    
    // GPU environment variables
    "remoteEnv": {
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "CUDA_VISIBLE_DEVICES": "all"
    },
    
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
                "visualstudioexptteam.vscodeintellicode",
                "ms-toolsai.vscode-ai" // AI workbench for GPU monitoring
            ],
            "settings": {
                "python.defaultInterpreterPath": "/opt/venv/bin/python",
                "python.testing.pytestEnabled": true,
                "github.copilot.chat.agent.enabled": true,
                "github.copilot.chat.codesearch.enabled": true,
                "github.copilot.chat.edits.enabled": true,
                "github.copilot.nextEditSuggestions.enabled": true,
                // GPU development settings
                "terminal.integrated.env.linux": {
                    "CUDA_VISIBLE_DEVICES": "all",
                    "NVIDIA_VISIBLE_DEVICES": "all"
                }
            }
        }
    }
}
```

#### **1.4 Post-Create Setup Script**

**Create `.devcontainer/post-create.sh`**:

```bash
#!/bin/bash
set -e

echo "🚀 Setting up RAGLite GPU development environment..."

# Fix ownership
sudo chown -R user:user /opt/

# GPU detection and validation
echo "🔍 Detecting GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    export RAGLITE_GPU_AVAILABLE=true
else
    echo "⚠️  No NVIDIA GPU detected - CPU fallback mode"
    export RAGLITE_GPU_AVAILABLE=false
fi

# Install Python dependencies with GPU optimization
echo "📦 Installing Python dependencies..."
uv sync --python ${PYTHON_VERSION:-3.11} --resolution ${RESOLUTION_STRATEGY:-highest} --all-extras

# Install GPU-accelerated packages if GPU is available
if [[ "$RAGLITE_GPU_AVAILABLE" == "true" ]]; then
    echo "🎮 Installing GPU-accelerated packages..."
    # Install CUDA-enabled PyTorch
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Install GPU-compiled llama-cpp-python
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
else
    echo "💻 Installing CPU-only packages..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    uv pip install llama-cpp-python
fi

# Install requirements files
pip install -r requirements.txt -r requirements-dev.txt

# Setup development tools
pre-commit install --install-hooks

# Validate GPU setup
echo "🧪 Validating GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo "✅ RAGLite GPU development environment ready!"
```

### **Phase 2: GitHub Codespaces GPU Configuration**

#### **2.1 Codespace Configuration**

**Create `.devcontainer/devcontainer.json` for Codespaces**:

```jsonc
{
    "name": "raglite-codespace",
    
    // Use custom image with GPU support
    "image": "mcr.microsoft.com/devcontainers/python:3.11-bullseye",
    
    // Codespace-specific features
    "features": {
        "ghcr.io/devcontainers/features/nvidia-cuda:1": {
            "installToolkit": true,
            "version": "12.1"
        },
        "ghcr.io/devcontainers/features/git-lfs:1": {}
    },
    
    // Codespace machine requirements
    "hostRequirements": {
        "gpu": true,
        "memory": "32gb",
        "storage": "32gb"
    },
    
    "postCreateCommand": "bash .devcontainer/codespace-setup.sh",
    
    // GitHub Copilot optimizations
    "customizations": {
        "codespaces": {
            "openFiles": [
                "README.md",
                "src/raglite/_gpu_utils.py"
            ]
        },
        "vscode": {
            "extensions": [
                "GitHub.copilot",
                "GitHub.copilot-chat",
                "ms-python.python",
                "ms-toolsai.jupyter"
            ]
        }
    }
}
```

#### **2.2 Codespace Setup Script**

**Create `.devcontainer/codespace-setup.sh`**:

```bash
#!/bin/bash
set -e

echo "🌥️ Setting up RAGLite in GitHub Codespaces with GPU support..."

# Detect Codespace GPU capabilities
if [ -n "$CODESPACE_NAME" ]; then
    echo "📡 Running in GitHub Codespaces: $CODESPACE_NAME"
    # Codespace-specific GPU detection
    if nvidia-smi &>/dev/null; then
        echo "✅ GPU-enabled Codespace detected"
        export RAGLITE_CODESPACE_GPU=true
    else
        echo "💻 CPU-only Codespace"
        export RAGLITE_CODESPACE_GPU=false
    fi
fi

# Install dependencies optimized for Codespace
pip install -r requirements.txt -r requirements-dev.txt

# GPU-specific setup for Codespaces
if [[ "$RAGLITE_CODESPACE_GPU" == "true" ]]; then
    # Install CUDA-enabled packages
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
else
    # CPU-only installation
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Setup RAGLite in development mode
pip install -e .

# Validate installation
python -c "
import raglite
print('✅ RAGLite imported successfully')

import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')

# Test RAGLite GPU detection
from raglite._gpu_utils import detect_gpu_info
gpu_info = detect_gpu_info()
print(f'RAGLite GPU detection: {gpu_info}')
"

echo "🎉 RAGLite Codespace setup complete!"
```

### **Phase 3: GitHub Copilot Integration Optimization**

#### **3.1 Copilot Context Configuration**

**Create `.github/copilot-instructions.md`** (enhanced):

```markdown
# GitHub Copilot Instructions for RAGLite GPU Development

## Context
You are developing RAGLite, a GPU-accelerated RAG system with the following GPU infrastructure:

### GPU Components
- `src/raglite/_gpu_utils.py`: GPU detection and management
- `src/raglite/_embedding_gpu.py`: GPU-accelerated embeddings
- Hardware-specific dependency resolution
- CUDA/ROCm support with fallbacks

### Development Environment
- **Local**: Docker with NVIDIA runtime
- **Codespaces**: GPU-enabled machines with CUDA 12.1
- **Testing**: Multi-GPU scenarios and CPU fallbacks

### Key Focus Areas
1. **GPU Detection**: Automatic NVIDIA/AMD/Intel GPU identification
2. **Performance**: Optimal memory usage and batch processing
3. **Fallbacks**: Graceful CPU degradation
4. **Compatibility**: Multiple CUDA versions and hardware configurations

## Coding Guidelines
- Always implement GPU detection with CPU fallbacks
- Use type hints and comprehensive error handling
- Follow existing patterns in `_gpu_utils.py`
- Test both GPU and CPU code paths
- Optimize for memory efficiency in GPU operations
</markdown>
```

---

## 🧪 **Validation & Testing**

### **Local Development Validation**

1. **Build and start dev container**:
   ```bash
   # VS Code Command Palette: "Dev Containers: Rebuild Container"
   ```

2. **Validate GPU access**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Test RAGLite GPU functionality**:
   ```bash
   pytest tests/test_gpu_acceleration.py -v
   ```

### **Codespaces Validation**

1. **Create GPU-enabled Codespace**:
   - Repository → Code → Codespaces → Configure and create
   - Select machine type with GPU support

2. **Validate Codespace GPU**:
   ```bash
   echo $CODESPACE_NAME
   nvidia-smi
   python -c "from raglite._gpu_utils import detect_gpu_info; print(detect_gpu_info())"
   ```

---

## 🚀 **Best Practices**

### **For GitHub Copilot Development**
1. **Use descriptive comments** for GPU-specific code
2. **Include context** about hardware requirements
3. **Document fallback behaviors** clearly
4. **Test suggestions** on both GPU and CPU environments

### **For GPU Development**
1. **Always implement CPU fallbacks**
2. **Monitor GPU memory usage**
3. **Use batch processing** for optimal performance
4. **Handle multiple GPU scenarios**

### **For Codespaces**
1. **Choose appropriate machine types** (4-core, 8GB RAM minimum for GPU)
2. **Precompile containers** for faster startup
3. **Use secrets** for API keys
4. **Monitor usage** to avoid quota limits

---

## 📚 **References**

- [Dev Containers Specification](https://containers.dev/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [GitHub Copilot Best Practices](https://docs.github.com/en/copilot/using-github-copilot/getting-the-best-results-with-github-copilot)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

This comprehensive setup ensures optimal GPU acceleration for RAGLite development while maintaining compatibility with GitHub Copilot's AI-assisted development workflow.
