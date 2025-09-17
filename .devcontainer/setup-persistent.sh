#!/bin/bash
set -e

echo "🚀 Setting up persistent RAGLite GPU development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Function to check if package is already installed
package_installed() {
    pip list | grep -q "^$1 " 2>/dev/null
}

# Function to check if virtual environment needs setup
venv_needs_setup() {
    [ ! -f "/opt/venv/pyvenv.cfg" ] || [ ! -f "/opt/venv/bin/python" ]
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Fix ownership (always needed)
print_info "Fixing file permissions..."
sudo chown -R user:user /opt/ /home/user/.cache /home/user/.local 2>/dev/null || true

# Check if we're in a fresh container or existing one
if [ -f "/home/user/.container_initialized" ]; then
    print_info "Container previously initialized, performing incremental setup..."
    INCREMENTAL=true
else
    print_info "Fresh container detected, performing full setup..."
    INCREMENTAL=false
fi

# Python environment setup
print_info "Checking Python environment..."
if venv_needs_setup || [ "$INCREMENTAL" = false ]; then
    print_info "Setting up Python virtual environment..."
    uv sync --python ${PYTHON_VERSION:-3.11} --resolution ${RESOLUTION_STRATEGY:-highest} --all-extras
    print_info "Installing base requirements..."
    pip install -r requirements.txt -r requirements-dev.txt
    print_info "Installing raglite in development mode..."
    pip install -e .
    print_status "Python environment configured"
else
    print_info "Virtual environment exists, checking for updates..."
    # Quick dependency check without full reinstall
    uv sync --no-install-package raglite || print_warning "Failed to update dependencies"
    # Ensure raglite is installed in development mode
    if ! python -c "import raglite" 2>/dev/null; then
        print_info "Installing raglite in development mode..."
        pip install -e .
    fi
    print_status "Python environment updated"
fi

# GPU detection and setup
print_info "Detecting GPU hardware..."
if check_gpu; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected - will install CPU-only versions"
    GPU_AVAILABLE=false
fi

# PyTorch installation with GPU support
print_info "Checking PyTorch installation..."
PYTORCH_NEEDS_INSTALL=false

if ! package_installed "torch"; then
    print_info "PyTorch not found, will install..."
    PYTORCH_NEEDS_INSTALL=true
elif [ "$GPU_AVAILABLE" = true ] && ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_warning "PyTorch found but GPU support not working, will reinstall..."
    PYTORCH_NEEDS_INSTALL=true
fi

if [ "$PYTORCH_NEEDS_INSTALL" = true ]; then
    print_info "Installing PyTorch..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_info "Installing GPU-accelerated PyTorch (CUDA 12.4)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    else
        print_info "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    print_status "PyTorch installation complete"
else
    print_status "PyTorch already installed and working"
fi

# llama-cpp-python installation
print_info "Checking llama-cpp-python installation..."
LLAMA_CPP_NEEDS_INSTALL=false

if ! package_installed "llama-cpp-python"; then
    print_info "llama-cpp-python not found, will install..."
    LLAMA_CPP_NEEDS_INSTALL=true
elif [ "$GPU_AVAILABLE" = true ] && [ "$INCREMENTAL" = false ]; then
    print_info "GPU available and fresh install, will install CUDA version..."
    LLAMA_CPP_NEEDS_INSTALL=true
fi

if [ "$LLAMA_CPP_NEEDS_INSTALL" = true ]; then
    if [ "$GPU_AVAILABLE" = true ]; then
        print_info "Installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python --force-reinstall --no-cache-dir
    else
        print_info "Installing llama-cpp-python (CPU-only)..."
        pip install llama-cpp-python
    fi
    print_status "llama-cpp-python installation complete"
else
    print_status "llama-cpp-python already installed"
fi

# Development tools setup (idempotent)
print_info "Setting up development tools..."
if [ ! -f ".git/hooks/pre-commit" ]; then
    print_info "Installing pre-commit hooks..."
    pre-commit install --install-hooks
    print_status "Pre-commit hooks installed"
else
    print_status "Pre-commit hooks already installed"
fi

# Install useful GPU monitoring tools
if [ "$GPU_AVAILABLE" = true ]; then
    print_info "Installing GPU monitoring tools..."
    tools=("gpustat" "nvitop")
    for tool in "${tools[@]}"; do
        if ! package_installed "$tool"; then
            print_info "Installing $tool..."
            pip install "$tool" || print_warning "Failed to install $tool (optional)"
        fi
    done
fi

# Validate installation
print_info "Validating installation..."
python -c "
import sys
print('🐍 Python version:', sys.version.split()[0])

# Test core imports
try:
    import raglite
    print('✅ RAGLite imported successfully')
except ImportError as e:
    print('⚠️ RAGLite import failed:', e)

# Test PyTorch
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA version: {torch.version.cuda}')
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('ℹ️ Running in CPU-only mode')
except ImportError as e:
    print('❌ PyTorch import failed:', e)

# Test llama-cpp-python
try:
    import llama_cpp
    print('✅ llama-cpp-python imported successfully')
    # Try to detect CUDA support
    try:
        from llama_cpp import llama_cuda
        print('✅ llama-cpp-python CUDA support detected')
    except:
        print('ℹ️ llama-cpp-python running in CPU mode')
except ImportError as e:
    print('⚠️ llama-cpp-python import failed:', e)
"

# Mark container as initialized
touch /home/user/.container_initialized
echo "$(date): Container initialized with GPU_AVAILABLE=$GPU_AVAILABLE" >> /home/user/.container_setup_log

print_status "RAGLite GPU development environment ready!"
print_info "Setup log available at: /home/user/.container_setup_log"

# Show quick access commands
echo ""
echo "🚀 Quick commands:"
echo "  nvidia-smi          - Check GPU status"
echo "  gpustat             - Monitor GPU usage" 
echo "  python -c 'import torch; print(torch.cuda.is_available())'  - Test PyTorch GPU"
echo "  pytest tests/test_gpu_acceleration.py  - Run GPU tests"
echo ""
