# Change Request: Smart Local Installation Script System

## 🎯 **Objective**

Create a comprehensive, intelligent installation script system for RAGLite that automatically detects system hardware, installs optimal dependencies, and sets up a complete development environment for local users who clone the repository.

## 📋 **Problem Statement**

### Current Issues:
1. **No Local Setup Automation**: Only GitHub CI/CD setup script exists (`.github/setup-environment.sh`) - nothing for local development
2. **Manual GPU Configuration**: Users must manually determine CUDA versions, llama-cpp-python variants, and PyTorch installations
3. **Complex Dependency Management**: Users need expert knowledge to choose correct optional extras and hardware-specific packages
4. **No System Detection**: No automatic hardware analysis or optimization
5. **Poor User Experience**: Multiple manual steps, no validation, no guidance for troubleshooting
6. **Missing Development Setup**: No automated virtual environment, git hooks, or development tool configuration

### Target User Experience:
```bash
git clone https://github.com/ArtemisAI/raglite.git
cd raglite
./install.sh
# Everything works perfectly with optimal configuration
```

## 🏗️ **Detailed Implementation Plan**

### **Phase 1: Core Installation Scripts**

#### **1.1 Main Installation Script (`install.sh`)**
**Location**: `/install.sh`  
**Purpose**: Unix/Linux/macOS installation entry point

**Key Features**:
- Bash script with error handling (`set -euo pipefail`)
- Colorized output with progress indicators
- Logging to `install.log` for debugging
- Root privilege detection and handling
- Interactive and non-interactive modes

**Script Structure**:
```bash
#!/bin/bash
# RAGLite Smart Installation Script
# Automatically detects hardware and installs optimal configuration

set -euo pipefail

# Color codes and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/install.log"
PYTHON_MIN_VERSION="3.10"
VENV_NAME="raglite-env"

# Functions to implement:
- print_header()
- log_message()
- check_prerequisites()
- detect_system()
- detect_gpu()
- create_virtual_environment()
- install_dependencies()
- configure_environment()
- run_validation_tests()
- print_summary()
```

#### **1.2 Windows Installation Script (`install.ps1`)**
**Location**: `/install.ps1`  
**Purpose**: Windows PowerShell installation entry point

**Key Features**:
- PowerShell 5.1+ compatibility
- Windows-specific GPU detection (WMI queries)
- Chocolatey/Winget integration for system dependencies
- Windows-specific path handling
- UAC elevation handling

#### **1.3 Python Detection Module (`scripts/detect_system.py`)**
**Location**: `/scripts/detect_system.py`  
**Purpose**: Cross-platform system detection using Python

**Detection Capabilities**:
```python
import platform
import subprocess
import json
from pathlib import Path

class SystemDetector:
    def detect_all(self) -> dict:
        return {
            'os': self.detect_os(),
            'architecture': self.detect_architecture(),
            'python': self.detect_python(),
            'gpu': self.detect_gpu(),
            'cuda': self.detect_cuda(),
            'rocm': self.detect_rocm(),
            'memory': self.detect_memory(),
            'storage': self.detect_storage()
        }
    
    def detect_os(self) -> dict:
        # Linux, macOS, Windows detection
        # Distribution detection for Linux
        
    def detect_gpu(self) -> dict:
        # NVIDIA: nvidia-ml-py, nvidia-smi
        # AMD: ROCm detection, rocm-smi
        # Intel: Intel GPU detection
        # Apple: Metal support detection
        # Multiple GPU support
        
    def detect_cuda(self) -> dict:
        # CUDA toolkit version
        # CUDA runtime version
        # cuDNN availability
        # Compute capability
        
    def detect_python(self) -> dict:
        # Python version validation
        # Virtual environment support
        # pip/uv availability
```

### **Phase 2: Hardware-Specific Dependency Resolution**

#### **2.1 Dependency Resolver (`scripts/resolve_dependencies.py`)**
**Location**: `/scripts/resolve_dependencies.py`

**Smart Resolution Logic**:
```python
class DependencyResolver:
    def resolve_pytorch(self, system_info: dict) -> str:
        """Return optimal PyTorch installation command"""
        gpu_type = system_info['gpu']['type']
        cuda_version = system_info.get('cuda', {}).get('version')
        
        if gpu_type == 'nvidia' and cuda_version:
            return f"torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
        elif gpu_type == 'amd':
            return "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
        elif system_info['os']['name'] == 'darwin':
            return "torch torchvision torchaudio"  # Metal support
        else:
            return "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    def resolve_llama_cpp_python(self, system_info: dict) -> str:
        """Return optimal llama-cpp-python precompiled wheel URL"""
        # Logic for selecting correct precompiled binary based on:
        # - OS (linux/macos/windows)
        # - Architecture (x86_64/arm64)
        # - GPU type (cuda121/cuda122/cuda123/cuda124/metal/rocm)
        # - Python version (cp310/cp311/cp312)
        
    def resolve_optional_extras(self, features: list) -> list:
        """Return list of optional dependencies based on selected features"""
```

#### **2.2 Hardware Configuration Templates**
**Location**: `/scripts/hardware_configs/`

**Configuration Files**:
- `nvidia_gpu.json`: NVIDIA GPU optimal settings
- `amd_gpu.json`: AMD GPU optimal settings  
- `apple_metal.json`: Apple Silicon optimal settings
- `cpu_only.json`: CPU-only fallback settings
- `high_memory.json`: High VRAM configurations
- `low_memory.json`: Low VRAM configurations

**Example Configuration** (`nvidia_gpu.json`):
```json
{
  "pytorch": {
    "index_url": "https://download.pytorch.org/whl/cu121",
    "packages": ["torch", "torchvision", "torchaudio"]
  },
  "llama_cpp_python": {
    "accelerator": "cu121",
    "compile_args": {
      "CMAKE_ARGS": "-DLLAMA_CUBLAS=on",
      "CUDACXX": "/usr/local/cuda/bin/nvcc"
    }
  },
  "environment_variables": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
  },
  "model_recommendations": {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "llm": "microsoft/DialoGPT-medium"
  }
}
```

### **Phase 3: Interactive Installation Experience**

#### **3.1 Interactive Feature Selection**
**Implementation**: Terminal-based UI using `dialog` or pure bash

**Features to Configure**:
```bash
# Core Features (always installed)
- RAGLite core functionality
- SQLite backend with sqlite-vec
- Basic embedding models

# Optional Features (user selects)
□ GPU Acceleration (auto-detected, user confirms)
□ Chainlit Web Interface
□ Pandoc Document Conversion  
□ Ragas Evaluation Framework
□ Benchmarking Tools
□ Development Tools (testing, linting)
□ Pre-commit Hooks
□ Jupyter Notebook Support

# Advanced Configuration
□ Custom model downloads
□ Database backend selection (SQLite/DuckDB/PostgreSQL)  
□ API key configuration (OpenAI, Anthropic, etc.)
□ Resource limits (VRAM, CPU cores)
```

#### **3.2 Progress Indicators and Logging**
**Implementation**: Rich progress bars with detailed logging

**Progress Tracking**:
```bash
# Installation phases with progress
[1/8] System Detection.................. ████████████████████ 100%
[2/8] Virtual Environment............... ████████████████████ 100%  
[3/8] Core Dependencies................. ████████████████████ 100%
[4/8] GPU-Specific Packages............ ████████████████████ 100%
[5/8] Optional Features................ ████████████████████ 100%
[6/8] Environment Configuration........ ████████████████████ 100%
[7/8] Model Downloads................... ████████████████████ 100%
[8/8] Validation Tests.................. ████████████████████ 100%

✅ Installation completed successfully!
```

### **Phase 4: Environment Configuration**

#### **4.1 Virtual Environment Management**
**Location**: `/scripts/venv_manager.py`

**Capabilities**:
```python
class VirtualEnvironmentManager:
    def create_environment(self, python_path: str, env_name: str):
        """Create virtual environment with optimal Python version"""
        
    def activate_environment(self, env_path: str):
        """Activate virtual environment in current shell"""
        
    def install_packages(self, packages: list, env_path: str):
        """Install packages with retry logic and fallbacks"""
        
    def verify_installation(self, env_path: str) -> bool:
        """Verify all packages installed correctly"""
```

#### **4.2 Environment Variable Configuration**
**Location**: `.env.template` and automated `.env` generation

**Generated Configuration**:
```bash
# RAGLite Configuration
RAGLITE_DB_URL=sqlite:///./raglite.db
RAGLITE_CACHE_DIR=./cache
RAGLITE_LOG_LEVEL=INFO

# GPU Configuration (auto-detected)
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model Configuration
RAGLITE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAGLITE_LLM_MODEL=gpt-3.5-turbo

# API Keys (user provided)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Development Settings
RAGLITE_DEBUG=false
RAGLITE_PROFILE=false
```

#### **4.3 Git Hooks and Development Setup**
**Pre-commit Configuration**:
- Code formatting (ruff)
- Type checking (mypy)  
- Testing (pytest)
- Documentation generation

### **Phase 5: Validation and Testing System**

#### **5.1 Post-Installation Validation (`scripts/validate_installation.py`)**
**Tests to Run**:
```python
class InstallationValidator:
    def validate_core_imports(self):
        """Test basic raglite imports"""
        import raglite
        from raglite import RAGLiteConfig, insert, search
        
    def validate_gpu_acceleration(self):
        """Test GPU functionality if available"""
        import torch
        if torch.cuda.is_available():
            # Test CUDA functionality
            # Test model loading on GPU
            
    def validate_database_functionality(self):
        """Test SQLite-vec and database operations"""
        # Create test database
        # Insert test documents  
        # Perform search operations
        
    def validate_embedding_models(self):
        """Test embedding model loading and inference"""
        # Load embedding model
        # Generate test embeddings
        # Verify embedding dimensions
        
    def validate_optional_features(self, enabled_features: list):
        """Test enabled optional features"""
        if 'chainlit' in enabled_features:
            import chainlit
        if 'pandoc' in enabled_features:
            import pypandoc
```

#### **5.2 Performance Benchmarking**
**Location**: `/scripts/benchmark_installation.py`

**Benchmark Tests**:
- Embedding generation speed
- Vector search performance  
- GPU vs CPU comparison
- Memory usage profiling
- Database query performance

### **Phase 6: Error Handling and Recovery**

#### **6.1 Comprehensive Error Handling**
**Error Categories**:
```bash
# System Compatibility Errors
- Unsupported OS/Architecture
- Insufficient Python version
- Missing system dependencies

# Hardware Detection Errors  
- GPU driver issues
- CUDA/ROCm installation problems
- Insufficient VRAM/RAM

# Network/Download Errors
- Package download failures
- Model download issues
- Repository access problems

# Permission Errors
- Virtual environment creation
- File system permissions
- Package installation rights
```

**Recovery Strategies**:
```bash
# Automatic Fallbacks
- GPU → CPU fallback
- Precompiled → Source compilation
- Latest → Stable versions
- Full → Minimal installation

# User-Guided Recovery
- Clear error messages
- Specific fix suggestions
- Alternative installation paths
- Manual installation steps
```

#### **6.2 Troubleshooting Guide Generation**
**Location**: `/TROUBLESHOOTING.md` (auto-generated)

**Content**:
- Common installation issues
- System-specific problems
- GPU driver installation guides
- Manual installation alternatives
- Community support resources

### **Phase 7: Documentation and User Experience**

#### **7.1 Interactive Installation Guide**
**Location**: `/docs/installation-guide.md`

**Sections**:
1. **Quick Start**: Single command installation
2. **System Requirements**: Detailed compatibility matrix
3. **Advanced Installation**: Custom configuration options
4. **Troubleshooting**: Common issues and solutions
5. **Development Setup**: Contributing guidelines

#### **7.2 Installation Summary and Next Steps**
**Post-Installation Output**:
```bash
🎉 RAGLite Installation Complete!

📊 Your Configuration:
   • OS: Ubuntu 22.04 LTS
   • GPU: NVIDIA RTX 4090 (24GB VRAM)
   • CUDA: 12.1
   • Python: 3.11.5
   • Virtual Environment: ./raglite-env

🚀 Installed Features:
   ✅ Core RAGLite functionality
   ✅ GPU acceleration (CUDA)
   ✅ Chainlit web interface
   ✅ Development tools
   ✅ Benchmarking suite

🔧 Environment Variables:
   • Configuration: .env
   • Database: ./raglite.db
   • Cache: ./cache

📝 Next Steps:
   1. Activate environment: source raglite-env/bin/activate
   2. Test installation: python -c "import raglite; print('✅ RAGLite ready!')"
   3. Quick start: python examples/quick_start.py
   4. Web interface: chainlit run src/raglite/_chainlit.py
   5. Documentation: https://github.com/superlinear-ai/raglite

🆘 Need Help?
   • Troubleshooting: cat TROUBLESHOOTING.md
   • Issues: https://github.com/superlinear-ai/raglite/issues
   • Documentation: README.md
```

### **Phase 8: Continuous Integration Integration**

#### **8.1 Installation Script Testing**
**GitHub Actions Workflow** (`.github/workflows/test-installation.yml`):
```yaml
name: Test Installation Scripts

on:
  push:
    paths:
      - 'install.*'
      - 'scripts/**'
  pull_request:
    paths:
      - 'install.*'
      - 'scripts/**'

jobs:
  test-installation:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
        gpu: [cpu-only, mock-nvidia, mock-amd]
    
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Test installation script
        run: |
          # Test installation in clean environment
          # Validate all features work
          # Run benchmark tests
```

## 🎯 **Expected Outcomes**

### **User Experience Improvements**:
1. **One-Command Setup**: `./install.sh` handles everything
2. **Hardware Optimization**: Automatic detection and optimal configuration
3. **Zero Manual Configuration**: No need to research CUDA versions, PyTorch variants, etc.
4. **Robust Error Handling**: Clear messages and automatic recovery
5. **Complete Development Environment**: Ready for contribution immediately

### **Technical Benefits**:
1. **Consistent Environments**: Same setup across all machines
2. **Optimal Performance**: Hardware-specific optimizations
3. **Reduced Support Burden**: Fewer installation-related issues
4. **Better Testing**: Automated validation of functionality
5. **Documentation Accuracy**: Auto-generated system-specific guides

### **Developer Benefits**:
1. **Faster Onboarding**: New contributors can start immediately
2. **Consistent Development**: Same environment for all team members
3. **Easy Testing**: Multiple configuration testing
4. **Better CI/CD**: Consistent testing across environments

## 📁 **File Structure**

```
raglite/
├── install.sh                          # Main Unix installation script
├── install.ps1                         # Windows PowerShell script
├── TROUBLESHOOTING.md                   # Auto-generated troubleshooting guide
├── .env.template                        # Environment variable template
├── scripts/
│   ├── detect_system.py                 # System detection module
│   ├── resolve_dependencies.py          # Dependency resolution
│   ├── venv_manager.py                  # Virtual environment management
│   ├── validate_installation.py        # Post-installation validation
│   ├── benchmark_installation.py       # Performance benchmarking
│   └── hardware_configs/               # Hardware-specific configurations
│       ├── nvidia_gpu.json
│       ├── amd_gpu.json
│       ├── apple_metal.json
│       └── cpu_only.json
├── docs/
│   └── installation-guide.md           # Comprehensive installation docs
└── .github/workflows/
    └── test-installation.yml           # CI for installation scripts
```

## 🚀 **Implementation Priority**

### **Phase 1** (Critical - Core Functionality):
1. Basic `install.sh` with system detection
2. Virtual environment creation
3. Core dependency installation
4. Basic validation

### **Phase 2** (Important - Smart Features):  
1. GPU detection and optimization
2. Hardware-specific dependency resolution
3. Interactive feature selection
4. Comprehensive error handling

### **Phase 3** (Enhancement - Polish):
1. Windows PowerShell script
2. Advanced benchmarking
3. Performance optimization
4. Comprehensive documentation

This comprehensive installation script system will transform the user experience from a complex manual process to a simple one-command setup that works optimally on any system.
