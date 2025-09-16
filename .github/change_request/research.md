# Research: Hardware Detection & Integration Analysis

**Date**: September 16, 2025  
**Objective**: Analyze cross-platform hardware detection approaches and integration patterns with existing RAGLite GPU infrastructure

## Current Infrastructure Analysis

### Existing GPU Detection (`src/raglite/_gpu_utils.py`)
RAGLite already implements solid GPU detection foundations:

**✅ Implemented Capabilities:**
- **CUDA Detection**: `detect_cuda_availability()` using PyTorch backend
- **Memory Analysis**: `get_gpu_memory_info()` returns total/available VRAM
- **Layer Optimization**: `calculate_optimal_gpu_layers()` for memory-aware configuration
- **Error Handling**: Graceful fallback when PyTorch unavailable

**🔄 Integration Opportunities:**
- Extend detection to cover AMD ROCm and Apple Metal
- Add system-level GPU detection before PyTorch dependency
- Include GPU driver version and compute capability detection
- Enhance cross-platform compatibility (currently CUDA-focused)

### GPU-Aware Implementation (`src/raglite/_embedding_gpu.py`)
**✅ Working Infrastructure:**
- `GPUAwareLlamaLLM.create()` factory with automatic GPU optimization
- Memory-based layer allocation for optimal performance
- Configuration integration with `RAGLiteConfig`

**❌ Known Issue (from TODO.md):**
- Model string parsing bug: `llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512` fails validation
- Need to strip `llama-cpp-python/` prefix before `Llama.from_pretrained()`
- **Installation Impact**: This bug affects post-installation validation tests

## Hardware Detection Research

### Cross-Platform GPU Detection Strategies

#### 1. System-Level Detection (Pre-PyTorch)
**NVIDIA Detection:**
```python
# Option A: nvidia-ml-py (preferred for detailed info)
import pynvml
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(handle)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

# Option B: subprocess nvidia-smi (fallback)
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                       capture_output=True, text=True)
```

**AMD ROCm Detection:**
```python  
# ROCm detection via rocm-smi
result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo'], 
                       capture_output=True, text=True)

# Alternative: Check /opt/rocm/ installation
rocm_path = Path('/opt/rocm')
rocm_available = rocm_path.exists() and (rocm_path / 'bin/rocm-smi').exists()
```

**Apple Metal Detection:**
```python
# macOS Metal detection
import platform
import subprocess

if platform.system() == 'Darwin':
    # Check for Apple Silicon
    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                           capture_output=True, text=True)
    is_apple_silicon = 'Apple' in result.stdout
    
    # Check Metal support
    result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                           capture_output=True, text=True)
```

#### 2. Recommended Detection Flow
```python
class SystemDetector:
    def detect_gpu(self) -> GPUInfo:
        """Comprehensive GPU detection with fallbacks"""
        # 1. Try system-level detection first (no dependencies)
        gpu_info = self._detect_system_level_gpu()
        
        # 2. If PyTorch available, enhance with runtime info  
        if self._pytorch_available():
            gpu_info = self._enhance_with_pytorch(gpu_info)
        
        # 3. Validate against existing RAGLite detection
        if gpu_info.has_cuda:
            gpu_info.cuda_functional = detect_cuda_availability()  # Use existing
            
        return gpu_info
```

### Dependency Resolution Patterns

#### PyTorch Installation Matrix
**CUDA Variants** (based on detected CUDA version):
```python
PYTORCH_CUDA_MATRIX = {
    "12.1": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "12.2": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122", 
    "12.4": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
    "cpu": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
}
```

**ROCm Variants:**
```python  
PYTORCH_ROCM_MATRIX = {
    "5.6": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6",
    "5.7": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7"
}
```

#### llama-cpp-python Binary Selection
**URL Pattern Analysis:**
```
https://github.com/abetlen/llama-cpp-python/releases/download/v{VERSION}-{ACCELERATOR}/
llama_cpp_python-{VERSION}-cp{PYTHON_VERSION}-cp{PYTHON_VERSION}-{PLATFORM}.whl

Variables:
- VERSION: 0.3.9, 0.3.10, etc.
- ACCELERATOR: metal, cu121, cu122, cu123, cu124, rocm  
- PYTHON_VERSION: 310, 311, 312
- PLATFORM: macosx_11_0_arm64, linux_x86_64, win_amd64
```

**Selection Logic:**
```python
def resolve_llama_cpp_python_url(system_info: SystemInfo, gpu_info: GPUInfo) -> str:
    """Generate optimal llama-cpp-python wheel URL"""
    version = "0.3.9"  # Latest stable
    
    # Determine accelerator
    if gpu_info.type == "nvidia" and gpu_info.cuda.version:
        accelerator = f"cu{gpu_info.cuda.version.replace('.', '')}"
    elif gpu_info.type == "amd" and gpu_info.rocm.available:
        accelerator = "rocm"  
    elif system_info.os.name == "darwin" and gpu_info.type == "apple":
        accelerator = "metal"
    else:
        accelerator = "cpu"  # Fallback
    
    # Build URL with proper error handling
    return f"https://github.com/abetlen/llama-cpp-python/releases/download/v{version}-{accelerator}/llama_cpp_python-{version}-cp{system_info.python.version}-cp{system_info.python.version}-{system_info.platform}.whl"
```

## Integration Strategy with Existing Infrastructure

### Leverage Current GPU Detection
**Enhanced Detection Pipeline:**
```python
def enhanced_gpu_detection(existing_detection: bool = None) -> GPUInfo:
    """Combine new system detection with existing RAGLite GPU utils"""
    
    # 1. System-level detection (new)
    gpu_info = detect_system_gpus()  
    
    # 2. Validate with existing detection (if PyTorch available)
    if existing_detection is not None:
        gpu_info.cuda_functional = existing_detection  # From detect_cuda_availability()
    
    # 3. Enhanced memory analysis (leverage existing)  
    if gpu_info.cuda_functional:
        memory_info = get_gpu_memory_info()  # Use existing function
        if memory_info:
            gpu_info.vram_total, gpu_info.vram_available = memory_info
            gpu_info.optimal_layers = calculate_optimal_gpu_layers(gpu_info.vram_total)
    
    return gpu_info
```

### Configuration Template Generation
**Hardware-Specific Optimization Files:**
```json
// scripts/hardware_configs/nvidia_gpu.json
{
  "pytorch": {
    "index_url": "https://download.pytorch.org/whl/cu121",
    "packages": ["torch", "torchvision", "torchaudio"]
  },
  "llama_cpp_python": {
    "url_template": "https://github.com/abetlen/llama-cpp-python/releases/download/v{version}-cu121/llama_cpp_python-{version}-cp{python}-cp{python}-linux_x86_64.whl"
  },
  "environment_variables": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
  },
  "raglite_config": {
    "gpu_layers": "auto",  // Will use calculate_optimal_gpu_layers()
    "gpu_memory_fraction": 0.8
  }
}
```

### Installation Validation Integration
**Post-Installation Testing Strategy:**
```python
def validate_gpu_installation(gpu_info: GPUInfo) -> ValidationResults:
    """Validate GPU setup using existing RAGLite infrastructure"""
    
    results = ValidationResults()
    
    # 1. Basic import validation
    try:
        import raglite
        from raglite import RAGLiteConfig
        results.basic_import = True
    except ImportError as e:
        results.basic_import = False
        results.errors.append(f"RAGLite import failed: {e}")
    
    # 2. GPU detection validation (use existing)
    if gpu_info.expected_cuda:
        cuda_detected = detect_cuda_availability()  # Existing function
        results.gpu_detection = cuda_detected
        if not cuda_detected:
            results.warnings.append("CUDA expected but not detected by PyTorch")
    
    # 3. Model loading test (address TODO.md bug)
    try:
        config = RAGLiteConfig()
        # Use a simple model string to avoid parsing bug
        test_model = "sentence-transformers/all-MiniLM-L6-v2"
        llm = GPUAwareLlamaLLM.create(test_model, config, embedding=True)  
        results.model_loading = True
    except Exception as e:
        results.model_loading = False
        results.errors.append(f"Model loading failed: {e}")
    
    return results
```

## Technical Recommendations

### 1. Modular Detection Architecture
- **Create**: `SystemDetector` class for cross-platform hardware detection
- **Extend**: Existing `_gpu_utils.py` with enhanced GPU type detection
- **Integrate**: New detection with existing CUDA functionality validation

### 2. Configuration Management
- **JSON Templates**: Hardware-specific configuration files for optimal settings
- **Environment Generation**: Automated `.env` file creation with detected optimizations  
- **Integration**: Respect existing `RAGLiteConfig` patterns and extend them

### 3. Error Handling Strategy
- **Graceful Degradation**: GPU detection failure → CPU-only installation
- **Specific Recovery**: Different fallback strategies for different failure types
- **User Guidance**: Clear error messages with actionable recovery steps

### 4. Cross-Platform Implementation
- **Bash Script**: Unix/Linux/macOS installation with comprehensive error handling
- **PowerShell Script**: Windows equivalent with WMI-based GPU detection
- **Python Modules**: Shared system detection logic used by both script types

## Next Phase Requirements

### Phase 1 Deliverables
1. **Data Model**: Standardized schemas for system/GPU detection results
2. **API Contracts**: Interfaces for detection, dependency resolution, validation
3. **Integration Guide**: How new installation scripts leverage existing GPU infrastructure
4. **Bug Resolution**: Address model string parsing issue identified in TODO.md

### Critical Integration Points
- **GPU Detection**: Enhance rather than replace existing `_gpu_utils.py` 
- **Configuration**: Extend `_config.py` patterns for installation-specific settings
- **Validation**: Integrate with existing test suite and GPU acceleration tests
- **Documentation**: Update existing docs to include installation automation

---

**Status**: ✅ Ready for Phase 1 design and contracts
**Key Finding**: Strong existing GPU infrastructure provides solid foundation for installation automation
