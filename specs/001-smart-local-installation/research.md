# Research: Smart Local Installation Script System

## Hardware Detection Research

### Decision: Multi-layered Detection Approach
**Rationale**: Different hardware types require different detection methods for accuracy
**Implementation**: Python libraries with platform-specific system calls

#### NVIDIA GPU Detection
- **Primary**: nvidia-ml-py library for programmatic access
- **Fallback**: nvidia-smi command parsing
- **CUDA Detection**: nvcc version parsing, CUDA runtime version
- **Capabilities**: VRAM, compute capability, driver compatibility

#### AMD GPU Detection  
- **Primary**: ROCm installation detection via rocm-smi
- **Fallback**: System device enumeration
- **ROCm Version**: Package manager queries, library detection
- **Capabilities**: VRAM, architecture support

#### Apple Silicon Detection
- **Primary**: system_profiler Metal support queries
- **Architecture**: platform.machine() for Apple Silicon vs Intel
- **Metal Support**: Core ML availability, GPU memory

#### CPU-Only Systems
- **Memory Detection**: psutil for available RAM
- **Architecture**: platform.architecture() for x86_64/ARM64
- **Performance**: CPU count, threading capabilities

### Alternatives Considered:
- **Single detection method**: Rejected due to platform diversity
- **Runtime-only detection**: Rejected due to installation-time optimization needs
- **External service calls**: Rejected for offline capability requirements

## Dependency Resolution Research

### Decision: Template-Based Configuration Mapping
**Rationale**: Hardware-specific optimizations require different package variants and settings
**Implementation**: JSON configuration templates with dynamic resolution

#### PyTorch Variant Selection
- **CUDA 12.1**: `--index-url https://download.pytorch.org/whl/cu121`
- **CUDA 12.4**: `--index-url https://download.pytorch.org/whl/cu124`  
- **ROCm 5.6**: `--index-url https://download.pytorch.org/whl/rocm5.6`
- **Metal (macOS)**: Default PyPI with Metal support
- **CPU-only**: `--index-url https://download.pytorch.org/whl/cpu`

#### llama-cpp-python Binary Selection
- **URL Pattern**: `https://github.com/abetlen/llama-cpp-python/releases/download/v{version}-{accelerator}/llama_cpp_python-{version}-cp{python}-cp{python}-{platform}.whl`
- **Accelerators**: cu121, cu122, cu123, cu124, metal, rocm, cpu
- **Platforms**: linux_x86_64, macosx_11_0_arm64, win_amd64
- **Python Versions**: cp310, cp311, cp312

### Alternatives Considered:
- **Compile from source**: Rejected due to complexity and time requirements
- **Single binary for all**: Rejected due to performance limitations
- **Runtime switching**: Rejected due to installation-time optimization goals

## Cross-Platform Shell Scripting Research

### Decision: Hybrid Python + Shell Approach
**Rationale**: Python provides cross-platform detection, shell scripts provide native integration
**Implementation**: Shell orchestration calling Python libraries

#### Unix Shell (Bash)
- **Error Handling**: `set -euo pipefail` for strict error catching
- **Progress Indicators**: ANSI escape codes for colors and progress bars
- **Privilege Handling**: sudo detection and delegation
- **Virtual Environment**: Python venv module integration

#### Windows PowerShell
- **Execution Policy**: Bypass for installation scripts
- **UAC Handling**: Administrative privilege detection and elevation
- **Progress Indicators**: Write-Progress cmdlets
- **GPU Detection**: WMI queries via Get-WmiObject

### Alternatives Considered:
- **Python-only solution**: Rejected due to platform-specific optimizations
- **Pure shell scripting**: Rejected due to complex hardware detection needs
- **Docker-based installation**: Rejected for local development focus

## Performance Optimization Research

### Decision: Hardware-Specific Environment Variables
**Rationale**: GPU memory management and threading optimization critical for RAGLite performance
**Implementation**: Automatic configuration based on detected hardware

#### NVIDIA Optimizations
- **Memory Management**: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- **GPU Selection**: `CUDA_VISIBLE_DEVICES` based on available GPUs
- **Threading**: `OMP_NUM_THREADS` based on CPU cores

#### AMD Optimizations  
- **ROCm Settings**: `HIP_VISIBLE_DEVICES` for GPU selection
- **Memory**: ROCm-specific allocation patterns
- **Compute Units**: Optimal thread counts for RDNA architectures

#### Apple Silicon Optimizations
- **Metal Memory**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- **Threading**: ARM64-optimized thread counts
- **Memory Pressure**: Automatic memory management settings

### Alternatives Considered:
- **Runtime auto-detection**: Rejected due to startup time impact
- **Manual configuration**: Rejected due to user complexity
- **One-size-fits-all**: Rejected due to performance differences

## Validation Strategy Research

### Decision: Multi-Level Testing Approach
**Rationale**: Installation success requires validation at multiple levels
**Implementation**: Progressive validation from basic imports to performance

#### Level 1: Import Validation
- **Core RAGLite**: Basic package imports and version verification
- **Dependencies**: PyTorch, transformers, sqlite-vec availability
- **GPU Libraries**: CUDA/ROCm/Metal accessibility

#### Level 2: Functionality Validation
- **Database**: SQLite-vec extension loading and basic operations
- **Embeddings**: Model loading and basic inference
- **GPU Acceleration**: Hardware-specific acceleration testing

#### Level 3: Performance Validation
- **Embedding Speed**: Benchmark against expected performance
- **Memory Usage**: Validate optimal memory configuration
- **GPU Utilization**: Confirm hardware acceleration active

### Alternatives Considered:
- **Basic import testing only**: Rejected due to configuration dependency
- **Full integration testing**: Rejected due to installation time constraints
- **External benchmarking**: Rejected for offline capability requirements
