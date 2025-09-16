# Data Model: Installation System Schemas

**Date**: September 16, 2025  
**Purpose**: Define standardized data structures for cross-platform system detection and dependency resolution

## Core Entity Schemas

### SystemInfo Entity
**Purpose**: Comprehensive system environment detection results

```python
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

@dataclass
class PythonInfo:
    """Python environment information"""
    version: str              # e.g., "3.11.5"
    version_tuple: tuple      # e.g., (3, 11, 5)
    executable_path: str      # Full path to Python executable
    is_virtual_env: bool      # Running in virtual environment
    venv_path: Optional[str]  # Path to virtual environment if applicable
    pip_available: bool       # pip command accessible
    uv_available: bool        # uv command accessible
    site_packages: str        # Site packages directory

@dataclass
class OSInfo:
    """Operating system information"""
    name: Literal["linux", "darwin", "windows"]    # Normalized OS name
    version: str                                   # OS version string
    distribution: Optional[str]                    # Linux distribution (ubuntu, fedora, etc.)
    architecture: Literal["x86_64", "arm64", "aarch64", "i386"]
    platform_string: str                         # Full platform identifier for wheel selection
    is_wsl: bool                                  # Windows Subsystem for Linux detection
    package_manager: Optional[str]                # apt, brew, choco, winget, etc.

@dataclass
class HardwareInfo:
    """System hardware specifications"""
    cpu_cores: int            # Physical CPU cores
    cpu_threads: int          # Logical CPU threads  
    total_ram_gb: float       # Total system RAM in GB
    available_ram_gb: float   # Available RAM in GB
    total_storage_gb: float   # Available storage in GB
    swap_size_gb: float       # Swap/page file size

@dataclass
class SystemInfo:
    """Complete system detection results"""
    os: OSInfo
    python: PythonInfo
    hardware: HardwareInfo
    detected_at: str          # ISO timestamp of detection
    detection_duration_ms: int # Time taken for detection
```

### GPUInfo Entity  
**Purpose**: Multi-vendor GPU detection with capability assessment

```python
from enum import Enum
from typing import List, Optional

class GPUType(Enum):
    """Supported GPU types"""
    NVIDIA = "nvidia"
    AMD = "amd" 
    INTEL = "intel"
    APPLE_METAL = "apple_metal"
    NONE = "none"

@dataclass
class CUDAInfo:
    """NVIDIA CUDA environment details"""
    available: bool           # CUDA toolkit installed
    version: Optional[str]    # e.g., "12.1"
    runtime_version: Optional[str]  # CUDA runtime version
    driver_version: Optional[str]   # NVIDIA driver version
    cudnn_available: bool     # cuDNN library available
    nvcc_available: bool      # NVCC compiler available
    
@dataclass  
class ROCmInfo:
    """AMD ROCm environment details"""
    available: bool           # ROCm installed
    version: Optional[str]    # e.g., "5.6.0"
    hip_version: Optional[str]  # HIP runtime version
    rocm_smi_available: bool  # rocm-smi tool available

@dataclass
class GPUDevice:
    """Individual GPU device information"""
    index: int                # Device index (0, 1, 2, ...)
    name: str                 # Device name/model
    vram_total_mb: int        # Total VRAM in MB
    vram_available_mb: int    # Available VRAM in MB  
    compute_capability: Optional[str]  # NVIDIA compute capability (e.g., "8.6")
    pci_id: Optional[str]     # PCI device identifier
    power_limit_w: Optional[int]  # Power limit in watts

@dataclass
class GPUInfo:
    """Complete GPU environment assessment"""
    type: GPUType
    devices: List[GPUDevice]  # All detected GPU devices
    primary_device: Optional[GPUDevice]  # Primary/recommended device
    cuda: Optional[CUDAInfo]  # NVIDIA-specific information
    rocm: Optional[ROCmInfo]  # AMD-specific information
    metal_available: bool     # Apple Metal support (macOS)
    pytorch_gpu_available: bool  # PyTorch GPU backend functional
    optimal_gpu_layers: Optional[int]  # Recommended layers for LLaMA models
    total_vram_gb: float      # Combined VRAM across all devices
```

### InstallationConfig Entity
**Purpose**: Installation settings and feature selections

```python
from typing import Set, Dict, Any
from enum import Enum

class InstallationMode(Enum):
    """Installation execution modes"""
    INTERACTIVE = "interactive"      # User prompts and feature selection
    AUTOMATIC = "automatic"          # Auto-detect optimal configuration  
    MINIMAL = "minimal"             # Core features only
    DEVELOPMENT = "development"      # Full development environment

class FeatureSet(Enum):
    """Available installation features"""
    CORE = "core"                   # Always included
    GPU_ACCELERATION = "gpu"        # GPU optimization
    CHAINLIT_UI = "chainlit"       # Web interface
    PANDOC_CONVERSION = "pandoc"    # Document conversion
    RAGAS_EVALUATION = "ragas"      # Evaluation framework
    BENCHMARKING = "bench"         # Performance benchmarking
    DEVELOPMENT_TOOLS = "dev"       # Testing, linting, pre-commit
    JUPYTER_SUPPORT = "jupyter"     # Notebook integration

@dataclass
class DependencyResolution:
    """Resolved package installation commands"""
    pytorch_install_cmd: str        # PyTorch installation command
    llama_cpp_python_url: str      # llama-cpp-python wheel URL
    extra_packages: List[str]       # Additional packages for selected features
    pip_index_urls: List[str]       # Additional package index URLs
    environment_variables: Dict[str, str]  # Required environment variables

@dataclass  
class InstallationConfig:
    """Complete installation configuration"""
    mode: InstallationMode
    features: Set[FeatureSet]
    system_info: SystemInfo
    gpu_info: GPUInfo
    dependencies: DependencyResolution
    
    # Environment setup
    venv_name: str                  # Virtual environment name
    venv_path: str                  # Full path to virtual environment
    install_path: str               # Installation directory
    
    # Configuration files
    env_file_content: Dict[str, str]    # .env file variables
    config_template: str               # Hardware-specific config template
    
    # Validation settings  
    validation_tests: List[str]        # Tests to run post-installation
    benchmark_tests: List[str]         # Performance benchmarks to execute
    
    # Installation tracking
    created_at: str                    # Installation timestamp
    estimated_duration_min: int       # Estimated installation time
```

### ValidationResults Entity
**Purpose**: Post-installation testing and verification results

```python  
@dataclass
class TestResult:
    """Individual test execution result"""
    name: str                       # Test name/identifier
    passed: bool                    # Test success status
    duration_ms: int                # Test execution time
    message: Optional[str]          # Success/failure message
    details: Optional[Dict[str, Any]]  # Additional test data

@dataclass
class BenchmarkResult:
    """Performance benchmark measurement"""
    metric_name: str                # Benchmark metric (e.g., "embedding_speed")
    value: float                    # Measured value
    unit: str                       # Unit of measurement (e.g., "tokens/sec")
    baseline_value: Optional[float] # Expected/baseline value for comparison
    improvement_factor: Optional[float]  # Performance improvement ratio

@dataclass
class ValidationResults:
    """Complete installation validation results"""
    overall_success: bool           # All critical tests passed
    
    # Core functionality tests
    basic_imports: TestResult       # RAGLite import test
    gpu_detection: Optional[TestResult]  # GPU detection validation
    model_loading: TestResult       # Basic model loading test
    database_connectivity: TestResult   # SQLite/vector DB test
    
    # Feature-specific tests
    feature_tests: Dict[FeatureSet, TestResult]  # Optional feature validation
    
    # Performance measurements
    benchmarks: List[BenchmarkResult]   # Performance benchmark results
    
    # Installation verification
    environment_setup: TestResult   # Virtual environment validation
    dependency_versions: TestResult # Package version verification
    configuration_files: TestResult # Config file generation validation
    
    # Error tracking
    warnings: List[str]             # Non-critical issues
    errors: List[str]               # Critical failures
    recovery_suggestions: List[str] # Actionable recovery steps
    
    # Summary metrics
    total_tests: int                # Total tests executed
    passed_tests: int               # Number of successful tests
    execution_time_min: float       # Total validation time
    installation_quality_score: float  # Overall quality rating (0-100)
```

## Hardware Configuration Templates Schema

### Template Structure
**Purpose**: JSON configuration files for hardware-specific optimizations

```python
@dataclass
class OptimizationTemplate:
    """Hardware-specific optimization settings"""
    
    # Package installation
    pytorch_config: Dict[str, Any]      # PyTorch installation settings
    llama_cpp_config: Dict[str, Any]    # llama-cpp-python configuration
    optional_packages: List[str]        # Hardware-specific optional packages
    
    # Environment variables  
    cuda_settings: Dict[str, str]       # CUDA-specific environment variables
    pytorch_settings: Dict[str, str]    # PyTorch optimization settings
    raglite_settings: Dict[str, str]    # RAGLite-specific settings
    
    # Model recommendations
    recommended_embedding_model: str     # Optimal embedding model
    recommended_llm_model: str          # Optimal LLM model  
    max_context_length: int             # Recommended context window
    
    # Performance tuning
    gpu_memory_fraction: float          # GPU memory allocation ratio
    cpu_threads: Optional[int]          # CPU thread count override
    batch_size_recommendations: Dict[str, int]  # Optimal batch sizes
    
    # Validation criteria
    expected_performance_metrics: Dict[str, float]  # Performance expectations
    critical_tests: List[str]           # Must-pass validation tests
```

### Template Examples

**NVIDIA GPU Template** (`scripts/hardware_configs/nvidia_gpu.json`):
```json
{
  "pytorch_config": {
    "index_url": "https://download.pytorch.org/whl/cu121",
    "packages": ["torch", "torchvision", "torchaudio"],
    "verify_cuda": true
  },
  "llama_cpp_config": {
    "accelerator": "cu121",
    "cmake_args": {
      "CMAKE_ARGS": "-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native"
    },
    "force_compile": false
  },
  "cuda_settings": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    "CUDA_LAUNCH_BLOCKING": "0"
  },
  "raglite_settings": {
    "RAGLITE_GPU_LAYERS": "auto",
    "RAGLITE_GPU_MEMORY_FRACTION": "0.8",
    "RAGLITE_BATCH_SIZE": "32"
  },
  "performance_expectations": {
    "embedding_tokens_per_sec": 10000.0,
    "gpu_memory_efficiency": 0.8,
    "inference_speedup_vs_cpu": 3.0
  }
}
```

**CPU-Only Template** (`scripts/hardware_configs/cpu_only.json`):
```json
{
  "pytorch_config": {
    "index_url": "https://download.pytorch.org/whl/cpu", 
    "packages": ["torch", "torchvision", "torchaudio"]
  },
  "llama_cpp_config": {
    "accelerator": "cpu",
    "cmake_args": {
      "CMAKE_ARGS": "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
    }
  },
  "raglite_settings": {
    "RAGLITE_CPU_THREADS": "auto",
    "RAGLITE_BATCH_SIZE": "8",
    "RAGLITE_USE_MLOCK": "true"
  },
  "performance_expectations": {
    "embedding_tokens_per_sec": 1000.0,
    "cpu_efficiency": 0.9
  }
}
```

## State Management Schema

### Installation State Tracking
**Purpose**: Enable installation recovery and progress monitoring

```python
@dataclass  
class InstallationStep:
    """Individual installation step tracking"""
    step_id: str                    # Unique step identifier
    name: str                       # Human-readable step name
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    started_at: Optional[str]       # ISO timestamp when started
    completed_at: Optional[str]     # ISO timestamp when finished  
    duration_ms: Optional[int]      # Step execution time
    error_message: Optional[str]    # Failure details if status == "failed"
    retry_count: int                # Number of retry attempts
    recoverable: bool               # Can this step be retried

@dataclass
class InstallationState:
    """Complete installation progress tracking"""
    installation_id: str           # Unique installation identifier
    config: InstallationConfig     # Installation configuration
    
    # Progress tracking
    current_step: str               # Currently executing step
    completed_steps: List[str]      # Successfully completed step IDs
    failed_steps: List[str]         # Failed step IDs
    total_steps: int                # Total number of steps
    progress_percentage: float      # Installation progress (0-100)
    
    # Execution details
    started_at: str                 # Installation start timestamp
    estimated_completion: str       # Estimated completion time
    steps: Dict[str, InstallationStep]  # Detailed step tracking
    
    # Recovery information
    can_resume: bool                # Installation can be resumed
    resume_from_step: Optional[str] # Step to resume from
    cleanup_required: bool          # Partial installation needs cleanup
    
    # Logging
    log_file_path: str              # Path to detailed installation log
    debug_info: Dict[str, Any]      # Additional debugging information
```

## Schema Validation & Serialization

### JSON Schema Validation
**Purpose**: Ensure data integrity across installation process

```python
import json
from jsonschema import validate, ValidationError

# Example schema for SystemInfo validation
SYSTEM_INFO_SCHEMA = {
    "type": "object",
    "required": ["os", "python", "hardware", "detected_at"],
    "properties": {
        "os": {
            "type": "object", 
            "required": ["name", "version", "architecture"],
            "properties": {
                "name": {"type": "string", "enum": ["linux", "darwin", "windows"]},
                "architecture": {"type": "string", "enum": ["x86_64", "arm64", "aarch64", "i386"]}
            }
        },
        "python": {
            "type": "object",
            "required": ["version", "version_tuple", "executable_path"],
            "properties": {
                "version_tuple": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 3,
                    "maxItems": 3
                }
            }
        }
    }
}

def validate_system_info(data: Dict[str, Any]) -> bool:
    """Validate SystemInfo data against schema"""
    try:
        validate(instance=data, schema=SYSTEM_INFO_SCHEMA)
        return True
    except ValidationError as e:
        logger.error(f"SystemInfo validation failed: {e.message}")
        return False
```

---

**Status**: ✅ Data model schemas defined and validated
**Next**: Create API contracts for system detection and dependency resolution interfaces
