# API Contract: DependencyResolver

**Interface**: `DependencyResolver`  
**Purpose**: Hardware-aware dependency resolution and package selection  
**Integration**: Uses SystemInfo and GPUInfo to select optimal packages

## Interface Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from .data_model import SystemInfo, GPUInfo, DependencyResolution, FeatureSet

class DependencyResolver(ABC):
    """Abstract interface for intelligent dependency resolution"""
    
    @abstractmethod
    def resolve_all_dependencies(
        self, 
        system_info: SystemInfo, 
        gpu_info: GPUInfo, 
        features: Set[FeatureSet]
    ) -> DependencyResolution:
        """
        Resolve complete dependency set for given system and features.
        
        Args:
            system_info: Detected system environment
            gpu_info: GPU hardware information  
            features: Selected installation features
            
        Returns:
            Complete dependency resolution with installation commands
            
        Raises:
            DependencyResolutionError: Cannot resolve compatible dependencies
        """
        pass
    
    @abstractmethod
    def resolve_pytorch(self, system_info: SystemInfo, gpu_info: GPUInfo) -> str:
        """
        Determine optimal PyTorch installation command.
        
        Args:
            system_info: System environment details
            gpu_info: GPU hardware configuration
            
        Returns:
            pip install command for optimal PyTorch variant
            Examples:
            - "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            - "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            
        Resolution Priority:
            1. NVIDIA GPU + CUDA: CUDA-optimized PyTorch
            2. AMD GPU + ROCm: ROCm-optimized PyTorch  
            3. Apple Silicon: Metal-optimized PyTorch
            4. Fallback: CPU-only PyTorch
        """
        pass
    
    @abstractmethod
    def resolve_llama_cpp_python(self, system_info: SystemInfo, gpu_info: GPUInfo) -> str:
        """
        Generate llama-cpp-python precompiled wheel URL.
        
        Args:
            system_info: System platform information
            gpu_info: GPU acceleration capabilities
            
        Returns:
            Direct URL to optimal llama-cpp-python wheel
            
        URL Format:
            https://github.com/abetlen/llama-cpp-python/releases/download/
            v{VERSION}-{ACCELERATOR}/llama_cpp_python-{VERSION}-cp{PYTHON}-cp{PYTHON}-{PLATFORM}.whl
            
        Accelerator Selection:
            - NVIDIA: cu121, cu122, cu123, cu124 (based on CUDA version)
            - AMD: rocm (if ROCm available)
            - Apple: metal (for Apple Silicon)
            - Fallback: cpu
        """
        pass
    
    @abstractmethod  
    def resolve_optional_features(self, features: Set[FeatureSet]) -> List[str]:
        """
        Generate package list for selected optional features.
        
        Args:
            features: Set of requested features
            
        Returns:
            List of pip package specifications
            
        Feature Mappings:
            - CHAINLIT_UI: ["chainlit>=2.0.0"]
            - PANDOC_CONVERSION: ["pypandoc-binary>=1.13"]  
            - RAGAS_EVALUATION: ["pandas>=2.1.1", "ragas>=0.1.12"]
            - BENCHMARKING: ["faiss-cpu>=1.11.0", "ir_datasets>=0.5.10", ...]
            - DEVELOPMENT_TOOLS: ["pytest>=8.3.4", "mypy>=1.14.1", "ruff>=0.10.0", ...]
        """
        pass
    
    @abstractmethod
    def generate_environment_variables(
        self, 
        system_info: SystemInfo, 
        gpu_info: GPUInfo,
        features: Set[FeatureSet]
    ) -> Dict[str, str]:
        """
        Generate optimal environment variables for configuration.
        
        Args:
            system_info: System details for path configuration
            gpu_info: GPU settings for optimization variables
            features: Enabled features requiring environment setup
            
        Returns:
            Dictionary of environment variable key-value pairs
            
        Variable Categories:
            - CUDA Settings: CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF
            - RAGLite Config: RAGLITE_DB_URL, RAGLITE_CACHE_DIR, RAGLITE_GPU_LAYERS
            - Performance: OMP_NUM_THREADS, TOKENIZERS_PARALLELISM
        """
        pass
    
    @abstractmethod
    def validate_dependency_compatibility(
        self, 
        resolution: DependencyResolution
    ) -> Dict[str, bool]:
        """
        Validate resolved dependencies for version compatibility.
        
        Args:
            resolution: Complete dependency resolution
            
        Returns:
            Dict mapping dependency name to compatibility status
            
        Validation Checks:
            - PyTorch CUDA version matches system CUDA
            - llama-cpp-python accelerator matches GPU type
            - Python version compatibility across all packages
            - No conflicting package versions
        """
        pass
```

## Hardware-Specific Resolution Logic

### CUDA Version Mapping Contract
```python
class CUDADependencyResolver:
    """NVIDIA CUDA-specific dependency resolution logic"""
    
    # CUDA version to PyTorch index mapping
    CUDA_PYTORCH_MATRIX = {
        "12.4": "https://download.pytorch.org/whl/cu124",
        "12.3": "https://download.pytorch.org/whl/cu123", 
        "12.2": "https://download.pytorch.org/whl/cu122",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "11.8": "https://download.pytorch.org/whl/cu118",
        "fallback": "https://download.pytorch.org/whl/cpu"
    }
    
    # CUDA version to llama-cpp-python accelerator mapping
    CUDA_LLAMACPP_MATRIX = {
        "12.4": "cu124",
        "12.3": "cu123",
        "12.2": "cu122", 
        "12.1": "cu121",
        "11.8": "cu118",
        "fallback": "cpu"
    }
    
    def resolve_cuda_pytorch(self, cuda_version: str) -> str:
        """
        Resolve CUDA-optimized PyTorch for specific CUDA version.
        
        Args:
            cuda_version: Detected CUDA version (e.g., "12.1")
            
        Returns:
            PyTorch installation command with correct CUDA index
            
        Fallback Strategy:
            1. Exact CUDA version match
            2. Nearest lower CUDA version  
            3. CPU-only PyTorch as final fallback
        """
        # Find best match or fallback
        index_url = self.CUDA_PYTORCH_MATRIX.get(cuda_version)
        if not index_url:
            # Try fallback to lower version
            index_url = self._find_compatible_cuda_version(cuda_version)
        
        return f"torch torchvision torchaudio --index-url {index_url}"
    
    def resolve_cuda_llamacpp(self, cuda_version: str, system_info: SystemInfo) -> str:
        """
        Generate CUDA-optimized llama-cpp-python wheel URL.
        
        Args:
            cuda_version: CUDA version for accelerator selection
            system_info: Platform details for wheel selection
            
        Returns:
            Direct URL to CUDA-optimized llama-cpp-python wheel
        """
        accelerator = self.CUDA_LLAMACPP_MATRIX.get(cuda_version, "cpu")
        platform = self._map_platform_string(system_info.os.platform_string)
        python_version = "".join(str(x) for x in system_info.python.version_tuple[:2])
        
        version = "0.3.9"  # Current stable version
        url = f"https://github.com/abetlen/llama-cpp-python/releases/download/v{version}-{accelerator}/llama_cpp_python-{version}-cp{python_version}-cp{python_version}-{platform}.whl"
        
        return url
```

### ROCm Resolution Contract
```python
class ROCmDependencyResolver:
    """AMD ROCm-specific dependency resolution logic"""
    
    ROCM_PYTORCH_MATRIX = {
        "5.7": "https://download.pytorch.org/whl/rocm5.7",
        "5.6": "https://download.pytorch.org/whl/rocm5.6",
        "5.5": "https://download.pytorch.org/whl/rocm5.5",
        "fallback": "https://download.pytorch.org/whl/cpu"
    }
    
    def resolve_rocm_dependencies(self, rocm_version: str, system_info: SystemInfo) -> DependencyResolution:
        """
        Resolve ROCm-optimized dependencies.
        
        Args:
            rocm_version: Detected ROCm version
            system_info: System platform information
            
        Returns:
            Complete dependency resolution for ROCm setup
            
        Additional Requirements:
            - ROCm-specific environment variables (HIP_VISIBLE_DEVICES)
            - AMD GPU optimization settings
            - ROCm library path configuration
        """
        pass
```

### Apple Silicon Resolution Contract  
```python
class AppleSiliconDependencyResolver:
    """Apple Silicon (Metal) dependency resolution logic"""
    
    def resolve_metal_dependencies(self, system_info: SystemInfo) -> DependencyResolution:
        """
        Resolve Metal-optimized dependencies for Apple Silicon.
        
        Args:
            system_info: macOS system information
            
        Returns:
            Metal-optimized dependency resolution
            
        Special Considerations:
            - Use default PyTorch (includes Metal support)
            - Select Metal-optimized llama-cpp-python wheel
            - Set Metal performance environment variables
            - Ensure Xcode Command Line Tools available
        """
        pass
```

## Configuration Template Integration

### Template Loading Contract
```python
class ConfigurationTemplateResolver:
    """Hardware configuration template resolution"""
    
    def __init__(self, templates_path: str = "scripts/hardware_configs/"):
        self.templates_path = templates_path
        
    def load_template_for_hardware(
        self, 
        gpu_info: GPUInfo, 
        system_info: SystemInfo
    ) -> Dict[str, Any]:
        """
        Load appropriate configuration template for detected hardware.
        
        Args:
            gpu_info: GPU hardware information
            system_info: System environment details
            
        Returns:
            Hardware-specific configuration template
            
        Template Selection Logic:
            1. NVIDIA GPU: nvidia_gpu.json
            2. AMD GPU: amd_gpu.json  
            3. Apple Silicon: apple_metal.json
            4. High VRAM (>16GB): high_memory.json overlay
            5. Low VRAM (<8GB): low_memory.json overlay
            6. Fallback: cpu_only.json
        """
        template_name = self._select_template_name(gpu_info, system_info)
        template_path = os.path.join(self.templates_path, template_name)
        
        with open(template_path, 'r') as f:
            template = json.load(f)
            
        # Apply memory-specific overlays
        template = self._apply_memory_overlays(template, gpu_info)
        
        return template
    
    def merge_template_with_resolution(
        self, 
        template: Dict[str, Any], 
        resolution: DependencyResolution
    ) -> DependencyResolution:
        """
        Merge hardware template settings with dependency resolution.
        
        Args:
            template: Hardware-specific configuration template
            resolution: Base dependency resolution
            
        Returns:
            Enhanced dependency resolution with template optimizations
        """
        pass
```

## Error Handling & Fallback Strategy

### Resolution Error Contract
```python
class DependencyResolutionError(Exception):
    """Dependency resolution failure"""
    
    def __init__(self, message: str, failed_component: str, fallback_available: bool = True):
        super().__init__(message)
        self.failed_component = failed_component  # "pytorch", "llama_cpp_python", etc.
        self.fallback_available = fallback_available

class IncompatibleSystemError(DependencyResolutionError):
    """System configuration cannot be resolved"""
    pass

class UnsupportedHardwareError(DependencyResolutionError):
    """Hardware configuration not supported"""
    pass
```

### Fallback Strategy Contract
```python
class FallbackDependencyResolver(DependencyResolver):
    """Dependency resolver with comprehensive fallback strategies"""
    
    def __init__(self, enable_fallbacks: bool = True, strict_compatibility: bool = False):
        self.enable_fallbacks = enable_fallbacks
        self.strict_compatibility = strict_compatibility
        
    def resolve_with_fallbacks(
        self, 
        system_info: SystemInfo, 
        gpu_info: GPUInfo, 
        features: Set[FeatureSet]
    ) -> DependencyResolution:
        """
        Resolve dependencies with automatic fallback handling.
        
        Fallback Strategies:
            1. GPU → CPU: If GPU packages unavailable, use CPU variants
            2. Latest → Stable: If latest versions fail, try stable releases
            3. Precompiled → Source: If wheels unavailable, allow source compilation
            4. Full → Minimal: If optional features fail, install core only
            
        Args:
            system_info: System environment
            gpu_info: GPU configuration  
            features: Requested features
            
        Returns:
            Dependency resolution with fallback options applied
        """
        try:
            # Try optimal resolution first
            return self.resolve_all_dependencies(system_info, gpu_info, features)
        except DependencyResolutionError as e:
            if self.enable_fallbacks and e.fallback_available:
                return self._apply_fallback_strategy(e, system_info, gpu_info, features)
            else:
                raise
```

## Testing Contract

### Resolution Testing Requirements
```python
class TestDependencyResolver:
    """Test contract for DependencyResolver implementations"""
    
    @pytest.mark.parametrize("gpu_type,expected_pytorch", [
        ("nvidia", "cu121"),
        ("amd", "rocm5.6"), 
        ("apple_metal", "default"),
        ("none", "cpu")
    ])
    def test_pytorch_resolution(self, gpu_type, expected_pytorch):
        """MUST resolve correct PyTorch variant for GPU type"""
        pass
    
    def test_llama_cpp_url_generation(self):
        """MUST generate valid llama-cpp-python wheel URLs"""
        # Test URL format and availability
        pass
    
    def test_dependency_compatibility(self):
        """MUST validate package version compatibility"""
        pass
    
    def test_fallback_strategies(self):
        """MUST provide working fallback options"""
        pass
```

---

**Status**: ✅ DependencyResolver interface contract defined  
**Next**: Create InstallationValidator contract and quickstart guide
