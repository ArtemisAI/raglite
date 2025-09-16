# API Contract: SystemDetector

**Interface**: `SystemDetector`  
**Purpose**: Cross-platform system environment detection  
**Integration**: Extends existing `_gpu_utils.py` detection capabilities

## Interface Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .data_model import SystemInfo, GPUInfo

class SystemDetector(ABC):
    """Abstract interface for cross-platform system detection"""
    
    @abstractmethod
    def detect_all(self) -> Dict[str, Any]:
        """
        Perform comprehensive system detection.
        
        Returns:
            Dict containing SystemInfo and GPUInfo data
            
        Raises:
            SystemDetectionError: Critical detection failure
        """
        pass
    
    @abstractmethod  
    def detect_os_info(self) -> Dict[str, Any]:
        """
        Detect operating system information.
        
        Returns:
            OS information conforming to OSInfo schema
            
        Raises:
            OSDetectionError: Cannot determine OS details
        """
        pass
    
    @abstractmethod
    def detect_python_info(self) -> Dict[str, Any]:
        """
        Analyze Python environment details.
        
        Returns:
            Python information conforming to PythonInfo schema
            
        Raises:
            PythonDetectionError: Invalid Python environment
        """
        pass
    
    @abstractmethod
    def detect_hardware_info(self) -> Dict[str, Any]:
        """
        Detect system hardware specifications.
        
        Returns:
            Hardware information conforming to HardwareInfo schema
        """
        pass
    
    @abstractmethod
    def detect_gpu_info(self) -> Dict[str, Any]:
        """
        Comprehensive GPU detection with multi-vendor support.
        
        Returns:
            GPU information conforming to GPUInfo schema
            
        Integration:
            MUST leverage existing detect_cuda_availability() from _gpu_utils.py
            SHOULD enhance with system-level GPU detection
        """
        pass
        
    @abstractmethod
    def validate_installation_requirements(self, system_info: SystemInfo) -> Dict[str, bool]:
        """
        Validate system meets RAGLite installation requirements.
        
        Args:
            system_info: Detected system information
            
        Returns:
            Dict mapping requirement name to satisfaction status
            Example: {"python_version": True, "disk_space": False}
            
        Requirements:
            - Python >= 3.10
            - Available disk space >= 2GB  
            - Sufficient RAM >= 4GB
        """
        pass
```

## Implementation Requirements

### Error Handling Contract
```python
class SystemDetectionError(Exception):
    """Base exception for system detection failures"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, recoverable: bool = True):
        super().__init__(message)
        self.details = details or {}
        self.recoverable = recoverable

class OSDetectionError(SystemDetectionError):
    """Operating system detection failure"""
    pass

class PythonDetectionError(SystemDetectionError):
    """Python environment detection failure"""
    pass

class GPUDetectionError(SystemDetectionError):
    """GPU hardware detection failure"""
    pass
```

### Performance Contract
- **Total Detection Time**: MUST complete within 30 seconds on standard hardware
- **GPU Detection Time**: SHOULD complete within 10 seconds
- **Retry Logic**: MUST implement exponential backoff for transient failures
- **Timeout Handling**: MUST provide configurable timeouts for external command execution

### Integration Contract  
```python
class EnhancedSystemDetector(SystemDetector):
    """Enhanced detector that integrates with existing RAGLite infrastructure"""
    
    def __init__(self, use_existing_gpu_detection: bool = True):
        """
        Initialize detector with RAGLite integration options.
        
        Args:
            use_existing_gpu_detection: Whether to use existing _gpu_utils functions
        """
        self.use_existing_gpu_detection = use_existing_gpu_detection
        
    def detect_gpu_info(self) -> Dict[str, Any]:
        """
        GPU detection that integrates with existing RAGLite detection.
        
        Implementation Requirements:
        1. Perform system-level GPU detection first (nvidia-smi, rocm-smi, etc.)
        2. If use_existing_gpu_detection=True and PyTorch available:
           - Call detect_cuda_availability() from _gpu_utils.py
           - Call get_gpu_memory_info() from _gpu_utils.py  
           - Call calculate_optimal_gpu_layers() from _gpu_utils.py
        3. Combine results into comprehensive GPUInfo structure
        4. Validate consistency between system and PyTorch detection
        """
        # Implementation details in concrete class
        pass
```

## Platform-Specific Implementation Contracts

### Linux Implementation Contract
```python
class LinuxSystemDetector(EnhancedSystemDetector):
    """Linux-specific system detection implementation"""
    
    def detect_os_info(self) -> Dict[str, Any]:
        """
        Linux OS detection requirements:
        - Parse /etc/os-release for distribution information
        - Use uname for kernel and architecture details
        - Detect package manager (apt, yum, dnf, pacman, zypper)
        - Check for WSL environment
        """
        pass
        
    def detect_gpu_info(self) -> Dict[str, Any]:
        """
        Linux GPU detection requirements:
        - NVIDIA: nvidia-smi command or /proc/driver/nvidia/version
        - AMD: rocm-smi command or /sys/class/drm inspection
        - Intel: lspci parsing for Intel Graphics
        - Fallback: /sys/class/drm device enumeration
        """
        pass
```

### macOS Implementation Contract  
```python
class MacOSSystemDetector(EnhancedSystemDetector):
    """macOS-specific system detection implementation"""
    
    def detect_os_info(self) -> Dict[str, Any]:
        """
        macOS OS detection requirements:
        - Use sw_vers for macOS version
        - Detect Apple Silicon vs Intel via sysctl
        - Check Xcode Command Line Tools availability
        - Determine Homebrew installation
        """
        pass
        
    def detect_gpu_info(self) -> Dict[str, Any]:
        """
        macOS GPU detection requirements:  
        - Apple Silicon: Metal support via system_profiler
        - Intel Mac: Discrete GPU detection via system_profiler
        - eGPU detection for external GPU enclosures
        """
        pass
```

### Windows Implementation Contract
```python
class WindowsSystemDetector(EnhancedSystemDetector):
    """Windows-specific system detection implementation"""
    
    def detect_os_info(self) -> Dict[str, Any]:
        """
        Windows OS detection requirements:
        - Use PowerShell Get-ComputerInfo for system details
        - Detect Windows version and build number
        - Check for WSL2 installation
        - Determine package manager availability (choco, winget)
        """
        pass
        
    def detect_gpu_info(self) -> Dict[str, Any]:
        """
        Windows GPU detection requirements:
        - WMI queries: Get-WmiObject Win32_VideoController
        - NVIDIA: nvidia-smi.exe if available
        - AMD: PowerShell GPU enumeration
        - Intel: DirectX diagnostic information
        """
        pass
```

## Testing Contract

### Unit Testing Requirements
```python
import pytest
from unittest.mock import Mock, patch

class TestSystemDetector:
    """Test contract for SystemDetector implementations"""
    
    @pytest.fixture
    def detector(self) -> SystemDetector:
        """Provide detector instance for testing"""
        # Implementation-specific detector
        pass
    
    def test_detect_all_returns_valid_schema(self, detector):
        """MUST return data conforming to SystemInfo + GPUInfo schemas"""
        result = detector.detect_all()
        assert validate_system_info(result["system_info"])
        assert validate_gpu_info(result["gpu_info"])
    
    def test_detect_all_performance(self, detector):
        """MUST complete within 30 seconds"""
        import time
        start = time.time()
        result = detector.detect_all()
        duration = time.time() - start
        assert duration < 30.0
    
    def test_gpu_detection_integration(self, detector):
        """MUST integrate with existing _gpu_utils functions when available"""
        with patch('raglite._gpu_utils.detect_cuda_availability') as mock_cuda:
            mock_cuda.return_value = True
            result = detector.detect_gpu_info()
            if result.get("cuda", {}).get("available"):
                mock_cuda.assert_called_once()
    
    def test_error_handling(self, detector):
        """MUST provide actionable error information on failure"""
        # Test with invalid environment
        try:
            detector.detect_all()
        except SystemDetectionError as e:
            assert e.recoverable is not None
            assert isinstance(e.details, dict)
            assert len(str(e)) > 0
```

### Integration Testing Requirements
```python
class TestRAGLiteIntegration:
    """Test contract for RAGLite ecosystem integration"""
    
    def test_existing_gpu_detection_compatibility(self):
        """MUST be compatible with existing _gpu_utils detection"""
        from raglite._gpu_utils import detect_cuda_availability, get_gpu_memory_info
        
        detector = EnhancedSystemDetector(use_existing_gpu_detection=True)
        gpu_info = detector.detect_gpu_info()
        
        # If CUDA detected by new system, existing detection should agree
        if gpu_info.get("cuda", {}).get("available"):
            existing_cuda = detect_cuda_availability()
            assert gpu_info["pytorch_gpu_available"] == existing_cuda
    
    def test_configuration_generation(self):
        """MUST generate valid RAGLite configuration"""
        detector = EnhancedSystemDetector()
        system_info = detector.detect_all()
        
        # Generated config should be compatible with RAGLiteConfig
        from raglite._config import RAGLiteConfig
        config = RAGLiteConfig()  # Should not raise with detected settings
```

## Backward Compatibility Contract

### Existing Code Integration
```python
def ensure_backward_compatibility():
    """
    Integration contract with existing RAGLite codebase:
    
    1. MUST NOT modify existing _gpu_utils.py functions
    2. SHOULD enhance detection by calling existing functions when available
    3. MUST provide same interface as existing detection for validation
    4. SHOULD gracefully handle missing PyTorch dependency
    5. MUST maintain same GPU memory calculation logic
    """
    pass
```

### Migration Strategy
- **Phase 1**: New detection runs alongside existing detection for validation
- **Phase 2**: Installation scripts use new detection, existing code unchanged  
- **Phase 3**: Optional migration of existing detection to use enhanced system detector

---

**Status**: ✅ SystemDetector interface contract defined
**Next**: Define DependencyResolver and InstallationValidator contracts
