# Integration Plan: Smart Installation Script + GPU Dev Container Setup

## 🎯 **Objective**

Integrate the Smart Local Installation Script System with the enhanced Dev Container and GitHub Codespaces configuration to provide a seamless development experience across all environments.

## 📋 **Integration Requirements**

### **Alignment with Current Project Status**

Based on the TODO.md, the project currently has:

✅ **Completed GPU Infrastructure**:
- GPU acceleration in `src/raglite/_embedding_gpu.py` and `src/raglite/_gpu_utils.py`
- CUDA detection and CPU fallback mechanisms
- Configuration management in `src/raglite/_config.py`
- GPU-aware tests and benchmarking

🚧 **Known Issues to Address**:
- Test failure in `test_database.py` with `huggingface_hub.errors.HFValidationError`
- Model string parsing bug in `GPUAwareLlamaLLM.create` method

🎯 **Installation Script Integration**:
- Leverage existing GPU detection infrastructure
- Build upon current configuration system
- Extend testing framework for validation
- Align with development workflow

## 🏗️ **Integration Architecture**

### **1. Unified System Detection**

The installation script will extend the existing GPU detection:

```python
# Enhanced integration with existing _gpu_utils.py
from raglite._gpu_utils import detect_gpu_info
from raglite._config import RAGLiteConfig

class EnhancedSystemDetector:
    def __init__(self):
        self.gpu_utils = detect_gpu_info()  # Use existing detection
        
    def get_installation_profile(self) -> dict:
        """Generate installation profile using existing GPU infrastructure"""
        return {
            'gpu_info': self.gpu_utils,
            'recommended_packages': self._resolve_packages(),
            'environment_config': self._generate_env_config(),
            'validation_tests': self._get_validation_suite()
        }
```

### **2. Container Configuration Integration**

The dev container setup will work seamlessly with the installation script:

```json
// .devcontainer/devcontainer.json - Enhanced integration
{
  "name": "RAGLite Development Environment",
  
  // Use installation script for setup
  "postCreateCommand": "./install.sh --dev-container --gpu-detect",
  "postStartCommand": "./scripts/validate_installation.py --container-mode",
  
  // Environment variables for installation script
  "containerEnv": {
    "RAGLITE_INSTALLATION_MODE": "dev-container",
    "RAGLITE_GPU_DETECTION": "auto",
    "RAGLITE_VALIDATION_LEVEL": "full"
  }
}
```

### **3. Testing Framework Integration**

Extend the existing test suite to include installation validation:

```python
# tests/test_installation_integration.py
class TestInstallationIntegration:
    def test_gpu_detection_consistency(self):
        """Ensure installation script uses same GPU detection as runtime"""
        from scripts.detect_system import SystemDetector
        from raglite._gpu_utils import detect_gpu_info
        
        install_gpu = SystemDetector().detect_gpu()
        runtime_gpu = detect_gpu_info()
        
        assert install_gpu['cuda_available'] == runtime_gpu.get('cuda_available')
        
    def test_dependency_resolution_with_existing_config(self):
        """Test that installation respects existing configuration"""
        # Test integration with RAGLiteConfig
        pass
```

## 📁 **Updated File Structure**

```
raglite/
├── install.sh                          # Main installation script (integrates with existing)
├── install.ps1                         # Windows version
├── .env.template                        # Environment template
├── .devcontainer/
│   ├── devcontainer.json               # GPU-enabled dev container
│   ├── docker-compose.dev.yml          # GPU runtime support
│   ├── Dockerfile                      # CUDA-enabled image
│   ├── post-create.sh                  # Calls install.sh --dev-container
│   └── codespaces-setup.sh             # CPU fallback setup
├── scripts/
│   ├── detect_system.py                # Extends _gpu_utils detection
│   ├── resolve_dependencies.py         # Uses RAGLiteConfig patterns
│   ├── validate_installation.py        # Extends existing test framework
│   └── hardware_configs/               # Configuration profiles
├── src/raglite/
│   ├── _gpu_utils.py                   # Existing GPU detection (enhanced)
│   ├── _embedding_gpu.py               # Existing GPU acceleration
│   └── _config.py                      # Existing configuration (extended)
├── tests/
│   ├── test_gpu_acceleration.py        # Existing GPU tests
│   ├── test_installation_integration.py # New integration tests
│   └── test_container_setup.py         # Container validation tests
└── docs/
    ├── DEV_CONTAINER_GPU_SETUP.md      # Container setup guide
    └── INSTALLATION_GUIDE.md           # Comprehensive installation docs
```

## 🔧 **Implementation Phases**

### **Phase 1: Core Integration (Priority 1)**

1. **Extend Existing GPU Detection**
   - Enhance `_gpu_utils.py` with installation-specific detection
   - Add container environment detection
   - Maintain backward compatibility

2. **Create Installation Script Foundation**
   - Build `install.sh` that uses existing `_gpu_utils.py`
   - Implement hardware-specific dependency resolution
   - Add basic validation using existing test patterns

3. **Update Dev Container Configuration**
   - Modify existing `.devcontainer/devcontainer.json`
   - Add GPU runtime support
   - Integrate installation script into container lifecycle

### **Phase 2: Advanced Features (Priority 2)**

1. **Interactive Installation Experience**
   - Feature selection using existing configuration patterns
   - Progress indicators with logging
   - Integration with existing benchmarking tools

2. **Enhanced Error Handling**
   - Extend existing fallback mechanisms
   - Container-specific error recovery
   - Integration with existing troubleshooting patterns

3. **Validation Framework**
   - Extend existing test suite
   - Container-specific validation
   - Performance benchmarking integration

### **Phase 3: Polish and Documentation (Priority 3)**

1. **Windows PowerShell Script**
   - Windows-specific GPU detection
   - Container and WSL2 integration
   - Compatibility with existing tools

2. **Comprehensive Documentation**
   - Update existing documentation
   - Container-specific guides
   - Integration examples

3. **CI/CD Integration**
   - Extend existing GitHub Actions
   - Container testing workflows
   - Installation script validation

## 🧪 **Testing Strategy**

### **Integration Test Matrix**

| Environment | GPU Support | Test Scenarios |
|------------|-------------|----------------|
| Local Dev Container | NVIDIA GPU | Full GPU acceleration |
| Local Dev Container | CPU Only | Graceful CPU fallback |
| GitHub Codespaces | CPU Only | Cloud development |
| CI/CD Environment | Mock GPU | Automated testing |
| Windows Container | NVIDIA GPU | Windows development |

### **Validation Checkpoints**

1. **System Detection Consistency**
   - Installation script and runtime detect same hardware
   - Container environment properly detected
   - Configuration consistency maintained

2. **Dependency Resolution Accuracy**
   - Correct PyTorch variant selected
   - llama-cpp-python compatibility verified
   - Optional dependencies properly resolved

3. **Environment Configuration**
   - Virtual environment creation
   - Environment variables properly set
   - Development tools configured

4. **Runtime Functionality**
   - GPU acceleration works when available
   - CPU fallback functions correctly
   - All existing tests pass

## 🎯 **Success Criteria**

### **Developer Experience**
- [ ] Single command setup: `./install.sh`
- [ ] Dev container starts with optimal configuration
- [ ] GitHub Codespaces works with CPU fallback
- [ ] All existing functionality preserved

### **Technical Requirements**
- [ ] GPU detection consistency across installation and runtime
- [ ] Seamless integration with existing configuration system
- [ ] No breaking changes to existing APIs
- [ ] Comprehensive test coverage

### **Documentation Standards**
- [ ] Clear setup instructions for all environments
- [ ] Troubleshooting guides for common issues
- [ ] Integration examples and best practices
- [ ] Contribution guidelines updated

This integration plan ensures that the smart installation script system builds upon the existing GPU acceleration infrastructure while providing a seamless development experience across all environments.
