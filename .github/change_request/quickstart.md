# QuickStart Guide: Smart Installation Script Implementation

**Objective**: Implement intelligent, hardware-aware installation script for RAGLite  
**Target**: Local developers seeking optimal RAGLite setup  
**Priority**: High - addresses critical gap in developer onboarding

## 🎯 Quick Overview

This implementation creates a smart installation script that:

1. **Automatically detects** your hardware (NVIDIA GPU, AMD GPU, Apple Silicon, CPU-only)
2. **Intelligently selects** optimal dependencies (PyTorch CUDA variant, llama-cpp-python wheels)
3. **Configures environment** with hardware-specific optimizations
4. **Validates installation** with comprehensive testing and benchmarking
5. **Provides fallback** strategies when optimal packages unavailable

## 📋 Implementation Checklist

### Phase 1: Foundation & Detection (Week 1)
- [ ] Create `scripts/install.py` main entry point script
- [ ] Implement `SystemDetector` class in `scripts/detection/system_detector.py`
- [ ] Implement `GPUDetector` class in `scripts/detection/gpu_detector.py`
- [ ] Create hardware detection tests
- [ ] Validate detection accuracy across platforms

### Phase 2: Dependency Resolution (Week 2)
- [ ] Implement `DependencyResolver` interface in `scripts/resolution/dependency_resolver.py`
- [ ] Create CUDA-specific resolver: `scripts/resolution/cuda_resolver.py`
- [ ] Create ROCm-specific resolver: `scripts/resolution/rocm_resolver.py`
- [ ] Create Apple Metal resolver: `scripts/resolution/metal_resolver.py`
- [ ] Create CPU fallback resolver: `scripts/resolution/cpu_resolver.py`
- [ ] Test dependency resolution logic

### Phase 3: Installation Management (Week 3)
- [ ] Implement `InstallationManager` class in `scripts/installation/manager.py`
- [ ] Create package installation handlers
- [ ] Implement environment variable configuration
- [ ] Create hardware configuration templates in `scripts/hardware_configs/`
- [ ] Test installation process

### Phase 4: Validation & Testing (Week 4)
- [ ] Implement `InstallationValidator` in `scripts/validation/validator.py`
- [ ] Create component-specific validators
- [ ] Implement performance benchmarking
- [ ] Create comprehensive validation reports
- [ ] Test validation accuracy

### Phase 5: User Interface (Week 5)
- [ ] Create interactive CLI interface
- [ ] Implement progress indicators and logging
- [ ] Create installation summary reports
- [ ] Add troubleshooting guidance
- [ ] Test user experience flows

### Phase 6: Configuration Templates (Week 6)
- [ ] Create hardware-specific config templates
- [ ] Implement configuration merging logic
- [ ] Create environment setup scripts
- [ ] Test configuration application
- [ ] Document configuration options

### Phase 7: Error Handling & Fallbacks (Week 7)
- [ ] Implement comprehensive error handling
- [ ] Create fallback installation strategies
- [ ] Add fix suggestion generation
- [ ] Test error recovery scenarios
- [ ] Validate fallback functionality

### Phase 8: Integration & Documentation (Week 8)
- [ ] Integrate with existing `requirements.txt` system
- [ ] Update main README with installation instructions
- [ ] Create detailed installation documentation
- [ ] Add troubleshooting guides
- [ ] Test complete installation process

## 🛠️ Key Implementation Notes

### Hardware Detection Priority
1. **NVIDIA GPU + CUDA**: Primary target for GPU acceleration
2. **AMD GPU + ROCm**: Secondary GPU acceleration option
3. **Apple Silicon**: Metal Performance Shaders optimization
4. **High-end CPU**: Multi-threading optimizations
5. **Standard CPU**: Basic installation fallback

### Dependency Resolution Strategy
```python
# Resolution priority example
if gpu_info.has_nvidia_gpu and gpu_info.cuda_version:
    pytorch_url = f"torch --index-url https://download.pytorch.org/whl/cu{gpu_info.cuda_version_short}"
    llamacpp_accelerator = f"cu{gpu_info.cuda_version_short}"
elif gpu_info.has_amd_gpu and gpu_info.rocm_version:
    pytorch_url = f"torch --index-url https://download.pytorch.org/whl/rocm{gpu_info.rocm_version}"
    llamacpp_accelerator = "rocm"
elif system_info.is_apple_silicon:
    pytorch_url = "torch"  # Default includes Metal support
    llamacpp_accelerator = "metal"
else:
    pytorch_url = "torch --index-url https://download.pytorch.org/whl/cpu"
    llamacpp_accelerator = "cpu"
```

### Configuration Template System
```json
// Example: scripts/hardware_configs/nvidia_gpu.json
{
    "environment_variables": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "RAGLITE_GPU_LAYERS": "-1",
        "RAGLITE_USE_GPU": "true"
    },
    "performance_settings": {
        "batch_size": 32,
        "max_tokens": 2048,
        "embedding_batch_size": 16
    },
    "memory_optimization": {
        "enable_gradient_checkpointing": true,
        "use_memory_efficient_attention": true
    }
}
```

### Validation Benchmarks
```python
# Example performance thresholds
PERFORMANCE_TARGETS = {
    "nvidia_gpu": {
        "embedding_speed": 200.0,  # embeddings/second
        "search_latency": 100.0,   # milliseconds
        "memory_usage": 4096.0     # MB
    },
    "apple_metal": {
        "embedding_speed": 100.0,
        "search_latency": 200.0,
        "memory_usage": 2048.0
    },
    "cpu_only": {
        "embedding_speed": 20.0,
        "search_latency": 500.0,
        "memory_usage": 1024.0
    }
}
```

## 🔗 Integration Points

### Existing RAGLite Infrastructure
- **GPU Detection**: Leverage `_gpu_utils.py` CUDA detection logic
- **Embedding**: Integrate with `_embed.py` and `_embedding_gpu.py`
- **Database**: Use existing `_database.py` abstraction
- **Requirements**: Build on `requirements.txt` and `requirements-dev.txt`

### Development Environment Integration
- **DevContainer**: Update `.devcontainer/devcontainer.json` to use installation script
- **GitHub Actions**: Integrate with CI/CD workflows
- **Documentation**: Update README and installation guides

## 📊 Success Metrics

### Primary Success Criteria
- [ ] **90%+ installation success rate** across supported platforms
- [ ] **<2 minute average installation time** on modern hardware
- [ ] **Automatic optimal configuration** for 95%+ of hardware configurations
- [ ] **Clear error messages and fix suggestions** for all failure modes

### Performance Validation Targets
- [ ] **NVIDIA GPU**: >200 embeddings/second, <100ms search latency
- [ ] **Apple Silicon**: >100 embeddings/second, <200ms search latency  
- [ ] **CPU-only**: >20 embeddings/second, <500ms search latency
- [ ] **Memory Usage**: <4GB peak memory usage during installation

## 🚀 Getting Started (Implementation)

### 1. Clone and Setup Development Environment
```bash
cd /workspaces/raglite
python scripts/install.py --dev-mode --validate-setup
```

### 2. Implement Core Detection Logic
```bash
# Start with system detection
touch scripts/detection/system_detector.py
touch scripts/detection/gpu_detector.py

# Implement following the contracts in contracts/ directory
```

### 3. Test Detection Across Platforms
```bash
# Test on different hardware
python -m pytest tests/test_installation_script.py::test_system_detection
python -m pytest tests/test_installation_script.py::test_gpu_detection
```

### 4. Implement Dependency Resolution
```bash
# Follow dependency-resolver.md contract
touch scripts/resolution/dependency_resolver.py
```

### 5. Build Installation Management
```bash
# Follow system-detector.md contract
touch scripts/installation/manager.py
```

## 📖 Related Documentation

- **[Technical Specification](./SPECIFICATION.md)**: Detailed requirements and architecture
- **[Implementation Plan](./plan.md)**: Phase-by-phase development approach
- **[Research Analysis](./research.md)**: Hardware detection and dependency resolution research
- **[Data Models](./data-model.md)**: Core data structures and schemas
- **[API Contracts](./contracts/)**: Interface definitions and implementation contracts

## ⚡ Quick Commands Reference

### Installation Script Usage (Once Implemented)
```bash
# Basic installation
python scripts/install.py

# Installation with specific features
python scripts/install.py --features gpu,chainlit,dev-tools

# Validate existing installation
python scripts/install.py --validate-only

# Force reinstallation with debugging
python scripts/install.py --force-reinstall --verbose --debug
```

### Development Commands
```bash
# Run installation tests
python -m pytest tests/test_installation_script.py -v

# Benchmark installation performance
python scripts/install.py --benchmark-only

# Generate installation report
python scripts/install.py --report-only --output install_report.html
```

---

**Status**: ✅ QuickStart guide created  
**Next Steps**: 
1. Review all specification documents for completeness
2. Use spec-kit to generate detailed implementation tasks
3. Hand off to GitHub Copilot coding agent for implementation
