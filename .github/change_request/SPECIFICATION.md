# Feature Specification: Smart Local Installation Script System

**Feature Branch**: `001-smart-local-installation`  
**Created**: September 16, 2025  
**Status**: Draft  
**Input**: Create comprehensive installation script system with GPU detection and optimal dependency resolution

## Context & Background

### Current State Assessment
RAGLite has achieved significant progress in GPU acceleration implementation:

**✅ Completed GPU Infrastructure:**
- GPU detection and CUDA support in `src/raglite/_embedding_gpu.py` and `src/raglite/_gpu_utils.py`
- Dynamic VRAM allocation and CPU fallback mechanisms
- GPU-aware configuration in `src/raglite/_config.py`
- Test framework for GPU acceleration validation
- Requirements files with GPU dependencies (torch, llama-cpp-python)

**❌ Missing Installation Automation:**
- No local installation script for developers cloning the repository
- Manual GPU configuration requiring expert knowledge of CUDA versions
- Complex dependency selection process for different hardware configurations
- No automated environment setup or validation

### Business Problem
**Primary Issue**: New developers and users face significant friction when setting up RAGLite locally, leading to:
- High abandonment rates during initial setup
- Support burden from installation issues  
- Inconsistent environments across development teams
- Suboptimal performance due to incorrect GPU configurations

**Target Solution**: One-command installation that automatically detects hardware and installs optimal configuration.

## User Scenarios & Testing

### Primary Users
1. **New Contributors**: Developers wanting to contribute to RAGLite
2. **Researchers**: Scientists using RAGLite for experiments  
3. **Enterprise Users**: Teams deploying RAGLite in production environments
4. **Students/Educators**: Academic users learning RAG techniques

### Core User Journey
```
1. User clones repository: `git clone https://github.com/ArtemisAI/raglite.git`
2. User runs installation: `cd raglite && ./install.sh`
3. System automatically:
   - Detects OS, architecture, Python version
   - Identifies GPU type (NVIDIA/AMD/Intel/CPU-only)  
   - Determines CUDA/ROCm versions if available
   - Selects optimal PyTorch and llama-cpp-python variants
   - Creates virtual environment
   - Installs all dependencies with hardware optimization
   - Configures environment variables
   - Validates installation with test suite
4. User receives summary with next steps
5. User can immediately start using RAGLite with optimal performance
```

### Success Criteria Testing
**Installation Success Rate**: >95% success rate across supported platforms
**Time to First Success**: <10 minutes from clone to working RAGLite
**Performance Optimization**: GPU users get >2x speedup vs CPU-only installation
**Error Recovery**: Failed installations provide actionable recovery steps

## Functional Requirements

### FR1: System Detection & Analysis
**Requirement**: Installation script SHALL automatically detect system configuration
**Acceptance Criteria**:
- Detects OS (Linux distributions, macOS, Windows) and architecture
- Identifies Python version and validates ≥3.10 requirement
- Discovers GPU hardware (NVIDIA, AMD, Intel, Apple Metal)
- Determines CUDA toolkit version (if NVIDIA GPU present)
- Measures available VRAM and system RAM
- Reports all findings to user before installation

### FR2: Intelligent Dependency Resolution  
**Requirement**: System SHALL select optimal dependencies based on detected hardware
**Acceptance Criteria**:
- Installs CUDA-optimized PyTorch for NVIDIA GPUs with correct CUDA version
- Installs ROCm-optimized PyTorch for AMD GPUs
- Installs Metal-optimized PyTorch for Apple Silicon
- Selects appropriate llama-cpp-python precompiled binary matching:
  - OS and architecture
  - GPU acceleration type (CUDA/ROCm/Metal/CPU)
  - Python version (3.10/3.11/3.12)
- Falls back gracefully when optimal packages unavailable

### FR3: Interactive Feature Selection
**Requirement**: Users SHALL choose optional features during installation
**Acceptance Criteria**:
- Presents clear feature selection menu with descriptions
- Required features: Core RAGLite, SQLite backend, basic embeddings
- Optional features: GPU acceleration, Chainlit UI, Pandoc conversion, Ragas evaluation, benchmarking tools, development tools
- Validates selections against system capabilities (e.g., GPU features only if GPU detected)
- Supports non-interactive mode with sensible defaults

### FR4: Environment Configuration
**Requirement**: Installation SHALL create complete development environment
**Acceptance Criteria**:
- Creates Python virtual environment with appropriate name
- Generates `.env` file with system-specific optimizations
- Sets CUDA/GPU environment variables based on detected hardware
- Configures model cache directories and database paths
- Installs pre-commit hooks if development tools selected
- Makes environment easily activatable by user

### FR5: Installation Validation
**Requirement**: System SHALL verify installation success before completion
**Acceptance Criteria**:
- Tests core RAGLite imports and basic functionality
- Validates GPU acceleration if enabled (loads test model, runs inference)
- Verifies database connectivity and SQLite-vec functionality
- Checks optional features if installed
- Runs performance benchmark to confirm optimization
- Reports validation results with pass/fail status

### FR6: Error Handling & Recovery
**Requirement**: Installation failures SHALL provide actionable recovery guidance
**Acceptance Criteria**:
- Categorizes errors: system compatibility, network issues, permission problems
- Provides specific fix instructions for common issues
- Offers automatic fallback options (GPU→CPU, latest→stable versions)
- Logs detailed error information for troubleshooting
- Generates system-specific troubleshooting guide
- Allows partial installation recovery without full restart

### FR7: Multi-Platform Support
**Requirement**: Installation SHALL work across all supported platforms
**Acceptance Criteria**:
- Unix/Linux: Bash script with error handling and progress indicators
- macOS: Same bash script with Apple Silicon optimizations  
- Windows: PowerShell script with equivalent functionality
- Handles platform-specific package managers (apt, brew, choco, winget)
- Manages platform-specific paths and permissions correctly

### FR8: Integration with Existing Infrastructure
**Requirement**: Installation SHALL leverage existing RAGLite GPU infrastructure
**Acceptance Criteria**:
- Uses existing GPU detection logic from `_gpu_utils.py`
- Integrates with current GPU-aware classes in `_embedding_gpu.py`
- Respects existing configuration patterns in `_config.py`
- Works with current requirements files and dependency structure
- Maintains compatibility with existing CI/CD workflows

## Key Entities & Data Model

### SystemInfo Entity
```
SystemInfo:
  - os: {name, version, distribution}
  - architecture: {x86_64, arm64, aarch64}
  - python: {version, executable_path, pip_available, uv_available}
  - memory: {total_ram_gb, available_ram_gb}
  - storage: {free_space_gb, install_path}
```

### GPUInfo Entity  
```
GPUInfo:
  - type: {nvidia, amd, intel, apple_metal, none}
  - devices: List[{name, vram_gb, compute_capability}]
  - cuda: {version, runtime_version, available}
  - rocm: {version, available}
  - drivers: {version, compatible}
```

### InstallationConfig Entity
```
InstallationConfig:
  - features: Set[{core, gpu, chainlit, pandoc, ragas, bench, dev}]
  - pytorch_variant: {cpu, cuda121, cuda122, rocm, metal}
  - llama_cpp_variant: {url, accelerator_type}
  - environment_variables: Dict[str, str]
  - virtual_env_path: str
  - validation_tests: List[str]
```

## Non-Functional Requirements

### NFR1: Performance
- Installation completes in <10 minutes on standard hardware
- GPU detection completes in <30 seconds
- Dependency resolution completes in <2 minutes
- Validation tests complete in <3 minutes

### NFR2: Reliability  
- 95% success rate across supported configurations
- Graceful degradation when optimal packages unavailable
- Comprehensive error logging and recovery guidance
- Idempotent installation (safe to re-run)

### NFR3: Usability
- Clear progress indicators with estimated time remaining
- Colored output with success/warning/error distinction  
- Interactive prompts with helpful descriptions
- Post-installation summary with next steps
- Auto-generated troubleshooting documentation

### NFR4: Maintainability
- Modular Python scripts for cross-platform system detection
- JSON configuration templates for different hardware profiles
- Automated testing of installation scripts in CI/CD
- Version compatibility matrix for supported configurations

## Business Rules & Constraints

### BR1: Compatibility Matrix
- Python 3.10+ required (aligns with existing codebase)
- CUDA 12.1+ recommended for NVIDIA GPUs (matches current GPU implementation)
- ROCm 5.6+ for AMD GPUs
- macOS 11+ for Apple Silicon support
- Windows 10+ for PowerShell script compatibility

### BR2: Dependency Prioritization
- Always prefer precompiled binaries over source compilation
- Prioritize stable releases over bleeding-edge versions  
- GPU-optimized packages take precedence when available
- Fallback to CPU-only variants maintains full functionality

### BR3: Security Requirements
- No automatic root/admin privilege escalation
- Package downloads from official repositories only
- Checksum validation for downloaded binaries
- No storage of API keys or credentials in installation artifacts

## Review Checklist

### Completeness Check
- [x] **Clear user scenarios**: Installation journey clearly defined with success criteria
- [x] **Testable requirements**: Each functional requirement has specific acceptance criteria  
- [x] **Business value**: Addresses key user friction and enables better RAGLite adoption
- [x] **Integration context**: Builds upon existing GPU infrastructure rather than replacing it

### Quality Check  
- [x] **Unambiguous language**: Requirements avoid technical implementation details
- [x] **Scope appropriateness**: Focuses on installation automation without expanding RAGLite core functionality
- [x] **Feasibility**: Leverages existing system detection patterns and dependency management
- [x] **Measurable outcomes**: Success criteria include quantifiable metrics (95% success rate, <10 minutes)

### Stakeholder Alignment
- [x] **Developer experience**: Dramatically reduces setup friction for new contributors
- [x] **User adoption**: Eliminates technical barriers for researchers and enterprise users
- [x] **Maintenance burden**: Reduces support requests related to installation issues
- [x] **Project roadmap**: Aligns with existing GPU acceleration work and enhances its accessibility

### Dependencies & Assumptions
- **Assumes**: Existing GPU detection infrastructure in `_gpu_utils.py` is reliable
- **Depends on**: Current requirements files structure and optional dependency groups
- **Assumes**: Users have basic command line familiarity for running installation script
- **Depends on**: Stable availability of PyTorch and llama-cpp-python precompiled binaries

---

**Status**: ✅ Ready for implementation planning
**Next Steps**: Generate detailed implementation tasks and technical architecture
