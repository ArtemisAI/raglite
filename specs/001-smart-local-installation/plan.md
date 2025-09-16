# Implementation Plan: Smart Local Installation Script System

**Branch**: `001-smart-local-installation` | **Date**: September 16, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-smart-local-installation/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✅
   → Feature spec loaded successfully from spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION) ✅
   → Project Type: CLI/Library hybrid with cross-platform installation scripts
   → Structure Decision: Single project structure with platform-specific scripts
3. Fill Constitution Check section ✅
4. Evaluate Constitution Check section ✅
   → No violations identified for installation script system
   → Update Progress Tracking: Initial Constitution Check ✅
5. Execute Phase 0 → research.md ✅
   → All technical decisions clarified from comprehensive change request
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, copilot-instructions.md ✅
7. Re-evaluate Constitution Check section ✅
   → No new violations after design phase
   → Update Progress Tracking: Post-Design Constitution Check ✅
8. Plan Phase 2 → Describe task generation approach ✅
9. STOP - Ready for /tasks command ✅
```

## Summary
Create a comprehensive installation script system that automatically detects system hardware (CPU, NVIDIA GPU, AMD GPU, Apple Silicon), selects optimal dependencies (PyTorch variants, llama-cpp-python binaries), configures virtual environments with hardware-specific optimizations, and validates complete RAGLite functionality. Technical approach uses Python for cross-platform system detection with platform-specific shell scripts (Bash/PowerShell) for execution and comprehensive validation testing.

## Technical Context
**Language/Version**: Python 3.10+ (system detection), Bash 4.0+ (Unix install), PowerShell 5.1+ (Windows install)  
**Primary Dependencies**: PyTorch (hardware-specific), llama-cpp-python (precompiled binaries), sqlite-vec, pynndescent, platform detection libraries  
**Storage**: Local file system for virtual environments, model cache, database files, configuration  
**Testing**: pytest for validation scripts, bash test framework for shell scripts, hardware-specific testing  
**Target Platform**: Linux (Ubuntu/CentOS/Debian/Fedora), macOS (Intel/Apple Silicon), Windows 10/11  
**Project Type**: CLI/Library hybrid - installation tooling with system detection libraries  
**Performance Goals**: <2 minutes total installation time, <30 seconds system detection, hardware-optimal configuration  
**Constraints**: No network dependencies for core detection, graceful fallbacks for all hardware types, resume capability for interrupted installs  
**Scale/Scope**: Support 6 major platforms, 15+ GPU configurations, 3 Python versions (3.10-3.12), 25+ dependency combinations

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Library-First Principle**: ✅ PASS
- System detection implemented as standalone Python library (`scripts/detect_system.py`)
- Dependency resolution as independent library (`scripts/resolve_dependencies.py`)
- Virtual environment management as reusable library (`scripts/venv_manager.py`)
- Each library self-contained with clear single responsibility

**CLI Interface Principle**: ✅ PASS
- Primary interface via shell scripts (`install.sh`, `install.ps1`)
- Python libraries expose CLI interfaces for individual functions
- JSON output format for programmatic integration
- Human-readable progress output for interactive use

**Test-First Principle**: ✅ PASS
- Contract tests for system detection across hardware configurations
- Integration tests for end-to-end installation scenarios
- Validation tests for hardware-specific optimizations
- Red-Green-Refactor cycle for each detection capability

**Cross-Platform Requirements**: ✅ PASS
- Unified Python libraries work across all platforms
- Platform-specific execution layers (Bash/PowerShell)
- Consistent JSON interfaces between platforms
- Graceful degradation for platform-specific features

## Project Structure

### Documentation (this feature)
```
specs/001-smart-local-installation/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
raglite/
├── install.sh                          # Main Unix installation entry point
├── install.ps1                         # Windows PowerShell installation entry point
├── .env.template                        # Environment variable template
├── TROUBLESHOOTING.md                   # Auto-generated troubleshooting guide
├── scripts/                            # Installation system libraries
│   ├── detect_system.py                # Cross-platform system detection library
│   ├── resolve_dependencies.py         # Hardware-specific dependency resolution
│   ├── venv_manager.py                 # Virtual environment management library
│   ├── validate_installation.py        # Post-installation validation library
│   ├── benchmark_installation.py       # Performance benchmarking library
│   └── hardware_configs/              # Hardware configuration templates
│       ├── nvidia_gpu.json            # NVIDIA GPU optimal settings
│       ├── amd_gpu.json               # AMD GPU optimal settings
│       ├── apple_metal.json           # Apple Silicon optimal settings
│       └── cpu_only.json              # CPU-only fallback settings
├── docs/
│   └── installation-guide.md          # Comprehensive installation documentation
├── tests/
│   ├── contract/                       # Contract tests for detection libraries
│   ├── integration/                    # End-to-end installation testing
│   └── unit/                          # Unit tests for individual components
└── .github/
    ├── workflows/
    │   └── test-installation.yml       # CI for installation script testing
    └── copilot-instructions.md         # Updated agent context
```

**Structure Decision**: Single project structure with platform-specific execution scripts and unified Python libraries

## Phase 0: Outline & Research

### Research Completed from Change Request Analysis:

1. **Hardware Detection Technologies**:
   - **NVIDIA**: nvidia-ml-py, nvidia-smi command, CUDA toolkit detection via nvcc
   - **AMD**: ROCm detection via rocm-smi, HIP version detection
   - **Apple Silicon**: Metal support via system profiler, architecture detection
   - **Intel**: Intel GPU detection via device queries, OpenCL support

2. **Dependency Resolution Strategies**:
   - **PyTorch**: Hardware-specific index URLs for CUDA/ROCm/Metal/CPU variants
   - **llama-cpp-python**: Precompiled wheel selection based on OS/arch/accelerator/Python version
   - **Package Management**: pip/uv compatibility, virtual environment isolation

3. **Cross-Platform Shell Scripting**:
   - **Unix**: Bash with POSIX compliance, error handling via set -euo pipefail
   - **Windows**: PowerShell with UAC handling, WMI queries for hardware detection
   - **Progress Indication**: ANSI color codes, progress bars, logging mechanisms

**Output**: All technical decisions resolved from comprehensive change request analysis

## Phase 1: Design & Contracts

### System Detection Library (`scripts/detect_system.py`)
**Purpose**: Cross-platform hardware and system detection
**Key Classes**:
- `SystemDetector`: Main detection orchestrator
- `GPUDetector`: GPU-specific detection logic
- `PlatformInfo`: System information container

### Dependency Resolution Library (`scripts/resolve_dependencies.py`)
**Purpose**: Hardware-optimal package selection
**Key Classes**:
- `DependencyResolver`: Main resolution logic
- `HardwareProfile`: Hardware configuration container
- `PackageMapping`: Dependency mapping logic

### Virtual Environment Manager (`scripts/venv_manager.py`)
**Purpose**: Environment creation and package installation
**Key Classes**:
- `VirtualEnvironmentManager`: Environment lifecycle management
- `PackageInstaller`: Installation with retry logic
- `InstallationValidator`: Environment verification

### Validation System (`scripts/validate_installation.py`)
**Purpose**: Post-installation functionality testing
**Key Classes**:
- `InstallationValidator`: Core functionality testing
- `HardwareValidator`: GPU acceleration testing
- `PerformanceBenchmark`: Speed and optimization validation

### Hardware Configuration Templates (`scripts/hardware_configs/`)
**Purpose**: Platform-specific optimal configurations
**Structure**: JSON configuration files with PyTorch settings, environment variables, model recommendations

### Installation Orchestration Scripts
**Purpose**: Platform-specific installation execution
**Components**:
- `install.sh`: Unix/Linux/macOS orchestration with Bash
- `install.ps1`: Windows orchestration with PowerShell
- Both scripts call Python libraries for detection and resolution

**Output**: Complete design with libraries, contracts, and platform-specific execution

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate setup tasks for project structure and dependencies
- Create TDD test tasks for each detection capability
- Generate implementation tasks for each library class
- Add integration tasks for end-to-end installation scenarios
- Include validation and benchmarking tasks

**Task Categories**:
1. **Setup Tasks**: Project initialization, directory structure, core dependencies
2. **Detection Test Tasks [P]**: Hardware detection tests for each platform/GPU combination
3. **Library Implementation Tasks**: Core detection, resolution, and management libraries
4. **Shell Script Tasks**: Platform-specific installation orchestration
5. **Integration Tasks [P]**: End-to-end installation testing across platforms
6. **Validation Tasks**: Performance benchmarking and functionality verification
7. **Documentation Tasks**: Installation guides, troubleshooting, CI/CD setup

**Ordering Strategy**:
- Test-first: Detection tests before implementation
- Library-first: Core libraries before shell scripts
- Platform independence: Detection libraries before platform-specific scripts
- Validation last: Integration tests after all components

**Estimated Output**: 30-35 numbered, dependency-ordered tasks focusing on TDD implementation of cross-platform installation system

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation following TDD principles with hardware-specific testing  
**Phase 5**: Validation across multiple hardware configurations and platforms

## Complexity Tracking
*No constitutional violations identified - installation tooling follows library-first and CLI principles*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None identified | N/A | N/A |

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Implementation plan ready for task generation and GitHub Copilot coding agent execution*
