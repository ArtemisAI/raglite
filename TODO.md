# Project Status & TODO

This document summarizes the current status of the LlamaCpp GPU acceleration implementation and outlines the next steps for development and testing on a new machine.

## Accomplishments

- **GPU Acceleration Implementation**: The core logic for GPU acceleration has been implemented in `src/raglite/_embedding_gpu.py` and `src/raglite/_gpu_utils.py`. This includes:
  - Automatic detection of CUDA-enabled GPUs.
  - Dynamic allocation of GPU layers based on available VRAM.
  - Graceful fallback to CPU processing when a GPU is not available.
- **Configuration**: The application's configuration in `src/raglite/_config.py` has been updated with options to manage GPU support.
- **Unit & Benchmark Tests**: New tests have been added in `tests/test_gpu_acceleration.py` and `tests/benchmark_gpu_performance.py`.
- **Dependency Management**: The `pyproject.toml` file has been updated to include optional dependencies for GPU support (`torch`, `llama-cpp-python`) and benchmarking (`ir_datasets`, etc.).
- **Build Configuration**: The `pyproject.toml` file has been updated to fix test discovery issues and to ignore non-critical warnings from `huggingface_hub`.

## Current Status

### What's Working
- The new GPU-related modules and functions have been created.
- The configuration and `pyproject.toml` files have been updated.
- The majority of the test suite is passing.

### Known Issues & Failures
- **Test Failure in `test_database.py`**: The test `test_in_memory_duckdb_creation` is failing with a `huggingface_hub.errors.HFValidationError`.
  - **Root Cause**: The model string `llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512` is being passed to `GPUAwareLlamaLLM.create` which then passes it to `Llama.from_pretrained` without stripping the `llama-cpp-python/` prefix. The `huggingface_hub` library correctly identifies this as an invalid repository ID.
  - **Location of Bug**: The `GPUAwareLlamaLLM.create` method in `src/raglite/_embedding_gpu.py` needs to be updated to correctly parse the model string, similar to how the old `LlamaCppPythonLLM.llm` method did.

## Next Steps for New Machine

1.  **Set up Environment**:
    *   Clone the repository.
    *   Create a Python virtual environment.
    *   Install the project with all optional dependencies for development, GPU, and benchmarking:
        ```bash
        pip install -e ".[dev,gpu,bench]"
        ```
2.  **Fix the Known Issue**:
    *   Modify the `GPUAwareLlamaLLM.create` method in `src/raglite/_embedding_gpu.py` to correctly parse the `repo_id` from the model string by removing the `llama-cpp-python/` prefix before passing it to `Llama.from_pretrained`.
3.  **Run Full Test Suite**:
    *   Execute the entire test suite to ensure all tests pass on the new machine with a GPU.
        ```bash
        pytest
        ```
4.  **Run Performance Benchmarks**:
    *   Execute the performance benchmarks to compare CPU vs. GPU performance and validate the speedup.
        ```bash
        pytest tests/benchmark_gpu_performance.py
        ```
5.  **Document Performance**:
    *   Document the performance improvements observed in the `LLAMACPP_GPU_ACCELERATION_SPECIFICATION.md` or a new performance report.