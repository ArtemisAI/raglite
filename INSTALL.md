# Installation Guide for Raglite

## Prerequisites
- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

### Using pip
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Using uv (recommended)
```bash
# Install from pyproject.toml with all optional dependencies
uv pip install -e ".[dev,gpu,bench,rerank,ragas,chainlit,pandoc]"

# Or install specific groups
uv pip install -e ".[dev,gpu]"  # Development + GPU support
```

## Optional Dependencies

### GPU Support
For GPU acceleration with LlamaCpp:
```bash
pip install torch llama-cpp-python
# or
uv pip install -e ".[gpu]"
```

### Benchmarking
For performance benchmarking:
```bash
pip install faiss-cpu ir_datasets ir_measures llama-index openai pandas
# or
uv pip install -e ".[bench]"
```

### Reranking
For advanced reranking capabilities:
```bash
pip install langdetect rerankers[api,flashrank]
# or
uv pip install -e ".[rerank]"
```

### Evaluation
For RAG evaluation with Ragas:
```bash
pip install pandas ragas
# or
uv pip install -e ".[ragas]"
```

### Frontend
For Chainlit web interface:
```bash
pip install chainlit
# or
uv pip install -e ".[chainlit]"
```

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install with development dependencies:
   ```bash
   uv pip install -e ".[dev,gpu,bench]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

## GPU Requirements

For GPU acceleration, ensure you have:
- CUDA-compatible GPU
- CUDA toolkit installed
- PyTorch with CUDA support
- llama-cpp-python compiled with CUDA

Note: In development containers without GPU access, the code will gracefully fall back to CPU processing.
