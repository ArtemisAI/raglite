# RAGLite Dependency Management Guide

This document describes the dependency management improvements implemented to address import conflicts and provide a better developer experience.

## Overview

RAGLite now supports optional dependencies and provides robust fallbacks for missing components, making it easier to get started and reducing conflicts with other ML libraries.

## Optional Dependencies

### Core Installation
```bash
pip install raglite
```

The core installation provides:
- SQLite database backend (default)
- Basic embedding and LLM functionality
- Document processing and search
- All core RAG features

### Optional Extras

#### Reranking Support
```bash
pip install raglite[rerank]
```

Adds:
- `rerankers[api,flashrank]` - Advanced reranking models
- `langdetect` - Language detection for multi-language reranking

#### Large Language Models
```bash
pip install raglite[llama-cpp-python]
```

Adds:
- `llama-cpp-python` - Local LLM support with GPU acceleration

#### Other Extras
```bash
pip install raglite[chainlit]  # Web UI
pip install raglite[pandoc]    # Enhanced markdown conversion
pip install raglite[ragas]     # Evaluation metrics
pip install raglite[bench]     # Benchmarking tools
```

## Fallback Behavior

### Without Optional Dependencies

When optional dependencies are not installed, RAGLite provides intelligent fallbacks:

| Component | Default (with deps) | Fallback (without deps) |
|-----------|-------------------|-------------------------|
| **Database** | SQLite | SQLite (always available) |
| **LLM** | llama-cpp-python models | OpenAI GPT-3.5-turbo |
| **Embedder** | llama-cpp-python BGE-M3 | OpenAI text-embedding-ada-002 |
| **Reranker** | FlashRank models | None (disabled gracefully) |

### Configuration Examples

#### Minimal Setup (No API Keys Required)
```python
from raglite import RAGLiteConfig

# Uses SQLite + fallbacks to OpenAI (requires OPENAI_API_KEY)
config = RAGLiteConfig()
```

#### Explicit Local Setup
```python
from raglite import RAGLiteConfig

# Force local models (requires llama-cpp-python)
config = RAGLiteConfig(
    embedder="llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@512",
    llm="llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192"
)
```

#### API-Based Setup
```python
from raglite import RAGLiteConfig

# Explicit API usage
config = RAGLiteConfig(
    embedder="openai/text-embedding-ada-002",
    llm="openai/gpt-4o-mini",
    reranker=None  # Disable reranking
)
```

## GPU Configuration

### Environment Variables

RAGLite automatically detects GPU availability but can be controlled via environment variables:

```bash
# Force GPU usage
export RAGLITE_FORCE_GPU=1

# Disable GPU usage
export RAGLITE_DISABLE_GPU=1

# Standard CUDA variables (auto-detected)
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0
export OLLAMA_CUDA_SUPPORT=true
```

### GPU Detection

RAGLite detects GPU support through:
1. Environment variables (`RAGLITE_FORCE_GPU`, `RAGLITE_DISABLE_GPU`)
2. CUDA environment variables (`CUDA_VISIBLE_DEVICES`, etc.)
3. NVIDIA driver presence (`nvidia-smi` command)
4. llama-cpp-python GPU support detection

## Lazy Loading

### Import Strategy

Heavy dependencies are loaded only when needed:

- **Rerankers**: Only imported when reranking is actually used
- **Transformers/TensorFlow**: Avoided through careful import management
- **llama-cpp-python**: Only accessed through lazy proxy objects

### Benefits

1. **Faster Imports**: Core package imports quickly without loading heavy ML libraries
2. **Reduced Conflicts**: Avoids version conflicts with TensorFlow/PyTorch
3. **Optional Features**: Missing dependencies don't break core functionality
4. **Better Error Messages**: Clear guidance on installing missing dependencies

## Database Support

### SQLite (Default)
- **File**: Single `.db` file
- **Performance**: WAL mode, optimized pragmas
- **Features**: FTS5 for text search, sqlite-vec for vector search
- **Fallbacks**: LIKE-based search when FTS5 unavailable

### DuckDB
```python
config = RAGLiteConfig(db_url="duckdb:///raglite.db")
```

### PostgreSQL
```python
config = RAGLiteConfig(db_url="postgresql://user:pass@host/db")
```

## Troubleshooting

### Import Errors

```python
# Check what's available
from raglite._config import RAGLiteConfig
config = RAGLiteConfig()
print(f"Reranker available: {config.reranker is not None}")
print(f"GPU support: {config.llm.startswith('llama-cpp-python')}")
```

### Missing Dependencies

```bash
# Install specific extras as needed
pip install raglite[rerank]  # For reranking
pip install raglite[llama-cpp-python]  # For local LLMs
```

### GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Set environment variables
export RAGLITE_FORCE_GPU=1

# Test GPU detection
python -c "from raglite._config import _detect_gpu_support; print(_detect_gpu_support())"
```

## Migration from Previous Versions

### Breaking Changes

1. **Default Database**: Changed from DuckDB to SQLite
   - **Action**: Existing DuckDB databases continue to work
   - **Migration**: Specify `db_url="duckdb:///raglite.db"` to keep DuckDB

2. **Optional Reranking**: Rerankers moved to optional dependency
   - **Action**: Install `raglite[rerank]` for reranking features
   - **Fallback**: Reranking disabled gracefully when unavailable

### Compatibility

- All existing database URLs continue to work
- Existing configurations are backward compatible
- Only the default values have changed