"""RAGLite config."""

import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rerankers.models.ranker import BaseRanker

from platformdirs import user_data_dir
from sqlalchemy.engine import URL

from raglite._lazy_llama import llama_supports_gpu_offload
from raglite._typing import ChunkId, SearchMethod


def _detect_gpu_support() -> bool:
    """Detect if GPU support is available and configured."""
    # Check environment variables first
    if os.getenv("RAGLITE_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
        return False
    if os.getenv("RAGLITE_FORCE_GPU", "").lower() in ("1", "true", "yes"):
        return True

    # Check for CUDA environment variables (commonly set in GPU environments)
    cuda_vars = ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "OLLAMA_CUDA_SUPPORT"]
    if any(os.getenv(var) for var in cuda_vars):
        return True

    # Try to detect nvidia-smi (indicates NVIDIA GPU driver presence)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fallback to llama-cpp-python detection if available
    try:
        return llama_supports_gpu_offload()
    except Exception:
        return False

# Lazy import helpers for heavy dependencies to avoid import conflicts
def _lazy_import_rerankers():
    """Lazily import rerankers to avoid heavy dependency loading on module import."""
    try:
        # Suppress rerankers output on import until [1] is fixed.
        # [1] https://github.com/AnswerDotAI/rerankers/issues/36
        with contextlib.redirect_stdout(StringIO()):
            from rerankers.models.flashrank_ranker import FlashRankRanker
            from rerankers.models.ranker import BaseRanker
        return FlashRankRanker, BaseRanker
    except ImportError as e:
        raise ImportError(
            "rerankers is required for reranking functionality. "
            "Install with: pip install 'raglite[rerank]' or pip install rerankers"
        ) from e


cache_path = Path(user_data_dir("raglite", ensure_exists=True))


def _default_llm():
    """Create default LLM with safe dependency checking."""
    try:
        gpu_available = _detect_gpu_support()
        return (
            "llama-cpp-python/unsloth/Qwen3-8B-GGUF/*Q4_K_M.gguf@8192"
            if gpu_available
            else "llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192"
        )
    except Exception:
        # Fallback to a simple LLM configuration that doesn't require llama-cpp-python
        return "openai/gpt-3.5-turbo"


def _default_embedder():
    """Create default embedder with safe dependency checking."""
    try:
        gpu_available = _detect_gpu_support()
        cpu_cores = os.cpu_count() or 1
        return (  # Nomic-embed may be better if only English is used.
            "llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512"
            if gpu_available or cpu_cores >= 4  # noqa: PLR2004
            else "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@512"
        )
    except Exception:
        # Fallback to OpenAI embeddings when llama-cpp-python is not available
        return "openai/text-embedding-ada-002"


def _default_reranker():
    """Create default reranker with lazy loading, returns None if rerankers not available."""
    try:
        # Just check if we can import, but don't initialize models yet
        _lazy_import_rerankers()
        # Return a lambda that will create the reranker when actually needed
        return "lazy_reranker"  # Placeholder that indicates reranking is available
    except ImportError:
        # Return None if rerankers are not available - this allows the system to work
        # without reranking functionality
        return None


# Lazily load the default search method to avoid circular imports.
# TODO: Replace with search_and_rerank_chunk_spans after benchmarking.
def _vector_search(
    query: str, *, num_results: int = 8, config: "RAGLiteConfig | None" = None
) -> tuple[list[ChunkId], list[float]]:
    from raglite._search import vector_search

    return vector_search(query, num_results=num_results, config=config)


@dataclass(frozen=True)
class RAGLiteConfig:
    """RAGLite config."""

    # Database config.
    db_url: str | URL = f"sqlite:///{(cache_path / 'raglite.db').as_posix()}"
    # LLM config used for generation.
    llm: str = field(default_factory=_default_llm)
    llm_max_tries: int = 4
    # Embedder config used for indexing.
    embedder: str = field(default_factory=_default_embedder)
    embedder_normalize: bool = True
    # Chunk config used to partition documents into chunks.
    chunk_max_size: int = 2048  # Max number of characters per chunk.
    # Vector search config.
    vector_search_distance_metric: Literal["cosine", "dot", "l2"] = "cosine"
    vector_search_multivector: bool = True
    vector_search_query_adapter: bool = True  # Only supported for "cosine" and "dot" metrics.
    # Reranking config.
    reranker: "BaseRanker | dict[str, BaseRanker] | str | None" = field(
        default_factory=_default_reranker,
        compare=False,  # Exclude the reranker from comparison to avoid lru_cache misses.
    )
    # Search config: you can pick any search method that returns (list[ChunkId], list[float]),
    # list[Chunk], or list[ChunkSpan].
    # GPU Configuration Options
    gpu_enabled: bool = True
    """Enable GPU acceleration when available."""

    gpu_layers: int | None = None
    """Number of layers to offload to GPU. If None, auto-detect optimal value."""

    gpu_memory_fraction: float = 0.8
    """Fraction of GPU memory to use (0.1-1.0)."""

    gpu_fallback_enabled: bool = True
    """Enable automatic fallback to CPU if GPU fails."""

    cpu_threads: int | None = None
    """Number of CPU threads for CPU-only mode. If None, auto-detect."""
    search_method: SearchMethod = field(default=_vector_search, compare=False)
