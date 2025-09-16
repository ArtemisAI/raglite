"""GPU-aware embedding generation for RAGLite."""

import logging
from typing import Optional, Any
from llama_cpp import Llama

from ._config import RAGLiteConfig
from ._gpu_utils import (
    detect_cuda_availability,
    get_gpu_memory_info,
    calculate_optimal_gpu_layers,
    log_gpu_info
)

logger = logging.getLogger(__name__)

class GPUAwareLlamaLLM:
    """GPU-aware LLM factory with automatic optimization."""

    @staticmethod
    def create(
        model: str,
        config: RAGLiteConfig,
        embedding: bool = False
    ) -> Llama:
        """
        Create optimized Llama instance with GPU support.

        Args:
            model: Model identifier or path
            config: RAGLite configuration
            embedding: Whether this is for embedding generation

        Returns:
            Llama: Configured Llama instance
        """
        # Log system information
        log_gpu_info()

        # Determine GPU configuration
        gpu_config = GPUAwareLlamaLLM._determine_gpu_config(config)

        # Create Llama instance
        try:
            llm = GPUAwareLlamaLLM._create_llama_instance(
                model=model,
                embedding=embedding,
                gpu_config=gpu_config,
                config=config
            )
            logger.info(f"LLM created successfully with {gpu_config['n_gpu_layers']} GPU layers")
            return llm

        except Exception as e:
            if config.gpu_fallback_enabled and gpu_config['n_gpu_layers'] > 0:
                logger.warning(f"GPU LLM creation failed, falling back to CPU: {e}")
                return GPUAwareLlamaLLM._create_cpu_fallback(model, embedding, config)
            else:
                raise

    @staticmethod
    def _determine_gpu_config(config: RAGLiteConfig) -> dict:
        """Determine optimal GPU configuration."""
        gpu_config = {
            'n_gpu_layers': 0,
            'use_mlock': False,
            'use_mmap': True,
            'verbose': False
        }

        # Check if GPU is enabled and available
        if not config.gpu_enabled or not detect_cuda_availability():
            logger.info("GPU disabled or unavailable, using CPU")
            return gpu_config

        # Get GPU memory information
        memory_info = get_gpu_memory_info()
        if memory_info is None:
            logger.warning("Could not detect GPU memory, using CPU")
            return gpu_config

        total_memory_mb, available_memory_mb = memory_info

        # Calculate optimal GPU layers
        if config.gpu_layers is not None:
            n_gpu_layers = min(config.gpu_layers, 32)
        else:
            n_gpu_layers = calculate_optimal_gpu_layers(
                int(available_memory_mb * config.gpu_memory_fraction)
            )

        gpu_config.update({
            'n_gpu_layers': n_gpu_layers,
            'use_mlock': True,
            'use_mmap': True,
        })

        logger.info(f"GPU configuration: {n_gpu_layers} layers, {available_memory_mb}MB available")
        return gpu_config

    @staticmethod
    def _create_llama_instance(
        model: str,
        embedding: bool,
        gpu_config: dict,
        config: RAGLiteConfig
    ) -> Llama:
        """Create Llama instance with specified configuration."""
        llama_kwargs = {
            'embedding': embedding,
            'n_ctx': 8192,
            **gpu_config
        }

        # Add CPU thread configuration if specified
        if config.cpu_threads is not None:
            llama_kwargs['n_threads'] = config.cpu_threads

        return Llama.from_pretrained(
            repo_id=model,
            filename="*q4_k_m.gguf",
            **llama_kwargs
        )

    @staticmethod
    def _create_cpu_fallback(
        model: str,
        embedding: bool,
        config: RAGLiteConfig
    ) -> Llama:
        """Create CPU-only fallback instance."""
        logger.info("Creating CPU fallback LLM")

        llama_kwargs = {
            'embedding': embedding,
            'n_ctx': 8192,
            'n_gpu_layers': 0,
            'use_mlock': False,
            'use_mmap': True,
            'verbose': False
        }

        if config.cpu_threads is not None:
            llama_kwargs['n_threads'] = config.cpu_threads

        return Llama.from_pretrained(
            repo_id=model,
            filename="*q4_k_m.gguf",
            **llama_kwargs
        )
class GPUMemoryManager:
    """Manages GPU memory for optimal performance."""

    @staticmethod
    def optimize_for_embedding(llm: Llama) -> None:
        """Optimize LLM instance for embedding generation."""
        try:
            import torch
            if torch.cuda.is_available():
                # Clear GPU cache before embedding generation
                torch.cuda.empty_cache()

                # Set memory growth to avoid fragmentation
                torch.cuda.set_per_process_memory_fraction(0.8)

        except Exception as e:
            logger.warning(f"GPU memory optimization failed: {e}")

    @staticmethod
    def cleanup_gpu_memory() -> None:
        """Clean up GPU memory after embedding generation."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.debug(f"GPU cleanup failed: {e}")

class GPUBatchProcessor:
    """Optimized batch processing for GPU embeddings."""

    def __init__(self, llm: Llama, config: RAGLiteConfig):
        self.llm = llm
        self.config = config
        self.gpu_available = detect_cuda_availability()

    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU memory."""
        if not self.gpu_available:
            return 1  # CPU processing is sequential

        memory_info = get_gpu_memory_info()
        if memory_info is None:
            return 1

        total_memory_mb, available_memory_mb = memory_info

        # Conservative batch sizing based on available memory
        if available_memory_mb > 8192:  # 8GB+
            return 8
        elif available_memory_mb > 4096:  # 4-8GB
            return 4
        elif available_memory_mb > 2048:  # 2-4GB
            return 2
        else:
            return 1

    def process_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Process embeddings in optimized batches."""
        batch_size = self.get_optimal_batch_size()
        embeddings = []

        # Pre-warm GPU if available
        if self.gpu_available:
            GPUMemoryManager.optimize_for_embedding(self.llm)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                embedding = self.llm.create_embedding(text)
                batch_embeddings.append(embedding['data'][0]['embedding'])

            embeddings.extend(batch_embeddings)

            # Periodic cleanup for large batches
            if i % (batch_size * 4) == 0 and self.gpu_available:
                GPUMemoryManager.cleanup_gpu_memory()

        return embeddings
