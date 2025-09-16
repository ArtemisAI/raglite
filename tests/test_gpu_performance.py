"""Performance benchmarks for GPU acceleration."""

import time
import logging
from typing import List, Tuple
import pytest

from raglite._config import RAGLiteConfig
from raglite._embedding_gpu import GPUAwareLlamaLLM, GPUBatchProcessor
from raglite._gpu_utils import detect_cuda_availability


logger = logging.getLogger(__name__)


class GPUPerformanceBenchmark:
    """Benchmark GPU acceleration performance."""

    def __init__(self):
        self.test_texts = [
            "This is a test document for embedding generation.",
            "RAGLite provides efficient retrieval-augmented generation.",
            "GPU acceleration significantly improves performance.",
            "Embeddings are essential for semantic search.",
            "Vector databases enable fast similarity search."
        ] * 20  # 100 test documents

    def benchmark_cpu_vs_gpu(self) -> Tuple[float, float]:
        """
        Benchmark CPU vs GPU performance.

        Returns:
            Tuple[float, float]: (cpu_time, gpu_time) in seconds
        """
        # CPU benchmark
        cpu_config = RAGLiteConfig(gpu_enabled=False)
        cpu_time = self._benchmark_embedding_generation(cpu_config)

        # GPU benchmark (if available)
        if detect_cuda_availability():
            gpu_config = RAGLiteConfig(gpu_enabled=True)
            gpu_time = self._benchmark_embedding_generation(gpu_config)
        else:
            gpu_time = cpu_time  # Same as CPU if no GPU
            logger.warning("No GPU available for benchmarking")

        return cpu_time, gpu_time

    def _benchmark_embedding_generation(self, config: RAGLiteConfig) -> float:
        """Benchmark embedding generation with given configuration."""
        try:
            # Create LLM instance
            llm = GPUAwareLlamaLLM.create(
                model="sentence-transformers/all-MiniLM-L6-v2",
                config=config,
                embedding=True
            )

            # Create batch processor
            processor = GPUBatchProcessor(llm, config)

            # Benchmark embedding generation
            start_time = time.time()
            embeddings = processor.process_embeddings_batch(self.test_texts)
            end_time = time.time()

            total_time = end_time - start_time
            documents_per_second = len(self.test_texts) / total_time

            logger.info(f"Processed {len(embeddings)} embeddings in {total_time:.2f}s")
            logger.info(f"Throughput: {documents_per_second:.2f} documents/second")

            return total_time

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return float('inf')

    def run_comprehensive_benchmark(self) -> dict:
        """Run comprehensive performance benchmark."""
        results = {}

        # Basic CPU vs GPU benchmark
        cpu_time, gpu_time = self.benchmark_cpu_vs_gpu()
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

        results['cpu_time'] = cpu_time
        results['gpu_time'] = gpu_time
        results['speedup'] = speedup
        results['gpu_available'] = detect_cuda_availability()

        # Batch size optimization
        if detect_cuda_availability():
            results['optimal_batch_sizes'] = self._benchmark_batch_sizes()

        return results

    def _benchmark_batch_sizes(self) -> dict:
        """Benchmark different batch sizes for GPU."""
        config = RAGLiteConfig(gpu_enabled=True)
        batch_results = {}

        for batch_size in [1, 2, 4, 8, 16]:
            try:
                # This would test different batch sizes
                # Implementation depends on batch processing capabilities
                batch_results[batch_size] = 1.0  # Placeholder
            except Exception as e:
                logger.warning(f"Batch size {batch_size} failed: {e}")

        return batch_results


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_gpu_performance_improvement(self):
        """Test that GPU provides performance improvement."""
        benchmark = GPUPerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark()

        if results['gpu_available']:
            # GPU should be faster than CPU
            assert results['speedup'] > 1.0, f"GPU speedup {results['speedup']} should be > 1.0"

            # Log performance results
            logger.info(f"Performance Results:")
            logger.info(f"  CPU Time: {results['cpu_time']:.2f}s")
            logger.info(f"  GPU Time: {results['gpu_time']:.2f}s")
            logger.info(f"  Speedup: {results['speedup']:.2f}x")
        else:
            pytest.skip("No GPU available for performance testing")
