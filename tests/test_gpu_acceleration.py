"""Tests for GPU acceleration functionality."""

import pytest
from unittest.mock import Mock, patch
import logging

from raglite._config import RAGLiteConfig
from raglite._gpu_utils import (
    detect_cuda_availability,
    get_gpu_memory_info,
    calculate_optimal_gpu_layers
)
from raglite._embedding_gpu import GPUAwareLlamaLLM, GPUMemoryManager


class TestGPUDetection:
    """Test GPU detection and configuration."""

    def test_cuda_detection_with_torch(self):
        """Test CUDA detection when PyTorch is available."""
        with patch('torch.cuda.is_available', return_value=True):
            assert detect_cuda_availability() is True

    def test_cuda_detection_without_torch(self):
        """Test CUDA detection when PyTorch is not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            assert detect_cuda_availability() is False

    def test_gpu_memory_info_success(self):
        """Test successful GPU memory detection."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8192 * 1024 * 1024
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024

        with patch.dict('sys.modules', {'torch': mock_torch}):
            total, available = get_gpu_memory_info()
            assert total == 8192
            assert available == 7168  # 8192 - 1024

    def test_optimal_gpu_layers_calculation(self):
        """Test optimal GPU layer calculation."""
        assert calculate_optimal_gpu_layers(1024) == 0    # < 2GB
        assert calculate_optimal_gpu_layers(3072) == 16   # 2-4GB
        assert calculate_optimal_gpu_layers(6144) == 24   # 4-8GB
        assert calculate_optimal_gpu_layers(10240) == 32  # > 8GB


class TestGPUAwareLLM:
    """Test GPU-aware LLM creation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGLiteConfig(
            gpu_enabled=True,
            gpu_fallback_enabled=True,
            gpu_memory_fraction=0.8
        )

    def test_cpu_fallback_when_gpu_disabled(self, config):
        """Test CPU fallback when GPU is disabled."""
        config.gpu_enabled = False

        with patch('raglite._embedding_gpu.Llama') as mock_llama:
            mock_instance = Mock()
            mock_llama.from_pretrained.return_value = mock_instance

            result = GPUAwareLlamaLLM.create("test-model", config, embedding=True)

            assert result == mock_instance
            # Verify CPU-only configuration
            call_kwargs = mock_llama.from_pretrained.call_args[1]
            assert call_kwargs['n_gpu_layers'] == 0

    @patch('raglite._embedding_gpu.detect_cuda_availability', return_value=True)
    @patch('raglite._embedding_gpu.get_gpu_memory_info', return_value=(8192, 6144))
    def test_gpu_configuration_with_available_gpu(self, mock_memory, mock_cuda, config):
        """Test GPU configuration when GPU is available."""
        with patch('raglite._embedding_gpu.Llama') as mock_llama:
            mock_instance = Mock()
            mock_llama.from_pretrained.return_value = mock_instance

            result = GPUAwareLlamaLLM.create("test-model", config, embedding=True)

            assert result == mock_instance
            # Verify GPU configuration
            call_kwargs = mock_llama.from_pretrained.call_args[1]
            assert call_kwargs['n_gpu_layers'] > 0
            assert call_kwargs['use_mlock'] is True

    def test_fallback_on_gpu_failure(self, config):
        """Test fallback to CPU when GPU creation fails."""
        with patch('raglite._embedding_gpu.detect_cuda_availability', return_value=True), \
             patch('raglite._embedding_gpu.get_gpu_memory_info', return_value=(8192, 6144)), \
             patch('raglite._embedding_gpu.Llama') as mock_llama:

            # First call (GPU) fails, second call (CPU) succeeds
            mock_instance = Mock()
            mock_llama.from_pretrained.side_effect = [Exception("GPU error"), mock_instance]

            result = GPUAwareLlamaLLM.create("test-model", config, embedding=True)

            assert result == mock_instance
            assert mock_llama.from_pretrained.call_count == 2


class TestGPUMemoryManager:
    """Test GPU memory management."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.set_per_process_memory_fraction')
    def test_embedding_optimization(self, mock_fraction, mock_cache, mock_available):
        """Test embedding optimization."""
        mock_llm = Mock()

        with patch.dict('sys.modules', {'torch': Mock()}):
            GPUMemoryManager.optimize_for_embedding(mock_llm)

        mock_cache.assert_called_once()
        mock_fraction.assert_called_once_with(0.8)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_memory_cleanup(self, mock_sync, mock_cache, mock_available):
        """Test GPU memory cleanup."""
        with patch.dict('sys.modules', {'torch': Mock()}):
            GPUMemoryManager.cleanup_gpu_memory()

        mock_cache.assert_called_once()
        mock_sync.assert_called_once()


class TestIntegration:
    """Integration tests for GPU acceleration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGLiteConfig(
            gpu_enabled=True,
            gpu_fallback_enabled=True
        )

    def test_end_to_end_embedding_generation(self, config):
        """Test complete embedding generation flow."""
        # This would be a real integration test with actual model loading
        # Skipped in unit tests but important for validation
        pytest.skip("Integration test - requires actual model and GPU")

    def test_performance_benchmark(self, config):
        """Test performance comparison between CPU and GPU."""
        # This would benchmark actual performance
        pytest.skip("Performance test - requires actual model and GPU")
