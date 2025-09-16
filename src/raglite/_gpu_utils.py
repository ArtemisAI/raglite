"""GPU utilities and detection for RAGLite."""

import logging
from typing import Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

def detect_cuda_availability() -> bool:
    """
    Detect if CUDA is available and functional.

    Returns:
        bool: True if CUDA is available and working, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.info("PyTorch not installed, CUDA detection disabled")
        return False
    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")
        return False

def get_gpu_memory_info() -> Optional[Tuple[int, int]]:
    """
    Get GPU memory information.

    Returns:
        Optional[Tuple[int, int]]: (total_memory_mb, available_memory_mb) or None
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = total_memory - allocated_memory

        return (total_memory // 1024 // 1024, available_memory // 1024 // 1024)
    except Exception as e:
        logger.warning(f"GPU memory detection failed: {e}")
        return None

def calculate_optimal_gpu_layers(total_memory_mb: int, model_size_estimate_mb: int = 2048) -> int:
    """
    Calculate optimal number of GPU layers based on available memory.

    Args:
        total_memory_mb: Total GPU memory in MB
        model_size_estimate_mb: Estimated model size in MB

    Returns:
        int: Optimal number of GPU layers (0-32)
    """
    if total_memory_mb < 2048:  # Less than 2GB
        return 0
    elif total_memory_mb < 4096:  # 2-4GB
        return 16
    elif total_memory_mb < 8192:  # 4-8GB
        return 24
    else:  # 8GB+
        return 32

def log_gpu_info() -> None:
    """Log detailed GPU information for debugging."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            logger.info(f"GPU detected: {props.name}")
            logger.info(f"GPU memory: {props.total_memory // 1024 // 1024} MB")
            logger.info(f"GPU compute capability: {props.major}.{props.minor}")
        else:
            logger.info("No GPU detected, using CPU")
    except Exception as e:
        logger.warning(f"GPU info logging failed: {e}")
