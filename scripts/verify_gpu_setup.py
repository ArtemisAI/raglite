#!/usr/bin/env python3
"""Verify GPU setup in the development container."""

import sys
import subprocess
from typing import Dict, Any

def check_nvidia_smi() -> Dict[str, Any]:
    """Check if nvidia-smi works."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return {
            'available': result.returncode == 0,
            'output': result.stdout if result.returncode == 0 else result.stderr
        }
    except FileNotFoundError:
        return {'available': False, 'output': 'nvidia-smi not found'}

def check_cuda_toolkit() -> Dict[str, Any]:
    """Check CUDA toolkit installation."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        return {
            'available': result.returncode == 0,
            'output': result.stdout if result.returncode == 0 else result.stderr
        }
    except FileNotFoundError:
        return {'available': False, 'output': 'nvcc not found'}

def check_pytorch_cuda() -> Dict[str, Any]:
    """Check PyTorch CUDA support."""
    try:
        import torch
        return {
            'installed': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A',
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }
    except ImportError:
        return {'installed': False, 'error': 'PyTorch not installed'}
    except Exception as e:
        return {'installed': True, 'error': str(e)}

def check_llama_cpp() -> Dict[str, Any]:
    """Check llama-cpp-python installation."""
    try:
        import llama_cpp
        return {
            'installed': True,
            'version': getattr(llama_cpp, '__version__', 'unknown'),
            'cuda_support': hasattr(llama_cpp, '_lib') and hasattr(llama_cpp._lib, 'llama_supports_gpu_offload')
        }
    except ImportError:
        return {'installed': False}
    except Exception as e:
        return {'installed': True, 'error': str(e)}

def main():
    """Run all GPU verification checks."""
    print("🚀 GPU Setup Verification")
    print("=" * 50)
    
    # Check nvidia-smi
    print("\n1. NVIDIA SMI:")
    nvidia_result = check_nvidia_smi()
    if nvidia_result['available']:
        print("✅ nvidia-smi available")
        print(nvidia_result['output'].split('\n')[0])  # First line only
    else:
        print("❌ nvidia-smi not available")
        print(f"Error: {nvidia_result['output']}")
    
    # Check CUDA toolkit
    print("\n2. CUDA Toolkit:")
    cuda_result = check_cuda_toolkit()
    if cuda_result['available']:
        print("✅ CUDA toolkit available")
        print(cuda_result['output'].strip())
    else:
        print("❌ CUDA toolkit not available")
        print(f"Error: {cuda_result['output']}")
    
    # Check PyTorch
    print("\n3. PyTorch CUDA:")
    pytorch_result = check_pytorch_cuda()
    if pytorch_result['installed']:
        print(f"✅ PyTorch installed: {pytorch_result['version']}")
        if pytorch_result.get('cuda_available'):
            print(f"✅ CUDA available: {pytorch_result['cuda_version']}")
            print(f"📱 GPU devices ({pytorch_result['device_count']}):")
            for device in pytorch_result['devices']:
                print(f"   - {device}")
        else:
            print("❌ CUDA not available in PyTorch")
            if 'error' in pytorch_result:
                print(f"Error: {pytorch_result['error']}")
    else:
        print("❌ PyTorch not installed")
    
    # Check llama-cpp-python
    print("\n4. llama-cpp-python:")
    llama_result = check_llama_cpp()
    if llama_result['installed']:
        print(f"✅ llama-cpp-python installed: {llama_result.get('version', 'unknown')}")
        if llama_result.get('cuda_support'):
            print("✅ CUDA support available")
        else:
            print("⚠️  CUDA support unclear or not available")
    else:
        print("❌ llama-cpp-python not installed")
    
    # Environment variables
    print("\n5. Environment Variables:")
    import os
    env_vars = ['CUDA_HOME', 'LD_LIBRARY_PATH', 'NVIDIA_VISIBLE_DEVICES', 'PATH']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # Summary
    print("\n" + "=" * 50)
    gpu_ready = (
        nvidia_result['available'] and 
        pytorch_result.get('cuda_available', False) and
        llama_result.get('installed', False)
    )
    
    if gpu_ready:
        print("🎉 GPU setup appears to be working!")
        print("You can now use GPU acceleration in RAGLite.")
    else:
        print("⚠️  GPU setup needs attention.")
        print("Please check the issues above and rebuild the container.")

if __name__ == '__main__':
    main()
