# GitHub Codespaces Configuration for RAGLite

## Overview
This configuration enables GPU-accelerated development of RAGLite in GitHub Codespaces with NVIDIA CUDA support.

## Machine Requirements
For optimal GPU performance, request a Codespace with:
- **4-core** or higher machine type
- **GPU-enabled** machines (when available in your region)
- **8GB+** memory minimum

## GPU Support
The Codespace is configured to automatically detect and use GPU acceleration when available:

### Automatic Detection
- Checks for NVIDIA drivers and CUDA support
- Installs GPU-enabled PyTorch (CUDA 12.4)
- Compiles llama-cpp-python with CUDA support
- Falls back gracefully to CPU if GPU unavailable

### Environment Variables
```bash
RAGLITE_GPU_ENABLED=true/false    # Auto-detected
CUDA_AVAILABLE=true/false         # Auto-detected  
OLLAMA_CUDA_SUPPORT=true/false    # Auto-detected
RAGLITE_ENV=codespaces           # Set automatically
```

## Quick Start
1. **Open in Codespace**: Click "Code" → "Codespaces" → "Create codespace"
2. **Wait for setup**: The environment will automatically install GPU support
3. **Verify setup**: Run `python scripts/verify_gpu_setup.py`
4. **Start developing**: All GPU features are ready!

## Performance Expectations

### With GPU (when available)
- **Embedding generation**: ~80 docs/second
- **Model inference**: 2-4x faster
- **Memory**: Utilizes GPU VRAM efficiently

### CPU Fallback
- **Embedding generation**: ~22 docs/second  
- **Model inference**: Standard CPU performance
- **Memory**: Uses system RAM only

## Verification Commands
```bash
# Check GPU status
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Full GPU verification
python scripts/verify_gpu_setup.py

# Check environment
echo $RAGLITE_GPU_ENABLED
```

## Extensions Included
- GitHub Copilot & Copilot Chat
- Python development tools
- NVIDIA Nsight (GPU debugging)
- Docker support
- Jupyter notebooks

## Troubleshooting

### GPU Not Detected
1. Check Codespace machine type supports GPU
2. Verify with `nvidia-smi`
3. Restart Codespace if needed

### Package Installation Issues
The setup script includes fallbacks for:
- PyTorch (GPU → CPU fallback)
- llama-cpp-python (CUDA → CPU fallback)
- All dependencies with retry logic

### Performance Issues
1. Ensure GPU is detected: `echo $RAGLITE_GPU_ENABLED`
2. Check VRAM usage: `nvidia-smi`
3. Monitor with: `watch -n 1 nvidia-smi`

## Development Tips
- GPU resources are shared in Codespaces
- Use `torch.cuda.empty_cache()` to free VRAM
- Monitor usage to avoid limits
- CPU fallback maintains full functionality

## Support
- GPU support is automatic when hardware available
- All RAGLite features work in CPU mode  
- Container optimized for development workflow
