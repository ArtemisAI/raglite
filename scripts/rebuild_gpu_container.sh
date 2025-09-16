#!/bin/bash

# GPU-enabled Development Container Rebuild Script
# This script rebuilds the RAGLite development container with GPU support

set -e

echo "🚀 Rebuilding RAGLite Development Container with GPU Support"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run this from the RAGLite root directory."
    exit 1
fi

# Stop existing containers
echo "📦 Stopping existing containers..."
docker-compose down

# Clean up existing images (optional - uncomment if you want fresh build)
# echo "🧹 Cleaning up existing images..."
# docker system prune -f
# docker image rm raglite-devcontainer 2>/dev/null || true

# Rebuild with GPU support
echo "🔨 Building GPU-enabled container..."
docker-compose build --no-cache devcontainer

# Start the container
echo "🚀 Starting GPU-enabled container..."
docker-compose up -d devcontainer

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 10

# Verify GPU setup
echo "🔍 Verifying GPU setup..."
docker-compose exec devcontainer python /workspaces/raglite/scripts/verify_gpu_setup.py

echo ""
echo "✅ Container rebuild complete!"
echo ""
echo "Next steps:"
echo "1. Open VS Code"
echo "2. Use 'Dev Containers: Reopen in Container'"
echo "3. Run the verification script: python scripts/verify_gpu_setup.py"
echo ""
echo "If you encounter issues:"
echo "- Check Docker daemon has GPU support: docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi"
echo "- Ensure NVIDIA Container Runtime is installed"
echo "- Verify your host driver version supports the container CUDA version"
