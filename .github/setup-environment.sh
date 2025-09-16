#!/bin/bash
# Environment setup script for Raglite development and testing
# This script ensures all dependencies are installed and configured

set -e

echo "🚀 Setting up Raglite development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $python_version"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Install core and development requirements
echo "🔄 Installing from requirements files..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || {
        echo "⚠️  Core requirements install failed, trying individual packages..."
        pip install sqlite-vec pynndescent torch llama-cpp-python faiss-cpu openai pandas
    }
else
    echo "⚠️  requirements.txt not found, installing essential packages..."
    pip install sqlite-vec pynndescent torch llama-cpp-python
fi

if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt || {
        echo "⚠️  Dev requirements install failed, trying essential dev packages..."
        pip install pytest mypy ruff pre-commit
    }
else
    echo "⚠️  requirements-dev.txt not found, installing essential dev packages..."
    pip install pytest mypy ruff
fi

# Install with retry mechanism for editable package
echo "🔄 Installing package in development mode..."
pip install -e .[dev,gpu,bench] || {
    echo "⚠️  Full install failed, trying basic development install..."
    pip install -e . || {
        echo "❌ Development install failed, trying production install..."
        pip install .
    }
}

# Install SQLite dependencies with fallbacks
echo "🗄️  Installing SQLite dependencies..."
pip install sqlite-vec pynndescent || {
    echo "⚠️  Direct install failed, trying individual packages..."
    pip install sqlite-vec || echo "⚠️  sqlite-vec install failed - will use fallback"
    pip install pynndescent || echo "⚠️  pynndescent install failed - will use fallback"
}

# Install Node.js dependencies with yarn fallback if needed
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    if command -v npm &> /dev/null; then
        npm install || {
            echo "⚠️  npm install failed, trying yarn..."
            if command -v yarn &> /dev/null; then
                yarn install
            else
                echo "🔄 Installing yarn..."
                npm install -g yarn || curl -o- -L https://yarnpkg.com/install.sh | bash
                yarn install
            fi
        }
    else
        echo "🔄 Installing npm..."
        # Install Node.js if not present
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs
        npm install
    fi
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "🤖 Installing Ollama..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo "📥 Please install Ollama manually from https://ollama.ai/download"
        echo "⚠️  Continuing without Ollama - will use fallback embeddings"
    else
        echo "⚠️  Unknown OS type, skipping Ollama install"
    fi
fi

# Pull embedding model with fallback
echo "📥 Pulling Ollama embedding model..."
ollama pull nomic-embed-text || {
    echo "⚠️  Failed to pull nomic-embed-text, trying alternative..."
    ollama pull all-minilm || {
        echo "⚠️  Failed to pull embedding models - will use OpenAI fallback"
        export RAGLITE_EMBEDDER="openai"
    }
}

# Check GPU support
echo "🔍 Checking GPU support..."
export RAGLITE_GPU_ENABLED="false"
export OLLAMA_CUDA_SUPPORT="false"

if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || {
        echo "⚠️  nvidia-smi found but not working - may be in container without GPU access"
    }
    export OLLAMA_CUDA_SUPPORT=true
    export RAGLITE_GPU_ENABLED="true"
    echo "✅ GPU support enabled for Ollama and RAGLite"

    # Check CUDA availability
    if command -v nvcc &> /dev/null; then
        echo "🔧 CUDA toolkit detected"
        nvcc --version | grep "release"
        export CUDA_AVAILABLE="true"
    else
        echo "⚠️  CUDA toolkit not found - checking if CUDA libs are available"
        if [ -d "/usr/local/cuda" ] || [ -d "/usr/local/cuda-12.4" ]; then
            echo "✅ CUDA libraries found"
            export CUDA_AVAILABLE="true"
        else
            echo "⚠️  CUDA not found - GPU acceleration may be limited"
            export CUDA_AVAILABLE="false"
        fi
    fi
    
    # Install GPU-enabled PyTorch for Codespaces/container environments
    echo "🚀 Installing GPU-enabled PyTorch..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || {
        echo "⚠️  GPU PyTorch install failed, falling back to CPU version"
        pip install torch torchvision torchaudio
        export RAGLITE_GPU_ENABLED="false"
    }
    
    # Install GPU-enabled llama-cpp-python
    echo "🦙 Installing GPU-enabled llama-cpp-python..."
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir || {
        echo "⚠️  GPU llama-cpp-python install failed, falling back to CPU version"
        pip install llama-cpp-python --force-reinstall --no-cache-dir
    }
    
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "🔥 PyTorch reports CUDA is available (container GPU support)"
    export RAGLITE_GPU_ENABLED="true"
    export OLLAMA_CUDA_SUPPORT="true"
    export CUDA_AVAILABLE="true"
    echo "✅ Container GPU support detected"
else
    echo "💻 No NVIDIA GPU detected, using CPU mode"
    export OLLAMA_CUDA_SUPPORT=false
    export RAGLITE_GPU_ENABLED="false"
    export CUDA_AVAILABLE="false"
fi

# Set environment variables in bashrc for persistence
echo "📝 Setting persistent environment variables..."
{
    echo "export RAGLITE_GPU_ENABLED='$RAGLITE_GPU_ENABLED'"
    echo "export OLLAMA_CUDA_SUPPORT='$OLLAMA_CUDA_SUPPORT'"
    echo "export CUDA_AVAILABLE='$CUDA_AVAILABLE'"
    if [ "$CUDA_AVAILABLE" = "true" ]; then
        echo "export CUDA_HOME='/usr/local/cuda-12.4'"
        echo "export PATH='/usr/local/cuda-12.4/bin:\$PATH'"
        echo "export LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH'"
    fi
} >> ~/.bashrc

# Validate SQLite-vec
echo "🔍 Validating SQLite-vec installation..."
python3 -c "
import sqlite3
import sqlite_vec
import tempfile
import os

# Create temporary database
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

try:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    
    # Create test table
    conn.execute('CREATE VIRTUAL TABLE test_vec USING vec0(id INTEGER PRIMARY KEY, vec FLOAT[3])')
    
    # Insert test data
    conn.execute('INSERT INTO test_vec VALUES (1, vec_f32([0.1, 0.2, 0.3]))')
    conn.commit()
    
    # Test search
    result = conn.execute('SELECT id FROM test_vec ORDER BY vec_distance_cosine(vec, vec_f32([0.1, 0.2, 0.3])) LIMIT 1').fetchone()
    
    if result:
        print('✅ SQLite-vec working correctly')
    else:
        print('❌ SQLite-vec test failed')
        
finally:
    conn.close()
    os.unlink(db_path)
"

# Validate test database
echo "🗄️  Validating test database..."
if [ -f "tests/test_raglite.db" ]; then
    python3 -c "
import sqlite3
conn = sqlite3.connect('tests/test_raglite.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM documents')
doc_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM chunks')
chunk_count = cursor.fetchone()[0]
print(f'✅ Test database: {doc_count} documents, {chunk_count} chunks')
conn.close()
"
else
    echo "📝 Creating test database..."
    python3 tests/create_test_db.py
fi

# Run basic validation
echo "🧪 Running basic validation..."
python3 -c "
import raglite
print('✅ Raglite imports successfully')

# Test basic functionality
from raglite import RAGLiteConfig
config = RAGLiteConfig()
print(f'✅ Config created with database: {config.db_url}')
"

# GPU validation if enabled
if [ "$RAGLITE_GPU_ENABLED" = "true" ]; then
    echo "🔥 Running GPU validation..."
    python3 -c "
import torch
print('🚀 PyTorch version:', torch.__version__)
print('� CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('📱 GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'   - GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'     Memory: {props.total_memory / 1024**3:.1f}GB')
else:
    print('⚠️  CUDA not available in PyTorch')

# Test llama-cpp-python
try:
    import llama_cpp
    print('🦙 llama-cpp-python available:', hasattr(llama_cpp, '__version__'))
except ImportError:
    print('⚠️  llama-cpp-python not available')
"
fi

echo "�🎉 Environment setup complete!"
echo ""
echo "🌟 Environment Summary:"
echo "  GPU Enabled: $RAGLITE_GPU_ENABLED"
echo "  CUDA Available: $CUDA_AVAILABLE"
echo "  Ollama CUDA: $OLLAMA_CUDA_SUPPORT"
if [ "$RAGLITE_ENV" = "codespaces" ]; then
    echo "  Environment: GitHub Codespaces"
else
    echo "  Environment: Local Development"
fi
echo ""
echo "📋 Next steps:"
echo "  1. Run tests: pytest tests/ -v"
echo "  2. Test SQLite backend: python tests/test_sqlite_backend.py"
if [ -f "scripts/verify_gpu_setup.py" ]; then
    echo "  3. Verify GPU setup: python scripts/verify_gpu_setup.py"
fi
echo "  4. Start development: python -m raglite --help"
