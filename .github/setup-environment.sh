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
pip install -e .[dev,test]
pip install sqlite-vec pynndescent

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "🤖 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull embedding model
echo "📥 Pulling Ollama embedding model..."
ollama pull embedding-embeddingemma

# Check GPU support
echo "🔍 Checking GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    export OLLAMA_CUDA_SUPPORT=true
    echo "✅ GPU support enabled for Ollama"

    # Check CUDA availability
    if command -v nvcc &> /dev/null; then
        echo "🔧 CUDA toolkit detected"
        nvcc --version | grep "release"
    else
        echo "⚠️  CUDA toolkit not found - GPU acceleration may be limited"
    fi
else
    echo "💻 No NVIDIA GPU detected, using CPU mode"
    export OLLAMA_CUDA_SUPPORT=false
fi

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

echo "🎉 Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Run tests: pytest tests/ -v"
echo "  2. Test SQLite backend: python tests/test_sqlite_backend.py"
echo "  3. Start development: python -m raglite --help"
