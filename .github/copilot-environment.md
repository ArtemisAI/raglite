# GitHub Copilot Environment Configuration for Raglite

## Environment Variables Configuration

### Core Configuration
```bash
# Python Environment
PYTHONPATH=/workspace/src
PYTHONUNBUFFERED=1

# Database Configuration
DATABASE_URL=sqlite:///data/raglite.db
DATABASE_POOL_SIZE=10
DATABASE_TIMEOUT=30

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_CACHE_DIR=./models/cache
EMBEDDING_BATCH_SIZE=32

# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
API_TIMEOUT=60

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=./logs/raglite.log
```

### Development Environment Variables
```bash
# Development Mode
NODE_ENV=development
DEBUG=true

# Testing Configuration
TEST_DATABASE_URL=sqlite:///tests/test.db
TEST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
PYTEST_VERBOSE=true

# Performance Testing
BENCHMARK_ITERATIONS=100
BENCHMARK_WARMUP=10
```

## Development Dependencies

### Python Dependencies (pyproject.toml)
```toml
[project]
dependencies = [
    "numpy>=1.24.0",
    "sqlite3",
    "sentence-transformers>=2.2.0",
    "openai>=1.0.0",
    "click>=8.0.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "tqdm>=4.65.0"
]

[project.optional-dependencies]
dev = [
    "mypy>=1.5.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pre-commit>=3.0.0"
]
test = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0"
]
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# macOS
brew install python@3.11

# Windows
# Install Python 3.11 from python.org
```

## GitHub Actions Runner Configuration

### Runner Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11+
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 50GB free space
- **Network**: Stable internet connection for API calls

### Runner Setup Steps
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', 'pyproject.toml') }}

- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e .[dev,test]
```

## Testing Environment

### Test Database Setup
```python
# tests/conftest.py
import pytest
import sqlite3
import tempfile
import os

@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database."""
    db_fd, db_path = tempfile.mkstemp()
    conn = sqlite3.connect(db_path)
    # Setup test schema
    yield conn
    conn.close()
    os.close(db_fd)
    os.unlink(db_path)
```

### Test Configuration
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=raglite
    --cov-report=term-missing
    --cov-report=html:htmlcov
```

## Performance Targets

### Benchmark Targets
- **Text Processing**: < 100ms for 10KB documents
- **Embedding Generation**: < 500ms for 1000 tokens
- **Vector Search**: < 50ms for 100K vectors
- **Memory Usage**: < 500MB for 1M vectors
- **Database Query**: < 10ms for simple queries

### Performance Monitoring
```python
# Performance monitoring utilities
import time
from functools import wraps

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
```

## Environment Validation

### Validation Script
```bash
#!/bin/bash
# validate_environment.sh

echo "🔍 Validating Raglite development environment..."

# Check Python version
python --version
if [[ $? -ne 0 ]]; then
    echo "❌ Python not found"
    exit 1
fi

# Check pip
pip --version
if [[ $? -ne 0 ]]; then
    echo "❌ pip not found"
    exit 1
fi

# Install dependencies
pip install -e .[dev,test]
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Run basic import test
python -c "import raglite; print('✅ Raglite imports successfully')"
if [[ $? -ne 0 ]]; then
    echo "❌ Raglite import failed"
    exit 1
fi

echo "🎉 Environment validation complete!"
```
