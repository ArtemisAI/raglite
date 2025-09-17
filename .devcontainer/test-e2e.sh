#!/bin/bash
# End-to-end devcontainer test script
# This script tests the complete devcontainer workflow

set -e

echo "🚀 End-to-End DevContainer Test"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_test() {
    echo -e "${BLUE}🔍 Testing:${NC} $1"
}

print_pass() {
    echo -e "${GREEN}✅ PASS:${NC} $1"
}

print_fail() {
    echo -e "${RED}❌ FAIL:${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠️ WARN:${NC} $1"
}

# Test the complete container build and run process
print_test "Building devcontainer image"
if docker build -t raglite-e2e-test --target dev . > /tmp/docker_build.log 2>&1; then
    print_pass "DevContainer image built successfully"
else
    print_fail "DevContainer build failed"
    echo "Build log:"
    cat /tmp/docker_build.log
    exit 1
fi

# Test running the container
print_test "Running container and testing environment"
docker run --rm -v "$(pwd)":/workspaces/raglite raglite-e2e-test bash -c "
    set -e
    echo '🔍 Testing Python environment in container'
    
    # Test virtual environment path
    echo 'Python executable:' \$(which python)
    echo 'Virtual env path:' \$VIRTUAL_ENV
    
    # Simulate devcontainer setup process
    cd /workspaces/raglite
    echo '🔍 Running setup script simulation'
    pip install -e . > /dev/null 2>&1
    
    # Test raglite import
    python -c 'import raglite; print(\"✅ RAGLite imports successfully\")'
    
    # Test core dependencies
    python -c 'import torch; print(\"✅ PyTorch available:\", torch.__version__)'
    python -c 'import numpy; print(\"✅ NumPy available:\", numpy.__version__)'
    python -c 'import duckdb; print(\"✅ DuckDB available:\", duckdb.__version__)'
    
    # Test GPU detection (should work even without GPU hardware)
    python -c 'from raglite._config import _detect_gpu_support; print(\"✅ GPU detection function works:\", _detect_gpu_support())'
    
    # Test that tests can run
    python -m pytest tests/test_import.py -v
    
    echo '✅ All container tests passed!'
" > /tmp/container_test.log 2>&1

if [ $? -eq 0 ]; then
    print_pass "Container environment test passed"
    echo "Container test output:"
    cat /tmp/container_test.log
else
    print_fail "Container environment test failed"
    echo "Container test error log:"
    cat /tmp/container_test.log
    exit 1
fi

# Cleanup
print_test "Cleaning up test image"
docker rmi raglite-e2e-test > /dev/null 2>&1
print_pass "Cleanup completed"

echo ""
echo "🎯 End-to-End Test Summary:"
echo "=========================="
echo "✅ DevContainer builds successfully"
echo "✅ Python environment works in container"
echo "✅ RAGLite imports correctly"
echo "✅ Core dependencies available"
echo "✅ GPU detection functions properly"
echo "✅ Tests can run in container"
echo ""
echo "🚀 DevContainer configuration is ready for production use!"
echo ""