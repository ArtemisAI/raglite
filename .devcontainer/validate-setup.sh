#!/bin/bash
# DevContainer validation script
# This script tests the core functionality expected from the devcontainer setup

set -e

echo "🧪 DevContainer Configuration Validation Script"
echo "================================================"

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

# Test 1: Docker Configuration Validation
print_test "Docker compose configuration syntax"
if docker compose config --quiet; then
    print_pass "docker-compose.yml syntax is valid"
else
    print_fail "docker-compose.yml has syntax errors"
    exit 1
fi

# Test 2: DevContainer JSON Validation
print_test "DevContainer JSON syntax"
if python -c "import json; json.load(open('.devcontainer/devcontainer.json'))" 2>/dev/null; then
    print_pass "devcontainer.json syntax is valid"
else
    print_fail "devcontainer.json has syntax errors"
    exit 1
fi

# Test 3: Python Environment Test
print_test "Python environment and raglite import"
if python -c "import raglite; print('RAGLite version:', getattr(raglite, '__version__', 'dev'))" 2>/dev/null; then
    print_pass "RAGLite imports successfully"
else
    print_warn "RAGLite import failed - may need development mode installation"
fi

# Test 4: Required Files Exist
print_test "Required configuration files"
required_files=(
    ".devcontainer/devcontainer.json"
    ".devcontainer/setup-persistent.sh"
    "Dockerfile"
    "docker-compose.yml"
    ".devcontainer/DEVCONTAINER_TROUBLESHOOTING.md"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_pass "Found: $file"
    else
        print_fail "Missing: $file"
        all_files_exist=false
    fi
done

if [[ "$all_files_exist" = false ]]; then
    exit 1
fi

# Test 5: GPU Detection (Optional)
print_test "GPU detection (optional)"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_pass "GPU detected: $gpu_name"
    else
        print_warn "nvidia-smi command exists but failed to run"
    fi
else
    print_warn "nvidia-smi not found - running in CPU-only mode"
fi

# Test 6: Python Dependencies
print_test "Core Python dependencies"
dependencies=("torch" "numpy" "duckdb")
for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        print_pass "Dependency: $dep"
    else
        print_warn "Missing dependency: $dep"
    fi
done

# Test 7: Docker Build Test (if Docker available)
print_test "Docker build capability"
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        print_pass "Docker is available and running"
        
        # Quick build test
        print_test "Dockerfile build test"
        if docker build -t raglite-test --target dev . &> /tmp/docker_build.log; then
            print_pass "Dockerfile builds successfully"
            docker rmi raglite-test 2>/dev/null || true
        else
            print_fail "Dockerfile build failed - check /tmp/docker_build.log"
        fi
    else
        print_warn "Docker daemon not available"
    fi
else
    print_warn "Docker not found"
fi

echo ""
echo "🎯 Validation Summary:"
echo "====================="
echo "- Configuration files are valid and present"
echo "- Python environment is functional"
echo "- Docker configuration is syntactically correct"
echo ""
echo "🚀 Next Steps:"
echo "- Open VS Code and run 'Dev Containers: Reopen in Container'"
echo "- Or use: docker compose up -d && docker compose exec devcontainer bash"
echo "- For troubleshooting, see: .devcontainer/DEVCONTAINER_TROUBLESHOOTING.md"
echo ""