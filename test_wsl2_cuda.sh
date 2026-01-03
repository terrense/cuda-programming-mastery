#!/bin/bash

echo "CUDA WSL2 Environment Test"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_test() {
    echo -e "${BLUE}Testing:${NC} $1"
}

print_pass() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
}

print_fail() {
    echo -e "${RED}✗ FAIL:${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠ WARN:${NC} $1"
}

# Test 1: WSL2 Detection
print_test "WSL2 Environment Detection"
if grep -qi microsoft /proc/version 2>/dev/null; then
    print_pass "Running in WSL2"
else
    print_warn "Not running in WSL2 (this is OK for native Linux)"
fi

# Test 2: CUDA Compiler
print_test "CUDA Compiler (nvcc)"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_pass "NVCC found - Version $CUDA_VERSION"
else
    print_fail "NVCC not found - Install CUDA Toolkit"
    exit 1
fi

# Test 3: C++ Compiler
print_test "C++ Compiler (g++)"
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | sed 's/.*) //')
    print_pass "g++ found - Version $GCC_VERSION"
else
    print_fail "g++ not found - Install build-essential"
    exit 1
fi

# Test 4: CMake
print_test "CMake Build System"
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
    print_pass "CMake found - Version $CMAKE_VERSION"
else
    print_fail "CMake not found - Install cmake"
    exit 1
fi

# Test 5: GPU Detection
print_test "GPU Detection"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    if [ "$GPU_COUNT" -gt 0 ] 2>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_pass "GPU detected: $GPU_NAME"
    else
        print_fail "No GPUs detected"
    fi
else
    print_warn "nvidia-smi not available (may be normal in some WSL2 setups)"
fi

# Test 6: CUDA Runtime Test
print_test "CUDA Runtime Compilation Test"
cat > /tmp/cuda_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "CUDA Devices: " << deviceCount << std::endl;

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    }

    return 0;
}
EOF

if nvcc -o /tmp/cuda_test /tmp/cuda_test.cu 2>/dev/null; then
    print_pass "CUDA compilation successful"

    # Test 7: CUDA Runtime Execution
    print_test "CUDA Runtime Execution"
    if /tmp/cuda_test 2>/dev/null; then
        print_pass "CUDA runtime test successful"
    else
        print_warn "CUDA runtime test failed (GPU may not be available)"
    fi

    rm -f /tmp/cuda_test /tmp/cuda_test.cu
else
    print_fail "CUDA compilation failed"
    rm -f /tmp/cuda_test.cu
fi

# Test 8: Project Build Test
print_test "Project Build System"
if [ -f "CMakeLists.txt" ] && [ -f "build.sh" ]; then
    print_pass "Build system files found"

    if [ -f "compile_commands.json" ]; then
        print_pass "IntelliSense database found"
    else
        print_warn "compile_commands.json not found - run ./build.sh to generate"
    fi
else
    print_warn "Not in project root directory"
fi

# Test 9: VS Code Integration
print_test "VS Code Integration"
if [ -d ".vscode" ]; then
    print_pass "VS Code configuration found"

    if command -v code &> /dev/null; then
        print_pass "VS Code CLI available"
    else
        print_warn "VS Code CLI not found - install Remote-WSL extension"
    fi
else
    print_warn "VS Code configuration not found"
fi

echo ""
echo "=========================="
echo "Environment Test Complete"
echo "=========================="
echo ""

# Summary
echo "Summary:"
echo "- CUDA Toolkit: $(command -v nvcc &> /dev/null && echo "✓ Installed" || echo "✗ Missing")"
echo "- Build Tools: $(command -v g++ &> /dev/null && command -v cmake &> /dev/null && echo "✓ Ready" || echo "✗ Missing")"
echo "- GPU Access: $(command -v nvidia-smi &> /dev/null && echo "✓ Available" || echo "? Unknown")"
echo ""

if command -v nvcc &> /dev/null && command -v g++ &> /dev/null && command -v cmake &> /dev/null; then
    echo -e "${GREEN}✓ Environment is ready for CUDA development!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Build the project: ./build.sh"
    echo "2. Run tests: ./build/tests/unit_tests"
    echo "3. Try examples: ./build/examples/basic/hello_world_cuda"
else
    echo -e "${RED}✗ Environment setup incomplete${NC}"
    echo ""
    echo "Run the setup script: ./setup_wsl2.sh"
fi
