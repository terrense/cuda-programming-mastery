#!/bin/bash

echo "CUDA Programming Mastery Build Script for WSL2 Ubuntu"
echo "====================================================="

# Check if running in WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "✓ Detected WSL2 environment"
    WSL2_ENV=true
else
    echo "ℹ Running on native Linux"
    WSL2_ENV=false
fi

# Check for CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: NVCC not found. Please install CUDA Toolkit."
    if [ "$WSL2_ENV" = true ]; then
        echo "WSL2 CUDA Setup Guide:"
        echo "1. Install CUDA Toolkit in WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/"
        echo "2. Ensure Windows has NVIDIA drivers with WSL2 support"
    fi
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "✓ CUDA version: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found. Please install CMake 3.18 or later."
    echo "Install with: sudo apt update && sudo apt install cmake"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
echo "✓ CMake version: $CMAKE_VERSION"

# Check for essential build tools
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found. Please install build-essential."
    echo "Install with: sudo apt update && sudo apt install build-essential"
    exit 1
fi

echo "✓ g++ compiler found"

# WSL2 specific checks
if [ "$WSL2_ENV" = true ]; then
    # Check if nvidia-smi works in WSL2
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ nvidia-smi available in WSL2"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits | head -1
    else
        echo "⚠ nvidia-smi not available - this is normal in some WSL2 setups"
    fi
fi

echo "Creating build directory..."
mkdir -p build

echo "Configuring project with WSL2 optimizations..."
cd build

# Configure with WSL2-specific options
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86;89;90"

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed."
    exit 1
fi

echo "Building project..."
cmake --build . --config Release -j$(nproc)
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed."
    exit 1
fi

echo "✓ Build completed successfully!"

# Copy compile_commands.json to root for IntelliSense
if [ -f compile_commands.json ]; then
    cp compile_commands.json ../
    echo "✓ compile_commands.json copied to root for IntelliSense"
fi

echo ""
echo "Next steps:"
echo "1. Test environment: ./build/examples/basic/cuda_env_test"
echo "2. Run hello world: ./build/examples/basic/hello_world_cuda"
echo "3. Run tests: ./build/tests/unit_tests"
echo ""
if [ "$WSL2_ENV" = true ]; then
    echo "WSL2 Tips:"
    echo "- Use VS Code with Remote-WSL extension for best experience"
    echo "- IntelliSense should work with the generated compile_commands.json"
    echo "- GPU memory is shared with Windows host"
fi