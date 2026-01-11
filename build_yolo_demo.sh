#!/bin/bash

# YOLO模型加载器演示构建脚本

set -e  # 遇到错误时退出

echo "=== YOLO Model Loader Demo Build Script ==="

# 检查CUDA环境
echo "Checking CUDA environment..."
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found. Please install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "CUDA Version: $CUDA_VERSION"

# 检查GPU
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not found. GPU detection skipped."
fi

# 创建构建目录
BUILD_DIR="build_yolo"
echo "Creating build directory: $BUILD_DIR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# 配置CMake
echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
      -DBUILD_TESTING=ON \
      ../src/yolo

# 编译
echo "Building YOLO module..."
make -j$(nproc)

# 检查编译结果
if [ -f "yolo_model_loader_demo" ]; then
    echo "Build successful! Executable created: yolo_model_loader_demo"
else
    echo "Build failed! Executable not found."
    exit 1
fi

# 运行演示程序
echo "Running YOLO model loader demo..."
echo "=================================="
./yolo_model_loader_demo

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "=================================="
    echo "Demo completed successfully!"
else
    echo "Demo failed with error code: $?"
    exit 1
fi

# 运行测试（如果启用）
if [ -f "CTestTestfile.cmake" ]; then
    echo "Running tests..."
    ctest --output-on-failure
fi

echo "=== Build and Demo Completed ==="
