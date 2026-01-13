#!/bin/bash

# 编译高级内存优化演示程序
echo "编译高级内存优化演示程序..."

# 创建临时目录
mkdir -p temp_build

# 编译核心文件 - 需要包含必要的头文件
nvcc -c src/core/advanced_memory_optimizer.cu -o temp_build/advanced_memory_optimizer.o \
    -std=c++17 -arch=sm_75 --use_fast_math -lineinfo -I src/core \
    -I/usr/local/cuda/include

# 检查第一步编译是否成功
if [ $? -ne 0 ]; then
    echo "核心文件编译失败！"
    exit 1
fi

# 编译演示程序并链接
nvcc examples/advanced/advanced_memory_optimization_demo.cu temp_build/advanced_memory_optimizer.o \
    -o temp_build/advanced_memory_optimization_demo \
    -std=c++17 -arch=sm_75 --use_fast_math -lineinfo -I src/core \
    -I/usr/local/cuda/include -lcudart

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo "运行演示程序..."
    ./temp_build/advanced_memory_optimization_demo
else
    echo "演示程序编译失败！"
    exit 1
fi
