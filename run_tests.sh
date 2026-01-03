#!/bin/bash

# CUDA编程学习系统测试运行脚本
#
# 这个脚本用于运行所有的单元测试和集成测试
# 支持在WSL2 Ubuntu环境下运行

echo "=== CUDA编程学习系统测试套件 ==="
echo "开始运行测试..."
echo

# 检查构建目录是否存在
if [ ! -d "build" ]; then
    echo "构建目录不存在，正在创建并构建项目..."
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
    make -j$(nproc)
    cd ..
    echo
fi

# 进入构建目录
cd build

echo "--- 运行环境检查 ---"
echo "检查CUDA环境..."

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
else
    echo "⚠ NVIDIA驱动未找到或不可用"
fi

if command -v nvcc &> /dev/null; then
    echo "✓ CUDA编译器可用"
    nvcc --version | grep "release"
else
    echo "⚠ CUDA编译器未找到"
fi

echo

# 运行测试的函数
run_test() {
    local test_name=$1
    local test_executable=$2

    echo "--- 运行 $test_name ---"

    if [ -f "$test_executable" ]; then
        ./$test_executable
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "✓ $test_name 通过"
        else
            echo "✗ $test_name 失败 (退出码: $exit_code)"
            FAILED_TESTS+=("$test_name")
        fi
    else
        echo "✗ $test_name 可执行文件不存在: $test_executable"
        FAILED_TESTS+=("$test_name")
    fi

    echo
}

# 初始化失败测试列表
FAILED_TESTS=()

# 运行各个测试
run_test "CUDA环境测试" "test_cuda_environment"
run_test "基础示例测试" "test_basic_examples"
run_test "示例编译测试" "test_example_compilation"
run_test "示例执行测试" "test_example_execution"

# 如果存在，运行集成测试
if [ -f "unit_tests" ]; then
    run_test "完整单元测试套件" "unit_tests"
fi

# 运行CTest（如果可用）
echo "--- 运行CTest ---"
if command -v ctest &> /dev/null; then
    ctest --output-on-failure --verbose
    CTEST_EXIT_CODE=$?

    if [ $CTEST_EXIT_CODE -eq 0 ]; then
        echo "✓ CTest 通过"
    else
        echo "✗ CTest 失败"
        FAILED_TESTS+=("CTest")
    fi
else
    echo "⚠ CTest 不可用"
fi

echo

# 运行示例程序（如果存在）
echo "--- 运行示例程序 ---"
EXAMPLE_DIR="../examples"

if [ -d "$EXAMPLE_DIR" ]; then
    # 尝试运行基础示例
    for example in "basic/hello_world_cuda" "basic/vector_addition" "basic/matrix_operations"; do
        if [ -f "$example" ]; then
            echo "运行示例: $example"
            timeout 30s ./$example || echo "示例 $example 运行超时或失败"
            echo
        fi
    done
fi

# 生成测试报告
echo "=== 测试总结 ==="

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "🎉 所有测试都通过了！"
    echo "您的CUDA编程学习环境已经准备就绪。"
    EXIT_CODE=0
else
    echo "❌ 以下测试失败了："
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo
    echo "请检查错误信息并修复问题。"
    EXIT_CODE=1
fi

echo
echo "测试完成。"

# 返回到原始目录
cd ..

exit $EXIT_CODE
