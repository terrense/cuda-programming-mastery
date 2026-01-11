#!/bin/bash

# CUDA算子测试运行脚本
# 运行所有自定义算子的单元测试和性能测试

set -e  # 遇到错误时退出

echo "========================================"
echo "CUDA算子测试套件"
echo "========================================"

# 检查CUDA环境
echo "检查CUDA环境..."
if ! command -v nvcc &> /dev/null; then
    echo "警告: NVCC未找到，某些测试可能会跳过"
fi

# 检查GPU可用性
if ! nvidia-smi &> /dev/null; then
    echo "警告: 未检测到NVIDIA GPU，CUDA测试将被跳过"
fi

# 创建测试结果目录
mkdir -p test_results

# 运行数学算子测试
echo ""
echo "=========================================="
echo "运行数学算子单元测试"
echo "=========================================="
if [ -f "./test_math_operators" ]; then
    ./test_math_operators | tee test_results/math_operators_test.log
    MATH_EXIT_CODE=$?
else
    echo "错误: test_math_operators 可执行文件未找到"
    MATH_EXIT_CODE=1
fi

# 运行卷积池化算子测试
echo ""
echo "=========================================="
echo "运行卷积池化算子单元测试"
echo "=========================================="
if [ -f "./test_conv_pool_operators" ]; then
    ./test_conv_pool_operators | tee test_results/conv_pool_operators_test.log
    CONV_EXIT_CODE=$?
else
    echo "错误: test_conv_pool_operators 可执行文件未找到"
    CONV_EXIT_CODE=1
fi

# 运行性能测试
echo ""
echo "=========================================="
echo "运行算子性能测试"
echo "=========================================="
if [ -f "./operator_performance_tests" ]; then
    ./operator_performance_tests | tee test_results/operator_performance_test.log
    PERF_EXIT_CODE=$?

    # 移动性能结果文件
    if [ -f "operator_performance_results.csv" ]; then
        mv operator_performance_results.csv test_results/
        echo "性能测试结果已保存到 test_results/operator_performance_results.csv"
    fi
else
    echo "错误: operator_performance_tests 可执行文件未找到"
    PERF_EXIT_CODE=1
fi

# 生成测试报告
echo ""
echo "=========================================="
echo "生成测试报告"
echo "=========================================="

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 统计测试结果
if [ -f "test_results/math_operators_test.log" ]; then
    MATH_TOTAL=$(grep "Total:" test_results/math_operators_test.log | awk '{print $2}' || echo "0")
    MATH_PASSED=$(grep "Passed:" test_results/math_operators_test.log | awk '{print $2}' || echo "0")
    MATH_FAILED=$(grep "Failed:" test_results/math_operators_test.log | awk '{print $2}' || echo "0")

    TOTAL_TESTS=$((TOTAL_TESTS + MATH_TOTAL))
    PASSED_TESTS=$((PASSED_TESTS + MATH_PASSED))
    FAILED_TESTS=$((FAILED_TESTS + MATH_FAILED))
fi

if [ -f "test_results/conv_pool_operators_test.log" ]; then
    CONV_TOTAL=$(grep "Total:" test_results/conv_pool_operators_test.log | awk '{print $2}' || echo "0")
    CONV_PASSED=$(grep "Passed:" test_results/conv_pool_operators_test.log | awk '{print $2}' || echo "0")
    CONV_FAILED=$(grep "Failed:" test_results/conv_pool_operators_test.log | awk '{print $2}' || echo "0")

    TOTAL_TESTS=$((TOTAL_TESTS + CONV_TOTAL))
    PASSED_TESTS=$((PASSED_TESTS + CONV_PASSED))
    FAILED_TESTS=$((FAILED_TESTS + CONV_FAILED))
fi

if [ -f "test_results/operator_performance_test.log" ]; then
    PERF_TOTAL=$(grep "Total:" test_results/operator_performance_test.log | awk '{print $2}' || echo "0")
    PERF_PASSED=$(grep "Passed:" test_results/operator_performance_test.log | awk '{print $2}' || echo "0")
    PERF_FAILED=$(grep "Failed:" test_results/operator_performance_test.log | awk '{print $2}' || echo "0")

    TOTAL_TESTS=$((TOTAL_TESTS + PERF_TOTAL))
    PASSED_TESTS=$((PASSED_TESTS + PERF_PASSED))
    FAILED_TESTS=$((FAILED_TESTS + PERF_FAILED))
fi

# 生成HTML报告
cat > test_results/test_report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>CUDA算子测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .failed { background-color: #ffe8e8; }
        .test-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CUDA算子测试报告</h1>
        <p>生成时间: $(date)</p>
    </div>

    <div class="summary $([ $FAILED_TESTS -gt 0 ] && echo "failed")">
        <h2>测试总结</h2>
        <table>
            <tr><th>测试类型</th><th>总数</th><th>通过</th><th>失败</th><th>状态</th></tr>
            <tr>
                <td>数学算子</td>
                <td>$MATH_TOTAL</td>
                <td class="pass">$MATH_PASSED</td>
                <td class="fail">$MATH_FAILED</td>
                <td>$([ $MATH_EXIT_CODE -eq 0 ] && echo '<span class="pass">PASS</span>' || echo '<span class="fail">FAIL</span>')</td>
            </tr>
            <tr>
                <td>卷积池化算子</td>
                <td>$CONV_TOTAL</td>
                <td class="pass">$CONV_PASSED</td>
                <td class="fail">$CONV_FAILED</td>
                <td>$([ $CONV_EXIT_CODE -eq 0 ] && echo '<span class="pass">PASS</span>' || echo '<span class="fail">FAIL</span>')</td>
            </tr>
            <tr>
                <td>性能测试</td>
                <td>$PERF_TOTAL</td>
                <td class="pass">$PERF_PASSED</td>
                <td class="fail">$PERF_FAILED</td>
                <td>$([ $PERF_EXIT_CODE -eq 0 ] && echo '<span class="pass">PASS</span>' || echo '<span class="fail">FAIL</span>')</td>
            </tr>
            <tr style="font-weight: bold; background-color: #f9f9f9;">
                <td>总计</td>
                <td>$TOTAL_TESTS</td>
                <td class="pass">$PASSED_TESTS</td>
                <td class="fail">$FAILED_TESTS</td>
                <td>$([ $FAILED_TESTS -eq 0 ] && echo '<span class="pass">PASS</span>' || echo '<span class="fail">FAIL</span>')</td>
            </tr>
        </table>
    </div>

    <div class="test-section">
        <h3>测试详情</h3>
        <p><strong>数学算子测试:</strong> 测试基础数学运算、激活函数、矩阵乘法和归约操作的正确性</p>
        <p><strong>卷积池化算子测试:</strong> 测试2D卷积、池化和批量归一化操作的正确性</p>
        <p><strong>性能测试:</strong> 测量各算子的执行时间、吞吐量和内存使用情况</p>
    </div>

    <div class="test-section">
        <h3>测试文件</h3>
        <ul>
            <li><a href="math_operators_test.log">数学算子测试日志</a></li>
            <li><a href="conv_pool_operators_test.log">卷积池化算子测试日志</a></li>
            <li><a href="operator_performance_test.log">性能测试日志</a></li>
            <li><a href="operator_performance_results.csv">性能测试结果CSV</a></li>
        </ul>
    </div>
</body>
</html>
EOF

echo "测试报告已生成: test_results/test_report.html"

# 打印最终结果
echo ""
echo "=========================================="
echo "最终测试结果"
echo "=========================================="
echo "总测试数: $TOTAL_TESTS"
echo "通过测试: $PASSED_TESTS"
echo "失败测试: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    echo "✅ 所有测试通过!"
    exit 0
else
    echo "❌ 有 $FAILED_TESTS 个测试失败"
    exit 1
fi
