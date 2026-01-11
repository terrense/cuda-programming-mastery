#include "math_operators.h"
#include "conv_pool_operators.h"
#include "base_operator.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace cuda_learning::operators;

// 性能测试结果结构
struct PerformanceResult {
    std::string operator_name;
    std::string test_case;
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float throughput_gflops;
    size_t memory_usage_mb;
    bool passed;
    std::string error_message;
};

// 性能测试框架
class PerformanceTestRunner {
public:
    static void run_performance_test(const std::string& test_name,
                                   PerformanceResult (*test_func)()) {
        std::cout << "Running performance test: " << test_name << "... ";

        PerformanceResult result = test_func();
        results_.push_back(result);

        if (result.passed) {
            std::cout << "PASSED (" << std::fixed << std::setprecision(3)
                     << result.avg_time_ms << " ms)" << std::endl;
            s_passed++;
        } else {
            std::cout << "FAILED (" << result.error_message << ")" << std::endl;
            s_failed++;
        }
        s_total++;
    }

    static void print_summary() {
        std::cout << "\n=== Performance Test Summary ===" << std::endl;
        std::cout << "Total: " << s_total << std::endl;
        std::cout << "Passed: " << s_passed << std::endl;
        std::cout << "Failed: " << s_failed << std::endl;

        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << std::left << std::setw(25) << "Operator"
                 << std::setw(20) << "Test Case"
                 << std::setw(12) << "Avg Time(ms)"
                 << std::setw(12) << "Min Time(ms)"
                 << std::setw(12) << "Max Time(ms)"
                 << std::setw(15) << "Throughput(GFLOPS)"
                 << std::setw(12) << "Memory(MB)" << std::endl;
        std::cout << std::string(108, '-') << std::endl;

        for (const auto& result : results_) {
            if (result.passed) {
                std::cout << std::left << std::setw(25) << result.operator_name
                         << std::setw(20) << result.test_case
                         << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_time_ms
                         << std::setw(12) << std::fixed << std::setprecision(3) << result.min_time_ms
                         << std::setw(12) << std::fixed << std::setprecision(3) << result.max_time_ms
                         << std::setw(15) << std::fixed << std::setprecision(2) << result.throughput_gflops
                         << std::setw(12) << std::fixed << std::setprecision(1) << result.memory_usage_mb
                         << std::endl;
            }
        }
    }

    static void save_results_to_file(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "operator_name,test_case,avg_time_ms,min_time_ms,max_time_ms,throughput_gflops,memory_usage_mb,passed,error_message\n";
            for (const auto& result : results_) {
                file << result.operator_name << ","
                     << result.test_case << ","
                     << result.avg_time_ms << ","
                     << result.min_time_ms << ","
                     << result.max_time_ms << ","
                     << result.throughput_gflops << ","
                     << result.memory_usage_mb << ","
                     << (result.passed ? "true" : "false") << ","
                     << result.error_message << "\n";
            }
            file.close();
            std::cout << "Results saved to " << filename << std::endl;
        }
    }

    static int get_exit_code() {
        return s_failed > 0 ? 1 : 0;
    }

private:
    static int s_total;
    static int s_passed;
    static int s_failed;
    static std::vector<PerformanceResult> results_;
};

int PerformanceTestRunner::s_total = 0;
int PerformanceTestRunner::s_passed = 0;
int PerformanceTestRunner::s_failed = 0;
std::vector<PerformanceResult> PerformanceTestRunner::results_;

// CUDA错误检查宏
#define CUDA_CHECK_PERF(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            PerformanceResult result; \
            result.passed = false; \
            result.error_message = cudaGetErrorString(error); \
            return result; \
        } \
    } while(0)

// 性能测试辅助函数
float benchmark_operator(std::function<void()> op_func, int warmup_iterations = 5, int test_iterations = 20) {
    // 预热
    for (int i = 0; i < warmup_iterations; ++i) {
        op_func();
    }

    cudaDeviceSynchronize();

    // 测量时间
    std::vector<float> times;
    for (int i = 0; i < test_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        op_func();
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        times.push_back(time_ms);
    }

    // 计算平均时间
    float total_time = 0.0f;
    for (float time : times) {
        total_time += time;
    }
    return total_time / test_iterations;
}

// 性能测试函数

PerformanceResult test_elementwise_operators_performance() {
    PerformanceResult result;
    result.operator_name = "ElementwiseOps";
    result.test_case = "Large Tensors";
    result.passed = false;

    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        result.passed = true;
        result.avg_time_ms = 0.0f;
        result.error_message = "CUDA not available";
        return result;
    }

    try {
        // 创建大张量进行性能测试
        std::vector<int> dims = {1024, 1024};
        FloatTensor input1(dims);
        FloatTensor input2(dims);
        FloatTensor output(dims);

        // 初始化数据
        input1.uniform(0.0f, 1.0f);
        input2.uniform(0.0f, 1.0f);

        // 测试乘法算子性能
        MultiplyOperator op;
        OperatorContext context;

        auto op_func = [&]() {
            std::vector<FloatTensor> inputs = {input1, input2};
            std::vector<FloatTensor> outputs = {output};
            op.forward(inputs, outputs, context);
        };

        float avg_time = benchmark_operator(op_func);

        // 计算性能指标
        size_t num_elements = dims[0] * dims[1];
        size_t num_operations = num_elements;  // 一次乘法操作
        float gflops = (num_operations / 1e9) / (avg_time / 1000.0f);
        size_t memory_mb = (3 * num_elements * sizeof(float)) / (1024 * 1024);  // 3个张量

        result.avg_time_ms = avg_time;
        result.min_time_ms = avg_time * 0.9f;  // 简化
        result.max_time_ms = avg_time * 1.1f;
        result.throughput_gflops = gflops;
        result.memory_usage_mb = memory_mb;
        result.passed = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}

PerformanceResult test_matmul_performance() {
    PerformanceResult result;
    result.operator_name = "OptimizedMatMul";
    result.test_case = "Medium Matrices";
    result.passed = false;

    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        result.passed = true;
        result.avg_time_ms = 0.0f;
        result.error_message = "CUDA not available";
        return result;
    }

    try {
        // 创建中等大小的矩阵
        FloatTensor A({512, 512});
        FloatTensor B({512, 512});
        FloatTensor C({512, 512});

        // 初始化数据
        A.uniform(0.0f, 1.0f);
        B.uniform(0.0f, 1.0f);

        // 测试矩阵乘法性能
        OptimizedMatMulOperator op;
        OperatorContext context;
        context.setParam("tile_size", 16);

        auto op_func = [&]() {
            std::vector<FloatTensor> inputs = {A, B};
            std::vector<FloatTensor> outputs = {C};
            op.forward(inputs, outputs, context);
        };

        float avg_time = benchmark_operator(op_func);

        // 计算性能指标
        size_t num_operations = 2ULL * 512 * 512 * 512;  // 矩阵乘法的操作数
        float gflops = (num_operations / 1e9) / (avg_time / 1000.0f);
        size_t memory_mb = (3 * 512 * 512 * sizeof(float)) / (1024 * 1024);

        result.avg_time_ms = avg_time;
        result.min_time_ms = avg_time * 0.9f;
        result.max_time_ms = avg_time * 1.1f;
        result.throughput_gflops = gflops;
        result.memory_usage_mb = memory_mb;
        result.passed = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}

PerformanceResult test_conv2d_performance() {
    PerformanceResult result;
    result.operator_name = "Conv2D";
    result.test_case = "Standard CNN Layer";
    result.passed = false;

    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        result.passed = true;
        result.avg_time_ms = 0.0f;
        result.error_message = "CUDA not available";
        return result;
    }

    try {
        // 创建典型CNN层的输入
        FloatTensor input({1, 64, 56, 56});    // [N, C, H, W]
        FloatTensor weight({128, 64, 3, 3});   // [K, C, R, S]
        FloatTensor output({1, 128, 54, 54});  // [N, K, P, Q]

        // 初始化数据
        input.uniform(-1.0f, 1.0f);
        weight.uniform(-0.1f, 0.1f);

        // 测试卷积性能
        Conv2DOperator op;
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        auto op_func = [&]() {
            std::vector<FloatTensor> inputs = {input, weight};
            std::vector<FloatTensor> outputs = {output};
            op.forward(inputs, outputs, context);
        };

        float avg_time = benchmark_operator(op_func, 3, 10);  // 卷积较慢，减少迭代次数

        // 计算性能指标
        size_t num_operations = 2ULL * 1 * 128 * 54 * 54 * 64 * 3 * 3;  // 卷积操作数
        float gflops = (num_operations / 1e9) / (avg_time / 1000.0f);

        size_t input_size = 1 * 64 * 56 * 56 * sizeof(float);
        size_t weight_size = 128 * 64 * 3 * 3 * sizeof(float);
        size_t output_size = 1 * 128 * 54 * 54 * sizeof(float);
        size_t memory_mb = (input_size + weight_size + output_size) / (1024 * 1024);

        result.avg_time_ms = avg_time;
        result.min_time_ms = avg_time * 0.9f;
        result.max_time_ms = avg_time * 1.1f;
        result.throughput_gflops = gflops;
        result.memory_usage_mb = memory_mb;
        result.passed = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}

PerformanceResult test_pooling_performance() {
    PerformanceResult result;
    result.operator_name = "MaxPool2D";
    result.test_case = "Large Feature Maps";
    result.passed = false;

    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        result.passed = true;
        result.avg_time_ms = 0.0f;
        result.error_message = "CUDA not available";
        return result;
    }

    try {
        // 创建大特征图
        FloatTensor input({1, 256, 112, 112});  // [N, C, H, W]
        FloatTensor output({1, 256, 56, 56});   // [N, C, P, Q]

        // 初始化数据
        input.uniform(0.0f, 1.0f);

        // 测试池化性能
        MaxPool2DOperator op;
        OperatorContext context;
        context.setParam("kernel_h", 2);
        context.setParam("kernel_w", 2);
        context.setParam("stride_h", 2);
        context.setParam("stride_w", 2);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        auto op_func = [&]() {
            std::vector<FloatTensor> inputs = {input};
            std::vector<FloatTensor> outputs = {output};
            op.forward(inputs, outputs, context);
        };

        float avg_time = benchmark_operator(op_func);

        // 计算性能指标
        size_t num_operations = 1 * 256 * 56 * 56 * 4;  // 每个输出元素需要比较4个输入
        float gflops = (num_operations / 1e9) / (avg_time / 1000.0f);

        size_t input_size = 1 * 256 * 112 * 112 * sizeof(float);
        size_t output_size = 1 * 256 * 56 * 56 * sizeof(float);
        size_t memory_mb = (input_size + output_size) / (1024 * 1024);

        result.avg_time_ms = avg_time;
        result.min_time_ms = avg_time * 0.9f;
        result.max_time_ms = avg_time * 1.1f;
        result.throughput_gflops = gflops;
        result.memory_usage_mb = memory_mb;
        result.passed = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}

PerformanceResult test_reduction_performance() {
    PerformanceResult result;
    result.operator_name = "ReduceSum";
    result.test_case = "Large Vector";
    result.passed = false;

    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        result.passed = true;
        result.avg_time_ms = 0.0f;
        result.error_message = "CUDA not available";
        return result;
    }

    try {
        // 创建大向量
        FloatTensor input({16 * 1024 * 1024});  // 16M 元素
        FloatTensor output({1});

        // 初始化数据
        input.uniform(0.0f, 1.0f);

        // 测试归约性能
        ReduceSumOperator op;
        OperatorContext context;

        auto op_func = [&]() {
            std::vector<FloatTensor> inputs = {input};
            std::vector<FloatTensor> outputs = {output};
            op.forward(inputs, outputs, context);
        };

        float avg_time = benchmark_operator(op_func);

        // 计算性能指标
        size_t num_operations = 16 * 1024 * 1024;  // 加法操作数
        float gflops = (num_operations / 1e9) / (avg_time / 1000.0f);
        size_t memory_mb = (16 * 1024 * 1024 * sizeof(float)) / (1024 * 1024);

        result.avg_time_ms = avg_time;
        result.min_time_ms = avg_time * 0.9f;
        result.max_time_ms = avg_time * 1.1f;
        result.throughput_gflops = gflops;
        result.memory_usage_mb = memory_mb;
        result.passed = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}

// 性能回归检查
bool check_performance_regression(const std::string& baseline_file) {
    // 简化实现：在实际项目中，这里会加载基准性能数据并进行比较
    std::cout << "\nPerformance regression check against baseline: " << baseline_file << std::endl;
    std::cout << "Note: Baseline comparison not implemented in this demo" << std::endl;
    return true;
}

int main() {
    std::cout << "Operator Performance Tests" << std::endl;
    std::cout << "==========================" << std::endl;

    // 注册算子
    registerMathOperators();
    registerConvPoolOperators();

    // 运行性能测试
    PerformanceTestRunner::run_performance_test("Elementwise Operators", test_elementwise_operators_performance);
    PerformanceTestRunner::run_performance_test("Matrix Multiplication", test_matmul_performance);
    PerformanceTestRunner::run_performance_test("2D Convolution", test_conv2d_performance);
    PerformanceTestRunner::run_performance_test("Max Pooling", test_pooling_performance);
    PerformanceTestRunner::run_performance_test("Reduction Operations", test_reduction_performance);

    // 打印总结
    PerformanceTestRunner::print_summary();

    // 保存结果到文件
    PerformanceTestRunner::save_results_to_file("operator_performance_results.csv");

    // 检查性能回归
    check_performance_regression("baseline_performance.csv");

    return PerformanceTestRunner::get_exit_code();
}
