#include "math_operators.h"
#include "operators.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace cuda_learning::operators;

// 性能测试辅助函数
template<typename Func>
double measureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // 返回毫秒
}

// 打印张量内容（仅用于小张量）
void printTensor(const FloatTensor& tensor, const std::string& name) {
    if (tensor.size() > 16) {
        std::cout << name << ": [" << tensor.size() << " elements, showing first 8]" << std::endl;
        auto host_data = tensor.toHost();
        for (size_t i = 0; i < std::min(size_t(8), host_data.size()); ++i) {
            std::cout << host_data[i] << " ";
        }
        std::cout << "..." << std::endl;
    } else {
        std::cout << name << ": ";
        auto host_data = tensor.toHost();
        for (float val : host_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// 测试元素级运算
void testElementwiseOperations() {
    std::cout << "\n=== 测试元素级运算算子 ===" << std::endl;

    // 创建测试数据
    FloatTensor a({4, 4});
    FloatTensor b({4, 4});
    a.uniform(1.0f, 5.0f);
    b.uniform(0.5f, 2.0f);

    printTensor(a, "输入A");
    printTensor(b, "输入B");

    OperatorContext context;
    context.setDevice(0);

    // 测试减法
    {
        std::vector<FloatTensor> inputs = {a, b};
        std::vector<FloatTensor> outputs = {FloatTensor(a.shape())};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("subtract", inputs, outputs, context);
        });

        printTensor(outputs[0], "A - B");
        std::cout << "减法执行时间: " << time << " ms" << std::endl;
    }

    // 测试乘法
    {
        std::vector<FloatTensor> inputs = {a, b};
        std::vector<FloatTensor> outputs = {FloatTensor(a.shape())};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("multiply", inputs, outputs, context);
        });

        printTensor(outputs[0], "A * B");
        std::cout << "乘法执行时间: " << time << " ms" << std::endl;
    }

    // 测试除法
    {
        std::vector<FloatTensor> inputs = {a, b};
        std::vector<FloatTensor> outputs = {FloatTensor(a.shape())};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("divide", inputs, outputs, context);
        });

        printTensor(outputs[0], "A / B");
        std::cout << "除法执行时间: " << time << " ms" << std::endl;
    }
}

// 测试激活函数
void testActivationFunctions() {
    std::cout << "\n=== 测试激活函数算子 ===" << std::endl;

    // 创建测试数据
    FloatTensor input({3, 3});
    input.uniform(-2.0f, 2.0f);

    printTensor(input, "输入");

    OperatorContext context;
    context.setDevice(0);

    // 测试Sigmoid
    {
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {FloatTensor(input.shape())};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("sigmoid", inputs, outputs, context);
        });

        printTensor(outputs[0], "Sigmoid输出");
        std::cout << "Sigmoid执行时间: " << time << " ms" << std::endl;
    }

    // 测试Tanh
    {
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {FloatTensor(input.shape())};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("tanh", inputs, outputs, context);
        });

        printTensor(outputs[0], "Tanh输出");
        std::cout << "Tanh执行时间: " << time << " ms" << std::endl;
    }
}

// 测试优化矩阵乘法
void testOptimizedMatMul() {
    std::cout << "\n=== 测试优化矩阵乘法算子 ===" << std::endl;

    // 创建测试矩阵
    int M = 512, K = 256, N = 512;
    FloatTensor A({M, K});
    FloatTensor B({K, N});
    A.uniform(-1.0f, 1.0f);
    B.uniform(-1.0f, 1.0f);

    std::cout << "矩阵A: " << M << "x" << K << std::endl;
    std::cout << "矩阵B: " << K << "x" << N << std::endl;

    OperatorContext context;
    context.setDevice(0);

    // 测试不同分块大小
    std::vector<int> tile_sizes = {16, 32};

    for (int tile_size : tile_sizes) {
        context.setParam("tile_size", tile_size);

        std::vector<FloatTensor> inputs = {A, B};
        std::vector<FloatTensor> outputs = {FloatTensor({M, N})};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("optimized_matmul", inputs, outputs, context);
        });

        std::cout << "分块大小 " << tile_size << "x" << tile_size
                  << " 执行时间: " << time << " ms" << std::endl;

        // 验证结果的一些统计信息
        auto host_result = outputs[0].toHost();
        float sum = 0.0f, max_val = host_result[0], min_val = host_result[0];
        for (float val : host_result) {
            sum += val;
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
        }
        float mean = sum / host_result.size();

        std::cout << "结果统计 - 均值: " << mean << ", 最大值: " << max_val
                  << ", 最小值: " << min_val << std::endl;
    }
}

// 测试归约运算
void testReductionOperations() {
    std::cout << "\n=== 测试归约运算算子 ===" << std::endl;

    // 创建测试数据
    FloatTensor input({1000});
    input.uniform(1.0f, 10.0f);

    std::cout << "输入张量大小: " << input.size() << " 元素" << std::endl;

    OperatorContext context;
    context.setDevice(0);

    // 测试求和
    {
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {FloatTensor({1})};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("reduce_sum", inputs, outputs, context);
        });

        auto result = outputs[0].toHost();
        std::cout << "求和结果: " << result[0] << ", 执行时间: " << time << " ms" << std::endl;
    }

    // 测试最大值
    {
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {FloatTensor({1})};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("reduce_max", inputs, outputs, context);
        });

        auto result = outputs[0].toHost();
        std::cout << "最大值结果: " << result[0] << ", 执行时间: " << time << " ms" << std::endl;
    }

    // 测试平均值
    {
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {FloatTensor({1})};

        auto time = measureTime([&]() {
            OperatorDispatcher::execute("reduce_mean", inputs, outputs, context);
        });

        auto result = outputs[0].toHost();
        std::cout << "平均值结果: " << result[0] << ", 执行时间: " << time << " ms" << std::endl;
    }
}

// 性能对比测试
void performanceBenchmark() {
    std::cout << "\n=== 性能基准测试 ===" << std::endl;

    std::vector<int> sizes = {1024, 4096, 16384};

    for (int size : sizes) {
        std::cout << "\n测试大小: " << size << "x" << size << std::endl;

        FloatTensor A({size, size});
        FloatTensor B({size, size});
        A.uniform(-1.0f, 1.0f);
        B.uniform(-1.0f, 1.0f);

        OperatorContext context;
        context.setDevice(0);

        // 测试元素级乘法性能
        {
            std::vector<FloatTensor> inputs = {A, B};
            std::vector<FloatTensor> outputs = {FloatTensor(A.shape())};

            auto time = measureTime([&]() {
                OperatorDispatcher::execute("multiply", inputs, outputs, context);
            });

            double throughput = (2.0 * size * size) / (time * 1e6); // GFLOPS
            std::cout << "元素级乘法: " << time << " ms, " << throughput << " GFLOPS" << std::endl;
        }

        // 测试矩阵乘法性能（仅对较小的矩阵）
        if (size <= 1024) {
            context.setParam("tile_size", 16);

            std::vector<FloatTensor> inputs = {A, B};
            std::vector<FloatTensor> outputs = {FloatTensor({size, size})};

            auto time = measureTime([&]() {
                OperatorDispatcher::execute("optimized_matmul", inputs, outputs, context);
            });

            double flops = 2.0 * size * size * size; // 矩阵乘法的浮点运算数
            double throughput = flops / (time * 1e6); // GFLOPS
            std::cout << "矩阵乘法: " << time << " ms, " << throughput << " GFLOPS" << std::endl;
        }
    }
}

int main() {
    std::cout << "CUDA基础数学算子演示程序" << std::endl;
    std::cout << "==============================" << std::endl;

    try {
        // 初始化算子系统
        initializeOperatorSystem();
        registerMathOperators();

        // 运行各项测试
        testElementwiseOperations();
        testActivationFunctions();
        testOptimizedMatMul();
        testReductionOperations();
        performanceBenchmark();

        std::cout << "\n所有测试完成！" << std::endl;

        // 清理算子系统
        shutdownOperatorSystem();

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
