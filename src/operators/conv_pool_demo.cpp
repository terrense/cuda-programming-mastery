#include "conv_pool_operators.h"
#include "operators.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace cuda_learning::operators;

void printTensorInfo(const std::string& name, const FloatTensor& tensor) {
    std::cout << name << " shape: [";
    for (int i = 0; i < tensor.shape().ndim(); ++i) {
        std::cout << tensor.shape().dim(i);
        if (i < tensor.shape().ndim() - 1) std::cout << ", ";
    }
    std::cout << "], size: " << tensor.size() << " elements" << std::endl;
}

void testConv2D() {
    std::cout << "\n=== Testing 2D Convolution ===" << std::endl;

    try {
        // 创建输入张量 [N=1, C=3, H=32, W=32]
        FloatTensor input({1, 3, 32, 32});
        input.uniform(-1.0f, 1.0f);

        // 创建卷积核 [K=16, C=3, R=3, S=3]
        FloatTensor weight({16, 3, 3, 3});
        weight.uniform(-0.1f, 0.1f);

        // 设置卷积参数
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 1);
        context.setParam("pad_w", 1);

        // 创建卷积算子
        Conv2DOperator conv_op;

        // 推断输出形状
        auto output_shapes = conv_op.inferOutputShapes({input.shape(), weight.shape()}, context);
        FloatTensor output(output_shapes[0].dims());

        printTensorInfo("Input", input);
        printTensorInfo("Weight", weight);
        printTensorInfo("Output", output);

        // 执行卷积
        auto start = std::chrono::high_resolution_clock::now();
        conv_op.forward({input, weight}, {output}, context);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Conv2D execution time: " << duration.count() << " microseconds" << std::endl;

        // 验证输出
        auto output_host = output.toHost();
        bool has_nan = false;
        float min_val = output_host[0], max_val = output_host[0];

        for (float val : output_host) {
            if (std::isnan(val)) {
                has_nan = true;
                break;
            }
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        std::cout << "Output range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "Has NaN: " << (has_nan ? "Yes" : "No") << std::endl;
        std::cout << "Conv2D test: " << (has_nan ? "FAILED" : "PASSED") << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Conv2D test FAILED: " << e.what() << std::endl;
    }
}

void testMaxPool2D() {
    std::cout << "\n=== Testing Max Pooling 2D ===" << std::endl;

    try {
        // 创建输入张量 [N=1, C=16, H=32, W=32]
        FloatTensor input({1, 16, 32, 32});
        input.uniform(-2.0f, 2.0f);

        // 设置池化参数
        OperatorContext context;
        context.setParam("kernel_h", 2);
        context.setParam("kernel_w", 2);
        context.setParam("stride_h", 2);
        context.setParam("stride_w", 2);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        // 创建池化算子
        MaxPool2DOperator pool_op;

        // 推断输出形状
        auto output_shapes = pool_op.inferOutputShapes({input.shape()}, context);
        FloatTensor output(output_shapes[0].dims());

        printTensorInfo("Input", input);
        printTensorInfo("Output", output);

        // 执行池化
        auto start = std::chrono::high_resolution_clock::now();
        pool_op.forward({input}, {output}, context);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "MaxPool2D execution time: " << duration.count() << " microseconds" << std::endl;

        // 验证输出
        auto input_host = input.toHost();
        auto output_host = output.toHost();

        bool test_passed = true;
        float min_val = output_host[0], max_val = output_host[0];

        for (float val : output_host) {
            if (std::isnan(val)) {
                test_passed = false;
                break;
            }
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        std::cout << "Output range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "MaxPool2D test: " << (test_passed ? "PASSED" : "FAILED") << std::endl;

    } catch (const std::exception& e) {
        std::cout << "MaxPool2D test FAILED: " << e.what() << std::endl;
    }
}

void testAvgPool2D() {
    std::cout << "\n=== Testing Average Pooling 2D ===" << std::endl;

    try {
        // 创建输入张量 [N=1, C=16, H=32, W=32]
        FloatTensor input({1, 16, 32, 32});
        input.uniform(0.0f, 1.0f);

        // 设置池化参数
        OperatorContext context;
        context.setParam("kernel_h", 2);
        context.setParam("kernel_w", 2);
        context.setParam("stride_h", 2);
        context.setParam("stride_w", 2);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);
        context.setParam("count_include_pad", true);

        // 创建池化算子
        AvgPool2DOperator pool_op;

        // 推断输出形状
        auto output_shapes = pool_op.inferOutputShapes({input.shape()}, context);
        FloatTensor output(output_shapes[0].dims());

        printTensorInfo("Input", input);
        printTensorInfo("Output", output);

        // 执行池化
        auto start = std::chrono::high_resolution_clock::now();
        pool_op.forward({input}, {output}, context);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "AvgPool2D execution time: " << duration.count() << " microseconds" << std::endl;

        // 验证输出
        auto output_host = output.toHost();

        bool test_passed = true;
        float min_val = output_host[0], max_val = output_host[0];

        for (float val : output_host) {
            if (std::isnan(val)) {
                test_passed = false;
                break;
            }
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        std::cout << "Output range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "AvgPool2D test: " << (test_passed ? "PASSED" : "FAILED") << std::endl;

    } catch (const std::exception& e) {
        std::cout << "AvgPool2D test FAILED: " << e.what() << std::endl;
    }
}

void testBatchNorm2D() {
    std::cout << "\n=== Testing Batch Normalization 2D ===" << std::endl;

    try {
        // 创建输入张量 [N=2, C=16, H=32, W=32]
        FloatTensor input({2, 16, 32, 32});
        input.normal(0.0f, 1.0f);

        // 创建gamma和beta参数 [C=16]
        FloatTensor gamma({16});
        FloatTensor beta({16});
        gamma.ones();
        beta.zero();

        // 设置批量归一化参数
        OperatorContext context;
        context.setParam("eps", 1e-5f);
        context.setParam("training", true);

        // 创建批量归一化算子
        BatchNorm2DOperator bn_op;

        // 推断输出形状
        auto output_shapes = bn_op.inferOutputShapes({input.shape()}, context);
        FloatTensor output(output_shapes[0].dims());

        printTensorInfo("Input", input);
        printTensorInfo("Gamma", gamma);
        printTensorInfo("Beta", beta);
        printTensorInfo("Output", output);

        // 执行批量归一化
        auto start = std::chrono::high_resolution_clock::now();
        bn_op.forward({input, gamma, beta}, {output}, context);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "BatchNorm2D execution time: " << duration.count() << " microseconds" << std::endl;

        // 验证输出
        auto output_host = output.toHost();

        bool test_passed = true;
        float sum = 0.0f, sum_sq = 0.0f;
        int count = 0;

        for (float val : output_host) {
            if (std::isnan(val)) {
                test_passed = false;
                break;
            }
            sum += val;
            sum_sq += val * val;
            count++;
        }

        float mean = sum / count;
        float var = (sum_sq / count) - (mean * mean);

        std::cout << "Output mean: " << mean << ", variance: " << var << std::endl;
        std::cout << "BatchNorm2D test: " << (test_passed ? "PASSED" : "FAILED") << std::endl;

    } catch (const std::exception& e) {
        std::cout << "BatchNorm2D test FAILED: " << e.what() << std::endl;
    }
}

void testCuDNNBenchmark() {
    std::cout << "\n=== Testing cuDNN Performance Comparison ===" << std::endl;

    try {
        // 测试卷积性能
        std::cout << "\n--- Convolution Benchmark ---" << std::endl;
        TensorShape input_shape({1, 64, 224, 224});
        TensorShape weight_shape({128, 64, 3, 3});

        auto conv_result = CuDNNConvBenchmark::compareConvPerformance(
            input_shape, weight_shape, 1, 1, 1, 1, 10);

        std::cout << "CUDA Conv2D time: " << std::fixed << std::setprecision(3)
                  << conv_result.cuda_time_ms << " ms" << std::endl;

        if (conv_result.cudnn_available) {
            std::cout << "cuDNN Conv2D time: " << conv_result.cudnn_time_ms << " ms" << std::endl;
            std::cout << "Speedup ratio: " << conv_result.speedup_ratio << "x" << std::endl;
        } else {
            std::cout << "cuDNN not available: " << conv_result.error_message << std::endl;
        }

        // 测试池化性能
        std::cout << "\n--- Pooling Benchmark ---" << std::endl;
        TensorShape pool_input_shape({1, 128, 112, 112});

        auto pool_result = CuDNNConvBenchmark::comparePoolingPerformance(
            pool_input_shape, 2, 2, 2, 2, 0, 0, true, 10);

        std::cout << "CUDA MaxPool2D time: " << pool_result.cuda_time_ms << " ms" << std::endl;

        if (pool_result.cudnn_available) {
            std::cout << "cuDNN MaxPool2D time: " << pool_result.cudnn_time_ms << " ms" << std::endl;
            std::cout << "Speedup ratio: " << pool_result.speedup_ratio << "x" << std::endl;
        } else {
            std::cout << "cuDNN not available: " << pool_result.error_message << std::endl;
        }

        // 测试批量归一化性能
        std::cout << "\n--- Batch Normalization Benchmark ---" << std::endl;
        TensorShape bn_input_shape({32, 256, 56, 56});

        auto bn_result = CuDNNConvBenchmark::compareBatchNormPerformance(bn_input_shape, 10);

        std::cout << "CUDA BatchNorm2D time: " << bn_result.cuda_time_ms << " ms" << std::endl;

        if (bn_result.cudnn_available) {
            std::cout << "cuDNN BatchNorm2D time: " << bn_result.cudnn_time_ms << " ms" << std::endl;
            std::cout << "Speedup ratio: " << bn_result.speedup_ratio << "x" << std::endl;
        } else {
            std::cout << "cuDNN not available: " << bn_result.error_message << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Benchmark test FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CUDA Convolution and Pooling Operators Demo" << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        // 初始化算子系统
        initializeOperatorSystem();

        // 注册卷积和池化算子
        registerConvPoolOperators();

        // 运行测试
        testConv2D();
        testMaxPool2D();
        testAvgPool2D();
        testBatchNorm2D();
        testCuDNNBenchmark();

        std::cout << "\n=== Demo Completed ===" << std::endl;

        // 清理算子系统
        shutdownOperatorSystem();

    } catch (const std::exception& e) {
        std::cout << "Demo FAILED: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
