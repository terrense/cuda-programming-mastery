#include "conv_pool_operators.h"
#include "base_operator.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cuda_learning::operators;

// Simple test framework
class TestRunner {
public:
    static void run_test(const std::string& test_name, bool (*test_func)()) {
        std::cout << "Running " << test_name << "... ";
        if (test_func()) {
            std::cout << "PASSED" << std::endl;
            s_passed++;
        } else {
            std::cout << "FAILED" << std::endl;
            s_failed++;
        }
        s_total++;
    }

    static void print_summary() {
        std::cout << "\n=== Conv/Pool Operators Test Summary ===" << std::endl;
        std::cout << "Total: " << s_total << std::endl;
        std::cout << "Passed: " << s_passed << std::endl;
        std::cout << "Failed: " << s_failed << std::endl;
    }

    static int get_exit_code() {
        return s_failed > 0 ? 1 : 0;
    }

private:
    static int s_total;
    static int s_passed;
    static int s_failed;
};

int TestRunner::s_total = 0;
int TestRunner::s_passed = 0;
int TestRunner::s_failed = 0;

// CUDA错误检查宏
#define CUDA_CHECK_TEST(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误: " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

// 浮点数比较函数
bool isClose(float a, float b, float tolerance = 1e-4f) {
    return std::abs(a - b) < tolerance;
}

// 测试函数

bool test_conv2d_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据 - 简单的1x1x3x3输入，1x1x2x2卷积核
        FloatTensor input({1, 1, 3, 3});   // [N, C, H, W]
        FloatTensor weight({1, 1, 2, 2});  // [K, C, R, S]
        FloatTensor output({1, 1, 2, 2});  // [N, K, P, Q]

        // 初始化输入数据
        std::vector<float> input_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f
        };
        std::vector<float> weight_data = {
            1.0f, 0.0f,
            0.0f, 1.0f
        };
        input.fromHost(input_data);
        weight.fromHost(weight_data);

        // 创建算子和上下文
        Conv2DOperator op;
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input, weight};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果 - 手动计算期望值
        std::vector<float> result = output.toHost();
        // 卷积计算: 对于每个2x2窗口，应用卷积核[1,0;0,1]
        // 位置(0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
        // 位置(0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
        // 位置(1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
        // 位置(1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
        std::vector<float> expected = {6.0f, 8.0f, 12.0f, 14.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i])) {
                std::cerr << "Conv2D mismatch at " << i << ": got " << result[i]
                         << ", expected " << expected[i] << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_conv2d_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_maxpool2d_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据 - 1x1x4x4输入，2x2池化
        FloatTensor input({1, 1, 4, 4});   // [N, C, H, W]
        FloatTensor output({1, 1, 2, 2});  // [N, C, P, Q]

        // 初始化输入数据
        std::vector<float> input_data = {
            1.0f, 3.0f, 2.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 2.0f, 1.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f
        };
        input.fromHost(input_data);

        // 创建算子和上下文
        MaxPool2DOperator op;
        OperatorContext context;
        context.setParam("kernel_h", 2);
        context.setParam("kernel_w", 2);
        context.setParam("stride_h", 2);
        context.setParam("stride_w", 2);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果 - 手动计算期望值
        std::vector<float> result = output.toHost();
        // 最大池化计算: 对于每个2x2窗口，取最大值
        // 左上窗口: max(1,3,5,6) = 6
        // 右上窗口: max(2,4,7,8) = 8
        // 左下窗口: max(9,2,4,5) = 9
        // 右下窗口: max(1,3,6,7) = 7
        std::vector<float> expected = {6.0f, 8.0f, 9.0f, 7.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i])) {
                std::cerr << "MaxPool2D mismatch at " << i << ": got " << result[i]
                         << ", expected " << expected[i] << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_maxpool2d_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_avgpool2d_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据 - 1x1x4x4输入，2x2池化
        FloatTensor input({1, 1, 4, 4});   // [N, C, H, W]
        FloatTensor output({1, 1, 2, 2});  // [N, C, P, Q]

        // 初始化输入数据
        std::vector<float> input_data = {
            1.0f, 3.0f, 2.0f, 4.0f,
            5.0f, 7.0f, 6.0f, 8.0f,
            9.0f, 1.0f, 2.0f, 4.0f,
            3.0f, 5.0f, 6.0f, 8.0f
        };
        input.fromHost(input_data);

        // 创建算子和上下文
        AvgPool2DOperator op;
        OperatorContext context;
        context.setParam("kernel_h", 2);
        context.setParam("kernel_w", 2);
        context.setParam("stride_h", 2);
        context.setParam("stride_w", 2);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);
        context.setParam("count_include_pad", true);

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果 - 手动计算期望值
        std::vector<float> result = output.toHost();
        // 平均池化计算: 对于每个2x2窗口，取平均值
        // 左上窗口: (1+3+5+7)/4 = 4.0
        // 右上窗口: (2+4+6+8)/4 = 5.0
        // 左下窗口: (9+1+3+5)/4 = 4.5
        // 右下窗口: (2+4+6+8)/4 = 5.0
        std::vector<float> expected = {4.0f, 5.0f, 4.5f, 5.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i])) {
                std::cerr << "AvgPool2D mismatch at " << i << ": got " << result[i]
                         << ", expected " << expected[i] << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_avgpool2d_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_batchnorm2d_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据 - 1x2x2x2输入（2个通道）
        FloatTensor input({1, 2, 2, 2});   // [N, C, H, W]
        FloatTensor gamma({2});            // [C]
        FloatTensor beta({2});             // [C]
        FloatTensor output({1, 2, 2, 2});  // [N, C, H, W]

        // 初始化输入数据
        std::vector<float> input_data = {
            // 通道0
            1.0f, 2.0f,
            3.0f, 4.0f,
            // 通道1
            5.0f, 6.0f,
            7.0f, 8.0f
        };
        std::vector<float> gamma_data = {1.0f, 1.0f};  // 不缩放
        std::vector<float> beta_data = {0.0f, 0.0f};   // 不偏移

        input.fromHost(input_data);
        gamma.fromHost(gamma_data);
        beta.fromHost(beta_data);

        // 创建算子和上下文
        BatchNorm2DOperator op;
        OperatorContext context;
        context.setParam("eps", 1e-5f);

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input, gamma, beta};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果 - 检查输出是否合理（归一化后的值）
        std::vector<float> result = output.toHost();

        // 对于批量归一化，我们主要检查：
        // 1. 输出不包含NaN或无穷大
        // 2. 输出值在合理范围内
        for (size_t i = 0; i < result.size(); ++i) {
            if (std::isnan(result[i]) || std::isinf(result[i])) {
                std::cerr << "BatchNorm2D produced invalid value at " << i
                         << ": " << result[i] << std::endl;
                return false;
            }
            if (std::abs(result[i]) > 10.0f) {  // 合理的范围检查
                std::cerr << "BatchNorm2D produced unreasonable value at " << i
                         << ": " << result[i] << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_batchnorm2d_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_conv2d_input_validation() {
    try {
        Conv2DOperator op;
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        // 测试错误的输入数量
        std::vector<FloatTensor> wrong_inputs = {FloatTensor({1, 1, 3, 3})};
        if (op.validateInputs(wrong_inputs, context)) {
            return false;  // 应该验证失败
        }

        // 测试维度不匹配
        FloatTensor input({1, 2, 3, 3});   // 2个输入通道
        FloatTensor weight({1, 3, 2, 2});  // 3个输入通道
        std::vector<FloatTensor> mismatched_inputs = {input, weight};
        if (op.validateInputs(mismatched_inputs, context)) {
            return false;  // 应该验证失败
        }

        // 测试正确的输入
        FloatTensor correct_input({1, 2, 3, 3});
        FloatTensor correct_weight({1, 2, 2, 2});
        std::vector<FloatTensor> correct_inputs = {correct_input, correct_weight};
        if (!op.validateInputs(correct_inputs, context)) {
            return false;  // 应该验证成功
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_conv2d_input_validation: " << e.what() << std::endl;
        return false;
    }
}

bool test_conv2d_output_shape_inference() {
    try {
        Conv2DOperator op;
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        std::vector<TensorShape> input_shapes = {
            TensorShape({2, 3, 5, 5}),  // [N, C, H, W]
            TensorShape({4, 3, 3, 3})   // [K, C, R, S]
        };

        auto output_shapes = op.inferOutputShapes(input_shapes, context);

        if (output_shapes.size() != 1) {
            return false;
        }

        // 期望输出形状: [2, 4, 3, 3]
        // P = (H + 2*pad_h - R) / stride_h + 1 = (5 + 0 - 3) / 1 + 1 = 3
        // Q = (W + 2*pad_w - S) / stride_w + 1 = (5 + 0 - 3) / 1 + 1 = 3
        TensorShape expected({2, 4, 3, 3});
        if (output_shapes[0] != expected) {
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_conv2d_output_shape_inference: " << e.what() << std::endl;
        return false;
    }
}

bool test_maxpool2d_output_shape_inference() {
    try {
        MaxPool2DOperator op;
        OperatorContext context;
        context.setParam("kernel_h", 2);
        context.setParam("kernel_w", 2);
        context.setParam("stride_h", 2);
        context.setParam("stride_w", 2);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        std::vector<TensorShape> input_shapes = {
            TensorShape({1, 3, 4, 4})  // [N, C, H, W]
        };

        auto output_shapes = op.inferOutputShapes(input_shapes, context);

        if (output_shapes.size() != 1) {
            return false;
        }

        // 期望输出形状: [1, 3, 2, 2]
        // P = (H + 2*pad_h - kernel_h) / stride_h + 1 = (4 + 0 - 2) / 2 + 1 = 2
        // Q = (W + 2*pad_w - kernel_w) / stride_w + 1 = (4 + 0 - 2) / 2 + 1 = 2
        TensorShape expected({1, 3, 2, 2});
        if (output_shapes[0] != expected) {
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_maxpool2d_output_shape_inference: " << e.what() << std::endl;
        return false;
    }
}

bool test_memory_usage_estimation() {
    try {
        Conv2DOperator conv_op;
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 0);
        context.setParam("pad_w", 0);

        std::vector<TensorShape> input_shapes = {
            TensorShape({1, 3, 5, 5}),  // [N, C, H, W]
            TensorShape({2, 3, 3, 3})   // [K, C, R, S]
        };

        size_t memory_usage = conv_op.estimateMemoryUsage(input_shapes, context);

        // 期望输出形状: [1, 2, 3, 3] = 18个float = 72字节
        size_t expected_memory = 18 * sizeof(float);
        if (memory_usage != expected_memory) {
            std::cerr << "Memory usage estimation mismatch: got " << memory_usage
                     << ", expected " << expected_memory << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_memory_usage_estimation: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "Conv/Pool Operators Unit Tests" << std::endl;
    std::cout << "==============================" << std::endl;

    // 注册算子
    registerConvPoolOperators();

    // 运行测试
    TestRunner::run_test("Conv2D Operator", test_conv2d_operator);
    TestRunner::run_test("MaxPool2D Operator", test_maxpool2d_operator);
    TestRunner::run_test("AvgPool2D Operator", test_avgpool2d_operator);
    TestRunner::run_test("BatchNorm2D Operator", test_batchnorm2d_operator);
    TestRunner::run_test("Conv2D Input Validation", test_conv2d_input_validation);
    TestRunner::run_test("Conv2D Output Shape Inference", test_conv2d_output_shape_inference);
    TestRunner::run_test("MaxPool2D Output Shape Inference", test_maxpool2d_output_shape_inference);
    TestRunner::run_test("Memory Usage Estimation", test_memory_usage_estimation);

    // 打印总结
    TestRunner::print_summary();

    return TestRunner::get_exit_code();
}
