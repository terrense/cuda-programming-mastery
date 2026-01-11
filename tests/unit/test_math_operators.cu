#include "math_operators.h"
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
        std::cout << "\n=== Math Operators Test Summary ===" << std::endl;
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
bool isClose(float a, float b, float tolerance = 1e-5f) {
    return std::abs(a - b) < tolerance;
}

// 测试函数

bool test_subtract_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {2, 3};
        FloatTensor input1(dims);
        FloatTensor input2(dims);
        FloatTensor output(dims);

        // 初始化输入数据
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> data2 = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
        input1.fromHost(data1);
        input2.fromHost(data2);

        // 创建算子和上下文
        SubtractOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input1, input2};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        std::vector<float> expected = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i])) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_subtract_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_multiply_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {2, 2};
        FloatTensor input1(dims);
        FloatTensor input2(dims);
        FloatTensor output(dims);

        // 初始化输入数据
        std::vector<float> data1 = {2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> data2 = {1.5f, 2.0f, 2.5f, 3.0f};
        input1.fromHost(data1);
        input2.fromHost(data2);

        // 创建算子和上下文
        MultiplyOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input1, input2};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        std::vector<float> expected = {3.0f, 6.0f, 10.0f, 15.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i])) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_multiply_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_divide_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {2, 2};
        FloatTensor input1(dims);
        FloatTensor input2(dims);
        FloatTensor output(dims);

        // 初始化输入数据
        std::vector<float> data1 = {6.0f, 8.0f, 10.0f, 12.0f};
        std::vector<float> data2 = {2.0f, 4.0f, 5.0f, 3.0f};
        input1.fromHost(data1);
        input2.fromHost(data2);

        // 创建算子和上下文
        DivideOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input1, input2};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        std::vector<float> expected = {3.0f, 2.0f, 2.0f, 4.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i])) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_divide_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_sigmoid_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {4};
        FloatTensor input(dims);
        FloatTensor output(dims);

        // 初始化输入数据
        std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.0f};
        input.fromHost(data);

        // 创建算子和上下文
        SigmoidOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        std::vector<float> expected = {
            0.5f,                           // sigmoid(0)
            1.0f / (1.0f + expf(-1.0f)),   // sigmoid(1)
            1.0f / (1.0f + expf(1.0f)),    // sigmoid(-1)
            1.0f / (1.0f + expf(-2.0f))    // sigmoid(2)
        };

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i], 1e-4f)) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_sigmoid_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_tanh_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {4};
        FloatTensor input(dims);
        FloatTensor output(dims);

        // 初始化输入数据
        std::vector<float> data = {0.0f, 1.0f, -1.0f, 0.5f};
        input.fromHost(data);

        // 创建算子和上下文
        TanhOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        std::vector<float> expected = {
            tanhf(0.0f),   // tanh(0)
            tanhf(1.0f),   // tanh(1)
            tanhf(-1.0f),  // tanh(-1)
            tanhf(0.5f)    // tanh(0.5)
        };

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i], 1e-4f)) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_tanh_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_optimized_matmul_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据 - 小矩阵乘法
        FloatTensor A({2, 3});  // 2x3 矩阵
        FloatTensor B({3, 2});  // 3x2 矩阵
        FloatTensor C({2, 2});  // 2x2 结果矩阵

        // 初始化输入数据
        std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> dataB = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        A.fromHost(dataA);
        B.fromHost(dataB);

        // 创建算子和上下文
        OptimizedMatMulOperator op;
        OperatorContext context;
        context.setParam("tile_size", 16);

        // 执行前向传播
        std::vector<FloatTensor> inputs = {A, B};
        std::vector<FloatTensor> outputs = {C};
        op.forward(inputs, outputs, context);

        // 验证结果 - 手动计算期望值
        std::vector<float> result = C.toHost();
        // A * B = [[1,2,3], [4,5,6]] * [[1,2], [3,4], [5,6]]
        //       = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        //       = [[22, 28], [49, 64]]
        std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};

        for (size_t i = 0; i < expected.size(); ++i) {
            if (!isClose(result[i], expected[i], 1e-3f)) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_optimized_matmul_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_reduce_sum_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {6};
        FloatTensor input(dims);
        FloatTensor output({1});

        // 初始化输入数据
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        input.fromHost(data);

        // 创建算子和上下文
        ReduceSumOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        float expected = 21.0f;  // 1+2+3+4+5+6 = 21

        if (!isClose(result[0], expected, 1e-3f)) {
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_reduce_sum_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_reduce_max_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {6};
        FloatTensor input(dims);
        FloatTensor output({1});

        // 初始化输入数据
        std::vector<float> data = {3.0f, 1.0f, 7.0f, 2.0f, 5.0f, 4.0f};
        input.fromHost(data);

        // 创建算子和上下文
        ReduceMaxOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        float expected = 7.0f;  // max(3,1,7,2,5,4) = 7

        if (!isClose(result[0], expected, 1e-3f)) {
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_reduce_max_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_reduce_mean_operator() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }

    try {
        // 创建测试数据
        std::vector<int> dims = {4};
        FloatTensor input(dims);
        FloatTensor output({1});

        // 初始化输入数据
        std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
        input.fromHost(data);

        // 创建算子和上下文
        ReduceMeanOperator op;
        OperatorContext context;

        // 执行前向传播
        std::vector<FloatTensor> inputs = {input};
        std::vector<FloatTensor> outputs = {output};
        op.forward(inputs, outputs, context);

        // 验证结果
        std::vector<float> result = output.toHost();
        float expected = 5.0f;  // (2+4+6+8)/4 = 5

        if (!isClose(result[0], expected, 1e-3f)) {
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_reduce_mean_operator: " << e.what() << std::endl;
        return false;
    }
}

bool test_input_validation() {
    try {
        // 测试输入验证
        SubtractOperator subtract_op;
        OperatorContext context;

        // 测试错误的输入数量
        std::vector<FloatTensor> wrong_inputs = {FloatTensor({2, 2})};
        if (subtract_op.validateInputs(wrong_inputs, context)) {
            return false;  // 应该验证失败
        }

        // 测试形状不匹配
        FloatTensor input1({2, 3});
        FloatTensor input2({3, 2});
        std::vector<FloatTensor> mismatched_inputs = {input1, input2};
        if (subtract_op.validateInputs(mismatched_inputs, context)) {
            return false;  // 应该验证失败
        }

        // 测试正确的输入
        FloatTensor input3({2, 2});
        FloatTensor input4({2, 2});
        std::vector<FloatTensor> correct_inputs = {input3, input4};
        if (!subtract_op.validateInputs(correct_inputs, context)) {
            return false;  // 应该验证成功
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_input_validation: " << e.what() << std::endl;
        return false;
    }
}

bool test_output_shape_inference() {
    try {
        // 测试输出形状推断
        SubtractOperator subtract_op;
        OperatorContext context;

        std::vector<TensorShape> input_shapes = {
            TensorShape({2, 3, 4}),
            TensorShape({2, 3, 4})
        };

        auto output_shapes = subtract_op.inferOutputShapes(input_shapes, context);

        if (output_shapes.size() != 1) {
            return false;
        }

        if (output_shapes[0] != input_shapes[0]) {
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_output_shape_inference: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "Math Operators Unit Tests" << std::endl;
    std::cout << "=========================" << std::endl;

    // 注册算子
    registerMathOperators();

    // 运行测试
    TestRunner::run_test("Subtract Operator", test_subtract_operator);
    TestRunner::run_test("Multiply Operator", test_multiply_operator);
    TestRunner::run_test("Divide Operator", test_divide_operator);
    TestRunner::run_test("Sigmoid Operator", test_sigmoid_operator);
    TestRunner::run_test("Tanh Operator", test_tanh_operator);
    TestRunner::run_test("Optimized MatMul Operator", test_optimized_matmul_operator);
    TestRunner::run_test("Reduce Sum Operator", test_reduce_sum_operator);
    TestRunner::run_test("Reduce Max Operator", test_reduce_max_operator);
    TestRunner::run_test("Reduce Mean Operator", test_reduce_mean_operator);
    TestRunner::run_test("Input Validation", test_input_validation);
    TestRunner::run_test("Output Shape Inference", test_output_shape_inference);

    // 打印总结
    TestRunner::print_summary();

    return TestRunner::get_exit_code();
}
