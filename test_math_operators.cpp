#include "src/operators/math_operators.h"
#include "src/operators/operators.h"
#include <iostream>
#include <vector>

using namespace cuda_learning::operators;

int main() {
    std::cout << "Testing CUDA Mathematical Operators" << std::endl;
    std::cout << "===================================" << std::endl;

    try {
        // 初始化算子系统
        initializeOperatorSystem();

        // 创建测试数据
        FloatTensor a(std::vector<int>{2, 2});
        FloatTensor b(std::vector<int>{2, 2});

        // 填充测试数据
        std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data_b = {0.5f, 1.0f, 1.5f, 2.0f};

        a.fromHost(data_a);
        b.fromHost(data_b);

        std::cout << "Input A: ";
        auto host_a = a.toHost();
        for (float val : host_a) std::cout << val << " ";
        std::cout << std::endl;

        std::cout << "Input B: ";
        auto host_b = b.toHost();
        for (float val : host_b) std::cout << val << " ";
        std::cout << std::endl;

        OperatorContext context;
        context.setDevice(0);

        // 测试减法
        std::cout << "\nTesting Subtract Operator..." << std::endl;
        {
            std::vector<FloatTensor> inputs = {a, b};
            std::vector<FloatTensor> outputs = {FloatTensor(a.shape())};

            OperatorDispatcher::execute("subtract", inputs, outputs, context);

            std::cout << "A - B: ";
            auto result = outputs[0].toHost();
            for (float val : result) std::cout << val << " ";
            std::cout << std::endl;
        }

        // 测试乘法
        std::cout << "\nTesting Multiply Operator..." << std::endl;
        {
            std::vector<FloatTensor> inputs = {a, b};
            std::vector<FloatTensor> outputs = {FloatTensor(a.shape())};

            OperatorDispatcher::execute("multiply", inputs, outputs, context);

            std::cout << "A * B: ";
            auto result = outputs[0].toHost();
            for (float val : result) std::cout << val << " ";
            std::cout << std::endl;
        }

        // 测试Sigmoid
        std::cout << "\nTesting Sigmoid Operator..." << std::endl;
        {
            std::vector<FloatTensor> inputs = {a};
            std::vector<FloatTensor> outputs = {FloatTensor(a.shape())};

            OperatorDispatcher::execute("sigmoid", inputs, outputs, context);

            std::cout << "Sigmoid(A): ";
            auto result = outputs[0].toHost();
            for (float val : result) std::cout << val << " ";
            std::cout << std::endl;
        }

        // 测试归约求和
        std::cout << "\nTesting Reduce Sum Operator..." << std::endl;
        {
            std::vector<FloatTensor> inputs = {a};
            std::vector<FloatTensor> outputs = {FloatTensor(std::vector<int>{1})};

            OperatorDispatcher::execute("reduce_sum", inputs, outputs, context);

            std::cout << "Sum(A): ";
            auto result = outputs[0].toHost();
            std::cout << result[0] << std::endl;
        }

        std::cout << "\nAll tests completed successfully!" << std::endl;

        // 清理
        shutdownOperatorSystem();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
