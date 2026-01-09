#include "src/operators/operators.h"
#include <iostream>

int main() {
    std::cout << "Testing CUDA Operator System...\n";

    try {
        // 初始化系统
        cuda_learning::operators::initializeOperatorSystem();

        // 创建简单张量
        cuda_learning::operators::FloatTensor tensor1({2, 2}, cuda_learning::operators::DeviceType::GPU);
        tensor1.fill(1.0f);

        cuda_learning::operators::FloatTensor tensor2({2, 2}, cuda_learning::operators::DeviceType::GPU);
        tensor2.fill(2.0f);

        std::cout << "Created tensors successfully\n";
        std::cout << "Tensor1: " << tensor1.toString() << "\n";
        std::cout << "Tensor2: " << tensor2.toString() << "\n";

        // 测试算子注册
        auto& registry = cuda_learning::operators::OperatorRegistry::getInstance();
        if (registry.hasOperator("add")) {
            std::cout << "Add operator is registered\n";
        } else {
            std::cout << "Add operator is NOT registered\n";
        }

        // 列出所有算子
        auto operators = registry.getRegisteredOperators();
        std::cout << "Registered operators: ";
        for (const auto& op : operators) {
            std::cout << op << " ";
        }
        std::cout << "\n";

        // 清理系统
        cuda_learning::operators::shutdownOperatorSystem();

        std::cout << "Test completed successfully!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
