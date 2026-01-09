#include "operators.h"
#include "math_operators.h"
#include "../core/error_handler.h"
#include <iostream>
#include <iomanip>

namespace cuda_learning {
namespace operators {

void initializeOperatorSystem() {
    ErrorHandler::logInfo("Initializing CUDA Operator System...");

    try {
        // 注册所有示例算子
        registerExampleOperators();

        // 注册所有基础数学算子
        registerMathOperators();

        // 注册卷积和池化算子
        registerConvPoolOperators();

        // 检查CUDA设备
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            ErrorHandler::logWarning("No CUDA devices found. Some operators may not work.");
        } else {
            ErrorHandler::logInfo("Found " + std::to_string(deviceCount) + " CUDA device(s)");
        }

        ErrorHandler::logInfo("Operator system initialized successfully");

    } catch (const std::exception& e) {
        ErrorHandler::logError("Failed to initialize operator system: " + std::string(e.what()));
        throw;
    }
}

void shutdownOperatorSystem() {
    ErrorHandler::logInfo("Shutting down CUDA Operator System...");

    try {
        // 清空算子注册表
        OperatorRegistry::getInstance().clear();

        // 重置CUDA设备
        cudaDeviceReset();

        ErrorHandler::logInfo("Operator system shutdown complete");

    } catch (const std::exception& e) {
        ErrorHandler::logError("Error during operator system shutdown: " + std::string(e.what()));
    }
}

std::string getOperatorSystemInfo() {
    std::ostringstream info;

    info << "=== CUDA Operator System Information ===\n";

    // CUDA设备信息
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    info << "CUDA Devices: " << deviceCount << "\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        info << "  Device " << i << ": " << prop.name
             << " (Compute " << prop.major << "." << prop.minor << ")\n";
        info << "    Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
        info << "    Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
        info << "    Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    }

    // 注册的算子信息
    auto& registry = OperatorRegistry::getInstance();
    auto operators = registry.getRegisteredOperators();
    info << "\nRegistered Operators: " << operators.size() << "\n";

    for (const auto& op_name : operators) {
        const auto& op_info = registry.getOperatorInfo(op_name);
        info << "  - " << op_name << ": " << op_info.description << "\n";

        if (!op_info.required_params.empty()) {
            info << "    Required params: ";
            for (size_t i = 0; i < op_info.required_params.size(); ++i) {
                if (i > 0) info << ", ";
                info << op_info.required_params[i];
            }
            info << "\n";
        }

        if (!op_info.optional_params.empty()) {
            info << "    Optional params: ";
            for (size_t i = 0; i < op_info.optional_params.size(); ++i) {
                if (i > 0) info << ", ";
                info << op_info.optional_params[i];
            }
            info << "\n";
        }
    }

    return info.str();
}

void listAvailableOperators() {
    std::cout << "\n=== Available CUDA Operators ===\n";

    auto& registry = OperatorRegistry::getInstance();
    auto operators = registry.getRegisteredOperators();

    if (operators.empty()) {
        std::cout << "No operators registered.\n";
        return;
    }

    std::cout << std::left << std::setw(15) << "Name"
              << std::setw(50) << "Description"
              << "Parameters\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& op_name : operators) {
        const auto& op_info = registry.getOperatorInfo(op_name);

        std::cout << std::setw(15) << op_name
                  << std::setw(50) << op_info.description;

        // 显示参数
        std::string params;
        if (!op_info.required_params.empty()) {
            params += "Required: ";
            for (size_t i = 0; i < op_info.required_params.size(); ++i) {
                if (i > 0) params += ", ";
                params += op_info.required_params[i];
            }
        }
        if (!op_info.optional_params.empty()) {
            if (!params.empty()) params += "; ";
            params += "Optional: ";
            for (size_t i = 0; i < op_info.optional_params.size(); ++i) {
                if (i > 0) params += ", ";
                params += op_info.optional_params[i];
            }
        }

        std::cout << params << "\n";
    }

    std::cout << "\nTotal: " << operators.size() << " operators\n\n";
}

bool runOperatorSystemTests() {
    ErrorHandler::logInfo("Running operator system tests...");

    try {
        // 测试1: 张量创建和基本操作
        std::cout << "Test 1: Tensor creation and basic operations...\n";

        FloatTensor tensor1({2, 3}, DeviceType::GPU);
        tensor1.fill(1.0f);

        FloatTensor tensor2({2, 3}, DeviceType::GPU);
        tensor2.fill(2.0f);

        std::cout << "  Created tensors: " << tensor1.toString() << "\n";
        std::cout << "  Created tensors: " << tensor2.toString() << "\n";

        // 测试2: 算子注册和创建
        std::cout << "Test 2: Operator registration and creation...\n";

        auto& registry = OperatorRegistry::getInstance();
        if (!registry.hasOperator("add")) {
            std::cout << "  ERROR: 'add' operator not registered\n";
            return false;
        }

        auto add_op = registry.createOperator("add");
        if (!add_op) {
            std::cout << "  ERROR: Failed to create 'add' operator\n";
            return false;
        }

        std::cout << "  Successfully created operator: " << add_op->getName() << "\n";

        // 测试3: 形状推断
        std::cout << "Test 3: Shape inference...\n";

        OperatorContext context;
        std::vector<TensorShape> input_shapes = {tensor1.shape(), tensor2.shape()};
        auto output_shapes = add_op->inferOutputShapes(input_shapes, context);

        if (output_shapes.size() != 1) {
            std::cout << "  ERROR: Expected 1 output shape, got " << output_shapes.size() << "\n";
            return false;
        }

        std::cout << "  Input shapes: " << input_shapes[0].toString()
                  << ", " << input_shapes[1].toString() << "\n";
        std::cout << "  Output shape: " << output_shapes[0].toString() << "\n";

        // 测试4: 算子执行
        std::cout << "Test 4: Operator execution...\n";

        std::vector<FloatTensor> inputs = {tensor1, tensor2};
        std::vector<FloatTensor> outputs;

        try {
            add_op->forward(inputs, outputs, context);

            if (outputs.empty()) {
                std::cout << "  ERROR: No output produced\n";
                return false;
            }

            std::cout << "  Successfully executed add operator\n";
            std::cout << "  Output tensor: " << outputs[0].toString() << "\n";

            // 验证结果
            auto result_host = outputs[0].toHost();
            bool correct = true;
            for (float val : result_host) {
                if (std::abs(val - 3.0f) > 1e-6) {
                    correct = false;
                    break;
                }
            }

            if (correct) {
                std::cout << "  Result verification: PASSED\n";
            } else {
                std::cout << "  Result verification: FAILED\n";
                return false;
            }

        } catch (const std::exception& e) {
            std::cout << "  ERROR: Operator execution failed: " << e.what() << "\n";
            return false;
        }

        std::cout << "\nAll tests PASSED!\n";
        ErrorHandler::logInfo("Operator system tests completed successfully");
        return true;

    } catch (const std::exception& e) {
        std::cout << "Test FAILED with exception: " << e.what() << "\n";
        ErrorHandler::logError("Operator system tests failed: " + std::string(e.what()));
        return false;
    }
}

} // namespace operators
} // namespace cuda_learning
