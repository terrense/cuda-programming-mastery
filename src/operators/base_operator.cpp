#include "base_operator.h"
#include <iostream>
#include <algorithm>

namespace cuda_learning {
namespace operators {

// OperatorContext 的额外实现（如果需要）

// BaseOperator 的默认实现已在头文件中提供

// OperatorRegistry 的额外实现（如果需要）

// OperatorDispatcher 的额外实现
void OperatorDispatcher::execute(const std::string& op_name,
                               const std::vector<FloatTensor>& inputs,
                               std::vector<FloatTensor>& outputs,
                               const OperatorContext& context) {
    try {
        auto op = OperatorRegistry::getInstance().createOperator(op_name);

        // 验证输入
        if (!op->validateInputs(inputs, context)) {
            throw std::runtime_error("Input validation failed for operator: " + op_name);
        }

        // 验证必需参数
        auto required_params = op->getRequiredParams();
        for (const auto& param : required_params) {
            if (!context.hasParam(param)) {
                throw std::runtime_error("Missing required parameter '" + param +
                                       "' for operator: " + op_name);
            }
        }

        // 设置CUDA设备
        int device_id = context.getDevice();
        if (device_id >= 0) {
            cudaSetDevice(device_id);
        }

        // 执行前向传播
        op->forward(inputs, outputs, context);

        // 同步CUDA流（如果指定了流）
        cudaStream_t stream = context.getStream();
        if (stream != 0) {
            cudaStreamSynchronize(stream);
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to execute operator '" + op_name + "': " + e.what());
    }
}

std::vector<TensorShape> OperatorDispatcher::inferOutputShapes(
    const std::string& op_name,
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    try {
        auto op = OperatorRegistry::getInstance().createOperator(op_name);
        return op->inferOutputShapes(input_shapes, context);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to infer output shapes for operator '" +
                               op_name + "': " + e.what());
    }
}

size_t OperatorDispatcher::estimateMemoryUsage(
    const std::string& op_name,
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    try {
        auto op = OperatorRegistry::getInstance().createOperator(op_name);
        return op->estimateMemoryUsage(input_shapes, context);
    } catch (const std::exception& e) {
        // 内存估计失败不应该抛出异常，返回0即可
        std::cerr << "Warning: Failed to estimate memory usage for operator '"
                  << op_name << "': " << e.what() << std::endl;
        return 0;
    }
}

} // namespace operators
} // namespace cuda_learning
