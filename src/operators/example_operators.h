#pragma once

#include "base_operator.h"

namespace cuda_learning {
namespace operators {

// 示例算子：元素级加法
class AddOperator : public BaseOperator {
public:
    AddOperator() : BaseOperator("add") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::string getDescription() const override {
        return "Element-wise addition of two tensors";
    }
};

// 示例算子：矩阵乘法
class MatMulOperator : public BaseOperator {
public:
    MatMulOperator() : BaseOperator("matmul") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::string getDescription() const override {
        return "Matrix multiplication of two 2D tensors";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// 示例算子：ReLU激活函数
class ReLUOperator : public BaseOperator {
public:
    ReLUOperator() : BaseOperator("relu") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    void backward(const std::vector<FloatTensor>& grad_outputs,
                 std::vector<FloatTensor>& grad_inputs,
                 const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::string getDescription() const override {
        return "ReLU activation function: max(0, x)";
    }
};

// 示例算子：卷积操作
class Conv2DOperator : public BaseOperator {
public:
    Conv2DOperator() : BaseOperator("conv2d") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::vector<std::string> getRequiredParams() const override {
        return {"kernel_size", "stride", "padding"};
    }

    std::vector<std::string> getOptionalParams() const override {
        return {"dilation", "groups"};
    }

    std::string getDescription() const override {
        return "2D convolution operation";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// 注册所有示例算子的函数
void registerExampleOperators();

} // namespace operators
} // namespace cuda_learning
