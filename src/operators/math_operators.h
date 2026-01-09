#pragma once

#include "base_operator.h"

namespace cuda_learning {
namespace operators {

// ============================================================================
// 元素级运算算子 (Element-wise Operations)
// ============================================================================

// 元素级减法算子
class SubtractOperator : public BaseOperator {
public:
    SubtractOperator() : BaseOperator("subtract") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::string getDescription() const override {
        return "Element-wise subtraction: output = input1 - input2";
    }
};

// 元素级乘法算子
class MultiplyOperator : public BaseOperator {
public:
    MultiplyOperator() : BaseOperator("multiply") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::string getDescription() const override {
        return "Element-wise multiplication: output = input1 * input2";
    }
};

// 元素级除法算子
class DivideOperator : public BaseOperator {
public:
    DivideOperator() : BaseOperator("divide") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::string getDescription() const override {
        return "Element-wise division: output = input1 / input2";
    }
};

// Sigmoid激活函数算子
class SigmoidOperator : public BaseOperator {
public:
    SigmoidOperator() : BaseOperator("sigmoid") {}

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
        return "Sigmoid activation function: output = 1 / (1 + exp(-x))";
    }
};

// Tanh激活函数算子
class TanhOperator : public BaseOperator {
public:
    TanhOperator() : BaseOperator("tanh") {}

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
        return "Tanh activation function: output = tanh(x)";
    }
};

// ============================================================================
// 优化矩阵乘法算子 (Optimized Matrix Multiplication)
// ============================================================================

// 优化的矩阵乘法算子（使用共享内存和分块）
class OptimizedMatMulOperator : public BaseOperator {
public:
    OptimizedMatMulOperator() : BaseOperator("optimized_matmul") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::vector<std::string> getOptionalParams() const override {
        return {"tile_size", "use_shared_memory"};
    }

    std::string getDescription() const override {
        return "Optimized matrix multiplication using shared memory and tiling";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// ============================================================================
// 归约运算算子 (Reduction Operations)
// ============================================================================

// 求和归约算子
class ReduceSumOperator : public BaseOperator {
public:
    ReduceSumOperator() : BaseOperator("reduce_sum") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::vector<std::string> getOptionalParams() const override {
        return {"axis", "keepdims"};
    }

    std::string getDescription() const override {
        return "Reduce sum along specified axis or all axes";
    }
};

// 最大值归约算子
class ReduceMaxOperator : public BaseOperator {
public:
    ReduceMaxOperator() : BaseOperator("reduce_max") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::vector<std::string> getOptionalParams() const override {
        return {"axis", "keepdims"};
    }

    std::string getDescription() const override {
        return "Reduce max along specified axis or all axes";
    }
};

// 平均值归约算子
class ReduceMeanOperator : public BaseOperator {
public:
    ReduceMeanOperator() : BaseOperator("reduce_mean") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override;

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override;

    bool validateInputs(const std::vector<FloatTensor>& inputs,
                       const OperatorContext& context) override;

    std::vector<std::string> getOptionalParams() const override {
        return {"axis", "keepdims"};
    }

    std::string getDescription() const override {
        return "Reduce mean along specified axis or all axes";
    }
};

// ============================================================================
// 注册函数
// ============================================================================

// 注册所有基础数学算子
void registerMathOperators();

} // namespace operators
} // namespace cuda_learning
