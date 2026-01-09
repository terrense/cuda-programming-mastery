#pragma once

#include "base_operator.h"

namespace cuda_learning {
namespace operators {

// ============================================================================
// 2D卷积算子 (2D Convolution Operator)
// ============================================================================

class Conv2DOperator : public BaseOperator {
public:
    Conv2DOperator() : BaseOperator("conv2d") {}

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

    std::vector<std::string> getRequiredParams() const override {
        return {"stride_h", "stride_w", "pad_h", "pad_w"};
    }

    std::vector<std::string> getOptionalParams() const override {
        return {"dilation_h", "dilation_w", "groups"};
    }

    std::string getDescription() const override {
        return "2D convolution operation with configurable stride, padding, and dilation";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// ============================================================================
// 最大池化算子 (Max Pooling Operator)
// ============================================================================

class MaxPool2DOperator : public BaseOperator {
public:
    MaxPool2DOperator() : BaseOperator("maxpool2d") {}

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

    std::vector<std::string> getRequiredParams() const override {
        return {"kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w"};
    }

    std::string getDescription() const override {
        return "2D max pooling operation with configurable kernel size, stride, and padding";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// ============================================================================
// 平均池化算子 (Average Pooling Operator)
// ============================================================================

class AvgPool2DOperator : public BaseOperator {
public:
    AvgPool2DOperator() : BaseOperator("avgpool2d") {}

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

    std::vector<std::string> getRequiredParams() const override {
        return {"kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w"};
    }

    std::vector<std::string> getOptionalParams() const override {
        return {"count_include_pad"};
    }

    std::string getDescription() const override {
        return "2D average pooling operation with configurable kernel size, stride, and padding";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// ============================================================================
// 批量归一化算子 (Batch Normalization Operator)
// ============================================================================

class BatchNorm2DOperator : public BaseOperator {
public:
    BatchNorm2DOperator() : BaseOperator("batchnorm2d") {}

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

    std::vector<std::string> getOptionalParams() const override {
        return {"eps", "momentum", "training"};
    }

    std::string getDescription() const override {
        return "Batch normalization for 2D feature maps with learnable parameters";
    }

    size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                              const OperatorContext& context) const override;
};

// ============================================================================
// cuDNN性能对比工具 (cuDNN Performance Comparison)
// ============================================================================

class CuDNNConvBenchmark {
public:
    struct BenchmarkResult {
        float cuda_time_ms;
        float cudnn_time_ms;
        float speedup_ratio;
        bool cudnn_available;
        std::string error_message;
    };

    // 对比CUDA实现和cuDNN实现的性能
    static BenchmarkResult compareConvPerformance(
        const TensorShape& input_shape,
        const TensorShape& weight_shape,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int num_iterations = 100);

    // 对比池化操作性能
    static BenchmarkResult comparePoolingPerformance(
        const TensorShape& input_shape,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        bool is_max_pool = true,
        int num_iterations = 100);

    // 对比批量归一化性能
    static BenchmarkResult compareBatchNormPerformance(
        const TensorShape& input_shape,
        int num_iterations = 100);

private:
    static bool initializeCuDNN();
    static void cleanupCuDNN();
    static float measureCudaKernelTime(std::function<void()> kernel_func, int iterations);
};

// ============================================================================
// 注册函数
// ============================================================================

// 注册所有卷积和池化算子
void registerConvPoolOperators();

} // namespace operators
} // namespace cuda_learning
