#pragma once

#include "../../operators/base_operator.h"
#include "../../operators/tensor.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>

namespace cuda_learning {
namespace yolo {

// 激活函数类型
enum class ActivationType {
    NONE,
    RELU,
    LEAKY_RELU,
    SWISH,
    MISH
};

// 融合算子配置
struct FusionConfig {
    // 卷积参数
    int kernel_h = 3;
    int kernel_w = 3;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int groups = 1;

    // 批归一化参数
    float bn_eps = 1e-5f;
    bool bn_training = false;

    // 激活函数参数
    ActivationType activation = ActivationType::RELU;
    float leaky_relu_alpha = 0.1f;

    FusionConfig() = default;
};

// Conv+BN+Activation 融合算子
class ConvBNActivationFusedOperator : public operators::BaseOperator {
private:
    FusionConfig config_;

    // cuDNN 句柄和描述符
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnActivationDescriptor_t activation_desc_;

    // 工作空间
    void* workspace_;
    size_t workspace_size_;

    // 是否已初始化
    bool initialized_;

public:
    ConvBNActivationFusedOperator(const FusionConfig& config = FusionConfig());
    ~ConvBNActivationFusedOperator();

    void forward(const std::vector<operators::FloatTensor>& inputs,
                std::vector<operators::FloatTensor>& outputs,
                const operators::OperatorContext& context) override;

    std::vector<operators::TensorShape> inferOutputShapes(
        const std::vector<operators::TensorShape>& input_shapes,
        const operators::OperatorContext& context) override;

    bool validateInputs(const std::vector<operators::FloatTensor>& inputs,
                       const operators::OperatorContext& context) override;

    std::vector<std::string> getRequiredParams() const override {
        return {"weight", "bn_weight", "bn_bias", "bn_mean", "bn_var"};
    }

    std::string getDescription() const override {
        return "Fused Conv+BatchNorm+Activation operator for optimized inference";
    }

    size_t estimateMemoryUsage(const std::vector<operators::TensorShape>& input_shapes,
                              const operators::OperatorContext& context) const override;

    // 设置融合配置
    void setConfig(const FusionConfig& config);
    const FusionConfig& getConfig() const { return config_; }

private:
    // 初始化cuDNN
    bool initializeCuDNN(const operators::TensorShape& input_shape,
                        const operators::TensorShape& weight_shape);

    // 清理cuDNN资源
    void cleanupCuDNN();

    // 执行融合卷积
    bool executeFusedConvolution(const operators::FloatTensor& input,
                                const operators::FloatTensor& weight,
                                const operators::FloatTensor& bn_weight,
                                const operators::FloatTensor& bn_bias,
                                const operators::FloatTensor& bn_mean,
                                const operators::FloatTensor& bn_var,
                                operators::FloatTensor& output,
                                cudaStream_t stream);

    // 计算输出形状
    operators::TensorShape calculateOutputShape(const operators::TensorShape& input_shape,
                                              const operators::TensorShape& weight_shape) const;

    // 设置激活函数
    void setupActivation();
};

// 算子融合优化器
class OperatorFusionOptimizer {
public:
    struct FusionPattern {
        std::vector<std::string> operator_sequence;
        std::string fused_operator_name;
        std::function<std::unique_ptr<operators::BaseOperator>()> factory;
    };

private:
    std::vector<FusionPattern> fusion_patterns_;

public:
    OperatorFusionOptimizer();

    // 注册融合模式
    void registerFusionPattern(const FusionPattern& pattern);

    // 检测可融合的算子序列
    std::vector<std::pair<size_t, size_t>> detectFusableSequences(
        const std::vector<std::string>& operator_names) const;

    // 应用融合优化
    bool applyFusion(std::vector<std::unique_ptr<operators::BaseOperator>>& operators,
                    const std::vector<std::pair<size_t, size_t>>& fusion_ranges) const;

    // 获取支持的融合模式
    std::vector<std::string> getSupportedPatterns() const;

    // 打印融合统计
    void printFusionStats() const;

private:
    // 检查模式匹配
    bool matchesPattern(const std::vector<std::string>& operator_names,
                       size_t start_idx,
                       const std::vector<std::string>& pattern) const;
};

// CUDA核函数声明
extern "C" {
    // 融合的Conv+BN+ReLU核函数
    void launch_fused_conv_bn_relu_kernel(
        const float* input, const float* weight, const float* bias,
        const float* bn_weight, const float* bn_bias,
        const float* bn_mean, const float* bn_var,
        float* output,
        int batch_size, int in_channels, int out_channels,
        int input_h, int input_w, int output_h, int output_w,
        int kernel_h, int kernel_w, int stride_h, int stride_w,
        int pad_h, int pad_w, float bn_eps,
        cudaStream_t stream);

    // 融合的Conv+BN+LeakyReLU核函数
    void launch_fused_conv_bn_leaky_relu_kernel(
        const float* input, const float* weight, const float* bias,
        const float* bn_weight, const float* bn_bias,
        const float* bn_mean, const float* bn_var,
        float* output,
        int batch_size, int in_channels, int out_channels,
        int input_h, int input_w, int output_h, int output_w,
        int kernel_h, int kernel_w, int stride_h, int stride_w,
        int pad_h, int pad_w, float bn_eps, float leaky_alpha,
        cudaStream_t stream);

    // 融合的Conv+BN+Swish核函数
    void launch_fused_conv_bn_swish_kernel(
        const float* input, const float* weight, const float* bias,
        const float* bn_weight, const float* bn_bias,
        const float* bn_mean, const float* bn_var,
        float* output,
        int batch_size, int in_channels, int out_channels,
        int input_h, int input_w, int output_h, int output_w,
        int kernel_h, int kernel_w, int stride_h, int stride_w,
        int pad_h, int pad_w, float bn_eps,
        cudaStream_t stream);
}

// 性能基准测试
class FusionBenchmark {
public:
    struct BenchmarkResult {
        float unfused_time_ms;
        float fused_time_ms;
        float speedup_ratio;
        float memory_saved_mb;
        bool success;
        std::string error_message;
    };

    // 对比融合前后的性能
    static BenchmarkResult compareFusionPerformance(
        const operators::TensorShape& input_shape,
        const operators::TensorShape& weight_shape,
        const FusionConfig& config,
        int num_iterations = 100);

    // 测试不同激活函数的融合性能
    static std::vector<BenchmarkResult> benchmarkActivationFusions(
        const operators::TensorShape& input_shape,
        const operators::TensorShape& weight_shape,
        int num_iterations = 100);

private:
    // 执行未融合的算子序列
    static float benchmarkUnfusedSequence(
        const operators::FloatTensor& input,
        const operators::FloatTensor& weight,
        const operators::FloatTensor& bn_weight,
        const operators::FloatTensor& bn_bias,
        const operators::FloatTensor& bn_mean,
        const operators::FloatTensor& bn_var,
        const FusionConfig& config,
        int num_iterations);

    // 执行融合算子
    static float benchmarkFusedOperator(
        const operators::FloatTensor& input,
        const operators::FloatTensor& weight,
        const operators::FloatTensor& bn_weight,
        const operators::FloatTensor& bn_bias,
        const operators::FloatTensor& bn_mean,
        const operators::FloatTensor& bn_var,
        const FusionConfig& config,
        int num_iterations);
};

} // namespace yolo
} // namespace cuda_learning
