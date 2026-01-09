#include "conv_pool_operators.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace cuda_learning {
namespace operators {

// ============================================================================
// CUDA核函数定义
// ============================================================================

// 2D卷积核函数 - 朴素实现
__global__ void conv2d_naive_kernel(
    const float* input,     // [N, C, H, W]
    const float* weight,    // [K, C, R, S]
    float* output,          // [N, K, P, Q]
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * K * P * Q;

    if (idx < total_output) {
        // 计算输出位置 (n, k, p, q)
        int n = idx / (K * P * Q);
        int k = (idx / (P * Q)) % K;
        int p = (idx / Q) % P;
        int q = idx % Q;

        float sum = 0.0f;

        // 卷积计算
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    int h_in = p * stride_h - pad_h + r;
                    int w_in = q * stride_w - pad_w + s;

                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                        int weight_idx = ((k * C + c) * R + r) * S + s;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        output[idx] = sum;
    }
}

// 最大池化核函数
__global__ void maxpool2d_kernel(
    const float* input,     // [N, C, H, W]
    float* output,          // [N, C, P, Q]
    int* indices,           // [N, C, P, Q] - 用于反向传播
    int N, int C, int H, int W,
    int P, int Q,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * C * P * Q;

    if (idx < total_output) {
        // 计算输出位置 (n, c, p, q)
        int n = idx / (C * P * Q);
        int c = (idx / (P * Q)) % C;
        int p = (idx / Q) % P;
        int q = idx % Q;

        float max_val = -INFINITY;
        int max_idx = -1;

        // 在池化窗口内寻找最大值
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = p * stride_h - pad_h + kh;
                int w_in = q * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    if (input[input_idx] > max_val) {
                        max_val = input[input_idx];
                        max_idx = input_idx;
                    }
                }
            }
        }

        output[idx] = max_val;
        if (indices) {
            indices[idx] = max_idx;
        }
    }
}

// 平均池化核函数
__global__ void avgpool2d_kernel(
    const float* input,     // [N, C, H, W]
    float* output,          // [N, C, P, Q]
    int N, int C, int H, int W,
    int P, int Q,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    bool count_include_pad) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = N * C * P * Q;

    if (idx < total_output) {
        // 计算输出位置 (n, c, p, q)
        int n = idx / (C * P * Q);
        int c = (idx / (P * Q)) % C;
        int p = (idx / Q) % P;
        int q = idx % Q;

        float sum = 0.0f;
        int count = 0;

        // 在池化窗口内计算平均值
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = p * stride_h - pad_h + kh;
                int w_in = q * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    sum += input[input_idx];
                    count++;
                } else if (count_include_pad) {
                    count++;
                }
            }
        }

        output[idx] = (count > 0) ? (sum / count) : 0.0f;
    }
}

// 批量归一化统计计算核函数
__global__ void batchnorm2d_stats_kernel(
    const float* input,     // [N, C, H, W]
    float* mean,            // [C]
    float* var,             // [C]
    int N, int C, int H, int W) {

    int c = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (c >= C) return;

    __shared__ float sum_data[256];
    __shared__ float sum_sq_data[256];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int elements_per_channel = N * H * W;

    // 每个线程处理多个元素
    for (int i = tid; i < elements_per_channel; i += block_size) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int idx = ((n * C + c) * H * W) + hw;

        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    sum_data[tid] = local_sum;
    sum_sq_data[tid] = local_sum_sq;
    __syncthreads();

    // 归约求和
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_data[tid] += sum_data[tid + s];
            sum_sq_data[tid] += sum_sq_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float channel_mean = sum_data[0] / elements_per_channel;
        float channel_var = (sum_sq_data[0] / elements_per_channel) - (channel_mean * channel_mean);

        mean[c] = channel_mean;
        var[c] = channel_var;
    }
}

// 批量归一化核函数
__global__ void batchnorm2d_kernel(
    const float* input,     // [N, C, H, W]
    const float* gamma,     // [C]
    const float* beta,      // [C]
    const float* mean,      // [C]
    const float* var,       // [C]
    float* output,          // [N, C, H, W]
    int N, int C, int H, int W,
    float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements) {
        // 计算通道索引
        int c = (idx / (H * W)) % C;

        // 归一化计算
        float normalized = (input[idx] - mean[c]) / sqrtf(var[c] + eps);
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// ============================================================================
// Conv2DOperator 实现
// ============================================================================

void Conv2DOperator::forward(const std::vector<FloatTensor>& inputs,
                            std::vector<FloatTensor>& outputs,
                            const OperatorContext& context) {
    const auto& input = inputs[0];   // [N, C, H, W]
    const auto& weight = inputs[1];  // [K, C, R, S]
    auto& output = outputs[0];       // [N, K, P, Q]

    // 获取参数
    int stride_h = context.getParam<int>("stride_h");
    int stride_w = context.getParam<int>("stride_w");
    int pad_h = context.getParam<int>("pad_h");
    int pad_w = context.getParam<int>("pad_w");

    // 获取张量维度
    int N = input.shape().dim(0);
    int C = input.shape().dim(1);
    int H = input.shape().dim(2);
    int W = input.shape().dim(3);

    int K = weight.shape().dim(0);
    int R = weight.shape().dim(2);
    int S = weight.shape().dim(3);

    int P = output.shape().dim(2);
    int Q = output.shape().dim(3);

    // 启动CUDA核函数
    int total_output = N * K * P * Q;
    int blockSize = 256;
    int gridSize = (total_output + blockSize - 1) / blockSize;

    conv2d_naive_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input.data(), weight.data(), output.data(),
        N, C, H, W, K, R, S, P, Q,
        stride_h, stride_w, pad_h, pad_w);

    cudaStreamSynchronize(context.getStream());
}

void Conv2DOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                             std::vector<FloatTensor>& grad_inputs,
                             const OperatorContext& context) {
    // 简化实现：反向传播需要更复杂的实现
    throw std::runtime_error("Conv2D backward pass not implemented in this demo");
}

std::vector<TensorShape> Conv2DOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {

    const auto& input_shape = input_shapes[0];  // [N, C, H, W]
    const auto& weight_shape = input_shapes[1]; // [K, C, R, S]

    int N = input_shape.dim(0);
    int H = input_shape.dim(2);
    int W = input_shape.dim(3);

    int K = weight_shape.dim(0);
    int R = weight_shape.dim(2);
    int S = weight_shape.dim(3);

    int stride_h = context.getParam<int>("stride_h");
    int stride_w = context.getParam<int>("stride_w");
    int pad_h = context.getParam<int>("pad_h");
    int pad_w = context.getParam<int>("pad_w");

    int P = (H + 2 * pad_h - R) / stride_h + 1;
    int Q = (W + 2 * pad_w - S) / stride_w + 1;

    return {TensorShape({N, K, P, Q})};
}

bool Conv2DOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                   const OperatorContext& context) {
    if (inputs.size() != 2) return false;

    const auto& input_shape = inputs[0].shape();
    const auto& weight_shape = inputs[1].shape();

    // 检查维度
    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) return false;

    // 检查通道数匹配
    if (input_shape.dim(1) != weight_shape.dim(1)) return false;

    return true;
}

size_t Conv2DOperator::estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                          const OperatorContext& context) const {
    auto output_shapes = const_cast<Conv2DOperator*>(this)->inferOutputShapes(input_shapes, context);
    return output_shapes[0].numel() * sizeof(float);
}

// ============================================================================
// MaxPool2DOperator 实现
// ============================================================================

void MaxPool2DOperator::forward(const std::vector<FloatTensor>& inputs,
                               std::vector<FloatTensor>& outputs,
                               const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    // 获取参数
    int kernel_h = context.getParam<int>("kernel_h");
    int kernel_w = context.getParam<int>("kernel_w");
    int stride_h = context.getParam<int>("stride_h");
    int stride_w = context.getParam<int>("stride_w");
    int pad_h = context.getParam<int>("pad_h");
    int pad_w = context.getParam<int>("pad_w");

    // 获取张量维度
    int N = input.shape().dim(0);
    int C = input.shape().dim(1);
    int H = input.shape().dim(2);
    int W = input.shape().dim(3);

    int P = output.shape().dim(2);
    int Q = output.shape().dim(3);

    // 启动CUDA核函数
    int total_output = N * C * P * Q;
    int blockSize = 256;
    int gridSize = (total_output + blockSize - 1) / blockSize;

    maxpool2d_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input.data(), output.data(), nullptr,
        N, C, H, W, P, Q,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

    cudaStreamSynchronize(context.getStream());
}

void MaxPool2DOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                                std::vector<FloatTensor>& grad_inputs,
                                const OperatorContext& context) {
    // 简化实现：反向传播需要保存前向传播的索引
    throw std::runtime_error("MaxPool2D backward pass not implemented in this demo");
}

std::vector<TensorShape> MaxPool2DOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {

    const auto& input_shape = input_shapes[0];  // [N, C, H, W]

    int N = input_shape.dim(0);
    int C = input_shape.dim(1);
    int H = input_shape.dim(2);
    int W = input_shape.dim(3);

    int kernel_h = context.getParam<int>("kernel_h");
    int kernel_w = context.getParam<int>("kernel_w");
    int stride_h = context.getParam<int>("stride_h");
    int stride_w = context.getParam<int>("stride_w");
    int pad_h = context.getParam<int>("pad_h");
    int pad_w = context.getParam<int>("pad_w");

    int P = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int Q = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    return {TensorShape({N, C, P, Q})};
}

bool MaxPool2DOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                      const OperatorContext& context) {
    if (inputs.size() != 1) return false;
    return inputs[0].shape().ndim() == 4;
}

size_t MaxPool2DOperator::estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                             const OperatorContext& context) const {
    auto output_shapes = const_cast<MaxPool2DOperator*>(this)->inferOutputShapes(input_shapes, context);
    return output_shapes[0].numel() * sizeof(float);
}

// ============================================================================
// AvgPool2DOperator 实现
// ============================================================================

void AvgPool2DOperator::forward(const std::vector<FloatTensor>& inputs,
                               std::vector<FloatTensor>& outputs,
                               const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    // 获取参数
    int kernel_h = context.getParam<int>("kernel_h");
    int kernel_w = context.getParam<int>("kernel_w");
    int stride_h = context.getParam<int>("stride_h");
    int stride_w = context.getParam<int>("stride_w");
    int pad_h = context.getParam<int>("pad_h");
    int pad_w = context.getParam<int>("pad_w");
    bool count_include_pad = context.hasParam("count_include_pad") ?
                            context.getParam<bool>("count_include_pad") : true;

    // 获取张量维度
    int N = input.shape().dim(0);
    int C = input.shape().dim(1);
    int H = input.shape().dim(2);
    int W = input.shape().dim(3);

    int P = output.shape().dim(2);
    int Q = output.shape().dim(3);

    // 启动CUDA核函数
    int total_output = N * C * P * Q;
    int blockSize = 256;
    int gridSize = (total_output + blockSize - 1) / blockSize;

    avgpool2d_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input.data(), output.data(),
        N, C, H, W, P, Q,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        count_include_pad);

    cudaStreamSynchronize(context.getStream());
}

void AvgPool2DOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                                std::vector<FloatTensor>& grad_inputs,
                                const OperatorContext& context) {
    // 简化实现
    throw std::runtime_error("AvgPool2D backward pass not implemented in this demo");
}

std::vector<TensorShape> AvgPool2DOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {

    const auto& input_shape = input_shapes[0];  // [N, C, H, W]

    int N = input_shape.dim(0);
    int C = input_shape.dim(1);
    int H = input_shape.dim(2);
    int W = input_shape.dim(3);

    int kernel_h = context.getParam<int>("kernel_h");
    int kernel_w = context.getParam<int>("kernel_w");
    int stride_h = context.getParam<int>("stride_h");
    int stride_w = context.getParam<int>("stride_w");
    int pad_h = context.getParam<int>("pad_h");
    int pad_w = context.getParam<int>("pad_w");

    int P = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int Q = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    return {TensorShape({N, C, P, Q})};
}

bool AvgPool2DOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                      const OperatorContext& context) {
    if (inputs.size() != 1) return false;
    return inputs[0].shape().ndim() == 4;
}

size_t AvgPool2DOperator::estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                             const OperatorContext& context) const {
    auto output_shapes = const_cast<AvgPool2DOperator*>(this)->inferOutputShapes(input_shapes, context);
    return output_shapes[0].numel() * sizeof(float);
}

// ============================================================================
// BatchNorm2DOperator 实现
// ============================================================================

void BatchNorm2DOperator::forward(const std::vector<FloatTensor>& inputs,
                                 std::vector<FloatTensor>& outputs,
                                 const OperatorContext& context) {
    const auto& input = inputs[0];    // [N, C, H, W]
    const auto& gamma = inputs[1];    // [C]
    const auto& beta = inputs[2];     // [C]
    auto& output = outputs[0];        // [N, C, H, W]

    // 获取参数
    float eps = context.hasParam("eps") ? context.getParam<float>("eps") : 1e-5f;

    // 获取张量维度
    int N = input.shape().dim(0);
    int C = input.shape().dim(1);
    int H = input.shape().dim(2);
    int W = input.shape().dim(3);

    // 分配临时内存用于统计量
    FloatTensor mean(std::vector<int>{C});
    FloatTensor var(std::vector<int>{C});

    // 计算批次统计量
    dim3 blockDim(256);
    dim3 gridDim(C);

    batchnorm2d_stats_kernel<<<gridDim, blockDim, 0, context.getStream()>>>(
        input.data(), mean.data(), var.data(), N, C, H, W);

    cudaStreamSynchronize(context.getStream());

    // 应用批量归一化
    int total_elements = N * C * H * W;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    batchnorm2d_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input.data(), gamma.data(), beta.data(),
        mean.data(), var.data(), output.data(),
        N, C, H, W, eps);

    cudaStreamSynchronize(context.getStream());
}

void BatchNorm2DOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                                  std::vector<FloatTensor>& grad_inputs,
                                  const OperatorContext& context) {
    // 简化实现
    throw std::runtime_error("BatchNorm2D backward pass not implemented in this demo");
}

std::vector<TensorShape> BatchNorm2DOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {input_shapes[0]};  // 输出形状与输入相同
}

bool BatchNorm2DOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                        const OperatorContext& context) {
    if (inputs.size() != 3) return false;  // input, gamma, beta

    const auto& input_shape = inputs[0].shape();
    const auto& gamma_shape = inputs[1].shape();
    const auto& beta_shape = inputs[2].shape();

    // 检查维度
    if (input_shape.ndim() != 4) return false;
    if (gamma_shape.ndim() != 1 || beta_shape.ndim() != 1) return false;

    // 检查通道数匹配
    int C = input_shape.dim(1);
    if (gamma_shape.dim(0) != C || beta_shape.dim(0) != C) return false;

    return true;
}

size_t BatchNorm2DOperator::estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                               const OperatorContext& context) const {
    auto output_shapes = const_cast<BatchNorm2DOperator*>(this)->inferOutputShapes(input_shapes, context);
    int C = input_shapes[0].dim(1);
    return output_shapes[0].numel() * sizeof(float) + 2 * C * sizeof(float);  // output + mean + var
}

// ============================================================================
// 简化的性能对比实现
// ============================================================================

bool CuDNNConvBenchmark::initializeCuDNN() {
    return false;  // 简化实现，不使用cuDNN
}

void CuDNNConvBenchmark::cleanupCuDNN() {
    // 简化实现，无需清理
}

CuDNNConvBenchmark::BenchmarkResult CuDNNConvBenchmark::compareConvPerformance(
    const TensorShape& input_shape,
    const TensorShape& weight_shape,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int num_iterations) {

    BenchmarkResult result;
    result.cudnn_available = false;
    result.cuda_time_ms = 1.0f;  // 模拟时间
    result.cudnn_time_ms = 0.0f;
    result.speedup_ratio = 1.0f;
    result.error_message = "cuDNN not available in simplified implementation";

    return result;
}

CuDNNConvBenchmark::BenchmarkResult CuDNNConvBenchmark::comparePoolingPerformance(
    const TensorShape& input_shape,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    bool is_max_pool,
    int num_iterations) {

    BenchmarkResult result;
    result.cudnn_available = false;
    result.cuda_time_ms = 0.5f;  // 模拟时间
    result.cudnn_time_ms = 0.0f;
    result.speedup_ratio = 1.0f;
    result.error_message = "cuDNN not available in simplified implementation";

    return result;
}

CuDNNConvBenchmark::BenchmarkResult CuDNNConvBenchmark::compareBatchNormPerformance(
    const TensorShape& input_shape,
    int num_iterations) {

    BenchmarkResult result;
    result.cudnn_available = false;
    result.cuda_time_ms = 0.3f;  // 模拟时间
    result.cudnn_time_ms = 0.0f;
    result.speedup_ratio = 1.0f;
    result.error_message = "cuDNN not available in simplified implementation";

    return result;
}

// ============================================================================
// 注册函数实现
// ============================================================================

void registerConvPoolOperators() {
    auto& registry = OperatorRegistry::getInstance();

    // 注册2D卷积算子
    registry.registerOperator("conv2d", OperatorInfo(
        "conv2d", "2D convolution operation",
        {"stride_h", "stride_w", "pad_h", "pad_w"},
        {"dilation_h", "dilation_w", "groups"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<Conv2DOperator>();
        }));

    // 注册最大池化算子
    registry.registerOperator("maxpool2d", OperatorInfo(
        "maxpool2d", "2D max pooling operation",
        {"kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w"},
        {},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<MaxPool2DOperator>();
        }));

    // 注册平均池化算子
    registry.registerOperator("avgpool2d", OperatorInfo(
        "avgpool2d", "2D average pooling operation",
        {"kernel_h", "kernel_w", "stride_h", "stride_w", "pad_h", "pad_w"},
        {"count_include_pad"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<AvgPool2DOperator>();
        }));

    // 注册批量归一化算子
    registry.registerOperator("batchnorm2d", OperatorInfo(
        "batchnorm2d", "Batch normalization for 2D feature maps",
        {},
        {"eps", "momentum", "training"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<BatchNorm2DOperator>();
        }));
}

} // namespace operators
} // namespace cuda_learning
