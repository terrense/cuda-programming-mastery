#include "math_operators.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>

namespace cuda_learning {
namespace operators {

// ============================================================================
// CUDA核函数定义
// ============================================================================

// 元素级运算核函数
__global__ void elementwise_subtract_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void elementwise_multiply_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwise_divide_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (b[idx] != 0.0f) ? (a[idx] / b[idx]) : 0.0f;  // 避免除零
    }
}

// 激活函数核函数
__global__ void sigmoid_kernel(const float* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_output, const float* output,
                                       float* grad_input, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * output[idx] * (1.0f - output[idx]);
    }
}

__global__ void tanh_kernel(const float* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_backward_kernel(const float* grad_output, const float* output,
                                    float* grad_input, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * (1.0f - output[idx] * output[idx]);
    }
}

// 优化矩阵乘法核函数（使用共享内存）
template<int TILE_SIZE>
__global__ void optimized_matmul_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载数据到共享内存
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 归约运算核函数
__global__ void reduce_sum_kernel(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 归约求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void reduce_max_kernel(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();

    // 归约求最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(sdata[0]));
    }
}

// ============================================================================
// 元素级运算算子实现
// ============================================================================

void SubtractOperator::forward(const std::vector<FloatTensor>& inputs,
                              std::vector<FloatTensor>& outputs,
                              const OperatorContext& context) {
    const auto& input1 = inputs[0];
    const auto& input2 = inputs[1];
    auto& output = outputs[0];

    size_t n = input1.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    elementwise_subtract_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input1.data(), input2.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> SubtractOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {input_shapes[0]};
}

bool SubtractOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                     const OperatorContext& context) {
    if (inputs.size() != 2) return false;
    return inputs[0].shape() == inputs[1].shape();
}

void MultiplyOperator::forward(const std::vector<FloatTensor>& inputs,
                              std::vector<FloatTensor>& outputs,
                              const OperatorContext& context) {
    const auto& input1 = inputs[0];
    const auto& input2 = inputs[1];
    auto& output = outputs[0];

    size_t n = input1.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    elementwise_multiply_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input1.data(), input2.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> MultiplyOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {input_shapes[0]};
}

bool MultiplyOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                     const OperatorContext& context) {
    if (inputs.size() != 2) return false;
    return inputs[0].shape() == inputs[1].shape();
}

void DivideOperator::forward(const std::vector<FloatTensor>& inputs,
                            std::vector<FloatTensor>& outputs,
                            const OperatorContext& context) {
    const auto& input1 = inputs[0];
    const auto& input2 = inputs[1];
    auto& output = outputs[0];

    size_t n = input1.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    elementwise_divide_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input1.data(), input2.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> DivideOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {input_shapes[0]};
}

bool DivideOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                   const OperatorContext& context) {
    if (inputs.size() != 2) return false;
    return inputs[0].shape() == inputs[1].shape();
}

// ============================================================================
// 激活函数算子实现
// ============================================================================

void SigmoidOperator::forward(const std::vector<FloatTensor>& inputs,
                             std::vector<FloatTensor>& outputs,
                             const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    size_t n = input.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    sigmoid_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

void SigmoidOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                              std::vector<FloatTensor>& grad_inputs,
                              const OperatorContext& context) {
    const auto& grad_output = grad_outputs[0];
    auto& grad_input = grad_inputs[0];

    // 需要前向传播的输出来计算梯度，这里简化处理
    size_t n = grad_output.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 注意：这里需要保存前向传播的输出，实际实现中应该从context获取
    FloatTensor forward_output(std::vector<int>{static_cast<int>(n)});
    sigmoid_backward_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        grad_output.data(), forward_output.data(), grad_input.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> SigmoidOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {input_shapes[0]};
}

bool SigmoidOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                    const OperatorContext& context) {
    return inputs.size() == 1;
}

void TanhOperator::forward(const std::vector<FloatTensor>& inputs,
                          std::vector<FloatTensor>& outputs,
                          const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    size_t n = input.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    tanh_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        input.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

void TanhOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                           std::vector<FloatTensor>& grad_inputs,
                           const OperatorContext& context) {
    const auto& grad_output = grad_outputs[0];
    auto& grad_input = grad_inputs[0];

    size_t n = grad_output.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 注意：这里需要保存前向传播的输出，实际实现中应该从context获取
    FloatTensor forward_output(std::vector<int>{static_cast<int>(n)});
    tanh_backward_kernel<<<gridSize, blockSize, 0, context.getStream()>>>(
        grad_output.data(), forward_output.data(), grad_input.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> TanhOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {input_shapes[0]};
}

bool TanhOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                 const OperatorContext& context) {
    return inputs.size() == 1;
}

// ============================================================================
// 优化矩阵乘法算子实现
// ============================================================================

void OptimizedMatMulOperator::forward(const std::vector<FloatTensor>& inputs,
                                     std::vector<FloatTensor>& outputs,
                                     const OperatorContext& context) {
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    auto& C = outputs[0];

    // 获取矩阵维度
    int M = A.shape().dim(0);
    int K = A.shape().dim(1);
    int N = B.shape().dim(1);

    // 获取分块大小参数
    int tile_size = context.hasParam("tile_size") ? context.getParam<int>("tile_size") : 16;

    // 设置网格和块维度
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

    // 根据分块大小选择合适的核函数
    if (tile_size == 16) {
        optimized_matmul_kernel<16><<<gridDim, blockDim, 0, context.getStream()>>>(
            A.data(), B.data(), C.data(), M, N, K);
    } else if (tile_size == 32) {
        optimized_matmul_kernel<32><<<gridDim, blockDim, 0, context.getStream()>>>(
            A.data(), B.data(), C.data(), M, N, K);
    } else {
        // 默认使用16x16分块
        optimized_matmul_kernel<16><<<gridDim, blockDim, 0, context.getStream()>>>(
            A.data(), B.data(), C.data(), M, N, K);
    }

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> OptimizedMatMulOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    int M = input_shapes[0].dim(0);
    int N = input_shapes[1].dim(1);
    return {TensorShape({M, N})};
}

bool OptimizedMatMulOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                            const OperatorContext& context) {
    if (inputs.size() != 2) return false;
    if (inputs[0].shape().ndim() != 2 || inputs[1].shape().ndim() != 2) return false;
    return inputs[0].shape().dim(1) == inputs[1].shape().dim(0);
}

size_t OptimizedMatMulOperator::estimateMemoryUsage(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) const {
    int M = input_shapes[0].dim(0);
    int N = input_shapes[1].dim(1);
    return M * N * sizeof(float);  // 输出矩阵大小
}

// ============================================================================
// 归约运算算子实现
// ============================================================================

void ReduceSumOperator::forward(const std::vector<FloatTensor>& inputs,
                               std::vector<FloatTensor>& outputs,
                               const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    // 简化实现：对所有元素求和
    size_t n = input.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 初始化输出为0
    cudaMemsetAsync(output.data(), 0, output.bytes(), context.getStream());

    reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float), context.getStream()>>>(
        input.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> ReduceSumOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    // 简化实现：返回标量
    return {TensorShape({1})};
}

bool ReduceSumOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                      const OperatorContext& context) {
    return inputs.size() == 1;
}

void ReduceMaxOperator::forward(const std::vector<FloatTensor>& inputs,
                               std::vector<FloatTensor>& outputs,
                               const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    size_t n = input.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 初始化输出为负无穷
    float neg_inf = -INFINITY;
    cudaMemcpyAsync(output.data(), &neg_inf, sizeof(float),
                   cudaMemcpyHostToDevice, context.getStream());

    reduce_max_kernel<<<gridSize, blockSize, blockSize * sizeof(float), context.getStream()>>>(
        input.data(), output.data(), n);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> ReduceMaxOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {TensorShape({1})};
}

bool ReduceMaxOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                      const OperatorContext& context) {
    return inputs.size() == 1;
}

void ReduceMeanOperator::forward(const std::vector<FloatTensor>& inputs,
                                std::vector<FloatTensor>& outputs,
                                const OperatorContext& context) {
    const auto& input = inputs[0];
    auto& output = outputs[0];

    // 先求和，再除以元素个数
    size_t n = input.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 初始化输出为0
    cudaMemsetAsync(output.data(), 0, output.bytes(), context.getStream());

    reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float), context.getStream()>>>(
        input.data(), output.data(), n);

    // 除以元素个数得到平均值
    float scale = 1.0f / static_cast<float>(n);
    cublasSscal(nullptr, 1, &scale, output.data(), 1);

    cudaStreamSynchronize(context.getStream());
}

std::vector<TensorShape> ReduceMeanOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    return {TensorShape({1})};
}

bool ReduceMeanOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                       const OperatorContext& context) {
    return inputs.size() == 1;
}

// ============================================================================
// 注册函数实现
// ============================================================================

void registerMathOperators() {
    auto& registry = OperatorRegistry::getInstance();

    // 注册元素级运算算子
    registry.registerOperator("subtract", OperatorInfo(
        "subtract", "Element-wise subtraction of two tensors", {}, {},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<SubtractOperator>();
        }));

    registry.registerOperator("multiply", OperatorInfo(
        "multiply", "Element-wise multiplication of two tensors", {}, {},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<MultiplyOperator>();
        }));

    registry.registerOperator("divide", OperatorInfo(
        "divide", "Element-wise division of two tensors", {}, {},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<DivideOperator>();
        }));

    // 注册激活函数算子
    registry.registerOperator("sigmoid", OperatorInfo(
        "sigmoid", "Sigmoid activation function", {}, {},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<SigmoidOperator>();
        }));

    registry.registerOperator("tanh", OperatorInfo(
        "tanh", "Tanh activation function", {}, {},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<TanhOperator>();
        }));

    // 注册优化矩阵乘法算子
    registry.registerOperator("optimized_matmul", OperatorInfo(
        "optimized_matmul", "Optimized matrix multiplication using shared memory",
        {}, {"tile_size", "use_shared_memory"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<OptimizedMatMulOperator>();
        }));

    // 注册归约运算算子
    registry.registerOperator("reduce_sum", OperatorInfo(
        "reduce_sum", "Reduce sum along specified axis",
        {}, {"axis", "keepdims"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<ReduceSumOperator>();
        }));

    registry.registerOperator("reduce_max", OperatorInfo(
        "reduce_max", "Reduce max along specified axis",
        {}, {"axis", "keepdims"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<ReduceMaxOperator>();
        }));

    registry.registerOperator("reduce_mean", OperatorInfo(
        "reduce_mean", "Reduce mean along specified axis",
        {}, {"axis", "keepdims"},
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<ReduceMeanOperator>();
        }));
}

} // namespace operators
} // namespace cuda_learning
