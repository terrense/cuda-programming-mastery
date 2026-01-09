#include "example_operators.h"
#include <cublas_v2.h>
#include <cudnn.h>

namespace cuda_learning {
namespace operators {

// CUDA 核函数定义
__global__ void addKernel(const float* a, const float* b, float* c, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void reluKernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void reluBackwardKernel(const float* grad_output, const float* input,
                                  float* grad_input, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

// AddOperator 实现
void AddOperator::forward(const std::vector<FloatTensor>& inputs,
                         std::vector<FloatTensor>& outputs,
                         const OperatorContext& context) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddOperator requires exactly 2 inputs");
    }

    const auto& a = inputs[0];
    const auto& b = inputs[1];

    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Input tensors must have the same shape");
    }

    // 确保输出张量已分配
    if (outputs.empty()) {
        outputs.resize(1);
        outputs[0] = FloatTensor(a.shape(), DeviceType::GPU);
    }

    auto& output = outputs[0];

    // 启动CUDA核函数
    int blockSize = 256;
    int gridSize = (a.size() + blockSize - 1) / blockSize;

    cudaStream_t stream = context.getStream();
    addKernel<<<gridSize, blockSize, 0, stream>>>(
        a.data(), b.data(), output.data(), a.size());

    if (stream == 0) {
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
}

std::vector<TensorShape> AddOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    if (input_shapes.size() != 2) {
        throw std::invalid_argument("AddOperator requires exactly 2 input shapes");
    }

    if (input_shapes[0] != input_shapes[1]) {
        throw std::invalid_argument("Input shapes must be identical for addition");
    }

    return {input_shapes[0]};
}

bool AddOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                const OperatorContext& context) {
    return inputs.size() == 2 &&
           inputs[0].isValid() &&
           inputs[1].isValid() &&
           inputs[0].shape() == inputs[1].shape();
}

// MatMulOperator 实现
void MatMulOperator::forward(const std::vector<FloatTensor>& inputs,
                            std::vector<FloatTensor>& outputs,
                            const OperatorContext& context) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatMulOperator requires exactly 2 inputs");
    }

    const auto& a = inputs[0];
    const auto& b = inputs[1];

    if (a.shape().ndim() != 2 || b.shape().ndim() != 2) {
        throw std::invalid_argument("MatMul inputs must be 2D tensors");
    }

    int m = a.shape().dim(0);
    int k = a.shape().dim(1);
    int n = b.shape().dim(1);

    if (k != b.shape().dim(0)) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    // 确保输出张量已分配
    if (outputs.empty()) {
        outputs.resize(1);
        outputs[0] = FloatTensor({m, n}, DeviceType::GPU);
    }

    auto& output = outputs[0];

    // 使用cuBLAS进行矩阵乘法
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream = context.getStream();
    if (stream != 0) {
        cublasSetStream(handle, stream);
    }

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                b.data(), n,
                a.data(), k,
                &beta,
                output.data(), n);

    cublasDestroy(handle);

    if (stream == 0) {
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
}

std::vector<TensorShape> MatMulOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    if (input_shapes.size() != 2) {
        throw std::invalid_argument("MatMulOperator requires exactly 2 input shapes");
    }

    const auto& shape_a = input_shapes[0];
    const auto& shape_b = input_shapes[1];

    if (shape_a.ndim() != 2 || shape_b.ndim() != 2) {
        throw std::invalid_argument("MatMul inputs must be 2D");
    }

    if (shape_a.dim(1) != shape_b.dim(0)) {
        throw std::invalid_argument("Matrix dimensions incompatible");
    }

    return {TensorShape({shape_a.dim(0), shape_b.dim(1)})};
}

bool MatMulOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                   const OperatorContext& context) {
    if (inputs.size() != 2) return false;

    const auto& a = inputs[0];
    const auto& b = inputs[1];

    return a.isValid() && b.isValid() &&
           a.shape().ndim() == 2 && b.shape().ndim() == 2 &&
           a.shape().dim(1) == b.shape().dim(0);
}

size_t MatMulOperator::estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                          const OperatorContext& context) const {
    if (input_shapes.size() != 2) return 0;

    const auto& shape_a = input_shapes[0];
    const auto& shape_b = input_shapes[1];

    if (shape_a.ndim() != 2 || shape_b.ndim() != 2) return 0;

    size_t output_size = shape_a.dim(0) * shape_b.dim(1);
    return output_size * sizeof(float); // 输出张量的内存需求
}

// ReLUOperator 实现
void ReLUOperator::forward(const std::vector<FloatTensor>& inputs,
                          std::vector<FloatTensor>& outputs,
                          const OperatorContext& context) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReLUOperator requires exactly 1 input");
    }

    const auto& input = inputs[0];

    // 确保输出张量已分配
    if (outputs.empty()) {
        outputs.resize(1);
        outputs[0] = FloatTensor(input.shape(), DeviceType::GPU);
    }

    auto& output = outputs[0];

    // 启动CUDA核函数
    int blockSize = 256;
    int gridSize = (input.size() + blockSize - 1) / blockSize;

    cudaStream_t stream = context.getStream();
    reluKernel<<<gridSize, blockSize, 0, stream>>>(
        input.data(), output.data(), input.size());

    if (stream == 0) {
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
    }
}

void ReLUOperator::backward(const std::vector<FloatTensor>& grad_outputs,
                           std::vector<FloatTensor>& grad_inputs,
                           const OperatorContext& context) {
    if (grad_outputs.size() != 1) {
        throw std::invalid_argument("ReLU backward requires exactly 1 grad_output");
    }

    // 需要原始输入来计算梯度，这里简化处理
    // 实际实现中应该保存前向传播的输入
    throw std::runtime_error("ReLU backward pass requires forward input to be saved");
}

std::vector<TensorShape> ReLUOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    if (input_shapes.size() != 1) {
        throw std::invalid_argument("ReLUOperator requires exactly 1 input shape");
    }

    return {input_shapes[0]}; // 输出形状与输入相同
}

bool ReLUOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                 const OperatorContext& context) {
    return inputs.size() == 1 && inputs[0].isValid();
}

// Conv2DOperator 实现（简化版本）
void Conv2DOperator::forward(const std::vector<FloatTensor>& inputs,
                            std::vector<FloatTensor>& outputs,
                            const OperatorContext& context) {
    // 这里提供一个简化的实现框架
    // 实际的卷积实现会更复杂，通常使用cuDNN
    throw std::runtime_error("Conv2D operator implementation requires cuDNN integration");
}

std::vector<TensorShape> Conv2DOperator::inferOutputShapes(
    const std::vector<TensorShape>& input_shapes,
    const OperatorContext& context) {
    if (input_shapes.size() < 2) {
        throw std::invalid_argument("Conv2D requires at least 2 inputs (input and weight)");
    }

    const auto& input_shape = input_shapes[0];  // [N, C, H, W]
    const auto& weight_shape = input_shapes[1]; // [out_channels, in_channels, kH, kW]

    if (input_shape.ndim() != 4 || weight_shape.ndim() != 4) {
        throw std::invalid_argument("Conv2D inputs must be 4D tensors");
    }

    int kernel_size = context.getParam<int>("kernel_size");
    int stride = context.getParam<int>("stride");
    int padding = context.getParam<int>("padding");

    int N = input_shape.dim(0);
    int out_channels = weight_shape.dim(0);
    int H = input_shape.dim(2);
    int W = input_shape.dim(3);

    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;

    return {TensorShape({N, out_channels, out_H, out_W})};
}

bool Conv2DOperator::validateInputs(const std::vector<FloatTensor>& inputs,
                                   const OperatorContext& context) {
    if (inputs.size() < 2) return false;

    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    return input.isValid() && weight.isValid() &&
           input.shape().ndim() == 4 && weight.shape().ndim() == 4;
}

size_t Conv2DOperator::estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                          const OperatorContext& context) const {
    if (input_shapes.size() < 2) return 0;

    auto output_shapes = const_cast<Conv2DOperator*>(this)->inferOutputShapes(input_shapes, context);
    if (output_shapes.empty()) return 0;

    return output_shapes[0].numel() * sizeof(float);
}

// 注册所有示例算子
void registerExampleOperators() {
    auto& registry = OperatorRegistry::getInstance();

    // 注册AddOperator
    registry.registerOperator("add", OperatorInfo(
        "add",
        "Element-wise addition of two tensors",
        {}, // 无必需参数
        {}, // 无可选参数
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<AddOperator>();
        }
    ));

    // 注册MatMulOperator
    registry.registerOperator("matmul", OperatorInfo(
        "matmul",
        "Matrix multiplication of two 2D tensors",
        {}, // 无必需参数
        {}, // 无可选参数
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<MatMulOperator>();
        }
    ));

    // 注册ReLUOperator
    registry.registerOperator("relu", OperatorInfo(
        "relu",
        "ReLU activation function: max(0, x)",
        {}, // 无必需参数
        {}, // 无可选参数
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<ReLUOperator>();
        }
    ));

    // 注册Conv2DOperator
    registry.registerOperator("conv2d", OperatorInfo(
        "conv2d",
        "2D convolution operation",
        {"kernel_size", "stride", "padding"}, // 必需参数
        {"dilation", "groups"}, // 可选参数
        []() -> std::unique_ptr<BaseOperator> {
            return std::make_unique<Conv2DOperator>();
        }
    ));
}

} // namespace operators
} // namespace cuda_learning
