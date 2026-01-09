# 自定义算子开发框架

## 概述

本文档介绍如何构建一个完整的自定义CUDA算子开发框架，包括算子基础架构、基础数学算子实现、卷积和池化算子开发，以及与cuDNN库的集成。

## 学习目标

完成本模块后，你将能够：
- 设计和实现通用的算子基础架构
- 开发高效的基础数学算子
- 实现卷积和池化等深度学习核心算子
- 集成和使用cuDNN库进行性能对比
- 编写完整的算子测试套件

## 算子基础架构设计

### 通用算子基类

```cpp
// operator_base.h
#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

class Tensor {
public:
    enum DataType { FLOAT32, FLOAT16, INT32, INT8 };

private:
    void* data_;
    std::vector<int> shape_;
    DataType dtype_;
    bool is_gpu_;

public:
    Tensor(const std::vector<int>& shape, DataType dtype, bool gpu = true);
    ~Tensor();

    void* data() const { return data_; }
    const std::vector<int>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    bool is_gpu() const { return is_gpu_; }

    size_t size() const;
    size_t bytes() const;

    void copyFrom(const Tensor& src);
    void copyToHost(void* host_ptr) const;
    void copyFromHost(const void* host_ptr);
};

class OperatorBase {
protected:
    std::string name_;
    cudaStream_t stream_;

public:
    OperatorBase(const std::string& name) : name_(name), stream_(0) {}
    virtual ~OperatorBase() = default;

    virtual void forward(const std::vector<Tensor*>& inputs,
                        std::vector<Tensor*>& outputs) = 0;

    virtual void backward(const std::vector<Tensor*>& grad_outputs,
                         const std::vector<Tensor*>& inputs,
                         std::vector<Tensor*>& grad_inputs) {}

    void setStream(cudaStream_t stream) { stream_ = stream; }
    const std::string& name() const { return name_; }
};
```

### 算子注册系统

```cpp
// operator_registry.h
#pragma once
#include "operator_base.h"
#include <unordered_map>
#include <functional>

class OperatorRegistry {
public:
    using CreateFunc = std::function<std::unique_ptr<OperatorBase>(const std::string&)>;

    static OperatorRegistry& getInstance() {
        static OperatorRegistry instance;
        return instance;
    }

    void registerOperator(const std::string& type, CreateFunc create_func) {
        registry_[type] = create_func;
    }

    std::unique_ptr<OperatorBase> createOperator(const std::string& type,
                                               const std::string& name) {
        auto it = registry_.find(type);
        if (it != registry_.end()) {
            return it->second(name);
        }
        return nullptr;
    }

private:
    std::unordered_map<std::string, CreateFunc> registry_;
};

#define REGISTER_OPERATOR(type, class_name) \
    static bool registered_##class_name = []() { \
        OperatorRegistry::getInstance().registerOperator(#type, \
            [](const std::string& name) -> std::unique_ptr<OperatorBase> { \
                return std::make_unique<class_name>(name); \
            }); \
        return true; \
    }();
```

### 内存管理器

```cpp
// memory_manager.h
#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>

class MemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks_;
    size_t total_allocated_;
    size_t peak_usage_;

public:
    MemoryPool() : total_allocated_(0), peak_usage_(0) {}
    ~MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);
    void clear();

    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getPeakUsage() const { return peak_usage_; }
};

class MemoryManager {
public:
    static MemoryManager& getInstance() {
        static MemoryManager instance;
        return instance;
    }

    void* allocate(size_t size);
    void deallocate(void* ptr);
    void setPoolSize(size_t size);

private:
    std::unique_ptr<MemoryPool> pool_;
};
```

## 基础数学算子实现

### 元素级运算算子

```cpp
// elementwise_ops.cu
#include "operator_base.h"

// 加法算子
template<typename T>
__global__ void addKernel(const T* a, const T* b, T* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

class AddOperator : public OperatorBase {
public:
    AddOperator(const std::string& name) : OperatorBase(name) {}

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input_a = inputs[0];
        auto* input_b = inputs[1];
        auto* output = outputs[0];

        int n = input_a->size();
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        if (input_a->dtype() == Tensor::FLOAT32) {
            addKernel<float><<<gridSize, blockSize, 0, stream_>>>(
                static_cast<const float*>(input_a->data()),
                static_cast<const float*>(input_b->data()),
                static_cast<float*>(output->data()),
                n
            );
        }

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(add, AddOperator);

// 激活函数算子
template<typename T>
__global__ void reluKernel(const T* input, T* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], T(0));
    }
}

template<typename T>
__global__ void sigmoidKernel(const T* input, T* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = T(1) / (T(1) + expf(-input[idx]));
    }
}

class ActivationOperator : public OperatorBase {
public:
    enum Type { RELU, SIGMOID, TANH };

private:
    Type type_;

public:
    ActivationOperator(const std::string& name, Type type)
        : OperatorBase(name), type_(type) {}

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input = inputs[0];
        auto* output = outputs[0];

        int n = input->size();
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        switch (type_) {
            case RELU:
                reluKernel<float><<<gridSize, blockSize, 0, stream_>>>(
                    static_cast<const float*>(input->data()),
                    static_cast<float*>(output->data()),
                    n
                );
                break;
            case SIGMOID:
                sigmoidKernel<float><<<gridSize, blockSize, 0, stream_>>>(
                    static_cast<const float*>(input->data()),
                    static_cast<float*>(output->data()),
                    n
                );
                break;
        }

        cudaStreamSynchronize(stream_);
    }
};
```

### 矩阵乘法算子

```cpp
// matmul_op.cu
#include "operator_base.h"
#include <cublas_v2.h>

class MatMulOperator : public OperatorBase {
private:
    cublasHandle_t cublas_handle_;

public:
    MatMulOperator(const std::string& name) : OperatorBase(name) {
        cublasCreate(&cublas_handle_);
    }

    ~MatMulOperator() {
        cublasDestroy(cublas_handle_);
    }

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input_a = inputs[0];  // [M, K]
        auto* input_b = inputs[1];  // [K, N]
        auto* output = outputs[0];  // [M, N]

        const auto& shape_a = input_a->shape();
        const auto& shape_b = input_b->shape();

        int M = shape_a[0];
        int K = shape_a[1];
        int N = shape_b[1];

        const float alpha = 1.0f, beta = 0.0f;

        cublasSetStream(cublas_handle_, stream_);

        cublasSgemm(cublas_handle_,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K,
                   &alpha,
                   static_cast<const float*>(input_b->data()), N,
                   static_cast<const float*>(input_a->data()), K,
                   &beta,
                   static_cast<float*>(output->data()), N);

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(matmul, MatMulOperator);
```

### 归约运算算子

```cpp
// reduction_ops.cu
#include "operator_base.h"

template<typename T>
__global__ void sumReductionKernel(const T* input, T* output, int n) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (idx < n) ? input[idx] : T(0);
    __syncthreads();

    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

class ReductionOperator : public OperatorBase {
public:
    enum Type { SUM, MAX, MIN, MEAN };

private:
    Type type_;

public:
    ReductionOperator(const std::string& name, Type type)
        : OperatorBase(name), type_(type) {}

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input = inputs[0];
        auto* output = outputs[0];

        int n = input->size();
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        // 初始化输出为0
        cudaMemsetAsync(output->data(), 0, output->bytes(), stream_);

        switch (type_) {
            case SUM:
                sumReductionKernel<float><<<gridSize, blockSize,
                                          blockSize * sizeof(float), stream_>>>(
                    static_cast<const float*>(input->data()),
                    static_cast<float*>(output->data()),
                    n
                );
                break;
        }

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(reduction, ReductionOperator);
```

## 卷积和池化算子

### 2D卷积算子

```cpp
// conv2d_op.cu
#include "operator_base.h"

template<typename T>
__global__ void conv2dKernel(const T* input, const T* weight, T* output,
                            int batch, int in_channels, int out_channels,
                            int in_height, int in_width,
                            int out_height, int out_width,
                            int kernel_h, int kernel_w,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * out_channels * out_height * out_width;

    if (idx < total_output) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c_out = (idx / (out_width * out_height)) % out_channels;
        int n = idx / (out_channels * out_height * out_width);

        T sum = T(0);

        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;

                    if (h_in >= 0 && h_in < in_height &&
                        w_in >= 0 && w_in < in_width) {

                        int input_idx = n * in_channels * in_height * in_width +
                                       c_in * in_height * in_width +
                                       h_in * in_width + w_in;

                        int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                        c_in * kernel_h * kernel_w +
                                        kh * kernel_w + kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        output[idx] = sum;
    }
}

class Conv2dOperator : public OperatorBase {
private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;

public:
    Conv2dOperator(const std::string& name,
                   int kernel_h, int kernel_w,
                   int stride_h = 1, int stride_w = 1,
                   int pad_h = 0, int pad_w = 0)
        : OperatorBase(name), kernel_h_(kernel_h), kernel_w_(kernel_w),
          stride_h_(stride_h), stride_w_(stride_w),
          pad_h_(pad_h), pad_w_(pad_w) {}

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input = inputs[0];   // [N, C, H, W]
        auto* weight = inputs[1];  // [OC, IC, KH, KW]
        auto* output = outputs[0]; // [N, OC, OH, OW]

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int batch = input_shape[0];
        int in_channels = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];

        int out_channels = output_shape[1];
        int out_height = output_shape[2];
        int out_width = output_shape[3];

        int total_output = batch * out_channels * out_height * out_width;
        int blockSize = 256;
        int gridSize = (total_output + blockSize - 1) / blockSize;

        conv2dKernel<float><<<gridSize, blockSize, 0, stream_>>>(
            static_cast<const float*>(input->data()),
            static_cast<const float*>(weight->data()),
            static_cast<float*>(output->data()),
            batch, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_h_, kernel_w_, stride_h_, stride_w_,
            pad_h_, pad_w_
        );

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(conv2d, Conv2dOperator);
```

### 池化算子

```cpp
// pooling_ops.cu
#include "operator_base.h"

template<typename T>
__global__ void maxPool2dKernel(const T* input, T* output,
                               int batch, int channels,
                               int in_height, int in_width,
                               int out_height, int out_width,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * channels * out_height * out_width;

    if (idx < total_output) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int n = idx / (channels * out_height * out_width);

        T max_val = T(-INFINITY);

        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < in_height &&
                    w_in >= 0 && w_in < in_width) {

                    int input_idx = n * channels * in_height * in_width +
                                   c * in_height * in_width +
                                   h_in * in_width + w_in;

                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }

        output[idx] = max_val;
    }
}

class MaxPool2dOperator : public OperatorBase {
private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;

public:
    MaxPool2dOperator(const std::string& name,
                      int kernel_h, int kernel_w,
                      int stride_h = -1, int stride_w = -1,
                      int pad_h = 0, int pad_w = 0)
        : OperatorBase(name), kernel_h_(kernel_h), kernel_w_(kernel_w),
          pad_h_(pad_h), pad_w_(pad_w) {
        stride_h_ = (stride_h == -1) ? kernel_h : stride_h;
        stride_w_ = (stride_w == -1) ? kernel_w : stride_w;
    }

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input = inputs[0];
        auto* output = outputs[0];

        const auto& input_shape = input->shape();
        const auto& output_shape = output->shape();

        int batch = input_shape[0];
        int channels = input_shape[1];
        int in_height = input_shape[2];
        int in_width = input_shape[3];

        int out_height = output_shape[2];
        int out_width = output_shape[3];

        int total_output = batch * channels * out_height * out_width;
        int blockSize = 256;
        int gridSize = (total_output + blockSize - 1) / blockSize;

        maxPool2dKernel<float><<<gridSize, blockSize, 0, stream_>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            batch, channels, in_height, in_width,
            out_height, out_width, kernel_h_, kernel_w_,
            stride_h_, stride_w_, pad_h_, pad_w_
        );

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(maxpool2d, MaxPool2dOperator);
```

### 批量归一化算子

```cpp
// batch_norm_op.cu
#include "operator_base.h"

template<typename T>
__global__ void batchNormKernel(const T* input, T* output,
                               const T* scale, const T* bias,
                               const T* mean, const T* var,
                               int batch, int channels,
                               int height, int width,
                               T epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * channels * height * width;

    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;

        T normalized = (input[idx] - mean[c]) / sqrtf(var[c] + epsilon);
        output[idx] = scale[c] * normalized + bias[c];
    }
}

class BatchNormOperator : public OperatorBase {
private:
    float epsilon_;

public:
    BatchNormOperator(const std::string& name, float epsilon = 1e-5f)
        : OperatorBase(name), epsilon_(epsilon) {}

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input = inputs[0];   // [N, C, H, W]
        auto* scale = inputs[1];   // [C]
        auto* bias = inputs[2];    // [C]
        auto* mean = inputs[3];    // [C]
        auto* var = inputs[4];     // [C]
        auto* output = outputs[0]; // [N, C, H, W]

        const auto& shape = input->shape();
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int total_elements = batch * channels * height * width;
        int blockSize = 256;
        int gridSize = (total_elements + blockSize - 1) / blockSize;

        batchNormKernel<float><<<gridSize, blockSize, 0, stream_>>>(
            static_cast<const float*>(input->data()),
            static_cast<float*>(output->data()),
            static_cast<const float*>(scale->data()),
            static_cast<const float*>(bias->data()),
            static_cast<const float*>(mean->data()),
            static_cast<const float*>(var->data()),
            batch, channels, height, width, epsilon_
        );

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(batch_norm, BatchNormOperator);
```

## cuDNN集成

### cuDNN卷积算子

```cpp
// cudnn_conv2d_op.cu
#include "operator_base.h"
#include <cudnn.h>

class CuDNNConv2dOperator : public OperatorBase {
private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_, bias_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnActivationDescriptor_t activation_desc_;

    int pad_h_, pad_w_;
    int stride_h_, stride_w_;
    int dilation_h_, dilation_w_;

public:
    CuDNNConv2dOperator(const std::string& name,
                        int pad_h = 0, int pad_w = 0,
                        int stride_h = 1, int stride_w = 1,
                        int dilation_h = 1, int dilation_w = 1)
        : OperatorBase(name), pad_h_(pad_h), pad_w_(pad_w),
          stride_h_(stride_h), stride_w_(stride_w),
          dilation_h_(dilation_h), dilation_w_(dilation_w) {

        cudnnCreate(&cudnn_handle_);
        cudnnCreateTensorDescriptor(&input_desc_);
        cudnnCreateTensorDescriptor(&output_desc_);
        cudnnCreateTensorDescriptor(&bias_desc_);
        cudnnCreateFilterDescriptor(&filter_desc_);
        cudnnCreateConvolutionDescriptor(&conv_desc_);
        cudnnCreateActivationDescriptor(&activation_desc_);
    }

    ~CuDNNConv2dOperator() {
        cudnnDestroyTensorDescriptor(input_desc_);
        cudnnDestroyTensorDescriptor(output_desc_);
        cudnnDestroyTensorDescriptor(bias_desc_);
        cudnnDestroyFilterDescriptor(filter_desc_);
        cudnnDestroyConvolutionDescriptor(conv_desc_);
        cudnnDestroyActivationDescriptor(activation_desc_);
        cudnnDestroy(cudnn_handle_);
    }

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs) override {
        auto* input = inputs[0];   // [N, C, H, W]
        auto* weight = inputs[1];  // [OC, IC, KH, KW]
        auto* bias = inputs[2];    // [OC]
        auto* output = outputs[0]; // [N, OC, OH, OW]

        const auto& input_shape = input->shape();
        const auto& weight_shape = weight->shape();
        const auto& output_shape = output->shape();

        // 设置描述符
        cudnnSetTensorNdDescriptor(input_desc_, CUDNN_DATA_FLOAT, 4,
                                  input_shape.data(), nullptr);

        cudnnSetFilterNdDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                  4, weight_shape.data());

        cudnnSetConvolutionNdDescriptor(conv_desc_, 2,
                                       &pad_h_, &stride_h_, &dilation_h_,
                                       CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        cudnnSetTensorNdDescriptor(output_desc_, CUDNN_DATA_FLOAT, 4,
                                  output_shape.data(), nullptr);

        cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_FLOAT, 4,
                                  &output_shape[1], nullptr);

        // 选择最优算法
        cudnnConvolutionFwdAlgo_t algo;
        cudnnGetConvolutionForwardAlgorithm(cudnn_handle_, input_desc_, filter_desc_,
                                           conv_desc_, output_desc_,
                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                           0, &algo);

        // 执行卷积
        const float alpha = 1.0f, beta = 0.0f;
        cudnnSetStream(cudnn_handle_, stream_);

        cudnnConvolutionForward(cudnn_handle_, &alpha,
                               input_desc_, input->data(),
                               filter_desc_, weight->data(),
                               conv_desc_, algo, nullptr, 0,
                               &beta, output_desc_, output->data());

        // 添加偏置
        cudnnAddTensor(cudnn_handle_, &alpha,
                      bias_desc_, bias->data(),
                      &alpha, output_desc_, output->data());

        cudaStreamSynchronize(stream_);
    }
};

REGISTER_OPERATOR(cudnn_conv2d, CuDNNConv2dOperator);
```

## 性能测试和验证

### 算子正确性测试

```cpp
// operator_test.cpp
#include "operator_base.h"
#include "operator_registry.h"
#include <gtest/gtest.h>
#include <random>

class OperatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    void fillRandom(Tensor& tensor, float min_val = -1.0f, float max_val = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);

        std::vector<float> host_data(tensor.size());
        for (size_t i = 0; i < host_data.size(); i++) {
            host_data[i] = dis(gen);
        }

        tensor.copyFromHost(host_data.data());
    }

    bool compareWithTolerance(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        if (a.size() != b.size()) return false;

        std::vector<float> data_a(a.size()), data_b(b.size());
        a.copyToHost(data_a.data());
        b.copyToHost(data_b.data());

        for (size_t i = 0; i < data_a.size(); i++) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(OperatorTest, AddOperatorTest) {
    // 创建算子
    auto add_op = OperatorRegistry::getInstance().createOperator("add", "test_add");
    ASSERT_NE(add_op, nullptr);

    // 创建输入输出张量
    Tensor input_a({1000}, Tensor::FLOAT32);
    Tensor input_b({1000}, Tensor::FLOAT32);
    Tensor output({1000}, Tensor::FLOAT32);

    fillRandom(input_a);
    fillRandom(input_b);

    // 执行算子
    std::vector<Tensor*> inputs = {&input_a, &input_b};
    std::vector<Tensor*> outputs = {&output};
    add_op->forward(inputs, outputs);

    // 验证结果
    std::vector<float> host_a(1000), host_b(1000), host_out(1000);
    input_a.copyToHost(host_a.data());
    input_b.copyToHost(host_b.data());
    output.copyToHost(host_out.data());

    for (int i = 0; i < 1000; i++) {
        EXPECT_NEAR(host_out[i], host_a[i] + host_b[i], 1e-5f);
    }
}

TEST_F(OperatorTest, Conv2dOperatorTest) {
    auto conv_op = OperatorRegistry::getInstance().createOperator("conv2d", "test_conv");
    ASSERT_NE(conv_op, nullptr);

    // 创建输入输出张量
    Tensor input({1, 3, 32, 32}, Tensor::FLOAT32);    // [N, C, H, W]
    Tensor weight({64, 3, 3, 3}, Tensor::FLOAT32);    // [OC, IC, KH, KW]
    Tensor output({1, 64, 30, 30}, Tensor::FLOAT32);  // [N, OC, OH, OW]

    fillRandom(input);
    fillRandom(weight);

    // 执行算子
    std::vector<Tensor*> inputs = {&input, &weight};
    std::vector<Tensor*> outputs = {&output};
    conv_op->forward(inputs, outputs);

    // 验证输出形状
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 30);
    EXPECT_EQ(output.shape()[3], 30);
}
```

### 性能基准测试

```cpp
// benchmark.cpp
#include "operator_base.h"
#include "operator_registry.h"
#include <chrono>
#include <iostream>

class Benchmark {
private:
    cudaEvent_t start_, stop_;

public:
    Benchmark() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~Benchmark() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    template<typename Func>
    float measureTime(Func func, int iterations = 100) {
        // 预热
        for (int i = 0; i < 10; i++) {
            func();
        }
        cudaDeviceSynchronize();

        // 测量
        cudaEventRecord(start_);
        for (int i = 0; i < iterations; i++) {
            func();
        }
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);

        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms / iterations;
    }
};

void benchmarkConvolution() {
    Benchmark bench;

    // 创建算子
    auto custom_conv = OperatorRegistry::getInstance().createOperator("conv2d", "custom");
    auto cudnn_conv = OperatorRegistry::getInstance().createOperator("cudnn_conv2d", "cudnn");

    // 创建测试数据
    Tensor input({32, 128, 56, 56}, Tensor::FLOAT32);
    Tensor weight({256, 128, 3, 3}, Tensor::FLOAT32);
    Tensor bias({256}, Tensor::FLOAT32);
    Tensor output({32, 256, 54, 54}, Tensor::FLOAT32);

    // 填充随机数据
    // ... (省略填充代码)

    // 测试自定义实现
    auto custom_time = bench.measureTime([&]() {
        std::vector<Tensor*> inputs = {&input, &weight};
        std::vector<Tensor*> outputs = {&output};
        custom_conv->forward(inputs, outputs);
    });

    // 测试cuDNN实现
    auto cudnn_time = bench.measureTime([&]() {
        std::vector<Tensor*> inputs = {&input, &weight, &bias};
        std::vector<Tensor*> outputs = {&output};
        cudnn_conv->forward(inputs, outputs);
    });

    std::cout << "Custom Conv2d: " << custom_time << " ms" << std::endl;
    std::cout << "cuDNN Conv2d: " << cudnn_time << " ms" << std::endl;
    std::cout << "Speedup: " << custom_time / cudnn_time << "x" << std::endl;
}
```

## 实践练习

### 练习1：实现Softmax算子
编写Softmax激活函数的CUDA实现，注意数值稳定性。

### 练习2：优化矩阵乘法
使用共享内存和分块技术优化矩阵乘法性能。

### 练习3：实现Group Convolution
实现分组卷积算子，支持不同的分组数。

### 练习4：算子融合
实现Conv+BN+ReLU的融合算子，减少内存访问。

## 总结

本模块介绍了完整的自定义算子开发框架，包括：
- 通用的算子基础架构设计
- 基础数学算子的高效实现
- 深度学习核心算子的开发
- cuDNN库的集成和性能对比
- 完整的测试和验证体系

掌握这些内容后，你将能够开发高性能的自定义CUDA算子，为深度学习模型提供优化的计算支持。

---

**下一步**：学习[YOLO模型GPU加速系统](05-yolo-gpu-acceleration.md)
