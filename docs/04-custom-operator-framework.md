# 第4课：自定义算子开发框架

## 🎯 学习目标

通过本课学习，你将掌握：
- 自定义CUDA算子的设计原理和架构
- 张量数据结构和内存管理技术
- 算子注册和调度系统的实现
- 基础数学算子的CUDA优化实现
- 卷积和池化算子的高效开发

## 📚 理论基础

### 什么是自定义算子？

**自定义算子(Custom Operator)** 是深度学习框架中用户定义的计算单元，它封装了特定的数学运算逻辑，可以在GPU上高效执行。

### 算子框架架构

```
算子框架架构
├── 算子基类 (Operator Base)
│   ├── 前向计算接口
│   ├── 反向计算接口
│   └── 参数管理
├── 张量系统 (Tensor System)
│   ├── 数据存储
│   ├── 形状管理
│   └── 内存分配
├── 调度系统 (Dispatcher)
│   ├── 算子注册
│   ├── 类型推导
│   └── 执行调度
└── 优化引擎 (Optimization Engine)
    ├── 内存优化
    ├── 计算融合
    └── 并行策略
```

## 🛠️ 核心组件实现

### 1. 算子基础架构

#### 通用算子基类
```cpp
class OperatorBase {
public:
    virtual ~OperatorBase() = default;

    // 前向计算
    virtual void forward(const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs) = 0;

    // 反向计算（可选）
    virtual void backward(const std::vector<Tensor>& grad_outputs,
                         std::vector<Tensor>& grad_inputs) {}

    // 形状推导
    virtual std::vector<Shape> inferShape(const std::vector<Shape>& input_shapes) = 0;

    // 内存需求估算
    virtual size_t estimateMemory(const std::vector<Shape>& shapes) = 0;

protected:
    std::string name_;
    std::map<std::string, std::any> attributes_;
};
```

#### 张量数据结构
```cpp
class Tensor {
public:
    Tensor(const Shape& shape, DataType dtype, Device device);
    ~Tensor();

    // 数据访问
    template<typename T>
    T* data() { return static_cast<T*>(data_ptr_); }

    template<typename T>
    const T* data() const { return static_cast<const T*>(data_ptr_); }

    // 形状信息
    const Shape& shape() const { return shape_; }
    size_t size() const { return shape_.size(); }
    size_t bytes() const { return size() * dtype_size(dtype_); }

    // 设备管理
    Device device() const { return device_; }
    void to(Device target_device);

    // 内存管理
    void allocate();
    void deallocate();
    bool is_allocated() const { return data_ptr_ != nullptr; }

private:
    Shape shape_;
    DataType dtype_;
    Device device_;
    void* data_ptr_;
    bool owns_data_;
};
```

### 2. 算子注册和调度系统

#### 算子注册器
```cpp
class OperatorRegistry {
public:
    static OperatorRegistry& getInstance() {
        static OperatorRegistry instance;
        return instance;
    }

    // 注册算子
    template<typename OpType>
    void registerOperator(const std::string& name) {
        operators_[name] = []() -> std::unique_ptr<OperatorBase> {
            return std::make_unique<OpType>();
        };
    }

    // 创建算子实例
    std::unique_ptr<OperatorBase> createOperator(const std::string& name) {
        auto it = operators_.find(name);
        if (it != operators_.end()) {
            return it->second();
        }
        throw std::runtime_error("Unknown operator: " + name);
    }

    // 列出所有注册的算子
    std::vector<std::string> listOperators() const {
        std::vector<std::string> names;
        for (const auto& pair : operators_) {
            names.push_back(pair.first);
        }
        return names;
    }

private:
    std::map<std::string, std::function<std::unique_ptr<OperatorBase>()>> operators_;
};

// 注册宏
#define REGISTER_OPERATOR(name, class_name) \
    static bool _register_##class_name = []() { \
        OperatorRegistry::getInstance().registerOperator<class_name>(#name); \
        return true; \
    }();
```

#### 算子调度器
```cpp
class OperatorDispatcher {
public:
    // 执行算子
    void dispatch(const std::string& op_name,
                 const std::vector<Tensor>& inputs,
                 std::vector<Tensor>& outputs,
                 const std::map<std::string, std::any>& attributes = {}) {

        // 创建算子实例
        auto op = OperatorRegistry::getInstance().createOperator(op_name);

        // 设置属性
        for (const auto& attr : attributes) {
            op->setAttribute(attr.first, attr.second);
        }

        // 形状推导
        std::vector<Shape> input_shapes;
        for (const auto& tensor : inputs) {
            input_shapes.push_back(tensor.shape());
        }

        auto output_shapes = op->inferShape(input_shapes);

        // 分配输出张量
        outputs.resize(output_shapes.size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            outputs[i] = Tensor(output_shapes[i], inputs[0].dtype(), inputs[0].device());
            outputs[i].allocate();
        }

        // 执行计算
        op->forward(inputs, outputs);
    }
};
```

## 🧮 基础数学算子实现

### 1. 元素级运算算子

#### 加法算子
```cpp
// CUDA核函数
__global__ void elementwise_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 算子类实现
class AddOperator : public OperatorBase {
public:
    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs) override {

        const auto& a = inputs[0];
        const auto& b = inputs[1];
        auto& c = outputs[0];

        int n = a.size();
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        elementwise_add_kernel<<<gridSize, blockSize>>>(
            a.data<float>(), b.data<float>(), c.data<float>(), n);

        cudaDeviceSynchronize();
    }

    std::vector<Shape> inferShape(const std::vector<Shape>& input_shapes) override {
        // 广播规则实现
        return {broadcastShapes(input_shapes[0], input_shapes[1])};
    }

    size_t estimateMemory(const std::vector<Shape>& shapes) override {
        return shapes[0].size() * sizeof(float) * 3; // 两个输入 + 一个输出
    }
};

REGISTER_OPERATOR(add, AddOperator);
```

#### 激活函数算子
```cpp
// ReLU激活函数
__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Sigmoid激活函数
__global__ void sigmoid_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

class ActivationOperator : public OperatorBase {
public:
    ActivationOperator(const std::string& activation_type)
        : activation_type_(activation_type) {}

    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs) override {

        const auto& input = inputs[0];
        auto& output = outputs[0];

        int n = input.size();
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        if (activation_type_ == "relu") {
            relu_kernel<<<gridSize, blockSize>>>(
                input.data<float>(), output.data<float>(), n);
        } else if (activation_type_ == "sigmoid") {
            sigmoid_kernel<<<gridSize, blockSize>>>(
                input.data<float>(), output.data<float>(), n);
        }

        cudaDeviceSynchronize();
    }

private:
    std::string activation_type_;
};
```

### 2. 矩阵乘法算子

#### 优化的矩阵乘法实现
```cpp
// 使用共享内存的矩阵乘法核函数
template<int TILE_SIZE>
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载数据到共享内存
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算部分结果
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

class MatMulOperator : public OperatorBase {
public:
    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs) override {

        const auto& A = inputs[0];
        const auto& B = inputs[1];
        auto& C = outputs[0];

        auto shape_a = A.shape();
        auto shape_b = B.shape();

        int M = shape_a.dims[0];
        int K = shape_a.dims[1];
        int N = shape_b.dims[1];

        const int TILE_SIZE = 16;
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

        matmul_shared_kernel<TILE_SIZE><<<gridSize, blockSize>>>(
            A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);

        cudaDeviceSynchronize();
    }

    std::vector<Shape> inferShape(const std::vector<Shape>& input_shapes) override {
        int M = input_shapes[0].dims[0];
        int N = input_shapes[1].dims[1];
        return {Shape({M, N})};
    }
};

REGISTER_OPERATOR(matmul, MatMulOperator);
```

### 3. 归约运算算子

#### 高效的归约实现
```cpp
// 使用共享内存的归约核函数
template<int BLOCK_SIZE>
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 归约操作
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

class ReduceOperator : public OperatorBase {
public:
    ReduceOperator(const std::string& reduce_type) : reduce_type_(reduce_type) {}

    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs) override {

        const auto& input = inputs[0];
        auto& output = outputs[0];

        int n = input.size();
        const int BLOCK_SIZE = 256;
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 临时缓冲区用于多步归约
        Tensor temp({gridSize}, input.dtype(), input.device());
        temp.allocate();

        if (reduce_type_ == "sum") {
            reduce_sum_kernel<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>(
                input.data<float>(), temp.data<float>(), n);

            // 如果需要多步归约
            while (gridSize > 1) {
                int new_gridSize = (gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                reduce_sum_kernel<BLOCK_SIZE><<<new_gridSize, BLOCK_SIZE>>>(
                    temp.data<float>(), temp.data<float>(), gridSize);
                gridSize = new_gridSize;
            }

            // 复制最终结果
            cudaMemcpy(output.data<float>(), temp.data<float>(),
                      sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaDeviceSynchronize();
    }

private:
    std::string reduce_type_;
};

REGISTER_OPERATOR(reduce_sum, ReduceOperator);
```

## 🔄 卷积和池化算子

### 1. 2D卷积算子

#### 基础卷积实现
```cpp
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                             int batch_size, int in_channels, int out_channels,
                             int input_height, int input_width,
                             int kernel_height, int kernel_width,
                             int output_height, int output_width,
                             int stride_h, int stride_w,
                             int pad_h, int pad_w) {

    int batch = blockIdx.z;
    int out_c = blockIdx.y;
    int out_h = blockIdx.x / output_width;
    int out_w = blockIdx.x % output_width;
    int tid = threadIdx.x;

    if (batch >= batch_size || out_c >= out_channels ||
        out_h >= output_height || out_w >= output_width) return;

    float sum = 0.0f;

    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int in_h = out_h * stride_h - pad_h + kh;
                int in_w = out_w * stride_w - pad_w + kw;

                if (in_h >= 0 && in_h < input_height &&
                    in_w >= 0 && in_w < input_width) {

                    int input_idx = batch * (in_channels * input_height * input_width) +
                                   in_c * (input_height * input_width) +
                                   in_h * input_width + in_w;

                    int kernel_idx = out_c * (in_channels * kernel_height * kernel_width) +
                                    in_c * (kernel_height * kernel_width) +
                                    kh * kernel_width + kw;

                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }

    int output_idx = batch * (out_channels * output_height * output_width) +
                    out_c * (output_height * output_width) +
                    out_h * output_width + out_w;

    output[output_idx] = sum;
}

class Conv2DOperator : public OperatorBase {
public:
    Conv2DOperator(int kernel_h, int kernel_w, int stride_h, int stride_w,
                  int pad_h, int pad_w)
        : kernel_h_(kernel_h), kernel_w_(kernel_w),
          stride_h_(stride_h), stride_w_(stride_w),
          pad_h_(pad_h), pad_w_(pad_w) {}

    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs) override {

        const auto& input = inputs[0];
        const auto& kernel = inputs[1];
        auto& output = outputs[0];

        auto input_shape = input.shape();
        auto kernel_shape = kernel.shape();
        auto output_shape = output.shape();

        int batch_size = input_shape.dims[0];
        int in_channels = input_shape.dims[1];
        int input_height = input_shape.dims[2];
        int input_width = input_shape.dims[3];

        int out_channels = kernel_shape.dims[0];
        int output_height = output_shape.dims[2];
        int output_width = output_shape.dims[3];

        dim3 blockSize(256);
        dim3 gridSize(output_height * output_width, out_channels, batch_size);

        conv2d_kernel<<<gridSize, blockSize>>>(
            input.data<float>(), kernel.data<float>(), output.data<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width, kernel_h_, kernel_w_,
            output_height, output_width, stride_h_, stride_w_, pad_h_, pad_w_);

        cudaDeviceSynchronize();
    }

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
};

REGISTER_OPERATOR(conv2d, Conv2DOperator);
```

### 2. 池化算子

#### 最大池化实现
```cpp
__global__ void maxpool2d_kernel(const float* input, float* output,
                                int batch_size, int channels,
                                int input_height, int input_width,
                                int output_height, int output_width,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w) {

    int batch = blockIdx.z;
    int channel = blockIdx.y;
    int out_h = blockIdx.x / output_width;
    int out_w = blockIdx.x % output_width;

    if (batch >= batch_size || channel >= channels ||
        out_h >= output_height || out_w >= output_width) return;

    float max_val = -FLT_MAX;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_h = out_h * stride_h - pad_h + kh;
            int in_w = out_w * stride_w - pad_w + kw;

            if (in_h >= 0 && in_h < input_height &&
                in_w >= 0 && in_w < input_width) {

                int input_idx = batch * (channels * input_height * input_width) +
                               channel * (input_height * input_width) +
                               in_h * input_width + in_w;

                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    int output_idx = batch * (channels * output_height * output_width) +
                    channel * (output_height * output_width) +
                    out_h * output_width + out_w;

    output[output_idx] = max_val;
}

class MaxPool2DOperator : public OperatorBase {
public:
    MaxPool2DOperator(int kernel_h, int kernel_w, int stride_h, int stride_w,
                     int pad_h, int pad_w)
        : kernel_h_(kernel_h), kernel_w_(kernel_w),
          stride_h_(stride_h), stride_w_(stride_w),
          pad_h_(pad_h), pad_w_(pad_w) {}

    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs) override {

        const auto& input = inputs[0];
        auto& output = outputs[0];

        auto input_shape = input.shape();
        auto output_shape = output.shape();

        int batch_size = input_shape.dims[0];
        int channels = input_shape.dims[1];
        int input_height = input_shape.dims[2];
        int input_width = input_shape.dims[3];
        int output_height = output_shape.dims[2];
        int output_width = output_shape.dims[3];

        dim3 blockSize(256);
        dim3 gridSize(output_height * output_width, channels, batch_size);

        maxpool2d_kernel<<<gridSize, blockSize>>>(
            input.data<float>(), output.data<float>(),
            batch_size, channels, input_height, input_width,
            output_height, output_width, kernel_h_, kernel_w_,
            stride_h_, stride_w_, pad_h_, pad_w_);

        cudaDeviceSynchronize();
    }

private:
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
};

REGISTER_OPERATOR(maxpool2d, MaxPool2DOperator);
```

## 🧪 使用示例

### 完整的算子使用流程

```cpp
#include "custom_operator_framework.h"

int main() {
    // 初始化设备
    cudaSetDevice(0);

    // 创建输入张量
    Shape input_shape({1, 3, 224, 224}); // NCHW格式
    Tensor input(input_shape, DataType::FLOAT32, Device::CUDA);
    input.allocate();

    // 创建卷积核
    Shape kernel_shape({64, 3, 3, 3}); // 64个3x3x3的卷积核
    Tensor kernel(kernel_shape, DataType::FLOAT32, Device::CUDA);
    kernel.allocate();

    // 初始化数据（省略具体初始化代码）
    // ...

    // 创建算子调度器
    OperatorDispatcher dispatcher;

    // 执行卷积操作
    std::vector<Tensor> conv_outputs;
    std::map<std::string, std::any> conv_attrs = {
        {"stride_h", 1}, {"stride_w", 1},
        {"pad_h", 1}, {"pad_w", 1}
    };

    dispatcher.dispatch("conv2d", {input, kernel}, conv_outputs, conv_attrs);

    // 执行ReLU激活
    std::vector<Tensor> relu_outputs;
    std::map<std::string, std::any> relu_attrs = {{"activation_type", std::string("relu")}};

    dispatcher.dispatch("activation", conv_outputs, relu_outputs, relu_attrs);

    // 执行最大池化
    std::vector<Tensor> pool_outputs;
    std::map<std::string, std::any> pool_attrs = {
        {"kernel_h", 2}, {"kernel_w", 2},
        {"stride_h", 2}, {"stride_w", 2},
        {"pad_h", 0}, {"pad_w", 0}
    };

    dispatcher.dispatch("maxpool2d", relu_outputs, pool_outputs, pool_attrs);

    std::cout << "算子执行完成!" << std::endl;
    std::cout << "输出形状: " << pool_outputs[0].shape().toString() << std::endl;

    return 0;
}
```

## 🎓 学习检查点

完成本课后，你应该能够：

- [ ] 理解自定义算子框架的整体架构
- [ ] 实现基础的张量数据结构
- [ ] 掌握算子注册和调度机制
- [ ] 编写高效的元素级运算算子
- [ ] 实现优化的矩阵乘法算子
- [ ] 开发卷积和池化算子
- [ ] 使用框架构建完整的计算图

## 🚀 下一步

完成本课学习后，继续学习：
- **第5课**: YOLO模型GPU加速系统
- **第6课**: 高级优化技术模块

## 📝 练习题

1. **实现题**: 编写一个批量归一化(BatchNorm)算子
2. **优化题**: 优化卷积算子的内存访问模式
3. **扩展题**: 添加对半精度(FP16)数据类型的支持

## 🔗 相关代码文件

- `src/operators/operator_base.h/cpp` - 算子基类实现
- `src/operators/tensor.h/cpp` - 张量数据结构
- `src/operators/registry.h/cpp` - 算子注册系统
- `src/operators/math_ops.h/cpp` - 数学算子实现
- `src/operators/conv_ops.h/cpp` - 卷积算子实现
