# CUDA算子基础架构

本模块实现了CUDA编程学习系统的算子基础架构，提供了通用的算子开发框架、张量数据结构和算子注册调度系统。

## 核心组件

### 1. 张量系统 (Tensor System)

#### 特性
- 支持多维张量数据结构
- GPU/CPU内存管理
- 自动内存分配和释放
- 设备间数据传输
- 多种数据类型支持

#### 基本用法
```cpp
#include "tensor.h"

// 创建GPU张量
FloatTensor tensor({2, 3, 4}, DeviceType::GPU);

// 填充数据
tensor.fill(1.0f);
tensor.uniform(0.0f, 1.0f);  // 均匀分布随机数
tensor.normal(0.0f, 1.0f);   // 正态分布随机数

// 形状操作
auto reshaped = tensor.reshape({6, 4});
auto transposed = tensor.transpose(0, 1);

// 设备转换
auto cpu_tensor = tensor.to(DeviceType::CPU);

// 数据访问
auto host_data = tensor.toHost();
tensor.fromHost(host_data);
```

### 2. 算子基类 (Base Operator)

#### 核心接口
```cpp
class BaseOperator {
public:
    // 前向传播（必须实现）
    virtual void forward(const std::vector<FloatTensor>& inputs,
                        std::vector<FloatTensor>& outputs,
                        const OperatorContext& context) = 0;

    // 推断输出形状（必须实现）
    virtual std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) = 0;

    // 反向传播（可选实现）
    virtual void backward(const std::vector<FloatTensor>& grad_outputs,
                         std::vector<FloatTensor>& grad_inputs,
                         const OperatorContext& context);

    // 输入验证（可选实现）
    virtual bool validateInputs(const std::vector<FloatTensor>& inputs,
                               const OperatorContext& context);
};
```

#### 自定义算子示例
```cpp
class MyCustomOperator : public BaseOperator {
public:
    MyCustomOperator() : BaseOperator("my_custom_op") {}

    void forward(const std::vector<FloatTensor>& inputs,
                std::vector<FloatTensor>& outputs,
                const OperatorContext& context) override {
        // 实现前向传播逻辑
        const auto& input = inputs[0];
        auto& output = outputs[0];

        // 启动CUDA核函数
        int blockSize = 256;
        int gridSize = (input.size() + blockSize - 1) / blockSize;
        myKernel<<<gridSize, blockSize>>>(input.data(), output.data(), input.size());
    }

    std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) override {
        return {input_shapes[0]}; // 输出形状与输入相同
    }
};
```

### 3. 算子注册系统 (Operator Registry)

#### 注册算子
```cpp
// 方法1: 使用注册宏
REGISTER_OPERATOR("my_op", MyCustomOperator);

// 方法2: 手动注册
auto& registry = OperatorRegistry::getInstance();
registry.registerOperator("my_op", OperatorInfo(
    "my_op",
    "My custom operator description",
    {"required_param1", "required_param2"}, // 必需参数
    {"optional_param1"},                     // 可选参数
    []() -> std::unique_ptr<BaseOperator> {
        return std::make_unique<MyCustomOperator>();
    }
));
```

#### 使用注册的算子
```cpp
// 创建算子实例
auto op = OperatorRegistry::getInstance().createOperator("my_op");

// 设置执行上下文
OperatorContext context;
context.setParam("required_param1", 42);
context.setParam("optional_param1", 3.14f);
context.setDevice(0);  // 使用GPU 0

// 执行算子
std::vector<FloatTensor> inputs = {input_tensor};
std::vector<FloatTensor> outputs;
op->forward(inputs, outputs, context);
```

### 4. 算子调度器 (Operator Dispatcher)

#### 统一调度接口
```cpp
// 推断输出形状
auto output_shapes = OperatorDispatcher::inferOutputShapes(
    "add", {tensor1.shape(), tensor2.shape()}, context);

// 估计内存使用
size_t memory_usage = OperatorDispatcher::estimateMemoryUsage(
    "matmul", {shape_a, shape_b}, context);

// 执行算子
OperatorDispatcher::execute("add", inputs, outputs, context);
```

## 内置算子

### 1. AddOperator (add)
- **功能**: 元素级张量加法
- **输入**: 2个相同形状的张量
- **输出**: 1个与输入形状相同的张量
- **参数**: 无

### 2. MatMulOperator (matmul)
- **功能**: 矩阵乘法
- **输入**: 2个2D张量 [M,K] 和 [K,N]
- **输出**: 1个2D张量 [M,N]
- **参数**: 无
- **依赖**: cuBLAS

### 3. ReLUOperator (relu)
- **功能**: ReLU激活函数
- **输入**: 1个任意形状的张量
- **输出**: 1个与输入形状相同的张量
- **参数**: 无

### 4. Conv2DOperator (conv2d)
- **功能**: 2D卷积操作
- **输入**: 输入张量[N,C,H,W] 和 权重张量[out_C,in_C,kH,kW]
- **输出**: 输出张量[N,out_C,out_H,out_W]
- **必需参数**: kernel_size, stride, padding
- **可选参数**: dilation, groups
- **依赖**: cuDNN (推荐)

## 编译和使用

### 依赖项
- CUDA Toolkit (>= 11.0)
- cuBLAS
- cuDNN (可选，用于卷积操作)
- CMake (>= 3.18)

### 编译
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行演示
```bash
./operator_demo
```

## 性能优化建议

### 1. 内存管理
- 预分配输出张量避免动态分配
- 使用内存池减少分配开销
- 合理使用CUDA流进行异步操作

### 2. 核函数优化
- 选择合适的线程块大小
- 优化内存访问模式
- 使用共享内存减少全局内存访问

### 3. 算子融合
- 将多个简单算子融合为一个复杂算子
- 减少中间结果的内存读写
- 提高计算密度

## 扩展开发

### 添加新算子
1. 继承 `BaseOperator` 类
2. 实现必需的虚函数
3. 编写CUDA核函数
4. 注册算子到系统
5. 添加单元测试

### 添加新数据类型
1. 在 `DataType` 枚举中添加新类型
2. 更新 `getDataTypeSize()` 函数
3. 为新类型创建张量别名
4. 更新相关算子支持新类型

## 调试和测试

### 启用调试信息
```cpp
ErrorHandler::setLogLevel(LogLevel::DEBUG);
ErrorHandler::enableConsoleOutput(true);
```

### 运行测试套件
```cpp
bool success = runOperatorSystemTests();
```

### 性能分析
- 使用CUDA事件测量执行时间
- 使用Nsight Compute分析核函数性能
- 监控内存使用和带宽利用率

## 常见问题

### Q: 如何处理不同形状的张量运算？
A: 实现广播机制或在算子中添加形状检查和转换逻辑。

### Q: 如何支持多GPU并行？
A: 在OperatorContext中设置不同的设备ID，并使用NCCL进行跨GPU通信。

### Q: 如何优化小张量的性能？
A: 对于小张量，考虑使用CPU计算或将多个操作批处理。

### Q: 如何添加自定义CUDA核函数？
A: 在.cu文件中定义核函数，然后在算子的forward方法中调用。

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
