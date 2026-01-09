# 卷积和池化算子实现文档

## 概述

本文档描述了CUDA编程精通学习系统中卷积和池化算子的实现。这些算子是深度学习中最重要的基础操作，包括：

- 2D卷积 (Conv2D)
- 最大池化 (MaxPool2D)
- 平均池化 (AvgPool2D)
- 批量归一化 (BatchNorm2D)
- cuDNN性能对比工具

## 实现的算子

### 1. Conv2DOperator - 2D卷积算子

#### 功能描述
实现2D卷积操作，支持可配置的步长、填充和膨胀参数。

#### 输入参数
- **输入张量**: `[N, C, H, W]` - 批次大小、输入通道数、高度、宽度
- **权重张量**: `[K, C, R, S]` - 输出通道数、输入通道数、卷积核高度、卷积核宽度
- **必需参数**:
  - `stride_h`, `stride_w`: 垂直和水平步长
  - `pad_h`, `pad_w`: 垂直和水平填充
- **可选参数**:
  - `dilation_h`, `dilation_w`: 膨胀参数
  - `groups`: 分组卷积参数

#### 输出
- **输出张量**: `[N, K, P, Q]` - 其中 P = (H + 2*pad_h - R) / stride_h + 1, Q = (W + 2*pad_w - S) / stride_w + 1

#### CUDA核函数实现
```cuda
__global__ void conv2d_naive_kernel(
    const float* input,     // [N, C, H, W]
    const float* weight,    // [K, C, R, S]
    float* output,          // [N, K, P, Q]
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
```

#### 优化特性
- 朴素实现：直接计算，易于理解
- 优化版本：使用共享内存分块计算（模板实现）
- 支持边界填充和步长控制

### 2. MaxPool2DOperator - 最大池化算子

#### 功能描述
在指定窗口内寻找最大值，常用于降采样和特征提取。

#### 输入参数
- **输入张量**: `[N, C, H, W]`
- **必需参数**:
  - `kernel_h`, `kernel_w`: 池化窗口大小
  - `stride_h`, `stride_w`: 步长
  - `pad_h`, `pad_w`: 填充

#### 输出
- **输出张量**: `[N, C, P, Q]` - 其中 P = (H + 2*pad_h - kernel_h) / stride_h + 1

#### CUDA核函数实现
```cuda
__global__ void maxpool2d_kernel(
    const float* input,     // [N, C, H, W]
    float* output,          // [N, C, P, Q]
    int* indices,           // [N, C, P, Q] - 用于反向传播
    int N, int C, int H, int W,
    int P, int Q,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
```

#### 特性
- 支持任意大小的池化窗口
- 可选择保存最大值索引用于反向传播
- 处理边界填充

### 3. AvgPool2DOperator - 平均池化算子

#### 功能描述
计算指定窗口内的平均值，提供平滑的降采样。

#### 输入参数
- **输入张量**: `[N, C, H, W]`
- **必需参数**: 与MaxPool2D相同
- **可选参数**:
  - `count_include_pad`: 是否在平均值计算中包含填充区域

#### CUDA核函数实现
```cuda
__global__ void avgpool2d_kernel(
    const float* input,     // [N, C, H, W]
    float* output,          // [N, C, P, Q]
    int N, int C, int H, int W,
    int P, int Q,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    bool count_include_pad)
```

### 4. BatchNorm2DOperator - 批量归一化算子

#### 功能描述
对2D特征图进行批量归一化，加速训练收敛并提高模型稳定性。

#### 输入参数
- **输入张量**: `[N, C, H, W]`
- **Gamma参数**: `[C]` - 缩放参数
- **Beta参数**: `[C]` - 偏移参数
- **可选参数**:
  - `eps`: 数值稳定性参数 (默认: 1e-5)
  - `momentum`: 动量参数 (默认: 0.1)
  - `training`: 训练模式标志

#### CUDA核函数实现
```cuda
// 统计量计算
__global__ void batchnorm2d_stats_kernel(
    const float* input,     // [N, C, H, W]
    float* mean,            // [C]
    float* var,             // [C]
    int N, int C, int H, int W)

// 归一化应用
__global__ void batchnorm2d_kernel(
    const float* input,     // [N, C, H, W]
    const float* gamma,     // [C]
    const float* beta,      // [C]
    const float* mean,      // [C]
    const float* var,       // [C]
    float* output,          // [N, C, H, W]
    int N, int C, int H, int W,
    float eps)
```

#### 算法流程
1. 计算每个通道的均值和方差
2. 使用均值和方差对输入进行归一化
3. 应用可学习的缩放和偏移参数

## cuDNN性能对比工具

### CuDNNConvBenchmark 类

提供CUDA实现与cuDNN库性能对比的工具。

#### 主要功能
- `compareConvPerformance()`: 对比卷积操作性能
- `comparePoolingPerformance()`: 对比池化操作性能
- `compareBatchNormPerformance()`: 对比批量归一化性能

#### 基准测试结果结构
```cpp
struct BenchmarkResult {
    float cuda_time_ms;      // CUDA实现执行时间
    float cudnn_time_ms;     // cuDNN实现执行时间
    float speedup_ratio;     // 加速比
    bool cudnn_available;    // cuDNN是否可用
    std::string error_message; // 错误信息
};
```

## 使用示例

### 基本卷积操作
```cpp
// 创建输入和权重
FloatTensor input({1, 3, 32, 32});
FloatTensor weight({16, 3, 3, 3});
input.uniform(-1.0f, 1.0f);
weight.uniform(-0.1f, 0.1f);

// 设置参数
OperatorContext context;
context.setParam("stride_h", 1);
context.setParam("stride_w", 1);
context.setParam("pad_h", 1);
context.setParam("pad_w", 1);

// 执行卷积
Conv2DOperator conv_op;
auto output_shapes = conv_op.inferOutputShapes({input.shape(), weight.shape()}, context);
FloatTensor output(output_shapes[0].dims());
conv_op.forward({input, weight}, {output}, context);
```

### 池化操作
```cpp
// 最大池化
OperatorContext pool_context;
pool_context.setParam("kernel_h", 2);
pool_context.setParam("kernel_w", 2);
pool_context.setParam("stride_h", 2);
pool_context.setParam("stride_w", 2);
pool_context.setParam("pad_h", 0);
pool_context.setParam("pad_w", 0);

MaxPool2DOperator pool_op;
auto pool_output_shapes = pool_op.inferOutputShapes({input.shape()}, pool_context);
FloatTensor pool_output(pool_output_shapes[0].dims());
pool_op.forward({input}, {pool_output}, pool_context);
```

### 批量归一化
```cpp
// 创建参数
int channels = input.shape().dim(1);
FloatTensor gamma({channels});
FloatTensor beta({channels});
gamma.ones();
beta.zero();

// 设置上下文
OperatorContext bn_context;
bn_context.setParam("eps", 1e-5f);
bn_context.setParam("training", true);

// 执行批量归一化
BatchNorm2DOperator bn_op;
auto bn_output_shapes = bn_op.inferOutputShapes({input.shape()}, bn_context);
FloatTensor bn_output(bn_output_shapes[0].dims());
bn_op.forward({input, gamma, beta}, {bn_output}, bn_context);
```

### 性能基准测试
```cpp
// 卷积性能对比
TensorShape input_shape({1, 64, 224, 224});
TensorShape weight_shape({128, 64, 3, 3});

auto result = CuDNNConvBenchmark::compareConvPerformance(
    input_shape, weight_shape, 1, 1, 1, 1, 100);

std::cout << "CUDA time: " << result.cuda_time_ms << " ms" << std::endl;
std::cout << "cuDNN time: " << result.cudnn_time_ms << " ms" << std::endl;
std::cout << "Speedup: " << result.speedup_ratio << "x" << std::endl;
```

## 性能优化技术

### 1. 内存访问优化
- **合并访问**: 确保线程访问连续内存地址
- **共享内存**: 使用分块算法减少全局内存访问
- **寄存器优化**: 最小化寄存器使用以提高占用率

### 2. 计算优化
- **循环展开**: 减少循环开销
- **指令级并行**: 利用GPU的指令流水线
- **数据重用**: 最大化数据在缓存中的重用

### 3. 线程配置优化
- **块大小**: 选择合适的线程块大小以最大化占用率
- **网格大小**: 确保有足够的并行度
- **负载均衡**: 避免线程发散和不均匀的工作负载

## 编译和构建

### 依赖项
- CUDA Toolkit (>= 11.0)
- cuDNN (可选，用于性能对比)
- CMake (>= 3.18)

### 编译命令
```bash
mkdir build && cd build
cmake ..
make conv_pool_demo
./conv_pool_demo
```

### 测试
```bash
# 编译测试程序
nvcc -o test_conv_pool test_conv_pool_operators.cpp src/operators/*.cu src/core/*.cpp -lcudart -lcublas

# 运行测试
./test_conv_pool
```

## 扩展和定制

### 添加新的卷积变体
1. 继承 `BaseOperator` 类
2. 实现必需的虚函数
3. 编写对应的CUDA核函数
4. 在注册函数中添加新算子

### 性能调优建议
1. 使用NVIDIA Nsight Compute进行性能分析
2. 测试不同的线程块配置
3. 考虑使用Tensor Cores（在支持的GPU上）
4. 实现算子融合以减少内存访问

## 已知限制和未来改进

### 当前限制
- 反向传播未完全实现（仅前向传播）
- 不支持动态形状
- 有限的数据类型支持（仅float32）

### 计划改进
- 添加完整的反向传播支持
- 实现混合精度训练
- 支持3D卷积
- 添加更多激活函数融合
- 集成Tensor Core优化

## 参考资料

1. CUDA C++ Programming Guide
2. cuDNN Developer Guide
3. Deep Learning with PyTorch
4. Efficient ConvNet Architectures for Mobile Applications
5. NVIDIA GPU Architecture Whitepapers
