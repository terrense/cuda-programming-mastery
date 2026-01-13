# CUDA多GPU编程框架使用指南

## 概述

本文档介绍CUDA多GPU编程框架的使用方法，该框架提供了完整的多GPU编程解决方案，包括NCCL通信、负载均衡、数据分片和协同训练推理等功能。

## 框架架构

### 核心组件

1. **NCCLCommunicator**: NCCL通信管理器
2. **DataShardManager**: 数据分片管理器
3. **LoadBalancer**: 负载均衡器
4. **MultiGPUExecutor**: 多GPU执行器
5. **MultiGPUContext**: 多GPU上下文管理器
6. **MultiGPUTrainer**: 多GPU训练框架
7. **MultiGPUInference**: 多GPU推理框架

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MultiGPUContext                          │
├─────────────────────────────────────────────────────────────┤
│  NCCLCommunicator  │  DataShardManager  │  LoadBalancer    │
├─────────────────────────────────────────────────────────────┤
│                  MultiGPUExecutor                           │
├─────────────────────────────────────────────────────────────┤
│  MultiGPUTrainer   │              │  MultiGPUInference     │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境要求

- CUDA 11.0+
- NCCL 2.8+
- 支持CUDA的GPU设备（计算能力3.0+）
- CMake 3.18+
- C++17编译器

### 2. 编译安装

```bash
# 克隆项目
git clone <repository_url>
cd cuda_programming

# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON

# 编译
make -j$(nproc)

# 运行测试
make test

# 运行演示程序
./bin/multi_gpu_demo
```

### 3. 基本使用示例

```cpp
#include "multi_gpu_framework.h"
using namespace cuda_learning;

int main() {
    // 1. 初始化多GPU上下文
    MultiGPUContext context;
    std::vector<int> deviceIds = {0, 1}; // 使用GPU 0和1

    if (!context.initialize(deviceIds)) {
        std::cerr << "初始化失败: " << context.getLastError() << std::endl;
        return -1;
    }

    // 2. 创建训练数据
    std::vector<float> data(1024 * 1024, 1.0f);

    // 3. 执行数据并行训练
    MultiGPUTrainer trainer(&context);

    auto trainStep = [](void* data, size_t size, int batch) {
        // 训练逻辑
        cudaDeviceSynchronize();
    };

    bool success = trainer.trainDataParallel(
        data.data(), data.size() * sizeof(float),
        256, trainStep);

    if (success) {
        std::cout << "训练成功完成!" << std::endl;
    }

    // 4. 清理资源
    context.finalize();
    return 0;
}
```

## 详细功能介绍

### NCCL通信操作

#### AllReduce操作

```cpp
NCCLCommunicator communicator;
communicator.initialize({0, 1, 2, 3});

// 执行AllReduce求和
float* deviceData; // 设备数据指针
size_t count = 1024;
bool success = communicator.allReduce(
    deviceData, deviceData, count,
    ncclFloat, ncclSum, {0, 1, 2, 3}
);
```

#### 支持的通信操作

- **AllReduce**: 全归约操作
- **AllGather**: 全收集操作
- **ReduceScatter**: 归约散射操作
- **Broadcast**: 广播操作
- **Reduce**: 归约操作
- **Send/Recv**: 点对点通信

### 数据分片策略

#### 批次并行分片

```cpp
DataShardManager shardManager;

// 创建批次并行分片
auto shards = shardManager.createShards(
    data, dataSize, deviceIds,
    ShardingStrategy::BATCH_PARALLEL
);

// 分发数据到各GPU
shardManager.distributeData(shards);
```

#### 支持的分片策略

- **BATCH_PARALLEL**: 批次并行分片
- **DATA_PARALLEL**: 数据并行分片
- **MODEL_PARALLEL**: 模型并行分片
- **PIPELINE_PARALLEL**: 流水线并行分片
- **HYBRID_PARALLEL**: 混合并行分片

### 负载均衡策略

#### 动态自适应负载均衡

```cpp
LoadBalancer balancer(LoadBalanceStrategy::DYNAMIC_ADAPTIVE);
balancer.setAvailableDevices(devices);

// 选择最优设备
int selectedDevice = balancer.selectDevice(task);

// 更新设备负载
balancer.updateDeviceLoad(deviceId, 0.8f);
balancer.updateDeviceMemory(deviceId, freeMemory);
```

#### 支持的负载均衡策略

- **ROUND_ROBIN**: 轮询策略
- **MEMORY_AWARE**: 内存感知策略
- **COMPUTE_AWARE**: 计算感知策略
- **DYNAMIC_ADAPTIVE**: 动态自适应策略

### 多GPU训练

#### 数据并行训练

```cpp
MultiGPUTrainer trainer(&context);

// 定义训练步骤
auto trainStep = [](void* data, size_t size, int batchSize) {
    // 前向传播
    forward_pass(data, size);

    // 反向传播
    backward_pass(data, size);

    // 同步
    cudaDeviceSynchronize();
};

// 执行数据并行训练
bool success = trainer.trainDataParallel(
    trainingData.data(),
    trainingData.size() * sizeof(float),
    batchSize,
    trainStep
);
```

#### 模型并行训练

```cpp
// 模型分片
std::vector<void*> modelParts = {layer1, layer2, layer3, layer4};
std::vector<size_t> partSizes = {size1, size2, size3, size4};

// 定义训练步骤
auto trainStep = [](void* modelPart, int deviceId, int stageId) {
    // 执行对应阶段的训练
    execute_stage(modelPart, stageId);
};

// 执行流水线并行训练
bool success = trainer.trainPipelineParallel(
    modelParts, partSizes, trainStep
);
```

#### 梯度同步

```cpp
// 同步梯度
trainer.synchronizeGradients(gradients, gradientSize);

// 更新参数
trainer.updateParameters(parameters, parameterSize);
```

### 多GPU推理

#### 批处理推理

```cpp
MultiGPUInference inference(&context);

// 定义推理步骤
auto inferenceStep = [](void* input, void* output, int deviceId) {
    // 执行推理
    model_inference(input, output);
    cudaDeviceSynchronize();
};

// 执行批处理推理
bool success = inference.inferBatch(
    inputData.data(), inputSize,
    batchSize,
    outputData.data(), outputSize,
    inferenceStep
);
```

#### 流式推理

```cpp
// 流式推理（支持流水线处理）
bool success = inference.inferStream(
    inputData.data(), inputSize,
    outputData.data(), outputSize,
    inferenceStep
);
```

## 性能优化指南

### 1. 通信优化

#### 减少通信频率
```cpp
// 批量梯度累积，减少AllReduce频率
const int accumulation_steps = 4;
for (int step = 0; step < accumulation_steps; ++step) {
    // 前向和反向传播
    forward_backward(data[step]);
}
// 一次性同步累积的梯度
trainer.synchronizeGradients(accumulated_gradients, gradSize);
```

#### 通信与计算重叠
```cpp
// 使用异步通信
communicator.allReduce(gradients, gradients, count,
                      ncclFloat, ncclSum, deviceIds);

// 在通信进行时执行其他计算
perform_other_computations();

// 同步等待通信完成
communicator.synchronize(deviceIds);
```

### 2. 内存优化

#### 内存池管理
```cpp
// 预分配内存池
for (int deviceId : deviceIds) {
    void* memoryPool = context.allocateOnDevice(deviceId, poolSize);
    // 管理内存池分配
}
```

#### 数据预取
```cpp
// 异步数据传输
cudaMemcpyAsync(devicePtr, hostPtr, size,
               cudaMemcpyHostToDevice, stream);
```

### 3. 负载均衡优化

#### 动态负载监控
```cpp
// 定期更新设备负载信息
auto updateLoads = [&]() {
    for (int deviceId : deviceIds) {
        float load = measureDeviceLoad(deviceId);
        balancer.updateDeviceLoad(deviceId, load);

        size_t freeMemory = getDeviceFreeMemory(deviceId);
        balancer.updateDeviceMemory(deviceId, freeMemory);
    }
};

// 定期执行负载更新
std::thread loadMonitor(updateLoads);
```

## 错误处理和调试

### 1. 错误检查

```cpp
// 检查NCCL错误
if (!communicator.allReduce(...)) {
    std::cerr << "NCCL AllReduce失败" << std::endl;
    // 错误处理逻辑
}

// 检查CUDA错误
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
}
```

### 2. 性能分析

```cpp
// 生成性能报告
std::string report = multi_gpu_utils::generateMultiGPUReport(context);
std::cout << report << std::endl;

// 测量通信开销
float commCost = multi_gpu_utils::estimateCommunicationCost(
    CommType::ALL_REDUCE, dataSize, numDevices);

// 计算负载均衡效率
auto loads = balancer.getDeviceLoads();
float efficiency = multi_gpu_utils::calculateLoadBalanceEfficiency(loads);
```

### 3. 调试技巧

#### 启用详细日志
```cpp
// 设置NCCL日志级别
setenv("NCCL_DEBUG", "INFO", 1);

// 设置CUDA错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
```

#### 性能分析工具
```bash
# 使用Nsight Systems分析
nsys profile --trace=cuda,nvtx ./multi_gpu_demo

# 使用Nsight Compute分析核函数
ncu --set full ./multi_gpu_demo
```

## 最佳实践

### 1. 设备选择

```cpp
// 选择最优设备组合
std::vector<int> optimalDevices = multi_gpu_utils::getOptimalDeviceSet(4);

// 检查设备兼容性
for (int deviceId : optimalDevices) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    if (prop.major < 6) {
        std::cout << "警告: 设备 " << deviceId
                 << " 计算能力较低，可能影响性能" << std::endl;
    }
}
```

### 2. 数据布局优化

```cpp
// 使用连续内存布局
struct OptimizedData {
    float* data;
    size_t size;
    size_t stride;

    // 确保内存对齐
    OptimizedData(size_t n) : size(n) {
        stride = ((n + 31) / 32) * 32; // 32字节对齐
        cudaMalloc(&data, stride * sizeof(float));
    }
};
```

### 3. 流水线设计

```cpp
// 多阶段流水线
class MultiStagePipeline {
public:
    void addStage(int deviceId, std::function<void()> stage) {
        stages_[deviceId] = stage;
    }

    void execute() {
        std::vector<std::thread> threads;
        for (auto& pair : stages_) {
            threads.emplace_back([&pair]() {
                cudaSetDevice(pair.first);
                pair.second();
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

private:
    std::map<int, std::function<void()>> stages_;
};
```

## 常见问题解答

### Q1: NCCL初始化失败怎么办？

**A**: 检查以下几点：
1. 确保所有GPU设备支持P2P通信
2. 检查NCCL库是否正确安装
3. 验证设备间的网络连接
4. 检查防火墙设置

```bash
# 检查P2P支持
nvidia-smi topo -m

# 检查NCCL版本
nccl-test --version
```

### Q2: 内存不足错误如何处理？

**A**: 采用以下策略：
1. 减少批次大小
2. 使用梯度累积
3. 启用内存优化选项
4. 使用混合精度训练

```cpp
// 梯度累积示例
const int virtual_batch_size = 1024;
const int actual_batch_size = 256;
const int accumulation_steps = virtual_batch_size / actual_batch_size;

for (int step = 0; step < accumulation_steps; ++step) {
    // 前向传播
    forward(data + step * actual_batch_size, actual_batch_size);

    // 反向传播（不更新参数）
    backward(gradients, actual_batch_size, false);
}

// 同步梯度并更新参数
trainer.synchronizeGradients(gradients, gradSize);
update_parameters(gradients);
```

### Q3: 通信性能差怎么优化？

**A**: 尝试以下优化方法：
1. 使用更高带宽的互连（NVLink、InfiniBand）
2. 调整NCCL算法选择
3. 优化数据传输大小
4. 使用通信与计算重叠

```cpp
// 设置NCCL算法
setenv("NCCL_ALGO", "Ring", 1);  // 或 "Tree"

// 设置NCCL协议
setenv("NCCL_PROTO", "Simple", 1);  // 或 "LL", "LL128"
```

## 示例项目

### 完整的多GPU训练示例

参考 `examples/advanced/multi_gpu_demo.cu` 文件，包含：

1. 数据并行训练演示
2. 批处理推理演示
3. NCCL通信操作演示
4. 负载均衡演示

### 性能基准测试

参考 `tests/unit/test_multi_gpu_framework.cpp` 文件，包含：

1. 通信性能测试
2. 内存管理测试
3. 负载均衡测试
4. 端到端训练测试

## 总结

CUDA多GPU编程框架提供了完整的多GPU编程解决方案，支持：

- ✅ NCCL高性能通信
- ✅ 灵活的数据分片策略
- ✅ 智能负载均衡
- ✅ 数据并行和模型并行训练
- ✅ 高效批处理推理
- ✅ 完善的错误处理和性能分析

通过合理使用这些功能，可以显著提升深度学习和高性能计算应用的性能和可扩展性。

## 参考资料

- [NCCL官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [多GPU编程最佳实践](https://developer.nvidia.com/blog/multi-gpu-programming-models/)
- [性能优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
