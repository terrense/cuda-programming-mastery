# CUDA核函数开发框架教程

## 概述

本框架提供了一套完整的CUDA核函数开发工具，包括模板生成器、线程索引计算辅助函数、网格和块配置优化器等，帮助开发者快速编写高效的CUDA核函数。

## 主要组件

### 1. 核函数模板生成器 (KernelTemplateGenerator)

模板生成器可以自动生成常见类型的核函数代码框架，支持以下模板类型：

- **向量加法** (VECTOR_ADD)
- **矩阵乘法** (MATRIX_MULTIPLY)
- **归约操作** (REDUCTION)
- **2D卷积** (CONVOLUTION_2D)
- **元素级操作** (ELEMENT_WISE)
- **自定义模板** (CUSTOM)

#### 使用示例

```cpp
#include "kernel_framework.h"

// 生成向量加法模板
std::string vectorAddCode = KernelTemplateGenerator::generateVectorAddTemplate("myVectorAdd");
std::cout << vectorAddCode << std::endl;

// 生成矩阵乘法模板
std::string matMulCode = KernelTemplateGenerator::generateMatrixMultiplyTemplate("myMatMul");
std::cout << matMulCode << std::endl;
```

### 2. 线程索引计算辅助函数 (ThreadUtils)

提供了一系列设备端函数，简化线程索引计算：

#### 1D线程索引
```cpp
__global__ void myKernel1D(float* data, int n) {
    int idx = ThreadUtils::getGlobalThreadId1D();
    if (ThreadUtils::isValidThread1D(n)) {
        // 处理data[idx]
    }
}
```

#### 2D线程索引
```cpp
__global__ void myKernel2D(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (ThreadUtils::isValidThread2D(width, height)) {
        int idx = ThreadUtils::getGlobalThreadId2D(width);
        // 处理data[idx]
    }
}
```

#### 3D线程索引
```cpp
__global__ void myKernel3D(float* data, int width, int height, int depth) {
    if (ThreadUtils::isValidThread3D(width, height, depth)) {
        int idx = ThreadUtils::getGlobalThreadId3D(width, height);
        // 处理data[idx]
    }
}
```

### 3. 网格和块配置优化器 (GridBlockOptimizer)

自动计算最优的网格和块配置，提高核函数性能：

#### 1D数据优化
```cpp
int dataSize = 1000000;
GridBlockConfig config = GridBlockOptimizer::optimize1D(dataSize);
std::cout << "优化配置: " << config.toString() << std::endl;
```

#### 2D数据优化
```cpp
int width = 1024, height = 1024;
GridBlockConfig config = GridBlockOptimizer::optimize2D(width, height);
```

#### 矩阵乘法优化
```cpp
int M = 512, N = 512, K = 512;
GridBlockConfig config = GridBlockOptimizer::optimizeMatMul(M, N, K);
```

#### 归约操作优化
```cpp
int dataSize = 1000000;
GridBlockConfig config = GridBlockOptimizer::optimizeReduction(dataSize);
// 注意：归约操作会自动配置共享内存大小
```

### 4. 核函数启动器 (KernelLauncher)

提供了安全的核函数启动接口，包含错误检查：

#### 基本启动
```cpp
GridBlockConfig config = GridBlockOptimizer::optimize1D(n);
cudaError_t error = KernelLauncher::launch(myKernel, config, d_a, d_b, d_c, n);
```

#### 带错误检查的启动
```cpp
GridBlockConfig config = GridBlockOptimizer::optimize1D(n);
bool success = KernelLauncher::launchWithCheck(myKernel, config, d_a, d_b, d_c, n);
if (!success) {
    std::cerr << "核函数执行失败!" << std::endl;
}
```

## 完整示例：向量加法

以下是一个完整的向量加法示例，展示了如何使用框架的各个组件：

```cpp
#include "kernel_framework.h"
#include <vector>
#include <iostream>

// 使用框架辅助函数的向量加法核函数
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = ThreadUtils::getGlobalThreadId1D();
    if (ThreadUtils::isValidThread1D(n)) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000000;
    size_t bytes = n * sizeof(float);

    // 主机内存
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    // 设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 复制数据到设备
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // 使用优化器计算最佳配置
    GridBlockConfig config = GridBlockOptimizer::optimize1D(n);
    std::cout << "使用配置: " << config.toString() << std::endl;

    // 启动核函数
    bool success = KernelLauncher::launchWithCheck(vectorAdd, config, d_a, d_b, d_c, n);

    if (success) {
        // 复制结果回主机
        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        // 验证结果
        std::cout << "前5个结果: ";
        for (int i = 0; i < 5; i++) {
            std::cout << h_c[i] << " ";
        }
        std::cout << std::endl;
    }

    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

## 最佳实践

### 1. 选择合适的块大小
- 1D问题：通常使用256或512个线程
- 2D问题：通常使用16x16或32x32
- 3D问题：通常使用8x8x8或更小

### 2. 内存访问优化
- 确保内存访问是合并的
- 使用共享内存减少全局内存访问
- 避免银行冲突

### 3. 线程发散最小化
- 避免在warp内使用分支语句
- 使用条件赋值代替条件分支

### 4. 占用率优化
- 使用`GridBlockOptimizer::calculateOptimalBlockSize()`获取最优块大小
- 平衡寄存器使用和共享内存使用

## 性能分析

框架提供了基本的性能分析功能：

```cpp
// 获取设备属性
cudaDeviceProp prop = GridBlockOptimizer::getDeviceProperties(0);
std::cout << "设备: " << prop.name << std::endl;
std::cout << "多处理器数量: " << prop.multiProcessorCount << std::endl;

// 计算理论占用率
GridBlockConfig config = GridBlockOptimizer::optimize1D(dataSize);
int totalThreads = config.getTotalThreads();
int maxThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
float occupancy = (float)totalThreads / maxThreads;
std::cout << "理论占用率: " << occupancy * 100 << "%" << std::endl;
```

## 常见问题

### Q: 如何选择合适的模板类型？
A: 根据你的计算模式选择：
- 简单的元素级操作：使用ELEMENT_WISE
- 向量运算：使用VECTOR_ADD
- 矩阵运算：使用MATRIX_MULTIPLY
- 求和、最大值等：使用REDUCTION
- 图像处理：使用CONVOLUTION_2D

### Q: 优化器给出的配置不是最优的怎么办？
A: 优化器提供的是通用的最佳实践配置。对于特定应用，你可能需要：
- 考虑数据访问模式
- 测试不同的块大小
- 使用CUDA Profiler进行详细分析

### Q: 如何处理不规则的数据大小？
A: 框架自动处理边界检查，但你也可以：
- 使用填充将数据大小调整为块大小的倍数
- 在核函数中添加额外的边界检查
- 使用动态并行处理不规则区域

## 扩展框架

框架设计为可扩展的，你可以：

1. 添加新的模板类型
2. 实现自定义的优化策略
3. 集成更多的性能分析工具
4. 添加对新GPU架构的支持

通过继承和扩展现有类，可以轻松地为特定应用定制框架功能。
