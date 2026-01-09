# CUDA核函数编程基础

## 概述

本文档详细介绍CUDA核函数编程的基础知识，包括核函数开发框架、内存管理教学系统和性能分析工具的使用。

## 学习目标

完成本模块后，你将能够：
- 理解CUDA核函数的基本概念和编程模型
- 掌握线程索引计算和网格配置的最佳实践
- 熟练使用不同类型的GPU内存
- 进行基础的性能分析和优化

## 核函数开发框架

### 核函数基础

核函数是在GPU上并行执行的函数，使用`__global__`关键字声明：

```cuda
__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### 线程索引计算

#### 一维索引计算
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

#### 二维索引计算
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;
```

#### 三维索引计算
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * width * height + y * width + x;
```

### 网格和块配置最佳实践

#### 块大小选择原则
- 通常选择32的倍数（warp大小）
- 常用配置：128, 256, 512线程/块
- 考虑共享内存使用量
- 考虑寄存器使用量

#### 网格大小计算
```cuda
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(data, n);
```

## 内存管理教学系统

### GPU内存层次结构

#### 全局内存 (Global Memory)
- 最大容量，最慢访问速度
- 所有线程都可访问
- 生命周期：整个应用程序

```cuda
// 分配全局内存
float* d_data;
cudaMalloc(&d_data, size * sizeof(float));

// 数据传输
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

// 释放内存
cudaFree(d_data);
```

#### 共享内存 (Shared Memory)
- 块内线程共享
- 访问速度快，容量小
- 生命周期：块执行期间

```cuda
__global__ void sharedMemoryExample(float* data) {
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    shared_data[tid] = data[idx];
    __syncthreads();

    // 使用共享内存数据
    float result = shared_data[tid] * 2.0f;
    data[idx] = result;
}
```

#### 常量内存 (Constant Memory)
- 只读内存，有缓存
- 适合所有线程访问相同数据

```cuda
__constant__ float const_data[256];

// 主机端设置常量内存
cudaMemcpyToSymbol(const_data, h_data, 256 * sizeof(float));
```

#### 纹理内存 (Texture Memory)
- 只读，有缓存，支持插值
- 适合空间局部性访问

```cuda
texture<float, 1, cudaReadModeElementType> tex_data;

__global__ void textureExample(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tex1Dfetch(tex_data, idx);
    }
}
```

### 内存合并优化

#### 合并访问模式
```cuda
// 好的访问模式 - 合并访问
__global__ void coalescedAccess(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // 连续访问
}

// 坏的访问模式 - 非合并访问
__global__ void stridedAccess(float* data, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    data[idx] = data[idx] * 2.0f;  // 跨步访问
}
```

#### 内存带宽测试
```cuda
__global__ void bandwidthTest(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        for (int i = 0; i < iterations; i++) {
            value = value * 1.01f;
        }
        data[idx] = value;
    }
}
```

## 性能分析和优化工具

### CUDA事件计时

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<gridSize, blockSize>>>(data, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Kernel execution time: %f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 占用率计算

```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

int gridSize = (n + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(data, n);
```

### 性能分析工具集成

#### Nsight Compute集成
```bash
# 性能分析命令
ncu --set full --force-overwrite -o profile_output ./my_cuda_program

# 查看分析结果
ncu-ui profile_output.ncu-rep
```

#### 自定义性能监控
```cuda
class PerformanceMonitor {
private:
    cudaEvent_t start, stop;
    std::vector<float> timings;

public:
    void startTiming() {
        cudaEventRecord(start);
    }

    void stopTiming() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        timings.push_back(ms);
    }

    float getAverageTime() {
        float sum = 0;
        for (float t : timings) sum += t;
        return sum / timings.size();
    }
};
```

## 实践练习

### 练习1：向量加法优化
实现并优化向量加法核函数，比较不同块大小的性能。

### 练习2：矩阵转置
实现矩阵转置核函数，使用共享内存避免银行冲突。

### 练习3：归约运算
实现并行归约求和，使用共享内存和warp shuffle优化。

### 练习4：内存带宽测试
编写内存带宽测试程序，比较不同访问模式的性能。

## 常见问题和解决方案

### 问题1：银行冲突
**症状**：共享内存访问性能差
**解决**：使用填充或改变访问模式

### 问题2：分支发散
**症状**：条件语句导致性能下降
**解决**：重组算法减少分支

### 问题3：占用率低
**症状**：GPU利用率不高
**解决**：调整块大小和资源使用

## 进阶主题

- warp级原语使用
- 动态并行
- 统一内存管理
- 多流并发执行

## 参考资料

- CUDA C++ Programming Guide
- CUDA Best Practices Guide
- Nsight Compute User Guide
- GPU Architecture Whitepaper

---

**下一步**：学习[自定义算子开发框架](03-custom-operator-framework.md)
