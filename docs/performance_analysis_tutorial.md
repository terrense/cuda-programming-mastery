# CUDA性能分析和优化工具教程

## 概述

CUDA性能优化是一个系统性工程，需要从多个维度分析和改进。本教学系统提供了完整的性能分析工具链，包括计时器、占用率计算器、瓶颈检测器和自动优化建议生成器，帮助开发者系统性地优化CUDA应用性能。

## 核心工具组件

### 1. CUDA事件计时器 (CudaTimer)

精确测量CUDA核函数执行时间的工具。

#### 基本使用

```cpp
#include "performance_analyzer.h"

// 基本计时
CudaTimer timer;
timer.start();
myKernel<<<grid, block>>>(args...);
timer.stop();
float elapsedTime = timer.getElapsedTime(); // 毫秒

// 函数模板计时
float avgTime = CudaTimer::timeFunction([&]() {
    myKernel<<<grid, block>>>(args...);
}, 100); // 100次迭代的平均时间
```

#### 高级用法

```cpp
// 比较不同配置的性能
std::vector<int> blockSizes = {64, 128, 256, 512, 1024};
for (int blockSize : blockSizes) {
    int gridSize = (dataSize + blockSize - 1) / blockSize;

    float time = CudaTimer::timeKernel(
        myKernel, dim3(gridSize), dim3(blockSize), 0, 0, args...);

    std::cout << "块大小 " << blockSize << ": " << time << " ms" << std::endl;
}
```

### 2. 占用率计算器 (OccupancyCalculator)

分析和优化GPU占用率的工具。

#### 占用率分析

```cpp
// 获取最优块大小
int optimalBlockSize = OccupancyCalculator::getOptimalBlockSize(myKernel);

// 计算理论占用率
float theoretical = OccupancyCalculator::calculateTheoreticalOccupancy(
    blockSize, registersPerThread, sharedMemPerBlock);

// 计算实际占用率
float actual = OccupancyCalculator::calculateActualOccupancy(
    myKernel, blockSize, sharedMemPerBlock);

// 分析占用率限制因素
std::string analysis = OccupancyCalculator::analyzeOccupancyLimiters(
    blockSize, registersPerThread, sharedMemPerBlock);
```

#### 占用率优化策略

```cpp
// 比较不同块大小的占用率
std::vector<int> blockSizes = {64, 128, 256, 512, 1024};
auto occupancyMap = OccupancyCalculator::calculateOccupancyForBlockSizes(
    myKernel, blockSizes);

// 找到最佳配置
int bestBlockSize = 0;
float bestOccupancy = 0.0f;
for (const auto& pair : occupancyMap) {
    if (pair.second > bestOccupancy) {
        bestOccupancy = pair.second;
        bestBlockSize = pair.first;
    }
}
```

### 3. 性能瓶颈检测器

自动识别性能瓶颈并提供针对性建议。

#### 瓶颈检测

```cpp
// 收集性能指标
PerformanceMetrics metrics;
metrics.executionTime_ms = 15.5f;
metrics.bandwidth_GB_s = 320.0f;
metrics.occupancy_percent = 65.0f;
metrics.memoryUtilization_percent = 85.0f;
metrics.computeUtilization_percent = 45.0f;

// 检测瓶颈
BottleneckAnalysis analysis = PerformanceAnalyzer::detectBottlenecks(metrics);

std::cout << "主要瓶颈: " << analysis.toString() << std::endl;
```

#### 瓶颈类型和解决方案

| 瓶颈类型 | 特征 | 解决方案 |
|---------|------|----------|
| 内存受限 | 内存利用率高，计算利用率低 | 优化内存访问模式，使用共享内存 |
| 计算受限 | 计算利用率高，内存利用率低 | 算法优化，使用专用计算单元 |
| 占用率受限 | 占用率低于50% | 调整块大小，减少资源使用 |
| 寄存器受限 | 寄存器使用过多 | 减少局部变量，使用共享内存 |

### 4. 优化建议生成器

基于性能分析结果自动生成优化建议。

#### 自动建议生成

```cpp
// 生成优化建议
std::vector<OptimizationSuggestion> suggestions =
    PerformanceAnalyzer::generateOptimizationSuggestions(metrics, analysis);

// 按优先级排序并显示
std::sort(suggestions.begin(), suggestions.end(),
    [](const auto& a, const auto& b) { return a.priority > b.priority; });

for (const auto& suggestion : suggestions) {
    std::cout << suggestion.toString() << std::endl;
}
```

#### 优化建议类型

```cpp
// 内存合并优化
OptimizationSuggestion memCoalescing;
memCoalescing.type = OptimizationType::MEMORY_COALESCING;
memCoalescing.title = "优化内存合并访问";
memCoalescing.expectedImprovement = 30.0f;
memCoalescing.codeExample = R"(
// 优化前：跨步访问
data[idx * stride] = value;

// 优化后：合并访问
data[idx] = value;
)";

// 共享内存优化
OptimizationSuggestion sharedMem;
sharedMem.type = OptimizationType::SHARED_MEMORY_USAGE;
sharedMem.title = "使用共享内存缓存";
sharedMem.expectedImprovement = 25.0f;
```

## 实际应用案例

### 案例1: 向量加法优化

```cpp
// 原始版本
__global__ void vectorAddBasic(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 性能分析
void analyzeVectorAdd() {
    const int n = 1000000;

    // 测试不同块大小
    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

    for (int blockSize : blockSizes) {
        int gridSize = (n + blockSize - 1) / blockSize;

        // 计时测试
        float time = CudaTimer::timeKernel(
            vectorAddBasic, dim3(gridSize), dim3(blockSize), 0, 0,
            d_a, d_b, d_c, n);

        // 占用率分析
        float occupancy = OccupancyCalculator::calculateActualOccupancy(
            (const void*)vectorAddBasic, blockSize, 0);

        // 带宽计算
        float bandwidth = (3.0f * sizeof(float) * n) / (time * 1e-3) / (1024*1024*1024);

        std::cout << "块大小: " << blockSize
                  << ", 时间: " << time << " ms"
                  << ", 占用率: " << occupancy << "%"
                  << ", 带宽: " << bandwidth << " GB/s" << std::endl;
    }
}
```

### 案例2: 矩阵乘法优化

```cpp
// 未优化版本
__global__ void matMulNaive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 共享内存优化版本
__global__ void matMulOptimized(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (K + 15) / 16; tile++) {
        // 协作加载到共享内存
        if (row < M && tile * 16 + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * 16 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile * 16 + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 使用共享内存计算
        for (int k = 0; k < 16; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 性能比较分析
void compareMatMulPerformance() {
    const int M = 1024, N = 1024, K = 1024;
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);

    // 测试未优化版本
    float naiveTime = CudaTimer::timeKernel(
        matMulNaive, gridSize, blockSize, 0, 0, d_A, d_B, d_C, M, N, K);

    // 测试优化版本
    float optimizedTime = CudaTimer::timeKernel(
        matMulOptimized, gridSize, blockSize, 0, 0, d_A, d_B, d_C, M, N, K);

    // 计算GFLOPS
    long long flops = 2LL * M * N * K;
    float naiveGFLOPS = (flops / (naiveTime * 1e-3)) / 1e9;
    float optimizedGFLOPS = (flops / (optimizedTime * 1e-3)) / 1e9;

    std::cout << "未优化版本: " << naiveGFLOPS << " GFLOPS" << std::endl;
    std::cout << "优化版本: " << optimizedGFLOPS << " GFLOPS" << std::endl;
    std::cout << "性能提升: " << (optimizedGFLOPS / naiveGFLOPS) << "x" << std::endl;
}
```

## 自动调优工具

### AutoTuner使用

```cpp
// 配置调优参数
AutoTuner::TuningConfig config;
config.blockSizes = {64, 128, 256, 512, 1024};
config.maxIterations = 50;
config.convergenceThreshold = 0.01f;

// 执行自动调优
AutoTuner::TuningResult result = AutoTuner::autoTune(
    myKernel, config, args...);

std::cout << "最优配置: " << result.toString() << std::endl;
```

### 网格搜索调优

```cpp
// 网格搜索所有可能的配置组合
AutoTuner::TuningResult gridResult = AutoTuner::gridSearchTune(
    myKernel, config, args...);

// 分析所有测试结果
for (const auto& metrics : gridResult.allResults) {
    std::cout << "配置: " << metrics.toString() << std::endl;
}
```

## 性能基准测试

### 标准基准测试

```cpp
// 运行标准基准测试套件
std::vector<PerformanceBenchmark::BenchmarkResult> results =
    PerformanceBenchmark::runStandardBenchmarks(0);

// 生成基准测试报告
std::string report = PerformanceBenchmark::generateBenchmarkReport(results);
std::cout << report << std::endl;
```

### 自定义基准测试

```cpp
// 内存带宽基准测试
PerformanceBenchmark::BenchmarkResult memResult =
    PerformanceBenchmark::benchmarkMemoryBandwidth(0);

// 计算性能基准测试
PerformanceBenchmark::BenchmarkResult computeResult =
    PerformanceBenchmark::benchmarkComputePerformance(0);

// 占用率基准测试
PerformanceBenchmark::BenchmarkResult occupancyResult =
    PerformanceBenchmark::benchmarkOccupancy(0);
```

## 性能监控

### 实时性能监控

```cpp
PerformanceMonitor monitor;
monitor.startMonitoring();

// 在训练或推理循环中记录性能
for (int i = 0; i < iterations; i++) {
    CudaTimer timer;
    timer.start();

    myKernel<<<grid, block>>>(args...);

    timer.stop();

    PerformanceMetrics metrics;
    metrics.executionTime_ms = timer.getElapsedTime();
    // ... 设置其他指标

    monitor.recordMetrics(metrics);
}

monitor.stopMonitoring();

// 生成监控报告
std::string monitorReport = monitor.generateMonitoringReport();
std::cout << monitorReport << std::endl;

// 检测性能回归
if (monitor.detectPerformanceRegression(0.1f)) {
    std::cout << "检测到性能回归!" << std::endl;
}
```

## 综合性能报告

### 生成完整报告

```cpp
// 收集性能指标
PerformanceMetrics metrics = collectPerformanceMetrics();

// 检测瓶颈
BottleneckAnalysis bottlenecks = PerformanceAnalyzer::detectBottlenecks(metrics);

// 生成优化建议
std::vector<OptimizationSuggestion> suggestions =
    PerformanceAnalyzer::generateOptimizationSuggestions(metrics, bottlenecks);

// 生成综合报告
std::string report = PerformanceAnalyzer::generatePerformanceReport(
    metrics, bottlenecks, suggestions);

std::cout << report << std::endl;
```

## 最佳实践

### 1. 系统性性能分析流程

```cpp
void systematicPerformanceAnalysis() {
    // 1. 基线性能测试
    PerformanceMetrics baseline = measureBaseline();

    // 2. 瓶颈识别
    BottleneckAnalysis bottlenecks = PerformanceAnalyzer::detectBottlenecks(baseline);

    // 3. 优化实施
    auto suggestions = PerformanceAnalyzer::generateOptimizationSuggestions(baseline, bottlenecks);

    // 4. 优化验证
    for (const auto& suggestion : suggestions) {
        PerformanceMetrics optimized = implementOptimization(suggestion);
        float improvement = (optimized.bandwidth_GB_s - baseline.bandwidth_GB_s) / baseline.bandwidth_GB_s;

        if (improvement > 0.05f) { // 5%以上提升
            std::cout << "优化有效: " << suggestion.title
                      << ", 提升: " << (improvement * 100) << "%" << std::endl;
        }
    }
}
```

### 2. 多维度性能优化

```cpp
void multiDimensionalOptimization() {
    // 内存维度优化
    optimizeMemoryAccess();

    // 计算维度优化
    optimizeComputeIntensity();

    // 占用率维度优化
    optimizeOccupancy();

    // 指令维度优化
    optimizeInstructions();
}
```

### 3. 性能回归检测

```cpp
void performanceRegressionTesting() {
    // 建立性能基线
    std::vector<PerformanceMetrics> baseline = establishBaseline();

    // 定期性能测试
    PerformanceMonitor monitor;
    monitor.startMonitoring();

    // 在CI/CD中集成性能测试
    if (monitor.detectPerformanceRegression(0.05f)) {
        // 触发性能回归警报
        alertPerformanceRegression();
    }
}
```

## 工具集成

### 与CUDA Profiler集成

```cpp
// 结合Nsight Compute使用
void integrateWithNsightCompute() {
    // 1. 使用工具进行初步分析
    PerformanceMetrics metrics = PerformanceAnalyzer::analyzeKernel(myKernel, grid, block, 0, 0, args...);

    // 2. 识别需要深入分析的核函数
    BottleneckAnalysis analysis = PerformanceAnalyzer::detectBottlenecks(metrics);

    // 3. 使用Nsight Compute进行详细分析
    if (analysis.primaryBottleneck == BottleneckType::MEMORY_BOUND) {
        std::cout << "建议使用 ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum 分析内存访问" << std::endl;
    }
}
```

通过本教学系统的性能分析工具，开发者可以：

1. **系统性地分析CUDA应用性能**
2. **自动识别性能瓶颈**
3. **获得针对性的优化建议**
4. **验证优化效果**
5. **建立性能监控体系**

这些工具为CUDA性能优化提供了科学的方法论和实用的技术手段。
