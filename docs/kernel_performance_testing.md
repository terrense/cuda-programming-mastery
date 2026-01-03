# CUDA核函数性能测试指南

## 概述

本文档介绍了CUDA核函数性能测试框架的使用方法和最佳实践。该测试框架提供了全面的性能基准测试、内存访问模式验证、线程配置优化验证等功能。

## 测试框架结构

### 核心测试类

```cpp
class KernelPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override;    // 测试初始化
    void TearDown() override; // 测试清理

    // 辅助方法
    void allocateTestData();
    void initializeTestData();
    void cleanupTestData();
    bool verifyResults(float tolerance = 1e-5f);

    // 测试数据
    cudaDeviceProp deviceProp;
    int testDataSize;
    size_t testBytes;
    float *h_input1, *h_input2, *h_output, *h_reference;
    float *d_input1, *d_input2, *d_output;
};
```

## 测试用例详解

### 1. 内存访问模式基准测试

测试不同内存访问模式对性能的影响。

```cpp
TEST_F(KernelPerformanceTest, MemoryAccessPatternBenchmark) {
    // 测试合并访问 vs 跨步访问
    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

    for (int blockSize : blockSizes) {
        // 合并访问测试
        float coalescedTime = measureKernelTime(coalescedAccessKernel, blockSize);

        // 跨步访问测试
        float stridedTime = measureKernelTime(stridedAccessKernel, blockSize);

        // 性能比较
        float performanceRatio = stridedTime / coalescedTime;

        // 验证跨步访问确实更慢
        EXPECT_GT(performanceRatio, 1.2f);
    }
}
```

#### 测试核函数

```cpp
// 合并访问核函数
__global__ void coalescedAccessKernel(float* input1, float* input2, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input1[idx] + input2[idx]; // 连续访问
    }
}

// 跨步访问核函数
__global__ void stridedAccessKernel(float* input1, float* input2, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (strided_idx < n) {
        output[strided_idx] = input1[strided_idx] + input2[strided_idx]; // 跨步访问
    }
}
```

### 2. 线程配置优化验证测试

验证不同线程配置对性能和占用率的影响。

```cpp
TEST_F(KernelPerformanceTest, ThreadConfigurationOptimization) {
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};
    std::vector<std::pair<int, float>> results;

    for (int blockSize : blockSizes) {
        // 性能测试
        float avgTime = measureAverageTime(blockSize);
        results.push_back({blockSize, avgTime});

        // 占用率分析
        float occupancy = OccupancyCalculator::calculateActualOccupancy(
            (const void*)coalescedAccessKernel, blockSize, 0);

        // 验证结果正确性
        ASSERT_TRUE(verifyResults());
    }

    // 找到最佳配置
    auto bestConfig = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    // 验证优化器建议的合理性
    GridBlockConfig optimizedConfig = GridBlockOptimizer::optimize1D(testDataSize);
    float optimizerTime = measureOptimizedTime(optimizedConfig);

    EXPECT_LE(optimizerTime, bestConfig->second * 1.1f);
}
```

### 3. 共享内存性能测试

比较使用共享内存和不使用共享内存的性能差异。

```cpp
TEST_F(KernelPerformanceTest, SharedMemoryPerformance) {
    const int blockSize = 256;
    const int iterations = 100;

    // 全局内存版本
    float globalMemTime = CudaTimer::timeFunction([&]() {
        coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    // 共享内存版本
    size_t sharedMemSize = 2 * blockSize * sizeof(float);
    float sharedMemTime = CudaTimer::timeFunction([&]() {
        sharedMemoryKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    // 性能比较和验证
    ASSERT_TRUE(verifyResults());
    EXPECT_GT(sharedMemTime, 0.0f);
}
```

#### 共享内存核函数

```cpp
__global__ void sharedMemoryKernel(float* input1, float* input2, float* output, int n) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 加载到共享内存
    if (idx < n) {
        sdata[tid] = input1[idx];
        sdata[tid + blockDim.x] = input2[idx];
    } else {
        sdata[tid] = 0.0f;
        sdata[tid + blockDim.x] = 0.0f;
    }

    __syncthreads();

    // 从共享内存计算
    if (idx < n) {
        output[idx] = sdata[tid] + sdata[tid + blockDim.x];
    }
}
```

### 4. 占用率影响测试

分析不同块大小对占用率和性能的影响。

```cpp
TEST_F(KernelPerformanceTest, OccupancyImpactTest) {
    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

    for (int blockSize : blockSizes) {
        // 计算占用率
        float occupancy = OccupancyCalculator::calculateActualOccupancy(
            (const void*)coalescedAccessKernel, blockSize, 0);

        // 测试执行时间
        float execTime = measureExecutionTime(blockSize);

        // 计算效率指标
        float efficiency = occupancy / execTime;

        // 验证占用率合理性
        EXPECT_GT(occupancy, 0.0f);
        EXPECT_LE(occupancy, 100.0f);
    }
}
```

### 5. 内存带宽利用率测试

测试不同数据大小下的内存带宽利用率。

```cpp
TEST_F(KernelPerformanceTest, MemoryBandwidthUtilization) {
    std::vector<size_t> dataSizes = {
        1024 * 1024,      // 1MB
        4 * 1024 * 1024,  // 4MB
        16 * 1024 * 1024, // 16MB
        64 * 1024 * 1024  // 64MB
    };

    for (size_t dataSize : dataSizes) {
        // 测试执行时间
        float execTime = measureExecutionTimeForSize(dataSize);

        // 计算带宽
        float bandwidth = calculateBandwidth(dataSize, execTime);

        // 计算效率
        float theoreticalBW = MemoryBandwidthTester::getTheoreticalBandwidth(0);
        float efficiency = (bandwidth / theoreticalBW) * 100.0f;

        // 验证带宽合理性
        EXPECT_GT(bandwidth, 0.0f);
        EXPECT_LT(bandwidth, theoreticalBW * 1.1f);
    }
}
```

### 6. 寄存器使用影响测试

比较不同寄存器使用量对性能的影响。

```cpp
TEST_F(KernelPerformanceTest, RegisterUsageImpact) {
    // 简单核函数 (低寄存器使用)
    float simpleTime = measureKernelTime(coalescedAccessKernel);
    float simpleOccupancy = calculateOccupancy(coalescedAccessKernel);

    // 寄存器密集型核函数
    float registerIntensiveTime = measureKernelTime(registerIntensiveKernel);
    float registerOccupancy = calculateOccupancy(registerIntensiveKernel);

    // 寄存器密集型核函数通常占用率更低
    EXPECT_LE(registerOccupancy, simpleOccupancy * 1.1f);
}
```

#### 寄存器密集型核函数

```cpp
__global__ void registerIntensiveKernel(float* input1, float* input2, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 使用多个寄存器进行复杂计算
        float a = input1[idx];
        float b = input2[idx];
        float c = a * b;
        float d = a + b;
        float e = c * d;
        float f = c + d;
        float g = e * f;
        float h = e + f;
        output[idx] = g + h;
    }
}
```

### 7. 性能回归测试

检测性能是否出现回归。

```cpp
TEST_F(KernelPerformanceTest, PerformanceRegressionTest) {
    // 建立性能基线
    std::vector<float> baselineTimes;
    for (int i = 0; i < 10; i++) {
        float time = measureSingleRun();
        baselineTimes.push_back(time);
    }

    // 计算基线统计信息
    std::sort(baselineTimes.begin(), baselineTimes.end());
    float medianTime = baselineTimes[baselineTimes.size() / 2];

    // 当前性能测试
    float currentTime = measureCurrentPerformance();

    // 检测回归 (允许5%波动)
    float regressionThreshold = 1.05f;
    EXPECT_LE(currentTime, medianTime * regressionThreshold);
}
```

## 运行测试

### 使用Google Test

如果系统安装了Google Test：

```bash
# 构建测试
mkdir build && cd build
cmake ..
make kernel_performance_tests

# 运行测试
./tests/kernel_performance_tests

# 运行特定测试
./tests/kernel_performance_tests --gtest_filter="*MemoryAccessPattern*"
```

### 使用简单测试运行器

如果没有Google Test：

```bash
# 构建简单测试
make kernel_performance_tests_simple

# 运行测试
./tests/kernel_performance_tests_simple
```

## 测试结果分析

### 性能指标

测试框架会输出以下关键指标：

1. **执行时间**: 核函数执行的平均时间
2. **内存带宽**: 实际达到的内存带宽
3. **占用率**: GPU资源利用率
4. **效率比**: 相对于理论峰值的效率

### 示例输出

```
=== 内存访问模式性能基准测试 ===
    块大小    合并访问(ms)    跨步访问(ms)    性能比
------------------------------------------------------
        64           2.150           5.420      2.52
       128           1.980           4.890      2.47
       256           1.850           4.650      2.51
       512           1.920           4.780      2.49
      1024           2.100           5.100      2.43

=== 线程配置优化验证测试 ===
块大小:   64, 时间:    2.150 ms, 占用率:  50.0%
块大小:  128, 时间:    1.980 ms, 占用率:  75.0%
块大小:  256, 时间:    1.850 ms, 占用率:  87.5%
块大小:  512, 时间:    1.920 ms, 占用率:  75.0%
块大小: 1024, 时间:    2.100 ms, 占用率:  50.0%

最佳块大小: 256, 最佳时间: 1.850 ms
优化器建议: Grid: (4096), Block: (256), SharedMem: 0 bytes, TotalThreads: 1048576
优化器配置性能: 1.865 ms
```

## 最佳实践

### 1. 测试设计原则

- **隔离性**: 每个测试应该独立，不依赖其他测试的结果
- **可重复性**: 测试结果应该在相同环境下可重复
- **全面性**: 覆盖各种内存访问模式和线程配置
- **现实性**: 测试场景应该反映实际应用需求

### 2. 性能基准设定

```cpp
// 设定合理的性能期望
const float EXPECTED_BANDWIDTH_EFFICIENCY = 0.8f; // 80%带宽效率
const float EXPECTED_OCCUPANCY_THRESHOLD = 50.0f;  // 50%占用率
const float PERFORMANCE_REGRESSION_THRESHOLD = 1.05f; // 5%性能回归容忍度
```

### 3. 测试环境控制

```cpp
void setupTestEnvironment() {
    // 设置GPU频率为固定值（如果支持）
    // 禁用GPU Boost
    // 确保充足的散热
    // 关闭其他GPU应用程序
}
```

### 4. 统计分析

```cpp
struct PerformanceStatistics {
    float mean;
    float median;
    float stddev;
    float min;
    float max;

    static PerformanceStatistics calculate(const std::vector<float>& times) {
        // 计算统计指标
        // 排除异常值
        // 返回统计结果
    }
};
```

## 持续集成

### CI/CD集成

```yaml
# .github/workflows/performance-tests.yml
name: Performance Tests

on: [push, pull_request]

jobs:
  performance-tests:
    runs-on: [self-hosted, gpu]

    steps:
    - uses: actions/checkout@v2

    - name: Build tests
      run: |
        mkdir build && cd build
        cmake ..
        make kernel_performance_tests

    - name: Run performance tests
      run: |
        cd build
        ./tests/kernel_performance_tests --gtest_output=xml:performance_results.xml

    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: performance-results
        path: build/performance_results.xml
```

### 性能监控

```cpp
class PerformanceMonitor {
public:
    void recordBenchmark(const std::string& testName, float time) {
        // 记录到数据库或文件
        // 检测性能趋势
        // 发送警报（如果性能下降）
    }

    void generateTrendReport() {
        // 生成性能趋势报告
        // 可视化性能变化
    }
};
```

## 故障排除

### 常见问题

1. **测试不稳定**:
   - 检查GPU温度和频率
   - 增加测试迭代次数
   - 使用统计方法处理结果

2. **性能异常**:
   - 验证CUDA驱动版本
   - 检查系统负载
   - 确认GPU型号和配置

3. **内存错误**:
   - 检查内存分配和释放
   - 验证数据传输
   - 使用CUDA内存检查工具

### 调试技巧

```cpp
// 启用详细输出
#define VERBOSE_TESTING 1

// 使用CUDA错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 性能分析集成
void enableProfiling() {
    // 集成Nsight Compute
    // 启用详细的性能计数器
    // 生成性能报告
}
```

通过这个全面的测试框架，开发者可以系统性地验证CUDA核函数的性能特征，确保优化措施的有效性，并建立持续的性能监控体系。
