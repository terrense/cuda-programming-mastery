#include "../src/core/kernel_framework.h"
#include "../src/core/memory_management.h"
#include "../src/core/performance_analyzer.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <random>

using namespace cuda_learning;

class KernelPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA设备
        cudaError_t error = cudaSetDevice(0);
        ASSERT_EQ(error, cudaSuccess) << "Failed to set CUDA device";

        // 获取设备属性
        cudaGetDeviceProperties(&deviceProp, 0);

        // 设置测试数据大小
        testDataSize = 1024 * 1024; // 1M elements
        testBytes = testDataSize * sizeof(float);

        // 分配测试数据
        allocateTestData();
        initializeTestData();
    }

    void TearDown() override {
        // 清理测试数据
        cleanupTestData();
    }

    void allocateTestData() {
        // 主机内存
        h_input1 = new float[testDataSize];
        h_input2 = new float[testDataSize];
        h_output = new float[testDataSize];
        h_reference = new float[testDataSize];

        // 设备内存
        cudaMalloc(&d_input1, testBytes);
        cudaMalloc(&d_input2, testBytes);
        cudaMalloc(&d_output, testBytes);

        ASSERT_NE(h_input1, nullptr);
        ASSERT_NE(h_input2, nullptr);
        ASSERT_NE(h_output, nullptr);
        ASSERT_NE(h_reference, nullptr);
        ASSERT_NE(d_input1, nullptr);
        ASSERT_NE(d_input2, nullptr);
        ASSERT_NE(d_output, nullptr);
    }

    void initializeTestData() {
        // 初始化输入数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < testDataSize; i++) {
            h_input1[i] = dis(gen);
            h_input2[i] = dis(gen);
            h_reference[i] = h_input1[i] + h_input2[i]; // 参考结果
        }

        // 复制到设备
        cudaMemcpy(d_input1, h_input1, testBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, h_input2, testBytes, cudaMemcpyHostToDevice);
    }

    void cleanupTestData() {
        delete[] h_input1;
        delete[] h_input2;
        delete[] h_output;
        delete[] h_reference;

        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);
    }

    bool verifyResults(float tolerance = 1e-5f) {
        cudaMemcpy(h_output, d_output, testBytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < testDataSize; i++) {
            if (std::abs(h_output[i] - h_reference[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

protected:
    cudaDeviceProp deviceProp;
    int testDataSize;
    size_t testBytes;

    // 主机内存
    float *h_input1, *h_input2, *h_output, *h_reference;

    // 设备内存
    float *d_input1, *d_input2, *d_output;
};

// 测试核函数：合并内存访问
__global__ void coalescedAccessKernel(float* input1, float* input2, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input1[idx] + input2[idx];
    }
}

// 测试核函数：跨步内存访问
__global__ void stridedAccessKernel(float* input1, float* input2, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (strided_idx < n) {
        output[strided_idx] = input1[strided_idx] + input2[strided_idx];
    }
}

// 测试核函数：随机内存访问
__global__ void randomAccessKernel(float* input1, float* input2, float* output, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int random_idx = indices[idx];
        if (random_idx < n) {
            output[idx] = input1[random_idx] + input2[random_idx];
        }
    }
}

// 测试核函数：共享内存使用
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

// 测试核函数：寄存器密集型
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

// 内存访问模式基准测试
TEST_F(KernelPerformanceTest, MemoryAccessPatternBenchmark) {
    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};
    const int iterations = 100;

    std::cout << "\n=== 内存访问模式性能基准测试 ===" << std::endl;
    std::cout << std::setw(12) << "块大小"
              << std::setw(15) << "合并访问(ms)"
              << std::setw(15) << "跨步访问(ms)"
              << std::setw(12) << "性能比" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    for (int blockSize : blockSizes) {
        int gridSize = (testDataSize + blockSize - 1) / blockSize;

        // 测试合并访问
        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
        }
        timer.stop();
        float coalescedTime = timer.getElapsedTime() / iterations;

        // 验证结果正确性
        ASSERT_TRUE(verifyResults()) << "Coalesced access kernel produced incorrect results";

        // 测试跨步访问
        timer.start();
        for (int i = 0; i < iterations; i++) {
            stridedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize, 32);
        }
        timer.stop();
        float stridedTime = timer.getElapsedTime() / iterations;

        float performanceRatio = stridedTime / coalescedTime;

        std::cout << std::setw(12) << blockSize
                  << std::setw(15) << std::fixed << std::setprecision(3) << coalescedTime
                  << std::setw(15) << stridedTime
                  << std::setw(12) << std::setprecision(2) << performanceRatio << std::endl;

        // 性能断言：跨步访问应该明显慢于合并访问
        EXPECT_GT(performanceRatio, 1.2f) << "Strided access should be significantly slower than coalesced access";
    }
}

// 线程配置优化验证测试
TEST_F(KernelPerformanceTest, ThreadConfigurationOptimization) {
    std::cout << "\n=== 线程配置优化验证测试 ===" << std::endl;

    // 测试不同的块大小配置
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};
    std::vector<std::pair<int, float>> results;

    const int iterations = 50;

    for (int blockSize : blockSizes) {
        int gridSize = (testDataSize + blockSize - 1) / blockSize;

        // 测试性能
        float totalTime = CudaTimer::timeFunction([&]() {
            coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
        }, iterations);

        float avgTime = totalTime / iterations;
        results.push_back({blockSize, avgTime});

        // 计算占用率
        float occupancy = OccupancyCalculator::calculateActualOccupancy(
            (const void*)coalescedAccessKernel, blockSize, 0);

        std::cout << "块大小: " << std::setw(4) << blockSize
                  << ", 时间: " << std::setw(8) << std::fixed << std::setprecision(3) << avgTime << " ms"
                  << ", 占用率: " << std::setw(6) << std::setprecision(1) << occupancy << "%" << std::endl;

        // 验证结果正确性
        ASSERT_TRUE(verifyResults()) << "Kernel with block size " << blockSize << " produced incorrect results";
    }

    // 找到最佳配置
    auto bestConfig = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::cout << "最佳块大小: " << bestConfig->first
              << ", 最佳时间: " << bestConfig->second << " ms" << std::endl;

    // 验证优化器的建议
    GridBlockConfig optimizedConfig = GridBlockOptimizer::optimize1D(testDataSize);
    std::cout << "优化器建议: " << optimizedConfig.toString() << std::endl;

    // 测试优化器建议的性能
    float optimizerTime = CudaTimer::timeFunction([&]() {
        coalescedAccessKernel<<<optimizedConfig.gridDim, optimizedConfig.blockDim>>>(
            d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    std::cout << "优化器配置性能: " << optimizerTime << " ms" << std::endl;

    // 优化器的建议应该接近最佳性能
    EXPECT_LE(optimizerTime, bestConfig->second * 1.1f)
        << "Optimizer suggestion should be within 10% of best performance";
}

// 共享内存性能测试
TEST_F(KernelPerformanceTest, SharedMemoryPerformance) {
    std::cout << "\n=== 共享内存性能测试 ===" << std::endl;

    const int blockSize = 256;
    const int gridSize = (testDataSize + blockSize - 1) / blockSize;
    const int iterations = 100;

    // 测试全局内存版本
    float globalMemTime = CudaTimer::timeFunction([&]() {
        coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    // 测试共享内存版本
    size_t sharedMemSize = 2 * blockSize * sizeof(float);
    float sharedMemTime = CudaTimer::timeFunction([&]() {
        sharedMemoryKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    std::cout << "全局内存版本: " << std::fixed << std::setprecision(3) << globalMemTime << " ms" << std::endl;
    std::cout << "共享内存版本: " << sharedMemTime << " ms" << std::endl;
    std::cout << "性能比: " << std::setprecision(2) << (globalMemTime / sharedMemTime) << "x" << std::endl;

    // 验证结果正确性
    ASSERT_TRUE(verifyResults()) << "Shared memory kernel produced incorrect results";

    // 对于这个简单的例子，共享内存版本可能不会更快，但应该功能正确
    EXPECT_GT(sharedMemTime, 0.0f) << "Shared memory kernel should execute successfully";
}

// 占用率影响测试
TEST_F(KernelPerformanceTest, OccupancyImpactTest) {
    std::cout << "\n=== 占用率影响测试 ===" << std::endl;

    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};
    const int iterations = 50;

    std::cout << std::setw(12) << "块大小"
              << std::setw(15) << "占用率%"
              << std::setw(15) << "执行时间(ms)"
              << std::setw(15) << "效率指标" << std::endl;
    std::cout << std::string(57, '-') << std::endl;

    for (int blockSize : blockSizes) {
        int gridSize = (testDataSize + blockSize - 1) / blockSize;

        // 计算占用率
        float occupancy = OccupancyCalculator::calculateActualOccupancy(
            (const void*)coalescedAccessKernel, blockSize, 0);

        // 测试执行时间
        float execTime = CudaTimer::timeFunction([&]() {
            coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
        }, iterations) / iterations;

        // 计算效率指标 (占用率 / 执行时间)
        float efficiency = occupancy / execTime;

        std::cout << std::setw(12) << blockSize
                  << std::setw(15) << std::fixed << std::setprecision(1) << occupancy
                  << std::setw(15) << std::setprecision(3) << execTime
                  << std::setw(15) << std::setprecision(2) << efficiency << std::endl;

        // 验证占用率合理性
        EXPECT_GT(occupancy, 0.0f) << "Occupancy should be positive";
        EXPECT_LE(occupancy, 100.0f) << "Occupancy should not exceed 100%";
    }
}

// 内存带宽利用率测试
TEST_F(KernelPerformanceTest, MemoryBandwidthUtilization) {
    std::cout << "\n=== 内存带宽利用率测试 ===" << std::endl;

    std::vector<size_t> dataSizes = {
        1024 * 1024,      // 1MB
        4 * 1024 * 1024,  // 4MB
        16 * 1024 * 1024, // 16MB
        64 * 1024 * 1024  // 64MB
    };

    const int blockSize = 256;
    const int iterations = 50;

    std::cout << std::setw(12) << "数据大小"
              << std::setw(15) << "执行时间(ms)"
              << std::setw(15) << "带宽(GB/s)"
              << std::setw(15) << "效率%" << std::endl;
    std::cout << std::string(57, '-') << std::endl;

    for (size_t dataSize : dataSizes) {
        size_t bytes = dataSize * sizeof(float);
        int gridSize = (dataSize + blockSize - 1) / blockSize;

        // 重新分配内存以适应不同的数据大小
        float *d_test_input1, *d_test_input2, *d_test_output;
        cudaMalloc(&d_test_input1, bytes);
        cudaMalloc(&d_test_input2, bytes);
        cudaMalloc(&d_test_output, bytes);

        // 测试执行时间
        float execTime = CudaTimer::timeFunction([&]() {
            coalescedAccessKernel<<<gridSize, blockSize>>>(
                d_test_input1, d_test_input2, d_test_output, dataSize);
        }, iterations) / iterations;

        // 计算带宽 (3个数组的读写)
        float bandwidth = (3.0f * bytes) / (execTime * 1e-3) / (1024*1024*1024);

        // 计算效率 (相对于理论峰值)
        float theoreticalBW = MemoryBandwidthTester::getTheoreticalBandwidth(0);
        float efficiency = (bandwidth / theoreticalBW) * 100.0f;

        std::cout << std::setw(10) << (dataSize / (1024*1024)) << "MB"
                  << std::setw(15) << std::fixed << std::setprecision(3) << execTime
                  << std::setw(15) << std::setprecision(2) << bandwidth
                  << std::setw(15) << std::setprecision(1) << efficiency << std::endl;

        // 清理临时内存
        cudaFree(d_test_input1);
        cudaFree(d_test_input2);
        cudaFree(d_test_output);

        // 验证带宽合理性
        EXPECT_GT(bandwidth, 0.0f) << "Bandwidth should be positive";
        EXPECT_LT(bandwidth, theoreticalBW * 1.1f) << "Bandwidth should not exceed theoretical maximum";
    }
}

// 寄存器使用影响测试
TEST_F(KernelPerformanceTest, RegisterUsageImpact) {
    std::cout << "\n=== 寄存器使用影响测试 ===" << std::endl;

    const int blockSize = 256;
    const int gridSize = (testDataSize + blockSize - 1) / blockSize;
    const int iterations = 50;

    // 测试简单核函数 (低寄存器使用)
    float simpleTime = CudaTimer::timeFunction([&]() {
        coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    // 测试寄存器密集型核函数
    float registerIntensiveTime = CudaTimer::timeFunction([&]() {
        registerIntensiveKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    // 计算占用率
    float simpleOccupancy = OccupancyCalculator::calculateActualOccupancy(
        (const void*)coalescedAccessKernel, blockSize, 0);
    float registerOccupancy = OccupancyCalculator::calculateActualOccupancy(
        (const void*)registerIntensiveKernel, blockSize, 0);

    std::cout << "简单核函数:" << std::endl;
    std::cout << "  执行时间: " << std::fixed << std::setprecision(3) << simpleTime << " ms" << std::endl;
    std::cout << "  占用率: " << std::setprecision(1) << simpleOccupancy << "%" << std::endl;

    std::cout << "寄存器密集型核函数:" << std::endl;
    std::cout << "  执行时间: " << std::setprecision(3) << registerIntensiveTime << " ms" << std::endl;
    std::cout << "  占用率: " << std::setprecision(1) << registerOccupancy << "%" << std::endl;

    std::cout << "性能影响: " << std::setprecision(2) << (registerIntensiveTime / simpleTime) << "x" << std::endl;

    // 寄存器密集型核函数通常会有更低的占用率
    EXPECT_LE(registerOccupancy, simpleOccupancy * 1.1f)
        << "Register-intensive kernel should not have significantly higher occupancy";
}

// 综合性能基准测试
TEST_F(KernelPerformanceTest, ComprehensivePerformanceBenchmark) {
    std::cout << "\n=== 综合性能基准测试 ===" << std::endl;

    // 运行标准基准测试
    auto benchmarkResults = PerformanceBenchmark::runStandardBenchmarks(0);

    ASSERT_FALSE(benchmarkResults.empty()) << "Benchmark results should not be empty";

    for (const auto& result : benchmarkResults) {
        std::cout << result.toString() << std::endl;

        // 验证基准测试结果的合理性
        EXPECT_GT(result.metrics.bandwidth_GB_s, 0.0f)
            << "Bandwidth should be positive in " << result.testName;
        EXPECT_GE(result.metrics.occupancy_percent, 0.0f)
            << "Occupancy should be non-negative in " << result.testName;
        EXPECT_LE(result.metrics.occupancy_percent, 100.0f)
            << "Occupancy should not exceed 100% in " << result.testName;
    }

    // 生成基准测试报告
    std::string report = PerformanceBenchmark::generateBenchmarkReport(benchmarkResults);
    std::cout << "\n" << report << std::endl;

    EXPECT_FALSE(report.empty()) << "Benchmark report should not be empty";
}

// 性能回归测试
TEST_F(KernelPerformanceTest, PerformanceRegressionTest) {
    std::cout << "\n=== 性能回归测试 ===" << std::endl;

    const int blockSize = 256;
    const int gridSize = (testDataSize + blockSize - 1) / blockSize;
    const int iterations = 100;

    // 建立性能基线
    std::vector<float> baselineTimes;
    for (int i = 0; i < 10; i++) {
        float time = CudaTimer::timeFunction([&]() {
            coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
        }, iterations) / iterations;
        baselineTimes.push_back(time);
    }

    // 计算基线统计信息
    std::sort(baselineTimes.begin(), baselineTimes.end());
    float medianTime = baselineTimes[baselineTimes.size() / 2];
    float avgTime = std::accumulate(baselineTimes.begin(), baselineTimes.end(), 0.0f) / baselineTimes.size();

    std::cout << "基线性能:" << std::endl;
    std::cout << "  中位数时间: " << std::fixed << std::setprecision(3) << medianTime << " ms" << std::endl;
    std::cout << "  平均时间: " << avgTime << " ms" << std::endl;

    // 模拟性能测试
    float currentTime = CudaTimer::timeFunction([&]() {
        coalescedAccessKernel<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, testDataSize);
    }, iterations) / iterations;

    std::cout << "当前性能: " << currentTime << " ms" << std::endl;

    // 检测性能回归 (允许5%的波动)
    float regressionThreshold = 1.05f;
    bool hasRegression = currentTime > (medianTime * regressionThreshold);

    std::cout << "性能回归检测: " << (hasRegression ? "检测到回归" : "无回归") << std::endl;

    // 性能不应该显著退化
    EXPECT_LE(currentTime, medianTime * regressionThreshold)
        << "Performance regression detected: current time " << currentTime
        << " ms exceeds baseline " << medianTime << " ms by more than 5%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // 检查CUDA设备可用性
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Skipping tests." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    // 显示设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;

    return RUN_ALL_TESTS();
}
