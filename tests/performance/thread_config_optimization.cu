#include "../../src/core/kernel_framework.h"
#include "../../src/core/performance_analyzer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <map>

using namespace cuda_learning;

// 测试核函数：简单向量加法
__global__ void vectorAddKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 测试核函数：矩阵乘法 (简化版)
__global__ void matrixMulKernel(float* a, float* b, float* c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

// 测试核函数：归约操作
__global__ void reductionKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 归约操作
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
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

// 测试核函数：计算密集型操作
__global__ void computeIntensiveKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // 执行多次浮点运算
        for (int i = 0; i < 100; i++) {
            x = x * 1.001f + 0.001f;
            x = sqrtf(x);
            x = x * x;
        }
        output[idx] = x;
    }
}

// 线程配置优化验证测试类
class ThreadConfigOptimizationTests {
private:
    static const int DEFAULT_DATA_SIZE = 1024 * 1024; // 1M elements
    static const int DEFAULT_ITERATIONS = 50;

    float *h_a, *h_b, *h_c, *h_reference;
    float *d_a, *d_b, *d_c;
    size_t dataSize;
    size_t dataBytes;

public:
    ThreadConfigOptimizationTests(size_t size = DEFAULT_DATA_SIZE) : dataSize(size) {
        dataBytes = dataSize * sizeof(float);
        allocateMemory();
        initializeData();
    }

    ~ThreadConfigOptimizationTests() {
        cleanup();
    }

private:
    void allocateMemory() {
        // 主机内存分配
        h_a = new float[dataSize];
        h_b = new float[dataSize];
        h_c = new float[dataSize];
        h_reference = new float[dataSize];

        // 设备内存分配
        cudaMalloc(&d_a, dataBytes);
        cudaMalloc(&d_b, dataBytes);
        cudaMalloc(&d_c, dataBytes);

        assert(h_a && h_b && h_c && h_reference);
        assert(d_a && d_b && d_c);
    }

    void initializeData() {
        // 初始化测试数据
        for (size_t i = 0; i < dataSize; i++) {
            h_a[i] = static_cast<float>(i % 100) / 10.0f;
            h_b[i] = static_cast<float>((i + 1) % 100) / 10.0f;
            h_reference[i] = h_a[i] + h_b[i];
        }

        // 复制到设备
        cudaMemcpy(d_a, h_a, dataBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, dataBytes, cudaMemcpyHostToDevice);
    }

    void cleanup() {
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        delete[] h_reference;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    bool verifyResults(float tolerance = 1e-5f) {
        cudaMemcpy(h_c, d_c, dataBytes, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < dataSize; i++) {
            if (std::abs(h_c[i] - h_reference[i]) > tolerance) {
                std::cout << "验证失败: index " << i
                         << ", expected " << h_reference[i]
                         << ", got " << h_c[i] << std::endl;
                return false;
            }
        }
        return true;
    }

public:
    // 块大小优化测试
    void testBlockSizeOptimization() {
        std::cout << "\n=== 块大小优化测试 ===" << std::endl;

        std::vector<int> blockSizes = {32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 768, 896, 1024};
        std::vector<std::pair<int, float>> results;

        std::cout << std::setw(12) << "块大小"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(12) << "占用率%"
                  << std::setw(15) << "有效带宽"
                  << std::setw(12) << "效率指标" << std::endl;
        std::cout << std::string(66, '-') << std::endl;

        for (int blockSize : blockSizes) {
            // 检查块大小是否有效
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            if (blockSize > prop.maxThreadsPerBlock) {
                continue;
            }

            int gridSize = (dataSize + blockSize - 1) / blockSize;

            // 性能测试
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;
            results.push_back({blockSize, avgTime});

            // 计算占用率
            float occupancy = OccupancyCalculator::calculateActualOccupancy(
                (const void*)vectorAddKernel, blockSize, 0);

            // 计算有效带宽
            float bandwidth = (3.0f * dataBytes) / (avgTime * 1e-3) / (1024*1024*1024);

            // 计算效率指标 (占用率 * 带宽)
            float efficiency = occupancy * bandwidth / 100.0f;

            std::cout << std::setw(12) << blockSize
                      << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                      << std::setw(12) << std::setprecision(1) << occupancy
                      << std::setw(15) << std::setprecision(2) << bandwidth
                      << std::setw(12) << std::setprecision(2) << efficiency << std::endl;

            // 验证结果正确性
            assert(verifyResults());
        }

        // 找到最佳配置
        auto bestConfig = std::min_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        std::cout << "\n最佳块大小: " << bestConfig->first
                  << ", 最佳时间: " << std::fixed << std::setprecision(3) << bestConfig->second << " ms" << std::endl;

        // 测试优化器建议
        GridBlockConfig optimizedConfig = GridBlockOptimizer::optimize1D(dataSize);
        std::cout << "优化器建议: " << optimizedConfig.toString() << std::endl;
    }

    // 网格大小优化测试
    void testGridSizeOptimization() {
        std::cout << "\n=== 网格大小优化测试 ===" << std::endl;

        const int blockSize = 256;
        std::vector<int> gridMultipliers = {1, 2, 4, 8, 16, 32};

        std::cout << std::setw(12) << "网格大小"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(15) << "GPU利用率%"
                  << std::setw(15) << "线程效率%" << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int multiplier : gridMultipliers) {
            int gridSize = ((dataSize + blockSize - 1) / blockSize) * multiplier;

            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

            // 计算GPU利用率 (实际工作线程 / 总启动线程)
            int totalThreads = gridSize * blockSize;
            float utilization = (float)dataSize / totalThreads * 100.0f;

            // 计算线程效率
            float threadEfficiency = std::min(100.0f, utilization);

            std::cout << std::setw(12) << gridSize
                      << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                      << std::setw(15) << std::setprecision(1) << utilization
                      << std::setw(15) << std::setprecision(1) << threadEfficiency << std::endl;

            // 验证结果正确性
            assert(verifyResults());
        }
    }

    // 2D线程配置优化测试
    void test2DThreadConfiguration() {
        std::cout << "\n=== 2D线程配置优化测试 ===" << std::endl;

        // 为矩阵乘法测试不同的2D块配置
        int matrixSize = 512; // 512x512 矩阵
        size_t matrixBytes = matrixSize * matrixSize * sizeof(float);

        float *d_mat_a, *d_mat_b, *d_mat_c;
        cudaMalloc(&d_mat_a, matrixBytes);
        cudaMalloc(&d_mat_b, matrixBytes);
        cudaMalloc(&d_mat_c, matrixBytes);

        // 初始化矩阵数据
        std::vector<float> h_mat_a(matrixSize * matrixSize, 1.0f);
        std::vector<float> h_mat_b(matrixSize * matrixSize, 1.0f);
        cudaMemcpy(d_mat_a, h_mat_a.data(), matrixBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mat_b, h_mat_b.data(), matrixBytes, cudaMemcpyHostToDevice);

        std::vector<std::pair<int, int>> blockConfigs = {
            {8, 8}, {16, 16}, {32, 32}, {8, 32}, {32, 8}, {16, 32}, {32, 16}
        };

        std::cout << std::setw(15) << "块配置"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(12) << "占用率%"
                  << std::setw(15) << "GFLOPS" << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (auto config : blockConfigs) {
            int blockX = config.first;
            int blockY = config.second;

            // 检查块大小是否有效
            if (blockX * blockY > 1024) continue;

            dim3 blockDim(blockX, blockY);
            dim3 gridDim((matrixSize + blockX - 1) / blockX, (matrixSize + blockY - 1) / blockY);

            CudaTimer timer;
            timer.start();
            for (int i = 0; i < 10; i++) { // 减少迭代次数，因为矩阵乘法较慢
                matrixMulKernel<<<gridDim, blockDim>>>(d_mat_a, d_mat_b, d_mat_c, matrixSize);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / 10;

            // 计算占用率
            float occupancy = OccupancyCalculator::calculateActualOccupancy(
                (const void*)matrixMulKernel, blockX * blockY, 0);

            // 计算GFLOPS (2*N^3 operations for N×N matrix multiplication)
            long long operations = 2LL * matrixSize * matrixSize * matrixSize;
            float gflops = operations / (avgTime * 1e-3) / 1e9;

            std::cout << std::setw(6) << blockX << "x" << std::setw(6) << blockY
                      << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                      << std::setw(12) << std::setprecision(1) << occupancy
                      << std::setw(15) << std::setprecision(2) << gflops << std::endl;
        }

        cudaFree(d_mat_a);
        cudaFree(d_mat_b);
        cudaFree(d_mat_c);
    }

    // 共享内存使用对线程配置的影响测试
    void testSharedMemoryImpactOnThreadConfig() {
        std::cout << "\n=== 共享内存对线程配置影响测试 ===" << std::endl;

        std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

        std::cout << std::setw(12) << "块大小"
                  << std::setw(15) << "无共享内存"
                  << std::setw(15) << "使用共享内存"
                  << std::setw(12) << "性能比"
                  << std::setw(12) << "占用率%" << std::endl;
        std::cout << std::string(66, '-') << std::endl;

        for (int blockSize : blockSizes) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            if (blockSize > prop.maxThreadsPerBlock) continue;

            int gridSize = (dataSize + blockSize - 1) / blockSize;
            size_t sharedMemSize = blockSize * sizeof(float);

            // 测试无共享内存版本
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
            }
            timer.stop();
            float noSharedTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

            // 测试使用共享内存版本 (归约操作)
            float *d_temp_output;
            cudaMalloc(&d_temp_output, gridSize * sizeof(float));

            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                reductionKernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_temp_output, dataSize);
            }
            timer.stop();
            float sharedTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

            // 计算占用率 (使用共享内存时)
            float occupancy = OccupancyCalculator::calculateActualOccupancy(
                (const void*)reductionKernel, blockSize, sharedMemSize);

            float performanceRatio = noSharedTime / sharedTime;

            std::cout << std::setw(12) << blockSize
                      << std::setw(15) << std::fixed << std::setprecision(3) << noSharedTime
                      << std::setw(15) << sharedTime
                      << std::setw(12) << std::setprecision(2) << performanceRatio
                      << std::setw(12) << std::setprecision(1) << occupancy << std::endl;

            cudaFree(d_temp_output);
        }
    }

    // 计算密集型 vs 内存密集型核函数的线程配置优化
    void testComputeVsMemoryBoundOptimization() {
        std::cout << "\n=== 计算密集型 vs 内存密集型线程配置优化 ===" << std::endl;

        std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

        std::cout << std::setw(12) << "块大小"
                  << std::setw(15) << "内存密集型"
                  << std::setw(15) << "计算密集型"
                  << std::setw(15) << "最优比例" << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int blockSize : blockSizes) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            if (blockSize > prop.maxThreadsPerBlock) continue;

            int gridSize = (dataSize + blockSize - 1) / blockSize;

            // 测试内存密集型 (向量加法)
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
            }
            timer.stop();
            float memoryBoundTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

            // 测试计算密集型
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                computeIntensiveKernel<<<gridSize, blockSize>>>(d_a, d_c, dataSize);
            }
            timer.stop();
            float computeBoundTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

            float optimalRatio = computeBoundTime / memoryBoundTime;

            std::cout << std::setw(12) << blockSize
                      << std::setw(15) << std::fixed << std::setprecision(3) << memoryBoundTime
                      << std::setw(15) << computeBoundTime
                      << std::setw(15) << std::setprecision(2) << optimalRatio << std::endl;
        }
    }

    // 占用率与性能关系验证
    void testOccupancyPerformanceRelationship() {
        std::cout << "\n=== 占用率与性能关系验证 ===" << std::endl;

        std::vector<int> blockSizes = {32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024};
        std::vector<std::pair<float, float>> occupancyPerformance;

        std::cout << std::setw(12) << "块大小"
                  << std::setw(12) << "占用率%"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(15) << "性能指标"
                  << std::setw(15) << "效率比" << std::endl;
        std::cout << std::string(69, '-') << std::endl;

        float baselinePerformance = 0.0f;

        for (int blockSize : blockSizes) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            if (blockSize > prop.maxThreadsPerBlock) continue;

            int gridSize = (dataSize + blockSize - 1) / blockSize;

            // 性能测试
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;
            float performance = 1.0f / avgTime; // 性能指标 (1/时间)

            if (baselinePerformance == 0.0f) {
                baselinePerformance = performance;
            }

            // 计算占用率
            float occupancy = OccupancyCalculator::calculateActualOccupancy(
                (const void*)vectorAddKernel, blockSize, 0);

            float efficiencyRatio = performance / baselinePerformance;

            occupancyPerformance.push_back({occupancy, performance});

            std::cout << std::setw(12) << blockSize
                      << std::setw(12) << std::fixed << std::setprecision(1) << occupancy
                      << std::setw(15) << std::setprecision(3) << avgTime
                      << std::setw(15) << std::setprecision(2) << performance
                      << std::setw(15) << std::setprecision(2) << efficiencyRatio << std::endl;
        }

        // 分析占用率与性能的相关性
        if (occupancyPerformance.size() >= 3) {
            // 简单的相关性分析
            float avgOccupancy = 0.0f, avgPerformance = 0.0f;
            for (const auto& pair : occupancyPerformance) {
                avgOccupancy += pair.first;
                avgPerformance += pair.second;
            }
            avgOccupancy /= occupancyPerformance.size();
            avgPerformance /= occupancyPerformance.size();

            float correlation = 0.0f;
            float occVariance = 0.0f, perfVariance = 0.0f;

            for (const auto& pair : occupancyPerformance) {
                float occDiff = pair.first - avgOccupancy;
                float perfDiff = pair.second - avgPerformance;
                correlation += occDiff * perfDiff;
                occVariance += occDiff * occDiff;
                perfVariance += perfDiff * perfDiff;
            }

            if (occVariance > 0 && perfVariance > 0) {
                correlation /= sqrt(occVariance * perfVariance);
                std::cout << "\n占用率与性能相关系数: " << std::fixed << std::setprecision(3) << correlation << std::endl;

                if (correlation > 0.7f) {
                    std::cout << "结论: 占用率与性能呈强正相关" << std::endl;
                } else if (correlation > 0.3f) {
                    std::cout << "结论: 占用率与性能呈中等正相关" << std::endl;
                } else {
                    std::cout << "结论: 占用率与性能相关性较弱" << std::endl;
                }
            }
        }
    }

    // 运行所有线程配置优化测试
    void runAllTests() {
        std::cout << "开始线程配置优化验证测试..." << std::endl;
        std::cout << "数据大小: " << (dataSize / (1024*1024)) << " MB" << std::endl;
        std::cout << "迭代次数: " << DEFAULT_ITERATIONS << std::endl;

        testBlockSizeOptimization();
        testGridSizeOptimization();
        test2DThreadConfiguration();
        testSharedMemoryImpactOnThreadConfig();
        testComputeVsMemoryBoundOptimization();
        testOccupancyPerformanceRelationship();

        std::cout << "\n线程配置优化验证测试完成!" << std::endl;
    }
};

// 主函数用于运行测试
int main() {
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "未找到CUDA设备，跳过测试。" << std::endl;
        return 0;
    }

    // 显示设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "使用设备: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "最大线程/块: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "最大块维度: (" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "最大网格维度: (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

    try {
        // 运行线程配置优化测试
        ThreadConfigOptimizationTests tests;
        tests.runAllTests();

        std::cout << "\n所有测试成功完成!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
}
