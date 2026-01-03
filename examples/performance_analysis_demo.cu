#include "../src/core/performance_analyzer.h"
#include "../src/core/kernel_framework.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cuda_learning;

// 示例核函数：向量加法
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 示例核函数：矩阵乘法（未优化）
__global__ void matrixMulNaive(float* A, float* B, float* C, int M, int N, int K) {
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

// 示例核函数：矩阵乘法（共享内存优化）
__global__ void matrixMulOptimized(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 15) / 16; tile++) {
        // 加载数据到共享内存
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

        // 计算部分结果
        for (int k = 0; k < 16; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void demonstrateCudaTimer() {
    std::cout << "\n=== CUDA计时器演示 ===" << std::endl;

    const int n = 1000000;
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 初始化数据
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);

    // 使用CudaTimer测试不同块大小的性能
    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

    std::cout << std::setw(12) << "块大小" << std::setw(15) << "执行时间(ms)"
              << std::setw(15) << "带宽(GB/s)" << std::endl;
    std::cout << std::string(42, '-') << std::endl;

    for (int blockSize : blockSizes) {
        int gridSize = (n + blockSize - 1) / blockSize;

        CudaTimer timer;
        timer.start();

        const int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        }

        timer.stop();
        float avgTime = timer.getElapsedTime() / iterations;

        // 计算带宽 (3个数组 * 4字节 * n个元素)
        float bandwidth = (3.0f * sizeof(float) * n) / (avgTime * 1e-3) / (1024*1024*1024);

        std::cout << std::setw(12) << blockSize
                  << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                  << std::setw(15) << std::setprecision(2) << bandwidth << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void demonstrateOccupancyCalculator() {
    std::cout << "\n=== 占用率计算器演示 ===" << std::endl;

    // 获取向量加法核函数的最优块大小
    int optimalBlockSize = OccupancyCalculator::getOptimalBlockSize((const void*)vectorAdd);
    std::cout << "向量加法最优块大小: " << optimalBlockSize << std::endl;

    // 分析不同块大小的占用率
    std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

    std::cout << std::setw(12) << "块大小" << std::setw(15) << "理论占用率%"
              << std::setw(15) << "实际占用率%" << std::endl;
    std::cout << std::string(42, '-') << std::endl;

    for (int blockSize : blockSizes) {
        float theoretical = OccupancyCalculator::calculateTheoreticalOccupancy(
            blockSize, 16, 0); // 假设16个寄存器，无共享内存

        float actual = OccupancyCalculator::calculateActualOccupancy(
            (const void*)vectorAdd, blockSize, 0);

        std::cout << std::setw(12) << blockSize
                  << std::setw(15) << std::fixed << std::setprecision(1) << theoretical
                  << std::setw(15) << actual << std::endl;
    }

    // 分析占用率限制因素
    std::cout << "\n占用率限制分析 (块大小=256):" << std::endl;
    std::string analysis = OccupancyCalculator::analyzeOccupancyLimiters(256, 16, 1024);
    std::cout << analysis << std::endl;
}

void demonstrateBottleneckDetection() {
    std::cout << "\n=== 性能瓶颈检测演示 ===" << std::endl;

    // 模拟不同的性能场景
    std::vector<std::pair<std::string, PerformanceMetrics>> scenarios = {
        {"内存受限场景", PerformanceMetrics()},
        {"计算受限场景", PerformanceMetrics()},
        {"占用率受限场景", PerformanceMetrics()},
        {"寄存器受限场景", PerformanceMetrics()}
    };

    // 设置内存受限场景
    scenarios[0].second.memoryUtilization_percent = 90.0f;
    scenarios[0].second.computeUtilization_percent = 30.0f;
    scenarios[0].second.occupancy_percent = 75.0f;
    scenarios[0].second.bandwidth_GB_s = 450.0f;

    // 设置计算受限场景
    scenarios[1].second.memoryUtilization_percent = 40.0f;
    scenarios[1].second.computeUtilization_percent = 95.0f;
    scenarios[1].second.occupancy_percent = 80.0f;
    scenarios[1].second.bandwidth_GB_s = 200.0f;

    // 设置占用率受限场景
    scenarios[2].second.memoryUtilization_percent = 60.0f;
    scenarios[2].second.computeUtilization_percent = 60.0f;
    scenarios[2].second.occupancy_percent = 25.0f;
    scenarios[2].second.bandwidth_GB_s = 300.0f;

    // 设置寄存器受限场景
    scenarios[3].second.memoryUtilization_percent = 50.0f;
    scenarios[3].second.computeUtilization_percent = 50.0f;
    scenarios[3].second.occupancy_percent = 40.0f;
    scenarios[3].second.registersPerThread = 48;
    scenarios[3].second.bandwidth_GB_s = 250.0f;

    for (const auto& scenario : scenarios) {
        std::cout << "\n--- " << scenario.first << " ---" << std::endl;

        BottleneckAnalysis analysis = PerformanceAnalyzer::detectBottlenecks(scenario.second);
        std::cout << analysis.toString() << std::endl;

        // 生成优化建议
        auto suggestions = PerformanceAnalyzer::generateOptimizationSuggestions(
            scenario.second, analysis);

        if (!suggestions.empty()) {
            std::cout << "优化建议:" << std::endl;
            for (size_t i = 0; i < suggestions.size() && i < 2; i++) {
                std::cout << "  " << (i+1) << ". " << suggestions[i].title
                          << " (预期提升: " << suggestions[i].expectedImprovement << "%)" << std::endl;
            }
        }
    }
}

void demonstrateMatrixMultiplicationComparison() {
    std::cout << "\n=== 矩阵乘法性能比较演示 ===" << std::endl;

    const int M = 512, N = 512, K = 512;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // 初始化数据
    cudaMemset(d_A, 1, bytes_A);
    cudaMemset(d_B, 1, bytes_B);

    // 测试未优化版本
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    CudaTimer timer;

    // 未优化版本
    timer.start();
    const int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop();
    float naiveTime = timer.getElapsedTime() / iterations;

    // 优化版本
    timer.start();
    for (int i = 0; i < iterations; i++) {
        matrixMulOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop();
    float optimizedTime = timer.getElapsedTime() / iterations;

    // 计算GFLOPS
    long long flops = 2LL * M * N * K; // 每个元素需要K次乘加操作
    float naiveGFLOPS = (flops / (naiveTime * 1e-3)) / 1e9;
    float optimizedGFLOPS = (flops / (optimizedTime * 1e-3)) / 1e9;

    std::cout << "矩阵大小: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "未优化版本:" << std::endl;
    std::cout << "  执行时间: " << std::fixed << std::setprecision(3) << naiveTime << " ms" << std::endl;
    std::cout << "  性能: " << std::setprecision(2) << naiveGFLOPS << " GFLOPS" << std::endl;

    std::cout << "优化版本:" << std::endl;
    std::cout << "  执行时间: " << std::setprecision(3) << optimizedTime << " ms" << std::endl;
    std::cout << "  性能: " << std::setprecision(2) << optimizedGFLOPS << " GFLOPS" << std::endl;

    std::cout << "性能提升: " << std::setprecision(1) << (optimizedGFLOPS / naiveGFLOPS) << "x" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void demonstrateMemoryBandwidthAnalysis() {
    std::cout << "\n=== 内存带宽分析演示 ===" << std::endl;

    std::vector<size_t> dataSizes = {
        1 * 1024 * 1024,    // 1MB
        16 * 1024 * 1024,   // 16MB
        64 * 1024 * 1024,   // 64MB
        256 * 1024 * 1024   // 256MB
    };

    std::cout << std::setw(12) << "数据大小" << std::setw(15) << "带宽(GB/s)"
              << std::setw(15) << "效率%" << std::endl;
    std::cout << std::string(42, '-') << std::endl;

    for (size_t dataSize : dataSizes) {
        float bandwidth = MemoryBandwidthAnalyzer::measureBandwidth(dataSize);
        float efficiency = MemoryBandwidthAnalyzer::analyzeMemoryEfficiency(bandwidth);

        std::cout << std::setw(10) << (dataSize / (1024*1024)) << "MB"
                  << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth
                  << std::setw(15) << std::setprecision(1) << efficiency << std::endl;
    }
}

void demonstratePerformanceBenchmark() {
    std::cout << "\n=== 性能基准测试演示 ===" << std::endl;

    // 运行标准基准测试
    auto results = PerformanceBenchmark::runStandardBenchmarks(0);

    for (const auto& result : results) {
        std::cout << "\n" << result.toString() << std::endl;
    }

    // 生成基准测试报告
    std::string report = PerformanceBenchmark::generateBenchmarkReport(results);
    std::cout << "\n=== 基准测试报告 ===" << std::endl;
    std::cout << report << std::endl;
}

void generateComprehensivePerformanceReport() {
    std::cout << "\n=== 综合性能报告生成 ===" << std::endl;

    // 创建示例性能指标
    PerformanceMetrics metrics;
    metrics.executionTime_ms = 15.5f;
    metrics.bandwidth_GB_s = 320.0f;
    metrics.occupancy_percent = 65.0f;
    metrics.activeWarps = 26;
    metrics.maxWarps = 40;
    metrics.registersPerThread = 24;
    metrics.sharedMemoryUsed = 2048;
    metrics.sharedMemoryAvailable = 49152;
    metrics.computeUtilization_percent = 70.0f;
    metrics.memoryUtilization_percent = 85.0f;

    // 检测瓶颈
    BottleneckAnalysis bottlenecks = PerformanceAnalyzer::detectBottlenecks(metrics);

    // 生成优化建议
    auto suggestions = PerformanceAnalyzer::generateOptimizationSuggestions(metrics, bottlenecks);

    // 生成完整报告
    std::string report = PerformanceAnalyzer::generatePerformanceReport(metrics, bottlenecks, suggestions);

    std::cout << report << std::endl;
}

int main() {
    // 初始化CUDA
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "CUDA初始化失败: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "=== CUDA性能分析和优化工具演示 ===" << std::endl;

    try {
        // 1. CUDA计时器演示
        demonstrateCudaTimer();

        // 2. 占用率计算器演示
        demonstrateOccupancyCalculator();

        // 3. 性能瓶颈检测演示
        demonstrateBottleneckDetection();

        // 4. 矩阵乘法性能比较
        demonstrateMatrixMultiplicationComparison();

        // 5. 内存带宽分析
        demonstrateMemoryBandwidthAnalysis();

        // 6. 性能基准测试
        demonstratePerformanceBenchmark();

        // 7. 综合性能报告
        generateComprehensivePerformanceReport();

    } catch (const std::exception& e) {
        std::cerr << "演示过程中发生错误: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n性能分析和优化工具演示完成!" << std::endl;
    std::cout << "建议：结合CUDA Profiler (Nsight Compute/Systems) 进行更深入的分析" << std::endl;

    return 0;
}
