#include "../../src/core/kernel_framework.h"
#include "../../src/core/memory_management.h"
#include "../../src/core/performance_analyzer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cassert>

using namespace cuda_learning;

// 测试核函数：合并内存访问
__global__ void coalescedAccessKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// 测试核函数：跨步内存访问
__global__ void stridedAccessKernel(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (strided_idx < n) {
        output[idx] = input[strided_idx] * 2.0f + 1.0f;
    }
}

// 测试核函数：随机内存访问
__global__ void randomAccessKernel(float* input, float* output, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int random_idx = indices[idx] % n;
        output[idx] = input[random_idx] * 2.0f + 1.0f;
    }
}

// 测试核函数：非对齐内存访问
__global__ void misalignedAccessKernel(float* input, float* output, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int aligned_idx = (idx + offset) % n;
    if (aligned_idx < n) {
        output[idx] = input[aligned_idx] * 2.0f + 1.0f;
    }
}

// 测试核函数：共享内存优化访问
__global__ void sharedMemoryOptimizedKernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 合并加载到共享内存
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // 从共享内存计算
    if (idx < n) {
        output[idx] = sdata[tid] * 2.0f + 1.0f;
    }
}

// 内存访问模式基准测试类
class MemoryAccessBenchmarks {
private:
    static const int DEFAULT_DATA_SIZE = 1024 * 1024; // 1M elements
    static const int DEFAULT_ITERATIONS = 100;

    float *h_input, *h_output, *h_reference;
    float *d_input, *d_output;
    int *d_random_indices;
    size_t dataSize;
    size_t dataBytes;

public:
    MemoryAccessBenchmarks(size_t size = DEFAULT_DATA_SIZE) : dataSize(size) {
        dataBytes = dataSize * sizeof(float);
        allocateMemory();
        initializeData();
    }

    ~MemoryAccessBenchmarks() {
        cleanup();
    }

private:
    void allocateMemory() {
        // 主机内存分配
        h_input = new float[dataSize];
        h_output = new float[dataSize];
        h_reference = new float[dataSize];

        // 设备内存分配
        cudaMalloc(&d_input, dataBytes);
        cudaMalloc(&d_output, dataBytes);
        cudaMalloc(&d_random_indices, dataSize * sizeof(int));

        assert(h_input && h_output && h_reference);
        assert(d_input && d_output && d_random_indices);
    }

    void initializeData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 100.0f);

        // 初始化输入数据
        for (size_t i = 0; i < dataSize; i++) {
            h_input[i] = dis(gen);
            h_reference[i] = h_input[i] * 2.0f + 1.0f;
        }

        // 生成随机索引
        std::vector<int> randomIndices(dataSize);
        std::iota(randomIndices.begin(), randomIndices.end(), 0);
        std::shuffle(randomIndices.begin(), randomIndices.end(), gen);

        // 复制到设备
        cudaMemcpy(d_input, h_input, dataBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_random_indices, randomIndices.data(), dataSize * sizeof(int), cudaMemcpyHostToDevice);
    }

    void cleanup() {
        delete[] h_input;
        delete[] h_output;
        delete[] h_reference;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_random_indices);
    }

    bool verifyResults(float tolerance = 1e-5f) {
        cudaMemcpy(h_output, d_output, dataBytes, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < dataSize; i++) {
            if (std::abs(h_output[i] - h_reference[i]) > tolerance) {
                std::cout << "验证失败: index " << i
                         << ", expected " << h_reference[i]
                         << ", got " << h_output[i] << std::endl;
                return false;
            }
        }
        return true;
    }

public:
    // 合并访问基准测试
    void benchmarkCoalescedAccess() {
        std::cout << "\n=== 合并内存访问基准测试 ===" << std::endl;

        std::vector<int> blockSizes = {64, 128, 256, 512, 1024};

        std::cout << std::setw(12) << "块大小"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(15) << "带宽(GB/s)"
                  << std::setw(12) << "占用率%" << std::endl;
        std::cout << std::string(54, '-') << std::endl;

        for (int blockSize : blockSizes) {
            int gridSize = (dataSize + blockSize - 1) / blockSize;

            // 性能测试
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                coalescedAccessKernel<<<gridSize, blockSize>>>(d_input, d_output, dataSize);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

            // 计算带宽 (读取input + 写入output)
            float bandwidth = (2.0f * dataBytes) / (avgTime * 1e-3) / (1024*1024*1024);

            // 计算占用率
            float occupancy = OccupancyCalculator::calculateActualOccupancy(
                (const void*)coalescedAccessKernel, blockSize, 0);

            std::cout << std::setw(12) << blockSize
                      << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                      << std::setw(15) << std::setprecision(2) << bandwidth
                      << std::setw(12) << std::setprecision(1) << occupancy << std::endl;

            // 验证结果正确性
            assert(verifyResults());
        }
    }

    // 跨步访问基准测试
    void benchmarkStridedAccess() {
        std::cout << "\n=== 跨步内存访问基准测试 ===" << std::endl;

        std::vector<int> strides = {1, 2, 4, 8, 16, 32};
        const int blockSize = 256;
        const int gridSize = (dataSize + blockSize - 1) / blockSize;

        std::cout << std::setw(12) << "跨步大小"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(15) << "带宽(GB/s)"
                  << std::setw(15) << "相对性能" << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        float baselineTime = 0.0f;

        for (int stride : strides) {
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                stridedAccessKernel<<<gridSize, blockSize>>>(d_input, d_output, dataSize, stride);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;
            float bandwidth = (2.0f * dataBytes) / (avgTime * 1e-3) / (1024*1024*1024);

            if (stride == 1) {
                baselineTime = avgTime;
            }

            float relativePerf = baselineTime / avgTime;

            std::cout << std::setw(12) << stride
                      << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                      << std::setw(15) << std::setprecision(2) << bandwidth
                      << std::setw(15) << std::setprecision(2) << relativePerf << std::endl;
        }
    }

    // 随机访问基准测试
    void benchmarkRandomAccess() {
        std::cout << "\n=== 随机内存访问基准测试 ===" << std::endl;

        const int blockSize = 256;
        const int gridSize = (dataSize + blockSize - 1) / blockSize;

        // 测试合并访问作为基线
        CudaTimer timer;
        timer.start();
        for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
            coalescedAccessKernel<<<gridSize, blockSize>>>(d_input, d_output, dataSize);
        }
        timer.stop();
        float coalescedTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

        // 测试随机访问
        timer.start();
        for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
            randomAccessKernel<<<gridSize, blockSize>>>(d_input, d_output, d_random_indices, dataSize);
        }
        timer.stop();
        float randomTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

        float coalescedBW = (2.0f * dataBytes) / (coalescedTime * 1e-3) / (1024*1024*1024);
        float randomBW = (2.0f * dataBytes) / (randomTime * 1e-3) / (1024*1024*1024);

        std::cout << "合并访问:" << std::endl;
        std::cout << "  执行时间: " << std::fixed << std::setprecision(3) << coalescedTime << " ms" << std::endl;
        std::cout << "  带宽: " << std::setprecision(2) << coalescedBW << " GB/s" << std::endl;

        std::cout << "随机访问:" << std::endl;
        std::cout << "  执行时间: " << std::setprecision(3) << randomTime << " ms" << std::endl;
        std::cout << "  带宽: " << std::setprecision(2) << randomBW << " GB/s" << std::endl;

        std::cout << "性能比 (合并/随机): " << std::setprecision(2) << (randomTime / coalescedTime) << "x" << std::endl;
    }

    // 内存对齐影响测试
    void benchmarkMemoryAlignment() {
        std::cout << "\n=== 内存对齐影响基准测试 ===" << std::endl;

        std::vector<int> offsets = {0, 1, 2, 4, 8, 16};
        const int blockSize = 256;
        const int gridSize = (dataSize + blockSize - 1) / blockSize;

        std::cout << std::setw(12) << "偏移量"
                  << std::setw(15) << "执行时间(ms)"
                  << std::setw(15) << "带宽(GB/s)"
                  << std::setw(15) << "相对性能" << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        float baselineTime = 0.0f;

        for (int offset : offsets) {
            CudaTimer timer;
            timer.start();
            for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
                misalignedAccessKernel<<<gridSize, blockSize>>>(d_input, d_output, dataSize, offset);
            }
            timer.stop();

            float avgTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;
            float bandwidth = (2.0f * dataBytes) / (avgTime * 1e-3) / (1024*1024*1024);

            if (offset == 0) {
                baselineTime = avgTime;
            }

            float relativePerf = baselineTime / avgTime;

            std::cout << std::setw(12) << offset
                      << std::setw(15) << std::fixed << std::setprecision(3) << avgTime
                      << std::setw(15) << std::setprecision(2) << bandwidth
                      << std::setw(15) << std::setprecision(2) << relativePerf << std::endl;
        }
    }

    // 共享内存优化效果测试
    void benchmarkSharedMemoryOptimization() {
        std::cout << "\n=== 共享内存优化效果基准测试 ===" << std::endl;

        const int blockSize = 256;
        const int gridSize = (dataSize + blockSize - 1) / blockSize;
        const size_t sharedMemSize = blockSize * sizeof(float);

        // 测试全局内存版本
        CudaTimer timer;
        timer.start();
        for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
            coalescedAccessKernel<<<gridSize, blockSize>>>(d_input, d_output, dataSize);
        }
        timer.stop();
        float globalTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

        // 测试共享内存版本
        timer.start();
        for (int i = 0; i < DEFAULT_ITERATIONS; i++) {
            sharedMemoryOptimizedKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, dataSize);
        }
        timer.stop();
        float sharedTime = timer.getElapsedTime() / DEFAULT_ITERATIONS;

        float globalBW = (2.0f * dataBytes) / (globalTime * 1e-3) / (1024*1024*1024);
        float sharedBW = (2.0f * dataBytes) / (sharedTime * 1e-3) / (1024*1024*1024);

        std::cout << "全局内存版本:" << std::endl;
        std::cout << "  执行时间: " << std::fixed << std::setprecision(3) << globalTime << " ms" << std::endl;
        std::cout << "  带宽: " << std::setprecision(2) << globalBW << " GB/s" << std::endl;

        std::cout << "共享内存版本:" << std::endl;
        std::cout << "  执行时间: " << std::setprecision(3) << sharedTime << " ms" << std::endl;
        std::cout << "  带宽: " << std::setprecision(2) << sharedBW << " GB/s" << std::endl;

        std::cout << "性能比 (全局/共享): " << std::setprecision(2) << (globalTime / sharedTime) << "x" << std::endl;

        // 验证结果正确性
        assert(verifyResults());
    }

    // 运行所有内存访问模式基准测试
    void runAllBenchmarks() {
        std::cout << "开始内存访问模式基准测试..." << std::endl;
        std::cout << "数据大小: " << (dataSize / (1024*1024)) << " MB" << std::endl;
        std::cout << "迭代次数: " << DEFAULT_ITERATIONS << std::endl;

        benchmarkCoalescedAccess();
        benchmarkStridedAccess();
        benchmarkRandomAccess();
        benchmarkMemoryAlignment();
        benchmarkSharedMemoryOptimization();

        std::cout << "\n内存访问模式基准测试完成!" << std::endl;
    }
};

// 主函数用于运行基准测试
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
    std::cout << "全局内存: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    std::cout << "共享内存/块: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;

    try {
        // 运行内存访问模式基准测试
        MemoryAccessBenchmarks benchmarks;
        benchmarks.runAllBenchmarks();

        std::cout << "\n所有测试成功完成!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
}
