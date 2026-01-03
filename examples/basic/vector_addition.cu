/**
 * 向量加法完整示例和变体
 * 
 * 这是CUDA编程中最经典的示例，展示：
 * 1. 基本向量加法实现
 * 2. 错误检查和性能测量
 * 3. 不同的实现变体
 * 4. 优化技巧和最佳实践
 * 
 * 学习目标：
 * - 掌握CUDA并行计算的基本模式
 * - 学会性能测量和优化
 * - 理解内存访问模式的重要性
 * - 掌握错误处理的最佳实践
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cassert>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * 基本向量加法核函数
 * C[i] = A[i] + B[i]
 */
__global__ void vectorAddBasic(const float* A, const float* B, float* C, int N) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * 优化版本1：每个线程处理多个元素
 * 减少线程启动开销，提高内存带宽利用率
 */
__global__ void vectorAddOptimized1(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 每个线程处理多个元素
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

/**
 * 优化版本2：使用向量化加载
 * 利用float4进行向量化内存访问
 */
__global__ void vectorAddOptimized2(const float4* A, const float4* B, float4* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float4 a = A[idx];
        float4 b = B[idx];
        float4 c;
        
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        
        C[idx] = c;
    }
}

/**
 * 优化版本3：使用共享内存（演示用途）
 * 注意：对于向量加法，共享内存通常不会提升性能
 * 但这里展示如何使用共享内存
 */
__global__ void vectorAddWithSharedMem(const float* A, const float* B, float* C, int N) {
    extern __shared__ float shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 加载数据到共享内存
    if (idx < N) {
        shared_data[tid] = A[idx];
        shared_data[tid + blockDim.x] = B[idx];
    }
    
    __syncthreads();
    
    // 从共享内存计算结果
    if (idx < N) {
        C[idx] = shared_data[tid] + shared_data[tid + blockDim.x];
    }
}

class VectorAdditionDemo {
public:
    VectorAdditionDemo(int size) : N(size) {
        // 分配主机内存
        h_A.resize(N);
        h_B.resize(N);
        h_C.resize(N);
        h_C_ref.resize(N);
        
        // 初始化数据
        initializeData();
        
        // 分配设备内存
        size_t bytes = N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        
        // 复制数据到设备
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    }
    
    ~VectorAdditionDemo() {
        // 释放设备内存
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    }
    
    void runAllTests() {
        std::cout << "=== 向量加法完整示例 ===" << std::endl;
        std::cout << "向量大小: " << N << " 个元素" << std::endl;
        std::cout << "数据大小: " << (N * sizeof(float) / 1024.0 / 1024.0) << " MB\n" << std::endl;
        
        // 计算CPU参考结果
        computeCPUReference();
        
        // 测试不同的CUDA实现
        testBasicVersion();
        testOptimizedVersion1();
        testOptimizedVersion2();
        testSharedMemoryVersion();
        
        // 性能对比
        performanceComparison();
    }

private:
    int N;
    std::vector<float> h_A, h_B, h_C, h_C_ref;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    void initializeData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int i = 0; i < N; ++i) {
            h_A[i] = dis(gen);
            h_B[i] = dis(gen);
        }
    }
    
    void computeCPUReference() {
        std::cout << "计算CPU参考结果..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < N; ++i) {
            h_C_ref[i] = h_A[i] + h_B[i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "CPU执行时间: " << duration.count() / 1000.0 << " ms\n" << std::endl;
    }
    
    void testBasicVersion() {
        std::cout << "--- 测试基本版本 ---" << std::endl;
        
        // 配置执行参数
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        std::cout << "执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
        
        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // 记录开始时间
        CUDA_CHECK(cudaEventRecord(start));
        
        // 启动核函数
        vectorAddBasic<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        
        // 记录结束时间
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        // 计算执行时间
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // 检查结果
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool correct = verifyResult();
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        std::cout << "结果正确性: " << (correct ? "✓ 正确" : "✗ 错误") << std::endl;
        std::cout << "带宽: " << calculateBandwidth(milliseconds) << " GB/s\n" << std::endl;
        
        // 清理事件
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void testOptimizedVersion1() {
        std::cout << "--- 测试优化版本1 (网格跨步循环) ---" << std::endl;
        
        // 使用较少的线程块，让每个线程处理更多元素
        int blockSize = 256;
        int gridSize = std::min((N + blockSize - 1) / blockSize, 65535);
        
        std::cout << "执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
        std::cout << "每个线程平均处理: " << (float)N / (gridSize * blockSize) << " 个元素" << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        vectorAddOptimized1<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool correct = verifyResult();
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        std::cout << "结果正确性: " << (correct ? "✓ 正确" : "✗ 错误") << std::endl;
        std::cout << "带宽: " << calculateBandwidth(milliseconds) << " GB/s\n" << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void testOptimizedVersion2() {
        std::cout << "--- 测试优化版本2 (向量化访问) ---" << std::endl;
        
        // 确保N是4的倍数
        if (N % 4 != 0) {
            std::cout << "跳过：向量大小不是4的倍数\n" << std::endl;
            return;
        }
        
        int vectorN = N / 4;
        int blockSize = 256;
        int gridSize = (vectorN + blockSize - 1) / blockSize;
        
        std::cout << "执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
        std::cout << "处理 " << vectorN << " 个float4向量" << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        vectorAddOptimized2<<<gridSize, blockSize>>>(
            reinterpret_cast<float4*>(d_A),
            reinterpret_cast<float4*>(d_B),
            reinterpret_cast<float4*>(d_C),
            vectorN
        );
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool correct = verifyResult();
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        std::cout << "结果正确性: " << (correct ? "✓ 正确" : "✗ 错误") << std::endl;
        std::cout << "带宽: " << calculateBandwidth(milliseconds) << " GB/s\n" << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void testSharedMemoryVersion() {
        std::cout << "--- 测试共享内存版本 (演示用途) ---" << std::endl;
        
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        size_t sharedMemSize = 2 * blockSize * sizeof(float);
        
        std::cout << "执行配置: <<<" << gridSize << ", " << blockSize 
                  << ", " << sharedMemSize << ">>>" << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        vectorAddWithSharedMem<<<gridSize, blockSize, sharedMemSize>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool correct = verifyResult();
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        std::cout << "结果正确性: " << (correct ? "✓ 正确" : "✗ 错误") << std::endl;
        std::cout << "带宽: " << calculateBandwidth(milliseconds) << " GB/s" << std::endl;
        std::cout << "注意: 对于向量加法，共享内存通常不会提升性能\n" << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void performanceComparison() {
        std::cout << "--- 性能分析和优化建议 ---" << std::endl;
        
        // 获取设备属性
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        
        // 计算理论峰值带宽
        float memClockRate = prop.memoryClockRate * 1000.0f; // Hz
        float memBusWidth = prop.memoryBusWidth; // bits
        float peakBandwidth = 2.0f * memClockRate * (memBusWidth / 8) / 1.0e9f; // GB/s
        
        std::cout << "设备信息:" << std::endl;
        std::cout << "  • GPU: " << prop.name << std::endl;
        std::cout << "  • 理论峰值带宽: " << std::fixed << std::setprecision(1) 
                  << peakBandwidth << " GB/s" << std::endl;
        
        std::cout << "\n优化建议:" << std::endl;
        std::cout << "  • 向量加法是内存带宽受限的操作" << std::endl;
        std::cout << "  • 关键是最大化内存带宽利用率" << std::endl;
        std::cout << "  • 合并内存访问模式很重要" << std::endl;
        std::cout << "  • 线程块大小应该是32的倍数" << std::endl;
        std::cout << "  • 考虑使用向量化数据类型(float2, float4)" << std::endl;
    }
    
    bool verifyResult() {
        const float epsilon = 1e-5f;
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_C[i] - h_C_ref[i]) > epsilon) {
                std::cout << "错误在索引 " << i << ": GPU=" << h_C[i] 
                          << ", CPU=" << h_C_ref[i] << std::endl;
                return false;
            }
        }
        return true;
    }
    
    float calculateBandwidth(float milliseconds) {
        // 向量加法需要读取2个向量，写入1个向量
        float bytes = 3.0f * N * sizeof(float);
        float seconds = milliseconds / 1000.0f;
        return (bytes / seconds) / 1.0e9f; // GB/s
    }
};

int main() {
    // 检查CUDA设备
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "没有找到CUDA设备！" << std::endl;
        return 1;
    }
    
    // 设置设备
    CUDA_CHECK(cudaSetDevice(0));
    
    // 运行不同大小的测试
    std::vector<int> testSizes = {1024, 1024*1024, 16*1024*1024};
    
    for (int size : testSizes) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        VectorAdditionDemo demo(size);
        demo.runAllTests();
    }
    
    std::cout << "\n=== 总结 ===" << std::endl;
    std::cout << "您已经学会了：" << std::endl;
    std::cout << "  • CUDA向量加法的基本实现" << std::endl;
    std::cout << "  • 性能测量和优化技巧" << std::endl;
    std::cout << "  • 不同的内存访问模式" << std::endl;
    std::cout << "  • 错误检查的重要性" << std::endl;
    std::cout << "  • 带宽计算和性能分析" << std::endl;
    
    return 0;
}

/*
编译和运行说明：

1. 编译命令：
   nvcc -O3 -o vector_addition vector_addition.cu

2. 运行：
   ./vector_addition

3. 预期输出：
   - 不同大小向量的性能测试结果
   - 各种优化版本的性能对比
   - 带宽利用率分析

4. 学习要点：
   - 理解内存带宽对性能的影响
   - 学会使用CUDA事件进行精确计时
   - 掌握结果验证的重要性
   - 了解不同优化策略的适用场景

5. 扩展练习：
   - 尝试不同的线程块大小
   - 实现其他向量运算（减法、乘法等）
   - 测试不同数据类型（double, int等）
   - 添加更多的优化版本
*/