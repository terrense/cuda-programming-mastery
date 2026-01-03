#include "memory_management.h"
#include "error_handler.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>

namespace cuda_learning {

// 常量内存声明
__constant__ float const_coefficients[256];

// MemoryBenchmarkResult 实现
MemoryBenchmarkResult::MemoryBenchmarkResult()
    : memoryType(MemoryType::GLOBAL), accessPattern(AccessPattern::COALESCED),
      dataSize(0), bandwidth_GB_s(0.0f), executionTime_ms(0.0f),
      blockSize(256), gridSize(1) {}

std::string MemoryBenchmarkResult::toString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "内存类型: ";

    switch (memoryType) {
        case MemoryType::GLOBAL: ss << "全局内存"; break;
        case MemoryType::SHARED: ss << "共享内存"; break;
        case MemoryType::CONSTANT: ss << "常量内存"; break;
        case MemoryType::TEXTURE: ss << "纹理内存"; break;
        case MemoryType::UNIFIED: ss << "统一内存"; break;
    }

    ss << ", 访问模式: ";
    switch (accessPattern) {
        case AccessPattern::COALESCED: ss << "合并访问"; break;
        case AccessPattern::STRIDED: ss << "跨步访问"; break;
        case AccessPattern::RANDOM: ss << "随机访问"; break;
        case AccessPattern::BROADCAST: ss << "广播访问"; break;
    }

    ss << ", 数据大小: " << dataSize / (1024*1024) << " MB";
    ss << ", 带宽: " << bandwidth_GB_s << " GB/s";
    ss << ", 执行时间: " << executionTime_ms << " ms";
    ss << ", 配置: " << gridSize << "x" << blockSize;

    if (!description.empty()) {
        ss << ", " << description;
    }

    return ss.str();
}

// 全局内存核函数实现
namespace GlobalMemoryKernels {
    __global__ void coalescedAccess(float* input, float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // 连续访问，warp内线程访问连续内存地址
            output[idx] = input[idx] * 2.0f;
        }
    }

    __global__ void uncoalescedAccess(float* input, float* output, int n, int stride) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int strided_idx = (idx * stride) % n;
        if (strided_idx < n) {
            // 跨步访问，破坏内存合并
            output[strided_idx] = input[strided_idx] * 2.0f;
        }
    }

    __global__ void alignedAccess(float* input, float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // 对齐访问
            output[idx] = input[idx] + 1.0f;
        }
    }

    __global__ void misalignedAccess(float* input, float* output, int n, int offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx + offset < n) {
            // 非对齐访问
            output[idx] = input[idx + offset] + 1.0f;
        }
    }
}

// 共享内存核函数实现
namespace SharedMemoryKernels {
    __global__ void basicSharedMemory(float* input, float* output, int n) {
        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // 加载数据到共享内存
        if (idx < n) {
            sdata[tid] = input[idx];
        } else {
            sdata[tid] = 0.0f;
        }

        __syncthreads();

        // 使用共享内存中的数据进行计算
        if (idx < n) {
            float sum = 0.0f;
            // 计算邻近元素的平均值
            for (int i = -1; i <= 1; i++) {
                int neighbor = tid + i;
                if (neighbor >= 0 && neighbor < blockDim.x) {
                    sum += sdata[neighbor];
                }
            }
            output[idx] = sum / 3.0f;
        }
    }

    __global__ void bankConflictDemo(float* output, int n) {
        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            // 产生银行冲突的访问模式
            // 所有线程访问同一个银行
            int conflicted_index = (tid * 2) % 32; // 2-way bank conflict
            sdata[conflicted_index] = tid;

            __syncthreads();

            output[idx] = sdata[conflicted_index];
        }
    }

    __global__ void noBankConflictDemo(float* output, int n) {
        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            // 避免银行冲突的访问模式
            sdata[tid] = tid;

            __syncthreads();

            output[idx] = sdata[tid];
        }
    }

    __global__ void sharedMemoryTranspose(float* input, float* output, int width, int height) {
        __shared__ float tile[32][33]; // 添加padding避免银行冲突

        int x = blockIdx.x * 32 + threadIdx.x;
        int y = blockIdx.y * 32 + threadIdx.y;

        // 读取到共享内存
        if (x < width && y < height) {
            tile[threadIdx.y][threadIdx.x] = input[y * width + x];
        }

        __syncthreads();

        // 转置后写回
        x = blockIdx.y * 32 + threadIdx.x;
        y = blockIdx.x * 32 + threadIdx.y;

        if (x < height && y < width) {
            output[y * height + x] = tile[threadIdx.x][threadIdx.y];
        }
    }

    __global__ void sharedMemoryReduction(float* input, float* output, int n) {
        extern __shared__ float sdata[];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // 加载数据到共享内存
        sdata[tid] = (idx < n) ? input[idx] : 0.0f;
        __syncthreads();

        // 树形归约
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
}

// 常量内存核函数实现
namespace ConstantMemoryKernels {
    __global__ void constantMemoryConvolution(float* input, float* output,
                                            int width, int height, int kernelSize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            float sum = 0.0f;
            int halfKernel = kernelSize / 2;

            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int inputY = y + ky - halfKernel;
                    int inputX = x + kx - halfKernel;

                    if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width) {
                        float inputVal = input[inputY * width + inputX];
                        float kernelVal = const_coefficients[ky * kernelSize + kx];
                        sum += inputVal * kernelVal;
                    }
                }
            }

            output[y * width + x] = sum;
        }
    }

    __global__ void constantMemoryBroadcast(float* input, float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            // 所有线程读取相同的常量内存值（广播）
            float coefficient = const_coefficients[0];
            output[idx] = input[idx] * coefficient;
        }
    }
}

// MemoryManagementTeacher 实现
void MemoryManagementTeacher::demonstrateGlobalMemory() {
    std::cout << "\n=== 全局内存演示 ===" << std::endl;

    const int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);

    // 分配内存
    float *h_input = new float[n];
    float *h_output = new float[n];
    float *d_input, *d_output;

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // 测试合并访问
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 合并访问测试
    cudaEventRecord(start);
    GlobalMemoryKernels::coalescedAccess<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coalescedTime;
    cudaEventElapsedTime(&coalescedTime, start, stop);

    // 非合并访问测试
    cudaEventRecord(start);
    GlobalMemoryKernels::uncoalescedAccess<<<gridSize, blockSize>>>(d_input, d_output, n, 32);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float uncoalescedTime;
    cudaEventElapsedTime(&uncoalescedTime, start, stop);

    std::cout << "合并访问时间: " << coalescedTime << " ms" << std::endl;
    std::cout << "非合并访问时间: " << uncoalescedTime << " ms" << std::endl;
    std::cout << "性能差异: " << (uncoalescedTime / coalescedTime) << "x" << std::endl;

    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void MemoryManagementTeacher::demonstrateSharedMemory() {
    std::cout << "\n=== 共享内存演示 ===" << std::endl;

    const int n = 1024;
    size_t bytes = n * sizeof(float);

    float *h_input = new float[n];
    float *h_output = new float[n];
    float *d_input, *d_output;

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // 基本共享内存使用
    SharedMemoryKernels::basicSharedMemory<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    std::cout << "共享内存平滑滤波结果 (前10个): ";
    for (int i = 0; i < 10; i++) {
        std::cout << std::fixed << std::setprecision(2) << h_output[i] << " ";
    }
    std::cout << std::endl;

    // 银行冲突演示
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 有银行冲突
    cudaEventRecord(start);
    SharedMemoryKernels::bankConflictDemo<<<gridSize, blockSize, sharedMemSize>>>(d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float conflictTime;
    cudaEventElapsedTime(&conflictTime, start, stop);

    // 无银行冲突
    cudaEventRecord(start);
    SharedMemoryKernels::noBankConflictDemo<<<gridSize, blockSize, sharedMemSize>>>(d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float noConflictTime;
    cudaEventElapsedTime(&noConflictTime, start, stop);

    std::cout << "银行冲突访问时间: " << conflictTime << " ms" << std::endl;
    std::cout << "无银行冲突访问时间: " << noConflictTime << " ms" << std::endl;
    std::cout << "性能差异: " << (conflictTime / noConflictTime) << "x" << std::endl;

    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void MemoryManagementTeacher::demonstrateConstantMemory() {
    std::cout << "\n=== 常量内存演示 ===" << std::endl;

    const int width = 512, height = 512;
    const int kernelSize = 3;
    const int n = width * height;
    size_t bytes = n * sizeof(float);

    // 创建卷积核
    float h_kernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    // 复制到常量内存
    cudaMemcpyToSymbol(const_coefficients, h_kernel, kernelSize * kernelSize * sizeof(float));

    float *h_input = new float[n];
    float *h_output = new float[n];
    float *d_input, *d_output;

    // 初始化输入图像
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 使用常量内存的卷积
    cudaEventRecord(start);
    ConstantMemoryKernels::constantMemoryConvolution<<<gridSize, blockSize>>>(
        d_input, d_output, width, height, kernelSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float convTime;
    cudaEventElapsedTime(&convTime, start, stop);

    std::cout << "常量内存卷积执行时间: " << convTime << " ms" << std::endl;

    // 广播访问演示
    float broadcastCoeff = 2.5f;
    cudaMemcpyToSymbol(const_coefficients, &broadcastCoeff, sizeof(float));

    int blockSize1D = 256;
    int gridSize1D = (n + blockSize1D - 1) / blockSize1D;

    cudaEventRecord(start);
    ConstantMemoryKernels::constantMemoryBroadcast<<<gridSize1D, blockSize1D>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float broadcastTime;
    cudaEventElapsedTime(&broadcastTime, start, stop);

    std::cout << "常量内存广播访问时间: " << broadcastTime << " ms" << std::endl;

    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// MemoryBandwidthTester 实现
float MemoryBandwidthTester::testGlobalMemoryBandwidth(size_t dataSize, AccessPattern pattern) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);

    int blockSize = 256;
    int gridSize = (dataSize / sizeof(float) + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        switch (pattern) {
            case AccessPattern::COALESCED:
                GlobalMemoryKernels::coalescedAccess<<<gridSize, blockSize>>>(
                    d_input, d_output, dataSize / sizeof(float));
                break;
            case AccessPattern::STRIDED:
                GlobalMemoryKernels::uncoalescedAccess<<<gridSize, blockSize>>>(
                    d_input, d_output, dataSize / sizeof(float), 32);
                break;
            default:
                GlobalMemoryKernels::coalescedAccess<<<gridSize, blockSize>>>(
                    d_input, d_output, dataSize / sizeof(float));
                break;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 计算带宽 (GB/s)
    // 每次迭代读写两次数据
    float bandwidth = (2.0f * dataSize * iterations) / (elapsedTime * 1e-3) / (1024*1024*1024);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidth;
}

float MemoryBandwidthTester::getTheoreticalBandwidth(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    // 理论带宽 = 内存时钟频率 * 内存总线宽度 * 2 (DDR) / 8 (bits to bytes)
    // 这里使用一个简化的估算
    return prop.memoryBusWidth * prop.memoryClockRate * 2.0f / (8 * 1e6);
}

// MemoryDemo 实现
void MemoryDemo::runCompleteDemo() {
    std::cout << "=== CUDA内存管理完整演示 ===" << std::endl;

    printSectionHeader("1. 全局内存演示");
    MemoryManagementTeacher::demonstrateGlobalMemory();

    printSectionHeader("2. 共享内存演示");
    MemoryManagementTeacher::demonstrateSharedMemory();

    printSectionHeader("3. 常量内存演示");
    MemoryManagementTeacher::demonstrateConstantMemory();

    printSectionHeader("4. 内存带宽比较");
    std::vector<MemoryBenchmarkResult> results = MemoryManagementTeacher::compareMemoryTypes(64 * 1024 * 1024);
    printBenchmarkResults(results);

    printSectionHeader("5. 优化建议");
    std::cout << MemoryManagementTeacher::generateOptimizationReport(results) << std::endl;
}

void MemoryDemo::printSectionHeader(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void MemoryDemo::printBenchmarkResults(const std::vector<MemoryBenchmarkResult>& results) {
    std::cout << "\n内存性能测试结果:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& result : results) {
        std::cout << result.toString() << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
}

std::vector<MemoryBenchmarkResult> MemoryManagementTeacher::compareMemoryTypes(size_t dataSize) {
    std::vector<MemoryBenchmarkResult> results;

    // 测试全局内存合并访问
    MemoryBenchmarkResult globalCoalesced;
    globalCoalesced.memoryType = MemoryType::GLOBAL;
    globalCoalesced.accessPattern = AccessPattern::COALESCED;
    globalCoalesced.dataSize = dataSize;
    globalCoalesced.bandwidth_GB_s = MemoryBandwidthTester::testGlobalMemoryBandwidth(dataSize, AccessPattern::COALESCED);
    globalCoalesced.description = "全局内存合并访问";
    results.push_back(globalCoalesced);

    // 测试全局内存跨步访问
    MemoryBenchmarkResult globalStrided;
    globalStrided.memoryType = MemoryType::GLOBAL;
    globalStrided.accessPattern = AccessPattern::STRIDED;
    globalStrided.dataSize = dataSize;
    globalStrided.bandwidth_GB_s = MemoryBandwidthTester::testGlobalMemoryBandwidth(dataSize, AccessPattern::STRIDED);
    globalStrided.description = "全局内存跨步访问";
    results.push_back(globalStrided);

    return results;
}

std::string MemoryManagementTeacher::generateOptimizationReport(const std::vector<MemoryBenchmarkResult>& results) {
    std::stringstream ss;
    ss << "\n=== 内存优化建议报告 ===" << std::endl;

    if (results.size() >= 2) {
        float coalescedBW = results[0].bandwidth_GB_s;
        float stridedBW = results[1].bandwidth_GB_s;
        float efficiency = stridedBW / coalescedBW;

        ss << "1. 内存访问模式影响:" << std::endl;
        ss << "   - 合并访问带宽: " << std::fixed << std::setprecision(2) << coalescedBW << " GB/s" << std::endl;
        ss << "   - 跨步访问带宽: " << stridedBW << " GB/s" << std::endl;
        ss << "   - 效率比: " << (efficiency * 100) << "%" << std::endl;

        ss << "\n2. 优化建议:" << std::endl;
        if (efficiency < 0.8f) {
            ss << "   - 重新组织数据布局以实现合并访问" << std::endl;
            ss << "   - 考虑使用共享内存缓存频繁访问的数据" << std::endl;
            ss << "   - 使用向量化数据类型 (float2, float4)" << std::endl;
        }

        ss << "   - 确保内存对齐 (128字节边界)" << std::endl;
        ss << "   - 使用常量内存存储只读数据" << std::endl;
        ss << "   - 避免共享内存银行冲突" << std::endl;
    }

    return ss.str();
}

} // namespace cuda_learning
