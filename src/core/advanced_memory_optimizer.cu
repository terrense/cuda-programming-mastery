#include "advanced_memory_optimizer.h"
#include "error_handler.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>
#include <set>

namespace cuda_learning {

// 内存合并检测核函数
__global__ void testCoalescedAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // 合并访问
    }
}

__global__ void testStridedAccess(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (strided_idx < n) {
        output[idx] = input[strided_idx] * 2.0f;  // 跨步访问
    }
}

// 银行冲突测试核函数
__global__ void testBankConflicts(float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        // 产生银行冲突的访问模式
        int conflicted_index = (tid * 2) % 32;  // 2-way bank conflict
        sdata[conflicted_index] = tid * 1.0f;
        __syncthreads();

        // 读取数据
        output[tid] = sdata[conflicted_index];
    }
}

__global__ void testBankConflictFree(float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        // 无银行冲突的访问模式
        sdata[tid] = tid * 1.0f;
        __syncthreads();

        // 读取数据
        output[tid] = sdata[tid];
    }
}

// 纹理内存示例核函数
__global__ void testTextureMemory(cudaTextureObject_t texObj, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;

        // 使用纹理内存进行双线性插值
        float value = tex2D<float>(texObj, u, v);
        output[y * width + x] = value;
    }
}

// 表面内存写入核函数
__global__ void testSurfaceMemory(cudaSurfaceObject_t surfObj, float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = input[y * width + x] * 2.0f;
        surf2Dwrite(value, surfObj, x * sizeof(float), y);
    }
}

Adv
ancedMemoryOptimizer::AdvancedMemoryOptimizer() {
    // 初始化CUDA事件用于性能测量
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

AdvancedMemoryOptimizer::~AdvancedMemoryOptimizer() {
    // 清理资源
    CUDA_CHECK(cudaEventDestroy(start_event_));
    CUDA_CHECK(cudaEventDestroy(stop_event_));
}

CoalescingAnalysis AdvancedMemoryOptimizer::analyzeMemoryCoalescing(size_t data_size, int access_pattern) {
    CoalescingAnalysis result;
    result.data_size = data_size;
    result.access_pattern = access_pattern;

    // 分配GPU内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, data_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, data_size * sizeof(float)));

    // 初始化输入数据
    std::vector<float> h_input(data_size);
    std::iota(h_input.begin(), h_input.end(), 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size * sizeof(float), cudaMemcpyHostToDevice));

    // 配置执行参数
    int block_size = 256;
    int grid_size = (data_size + block_size - 1) / block_size;

    // 测试合并访问
    CUDA_CHECK(cudaEventRecord(start_event_));
    testCoalescedAccess<<<grid_size, block_size>>>(d_input, d_output, data_size);
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float coalesced_time;
    CUDA_CHECK(cudaEventElapsedTime(&coalesced_time, start_event_, stop_event_));
    result.coalesced_time_ms = coalesced_time;

    // 测试跨步访问
    int stride = access_pattern;
    CUDA_CHECK(cudaEventRecord(start_event_));
    testStridedAccess<<<grid_size, block_size>>>(d_input, d_output, data_size, stride);
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float strided_time;
    CUDA_CHECK(cudaEventElapsedTime(&strided_time, start_event_, stop_event_));
    result.strided_time_ms = strided_time;

    // 计算效率
    result.efficiency = coalesced_time / strided_time;
    result.bandwidth_gb_s = (data_size * sizeof(float) * 2) / (coalesced_time * 1e6);  // 读写两次

    // 生成优化建议
    if (result.efficiency > 0.8f) {
        result.optimization_suggestions.push_back("内存访问模式已经很好地合并");
    } else if (result.efficiency > 0.5f) {
        result.optimization_suggestions.push_back("考虑重新排列数据布局以提高合并度");
    } else {
        result.optimization_suggestions.push_back("严重的内存访问不合并，需要重新设计算法");
        result.optimization_suggestions.push_back("考虑使用共享内存或纹理内存");
    }

    // 清理内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

BankConflictAnalysis AdvancedMemoryOptimizer::analyzeBankConflicts(int shared_memory_size, int access_pattern) {
    BankConflictAnalysis result;
    result.shared_memory_size = shared_memory_size;
    result.access_pattern = access_pattern;

    // 分配GPU内存
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, shared_memory_size * sizeof(float)));

    int block_size = std::min(shared_memory_size, 256);
    size_t shared_mem_bytes = shared_memory_size * sizeof(float);

    // 测试有银行冲突的访问
    CUDA_CHECK(cudaEventRecord(start_event_));
    testBankConflicts<<<1, block_size, shared_mem_bytes>>>(d_output, shared_memory_size);
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float conflict_time;
    CUDA_CHECK(cudaEventElapsedTime(&conflict_time, start_event_, stop_event_));
    result.conflict_time_ms = conflict_time;

    // 测试无银行冲突的访问
    CUDA_CHECK(cudaEventRecord(start_event_));
    testBankConflictFree<<<1, block_size, shared_mem_bytes>>>(d_output, shared_memory_size);
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float conflict_free_time;
    CUDA_CHECK(cudaEventElapsedTime(&conflict_free_time, start_event_, stop_event_));
    result.conflict_free_time_ms = conflict_free_time;

    // 计算银行冲突程度
    result.conflict_ratio = conflict_time / conflict_free_time;

    // 估算银行冲突数量
    if (result.conflict_ratio > 2.0f) {
        result.estimated_conflicts = static_cast<int>((result.conflict_ratio - 1.0f) * 16);  // 假设16路冲突
    } else {
        result.estimated_conflicts = 0;
    }

    // 生成优化建议
    if (result.conflict_ratio < 1.2f) {
        result.optimization_suggestions.push_back("共享内存访问模式良好，无明显银行冲突");
    } else if (result.conflict_ratio < 2.0f) {
        result.optimization_suggestions.push_back("存在轻微银行冲突，考虑填充数组避免冲突");
    } else {
        result.optimization_suggestions.push_back("严重的银行冲突，需要重新设计共享内存访问模式");
        result.optimization_suggestions.push_back("使用数组填充或改变索引计算方式");
    }

    // 清理内存
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

TextureMemoryDemo AdvancedMemoryOptimizer::demonstrateTextureMemory(int width, int height) {
    TextureMemoryDemo result;
    result.width = width;
    result.height = height;

    size_t data_size = width * height;

    // 分配主机和设备内存
    std::vector<float> h_input(data_size);
    std::vector<float> h_output(data_size);

    // 初始化输入数据（创建一个简单的图案）
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_input[y * width + x] = sinf(x * 0.1f) * cosf(y * 0.1f);
        }
    }

    // 创建CUDA数组用于纹理
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // 复制数据到CUDA数组
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_input.data(),
                                   width * sizeof(float), width * sizeof(float), height,
                                   cudaMemcpyHostToDevice));

    // 创建纹理对象
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // 分配输出内存
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_size * sizeof(float)));

    // 配置执行参数
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    // 执行纹理内存测试
    CUDA_CHECK(cudaEventRecord(start_event_));
    testTextureMemory<<<grid_size, block_size>>>(texObj, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float execution_time;
    CUDA_CHECK(cudaEventElapsedTime(&execution_time, start_event_, stop_event_));
    result.execution_time_ms = execution_time;

    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, data_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算缓存命中率（简化估算）
    result.cache_hit_rate = 0.85f;  // 纹理缓存通常有很高的命中率

    // 生成使用建议
    result.usage_recommendations.push_back("纹理内存适用于具有空间局部性的2D数据访问");
    result.usage_recommendations.push_back("自动处理边界条件和插值，减少边界检查开销");
    result.usage_recommendations.push_back("对于随机访问模式，纹理缓存可以显著提高性能");

    // 清理资源
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

SurfaceMemoryDemo AdvancedMemoryOptimizer::demonstrateSurfaceMemory(int width, int height) {
    SurfaceMemoryDemo result;
    result.width = width;
    result.height = height;

    size_t data_size = width * height;

    // 分配主机内存
    std::vector<float> h_input(data_size);
    std::vector<float> h_output(data_size);

    // 初始化输入数据
    for (int i = 0; i < data_size; ++i) {
        h_input[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // 创建CUDA数组用于表面
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

    // 创建表面对象
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaSurfaceObject_t surfObj = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &resDesc));

    // 分配输入内存
    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, data_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size * sizeof(float), cudaMemcpyHostToDevice));

    // 配置执行参数
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    // 执行表面内存写入测试
    CUDA_CHECK(cudaEventRecord(start_event_));
    testSurfaceMemory<<<grid_size, block_size>>>(surfObj, d_input, width, height);
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float execution_time;
    CUDA_CHECK(cudaEventElapsedTime(&execution_time, start_event_, stop_event_));
    result.execution_time_ms = execution_time;

    // 从CUDA数组读取结果
    CUDA_CHECK(cudaMemcpy2DFromArray(h_output.data(), width * sizeof(float),
                                     cuArray, 0, 0, width * sizeof(float), height,
                                     cudaMemcpyDeviceToHost));

    // 计算写入带宽
    result.write_bandwidth_gb_s = (data_size * sizeof(float)) / (execution_time * 1e6);

    // 生成使用建议
    result.usage_recommendations.push_back("表面内存允许对CUDA数组进行读写操作");
    result.usage_recommendations.push_back("适用于需要就地修改2D数据的算法");
    result.usage_recommendations.push_back("与纹理内存结合使用可以实现高效的图像处理管道");

    // 清理资源
    CUDA_CHECK(cudaDestroySurfaceObject(surfObj));
    CUDA_CHECK(cudaFreeArray(cuArray));
    CUDA_CHECK(cudaFree(d_input));

    return result;
}

std::string AdvancedMemoryOptimizer::generateOptimizationReport(const CoalescingAnalysis& coalescing,
                                                               const BankConflictAnalysis& bank_conflict) {
    std::ostringstream report;

    report << "=== 高级内存优化分析报告 ===\n\n";

    // 内存合并分析
    report << "1. 内存合并分析:\n";
    report << "   数据大小: " << coalescing.data_size << " 元素\n";
    report << "   合并访问时间: " << std::fixed << std::setprecision(3)
           << coalescing.coalesced_time_ms << " ms\n";
    report << "   跨步访问时间: " << coalescing.strided_time_ms << " ms\n";
    report << "   访问效率: " << std::setprecision(2) << (coalescing.efficiency * 100) << "%\n";
    report << "   内存带宽: " << std::setprecision(1) << coalescing.bandwidth_gb_s << " GB/s\n";

    report << "   优化建议:\n";
    for (const auto& suggestion : coalescing.optimization_suggestions) {
        report << "   - " << suggestion << "\n";
    }
    report << "\n";

    // 银行冲突分析
    report << "2. 共享内存银行冲突分析:\n";
    report << "   共享内存大小: " << bank_conflict.shared_memory_size << " 元素\n";
    report << "   冲突访问时间: " << std::setprecision(3)
           << bank_conflict.conflict_time_ms << " ms\n";
    report << "   无冲突访问时间: " << bank_conflict.conflict_free_time_ms << " ms\n";
    report << "   冲突比率: " << std::setprecision(2) << bank_conflict.conflict_ratio << "x\n";
    report << "   估算冲突数: " << bank_conflict.estimated_conflicts << "\n";

    report << "   优化建议:\n";
    for (const auto& suggestion : bank_conflict.optimization_suggestions) {
        report << "   - " << suggestion << "\n";
    }
    report << "\n";

    // 总体建议
    report << "3. 总体优化建议:\n";
    if (coalescing.efficiency > 0.8f && bank_conflict.conflict_ratio < 1.2f) {
        report << "   - 内存访问模式已经很好优化\n";
        report << "   - 可以考虑进一步的算法级优化\n";
    } else {
        report << "   - 优先解决内存访问模式问题\n";
        report << "   - 考虑使用性能分析工具进行深入分析\n";
        report << "   - 实验不同的数据布局和访问模式\n";
    }

    return report.str();
}

} // namespace cuda_learning
