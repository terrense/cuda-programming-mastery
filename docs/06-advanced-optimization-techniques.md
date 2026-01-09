# 高级优化技术

## 概述

本文档介绍CUDA编程中的高级优化技术，包括内存优化、多GPU编程、性能分析工具集成等内容。这些技术将帮助你将CUDA程序的性能提升到极致。

## 学习目标

完成本模块后，你将能够：
- 掌握高级内存优化技术
- 实现多GPU协同编程
- 使用专业性能分析工具
- 应用各种高级优化策略
- 进行系统级性能调优

## 内存优化高级技术

### 内存合并模式检测和优化

#### 内存访问模式分析器

```cpp
// memory_analyzer.h
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

class MemoryAccessAnalyzer {
public:
    enum AccessPattern {
        COALESCED,      // 合并访问
        STRIDED,        // 跨步访问
        RANDOM,         // 随机访问
        BROADCAST       // 广播访问
    };

    struct AccessInfo {
        AccessPattern pattern;
        float efficiency;
        size_t transactions;
        std::string recommendation;
    };

    static AccessInfo analyzeKernel(const std::string& kernel_name,
                                   const std::vector<void*>& arrays,
                                   const std::vector<size_t>& sizes);

    static void optimizeAccessPattern(void* data, size_t size,
                                    AccessPattern current_pattern);
};

// 内存访问模式测试核函数
template<typename T>
__global__ void coalescedAccessKernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // 连续访问
    }
}

template<typename T>
__global__ void stridedAccessKernel(T* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // 跨步访问
    }
}

template<typename T>
__global__ void randomAccessKernel(T* data, int* indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int idx = indices[tid];
        if (idx < n) {
            data[idx] = data[idx] * 2.0f;  // 随机访问
        }
    }
}
```

#### 内存访问优化工具

```cpp
// memory_optimizer.cu
#include "memory_analyzer.h"

class MemoryOptimizer {
public:
    // 数据重排优化
    template<typename T>
    static void optimizeDataLayout(T* input, T* output,
                                  const std::vector<int>& old_shape,
                                  const std::vector<int>& new_shape) {
        // 实现数据重排以优化访问模式
        int total_elements = 1;
        for (int dim : old_shape) total_elements *= dim;

        dim3 blockSize(256);
        dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);

        transposeKernel<<<gridSize, blockSize>>>(input, output,
                                                old_shape.data(),
                                                new_shape.data(),
                                                old_shape.size());
    }

    // 内存预取优化
    static void prefetchMemory(void* ptr, size_t size, int device) {
        cudaMemPrefetchAsync(ptr, size, device);
    }

    // 内存对齐优化
    template<typename T>
    static T* allocateAligned(size_t count, size_t alignment = 256) {
        T* ptr;
        size_t size = count * sizeof(T);
        size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

        cudaMalloc(&ptr, aligned_size);
        return ptr;
    }
};

template<typename T>
__global__ void transposeKernel(const T* input, T* output,
                               const int* old_shape, const int* new_shape,
                               int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算多维索引转换
    // 实现复杂的数据重排逻辑
    // ...
}
```

### 共享内存银行冲突避免

#### 银行冲突检测器

```cpp
// bank_conflict_detector.cu
#include <cuda_runtime.h>

class BankConflictDetector {
public:
    // 检测共享内存访问的银行冲突
    template<int BLOCK_SIZE>
    static int detectConflicts(const std::vector<int>& access_indices) {
        const int BANK_SIZE = 32;  // 每个银行的大小
        const int NUM_BANKS = 32;  // 银行数量

        std::vector<int> bank_access_count(NUM_BANKS, 0);

        for (int idx : access_indices) {
            int bank = (idx / sizeof(float)) % NUM_BANKS;
            bank_access_count[bank]++;
        }

        int max_conflicts = 0;
        for (int count : bank_access_count) {
            max_conflicts = std::max(max_conflicts, count - 1);
        }

        return max_conflicts;
    }
};

// 避免银行冲突的矩阵转置
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transposeNoBankConflicts(float* odata, const float* idata,
                                        int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免银行冲突

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 读取数据到共享内存
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    // 转置写入
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

### 纹理内存和表面内存优化

#### 纹理内存使用示例

```cpp
// texture_memory.cu
#include <cuda_runtime.h>

// 1D纹理内存
texture<float, 1, cudaReadModeElementType> tex1D;

// 2D纹理内存
texture<float, 2, cudaReadModeElementType> tex2D;

// 3D纹理内存
texture<float, 3, cudaReadModeElementType> tex3D;

class TextureMemoryManager {
public:
    // 绑定1D纹理
    static void bind1DTexture(float* data, size_t size) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaBindTexture(0, tex1D, data, channelDesc, size * sizeof(float));
    }

    // 绑定2D纹理
    static void bind2DTexture(float* data, int width, int height) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaBindTexture2D(0, tex2D, data, channelDesc,
                         width, height, width * sizeof(float));
    }

    // 使用纹理内存的卷积核函数
    template<int KERNEL_SIZE>
    static void launchTextureConvolution(float* output, int width, int height) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);

        textureConvolutionKernel<KERNEL_SIZE><<<gridSize, blockSize>>>(
            output, width, height);
    }
};

template<int KERNEL_SIZE>
__global__ void textureConvolutionKernel(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        for (int ky = -KERNEL_SIZE/2; ky <= KERNEL_SIZE/2; ky++) {
            for (int kx = -KERNEL_SIZE/2; kx <= KERNEL_SIZE/2; kx++) {
                // 使用纹理内存，自动处理边界条件
                sum += tex2D(tex2D, x + kx + 0.5f, y + ky + 0.5f);
            }
        }

        output[y * width + x] = sum / (KERNEL_SIZE * KERNEL_SIZE);
    }
}
```

#### 表面内存使用

```cpp
// surface_memory.cu
#include <cuda_runtime.h>

// 表面内存声明
surface<void, 2> surf2D;

class SurfaceMemoryManager {
public:
    static cudaArray* createSurfaceArray(int width, int height) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray* cuArray;

        cudaMallocArray(&cuArray, &channelDesc, width, height,
                       cudaArraySurfaceLoadStore);

        // 绑定到表面内存
        cudaBindSurfaceToArray(surf2D, cuArray);

        return cuArray;
    }

    static void launchSurfaceWrite(int width, int height) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);

        surfaceWriteKernel<<<gridSize, blockSize>>>(width, height);
    }
};

__global__ void surfaceWriteKernel(int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = x * y * 0.001f;
        surf2Dwrite(value, surf2D, x * sizeof(float), y);
    }
}

__global__ void surfaceReadKernel(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value;
        surf2Dread(&value, surf2D, x * sizeof(float), y);
        output[y * width + x] = value;
    }
}
```

## 多GPU编程框架

### NCCL集成多GPU通信

```cpp
// multi_gpu_manager.h
#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

class MultiGPUManager {
private:
    std::vector<int> device_ids_;
    std::vector<cudaStream_t> streams_;
    std::vector<ncclComm_t> nccl_comms_;
    ncclUniqueId nccl_id_;

public:
    MultiGPUManager(const std::vector<int>& device_ids);
    ~MultiGPUManager();

    // 初始化NCCL通信
    void initializeNCCL();

    // 全归约操作
    void allReduce(const std::vector<void*>& send_buffs,
                   const std::vector<void*>& recv_buffs,
                   size_t count, ncclDataType_t datatype,
                   ncclRedOp_t op = ncclSum);

    // 全收集操作
    void allGather(const std::vector<void*>& send_buffs,
                   const std::vector<void*>& recv_buffs,
                   size_t count, ncclDataType_t datatype);

    // 广播操作
    void broadcast(const std::vector<void*>& buffs,
                   size_t count, ncclDataType_t datatype,
                   int root);

    // 点对点通信
    void send(void* sendbuff, size_t count, ncclDataType_t datatype,
              int peer, int device_id);
    void recv(void* recvbuff, size_t count, ncclDataType_t datatype,
              int peer, int device_id);

    // 同步所有设备
    void synchronizeAll();

    // 获取设备数量
    int getDeviceCount() const { return device_ids_.size(); }

    // 获取流
    cudaStream_t getStream(int device_idx) const { return streams_[device_idx]; }
};

// multi_gpu_manager.cu
MultiGPUManager::MultiGPUManager(const std::vector<int>& device_ids)
    : device_ids_(device_ids) {

    streams_.resize(device_ids_.size());
    nccl_comms_.resize(device_ids_.size());

    // 为每个设备创建流
    for (size_t i = 0; i < device_ids_.size(); i++) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamCreate(&streams_[i]);
    }

    initializeNCCL();
}

void MultiGPUManager::initializeNCCL() {
    // 生成唯一ID
    ncclGetUniqueId(&nccl_id_);

    // 初始化通信器
    ncclGroupStart();
    for (size_t i = 0; i < device_ids_.size(); i++) {
        cudaSetDevice(device_ids_[i]);
        ncclCommInitRank(&nccl_comms_[i], device_ids_.size(), nccl_id_, i);
    }
    ncclGroupEnd();
}

void MultiGPUManager::allReduce(const std::vector<void*>& send_buffs,
                               const std::vector<void*>& recv_buffs,
                               size_t count, ncclDataType_t datatype,
                               ncclRedOp_t op) {
    ncclGroupStart();
    for (size_t i = 0; i < device_ids_.size(); i++) {
        cudaSetDevice(device_ids_[i]);
        ncclAllReduce(send_buffs[i], recv_buffs[i], count, datatype, op,
                     nccl_comms_[i], streams_[i]);
    }
    ncclGroupEnd();
}
```

### 负载均衡和数据分片策略

```cpp
// load_balancer.h
#pragma once
#include "multi_gpu_manager.h"
#include <vector>
#include <memory>

class LoadBalancer {
public:
    enum Strategy {
        ROUND_ROBIN,    // 轮询分配
        CAPABILITY_BASED, // 基于能力分配
        DYNAMIC_LOAD    // 动态负载均衡
    };

    struct DeviceInfo {
        int device_id;
        size_t memory_total;
        size_t memory_free;
        int compute_capability_major;
        int compute_capability_minor;
        int multiprocessor_count;
        float utilization;
    };

private:
    std::vector<DeviceInfo> devices_;
    Strategy strategy_;
    std::unique_ptr<MultiGPUManager> gpu_manager_;

public:
    LoadBalancer(const std::vector<int>& device_ids, Strategy strategy);

    // 获取设备信息
    void updateDeviceInfo();

    // 数据分片
    std::vector<size_t> calculateDataShards(size_t total_size);

    // 任务分配
    std::vector<int> assignTasks(const std::vector<size_t>& task_sizes);

    // 动态重平衡
    void rebalanceTasks();

    // 性能监控
    void startPerformanceMonitoring();
    void stopPerformanceMonitoring();

private:
    std::vector<size_t> roundRobinShard(size_t total_size);
    std::vector<size_t> capabilityBasedShard(size_t total_size);
    std::vector<size_t> dynamicLoadShard(size_t total_size);
};

// load_balancer.cu
std::vector<size_t> LoadBalancer::calculateDataShards(size_t total_size) {
    switch (strategy_) {
        case ROUND_ROBIN:
            return roundRobinShard(total_size);
        case CAPABILITY_BASED:
            return capabilityBasedShard(total_size);
        case DYNAMIC_LOAD:
            return dynamicLoadShard(total_size);
        default:
            return roundRobinShard(total_size);
    }
}

std::vector<size_t> LoadBalancer::capabilityBasedShard(size_t total_size) {
    std::vector<size_t> shards(devices_.size());

    // 计算总计算能力
    float total_capability = 0.0f;
    for (const auto& device : devices_) {
        float capability = device.multiprocessor_count *
                          (device.compute_capability_major * 10 +
                           device.compute_capability_minor);
        total_capability += capability;
    }

    // 按能力比例分配
    size_t allocated = 0;
    for (size_t i = 0; i < devices_.size() - 1; i++) {
        float capability = devices_[i].multiprocessor_count *
                          (devices_[i].compute_capability_major * 10 +
                           devices_[i].compute_capability_minor);

        shards[i] = static_cast<size_t>(total_size * capability / total_capability);
        allocated += shards[i];
    }

    // 最后一个设备分配剩余数据
    shards.back() = total_size - allocated;

    return shards;
}
```

### 多GPU协同训练系统

```cpp
// distributed_training.h
#pragma once
#include "multi_gpu_manager.h"
#include "load_balancer.h"

class DistributedTrainingManager {
private:
    std::unique_ptr<MultiGPUManager> gpu_manager_;
    std::unique_ptr<LoadBalancer> load_balancer_;

    struct TrainingState {
        std::vector<void*> model_parameters;
        std::vector<void*> gradients;
        std::vector<void*> optimizer_states;
        float learning_rate;
        int current_epoch;
        int global_step;
    };

    TrainingState training_state_;

public:
    DistributedTrainingManager(const std::vector<int>& device_ids);

    // 初始化分布式训练
    void initializeTraining(const std::vector<size_t>& parameter_sizes);

    // 前向传播
    void forwardPass(const std::vector<void*>& inputs,
                    std::vector<void*>& outputs);

    // 反向传播
    void backwardPass(const std::vector<void*>& grad_outputs);

    // 梯度同步
    void synchronizeGradients();

    // 参数更新
    void updateParameters();

    // 模型同步
    void synchronizeModel();

    // 检查点保存/加载
    void saveCheckpoint(const std::string& path);
    void loadCheckpoint(const std::string& path);

private:
    void allReduceGradients();
    void applyGradientClipping(float max_norm);
};

// distributed_training.cu
void DistributedTrainingManager::synchronizeGradients() {
    // 梯度全归约
    gpu_manager_->allReduce(training_state_.gradients,
                           training_state_.gradients,
                           getTotalParameterCount(),
                           ncclFloat32, ncclSum);

    // 梯度平均
    int num_devices = gpu_manager_->getDeviceCount();
    for (size_t i = 0; i < training_state_.gradients.size(); i++) {
        cudaSetDevice(gpu_manager_->getDeviceIds()[i]);

        dim3 blockSize(256);
        dim3 gridSize((parameter_sizes_[i] + blockSize.x - 1) / blockSize.x);

        scaleGradientsKernel<<<gridSize, blockSize, 0,
                              gpu_manager_->getStream(i)>>>(
            static_cast<float*>(training_state_.gradients[i]),
            parameter_sizes_[i], 1.0f / num_devices);
    }

    gpu_manager_->synchronizeAll();
}

__global__ void scaleGradientsKernel(float* gradients, size_t count, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        gradients[idx] *= scale;
    }
}
```

## Nsight性能分析工具集成

### Nsight Compute集成接口

```cpp
// nsight_profiler.h
#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>

class NsightProfiler {
public:
    struct KernelMetrics {
        std::string kernel_name;
        float execution_time_ms;
        float achieved_occupancy;
        float memory_throughput_gbps;
        float compute_throughput_percent;
        size_t shared_memory_usage;
        size_t register_usage;
        std::map<std::string, float> custom_metrics;
    };

    struct ProfilingSession {
        std::string session_name;
        std::vector<KernelMetrics> kernel_metrics;
        float total_time_ms;
        std::string report_path;
    };

private:
    bool profiling_enabled_;
    std::string output_directory_;
    ProfilingSession current_session_;

public:
    NsightProfiler(const std::string& output_dir = "./profiling_results");

    // 开始性能分析会话
    void startSession(const std::string& session_name);

    // 结束性能分析会话
    void endSession();

    // 标记核函数开始
    void markKernelStart(const std::string& kernel_name);

    // 标记核函数结束
    void markKernelEnd(const std::string& kernel_name);

    // 添加自定义指标
    void addCustomMetric(const std::string& kernel_name,
                        const std::string& metric_name,
                        float value);

    // 生成性能报告
    void generateReport();

    // 自动化性能分析
    template<typename KernelFunc>
    KernelMetrics profileKernel(const std::string& kernel_name,
                               KernelFunc kernel_func);

    // 比较不同实现的性能
    void compareImplementations(const std::vector<std::string>& impl_names,
                               const std::vector<std::function<void()>>& impls);

private:
    void collectKernelMetrics(const std::string& kernel_name);
    void saveMetricsToFile();
    std::string generateHTMLReport();
};

// nsight_profiler.cu
template<typename KernelFunc>
NsightProfiler::KernelMetrics NsightProfiler::profileKernel(
    const std::string& kernel_name, KernelFunc kernel_func) {

    KernelMetrics metrics;
    metrics.kernel_name = kernel_name;

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    for (int i = 0; i < 5; i++) {
        kernel_func();
    }
    cudaDeviceSynchronize();

    // 开始性能分析
    markKernelStart(kernel_name);

    cudaEventRecord(start);
    kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    markKernelEnd(kernel_name);

    // 计算执行时间
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    metrics.execution_time_ms = ms;

    // 收集其他指标
    collectKernelMetrics(kernel_name);

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return metrics;
}

void NsightProfiler::collectKernelMetrics(const std::string& kernel_name) {
    // 使用CUPTI或其他工具收集详细指标
    // 这里简化实现，实际需要集成CUPTI API

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 估算占用率
    int blockSize = 256;  // 假设的块大小
    int minGridSize, optimalBlockSize;

    // 这里需要实际的核函数指针，简化处理
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize,
    //                                   kernel_func, 0, 0);

    // 添加到当前会话
    KernelMetrics& metrics = current_session_.kernel_metrics.back();
    metrics.achieved_occupancy = 0.75f;  // 示例值
    metrics.memory_throughput_gbps = 500.0f;  // 示例值
    metrics.compute_throughput_percent = 80.0f;  // 示例值
}
```

### 自动化性能报告生成

```cpp
// performance_reporter.h
#pragma once
#include "nsight_profiler.h"
#include <fstream>
#include <sstream>

class PerformanceReporter {
public:
    struct OptimizationSuggestion {
        std::string category;
        std::string issue;
        std::string suggestion;
        float potential_speedup;
        int priority;  // 1-5, 5最高
    };

    struct PerformanceAnalysis {
        std::vector<OptimizationSuggestion> suggestions;
        float overall_efficiency;
        std::string bottleneck_analysis;
        std::map<std::string, float> metric_scores;
    };

private:
    NsightProfiler profiler_;

public:
    PerformanceReporter(const std::string& output_dir);

    // 分析性能数据
    PerformanceAnalysis analyzePerformance(
        const NsightProfiler::ProfilingSession& session);

    // 生成优化建议
    std::vector<OptimizationSuggestion> generateSuggestions(
        const NsightProfiler::KernelMetrics& metrics);

    // 生成HTML报告
    void generateHTMLReport(const PerformanceAnalysis& analysis,
                           const std::string& output_path);

    // 生成JSON报告
    void generateJSONReport(const PerformanceAnalysis& analysis,
                           const std::string& output_path);

    // 性能趋势分析
    void analyzeTrends(const std::vector<NsightProfiler::ProfilingSession>& sessions);

private:
    float calculateEfficiencyScore(const NsightProfiler::KernelMetrics& metrics);
    std::string identifyBottleneck(const NsightProfiler::KernelMetrics& metrics);
    std::vector<OptimizationSuggestion> analyzeMemoryAccess(
        const NsightProfiler::KernelMetrics& metrics);
    std::vector<OptimizationSuggestion> analyzeOccupancy(
        const NsightProfiler::KernelMetrics& metrics);
};

// performance_reporter.cu
std::vector<PerformanceReporter::OptimizationSuggestion>
PerformanceReporter::generateSuggestions(const NsightProfiler::KernelMetrics& metrics) {

    std::vector<OptimizationSuggestion> suggestions;

    // 分析占用率
    if (metrics.achieved_occupancy < 0.5f) {
        OptimizationSuggestion suggestion;
        suggestion.category = "Occupancy";
        suggestion.issue = "Low occupancy detected (" +
                          std::to_string(metrics.achieved_occupancy * 100) + "%)";
        suggestion.suggestion = "Consider reducing register usage or shared memory usage, "
                               "or increasing block size";
        suggestion.potential_speedup = 1.5f;
        suggestion.priority = 4;
        suggestions.push_back(suggestion);
    }

    // 分析内存吞吐量
    if (metrics.memory_throughput_gbps < 300.0f) {  // 假设峰值为800GB/s
        OptimizationSuggestion suggestion;
        suggestion.category = "Memory";
        suggestion.issue = "Low memory throughput (" +
                          std::to_string(metrics.memory_throughput_gbps) + " GB/s)";
        suggestion.suggestion = "Optimize memory access patterns for coalescing, "
                               "consider using shared memory";
        suggestion.potential_speedup = 2.0f;
        suggestion.priority = 5;
        suggestions.push_back(suggestion);
    }

    // 分析计算吞吐量
    if (metrics.compute_throughput_percent < 60.0f) {
        OptimizationSuggestion suggestion;
        suggestion.category = "Compute";
        suggestion.issue = "Low compute utilization (" +
                          std::to_string(metrics.compute_throughput_percent) + "%)";
        suggestion.suggestion = "Increase arithmetic intensity, reduce memory-bound operations";
        suggestion.potential_speedup = 1.3f;
        suggestion.priority = 3;
        suggestions.push_back(suggestion);
    }

    return suggestions;
}

void PerformanceReporter::generateHTMLReport(const PerformanceAnalysis& analysis,
                                           const std::string& output_path) {
    std::ofstream file(output_path);

    file << R"(
<!DOCTYPE html>
<html>
<head>
    <title>CUDA Performance Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .suggestion { background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }
        .high-priority { border-left-color: #dc3545; }
        .chart { width: 100%; height: 300px; background: #f8f9fa; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>CUDA Performance Analysis Report</h1>

    <h2>Overall Performance</h2>
    <div class="metric">
        <strong>Overall Efficiency:</strong> )" << analysis.overall_efficiency * 100 << R"(%
    </div>

    <h2>Bottleneck Analysis</h2>
    <div class="metric">)" << analysis.bottleneck_analysis << R"(</div>

    <h2>Optimization Suggestions</h2>)";

    for (const auto& suggestion : analysis.suggestions) {
        std::string priority_class = (suggestion.priority >= 4) ? " high-priority" : "";
        file << R"(
    <div class="suggestion)" << priority_class << R"(">
        <h3>)" << suggestion.category << R"(</h3>
        <p><strong>Issue:</strong> )" << suggestion.issue << R"(</p>
        <p><strong>Suggestion:</strong> )" << suggestion.suggestion << R"(</p>
        <p><strong>Potential Speedup:</strong> )" << suggestion.potential_speedup << R"(x</p>
        <p><strong>Priority:</strong> )" << suggestion.priority << R"(/5</p>
    </div>)";
    }

    file << R"(
</body>
</html>)";

    file.close();
}
```

## 高级优化策略

### Warp级原语优化

```cpp
// warp_primitives.cu
#include <cuda_runtime.h>

// Warp级归约
template<typename T>
__device__ T warpReduce(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp级前缀和
template<typename T>
__device__ T warpPrefixSum(T val) {
    for (int offset = 1; offset < warpSize; offset *= 2) {
        T temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x >= offset) {
            val += temp;
        }
    }
    return val;
}

// 使用warp原语的快速归约
template<typename T>
__global__ void fastReductionKernel(const T* input, T* output, int n) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据
    T val = (idx < n) ? input[idx] : T(0);

    // Warp级归约
    val = warpReduce(val);

    // 每个warp的第一个线程写入共享内存
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = val;
    }

    __syncthreads();

    // 最后一个warp处理剩余数据
    if (tid < blockDim.x / warpSize) {
        val = sdata[tid];
        val = warpReduce(val);

        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}
```

### 动态并行优化

```cpp
// dynamic_parallelism.cu
#include <cuda_runtime.h>

// 父核函数
__global__ void parentKernel(float* data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 处理当前层数据
        data[idx] = data[idx] * 2.0f;

        // 如果需要进一步处理，启动子核函数
        if (depth > 0 && data[idx] > 100.0f) {
            // 动态启动子核函数
            childKernel<<<1, 32>>>(data + idx, min(32, n - idx), depth - 1);
        }
    }
}

// 子核函数
__global__ void childKernel(float* data, int n, int depth) {
    int idx = threadIdx.x;

    if (idx < n) {
        // 子任务处理
        data[idx] = sqrtf(data[idx]);

        // 递归调用
        if (depth > 0) {
            childKernel<<<1, min(n, 32)>>>(data, n, depth - 1);
        }
    }
}

// 自适应网格大小的动态并行
__global__ void adaptiveKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float value = data[idx];

        // 根据数据特征决定是否启动子任务
        if (value > 1000.0f) {
            // 复杂计算需要更多线程
            int child_threads = min(256, n - idx);
            int child_blocks = (child_threads + 31) / 32;

            complexComputeKernel<<<child_blocks, 32>>>(data + idx, child_threads);
        } else {
            // 简单计算
            data[idx] = value * 1.1f;
        }
    }
}

__global__ void complexComputeKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 复杂的数学运算
        float result = data[idx];
        for (int i = 0; i < 100; i++) {
            result = sinf(result) * cosf(result);
        }
        data[idx] = result;
    }
}
```

### 统一内存管理优化

```cpp
// unified_memory.cu
#include <cuda_runtime.h>

class UnifiedMemoryManager {
private:
    struct MemoryRegion {
        void* ptr;
        size_t size;
        bool is_prefetched;
        int preferred_device;
    };

    std::vector<MemoryRegion> regions_;

public:
    // 分配统一内存
    void* allocateUnified(size_t size, int preferred_device = -1) {
        void* ptr;
        cudaMallocManaged(&ptr, size);

        MemoryRegion region;
        region.ptr = ptr;
        region.size = size;
        region.is_prefetched = false;
        region.preferred_device = preferred_device;

        regions_.push_back(region);

        // 设置内存访问提示
        if (preferred_device >= 0) {
            cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, preferred_device);
        }

        return ptr;
    }

    // 预取内存到指定设备
    void prefetchToDevice(void* ptr, int device) {
        auto it = std::find_if(regions_.begin(), regions_.end(),
                              [ptr](const MemoryRegion& region) {
                                  return region.ptr == ptr;
                              });

        if (it != regions_.end()) {
            cudaMemPrefetchAsync(ptr, it->size, device);
            it->is_prefetched = true;
        }
    }

    // 设置内存访问模式
    void setAccessPattern(void* ptr, cudaMemoryAdvise advice) {
        auto it = std::find_if(regions_.begin(), regions_.end(),
                              [ptr](const MemoryRegion& region) {
                                  return region.ptr == ptr;
                              });

        if (it != regions_.end()) {
            cudaMemAdvise(ptr, it->size, advice, 0);
        }
    }

    // 智能内存迁移
    void smartMigration() {
        for (auto& region : regions_) {
            // 分析访问模式并决定是否迁移
            // 这里简化实现
            if (!region.is_prefetched && region.preferred_device >= 0) {
                prefetchToDevice(region.ptr, region.preferred_device);
            }
        }
    }

    // 释放统一内存
    void deallocate(void* ptr) {
        auto it = std::find_if(regions_.begin(), regions_.end(),
                              [ptr](const MemoryRegion& region) {
                                  return region.ptr == ptr;
                              });

        if (it != regions_.end()) {
            cudaFree(ptr);
            regions_.erase(it);
        }
    }
};

// 使用统一内存的示例
__global__ void unifiedMemoryKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 直接访问统一内存，无需显式数据传输
        data[idx] = data[idx] * data[idx] + 1.0f;
    }
}

void demonstrateUnifiedMemory() {
    UnifiedMemoryManager um_manager;

    const int n = 1000000;
    float* data = static_cast<float*>(um_manager.allocateUnified(n * sizeof(float), 0));

    // 初始化数据
    for (int i = 0; i < n; i++) {
        data[i] = i * 0.001f;
    }

    // 设置访问提示
    um_manager.setAccessPattern(data, cudaMemAdviseSetReadMostly);

    // 预取到GPU
    um_manager.prefetchToDevice(data, 0);

    // 执行核函数
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    unifiedMemoryKernel<<<gridSize, blockSize>>>(data, n);
    cudaDeviceSynchronize();

    // 预取回CPU进行结果验证
    um_manager.prefetchToDevice(data, cudaCpuDeviceId);

    // 验证结果
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %f\n", i, data[i]);
    }

    um_manager.deallocate(data);
}
```

## 实践练习

### 练习1：内存访问优化
实现一个内存访问模式分析器，自动检测和优化非合并访问。

### 练习2：多GPU矩阵乘法
使用NCCL实现多GPU协同的大规模矩阵乘法。

### 练习3：性能分析工具
集成Nsight Compute，创建自动化性能分析流水线。

### 练习4：动态负载均衡
实现基于实时性能监控的动态负载均衡系统。

## 总结

本模块介绍了CUDA编程的高级优化技术，包括：
- 内存访问模式的深度优化
- 多GPU协同编程框架
- 专业性能分析工具的集成
- 各种高级优化策略的应用

掌握这些技术后，你将能够开发出性能极致的CUDA应用程序，充分发挥GPU的计算潜力。

---

**下一步**：学习[学习管理和进度跟踪系统](07-learning-management-system.md)
