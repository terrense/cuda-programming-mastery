#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace cuda_learning {
namespace yolo {

// GPU内存块信息
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    cudaStream_t stream;
    std::string tag;  // 用于调试和跟踪

    MemoryBlock() : ptr(nullptr), size(0), in_use(false), stream(0) {}
    MemoryBlock(void* p, size_t s, const std::string& t = "")
        : ptr(p), size(s), in_use(false), stream(0), tag(t) {}
};

// GPU内存管理器
class GPUMemoryManager {
private:
    std::vector<MemoryBlock> blocks_;
    mutable std::mutex memory_mutex_;
    size_t total_allocated_;
    size_t peak_usage_;
    size_t current_usage_;

    // 内存对齐大小
    static const size_t ALIGNMENT = 256;

public:
    GPUMemoryManager();
    ~GPUMemoryManager();

    // 分配内存
    void* allocate(size_t size, const std::string& tag = "", cudaStream_t stream = 0);

    // 释放内存
    bool deallocate(void* ptr);

    // 预分配内存池
    bool preallocate(const std::vector<size_t>& sizes, const std::vector<std::string>& tags = {});

    // 清理未使用的内存
    void cleanup();

    // 内存碎片整理
    void defragment();

    // 获取内存使用统计
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getPeakUsage() const { return peak_usage_; }
    size_t getCurrentUsage() const { return current_usage_; }
    size_t getAvailableMemory() const;

    // 打印内存使用情况
    void printMemoryInfo() const;

    // 检查内存泄漏
    bool checkMemoryLeaks() const;

    // 重置统计信息
    void resetStats();

    // 设置内存限制
    void setMemoryLimit(size_t limit_bytes);

private:
    // 查找合适的空闲块
    MemoryBlock* findFreeBlock(size_t size);

    // 分割内存块
    void splitBlock(MemoryBlock* block, size_t size);

    // 合并相邻的空闲块
    void mergeBlocks();

    // 内存对齐
    size_t alignSize(size_t size) const;

    // 更新使用统计
    void updateUsageStats();
};

// 权重数据管理器
class WeightDataManager {
private:
    std::map<std::string, void*> weight_data_;
    std::map<std::string, size_t> weight_sizes_;
    std::map<std::string, cudaStream_t> weight_streams_;
    GPUMemoryManager* memory_manager_;
    mutable std::mutex weights_mutex_;

public:
    WeightDataManager(GPUMemoryManager* mem_mgr);
    ~WeightDataManager();

    // 加载权重数据到GPU
    bool loadWeights(const std::string& name, const void* host_data, size_t size,
                    cudaStream_t stream = 0);

    // 获取权重数据
    void* getWeights(const std::string& name) const;

    // 获取权重大小
    size_t getWeightSize(const std::string& name) const;

    // 检查权重是否存在
    bool hasWeights(const std::string& name) const;

    // 卸载权重数据
    bool unloadWeights(const std::string& name);

    // 卸载所有权重
    void unloadAllWeights();

    // 预加载权重数据
    bool preloadWeights(const std::map<std::string, std::pair<const void*, size_t>>& weights_data);

    // 获取权重信息
    std::vector<std::string> getWeightNames() const;
    size_t getTotalWeightSize() const;

    // 打印权重信息
    void printWeightInfo() const;

private:
    // 验证权重数据
    bool validateWeightData(const void* data, size_t size) const;
};

// 张量内存池
class TensorMemoryPool {
private:
    struct TensorSlot {
        void* ptr;
        size_t size;
        std::vector<int> shape;
        bool in_use;
        std::string owner;

        TensorSlot() : ptr(nullptr), size(0), in_use(false) {}
    };

    std::vector<TensorSlot> tensor_slots_;
    GPUMemoryManager* memory_manager_;
    mutable std::mutex pool_mutex_;
    size_t max_pool_size_;

public:
    TensorMemoryPool(GPUMemoryManager* mem_mgr, size_t max_size = 1024 * 1024 * 1024); // 1GB默认
    ~TensorMemoryPool();

    // 分配张量内存
    void* allocateTensor(const std::vector<int>& shape, size_t element_size,
                        const std::string& owner = "");

    // 释放张量内存
    bool deallocateTensor(void* ptr);

    // 预分配张量池
    bool preallocateTensors(const std::vector<std::pair<std::vector<int>, size_t>>& tensor_specs);

    // 清理未使用的张量
    void cleanupUnusedTensors();

    // 获取池使用情况
    size_t getPoolUsage() const;
    size_t getPoolCapacity() const { return max_pool_size_; }

    // 打印池信息
    void printPoolInfo() const;

private:
    // 查找合适的张量槽
    TensorSlot* findSuitableSlot(const std::vector<int>& shape, size_t element_size);

    // 计算张量大小
    size_t calculateTensorSize(const std::vector<int>& shape, size_t element_size) const;

    // 形状匹配
    bool shapesMatch(const std::vector<int>& shape1, const std::vector<int>& shape2) const;
};

// 全局内存管理器实例
class GlobalMemoryManager {
private:
    static std::unique_ptr<GPUMemoryManager> gpu_memory_manager_;
    static std::unique_ptr<WeightDataManager> weight_data_manager_;
    static std::unique_ptr<TensorMemoryPool> tensor_memory_pool_;
    static std::mutex init_mutex_;
    static bool initialized_;

public:
    // 初始化全局内存管理器
    static bool initialize(size_t gpu_memory_limit = 0, size_t tensor_pool_size = 1024 * 1024 * 1024);

    // 获取管理器实例
    static GPUMemoryManager* getGPUMemoryManager();
    static WeightDataManager* getWeightDataManager();
    static TensorMemoryPool* getTensorMemoryPool();

    // 清理所有资源
    static void cleanup();

    // 打印全局内存使用情况
    static void printGlobalMemoryInfo();

private:
    GlobalMemoryManager() = default;
};

} // namespace yolo
} // namespace cuda_learning
