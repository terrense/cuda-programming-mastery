#pragma once

#include "../utils/gpu_memory_manager.h"
#include "../../operators/tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <future>
#include <chrono>
#include <cuda_runtime.h>

namespace cuda_learning {
namespace yolo {

// 批处理推理配置
struct BatchInferenceConfig {
    int max_batch_size = 8;
    int input_height = 640;
    int input_width = 640;
    int input_channels = 3;
    bool enable_dynamic_batching = true;
    float timeout_ms = 10.0f;  // 动态批处理超时时间

    BatchInferenceConfig() = default;
    BatchInferenceConfig(int batch_size, int h, int w, int c = 3)
        : max_batch_size(batch_size), input_height(h), input_width(w), input_channels(c) {}
};

// 推理请求
struct InferenceRequest {
    std::string request_id;
    operators::FloatTensor input_tensor;
    std::promise<std::vector<operators::FloatTensor>> result_promise;
    std::chrono::high_resolution_clock::time_point submit_time;

    InferenceRequest(const std::string& id, const operators::FloatTensor& input)
        : request_id(id), input_tensor(input), submit_time(std::chrono::high_resolution_clock::now()) {}
};

// 批处理内存池管理器
class BatchMemoryPool {
private:
    struct BatchBuffer {
        void* input_buffer;      // 批处理输入缓冲区
        void* output_buffer;     // 批处理输出缓冲区
        void* workspace_buffer;  // 工作空间缓冲区
        size_t input_size;
        size_t output_size;
        size_t workspace_size;
        bool in_use;
        cudaStream_t stream;

        BatchBuffer() : input_buffer(nullptr), output_buffer(nullptr), workspace_buffer(nullptr),
                       input_size(0), output_size(0), workspace_size(0), in_use(false), stream(0) {}
    };

    std::vector<BatchBuffer> batch_buffers_;
    GPUMemoryManager* memory_manager_;
    BatchInferenceConfig config_;
    mutable std::mutex pool_mutex_;

public:
    BatchMemoryPool(GPUMemoryManager* mem_mgr, const BatchInferenceConfig& config);
    ~BatchMemoryPool();

    // 获取批处理缓冲区
    BatchBuffer* acquireBatchBuffer(cudaStream_t stream = 0);

    // 释放批处理缓冲区
    void releaseBatchBuffer(BatchBuffer* buffer);

    // 预分配缓冲区
    bool preallocateBuffers(int num_buffers);

    // 获取池使用情况
    size_t getPoolUsage() const;
    void printPoolInfo() const;

private:
    // 计算缓冲区大小
    size_t calculateInputBufferSize() const;
    size_t calculateOutputBufferSize() const;
    size_t calculateWorkspaceSize() const;

    // 分配单个缓冲区
    bool allocateBuffer(BatchBuffer& buffer);
    void deallocateBuffer(BatchBuffer& buffer);
};

// 批处理推理管理器
class BatchInferenceManager {
private:
    BatchInferenceConfig config_;
    std::unique_ptr<BatchMemoryPool> memory_pool_;

    // 请求队列
    std::vector<std::unique_ptr<InferenceRequest>> pending_requests_;
    mutable std::mutex request_mutex_;

    // 批处理线程
    std::thread batch_thread_;
    std::atomic<bool> running_;
    std::condition_variable batch_cv_;

    // 性能统计
    struct BatchStats {
        size_t total_requests = 0;
        size_t total_batches = 0;
        float avg_batch_size = 0.0f;
        float avg_latency_ms = 0.0f;
        float throughput_qps = 0.0f;
        std::chrono::high_resolution_clock::time_point start_time;

        BatchStats() : start_time(std::chrono::high_resolution_clock::now()) {}
    } stats_;

public:
    BatchInferenceManager(const BatchInferenceConfig& config, GPUMemoryManager* mem_mgr);
    ~BatchInferenceManager();

    // 启动批处理服务
    bool start();

    // 停止批处理服务
    void stop();

    // 提交推理请求
    std::future<std::vector<operators::FloatTensor>> submitRequest(
        const std::string& request_id,
        const operators::FloatTensor& input);

    // 获取配置
    const BatchInferenceConfig& getConfig() const { return config_; }

    // 获取统计信息
    BatchStats getStats() const;
    void printStats() const;
    void resetStats();

private:
    // 批处理主循环
    void batchProcessingLoop();

    // 收集批处理请求
    std::vector<std::unique_ptr<InferenceRequest>> collectBatch();

    // 执行批处理推理
    void executeBatch(std::vector<std::unique_ptr<InferenceRequest>>& batch);

    // 准备批处理输入
    bool prepareBatchInput(const std::vector<std::unique_ptr<InferenceRequest>>& batch,
                          BatchMemoryPool::BatchBuffer* buffer);

    // 分发批处理结果
    void distributeBatchResults(const std::vector<std::unique_ptr<InferenceRequest>>& batch,
                               const std::vector<operators::FloatTensor>& batch_outputs);

    // 更新统计信息
    void updateStats(size_t batch_size, float latency_ms);

    // 检查超时
    bool shouldProcessBatch() const;
};

} // namespace yolo
} // namespace cuda_learning
