#include "batch_inference_manager.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>

namespace cuda_learning {
namespace yolo {

// BatchMemoryPool 实现
BatchMemoryPool::BatchMemoryPool(GPUMemoryManager* mem_mgr, const BatchInferenceConfig& config)
    : memory_manager_(mem_mgr), config_(config) {
    std::cout << "Batch Memory Pool initialized for batch size: " << config_.max_batch_size << std::endl;
}

BatchMemoryPool::~BatchMemoryPool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto& buffer : batch_buffers_) {
        deallocateBuffer(buffer);
    }
    std::cout << "Batch Memory Pool destroyed" << std::endl;
}

BatchMemoryPool::BatchBuffer* BatchMemoryPool::acquireBatchBuffer(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // 查找空闲缓冲区
    for (auto& buffer : batch_buffers_) {
        if (!buffer.in_use) {
            buffer.in_use = true;
            buffer.stream = stream;
            return &buffer;
        }
    }

    // 没有空闲缓冲区，创建新的
    BatchBuffer new_buffer;
    if (allocateBuffer(new_buffer)) {
        new_buffer.in_use = true;
        new_buffer.stream = stream;
        batch_buffers_.push_back(new_buffer);
        return &batch_buffers_.back();
    }

    std::cerr << "Failed to acquire batch buffer" << std::endl;
    return nullptr;
}

void BatchMemoryPool::releaseBatchBuffer(BatchBuffer* buffer) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (buffer) {
        buffer->in_use = false;
        buffer->stream = 0;
    }
}

bool BatchMemoryPool::preallocateBuffers(int num_buffers) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    std::cout << "Preallocating " << num_buffers << " batch buffers..." << std::endl;

    for (int i = 0; i < num_buffers; ++i) {
        BatchBuffer buffer;
        if (!allocateBuffer(buffer)) {
            std::cerr << "Failed to preallocate buffer " << i << std::endl;
            return false;
        }
        batch_buffers_.push_back(buffer);
    }

    std::cout << "Batch buffer preallocation completed" << std::endl;
    return true;
}

size_t BatchMemoryPool::getPoolUsage() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    size_t total_usage = 0;
    for (const auto& buffer : batch_buffers_) {
        total_usage += buffer.input_size + buffer.output_size + buffer.workspace_size;
    }
    return total_usage;
}

void BatchMemoryPool::printPoolInfo() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    std::cout << "\n=== Batch Memory Pool Statistics ===" << std::endl;
    std::cout << "Number of buffers: " << batch_buffers_.size() << std::endl;
    std::cout << "Total pool usage: " << getPoolUsage() / (1024 * 1024) << " MB" << std::endl;

    size_t used_buffers = 0;
    for (const auto& buffer : batch_buffers_) {
        if (buffer.in_use) used_buffers++;
    }

    std::cout << "Used buffers: " << used_buffers << std::endl;
    std::cout << "Free buffers: " << (batch_buffers_.size() - used_buffers) << std::endl;
    std::cout << "===================================\n" << std::endl;
}

size_t BatchMemoryPool::calculateInputBufferSize() const {
    return config_.max_batch_size * config_.input_channels *
           config_.input_height * config_.input_width * sizeof(float);
}

size_t BatchMemoryPool::calculateOutputBufferSize() const {
    // 假设YOLO输出：3个尺度 * (5 + num_classes) * grid_cells
    // 简化计算：输入大小的1/4作为输出大小
    return calculateInputBufferSize() / 4;
}

size_t BatchMemoryPool::calculateWorkspaceSize() const {
    // 工作空间大小：输入大小的2倍（用于中间计算）
    return calculateInputBufferSize() * 2;
}

bool BatchMemoryPool::allocateBuffer(BatchBuffer& buffer) {
    buffer.input_size = calculateInputBufferSize();
    buffer.output_size = calculateOutputBufferSize();
    buffer.workspace_size = calculateWorkspaceSize();

    // 分配输入缓冲区
    buffer.input_buffer = memory_manager_->allocate(buffer.input_size, "batch_input");
    if (!buffer.input_buffer) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return false;
    }

    // 分配输出缓冲区
    buffer.output_buffer = memory_manager_->allocate(buffer.output_size, "batch_output");
    if (!buffer.output_buffer) {
        std::cerr << "Failed to allocate output buffer" << std::endl;
        memory_manager_->deallocate(buffer.input_buffer);
        return false;
    }

    // 分配工作空间缓冲区
    buffer.workspace_buffer = memory_manager_->allocate(buffer.workspace_size, "batch_workspace");
    if (!buffer.workspace_buffer) {
        std::cerr << "Failed to allocate workspace buffer" << std::endl;
        memory_manager_->deallocate(buffer.input_buffer);
        memory_manager_->deallocate(buffer.output_buffer);
        return false;
    }

    std::cout << "Allocated batch buffer: input=" << buffer.input_size / (1024 * 1024)
              << "MB, output=" << buffer.output_size / (1024 * 1024)
              << "MB, workspace=" << buffer.workspace_size / (1024 * 1024) << "MB" << std::endl;

    return true;
}

void BatchMemoryPool::deallocateBuffer(BatchBuffer& buffer) {
    if (buffer.input_buffer) {
        memory_manager_->deallocate(buffer.input_buffer);
        buffer.input_buffer = nullptr;
    }
    if (buffer.output_buffer) {
        memory_manager_->deallocate(buffer.output_buffer);
        buffer.output_buffer = nullptr;
    }
    if (buffer.workspace_buffer) {
        memory_manager_->deallocate(buffer.workspace_buffer);
        buffer.workspace_buffer = nullptr;
    }
    buffer.input_size = buffer.output_size = buffer.workspace_size = 0;
}

// BatchInferenceManager 实现
BatchInferenceManager::BatchInferenceManager(const BatchInferenceConfig& config, GPUMemoryManager* mem_mgr)
    : config_(config), running_(false) {
    memory_pool_ = std::make_unique<BatchMemoryPool>(mem_mgr, config);
    std::cout << "Batch Inference Manager initialized" << std::endl;
}

BatchInferenceManager::~BatchInferenceManager() {
    stop();
    std::cout << "Batch Inference Manager destroyed" << std::endl;
}

bool BatchInferenceManager::start() {
    if (running_.load()) {
        std::cout << "Batch Inference Manager already running" << std::endl;
        return true;
    }

    // 预分配缓冲区
    if (!memory_pool_->preallocateBuffers(4)) {  // 预分配4个缓冲区
        std::cerr << "Failed to preallocate batch buffers" << std::endl;
        return false;
    }

    running_.store(true);
    batch_thread_ = std::thread(&BatchInferenceManager::batchProcessingLoop, this);

    std::cout << "Batch Inference Manager started" << std::endl;
    return true;
}

void BatchInferenceManager::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    batch_cv_.notify_all();

    if (batch_thread_.joinable()) {
        batch_thread_.join();
    }

    // 处理剩余请求
    std::lock_guard<std::mutex> lock(request_mutex_);
    for (auto& request : pending_requests_) {
        try {
            request->result_promise.set_exception(
                std::make_exception_ptr(std::runtime_error("Batch manager stopped")));
        } catch (...) {
            // 忽略已设置的promise异常
        }
    }
    pending_requests_.clear();

    std::cout << "Batch Inference Manager stopped" << std::endl;
}

std::future<std::vector<operators::FloatTensor>> BatchInferenceManager::submitRequest(
    const std::string& request_id, const operators::FloatTensor& input) {

    if (!running_.load()) {
        throw std::runtime_error("Batch manager not running");
    }

    auto request = std::make_unique<InferenceRequest>(request_id, input);
    auto future = request->result_promise.get_future();

    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        pending_requests_.push_back(std::move(request));
    }

    batch_cv_.notify_one();
    return future;
}

BatchInferenceManager::BatchStats BatchInferenceManager::getStats() const {
    return stats_;
}

void BatchInferenceManager::printStats() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - stats_.start_time);

    std::cout << "\n=== Batch Inference Statistics ===" << std::endl;
    std::cout << "Total requests: " << stats_.total_requests << std::endl;
    std::cout << "Total batches: " << stats_.total_batches << std::endl;
    std::cout << "Average batch size: " << stats_.avg_batch_size << std::endl;
    std::cout << "Average latency: " << stats_.avg_latency_ms << " ms" << std::endl;
    std::cout << "Throughput: " << stats_.throughput_qps << " QPS" << std::endl;
    std::cout << "Running time: " << duration.count() << " seconds" << std::endl;
    std::cout << "=================================\n" << std::endl;
}

void BatchInferenceManager::resetStats() {
    stats_ = BatchStats();
}

void BatchInferenceManager::batchProcessingLoop() {
    std::cout << "Batch processing loop started" << std::endl;

    while (running_.load()) {
        auto batch = collectBatch();

        if (!batch.empty()) {
            executeBatch(batch);
        } else {
            // 等待新请求或超时
            std::unique_lock<std::mutex> lock(request_mutex_);
            batch_cv_.wait_for(lock, std::chrono::milliseconds(static_cast<int>(config_.timeout_ms)));
        }
    }

    std::cout << "Batch processing loop ended" << std::endl;
}

std::vector<std::unique_ptr<InferenceRequest>> BatchInferenceManager::collectBatch() {
    std::lock_guard<std::mutex> lock(request_mutex_);

    std::vector<std::unique_ptr<InferenceRequest>> batch;

    if (pending_requests_.empty()) {
        return batch;
    }

    // 收集批处理请求
    size_t batch_size = std::min(static_cast<size_t>(config_.max_batch_size), pending_requests_.size());

    // 检查是否应该处理批次（基于超时或批次大小）
    if (!shouldProcessBatch() && batch_size < static_cast<size_t>(config_.max_batch_size)) {
        return batch;  // 返回空批次，继续等待
    }

    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(std::move(pending_requests_[i]));
    }

    pending_requests_.erase(pending_requests_.begin(), pending_requests_.begin() + batch_size);

    return batch;
}

void BatchInferenceManager::executeBatch(std::vector<std::unique_ptr<InferenceRequest>>& batch) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 获取批处理缓冲区
    auto* buffer = memory_pool_->acquireBatchBuffer();
    if (!buffer) {
        std::cerr << "Failed to acquire batch buffer" << std::endl;
        // 设置错误结果
        for (auto& request : batch) {
            try {
                request->result_promise.set_exception(
                    std::make_exception_ptr(std::runtime_error("Failed to acquire batch buffer")));
            } catch (...) {}
        }
        return;
    }

    try {
        // 准备批处理输入
        if (!prepareBatchInput(batch, buffer)) {
            throw std::runtime_error("Failed to prepare batch input");
        }

        // 执行推理（这里是简化实现，实际应该调用YOLO模型）
        std::vector<operators::FloatTensor> batch_outputs;

        // 模拟推理过程
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 模拟推理时间

        // 创建输出张量（简化实现）
        for (size_t i = 0; i < batch.size(); ++i) {
            operators::FloatTensor output({1, 85, 8400});  // YOLO输出格式示例
            output.zero();
            batch_outputs.push_back(std::move(output));
        }

        // 分发结果
        distributeBatchResults(batch, batch_outputs);

        // 计算延迟并更新统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float latency_ms = latency.count() / 1000.0f;

        updateStats(batch.size(), latency_ms);

    } catch (const std::exception& e) {
        std::cerr << "Batch execution failed: " << e.what() << std::endl;

        // 设置错误结果
        for (auto& request : batch) {
            try {
                request->result_promise.set_exception(std::current_exception());
            } catch (...) {}
        }
    }

    // 释放缓冲区
    memory_pool_->releaseBatchBuffer(buffer);
}

bool BatchInferenceManager::prepareBatchInput(
    const std::vector<std::unique_ptr<InferenceRequest>>& batch,
    BatchMemoryPool::BatchBuffer* buffer) {

    // 将单个输入张量合并到批处理缓冲区
    float* batch_input = static_cast<float*>(buffer->input_buffer);

    size_t single_input_size = config_.input_channels * config_.input_height * config_.input_width;

    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& input_tensor = batch[i]->input_tensor;

        // 验证输入张量形状
        if (input_tensor.shape().numel() != single_input_size) {
            std::cerr << "Invalid input tensor shape for request: " << batch[i]->request_id << std::endl;
            return false;
        }

        // 复制数据到批处理缓冲区
        cudaError_t error = cudaMemcpy(
            batch_input + i * single_input_size,
            input_tensor.data(),
            single_input_size * sizeof(float),
            cudaMemcpyDeviceToDevice);

        if (error != cudaSuccess) {
            std::cerr << "Failed to copy input data: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }

    return true;
}

void BatchInferenceManager::distributeBatchResults(
    const std::vector<std::unique_ptr<InferenceRequest>>& batch,
    const std::vector<operators::FloatTensor>& batch_outputs) {

    for (size_t i = 0; i < batch.size(); ++i) {
        try {
            std::vector<operators::FloatTensor> result = {batch_outputs[i]};
            batch[i]->result_promise.set_value(std::move(result));
        } catch (...) {
            // 忽略已设置的promise异常
        }
    }
}

void BatchInferenceManager::updateStats(size_t batch_size, float latency_ms) {
    stats_.total_requests += batch_size;
    stats_.total_batches++;

    // 更新平均批次大小
    stats_.avg_batch_size = static_cast<float>(stats_.total_requests) / stats_.total_batches;

    // 更新平均延迟（指数移动平均）
    const float alpha = 0.1f;
    stats_.avg_latency_ms = alpha * latency_ms + (1.0f - alpha) * stats_.avg_latency_ms;

    // 计算吞吐量
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - stats_.start_time);
    if (duration.count() > 0) {
        stats_.throughput_qps = static_cast<float>(stats_.total_requests) / duration.count();
    }
}

bool BatchInferenceManager::shouldProcessBatch() const {
    if (pending_requests_.empty()) {
        return false;
    }

    if (!config_.enable_dynamic_batching) {
        return pending_requests_.size() >= static_cast<size_t>(config_.max_batch_size);
    }

    // 检查最早请求的超时
    auto current_time = std::chrono::high_resolution_clock::now();
    auto oldest_request_time = pending_requests_[0]->submit_time;
    auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - oldest_request_time);

    return wait_time.count() >= config_.timeout_ms;
}

} // namespace yolo
} // namespace cuda_learning
