#include "multi_stream_inference.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>

// 添加缺失的宏定义
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)
#endif

namespace cuda_learning {
namespace yolo {

// InferenceStream 实现
InferenceStream::InferenceStream(const StreamConfig& config, GPUMemoryManager* global_mem_mgr)
    : config_(config), cuda_stream_(nullptr), state_(StreamState::IDLE), running_(false) {

    // 创建CUDA流
    if (config_.priority != 0) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&cuda_stream_, cudaStreamNonBlocking, config_.priority));
    } else {
        CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    }

    // 创建流专用内存管理器（如果有内存限制）
    if (config_.memory_limit > 0) {
        stream_memory_manager_ = std::make_unique<GPUMemoryManager>();
        stream_memory_manager_->setMemoryLimit(config_.memory_limit);
    }

    // 创建批处理管理器
    BatchInferenceConfig batch_config;
    batch_config.max_batch_size = 4;  // 每个流的批处理大小
    batch_config.enable_dynamic_batching = true;
    batch_config.timeout_ms = 5.0f;

    GPUMemoryManager* mem_mgr = stream_memory_manager_ ? stream_memory_manager_.get() : global_mem_mgr;
    batch_manager_ = std::make_unique<BatchInferenceManager>(batch_config, mem_mgr);

    std::cout << "Inference Stream " << config_.stream_id << " created" << std::endl;
}

InferenceStream::~InferenceStream() {
    stop();

    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
    }

    std::cout << "Inference Stream " << config_.stream_id << " destroyed" << std::endl;
}

bool InferenceStream::start() {
    if (running_.load()) {
        std::cout << "Stream " << config_.stream_id << " already running" << std::endl;
        return true;
    }

    // 启动批处理管理器
    if (!batch_manager_->start()) {
        std::cerr << "Failed to start batch manager for stream " << config_.stream_id << std::endl;
        return false;
    }

    running_.store(true);
    state_ = StreamState::IDLE;
    worker_thread_ = std::thread(&InferenceStream::workerLoop, this);

    std::cout << "Inference Stream " << config_.stream_id << " started" << std::endl;
    return true;
}

void InferenceStream::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    queue_cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    // 停止批处理管理器
    batch_manager_->stop();

    // 处理剩余任务
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!task_queue_.empty()) {
        auto task = std::move(task_queue_.front());
        task_queue_.pop();

        try {
            task->result_promise.set_exception(
                std::make_exception_ptr(std::runtime_error("Stream stopped")));
        } catch (...) {
            // 忽略已设置的promise异常
        }
    }

    state_ = StreamState::IDLE;
    std::cout << "Inference Stream " << config_.stream_id << " stopped" << std::endl;
}

bool InferenceStream::submitTask(std::unique_ptr<InferenceTask> task) {
    if (!running_.load()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
        stats_.total_tasks++;
    }

    queue_cv_.notify_one();
    return true;
}

size_t InferenceStream::getQueueLength() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

void InferenceStream::printStats() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - stats_.start_time);

    std::cout << "\n=== Stream " << config_.stream_id << " Statistics ===" << std::endl;
    std::cout << "Total tasks: " << stats_.total_tasks << std::endl;
    std::cout << "Completed tasks: " << stats_.completed_tasks << std::endl;
    std::cout << "Failed tasks: " << stats_.failed_tasks << std::endl;
    std::cout << "Success rate: " << (stats_.total_tasks > 0 ?
        (float)stats_.completed_tasks / stats_.total_tasks * 100.0f : 0.0f) << "%" << std::endl;
    std::cout << "Average latency: " << stats_.avg_latency_ms << " ms" << std::endl;
    std::cout << "Utilization: " << stats_.utilization_ratio * 100.0f << "%" << std::endl;
    std::cout << "Queue length: " << getQueueLength() << std::endl;
    std::cout << "Running time: " << duration.count() << " seconds" << std::endl;
    std::cout << "==============================\n" << std::endl;
}

void InferenceStream::resetStats() {
    stats_ = StreamStats();
}

bool InferenceStream::isIdle() const {
    return state_ == StreamState::IDLE && getQueueLength() == 0;
}

float InferenceStream::getLoad() const {
    // 基于队列长度和利用率计算负载
    size_t queue_len = getQueueLength();
    float queue_factor = std::min(1.0f, queue_len / 10.0f);  // 队列长度因子
    float util_factor = stats_.utilization_ratio;  // 利用率因子

    return (queue_factor + util_factor) / 2.0f;
}

void InferenceStream::workerLoop() {
    std::cout << "Stream " << config_.stream_id << " worker loop started" << std::endl;

    while (running_.load()) {
        std::unique_ptr<InferenceTask> task;

        // 获取任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !task_queue_.empty() || !running_.load(); });

            if (!running_.load()) {
                break;
            }

            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }

        if (task) {
            processTask(std::move(task));
        }
    }

    std::cout << "Stream " << config_.stream_id << " worker loop ended" << std::endl;
}

void InferenceStream::processTask(std::unique_ptr<InferenceTask> task) {
    auto start_time = std::chrono::high_resolution_clock::now();
    state_ = StreamState::BUSY;

    try {
        // 通过批处理管理器执行推理
        auto future = batch_manager_->submitRequest(task->task_id, task->input_tensor);

        // 等待结果
        auto result = future.get();

        // 设置结果
        task->result_promise.set_value(std::move(result));

        // 计算延迟并更新统计
        auto end_time = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float latency_ms = latency.count() / 1000.0f;

        updateStats(true, latency_ms);

    } catch (const std::exception& e) {
        std::cerr << "Task processing failed in stream " << config_.stream_id
                  << ": " << e.what() << std::endl;

        try {
            task->result_promise.set_exception(std::current_exception());
        } catch (...) {
            // 忽略已设置的promise异常
        }

        updateStats(false, 0.0f);
        state_ = StreamState::ERROR;
    }

    state_ = StreamState::IDLE;
    updateUtilization();
}

void InferenceStream::updateStats(bool success, float latency_ms) {
    if (success) {
        stats_.completed_tasks++;

        // 更新平均延迟（指数移动平均）
        const float alpha = 0.1f;
        stats_.avg_latency_ms = alpha * latency_ms + (1.0f - alpha) * stats_.avg_latency_ms;
    } else {
        stats_.failed_tasks++;
    }

    stats_.last_task_time = std::chrono::high_resolution_clock::now();
}

void InferenceStream::updateUtilization() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - stats_.start_time);

    if (total_duration.count() > 0) {
        // 简化的利用率计算：基于完成任务数和平均延迟
        float active_time = stats_.completed_tasks * stats_.avg_latency_ms;
        stats_.utilization_ratio = std::min(1.0f, active_time / total_duration.count());
    }
}

// MultiStreamInferenceSystem 实现
MultiStreamInferenceSystem::MultiStreamInferenceSystem(
    GPUMemoryManager* mem_mgr, const SystemConfig& config)
    : global_memory_manager_(mem_mgr), system_config_(config),
      load_balance_strategy_(config.strategy), round_robin_counter_(0) {

    std::cout << "Multi-Stream Inference System created with " << config.num_streams << " streams" << std::endl;
}

MultiStreamInferenceSystem::~MultiStreamInferenceSystem() {
    shutdown();
    std::cout << "Multi-Stream Inference System destroyed" << std::endl;
}

bool MultiStreamInferenceSystem::initialize() {
    std::cout << "Initializing Multi-Stream Inference System..." << std::endl;

    // 创建推理流
    for (int i = 0; i < system_config_.num_streams; ++i) {
        StreamConfig stream_config(i);
        stream_config.device_id = 0;  // 假设使用设备0
        stream_config.priority = 0;   // 默认优先级

        auto stream = std::make_unique<InferenceStream>(stream_config, global_memory_manager_);
        if (!stream->start()) {
            std::cerr << "Failed to start stream " << i << std::endl;
            return false;
        }

        streams_.push_back(std::move(stream));
    }

    std::cout << "Multi-Stream Inference System initialized successfully" << std::endl;
    return true;
}

void MultiStreamInferenceSystem::shutdown() {
    std::cout << "Shutting down Multi-Stream Inference System..." << std::endl;

    for (auto& stream : streams_) {
        stream->stop();
    }

    streams_.clear();
    std::cout << "Multi-Stream Inference System shutdown completed" << std::endl;
}

std::future<std::vector<operators::FloatTensor>> MultiStreamInferenceSystem::submitInference(
    const std::string& task_id,
    const operators::FloatTensor& input,
    int preferred_stream_id) {

    // 选择最佳流
    int stream_id = selectBestStream(preferred_stream_id);
    if (stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        throw std::runtime_error("No available stream for inference");
    }

    // 创建任务
    auto task = std::make_unique<InferenceTask>(task_id, input);
    task->preferred_stream_id = preferred_stream_id;
    auto future = task->result_promise.get_future();

    // 提交任务
    if (!streams_[stream_id]->submitTask(std::move(task))) {
        throw std::runtime_error("Failed to submit task to stream " + std::to_string(stream_id));
    }

    // 更新系统统计
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        system_stats_.total_tasks++;
    }

    return future;
}

std::vector<std::future<std::vector<operators::FloatTensor>>>
MultiStreamInferenceSystem::submitBatchInference(
    const std::vector<std::string>& task_ids,
    const std::vector<operators::FloatTensor>& inputs) {

    if (task_ids.size() != inputs.size()) {
        throw std::invalid_argument("Task IDs and inputs size mismatch");
    }

    std::vector<std::future<std::vector<operators::FloatTensor>>> futures;
    futures.reserve(task_ids.size());

    for (size_t i = 0; i < task_ids.size(); ++i) {
        futures.push_back(submitInference(task_ids[i], inputs[i]));
    }

    return futures;
}

void MultiStreamInferenceSystem::setLoadBalanceStrategy(LoadBalanceStrategy strategy) {
    load_balance_strategy_ = strategy;
    std::cout << "Load balance strategy changed to: " << static_cast<int>(strategy) << std::endl;
}

InferenceStream* MultiStreamInferenceSystem::getStream(int stream_id) const {
    if (stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return nullptr;
    }
    return streams_[stream_id].get();
}

bool MultiStreamInferenceSystem::isHealthy() const {
    for (const auto& stream : streams_) {
        if (stream->getState() == StreamState::ERROR) {
            return false;
        }
    }
    return true;
}

std::vector<StreamState> MultiStreamInferenceSystem::getStreamStates() const {
    std::vector<StreamState> states;
    states.reserve(streams_.size());

    for (const auto& stream : streams_) {
        states.push_back(stream->getState());
    }

    return states;
}

MultiStreamInferenceSystem::SystemStats MultiStreamInferenceSystem::getSystemStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // 计算系统级统计
    SystemStats stats = system_stats_;

    // 聚合流统计
    size_t total_completed = 0;
    size_t total_failed = 0;
    float total_latency = 0.0f;

    for (const auto& stream : streams_) {
        auto stream_stats = stream->getStats();
        total_completed += stream_stats.completed_tasks;
        total_failed += stream_stats.failed_tasks;
        total_latency += stream_stats.avg_latency_ms;
    }

    stats.completed_tasks = total_completed;
    stats.failed_tasks = total_failed;

    if (!streams_.empty()) {
        stats.avg_system_latency_ms = total_latency / streams_.size();
    }

    // 计算吞吐量
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - stats.start_time);
    if (duration.count() > 0) {
        stats.system_throughput_qps = static_cast<float>(total_completed) / duration.count();
    }

    return stats;
}

void MultiStreamInferenceSystem::printSystemStats() const {
    auto stats = getSystemStats();

    std::cout << "\n=== Multi-Stream System Statistics ===" << std::endl;
    std::cout << "Number of streams: " << streams_.size() << std::endl;
    std::cout << "Total tasks: " << stats.total_tasks << std::endl;
    std::cout << "Completed tasks: " << stats.completed_tasks << std::endl;
    std::cout << "Failed tasks: " << stats.failed_tasks << std::endl;
    std::cout << "System success rate: " << (stats.total_tasks > 0 ?
        (float)stats.completed_tasks / stats.total_tasks * 100.0f : 0.0f) << "%" << std::endl;
    std::cout << "Average system latency: " << stats.avg_system_latency_ms << " ms" << std::endl;
    std::cout << "System throughput: " << stats.system_throughput_qps << " QPS" << std::endl;
    std::cout << "System utilization: " << getSystemUtilization() * 100.0f << "%" << std::endl;
    std::cout << "System health: " << (isHealthy() ? "Healthy" : "Unhealthy") << std::endl;
    std::cout << "====================================\n" << std::endl;
}

void MultiStreamInferenceSystem::printAllStreamStats() const {
    printSystemStats();

    for (const auto& stream : streams_) {
        stream->printStats();
    }
}

void MultiStreamInferenceSystem::resetAllStats() {
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        system_stats_ = SystemStats();
    }

    for (auto& stream : streams_) {
        stream->resetStats();
    }

    std::cout << "All statistics reset" << std::endl;
}

float MultiStreamInferenceSystem::getSystemUtilization() const {
    if (streams_.empty()) return 0.0f;

    float total_utilization = 0.0f;
    for (const auto& stream : streams_) {
        total_utilization += stream->getStats().utilization_ratio;
    }

    return total_utilization / streams_.size();
}

std::vector<float> MultiStreamInferenceSystem::getStreamLoads() const {
    std::vector<float> loads;
    loads.reserve(streams_.size());

    for (const auto& stream : streams_) {
        loads.push_back(stream->getLoad());
    }

    return loads;
}

int MultiStreamInferenceSystem::selectBestStream(int preferred_stream_id) const {
    // 如果指定了首选流且可用，使用首选流
    if (preferred_stream_id >= 0 && preferred_stream_id < static_cast<int>(streams_.size())) {
        if (streams_[preferred_stream_id]->getState() != StreamState::ERROR) {
            return preferred_stream_id;
        }
    }

    // 根据负载均衡策略选择流
    switch (load_balance_strategy_) {
        case LoadBalanceStrategy::ROUND_ROBIN:
            return roundRobinSelect();
        case LoadBalanceStrategy::LEAST_LOADED:
            return leastLoadedSelect();
        case LoadBalanceStrategy::SHORTEST_QUEUE:
            return shortestQueueSelect();
        case LoadBalanceStrategy::ADAPTIVE:
            return adaptiveSelect();
        default:
            return roundRobinSelect();
    }
}

int MultiStreamInferenceSystem::roundRobinSelect() const {
    size_t start_idx = round_robin_counter_.fetch_add(1) % streams_.size();

    // 从当前位置开始查找可用流
    for (size_t i = 0; i < streams_.size(); ++i) {
        size_t idx = (start_idx + i) % streams_.size();
        if (streams_[idx]->getState() != StreamState::ERROR) {
            return static_cast<int>(idx);
        }
    }

    return -1;  // 没有可用流
}

int MultiStreamInferenceSystem::leastLoadedSelect() const {
    int best_stream = -1;
    float min_load = 1.0f;

    for (size_t i = 0; i < streams_.size(); ++i) {
        if (streams_[i]->getState() == StreamState::ERROR) {
            continue;
        }

        float load = streams_[i]->getLoad();
        if (load < min_load) {
            min_load = load;
            best_stream = static_cast<int>(i);
        }
    }

    return best_stream;
}

int MultiStreamInferenceSystem::shortestQueueSelect() const {
    int best_stream = -1;
    size_t min_queue_len = SIZE_MAX;

    for (size_t i = 0; i < streams_.size(); ++i) {
        if (streams_[i]->getState() == StreamState::ERROR) {
            continue;
        }

        size_t queue_len = streams_[i]->getQueueLength();
        if (queue_len < min_queue_len) {
            min_queue_len = queue_len;
            best_stream = static_cast<int>(i);
        }
    }

    return best_stream;
}

int MultiStreamInferenceSystem::adaptiveSelect() const {
    // 自适应策略：综合考虑负载和队列长度
    int best_stream = -1;
    float min_score = 1.0f;

    for (size_t i = 0; i < streams_.size(); ++i) {
        if (streams_[i]->getState() == StreamState::ERROR) {
            continue;
        }

        float load = streams_[i]->getLoad();
        size_t queue_len = streams_[i]->getQueueLength();

        // 综合评分：负载权重0.7，队列长度权重0.3
        float queue_score = std::min(1.0f, queue_len / 10.0f);
        float total_score = 0.7f * load + 0.3f * queue_score;

        if (total_score < min_score) {
            min_score = total_score;
            best_stream = static_cast<int>(i);
        }
    }

    return best_stream;
}

void MultiStreamInferenceSystem::updateSystemStats(bool success, float latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (success) {
        system_stats_.completed_tasks++;

        // 更新系统平均延迟
        const float alpha = 0.1f;
        system_stats_.avg_system_latency_ms = alpha * latency_ms +
            (1.0f - alpha) * system_stats_.avg_system_latency_ms;
    } else {
        system_stats_.failed_tasks++;
    }
}

} // namespace yolo
} // namespace cuda_learning
// InferencePerformanceMonitor 实现
InferencePerformanceMonitor::InferencePerformanceMonitor(
    MultiStreamInferenceSystem* system, const MonitorConfig& config)
    : inference_system_(system), monitor_config_(config), monitoring_(false) {

    std::cout << "Inference Performance Monitor created" << std::endl;
}

InferencePerformanceMonitor::~InferencePerformanceMonitor() {
    stopMonitoring();
    std::cout << "Inference Performance Monitor destroyed" << std::endl;
}

bool InferencePerformanceMonitor::startMonitoring() {
    if (monitoring_.load()) {
        std::cout << "Performance monitor already running" << std::endl;
        return true;
    }

    monitoring_.store(true);
    monitor_thread_ = std::thread(&InferencePerformanceMonitor::monitorLoop, this);

    std::cout << "Performance monitoring started" << std::endl;
    return true;
}

void InferencePerformanceMonitor::stopMonitoring() {
    if (!monitoring_.load()) {
        return;
    }

    monitoring_.store(false);

    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }

    std::cout << "Performance monitoring stopped" << std::endl;
}

void InferencePerformanceMonitor::setMonitorConfig(const MonitorConfig& config) {
    monitor_config_ = config;
}

float InferencePerformanceMonitor::getAverageLatency(size_t window_size) const {
    std::lock_guard<std::mutex> lock(history_mutex_);

    if (performance_history_.latency_history.empty()) {
        return 0.0f;
    }

    size_t start_idx = performance_history_.latency_history.size() > window_size ?
        performance_history_.latency_history.size() - window_size : 0;

    float sum = 0.0f;
    size_t count = 0;

    for (size_t i = start_idx; i < performance_history_.latency_history.size(); ++i) {
        sum += performance_history_.latency_history[i];
        count++;
    }

    return count > 0 ? sum / count : 0.0f;
}

float InferencePerformanceMonitor::getAverageThroughput(size_t window_size) const {
    std::lock_guard<std::mutex> lock(history_mutex_);

    if (performance_history_.throughput_history.empty()) {
        return 0.0f;
    }

    size_t start_idx = performance_history_.throughput_history.size() > window_size ?
        performance_history_.throughput_history.size() - window_size : 0;

    float sum = 0.0f;
    size_t count = 0;

    for (size_t i = start_idx; i < performance_history_.throughput_history.size(); ++i) {
        sum += performance_history_.throughput_history[i];
        count++;
    }

    return count > 0 ? sum / count : 0.0f;
}

float InferencePerformanceMonitor::getAverageUtilization(size_t window_size) const {
    std::lock_guard<std::mutex> lock(history_mutex_);

    if (performance_history_.utilization_history.empty()) {
        return 0.0f;
    }

    size_t start_idx = performance_history_.utilization_history.size() > window_size ?
        performance_history_.utilization_history.size() - window_size : 0;

    float sum = 0.0f;
    size_t count = 0;

    for (size_t i = start_idx; i < performance_history_.utilization_history.size(); ++i) {
        sum += performance_history_.utilization_history[i];
        count++;
    }

    return count > 0 ? sum / count : 0.0f;
}

bool InferencePerformanceMonitor::detectLatencyAnomaly() const {
    float avg_latency = getAverageLatency(50);
    return avg_latency > monitor_config_.alert_threshold_latency_ms;
}

bool InferencePerformanceMonitor::detectUtilizationAnomaly() const {
    float avg_utilization = getAverageUtilization(50);
    return avg_utilization > monitor_config_.alert_threshold_utilization;
}

void InferencePerformanceMonitor::generatePerformanceReport() const {
    std::cout << "\n=== Performance Monitor Report ===" << std::endl;

    float avg_latency = getAverageLatency(100);
    float avg_throughput = getAverageThroughput(100);
    float avg_utilization = getAverageUtilization(100);

    std::cout << "Average Latency (last 100): " << avg_latency << " ms" << std::endl;
    std::cout << "Average Throughput (last 100): " << avg_throughput << " QPS" << std::endl;
    std::cout << "Average Utilization (last 100): " << avg_utilization * 100.0f << "%" << std::endl;

    if (detectLatencyAnomaly()) {
        std::cout << "⚠️  ALERT: High latency detected!" << std::endl;
    }

    if (detectUtilizationAnomaly()) {
        std::cout << "⚠️  ALERT: High utilization detected!" << std::endl;
    }

    std::cout << "================================\n" << std::endl;
}

void InferencePerformanceMonitor::exportPerformanceData(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(history_mutex_);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for export: " << filename << std::endl;
        return;
    }

    file << "timestamp,latency_ms,throughput_qps,utilization\n";

    size_t max_size = std::max({
        performance_history_.latency_history.size(),
        performance_history_.throughput_history.size(),
        performance_history_.utilization_history.size()
    });

    for (size_t i = 0; i < max_size; ++i) {
        file << i << ",";

        if (i < performance_history_.latency_history.size()) {
            file << performance_history_.latency_history[i];
        }
        file << ",";

        if (i < performance_history_.throughput_history.size()) {
            file << performance_history_.throughput_history[i];
        }
        file << ",";

        if (i < performance_history_.utilization_history.size()) {
            file << performance_history_.utilization_history[i];
        }
        file << "\n";
    }

    file.close();
    std::cout << "Performance data exported to: " << filename << std::endl;
}

void InferencePerformanceMonitor::monitorLoop() {
    std::cout << "Performance monitor loop started" << std::endl;

    while (monitoring_.load()) {
        collectPerformanceData();
        checkAlerts();

        if (monitor_config_.enable_auto_scaling) {
            autoScale();
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(monitor_config_.monitor_interval_ms)));
    }

    std::cout << "Performance monitor loop ended" << std::endl;
}

void InferencePerformanceMonitor::collectPerformanceData() {
    if (!inference_system_) return;

    auto stats = inference_system_->getSystemStats();
    float utilization = inference_system_->getSystemUtilization();

    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        performance_history_.addLatency(stats.avg_system_latency_ms);
        performance_history_.addThroughput(stats.system_throughput_qps);
        performance_history_.addUtilization(utilization);
    }
}

void InferencePerformanceMonitor::checkAlerts() {
    if (detectLatencyAnomaly()) {
        std::cout << "🚨 ALERT: High latency detected - "
                  << getAverageLatency(10) << " ms" << std::endl;
    }

    if (detectUtilizationAnomaly()) {
        std::cout << "🚨 ALERT: High utilization detected - "
                  << getAverageUtilization(10) * 100.0f << "%" << std::endl;
    }
}

void InferencePerformanceMonitor::autoScale() {
    // 简化的自动扩缩容逻辑
    float avg_utilization = getAverageUtilization(20);

    if (avg_utilization > 0.9f) {
        std::cout << "🔄 Auto-scaling: High utilization, consider adding more streams" << std::endl;
    } else if (avg_utilization < 0.3f) {
        std::cout << "🔄 Auto-scaling: Low utilization, consider reducing streams" << std::endl;
    }
}

} // namespace yolo
} // namespace cuda_learning
