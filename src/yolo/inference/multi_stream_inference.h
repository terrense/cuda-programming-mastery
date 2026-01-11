#pragma once

#include "../utils/gpu_memory_manager.h"
#include "../../operators/tensor.h"
#include "batch_inference_manager.h"
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <future>
#include <cuda_runtime.h>

namespace cuda_learning {
namespace yolo {

// 流配置
struct StreamConfig {
    int stream_id;
    int device_id = 0;
    int priority = 0;  // CUDA流优先级
    size_t memory_limit = 0;  // 每个流的内存限制（0表示无限制）

    StreamConfig(int id = 0) : stream_id(id) {}
};

// 推理任务
struct InferenceTask {
    std::string task_id;
    operators::FloatTensor input_tensor;
    std::promise<std::vector<operators::FloatTensor>> result_promise;
    std::chrono::high_resolution_clock::time_point submit_time;
    int preferred_stream_id = -1;  // 首选流ID，-1表示自动分配

    InferenceTask(const std::string& id, const operators::FloatTensor& input)
        : task_id(id), input_tensor(input),
          submit_time(std::chrono::high_resolution_clock::now()) {}
};

// 流状态
enum class StreamState {
    IDLE,
    BUSY,
    ERROR
};

// 推理流管理器
class InferenceStream {
private:
    StreamConfig config_;
    cudaStream_t cuda_stream_;
    StreamState state_;

    // 流专用内存管理
    std::unique_ptr<GPUMemoryManager> stream_memory_manager_;
    std::unique_ptr<BatchInferenceManager> batch_manager_;

    // 任务队列
    std::queue<std::unique_ptr<InferenceTask>> task_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 工作线程
    std::thread worker_thread_;
    std::atomic<bool> running_;

    // 性能统计
    struct StreamStats {
        size_t total_tasks = 0;
        size_t completed_tasks = 0;
        size_t failed_tasks = 0;
        float avg_latency_ms = 0.0f;
        float utilization_ratio = 0.0f;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point last_task_time;

        StreamStats() : start_time(std::chrono::high_resolution_clock::now()),
                       last_task_time(start_time) {}
    } stats_;

public:
    InferenceStream(const StreamConfig& config, GPUMemoryManager* global_mem_mgr);
    ~InferenceStream();

    // 启动和停止流
    bool start();
    void stop();

    // 提交任务
    bool submitTask(std::unique_ptr<InferenceTask> task);

    // 获取流状态
    StreamState getState() const { return state_; }
    int getStreamId() const { return config_.stream_id; }
    cudaStream_t getCudaStream() const { return cuda_stream_; }

    // 获取队列长度
    size_t getQueueLength() const;

    // 获取统计信息
    StreamStats getStats() const { return stats_; }
    void printStats() const;
    void resetStats();

    // 检查流是否空闲
    bool isIdle() const;

    // 获取流负载（0.0-1.0）
    float getLoad() const;

private:
    // 工作线程主循环
    void workerLoop();

    // 处理单个任务
    void processTask(std::unique_ptr<InferenceTask> task);

    // 更新统计信息
    void updateStats(bool success, float latency_ms);

    // 计算利用率
    void updateUtilization();
};

// 多流推理系统
class MultiStreamInferenceSystem {
private:
    std::vector<std::unique_ptr<InferenceStream>> streams_;
    GPUMemoryManager* global_memory_manager_;

    // 负载均衡策略
    enum class LoadBalanceStrategy {
        ROUND_ROBIN,
        LEAST_LOADED,
        SHORTEST_QUEUE,
        ADAPTIVE
    };

    LoadBalanceStrategy load_balance_strategy_;
    std::atomic<size_t> round_robin_counter_;

    // 系统配置
    struct SystemConfig {
        int num_streams = 4;
        LoadBalanceStrategy strategy = LoadBalanceStrategy::LEAST_LOADED;
        bool enable_stream_synchronization = true;
        float load_balance_threshold = 0.8f;

        SystemConfig() = default;
    } system_config_;

    // 系统统计
    struct SystemStats {
        size_t total_tasks = 0;
        size_t completed_tasks = 0;
        size_t failed_tasks = 0;
        float avg_system_latency_ms = 0.0f;
        float system_throughput_qps = 0.0f;
        std::chrono::high_resolution_clock::time_point start_time;

        SystemStats() : start_time(std::chrono::high_resolution_clock::now()) {}
    } system_stats_;

    mutable std::mutex stats_mutex_;

public:
    MultiStreamInferenceSystem(GPUMemoryManager* mem_mgr, const SystemConfig& config = SystemConfig());
    ~MultiStreamInferenceSystem();

    // 初始化和清理
    bool initialize();
    void shutdown();

    // 提交推理任务
    std::future<std::vector<operators::FloatTensor>> submitInference(
        const std::string& task_id,
        const operators::FloatTensor& input,
        int preferred_stream_id = -1);

    // 批量提交任务
    std::vector<std::future<std::vector<operators::FloatTensor>>> submitBatchInference(
        const std::vector<std::string>& task_ids,
        const std::vector<operators::FloatTensor>& inputs);

    // 配置管理
    void setLoadBalanceStrategy(LoadBalanceStrategy strategy);
    LoadBalanceStrategy getLoadBalanceStrategy() const { return load_balance_strategy_; }

    // 流管理
    size_t getNumStreams() const { return streams_.size(); }
    InferenceStream* getStream(int stream_id) const;

    // 系统状态
    bool isHealthy() const;
    std::vector<StreamState> getStreamStates() const;

    // 统计信息
    SystemStats getSystemStats() const;
    void printSystemStats() const;
    void printAllStreamStats() const;
    void resetAllStats();

    // 性能监控
    float getSystemUtilization() const;
    std::vector<float> getStreamLoads() const;

    // 动态调整
    bool addStream(const StreamConfig& config);
    bool removeStream(int stream_id);

private:
    // 选择最佳流
    int selectBestStream(int preferred_stream_id = -1) const;

    // 负载均衡算法
    int roundRobinSelect() const;
    int leastLoadedSelect() const;
    int shortestQueueSelect() const;
    int adaptiveSelect() const;

    // 更新系统统计
    void updateSystemStats(bool success, float latency_ms);

    // 健康检查
    void performHealthCheck();

    // 流同步
    void synchronizeStreams();
};

// 推理性能监控器
class InferencePerformanceMonitor {
private:
    MultiStreamInferenceSystem* inference_system_;

    // 监控线程
    std::thread monitor_thread_;
    std::atomic<bool> monitoring_;

    // 监控配置
    struct MonitorConfig {
        float monitor_interval_ms = 1000.0f;  // 监控间隔
        float alert_threshold_latency_ms = 100.0f;  // 延迟告警阈值
        float alert_threshold_utilization = 0.9f;   // 利用率告警阈值
        bool enable_auto_scaling = false;  // 自动扩缩容

        MonitorConfig() = default;
    } monitor_config_;

    // 性能历史
    struct PerformanceHistory {
        std::vector<float> latency_history;
        std::vector<float> throughput_history;
        std::vector<float> utilization_history;
        size_t max_history_size = 1000;

        void addLatency(float latency) {
            latency_history.push_back(latency);
            if (latency_history.size() > max_history_size) {
                latency_history.erase(latency_history.begin());
            }
        }

        void addThroughput(float throughput) {
            throughput_history.push_back(throughput);
            if (throughput_history.size() > max_history_size) {
                throughput_history.erase(throughput_history.begin());
            }
        }

        void addUtilization(float utilization) {
            utilization_history.push_back(utilization);
            if (utilization_history.size() > max_history_size) {
                utilization_history.erase(utilization_history.begin());
            }
        }
    } performance_history_;

    mutable std::mutex history_mutex_;

public:
    InferencePerformanceMonitor(MultiStreamInferenceSystem* system,
                               const MonitorConfig& config = MonitorConfig());
    ~InferencePerformanceMonitor();

    // 启动和停止监控
    bool startMonitoring();
    void stopMonitoring();

    // 配置管理
    void setMonitorConfig(const MonitorConfig& config);
    const MonitorConfig& getMonitorConfig() const { return monitor_config_; }

    // 性能分析
    float getAverageLatency(size_t window_size = 100) const;
    float getAverageThroughput(size_t window_size = 100) const;
    float getAverageUtilization(size_t window_size = 100) const;

    // 异常检测
    bool detectLatencyAnomaly() const;
    bool detectUtilizationAnomaly() const;

    // 性能报告
    void generatePerformanceReport() const;
    void exportPerformanceData(const std::string& filename) const;

private:
    // 监控主循环
    void monitorLoop();

    // 收集性能数据
    void collectPerformanceData();

    // 检查告警条件
    void checkAlerts();

    // 自动扩缩容
    void autoScale();
};

} // namespace yolo
} // namespace cuda_learning
