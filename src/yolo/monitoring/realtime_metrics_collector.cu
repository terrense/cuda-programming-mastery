#include "realtime_metrics_collector.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cmath>
#include <numeric>

namespace yolo_acceleration {

// ExtendedRealtimeMetrics 实现
ExtendedRealtimeMetrics::ExtendedRealtimeMetrics() : RealtimeMetrics(),
    gpu_temperature_celsius(0.0f), gpu_power_usage_watts(0.0f),
    gpu_clock_mhz(0.0f), memory_clock_mhz(0.0f),
    total_memory_mb(0), free_memory_mb(0), used_memory_mb(0),
    memory_fragmentation_percent(0.0f),
    tensor_core_utilization_percent(0.0f), cuda_core_utilization_percent(0.0f),
    active_sm_count(0), total_sm_count(0),
    preprocessing_time_ms(0.0f), inference_time_ms(0.0f),
    postprocessing_time_ms(0.0f), total_pipeline_time_ms(0.0f),
    current_batch_size(1), batch_efficiency_percent(0.0f),
    queue_wait_time_ms(0.0f) {}

std::string ExtendedRealtimeMetrics::toDetailedString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "=== 扩展实时性能指标 ===\n";
    ss << "基础指标:\n";
    ss << "  " << RealtimeMetrics::toString() << "\n\n";

    ss << "GPU硬件指标:\n";
    ss << "  温度: " << gpu_temperature_celsius << "°C\n";
    ss << "  功耗: " << gpu_power_usage_watts << "W\n";
    ss << "  GPU时钟: " << gpu_clock_mhz << "MHz\n";
    ss << "  内存时钟: " << memory_clock_mhz << "MHz\n\n";

    ss << "内存详细指标:\n";
    ss << "  总内存: " << total_memory_mb << "MB\n";
    ss << "  空闲内存: " << free_memory_mb << "MB\n";
    ss << "  已用内存: " << used_memory_mb << "MB\n";
    ss << "  内存碎片化: " << memory_fragmentation_percent << "%\n\n";

    ss << "计算指标:\n";
    ss << "  Tensor Core利用率: " << tensor_core_utilization_percent << "%\n";
    ss << "  CUDA Core利用率: " << cuda_core_utilization_percent << "%\n";
    ss << "  活跃SM: " << active_sm_count << "/" << total_sm_count << "\n\n";

    ss << "推理管道指标:\n";
    ss << "  预处理时间: " << preprocessing_time_ms << "ms\n";
    ss << "  推理时间: " << inference_time_ms << "ms\n";
    ss << "  后处理时间: " << postprocessing_time_ms << "ms\n";
    ss << "  总管道时间: " << total_pipeline_time_ms << "ms\n\n";

    ss << "批处理指标:\n";
    ss << "  当前批大小: " << current_batch_size << "\n";
    ss << "  批效率: " << batch_efficiency_percent << "%\n";
    ss << "  队列等待时间: " << queue_wait_time_ms << "ms\n";

    if (!layer_times_ms.empty()) {
        ss << "\n网络层耗时:\n";
        for (const auto& [layer, time] : layer_times_ms) {
            ss << "  " << layer << ": " << time << "ms\n";
        }
    }

    return ss.str();
}

std::string ExtendedRealtimeMetrics::toJSON() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);

    ss << "{\n";
    ss << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch()).count() << ",\n";
    ss << "  \"fps\": " << fps << ",\n";
    ss << "  \"latency_ms\": " << latency_ms << ",\n";
    ss << "  \"throughput_images_per_sec\": " << throughput_images_per_sec << ",\n";
    ss << "  \"gpu_utilization_percent\": " << gpu_utilization_percent << ",\n";
    ss << "  \"memory_usage_mb\": " << memory_usage_mb << ",\n";
    ss << "  \"gpu_temperature_celsius\": " << gpu_temperature_celsius << ",\n";
    ss << "  \"gpu_power_usage_watts\": " << gpu_power_usage_watts << ",\n";
    ss << "  \"total_memory_mb\": " << total_memory_mb << ",\n";
    ss << "  \"free_memory_mb\": " << free_memory_mb << ",\n";
    ss << "  \"preprocessing_time_ms\": " << preprocessing_time_ms << ",\n";
    ss << "  \"inference_time_ms\": " << inference_time_ms << ",\n";
    ss << "  \"postprocessing_time_ms\": " << postprocessing_time_ms << ",\n";
    ss << "  \"current_batch_size\": " << current_batch_size << ",\n";
    ss << "  \"batch_efficiency_percent\": " << batch_efficiency_percent;

    if (!layer_times_ms.empty()) {
        ss << ",\n  \"layer_times_ms\": {\n";
        bool first = true;
        for (const auto& [layer, time] : layer_times_ms) {
            if (!first) ss << ",\n";
            ss << "    \"" << layer << "\": " << time;
            first = false;
        }
        ss << "\n  }";
    }

    ss << "\n}";
    return ss.str();
}

// RealtimeMetricsCollector 实现
RealtimeMetricsCollector::RealtimeMetricsCollector(size_t max_buffer_size)
    : collecting_(false), max_buffer_size_(max_buffer_size),
      nvml_initialized_(false), collection_interval_(std::chrono::milliseconds(100)),
      total_samples_(0), anomaly_detection_enabled_(false) {

    // 初始化默认启用的指标
    enabled_metrics_ = {
        "fps", "latency", "gpu_utilization", "memory_usage",
        "gpu_temperature", "gpu_power", "pipeline_times"
    };

    start_time_ = std::chrono::steady_clock::now();
}

RealtimeMetricsCollector::~RealtimeMetricsCollector() {
    stopCollection();
    cleanupNVML();

    // 清理CUDA事件
    for (auto& [name, event] : timing_events_) {
        cudaEventDestroy(event);
    }
}

bool RealtimeMetricsCollector::startCollection(std::chrono::milliseconds interval) {
    if (collecting_) {
        return false;
    }

    collection_interval_ = interval;

    // 初始化NVML
    if (!initializeNVML()) {
        std::cerr << "警告: NVML初始化失败，部分GPU指标将不可用\n";
    }

    collecting_ = true;
    collection_thread_ = std::thread(&RealtimeMetricsCollector::collectionLoop, this);

    return true;
}

void RealtimeMetricsCollector::stopCollection() {
    if (collecting_) {
        collecting_ = false;
        if (collection_thread_.joinable()) {
            collection_thread_.join();
        }
    }
}

void RealtimeMetricsCollector::setCollectionInterval(std::chrono::milliseconds interval) {
    collection_interval_ = interval;
}

void RealtimeMetricsCollector::enableMetric(const std::string& metric_name) {
    auto it = std::find(enabled_metrics_.begin(), enabled_metrics_.end(), metric_name);
    if (it == enabled_metrics_.end()) {
        enabled_metrics_.push_back(metric_name);
    }
}

void RealtimeMetricsCollector::disableMetric(const std::string& metric_name) {
    enabled_metrics_.erase(
        std::remove(enabled_metrics_.begin(), enabled_metrics_.end(), metric_name),
        enabled_metrics_.end());
}

void RealtimeMetricsCollector::setEnabledMetrics(const std::vector<std::string>& metrics) {
    enabled_metrics_ = metrics;
}

void RealtimeMetricsCollector::recordPipelineStart(const std::string& pipeline_id) {
    cpu_timers_[pipeline_id + "_start"] = std::chrono::steady_clock::now();

    // 创建CUDA事件（如果不存在）
    if (timing_events_.find(pipeline_id + "_start") == timing_events_.end()) {
        cudaEvent_t start_event, end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        timing_events_[pipeline_id + "_start"] = start_event;
        timing_events_[pipeline_id + "_end"] = end_event;
    }

    cudaEventRecord(timing_events_[pipeline_id + "_start"]);
}

void RealtimeMetricsCollector::recordPipelineEnd(const std::string& pipeline_id, int batch_size) {
    auto end_time = std::chrono::steady_clock::now();
    cpu_timers_[pipeline_id + "_end"] = end_time;

    if (timing_events_.find(pipeline_id + "_end") != timing_events_.end()) {
        cudaEventRecord(timing_events_[pipeline_id + "_end"]);
        cudaEventSynchronize(timing_events_[pipeline_id + "_end"]);

        // 计算GPU时间
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms,
                           timing_events_[pipeline_id + "_start"],
                           timing_events_[pipeline_id + "_end"]);

        // 计算CPU时间
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - cpu_timers_[pipeline_id + "_start"]);
        float cpu_time_ms = cpu_duration.count() / 1000.0f;

        // 创建指标记录
        ExtendedRealtimeMetrics metrics;
        metrics.total_pipeline_time_ms = gpu_time_ms;
        metrics.current_batch_size = batch_size;
        metrics.fps = batch_size * 1000.0f / gpu_time_ms;
        metrics.latency_ms = gpu_time_ms / batch_size;
        metrics.timestamp = end_time;

        // 添加到缓冲区
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_buffer_.push(metrics);
        if (metrics_buffer_.size() > max_buffer_size_) {
            metrics_buffer_.pop();
        }

        // 调用回调函数
        for (const auto& callback : callbacks_) {
            callback(metrics);
        }
    }
}

void RealtimeMetricsCollector::recordLayerStart(const std::string& layer_name) {
    cpu_timers_[layer_name + "_layer_start"] = std::chrono::steady_clock::now();

    if (timing_events_.find(layer_name + "_layer_start") == timing_events_.end()) {
        cudaEvent_t start_event, end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        timing_events_[layer_name + "_layer_start"] = start_event;
        timing_events_[layer_name + "_layer_end"] = end_event;
    }

    cudaEventRecord(timing_events_[layer_name + "_layer_start"]);
}

void RealtimeMetricsCollector::recordLayerEnd(const std::string& layer_name) {
    if (timing_events_.find(layer_name + "_layer_end") != timing_events_.end()) {
        cudaEventRecord(timing_events_[layer_name + "_layer_end"]);
        cudaEventSynchronize(timing_events_[layer_name + "_layer_end"]);

        float layer_time_ms;
        cudaEventElapsedTime(&layer_time_ms,
                           timing_events_[layer_name + "_layer_start"],
                           timing_events_[layer_name + "_layer_end"]);

        // 记录层时间到最新的指标中
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        if (!metrics_buffer_.empty()) {
            auto& latest_metrics = const_cast<ExtendedRealtimeMetrics&>(metrics_buffer_.back());
            latest_metrics.layer_times_ms[layer_name] = layer_time_ms;
        }
    }
}

void RealtimeMetricsCollector::recordCustomMetric(const std::string& name, float value) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!metric_history_[name].empty()) {
        metric_history_[name].push_back(value);

        // 限制历史记录大小
        if (metric_history_[name].size() > max_buffer_size_) {
            metric_history_[name].erase(metric_history_[name].begin());
        }
    } else {
        metric_history_[name] = {value};
    }
}

ExtendedRealtimeMetrics RealtimeMetricsCollector::getCurrentMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!metrics_buffer_.empty()) {
        return metrics_buffer_.back();
    }
    return ExtendedRealtimeMetrics();
}

std::vector<ExtendedRealtimeMetrics> RealtimeMetricsCollector::getRecentMetrics(size_t count) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    std::vector<ExtendedRealtimeMetrics> result;

    std::queue<ExtendedRealtimeMetrics> temp_queue = metrics_buffer_;
    while (!temp_queue.empty() && result.size() < count) {
        result.insert(result.begin(), temp_queue.back());
        temp_queue.pop();
    }

    return result;
}

ExtendedRealtimeMetrics RealtimeMetricsCollector::calculateAverageMetrics(size_t window_size) {
    auto recent_metrics = getRecentMetrics(window_size);
    if (recent_metrics.empty()) {
        return ExtendedRealtimeMetrics();
    }

    ExtendedRealtimeMetrics avg;
    for (const auto& metrics : recent_metrics) {
        avg.fps += metrics.fps;
        avg.latency_ms += metrics.latency_ms;
        avg.gpu_utilization_percent += metrics.gpu_utilization_percent;
        avg.memory_usage_mb += metrics.memory_usage_mb;
        avg.gpu_temperature_celsius += metrics.gpu_temperature_celsius;
        avg.gpu_power_usage_watts += metrics.gpu_power_usage_watts;
        avg.preprocessing_time_ms += metrics.preprocessing_time_ms;
        avg.inference_time_ms += metrics.inference_time_ms;
        avg.postprocessing_time_ms += metrics.postprocessing_time_ms;
    }

    float count = static_cast<float>(recent_metrics.size());
    avg.fps /= count;
    avg.latency_ms /= count;
    avg.gpu_utilization_percent /= count;
    avg.memory_usage_mb /= count;
    avg.gpu_temperature_celsius /= count;
    avg.gpu_power_usage_watts /= count;
    avg.preprocessing_time_ms /= count;
    avg.inference_time_ms /= count;
    avg.postprocessing_time_ms /= count;

    return avg;
}

void RealtimeMetricsCollector::registerCallback(
    std::function<void(const ExtendedRealtimeMetrics&)> callback) {
    callbacks_.push_back(callback);
}

bool RealtimeMetricsCollector::exportMetricsToCSV(const std::string& filename, size_t max_records) {
    auto metrics = getRecentMetrics(max_records > 0 ? max_records : max_buffer_size_);

    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // 写入CSV头部
    file << "timestamp,fps,latency_ms,gpu_utilization_percent,memory_usage_mb,"
         << "gpu_temperature_celsius,gpu_power_usage_watts,preprocessing_time_ms,"
         << "inference_time_ms,postprocessing_time_ms,current_batch_size\n";

    // 写入数据
    for (const auto& metric : metrics) {
        auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            metric.timestamp.time_since_epoch()).count();

        file << timestamp_ms << ","
             << metric.fps << ","
             << metric.latency_ms << ","
             << metric.gpu_utilization_percent << ","
             << metric.memory_usage_mb << ","
             << metric.gpu_temperature_celsius << ","
             << metric.gpu_power_usage_watts << ","
             << metric.preprocessing_time_ms << ","
             << metric.inference_time_ms << ","
             << metric.postprocessing_time_ms << ","
             << metric.current_batch_size << "\n";
    }

    file.close();
    return true;
}

std::string RealtimeMetricsCollector::generateMetricsSummary() {
    auto avg_metrics = calculateAverageMetrics(100);
    auto recent_metrics = getRecentMetrics(10);

    std::stringstream ss;
    ss << "=== 实时指标收集器摘要 ===\n";
    ss << "收集状态: " << (collecting_ ? "运行中" : "已停止") << "\n";
    ss << "总样本数: " << total_samples_ << "\n";
    ss << "缓冲区大小: " << metrics_buffer_.size() << "/" << max_buffer_size_ << "\n";
    ss << "收集间隔: " << collection_interval_.count() << "ms\n\n";

    ss << "平均性能指标 (最近100次):\n";
    ss << avg_metrics.toString() << "\n\n";

    if (!recent_metrics.empty()) {
        ss << "最新指标:\n";
        ss << recent_metrics.back().toDetailedString() << "\n";
    }

    return ss.str();
}

// 私有方法实现
void RealtimeMetricsCollector::collectionLoop() {
    while (collecting_) {
        ExtendedRealtimeMetrics metrics;

        // 收集各种指标
        if (std::find(enabled_metrics_.begin(), enabled_metrics_.end(), "gpu_utilization") != enabled_metrics_.end()) {
            collectGPUHardwareMetrics(metrics);
        }

        if (std::find(enabled_metrics_.begin(), enabled_metrics_.end(), "memory_usage") != enabled_metrics_.end()) {
            collectMemoryMetrics(metrics);
        }

        if (std::find(enabled_metrics_.begin(), enabled_metrics_.end(), "compute_metrics") != enabled_metrics_.end()) {
            collectComputeMetrics(metrics);
        }

        calculateDerivedMetrics(metrics);
        updateRunningStatistics(metrics);

        // 异常检测
        if (anomaly_detection_enabled_) {
            checkForAnomalies(metrics);
        }

        // 添加到缓冲区
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_buffer_.push(metrics);
            if (metrics_buffer_.size() > max_buffer_size_) {
                metrics_buffer_.pop();
            }
            total_samples_++;
        }

        // 调用回调函数
        for (const auto& callback : callbacks_) {
            callback(metrics);
        }

        std::this_thread::sleep_for(collection_interval_);
    }
}

bool RealtimeMetricsCollector::initializeNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        return false;
    }

    result = nvmlDeviceGetHandleByIndex(0, &nvml_device_);
    if (result != NVML_SUCCESS) {
        nvmlShutdown();
        return false;
    }

    nvml_initialized_ = true;
    return true;
}

void RealtimeMetricsCollector::cleanupNVML() {
    if (nvml_initialized_) {
        nvmlShutdown();
        nvml_initialized_ = false;
    }
}

void RealtimeMetricsCollector::collectGPUHardwareMetrics(ExtendedRealtimeMetrics& metrics) {
    if (!nvml_initialized_) {
        return;
    }

    // 温度
    unsigned int temperature;
    if (nvmlDeviceGetTemperature(nvml_device_, NVML_TEMPERATURE_GPU, &temperature) == NVML_SUCCESS) {
        metrics.gpu_temperature_celsius = static_cast<float>(temperature);
    }

    // 功耗
    unsigned int power;
    if (nvmlDeviceGetPowerUsage(nvml_device_, &power) == NVML_SUCCESS) {
        metrics.gpu_power_usage_watts = static_cast<float>(power) / 1000.0f;
    }

    // 时钟频率
    unsigned int clock;
    if (nvmlDeviceGetClockInfo(nvml_device_, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS) {
        metrics.gpu_clock_mhz = static_cast<float>(clock);
    }

    if (nvmlDeviceGetClockInfo(nvml_device_, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
        metrics.memory_clock_mhz = static_cast<float>(clock);
    }

    // GPU利用率
    nvmlUtilization_t utilization;
    if (nvmlDeviceGetUtilizationRates(nvml_device_, &utilization) == NVML_SUCCESS) {
        metrics.gpu_utilization_percent = static_cast<float>(utilization.gpu);
    }
}

void RealtimeMetricsCollector::collectMemoryMetrics(ExtendedRealtimeMetrics& metrics) {
    // CUDA内存信息
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        metrics.total_memory_mb = total_mem / (1024 * 1024);
        metrics.free_memory_mb = free_mem / (1024 * 1024);
        metrics.used_memory_mb = metrics.total_memory_mb - metrics.free_memory_mb;
        metrics.memory_usage_mb = static_cast<float>(metrics.used_memory_mb);

        // 简单的碎片化估算
        float usage_ratio = static_cast<float>(metrics.used_memory_mb) / metrics.total_memory_mb;
        metrics.memory_fragmentation_percent = usage_ratio * 10.0f; // 简化估算
    }
}

void RealtimeMetricsCollector::collectComputeMetrics(ExtendedRealtimeMetrics& metrics) {
    // 获取设备属性
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        metrics.total_sm_count = prop.multiProcessorCount;

        // 简化的活跃SM估算（基于GPU利用率）
        metrics.active_sm_count = static_cast<int>(
            metrics.gpu_utilization_percent / 100.0f * metrics.total_sm_count);

        // 估算Tensor Core和CUDA Core利用率
        if (prop.major >= 7) { // Volta及以上架构支持Tensor Core
            metrics.tensor_core_utilization_percent = metrics.gpu_utilization_percent * 0.3f;
            metrics.cuda_core_utilization_percent = metrics.gpu_utilization_percent * 0.7f;
        } else {
            metrics.tensor_core_utilization_percent = 0.0f;
            metrics.cuda_core_utilization_percent = metrics.gpu_utilization_percent;
        }
    }
}

void RealtimeMetricsCollector::calculateDerivedMetrics(ExtendedRealtimeMetrics& metrics) {
    metrics.timestamp = std::chrono::steady_clock::now();

    // 计算批效率
    if (metrics.current_batch_size > 0 && metrics.total_pipeline_time_ms > 0) {
        float ideal_time = metrics.total_pipeline_time_ms / metrics.current_batch_size;
        float actual_time = metrics.total_pipeline_time_ms;
        metrics.batch_efficiency_percent = (ideal_time / actual_time) * 100.0f;
    }

    // 计算吞吐量
    if (metrics.total_pipeline_time_ms > 0) {
        metrics.throughput_images_per_sec =
            (metrics.current_batch_size * 1000.0f) / metrics.total_pipeline_time_ms;
    }
}

void RealtimeMetricsCollector::updateRunningStatistics(const ExtendedRealtimeMetrics& metrics) {
    // 更新运行统计信息
    std::vector<std::pair<std::string, float>> metric_pairs = {
        {"fps", metrics.fps},
        {"latency_ms", metrics.latency_ms},
        {"gpu_utilization_percent", metrics.gpu_utilization_percent},
        {"memory_usage_mb", metrics.memory_usage_mb},
        {"gpu_temperature_celsius", metrics.gpu_temperature_celsius}
    };

    for (const auto& [name, value] : metric_pairs) {
        if (running_means_.find(name) == running_means_.end()) {
            running_means_[name] = value;
            running_variances_[name] = 0.0f;
        } else {
            // 使用Welford算法更新均值和方差
            float delta = value - running_means_[name];
            running_means_[name] += delta / total_samples_;
            float delta2 = value - running_means_[name];
            running_variances_[name] += delta * delta2;
        }
    }
}

// AnomalyDetectionConfig 实现
RealtimeMetricsCollector::AnomalyDetectionConfig::AnomalyDetectionConfig()
    : threshold_multiplier(2.0f), window_size(50),
      monitored_metrics({"fps", "latency_ms", "gpu_temperature_celsius"}) {}

// AnomalyAlert 实现
std::string RealtimeMetricsCollector::AnomalyAlert::toString() const {
    std::stringstream ss;
    ss << "异常告警 [" << severity << "]: " << metric_name
       << " = " << std::fixed << std::setprecision(2) << current_value
       << " (期望: " << expected_value
       << ", 偏差: " << deviation_percent << "%)";
    return ss.str();
}

void RealtimeMetricsCollector::enableAnomalyDetection(const AnomalyDetectionConfig& config) {
    anomaly_config_ = config;
    anomaly_detection_enabled_ = true;
}

void RealtimeMetricsCollector::disableAnomalyDetection() {
    anomaly_detection_enabled_ = false;
}

void RealtimeMetricsCollector::checkForAnomalies(const ExtendedRealtimeMetrics& metrics) {
    for (const std::string& metric_name : anomaly_config_.monitored_metrics) {
        float current_value = 0.0f;

        if (metric_name == "fps") current_value = metrics.fps;
        else if (metric_name == "latency_ms") current_value = metrics.latency_ms;
        else if (metric_name == "gpu_temperature_celsius") current_value = metrics.gpu_temperature_celsius;
        else if (metric_name == "gpu_utilization_percent") current_value = metrics.gpu_utilization_percent;

        if (running_means_.find(metric_name) != running_means_.end()) {
            float expected_value = running_means_[metric_name];

            if (isAnomalous(metric_name, current_value, expected_value)) {
                AnomalyAlert alert;
                alert.metric_name = metric_name;
                alert.current_value = current_value;
                alert.expected_value = expected_value;
                alert.deviation_percent = std::abs((current_value - expected_value) / expected_value) * 100.0f;
                alert.timestamp = metrics.timestamp;

                if (alert.deviation_percent > 50.0f) alert.severity = "high";
                else if (alert.deviation_percent > 25.0f) alert.severity = "medium";
                else alert.severity = "low";

                recent_anomalies_.push_back(alert);
                if (recent_anomalies_.size() > 100) {
                    recent_anomalies_.erase(recent_anomalies_.begin());
                }

                // 调用异常回调
                for (const auto& callback : anomaly_callbacks_) {
                    callback(alert);
                }
            }
        }
    }
}

bool RealtimeMetricsCollector::isAnomalous(const std::string& metric_name,
                                          float current_value, float expected_value) {
    if (running_variances_.find(metric_name) == running_variances_.end()) {
        return false;
    }

    float variance = running_variances_[metric_name] / (total_samples_ - 1);
    float std_dev = std::sqrt(variance);
    float threshold = anomaly_config_.threshold_multiplier * std_dev;

    return std::abs(current_value - expected_value) > threshold;
}

} // namespace yolo_acceleration
