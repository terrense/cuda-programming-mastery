#ifndef REALTIME_METRICS_COLLECTOR_H
#define REALTIME_METRICS_COLLECTOR_H

#include "performance_monitor.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <functional>

namespace yolo_acceleration {

// 扩展的实时性能指标
struct ExtendedRealtimeMetrics : public RealtimeMetrics {
    // GPU硬件指标
    float gpu_temperature_celsius;
    float gpu_power_usage_watts;
    float gpu_clock_mhz;
    float memory_clock_mhz;

    // 内存详细指标
    size_t total_memory_mb;
    size_t free_memory_mb;
    size_t used_memory_mb;
    float memory_fragmentation_percent;

    // 计算指标
    float tensor_core_utilization_percent;
    float cuda_core_utilization_percent;
    int active_sm_count;
    int total_sm_count;

    // 推理管道指标
    float preprocessing_time_ms;
    float inference_time_ms;
    float postprocessing_time_ms;
    float total_pipeline_time_ms;

    // 批处理指标
    int current_batch_size;
    float batch_efficiency_percent;
    float queue_wait_time_ms;

    // 网络层级指标
    std::map<std::string, float> layer_times_ms;
    std::map<std::string, float> layer_memory_usage_mb;

    ExtendedRealtimeMetrics();
    std::string toDetailedString() const;
    std::string toJSON() const;
};

// 实时指标收集器
class RealtimeMetricsCollector {
private:
    std::atomic<bool> collecting_;
    std::thread collection_thread_;
    std::mutex metrics_mutex_;
    std::queue<ExtendedRealtimeMetrics> metrics_buffer_;
    size_t max_buffer_size_;

    // NVML句柄
    nvmlDevice_t nvml_device_;
    bool nvml_initialized_;

    // CUDA事件
    std::map<std::string, cudaEvent_t> timing_events_;
    std::map<std::string, std::chrono::steady_clock::time_point> cpu_timers_;

    // 采样配置
    std::chrono::milliseconds collection_interval_;
    std::vector<std::string> enabled_metrics_;

    // 回调函数
    std::vector<std::function<void(const ExtendedRealtimeMetrics&)>> callbacks_;

    // 统计信息
    size_t total_samples_;
    std::chrono::steady_clock::time_point start_time_;

public:
    RealtimeMetricsCollector(size_t max_buffer_size = 10000);
    ~RealtimeMetricsCollector();

    // 启动/停止收集
    bool startCollection(std::chrono::milliseconds interval = std::chrono::milliseconds(100));
    void stopCollection();
    bool isCollecting() const { return collecting_; }

    // 配置收集器
    void setCollectionInterval(std::chrono::milliseconds interval);
    void enableMetric(const std::string& metric_name);
    void disableMetric(const std::string& metric_name);
    void setEnabledMetrics(const std::vector<std::string>& metrics);

    // 手动记录指标
    void recordPipelineStart(const std::string& pipeline_id);
    void recordPipelineEnd(const std::string& pipeline_id, int batch_size);
    void recordLayerStart(const std::string& layer_name);
    void recordLayerEnd(const std::string& layer_name);
    void recordCustomMetric(const std::string& name, float value);

    // 获取指标数据
    ExtendedRealtimeMetrics getCurrentMetrics();
    std::vector<ExtendedRealtimeMetrics> getRecentMetrics(size_t count = 100);
    std::vector<ExtendedRealtimeMetrics> getMetricsInTimeRange(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end);

    // 统计分析
    ExtendedRealtimeMetrics calculateAverageMetrics(size_t window_size = 100);
    ExtendedRealtimeMetrics calculatePercentileMetrics(float percentile, size_t window_size = 100);
    std::map<std::string, float> calculateMetricTrends(size_t window_size = 100);

    // 异常检测
    struct AnomalyDetectionConfig {
        float threshold_multiplier;
        size_t window_size;
        std::vector<std::string> monitored_metrics;

        AnomalyDetectionConfig();
    };

    struct AnomalyAlert {
        std::string metric_name;
        float current_value;
        float expected_value;
        float deviation_percent;
        std::chrono::steady_clock::time_point timestamp;
        std::string severity; // "low", "medium", "high"

        std::string toString() const;
    };

    void enableAnomalyDetection(const AnomalyDetectionConfig& config);
    void disableAnomalyDetection();
    std::vector<AnomalyAlert> getRecentAnomalies(size_t count = 10);

    // 回调注册
    void registerCallback(std::function<void(const ExtendedRealtimeMetrics&)> callback);
    void registerAnomalyCallback(std::function<void(const AnomalyAlert&)> callback);

    // 数据导出
    bool exportMetricsToCSV(const std::string& filename, size_t max_records = 0);
    bool exportMetricsToJSON(const std::string& filename, size_t max_records = 0);
    std::string generateMetricsSummary();

private:
    // 收集循环
    void collectionLoop();

    // 硬件指标收集
    bool initializeNVML();
    void cleanupNVML();
    void collectGPUHardwareMetrics(ExtendedRealtimeMetrics& metrics);
    void collectMemoryMetrics(ExtendedRealtimeMetrics& metrics);
    void collectComputeMetrics(ExtendedRealtimeMetrics& metrics);

    // 性能计算
    void calculateDerivedMetrics(ExtendedRealtimeMetrics& metrics);
    void updateRunningStatistics(const ExtendedRealtimeMetrics& metrics);

    // 异常检测
    AnomalyDetectionConfig anomaly_config_;
    bool anomaly_detection_enabled_;
    std::vector<AnomalyAlert> recent_anomalies_;
    std::vector<std::function<void(const AnomalyAlert&)>> anomaly_callbacks_;

    void checkForAnomalies(const ExtendedRealtimeMetrics& metrics);
    bool isAnomalous(const std::string& metric_name, float current_value, float expected_value);

    // 统计数据
    std::map<std::string, std::vector<float>> metric_history_;
    std::map<std::string, float> running_means_;
    std::map<std::string, float> running_variances_;
};

// 性能指标聚合器
class MetricsAggregator {
public:
    // 聚合配置
    struct AggregationConfig {
        std::chrono::seconds window_size;
        std::vector<std::string> aggregation_functions; // "mean", "max", "min", "p95", "p99"
        std::vector<std::string> group_by_fields;
        bool enable_real_time_aggregation;

        AggregationConfig();
    };

    // 聚合结果
    struct AggregatedMetrics {
        std::chrono::steady_clock::time_point window_start;
        std::chrono::steady_clock::time_point window_end;
        std::map<std::string, float> aggregated_values;
        size_t sample_count;

        std::string toString() const;
    };

private:
    AggregationConfig config_;
    std::vector<ExtendedRealtimeMetrics> current_window_;
    std::vector<AggregatedMetrics> aggregation_history_;
    std::mutex aggregation_mutex_;

    std::thread aggregation_thread_;
    std::atomic<bool> aggregating_;

public:
    MetricsAggregator(const AggregationConfig& config);
    ~MetricsAggregator();

    void startAggregation();
    void stopAggregation();

    void addMetrics(const ExtendedRealtimeMetrics& metrics);
    std::vector<AggregatedMetrics> getAggregatedHistory(size_t count = 100);
    AggregatedMetrics getCurrentAggregation();

    // 自定义聚合函数
    void registerCustomAggregationFunction(
        const std::string& name,
        std::function<float(const std::vector<float>&)> func);

private:
    void aggregationLoop();
    AggregatedMetrics performAggregation(const std::vector<ExtendedRealtimeMetrics>& metrics);
    float calculatePercentile(const std::vector<float>& values, float percentile);

    std::map<std::string, std::function<float(const std::vector<float>&)>> custom_functions_;
};

// 实时仪表板数据生成器
class RealtimeDashboardGenerator {
private:
    std::shared_ptr<RealtimeMetricsCollector> collector_;
    std::shared_ptr<MetricsAggregator> aggregator_;

    // 仪表板配置
    struct DashboardConfig {
        std::vector<std::string> displayed_metrics;
        std::chrono::seconds refresh_interval;
        size_t max_data_points;
        bool enable_alerts;

        DashboardConfig();
    } config_;

public:
    RealtimeDashboardGenerator(
        std::shared_ptr<RealtimeMetricsCollector> collector,
        std::shared_ptr<MetricsAggregator> aggregator);

    // 生成仪表板数据
    std::string generateDashboardJSON();
    std::string generateMetricsChartData(const std::string& metric_name, size_t data_points = 100);
    std::string generatePerformanceSummary();
    std::string generateAlertsJSON();

    // 配置仪表板
    void setDisplayedMetrics(const std::vector<std::string>& metrics);
    void setRefreshInterval(std::chrono::seconds interval);
    void enableAlerts(bool enable);

    // 导出功能
    bool exportDashboardData(const std::string& filename, const std::string& format = "json");
};

} // namespace yolo_acceleration

#endif // REALTIME_METRICS_COLLECTOR_H
