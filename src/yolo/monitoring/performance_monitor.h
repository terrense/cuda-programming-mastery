#ifndef YOLO_PERFORMANCE_MONITOR_H
#define YOLO_PERFORMANCE_MONITOR_H

#include <cuda_runtime.h>
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

// 实时性能指标
struct RealtimeMetrics {
    float fps;                          // 帧率
    float latency_ms;                   // 延迟
    float throughput_images_per_sec;    // 吞吐量
    float gpu_utilization_percent;      // GPU利用率
    float memory_usage_mb;              // 内存使用量
    float memory_bandwidth_gb_s;        // 内存带宽
    float compute_utilization_percent;  // 计算利用率
    std::chrono::steady_clock::time_point timestamp;

    RealtimeMetrics();
    std::string toString() const;
};

// 性能瓶颈类型
enum class YOLOBottleneckType {
    PREPROCESSING,      // 预处理瓶颈
    BACKBONE,          // 骨干网络瓶颈
    NECK,              // 颈部网络瓶颈
    HEAD,              // 检测头瓶颈
    POSTPROCESSING,    // 后处理瓶颈
    MEMORY_TRANSFER,   // 内存传输瓶颈
    BATCH_SIZE,        // 批大小限制
    NONE               // 无瓶颈
};

// 瓶颈分析结果
struct YOLOBottleneckAnalysis {
    YOLOBottleneckType primaryBottleneck;
    YOLOBottleneckType secondaryBottleneck;
    float severity;                     // 严重程度 0-1
    std::string description;
    std::vector<std::string> suggestions;
    std::map<std::string, float> componentTimes; // 各组件耗时

    YOLOBottleneckAnalysis();
    std::string toString() const;
};

// 优化建议类型
enum class YOLOOptimizationType {
    TENSORRT_OPTIMIZATION,      // TensorRT优化
    BATCH_SIZE_TUNING,         // 批大小调优
    MEMORY_OPTIMIZATION,       // 内存优化
    OPERATOR_FUSION,           // 算子融合
    PRECISION_OPTIMIZATION,    // 精度优化
    STREAM_PARALLELISM,        // 流并行
    DYNAMIC_SHAPE_OPT,         // 动态形状优化
    CUSTOM_KERNEL_OPT          // 自定义核函数优化
};

// 优化建议
struct YOLOOptimizationSuggestion {
    YOLOOptimizationType type;
    std::string title;
    std::string description;
    std::string implementation;
    float expectedImprovement;  // 预期性能提升百分比
    int priority;              // 优先级 1-5
    float implementationCost;  // 实现成本 0-1

    YOLOOptimizationSuggestion();
    std::string toString() const;
};

// 实时性能监控器
class RealtimePerformanceMonitor {
private:
    std::atomic<bool> monitoring_;
    std::thread monitor_thread_;
    mutable std::mutex metrics_mutex_;
    std::queue<RealtimeMetrics> metrics_history_;
    size_t max_history_size_;

    // 监控回调函数
    std::vector<void(*)(const RealtimeMetrics&)> callbacks_;

    // GPU监控相关
    cudaEvent_t start_event_, stop_event_;
    std::chrono::steady_clock::time_point last_update_;

    // 统计信息
    float total_frames_;
    float total_time_ms_;

public:
    RealtimePerformanceMonitor(size_t max_history = 1000);
    ~RealtimePerformanceMonitor();

    // 启动/停止监控
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return monitoring_; }

    // 记录性能指标
    void recordInferenceStart();
    void recordInferenceEnd(int batch_size = 1);
    void recordCustomMetrics(const RealtimeMetrics& metrics);

    // 获取性能数据
    RealtimeMetrics getCurrentMetrics() const;
    std::vector<RealtimeMetrics> getRecentMetrics(size_t count = 100) const;
    RealtimeMetrics getAverageMetrics(size_t window_size = 100) const;

    // 注册回调函数
    void registerCallback(void(*callback)(const RealtimeMetrics&));

    // 性能报告
    std::string generateRealtimeReport() const;

    // 检测性能异常
    bool detectPerformanceAnomaly(float threshold = 0.2f) const;

private:
    void monitoringLoop();
    void updateGPUMetrics(RealtimeMetrics& metrics);
    void calculateDerivedMetrics(RealtimeMetrics& metrics);
};

// YOLO瓶颈分析器
class YOLOBottleneckAnalyzer {
public:
    // 分析YOLO推理瓶颈
    static YOLOBottleneckAnalysis analyzeYOLOBottlenecks(
        const std::vector<RealtimeMetrics>& metrics_history,
        const std::map<std::string, float>& component_times);

    // 分析特定组件瓶颈
    static YOLOBottleneckType analyzeComponentBottleneck(
        const std::map<std::string, float>& component_times);

    // 生成优化建议
    static std::vector<YOLOOptimizationSuggestion> generateYOLOOptimizations(
        const YOLOBottleneckAnalysis& analysis,
        const RealtimeMetrics& current_metrics);

    // 评估优化效果
    static float evaluateOptimizationImpact(
        const YOLOOptimizationSuggestion& suggestion,
        const RealtimeMetrics& baseline_metrics);

private:
    static float calculateComponentBottleneckSeverity(
        const std::string& component,
        const std::map<std::string, float>& times);
};

// A/B测试框架
class ABTestingFramework {
public:
    // A/B测试配置
    struct ABTestConfig {
        std::string test_name;
        std::string description;
        int sample_size;                    // 每组样本数量
        float significance_level;           // 显著性水平
        std::vector<std::string> metrics;   // 要比较的指标

        ABTestConfig();
    };

    // A/B测试结果
    struct ABTestResult {
        std::string test_name;
        bool is_significant;                // 是否显著
        float p_value;                     // p值
        float effect_size;                 // 效应大小
        std::map<std::string, float> baseline_metrics;  // 基线指标
        std::map<std::string, float> variant_metrics;   // 变体指标
        std::map<std::string, float> improvement_percent; // 改进百分比
        std::string conclusion;

        ABTestResult();
        std::string toString() const;
    };

private:
    std::map<std::string, std::vector<RealtimeMetrics>> test_groups_;
    std::map<std::string, ABTestConfig> test_configs_;
    std::mutex test_mutex_;

public:
    ABTestingFramework();

    // 创建A/B测试
    bool createABTest(const ABTestConfig& config);

    // 记录测试数据
    void recordTestData(const std::string& test_name,
                       const std::string& group_name,
                       const RealtimeMetrics& metrics);

    // 执行统计分析
    ABTestResult analyzeABTest(const std::string& test_name);

    // 批量测试多个优化方案
    std::vector<ABTestResult> runOptimizationComparison(
        const std::vector<YOLOOptimizationSuggestion>& optimizations,
        std::function<RealtimeMetrics(const YOLOOptimizationSuggestion&)> test_function);

    // 获取最佳优化方案
    YOLOOptimizationSuggestion selectBestOptimization(
        const std::vector<ABTestResult>& results);

    // 生成A/B测试报告
    std::string generateABTestReport(const std::vector<ABTestResult>& results);

private:
    // 统计分析辅助函数
    float calculateMean(const std::vector<float>& values);
    float calculateStdDev(const std::vector<float>& values, float mean);
    float calculateTStatistic(const std::vector<float>& group1,
                             const std::vector<float>& group2);
    float calculatePValue(float t_stat, int df);
    float calculateEffectSize(const std::vector<float>& group1,
                             const std::vector<float>& group2);
};

// 自动调优系统
class AutoTuningSystem {
public:
    // 调优配置
    struct TuningConfig {
        std::vector<int> batch_sizes;           // 批大小候选
        std::vector<std::string> precisions;   // 精度候选
        std::vector<int> stream_counts;        // 流数量候选
        int max_iterations;                    // 最大迭代次数
        float convergence_threshold;           // 收敛阈值
        std::string optimization_target;       // 优化目标 (fps, latency, throughput)

        TuningConfig();
    };

    // 调优结果
    struct TuningResult {
        int optimal_batch_size;
        std::string optimal_precision;
        int optimal_stream_count;
        RealtimeMetrics best_metrics;
        std::vector<ABTestingFramework::ABTestResult> test_results;
        std::string tuning_summary;

        TuningResult();
        std::string toString() const;
    };

private:
    std::unique_ptr<RealtimePerformanceMonitor> monitor_;
    std::unique_ptr<ABTestingFramework> ab_tester_;
    std::unique_ptr<YOLOBottleneckAnalyzer> analyzer_;

public:
    AutoTuningSystem();
    ~AutoTuningSystem();

    // 执行自动调优
    TuningResult autoTune(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 网格搜索调优
    TuningResult gridSearchTuning(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 贝叶斯优化调优
    TuningResult bayesianOptimization(
        const TuningConfig& config,
        std::function<RealtimeMetrics(int, const std::string&, int)> inference_function);

    // 生成调优报告
    std::string generateTuningReport(const TuningResult& result);

private:
    float evaluateConfiguration(int batch_size, const std::string& precision,
                               int stream_count, const std::string& target);
    std::vector<std::tuple<int, std::string, int>> generateConfigurationGrid(
        const TuningConfig& config);
};

// 性能监控仪表板
class PerformanceDashboard {
private:
    std::unique_ptr<RealtimePerformanceMonitor> monitor_;
    std::map<std::string, std::vector<float>> metric_history_;
    std::mutex dashboard_mutex_;

public:
    PerformanceDashboard();
    ~PerformanceDashboard();

    // 启动仪表板
    void start();
    void stop();

    // 生成实时仪表板数据
    std::string generateDashboardJSON();

    // 生成性能图表
    std::string generatePerformanceChart(const std::string& metric_name,
                                        int time_window_minutes = 10);

    // 生成瓶颈热力图
    std::string generateBottleneckHeatmap();

    // 导出性能数据
    bool exportPerformanceData(const std::string& filename,
                              const std::string& format = "csv");

    // 设置性能告警
    void setPerformanceAlert(const std::string& metric_name,
                           float threshold,
                           std::function<void(float)> callback);

private:
    void updateMetricHistory(const RealtimeMetrics& metrics);
    std::string formatMetricForChart(const std::vector<float>& values);
};

} // namespace yolo_acceleration

#endif // YOLO_PERFORMANCE_MONITOR_H
