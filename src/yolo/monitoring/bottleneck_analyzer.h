#ifndef BOTTLENECK_ANALYZER_H
#define BOTTLENECK_ANALYZER_H

#include "realtime_metrics_collector.h"
#include "performance_monitor.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace yolo_acceleration {

// 高级瓶颈分析器
class AdvancedBottleneckAnalyzer {
public:
    // 瓶颈分析配置
    struct AnalysisConfig {
        size_t analysis_window_size;        // 分析窗口大小
        float bottleneck_threshold;         // 瓶颈阈值
        bool enable_predictive_analysis;    // 启用预测分析
        bool enable_root_cause_analysis;    // 启用根因分析
        std::vector<std::string> focus_components; // 重点分析组件

        AnalysisConfig();
    };

    // 详细瓶颈分析结果
    struct DetailedBottleneckAnalysis {
        YOLOBottleneckAnalysis base_analysis;

        // 瓶颈严重程度分级
        std::map<std::string, float> component_bottleneck_scores;
        std::map<std::string, std::string> bottleneck_categories; // "critical", "major", "minor"

        // 根因分析
        struct RootCauseAnalysis {
            std::string primary_cause;
            std::vector<std::string> contributing_factors;
            std::map<std::string, float> factor_weights;
            std::string analysis_confidence; // "high", "medium", "low"

            std::string toString() const;
        } root_cause;

        // 预测分析
        struct PredictiveAnalysis {
            std::vector<std::string> potential_future_bottlenecks;
            std::map<std::string, float> bottleneck_probabilities;
            std::string trend_analysis;
            float prediction_confidence;

            std::string toString() const;
        } predictive;

        // 影响评估
        struct ImpactAssessment {
            float performance_impact_percent;
            float resource_waste_percent;
            std::map<std::string, float> metric_degradation;
            std::string business_impact_level; // "low", "medium", "high", "critical"

            std::string toString() const;
        } impact;

        std::string toComprehensiveReport() const;
    };

    // 智能优化建议生成器
    struct IntelligentOptimizationSuggestion {
        YOLOOptimizationSuggestion base_suggestion;

        // 实施计划
        struct ImplementationPlan {
            std::vector<std::string> implementation_steps;
            std::map<std::string, std::string> step_details;
            std::vector<std::string> prerequisites;
            std::vector<std::string> potential_risks;
            float estimated_implementation_time_hours;

            std::string toString() const;
        } implementation;

        // 效果预测
        struct EffectPrediction {
            std::map<std::string, float> predicted_improvements;
            std::map<std::string, float> confidence_intervals;
            std::vector<std::string> success_indicators;
            std::vector<std::string> failure_indicators;

            std::string toString() const;
        } prediction;

        // 成本效益分析
        struct CostBenefitAnalysis {
            float implementation_cost_score;
            float expected_benefit_score;
            float roi_estimate;
            std::string cost_breakdown;
            std::string benefit_breakdown;

            std::string toString() const;
        } cost_benefit;

        std::string toDetailedReport() const;
    };

private:
    AnalysisConfig config_;
    std::shared_ptr<RealtimeMetricsCollector> metrics_collector_;

    // 历史分析数据
    std::vector<DetailedBottleneckAnalysis> analysis_history_;
    std::map<std::string, std::vector<float>> component_performance_trends_;

    // 机器学习模型（简化实现）
    struct BottleneckPredictionModel {
        std::map<std::string, std::vector<float>> feature_weights;
        std::map<std::string, float> bias_terms;
        bool is_trained;

        BottleneckPredictionModel() : is_trained(false) {}
    } prediction_model_;

public:
    AdvancedBottleneckAnalyzer(std::shared_ptr<RealtimeMetricsCollector> collector,
                              const AnalysisConfig& config = AnalysisConfig());

    // 执行综合瓶颈分析
    DetailedBottleneckAnalysis performComprehensiveAnalysis();

    // 生成智能优化建议
    std::vector<IntelligentOptimizationSuggestion> generateIntelligentSuggestions(
        const DetailedBottleneckAnalysis& analysis);

    // 实时瓶颈监控
    void startRealtimeBottleneckMonitoring(
        std::function<void(const DetailedBottleneckAnalysis&)> callback);
    void stopRealtimeBottleneckMonitoring();

    // 趋势分析
    std::string analyzePerformanceTrends(size_t history_window = 1000);
    std::map<std::string, float> predictFutureBottlenecks(int prediction_horizon_minutes = 30);

    // 对比分析
    std::string compareWithBaseline(const DetailedBottleneckAnalysis& current,
                                   const DetailedBottleneckAnalysis& baseline);
    std::string generateImprovementReport(const std::vector<DetailedBottleneckAnalysis>& history);

    // 配置管理
    void updateAnalysisConfig(const AnalysisConfig& new_config);
    AnalysisConfig getAnalysisConfig() const { return config_; }

    // 模型训练和优化
    bool trainPredictionModel(const std::vector<ExtendedRealtimeMetrics>& training_data);
    float evaluatePredictionAccuracy(const std::vector<ExtendedRealtimeMetrics>& test_data);

private:
    // 核心分析方法
    DetailedBottleneckAnalysis::RootCauseAnalysis performRootCauseAnalysis(
        const std::vector<ExtendedRealtimeMetrics>& metrics_history);

    DetailedBottleneckAnalysis::PredictiveAnalysis performPredictiveAnalysis(
        const std::vector<ExtendedRealtimeMetrics>& metrics_history);

    DetailedBottleneckAnalysis::ImpactAssessment assessPerformanceImpact(
        const YOLOBottleneckAnalysis& base_analysis,
        const std::vector<ExtendedRealtimeMetrics>& metrics_history);

    // 瓶颈评分算法
    std::map<std::string, float> calculateBottleneckScores(
        const std::vector<ExtendedRealtimeMetrics>& metrics_history);

    std::string categorizeBottleneckSeverity(float score);

    // 优化建议生成
    IntelligentOptimizationSuggestion::ImplementationPlan generateImplementationPlan(
        const YOLOOptimizationSuggestion& base_suggestion);

    IntelligentOptimizationSuggestion::EffectPrediction predictOptimizationEffect(
        const YOLOOptimizationSuggestion& base_suggestion,
        const DetailedBottleneckAnalysis& analysis);

    IntelligentOptimizationSuggestion::CostBenefitAnalysis analyzeCostBenefit(
        const YOLOOptimizationSuggestion& base_suggestion);

    // 机器学习辅助方法
    std::vector<float> extractFeatures(const ExtendedRealtimeMetrics& metrics);
    float predictBottleneckProbability(const std::string& component,
                                      const std::vector<float>& features);

    // 统计分析方法
    float calculateTrendSlope(const std::vector<float>& values);
    float calculateCorrelation(const std::vector<float>& x, const std::vector<float>& y);
    std::vector<float> smoothData(const std::vector<float>& data, int window_size = 5);

    // 实时监控
    std::atomic<bool> realtime_monitoring_;
    std::thread monitoring_thread_;
    std::vector<std::function<void(const DetailedBottleneckAnalysis&)>> monitoring_callbacks_;

    void realtimeMonitoringLoop();
};

// 自动优化建议执行器
class OptimizationSuggestionExecutor {
public:
    // 执行结果
    struct ExecutionResult {
        std::string suggestion_id;
        bool execution_successful;
        std::string execution_log;
        ExtendedRealtimeMetrics before_metrics;
        ExtendedRealtimeMetrics after_metrics;
        std::map<std::string, float> improvement_achieved;
        std::vector<std::string> issues_encountered;
        float execution_time_minutes;

        std::string toString() const;
    };

    // 执行配置
    struct ExecutionConfig {
        bool enable_rollback;               // 启用回滚
        float success_threshold;            // 成功阈值
        int max_execution_time_minutes;     // 最大执行时间
        bool enable_safety_checks;          // 启用安全检查
        std::vector<std::string> excluded_optimizations; // 排除的优化类型

        ExecutionConfig();
    };

private:
    ExecutionConfig config_;
    std::shared_ptr<RealtimeMetricsCollector> metrics_collector_;
    std::shared_ptr<ABTestingFramework> ab_tester_;

    // 执行历史
    std::vector<ExecutionResult> execution_history_;
    std::map<std::string, std::string> rollback_states_;

public:
    OptimizationSuggestionExecutor(
        std::shared_ptr<RealtimeMetricsCollector> collector,
        std::shared_ptr<ABTestingFramework> ab_tester,
        const ExecutionConfig& config = ExecutionConfig());

    // 执行单个优化建议
    ExecutionResult executeSuggestion(
        const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion);

    // 批量执行优化建议
    std::vector<ExecutionResult> executeSuggestionBatch(
        const std::vector<AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion>& suggestions);

    // 自动优化流程
    std::vector<ExecutionResult> performAutomaticOptimization(
        const AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis& analysis);

    // 回滚功能
    bool rollbackOptimization(const std::string& suggestion_id);
    bool rollbackAllOptimizations();

    // 验证和测试
    bool validateOptimizationSafety(
        const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion);

    ExecutionResult performABTest(
        const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion,
        int test_duration_minutes = 10);

    // 执行历史管理
    std::vector<ExecutionResult> getExecutionHistory() const { return execution_history_; }
    std::string generateExecutionReport();

    // 配置管理
    void updateExecutionConfig(const ExecutionConfig& new_config);

private:
    // 具体优化实现
    bool executeTensorRTOptimization(const std::string& implementation_details);
    bool executeBatchSizeOptimization(int new_batch_size);
    bool executeMemoryOptimization(const std::string& optimization_type);
    bool executeStreamOptimization(int stream_count);
    bool executePrecisionOptimization(const std::string& precision);

    // 安全检查
    bool performSafetyChecks(
        const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion);

    // 状态管理
    std::string captureCurrentState();
    bool restoreState(const std::string& state_data);

    // 执行监控
    void monitorExecution(const std::string& suggestion_id,
                         std::function<void(const std::string&)> progress_callback);
};

// 性能基准测试和对比系统
class PerformanceBenchmarkSystem {
public:
    // 基准测试配置
    struct BenchmarkConfig {
        std::vector<std::string> test_scenarios;    // 测试场景
        std::vector<int> batch_sizes;              // 批大小
        std::vector<std::string> model_variants;   // 模型变体
        int iterations_per_test;                   // 每个测试的迭代次数
        bool enable_warmup;                        // 启用预热
        int warmup_iterations;                     // 预热迭代次数

        BenchmarkConfig();
    };

    // 基准测试结果
    struct BenchmarkResult {
        std::string test_name;
        std::string configuration;
        ExtendedRealtimeMetrics average_metrics;
        ExtendedRealtimeMetrics best_metrics;
        ExtendedRealtimeMetrics worst_metrics;
        std::vector<ExtendedRealtimeMetrics> all_measurements;
        float coefficient_of_variation;
        std::string stability_assessment;

        std::string toString() const;
    };

private:
    BenchmarkConfig config_;
    std::shared_ptr<RealtimeMetricsCollector> metrics_collector_;
    std::vector<BenchmarkResult> benchmark_history_;

public:
    PerformanceBenchmarkSystem(
        std::shared_ptr<RealtimeMetricsCollector> collector,
        const BenchmarkConfig& config = BenchmarkConfig());

    // 执行基准测试
    std::vector<BenchmarkResult> runComprehensiveBenchmark();
    BenchmarkResult runSingleBenchmark(const std::string& test_name,
                                      const std::string& configuration);

    // 对比分析
    std::string compareConfigurations(const std::vector<BenchmarkResult>& results);
    std::string generatePerformanceRanking(const std::vector<BenchmarkResult>& results);

    // 回归测试
    bool detectPerformanceRegression(const BenchmarkResult& current,
                                   const BenchmarkResult& baseline,
                                   float regression_threshold = 0.05f);

    std::string generateRegressionReport(const std::vector<BenchmarkResult>& current_results,
                                       const std::vector<BenchmarkResult>& baseline_results);

    // 基准数据管理
    bool saveBenchmarkResults(const std::string& filename);
    bool loadBenchmarkResults(const std::string& filename);

    // 统计分析
    std::map<std::string, float> calculatePerformanceStatistics(
        const std::vector<BenchmarkResult>& results);

    std::string generateStatisticalReport(const std::vector<BenchmarkResult>& results);

private:
    // 测试执行
    BenchmarkResult executeBenchmarkTest(const std::string& test_name,
                                       const std::string& configuration);

    void performWarmup(int iterations);

    // 统计计算
    float calculateCoefficientOfVariation(const std::vector<float>& values);
    std::string assessStability(float cv);

    // 数据处理
    ExtendedRealtimeMetrics calculateStatistics(
        const std::vector<ExtendedRealtimeMetrics>& measurements,
        const std::string& statistic_type);
};

} // namespace yolo_acceleration

#endif // BOTTLENECK_ANALYZER_H
