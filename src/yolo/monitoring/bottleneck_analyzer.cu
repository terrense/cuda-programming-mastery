#include "bottleneck_analyzer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>
#include <chrono>

namespace yolo_acceleration {

// 前向声明
class ABTestingFramework; // 简单的前向声明，避免循环依赖

// AnalysisConfig 实现
AdvancedBottleneckAnalyzer::AnalysisConfig::AnalysisConfig()
    : analysis_window_size(100), bottleneck_threshold(0.3f),
      enable_predictive_analysis(true), enable_root_cause_analysis(true),
      focus_components({"backbone", "neck", "head", "nms"}) {
}

// DetailedBottleneckAnalysis 嵌套结构的 toString 方法实现
std::string AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::RootCauseAnalysis::toString() const {
    std::stringstream ss;
    ss << "Root Cause Analysis:\n";
    ss << "  Primary Cause: " << primary_cause << "\n";
    ss << "  Contributing Factors:\n";
    for (const auto& factor : contributing_factors) {
        auto weight_it = factor_weights.find(factor);
        float weight = (weight_it != factor_weights.end()) ? weight_it->second : 0.0f;
        ss << "    - " << factor << " (weight: " << std::fixed << std::setprecision(2) << weight << ")\n";
    }
    ss << "  Confidence: " << analysis_confidence << "\n";
    return ss.str();
}

std::string AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::PredictiveAnalysis::toString() const {
    std::stringstream ss;
    ss << "Predictive Analysis:\n";
    ss << "  Potential Future Bottlenecks:\n";
    for (const auto& bottleneck : potential_future_bottlenecks) {
        auto prob_it = bottleneck_probabilities.find(bottleneck);
        float prob = (prob_it != bottleneck_probabilities.end()) ? prob_it->second : 0.0f;
        ss << "    - " << bottleneck << " (probability: " << std::fixed << std::setprecision(2) << prob << ")\n";
    }
    ss << "  Trend Analysis: " << trend_analysis << "\n";
    ss << "  Prediction Confidence: " << std::fixed << std::setprecision(2) << prediction_confidence << "\n";
    return ss.str();
}

std::string AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::ImpactAssessment::toString() const {
    std::stringstream ss;
    ss << "Impact Assessment:\n";
    ss << "  Performance Impact: " << std::fixed << std::setprecision(1) << performance_impact_percent << "%\n";
    ss << "  Resource Waste: " << std::fixed << std::setprecision(1) << resource_waste_percent << "%\n";
    ss << "  Business Impact Level: " << business_impact_level << "\n";
    ss << "  Metric Degradation:\n";
    for (const auto& [metric, degradation] : metric_degradation) {
        ss << "    - " << metric << ": " << std::fixed << std::setprecision(1) << degradation << "%\n";
    }
    return ss.str();
}

std::string AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::toComprehensiveReport() const {
    std::stringstream ss;
    ss << "=== COMPREHENSIVE BOTTLENECK ANALYSIS REPORT ===\n\n";

    // 基础分析
    ss << "Base Analysis:\n";
    ss << "  Primary Bottleneck: " << base_analysis.primary_bottleneck << "\n";
    ss << "  Severity: " << base_analysis.severity_level << "\n\n";

    // 组件瓶颈评分
    ss << "Component Bottleneck Scores:\n";
    for (const auto& [component, score] : component_bottleneck_scores) {
        auto category_it = bottleneck_categories.find(component);
        std::string category = (category_it != bottleneck_categories.end()) ? category_it->second : "unknown";
        ss << "  " << component << ": " << std::fixed << std::setprecision(3) << score
           << " (" << category << ")\n";
    }
    ss << "\n";

    // 根因分析
    ss << root_cause.toString() << "\n";

    // 预测分析
    ss << predictive.toString() << "\n";

    // 影响评估
    ss << impact.toString() << "\n";

    return ss.str();
}

// IntelligentOptimizationSuggestion 嵌套结构的 toString 方法实现
std::string AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::ImplementationPlan::toString() const {
    std::stringstream ss;
    ss << "Implementation Plan:\n";
    ss << "  Estimated Time: " << std::fixed << std::setprecision(1) << estimated_implementation_time_hours << " hours\n";
    ss << "  Steps:\n";
    for (size_t i = 0; i < implementation_steps.size(); ++i) {
        ss << "    " << (i + 1) << ". " << implementation_steps[i] << "\n";
        auto detail_it = step_details.find(implementation_steps[i]);
        if (detail_it != step_details.end()) {
            ss << "       Details: " << detail_it->second << "\n";
        }
    }
    if (!prerequisites.empty()) {
        ss << "  Prerequisites:\n";
        for (const auto& prereq : prerequisites) {
            ss << "    - " << prereq << "\n";
        }
    }
    if (!potential_risks.empty()) {
        ss << "  Potential Risks:\n";
        for (const auto& risk : potential_risks) {
            ss << "    - " << risk << "\n";
        }
    }
    return ss.str();
}

std::string AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::EffectPrediction::toString() const {
    std::stringstream ss;
    ss << "Effect Prediction:\n";
    ss << "  Predicted Improvements:\n";
    for (const auto& [metric, improvement] : predicted_improvements) {
        auto confidence_it = confidence_intervals.find(metric);
        float confidence = (confidence_it != confidence_intervals.end()) ? confidence_it->second : 0.0f;
        ss << "    " << metric << ": +" << std::fixed << std::setprecision(1) << improvement
           << "% (confidence: " << std::setprecision(2) << confidence << ")\n";
    }
    if (!success_indicators.empty()) {
        ss << "  Success Indicators:\n";
        for (const auto& indicator : success_indicators) {
            ss << "    - " << indicator << "\n";
        }
    }
    return ss.str();
}

std::string AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::CostBenefitAnalysis::toString() const {
    std::stringstream ss;
    ss << "Cost-Benefit Analysis:\n";
    ss << "  Implementation Cost Score: " << std::fixed << std::setprecision(2) << implementation_cost_score << "\n";
    ss << "  Expected Benefit Score: " << std::fixed << std::setprecision(2) << expected_benefit_score << "\n";
    ss << "  ROI Estimate: " << std::fixed << std::setprecision(1) << roi_estimate << "%\n";
    ss << "  Cost Breakdown: " << cost_breakdown << "\n";
    ss << "  Benefit Breakdown: " << benefit_breakdown << "\n";
    return ss.str();
}

std::string AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::toDetailedReport() const {
    std::stringstream ss;
    ss << "=== INTELLIGENT OPTIMIZATION SUGGESTION ===\n\n";

    // 基础建议
    ss << "Optimization Type: " << base_suggestion.optimization_type << "\n";
    ss << "Priority: " << base_suggestion.priority_level << "\n";
    ss << "Description: " << base_suggestion.description << "\n\n";

    // 实施计划
    ss << implementation.toString() << "\n";

    // 效果预测
    ss << prediction.toString() << "\n";

    // 成本效益分析
    ss << cost_benefit.toString() << "\n";

    return ss.str();
}

// AdvancedBottleneckAnalyzer 主类实现
AdvancedBottleneckAnalyzer::AdvancedBottleneckAnalyzer(
    std::shared_ptr<RealtimeMetricsCollector> collector,
    const AnalysisConfig& config)
    : config_(config), metrics_collector_(collector), realtime_monitoring_(false) {

    // 初始化预测模型
    prediction_model_.is_trained = false;

    // 预设一些基础权重（简化的机器学习模型）
    prediction_model_.feature_weights["backbone"] = {0.3f, 0.2f, 0.1f, 0.4f};
    prediction_model_.feature_weights["neck"] = {0.25f, 0.25f, 0.25f, 0.25f};
    prediction_model_.feature_weights["head"] = {0.2f, 0.3f, 0.3f, 0.2f};
    prediction_model_.feature_weights["nms"] = {0.1f, 0.1f, 0.4f, 0.4f};

    prediction_model_.bias_terms["backbone"] = 0.1f;
    prediction_model_.bias_terms["neck"] = 0.05f;
    prediction_model_.bias_terms["head"] = 0.08f;
    prediction_model_.bias_terms["nms"] = 0.15f;
}

AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis
AdvancedBottleneckAnalyzer::performComprehensiveAnalysis() {
    DetailedBottleneckAnalysis analysis;

    // 获取最近的性能指标历史
    auto metrics_history = metrics_collector_->getMetricsHistory(config_.analysis_window_size);

    if (metrics_history.empty()) {
        // 如果没有历史数据，返回默认分析
        analysis.base_analysis.primary_bottleneck = "insufficient_data";
        analysis.base_analysis.severity_level = "unknown";
        return analysis;
    }

    // 执行基础瓶颈分析
    analysis.base_analysis = metrics_collector_->analyzeBottlenecks();

    // 计算组件瓶颈评分
    analysis.component_bottleneck_scores = calculateBottleneckScores(metrics_history);

    // 分类瓶颈严重程度
    for (const auto& [component, score] : analysis.component_bottleneck_scores) {
        analysis.bottleneck_categories[component] = categorizeBottleneckSeverity(score);
    }

    // 执行根因分析
    if (config_.enable_root_cause_analysis) {
        analysis.root_cause = performRootCauseAnalysis(metrics_history);
    }

    // 执行预测分析
    if (config_.enable_predictive_analysis) {
        analysis.predictive = performPredictiveAnalysis(metrics_history);
    }

    // 评估性能影响
    analysis.impact = assessPerformanceImpact(analysis.base_analysis, metrics_history);

    // 保存到历史记录
    analysis_history_.push_back(analysis);
    if (analysis_history_.size() > config_.analysis_window_size) {
        analysis_history_.erase(analysis_history_.begin());
    }

    return analysis;
}

std::vector<AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion>
AdvancedBottleneckAnalyzer::generateIntelligentSuggestions(const DetailedBottleneckAnalysis& analysis) {
    std::vector<IntelligentOptimizationSuggestion> suggestions;

    // 基于基础分析生成建议
    auto base_suggestions = metrics_collector_->generateOptimizationSuggestions();

    for (const auto& base_suggestion : base_suggestions) {
        IntelligentOptimizationSuggestion intelligent_suggestion;
        intelligent_suggestion.base_suggestion = base_suggestion;

        // 生成实施计划
        intelligent_suggestion.implementation = generateImplementationPlan(base_suggestion);

        // 预测优化效果
        intelligent_suggestion.prediction = predictOptimizationEffect(base_suggestion, analysis);

        // 分析成本效益
        intelligent_suggestion.cost_benefit = analyzeCostBenefit(base_suggestion);

        suggestions.push_back(intelligent_suggestion);
    }

    // 根据ROI排序
    std::sort(suggestions.begin(), suggestions.end(),
        [](const IntelligentOptimizationSuggestion& a, const IntelligentOptimizationSuggestion& b) {
            return a.cost_benefit.roi_estimate > b.cost_benefit.roi_estimate;
        });

    return suggestions;
}

void AdvancedBottleneckAnalyzer::startRealtimeBottleneckMonitoring(
    std::function<void(const DetailedBottleneckAnalysis&)> callback) {

    if (realtime_monitoring_.load()) {
        return; // 已经在监控中
    }

    monitoring_callbacks_.push_back(callback);
    realtime_monitoring_.store(true);

    monitoring_thread_ = std::thread(&AdvancedBottleneckAnalyzer::realtimeMonitoringLoop, this);
}

void AdvancedBottleneckAnalyzer::stopRealtimeBottleneckMonitoring() {
    realtime_monitoring_.store(false);
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    monitoring_callbacks_.clear();
}

std::string AdvancedBottleneckAnalyzer::analyzePerformanceTrends(size_t history_window) {
    std::stringstream ss;
    ss << "=== PERFORMANCE TREND ANALYSIS ===\n\n";

    if (analysis_history_.size() < 2) {
        ss << "Insufficient historical data for trend analysis.\n";
        return ss.str();
    }

    size_t window_size = std::min(history_window, analysis_history_.size());
    auto recent_history = std::vector<DetailedBottleneckAnalysis>(
        analysis_history_.end() - window_size, analysis_history_.end());

    // 分析各组件的趋势
    for (const auto& component : config_.focus_components) {
        std::vector<float> scores;
        for (const auto& analysis : recent_history) {
            auto it = analysis.component_bottleneck_scores.find(component);
            if (it != analysis.component_bottleneck_scores.end()) {
                scores.push_back(it->second);
            }
        }

        if (!scores.empty()) {
            float trend_slope = calculateTrendSlope(scores);
            std::string trend_direction = (trend_slope > 0.01f) ? "worsening" :
                                        (trend_slope < -0.01f) ? "improving" : "stable";

            ss << component << " Component:\n";
            ss << "  Trend: " << trend_direction << " (slope: " << std::fixed << std::setprecision(4) << trend_slope << ")\n";
            ss << "  Current Score: " << std::fixed << std::setprecision(3) << scores.back() << "\n";
            ss << "  Average Score: " << std::fixed << std::setprecision(3)
               << (std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size()) << "\n\n";
        }
    }

    return ss.str();
}

std::map<std::string, float> AdvancedBottleneckAnalyzer::predictFutureBottlenecks(int prediction_horizon_minutes) {
    std::map<std::string, float> predictions;

    if (!prediction_model_.is_trained) {
        // 使用简化的预测逻辑
        for (const auto& component : config_.focus_components) {
            predictions[component] = 0.1f; // 默认低概率
        }
        return predictions;
    }

    // 获取最新指标作为特征
    auto latest_metrics = metrics_collector_->getCurrentMetrics();
    auto features = extractFeatures(latest_metrics);

    // 为每个组件预测瓶颈概率
    for (const auto& component : config_.focus_components) {
        float probability = predictBottleneckProbability(component, features);

        // 根据预测时间范围调整概率
        float time_factor = std::min(1.0f, prediction_horizon_minutes / 60.0f);
        predictions[component] = probability * time_factor;
    }

    return predictions;
}

// 私有方法实现
AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::RootCauseAnalysis
AdvancedBottleneckAnalyzer::performRootCauseAnalysis(const std::vector<ExtendedRealtimeMetrics>& metrics_history) {
    DetailedBottleneckAnalysis::RootCauseAnalysis root_cause;

    if (metrics_history.empty()) {
        root_cause.primary_cause = "insufficient_data";
        root_cause.analysis_confidence = "low";
        return root_cause;
    }

    // 分析最近的指标趋势
    std::map<std::string, std::vector<float>> component_trends;

    for (const auto& metrics : metrics_history) {
        component_trends["gpu_utilization"].push_back(metrics.gpu_utilization);
        component_trends["memory_utilization"].push_back(metrics.memory_utilization);
        component_trends["inference_time"].push_back(metrics.inference_time_ms);
        component_trends["throughput"].push_back(metrics.throughput_fps);
    }

    // 计算各指标的变化趋势和相关性
    std::map<std::string, float> trend_slopes;
    for (const auto& [metric, values] : component_trends) {
        trend_slopes[metric] = calculateTrendSlope(values);
    }

    // 确定主要原因
    float max_negative_trend = -0.01f;
    std::string primary_cause_metric;

    for (const auto& [metric, slope] : trend_slopes) {
        if (slope < max_negative_trend) {
            max_negative_trend = slope;
            primary_cause_metric = metric;
        }
    }

    if (!primary_cause_metric.empty()) {
        if (primary_cause_metric == "gpu_utilization") {
            root_cause.primary_cause = "GPU underutilization due to CPU bottleneck or memory bandwidth limitation";
        } else if (primary_cause_metric == "memory_utilization") {
            root_cause.primary_cause = "Memory bandwidth saturation or inefficient memory access patterns";
        } else if (primary_cause_metric == "inference_time") {
            root_cause.primary_cause = "Computational bottleneck in model execution";
        } else {
            root_cause.primary_cause = "Throughput degradation due to system resource contention";
        }
    } else {
        root_cause.primary_cause = "No clear performance degradation detected";
    }

    // 识别贡献因素
    for (const auto& [metric, slope] : trend_slopes) {
        if (std::abs(slope) > 0.005f && metric != primary_cause_metric) {
            root_cause.contributing_factors.push_back(metric + "_trend");
            root_cause.factor_weights[metric + "_trend"] = std::abs(slope);
        }
    }

    // 设置置信度
    float confidence_score = std::min(1.0f, std::abs(max_negative_trend) * 100.0f);
    if (confidence_score > 0.7f) {
        root_cause.analysis_confidence = "high";
    } else if (confidence_score > 0.3f) {
        root_cause.analysis_confidence = "medium";
    } else {
        root_cause.analysis_confidence = "low";
    }

    return root_cause;
}

AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::PredictiveAnalysis
AdvancedBottleneckAnalyzer::performPredictiveAnalysis(const std::vector<ExtendedRealtimeMetrics>& metrics_history) {
    DetailedBottleneckAnalysis::PredictiveAnalysis predictive;

    if (metrics_history.size() < 10) {
        predictive.trend_analysis = "Insufficient data for reliable prediction";
        predictive.prediction_confidence = 0.1f;
        return predictive;
    }

    // 预测未来瓶颈
    auto future_bottlenecks = predictFutureBottlenecks(30); // 30分钟预测

    for (const auto& [component, probability] : future_bottlenecks) {
        if (probability > 0.3f) {
            predictive.potential_future_bottlenecks.push_back(component);
            predictive.bottleneck_probabilities[component] = probability;
        }
    }

    // 趋势分析
    std::vector<float> recent_throughput;
    for (size_t i = std::max(0, (int)metrics_history.size() - 20); i < metrics_history.size(); ++i) {
        recent_throughput.push_back(metrics_history[i].throughput_fps);
    }

    float throughput_trend = calculateTrendSlope(recent_throughput);
    if (throughput_trend > 0.01f) {
        predictive.trend_analysis = "Performance is improving";
    } else if (throughput_trend < -0.01f) {
        predictive.trend_analysis = "Performance is degrading";
    } else {
        predictive.trend_analysis = "Performance is stable";
    }

    // 计算预测置信度
    float data_quality = std::min(1.0f, metrics_history.size() / 100.0f);
    float trend_consistency = 1.0f - std::abs(throughput_trend) * 10.0f;
    predictive.prediction_confidence = (data_quality + trend_consistency) / 2.0f;

    return predictive;
}

AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis::ImpactAssessment
AdvancedBottleneckAnalyzer::assessPerformanceImpact(
    const YOLOBottleneckAnalysis& base_analysis,
    const std::vector<ExtendedRealtimeMetrics>& metrics_history) {

    DetailedBottleneckAnalysis::ImpactAssessment impact;

    if (metrics_history.empty()) {
        impact.performance_impact_percent = 0.0f;
        impact.resource_waste_percent = 0.0f;
        impact.business_impact_level = "unknown";
        return impact;
    }

    // 计算性能影响
    auto latest_metrics = metrics_history.back();
    float baseline_throughput = 60.0f; // 假设基准吞吐量为60 FPS

    impact.performance_impact_percent =
        std::max(0.0f, (baseline_throughput - latest_metrics.throughput_fps) / baseline_throughput * 100.0f);

    // 计算资源浪费
    float ideal_gpu_utilization = 95.0f;
    impact.resource_waste_percent =
        std::max(0.0f, (ideal_gpu_utilization - latest_metrics.gpu_utilization) / ideal_gpu_utilization * 100.0f);

    // 指标退化分析
    impact.metric_degradation["throughput"] = impact.performance_impact_percent;
    impact.metric_degradation["gpu_efficiency"] = impact.resource_waste_percent;
    impact.metric_degradation["memory_efficiency"] =
        std::max(0.0f, (90.0f - latest_metrics.memory_utilization) / 90.0f * 100.0f);

    // 业务影响级别
    if (impact.performance_impact_percent > 30.0f) {
        impact.business_impact_level = "critical";
    } else if (impact.performance_impact_percent > 15.0f) {
        impact.business_impact_level = "high";
    } else if (impact.performance_impact_percent > 5.0f) {
        impact.business_impact_level = "medium";
    } else {
        impact.business_impact_level = "low";
    }

    return impact;
}

std::map<std::string, float> AdvancedBottleneckAnalyzer::calculateBottleneckScores(
    const std::vector<ExtendedRealtimeMetrics>& metrics_history) {

    std::map<std::string, float> scores;

    if (metrics_history.empty()) {
        for (const auto& component : config_.focus_components) {
            scores[component] = 0.0f;
        }
        return scores;
    }

    // 计算各组件的瓶颈评分
    auto latest_metrics = metrics_history.back();

    // GPU利用率相关评分
    float gpu_score = 1.0f - (latest_metrics.gpu_utilization / 100.0f);
    scores["backbone"] = gpu_score * 0.6f + (latest_metrics.inference_time_ms / 50.0f) * 0.4f;

    // 内存利用率相关评分
    float memory_score = latest_metrics.memory_utilization / 100.0f;
    scores["neck"] = memory_score * 0.7f + gpu_score * 0.3f;

    // 吞吐量相关评分
    float throughput_score = std::max(0.0f, (60.0f - latest_metrics.throughput_fps) / 60.0f);
    scores["head"] = throughput_score * 0.8f + gpu_score * 0.2f;

    // NMS相关评分（假设与后处理时间相关）
    scores["nms"] = std::min(1.0f, latest_metrics.inference_time_ms / 100.0f) * 0.5f + throughput_score * 0.5f;

    // 限制评分范围
    for (auto& [component, score] : scores) {
        score = std::max(0.0f, std::min(1.0f, score));
    }

    return scores;
}

std::string AdvancedBottleneckAnalyzer::categorizeBottleneckSeverity(float score) {
    if (score > 0.7f) {
        return "critical";
    } else if (score > 0.4f) {
        return "major";
    } else if (score > 0.1f) {
        return "minor";
    } else {
        return "none";
    }
}

// 优化建议生成方法
AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::ImplementationPlan
AdvancedBottleneckAnalyzer::generateImplementationPlan(const YOLOOptimizationSuggestion& base_suggestion) {
    IntelligentOptimizationSuggestion::ImplementationPlan plan;

    if (base_suggestion.optimization_type == "tensorrt_optimization") {
        plan.implementation_steps = {
            "Export model to ONNX format",
            "Create TensorRT engine with FP16 precision",
            "Implement custom plugins for unsupported layers",
            "Optimize batch size and workspace size",
            "Validate accuracy and performance"
        };
        plan.estimated_implementation_time_hours = 4.0f;
        plan.prerequisites = {"TensorRT SDK installed", "ONNX model available"};
        plan.potential_risks = {"Accuracy degradation with FP16", "Memory constraints with large batch sizes"};
    } else if (base_suggestion.optimization_type == "memory_optimization") {
        plan.implementation_steps = {
            "Analyze current memory usage patterns",
            "Implement memory pooling",
            "Optimize tensor allocation and deallocation",
            "Enable memory mapping for large tensors"
        };
        plan.estimated_implementation_time_hours = 2.5f;
        plan.prerequisites = {"Memory profiling tools"};
        plan.potential_risks = {"Memory fragmentation", "Increased complexity"};
    } else {
        // 默认实施计划
        plan.implementation_steps = {"Analyze current implementation", "Apply optimization", "Test and validate"};
        plan.estimated_implementation_time_hours = 1.0f;
    }

    // 添加步骤详情
    for (const auto& step : plan.implementation_steps) {
        plan.step_details[step] = "Detailed implementation guidance for: " + step;
    }

    return plan;
}

AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::EffectPrediction
AdvancedBottleneckAnalyzer::predictOptimizationEffect(
    const YOLOOptimizationSuggestion& base_suggestion,
    const DetailedBottleneckAnalysis& analysis) {

    IntelligentOptimizationSuggestion::EffectPrediction prediction;

    // 基于优化类型预测效果
    if (base_suggestion.optimization_type == "tensorrt_optimization") {
        prediction.predicted_improvements["inference_time"] = 40.0f;
        prediction.predicted_improvements["throughput"] = 35.0f;
        prediction.predicted_improvements["memory_usage"] = -10.0f; // 可能增加内存使用

        prediction.confidence_intervals["inference_time"] = 0.8f;
        prediction.confidence_intervals["throughput"] = 0.75f;
        prediction.confidence_intervals["memory_usage"] = 0.6f;

        prediction.success_indicators = {
            "Inference time reduced by >30%",
            "Throughput increased by >25%",
            "Accuracy maintained within 1%"
        };
    } else if (base_suggestion.optimization_type == "memory_optimization") {
        prediction.predicted_improvements["memory_usage"] = 25.0f;
        prediction.predicted_improvements["memory_bandwidth"] = 15.0f;

        prediction.confidence_intervals["memory_usage"] = 0.9f;
        prediction.confidence_intervals["memory_bandwidth"] = 0.7f;

        prediction.success_indicators = {
            "Memory usage reduced by >20%",
            "Memory allocation time improved"
        };
    }

    return prediction;
}

AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion::CostBenefitAnalysis
AdvancedBottleneckAnalyzer::analyzeCostBenefit(const YOLOOptimizationSuggestion& base_suggestion) {
    IntelligentOptimizationSuggestion::CostBenefitAnalysis analysis;

    // 基于优化类型分析成本效益
    if (base_suggestion.optimization_type == "tensorrt_optimization") {
        analysis.implementation_cost_score = 7.0f; // 较高实施成本
        analysis.expected_benefit_score = 9.0f;    // 高预期收益
        analysis.roi_estimate = 180.0f;            // 180% ROI

        analysis.cost_breakdown = "Development time: 4 hours, Testing: 2 hours, Integration: 1 hour";
        analysis.benefit_breakdown = "40% inference speedup, 35% throughput improvement, reduced hardware requirements";
    } else if (base_suggestion.optimization_type == "memory_optimization") {
        analysis.implementation_cost_score = 4.0f; // 中等实施成本
        analysis.expected_benefit_score = 6.0f;    // 中等预期收益
        analysis.roi_estimate = 120.0f;            // 120% ROI

        analysis.cost_breakdown = "Development time: 2.5 hours, Testing: 1 hour";
        analysis.benefit_breakdown = "25% memory reduction, improved stability, better scalability";
    } else {
        // 默认分析
        analysis.implementation_cost_score = 3.0f;
        analysis.expected_benefit_score = 5.0f;
        analysis.roi_estimate = 100.0f;

        analysis.cost_breakdown = "Minimal implementation effort";
        analysis.benefit_breakdown = "General performance improvement";
    }

    return analysis;
}

// 机器学习辅助方法
std::vector<float> AdvancedBottleneckAnalyzer::extractFeatures(const ExtendedRealtimeMetrics& metrics) {
    return {
        metrics.gpu_utilization / 100.0f,
        metrics.memory_utilization / 100.0f,
        metrics.inference_time_ms / 100.0f,
        metrics.throughput_fps / 60.0f
    };
}

float AdvancedBottleneckAnalyzer::predictBottleneckProbability(
    const std::string& component, const std::vector<float>& features) {

    auto weight_it = prediction_model_.feature_weights.find(component);
    auto bias_it = prediction_model_.bias_terms.find(component);

    if (weight_it == prediction_model_.feature_weights.end() ||
        bias_it == prediction_model_.bias_terms.end()) {
        return 0.1f; // 默认低概率
    }

    const auto& weights = weight_it->second;
    float bias = bias_it->second;

    float score = bias;
    for (size_t i = 0; i < std::min(features.size(), weights.size()); ++i) {
        score += features[i] * weights[i];
    }

    // 使用sigmoid函数转换为概率
    return 1.0f / (1.0f + std::exp(-score));
}

// 统计分析方法
float AdvancedBottleneckAnalyzer::calculateTrendSlope(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0f;

    float n = static_cast<float>(values.size());
    float sum_x = n * (n - 1) / 2.0f;
    float sum_y = std::accumulate(values.begin(), values.end(), 0.0f);
    float sum_xy = 0.0f;
    float sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0f;

    for (size_t i = 0; i < values.size(); ++i) {
        sum_xy += i * values[i];
    }

    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

float AdvancedBottleneckAnalyzer::calculateCorrelation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0f;

    float n = static_cast<float>(x.size());
    float sum_x = std::accumulate(x.begin(), x.end(), 0.0f);
    float sum_y = std::accumulate(y.begin(), y.end(), 0.0f);
    float sum_xy = 0.0f, sum_x2 = 0.0f, sum_y2 = 0.0f;

    for (size_t i = 0; i < x.size(); ++i) {
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    float numerator = n * sum_xy - sum_x * sum_y;
    float denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    return (denominator != 0.0f) ? numerator / denominator : 0.0f;
}

std::vector<float> AdvancedBottleneckAnalyzer::smoothData(const std::vector<float>& data, int window_size) {
    std::vector<float> smoothed;
    smoothed.reserve(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - window_size / 2);
        int end = std::min(static_cast<int>(data.size()), static_cast<int>(i) + window_size / 2 + 1);

        float sum = 0.0f;
        int count = 0;
        for (int j = start; j < end; ++j) {
            sum += data[j];
            count++;
        }

        smoothed.push_back(sum / count);
    }

    return smoothed;
}

// 实时监控循环
void AdvancedBottleneckAnalyzer::realtimeMonitoringLoop() {
    while (realtime_monitoring_.load()) {
        try {
            auto analysis = performComprehensiveAnalysis();

            // 通知所有回调函数
            for (const auto& callback : monitoring_callbacks_) {
                callback(analysis);
            }

            // 等待一段时间再进行下次分析
            std::this_thread::sleep_for(std::chrono::seconds(5));
        } catch (const std::exception& e) {
            std::cerr << "Error in realtime monitoring: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void AdvancedBottleneckAnalyzer::updateAnalysisConfig(const AnalysisConfig& new_config) {
    config_ = new_config;
}

bool AdvancedBottleneckAnalyzer::trainPredictionModel(const std::vector<ExtendedRealtimeMetrics>& training_data) {
    if (training_data.size() < 10) {
        return false; // 训练数据不足
    }

    // 简化的模型训练逻辑
    // 在实际实现中，这里会使用更复杂的机器学习算法

    std::map<std::string, std::vector<std::vector<float>>> component_features;
    std::map<std::string, std::vector<float>> component_labels;

    // 准备训练数据
    for (size_t i = 0; i < training_data.size() - 1; ++i) {
        auto features = extractFeatures(training_data[i]);

        // 计算下一时刻的瓶颈标签（简化）
        for (const auto& component : config_.focus_components) {
            component_features[component].push_back(features);

            // 简化的标签生成逻辑
            float label = (training_data[i + 1].throughput_fps < training_data[i].throughput_fps) ? 1.0f : 0.0f;
            component_labels[component].push_back(label);
        }
    }

    // 更新模型权重（简化的梯度下降）
    float learning_rate = 0.01f;
    for (const auto& component : config_.focus_components) {
        if (component_features[component].empty()) continue;

        auto& weights = prediction_model_.feature_weights[component];
        auto& bias = prediction_model_.bias_terms[component];

        // 简化的权重更新
        for (size_t i = 0; i < component_features[component].size(); ++i) {
            const auto& features = component_features[component][i];
            float label = component_labels[component][i];

            float prediction = predictBottleneckProbability(component, features);
            float error = label - prediction;

            // 更新权重
            for (size_t j = 0; j < std::min(features.size(), weights.size()); ++j) {
                weights[j] += learning_rate * error * features[j];
            }
            bias += learning_rate * error;
        }
    }

    prediction_model_.is_trained = true;
    return true;
}

float AdvancedBottleneckAnalyzer::evaluatePredictionAccuracy(const std::vector<ExtendedRealtimeMetrics>& test_data) {
    if (!prediction_model_.is_trained || test_data.size() < 2) {
        return 0.0f;
    }

    int correct_predictions = 0;
    int total_predictions = 0;

    for (size_t i = 0; i < test_data.size() - 1; ++i) {
        auto features = extractFeatures(test_data[i]);

        for (const auto& component : config_.focus_components) {
            float predicted_prob = predictBottleneckProbability(component, features);
            bool predicted_bottleneck = predicted_prob > 0.5f;

            // 简化的真实标签
            bool actual_bottleneck = (test_data[i + 1].throughput_fps < test_data[i].throughput_fps);

            if (predicted_bottleneck == actual_bottleneck) {
                correct_predictions++;
            }
            total_predictions++;
        }
    }

    return (total_predictions > 0) ? static_cast<float>(correct_predictions) / total_predictions : 0.0f;
}

} // namespace yolo_acceleration//
 OptimizationSuggestionExecutor 实现
OptimizationSuggestionExecutor::ExecutionConfig::ExecutionConfig()
    : enable_rollback(true), success_threshold(0.05f), max_execution_time_minutes(30),
      enable_safety_checks(true), excluded_optimizations({}) {
}

std::string OptimizationSuggestionExecutor::ExecutionResult::toString() const {
    std::stringstream ss;
    ss << "=== OPTIMIZATION EXECUTION RESULT ===\n";
    ss << "Suggestion ID: " << suggestion_id << "\n";
    ss << "Execution Status: " << (execution_successful ? "SUCCESS" : "FAILED") << "\n";
    ss << "Execution Time: " << std::fixed << std::setprecision(1) << execution_time_minutes << " minutes\n";

    if (!improvement_achieved.empty()) {
        ss << "Improvements Achieved:\n";
        for (const auto& [metric, improvement] : improvement_achieved) {
            ss << "  " << metric << ": " << std::fixed << std::setprecision(1) << improvement << "%\n";
        }
    }

    if (!issues_encountered.empty()) {
        ss << "Issues Encountered:\n";
        for (const auto& issue : issues_encountered) {
            ss << "  - " << issue << "\n";
        }
    }

    ss << "Execution Log:\n" << execution_log << "\n";
    return ss.str();
}

OptimizationSuggestionExecutor::OptimizationSuggestionExecutor(
    std::shared_ptr<RealtimeMetricsCollector> collector,
    std::shared_ptr<ABTestingFramework> ab_tester,
    const ExecutionConfig& config)
    : config_(config), metrics_collector_(collector), ab_tester_(ab_tester) {
}

OptimizationSuggestionExecutor::ExecutionResult
OptimizationSuggestionExecutor::executeSuggestion(
    const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion) {

    ExecutionResult result;
    result.suggestion_id = suggestion.base_suggestion.optimization_type + "_" +
                          std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count());

    auto start_time = std::chrono::steady_clock::now();

    try {
        // 记录执行前的指标
        result.before_metrics = metrics_collector_->getCurrentMetrics();

        // 执行安全检查
        if (config_.enable_safety_checks && !performSafetyChecks(suggestion)) {
            result.execution_successful = false;
            result.execution_log = "Safety checks failed";
            result.issues_encountered.push_back("Safety validation failed");
            return result;
        }

        // 保存当前状态用于回滚
        if (config_.enable_rollback) {
            rollback_states_[result.suggestion_id] = captureCurrentState();
        }

        // 执行具体的优化
        bool execution_success = false;
        std::stringstream log_stream;

        const auto& opt_type = suggestion.base_suggestion.optimization_type;

        if (opt_type == "tensorrt_optimization") {
            log_stream << "Executing TensorRT optimization...\n";
            execution_success = executeTensorRTOptimization(suggestion.base_suggestion.implementation_details);
        } else if (opt_type == "batch_size_optimization") {
            log_stream << "Executing batch size optimization...\n";
            // 从实施详情中解析批大小
            int new_batch_size = 8; // 默认值，实际应从suggestion中解析
            execution_success = executeBatchSizeOptimization(new_batch_size);
        } else if (opt_type == "memory_optimization") {
            log_stream << "Executing memory optimization...\n";
            execution_success = executeMemoryOptimization(suggestion.base_suggestion.implementation_details);
        } else if (opt_type == "stream_optimization") {
            log_stream << "Executing stream optimization...\n";
            execution_success = executeStreamOptimization(4); // 默认4个流
        } else if (opt_type == "precision_optimization") {
            log_stream << "Executing precision optimization...\n";
            execution_success = executePrecisionOptimization("fp16");
        } else {
            log_stream << "Unknown optimization type: " << opt_type << "\n";
            execution_success = false;
        }

        result.execution_successful = execution_success;
        result.execution_log = log_stream.str();

        if (execution_success) {
            // 等待一段时间让优化生效
            std::this_thread::sleep_for(std::chrono::seconds(2));

            // 记录执行后的指标
            result.after_metrics = metrics_collector_->getCurrentMetrics();

            // 计算改进效果
            float throughput_improvement =
                ((result.after_metrics.throughput_fps - result.before_metrics.throughput_fps) /
                 result.before_metrics.throughput_fps) * 100.0f;

            float latency_improvement =
                ((result.before_metrics.inference_time_ms - result.after_metrics.inference_time_ms) /
                 result.before_metrics.inference_time_ms) * 100.0f;

            result.improvement_achieved["throughput"] = throughput_improvement;
            result.improvement_achieved["latency"] = latency_improvement;

            // 检查是否达到成功阈值
            if (throughput_improvement < config_.success_threshold * 100.0f) {
                result.issues_encountered.push_back("Improvement below success threshold");
            }
        } else {
            result.issues_encountered.push_back("Optimization execution failed");
        }

    } catch (const std::exception& e) {
        result.execution_successful = false;
        result.execution_log += "Exception: " + std::string(e.what());
        result.issues_encountered.push_back("Exception during execution");
    }

    auto end_time = std::chrono::steady_clock::now();
    result.execution_time_minutes =
        std::chrono::duration<float, std::ratio<60>>(end_time - start_time).count();

    // 保存执行历史
    execution_history_.push_back(result);

    return result;
}

std::vector<OptimizationSuggestionExecutor::ExecutionResult>
OptimizationSuggestionExecutor::executeSuggestionBatch(
    const std::vector<AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion>& suggestions) {

    std::vector<ExecutionResult> results;
    results.reserve(suggestions.size());

    for (const auto& suggestion : suggestions) {
        // 检查是否在排除列表中
        bool excluded = std::find(config_.excluded_optimizations.begin(),
                                 config_.excluded_optimizations.end(),
                                 suggestion.base_suggestion.optimization_type) !=
                       config_.excluded_optimizations.end();

        if (excluded) {
            ExecutionResult skipped_result;
            skipped_result.suggestion_id = suggestion.base_suggestion.optimization_type + "_skipped";
            skipped_result.execution_successful = false;
            skipped_result.execution_log = "Optimization type excluded from execution";
            results.push_back(skipped_result);
            continue;
        }

        auto result = executeSuggestion(suggestion);
        results.push_back(result);

        // 如果执行失败且启用了回滚，则回滚
        if (!result.execution_successful && config_.enable_rollback) {
            rollbackOptimization(result.suggestion_id);
        }
    }

    return results;
}

std::vector<OptimizationSuggestionExecutor::ExecutionResult>
OptimizationSuggestionExecutor::performAutomaticOptimization(
    const AdvancedBottleneckAnalyzer::DetailedBottleneckAnalysis& analysis) {

    std::vector<ExecutionResult> results;

    // 基于分析结果自动选择优化策略
    std::vector<AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion> auto_suggestions;

    // 根据瓶颈类型自动生成建议
    for (const auto& [component, score] : analysis.component_bottleneck_scores) {
        if (score > config_.success_threshold) {
            AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion suggestion;

            if (component == "backbone" && score > 0.5f) {
                suggestion.base_suggestion.optimization_type = "tensorrt_optimization";
                suggestion.base_suggestion.priority_level = "high";
                suggestion.base_suggestion.description = "Optimize backbone with TensorRT";
            } else if (component == "memory" && score > 0.4f) {
                suggestion.base_suggestion.optimization_type = "memory_optimization";
                suggestion.base_suggestion.priority_level = "medium";
                suggestion.base_suggestion.description = "Optimize memory usage patterns";
            }

            if (!suggestion.base_suggestion.optimization_type.empty()) {
                auto_suggestions.push_back(suggestion);
            }
        }
    }

    // 执行自动生成的建议
    if (!auto_suggestions.empty()) {
        results = executeSuggestionBatch(auto_suggestions);
    }

    return results;
}

bool OptimizationSuggestionExecutor::rollbackOptimization(const std::string& suggestion_id) {
    auto state_it = rollback_states_.find(suggestion_id);
    if (state_it == rollback_states_.end()) {
        return false;
    }

    try {
        bool success = restoreState(state_it->second);
        if (success) {
            rollback_states_.erase(state_it);
        }
        return success;
    } catch (const std::exception& e) {
        std::cerr << "Rollback failed: " << e.what() << std::endl;
        return false;
    }
}

bool OptimizationSuggestionExecutor::rollbackAllOptimizations() {
    bool all_success = true;

    for (const auto& [suggestion_id, state] : rollback_states_) {
        if (!restoreState(state)) {
            all_success = false;
        }
    }

    rollback_states_.clear();
    return all_success;
}

bool OptimizationSuggestionExecutor::validateOptimizationSafety(
    const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion) {

    // 检查优化类型是否安全
    const auto& opt_type = suggestion.base_suggestion.optimization_type;

    if (opt_type == "tensorrt_optimization") {
        // 检查TensorRT是否可用
        // 这里应该检查实际的TensorRT环境
        return true; // 简化实现
    } else if (opt_type == "memory_optimization") {
        // 检查内存使用情况
        auto current_metrics = metrics_collector_->getCurrentMetrics();
        return current_metrics.memory_utilization < 90.0f; // 内存使用率不超过90%
    }

    return true; // 默认认为安全
}

OptimizationSuggestionExecutor::ExecutionResult
OptimizationSuggestionExecutor::performABTest(
    const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion,
    int test_duration_minutes) {

    ExecutionResult result;
    result.suggestion_id = suggestion.base_suggestion.optimization_type + "_abtest";

    if (!ab_tester_) {
        result.execution_successful = false;
        result.execution_log = "A/B testing framework not available";
        return result;
    }

    try {
        // 记录基准性能
        result.before_metrics = metrics_collector_->getCurrentMetrics();

        // 执行A/B测试
        // 这里应该调用实际的A/B测试框架
        result.execution_log = "A/B test executed for " + std::to_string(test_duration_minutes) + " minutes";

        // 模拟测试结果
        std::this_thread::sleep_for(std::chrono::seconds(std::min(test_duration_minutes * 6, 30))); // 最多等待30秒

        result.after_metrics = metrics_collector_->getCurrentMetrics();
        result.execution_successful = true;

        // 计算A/B测试结果
        float improvement = ((result.after_metrics.throughput_fps - result.before_metrics.throughput_fps) /
                           result.before_metrics.throughput_fps) * 100.0f;
        result.improvement_achieved["throughput"] = improvement;

    } catch (const std::exception& e) {
        result.execution_successful = false;
        result.execution_log = "A/B test failed: " + std::string(e.what());
    }

    return result;
}

std::string OptimizationSuggestionExecutor::generateExecutionReport() {
    std::stringstream ss;
    ss << "=== OPTIMIZATION EXECUTION REPORT ===\n\n";

    if (execution_history_.empty()) {
        ss << "No optimizations executed yet.\n";
        return ss.str();
    }

    int successful_executions = 0;
    float total_throughput_improvement = 0.0f;
    float total_latency_improvement = 0.0f;

    for (const auto& result : execution_history_) {
        if (result.execution_successful) {
            successful_executions++;

            auto throughput_it = result.improvement_achieved.find("throughput");
            if (throughput_it != result.improvement_achieved.end()) {
                total_throughput_improvement += throughput_it->second;
            }

            auto latency_it = result.improvement_achieved.find("latency");
            if (latency_it != result.improvement_achieved.end()) {
                total_latency_improvement += latency_it->second;
            }
        }
    }

    ss << "Total Executions: " << execution_history_.size() << "\n";
    ss << "Successful Executions: " << successful_executions << "\n";
    ss << "Success Rate: " << std::fixed << std::setprecision(1)
       << (static_cast<float>(successful_executions) / execution_history_.size() * 100.0f) << "%\n\n";

    if (successful_executions > 0) {
        ss << "Average Throughput Improvement: " << std::fixed << std::setprecision(1)
           << (total_throughput_improvement / successful_executions) << "%\n";
        ss << "Average Latency Improvement: " << std::fixed << std::setprecision(1)
           << (total_latency_improvement / successful_executions) << "%\n\n";
    }

    ss << "Recent Executions:\n";
    size_t recent_count = std::min(static_cast<size_t>(5), execution_history_.size());
    for (size_t i = execution_history_.size() - recent_count; i < execution_history_.size(); ++i) {
        const auto& result = execution_history_[i];
        ss << "  " << result.suggestion_id << ": "
           << (result.execution_successful ? "SUCCESS" : "FAILED") << "\n";
    }

    return ss.str();
}

void OptimizationSuggestionExecutor::updateExecutionConfig(const ExecutionConfig& new_config) {
    config_ = new_config;
}

// 私有方法实现
bool OptimizationSuggestionExecutor::executeTensorRTOptimization(const std::string& implementation_details) {
    // 这里应该实现实际的TensorRT优化逻辑
    // 简化实现，假设总是成功
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 模拟执行时间
    return true;
}

bool OptimizationSuggestionExecutor::executeBatchSizeOptimization(int new_batch_size) {
    // 这里应该实现实际的批大小优化逻辑
    if (new_batch_size <= 0 || new_batch_size > 32) {
        return false; // 无效的批大小
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 模拟执行时间
    return true;
}

bool OptimizationSuggestionExecutor::executeMemoryOptimization(const std::string& optimization_type) {
    // 这里应该实现实际的内存优化逻辑
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); // 模拟执行时间
    return true;
}

bool OptimizationSuggestionExecutor::executeStreamOptimization(int stream_count) {
    // 这里应该实现实际的流优化逻辑
    if (stream_count <= 0 || stream_count > 16) {
        return false; // 无效的流数量
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(150)); // 模拟执行时间
    return true;
}

bool OptimizationSuggestionExecutor::executePrecisionOptimization(const std::string& precision) {
    // 这里应该实现实际的精度优化逻辑
    if (precision != "fp16" && precision != "int8" && precision != "fp32") {
        return false; // 不支持的精度
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400)); // 模拟执行时间
    return true;
}

bool OptimizationSuggestionExecutor::performSafetyChecks(
    const AdvancedBottleneckAnalyzer::IntelligentOptimizationSuggestion& suggestion) {

    // 检查系统资源
    auto current_metrics = metrics_collector_->getCurrentMetrics();

    // 检查GPU利用率
    if (current_metrics.gpu_utilization > 95.0f) {
        return false; // GPU利用率过高，不适合进行优化
    }

    // 检查内存使用率
    if (current_metrics.memory_utilization > 90.0f) {
        return false; // 内存使用率过高
    }

    // 检查优化类型的风险级别
    const auto& opt_type = suggestion.base_suggestion.optimization_type;
    if (opt_type == "precision_optimization") {
        // 精度优化可能影响准确性，需要额外检查
        return suggestion.cost_benefit.roi_estimate > 150.0f; // 只有高ROI才执行
    }

    return true;
}

std::string OptimizationSuggestionExecutor::captureCurrentState() {
    // 这里应该捕获当前系统状态用于回滚
    // 简化实现，返回时间戳作为状态标识
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    return "state_" + std::to_string(timestamp);
}

bool OptimizationSuggestionExecutor::restoreState(const std::string& state_data) {
    // 这里应该实现实际的状态恢复逻辑
    // 简化实现，假设总是成功
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 模拟恢复时间
    return !state_data.empty();
}

void OptimizationSuggestionExecutor::monitorExecution(
    const std::string& suggestion_id,
    std::function<void(const std::string&)> progress_callback) {

    // 这里应该实现执行监控逻辑
    if (progress_callback) {
        progress_callback("Execution started for " + suggestion_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        progress_callback("Execution in progress for " + suggestion_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        progress_callback("Execution completed for " + suggestion_id);
    }
}

// PerformanceBenchmarkSystem 实现
PerformanceBenchmarkSystem::BenchmarkConfig::BenchmarkConfig()
    : test_scenarios({"single_image", "batch_inference", "streaming"}),
      batch_sizes({1, 4, 8, 16}),
      model_variants({"yolov5s", "yolov5m", "yolov5l"}),
      iterations_per_test(10),
      enable_warmup(true),
      warmup_iterations(5) {
}

std::string PerformanceBenchmarkSystem::BenchmarkResult::toString() const {
    std::stringstream ss;
    ss << "=== BENCHMARK RESULT ===\n";
    ss << "Test: " << test_name << "\n";
    ss << "Configuration: " << configuration << "\n";
    ss << "Stability: " << stability_assessment << " (CV: " << std::fixed << std::setprecision(3) << coefficient_of_variation << ")\n";

    ss << "Average Performance:\n";
    ss << "  Throughput: " << std::fixed << std::setprecision(1) << average_metrics.throughput_fps << " FPS\n";
    ss << "  Inference Time: " << std::fixed << std::setprecision(2) << average_metrics.inference_time_ms << " ms\n";
    ss << "  GPU Utilization: " << std::fixed << std::setprecision(1) << average_metrics.gpu_utilization << "%\n";
    ss << "  Memory Utilization: " << std::fixed << std::setprecision(1) << average_metrics.memory_utilization << "%\n";

    ss << "Best Performance:\n";
    ss << "  Throughput: " << std::fixed << std::setprecision(1) << best_metrics.throughput_fps << " FPS\n";
    ss << "  Inference Time: " << std::fixed << std::setprecision(2) << best_metrics.inference_time_ms << " ms\n";

    return ss.str();
}

PerformanceBenchmarkSystem::PerformanceBenchmarkSystem(
    std::shared_ptr<RealtimeMetricsCollector> collector,
    const BenchmarkConfig& config)
    : config_(config), metrics_collector_(collector) {
}

std::vector<PerformanceBenchmarkSystem::BenchmarkResult>
PerformanceBenchmarkSystem::runComprehensiveBenchmark() {
    std::vector<BenchmarkResult> results;

    for (const auto& scenario : config_.test_scenarios) {
        for (const auto& model : config_.model_variants) {
            for (int batch_size : config_.batch_sizes) {
                std::string configuration = model + "_batch" + std::to_string(batch_size);

                auto result = runSingleBenchmark(scenario, configuration);
                results.push_back(result);
            }
        }
    }

    // 保存到历史记录
    benchmark_history_.insert(benchmark_history_.end(), results.begin(), results.end());

    return results;
}

PerformanceBenchmarkSystem::BenchmarkResult
PerformanceBenchmarkSystem::runSingleBenchmark(const std::string& test_name, const std::string& configuration) {
    BenchmarkResult result;
    result.test_name = test_name;
    result.configuration = configuration;

    // 预热
    if (config_.enable_warmup) {
        performWarmup(config_.warmup_iterations);
    }

    // 执行基准测试
    std::vector<ExtendedRealtimeMetrics> measurements;
    measurements.reserve(config_.iterations_per_test);

    for (int i = 0; i < config_.iterations_per_test; ++i) {
        // 执行单次测试
        auto metrics = executeBenchmarkTest(test_name, configuration);
        measurements.push_back(metrics);

        // 短暂等待以避免热效应
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    result.all_measurements = measurements;

    // 计算统计数据
    result.average_metrics = calculateStatistics(measurements, "average");
    result.best_metrics = calculateStatistics(measurements, "best");
    result.worst_metrics = calculateStatistics(measurements, "worst");

    // 计算变异系数
    std::vector<float> throughput_values;
    for (const auto& m : measurements) {
        throughput_values.push_back(m.throughput_fps);
    }
    result.coefficient_of_variation = calculateCoefficientOfVariation(throughput_values);
    result.stability_assessment = assessStability(result.coefficient_of_variation);

    return result;
}

std::string PerformanceBenchmarkSystem::compareConfigurations(const std::vector<BenchmarkResult>& results) {
    std::stringstream ss;
    ss << "=== CONFIGURATION COMPARISON ===\n\n";

    if (results.empty()) {
        ss << "No results to compare.\n";
        return ss.str();
    }

    // 按吞吐量排序
    auto sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.average_metrics.throughput_fps > b.average_metrics.throughput_fps;
        });

    ss << "Performance Ranking (by throughput):\n";
    for (size_t i = 0; i < sorted_results.size(); ++i) {
        const auto& result = sorted_results[i];
        ss << (i + 1) << ". " << result.configuration << " (" << result.test_name << ")\n";
        ss << "   Throughput: " << std::fixed << std::setprecision(1) << result.average_metrics.throughput_fps << " FPS\n";
        ss << "   Latency: " << std::fixed << std::setprecision(2) << result.average_metrics.inference_time_ms << " ms\n";
        ss << "   Stability: " << result.stability_assessment << "\n\n";
    }

    // 分析最佳配置
    if (!sorted_results.empty()) {
        const auto& best = sorted_results[0];
        ss << "Best Overall Configuration: " << best.configuration << "\n";
        ss << "Performance Advantage: ";

        if (sorted_results.size() > 1) {
            const auto& second_best = sorted_results[1];
            float advantage = ((best.average_metrics.throughput_fps - second_best.average_metrics.throughput_fps) /
                              second_best.average_metrics.throughput_fps) * 100.0f;
            ss << "+" << std::fixed << std::setprecision(1) << advantage << "% over second best\n";
        } else {
            ss << "No comparison available\n";
        }
    }

    return ss.str();
}

std::string PerformanceBenchmarkSystem::generatePerformanceRanking(const std::vector<BenchmarkResult>& results) {
    std::stringstream ss;
    ss << "=== PERFORMANCE RANKING ===\n\n";

    // 创建综合评分
    struct ScoredResult {
        BenchmarkResult result;
        float composite_score;
    };

    std::vector<ScoredResult> scored_results;

    for (const auto& result : results) {
        ScoredResult scored;
        scored.result = result;

        // 综合评分：吞吐量权重60%，延迟权重30%，稳定性权重10%
        float throughput_score = result.average_metrics.throughput_fps / 100.0f; // 归一化到0-1
        float latency_score = std::max(0.0f, (100.0f - result.average_metrics.inference_time_ms) / 100.0f);
        float stability_score = 1.0f - std::min(1.0f, result.coefficient_of_variation);

        scored.composite_score = throughput_score * 0.6f + latency_score * 0.3f + stability_score * 0.1f;
        scored_results.push_back(scored);
    }

    // 按综合评分排序
    std::sort(scored_results.begin(), scored_results.end(),
        [](const ScoredResult& a, const ScoredResult& b) {
            return a.composite_score > b.composite_score;
        });

    ss << "Composite Performance Ranking:\n";
    for (size_t i = 0; i < scored_results.size(); ++i) {
        const auto& scored = scored_results[i];
        ss << (i + 1) << ". " << scored.result.configuration << " (Score: "
           << std::fixed << std::setprecision(3) << scored.composite_score << ")\n";
        ss << "   " << scored.result.test_name << "\n";
        ss << "   Throughput: " << std::fixed << std::setprecision(1)
           << scored.result.average_metrics.throughput_fps << " FPS\n";
        ss << "   Latency: " << std::fixed << std::setprecision(2)
           << scored.result.average_metrics.inference_time_ms << " ms\n";
        ss << "   Stability: " << scored.result.stability_assessment << "\n\n";
    }

    return ss.str();
}

bool PerformanceBenchmarkSystem::detectPerformanceRegression(
    const BenchmarkResult& current, const BenchmarkResult& baseline, float regression_threshold) {

    float throughput_change = (current.average_metrics.throughput_fps - baseline.average_metrics.throughput_fps) /
                             baseline.average_metrics.throughput_fps;

    float latency_change = (current.average_metrics.inference_time_ms - baseline.average_metrics.inference_time_ms) /
                          baseline.average_metrics.inference_time_ms;

    // 检测回归：吞吐量下降或延迟增加超过阈值
    return (throughput_change < -regression_threshold) || (latency_change > regression_threshold);
}

std::string PerformanceBenchmarkSystem::generateRegressionReport(
    const std::vector<BenchmarkResult>& current_results,
    const std::vector<BenchmarkResult>& baseline_results) {

    std::stringstream ss;
    ss << "=== PERFORMANCE REGRESSION REPORT ===\n\n";

    int regressions_detected = 0;
    int improvements_detected = 0;

    // 创建配置映射以便比较
    std::map<std::string, BenchmarkResult> baseline_map;
    for (const auto& result : baseline_results) {
        std::string key = result.test_name + "_" + result.configuration;
        baseline_map[key] = result;
    }

    for (const auto& current : current_results) {
        std::string key = current.test_name + "_" + current.configuration;
        auto baseline_it = baseline_map.find(key);

        if (baseline_it != baseline_map.end()) {
            const auto& baseline = baseline_it->second;

            bool is_regression = detectPerformanceRegression(current, baseline);

            float throughput_change = ((current.average_metrics.throughput_fps - baseline.average_metrics.throughput_fps) /
                                     baseline.average_metrics.throughput_fps) * 100.0f;

            float latency_change = ((current.average_metrics.inference_time_ms - baseline.average_metrics.inference_time_ms) /
                                  baseline.average_metrics.inference_time_ms) * 100.0f;

            if (is_regression) {
                regressions_detected++;
                ss << "REGRESSION DETECTED: " << current.configuration << " (" << current.test_name << ")\n";
            } else if (throughput_change > 5.0f || latency_change < -5.0f) {
                improvements_detected++;
                ss << "IMPROVEMENT DETECTED: " << current.configuration << " (" << current.test_name << ")\n";
            } else {
                ss << "STABLE: " << current.configuration << " (" << current.test_name << ")\n";
            }

            ss << "  Throughput: " << std::fixed << std::setprecision(1) << throughput_change << "%\n";
            ss << "  Latency: " << std::fixed << std::setprecision(1) << latency_change << "%\n\n";
        }
    }

    ss << "Summary:\n";
    ss << "  Regressions: " << regressions_detected << "\n";
    ss << "  Improvements: " << improvements_detected << "\n";
    ss << "  Stable: " << (current_results.size() - regressions_detected - improvements_detected) << "\n";

    return ss.str();
}

// 私有方法实现
ExtendedRealtimeMetrics PerformanceBenchmarkSystem::executeBenchmarkTest(
    const std::string& test_name, const std::string& configuration) {

    // 这里应该执行实际的基准测试
    // 简化实现，返回模拟的指标

    ExtendedRealtimeMetrics metrics;

    // 模拟不同配置的性能差异
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.05f); // 5%的噪声

    // 基础性能值（根据配置调整）
    float base_throughput = 50.0f;
    float base_latency = 20.0f;

    if (configuration.find("yolov5s") != std::string::npos) {
        base_throughput = 80.0f;
        base_latency = 12.5f;
    } else if (configuration.find("yolov5m") != std::string::npos) {
        base_throughput = 60.0f;
        base_latency = 16.7f;
    } else if (configuration.find("yolov5l") != std::string::npos) {
        base_throughput = 40.0f;
        base_latency = 25.0f;
    }

    // 批大小影响
    if (configuration.find("batch4") != std::string::npos) {
        base_throughput *= 1.8f;
        base_latency *= 1.1f;
    } else if (configuration.find("batch8") != std::string::npos) {
        base_throughput *= 3.2f;
        base_latency *= 1.3f;
    } else if (configuration.find("batch16") != std::string::npos) {
        base_throughput *= 5.5f;
        base_latency *= 1.6f;
    }

    // 添加噪声
    metrics.throughput_fps = base_throughput * (1.0f + noise(gen));
    metrics.inference_time_ms = base_latency * (1.0f + noise(gen));
    metrics.gpu_utilization = 85.0f + noise(gen) * 100.0f;
    metrics.memory_utilization = 70.0f + noise(gen) * 100.0f;

    // 确保值在合理范围内
    metrics.throughput_fps = std::max(1.0f, metrics.throughput_fps);
    metrics.inference_time_ms = std::max(1.0f, metrics.inference_time_ms);
    metrics.gpu_utilization = std::max(0.0f, std::min(100.0f, metrics.gpu_utilization));
    metrics.memory_utilization = std::max(0.0f, std::min(100.0f, metrics.memory_utilization));

    // 模拟测试执行时间
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    return metrics;
}

void PerformanceBenchmarkSystem::performWarmup(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // 执行预热推理
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

float PerformanceBenchmarkSystem::calculateCoefficientOfVariation(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;

    float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();

    float variance = 0.0f;
    for (float value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();

    float std_dev = std::sqrt(variance);
    return (mean != 0.0f) ? (std_dev / mean) : 0.0f;
}

std::string PerformanceBenchmarkSystem::assessStability(float cv) {
    if (cv < 0.05f) {
        return "excellent";
    } else if (cv < 0.10f) {
        return "good";
    } else if (cv < 0.20f) {
        return "fair";
    } else {
        return "poor";
    }
}

ExtendedRealtimeMetrics PerformanceBenchmarkSystem::calculateStatistics(
    const std::vector<ExtendedRealtimeMetrics>& measurements, const std::string& statistic_type) {

    if (measurements.empty()) {
        return ExtendedRealtimeMetrics{};
    }

    ExtendedRealtimeMetrics result;

    if (statistic_type == "average") {
        float sum_throughput = 0.0f, sum_latency = 0.0f, sum_gpu = 0.0f, sum_memory = 0.0f;

        for (const auto& m : measurements) {
            sum_throughput += m.throughput_fps;
            sum_latency += m.inference_time_ms;
            sum_gpu += m.gpu_utilization;
            sum_memory += m.memory_utilization;
        }

        float count = static_cast<float>(measurements.size());
        result.throughput_fps = sum_throughput / count;
        result.inference_time_ms = sum_latency / count;
        result.gpu_utilization = sum_gpu / count;
        result.memory_utilization = sum_memory / count;

    } else if (statistic_type == "best") {
        result = *std::max_element(measurements.begin(), measurements.end(),
            [](const ExtendedRealtimeMetrics& a, const ExtendedRealtimeMetrics& b) {
                return a.throughput_fps < b.throughput_fps;
            });

    } else if (statistic_type == "worst") {
        result = *std::min_element(measurements.begin(), measurements.end(),
            [](const ExtendedRealtimeMetrics& a, const ExtendedRealtimeMetrics& b) {
                return a.throughput_fps < b.throughput_fps;
            });
    }

    return result;
}

bool PerformanceBenchmarkSystem::saveBenchmarkResults(const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        file << "# Benchmark Results\n";
        file << "# Generated at: " << std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << "\n\n";

        for (const auto& result : benchmark_history_) {
            file << result.toString() << "\n";
        }

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save benchmark results: " << e.what() << std::endl;
        return false;
    }
}

bool PerformanceBenchmarkSystem::loadBenchmarkResults(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        // 简化实现：这里应该解析文件内容并重建BenchmarkResult对象
        // 当前只是验证文件可以打开

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load benchmark results: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, float> PerformanceBenchmarkSystem::calculatePerformanceStatistics(
    const std::vector<BenchmarkResult>& results) {

    std::map<std::string, float> statistics;

    if (results.empty()) {
        return statistics;
    }

    std::vector<float> throughput_values, latency_values, gpu_values, memory_values;

    for (const auto& result : results) {
        throughput_values.push_back(result.average_metrics.throughput_fps);
        latency_values.push_back(result.average_metrics.inference_time_ms);
        gpu_values.push_back(result.average_metrics.gpu_utilization);
        memory_values.push_back(result.average_metrics.memory_utilization);
    }

    // 计算各种统计指标
    statistics["throughput_mean"] = std::accumulate(throughput_values.begin(), throughput_values.end(), 0.0f) / throughput_values.size();
    statistics["throughput_max"] = *std::max_element(throughput_values.begin(), throughput_values.end());
    statistics["throughput_min"] = *std::min_element(throughput_values.begin(), throughput_values.end());
    statistics["throughput_cv"] = calculateCoefficientOfVariation(throughput_values);

    statistics["latency_mean"] = std::accumulate(latency_values.begin(), latency_values.end(), 0.0f) / latency_values.size();
    statistics["latency_max"] = *std::max_element(latency_values.begin(), latency_values.end());
    statistics["latency_min"] = *std::min_element(latency_values.begin(), latency_values.end());
    statistics["latency_cv"] = calculateCoefficientOfVariation(latency_values);

    return statistics;
}

std::string PerformanceBenchmarkSystem::generateStatisticalReport(const std::vector<BenchmarkResult>& results) {
    std::stringstream ss;
    ss << "=== STATISTICAL ANALYSIS REPORT ===\n\n";

    auto stats = calculatePerformanceStatistics(results);

    if (stats.empty()) {
        ss << "No data available for statistical analysis.\n";
        return ss.str();
    }

    ss << "Throughput Statistics:\n";
    ss << "  Mean: " << std::fixed << std::setprecision(1) << stats["throughput_mean"] << " FPS\n";
    ss << "  Range: " << std::fixed << std::setprecision(1) << stats["throughput_min"]
       << " - " << stats["throughput_max"] << " FPS\n";
    ss << "  Coefficient of Variation: " << std::fixed << std::setprecision(3) << stats["throughput_cv"] << "\n\n";

    ss << "Latency Statistics:\n";
    ss << "  Mean: " << std::fixed << std::setprecision(2) << stats["latency_mean"] << " ms\n";
    ss << "  Range: " << std::fixed << std::setprecision(2) << stats["latency_min"]
       << " - " << stats["latency_max"] << " ms\n";
    ss << "  Coefficient of Variation: " << std::fixed << std::setprecision(3) << stats["latency_cv"] << "\n\n";

    // 性能分布分析
    ss << "Performance Distribution:\n";
    int high_performance = 0, medium_performance = 0, low_performance = 0;

    for (const auto& result : results) {
        if (result.average_metrics.throughput_fps > stats["throughput_mean"] * 1.2f) {
            high_performance++;
        } else if (result.average_metrics.throughput_fps > stats["throughput_mean"] * 0.8f) {
            medium_performance++;
        } else {
            low_performance++;
        }
    }

    ss << "  High Performance (>120% of mean): " << high_performance << " configurations\n";
    ss << "  Medium Performance (80-120% of mean): " << medium_performance << " configurations\n";
    ss << "  Low Performance (<80% of mean): " << low_performance << " configurations\n";

    return ss.str();
}

} // namespace yolo_acceleration
