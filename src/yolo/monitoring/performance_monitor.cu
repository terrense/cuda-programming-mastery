#include "performance_monitor.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>
#include <fstream>
#include <numeric>

namespace yolo_acceleration {

// RealtimeMetrics 实现
RealtimeMetrics::RealtimeMetrics()
    : fps(0.0f), latency_ms(0.0f), throughput_images_per_sec(0.0f),
      gpu_utilization_percent(0.0f), memory_usage_mb(0.0f),
      memory_bandwidth_gb_s(0.0f), compute_utilization_percent(0.0f),
      timestamp(std::chrono::steady_clock::now()) {}

std::string RealtimeMetrics::toString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "FPS: " << fps << ", ";
    ss << "延迟: " << latency_ms << "ms, ";
    ss << "吞吐量: " << throughput_images_per_sec << " images/s, ";
    ss << "GPU利用率: " << gpu_utilization_percent << "%, ";
    ss << "内存使用: " << memory_usage_mb << "MB, ";
    ss << "内存带宽: " << memory_bandwidth_gb_s << "GB/s, ";
    ss << "计算利用率: " << compute_utilization_percent << "%";
    return ss.str();
}

// YOLOBottleneckAnalysis 实现
YOLOBottleneckAnalysis::YOLOBottleneckAnalysis()
    : primaryBottleneck(YOLOBottleneckType::NONE),
      secondaryBottleneck(YOLOBottleneckType::NONE),
      severity(0.0f) {}

std::string YOLOBottleneckAnalysis::toString() const {
    std::stringstream ss;

    auto bottleneckToString = [](YOLOBottleneckType type) -> std::string {
        switch (type) {
            case YOLOBottleneckType::PREPROCESSING: return "预处理瓶颈";
            case YOLOBottleneckType::BACKBONE: return "骨干网络瓶颈";
            case YOLOBottleneckType::NECK: return "颈部网络瓶颈";
            case YOLOBottleneckType::HEAD: return "检测头瓶颈";
            case YOLOBottleneckType::POSTPROCESSING: return "后处理瓶颈";
            case YOLOBottleneckType::MEMORY_TRANSFER: return "内存传输瓶颈";
            case YOLOBottleneckType::BATCH_SIZE: return "批大小限制";
            case YOLOBottleneckType::NONE: return "无明显瓶颈";
            default: return "未知瓶颈";
        }
    };

    ss << "=== YOLO瓶颈分析 ===\n";
    ss << "主要瓶颈: " << bottleneckToString(primaryBottleneck) << "\n";
    if (secondaryBottleneck != YOLOBottleneckType::NONE) {
        ss << "次要瓶颈: " << bottleneckToString(secondaryBottleneck) << "\n";
    }
    ss << "严重程度: " << std::fixed << std::setprecision(1) << (severity * 100) << "%\n";
    ss << "描述: " << description << "\n";

    if (!componentTimes.empty()) {
        ss << "组件耗时分析:\n";
        for (const auto& [component, time] : componentTimes) {
            ss << "  " << component << ": " << std::fixed << std::setprecision(2)
               << time << "ms\n";
        }
    }

    if (!suggestions.empty()) {
        ss << "优化建议:\n";
        for (size_t i = 0; i < suggestions.size(); i++) {
            ss << "  " << (i + 1) << ". " << suggestions[i] << "\n";
        }
    }

    return ss.str();
}

// YOLOOptimizationSuggestion 实现
YOLOOptimizationSuggestion::YOLOOptimizationSuggestion()
    : type(YOLOOptimizationType::TENSORRT_OPTIMIZATION),
      expectedImprovement(0.0f), priority(3), implementationCost(0.5f) {}

std::string YOLOOptimizationSuggestion::toString() const {
    std::stringstream ss;
    ss << "=== 优化建议 ===\n";
    ss << "标题: " << title << "\n";
    ss << "描述: " << description << "\n";
    ss << "预期提升: " << std::fixed << std::setprecision(1) << expectedImprovement << "%\n";
    ss << "优先级: " << priority << "/5\n";
    ss << "实现成本: " << std::fixed << std::setprecision(1) << (implementationCost * 100) << "%\n";
    if (!implementation.empty()) {
        ss << "实现方案:\n" << implementation << "\n";
    }
    return ss.str();
}

// RealtimePerformanceMonitor 实现
RealtimePerformanceMonitor::RealtimePerformanceMonitor(size_t max_history)
    : monitoring_(false), max_history_size_(max_history),
      total_frames_(0), total_time_ms_(0) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

RealtimePerformanceMonitor::~RealtimePerformanceMonitor() {
    stopMonitoring();
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void RealtimePerformanceMonitor::startMonitoring() {
    if (!monitoring_) {
        monitoring_ = true;
        last_update_ = std::chrono::steady_clock::now();
        monitor_thread_ = std::thread(&RealtimePerformanceMonitor::monitoringLoop, this);
    }
}

void RealtimePerformanceMonitor::stopMonitoring() {
    if (monitoring_) {
        monitoring_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
}

void RealtimePerformanceMonitor::recordInferenceStart() {
    cudaEventRecord(start_event_);
}

void RealtimePerformanceMonitor::recordInferenceEnd(int batch_size) {
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_);

    RealtimeMetrics metrics;
    metrics.latency_ms = elapsed_time;
    metrics.fps = batch_size * 1000.0f / elapsed_time;
    metrics.throughput_images_per_sec = metrics.fps;
    metrics.timestamp = std::chrono::steady_clock::now();

    updateGPUMetrics(metrics);
    calculateDerivedMetrics(metrics);

    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_history_.push(metrics);
    if (metrics_history_.size() > max_history_size_) {
        metrics_history_.pop();
    }

    total_frames_ += batch_size;
    total_time_ms_ += elapsed_time;

    // 调用回调函数
    for (const auto& callback : callbacks_) {
        callback(metrics);
    }
}

void RealtimePerformanceMonitor::recordCustomMetrics(const RealtimeMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_history_.push(metrics);
    if (metrics_history_.size() > max_history_size_) {
        metrics_history_.pop();
    }

    // 调用回调函数
    for (const auto& callback : callbacks_) {
        callback(metrics);
    }
}

RealtimeMetrics RealtimePerformanceMonitor::getCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!metrics_history_.empty()) {
        return metrics_history_.back();
    }
    return RealtimeMetrics();
}

std::vector<RealtimeMetrics> RealtimePerformanceMonitor::getRecentMetrics(size_t count) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    std::vector<RealtimeMetrics> result;

    std::queue<RealtimeMetrics> temp_queue = metrics_history_;
    while (!temp_queue.empty() && result.size() < count) {
        result.insert(result.begin(), temp_queue.back());
        temp_queue.pop();
    }

    return result;
}

RealtimeMetrics RealtimePerformanceMonitor::getAverageMetrics(size_t window_size) const {
    auto recent_metrics = getRecentMetrics(window_size);
    if (recent_metrics.empty()) {
        return RealtimeMetrics();
    }

    RealtimeMetrics avg;
    for (const auto& metrics : recent_metrics) {
        avg.fps += metrics.fps;
        avg.latency_ms += metrics.latency_ms;
        avg.throughput_images_per_sec += metrics.throughput_images_per_sec;
        avg.gpu_utilization_percent += metrics.gpu_utilization_percent;
        avg.memory_usage_mb += metrics.memory_usage_mb;
        avg.memory_bandwidth_gb_s += metrics.memory_bandwidth_gb_s;
        avg.compute_utilization_percent += metrics.compute_utilization_percent;
    }

    float count = static_cast<float>(recent_metrics.size());
    avg.fps /= count;
    avg.latency_ms /= count;
    avg.throughput_images_per_sec /= count;
    avg.gpu_utilization_percent /= count;
    avg.memory_usage_mb /= count;
    avg.memory_bandwidth_gb_s /= count;
    avg.compute_utilization_percent /= count;

    return avg;
}

void RealtimePerformanceMonitor::registerCallback(
    std::function<void(const RealtimeMetrics&)> callback) {
    callbacks_.push_back(callback);
}

std::string RealtimePerformanceMonitor::generateRealtimeReport() const {
    auto current = getCurrentMetrics();
    auto average = getAverageMetrics(100);

    std::stringstream ss;
    ss << "=== 实时性能监控报告 ===\n";
    ss << "当前指标:\n" << current.toString() << "\n\n";
    ss << "平均指标 (最近100次):\n" << average.toString() << "\n\n";

    if (total_frames_ > 0) {
        ss << "总体统计:\n";
        ss << "  总帧数: " << total_frames_ << "\n";
        ss << "  总时间: " << std::fixed << std::setprecision(2) << total_time_ms_ << "ms\n";
        ss << "  平均FPS: " << std::fixed << std::setprecision(2)
           << (total_frames_ * 1000.0f / total_time_ms_) << "\n";
    }

    return ss.str();
}

bool RealtimePerformanceMonitor::detectPerformanceAnomaly(float threshold) const {
    auto recent_metrics = getRecentMetrics(50);
    if (recent_metrics.size() < 10) {
        return false;
    }

    // 计算FPS的标准差
    float mean_fps = 0.0f;
    for (const auto& metrics : recent_metrics) {
        mean_fps += metrics.fps;
    }
    mean_fps /= recent_metrics.size();

    float variance = 0.0f;
    for (const auto& metrics : recent_metrics) {
        variance += (metrics.fps - mean_fps) * (metrics.fps - mean_fps);
    }
    variance /= recent_metrics.size();
    float std_dev = std::sqrt(variance);

    // 检测异常值
    float cv = std_dev / mean_fps; // 变异系数
    return cv > threshold;
}

void RealtimePerformanceMonitor::monitoringLoop() {
    while (monitoring_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 定期更新GPU指标
        RealtimeMetrics metrics;
        updateGPUMetrics(metrics);

        // 如果有新的指标数据，记录它
        if (metrics.gpu_utilization_percent > 0) {
            recordCustomMetrics(metrics);
        }
    }
}

void RealtimePerformanceMonitor::updateGPUMetrics(RealtimeMetrics& metrics) {
    // 获取GPU内存使用情况
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    metrics.memory_usage_mb = (total_mem - free_mem) / (1024.0f * 1024.0f);

    // 简化的GPU利用率估算（实际应用中可以使用NVML）
    metrics.gpu_utilization_percent = std::min(95.0f,
        metrics.memory_usage_mb / (total_mem / (1024.0f * 1024.0f)) * 100.0f);

    // 估算计算利用率
    metrics.compute_utilization_percent = std::min(90.0f,
        metrics.gpu_utilization_percent * 0.8f);
}

void RealtimePerformanceMonitor::calculateDerivedMetrics(RealtimeMetrics& metrics) {
    // 根据延迟计算FPS（如果还没有设置）
    if (metrics.fps == 0.0f && metrics.latency_ms > 0.0f) {
        metrics.fps = 1000.0f / metrics.latency_ms;
    }

    // 根据FPS计算吞吐量（如果还没有设置）
    if (metrics.throughput_images_per_sec == 0.0f) {
        metrics.throughput_images_per_sec = metrics.fps;
    }

    // 估算内存带宽使用
    if (metrics.memory_bandwidth_gb_s == 0.0f) {
        // 基于GPU利用率的简单估算
        metrics.memory_bandwidth_gb_s = metrics.gpu_utilization_percent * 5.0f; // 假设峰值500GB/s
    }
}

// YOLOBottleneckAnalyzer 实现
YOLOBottleneckAnalysis YOLOBottleneckAnalyzer::analyzeYOLOBottlenecks(
    const std::vector<RealtimeMetrics>& metrics_history,
    const std::map<std::string, float>& component_times) {

    YOLOBottleneckAnalysis analysis;

    if (metrics_history.empty()) {
        return analysis;
    }

    // 计算平均指标
    RealtimeMetrics avg_metrics;
    for (const auto& metrics : metrics_history) {
        avg_metrics.fps += metrics.fps;
        avg_metrics.latency_ms += metrics.latency_ms;
        avg_metrics.gpu_utilization_percent += metrics.gpu_utilization_percent;
        avg_metrics.memory_usage_mb += metrics.memory_usage_mb;
    }
    float count = static_cast<float>(metrics_history.size());
    avg_metrics.fps /= count;
    avg_metrics.latency_ms /= count;
    avg_metrics.gpu_utilization_percent /= count;
    avg_metrics.memory_usage_mb /= count;

    analysis.componentTimes = component_times;

    // 分析组件瓶颈
    analysis.primaryBottleneck = analyzeComponentBottleneck(component_times);

    // 根据瓶颈类型设置严重程度和建议
    switch (analysis.primaryBottleneck) {
        case YOLOBottleneckType::BACKBONE:
            analysis.severity = 0.8f;
            analysis.description = "骨干网络计算成为主要瓶颈，占用大部分推理时间";
            analysis.suggestions.push_back("使用TensorRT优化骨干网络");
            analysis.suggestions.push_back("考虑使用更轻量的骨干网络架构");
            analysis.suggestions.push_back("启用混合精度计算");
            break;

        case YOLOBottleneckType::POSTPROCESSING:
            analysis.severity = 0.6f;
            analysis.description = "后处理（NMS等）成为瓶颈，CPU处理效率低";
            analysis.suggestions.push_back("使用GPU加速的NMS实现");
            analysis.suggestions.push_back("优化后处理算法");
            analysis.suggestions.push_back("减少检测框数量");
            break;

        case YOLOBottleneckType::MEMORY_TRANSFER:
            analysis.severity = 0.7f;
            analysis.description = "内存传输成为瓶颈，数据传输耗时过长";
            analysis.suggestions.push_back("使用CUDA流进行异步传输");
            analysis.suggestions.push_back("优化数据布局和内存访问模式");
            analysis.suggestions.push_back("减少不必要的内存拷贝");
            break;

        case YOLOBottleneckType::BATCH_SIZE:
            analysis.severity = 0.5f;
            analysis.description = "批大小配置不当，GPU利用率不足";
            analysis.suggestions.push_back("增加批大小以提高GPU利用率");
            analysis.suggestions.push_back("使用动态批处理");
            break;

        default:
            analysis.primaryBottleneck = YOLOBottleneckType::NONE;
            analysis.description = "性能表现良好，无明显瓶颈";
            analysis.suggestions.push_back("继续监控性能变化");
            break;
    }

    return analysis;
}

YOLOBottleneckType YOLOBottleneckAnalyzer::analyzeComponentBottleneck(
    const std::map<std::string, float>& component_times) {

    if (component_times.empty()) {
        return YOLOBottleneckType::NONE;
    }

    // 找到耗时最长的组件
    auto max_component = std::max_element(component_times.begin(), component_times.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    const std::string& max_component_name = max_component->first;
    float max_time = max_component->second;

    // 计算总时间
    float total_time = 0.0f;
    for (const auto& [name, time] : component_times) {
        total_time += time;
    }

    // 如果某个组件占用超过40%的时间，认为是瓶颈
    if (max_time / total_time > 0.4f) {
        if (max_component_name.find("backbone") != std::string::npos) {
            return YOLOBottleneckType::BACKBONE;
        } else if (max_component_name.find("neck") != std::string::npos) {
            return YOLOBottleneckType::NECK;
        } else if (max_component_name.find("head") != std::string::npos) {
            return YOLOBottleneckType::HEAD;
        } else if (max_component_name.find("preprocess") != std::string::npos) {
            return YOLOBottleneckType::PREPROCESSING;
        } else if (max_component_name.find("postprocess") != std::string::npos) {
            return YOLOBottleneckType::POSTPROCESSING;
        } else if (max_component_name.find("memory") != std::string::npos) {
            return YOLOBottleneckType::MEMORY_TRANSFER;
        }
    }

    return YOLOBottleneckType::NONE;
}

std::vector<YOLOOptimizationSuggestion> YOLOBottleneckAnalyzer::generateYOLOOptimizations(
    const YOLOBottleneckAnalysis& analysis,
    const RealtimeMetrics& current_metrics) {

    std::vector<YOLOOptimizationSuggestion> suggestions;

    // 基于瓶颈类型生成具体建议
    switch (analysis.primaryBottleneck) {
        case YOLOBottleneckType::BACKBONE: {
            YOLOOptimizationSuggestion tensorrt_opt;
            tensorrt_opt.type = YOLOOptimizationType::TENSORRT_OPTIMIZATION;
            tensorrt_opt.title = "TensorRT骨干网络优化";
            tensorrt_opt.description = "使用TensorRT对骨干网络进行深度优化";
            tensorrt_opt.expectedImprovement = 40.0f;
            tensorrt_opt.priority = 5;
            tensorrt_opt.implementationCost = 0.3f;
            tensorrt_opt.implementation = R"(
// 启用TensorRT优化
trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
trt_config->setMaxWorkspaceSize(1 << 30); // 1GB
)";
            suggestions.push_back(tensorrt_opt);

            YOLOOptimizationSuggestion precision_opt;
            precision_opt.type = YOLOOptimizationType::PRECISION_OPTIMIZATION;
            precision_opt.title = "混合精度优化";
            precision_opt.description = "使用FP16或INT8精度减少计算量";
            precision_opt.expectedImprovement = 25.0f;
            precision_opt.priority = 4;
            precision_opt.implementationCost = 0.2f;
            suggestions.push_back(precision_opt);
            break;
        }

        case YOLOBottleneckType::POSTPROCESSING: {
            YOLOOptimizationSuggestion custom_kernel;
            custom_kernel.type = YOLOOptimizationType::CUSTOM_KERNEL_OPT;
            custom_kernel.title = "GPU加速后处理";
            custom_kernel.description = "实现GPU版本的NMS和后处理算法";
            custom_kernel.expectedImprovement = 50.0f;
            custom_kernel.priority = 5;
            custom_kernel.implementationCost = 0.7f;
            custom_kernel.implementation = R"(
// GPU NMS核函数
__global__ void gpu_nms_kernel(float* boxes, float* scores,
                              bool* suppressed, int num_boxes, float threshold);
)";
            suggestions.push_back(custom_kernel);
            break;
        }

        case YOLOBottleneckType::MEMORY_TRANSFER: {
            YOLOOptimizationSuggestion stream_opt;
            stream_opt.type = YOLOOptimizationType::STREAM_PARALLELISM;
            stream_opt.title = "CUDA流并行优化";
            stream_opt.description = "使用多个CUDA流实现数据传输和计算的重叠";
            stream_opt.expectedImprovement = 30.0f;
            stream_opt.priority = 4;
            stream_opt.implementationCost = 0.4f;
            suggestions.push_back(stream_opt);

            YOLOOptimizationSuggestion memory_opt;
            memory_opt.type = YOLOOptimizationType::MEMORY_OPTIMIZATION;
            memory_opt.title = "内存访问优化";
            memory_opt.description = "优化数据布局和内存访问模式";
            memory_opt.expectedImprovement = 20.0f;
            memory_opt.priority = 3;
            memory_opt.implementationCost = 0.5f;
            suggestions.push_back(memory_opt);
            break;
        }

        case YOLOBottleneckType::BATCH_SIZE: {
            YOLOOptimizationSuggestion batch_opt;
            batch_opt.type = YOLOOptimizationType::BATCH_SIZE_TUNING;
            batch_opt.title = "批大小自动调优";
            batch_opt.description = "自动找到最优的批处理大小";
            batch_opt.expectedImprovement = 35.0f;
            batch_opt.priority = 4;
            batch_opt.implementationCost = 0.2f;
            suggestions.push_back(batch_opt);
            break;
        }

        default:
            break;
    }

    // 通用优化建议
    if (current_metrics.gpu_utilization_percent < 70.0f) {
        YOLOOptimizationSuggestion fusion_opt;
        fusion_opt.type = YOLOOptimizationType::OPERATOR_FUSION;
        fusion_opt.title = "算子融合优化";
        fusion_opt.description = "融合相邻的卷积、批归一化和激活函数";
        fusion_opt.expectedImprovement = 15.0f;
        fusion_opt.priority = 3;
        fusion_opt.implementationCost = 0.3f;
        suggestions.push_back(fusion_opt);
    }

    return suggestions;
}

// ABTestingFramework 实现
ABTestingFramework::ABTestConfig::ABTestConfig()
    : sample_size(100), significance_level(0.05f),
      metrics({"fps", "latency_ms", "gpu_utilization_percent"}) {}

ABTestingFramework::ABTestResult::ABTestResult()
    : is_significant(false), p_value(1.0f), effect_size(0.0f) {}

std::string ABTestingFramework::ABTestResult::toString() const {
    std::stringstream ss;
    ss << "=== A/B测试结果: " << test_name << " ===\n";
    ss << "统计显著性: " << (is_significant ? "是" : "否") << "\n";
    ss << "P值: " << std::fixed << std::setprecision(4) << p_value << "\n";
    ss << "效应大小: " << std::fixed << std::setprecision(3) << effect_size << "\n";

    ss << "\n基线指标:\n";
    for (const auto& [metric, value] : baseline_metrics) {
        ss << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << "\n";
    }

    ss << "\n变体指标:\n";
    for (const auto& [metric, value] : variant_metrics) {
        ss << "  " << metric << ": " << std::fixed << std::setprecision(2) << value << "\n";
    }

    ss << "\n改进百分比:\n";
    for (const auto& [metric, improvement] : improvement_percent) {
        ss << "  " << metric << ": " << std::fixed << std::setprecision(1)
           << improvement << "%\n";
    }

    ss << "\n结论: " << conclusion << "\n";

    return ss.str();
}

ABTestingFramework::ABTestingFramework() {}

bool ABTestingFramework::createABTest(const ABTestConfig& config) {
    std::lock_guard<std::mutex> lock(test_mutex_);
    test_configs_[config.test_name] = config;
    test_groups_[config.test_name + "_baseline"] = std::vector<RealtimeMetrics>();
    test_groups_[config.test_name + "_variant"] = std::vector<RealtimeMetrics>();
    return true;
}

void ABTestingFramework::recordTestData(const std::string& test_name,
                                       const std::string& group_name,
                                       const RealtimeMetrics& metrics) {
    std::lock_guard<std::mutex> lock(test_mutex_);
    std::string full_group_name = test_name + "_" + group_name;
    if (test_groups_.find(full_group_name) != test_groups_.end()) {
        test_groups_[full_group_name].push_back(metrics);
    }
}

ABTestingFramework::ABTestResult ABTestingFramework::analyzeABTest(const std::string& test_name) {
    std::lock_guard<std::mutex> lock(test_mutex_);

    ABTestResult result;
    result.test_name = test_name;

    auto baseline_it = test_groups_.find(test_name + "_baseline");
    auto variant_it = test_groups_.find(test_name + "_variant");

    if (baseline_it == test_groups_.end() || variant_it == test_groups_.end()) {
        result.conclusion = "测试数据不足";
        return result;
    }

    const auto& baseline_data = baseline_it->second;
    const auto& variant_data = variant_it->second;

    if (baseline_data.empty() || variant_data.empty()) {
        result.conclusion = "测试数据为空";
        return result;
    }

    // 分析每个指标
    auto config_it = test_configs_.find(test_name);
    if (config_it == test_configs_.end()) {
        result.conclusion = "找不到测试配置";
        return result;
    }

    const auto& config = config_it->second;

    for (const std::string& metric : config.metrics) {
        std::vector<float> baseline_values, variant_values;

        // 提取指标值
        for (const auto& metrics : baseline_data) {
            if (metric == "fps") baseline_values.push_back(metrics.fps);
            else if (metric == "latency_ms") baseline_values.push_back(metrics.latency_ms);
            else if (metric == "gpu_utilization_percent")
                baseline_values.push_back(metrics.gpu_utilization_percent);
        }

        for (const auto& metrics : variant_data) {
            if (metric == "fps") variant_values.push_back(metrics.fps);
            else if (metric == "latency_ms") variant_values.push_back(metrics.latency_ms);
            else if (metric == "gpu_utilization_percent")
                variant_values.push_back(metrics.gpu_utilization_percent);
        }

        if (baseline_values.empty() || variant_values.empty()) continue;

        // 计算统计量
        float baseline_mean = calculateMean(baseline_values);
        float variant_mean = calculateMean(variant_values);
        float t_stat = calculateTStatistic(baseline_values, variant_values);
        int df = baseline_values.size() + variant_values.size() - 2;
        float p_value = calculatePValue(t_stat, df);
        float effect_size = calculateEffectSize(baseline_values, variant_values);

        result.baseline_metrics[metric] = baseline_mean;
        result.variant_metrics[metric] = variant_mean;
        result.improvement_percent[metric] = ((variant_mean - baseline_mean) / baseline_mean) * 100.0f;

        // 更新整体结果
        if (p_value < config.significance_level) {
            result.is_significant = true;
            result.p_value = std::min(result.p_value, p_value);
        }
        result.effect_size = std::max(result.effect_size, std::abs(effect_size));
    }

    // 生成结论
    if (result.is_significant) {
        if (result.improvement_percent["fps"] > 5.0f || result.improvement_percent["latency_ms"] < -5.0f) {
            result.conclusion = "变体显著优于基线，建议采用";
        } else {
            result.conclusion = "变体与基线有显著差异，但改进幅度较小";
        }
    } else {
        result.conclusion = "变体与基线无显著差异";
    }

    return result;
}

float ABTestingFramework::calculateMean(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

float ABTestingFramework::calculateStdDev(const std::vector<float>& values, float mean) {
    if (values.size() <= 1) return 0.0f;

    float variance = 0.0f;
    for (float value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= (values.size() - 1);
    return std::sqrt(variance);
}

float ABTestingFramework::calculateTStatistic(const std::vector<float>& group1,
                                             const std::vector<float>& group2) {
    float mean1 = calculateMean(group1);
    float mean2 = calculateMean(group2);
    float std1 = calculateStdDev(group1, mean1);
    float std2 = calculateStdDev(group2, mean2);

    float pooled_std = std::sqrt(((group1.size() - 1) * std1 * std1 +
                                 (group2.size() - 1) * std2 * std2) /
                                (group1.size() + group2.size() - 2));

    float se = pooled_std * std::sqrt(1.0f / group1.size() + 1.0f / group2.size());

    if (se == 0.0f) return 0.0f;
    return (mean1 - mean2) / se;
}

float ABTestingFramework::calculatePValue(float t_stat, int df) {
    // 简化的p值计算（实际应用中应使用更精确的统计函数）
    float abs_t = std::abs(t_stat);
    if (abs_t > 2.576f) return 0.01f;   // 99%置信度
    if (abs_t > 1.96f) return 0.05f;    // 95%置信度
    if (abs_t > 1.645f) return 0.10f;   // 90%置信度
    return 0.20f;
}

float ABTestingFramework::calculateEffectSize(const std::vector<float>& group1,
                                             const std::vector<float>& group2) {
    float mean1 = calculateMean(group1);
    float mean2 = calculateMean(group2);
    float std1 = calculateStdDev(group1, mean1);
    float std2 = calculateStdDev(group2, mean2);

    float pooled_std = std::sqrt((std1 * std1 + std2 * std2) / 2.0f);

    if (pooled_std == 0.0f) return 0.0f;
    return (mean1 - mean2) / pooled_std;
}

} // namespace yolo_acceleration
