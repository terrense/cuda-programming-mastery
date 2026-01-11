#include "bottleneck_analyzer.h"
#include "realtime_metrics_collector.h"
#include <iostream>
#include <memory>

using namespace yolo_acceleration;

int main() {
    std::cout << "=== YOLO Bottleneck Analyzer Test ===" << std::endl;

    try {
        // 创建实时指标收集器
        auto metrics_collector = std::make_shared<RealtimeMetricsCollector>();

        // 创建瓶颈分析器配置
        AdvancedBottleneckAnalyzer::AnalysisConfig config;
        config.analysis_window_size = 50;
        config.bottleneck_threshold = 0.2f;
        config.enable_predictive_analysis = true;
        config.enable_root_cause_analysis = true;

        std::cout << "Creating AdvancedBottleneckAnalyzer..." << std::endl;

        // 创建高级瓶颈分析器
        AdvancedBottleneckAnalyzer analyzer(metrics_collector, config);

        std::cout << "Analyzer created successfully!" << std::endl;

        // 执行综合分析
        std::cout << "Performing comprehensive analysis..." << std::endl;
        auto analysis = analyzer.performComprehensiveAnalysis();

        std::cout << "Analysis completed!" << std::endl;
        std::cout << "Primary bottleneck: " << analysis.base_analysis.primary_bottleneck << std::endl;
        std::cout << "Severity level: " << analysis.base_analysis.severity_level << std::endl;

        // 生成智能优化建议
        std::cout << "Generating optimization suggestions..." << std::endl;
        auto suggestions = analyzer.generateIntelligentSuggestions(analysis);

        std::cout << "Generated " << suggestions.size() << " optimization suggestions" << std::endl;

        // 创建优化建议执行器
        std::cout << "Creating OptimizationSuggestionExecutor..." << std::endl;
        OptimizationSuggestionExecutor executor(metrics_collector, nullptr);

        std::cout << "Executor created successfully!" << std::endl;

        // 创建性能基准测试系统
        std::cout << "Creating PerformanceBenchmarkSystem..." << std::endl;
        PerformanceBenchmarkSystem benchmark_system(metrics_collector);

        std::cout << "Benchmark system created successfully!" << std::endl;

        // 运行单个基准测试
        std::cout << "Running single benchmark test..." << std::endl;
        auto benchmark_result = benchmark_system.runSingleBenchmark("test_scenario", "yolov5s_batch1");

        std::cout << "Benchmark completed!" << std::endl;
        std::cout << "Test: " << benchmark_result.test_name << std::endl;
        std::cout << "Configuration: " << benchmark_result.configuration << std::endl;
        std::cout << "Average throughput: " << benchmark_result.average_metrics.throughput_fps << " FPS" << std::endl;

        // 分析性能趋势
        std::cout << "Analyzing performance trends..." << std::endl;
        auto trend_analysis = analyzer.analyzePerformanceTrends(20);
        std::cout << "Trend Analysis Result:" << std::endl;
        std::cout << trend_analysis << std::endl;

        // 预测未来瓶颈
        std::cout << "Predicting future bottlenecks..." << std::endl;
        auto future_bottlenecks = analyzer.predictFutureBottlenecks(30);

        std::cout << "Future bottleneck predictions:" << std::endl;
        for (const auto& [component, probability] : future_bottlenecks) {
            std::cout << "  " << component << ": " << (probability * 100.0f) << "%" << std::endl;
        }

        std::cout << "\n=== All tests completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
