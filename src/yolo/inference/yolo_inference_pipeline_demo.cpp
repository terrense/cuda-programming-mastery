#include "batch_inference_manager.h"
#include "operator_fusion.h"
#include "multi_stream_inference.h"
#include "../utils/gpu_memory_manager.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace cuda_learning::yolo;
using namespace cuda_learning::operators;

// 演示批处理推理内存池管理
void demonstrateBatchInferenceManager() {
    std::cout << "\n=== Batch Inference Manager Demonstration ===" << std::endl;

    // 初始化全局内存管理器
    if (!GlobalMemoryManager::initialize(4ULL * 1024 * 1024 * 1024, // 4GB GPU内存
                                        2ULL * 1024 * 1024 * 1024)) { // 2GB张量池
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return;
    }

    auto* gpu_mem_mgr = GlobalMemoryManager::getGPUMemoryManager();

    // 配置批处理推理
    BatchInferenceConfig config;
    config.max_batch_size = 8;
    config.input_height = 640;
    config.input_width = 640;
    config.input_channels = 3;
    config.enable_dynamic_batching = true;
    config.timeout_ms = 10.0f;

    // 创建批处理管理器
    BatchInferenceManager batch_manager(config, gpu_mem_mgr);

    if (!batch_manager.start()) {
        std::cerr << "Failed to start batch manager" << std::endl;
        return;
    }

    std::cout << "Batch manager started successfully" << std::endl;

    // 创建测试输入数据
    std::vector<std::future<std::vector<FloatTensor>>> futures;
    const int num_requests = 20;

    std::cout << "Submitting " << num_requests << " inference requests..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_requests; ++i) {
        // 创建输入张量
        FloatTensor input({1, config.input_channels, config.input_height, config.input_width});
        input.uniform(0.0f, 1.0f);  // 随机填充

        std::string request_id = "request_" + std::to_string(i);
        auto future = batch_manager.submitRequest(request_id, input);
        futures.push_back(std::move(future));

        // 模拟请求间隔
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // 等待所有结果
    std::cout << "Waiting for results..." << std::endl;
    int successful_requests = 0;

    for (auto& future : futures) {
        try {
            auto result = future.get();
            successful_requests++;
        } catch (const std::exception& e) {
            std::cerr << "Request failed: " << e.what() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Batch inference completed:" << std::endl;
    std::cout << "  Successful requests: " << successful_requests << "/" << num_requests << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Average latency: " << (float)duration.count() / num_requests << " ms/request" << std::endl;

    // 打印统计信息
    batch_manager.printStats();

    batch_manager.stop();
    GlobalMemoryManager::cleanup();
}

// 演示算子融合优化
void demonstrateOperatorFusion() {
    std::cout << "\n=== Operator Fusion Demonstration ===" << std::endl;

    // 创建融合优化器
    OperatorFusionOptimizer optimizer;
    optimizer.printFusionStats();

    // 测试融合模式检测
    std::vector<std::string> operator_sequence = {
        "Conv", "BatchNormalization", "Relu",
        "Conv", "BatchNormalization", "LeakyRelu",
        "MaxPool", "Conv", "BatchNormalization", "Relu"
    };

    std::cout << "Original operator sequence:" << std::endl;
    for (size_t i = 0; i < operator_sequence.size(); ++i) {
        std::cout << "  " << i << ": " << operator_sequence[i] << std::endl;
    }

    // 检测可融合序列
    auto fusion_ranges = optimizer.detectFusableSequences(operator_sequence);

    std::cout << "Detected " << fusion_ranges.size() << " fusable sequences" << std::endl;

    // 创建融合算子进行性能测试
    std::cout << "\nTesting fused Conv+BN+ReLU performance..." << std::endl;

    TensorShape input_shape({1, 64, 224, 224});   // 输入形状
    TensorShape weight_shape({128, 64, 3, 3});    // 权重形状

    FusionConfig fusion_config;
    fusion_config.kernel_h = 3;
    fusion_config.kernel_w = 3;
    fusion_config.stride_h = 1;
    fusion_config.stride_w = 1;
    fusion_config.pad_h = 1;
    fusion_config.pad_w = 1;
    fusion_config.activation = ActivationType::RELU;

    // 性能基准测试
    auto benchmark_result = FusionBenchmark::compareFusionPerformance(
        input_shape, weight_shape, fusion_config, 50);

    if (benchmark_result.success) {
        std::cout << "Fusion benchmark results:" << std::endl;
        std::cout << "  Unfused time: " << benchmark_result.unfused_time_ms << " ms" << std::endl;
        std::cout << "  Fused time: " << benchmark_result.fused_time_ms << " ms" << std::endl;
        std::cout << "  Speedup ratio: " << benchmark_result.speedup_ratio << "x" << std::endl;
        std::cout << "  Memory saved: " << benchmark_result.memory_saved_mb << " MB" << std::endl;
    } else {
        std::cout << "Fusion benchmark failed: " << benchmark_result.error_message << std::endl;
    }

    // 测试不同激活函数的融合性能
    std::cout << "\nTesting different activation function fusions..." << std::endl;
    auto activation_benchmarks = FusionBenchmark::benchmarkActivationFusions(
        input_shape, weight_shape, 20);

    for (size_t i = 0; i < activation_benchmarks.size(); ++i) {
        const auto& result = activation_benchmarks[i];
        std::string activation_name;

        switch (static_cast<ActivationType>(i)) {
            case ActivationType::RELU: activation_name = "ReLU"; break;
            case ActivationType::LEAKY_RELU: activation_name = "LeakyReLU"; break;
            case ActivationType::SWISH: activation_name = "Swish"; break;
            default: activation_name = "Unknown"; break;
        }

        if (result.success) {
            std::cout << "  " << activation_name << " fusion speedup: "
                      << result.speedup_ratio << "x" << std::endl;
        }
    }
}

// 演示多流并行推理系统
void demonstrateMultiStreamInference() {
    std::cout << "\n=== Multi-Stream Inference Demonstration ===" << std::endl;

    // 初始化全局内存管理器
    if (!GlobalMemoryManager::initialize(8ULL * 1024 * 1024 * 1024, // 8GB GPU内存
                                        4ULL * 1024 * 1024 * 1024)) { // 4GB张量池
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return;
    }

    auto* gpu_mem_mgr = GlobalMemoryManager::getGPUMemoryManager();

    // 配置多流系统
    MultiStreamInferenceSystem::SystemConfig system_config;
    system_config.num_streams = 4;
    system_config.strategy = MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED;
    system_config.enable_stream_synchronization = true;

    // 创建多流推理系统
    MultiStreamInferenceSystem inference_system(gpu_mem_mgr, system_config);

    if (!inference_system.initialize()) {
        std::cerr << "Failed to initialize multi-stream system" << std::endl;
        return;
    }

    std::cout << "Multi-stream system initialized with " << inference_system.getNumStreams()
              << " streams" << std::endl;

    // 创建性能监控器
    InferencePerformanceMonitor::MonitorConfig monitor_config;
    monitor_config.monitor_interval_ms = 500.0f;
    monitor_config.alert_threshold_latency_ms = 50.0f;
    monitor_config.alert_threshold_utilization = 0.8f;

    InferencePerformanceMonitor monitor(&inference_system, monitor_config);
    monitor.startMonitoring();

    // 提交大量并发推理请求
    const int num_concurrent_requests = 100;
    std::vector<std::future<std::vector<FloatTensor>>> futures;
    futures.reserve(num_concurrent_requests);

    std::cout << "Submitting " << num_concurrent_requests << " concurrent requests..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(416, 640);  // 随机输入尺寸

    for (int i = 0; i < num_concurrent_requests; ++i) {
        // 创建随机大小的输入张量
        int input_size = size_dist(gen);
        FloatTensor input({1, 3, input_size, input_size});
        input.uniform(0.0f, 1.0f);

        std::string task_id = "task_" + std::to_string(i);

        try {
            auto future = inference_system.submitInference(task_id, input);
            futures.push_back(std::move(future));
        } catch (const std::exception& e) {
            std::cerr << "Failed to submit task " << i << ": " << e.what() << std::endl;
        }
    }

    // 等待所有结果
    std::cout << "Waiting for all results..." << std::endl;
    int successful_tasks = 0;
    int failed_tasks = 0;

    for (auto& future : futures) {
        try {
            auto result = future.get();
            successful_tasks++;
        } catch (const std::exception& e) {
            failed_tasks++;
            // std::cerr << "Task failed: " << e.what() << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Multi-stream inference completed:" << std::endl;
    std::cout << "  Successful tasks: " << successful_tasks << std::endl;
    std::cout << "  Failed tasks: " << failed_tasks << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << (float)successful_tasks / (duration.count() / 1000.0f)
              << " QPS" << std::endl;

    // 等待一段时间让监控器收集数据
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 打印系统统计信息
    inference_system.printAllStreamStats();

    // 打印性能监控报告
    monitor.generatePerformanceReport();

    // 测试负载均衡策略
    std::cout << "\nTesting different load balance strategies..." << std::endl;

    std::vector<MultiStreamInferenceSystem::LoadBalanceStrategy> strategies = {
        MultiStreamInferenceSystem::LoadBalanceStrategy::ROUND_ROBIN,
        MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED,
        MultiStreamInferenceSystem::LoadBalanceStrategy::SHORTEST_QUEUE,
        MultiStreamInferenceSystem::LoadBalanceStrategy::ADAPTIVE
    };

    std::vector<std::string> strategy_names = {
        "Round Robin", "Least Loaded", "Shortest Queue", "Adaptive"
    };

    for (size_t i = 0; i < strategies.size(); ++i) {
        inference_system.setLoadBalanceStrategy(strategies[i]);
        inference_system.resetAllStats();

        std::cout << "Testing " << strategy_names[i] << " strategy..." << std::endl;

        // 提交少量测试请求
        std::vector<std::future<std::vector<FloatTensor>>> test_futures;
        for (int j = 0; j < 20; ++j) {
            FloatTensor input({1, 3, 640, 640});
            input.uniform(0.0f, 1.0f);

            std::string task_id = "test_" + std::to_string(j);
            auto future = inference_system.submitInference(task_id, input);
            test_futures.push_back(std::move(future));
        }

        // 等待结果
        for (auto& future : test_futures) {
            try {
                future.get();
            } catch (...) {}
        }

        // 打印简要统计
        auto stats = inference_system.getSystemStats();
        std::cout << "  Completed: " << stats.completed_tasks
                  << ", Avg latency: " << stats.avg_system_latency_ms << " ms" << std::endl;
    }

    // 清理资源
    monitor.stopMonitoring();
    inference_system.shutdown();
    GlobalMemoryManager::cleanup();
}

// 综合性能对比测试
void comprehensivePerformanceComparison() {
    std::cout << "\n=== Comprehensive Performance Comparison ===" << std::endl;

    // 初始化内存管理器
    if (!GlobalMemoryManager::initialize(8ULL * 1024 * 1024 * 1024,
                                        4ULL * 1024 * 1024 * 1024)) {
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return;
    }

    auto* gpu_mem_mgr = GlobalMemoryManager::getGPUMemoryManager();

    // 测试配置
    const int num_test_requests = 50;
    const std::vector<int> batch_sizes = {1, 2, 4, 8};
    const std::vector<int> num_streams = {1, 2, 4, 8};

    std::cout << "Testing different configurations..." << std::endl;
    std::cout << std::setw(12) << "Batch Size"
              << std::setw(12) << "Streams"
              << std::setw(15) << "Throughput(QPS)"
              << std::setw(15) << "Avg Latency(ms)"
              << std::setw(15) << "Memory Usage(MB)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int batch_size : batch_sizes) {
        for (int stream_count : num_streams) {
            // 配置系统
            MultiStreamInferenceSystem::SystemConfig config;
            config.num_streams = stream_count;
            config.strategy = MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED;

            MultiStreamInferenceSystem system(gpu_mem_mgr, config);

            if (!system.initialize()) {
                std::cout << "Failed to initialize system for batch_size="
                          << batch_size << ", streams=" << stream_count << std::endl;
                continue;
            }

            // 执行测试
            std::vector<std::future<std::vector<FloatTensor>>> futures;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < num_test_requests; ++i) {
                FloatTensor input({batch_size, 3, 640, 640});
                input.uniform(0.0f, 1.0f);

                std::string task_id = "perf_test_" + std::to_string(i);
                auto future = system.submitInference(task_id, input);
                futures.push_back(std::move(future));
            }

            // 等待结果
            int successful = 0;
            for (auto& future : futures) {
                try {
                    future.get();
                    successful++;
                } catch (...) {}
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // 计算性能指标
            float throughput = (float)successful / (duration.count() / 1000.0f);
            float avg_latency = (float)duration.count() / successful;
            size_t memory_usage = gpu_mem_mgr->getCurrentUsage() / (1024 * 1024);

            std::cout << std::setw(12) << batch_size
                      << std::setw(12) << stream_count
                      << std::setw(15) << std::fixed << std::setprecision(2) << throughput
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_latency
                      << std::setw(15) << memory_usage << std::endl;

            system.shutdown();
        }
    }

    GlobalMemoryManager::cleanup();
}

int main() {
    std::cout << "YOLO Inference Pipeline Optimization Demo Starting..." << std::endl;

    try {
        // 演示各个优化组件
        demonstrateBatchInferenceManager();
        demonstrateOperatorFusion();
        demonstrateMultiStreamInference();
        comprehensivePerformanceComparison();

        std::cout << "\n=== All Demonstrations Completed Successfully ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
