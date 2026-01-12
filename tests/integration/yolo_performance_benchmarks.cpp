#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>
#include <sstream>

// YOLO system includes
#include "../../src/yolo/model_parser/onnx_parser.h"
#include "../../src/yolo/inference/batch_inference_manager.h"
#include "../../src/yolo/inference/multi_stream_inference.h"
#include "../../src/yolo/inference/operator_fusion.h"
#include "../../src/yolo/utils/gpu_memory_manager.h"

using namespace cuda_learning::yolo;
using namespace cuda_learning::operators;

class YOLOPerformanceBenchmark : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize GPU memory manager once for all tests
        if (!GlobalMemoryManager::initialize(
            8ULL * 1024 * 1024 * 1024,  // 8GB GPU memory
            4ULL * 1024 * 1024 * 1024   // 4GB tensor pool
        )) {
            throw std::runtime_error("Failed to initialize memory manager");
        }

        gpu_mem_mgr_ = GlobalMemoryManager::getGPUMemoryManager();

        // Create benchmark test data
        createBenchmarkData();
    }

    static void TearDownTestSuite() {
        GlobalMemoryManager::cleanup();
    }

    static void createBenchmarkData() {
        benchmark_images_.clear();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Create different sized test images for benchmarking
        std::vector<std::pair<int, int>> image_sizes = {
            {416, 416}, {512, 512}, {640, 640}, {832, 832}
        };

        for (const auto& size : image_sizes) {
            for (int i = 0; i < images_per_size_; ++i) {
                FloatTensor image({1, 3, size.first, size.second});
                image.uniform(0.0f, 1.0f);
                benchmark_images_[size].push_back(image);
            }
        }
    }

protected:
    static GPUMemoryManager* gpu_mem_mgr_;
    static std::map<std::pair<int, int>, std::vector<FloatTensor>> benchmark_images_;
    static constexpr int images_per_size_ = 20;

    // Performance tracking
    struct BenchmarkResult {
        std::string test_name;
        float avg_latency_ms;
        float throughput_qps;
        float memory_usage_mb;
        float gpu_utilization;
        std::string timestamp;
    };

    static std::vector<BenchmarkResult> benchmark_results_;

    void recordBenchmarkResult(const std::string& test_name,
                              float latency_ms,
                              float throughput_qps,
                              float memory_mb = 0.0f,
                              float gpu_util = 0.0f) {
        BenchmarkResult result;
        result.test_name = test_name;
        result.avg_latency_ms = latency_ms;
        result.throughput_qps = throughput_qps;
        result.memory_usage_mb = memory_mb;
        result.gpu_utilization = gpu_util;

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        result.timestamp = ss.str();

        benchmark_results_.push_back(result);
    }

    void saveBenchmarkResults() {
        std::ofstream results_file("tests/fixtures/yolo_benchmark_results.csv");
        if (results_file.is_open()) {
            // Write CSV header
            results_file << "Test Name,Avg Latency (ms),Throughput (QPS),Memory Usage (MB),GPU Utilization (%),Timestamp\n";

            // Write results
            for (const auto& result : benchmark_results_) {
                results_file << result.test_name << ","
                           << std::fixed << std::setprecision(2) << result.avg_latency_ms << ","
                           << std::fixed << std::setprecision(2) << result.throughput_qps << ","
                           << std::fixed << std::setprecision(2) << result.memory_usage_mb << ","
                           << std::fixed << std::setprecision(1) << result.gpu_utilization << ","
                           << result.timestamp << "\n";
            }
        }
    }
};

// Static member definitions
GPUMemoryManager* YOLOPerformanceBenchmark::gpu_mem_mgr_ = nullptr;
std::map<std::pair<int, int>, std::vector<FloatTensor>> YOLOPerformanceBenchmark::benchmark_images_;
std::vector<YOLOPerformanceBenchmark::BenchmarkResult> YOLOPerformanceBenchmark::benchmark_results_;

// Benchmark 1: Single Image Inference Latency
TEST_F(YOLOPerformanceBenchmark, SingleImageInferenceLatency) {
    ONNXParser parser;
    ASSERT_TRUE(parser.loadModel("yolo_model.onnx"));
    ASSERT_TRUE(parser.parseModel());

    // Test different image sizes
    for (const auto& size_images : benchmark_images_) {
        const auto& size = size_images.first;
        const auto& images = size_images.second;

        std::vector<float> latencies;

        // Warmup runs
        for (int i = 0; i < 5; ++i) {
            parser.inference(images[0]);
        }

        // Benchmark runs
        for (const auto& image : images) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto outputs = parser.inference(image);
            auto end_time = std::chrono::high_resolution_clock::now();

            float latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count() / 1000.0f;

            latencies.push_back(latency_ms);
        }

        // Calculate statistics
        float avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / latencies.size();
        float throughput = 1000.0f / avg_latency;

        std::string test_name = "SingleInference_" + std::to_string(size.first) + "x" + std::to_string(size.second);
        recordBenchmarkResult(test_name, avg_latency, throughput);

        // Performance assertions based on image size
        if (size.first <= 416) {
            EXPECT_LE(avg_latency, 15.0f) << "416x416 inference should be under 15ms";
        } else if (size.first <= 640) {
            EXPECT_LE(avg_latency, 25.0f) << "640x640 inference should be under 25ms";
        } else {
            EXPECT_LE(avg_latency, 40.0f) << "832x832 inference should be under 40ms";
        }

        std::cout << "Single inference " << size.first << "x" << size.second
                  << ": " << avg_latency << " ms, " << throughput << " QPS" << std::endl;
    }
}

// Benchmark 2: Batch Inference Throughput
TEST_F(YOLOPerformanceBenchmark, BatchInferenceThroughput) {
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16};

    for (int batch_size : batch_sizes) {
        BatchInferenceConfig config;
        config.max_batch_size = batch_size;
        config.input_height = 640;
        config.input_width = 640;
        config.input_channels = 3;
        config.enable_dynamic_batching = true;
        config.timeout_ms = 5.0f;

        BatchInferenceManager batch_manager(config, gpu_mem_mgr_);
        ASSERT_TRUE(batch_manager.start());

        const auto& test_images = benchmark_images_[{640, 640}];
        const int num_requests = 50;

        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto future = batch_manager.submitRequest("warmup_" + std::to_string(i), test_images[0]);
            future.get();
        }

        // Benchmark
        std::vector<std::future<std::vector<FloatTensor>>> futures;
        size_t initial_memory = gpu_mem_mgr_->getCurrentUsage();

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_requests; ++i) {
            size_t img_idx = i % test_images.size();
            std::string request_id = "batch_" + std::to_string(batch_size) + "_" + std::to_string(i);

            auto future = batch_manager.submitRequest(request_id, test_images[img_idx]);
            futures.push_back(std::move(future));
        }

        // Wait for results
        int completed = 0;
        for (auto& future : futures) {
            try {
                future.get();
                completed++;
            } catch (...) {}
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        size_t peak_memory = gpu_mem_mgr_->getPeakUsage();
        float memory_usage_mb = static_cast<float>(peak_memory - initial_memory) / (1024 * 1024);

        float avg_latency = static_cast<float>(total_time_ms) / completed;
        float throughput = (static_cast<float>(completed) * 1000.0f) / total_time_ms;

        std::string test_name = "BatchInference_BS" + std::to_string(batch_size);
        recordBenchmarkResult(test_name, avg_latency, throughput, memory_usage_mb);

        // Performance assertions
        if (batch_size > 1) {
            // Batch processing should improve throughput
            EXPECT_GT(throughput, 20.0f) << "Batch size " << batch_size << " should achieve >20 QPS";
        }

        std::cout << "Batch inference (BS=" << batch_size << "): "
                  << avg_latency << " ms, " << throughput << " QPS, "
                  << memory_usage_mb << " MB" << std::endl;

        batch_manager.stop();
    }
}

// Benchmark 3: Multi-Stream Concurrent Performance
TEST_F(YOLOPerformanceBenchmark, MultiStreamConcurrentPerformance) {
    std::vector<int> stream_counts = {1, 2, 4, 8};
    std::vector<MultiStreamInferenceSystem::LoadBalanceStrategy> strategies = {
        MultiStreamInferenceSystem::LoadBalanceStrategy::ROUND_ROBIN,
        MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED,
        MultiStreamInferenceSystem::LoadBalanceStrategy::SHORTEST_QUEUE
    };

    std::vector<std::string> strategy_names = {
        "RoundRobin", "LeastLoaded", "ShortestQueue"
    };

    for (size_t strategy_idx = 0; strategy_idx < strategies.size(); ++strategy_idx) {
        for (int num_streams : stream_counts) {
            MultiStreamInferenceSystem::SystemConfig config;
            config.num_streams = num_streams;
            config.strategy = strategies[strategy_idx];
            config.enable_stream_synchronization = true;

            MultiStreamInferenceSystem inference_system(gpu_mem_mgr_, config);
            ASSERT_TRUE(inference_system.initialize());

            const auto& test_images = benchmark_images_[{640, 640}];
            const int num_concurrent_requests = 100;

            // Warmup
            for (int i = 0; i < 10; ++i) {
                auto future = inference_system.submitInference("warmup_" + std::to_string(i), test_images[0]);
                future.get();
            }

            // Benchmark concurrent requests
            std::vector<std::future<std::vector<FloatTensor>>> futures;
            size_t initial_memory = gpu_mem_mgr_->getCurrentUsage();

            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < num_concurrent_requests; ++i) {
                size_t img_idx = i % test_images.size();
                std::string task_id = "concurrent_" + std::to_string(i);

                auto future = inference_system.submitInference(task_id, test_images[img_idx]);
                futures.push_back(std::move(future));
            }

            // Wait for all results
            int completed = 0;
            for (auto& future : futures) {
                try {
                    future.get();
                    completed++;
                } catch (...) {}
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

            size_t peak_memory = gpu_mem_mgr_->getPeakUsage();
            float memory_usage_mb = static_cast<float>(peak_memory - initial_memory) / (1024 * 1024);

            float avg_latency = static_cast<float>(total_time_ms) / completed;
            float throughput = (static_cast<float>(completed) * 1000.0f) / total_time_ms;

            // Get system statistics
            auto system_stats = inference_system.getSystemStats();
            float gpu_utilization = system_stats.avg_gpu_utilization * 100.0f;

            std::string test_name = "MultiStream_" + strategy_names[strategy_idx] +
                                   "_S" + std::to_string(num_streams);
            recordBenchmarkResult(test_name, avg_latency, throughput, memory_usage_mb, gpu_utilization);

            // Performance assertions
            if (num_streams > 1) {
                EXPECT_GT(throughput, 30.0f) << "Multi-stream should achieve >30 QPS";
                EXPECT_GT(gpu_utilization, 70.0f) << "GPU utilization should be >70%";
            }

            std::cout << "Multi-stream (" << strategy_names[strategy_idx]
                      << ", " << num_streams << " streams): "
                      << avg_latency << " ms, " << throughput << " QPS, "
                      << memory_usage_mb << " MB, " << gpu_utilization << "% GPU" << std::endl;

            inference_system.shutdown();
        }
    }
}

// Benchmark 4: Operator Fusion Performance Impact
TEST_F(YOLOPerformanceBenchmark, OperatorFusionPerformanceImpact) {
    std::vector<TensorShape> test_shapes = {
        {1, 64, 224, 224},
        {1, 128, 112, 112},
        {1, 256, 56, 56},
        {1, 512, 28, 28}
    };

    std::vector<ActivationType> activations = {
        ActivationType::RELU,
        ActivationType::LEAKY_RELU,
        ActivationType::SWISH
    };

    std::vector<std::string> activation_names = {
        "ReLU", "LeakyReLU", "Swish"
    };

    for (size_t act_idx = 0; act_idx < activations.size(); ++act_idx) {
        for (const auto& input_shape : test_shapes) {
            TensorShape weight_shape({input_shape[1] * 2, input_shape[1], 3, 3});

            FusionConfig fusion_config;
            fusion_config.kernel_h = 3;
            fusion_config.kernel_w = 3;
            fusion_config.stride_h = 1;
            fusion_config.stride_w = 1;
            fusion_config.pad_h = 1;
            fusion_config.pad_w = 1;
            fusion_config.activation = activations[act_idx];

            // Benchmark fusion performance
            const int num_iterations = 20;
            auto benchmark_result = FusionBenchmark::compareFusionPerformance(
                input_shape, weight_shape, fusion_config, num_iterations);

            if (benchmark_result.success) {
                float speedup = benchmark_result.speedup_ratio;
                float memory_saved = benchmark_result.memory_saved_mb;

                std::string test_name = "Fusion_" + activation_names[act_idx] +
                                       "_" + std::to_string(input_shape[1]) + "ch";

                // Record both fused and unfused performance
                recordBenchmarkResult(test_name + "_Unfused",
                                    benchmark_result.unfused_time_ms,
                                    1000.0f / benchmark_result.unfused_time_ms);

                recordBenchmarkResult(test_name + "_Fused",
                                    benchmark_result.fused_time_ms,
                                    1000.0f / benchmark_result.fused_time_ms,
                                    memory_saved);

                // Performance assertions
                EXPECT_GT(speedup, 1.1f) << "Fusion should provide >10% speedup";
                EXPECT_GT(memory_saved, 0) << "Fusion should save memory";

                std::cout << "Fusion " << activation_names[act_idx]
                          << " (" << input_shape[1] << " channels): "
                          << speedup << "x speedup, " << memory_saved << " MB saved" << std::endl;
            }
        }
    }
}

// Benchmark 5: Memory Usage Scaling
TEST_F(YOLOPerformanceBenchmark, MemoryUsageScaling) {
    std::vector<int> concurrent_requests = {1, 5, 10, 20, 50, 100};

    for (int num_requests : concurrent_requests) {
        MultiStreamInferenceSystem::SystemConfig config;
        config.num_streams = 4;
        config.strategy = MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED;

        MultiStreamInferenceSystem inference_system(gpu_mem_mgr_, config);
        ASSERT_TRUE(inference_system.initialize());

        const auto& test_images = benchmark_images_[{640, 640}];

        // Measure baseline memory
        size_t baseline_memory = gpu_mem_mgr_->getCurrentUsage();

        // Submit concurrent requests
        std::vector<std::future<std::vector<FloatTensor>>> futures;

        for (int i = 0; i < num_requests; ++i) {
            size_t img_idx = i % test_images.size();
            std::string task_id = "memory_test_" + std::to_string(i);

            auto future = inference_system.submitInference(task_id, test_images[img_idx]);
            futures.push_back(std::move(future));
        }

        // Measure peak memory during processing
        size_t peak_memory = gpu_mem_mgr_->getPeakUsage();

        // Wait for completion
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {}
        }

        // Measure final memory
        size_t final_memory = gpu_mem_mgr_->getCurrentUsage();

        float peak_usage_mb = static_cast<float>(peak_memory - baseline_memory) / (1024 * 1024);
        float final_usage_mb = static_cast<float>(final_memory - baseline_memory) / (1024 * 1024);
        float memory_per_request = peak_usage_mb / num_requests;

        std::string test_name = "MemoryScaling_" + std::to_string(num_requests) + "req";
        recordBenchmarkResult(test_name, 0, 0, peak_usage_mb);

        // Memory usage should scale reasonably
        EXPECT_LT(memory_per_request, 100.0f) << "Memory per request should be <100MB";
        EXPECT_LT(final_usage_mb, peak_usage_mb * 1.1f) << "Memory should be properly released";

        std::cout << "Memory scaling (" << num_requests << " requests): "
                  << peak_usage_mb << " MB peak, " << memory_per_request
                  << " MB/request, " << final_usage_mb << " MB final" << std::endl;

        inference_system.shutdown();

        // Force garbage collection
        gpu_mem_mgr_->garbageCollect();
    }
}

// Benchmark 6: Load Balance Strategy Comparison
TEST_F(YOLOPerformanceBenchmark, LoadBalanceStrategyComparison) {
    std::vector<MultiStreamInferenceSystem::LoadBalanceStrategy> strategies = {
        MultiStreamInferenceSystem::LoadBalanceStrategy::ROUND_ROBIN,
        MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED,
        MultiStreamInferenceSystem::LoadBalanceStrategy::SHORTEST_QUEUE,
        MultiStreamInferenceSystem::LoadBalanceStrategy::ADAPTIVE
    };

    std::vector<std::string> strategy_names = {
        "RoundRobin", "LeastLoaded", "ShortestQueue", "Adaptive"
    };

    const int num_streams = 4;
    const int num_requests = 100;
    const auto& test_images = benchmark_images_[{640, 640}];

    std::vector<float> strategy_throughputs;
    std::vector<float> strategy_latencies;

    for (size_t i = 0; i < strategies.size(); ++i) {
        MultiStreamInferenceSystem::SystemConfig config;
        config.num_streams = num_streams;
        config.strategy = strategies[i];
        config.enable_stream_synchronization = true;

        MultiStreamInferenceSystem inference_system(gpu_mem_mgr_, config);
        ASSERT_TRUE(inference_system.initialize());

        // Warmup
        for (int j = 0; j < 10; ++j) {
            auto future = inference_system.submitInference("warmup_" + std::to_string(j), test_images[0]);
            future.get();
        }

        // Reset statistics
        inference_system.resetAllStats();

        // Benchmark
        std::vector<std::future<std::vector<FloatTensor>>> futures;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < num_requests; ++j) {
            size_t img_idx = j % test_images.size();
            std::string task_id = "strategy_test_" + std::to_string(j);

            auto future = inference_system.submitInference(task_id, test_images[img_idx]);
            futures.push_back(std::move(future));
        }

        // Wait for results
        int completed = 0;
        for (auto& future : futures) {
            try {
                future.get();
                completed++;
            } catch (...) {}
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        float avg_latency = static_cast<float>(total_time_ms) / completed;
        float throughput = (static_cast<float>(completed) * 1000.0f) / total_time_ms;

        strategy_throughputs.push_back(throughput);
        strategy_latencies.push_back(avg_latency);

        // Get detailed statistics
        auto system_stats = inference_system.getSystemStats();

        std::string test_name = "LoadBalance_" + strategy_names[i];
        recordBenchmarkResult(test_name, avg_latency, throughput, 0,
                            system_stats.avg_gpu_utilization * 100.0f);

        std::cout << "Load balance " << strategy_names[i] << ": "
                  << avg_latency << " ms, " << throughput << " QPS, "
                  << "Load variance: " << system_stats.load_balance_variance << std::endl;

        inference_system.shutdown();
    }

    // Find best performing strategy
    auto best_throughput_idx = std::max_element(strategy_throughputs.begin(),
                                               strategy_throughputs.end()) - strategy_throughputs.begin();

    std::cout << "Best performing strategy: " << strategy_names[best_throughput_idx]
              << " (" << strategy_throughputs[best_throughput_idx] << " QPS)" << std::endl;

    // Verify that adaptive or least_loaded performs well
    EXPECT_TRUE(best_throughput_idx == 1 || best_throughput_idx == 3)
        << "Least loaded or adaptive strategy should perform best";
}

// Test to save all benchmark results
TEST_F(YOLOPerformanceBenchmark, SaveBenchmarkResults) {
    // This test runs last to save all accumulated results
    saveBenchmarkResults();

    std::cout << "\nBenchmark Summary:" << std::endl;
    std::cout << "Total tests run: " << benchmark_results_.size() << std::endl;

    // Calculate overall statistics
    float total_throughput = 0;
    float total_memory = 0;
    int throughput_count = 0;
    int memory_count = 0;

    for (const auto& result : benchmark_results_) {
        if (result.throughput_qps > 0) {
            total_throughput += result.throughput_qps;
            throughput_count++;
        }
        if (result.memory_usage_mb > 0) {
            total_memory += result.memory_usage_mb;
            memory_count++;
        }
    }

    if (throughput_count > 0) {
        std::cout << "Average throughput: " << (total_throughput / throughput_count) << " QPS" << std::endl;
    }
    if (memory_count > 0) {
        std::cout << "Average memory usage: " << (total_memory / memory_count) << " MB" << std::endl;
    }

    std::cout << "Results saved to: tests/fixtures/yolo_benchmark_results.csv" << std::endl;
}

// Google Benchmark integration for more detailed profiling
static void BM_SingleInference640x640(benchmark::State& state) {
    // This would integrate with Google Benchmark for micro-benchmarking
    ONNXParser parser;
    parser.loadModel("yolo_model.onnx");
    parser.parseModel();

    FloatTensor input({1, 3, 640, 640});
    input.uniform(0.0f, 1.0f);

    for (auto _ : state) {
        auto outputs = parser.inference(input);
        benchmark::DoNotOptimize(outputs);
    }

    state.SetItemsProcessed(state.iterations());
}

// Register Google Benchmark
BENCHMARK(BM_SingleInference640x640)->Unit(benchmark::kMillisecond);

// Main benchmark runner
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize CUDA context
    cudaSetDevice(0);

    // Run Google Test benchmarks
    int test_result = RUN_ALL_TESTS();

    // Run Google Benchmark if requested
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        ::benchmark::Initialize(&argc, argv);
        ::benchmark::RunSpecifiedBenchmarks();
    }

    return test_result;
}
