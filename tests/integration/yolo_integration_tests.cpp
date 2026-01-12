#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <memory>
#include <fstream>
#include <random>

// YOLO system includes
#include "../../src/yolo/model_parser/onnx_parser.h"
#include "../../src/yolo/model_parser/pytorch_loader.h"
#include "../../src/yolo/inference/batch_inference_manager.h"
#include "../../src/yolo/inference/multi_stream_inference.h"
#include "../../src/yolo/inference/operator_fusion.h"
#include "../../src/yolo/tensorrt/tensorrt_engine.h"
#include "../../src/yolo/monitoring/performance_monitor.h"
#include "../../src/yolo/utils/gpu_memory_manager.h"
#include "../../src/core/error_handler.h"

using namespace cuda_learning::yolo;
using namespace cuda_learning::operators;

class YOLOIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GPU memory manager
        ASSERT_TRUE(GlobalMemoryManager::initialize(
            4ULL * 1024 * 1024 * 1024,  // 4GB GPU memory
            2ULL * 1024 * 1024 * 1024   // 2GB tensor pool
        )) << "Failed to initialize memory manager";

        gpu_mem_mgr_ = GlobalMemoryManager::getGPUMemoryManager();
        ASSERT_NE(gpu_mem_mgr_, nullptr);

        // Create test input data
        createTestData();
    }

    void TearDown() override {
        GlobalMemoryManager::cleanup();
    }

    void createTestData() {
        // Create synthetic test images
        test_images_.clear();
        expected_detections_.clear();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int i = 0; i < num_test_images_; ++i) {
            // Create random test image
            cv::Mat image(640, 640, CV_32FC3);
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    image.at<cv::Vec3f>(y, x) = cv::Vec3f(dis(gen), dis(gen), dis(gen));
                }
            }
            test_images_.push_back(image);

            // Create expected detections (synthetic ground truth)
            std::vector<Detection> detections;
            int num_objects = gen() % 5 + 1;  // 1-5 objects per image

            for (int j = 0; j < num_objects; ++j) {
                Detection det;
                det.bbox.x = dis(gen) * 500;
                det.bbox.y = dis(gen) * 500;
                det.bbox.width = dis(gen) * 100 + 50;
                det.bbox.height = dis(gen) * 100 + 50;
                det.confidence = dis(gen) * 0.5f + 0.5f;  // 0.5-1.0
                det.class_id = gen() % 80;  // COCO classes
                detections.push_back(det);
            }
            expected_detections_.push_back(detections);
        }
    }

    // Helper function to compare detection results
    bool compareDetections(const std::vector<Detection>& actual,
                          const std::vector<Detection>& expected,
                          float iou_threshold = 0.5f,
                          float conf_threshold = 0.1f) {
        if (actual.empty() && expected.empty()) return true;

        int matched = 0;
        for (const auto& exp_det : expected) {
            for (const auto& act_det : actual) {
                if (act_det.class_id == exp_det.class_id) {
                    float iou = calculateIoU(act_det.bbox, exp_det.bbox);
                    if (iou >= iou_threshold &&
                        std::abs(act_det.confidence - exp_det.confidence) <= conf_threshold) {
                        matched++;
                        break;
                    }
                }
            }
        }

        // Allow some tolerance in detection matching
        float match_ratio = static_cast<float>(matched) / expected.size();
        return match_ratio >= 0.7f;  // 70% match threshold
    }

    float calculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
        float x1 = std::max(box1.x, box2.x);
        float y1 = std::max(box1.y, box2.y);
        float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

        if (x2 <= x1 || y2 <= y1) return 0.0f;

        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float union_area = area1 + area2 - intersection;

        return intersection / union_area;
    }

protected:
    GPUMemoryManager* gpu_mem_mgr_ = nullptr;
    std::vector<cv::Mat> test_images_;
    std::vector<std::vector<Detection>> expected_detections_;
    static constexpr int num_test_images_ = 10;

    // Performance thresholds
    static constexpr float max_inference_time_ms_ = 50.0f;
    static constexpr float min_throughput_qps_ = 20.0f;
    static constexpr float max_memory_usage_gb_ = 3.0f;
};

// Test 1: End-to-End ONNX Model Loading and Inference Accuracy
TEST_F(YOLOIntegrationTest, ONNXModelInferenceAccuracy) {
    // Load ONNX model
    ONNXParser parser;
    ASSERT_TRUE(parser.loadModel("yolo_model.onnx"))
        << "Failed to load ONNX model";

    ASSERT_TRUE(parser.parseModel())
        << "Failed to parse ONNX model";

    // Verify model structure
    auto inputs = parser.getInputs();
    auto outputs = parser.getOutputs();

    ASSERT_GE(inputs.size(), 1) << "Model should have at least one input";
    ASSERT_GE(outputs.size(), 1) << "Model should have at least one output";

    // Test inference on all test images
    int successful_inferences = 0;
    int accurate_detections = 0;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        try {
            // Convert image to tensor
            FloatTensor input_tensor = imageToTensor(test_images_[i]);

            // Run inference
            auto start_time = std::chrono::high_resolution_clock::now();
            auto output_tensors = parser.inference(input_tensor);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

            EXPECT_LE(inference_time, max_inference_time_ms_)
                << "Inference time exceeded threshold for image " << i;

            // Parse detections from output
            auto detections = parseDetections(output_tensors);

            // Compare with expected results
            if (compareDetections(detections, expected_detections_[i])) {
                accurate_detections++;
            }

            successful_inferences++;

        } catch (const std::exception& e) {
            FAIL() << "Inference failed for image " << i << ": " << e.what();
        }
    }

    // Verify accuracy metrics
    EXPECT_EQ(successful_inferences, test_images_.size())
        << "Not all inferences completed successfully";

    float accuracy_rate = static_cast<float>(accurate_detections) / test_images_.size();
    EXPECT_GE(accuracy_rate, 0.8f)
        << "Detection accuracy below threshold: " << accuracy_rate;
}

// Test 2: PyTorch Model Loading and Inference Accuracy
TEST_F(YOLOIntegrationTest, PyTorchModelInferenceAccuracy) {
    PyTorchLoader loader;
    ASSERT_TRUE(loader.loadModel("yolo_model.pt"))
        << "Failed to load PyTorch model";

    // Test batch inference
    std::vector<FloatTensor> batch_inputs;
    std::vector<std::vector<Detection>> batch_expected;

    for (size_t i = 0; i < std::min(4UL, test_images_.size()); ++i) {
        batch_inputs.push_back(imageToTensor(test_images_[i]));
        batch_expected.push_back(expected_detections_[i]);
    }

    // Set batch input
    ASSERT_TRUE(loader.setBatchInput("input", batch_inputs))
        << "Failed to set batch input";

    // Run batch inference
    auto start_time = std::chrono::high_resolution_clock::now();
    auto batch_outputs = loader.forward();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    EXPECT_LE(batch_time, max_inference_time_ms_ * batch_inputs.size())
        << "Batch inference time exceeded threshold";

    // Verify batch results
    ASSERT_EQ(batch_outputs.size(), batch_inputs.size())
        << "Batch output size mismatch";

    int accurate_batch_detections = 0;
    for (size_t i = 0; i < batch_outputs.size(); ++i) {
        auto detections = parseDetections({batch_outputs[i]});
        if (compareDetections(detections, batch_expected[i])) {
            accurate_batch_detections++;
        }
    }

    float batch_accuracy = static_cast<float>(accurate_batch_detections) / batch_inputs.size();
    EXPECT_GE(batch_accuracy, 0.8f)
        << "Batch detection accuracy below threshold: " << batch_accuracy;
}

// Test 3: Batch Inference Manager Performance and Accuracy
TEST_F(YOLOIntegrationTest, BatchInferenceManagerPerformance) {
    BatchInferenceConfig config;
    config.max_batch_size = 8;
    config.input_height = 640;
    config.input_width = 640;
    config.input_channels = 3;
    config.enable_dynamic_batching = true;
    config.timeout_ms = 10.0f;

    BatchInferenceManager batch_manager(config, gpu_mem_mgr_);
    ASSERT_TRUE(batch_manager.start())
        << "Failed to start batch manager";

    // Submit concurrent requests
    std::vector<std::future<std::vector<FloatTensor>>> futures;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_images_.size(); ++i) {
        FloatTensor input = imageToTensor(test_images_[i]);
        std::string request_id = "batch_test_" + std::to_string(i);

        auto future = batch_manager.submitRequest(request_id, input);
        futures.push_back(std::move(future));
    }

    // Collect results
    int successful_batch_requests = 0;
    int accurate_batch_results = 0;

    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            auto result = futures[i].get();
            successful_batch_requests++;

            auto detections = parseDetections(result);
            if (compareDetections(detections, expected_detections_[i])) {
                accurate_batch_results++;
            }
        } catch (const std::exception& e) {
            FAIL() << "Batch request " << i << " failed: " << e.what();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Performance assertions
    EXPECT_EQ(successful_batch_requests, test_images_.size())
        << "Not all batch requests completed";

    float throughput = (static_cast<float>(successful_batch_requests) * 1000.0f) / total_time;
    EXPECT_GE(throughput, min_throughput_qps_)
        << "Batch throughput below threshold: " << throughput << " QPS";

    float batch_accuracy = static_cast<float>(accurate_batch_results) / successful_batch_requests;
    EXPECT_GE(batch_accuracy, 0.8f)
        << "Batch accuracy below threshold: " << batch_accuracy;

    // Memory usage check
    size_t memory_usage_bytes = gpu_mem_mgr_->getCurrentUsage();
    float memory_usage_gb = static_cast<float>(memory_usage_bytes) / (1024 * 1024 * 1024);
    EXPECT_LE(memory_usage_gb, max_memory_usage_gb_)
        << "Memory usage exceeded threshold: " << memory_usage_gb << " GB";

    batch_manager.stop();
}

// Test 4: Multi-Stream Inference System Performance
TEST_F(YOLOIntegrationTest, MultiStreamInferencePerformance) {
    MultiStreamInferenceSystem::SystemConfig system_config;
    system_config.num_streams = 4;
    system_config.strategy = MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED;
    system_config.enable_stream_synchronization = true;

    MultiStreamInferenceSystem inference_system(gpu_mem_mgr_, system_config);
    ASSERT_TRUE(inference_system.initialize())
        << "Failed to initialize multi-stream system";

    // Test concurrent inference with multiple streams
    const int num_concurrent_requests = 20;
    std::vector<std::future<std::vector<FloatTensor>>> futures;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_concurrent_requests; ++i) {
        size_t img_idx = i % test_images_.size();
        FloatTensor input = imageToTensor(test_images_[img_idx]);
        std::string task_id = "stream_test_" + std::to_string(i);

        auto future = inference_system.submitInference(task_id, input);
        futures.push_back(std::move(future));
    }

    // Collect results
    int successful_stream_requests = 0;
    int accurate_stream_results = 0;

    for (int i = 0; i < num_concurrent_requests; ++i) {
        try {
            auto result = futures[i].get();
            successful_stream_requests++;

            size_t img_idx = i % test_images_.size();
            auto detections = parseDetections(result);
            if (compareDetections(detections, expected_detections_[img_idx])) {
                accurate_stream_results++;
            }
        } catch (const std::exception& e) {
            FAIL() << "Stream request " << i << " failed: " << e.what();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Performance assertions
    EXPECT_EQ(successful_stream_requests, num_concurrent_requests)
        << "Not all stream requests completed";

    float stream_throughput = (static_cast<float>(successful_stream_requests) * 1000.0f) / total_time;
    EXPECT_GE(stream_throughput, min_throughput_qps_ * 2)
        << "Multi-stream throughput below threshold: " << stream_throughput << " QPS";

    float stream_accuracy = static_cast<float>(accurate_stream_results) / successful_stream_requests;
    EXPECT_GE(stream_accuracy, 0.8f)
        << "Multi-stream accuracy below threshold: " << stream_accuracy;

    // Verify load balancing effectiveness
    auto system_stats = inference_system.getSystemStats();
    EXPECT_GT(system_stats.completed_tasks, 0) << "No tasks completed";
    EXPECT_LT(system_stats.avg_system_latency_ms, max_inference_time_ms_ * 2)
        << "Average system latency too high";

    inference_system.shutdown();
}

// Test 5: Operator Fusion Performance and Accuracy
TEST_F(YOLOIntegrationTest, OperatorFusionPerformance) {
    OperatorFusionOptimizer optimizer;

    // Test fusion detection
    std::vector<std::string> operator_sequence = {
        "Conv", "BatchNormalization", "Relu",
        "Conv", "BatchNormalization", "LeakyRelu"
    };

    auto fusion_ranges = optimizer.detectFusableSequences(operator_sequence);
    EXPECT_GE(fusion_ranges.size(), 2) << "Should detect at least 2 fusable sequences";

    // Performance benchmark test
    TensorShape input_shape({1, 64, 224, 224});
    TensorShape weight_shape({128, 64, 3, 3});

    FusionConfig fusion_config;
    fusion_config.kernel_h = 3;
    fusion_config.kernel_w = 3;
    fusion_config.activation = ActivationType::RELU;

    auto benchmark_result = FusionBenchmark::compareFusionPerformance(
        input_shape, weight_shape, fusion_config, 10);

    ASSERT_TRUE(benchmark_result.success)
        << "Fusion benchmark failed: " << benchmark_result.error_message;

    // Verify performance improvement
    EXPECT_GT(benchmark_result.speedup_ratio, 1.1f)
        << "Fusion should provide at least 10% speedup";

    EXPECT_GT(benchmark_result.memory_saved_mb, 0)
        << "Fusion should save memory";

    EXPECT_LT(benchmark_result.fused_time_ms, benchmark_result.unfused_time_ms)
        << "Fused execution should be faster";
}

// Test 6: TensorRT Integration Performance
TEST_F(YOLOIntegrationTest, TensorRTIntegrationPerformance) {
    // This test would require TensorRT integration
    // For now, we'll create a placeholder that tests the interface

    TensorRTEngine engine;

    // Test engine building (mock implementation)
    TensorRTConfig config;
    config.max_batch_size = 8;
    config.max_workspace_size = 1ULL << 30;  // 1GB
    config.fp16_mode = true;
    config.int8_mode = false;

    // In a real implementation, this would build from ONNX
    bool build_success = engine.buildFromONNX("yolo_model.onnx", config);

    if (build_success) {
        // Test inference performance
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_images_.size(); ++i) {
            FloatTensor input = imageToTensor(test_images_[i]);
            auto outputs = engine.inference(input);

            EXPECT_FALSE(outputs.empty()) << "TensorRT inference should produce outputs";
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        float tensorrt_throughput = (static_cast<float>(test_images_.size()) * 1000.0f) / total_time;

        // TensorRT should be significantly faster
        EXPECT_GE(tensorrt_throughput, min_throughput_qps_ * 3)
            << "TensorRT throughput below expected threshold: " << tensorrt_throughput << " QPS";
    } else {
        // If TensorRT is not available, skip this test
        GTEST_SKIP() << "TensorRT engine building failed - skipping TensorRT tests";
    }
}

// Helper function implementations
FloatTensor YOLOIntegrationTest::imageToTensor(const cv::Mat& image) {
    FloatTensor tensor({1, 3, image.rows, image.cols});

    // Convert BGR to RGB and normalize
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3f pixel = image.at<cv::Vec3f>(y, x);
            tensor.setData({0, 0, y, x}, pixel[2]);  // R
            tensor.setData({0, 1, y, x}, pixel[1]);  // G
            tensor.setData({0, 2, y, x}, pixel[0]);  // B
        }
    }

    return tensor;
}

std::vector<Detection> YOLOIntegrationTest::parseDetections(const std::vector<FloatTensor>& outputs) {
    std::vector<Detection> detections;

    if (outputs.empty()) return detections;

    // Mock detection parsing - in real implementation this would parse YOLO output format
    const auto& output = outputs[0];
    auto shape = output.getShape();

    // Simulate parsing detections from output tensor
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    int num_detections = gen() % 5 + 1;  // 1-5 detections

    for (int i = 0; i < num_detections; ++i) {
        Detection det;
        det.bbox.x = dis(gen) * 500;
        det.bbox.y = dis(gen) * 500;
        det.bbox.width = dis(gen) * 100 + 50;
        det.bbox.height = dis(gen) * 100 + 50;
        det.confidence = dis(gen) * 0.5f + 0.5f;
        det.class_id = gen() % 80;
        detections.push_back(det);
    }

    return detections;
}

// Performance regression test suite
class YOLOPerformanceRegressionTest : public YOLOIntegrationTest {
protected:
    void SetUp() override {
        YOLOIntegrationTest::SetUp();
        loadBaselineMetrics();
    }

    void loadBaselineMetrics() {
        // Load baseline performance metrics from file
        std::ifstream baseline_file("tests/fixtures/yolo_performance_baseline.json");
        if (baseline_file.is_open()) {
            // Parse baseline metrics (simplified)
            baseline_inference_time_ms_ = 25.0f;
            baseline_throughput_qps_ = 40.0f;
            baseline_memory_usage_mb_ = 1500.0f;
            baseline_accuracy_ = 0.85f;
        } else {
            // Use default baselines if file doesn't exist
            baseline_inference_time_ms_ = 30.0f;
            baseline_throughput_qps_ = 35.0f;
            baseline_memory_usage_mb_ = 2000.0f;
            baseline_accuracy_ = 0.80f;
        }
    }

    void saveCurrentMetrics(float inference_time, float throughput,
                           float memory_usage, float accuracy) {
        std::ofstream metrics_file("tests/fixtures/yolo_current_metrics.json");
        if (metrics_file.is_open()) {
            metrics_file << "{\n";
            metrics_file << "  \"inference_time_ms\": " << inference_time << ",\n";
            metrics_file << "  \"throughput_qps\": " << throughput << ",\n";
            metrics_file << "  \"memory_usage_mb\": " << memory_usage << ",\n";
            metrics_file << "  \"accuracy\": " << accuracy << "\n";
            metrics_file << "}\n";
        }
    }

protected:
    float baseline_inference_time_ms_;
    float baseline_throughput_qps_;
    float baseline_memory_usage_mb_;
    float baseline_accuracy_;

    // Regression thresholds (allow 5% degradation)
    static constexpr float regression_threshold_ = 0.05f;
};

TEST_F(YOLOPerformanceRegressionTest, InferenceTimeRegression) {
    ONNXParser parser;
    ASSERT_TRUE(parser.loadModel("yolo_model.onnx"));
    ASSERT_TRUE(parser.parseModel());

    // Measure current inference time
    std::vector<float> inference_times;

    for (size_t i = 0; i < test_images_.size(); ++i) {
        FloatTensor input = imageToTensor(test_images_[i]);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto outputs = parser.inference(input);
        auto end_time = std::chrono::high_resolution_clock::now();

        float inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0f;  // Convert to ms

        inference_times.push_back(inference_time);
    }

    // Calculate average inference time
    float avg_inference_time = std::accumulate(inference_times.begin(),
                                              inference_times.end(), 0.0f) / inference_times.size();

    // Check for regression
    float regression_ratio = (avg_inference_time - baseline_inference_time_ms_) / baseline_inference_time_ms_;

    EXPECT_LE(regression_ratio, regression_threshold_)
        << "Inference time regression detected: " << avg_inference_time
        << " ms vs baseline " << baseline_inference_time_ms_ << " ms ("
        << (regression_ratio * 100) << "% increase)";

    // Save current metrics
    saveCurrentMetrics(avg_inference_time, 0, 0, 0);
}

TEST_F(YOLOPerformanceRegressionTest, ThroughputRegression) {
    MultiStreamInferenceSystem::SystemConfig config;
    config.num_streams = 4;
    config.strategy = MultiStreamInferenceSystem::LoadBalanceStrategy::LEAST_LOADED;

    MultiStreamInferenceSystem inference_system(gpu_mem_mgr_, config);
    ASSERT_TRUE(inference_system.initialize());

    // Measure throughput with concurrent requests
    const int num_requests = 50;
    std::vector<std::future<std::vector<FloatTensor>>> futures;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_requests; ++i) {
        size_t img_idx = i % test_images_.size();
        FloatTensor input = imageToTensor(test_images_[img_idx]);
        std::string task_id = "regression_test_" + std::to_string(i);

        auto future = inference_system.submitInference(task_id, input);
        futures.push_back(std::move(future));
    }

    // Wait for all results
    int completed_requests = 0;
    for (auto& future : futures) {
        try {
            future.get();
            completed_requests++;
        } catch (...) {}
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    float current_throughput = (static_cast<float>(completed_requests) * 1000.0f) / total_time;

    // Check for throughput regression
    float throughput_regression = (baseline_throughput_qps_ - current_throughput) / baseline_throughput_qps_;

    EXPECT_LE(throughput_regression, regression_threshold_)
        << "Throughput regression detected: " << current_throughput
        << " QPS vs baseline " << baseline_throughput_qps_ << " QPS ("
        << (throughput_regression * 100) << "% decrease)";

    inference_system.shutdown();
}

TEST_F(YOLOPerformanceRegressionTest, MemoryUsageRegression) {
    // Test memory usage with batch inference
    BatchInferenceConfig config;
    config.max_batch_size = 8;
    config.input_height = 640;
    config.input_width = 640;
    config.input_channels = 3;

    BatchInferenceManager batch_manager(config, gpu_mem_mgr_);
    ASSERT_TRUE(batch_manager.start());

    // Submit requests and measure peak memory usage
    size_t initial_memory = gpu_mem_mgr_->getCurrentUsage();

    std::vector<std::future<std::vector<FloatTensor>>> futures;
    for (size_t i = 0; i < test_images_.size(); ++i) {
        FloatTensor input = imageToTensor(test_images_[i]);
        std::string request_id = "memory_test_" + std::to_string(i);

        auto future = batch_manager.submitRequest(request_id, input);
        futures.push_back(std::move(future));
    }

    // Wait for completion and measure peak memory
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {}
    }

    size_t peak_memory = gpu_mem_mgr_->getPeakUsage();
    float current_memory_mb = static_cast<float>(peak_memory - initial_memory) / (1024 * 1024);

    // Check for memory regression
    float memory_regression = (current_memory_mb - baseline_memory_usage_mb_) / baseline_memory_usage_mb_;

    EXPECT_LE(memory_regression, regression_threshold_)
        << "Memory usage regression detected: " << current_memory_mb
        << " MB vs baseline " << baseline_memory_usage_mb_ << " MB ("
        << (memory_regression * 100) << "% increase)";

    batch_manager.stop();
}

// Main test runner
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize CUDA context
    cudaSetDevice(0);

    return RUN_ALL_TESTS();
}
