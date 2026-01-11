#include "tensorrt_yolo_accelerator.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace yolo {
namespace tensorrt {

// TensorRTYOLOAccelerator Implementation
TensorRTYOLOAccelerator::TensorRTYOLOAccelerator(const AcceleratorConfig& config)
    : config_(config)
    , current_context_id_(-1)
    , num_classes_(80)
    , confidence_threshold_(0.5f)
    , nms_threshold_(0.45f)
    , input_buffer_gpu_(nullptr)
    , output_buffer_gpu_(nullptr)
    , input_buffer_size_(0)
    , output_buffer_size_(0)
    , inference_stream_(nullptr)
    , initialized_(false)
    , profiling_enabled_(config.enable_profiling) {

    // 初始化CUDA流
    cudaStreamCreate(&inference_stream_);

    // 创建GPU内存管理器
    memory_manager_ = std::make_shared<utils::GPUMemoryManager>();

    std::cout << "TensorRT YOLO Accelerator created" << std::endl;
}

TensorRTYOLOAccelerator::~TensorRTYOLOAccelerator() {
    // 清理GPU内存
    if (input_buffer_gpu_) {
        freeGPUMemory(input_buffer_gpu_);
    }
    if (output_buffer_gpu_) {
        freeGPUMemory(output_buffer_gpu_);
    }

    // 销毁CUDA流
    if (inference_stream_) {
        cudaStreamDestroy(inference_stream_);
    }

    std::cout << "TensorRT YOLO Accelerator destroyed" << std::endl;
}

bool TensorRTYOLOAccelerator::initializeFromONNX(const std::string& onnx_path,
                                                 const std::vector<std::string>& class_names) {

    std::cout << "Initializing TensorRT YOLO Accelerator from ONNX: " << onnx_path << std::endl;

    // 设置类别名称
    class_names_ = class_names;
    if (class_names_.empty()) {
        // 使用默认COCO类别
        num_classes_ = 80;
        for (int i = 0; i < num_classes_; ++i) {
            class_names_.push_back("class_" + std::to_string(i));
        }
    } else {
        num_classes_ = class_names_.size();
    }

    // 检查是否使用缓存的引擎
    if (config_.use_engine_cache && !config_.force_rebuild) {
        std::ifstream engine_file(config_.engine_cache_path);
        if (engine_file.good()) {
            std::cout << "Loading cached engine from: " << config_.engine_cache_path << std::endl;
            if (initializeFromEngine(config_.engine_cache_path, class_names)) {
                return true;
            }
        }
    }

    // 初始化TensorRT组件
    if (!initializeTensorRT()) {
        return false;
    }

    // 注册自定义插件
    registerCustomPlugins();

    // 构建引擎
    engine_ = engine_builder_->buildEngineFromONNX(onnx_path, config_.build_config);
    if (!engine_) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return false;
    }

    // 保存引擎缓存
    if (config_.use_engine_cache) {
        if (!engine_builder_->serializeEngine(engine_.get(), config_.engine_cache_path)) {
            std::cerr << "Failed to save engine cache" << std::endl;
        }
    }

    // 配置动态形状处理
    if (!configureDynamicShapes()) {
        return false;
    }

    // 创建执行上下文
    for (int i = 0; i < config_.num_execution_contexts; ++i) {
        int context_id = shape_handler_->createExecutionContext("context_" + std::to_string(i));
        if (context_id >= 0) {
            execution_context_ids_.push_back(context_id);
        }
    }

    if (execution_context_ids_.empty()) {
        std::cerr << "Failed to create execution contexts" << std::endl;
        return false;
    }

    current_context_id_ = execution_context_ids_[0];

    // 初始化动态批处理管理器
    if (config_.batch_config.enable_auto_batching) {
        batch_manager_ = std::make_unique<DynamicBatchManager>(shape_handler_, config_.batch_config);
        batch_manager_->start();
    }

    initialized_ = true;
    std::cout << "TensorRT YOLO Accelerator initialized successfully" << std::endl;

    return true;
}

bool TensorRTYOLOAccelerator::initializeFromEngine(const std::string& engine_path,
                                                  const std::vector<std::string>& class_names) {

    std::cout << "Initializing from serialized engine: " << engine_path << std::endl;

    // 设置类别名称
    class_names_ = class_names;
    if (class_names_.empty()) {
        num_classes_ = 80;
        for (int i = 0; i < num_classes_; ++i) {
            class_names_.push_back("class_" + std::to_string(i));
        }
    } else {
        num_classes_ = class_names_.size();
    }

    // 初始化TensorRT组件
    if (!initializeTensorRT()) {
        return false;
    }

    // 注册自定义插件
    registerCustomPlugins();

    // 加载序列化引擎
    engine_ = engine_builder_->loadSerializedEngine(engine_path);
    if (!engine_) {
        std::cerr << "Failed to load serialized engine" << std::endl;
        return false;
    }

    // 配置动态形状处理
    if (!configureDynamicShapes()) {
        return false;
    }

    // 创建执行上下文
    for (int i = 0; i < config_.num_execution_contexts; ++i) {
        int context_id = shape_handler_->createExecutionContext("context_" + std::to_string(i));
        if (context_id >= 0) {
            execution_context_ids_.push_back(context_id);
        }
    }

    if (execution_context_ids_.empty()) {
        std::cerr << "Failed to create execution contexts" << std::endl;
        return false;
    }

    current_context_id_ = execution_context_ids_[0];

    // 初始化动态批处理管理器
    if (config_.batch_config.enable_auto_batching) {
        batch_manager_ = std::make_unique<DynamicBatchManager>(shape_handler_, config_.batch_config);
        batch_manager_->start();
    }

    initialized_ = true;
    std::cout << "TensorRT YOLO Accelerator initialized from engine successfully" << std::endl;

    return true;
}

TensorRTYOLOAccelerator::InferenceResult TensorRTYOLOAccelerator::inferSingle(
    const float* image_data, int width, int height, int channels) {

    if (!initialized_) {
        std::cerr << "Accelerator not initialized" << std::endl;
        return {};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 设置输入形状
    if (!setInputShape(1, width, height, channels)) {
        std::cerr << "Failed to set input shape" << std::endl;
        return {};
    }

    // 预处理输入数据
    size_t input_size = 1 * channels * height * width * sizeof(float);
    if (input_buffer_size_ < input_size) {
        if (input_buffer_gpu_) {
            freeGPUMemory(input_buffer_gpu_);
        }
        input_buffer_gpu_ = allocateGPUMemory(input_size);
        input_buffer_size_ = input_size;
    }

    // 复制输入数据到GPU
    cudaMemcpyAsync(input_buffer_gpu_, image_data, input_size,
                   cudaMemcpyHostToDevice, inference_stream_);

    // 准备输出缓冲区
    size_t output_size = 1 * 1000 * 6 * sizeof(float); // 最大1000个检测，每个6个值
    if (output_buffer_size_ < output_size) {
        if (output_buffer_gpu_) {
            freeGPUMemory(output_buffer_gpu_);
        }
        output_buffer_gpu_ = allocateGPUMemory(output_size);
        output_buffer_size_ = output_size;
    }

    // 执行推理
    std::map<std::string, void*> input_buffers = {{input_tensor_name_, input_buffer_gpu_}};
    std::map<std::string, void*> output_buffers = {{output_tensor_name_, output_buffer_gpu_}};

    bool success = shape_handler_->executeInferenceAsync(
        current_context_id_, input_buffers, output_buffers, inference_stream_);

    if (!success) {
        std::cerr << "Inference execution failed" << std::endl;
        return {};
    }

    // 同步CUDA流
    cudaStreamSynchronize(inference_stream_);

    // 复制输出数据到CPU
    std::vector<float> output_data(1000 * 6);
    cudaMemcpy(output_data.data(), output_buffer_gpu_, output_size, cudaMemcpyDeviceToHost);

    // 后处理
    auto batch_detections = postprocessOutput(output_data.data(), 1);

    auto end_time = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    InferenceResult result;
    result.detections = batch_detections.empty() ? std::vector<Detection>() : batch_detections[0];
    result.inference_time_ms = inference_time;
    result.batch_size = 1;

    // 更新性能统计
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        performance_stats_["avg_inference_time_ms"] =
            (performance_stats_["avg_inference_time_ms"] * 0.9f + inference_time * 0.1f);
        performance_stats_["total_inferences"]++;
    }

    return result;
}

TensorRTYOLOAccelerator::InferenceResult TensorRTYOLOAccelerator::inferBatch(
    const std::vector<const float*>& batch_images,
    int batch_size, int width, int height, int channels) {

    if (!initialized_) {
        std::cerr << "Accelerator not initialized" << std::endl;
        return {};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 设置输入形状
    if (!setInputShape(batch_size, width, height, channels)) {
        std::cerr << "Failed to set input shape" << std::endl;
        return {};
    }

    // 准备批量输入数据
    size_t single_image_size = channels * height * width * sizeof(float);
    size_t batch_input_size = batch_size * single_image_size;

    if (input_buffer_size_ < batch_input_size) {
        if (input_buffer_gpu_) {
            freeGPUMemory(input_buffer_gpu_);
        }
        input_buffer_gpu_ = allocateGPUMemory(batch_input_size);
        input_buffer_size_ = batch_input_size;
    }

    // 复制批量数据到GPU
    for (int i = 0; i < batch_size; ++i) {
        void* dst = static_cast<char*>(input_buffer_gpu_) + i * single_image_size;
        cudaMemcpyAsync(dst, batch_images[i], single_image_size,
                       cudaMemcpyHostToDevice, inference_stream_);
    }

    // 准备输出缓冲区
    size_t output_size = batch_size * 1000 * 6 * sizeof(float);
    if (output_buffer_size_ < output_size) {
        if (output_buffer_gpu_) {
            freeGPUMemory(output_buffer_gpu_);
        }
        output_buffer_gpu_ = allocateGPUMemory(output_size);
        output_buffer_size_ = output_size;
    }

    // 执行推理
    std::map<std::string, void*> input_buffers = {{input_tensor_name_, input_buffer_gpu_}};
    std::map<std::string, void*> output_buffers = {{output_tensor_name_, output_buffer_gpu_}};

    bool success = shape_handler_->executeInferenceAsync(
        current_context_id_, input_buffers, output_buffers, inference_stream_);

    if (!success) {
        std::cerr << "Batch inference execution failed" << std::endl;
        return {};
    }

    // 同步CUDA流
    cudaStreamSynchronize(inference_stream_);

    // 复制输出数据到CPU
    std::vector<float> output_data(batch_size * 1000 * 6);
    cudaMemcpy(output_data.data(), output_buffer_gpu_, output_size, cudaMemcpyDeviceToHost);

    // 后处理
    auto batch_detections = postprocessOutput(output_data.data(), batch_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    InferenceResult result;
    // 合并所有批次的检测结果
    for (const auto& detections : batch_detections) {
        result.detections.insert(result.detections.end(), detections.begin(), detections.end());
    }
    result.inference_time_ms = inference_time;
    result.batch_size = batch_size;

    // 更新性能统计
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        performance_stats_["avg_batch_inference_time_ms"] =
            (performance_stats_["avg_batch_inference_time_ms"] * 0.9f + inference_time * 0.1f);
        performance_stats_["total_batch_inferences"]++;
    }

    return result;
}

std::future<TensorRTYOLOAccelerator::InferenceResult> TensorRTYOLOAccelerator::inferAsync(
    const float* image_data, int width, int height, int channels) {

    return std::async(std::launch::async, [this, image_data, width, height, channels]() {
        return inferSingle(image_data, width, height, channels);
    });
}

int TensorRTYOLOAccelerator::inferDynamicBatch(const float* image_data,
                                              int width, int height, int channels,
                                              std::function<void(const InferenceResult&)> callback) {

    if (!batch_manager_) {
        std::cerr << "Dynamic batch manager not initialized" << std::endl;
        return -1;
    }

    // 准备输入数据
    std::map<std::string, void*> input_data;
    size_t image_size = channels * height * width * sizeof(float);
    void* gpu_buffer = allocateGPUMemory(image_size);
    cudaMemcpy(gpu_buffer, image_data, image_size, cudaMemcpyHostToDevice);
    input_data[input_tensor_name_] = gpu_buffer;

    // 提交到批处理管理器
    return batch_manager_->submitRequest(input_data,
        [this, callback, gpu_buffer](const std::map<std::string, void*>& output_data) {
            // 处理输出并调用回调
            InferenceResult result; // 简化实现
            callback(result);

            // 清理GPU内存
            freeGPUMemory(gpu_buffer);
        });
}

bool TensorRTYOLOAccelerator::setInputShape(int batch_size, int width, int height, int channels) {
    std::vector<int> new_shape = {batch_size, channels, height, width};

    if (new_shape == input_shape_) {
        return true; // 形状未改变
    }

    input_shape_ = new_shape;

    // 更新所有执行上下文的输入形状
    for (int context_id : execution_context_ids_) {
        if (!shape_handler_->setInputShape(context_id, input_tensor_name_, input_shape_)) {
            std::cerr << "Failed to set input shape for context: " << context_id << std::endl;
            return false;
        }
    }

    // 获取输出形状
    output_shape_ = shape_handler_->getOutputShape(current_context_id_, output_tensor_name_);

    std::cout << "Input shape updated to: [" << batch_size << ", " << channels
              << ", " << height << ", " << width << "]" << std::endl;

    return true;
}

bool TensorRTYOLOAccelerator::warmupEngine(int warmup_iterations) {
    if (!initialized_) {
        std::cerr << "Accelerator not initialized" << std::endl;
        return false;
    }

    std::cout << "Warming up engine with " << warmup_iterations << " iterations..." << std::endl;

    // 使用默认输入形状进行预热
    std::vector<int> warmup_shape = {1, 3, 640, 640};
    if (!setInputShape(warmup_shape[0], warmup_shape[3], warmup_shape[2], warmup_shape[1])) {
        return false;
    }

    // 创建虚拟输入数据
    size_t input_size = warmup_shape[0] * warmup_shape[1] * warmup_shape[2] * warmup_shape[3];
    std::vector<float> dummy_input(input_size, 0.5f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < warmup_iterations; ++i) {
        inferSingle(dummy_input.data(), warmup_shape[3], warmup_shape[2], warmup_shape[1]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    std::cout << "Engine warmup completed. Average time per iteration: "
              << total_time / warmup_iterations << " ms" << std::endl;

    return true;
}

std::map<std::string, float> TensorRTYOLOAccelerator::getPerformanceStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

std::string TensorRTYOLOAccelerator::getEngineInfo() const {
    if (!engine_) {
        return "Engine not initialized";
    }

    std::ostringstream info;
    info << "=== TensorRT YOLO Accelerator Info ===" << std::endl;
    info << "Number of Classes: " << num_classes_ << std::endl;
    info << "Confidence Threshold: " << confidence_threshold_ << std::endl;
    info << "NMS Threshold: " << nms_threshold_ << std::endl;
    info << "Execution Contexts: " << execution_context_ids_.size() << std::endl;

    if (shape_handler_) {
        info << shape_handler_->getEngineInfo() << std::endl;
    }

    return info.str();
}

bool TensorRTYOLOAccelerator::saveEngine(const std::string& engine_path) {
    if (!engine_ || !engine_builder_) {
        std::cerr << "Engine not available for saving" << std::endl;
        return false;
    }

    return engine_builder_->serializeEngine(engine_.get(), engine_path);
}

void TensorRTYOLOAccelerator::setConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
    std::cout << "Confidence threshold set to: " << threshold << std::endl;
}

void TensorRTYOLOAccelerator::setNMSThreshold(float threshold) {
    nms_threshold_ = threshold;
    std::cout << "NMS threshold set to: " << threshold << std::endl;
}

void TensorRTYOLOAccelerator::enableProfiling(bool enable) {
    profiling_enabled_ = enable;
    std::cout << "Profiling " << (enable ? "enabled" : "disabled") << std::endl;
}

// Private methods implementation
bool TensorRTYOLOAccelerator::initializeTensorRT() {
    engine_builder_ = std::make_unique<TensorRTEngineBuilder>();
    if (!engine_builder_) {
        std::cerr << "Failed to create TensorRT engine builder" << std::endl;
        return false;
    }

    std::cout << "TensorRT components initialized" << std::endl;
    return true;
}

void TensorRTYOLOAccelerator::registerCustomPlugins() {
    auto& plugin_manager = PluginManager::getInstance();
    plugin_manager.registerYOLOPlugins();

    std::cout << "Custom plugins registered" << std::endl;
}

bool TensorRTYOLOAccelerator::configureDynamicShapes() {
    shape_handler_ = std::make_unique<DynamicShapeHandler>(engine_);

    // 添加配置的动态形状
    for (const auto& shape_config : config_.shape_configs) {
        if (!shape_handler_->addShapeConfig(shape_config)) {
            std::cerr << "Failed to add shape config for: " << shape_config.tensor_name << std::endl;
            return false;
        }
    }

    // 获取输入输出张量名称
    if (engine_->getNbBindings() >= 2) {
        input_tensor_name_ = engine_->getBindingName(0);
        output_tensor_name_ = engine_->getBindingName(1);
    }

    std::cout << "Dynamic shapes configured" << std::endl;
    return true;
}

void TensorRTYOLOAccelerator::preprocessInput(const float* image_data, float* processed_data,
                                             int width, int height, int channels) {
    // 简单的数据复制，实际应用中可能需要归一化等预处理
    size_t data_size = width * height * channels * sizeof(float);
    std::memcpy(processed_data, image_data, data_size);
}

std::vector<std::vector<TensorRTYOLOAccelerator::Detection>>
TensorRTYOLOAccelerator::postprocessOutput(const float* output_data, int batch_size) {

    std::vector<std::vector<Detection>> batch_results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        std::vector<Detection> detections;

        // 解析输出数据 (假设格式为 [x, y, w, h, conf, class])
        const float* batch_output = output_data + b * 1000 * 6;

        for (int i = 0; i < 1000; ++i) {
            const float* detection_data = batch_output + i * 6;

            float confidence = detection_data[4];
            if (confidence < confidence_threshold_) {
                continue;
            }

            Detection det;
            det.x = detection_data[0];
            det.y = detection_data[1];
            det.w = detection_data[2];
            det.h = detection_data[3];
            det.confidence = confidence;
            det.class_id = static_cast<int>(detection_data[5]);

            if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
                det.class_name = class_names_[det.class_id];
            } else {
                det.class_name = "unknown";
            }

            detections.push_back(det);
        }

        // 应用NMS
        batch_results[b] = applyNMS(detections, nms_threshold_);
    }

    return batch_results;
}

std::vector<TensorRTYOLOAccelerator::Detection>
TensorRTYOLOAccelerator::applyNMS(const std::vector<Detection>& detections, float nms_threshold) {

    if (detections.empty()) {
        return {};
    }

    // 按置信度排序
    std::vector<Detection> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> result;
    std::vector<bool> suppressed(sorted_detections.size(), false);

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(sorted_detections[i]);

        // 抑制重叠的检测框
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (suppressed[j]) continue;

            if (sorted_detections[i].class_id == sorted_detections[j].class_id) {
                float iou = calculateIoU(sorted_detections[i], sorted_detections[j]);
                if (iou > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

float TensorRTYOLOAccelerator::calculateIoU(const Detection& det1, const Detection& det2) {
    float x1_min = det1.x - det1.w / 2;
    float y1_min = det1.y - det1.h / 2;
    float x1_max = det1.x + det1.w / 2;
    float y1_max = det1.y + det1.h / 2;

    float x2_min = det2.x - det2.w / 2;
    float y2_min = det2.y - det2.h / 2;
    float x2_max = det2.x + det2.w / 2;
    float y2_max = det2.y + det2.h / 2;

    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);

    if (inter_x_max <= inter_x_min || inter_y_max <= inter_y_min) {
        return 0.0f;
    }

    float inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    float area1 = det1.w * det1.h;
    float area2 = det2.w * det2.h;
    float union_area = area1 + area2 - inter_area;

    return inter_area / union_area;
}

void* TensorRTYOLOAccelerator::allocateGPUMemory(size_t size) {
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return ptr;
}

void TensorRTYOLOAccelerator::freeGPUMemory(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// TensorRTYOLOAcceleratorFactory Implementation
std::unique_ptr<TensorRTYOLOAccelerator> TensorRTYOLOAcceleratorFactory::createYOLOv5Accelerator(
    const std::string& onnx_path,
    const TensorRTYOLOAccelerator::AcceleratorConfig& config) {

    auto accelerator_config = config;
    if (accelerator_config.shape_configs.empty()) {
        accelerator_config = getDefaultYOLOv5Config();
    }

    auto accelerator = std::make_unique<TensorRTYOLOAccelerator>(accelerator_config);

    if (!accelerator->initializeFromONNX(onnx_path)) {
        return nullptr;
    }

    return accelerator;
}

std::unique_ptr<TensorRTYOLOAccelerator> TensorRTYOLOAcceleratorFactory::createYOLOv8Accelerator(
    const std::string& onnx_path,
    const TensorRTYOLOAccelerator::AcceleratorConfig& config) {

    auto accelerator_config = config;
    if (accelerator_config.shape_configs.empty()) {
        accelerator_config = getDefaultYOLOv8Config();
    }

    auto accelerator = std::make_unique<TensorRTYOLOAccelerator>(accelerator_config);

    if (!accelerator->initializeFromONNX(onnx_path)) {
        return nullptr;
    }

    return accelerator;
}

TensorRTYOLOAccelerator::AcceleratorConfig
TensorRTYOLOAcceleratorFactory::getDefaultYOLOv5Config() {

    TensorRTYOLOAccelerator::AcceleratorConfig config;

    // 构建配置
    config.build_config.max_batch_size = 8;
    config.build_config.enable_fp16 = true;
    config.build_config.enable_dynamic_shapes = true;

    // 动态形状配置
    DynamicShapeHandler::ShapeConfig input_shape_config;
    input_shape_config.tensor_name = "images";
    input_shape_config.min_shape = {1, 3, 320, 320};
    input_shape_config.opt_shape = {4, 3, 640, 640};
    input_shape_config.max_shape = {8, 3, 1280, 1280};
    config.shape_configs.push_back(input_shape_config);

    // 批处理配置
    config.batch_config.min_batch_size = 1;
    config.batch_config.opt_batch_size = 4;
    config.batch_config.max_batch_size = 8;
    config.batch_config.enable_auto_batching = true;

    return config;
}

TensorRTYOLOAccelerator::AcceleratorConfig
TensorRTYOLOAcceleratorFactory::getDefaultYOLOv8Config() {

    TensorRTYOLOAccelerator::AcceleratorConfig config;

    // 构建配置
    config.build_config.max_batch_size = 16;
    config.build_config.enable_fp16 = true;
    config.build_config.enable_dynamic_shapes = true;

    // 动态形状配置
    DynamicShapeHandler::ShapeConfig input_shape_config;
    input_shape_config.tensor_name = "images";
    input_shape_config.min_shape = {1, 3, 320, 320};
    input_shape_config.opt_shape = {8, 3, 640, 640};
    input_shape_config.max_shape = {16, 3, 1280, 1280};
    config.shape_configs.push_back(input_shape_config);

    // 批处理配置
    config.batch_config.min_batch_size = 1;
    config.batch_config.opt_batch_size = 8;
    config.batch_config.max_batch_size = 16;
    config.batch_config.enable_auto_batching = true;

    return config;
}

} // namespace tensorrt
} // namespace yolo
