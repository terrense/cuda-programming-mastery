#include "dynamic_shape_handler.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>

namespace yolo {
namespace tensorrt {

// ShapeConfig Implementation
bool DynamicShapeHandler::ShapeConfig::isValid() const {
    if (min_shape.size() != opt_shape.size() ||
        opt_shape.size() != max_shape.size()) {
        return false;
    }

    for (size_t i = 0; i < min_shape.size(); ++i) {
        if (min_shape[i] <= 0 || opt_shape[i] <= 0 || max_shape[i] <= 0) {
            return false;
        }
        if (min_shape[i] > opt_shape[i] || opt_shape[i] > max_shape[i]) {
            return false;
        }
    }

    return true;
}

bool DynamicShapeHandler::ShapeConfig::isInRange(const std::vector<int>& shape) const {
    if (shape.size() != min_shape.size()) {
        return false;
    }

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < min_shape[i] || shape[i] > max_shape[i]) {
            return false;
        }
    }

    return true;
}

// DynamicShapeHandler Implementation
DynamicShapeHandler::DynamicShapeHandler(std::shared_ptr<nvinfer1::ICudaEngine> engine)
    : engine_(engine), next_context_id_(0) {

    if (!engine_) {
        throw std::runtime_error("Invalid TensorRT engine");
    }

    // 缓存引擎信息
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        std::string tensor_name = engine_->getBindingName(i);
        nvinfer1::DataType data_type = engine_->getBindingDataType(i);

        tensor_data_types_[tensor_name] = data_type;

        if (engine_->bindingIsInput(i)) {
            input_tensor_names_.push_back(tensor_name);
        } else {
            output_tensor_names_.push_back(tensor_name);
        }
    }

    std::cout << "DynamicShapeHandler initialized with "
              << input_tensor_names_.size() << " inputs and "
              << output_tensor_names_.size() << " outputs" << std::endl;
}

DynamicShapeHandler::~DynamicShapeHandler() {
    // 清理所有执行上下文
    for (auto& [context_id, context] : execution_contexts_) {
        if (context.workspace_ptr) {
            cudaFree(context.workspace_ptr);
        }
    }
    execution_contexts_.clear();

    std::cout << "DynamicShapeHandler destroyed" << std::endl;
}

bool DynamicShapeHandler::addShapeConfig(const ShapeConfig& config) {
    if (!config.isValid()) {
        std::cerr << "Invalid shape config for tensor: " << config.tensor_name << std::endl;
        return false;
    }

    // 验证张量名称是否存在
    bool tensor_exists = false;
    for (const auto& name : input_tensor_names_) {
        if (name == config.tensor_name) {
            tensor_exists = true;
            break;
        }
    }

    if (!tensor_exists) {
        std::cerr << "Tensor not found in engine: " << config.tensor_name << std::endl;
        return false;
    }

    shape_configs_[config.tensor_name] = config;
    std::cout << "Added shape config for tensor: " << config.tensor_name << std::endl;

    return true;
}

int DynamicShapeHandler::createExecutionContext(const std::string& context_name) {
    auto context = engine_->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return -1;
    }

    int context_id = next_context_id_++;

    ExecutionContext exec_context;
    exec_context.context.reset(context, [](nvinfer1::IExecutionContext* ctx) {
        if (ctx) ctx->destroy();
    });

    execution_contexts_[context_id] = std::move(exec_context);
    context_name_to_id_[context_name] = context_id;

    std::cout << "Created execution context: " << context_name
              << " (ID: " << context_id << ")" << std::endl;

    return context_id;
}

bool DynamicShapeHandler::setInputShape(int context_id, const std::string& tensor_name,
                                       const std::vector<int>& shape) {

    auto it = execution_contexts_.find(context_id);
    if (it == execution_contexts_.end()) {
        std::cerr << "Invalid context ID: " << context_id << std::endl;
        return false;
    }

    if (!validateShapeCompatibility(tensor_name, shape)) {
        return false;
    }

    auto& context = it->second;
    nvinfer1::Dims dims = vectorToDims(shape);

    if (!context.context->setBindingDimensions(
            engine_->getBindingIndex(tensor_name.c_str()), dims)) {
        std::cerr << "Failed to set binding dimensions for: " << tensor_name << std::endl;
        return false;
    }

    context.current_shapes[tensor_name] = shape;

    // 更新工作空间
    updateWorkspace(context_id);

    std::cout << "Set input shape for " << tensor_name << ": [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]" << std::endl;

    return true;
}

bool DynamicShapeHandler::setInputShapes(int context_id,
                                        const std::map<std::string, std::vector<int>>& shapes) {

    for (const auto& [tensor_name, shape] : shapes) {
        if (!setInputShape(context_id, tensor_name, shape)) {
            return false;
        }
    }

    return true;
}

std::vector<int> DynamicShapeHandler::getOutputShape(int context_id,
                                                    const std::string& tensor_name) {

    auto it = execution_contexts_.find(context_id);
    if (it == execution_contexts_.end()) {
        std::cerr << "Invalid context ID: " << context_id << std::endl;
        return {};
    }

    int binding_index = engine_->getBindingIndex(tensor_name.c_str());
    if (binding_index == -1) {
        std::cerr << "Tensor not found: " << tensor_name << std::endl;
        return {};
    }

    nvinfer1::Dims dims = it->second.context->getBindingDimensions(binding_index);
    return dimsToVector(dims);
}

std::map<std::string, std::vector<int>> DynamicShapeHandler::getAllOutputShapes(int context_id) {
    std::map<std::string, std::vector<int>> output_shapes;

    for (const auto& tensor_name : output_tensor_names_) {
        auto shape = getOutputShape(context_id, tensor_name);
        if (!shape.empty()) {
            output_shapes[tensor_name] = shape;
        }
    }

    return output_shapes;
}

bool DynamicShapeHandler::executeInference(int context_id,
                                          const std::map<std::string, void*>& input_buffers,
                                          const std::map<std::string, void*>& output_buffers,
                                          cudaStream_t stream) {

    auto it = execution_contexts_.find(context_id);
    if (it == execution_contexts_.end()) {
        std::cerr << "Invalid context ID: " << context_id << std::endl;
        return false;
    }

    auto& context = it->second;

    // 准备绑定数组
    std::vector<void*> bindings(engine_->getNbBindings(), nullptr);

    // 设置输入绑定
    for (const auto& [tensor_name, buffer] : input_buffers) {
        int binding_index = engine_->getBindingIndex(tensor_name.c_str());
        if (binding_index == -1) {
            std::cerr << "Input tensor not found: " << tensor_name << std::endl;
            return false;
        }
        bindings[binding_index] = buffer;
    }

    // 设置输出绑定
    for (const auto& [tensor_name, buffer] : output_buffers) {
        int binding_index = engine_->getBindingIndex(tensor_name.c_str());
        if (binding_index == -1) {
            std::cerr << "Output tensor not found: " << tensor_name << std::endl;
            return false;
        }
        bindings[binding_index] = buffer;
    }

    // 执行推理
    bool success;
    if (stream) {
        success = context.context->enqueueV2(bindings.data(), stream, nullptr);
    } else {
        success = context.context->executeV2(bindings.data());
    }

    if (!success) {
        std::cerr << "Inference execution failed" << std::endl;
        return false;
    }

    return true;
}

bool DynamicShapeHandler::executeInferenceAsync(int context_id,
                                               const std::map<std::string, void*>& input_buffers,
                                               const std::map<std::string, void*>& output_buffers,
                                               cudaStream_t stream) {

    if (!stream) {
        std::cerr << "CUDA stream required for async execution" << std::endl;
        return false;
    }

    return executeInference(context_id, input_buffers, output_buffers, stream);
}

bool DynamicShapeHandler::optimizeContext(int context_id,
                                         const std::map<std::string, std::vector<int>>& target_shapes) {

    auto it = execution_contexts_.find(context_id);
    if (it == execution_contexts_.end()) {
        std::cerr << "Invalid context ID: " << context_id << std::endl;
        return false;
    }

    // 设置优化配置文件
    for (const auto& [tensor_name, shape] : target_shapes) {
        if (!setInputShape(context_id, tensor_name, shape)) {
            return false;
        }
    }

    // 触发形状推断
    if (!it->second.context->allInputDimensionsSpecified()) {
        std::cerr << "Not all input dimensions specified" << std::endl;
        return false;
    }

    std::cout << "Context optimized for target shapes" << std::endl;
    return true;
}

size_t DynamicShapeHandler::getTensorSize(const std::string& tensor_name,
                                         const std::vector<int>& shape,
                                         nvinfer1::DataType data_type) {

    size_t volume = calculateTensorVolume(shape);
    size_t element_size = getDataTypeSize(data_type);

    return volume * element_size;
}

std::string DynamicShapeHandler::getEngineInfo() const {
    std::ostringstream info;
    info << "=== Dynamic Shape Handler Engine Info ===" << std::endl;
    info << "Input Tensors:" << std::endl;
    for (const auto& name : input_tensor_names_) {
        info << "  - " << name << " (" << static_cast<int>(tensor_data_types_.at(name)) << ")" << std::endl;
    }
    info << "Output Tensors:" << std::endl;
    for (const auto& name : output_tensor_names_) {
        info << "  - " << name << " (" << static_cast<int>(tensor_data_types_.at(name)) << ")" << std::endl;
    }
    info << "Shape Configurations: " << shape_configs_.size() << std::endl;
    info << "Active Contexts: " << execution_contexts_.size() << std::endl;

    return info.str();
}

std::string DynamicShapeHandler::getContextInfo(int context_id) const {
    auto it = execution_contexts_.find(context_id);
    if (it == execution_contexts_.end()) {
        return "Invalid context ID";
    }

    const auto& context = it->second;
    std::ostringstream info;
    info << "=== Context " << context_id << " Info ===" << std::endl;
    info << "Current Shapes:" << std::endl;
    for (const auto& [tensor_name, shape] : context.current_shapes) {
        info << "  " << tensor_name << ": [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) info << ", ";
            info << shape[i];
        }
        info << "]" << std::endl;
    }
    info << "Workspace Size: " << context.workspace_size << " bytes" << std::endl;

    return info.str();
}

void DynamicShapeHandler::destroyExecutionContext(int context_id) {
    auto it = execution_contexts_.find(context_id);
    if (it != execution_contexts_.end()) {
        if (it->second.workspace_ptr) {
            cudaFree(it->second.workspace_ptr);
        }
        execution_contexts_.erase(it);

        // 清理名称映射
        for (auto name_it = context_name_to_id_.begin();
             name_it != context_name_to_id_.end(); ++name_it) {
            if (name_it->second == context_id) {
                context_name_to_id_.erase(name_it);
                break;
            }
        }

        std::cout << "Destroyed execution context: " << context_id << std::endl;
    }
}

bool DynamicShapeHandler::validateShapeCompatibility(const std::string& tensor_name,
                                                    const std::vector<int>& shape) {

    auto config_it = shape_configs_.find(tensor_name);
    if (config_it != shape_configs_.end()) {
        if (!config_it->second.isInRange(shape)) {
            std::cerr << "Shape out of range for tensor: " << tensor_name << std::endl;
            return false;
        }
    }

    return true;
}

bool DynamicShapeHandler::updateWorkspace(int context_id) {
    auto it = execution_contexts_.find(context_id);
    if (it == execution_contexts_.end()) {
        return false;
    }

    auto& context = it->second;

    // 获取所需的工作空间大小
    size_t required_workspace = engine_->getWorkspaceSize();

    if (required_workspace > context.workspace_size) {
        // 释放旧的工作空间
        if (context.workspace_ptr) {
            cudaFree(context.workspace_ptr);
        }

        // 分配新的工作空间
        cudaError_t error = cudaMalloc(&context.workspace_ptr, required_workspace);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate workspace: " << cudaGetErrorString(error) << std::endl;
            context.workspace_ptr = nullptr;
            context.workspace_size = 0;
            return false;
        }

        context.workspace_size = required_workspace;
        std::cout << "Updated workspace size to: " << required_workspace << " bytes" << std::endl;
    }

    return true;
}

size_t DynamicShapeHandler::calculateTensorVolume(const std::vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

size_t DynamicShapeHandler::getDataTypeSize(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT: return sizeof(float);
        case nvinfer1::DataType::kHALF: return sizeof(__half);
        case nvinfer1::DataType::kINT8: return sizeof(int8_t);
        case nvinfer1::DataType::kINT32: return sizeof(int32_t);
        case nvinfer1::DataType::kBOOL: return sizeof(bool);
        default: return sizeof(float);
    }
}

nvinfer1::Dims DynamicShapeHandler::vectorToDims(const std::vector<int>& shape) {
    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

std::vector<int> DynamicShapeHandler::dimsToVector(const nvinfer1::Dims& dims) {
    std::vector<int> shape;
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

// DynamicBatchManager Implementation
DynamicBatchManager::DynamicBatchManager(std::shared_ptr<DynamicShapeHandler> shape_handler,
                                        const BatchConfig& config)
    : shape_handler_(shape_handler)
    , config_(config)
    , running_(false)
    , total_requests_(0)
    , total_batches_(0)
    , avg_batch_size_(0.0)
    , next_request_id_(0) {

    std::cout << "DynamicBatchManager initialized" << std::endl;
}

DynamicBatchManager::~DynamicBatchManager() {
    stop();
    std::cout << "DynamicBatchManager destroyed" << std::endl;
}

int DynamicBatchManager::submitRequest(const std::map<std::string, void*>& input_data,
                                      std::function<void(const std::map<std::string, void*>&)> callback) {

    std::lock_guard<std::mutex> lock(request_mutex_);

    int request_id = next_request_id_++;

    InferenceRequest request;
    request.request_id = request_id;
    request.input_data = input_data;
    request.callback = callback;
    request.submit_time = std::chrono::high_resolution_clock::now();

    pending_requests_[request_id] = request;
    request_queue_.push(request_id);

    total_requests_++;

    request_cv_.notify_one();

    return request_id;
}

int DynamicBatchManager::flushBatch() {
    std::lock_guard<std::mutex> lock(request_mutex_);

    std::vector<int> batch_requests;
    while (!request_queue_.empty() &&
           batch_requests.size() < static_cast<size_t>(config_.max_batch_size)) {
        batch_requests.push_back(request_queue_.front());
        request_queue_.pop();
    }

    if (!batch_requests.empty()) {
        executeBatch(batch_requests);
    }

    return batch_requests.size();
}

void DynamicBatchManager::setBatchConfig(const BatchConfig& config) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    config_ = config;
    std::cout << "Batch config updated" << std::endl;
}

std::string DynamicBatchManager::getBatchStatistics() const {
    std::ostringstream stats;
    stats << "=== Batch Manager Statistics ===" << std::endl;
    stats << "Total Requests: " << total_requests_.load() << std::endl;
    stats << "Total Batches: " << total_batches_.load() << std::endl;
    stats << "Average Batch Size: " << avg_batch_size_.load() << std::endl;
    stats << "Pending Requests: " << pending_requests_.size() << std::endl;

    return stats.str();
}

void DynamicBatchManager::start() {
    if (running_) {
        std::cout << "Batch manager already running" << std::endl;
        return;
    }

    running_ = true;
    batch_worker_ = std::thread(&DynamicBatchManager::batchWorkerThread, this);

    std::cout << "Batch manager started" << std::endl;
}

void DynamicBatchManager::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    request_cv_.notify_all();

    if (batch_worker_.joinable()) {
        batch_worker_.join();
    }

    std::cout << "Batch manager stopped" << std::endl;
}

void DynamicBatchManager::batchWorkerThread() {
    while (running_) {
        std::unique_lock<std::mutex> lock(request_mutex_);

        // 等待请求或超时
        request_cv_.wait_for(lock,
                            std::chrono::milliseconds(static_cast<int>(config_.batch_timeout_ms)),
                            [this] { return !request_queue_.empty() || !running_; });

        if (!running_) break;

        // 收集批次请求
        std::vector<int> batch_requests;
        while (!request_queue_.empty() &&
               batch_requests.size() < static_cast<size_t>(config_.max_batch_size)) {
            batch_requests.push_back(request_queue_.front());
            request_queue_.pop();
        }

        lock.unlock();

        // 执行批次
        if (!batch_requests.empty()) {
            executeBatch(batch_requests);
        }
    }
}

void DynamicBatchManager::executeBatch(const std::vector<int>& batch_requests) {
    if (batch_requests.empty()) return;

    // 更新统计信息
    total_batches_++;
    double current_avg = avg_batch_size_.load();
    double new_avg = (current_avg * (total_batches_ - 1) + batch_requests.size()) / total_batches_;
    avg_batch_size_ = new_avg;

    // 这里应该实现实际的批处理推理逻辑
    // 为了简化，我们只是调用回调函数
    for (int request_id : batch_requests) {
        auto it = pending_requests_.find(request_id);
        if (it != pending_requests_.end()) {
            // 模拟推理结果
            std::map<std::string, void*> output_data;
            it->second.callback(output_data);

            pending_requests_.erase(it);
        }
    }

    std::cout << "Executed batch with " << batch_requests.size() << " requests" << std::endl;
}

} // namespace tensorrt
} // namespace yolo
