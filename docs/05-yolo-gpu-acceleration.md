# YOLO模型GPU加速系统

## 概述

本文档介绍如何构建高效的YOLO模型GPU加速系统，包括模型解析、推理管道优化、TensorRT集成和性能监控等关键技术。

## 学习目标

完成本模块后，你将能够：
- 理解YOLO模型架构和GPU适配策略
- 实现ONNX/PyTorch模型的解析和加载
- 构建高效的GPU推理管道
- 集成TensorRT进行推理加速
- 开发实时性能监控和调优系统

## YOLO模型架构分析

### YOLO网络结构

YOLO (You Only Look Once) 是一种单阶段目标检测算法，其网络结构包括：

```
YOLO网络架构
├── Backbone (主干网络)
│   ├── CSPDarknet53/EfficientNet
│   ├── 卷积层 + 批归一化 + 激活
│   └── 残差连接
├── Neck (特征融合网络)
│   ├── FPN (特征金字塔网络)
│   ├── PAN (路径聚合网络)
│   └── 多尺度特征融合
└── Head (检测头)
    ├── 分类分支
    ├── 回归分支
    └── 置信度分支
```

### GPU加速关键点

1. **并行计算友好**: 卷积操作天然适合GPU并行
2. **内存访问优化**: 减少CPU-GPU数据传输
3. **算子融合**: 合并相邻操作减少内存访问
4. **批处理**: 提高GPU利用率
5. **混合精度**: 使用FP16加速推理

## YOLO模型解析和加载

### ONNX模型解析器

```cpp
// onnx_parser.h
#pragma once
#include <onnx/onnx.pb.h>
#include <string>
#include <vector>
#include <map>
#include <memory>

struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    int data_type;
    void* data;
    size_t size;
};

struct LayerInfo {
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::string> attributes;
};

class ONNXParser {
private:
    onnx::ModelProto model_;
    std::vector<LayerInfo> layers_;
    std::map<std::string, TensorInfo> tensors_;

public:
    ONNXParser();
    ~ONNXParser();

    // 加载ONNX模型
    bool loadModel(const std::string& model_path);

    // 解析模型结构
    bool parseModel();

    // 获取输入输出信息
    std::vector<TensorInfo> getInputs() const;
    std::vector<TensorInfo> getOutputs() const;

    // 获取层信息
    const std::vector<LayerInfo>& getLayers() const { return layers_; }

    // 获取权重数据
    bool getWeights(const std::string& name, void** data, size_t* size);

    // 模型验证
    bool validateModel();

private:
    void parseValueInfo(const onnx::ValueInfoProto& value_info, TensorInfo& tensor);
    void parseNode(const onnx::NodeProto& node, LayerInfo& layer);
    void parseInitializer(const onnx::TensorProto& tensor);
};

// onnx_parser.cpp
bool ONNXParser::loadModel(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return false;
    }

    if (!model_.ParseFromIstream(&file)) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return false;
    }

    return parseModel();
}

bool ONNXParser::parseModel() {
    const auto& graph = model_.graph();

    // 解析输入
    for (const auto& input : graph.input()) {
        TensorInfo tensor;
        parseValueInfo(input, tensor);
        tensors_[tensor.name] = tensor;
    }

    // 解析输出
    for (const auto& output : graph.output()) {
        TensorInfo tensor;
        parseValueInfo(output, tensor);
        tensors_[tensor.name] = tensor;
    }

    // 解析初始化器（权重）
    for (const auto& initializer : graph.initializer()) {
        parseInitializer(initializer);
    }

    // 解析节点（层）
    for (const auto& node : graph.node()) {
        LayerInfo layer;
        parseNode(node, layer);
        layers_.push_back(layer);
    }

    return true;
}

void ONNXParser::parseValueInfo(const onnx::ValueInfoProto& value_info, TensorInfo& tensor) {
    tensor.name = value_info.name();

    const auto& type = value_info.type().tensor_type();
    tensor.data_type = type.elem_type();

    for (const auto& dim : type.shape().dim()) {
        if (dim.has_dim_value()) {
            tensor.shape.push_back(dim.dim_value());
        } else {
            tensor.shape.push_back(-1); // 动态维度
        }
    }
}
```

### PyTorch模型加载器

```cpp
// pytorch_loader.h
#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>

class PyTorchLoader {
private:
    torch::jit::script::Module model_;
    std::vector<torch::jit::IValue> inputs_;
    bool model_loaded_;

public:
    PyTorchLoader();
    ~PyTorchLoader();

    // 加载TorchScript模型
    bool loadModel(const std::string& model_path);

    // 设置输入数据
    bool setInput(const std::vector<float>& data, const std::vector<int>& shape);

    // 执行推理
    std::vector<torch::Tensor> forward();

    // 获取模型信息
    void printModelInfo();

    // 转换为ONNX
    bool exportToONNX(const std::string& output_path, const std::vector<int>& input_shape);

private:
    torch::Tensor createTensorFromData(const std::vector<float>& data,
                                      const std::vector<int>& shape);
};

// pytorch_loader.cpp
bool PyTorchLoader::loadModel(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
        model_loaded_ = true;

        // 移动到GPU
        if (torch::cuda::is_available()) {
            model_.to(torch::kCUDA);
            std::cout << "Model loaded on GPU" << std::endl;
        } else {
            std::cout << "CUDA not available, using CPU" << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<torch::Tensor> PyTorchLoader::forward() {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }

    try {
        // 执行前向传播
        auto outputs = model_.forward(inputs_).toTuple();

        std::vector<torch::Tensor> result;
        for (const auto& output : outputs->elements()) {
            result.push_back(output.toTensor());
        }

        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return {};
    }
}
```

## GPU推理管道优化

### 批处理推理管道

```cpp
// inference_pipeline.h
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>

struct InferenceRequest {
    int request_id;
    void* input_data;
    size_t input_size;
    std::vector<int> input_shape;
    std::promise<std::vector<float>> result_promise;
};

struct InferenceResult {
    int request_id;
    std::vector<float> outputs;
    float inference_time;
    bool success;
};

class GPUInferencePipeline {
private:
    // GPU内存管理
    void* d_input_buffer_;
    void* d_output_buffer_;
    size_t max_batch_size_;
    size_t input_size_;
    size_t output_size_;

    // 流管理
    std::vector<cudaStream_t> streams_;
    int current_stream_;

    // 批处理队列
    std::queue<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 工作线程
    std::vector<std::thread> worker_threads_;
    bool running_;

    // 性能统计
    struct PerformanceStats {
        float avg_inference_time;
        float throughput;
        int total_requests;
        int failed_requests;
    } stats_;

public:
    GPUInferencePipeline(size_t max_batch_size, size_t input_size, size_t output_size);
    ~GPUInferencePipeline();

    // 初始化管道
    bool initialize();

    // 提交推理请求
    std::future<std::vector<float>> submitRequest(void* input_data,
                                                 const std::vector<int>& shape);

    // 批处理推理
    void processBatch(const std::vector<InferenceRequest>& batch);

    // 启动/停止管道
    void start();
    void stop();

    // 获取性能统计
    PerformanceStats getStats() const { return stats_; }

private:
    void workerThread();
    void collectBatch(std::vector<InferenceRequest>& batch);
    void executeInference(const std::vector<InferenceRequest>& batch);
    void updateStats(float inference_time, bool success);
};

// inference_pipeline.cpp
GPUInferencePipeline::GPUInferencePipeline(size_t max_batch_size,
                                          size_t input_size,
                                          size_t output_size)
    : max_batch_size_(max_batch_size), input_size_(input_size),
      output_size_(output_size), current_stream_(0), running_(false) {

    // 分配GPU内存
    cudaMalloc(&d_input_buffer_, max_batch_size_ * input_size_);
    cudaMalloc(&d_output_buffer_, max_batch_size_ * output_size_);

    // 创建CUDA流
    streams_.resize(4);
    for (auto& stream : streams_) {
        cudaStreamCreate(&stream);
    }

    // 初始化统计信息
    stats_ = {0.0f, 0.0f, 0, 0};
}

void GPUInferencePipeline::start() {
    running_ = true;

    // 启动工作线程
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back(&GPUInferencePipeline::workerThread, this);
    }
}

void GPUInferencePipeline::workerThread() {
    while (running_) {
        std::vector<InferenceRequest> batch;
        collectBatch(batch);

        if (!batch.empty()) {
            executeInference(batch);
        }
    }
}

void GPUInferencePipeline::executeInference(const std::vector<InferenceRequest>& batch) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 选择CUDA流
    cudaStream_t stream = streams_[current_stream_];
    current_stream_ = (current_stream_ + 1) % streams_.size();

    // 复制输入数据到GPU
    for (size_t i = 0; i < batch.size(); ++i) {
        void* dst = static_cast<char*>(d_input_buffer_) + i * input_size_;
        cudaMemcpyAsync(dst, batch[i].input_data, input_size_,
                       cudaMemcpyHostToDevice, stream);
    }

    // 执行推理（这里需要调用实际的推理引擎）
    // runInference(d_input_buffer_, d_output_buffer_, batch.size(), stream);

    // 复制输出数据回CPU
    std::vector<std::vector<float>> results(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        results[i].resize(output_size_ / sizeof(float));
        void* src = static_cast<char*>(d_output_buffer_) + i * output_size_;
        cudaMemcpyAsync(results[i].data(), src, output_size_,
                       cudaMemcpyDeviceToHost, stream);
    }

    // 等待完成
    cudaStreamSynchronize(stream);

    auto end_time = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // 返回结果
    for (size_t i = 0; i < batch.size(); ++i) {
        batch[i].result_promise.set_value(results[i]);
    }

    updateStats(inference_time, true);
}
```

### 内存池管理

```cpp
// memory_pool.h
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <unordered_map>

class GPUMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        cudaStream_t stream;
    };

    std::vector<MemoryBlock> blocks_;
    std::mutex pool_mutex_;
    size_t total_allocated_;
    size_t peak_usage_;

public:
    GPUMemoryPool();
    ~GPUMemoryPool();

    // 分配内存
    void* allocate(size_t size, cudaStream_t stream = 0);

    // 释放内存
    void deallocate(void* ptr);

    // 预分配内存池
    bool preallocate(const std::vector<size_t>& sizes);

    // 清理未使用的内存
    void cleanup();

    // 获取内存使用统计
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getPeakUsage() const { return peak_usage_; }

    // 内存碎片整理
    void defragment();

private:
    MemoryBlock* findFreeBlock(size_t size);
    void splitBlock(MemoryBlock* block, size_t size);
    void mergeBlocks();
};

// memory_pool.cpp
void* GPUMemoryPool::allocate(size_t size, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // 查找合适的空闲块
    MemoryBlock* block = findFreeBlock(size);

    if (block) {
        block->in_use = true;
        block->stream = stream;

        // 如果块太大，分割它
        if (block->size > size * 2) {
            splitBlock(block, size);
        }

        return block->ptr;
    }

    // 没有合适的块，分配新内存
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }

    MemoryBlock new_block;
    new_block.ptr = ptr;
    new_block.size = size;
    new_block.in_use = true;
    new_block.stream = stream;

    blocks_.push_back(new_block);
    total_allocated_ += size;
    peak_usage_ = std::max(peak_usage_, total_allocated_);

    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            block.stream = 0;

            // 尝试合并相邻的空闲块
            mergeBlocks();
            return;
        }
    }
}
```

## TensorRT集成和优化

### TensorRT引擎构建

```cpp
// tensorrt_engine.h
#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>

class TensorRTEngine {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::vector<void*> bindings_;
    std::vector<size_t> binding_sizes_;
    std::vector<nvinfer1::Dims> binding_dims_;

    cudaStream_t stream_;

public:
    TensorRTEngine();
    ~TensorRTEngine();

    // 从ONNX构建引擎
    bool buildFromONNX(const std::string& onnx_path,
                      const std::string& engine_path,
                      int max_batch_size = 1,
                      bool use_fp16 = false);

    // 加载预构建的引擎
    bool loadEngine(const std::string& engine_path);

    // 保存引擎
    bool saveEngine(const std::string& engine_path);

    // 执行推理
    bool infer(const std::vector<void*>& inputs, std::vector<void*>& outputs);

    // 获取绑定信息
    int getBindingIndex(const std::string& name);
    nvinfer1::Dims getBindingDimensions(int index);
    size_t getBindingSize(int index);

    // 性能优化
    void enableDynamicShapes(const std::vector<nvinfer1::Dims>& min_shapes,
                           const std::vector<nvinfer1::Dims>& opt_shapes,
                           const std::vector<nvinfer1::Dims>& max_shapes);

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    } logger_;

    bool allocateBindings();
    void deallocateBindings();
};

// tensorrt_engine.cpp
bool TensorRTEngine::buildFromONNX(const std::string& onnx_path,
                                  const std::string& engine_path,
                                  int max_batch_size,
                                  bool use_fp16) {

    // 创建构建器
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

    // 创建网络
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cerr << "Failed to create TensorRT network" << std::endl;
        return false;
    }

    // 创建ONNX解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return false;
    }

    // 解析ONNX模型
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return false;
    }

    // 创建构建配置
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }

    // 设置最大工作空间大小
    config->setMaxWorkspaceSize(1ULL << 30); // 1GB

    // 启用FP16精度
    if (use_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled" << std::endl;
    }

    // 构建引擎
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine_) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return false;
    }

    // 保存引擎
    if (!engine_path.empty()) {
        saveEngine(engine_path);
    }

    // 创建执行上下文
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // 分配绑定内存
    return allocateBindings();
}

bool TensorRTEngine::infer(const std::vector<void*>& inputs, std::vector<void*>& outputs) {
    if (!context_) {
        std::cerr << "Execution context not initialized" << std::endl;
        return false;
    }

    // 复制输入数据
    int input_count = 0;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            cudaMemcpyAsync(bindings_[i], inputs[input_count], binding_sizes_[i],
                           cudaMemcpyHostToDevice, stream_);
            input_count++;
        }
    }

    // 执行推理
    bool success = context_->enqueueV2(bindings_.data(), stream_, nullptr);
    if (!success) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return false;
    }

    // 复制输出数据
    int output_count = 0;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (!engine_->bindingIsInput(i)) {
            cudaMemcpyAsync(outputs[output_count], bindings_[i], binding_sizes_[i],
                           cudaMemcpyDeviceToHost, stream_);
            output_count++;
        }
    }

    // 同步流
    cudaStreamSynchronize(stream_);

    return true;
}
```

### 自定义TensorRT插件

```cpp
// custom_plugin.h
#pragma once
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

class YOLOLayerPlugin : public nvinfer1::IPluginV2DynamicExt {
private:
    int num_classes_;
    int num_anchors_;
    std::vector<float> anchors_;
    int input_width_;
    int input_height_;

public:
    YOLOLayerPlugin(int num_classes, int num_anchors, const std::vector<float>& anchors,
                   int input_width, int input_height);

    YOLOLayerPlugin(const void* data, size_t length);

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                           const nvinfer1::DimsExprs* inputs,
                                           int nbInputs,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                  int nbInputs, int nbOutputs) noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                           const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
               const nvinfer1::PluginTensorDesc* outputDesc,
               const void* const* inputs, void* const* outputs,
               void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                        int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override { return "YOLOLayer"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* pluginNamespace) noexcept override {}
    const char* getPluginNamespace() const noexcept override { return ""; }

private:
    void forwardGpu(const float* input, float* output, int batch_size,
                   int input_height, int input_width, cudaStream_t stream);
};

// YOLO层CUDA核函数
__global__ void yoloLayerKernel(const float* input, float* output,
                               int batch_size, int num_classes, int num_anchors,
                               int input_height, int input_width,
                               const float* anchors) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_anchors * input_height * input_width * (5 + num_classes);

    if (idx >= total_elements) return;

    // 计算索引
    int batch = idx / (num_anchors * input_height * input_width * (5 + num_classes));
    int remaining = idx % (num_anchors * input_height * input_width * (5 + num_classes));

    int anchor = remaining / (input_height * input_width * (5 + num_classes));
    remaining = remaining % (input_height * input_width * (5 + num_classes));

    int y = remaining / (input_width * (5 + num_classes));
    remaining = remaining % (input_width * (5 + num_classes));

    int x = remaining / (5 + num_classes);
    int channel = remaining % (5 + num_classes);

    float value = input[idx];

    if (channel == 0 || channel == 1) {
        // x, y坐标 - sigmoid激活
        output[idx] = 1.0f / (1.0f + expf(-value));
    } else if (channel == 2 || channel == 3) {
        // w, h - 指数激活
        int anchor_idx = anchor * 2 + (channel - 2);
        output[idx] = expf(value) * anchors[anchor_idx];
    } else if (channel == 4) {
        // 置信度 - sigmoid激活
        output[idx] = 1.0f / (1.0f + expf(-value));
    } else {
        // 类别概率 - sigmoid激活
        output[idx] = 1.0f / (1.0f + expf(-value));
    }
}

void YOLOLayerPlugin::forwardGpu(const float* input, float* output, int batch_size,
                                int input_height, int input_width, cudaStream_t stream) {

    int total_elements = batch_size * num_anchors_ * input_height * input_width * (5 + num_classes_);
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    // 复制anchors到GPU
    float* d_anchors;
    cudaMalloc(&d_anchors, anchors_.size() * sizeof(float));
    cudaMemcpyAsync(d_anchors, anchors_.data(), anchors_.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream);

    yoloLayerKernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, batch_size, num_classes_, num_anchors_,
        input_height, input_width, d_anchors);

    cudaFree(d_anchors);
}
```

## 性能监控和调优

### 实时性能监控

```cpp
// performance_monitor.h
#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <map>

struct PerformanceMetrics {
    float inference_time_ms;
    float preprocessing_time_ms;
    float postprocessing_time_ms;
    float total_time_ms;
    float gpu_utilization;
    float memory_usage_mb;
    float throughput_fps;
    int batch_size;
};

class PerformanceMonitor {
private:
    std::vector<PerformanceMetrics> metrics_history_;
    std::map<std::string, cudaEvent_t> start_events_;
    std::map<std::string, cudaEvent_t> stop_events_;

    // 性能统计
    struct Statistics {
        float avg_inference_time;
        float min_inference_time;
        float max_inference_time;
        float avg_throughput;
        float peak_memory_usage;
    } stats_;

public:
    PerformanceMonitor();
    ~PerformanceMonitor();

    // 开始计时
    void startTiming(const std::string& name);

    // 结束计时
    float stopTiming(const std::string& name);

    // 记录性能指标
    void recordMetrics(const PerformanceMetrics& metrics);

    // 获取统计信息
    Statistics getStatistics() const { return stats_; }

    // 生成性能报告
    std::string generateReport();

    // 检测性能瓶颈
    std::vector<std::string> detectBottlenecks();

    // 获取优化建议
    std::vector<std::string> getOptimizationSuggestions();

private:
    void updateStatistics();
    float getCurrentGPUUtilization();
    float getCurrentMemoryUsage();
};

// performance_monitor.cpp
void PerformanceMonitor::startTiming(const std::string& name) {
    if (start_events_.find(name) == start_events_.end()) {
        cudaEventCreate(&start_events_[name]);
        cudaEventCreate(&stop_events_[name]);
    }

    cudaEventRecord(start_events_[name]);
}

float PerformanceMonitor::stopTiming(const std::string& name) {
    if (start_events_.find(name) == start_events_.end()) {
        return -1.0f;
    }

    cudaEventRecord(stop_events_[name]);
    cudaEventSynchronize(stop_events_[name]);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_events_[name], stop_events_[name]);

    return elapsed_time;
}

std::vector<std::string> PerformanceMonitor::detectBottlenecks() {
    std::vector<std::string> bottlenecks;

    if (metrics_history_.empty()) {
        return bottlenecks;
    }

    // 分析最近的性能数据
    const auto& latest = metrics_history_.back();

    // 检测GPU利用率低
    if (latest.gpu_utilization < 70.0f) {
        bottlenecks.push_back("Low GPU utilization detected. Consider increasing batch size or optimizing memory access patterns.");
    }

    // 检测内存使用过高
    if (latest.memory_usage_mb > 8000.0f) { // 假设8GB为阈值
        bottlenecks.push_back("High memory usage detected. Consider reducing batch size or optimizing memory allocation.");
    }

    // 检测预处理时间过长
    if (latest.preprocessing_time_ms > latest.inference_time_ms * 0.5f) {
        bottlenecks.push_back("Preprocessing time is significant. Consider optimizing data loading and preprocessing pipeline.");
    }

    // 检测后处理时间过长
    if (latest.postprocessing_time_ms > latest.inference_time_ms * 0.3f) {
        bottlenecks.push_back("Postprocessing time is significant. Consider optimizing NMS and output processing.");
    }

    return bottlenecks;
}

std::vector<std::string> PerformanceMonitor::getOptimizationSuggestions() {
    std::vector<std::string> suggestions;

    if (metrics_history_.empty()) {
        return suggestions;
    }

    const auto& latest = metrics_history_.back();

    // 基于性能指标提供建议
    if (latest.throughput_fps < 30.0f) {
        suggestions.push_back("Consider using TensorRT FP16 precision for faster inference");
        suggestions.push_back("Implement dynamic batching to improve throughput");
        suggestions.push_back("Use CUDA streams for overlapping computation and memory transfer");
    }

    if (latest.batch_size == 1) {
        suggestions.push_back("Increase batch size to improve GPU utilization");
    }

    if (latest.memory_usage_mb < 4000.0f) {
        suggestions.push_back("GPU memory is underutilized. Consider increasing batch size or model complexity");
    }

    return suggestions;
}
```

### 自动调优系统

```cpp
// auto_tuner.h
#pragma once
#include "performance_monitor.h"
#include <vector>
#include <map>

struct TuningParameter {
    std::string name;
    float min_value;
    float max_value;
    float current_value;
    float step_size;
};

class AutoTuner {
private:
    std::vector<TuningParameter> parameters_;
    PerformanceMonitor* monitor_;

    // 调优历史
    struct TuningResult {
        std::map<std::string, float> parameter_values;
        float performance_score;
    };
    std::vector<TuningResult> tuning_history_;

    // 当前最佳配置
    std::map<std::string, float> best_config_;
    float best_score_;

public:
    AutoTuner(PerformanceMonitor* monitor);

    // 添加调优参数
    void addParameter(const std::string& name, float min_val, float max_val,
                     float initial_val, float step_size);

    // 开始自动调优
    void startTuning(int max_iterations = 100);

    // 评估当前配置
    float evaluateConfiguration();

    // 获取最佳配置
    std::map<std::string, float> getBestConfiguration() const { return best_config_; }

    // 应用配置
    void applyConfiguration(const std::map<std::string, float>& config);

private:
    // 网格搜索
    void gridSearch();

    // 随机搜索
    void randomSearch(int iterations);

    // 贝叶斯优化
    void bayesianOptimization(int iterations);

    // 计算性能分数
    float calculatePerformanceScore(const PerformanceMetrics& metrics);
};

// auto_tuner.cpp
void AutoTuner::startTuning(int max_iterations) {
    std::cout << "Starting auto-tuning with " << max_iterations << " iterations..." << std::endl;

    best_score_ = 0.0f;

    // 使用随机搜索进行调优
    randomSearch(max_iterations);

    std::cout << "Auto-tuning completed. Best score: " << best_score_ << std::endl;
    std::cout << "Best configuration:" << std::endl;
    for (const auto& param : best_config_) {
        std::cout << "  " << param.first << ": " << param.second << std::endl;
    }
}

void AutoTuner::randomSearch(int iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < iterations; ++i) {
        // 生成随机配置
        std::map<std::string, float> config;
        for (const auto& param : parameters_) {
            std::uniform_real_distribution<float> dist(param.min_value, param.max_value);
            config[param.name] = dist(gen);
        }

        // 应用配置
        applyConfiguration(config);

        // 评估性能
        float score = evaluateConfiguration();

        // 记录结果
        TuningResult result;
        result.parameter_values = config;
        result.performance_score = score;
        tuning_history_.push_back(result);

        // 更新最佳配置
        if (score > best_score_) {
            best_score_ = score;
            best_config_ = config;
            std::cout << "New best score: " << score << " at iteration " << i << std::endl;
        }
    }
}

float AutoTuner::calculatePerformanceScore(const PerformanceMetrics& metrics) {
    // 综合考虑多个性能指标
    float throughput_score = metrics.throughput_fps / 100.0f; // 归一化到0-1
    float utilization_score = metrics.gpu_utilization / 100.0f;
    float memory_efficiency = 1.0f - (metrics.memory_usage_mb / 16000.0f); // 假设16GB为满载

    // 加权平均
    float score = 0.5f * throughput_score + 0.3f * utilization_score + 0.2f * memory_efficiency;

    return std::max(0.0f, std::min(1.0f, score));
}
```

## 实践练习

### 练习1：YOLO模型加载
实现一个完整的YOLO模型加载器，支持ONNX和PyTorch格式。

### 练习2：批处理推理优化
开发一个高效的批处理推理系统，支持动态批大小。

### 练习3：TensorRT集成
将YOLO模型转换为TensorRT引擎，并实现自定义插件。

### 练习4：性能监控系统
构建一个实时性能监控和自动调优系统。

## 总结

本模块介绍了YOLO模型GPU加速的完整解决方案，包括：
- 模型解析和加载技术
- 高效的GPU推理管道设计
- TensorRT集成和自定义插件开发
- 实时性能监控和自动调优

通过这些技术，可以实现高性能的实时目标检测系统，满足工业级应用的需求。

---

**下一步**：学习[高级优化技术](06-advanced-optimization-techniques.md)

