#include "tensorrt_engine_builder.h"
#include "../utils/gpu_memory_manager.h"
#include "../../core/error_handler.h"

#include <fstream>
#include <iostream>
#include <cassert>

namespace yolo {
namespace tensorrt {

// TensorRT Logger Implementation
void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= log_level_) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[TensorRT INTERNAL_ERROR] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[TensorRT ERROR] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "[TensorRT WARNING] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[TensorRT INFO] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[TensorRT VERBOSE] " << msg << std::endl;
                break;
        }
    }
}

// TensorRT Engine Builder Implementation
TensorRTEngineBuilder::TensorRTEngineBuilder()
    : logger_(std::make_unique<TensorRTLogger>())
    , int8_calibrator_(nullptr) {

    // 创建TensorRT运行时
    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    std::cout << "TensorRT Engine Builder initialized successfully" << std::endl;
}

TensorRTEngineBuilder::~TensorRTEngineBuilder() {
    // 清理自定义插件
    custom_plugins_.clear();

    std::cout << "TensorRT Engine Builder destroyed" << std::endl;
}

std::shared_ptr<nvinfer1::ICudaEngine> TensorRTEngineBuilder::buildEngineFromONNX(
    const std::string& onnx_path,
    const BuildConfig& config) {

    std::cout << "Building TensorRT engine from ONNX: " << onnx_path << std::endl;

    // 创建网络定义
    auto [network, builder] = createNetworkDefinition(config);
    if (!network || !builder) {
        std::cerr << "Failed to create network definition" << std::endl;
        return nullptr;
    }

    // 创建ONNX解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, *logger_));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return nullptr;
    }

    // 解析ONNX模型
    if (!parser->parseFromFile(onnx_path.c_str(),
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model: " << onnx_path << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Parser error: " << parser->getError(i)->desc() << std::endl;
        }
        return nullptr;
    }

    std::cout << "ONNX model parsed successfully" << std::endl;

    // 优化网络
    optimizeNetwork(network, config);

    // 配置构建器
    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        configureBuilder(builder, config));
    if (!builder_config) {
        std::cerr << "Failed to configure builder" << std::endl;
        return nullptr;
    }

    // 构建引擎
    std::cout << "Building TensorRT engine..." << std::endl;
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *builder_config),
        [](nvinfer1::ICudaEngine* engine) {
            if (engine) engine->destroy();
        });

    if (!engine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return nullptr;
    }

    std::cout << "TensorRT engine built successfully" << std::endl;
    std::cout << getEngineInfo(engine.get()) << std::endl;

    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine> TensorRTEngineBuilder::buildEngineFromSerialized(
    const void* serialized_data,
    size_t data_size) {

    if (!serialized_data || data_size == 0) {
        std::cerr << "Invalid serialized data" << std::endl;
        return nullptr;
    }

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(serialized_data, data_size),
        [](nvinfer1::ICudaEngine* engine) {
            if (engine) engine->destroy();
        });

    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return nullptr;
    }

    std::cout << "TensorRT engine deserialized successfully" << std::endl;
    return engine;
}

bool TensorRTEngineBuilder::serializeEngine(
    const nvinfer1::ICudaEngine* engine,
    const std::string& output_path) {

    if (!engine) {
        std::cerr << "Invalid engine for serialization" << std::endl;
        return false;
    }

    // 序列化引擎
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        engine->serialize());
    if (!serialized_engine) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }

    // 写入文件
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return false;
    }

    file.write(static_cast<const char*>(serialized_engine->data()),
               serialized_engine->size());
    file.close();

    std::cout << "Engine serialized to: " << output_path
              << " (size: " << serialized_engine->size() << " bytes)" << std::endl;

    return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> TensorRTEngineBuilder::loadSerializedEngine(
    const std::string& engine_path) {

    // 读取序列化文件
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return nullptr;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);
    file.close();

    std::cout << "Loaded serialized engine from: " << engine_path
              << " (size: " << file_size << " bytes)" << std::endl;

    return buildEngineFromSerialized(buffer.data(), file_size);
}

void TensorRTEngineBuilder::registerCustomPlugin(
    const std::string& plugin_name,
    nvinfer1::IPluginCreator* plugin_creator) {

    if (!plugin_creator) {
        std::cerr << "Invalid plugin creator for: " << plugin_name << std::endl;
        return;
    }

    custom_plugins_[plugin_name] = plugin_creator;

    // 注册到TensorRT插件注册表
    auto plugin_registry = nvinfer1::getPluginRegistry();
    if (plugin_registry) {
        plugin_registry->registerCreator(*plugin_creator, "");
        std::cout << "Registered custom plugin: " << plugin_name << std::endl;
    }
}

void TensorRTEngineBuilder::setInt8Calibrator(nvinfer1::IInt8Calibrator* calibrator) {
    int8_calibrator_ = calibrator;
    std::cout << "INT8 calibrator set" << std::endl;
}

std::string TensorRTEngineBuilder::getEngineInfo(const nvinfer1::ICudaEngine* engine) {
    if (!engine) {
        return "Invalid engine";
    }

    std::ostringstream info;
    info << "=== TensorRT Engine Information ===" << std::endl;
    info << "Max Batch Size: " << engine->getMaxBatchSize() << std::endl;
    info << "Number of Bindings: " << engine->getNbBindings() << std::endl;

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        info << "Binding " << i << ": " << engine->getBindingName(i);
        info << " (";
        if (engine->bindingIsInput(i)) {
            info << "INPUT";
        } else {
            info << "OUTPUT";
        }
        info << ") - Shape: ";

        auto dims = engine->getBindingDimensions(i);
        info << "[";
        for (int j = 0; j < dims.nbDims; ++j) {
            if (j > 0) info << ", ";
            info << dims.d[j];
        }
        info << "]" << std::endl;
    }

    return info.str();
}

std::pair<nvinfer1::INetworkDefinition*, nvinfer1::IBuilder*>
TensorRTEngineBuilder::createNetworkDefinition(const BuildConfig& config) {

    // 创建构建器
    auto builder = nvinfer1::createInferBuilder(*logger_);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return {nullptr, nullptr};
    }

    // 创建网络定义
    uint32_t flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = builder->createNetworkV2(flags);
    if (!network) {
        std::cerr << "Failed to create network definition" << std::endl;
        builder->destroy();
        return {nullptr, nullptr};
    }

    return {network, builder};
}

nvinfer1::IBuilderConfig* TensorRTEngineBuilder::configureBuilder(
    nvinfer1::IBuilder* builder,
    const BuildConfig& config) {

    auto builder_config = builder->createBuilderConfig();
    if (!builder_config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return nullptr;
    }

    // 设置工作空间大小
    builder_config->setMaxWorkspaceSize(config.max_workspace_size);

    // 设置精度模式
    if (config.enable_fp16 && builder->platformHasFastFp16()) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 precision enabled" << std::endl;
    }

    if (config.enable_int8 && builder->platformHasFastInt8()) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (int8_calibrator_) {
            builder_config->setInt8Calibrator(int8_calibrator_);
        }
        std::cout << "INT8 precision enabled" << std::endl;
    }

    // 设置严格类型约束
    if (config.strict_type_constraints) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }

    // 设置性能分析
    if (config.enable_profiling) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kPROFILING_VERBOSITY_DETAILED);
    }

    // 设置DLA
    if (config.enable_dla && builder->getNbDLACores() > 0) {
        builder_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        builder_config->setDLACore(config.dla_core);
        std::cout << "DLA enabled on core " << config.dla_core << std::endl;
    }

    // 设置动态形状
    if (config.enable_dynamic_shapes) {
        setupDynamicShapes(builder_config, config);
    }

    return builder_config;
}

void TensorRTEngineBuilder::setupDynamicShapes(
    nvinfer1::IBuilderConfig* builder_config,
    const BuildConfig& config) {

    if (config.dynamic_shapes.empty()) {
        std::cout << "No dynamic shapes configured" << std::endl;
        return;
    }

    auto profile = builder_config->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Failed to create optimization profile" << std::endl;
        return;
    }

    for (const auto& shape_config : config.dynamic_shapes) {
        // 转换形状向量为TensorRT Dims
        auto toDims = [](const std::vector<int>& shape) {
            nvinfer1::Dims dims;
            dims.nbDims = shape.size();
            for (size_t i = 0; i < shape.size(); ++i) {
                dims.d[i] = shape[i];
            }
            return dims;
        };

        nvinfer1::Dims min_dims = toDims(shape_config.min_shape);
        nvinfer1::Dims opt_dims = toDims(shape_config.opt_shape);
        nvinfer1::Dims max_dims = toDims(shape_config.max_shape);

        profile->setDimensions(shape_config.tensor_name.c_str(),
                              nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(shape_config.tensor_name.c_str(),
                              nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(shape_config.tensor_name.c_str(),
                              nvinfer1::OptProfileSelector::kMAX, max_dims);

        std::cout << "Dynamic shape configured for " << shape_config.tensor_name << std::endl;
    }

    builder_config->addOptimizationProfile(profile);
    std::cout << "Dynamic shapes optimization profile added" << std::endl;
}

void TensorRTEngineBuilder::optimizeNetwork(
    nvinfer1::INetworkDefinition* network,
    const BuildConfig& config) {

    std::cout << "Optimizing network..." << std::endl;

    // 网络优化策略
    // 1. 标记输出张量
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto output = network->getOutput(i);
        if (output) {
            output->setName(("output_" + std::to_string(i)).c_str());
        }
    }

    // 2. 设置精度约束（如果需要）
    if (config.strict_type_constraints) {
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto layer = network->getLayer(i);
            if (layer) {
                // 为特定层设置精度约束
                if (config.enable_fp16) {
                    layer->setPrecision(nvinfer1::DataType::kHALF);
                }
            }
        }
    }

    std::cout << "Network optimization completed" << std::endl;
}

} // namespace tensorrt
} // namespace yolo
