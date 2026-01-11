#include "custom_plugins.h"
#include "plugin_kernels.cuh"
#include <iostream>
#include <cstring>
#include <algorithm>

namespace yolo {
namespace tensorrt {

// YOLODetectionPlugin Implementation
YOLODetectionPlugin::YOLODetectionPlugin(int num_classes, int num_anchors,
                                       float conf_threshold, float nms_threshold)
    : num_classes_(num_classes)
    , num_anchors_(num_anchors)
    , conf_threshold_(conf_threshold)
    , nms_threshold_(nms_threshold) {
}

YOLODetectionPlugin::YOLODetectionPlugin(const YOLODetectionPlugin& other)
    : num_classes_(other.num_classes_)
    , num_anchors_(other.num_anchors_)
    , conf_threshold_(other.conf_threshold_)
    , nms_threshold_(other.nms_threshold_)
    , namespace_(other.namespace_) {
}

nvinfer1::IPluginV2DynamicExt* YOLODetectionPlugin::clone() const noexcept {
    return new YOLODetectionPlugin(*this);
}

nvinfer1::DimsExprs YOLODetectionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs,
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {

    // 输出维度: [batch_size, max_detections, 6] (x, y, w, h, conf, class)
    nvinfer1::DimsExprs output;
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0]; // batch_size
    output.d[1] = exprBuilder.constant(1000); // max_detections
    output.d[2] = exprBuilder.constant(6); // detection info

    return output;
}

bool YOLODetectionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs, int nbOutputs) noexcept {

    // 支持FP32和FP16格式
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kHALF) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void YOLODetectionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    // 配置插件参数
}

size_t YOLODetectionPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {

    // 计算工作空间大小
    int batch_size = inputs[0].dims.d[0];
    int max_detections = 1000;

    // 需要临时存储空间用于NMS
    size_t workspace_size = batch_size * max_detections * 6 * sizeof(float);
    return workspace_size;
}

int YOLODetectionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    int batch_size = inputDesc[0].dims.d[0];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    // 启动CUDA核函数
    auto launch_params = calculateLaunchParams(batch_size, height * width);

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        launchYOLODetectionKernel(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]),
            static_cast<float*>(workspace),
            batch_size, height, width,
            num_classes_, num_anchors_,
            conf_threshold_, nms_threshold_,
            launch_params.grid_x, launch_params.grid_y,
            launch_params.block_x, launch_params.block_y,
            stream);
    } else {
        // FP16 version
        launchYOLODetectionKernelHalf(
            static_cast<const __half*>(inputs[0]),
            static_cast<__half*>(outputs[0]),
            static_cast<__half*>(workspace),
            batch_size, height, width,
            num_classes_, num_anchors_,
            conf_threshold_, nms_threshold_,
            launch_params.grid_x, launch_params.grid_y,
            launch_params.block_x, launch_params.block_y,
            stream);
    }

    return 0;
}

nvinfer1::DataType YOLODetectionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

const char* YOLODetectionPlugin::getPluginType() const noexcept {
    return "YOLODetection";
}

const char* YOLODetectionPlugin::getPluginVersion() const noexcept {
    return "1.0";
}

int YOLODetectionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int YOLODetectionPlugin::initialize() noexcept {
    return 0;
}

void YOLODetectionPlugin::terminate() noexcept {
}

size_t YOLODetectionPlugin::getSerializationSize() const noexcept {
    return sizeof(num_classes_) + sizeof(num_anchors_) +
           sizeof(conf_threshold_) + sizeof(nms_threshold_);
}

void YOLODetectionPlugin::serialize(void* buffer) const noexcept {
    char* data = static_cast<char*>(buffer);
    size_t offset = 0;

    std::memcpy(data + offset, &num_classes_, sizeof(num_classes_));
    offset += sizeof(num_classes_);

    std::memcpy(data + offset, &num_anchors_, sizeof(num_anchors_));
    offset += sizeof(num_anchors_);

    std::memcpy(data + offset, &conf_threshold_, sizeof(conf_threshold_));
    offset += sizeof(conf_threshold_);

    std::memcpy(data + offset, &nms_threshold_, sizeof(nms_threshold_));
}

void YOLODetectionPlugin::destroy() noexcept {
    delete this;
}

void YOLODetectionPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* YOLODetectionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

YOLODetectionPlugin::LaunchParams YOLODetectionPlugin::calculateLaunchParams(
    int batch_size, int num_detections) const {

    LaunchParams params;
    params.block_x = 256;
    params.block_y = 1;
    params.block_z = 1;

    params.grid_x = (num_detections + params.block_x - 1) / params.block_x;
    params.grid_y = batch_size;
    params.grid_z = 1;

    return params;
}

// YOLODetectionPluginCreator Implementation
YOLODetectionPluginCreator::YOLODetectionPluginCreator() {
    plugin_fields_.emplace_back("num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_fields_.emplace_back("num_anchors", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    plugin_fields_.emplace_back("conf_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    plugin_fields_.emplace_back("nms_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);

    field_collection_.nbFields = plugin_fields_.size();
    field_collection_.fields = plugin_fields_.data();
}

const char* YOLODetectionPluginCreator::getPluginName() const noexcept {
    return "YOLODetection";
}

const char* YOLODetectionPluginCreator::getPluginVersion() const noexcept {
    return "1.0";
}

const nvinfer1::PluginFieldCollection* YOLODetectionPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV2* YOLODetectionPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {

    int num_classes = 80;
    int num_anchors = 3;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;

    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& field = fc->fields[i];
        if (std::strcmp(field.name, "num_classes") == 0) {
            num_classes = *static_cast<const int*>(field.data);
        } else if (std::strcmp(field.name, "num_anchors") == 0) {
            num_anchors = *static_cast<const int*>(field.data);
        } else if (std::strcmp(field.name, "conf_threshold") == 0) {
            conf_threshold = *static_cast<const float*>(field.data);
        } else if (std::strcmp(field.name, "nms_threshold") == 0) {
            nms_threshold = *static_cast<const float*>(field.data);
        }
    }

    return new YOLODetectionPlugin(num_classes, num_anchors, conf_threshold, nms_threshold);
}

nvinfer1::IPluginV2* YOLODetectionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {

    const char* data = static_cast<const char*>(serialData);
    size_t offset = 0;

    int num_classes;
    std::memcpy(&num_classes, data + offset, sizeof(num_classes));
    offset += sizeof(num_classes);

    int num_anchors;
    std::memcpy(&num_anchors, data + offset, sizeof(num_anchors));
    offset += sizeof(num_anchors);

    float conf_threshold;
    std::memcpy(&conf_threshold, data + offset, sizeof(conf_threshold));
    offset += sizeof(conf_threshold);

    float nms_threshold;
    std::memcpy(&nms_threshold, data + offset, sizeof(nms_threshold));

    return new YOLODetectionPlugin(num_classes, num_anchors, conf_threshold, nms_threshold);
}

void YOLODetectionPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* YOLODetectionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// FocusPlugin Implementation
nvinfer1::IPluginV2DynamicExt* FocusPlugin::clone() const noexcept {
    return new FocusPlugin(*this);
}

nvinfer1::DimsExprs FocusPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs,
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {

    // Focus层输出: [B, C*4, H/2, W/2]
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0]; // batch_size
    output.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                                       *inputs[0].d[1], *exprBuilder.constant(4)); // channels * 4
    output.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
                                       *inputs[0].d[2], *exprBuilder.constant(2)); // height / 2
    output.d[3] = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
                                       *inputs[0].d[3], *exprBuilder.constant(2)); // width / 2

    return output;
}

bool FocusPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs, int nbOutputs) noexcept {

    return (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kHALF) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void FocusPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
}

size_t FocusPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0; // Focus层不需要额外工作空间
}

int FocusPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {

    int batch_size = inputDesc[0].dims.d[0];
    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        launchFocusKernel(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]),
            batch_size, channels, height, width,
            stream);
    } else {
        launchFocusKernelHalf(
            static_cast<const __half*>(inputs[0]),
            static_cast<__half*>(outputs[0]),
            batch_size, channels, height, width,
            stream);
    }

    return 0;
}

nvinfer1::DataType FocusPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

const char* FocusPlugin::getPluginType() const noexcept {
    return "Focus";
}

const char* FocusPlugin::getPluginVersion() const noexcept {
    return "1.0";
}

int FocusPlugin::getNbOutputs() const noexcept {
    return 1;
}

int FocusPlugin::initialize() noexcept {
    return 0;
}

void FocusPlugin::terminate() noexcept {
}

size_t FocusPlugin::getSerializationSize() const noexcept {
    return 0; // Focus层没有参数需要序列化
}

void FocusPlugin::serialize(void* buffer) const noexcept {
}

void FocusPlugin::destroy() noexcept {
    delete this;
}

void FocusPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* FocusPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// FocusPluginCreator Implementation
FocusPluginCreator::FocusPluginCreator() {
    field_collection_.nbFields = 0;
    field_collection_.fields = nullptr;
}

const char* FocusPluginCreator::getPluginName() const noexcept {
    return "Focus";
}

const char* FocusPluginCreator::getPluginVersion() const noexcept {
    return "1.0";
}

const nvinfer1::PluginFieldCollection* FocusPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV2* FocusPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
    return new FocusPlugin();
}

nvinfer1::IPluginV2* FocusPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
    return new FocusPlugin();
}

void FocusPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    namespace_ = pluginNamespace;
}

const char* FocusPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

// PluginManager Implementation
PluginManager& PluginManager::getInstance() {
    static PluginManager instance;
    return instance;
}

void PluginManager::registerYOLOPlugins() {
    std::cout << "Registering YOLO custom plugins..." << std::endl;

    // 注册YOLO检测层插件
    registerPlugin(std::make_unique<YOLODetectionPluginCreator>());

    // 注册Focus层插件
    registerPlugin(std::make_unique<FocusPluginCreator>());

    std::cout << "YOLO custom plugins registered successfully" << std::endl;
}

void PluginManager::registerPlugin(std::unique_ptr<nvinfer1::IPluginCreator> creator) {
    if (!creator) {
        std::cerr << "Invalid plugin creator" << std::endl;
        return;
    }

    // 注册到TensorRT插件注册表
    auto plugin_registry = nvinfer1::getPluginRegistry();
    if (plugin_registry) {
        plugin_registry->registerCreator(*creator, "");
        std::cout << "Registered plugin: " << creator->getPluginName() << std::endl;
    }

    plugin_creators_.push_back(std::move(creator));
}

nvinfer1::IPluginCreator* PluginManager::getPluginCreator(const std::string& plugin_name) {
    for (const auto& creator : plugin_creators_) {
        if (creator->getPluginName() == plugin_name) {
            return creator.get();
        }
    }
    return nullptr;
}

std::vector<std::string> PluginManager::listRegisteredPlugins() const {
    std::vector<std::string> plugin_names;
    for (const auto& creator : plugin_creators_) {
        plugin_names.push_back(creator->getPluginName());
    }
    return plugin_names;
}

} // namespace tensorrt
} // namespace yolo
