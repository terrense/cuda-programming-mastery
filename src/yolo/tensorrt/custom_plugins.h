#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <vector>
#include <string>
#include <memory>

namespace yolo {
namespace tensorrt {

/**
 * @brief 自定义YOLO检测层插件
 *
 * 实现YOLO模型的检测层，包括边界框解码和NMS后处理
 */
class YOLODetectionPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    /**
     * @brief 构造函数
     * @param num_classes 类别数量
     * @param num_anchors 锚点数量
     * @param conf_threshold 置信度阈值
     * @param nms_threshold NMS阈值
     */
    YOLODetectionPlugin(int num_classes, int num_anchors,
                       float conf_threshold, float nms_threshold);

    /**
     * @brief 拷贝构造函数
     */
    YOLODetectionPlugin(const YOLODetectionPlugin& other);

    /**
     * @brief 析构函数
     */
    ~YOLODetectionPlugin() override = default;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    int num_classes_;
    int num_anchors_;
    float conf_threshold_;
    float nms_threshold_;
    std::string namespace_;

    // CUDA kernel launch parameters
    struct LaunchParams {
        int grid_x, grid_y, grid_z;
        int block_x, block_y, block_z;
    };

    LaunchParams calculateLaunchParams(int batch_size, int num_detections) const;
};

/**
 * @brief YOLO检测层插件创建器
 */
class YOLODetectionPluginCreator : public nvinfer1::IPluginCreator {
public:
    YOLODetectionPluginCreator();
    ~YOLODetectionPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection field_collection_;
    std::vector<nvinfer1::PluginField> plugin_fields_;
    std::string namespace_;
};

/**
 * @brief 自定义Focus层插件
 *
 * 实现YOLOv5的Focus层，将空间信息压缩到通道维度
 */
class FocusPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    FocusPlugin() = default;
    FocusPlugin(const FocusPlugin& other) = default;
    ~FocusPlugin() override = default;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string namespace_;
};

/**
 * @brief Focus层插件创建器
 */
class FocusPluginCreator : public nvinfer1::IPluginCreator {
public:
    FocusPluginCreator();
    ~FocusPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection field_collection_;
    std::vector<nvinfer1::PluginField> plugin_fields_;
    std::string namespace_;
};

/**
 * @brief 插件管理器
 *
 * 负责注册和管理所有自定义插件
 */
class PluginManager {
public:
    /**
     * @brief 获取单例实例
     */
    static PluginManager& getInstance();

    /**
     * @brief 注册所有YOLO相关插件
     */
    void registerYOLOPlugins();

    /**
     * @brief 注册单个插件
     * @param creator 插件创建器
     */
    void registerPlugin(std::unique_ptr<nvinfer1::IPluginCreator> creator);

    /**
     * @brief 获取插件创建器
     * @param plugin_name 插件名称
     * @return 插件创建器指针
     */
    nvinfer1::IPluginCreator* getPluginCreator(const std::string& plugin_name);

    /**
     * @brief 列出所有已注册的插件
     */
    std::vector<std::string> listRegisteredPlugins() const;

private:
    PluginManager() = default;
    ~PluginManager() = default;

    std::vector<std::unique_ptr<nvinfer1::IPluginCreator>> plugin_creators_;
};

} // namespace tensorrt
} // namespace yolo
