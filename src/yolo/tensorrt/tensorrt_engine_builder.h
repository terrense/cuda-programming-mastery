#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace yolo {
namespace tensorrt {

/**
 * @brief TensorRT引擎构建器
 *
 * 负责从ONNX模型构建TensorRT引擎，支持动态形状和自定义插件
 */
class TensorRTEngineBuilder {
public:
    /**
     * @brief 构建配置参数
     */
    struct BuildConfig {
        // 基础配置
        int max_batch_size = 8;
        size_t max_workspace_size = 1ULL << 30; // 1GB

        // 精度配置
        bool enable_fp16 = true;
        bool enable_int8 = false;

        // 动态形状配置
        bool enable_dynamic_shapes = true;
        struct DynamicShapeProfile {
            std::string tensor_name;
            std::vector<int> min_shape;
            std::vector<int> opt_shape;
            std::vector<int> max_shape;
        };
        std::vector<DynamicShapeProfile> dynamic_shapes;

        // 优化配置
        bool enable_dla = false;
        int dla_core = 0;
        bool strict_type_constraints = false;

        // 调试配置
        bool enable_profiling = false;
        bool enable_debug = false;
    };

    /**
     * @brief 构造函数
     */
    TensorRTEngineBuilder();

    /**
     * @brief 析构函数
     */
    ~TensorRTEngineBuilder();

    /**
     * @brief 从ONNX文件构建TensorRT引擎
     * @param onnx_path ONNX模型文件路径
     * @param config 构建配置
     * @return 构建的TensorRT引擎
     */
    std::shared_ptr<nvinfer1::ICudaEngine> buildEngineFromONNX(
        const std::string& onnx_path,
        const BuildConfig& config);

    /**
     * @brief 从序列化数据构建TensorRT引擎
     * @param serialized_data 序列化的引擎数据
     * @param data_size 数据大小
     * @return 构建的TensorRT引擎
     */
    std::shared_ptr<nvinfer1::ICudaEngine> buildEngineFromSerialized(
        const void* serialized_data,
        size_t data_size);

    /**
     * @brief 序列化TensorRT引擎
     * @param engine TensorRT引擎
     * @param output_path 输出文件路径
     * @return 是否成功
     */
    bool serializeEngine(
        const nvinfer1::ICudaEngine* engine,
        const std::string& output_path);

    /**
     * @brief 从文件加载序列化的引擎
     * @param engine_path 引擎文件路径
     * @return 加载的TensorRT引擎
     */
    std::shared_ptr<nvinfer1::ICudaEngine> loadSerializedEngine(
        const std::string& engine_path);

    /**
     * @brief 注册自定义插件
     * @param plugin_name 插件名称
     * @param plugin_creator 插件创建器
     */
    void registerCustomPlugin(
        const std::string& plugin_name,
        nvinfer1::IPluginCreator* plugin_creator);

    /**
     * @brief 设置INT8校准器
     * @param calibrator INT8校准器
     */
    void setInt8Calibrator(nvinfer1::IInt8Calibrator* calibrator);

    /**
     * @brief 获取引擎信息
     * @param engine TensorRT引擎
     * @return 引擎信息字符串
     */
    std::string getEngineInfo(const nvinfer1::ICudaEngine* engine);

private:
    /**
     * @brief 创建网络定义
     * @param config 构建配置
     * @return 网络定义和构建器
     */
    std::pair<nvinfer1::INetworkDefinition*, nvinfer1::IBuilder*>
    createNetworkDefinition(const BuildConfig& config);

    /**
     * @brief 配置构建器
     * @param builder TensorRT构建器
     * @param config 构建配置
     * @return 构建器配置
     */
    nvinfer1::IBuilderConfig* configureBuilder(
        nvinfer1::IBuilder* builder,
        const BuildConfig& config);

    /**
     * @brief 设置动态形状配置
     * @param builder_config 构建器配置
     * @param config 构建配置
     */
    void setupDynamicShapes(
        nvinfer1::IBuilderConfig* builder_config,
        const BuildConfig& config);

    /**
     * @brief 优化网络
     * @param network 网络定义
     * @param config 构建配置
     */
    void optimizeNetwork(
        nvinfer1::INetworkDefinition* network,
        const BuildConfig& config);

private:
    std::unique_ptr<nvinfer1::ILogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::map<std::string, nvinfer1::IPluginCreator*> custom_plugins_;
    nvinfer1::IInt8Calibrator* int8_calibrator_;
};

/**
 * @brief TensorRT日志记录器
 */
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;

    void setLogLevel(Severity level) { log_level_ = level; }

private:
    Severity log_level_ = Severity::kWARNING;
};

} // namespace tensorrt
} // namespace yolo
