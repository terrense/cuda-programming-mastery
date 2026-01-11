#pragma once

#include "tensorrt_engine_builder.h"
#include "custom_plugins.h"
#include "dynamic_shape_handler.h"
#include "../utils/gpu_memory_manager.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <future>

namespace yolo {
namespace tensorrt {

/**
 * @brief TensorRT YOLO加速器
 *
 * 集成TensorRT引擎构建、自定义插件和动态形状处理的完整YOLO加速解决方案
 */
class TensorRTYOLOAccelerator {
public:
    /**
     * @brief 加速器配置
     */
    struct AcceleratorConfig {
        // 引擎构建配置
        TensorRTEngineBuilder::BuildConfig build_config;

        // 动态形状配置
        std::vector<DynamicShapeHandler::ShapeConfig> shape_configs;

        // 批处理配置
        DynamicBatchManager::BatchConfig batch_config;

        // 性能配置
        bool enable_profiling = false;
        bool enable_optimization = true;
        int num_execution_contexts = 2;

        // 缓存配置
        std::string engine_cache_path = "./yolo_engine.trt";
        bool use_engine_cache = true;
        bool force_rebuild = false;
    };

    /**
     * @brief 检测结果
     */
    struct Detection {
        float x, y, w, h;      // 边界框坐标 (归一化)
        float confidence;      // 置信度
        int class_id;          // 类别ID
        std::string class_name; // 类别名称
    };

    /**
     * @brief 推理结果
     */
    struct InferenceResult {
        std::vector<Detection> detections;
        float inference_time_ms;
        int batch_size;
        std::map<std::string, float> performance_metrics;
    };

public:
    /**
     * @brief 构造函数
     * @param config 加速器配置
     */
    explicit TensorRTYOLOAccelerator(const AcceleratorConfig& config);

    /**
     * @brief 析构函数
     */
    ~TensorRTYOLOAccelerator();

    /**
     * @brief 从ONNX模型初始化
     * @param onnx_path ONNX模型文件路径
     * @param class_names 类别名称列表
     * @return 是否成功
     */
    bool initializeFromONNX(const std::string& onnx_path,
                           const std::vector<std::string>& class_names = {});

    /**
     * @brief 从序列化引擎初始化
     * @param engine_path 引擎文件路径
     * @param class_names 类别名称列表
     * @return 是否成功
     */
    bool initializeFromEngine(const std::string& engine_path,
                             const std::vector<std::string>& class_names = {});

    /**
     * @brief 单张图像推理
     * @param image_data 图像数据 (CHW格式)
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     * @return 推理结果
     */
    InferenceResult inferSingle(const float* image_data,
                               int width, int height, int channels = 3);

    /**
     * @brief 批量图像推理
     * @param batch_images 批量图像数据
     * @param batch_size 批次大小
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     * @return 推理结果
     */
    InferenceResult inferBatch(const std::vector<const float*>& batch_images,
                              int batch_size, int width, int height, int channels = 3);

    /**
     * @brief 异步推理
     * @param image_data 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     * @return 推理结果的future
     */
    std::future<InferenceResult> inferAsync(const float* image_data,
                                           int width, int height, int channels = 3);

    /**
     * @brief 动态批处理推理
     * @param image_data 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     * @param callback 完成回调
     * @return 请求ID
     */
    int inferDynamicBatch(const float* image_data, int width, int height, int channels,
                         std::function<void(const InferenceResult&)> callback);

    /**
     * @brief 设置输入形状
     * @param batch_size 批次大小
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     * @return 是否成功
     */
    bool setInputShape(int batch_size, int width, int height, int channels = 3);

    /**
     * @brief 预热引擎
     * @param warmup_iterations 预热迭代次数
     * @return 是否成功
     */
    bool warmupEngine(int warmup_iterations = 10);

    /**
     * @brief 获取性能统计
     * @return 性能统计信息
     */
    std::map<std::string, float> getPerformanceStats() const;

    /**
     * @brief 获取引擎信息
     * @return 引擎信息字符串
     */
    std::string getEngineInfo() const;

    /**
     * @brief 保存引擎到文件
     * @param engine_path 保存路径
     * @return 是否成功
     */
    bool saveEngine(const std::string& engine_path);

    /**
     * @brief 设置置信度阈值
     * @param threshold 置信度阈值
     */
    void setConfidenceThreshold(float threshold);

    /**
     * @brief 设置NMS阈值
     * @param threshold NMS阈值
     */
    void setNMSThreshold(float threshold);

    /**
     * @brief 启用/禁用性能分析
     * @param enable 是否启用
     */
    void enableProfiling(bool enable);

private:
    /**
     * @brief 初始化TensorRT组件
     * @return 是否成功
     */
    bool initializeTensorRT();

    /**
     * @brief 注册自定义插件
     */
    void registerCustomPlugins();

    /**
     * @brief 配置动态形状
     * @return 是否成功
     */
    bool configureDynamicShapes();

    /**
     * @brief 预处理输入数据
     * @param image_data 原始图像数据
     * @param processed_data 预处理后的数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     */
    void preprocessInput(const float* image_data, float* processed_data,
                        int width, int height, int channels);

    /**
     * @brief 后处理输出数据
     * @param output_data 原始输出数据
     * @param batch_size 批次大小
     * @return 检测结果
     */
    std::vector<std::vector<Detection>> postprocessOutput(const float* output_data,
                                                         int batch_size);

    /**
     * @brief 执行NMS
     * @param detections 检测结果
     * @param nms_threshold NMS阈值
     * @return 过滤后的检测结果
     */
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections,
                                   float nms_threshold);

    /**
     * @brief 计算IoU
     * @param det1 检测框1
     * @param det2 检测框2
     * @return IoU值
     */
    float calculateIoU(const Detection& det1, const Detection& det2);

    /**
     * @brief 分配GPU内存
     * @param size 内存大小
     * @return GPU内存指针
     */
    void* allocateGPUMemory(size_t size);

    /**
     * @brief 释放GPU内存
     * @param ptr 内存指针
     */
    void freeGPUMemory(void* ptr);

private:
    AcceleratorConfig config_;

    // TensorRT组件
    std::unique_ptr<TensorRTEngineBuilder> engine_builder_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<DynamicShapeHandler> shape_handler_;
    std::unique_ptr<DynamicBatchManager> batch_manager_;

    // GPU内存管理
    std::shared_ptr<utils::GPUMemoryManager> memory_manager_;

    // 执行上下文
    std::vector<int> execution_context_ids_;
    int current_context_id_;

    // 模型信息
    std::vector<std::string> class_names_;
    int num_classes_;
    float confidence_threshold_;
    float nms_threshold_;

    // 输入输出信息
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;

    // GPU内存缓冲区
    void* input_buffer_gpu_;
    void* output_buffer_gpu_;
    size_t input_buffer_size_;
    size_t output_buffer_size_;

    // CUDA流
    cudaStream_t inference_stream_;

    // 性能统计
    mutable std::mutex stats_mutex_;
    std::map<std::string, float> performance_stats_;

    // 状态标志
    bool initialized_;
    bool profiling_enabled_;
};

/**
 * @brief TensorRT YOLO加速器工厂
 */
class TensorRTYOLOAcceleratorFactory {
public:
    /**
     * @brief 创建YOLOv5加速器
     * @param onnx_path ONNX模型路径
     * @param config 配置参数
     * @return 加速器实例
     */
    static std::unique_ptr<TensorRTYOLOAccelerator> createYOLOv5Accelerator(
        const std::string& onnx_path,
        const TensorRTYOLOAccelerator::AcceleratorConfig& config = {});

    /**
     * @brief 创建YOLOv8加速器
     * @param onnx_path ONNX模型路径
     * @param config 配置参数
     * @return 加速器实例
     */
    static std::unique_ptr<TensorRTYOLOAccelerator> createYOLOv8Accelerator(
        const std::string& onnx_path,
        const TensorRTYOLOAccelerator::AcceleratorConfig& config = {});

    /**
     * @brief 从配置文件创建加速器
     * @param config_path 配置文件路径
     * @return 加速器实例
     */
    static std::unique_ptr<TensorRTYOLOAccelerator> createFromConfig(
        const std::string& config_path);

private:
    /**
     * @brief 获取默认YOLOv5配置
     */
    static TensorRTYOLOAccelerator::AcceleratorConfig getDefaultYOLOv5Config();

    /**
     * @brief 获取默认YOLOv8配置
     */
    static TensorRTYOLOAccelerator::AcceleratorConfig getDefaultYOLOv8Config();
};

} // namespace tensorrt
} // namespace yolo
