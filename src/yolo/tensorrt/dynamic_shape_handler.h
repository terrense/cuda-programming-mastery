#pragma once

#include <NvInfer.h>
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace yolo {
namespace tensorrt {

/**
 * @brief 动态形状处理器
 *
 * 负责处理TensorRT引擎的动态输入形状，包括形状推断、内存管理和执行上下文优化
 */
class DynamicShapeHandler {
public:
    /**
     * @brief 形状配置
     */
    struct ShapeConfig {
        std::string tensor_name;
        std::vector<int> min_shape;
        std::vector<int> opt_shape;
        std::vector<int> max_shape;

        // 验证形状配置是否有效
        bool isValid() const;

        // 检查给定形状是否在范围内
        bool isInRange(const std::vector<int>& shape) const;
    };

    /**
     * @brief 执行上下文信息
     */
    struct ExecutionContext {
        std::shared_ptr<nvinfer1::IExecutionContext> context;
        std::map<std::string, std::vector<int>> current_shapes;
        size_t workspace_size;
        void* workspace_ptr;

        ExecutionContext() : workspace_size(0), workspace_ptr(nullptr) {}
    };

public:
    /**
     * @brief 构造函数
     * @param engine TensorRT引擎
     */
    explicit DynamicShapeHandler(std::shared_ptr<nvinfer1::ICudaEngine> engine);

    /**
     * @brief 析构函数
     */
    ~DynamicShapeHandler();

    /**
     * @brief 添加形状配置
     * @param config 形状配置
     * @return 是否成功
     */
    bool addShapeConfig(const ShapeConfig& config);

    /**
     * @brief 创建执行上下文
     * @param context_name 上下文名称
     * @return 执行上下文ID
     */
    int createExecutionContext(const std::string& context_name = "default");

    /**
     * @brief 设置输入形状
     * @param context_id 执行上下文ID
     * @param tensor_name 张量名称
     * @param shape 输入形状
     * @return 是否成功
     */
    bool setInputShape(int context_id, const std::string& tensor_name,
                      const std::vector<int>& shape);

    /**
     * @brief 批量设置输入形状
     * @param context_id 执行上下文ID
     * @param shapes 形状映射 (张量名 -> 形状)
     * @return 是否成功
     */
    bool setInputShapes(int context_id,
                       const std::map<std::string, std::vector<int>>& shapes);

    /**
     * @brief 获取输出形状
     * @param context_id 执行上下文ID
     * @param tensor_name 张量名称
     * @return 输出形状
     */
    std::vector<int> getOutputShape(int context_id, const std::string& tensor_name);

    /**
     * @brief 获取所有输出形状
     * @param context_id 执行上下文ID
     * @return 形状映射 (张量名 -> 形状)
     */
    std::map<std::string, std::vector<int>> getAllOutputShapes(int context_id);

    /**
     * @brief 执行推理
     * @param context_id 执行上下文ID
     * @param input_buffers 输入缓冲区
     * @param output_buffers 输出缓冲区
     * @param stream CUDA流
     * @return 是否成功
     */
    bool executeInference(int context_id,
                         const std::map<std::string, void*>& input_buffers,
                         const std::map<std::string, void*>& output_buffers,
                         cudaStream_t stream = nullptr);

    /**
     * @brief 异步执行推理
     * @param context_id 执行上下文ID
     * @param input_buffers 输入缓冲区
     * @param output_buffers 输出缓冲区
     * @param stream CUDA流
     * @return 是否成功
     */
    bool executeInferenceAsync(int context_id,
                              const std::map<std::string, void*>& input_buffers,
                              const std::map<std::string, void*>& output_buffers,
                              cudaStream_t stream);

    /**
     * @brief 优化执行上下文
     * @param context_id 执行上下文ID
     * @param target_shapes 目标形状
     * @return 是否成功
     */
    bool optimizeContext(int context_id,
                        const std::map<std::string, std::vector<int>>& target_shapes);

    /**
     * @brief 获取张量大小
     * @param tensor_name 张量名称
     * @param shape 张量形状
     * @param data_type 数据类型
     * @return 字节大小
     */
    size_t getTensorSize(const std::string& tensor_name,
                        const std::vector<int>& shape,
                        nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT);

    /**
     * @brief 获取引擎信息
     * @return 引擎信息字符串
     */
    std::string getEngineInfo() const;

    /**
     * @brief 获取上下文信息
     * @param context_id 执行上下文ID
     * @return 上下文信息字符串
     */
    std::string getContextInfo(int context_id) const;

    /**
     * @brief 销毁执行上下文
     * @param context_id 执行上下文ID
     */
    void destroyExecutionContext(int context_id);

private:
    /**
     * @brief 验证形状兼容性
     * @param tensor_name 张量名称
     * @param shape 输入形状
     * @return 是否兼容
     */
    bool validateShapeCompatibility(const std::string& tensor_name,
                                   const std::vector<int>& shape);

    /**
     * @brief 更新工作空间
     * @param context_id 执行上下文ID
     * @return 是否成功
     */
    bool updateWorkspace(int context_id);

    /**
     * @brief 计算张量元素数量
     * @param shape 张量形状
     * @return 元素数量
     */
    size_t calculateTensorVolume(const std::vector<int>& shape);

    /**
     * @brief 获取数据类型大小
     * @param data_type 数据类型
     * @return 字节大小
     */
    size_t getDataTypeSize(nvinfer1::DataType data_type);

    /**
     * @brief 转换形状格式
     * @param shape 标准形状向量
     * @return TensorRT Dims
     */
    nvinfer1::Dims vectorToDims(const std::vector<int>& shape);

    /**
     * @brief 转换形状格式
     * @param dims TensorRT Dims
     * @return 标准形状向量
     */
    std::vector<int> dimsToVector(const nvinfer1::Dims& dims);

private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::map<std::string, ShapeConfig> shape_configs_;
    std::map<int, ExecutionContext> execution_contexts_;
    std::map<std::string, int> context_name_to_id_;

    int next_context_id_;

    // 缓存的引擎信息
    std::vector<std::string> input_tensor_names_;
    std::vector<std::string> output_tensor_names_;
    std::map<std::string, nvinfer1::DataType> tensor_data_types_;
};

/**
 * @brief 动态批处理管理器
 *
 * 专门处理动态批处理大小的优化
 */
class DynamicBatchManager {
public:
    /**
     * @brief 批处理配置
     */
    struct BatchConfig {
        int min_batch_size = 1;
        int opt_batch_size = 4;
        int max_batch_size = 16;

        // 自动批处理参数
        bool enable_auto_batching = true;
        float batch_timeout_ms = 5.0f;
        int batch_accumulation_threshold = 8;
    };

public:
    /**
     * @brief 构造函数
     * @param shape_handler 动态形状处理器
     * @param config 批处理配置
     */
    DynamicBatchManager(std::shared_ptr<DynamicShapeHandler> shape_handler,
                       const BatchConfig& config);

    /**
     * @brief 析构函数
     */
    ~DynamicBatchManager();

    /**
     * @brief 提交单个推理请求
     * @param input_data 输入数据
     * @param callback 完成回调
     * @return 请求ID
     */
    int submitRequest(const std::map<std::string, void*>& input_data,
                     std::function<void(const std::map<std::string, void*>&)> callback);

    /**
     * @brief 强制执行当前批次
     * @return 执行的请求数量
     */
    int flushBatch();

    /**
     * @brief 设置批处理配置
     * @param config 新的批处理配置
     */
    void setBatchConfig(const BatchConfig& config);

    /**
     * @brief 获取批处理统计信息
     * @return 统计信息字符串
     */
    std::string getBatchStatistics() const;

    /**
     * @brief 启动批处理管理器
     */
    void start();

    /**
     * @brief 停止批处理管理器
     */
    void stop();

private:
    /**
     * @brief 批处理工作线程
     */
    void batchWorkerThread();

    /**
     * @brief 执行批处理推理
     * @param batch_requests 批处理请求
     */
    void executeBatch(const std::vector<int>& batch_requests);

private:
    std::shared_ptr<DynamicShapeHandler> shape_handler_;
    BatchConfig config_;

    // 请求管理
    struct InferenceRequest {
        int request_id;
        std::map<std::string, void*> input_data;
        std::map<std::string, void*> output_data;
        std::function<void(const std::map<std::string, void*>&)> callback;
        std::chrono::high_resolution_clock::time_point submit_time;
    };

    std::map<int, InferenceRequest> pending_requests_;
    std::queue<int> request_queue_;
    std::mutex request_mutex_;
    std::condition_variable request_cv_;

    // 线程管理
    std::thread batch_worker_;
    std::atomic<bool> running_;

    // 统计信息
    std::atomic<int> total_requests_;
    std::atomic<int> total_batches_;
    std::atomic<double> avg_batch_size_;

    int next_request_id_;
};

} // namespace tensorrt
} // namespace yolo
