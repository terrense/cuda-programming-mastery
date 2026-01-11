#pragma once

#include "tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace cuda_learning {
namespace operators {

// 前向声明
class OperatorContext;
class OperatorRegistry;

// 算子参数类型
enum class ParamType {
    INT,
    FLOAT,
    STRING,
    BOOL,
    TENSOR_SHAPE,
    DATA_TYPE
};

// 算子参数值
struct ParamValue {
    ParamType type;
    union {
        int int_val;
        float float_val;
        bool bool_val;
    };
    std::string string_val;
    TensorShape shape_val;
    DataType dtype_val;

    ParamValue() : type(ParamType::INT), int_val(0) {}
    ParamValue(int val) : type(ParamType::INT), int_val(val) {}
    ParamValue(float val) : type(ParamType::FLOAT), float_val(val) {}
    ParamValue(bool val) : type(ParamType::BOOL), bool_val(val) {}
    ParamValue(const std::string& val) : type(ParamType::STRING), string_val(val) {}
    ParamValue(const TensorShape& val) : type(ParamType::TENSOR_SHAPE), shape_val(val) {}
    ParamValue(DataType val) : type(ParamType::DATA_TYPE), dtype_val(val) {}
};

// 算子参数集合
using OperatorParams = std::unordered_map<std::string, ParamValue>;

// 算子执行上下文
class OperatorContext {
public:
    OperatorContext() = default;

    // 设置和获取参数
    void setParam(const std::string& name, const ParamValue& value) {
        params_[name] = value;
    }

    template<typename T>
    T getParam(const std::string& name) const {
        auto it = params_.find(name);
        if (it == params_.end()) {
            throw std::invalid_argument("Parameter not found: " + name);
        }

        const ParamValue& param = it->second;
        if constexpr (std::is_same_v<T, int>) {
            if (param.type != ParamType::INT) throw std::invalid_argument("Type mismatch for parameter: " + name);
            return param.int_val;
        } else if constexpr (std::is_same_v<T, float>) {
            if (param.type != ParamType::FLOAT) throw std::invalid_argument("Type mismatch for parameter: " + name);
            return param.float_val;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (param.type != ParamType::BOOL) throw std::invalid_argument("Type mismatch for parameter: " + name);
            return param.bool_val;
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (param.type != ParamType::STRING) throw std::invalid_argument("Type mismatch for parameter: " + name);
            return param.string_val;
        } else if constexpr (std::is_same_v<T, TensorShape>) {
            if (param.type != ParamType::TENSOR_SHAPE) throw std::invalid_argument("Type mismatch for parameter: " + name);
            return param.shape_val;
        } else if constexpr (std::is_same_v<T, DataType>) {
            if (param.type != ParamType::DATA_TYPE) throw std::invalid_argument("Type mismatch for parameter: " + name);
            return param.dtype_val;
        }
    }

    bool hasParam(const std::string& name) const {
        return params_.find(name) != params_.end();
    }

    // 设置CUDA流
    void setStream(cudaStream_t stream) { stream_ = stream; }
    cudaStream_t getStream() const { return stream_; }

    // 设置设备ID
    void setDevice(int device_id) { device_id_ = device_id; }
    int getDevice() const { return device_id_; }

private:
    OperatorParams params_;
    cudaStream_t stream_ = 0;
    int device_id_ = 0;
};

// 算子基类
class BaseOperator {
public:
    BaseOperator(const std::string& name) : name_(name) {}
    virtual ~BaseOperator() = default;

    // 获取算子名称
    const std::string& getName() const { return name_; }

    // 纯虚函数：前向传播
    virtual void forward(const std::vector<FloatTensor>& inputs,
                        std::vector<FloatTensor>& outputs,
                        const OperatorContext& context) = 0;

    // 虚函数：反向传播（可选实现）
    virtual void backward(const std::vector<FloatTensor>& grad_outputs,
                         std::vector<FloatTensor>& grad_inputs,
                         const OperatorContext& context) {
        throw std::runtime_error("Backward pass not implemented for operator: " + name_);
    }

    // 虚函数：推断输出形状
    virtual std::vector<TensorShape> inferOutputShapes(
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context) = 0;

    // 虚函数：验证输入
    virtual bool validateInputs(const std::vector<FloatTensor>& inputs,
                               const OperatorContext& context) {
        return true; // 默认不验证
    }

    // 虚函数：获取所需参数
    virtual std::vector<std::string> getRequiredParams() const {
        return {}; // 默认无必需参数
    }

    // 虚函数：获取可选参数
    virtual std::vector<std::string> getOptionalParams() const {
        return {}; // 默认无可选参数
    }

    // 虚函数：获取算子描述
    virtual std::string getDescription() const {
        return "Base operator: " + name_;
    }

    // 虚函数：获取内存需求估计
    virtual size_t estimateMemoryUsage(const std::vector<TensorShape>& input_shapes,
                                      const OperatorContext& context) const {
        return 0; // 默认返回0
    }

protected:
    std::string name_;
};

// 算子工厂函数类型
using OperatorFactory = std::function<std::unique_ptr<BaseOperator>()>;

// 算子注册信息
struct OperatorInfo {
    std::string name;
    std::string description;
    std::vector<std::string> required_params;
    std::vector<std::string> optional_params;
    OperatorFactory factory;

    OperatorInfo() = default;
    OperatorInfo(const std::string& n, const std::string& desc,
                const std::vector<std::string>& req_params,
                const std::vector<std::string>& opt_params,
                OperatorFactory f)
        : name(n), description(desc), required_params(req_params),
          optional_params(opt_params), factory(f) {}
};

// 算子注册表
class OperatorRegistry {
public:
    // 获取单例实例
    static OperatorRegistry& getInstance() {
        static OperatorRegistry instance;
        return instance;
    }

    // 注册算子
    void registerOperator(const std::string& name, const OperatorInfo& info) {
        operators_[name] = info;
    }

    // 创建算子实例
    std::unique_ptr<BaseOperator> createOperator(const std::string& name) const {
        auto it = operators_.find(name);
        if (it == operators_.end()) {
            throw std::invalid_argument("Unknown operator: " + name);
        }
        return it->second.factory();
    }

    // 检查算子是否存在
    bool hasOperator(const std::string& name) const {
        return operators_.find(name) != operators_.end();
    }

    // 获取算子信息
    const OperatorInfo& getOperatorInfo(const std::string& name) const {
        auto it = operators_.find(name);
        if (it == operators_.end()) {
            throw std::invalid_argument("Unknown operator: " + name);
        }
        return it->second;
    }

    // 获取所有注册的算子名称
    std::vector<std::string> getRegisteredOperators() const {
        std::vector<std::string> names;
        for (const auto& pair : operators_) {
            names.push_back(pair.first);
        }
        return names;
    }

    // 清空注册表
    void clear() {
        operators_.clear();
    }

private:
    OperatorRegistry() = default;
    std::unordered_map<std::string, OperatorInfo> operators_;
};

// 算子注册宏
#define REGISTER_OPERATOR(name, class_name) \
    namespace { \
        struct class_name##Registrar { \
            class_name##Registrar() { \
                auto factory = []() -> std::unique_ptr<BaseOperator> { \
                    return std::make_unique<class_name>(); \
                }; \
                class_name temp_instance; \
                OperatorInfo info(name, temp_instance.getDescription(), \
                                temp_instance.getRequiredParams(), \
                                temp_instance.getOptionalParams(), \
                                factory); \
                OperatorRegistry::getInstance().registerOperator(name, info); \
            } \
        }; \
        static class_name##Registrar g_##class_name##_registrar; \
    }

// 算子调度器
class OperatorDispatcher {
public:
    // 执行算子
    static void execute(const std::string& op_name,
                       const std::vector<FloatTensor>& inputs,
                       std::vector<FloatTensor>& outputs,
                       const OperatorContext& context);

    // 推断输出形状
    static std::vector<TensorShape> inferOutputShapes(
        const std::string& op_name,
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context);

    // 估计内存使用
    static size_t estimateMemoryUsage(
        const std::string& op_name,
        const std::vector<TensorShape>& input_shapes,
        const OperatorContext& context);
};

} // namespace operators
} // namespace cuda_learning
