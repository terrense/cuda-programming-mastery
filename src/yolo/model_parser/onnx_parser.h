#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <iostream>

namespace cuda_learning {
namespace yolo {

// 张量信息结构
struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    int data_type;  // 对应ONNX数据类型
    void* data;
    size_t size;

    TensorInfo() : data(nullptr), size(0), data_type(1) {} // 默认float32
    ~TensorInfo() {
        if (data) {
            free(data);
            data = nullptr;
        }
    }
};

// 层信息结构
struct LayerInfo {
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::string> attributes;

    LayerInfo() = default;
};

// ONNX模型解析器
class ONNXParser {
private:
    std::vector<LayerInfo> layers_;
    std::map<std::string, std::unique_ptr<TensorInfo>> tensors_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    bool model_loaded_;

public:
    ONNXParser();
    ~ONNXParser();

    // 加载ONNX模型
    bool loadModel(const std::string& model_path);

    // 解析模型结构
    bool parseModel();

    // 获取输入输出信息
    std::vector<TensorInfo*> getInputs() const;
    std::vector<TensorInfo*> getOutputs() const;

    // 获取层信息
    const std::vector<LayerInfo>& getLayers() const { return layers_; }

    // 获取权重数据
    bool getWeights(const std::string& name, void** data, size_t* size);

    // 模型验证
    bool validateModel();

    // 获取模型信息
    void printModelInfo() const;

    // 检查是否已加载模型
    bool isModelLoaded() const { return model_loaded_; }

private:
    // 简化的ONNX解析（不依赖protobuf）
    bool parseSimpleONNX(const std::string& model_path);
    void createDummyYOLOModel();

    // 辅助函数
    std::vector<int> parseShape(const std::string& shape_str);
    int parseDataType(const std::string& type_str);
};

} // namespace yolo
} // namespace cuda_learning
