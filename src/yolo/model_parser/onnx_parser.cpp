#include "onnx_parser.h"
#include "../../core/error_handler.h"
#include <algorithm>
#include <sstream>

namespace cuda_learning {
namespace yolo {

ONNXParser::ONNXParser() : model_loaded_(false) {
}

ONNXParser::~ONNXParser() {
    // 智能指针会自动清理tensors_
}

bool ONNXParser::loadModel(const std::string& model_path) {
    std::cout << "Loading ONNX model from: " << model_path << std::endl;

    // 检查文件是否存在
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        // 为演示目的，创建一个虚拟的YOLO模型结构
        createDummyYOLOModel();
        return true;
    }

    file.close();

    // 简化的ONNX解析（实际项目中应使用onnx库）
    if (!parseSimpleONNX(model_path)) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return false;
    }

    model_loaded_ = true;
    std::cout << "ONNX model loaded successfully" << std::endl;
    return true;
}

bool ONNXParser::parseModel() {
    if (!model_loaded_) {
        std::cerr << "No model loaded" << std::endl;
        return false;
    }

    std::cout << "Parsing model structure..." << std::endl;

    // 验证模型结构
    if (!validateModel()) {
        std::cerr << "Model validation failed" << std::endl;
        return false;
    }

    std::cout << "Model parsed successfully" << std::endl;
    std::cout << "Found " << layers_.size() << " layers" << std::endl;
    std::cout << "Found " << tensors_.size() << " tensors" << std::endl;

    return true;
}

std::vector<TensorInfo*> ONNXParser::getInputs() const {
    std::vector<TensorInfo*> inputs;
    for (const auto& name : input_names_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            inputs.push_back(it->second.get());
        }
    }
    return inputs;
}

std::vector<TensorInfo*> ONNXParser::getOutputs() const {
    std::vector<TensorInfo*> outputs;
    for (const auto& name : output_names_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            outputs.push_back(it->second.get());
        }
    }
    return outputs;
}

bool ONNXParser::getWeights(const std::string& name, void** data, size_t* size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return false;
    }

    *data = it->second->data;
    *size = it->second->size;
    return true;
}

bool ONNXParser::validateModel() {
    // 基本验证
    if (layers_.empty()) {
        std::cerr << "No layers found in model" << std::endl;
        return false;
    }

    if (input_names_.empty()) {
        std::cerr << "No inputs found in model" << std::endl;
        return false;
    }

    if (output_names_.empty()) {
        std::cerr << "No outputs found in model" << std::endl;
        return false;
    }

    // 验证输入输出张量存在
    for (const auto& name : input_names_) {
        if (tensors_.find(name) == tensors_.end()) {
            std::cerr << "Input tensor not found: " << name << std::endl;
            return false;
        }
    }

    for (const auto& name : output_names_) {
        if (tensors_.find(name) == tensors_.end()) {
            std::cerr << "Output tensor not found: " << name << std::endl;
            return false;
        }
    }

    std::cout << "Model validation passed" << std::endl;
    return true;
}

void ONNXParser::printModelInfo() const {
    std::cout << "\n=== ONNX Model Information ===" << std::endl;

    std::cout << "Inputs:" << std::endl;
    for (const auto& name : input_names_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            const auto& tensor = it->second;
            std::cout << "  " << name << ": [";
            for (size_t i = 0; i < tensor->shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << tensor->shape[i];
            }
            std::cout << "] (type: " << tensor->data_type << ")" << std::endl;
        }
    }

    std::cout << "Outputs:" << std::endl;
    for (const auto& name : output_names_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            const auto& tensor = it->second;
            std::cout << "  " << name << ": [";
            for (size_t i = 0; i < tensor->shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << tensor->shape[i];
            }
            std::cout << "] (type: " << tensor->data_type << ")" << std::endl;
        }
    }

    std::cout << "Layers: " << layers_.size() << std::endl;
    for (const auto& layer : layers_) {
        std::cout << "  " << layer.name << " (" << layer.type << ")" << std::endl;
    }

    std::cout << "==============================\n" << std::endl;
}

bool ONNXParser::parseSimpleONNX(const std::string& model_path) {
    // 简化实现：创建一个典型的YOLO模型结构
    createDummyYOLOModel();
    return true;
}

void ONNXParser::createDummyYOLOModel() {
    std::cout << "Creating dummy YOLO model structure for demonstration..." << std::endl;

    // 创建输入张量
    auto input_tensor = std::make_unique<TensorInfo>();
    input_tensor->name = "input";
    input_tensor->shape = {1, 3, 640, 640};  // NCHW格式
    input_tensor->data_type = 1;  // float32
    input_tensor->size = 1 * 3 * 640 * 640 * sizeof(float);
    input_names_.push_back("input");
    tensors_["input"] = std::move(input_tensor);

    // 创建输出张量（YOLO通常有多个输出）
    auto output1_tensor = std::make_unique<TensorInfo>();
    output1_tensor->name = "output1";
    output1_tensor->shape = {1, 255, 80, 80};  // 大尺度特征图
    output1_tensor->data_type = 1;
    output1_tensor->size = 1 * 255 * 80 * 80 * sizeof(float);
    output_names_.push_back("output1");
    tensors_["output1"] = std::move(output1_tensor);

    auto output2_tensor = std::make_unique<TensorInfo>();
    output2_tensor->name = "output2";
    output2_tensor->shape = {1, 255, 40, 40};  // 中尺度特征图
    output2_tensor->data_type = 1;
    output2_tensor->size = 1 * 255 * 40 * 40 * sizeof(float);
    output_names_.push_back("output2");
    tensors_["output2"] = std::move(output2_tensor);

    auto output3_tensor = std::make_unique<TensorInfo>();
    output3_tensor->name = "output3";
    output3_tensor->shape = {1, 255, 20, 20};  // 小尺度特征图
    output3_tensor->data_type = 1;
    output3_tensor->size = 1 * 255 * 20 * 20 * sizeof(float);
    output_names_.push_back("output3");
    tensors_["output3"] = std::move(output3_tensor);

    // 创建一些典型的YOLO层
    LayerInfo conv_layer;
    conv_layer.name = "backbone_conv1";
    conv_layer.type = "Conv";
    conv_layer.inputs = {"input"};
    conv_layer.outputs = {"conv1_output"};
    conv_layer.attributes["kernel_size"] = "3";
    conv_layer.attributes["stride"] = "1";
    conv_layer.attributes["padding"] = "1";
    layers_.push_back(conv_layer);

    LayerInfo bn_layer;
    bn_layer.name = "backbone_bn1";
    bn_layer.type = "BatchNormalization";
    bn_layer.inputs = {"conv1_output"};
    bn_layer.outputs = {"bn1_output"};
    layers_.push_back(bn_layer);

    LayerInfo relu_layer;
    relu_layer.name = "backbone_relu1";
    relu_layer.type = "Relu";
    relu_layer.inputs = {"bn1_output"};
    relu_layer.outputs = {"relu1_output"};
    layers_.push_back(relu_layer);

    // 添加更多层来模拟完整的YOLO结构
    for (int i = 2; i <= 10; ++i) {
        LayerInfo layer;
        layer.name = "backbone_conv" + std::to_string(i);
        layer.type = "Conv";
        layer.inputs = {"relu" + std::to_string(i-1) + "_output"};
        layer.outputs = {"conv" + std::to_string(i) + "_output"};
        layers_.push_back(layer);
    }

    // 添加检测头层
    LayerInfo detect_layer;
    detect_layer.name = "detect";
    detect_layer.type = "YOLODetect";
    detect_layer.inputs = {"conv8_output", "conv9_output", "conv10_output"};
    detect_layer.outputs = {"output1", "output2", "output3"};
    detect_layer.attributes["num_classes"] = "80";
    detect_layer.attributes["anchors"] = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    layers_.push_back(detect_layer);

    std::cout << "Dummy YOLO model created with " << layers_.size() << " layers" << std::endl;
}

std::vector<int> ONNXParser::parseShape(const std::string& shape_str) {
    std::vector<int> shape;
    std::stringstream ss(shape_str);
    std::string item;

    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoi(item));
    }

    return shape;
}

int ONNXParser::parseDataType(const std::string& type_str) {
    if (type_str == "float32" || type_str == "FLOAT") return 1;
    if (type_str == "uint8" || type_str == "UINT8") return 2;
    if (type_str == "int8" || type_str == "INT8") return 3;
    if (type_str == "uint16" || type_str == "UINT16") return 4;
    if (type_str == "int16" || type_str == "INT16") return 5;
    if (type_str == "int32" || type_str == "INT32") return 6;
    if (type_str == "int64" || type_str == "INT64") return 7;
    if (type_str == "string" || type_str == "STRING") return 8;
    if (type_str == "bool" || type_str == "BOOL") return 9;
    if (type_str == "float16" || type_str == "FLOAT16") return 10;
    return 1; // 默认float32
}

} // namespace yolo
} // namespace cuda_learning
