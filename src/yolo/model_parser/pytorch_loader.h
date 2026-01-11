#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <cuda_runtime.h>

namespace cuda_learning {
namespace yolo {

// PyTorch张量信息
struct PyTorchTensorInfo {
    std::string name;
    std::vector<int> shape;
    int data_type;
    void* data;
    size_t size;
    bool is_cuda;

    PyTorchTensorInfo() : data(nullptr), size(0), data_type(1), is_cuda(false) {}
    ~PyTorchTensorInfo() {
        if (data) {
            if (is_cuda) {
                cudaFree(data);
            } else {
                free(data);
            }
            data = nullptr;
        }
    }
};

// PyTorch模型加载器
class PyTorchLoader {
private:
    std::map<std::string, std::unique_ptr<PyTorchTensorInfo>> tensors_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    bool model_loaded_;
    bool use_cuda_;

public:
    PyTorchLoader();
    ~PyTorchLoader();

    // 加载TorchScript模型
    bool loadModel(const std::string& model_path);

    // 设置输入数据
    bool setInput(const std::string& name, const std::vector<float>& data, const std::vector<int>& shape);

    // 执行推理
    std::vector<PyTorchTensorInfo*> forward();

    // 获取模型信息
    void printModelInfo();

    // 转换为ONNX
    bool exportToONNX(const std::string& output_path, const std::vector<int>& input_shape);

    // 获取输入输出信息
    std::vector<PyTorchTensorInfo*> getInputs() const;
    std::vector<PyTorchTensorInfo*> getOutputs() const;

    // 检查CUDA可用性
    bool isCudaAvailable() const { return use_cuda_; }

    // 设置设备
    void setDevice(bool use_cuda) { use_cuda_ = use_cuda; }

    // 检查是否已加载模型
    bool isModelLoaded() const { return model_loaded_; }

private:
    // 创建虚拟PyTorch模型（用于演示）
    void createDummyPyTorchModel();

    // 创建张量
    PyTorchTensorInfo* createTensor(const std::string& name, const std::vector<int>& shape,
                                   int data_type = 1, bool on_cuda = false);

    // 数据类型转换
    size_t getDataTypeSize(int data_type);
    std::string getDataTypeName(int data_type);
};

} // namespace yolo
} // namespace cuda_learning
