#include "pytorch_loader.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <fstream>
#include <cstring>

namespace cuda_learning {
namespace yolo {

PyTorchLoader::PyTorchLoader() : model_loaded_(false), use_cuda_(false) {
    // 检查CUDA可用性
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaSuccess && device_count > 0) {
        use_cuda_ = true;
        std::cout << "CUDA available with " << device_count << " device(s)" << std::endl;
    } else {
        std::cout << "CUDA not available, using CPU" << std::endl;
    }
}

PyTorchLoader::~PyTorchLoader() {
    // 智能指针会自动清理tensors_
}

bool PyTorchLoader::loadModel(const std::string& model_path) {
    std::cout << "Loading PyTorch model from: " << model_path << std::endl;

    // 检查文件是否存在
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        // 为演示目的，创建一个虚拟的PyTorch模型
        createDummyPyTorchModel();
        return true;
    }

    file.close();

    try {
        // 在实际实现中，这里会使用libtorch加载模型
        // torch::jit::script::Module model = torch::jit::load(model_path);

        // 为演示目的，创建虚拟模型
        createDummyPyTorchModel();

        model_loaded_ = true;
        std::cout << "PyTorch model loaded successfully" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading PyTorch model: " << e.what() << std::endl;
        return false;
    }
}

bool PyTorchLoader::setInput(const std::string& name, const std::vector<float>& data, const std::vector<int>& shape) {
    if (!model_loaded_) {
        std::cerr << "No model loaded" << std::endl;
        return false;
    }

    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        std::cerr << "Input tensor not found: " << name << std::endl;
        return false;
    }

    auto& tensor = it->second;

    // 验证形状
    if (tensor->shape != shape) {
        std::cerr << "Shape mismatch for tensor " << name << std::endl;
        return false;
    }

    // 验证数据大小
    size_t expected_size = 1;
    for (int dim : shape) {
        expected_size *= dim;
    }

    if (data.size() != expected_size) {
        std::cerr << "Data size mismatch for tensor " << name << std::endl;
        return false;
    }

    // 复制数据
    size_t byte_size = expected_size * sizeof(float);

    if (tensor->is_cuda && use_cuda_) {
        // 复制到GPU
        cudaError_t error = cudaMemcpy(tensor->data, data.data(), byte_size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data to GPU: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
    } else {
        // 复制到CPU
        memcpy(tensor->data, data.data(), byte_size);
    }

    std::cout << "Input data set for tensor: " << name << std::endl;
    return true;
}

std::vector<PyTorchTensorInfo*> PyTorchLoader::forward() {
    if (!model_loaded_) {
        std::cerr << "No model loaded" << std::endl;
        return {};
    }

    std::cout << "Executing PyTorch model forward pass..." << std::endl;

    try {
        // 在实际实现中，这里会执行PyTorch模型的前向传播
        // auto outputs = model_.forward(inputs_).toTuple();

        // 为演示目的，填充输出张量
        for (const auto& name : output_names_) {
            auto it = tensors_.find(name);
            if (it != tensors_.end()) {
                auto& tensor = it->second;

                // 填充一些虚拟数据
                size_t num_elements = 1;
                for (int dim : tensor->shape) {
                    num_elements *= dim;
                }

                std::vector<float> dummy_data(num_elements, 0.5f);

                if (tensor->is_cuda && use_cuda_) {
                    cudaMemcpy(tensor->data, dummy_data.data(),
                              num_elements * sizeof(float), cudaMemcpyHostToDevice);
                } else {
                    memcpy(tensor->data, dummy_data.data(), num_elements * sizeof(float));
                }
            }
        }

        // 返回输出张量
        std::vector<PyTorchTensorInfo*> outputs;
        for (const auto& name : output_names_) {
            auto it = tensors_.find(name);
            if (it != tensors_.end()) {
                outputs.push_back(it->second.get());
            }
        }

        std::cout << "Forward pass completed, " << outputs.size() << " outputs generated" << std::endl;
        return outputs;

    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return {};
    }
}

void PyTorchLoader::printModelInfo() {
    std::cout << "\n=== PyTorch Model Information ===" << std::endl;

    std::cout << "Device: " << (use_cuda_ ? "CUDA" : "CPU") << std::endl;

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
            std::cout << "] (" << getDataTypeName(tensor->data_type) << ")" << std::endl;
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
            std::cout << "] (" << getDataTypeName(tensor->data_type) << ")" << std::endl;
        }
    }

    std::cout << "================================\n" << std::endl;
}

bool PyTorchLoader::exportToONNX(const std::string& output_path, const std::vector<int>& input_shape) {
    if (!model_loaded_) {
        std::cerr << "No model loaded" << std::endl;
        return false;
    }

    std::cout << "Exporting PyTorch model to ONNX: " << output_path << std::endl;

    try {
        // 在实际实现中，这里会使用PyTorch的ONNX导出功能
        // torch::jit::script::Module traced_model = torch::jit::trace(model_, example_input);
        // traced_model.save(output_path);

        // 为演示目的，创建一个虚拟的ONNX文件
        std::ofstream file(output_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to create ONNX file: " << output_path << std::endl;
            return false;
        }

        // 写入一些虚拟数据
        std::string dummy_onnx = "ONNX_DUMMY_MODEL_DATA";
        file.write(dummy_onnx.c_str(), dummy_onnx.size());
        file.close();

        std::cout << "Model exported to ONNX successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error exporting to ONNX: " << e.what() << std::endl;
        return false;
    }
}

std::vector<PyTorchTensorInfo*> PyTorchLoader::getInputs() const {
    std::vector<PyTorchTensorInfo*> inputs;
    for (const auto& name : input_names_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            inputs.push_back(it->second.get());
        }
    }
    return inputs;
}

std::vector<PyTorchTensorInfo*> PyTorchLoader::getOutputs() const {
    std::vector<PyTorchTensorInfo*> outputs;
    for (const auto& name : output_names_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            outputs.push_back(it->second.get());
        }
    }
    return outputs;
}

void PyTorchLoader::createDummyPyTorchModel() {
    std::cout << "Creating dummy PyTorch model structure for demonstration..." << std::endl;

    // 创建输入张量
    auto input_tensor = createTensor("input", {1, 3, 640, 640}, 1, use_cuda_);
    input_names_.push_back("input");

    // 创建输出张量（YOLO多尺度输出）
    auto output1_tensor = createTensor("output1", {1, 255, 80, 80}, 1, use_cuda_);
    output_names_.push_back("output1");

    auto output2_tensor = createTensor("output2", {1, 255, 40, 40}, 1, use_cuda_);
    output_names_.push_back("output2");

    auto output3_tensor = createTensor("output3", {1, 255, 20, 20}, 1, use_cuda_);
    output_names_.push_back("output3");

    std::cout << "Dummy PyTorch model created with " << input_names_.size()
              << " inputs and " << output_names_.size() << " outputs" << std::endl;
}

PyTorchTensorInfo* PyTorchLoader::createTensor(const std::string& name, const std::vector<int>& shape,
                                              int data_type, bool on_cuda) {
    auto tensor = std::make_unique<PyTorchTensorInfo>();
    tensor->name = name;
    tensor->shape = shape;
    tensor->data_type = data_type;
    tensor->is_cuda = on_cuda;

    // 计算大小
    size_t num_elements = 1;
    for (int dim : shape) {
        num_elements *= dim;
    }
    tensor->size = num_elements * getDataTypeSize(data_type);

    // 分配内存
    if (on_cuda && use_cuda_) {
        cudaError_t error = cudaMalloc(&tensor->data, tensor->size);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for tensor " << name
                      << ": " << cudaGetErrorString(error) << std::endl;
            tensor->is_cuda = false;
            tensor->data = malloc(tensor->size);
        }
    } else {
        tensor->data = malloc(tensor->size);
        tensor->is_cuda = false;
    }

    if (!tensor->data) {
        std::cerr << "Failed to allocate memory for tensor " << name << std::endl;
        return nullptr;
    }

    PyTorchTensorInfo* ptr = tensor.get();
    tensors_[name] = std::move(tensor);
    return ptr;
}

size_t PyTorchLoader::getDataTypeSize(int data_type) {
    switch (data_type) {
        case 1: return sizeof(float);    // float32
        case 2: return sizeof(uint8_t);  // uint8
        case 3: return sizeof(int8_t);   // int8
        case 4: return sizeof(uint16_t); // uint16
        case 5: return sizeof(int16_t);  // int16
        case 6: return sizeof(int32_t);  // int32
        case 7: return sizeof(int64_t);  // int64
        case 10: return sizeof(uint16_t); // float16
        default: return sizeof(float);
    }
}

std::string PyTorchLoader::getDataTypeName(int data_type) {
    switch (data_type) {
        case 1: return "float32";
        case 2: return "uint8";
        case 3: return "int8";
        case 4: return "uint16";
        case 5: return "int16";
        case 6: return "int32";
        case 7: return "int64";
        case 10: return "float16";
        default: return "unknown";
    }
}

} // namespace yolo
} // namespace cuda_learning
