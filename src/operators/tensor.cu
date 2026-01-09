#include "tensor.h"
#include <iostream>
#include <sstream>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

namespace cuda_learning {
namespace operators {

// TensorShape 实现
TensorShape TensorShape::reshape(const std::vector<int>& new_dims) const {
    size_t new_numel = 1;
    for (int d : new_dims) new_numel *= d;

    if (new_numel != numel()) {
        throw std::invalid_argument("Cannot reshape tensor: element count mismatch");
    }

    return TensorShape(new_dims);
}

TensorShape TensorShape::transpose(int dim0, int dim1) const {
    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();

    if (dim0 >= ndim() || dim1 >= ndim() || dim0 < 0 || dim1 < 0) {
        throw std::out_of_range("Dimension index out of range");
    }

    std::vector<int> new_dims = dims_;
    std::swap(new_dims[dim0], new_dims[dim1]);
    return TensorShape(new_dims);
}

std::string TensorShape::toString() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims_[i];
    }
    oss << "]";
    return oss.str();
}

// CUDA 核函数用于张量操作
__global__ void fillKernel(float* data, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void uniformKernel(float* data, float min_val, float max_val, size_t size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = min_val + (max_val - min_val) * curand_uniform(&state);
    }
}

__global__ void normalKernel(float* data, float mean, float std, size_t size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = mean + std * curand_normal(&state);
    }
}

// Tensor 模板特化实现
template<typename T>
void Tensor<T>::allocate() {
    if (size_ == 0) return;

    if (device_ == DeviceType::GPU) {
        CUDA_CHECK_THROW(cudaMalloc(&data_, bytes()));
    } else {
        data_ = new T[size_];
    }
}

template<typename T>
void Tensor<T>::deallocate() {
    if (data_ != nullptr) {
        if (device_ == DeviceType::GPU) {
            cudaFree(data_);
        } else {
            delete[] data_;
        }
        data_ = nullptr;
    }
}

template<typename T>
void Tensor<T>::copyFrom(const Tensor& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Tensor size mismatch");
    }

    cudaMemcpyKind kind;
    if (device_ == DeviceType::GPU && other.device_ == DeviceType::GPU) {
        kind = cudaMemcpyDeviceToDevice;
    } else if (device_ == DeviceType::GPU && other.device_ == DeviceType::CPU) {
        kind = cudaMemcpyHostToDevice;
    } else if (device_ == DeviceType::CPU && other.device_ == DeviceType::GPU) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyHostToHost;
    }

    CUDA_CHECK_THROW(cudaMemcpy(data_, other.data_, bytes(), kind));
}

template<typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& new_dims) const {
    TensorShape new_shape = shape_.reshape(new_dims);
    Tensor<T> result(new_shape, device_);
    result.copyFrom(*this);
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::to(DeviceType target_device) const {
    if (target_device == device_) {
        return *this; // 返回副本
    }

    Tensor<T> result(shape_, target_device);
    result.copyFrom(*this);
    return result;
}

template<typename T>
void Tensor<T>::copyTo(Tensor<T>& target) const {
    target.copyFrom(*this);
}

template<typename T>
void Tensor<T>::fromHost(const std::vector<T>& host_data) {
    if (host_data.size() != size_) {
        throw std::invalid_argument("Host data size mismatch");
    }

    if (device_ == DeviceType::GPU) {
        CUDA_CHECK_THROW(cudaMemcpy(data_, host_data.data(), bytes(), cudaMemcpyHostToDevice));
    } else {
        std::copy(host_data.begin(), host_data.end(), data_);
    }
}

template<typename T>
std::vector<T> Tensor<T>::toHost() const {
    std::vector<T> result(size_);

    if (device_ == DeviceType::GPU) {
        CUDA_CHECK_THROW(cudaMemcpy(result.data(), data_, bytes(), cudaMemcpyDeviceToHost));
    } else {
        std::copy(data_, data_ + size_, result.begin());
    }

    return result;
}

template<typename T>
void Tensor<T>::fill(T value) {
    if (device_ == DeviceType::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            int blockSize = 256;
            int gridSize = (size_ + blockSize - 1) / blockSize;
            fillKernel<<<gridSize, blockSize>>>(reinterpret_cast<float*>(data_),
                                               static_cast<float>(value), size_);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
        } else {
            // 对于其他类型，使用 cudaMemset 或回退到 CPU
            std::vector<T> host_data(size_, value);
            fromHost(host_data);
        }
    } else {
        std::fill(data_, data_ + size_, value);
    }
}

template<typename T>
void Tensor<T>::uniform(T min_val, T max_val) {
    if (device_ == DeviceType::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            int blockSize = 256;
            int gridSize = (size_ + blockSize - 1) / blockSize;
            unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            uniformKernel<<<gridSize, blockSize>>>(reinterpret_cast<float*>(data_),
                                                  static_cast<float>(min_val),
                                                  static_cast<float>(max_val),
                                                  size_, seed);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
        } else {
            // 回退到 CPU 实现
            auto cpu_tensor = to(DeviceType::CPU);
            cpu_tensor.uniform(min_val, max_val);
            copyFrom(cpu_tensor);
        }
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(static_cast<float>(min_val), static_cast<float>(max_val));

        for (size_t i = 0; i < size_; ++i) {
            data_[i] = static_cast<T>(dis(gen));
        }
    }
}

template<typename T>
void Tensor<T>::normal(T mean, T std) {
    if (device_ == DeviceType::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            int blockSize = 256;
            int gridSize = (size_ + blockSize - 1) / blockSize;
            unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            normalKernel<<<gridSize, blockSize>>>(reinterpret_cast<float*>(data_),
                                                 static_cast<float>(mean),
                                                 static_cast<float>(std),
                                                 size_, seed);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
        } else {
            // 回退到 CPU 实现
            auto cpu_tensor = to(DeviceType::CPU);
            cpu_tensor.normal(mean, std);
            copyFrom(cpu_tensor);
        }
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(static_cast<float>(mean), static_cast<float>(std));

        for (size_t i = 0; i < size_; ++i) {
            data_[i] = static_cast<T>(dis(gen));
        }
    }
}

template<typename T>
size_t Tensor<T>::getLinearIndex(const std::vector<int>& indices) const {
    if (indices.size() != shape_.ndim()) {
        throw std::invalid_argument("Index dimension mismatch");
    }

    size_t linear_idx = 0;
    size_t stride = 1;

    for (int i = shape_.ndim() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_.dim(i)) {
            throw std::out_of_range("Index out of range");
        }
        linear_idx += indices[i] * stride;
        stride *= shape_.dim(i);
    }

    return linear_idx;
}

template<typename T>
T& Tensor<T>::operator[](const std::vector<int>& indices) {
    if (device_ != DeviceType::CPU) {
        throw std::runtime_error("Direct indexing only supported for CPU tensors");
    }
    return data_[getLinearIndex(indices)];
}

template<typename T>
const T& Tensor<T>::operator[](const std::vector<int>& indices) const {
    if (device_ != DeviceType::CPU) {
        throw std::runtime_error("Direct indexing only supported for CPU tensors");
    }
    return data_[getLinearIndex(indices)];
}

template<typename T>
std::string Tensor<T>::toString() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_.toString()
        << ", device=" << (device_ == DeviceType::GPU ? "GPU" : "CPU")
        << ", size=" << size_ << ")";
    return oss.str();
}

// 工具函数实现
size_t getDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return sizeof(short);
        case DataType::INT32: return sizeof(int);
        case DataType::INT8: return sizeof(char);
        case DataType::BOOL: return sizeof(bool);
        default: return 0;
    }
}

std::string dataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT16: return "float16";
        case DataType::INT32: return "int32";
        case DataType::INT8: return "int8";
        case DataType::BOOL: return "bool";
        default: return "unknown";
    }
}

DataType stringToDataType(const std::string& str) {
    if (str == "float32") return DataType::FLOAT32;
    if (str == "float16") return DataType::FLOAT16;
    if (str == "int32") return DataType::INT32;
    if (str == "int8") return DataType::INT8;
    if (str == "bool") return DataType::BOOL;
    throw std::invalid_argument("Unknown data type: " + str);
}

// 显式模板实例化
template class Tensor<float>;
template class Tensor<int>;
template class Tensor<bool>;

} // namespace operators
} // namespace cuda_learning
