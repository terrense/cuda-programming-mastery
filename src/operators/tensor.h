#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include "../core/error_handler.h"

namespace cuda_learning {
namespace operators {

// 数据类型枚举
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT8,
    BOOL
};

// 设备类型枚举
enum class DeviceType {
    CPU,
    GPU
};

// 张量形状类
class TensorShape {
public:
    TensorShape() = default;
    TensorShape(const std::vector<int>& dims) : dims_(dims) {}
    TensorShape(std::initializer_list<int> dims) : dims_(dims) {}

    // 获取维度数量
    int ndim() const { return static_cast<int>(dims_.size()); }

    // 获取指定维度的大小
    int dim(int index) const {
        if (index < 0) index += ndim();
        return dims_[index];
    }

    // 获取所有维度
    const std::vector<int>& dims() const { return dims_; }

    // 计算总元素数量
    size_t numel() const {
        size_t total = 1;
        for (int d : dims_) total *= d;
        return total;
    }

    // 重塑形状
    TensorShape reshape(const std::vector<int>& new_dims) const;

    // 转置
    TensorShape transpose(int dim0, int dim1) const;

    // 字符串表示
    std::string toString() const;

    // 比较操作
    bool operator==(const TensorShape& other) const { return dims_ == other.dims_; }
    bool operator!=(const TensorShape& other) const { return !(*this == other); }

private:
    std::vector<int> dims_;
};

// 张量类
template<typename T>
class Tensor {
public:
    // 构造函数
    Tensor() : data_(nullptr), size_(0), device_(DeviceType::CPU) {}

    Tensor(const TensorShape& shape, DeviceType device = DeviceType::GPU)
        : shape_(shape), device_(device), size_(shape.numel()) {
        allocate();
    }

    Tensor(const std::vector<int>& dims, DeviceType device = DeviceType::GPU)
        : Tensor(TensorShape(dims), device) {}

    // 拷贝构造函数
    Tensor(const Tensor& other) : shape_(other.shape_), device_(other.device_), size_(other.size_) {
        allocate();
        copyFrom(other);
    }

    // 移动构造函数
    Tensor(Tensor&& other) noexcept
        : data_(other.data_), shape_(other.shape_), device_(other.device_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // 析构函数
    ~Tensor() { deallocate(); }

    // 赋值操作符
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            deallocate();
            shape_ = other.shape_;
            device_ = other.device_;
            size_ = other.size_;
            allocate();
            copyFrom(other);
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            deallocate();
            data_ = other.data_;
            shape_ = other.shape_;
            device_ = other.device_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 基本属性访问
    const TensorShape& shape() const { return shape_; }
    DeviceType device() const { return device_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }

    // 数据访问
    T* data() { return data_; }
    const T* data() const { return data_; }

    // 重塑张量
    Tensor<T> reshape(const std::vector<int>& new_dims) const;

    // 转置张量
    Tensor<T> transpose(int dim0, int dim1) const;

    // 设备间数据传输
    Tensor<T> to(DeviceType target_device) const;
    void copyTo(Tensor<T>& target) const;

    // 从主机数据初始化
    void fromHost(const std::vector<T>& host_data);

    // 复制到主机
    std::vector<T> toHost() const;

    // 填充数据
    void fill(T value);
    void zero() { fill(static_cast<T>(0)); }
    void ones() { fill(static_cast<T>(1)); }

    // 随机初始化
    void uniform(T min_val, T max_val);
    void normal(T mean, T std);

    // 切片操作
    Tensor<T> slice(int dim, int start, int end) const;

    // 索引操作
    T& operator[](const std::vector<int>& indices);
    const T& operator[](const std::vector<int>& indices) const;

    // 字符串表示
    std::string toString() const;

    // 验证张量有效性
    bool isValid() const { return data_ != nullptr && size_ > 0; }

private:
    T* data_;
    TensorShape shape_;
    DeviceType device_;
    size_t size_;

    void allocate();
    void deallocate();
    void copyFrom(const Tensor& other);
    size_t getLinearIndex(const std::vector<int>& indices) const;
};

// 张量工厂函数
template<typename T>
Tensor<T> zeros(const std::vector<int>& dims, DeviceType device = DeviceType::GPU) {
    Tensor<T> tensor(dims, device);
    tensor.zero();
    return tensor;
}

template<typename T>
Tensor<T> ones(const std::vector<int>& dims, DeviceType device = DeviceType::GPU) {
    Tensor<T> tensor(dims, device);
    tensor.ones();
    return tensor;
}

template<typename T>
Tensor<T> uniform(const std::vector<int>& dims, T min_val, T max_val, DeviceType device = DeviceType::GPU) {
    Tensor<T> tensor(dims, device);
    tensor.uniform(min_val, max_val);
    return tensor;
}

template<typename T>
Tensor<T> normal(const std::vector<int>& dims, T mean, T std, DeviceType device = DeviceType::GPU) {
    Tensor<T> tensor(dims, device);
    tensor.normal(mean, std);
    return tensor;
}

// 数据类型工具函数
size_t getDataTypeSize(DataType dtype);
std::string dataTypeToString(DataType dtype);
DataType stringToDataType(const std::string& str);

// 常用类型别名
using FloatTensor = Tensor<float>;
using IntTensor = Tensor<int>;
using BoolTensor = Tensor<bool>;

} // namespace operators
} // namespace cuda_learning
