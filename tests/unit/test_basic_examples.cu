#include "cuda_environment.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <fstream>

// Simple test framework (reusing from existing test)
class TestRunner {
public:
    static void run_test(const std::string& test_name, bool (*test_func)()) {
        std::cout << "Running " << test_name << "... ";
        if (test_func()) {
            std::cout << "PASSED" << std::endl;
            s_passed++;
        } else {
            std::cout << "FAILED" << std::endl;
            s_failed++;
        }
        s_total++;
    }
    
    static void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total: " << s_total << std::endl;
        std::cout << "Passed: " << s_passed << std::endl;
        std::cout << "Failed: " << s_failed << std::endl;
    }
    
    static int get_exit_code() {
        return s_failed > 0 ? 1 : 0;
    }
    
private:
    static int s_total;
    static int s_passed;
    static int s_failed;
};

int TestRunner::s_total = 0;
int TestRunner::s_passed = 0;
int TestRunner::s_failed = 0;

// CUDA错误检查宏
#define CUDA_CHECK_TEST(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误: " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

// 测试用的简单核函数
__global__ void test_vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void test_matrix_add(const float* a, const float* b, float* c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void test_thread_info(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 编码线程信息
        output[idx] = blockIdx.x * 1000.0f + threadIdx.x;
    }
}

// 测试函数

bool test_cuda_availability() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "CUDA不可用，跳过CUDA相关测试" << std::endl;
        return true; // 不将CUDA不可用视为测试失败
    }
    
    return deviceCount > 0;
}

bool test_basic_vector_addition() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }
    
    const int N = 1024;
    const size_t size = N * sizeof(float);
    
    // 分配主机内存
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // 分配设备内存
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_a, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_c, size));
    
    // 复制数据到设备
    CUDA_CHECK_TEST(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
    
    // 配置执行参数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // 启动核函数
    test_vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK_TEST(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    
    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return correct;
}

bool test_basic_matrix_addition() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }
    
    const int rows = 32, cols = 32;
    const int N = rows * cols;
    const size_t size = N * sizeof(float);
    
    // 分配主机内存
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 100);
        h_b[i] = static_cast<float>((i * 2) % 100);
    }
    
    // 分配设备内存
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_a, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_c, size));
    
    // 复制数据到设备
    CUDA_CHECK_TEST(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
    
    // 配置执行参数
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);
    
    // 启动核函数
    test_matrix_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK_TEST(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    
    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return correct;
}

bool test_thread_hierarchy() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }
    
    const int N = 128;
    const size_t size = N * sizeof(float);
    
    // 分配主机内存
    std::vector<float> h_output(N);
    
    // 分配设备内存
    float* d_output = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_output, size));
    
    // 配置执行参数
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // 启动核函数
    test_thread_info<<<gridSize, blockSize>>>(d_output, N);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK_TEST(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));
    
    // 验证结果：检查线程信息编码是否正确
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        int expectedBlockIdx = i / blockSize;
        int expectedThreadIdx = i % blockSize;
        float expected = expectedBlockIdx * 1000.0f + expectedThreadIdx;
        
        if (std::abs(h_output[i] - expected) > 1e-5f) {
            correct = false;
            break;
        }
    }
    
    // 清理内存
    cudaFree(d_output);
    
    return correct;
}

bool test_memory_management() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }
    
    const size_t size = 1024 * sizeof(float);
    
    // 测试内存分配和释放
    float* d_ptr = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_ptr, size));
    
    if (d_ptr == nullptr) {
        return false;
    }
    
    // 测试内存设置
    CUDA_CHECK_TEST(cudaMemset(d_ptr, 0, size));
    
    // 测试主机到设备的内存复制
    std::vector<float> h_data(1024, 42.0f);
    CUDA_CHECK_TEST(cudaMemcpy(d_ptr, h_data.data(), size, cudaMemcpyHostToDevice));
    
    // 测试设备到主机的内存复制
    std::vector<float> h_result(1024);
    CUDA_CHECK_TEST(cudaMemcpy(h_result.data(), d_ptr, size, cudaMemcpyDeviceToHost));
    
    // 验证数据
    bool correct = true;
    for (int i = 0; i < 1024; ++i) {
        if (std::abs(h_result[i] - 42.0f) > 1e-5f) {
            correct = false;
            break;
        }
    }
    
    // 释放内存
    CUDA_CHECK_TEST(cudaFree(d_ptr));
    
    return correct;
}

bool test_error_handling() {
    // 测试CUDA错误处理机制
    
    // 测试无效的内存分配（尝试分配过大的内存）
    float* d_ptr = nullptr;
    cudaError_t error = cudaMalloc(&d_ptr, SIZE_MAX);
    
    // 这应该失败，我们检查是否正确返回错误
    if (error == cudaSuccess) {
        cudaFree(d_ptr);
        return false; // 不应该成功
    }
    
    // 重置错误状态
    cudaGetLastError();
    
    return true;
}

bool test_device_properties() {
    // 如果CUDA不可用，跳过测试
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        return true;
    }
    
    // 测试设备属性查询
    cudaDeviceProp prop;
    CUDA_CHECK_TEST(cudaGetDeviceProperties(&prop, 0));
    
    // 基本的合理性检查
    if (prop.maxThreadsPerBlock < 32) {
        return false; // 现代GPU应该至少支持32个线程每块
    }
    
    if (prop.warpSize != 32) {
        return false; // CUDA线程束大小应该是32
    }
    
    if (prop.totalGlobalMem == 0) {
        return false; // 应该有一些全局内存
    }
    
    return true;
}

bool test_cuda_environment_integration() {
    // 测试CudaEnvironment类的基本功能
    cuda_learning::CudaEnvironment env;
    
    // 测试GPU检测
    bool hasGPU = env.isCudaAvailable();
    
    // 测试GPU信息获取
    auto gpus = env.getAvailableGPUs();
    
    // 如果有GPU，测试更多功能
    if (hasGPU && !gpus.empty()) {
        auto bestGPU = env.getBestGPU();
        
        // 基本合理性检查
        if (bestGPU.name.empty()) {
            return false;
        }
        
        if (bestGPU.totalMemory == 0) {
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "Basic Examples Unit Tests" << std::endl;
    std::cout << "==========================" << std::endl;
    
    // 初始化错误处理
    cuda_learning::ErrorHandler::initialize("test_basic_examples.log");
    
    // 运行测试
    TestRunner::run_test("CUDA Availability", test_cuda_availability);
    TestRunner::run_test("Basic Vector Addition", test_basic_vector_addition);
    TestRunner::run_test("Basic Matrix Addition", test_basic_matrix_addition);
    TestRunner::run_test("Thread Hierarchy", test_thread_hierarchy);
    TestRunner::run_test("Memory Management", test_memory_management);
    TestRunner::run_test("Error Handling", test_error_handling);
    TestRunner::run_test("Device Properties", test_device_properties);
    TestRunner::run_test("CudaEnvironment Integration", test_cuda_environment_integration);
    
    // 打印总结
    TestRunner::print_summary();
    
    // 清理
    cuda_learning::ErrorHandler::shutdown();
    
    return TestRunner::get_exit_code();
}