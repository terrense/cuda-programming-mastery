/**
 * 示例代码执行测试
 *
 * 这个测试文件验证基础示例代码能够正确编译和运行
 * 主要测试：
 * 1. 示例代码的编译能力
 * 2. 运行时的正确性
 * 3. 错误处理的有效性
 * 4. 性能基准的合理性
 */

#include "cuda_environment.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// Simple test framework
class TestRunner {
public:
    static void run_test(const std::string& test_name, bool (*test_func)()) {
        std::cout << "Running " << test_name << "... ";
        try {
            if (test_func()) {
                std::cout << "PASSED" << std::endl;
                s_passed++;
            } else {
                std::cout << "FAILED" << std::endl;
                s_failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (Exception: " << e.what() << ")" << std::endl;
            s_failed++;
        } catch (...) {
            std::cout << "FAILED (Unknown exception)" << std::endl;
            s_failed++;
        }
        s_total++;
    }

    static void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total: " << s_total << std::endl;
        std::cout << "Passed: " << s_passed << std::endl;
        std::cout << "Failed: " << s_failed << std::endl;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1)
                  << (s_total > 0 ? (100.0 * s_passed / s_total) : 0.0) << "%" << std::endl;
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

// 测试用的核函数（复制自示例代码）
__global__ void test_hello_world_kernel(char* message, int messageLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < messageLength) {
        if (message[idx] >= 'a' && message[idx] <= 'z') {
            message[idx] = message[idx] - 'a' + 'A';
        }
    }
}

__global__ void test_vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void test_matrix_add_kernel(const float* a, const float* b, float* c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void test_matrix_transpose_kernel(const float* a, float* b, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        b[col * rows + row] = a[row * cols + col];
    }
}

// 辅助函数：检查CUDA是否可用
bool is_cuda_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// 测试函数

bool test_hello_world_example() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
        return true;
    }

    std::string hostMessage = "hello cuda world!";
    int messageLength = hostMessage.length();

    // 分配设备内存
    char* deviceMessage = nullptr;
    size_t messageSize = messageLength * sizeof(char);
    CUDA_CHECK_TEST(cudaMalloc(&deviceMessage, messageSize));

    // 复制数据到设备
    CUDA_CHECK_TEST(cudaMemcpy(deviceMessage, hostMessage.c_str(), messageSize, cudaMemcpyHostToDevice));

    // 配置执行参数
    int blockSize = 256;
    int gridSize = (messageLength + blockSize - 1) / blockSize;

    // 启动核函数
    test_hello_world_kernel<<<gridSize, blockSize>>>(deviceMessage, messageLength);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    // 复制结果回主机
    std::vector<char> result(messageLength + 1, '\0');
    CUDA_CHECK_TEST(cudaMemcpy(result.data(), deviceMessage, messageSize, cudaMemcpyDeviceToHost));

    // 验证结果
    std::string expected = "HELLO CUDA WORLD!";
    bool correct = (std::string(result.data()) == expected);

    // 清理内存
    CUDA_CHECK_TEST(cudaFree(deviceMessage));

    return correct;
}

bool test_vector_addition_example() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
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
    test_vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
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

bool test_matrix_addition_example() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
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
    test_matrix_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
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

bool test_matrix_transpose_example() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
        return true;
    }

    const int rows = 16, cols = 16;
    const int N = rows * cols;
    const size_t size = N * sizeof(float);

    // 分配主机内存
    std::vector<float> h_a(N), h_b(N);

    // 初始化数据
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_a[i * cols + j] = static_cast<float>(i * cols + j);
        }
    }

    // 分配设备内存
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_a, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, size));

    // 复制数据到设备
    CUDA_CHECK_TEST(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));

    // 配置执行参数
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    // 启动核函数
    test_matrix_transpose_kernel<<<gridSize, blockSize>>>(d_a, d_b, rows, cols);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    // 复制结果回主机
    CUDA_CHECK_TEST(cudaMemcpy(h_b.data(), d_b, size, cudaMemcpyDeviceToHost));

    // 验证结果
    bool correct = true;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float original = h_a[i * cols + j];
            float transposed = h_b[j * rows + i];
            if (std::abs(original - transposed) > 1e-5f) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);

    return correct;
}

bool test_performance_measurement() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
        return true;
    }

    const int N = 1024 * 1024;
    const size_t size = N * sizeof(float);

    // 分配内存
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_a, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_c, size));

    // 初始化数据
    CUDA_CHECK_TEST(cudaMemset(d_a, 1, size));
    CUDA_CHECK_TEST(cudaMemset(d_b, 2, size));

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK_TEST(cudaEventCreate(&start));
    CUDA_CHECK_TEST(cudaEventCreate(&stop));

    // 配置执行参数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 测量执行时间
    CUDA_CHECK_TEST(cudaEventRecord(start));
    test_vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_TEST(cudaEventRecord(stop));
    CUDA_CHECK_TEST(cudaEventSynchronize(stop));

    // 计算执行时间
    float milliseconds = 0;
    CUDA_CHECK_TEST(cudaEventElapsedTime(&milliseconds, start, stop));

    // 验证时间是合理的（应该在几毫秒内完成）
    bool reasonable_time = (milliseconds > 0.0f && milliseconds < 1000.0f);

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return reasonable_time;
}

bool test_memory_bandwidth_calculation() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
        return true;
    }

    const int N = 1024 * 1024;
    const size_t size = N * sizeof(float);

    // 分配内存
    std::vector<float> h_data(N, 1.0f);
    float* d_data = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&d_data, size));

    // 测量主机到设备传输时间
    cudaEvent_t start, stop;
    CUDA_CHECK_TEST(cudaEventCreate(&start));
    CUDA_CHECK_TEST(cudaEventCreate(&stop));

    CUDA_CHECK_TEST(cudaEventRecord(start));
    CUDA_CHECK_TEST(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaEventRecord(stop));
    CUDA_CHECK_TEST(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK_TEST(cudaEventElapsedTime(&milliseconds, start, stop));

    // 计算带宽 (GB/s)
    float bandwidth = (size / (1024.0f * 1024.0f * 1024.0f)) / (milliseconds / 1000.0f);

    // 验证带宽是合理的（应该大于0，小于理论最大值）
    bool reasonable_bandwidth = (bandwidth > 0.0f && bandwidth < 1000.0f);

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return reasonable_bandwidth;
}

bool test_error_handling_robustness() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
        return true;
    }

    // 测试无效内存访问的错误处理
    float* invalid_ptr = nullptr;

    // 这应该失败
    cudaError_t error = cudaMemset(invalid_ptr, 0, 1024);
    if (error == cudaSuccess) {
        return false; // 不应该成功
    }

    // 重置错误状态
    cudaGetLastError();

    // 测试无效的核函数启动参数
    dim3 invalid_grid(0, 0, 0);
    dim3 valid_block(256, 1, 1);

    // 分配有效内存用于测试
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t size = 1024 * sizeof(float);

    if (cudaMalloc(&d_a, size) != cudaSuccess ||
        cudaMalloc(&d_b, size) != cudaSuccess ||
        cudaMalloc(&d_c, size) != cudaSuccess) {
        return false;
    }

    // 启动无效的核函数（网格大小为0）
    test_vector_add_kernel<<<invalid_grid, valid_block>>>(d_a, d_b, d_c, 1024);
    error = cudaGetLastError();

    bool error_detected = (error != cudaSuccess);

    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return error_detected;
}

bool test_thread_configuration_validation() {
    if (!is_cuda_available()) {
        std::cout << "CUDA不可用，跳过测试" << std::endl;
        return true;
    }

    // 获取设备属性
    cudaDeviceProp prop;
    CUDA_CHECK_TEST(cudaGetDeviceProperties(&prop, 0));

    // 测试有效的线程配置
    int validBlockSize = std::min(256, prop.maxThreadsPerBlock);
    int validGridSize = 64;

    // 分配测试内存
    const int N = validGridSize * validBlockSize;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t size = N * sizeof(float);

    CUDA_CHECK_TEST(cudaMalloc(&d_a, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, size));
    CUDA_CHECK_TEST(cudaMalloc(&d_c, size));

    // 启动有效配置的核函数
    test_vector_add_kernel<<<validGridSize, validBlockSize>>>(d_a, d_b, d_c, N);
    cudaError_t error1 = cudaGetLastError();
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    // 测试无效的线程配置（超过最大线程数）
    int invalidBlockSize = prop.maxThreadsPerBlock + 1;
    test_vector_add_kernel<<<1, invalidBlockSize>>>(d_a, d_b, d_c, N);
    cudaError_t error2 = cudaGetLastError();

    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 有效配置应该成功，无效配置应该失败
    return (error1 == cudaSuccess) && (error2 != cudaSuccess);
}

int main() {
    std::cout << "Example Execution Unit Tests" << std::endl;
    std::cout << "=============================" << std::endl;

    // 初始化错误处理
    cuda_learning::ErrorHandler::initialize("test_example_execution.log");

    // 运行测试
    TestRunner::run_test("Hello World Example", test_hello_world_example);
    TestRunner::run_test("Vector Addition Example", test_vector_addition_example);
    TestRunner::run_test("Matrix Addition Example", test_matrix_addition_example);
    TestRunner::run_test("Matrix Transpose Example", test_matrix_transpose_example);
    TestRunner::run_test("Performance Measurement", test_performance_measurement);
    TestRunner::run_test("Memory Bandwidth Calculation", test_memory_bandwidth_calculation);
    TestRunner::run_test("Error Handling Robustness", test_error_handling_robustness);
    TestRunner::run_test("Thread Configuration Validation", test_thread_configuration_validation);

    // 打印总结
    TestRunner::print_summary();

    // 清理
    cuda_learning::ErrorHandler::shutdown();

    return TestRunner::get_exit_code();
}
