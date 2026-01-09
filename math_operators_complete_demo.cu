#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <vector>
#include <cmath>

// ============================================================================
// CUDA核函数定义
// ============================================================================

// 元素级运算核函数
__global__ void elementwise_subtract_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void elementwise_multiply_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwise_divide_kernel(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (b[idx] != 0.0f) ? (a[idx] / b[idx]) : 0.0f;
    }
}

// 激活函数核函数
__global__ void sigmoid_kernel(const float* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(const float* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

// 优化矩阵乘法核函数（使用共享内存）
template<int TILE_SIZE>
__global__ void optimized_matmul_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载数据到共享内存
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 归约运算核函数
__global__ void reduce_sum_kernel(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 归约求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void reduce_max_kernel(const float* input, float* output, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();

    // 归约求最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(sdata[0]));
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

// 性能测试辅助函数
template<typename Func>
double measureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // 返回毫秒
}

// 打印数组（仅用于小数组）
void printArray(const float* data, int size, const std::string& name) {
    std::cout << name << ": ";
    int print_size = std::min(size, 16);
    for (int i = 0; i < print_size; ++i) {
        std::cout << data[i] << " ";
    }
    if (size > 16) std::cout << "...";
    std::cout << std::endl;
}

// ============================================================================
// 测试函数
// ============================================================================

void testElementwiseOperations() {
    std::cout << "\n=== 测试元素级运算算子 ===" << std::endl;

    const int N = 16;
    const size_t size = N * sizeof(float);

    // 主机内存分配
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i + 1);
        h_b[i] = static_cast<float>(i + 1) * 0.5f;
    }

    printArray(h_a, N, "输入A");
    printArray(h_b, N, "输入B");

    // 设备内存分配
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 数据传输到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 测试减法
    auto time = measureTime([&]() {
        elementwise_subtract_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    });
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printArray(h_c, N, "A - B");
    std::cout << "减法执行时间: " << time << " ms" << std::endl;

    // 测试乘法
    time = measureTime([&]() {
        elementwise_multiply_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    });
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printArray(h_c, N, "A * B");
    std::cout << "乘法执行时间: " << time << " ms" << std::endl;

    // 测试除法
    time = measureTime([&]() {
        elementwise_divide_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    });
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printArray(h_c, N, "A / B");
    std::cout << "除法执行时间: " << time << " ms" << std::endl;

    // 清理内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void testActivationFunctions() {
    std::cout << "\n=== 测试激活函数算子 ===" << std::endl;

    const int N = 9;
    const size_t size = N * sizeof(float);

    // 主机内存分配
    float *h_input = new float[N];
    float *h_output = new float[N];

    // 初始化数据（范围从-2到2）
    for (int i = 0; i < N; i++) {
        h_input[i] = -2.0f + i * 0.5f;
    }

    printArray(h_input, N, "输入");

    // 设备内存分配
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // 数据传输到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 测试Sigmoid
    auto time = measureTime([&]() {
        sigmoid_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    });
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printArray(h_output, N, "Sigmoid输出");
    std::cout << "Sigmoid执行时间: " << time << " ms" << std::endl;

    // 测试Tanh
    time = measureTime([&]() {
        tanh_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    });
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printArray(h_output, N, "Tanh输出");
    std::cout << "Tanh执行时间: " << time << " ms" << std::endl;

    // 清理内存
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

void testOptimizedMatMul() {
    std::cout << "\n=== 测试优化矩阵乘法算子 ===" << std::endl;

    const int M = 64, K = 32, N = 64;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // 主机内存分配
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // 初始化数据
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    std::cout << "矩阵A: " << M << "x" << K << std::endl;
    std::cout << "矩阵B: " << K << "x" << N << std::endl;

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 数据传输到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 测试不同分块大小
    std::vector<int> tile_sizes = {16, 32};

    for (int tile_size : tile_sizes) {
        dim3 blockDim(tile_size, tile_size);
        dim3 gridDim((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);

        auto time = measureTime([&]() {
            if (tile_size == 16) {
                optimized_matmul_kernel<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            } else if (tile_size == 32) {
                optimized_matmul_kernel<32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            }
        });

        std::cout << "分块大小 " << tile_size << "x" << tile_size
                  << " 执行时间: " << time << " ms" << std::endl;

        // 验证结果的一些统计信息
        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
        float sum = 0.0f, max_val = h_C[0], min_val = h_C[0];
        for (int i = 0; i < M * N; i++) {
            sum += h_C[i];
            max_val = std::max(max_val, h_C[i]);
            min_val = std::min(min_val, h_C[i]);
        }
        float mean = sum / (M * N);

        std::cout << "结果统计 - 均值: " << mean << ", 最大值: " << max_val
                  << ", 最小值: " << min_val << std::endl;
    }

    // 清理内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void testReductionOperations() {
    std::cout << "\n=== 测试归约运算算子 ===" << std::endl;

    const int N = 1000;
    const size_t size = N * sizeof(float);

    // 主机内存分配
    float *h_input = new float[N];
    float h_result;

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    std::cout << "输入张量大小: " << N << " 元素" << std::endl;

    // 设备内存分配
    float *d_input, *d_result;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_result, sizeof(float));

    // 数据传输到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 测试求和
    h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    auto time = measureTime([&]() {
        reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_result, N);
    });

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "求和结果: " << h_result << ", 执行时间: " << time << " ms" << std::endl;

    // 测试最大值
    h_result = -INFINITY;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    time = measureTime([&]() {
        reduce_max_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_result, N);
    });

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "最大值结果: " << h_result << ", 执行时间: " << time << " ms" << std::endl;

    // 计算平均值（求和后除以元素个数）
    h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_result, N);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    h_result /= N;
    std::cout << "平均值结果: " << h_result << std::endl;

    // 清理内存
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_result);
}

int main() {
    std::cout << "CUDA基础数学算子完整演示程序" << std::endl;
    std::cout << "==============================" << std::endl;

    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "错误: 未找到CUDA设备" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "使用设备: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;

    try {
        // 运行各项测试
        testElementwiseOperations();
        testActivationFunctions();
        testOptimizedMatMul();
        testReductionOperations();

        std::cout << "\n所有测试完成！" << std::endl;
        std::cout << "\n实现的算子包括:" << std::endl;
        std::cout << "1. 元素级运算: 减法、乘法、除法" << std::endl;
        std::cout << "2. 激活函数: Sigmoid、Tanh" << std::endl;
        std::cout << "3. 优化矩阵乘法: 使用共享内存和分块优化" << std::endl;
        std::cout << "4. 归约运算: 求和、最大值、平均值" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
