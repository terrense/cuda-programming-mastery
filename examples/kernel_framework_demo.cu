#include "../src/core/kernel_framework.h"
#include "../src/core/error_handler.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace cuda_learning;

// 示例：使用模板生成器创建向量加法核函数
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // 使用线程索引辅助函数
    int idx = ThreadUtils::getGlobalThreadId1D();

    // 边界检查
    if (ThreadUtils::isValidThread1D(n)) {
        c[idx] = a[idx] + b[idx];
    }
}

// 示例：使用模板生成器创建矩阵乘法核函数
__global__ void matrixMultiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (ThreadUtils::isValidThread2D(N, M)) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void demonstrateKernelFramework() {
    std::cout << "=== CUDA核函数开发框架演示 ===" << std::endl;

    // 1. 演示模板生成器
    std::cout << "\n1. 核函数模板生成器演示:" << std::endl;

    std::string vectorAddTemplate = KernelTemplateGenerator::generateVectorAddTemplate("myVectorAdd");
    std::cout << "向量加法模板:\n" << vectorAddTemplate << std::endl;

    std::string matMulTemplate = KernelTemplateGenerator::generateMatrixMultiplyTemplate("myMatMul");
    std::cout << "矩阵乘法模板:\n" << matMulTemplate << std::endl;

    // 2. 演示网格和块配置优化器
    std::cout << "\n2. 网格和块配置优化演示:" << std::endl;

    // 1D优化示例
    int dataSize = 1000000;
    GridBlockConfig config1D = GridBlockOptimizer::optimize1D(dataSize);
    std::cout << "1D数据优化配置 (size=" << dataSize << "): " << config1D.toString() << std::endl;

    // 2D优化示例
    int width = 1024, height = 1024;
    GridBlockConfig config2D = GridBlockOptimizer::optimize2D(width, height);
    std::cout << "2D数据优化配置 (" << width << "x" << height << "): " << config2D.toString() << std::endl;

    // 矩阵乘法优化示例
    int M = 512, N = 512, K = 512;
    GridBlockConfig configMatMul = GridBlockOptimizer::optimizeMatMul(M, N, K);
    std::cout << "矩阵乘法优化配置 (" << M << "x" << N << "x" << K << "): " << configMatMul.toString() << std::endl;

    // 3. 演示实际核函数执行
    std::cout << "\n3. 实际核函数执行演示:" << std::endl;

    // 向量加法示例
    const int vecSize = 1000;
    std::vector<float> h_a(vecSize, 1.0f);
    std::vector<float> h_b(vecSize, 2.0f);
    std::vector<float> h_c(vecSize, 0.0f);

    float *d_a, *d_b, *d_c;
    size_t bytes = vecSize * sizeof(float);

    // 分配GPU内存
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 复制数据到GPU
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // 使用优化器配置核函数
    GridBlockConfig vecConfig = GridBlockOptimizer::optimize1D(vecSize);
    std::cout << "向量加法配置: " << vecConfig.toString() << std::endl;

    // 启动核函数
    bool success = KernelLauncher::launchWithCheck(vectorAdd, vecConfig, d_a, d_b, d_c, vecSize);

    if (success) {
        // 复制结果回主机
        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        // 验证结果
        bool correct = true;
        for (int i = 0; i < 10; i++) { // 检查前10个元素
            if (abs(h_c[i] - 3.0f) > 1e-5) {
                correct = false;
                break;
            }
        }

        std::cout << "向量加法结果: " << (correct ? "正确" : "错误") << std::endl;
        std::cout << "前5个结果: ";
        for (int i = 0; i < 5; i++) {
            std::cout << h_c[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "核函数执行失败!" << std::endl;
    }

    // 清理内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 4. 演示设备属性查询
    std::cout << "\n4. 设备属性查询:" << std::endl;
    cudaDeviceProp prop = GridBlockOptimizer::getDeviceProperties(0);
    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "最大线程数/块: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "最大块维度: (" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "最大网格维度: (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "多处理器数量: " << prop.multiProcessorCount << std::endl;
}

int main() {
    // 初始化CUDA
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "CUDA初始化失败: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    try {
        demonstrateKernelFramework();
    } catch (const std::exception& e) {
        std::cerr << "演示过程中发生错误: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n核函数开发框架演示完成!" << std::endl;
    return 0;
}
