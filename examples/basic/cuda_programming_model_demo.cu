#include "cuda_environment.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

/**
 * CUDA编程模型可视化演示程序
 * 
 * 本程序演示CUDA编程模型的核心概念：
 * 1. 主机(Host)和设备(Device)的概念
 * 2. 内存管理模型
 * 3. 核函数执行模型
 * 4. 线程层次结构
 */

// 简单的CUDA核函数用于演示
__global__ void demoKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f + threadIdx.x * 0.1f;
    }
}

// 演示线程层次结构的核函数
__global__ void threadHierarchyDemo(float* output, int width, int height) {
    // 计算全局线程索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;
    
    if (col < width && row < height) {
        // 将线程信息编码到输出中
        float blockInfo = blockIdx.x * 1000.0f + blockIdx.y;
        float threadInfo = threadIdx.x * 10.0f + threadIdx.y;
        output[idx] = blockInfo + threadInfo * 0.001f;
    }
}

namespace cuda_learning {

class CudaProgrammingModelDemo {
public:
    CudaProgrammingModelDemo() : m_cudaEnv() {}
    
    void runDemo() {
        std::cout << "=== CUDA编程模型可视化演示 ===" << std::endl;
        std::cout << "本演示将展示CUDA编程的核心概念\n" << std::endl;
        
        if (!m_cudaEnv.isCudaAvailable()) {
            std::cerr << "CUDA不可用，无法运行演示" << std::endl;
            return;
        }
        
        // 1. 主机和设备概念演示
        demonstrateHostDevice();
        
        // 2. 内存管理模型演示
        demonstrateMemoryModel();
        
        // 3. 核函数执行模型演示
        demonstrateKernelExecution();
        
        // 4. 线程层次结构演示
        demonstrateThreadHierarchy();
        
        std::cout << "\n=== 演示完成 ===" << std::endl;
        std::cout << "您已经了解了CUDA编程模型的核心概念！" << std::endl;
    }

private:
    cuda_learning::CudaEnvironment m_cudaEnv;
    
    void demonstrateHostDevice() {
        std::cout << "1. 主机(Host)和设备(Device)概念" << std::endl;
        std::cout << "================================" << std::endl;
        
        std::cout << "CUDA编程模型基于主机-设备架构：" << std::endl;
        std::cout << "  • 主机(Host): CPU及其内存空间" << std::endl;
        std::cout << "  • 设备(Device): GPU及其内存空间" << std::endl;
        std::cout << "  • 两者有独立的内存空间，需要显式数据传输" << std::endl;
        
        // 显示当前主机和设备信息
        std::cout << "\n当前系统配置：" << std::endl;
        std::cout << "主机信息：" << std::endl;
        std::cout << "  • CPU: " << "x86_64 架构" << std::endl; // 简化显示
        std::cout << "  • 主机内存: 系统RAM" << std::endl;
        
        auto bestGPU = m_cudaEnv.getBestGPU();
        std::cout << "设备信息：" << std::endl;
        std::cout << "  • GPU: " << bestGPU.name << std::endl;
        std::cout << "  • 设备内存: " << bestGPU.totalMemory / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "  • 计算能力: " << bestGPU.computeCapabilityMajor 
                  << "." << bestGPU.computeCapabilityMinor << std::endl;
        
        waitForUser();
    }
    
    void demonstrateMemoryModel() {
        std::cout << "\n2. 内存管理模型演示" << std::endl;
        std::cout << "=====================" << std::endl;
        
        std::cout << "CUDA内存管理的基本步骤：" << std::endl;
        std::cout << "  1. 在设备上分配内存" << std::endl;
        std::cout << "  2. 将数据从主机复制到设备" << std::endl;
        std::cout << "  3. 在设备上执行计算" << std::endl;
        std::cout << "  4. 将结果从设备复制回主机" << std::endl;
        std::cout << "  5. 释放设备内存" << std::endl;
        
        // 实际演示内存操作
        const int N = 1024;
        const size_t size = N * sizeof(float);
        
        std::cout << "\n实际演示 (数组大小: " << N << " 个float):" << std::endl;
        
        // 1. 主机内存分配
        std::cout << "  1. 分配主机内存..." << std::endl;
        std::vector<float> h_data(N);
        for (int i = 0; i < N; ++i) {
            h_data[i] = static_cast<float>(i);
        }
        
        // 2. 设备内存分配
        std::cout << "  2. 分配设备内存..." << std::endl;
        float* d_data = nullptr;
        cudaError_t error = cudaMalloc(&d_data, size);
        if (error != cudaSuccess) {
            std::cerr << "设备内存分配失败: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        // 3. 数据传输：主机到设备
        std::cout << "  3. 数据传输: 主机 -> 设备..." << std::endl;
        error = cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "数据传输失败: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_data);
            return;
        }
        
        // 4. 在设备上执行简单计算
        std::cout << "  4. 在设备上执行计算..." << std::endl;
        dim3 blockSize(256);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
        demoKernel<<<gridSize, blockSize>>>(d_data, N);
        
        // 检查核函数执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "核函数执行失败: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_data);
            return;
        }
        
        // 等待GPU完成
        cudaDeviceSynchronize();
        
        // 5. 数据传输：设备到主机
        std::cout << "  5. 数据传输: 设备 -> 主机..." << std::endl;
        std::vector<float> h_result(N);
        error = cudaMemcpy(h_result.data(), d_data, size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "结果传输失败: " << cudaGetErrorString(error) << std::endl;
            cudaFree(d_data);
            return;
        }
        
        // 6. 释放设备内存
        std::cout << "  6. 释放设备内存..." << std::endl;
        cudaFree(d_data);
        
        // 显示部分结果
        std::cout << "\n计算结果示例 (前10个元素):" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "  h_result[" << i << "] = " << std::fixed 
                      << std::setprecision(2) << h_result[i] << std::endl;
        }
        
        waitForUser();
    }
    
    void demonstrateKernelExecution() {
        std::cout << "\n3. 核函数执行模型演示" << std::endl;
        std::cout << "=======================" << std::endl;
        
        std::cout << "核函数执行的关键概念：" << std::endl;
        std::cout << "  • 核函数在GPU上并行执行" << std::endl;
        std::cout << "  • 使用<<<grid, block>>>语法启动" << std::endl;
        std::cout << "  • grid定义线程块的组织方式" << std::endl;
        std::cout << "  • block定义每个线程块内线程的组织方式" << std::endl;
        
        const int N = 1000;
        std::cout << "\n演示不同的执行配置 (数组大小: " << N << "):" << std::endl;
        
        // 配置1: 单个大线程块
        int blockSize1 = 256;
        int gridSize1 = (N + blockSize1 - 1) / blockSize1;
        std::cout << "  配置1: <<<" << gridSize1 << ", " << blockSize1 << ">>>" << std::endl;
        std::cout << "    - 网格大小: " << gridSize1 << " 个线程块" << std::endl;
        std::cout << "    - 线程块大小: " << blockSize1 << " 个线程" << std::endl;
        std::cout << "    - 总线程数: " << gridSize1 * blockSize1 << std::endl;
        
        // 配置2: 多个小线程块
        int blockSize2 = 128;
        int gridSize2 = (N + blockSize2 - 1) / blockSize2;
        std::cout << "  配置2: <<<" << gridSize2 << ", " << blockSize2 << ">>>" << std::endl;
        std::cout << "    - 网格大小: " << gridSize2 << " 个线程块" << std::endl;
        std::cout << "    - 线程块大小: " << blockSize2 << " 个线程" << std::endl;
        std::cout << "    - 总线程数: " << gridSize2 * blockSize2 << std::endl;
        
        // 配置3: 2D网格配置
        int width = 32, height = 32;
        dim3 blockSize3(16, 16);
        dim3 gridSize3((width + blockSize3.x - 1) / blockSize3.x,
                       (height + blockSize3.y - 1) / blockSize3.y);
        std::cout << "  配置3 (2D): <<<(" << gridSize3.x << "," << gridSize3.y 
                  << "), (" << blockSize3.x << "," << blockSize3.y << ")>>>" << std::endl;
        std::cout << "    - 网格大小: " << gridSize3.x << "x" << gridSize3.y << " 个线程块" << std::endl;
        std::cout << "    - 线程块大小: " << blockSize3.x << "x" << blockSize3.y << " 个线程" << std::endl;
        std::cout << "    - 总线程数: " << gridSize3.x * gridSize3.y * blockSize3.x * blockSize3.y << std::endl;
        
        std::cout << "\n选择执行配置的考虑因素：" << std::endl;
        std::cout << "  • 线程块大小应该是32(warp大小)的倍数" << std::endl;
        std::cout << "  • 考虑GPU的SM数量和每个SM的最大线程数" << std::endl;
        std::cout << "  • 平衡占用率和资源使用" << std::endl;
        
        waitForUser();
    }
    
    void demonstrateThreadHierarchy() {
        std::cout << "\n4. 线程层次结构演示" << std::endl;
        std::cout << "=====================" << std::endl;
        
        std::cout << "CUDA线程层次结构：" << std::endl;
        std::cout << "  Grid (网格)" << std::endl;
        std::cout << "    └── Block (线程块)" << std::endl;
        std::cout << "          └── Thread (线程)" << std::endl;
        std::cout << "                └── Warp (线程束, 32个线程)" << std::endl;
        
        // 实际演示线程层次结构
        const int width = 8, height = 6;
        const size_t size = width * height * sizeof(float);
        
        std::cout << "\n实际演示 (2D网格: " << width << "x" << height << "):" << std::endl;
        
        // 分配内存
        float* h_output = new float[width * height];
        float* d_output = nullptr;
        cudaMalloc(&d_output, size);
        
        // 配置执行参数
        dim3 blockSize(4, 3);  // 每个线程块 4x3 = 12 个线程
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        std::cout << "执行配置：" << std::endl;
        std::cout << "  • 网格大小: " << gridSize.x << "x" << gridSize.y << " = " 
                  << gridSize.x * gridSize.y << " 个线程块" << std::endl;
        std::cout << "  • 线程块大小: " << blockSize.x << "x" << blockSize.y << " = " 
                  << blockSize.x * blockSize.y << " 个线程" << std::endl;
        std::cout << "  • 总线程数: " << gridSize.x * gridSize.y * blockSize.x * blockSize.y << std::endl;
        
        // 执行核函数
        threadHierarchyDemo<<<gridSize, blockSize>>>(d_output, width, height);
        cudaDeviceSynchronize();
        
        // 复制结果
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        
        // 显示结果矩阵，展示线程层次结构
        std::cout << "\n结果矩阵 (显示线程块和线程信息):" << std::endl;
        std::cout << "格式: [blockIdx.x*1000 + blockIdx.y + threadIdx.x*0.01 + threadIdx.y*0.001]" << std::endl;
        
        for (int row = 0; row < height; ++row) {
            std::cout << "  ";
            for (int col = 0; col < width; ++col) {
                int idx = row * width + col;
                std::cout << std::fixed << std::setprecision(3) << std::setw(8) << h_output[idx] << " ";
            }
            std::cout << std::endl;
        }
        
        // 解释结果
        std::cout << "\n结果解释：" << std::endl;
        std::cout << "  • 整数部分表示线程块索引 (blockIdx.x*1000 + blockIdx.y)" << std::endl;
        std::cout << "  • 小数部分表示线程在块内的索引" << std::endl;
        std::cout << "  • 相同整数部分的元素属于同一个线程块" << std::endl;
        
        // 清理内存
        delete[] h_output;
        cudaFree(d_output);
        
        // 显示线程束概念
        std::cout << "\n线程束(Warp)概念：" << std::endl;
        std::cout << "  • 32个连续线程组成一个线程束" << std::endl;
        std::cout << "  • 线程束是GPU调度和执行的基本单位" << std::endl;
        std::cout << "  • 同一线程束内的线程执行相同指令(SIMT)" << std::endl;
        std::cout << "  • 线程发散会降低性能" << std::endl;
        
        waitForUser();
    }
    
    void waitForUser() {
        std::cout << "\n按回车键继续..." << std::endl;
        std::cin.get();
    }
};

} // namespace cuda_learning

int main() {
    // 初始化错误处理
    cuda_learning::ErrorHandler::initialize("cuda_programming_model_demo.log");
    
    try {
        cuda_learning::CudaProgrammingModelDemo demo;
        demo.runDemo();
    } catch (const std::exception& e) {
        std::cerr << "演示执行出错: " << e.what() << std::endl;
        cuda_learning::ErrorHandler::shutdown();
        return 1;
    }
    
    cuda_learning::ErrorHandler::shutdown();
    return 0;
}