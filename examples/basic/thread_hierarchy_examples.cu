#include "cuda_environment.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

/**
 * 线程层次结构实际代码示例
 * 
 * 本程序提供多个实际的CUDA代码示例，展示：
 * 1. 1D线程索引计算
 * 2. 2D线程索引计算
 * 3. 3D线程索引计算
 * 4. 线程束(Warp)级别的操作
 * 5. 线程块内同步
 */

// 1D线程索引示例
__global__ void example1D_threadIndex(float* data, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 将线程信息存储到数据中
        data[idx] = blockIdx.x * 1000.0f + threadIdx.x;
    }
}

// 2D线程索引示例
__global__ void example2D_threadIndex(float* data, int width, int height) {
    // 计算2D线程索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        // 编码线程块和线程信息
        float blockInfo = blockIdx.x * 100.0f + blockIdx.y;
        float threadInfo = threadIdx.x * 10.0f + threadIdx.y;
        data[idx] = blockInfo + threadInfo * 0.01f;
    }
}

// 3D线程索引示例
__global__ void example3D_threadIndex(float* data, int width, int height, int depth) {
    // 计算3D线程索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        // 编码3D位置信息
        data[idx] = x * 10000.0f + y * 100.0f + z;
    }
}

// 线程束级别操作示例
__global__ void exampleWarpOperations(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = input[idx];
        
        // 线程束内的shuffle操作
        float neighbor = __shfl_down_sync(0xffffffff, value, 1);
        
        // 线程束内的归约操作
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
        // 只有每个线程束的第一个线程写入结果
        if (threadIdx.x % 32 == 0) {
            output[idx / 32] = value;
        }
    }
}

// 线程块内同步示例
__global__ void exampleBlockSync(float* data, int n) {
    extern __shared__ float shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 将数据加载到共享内存
    if (idx < n) {
        shared_data[tid] = data[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    // 同步线程块内所有线程
    __syncthreads();
    
    // 在共享内存中进行归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // 第一个线程写回结果
    if (tid == 0) {
        data[blockIdx.x] = shared_data[0];
    }
}

namespace cuda_learning {

class ThreadHierarchyExamples {
public:
    ThreadHierarchyExamples() : m_cudaEnv() {}
    
    void runExamples() {
        std::cout << "=== 线程层次结构实际代码示例 ===" << std::endl;
        std::cout << "本示例展示各种线程索引计算和操作\n" << std::endl;
        
        if (!m_cudaEnv.isCudaAvailable()) {
            std::cerr << "CUDA不可用，无法运行示例" << std::endl;
            return;
        }
        
        // 1. 1D线程索引示例
        example1D_ThreadIndex();
        
        // 2. 2D线程索引示例
        example2D_ThreadIndex();
        
        // 3. 3D线程索引示例
        example3D_ThreadIndex();
        
        // 4. 线程束操作示例
        exampleWarpOperations();
        
        // 5. 线程块同步示例
        exampleBlockSynchronization();
        
        std::cout << "\n=== 示例完成 ===" << std::endl;
        std::cout << "您已经学会了各种线程层次结构的使用方法！" << std::endl;
    }

private:
    cuda_learning::CudaEnvironment m_cudaEnv;
    
    void example1D_ThreadIndex() {
        std::cout << "1. 1D线程索引计算示例" << std::endl;
        std::cout << "========================" << std::endl;
        
        const int N = 32;
        const size_t size = N * sizeof(float);
        
        std::cout << "核函数代码：" << std::endl;
        std::cout << "  int idx = blockIdx.x * blockDim.x + threadIdx.x;" << std::endl;
        std::cout << "  data[idx] = blockIdx.x * 1000.0f + threadIdx.x;" << std::endl;
        
        // 分配内存
        std::vector<float> h_data(N, 0.0f);
        float* d_data = nullptr;
        cudaMalloc(&d_data, size);
        
        // 配置执行参数
        int blockSize = 8;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        std::cout << "\n执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
        std::cout << "  • 网格大小: " << gridSize << " 个线程块" << std::endl;
        std::cout << "  • 线程块大小: " << blockSize << " 个线程" << std::endl;
        
        // 执行核函数
        example1D_threadIndex<<<gridSize, blockSize>>>(d_data, N);
        cudaDeviceSynchronize();
        
        // 复制结果
        cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
        
        // 显示结果
        std::cout << "\n结果 (格式: blockIdx*1000 + threadIdx):" << std::endl;
        for (int i = 0; i < N; ++i) {
            if (i % 8 == 0) std::cout << "  ";
            std::cout << std::setw(6) << static_cast<int>(h_data[i]) << " ";
            if ((i + 1) % 8 == 0) std::cout << std::endl;
        }
        
        // 解释结果
        std::cout << "\n结果解释：" << std::endl;
        std::cout << "  • 0-7: 线程块0，线程0-7 (值: 0-7)" << std::endl;
        std::cout << "  • 8-15: 线程块1，线程0-7 (值: 1000-1007)" << std::endl;
        std::cout << "  • 以此类推..." << std::endl;
        
        cudaFree(d_data);
        waitForUser();
    }
    
    void example2D_ThreadIndex() {
        std::cout << "\n2. 2D线程索引计算示例" << std::endl;
        std::cout << "========================" << std::endl;
        
        const int width = 8, height = 6;
        const size_t size = width * height * sizeof(float);
        
        std::cout << "核函数代码：" << std::endl;
        std::cout << "  int col = blockIdx.x * blockDim.x + threadIdx.x;" << std::endl;
        std::cout << "  int row = blockIdx.y * blockDim.y + threadIdx.y;" << std::endl;
        std::cout << "  int idx = row * width + col;" << std::endl;
        
        // 分配内存
        std::vector<float> h_data(width * height, 0.0f);
        float* d_data = nullptr;
        cudaMalloc(&d_data, size);
        
        // 配置执行参数
        dim3 blockSize(4, 2);  // 4x2 = 8 个线程每块
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        std::cout << "\n执行配置: <<<(" << gridSize.x << "," << gridSize.y 
                  << "), (" << blockSize.x << "," << blockSize.y << ")>>>" << std::endl;
        std::cout << "  • 网格大小: " << gridSize.x << "x" << gridSize.y << " 个线程块" << std::endl;
        std::cout << "  • 线程块大小: " << blockSize.x << "x" << blockSize.y << " 个线程" << std::endl;
        
        // 执行核函数
        example2D_threadIndex<<<gridSize, blockSize>>>(d_data, width, height);
        cudaDeviceSynchronize();
        
        // 复制结果
        cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
        
        // 显示结果矩阵
        std::cout << "\n结果矩阵 (格式: blockIdx*100 + threadIdx*0.01):" << std::endl;
        for (int row = 0; row < height; ++row) {
            std::cout << "  ";
            for (int col = 0; col < width; ++col) {
                int idx = row * width + col;
                std::cout << std::fixed << std::setprecision(2) << std::setw(7) << h_data[idx] << " ";
            }
            std::cout << std::endl;
        }
        
        cudaFree(d_data);
        waitForUser();
    }
    
    void example3D_ThreadIndex() {
        std::cout << "\n3. 3D线程索引计算示例" << std::endl;
        std::cout << "========================" << std::endl;
        
        const int width = 4, height = 3, depth = 2;
        const size_t size = width * height * depth * sizeof(float);
        
        std::cout << "核函数代码：" << std::endl;
        std::cout << "  int x = blockIdx.x * blockDim.x + threadIdx.x;" << std::endl;
        std::cout << "  int y = blockIdx.y * blockDim.y + threadIdx.y;" << std::endl;
        std::cout << "  int z = blockIdx.z * blockDim.z + threadIdx.z;" << std::endl;
        std::cout << "  int idx = z * width * height + y * width + x;" << std::endl;
        
        // 分配内存
        std::vector<float> h_data(width * height * depth, 0.0f);
        float* d_data = nullptr;
        cudaMalloc(&d_data, size);
        
        // 配置执行参数
        dim3 blockSize(2, 2, 2);  // 2x2x2 = 8 个线程每块
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y,
                      (depth + blockSize.z - 1) / blockSize.z);
        
        std::cout << "\n执行配置: <<<(" << gridSize.x << "," << gridSize.y << "," << gridSize.z
                  << "), (" << blockSize.x << "," << blockSize.y << "," << blockSize.z << ")>>>" << std::endl;
        
        // 执行核函数
        example3D_threadIndex<<<gridSize, blockSize>>>(d_data, width, height, depth);
        cudaDeviceSynchronize();
        
        // 复制结果
        cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
        
        // 显示结果（按层显示）
        std::cout << "\n结果 (格式: x*10000 + y*100 + z):" << std::endl;
        for (int z = 0; z < depth; ++z) {
            std::cout << "  层 " << z << ":" << std::endl;
            for (int y = 0; y < height; ++y) {
                std::cout << "    ";
                for (int x = 0; x < width; ++x) {
                    int idx = z * width * height + y * width + x;
                    std::cout << std::setw(6) << static_cast<int>(h_data[idx]) << " ";
                }
                std::cout << std::endl;
            }
        }
        
        cudaFree(d_data);
        waitForUser();
    }
    
    void exampleWarpOperations() {
        std::cout << "\n4. 线程束(Warp)级别操作示例" << std::endl;
        std::cout << "==============================" << std::endl;
        
        const int N = 128;
        const size_t inputSize = N * sizeof(float);
        const size_t outputSize = (N / 32) * sizeof(float);
        
        std::cout << "线程束操作特点：" << std::endl;
        std::cout << "  • 32个线程组成一个线程束" << std::endl;
        std::cout << "  • 线程束内可以进行高效的数据交换" << std::endl;
        std::cout << "  • __shfl_*函数用于线程束内通信" << std::endl;
        
        // 分配内存
        std::vector<float> h_input(N);
        std::vector<float> h_output(N / 32, 0.0f);
        
        // 初始化输入数据
        for (int i = 0; i < N; ++i) {
            h_input[i] = i + 1.0f;
        }
        
        float* d_input = nullptr;
        float* d_output = nullptr;
        cudaMalloc(&d_input, inputSize);
        cudaMalloc(&d_output, outputSize);
        
        // 复制输入数据
        cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice);
        
        // 配置执行参数（确保每个线程块是32的倍数）
        int blockSize = 64;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        std::cout << "\n执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
        std::cout << "输入数据: 1, 2, 3, ..., " << N << std::endl;
        
        // 执行核函数
        exampleWarpOperations<<<gridSize, blockSize>>>(d_input, d_output, N);
        cudaDeviceSynchronize();
        
        // 复制结果
        cudaMemcpy(h_output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);
        
        // 显示结果
        std::cout << "\n线程束归约结果 (每个线程束的和):" << std::endl;
        for (size_t i = 0; i < h_output.size(); ++i) {
            int warpStart = i * 32 + 1;
            int warpEnd = (i + 1) * 32;
            std::cout << "  线程束 " << i << " (元素 " << warpStart << "-" << warpEnd 
                      << "): " << h_output[i] << std::endl;
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        waitForUser();
    }
    
    void exampleBlockSynchronization() {
        std::cout << "\n5. 线程块内同步示例" << std::endl;
        std::cout << "=====================" << std::endl;
        
        const int N = 256;
        const size_t size = N * sizeof(float);
        
        std::cout << "线程块同步概念：" << std::endl;
        std::cout << "  • __syncthreads()同步线程块内所有线程" << std::endl;
        std::cout << "  • 共享内存用于线程块内数据共享" << std::endl;
        std::cout << "  • 常用于归约、扫描等算法" << std::endl;
        
        // 分配内存
        std::vector<float> h_data(N);
        
        // 初始化数据
        for (int i = 0; i < N; ++i) {
            h_data[i] = 1.0f;  // 每个元素都是1
        }
        
        float* d_data = nullptr;
        cudaMalloc(&d_data, size);
        cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
        
        // 配置执行参数
        int blockSize = 64;
        int gridSize = (N + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(float);
        
        std::cout << "\n执行配置: <<<" << gridSize << ", " << blockSize 
                  << ", " << sharedMemSize << ">>>" << std::endl;
        std::cout << "输入: " << N << " 个 1.0" << std::endl;
        
        // 执行核函数
        exampleBlockSync<<<gridSize, blockSize, sharedMemSize>>>(d_data, N);
        cudaDeviceSynchronize();
        
        // 复制结果
        cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
        
        // 显示结果（只显示每个线程块的结果）
        std::cout << "\n每个线程块的归约结果:" << std::endl;
        for (int i = 0; i < gridSize; ++i) {
            std::cout << "  线程块 " << i << ": " << h_data[i] << std::endl;
        }
        
        std::cout << "\n说明：每个线程块将其负责的元素求和" << std::endl;
        std::cout << "      结果应该等于该线程块处理的元素数量" << std::endl;
        
        cudaFree(d_data);
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
    cuda_learning::ErrorHandler::initialize("thread_hierarchy_examples.log");
    
    try {
        cuda_learning::ThreadHierarchyExamples examples;
        examples.runExamples();
    } catch (const std::exception& e) {
        std::cerr << "示例执行出错: " << e.what() << std::endl;
        cuda_learning::ErrorHandler::shutdown();
        return 1;
    }
    
    cuda_learning::ErrorHandler::shutdown();
    return 0;
}