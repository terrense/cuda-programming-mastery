/**
 * Hello World CUDA程序模板
 * 
 * 这是最简单的CUDA程序示例，展示：
 * 1. CUDA程序的基本结构
 * 2. 核函数的定义和调用
 * 3. 基本的内存管理
 * 4. 错误检查的重要性
 * 
 * 学习目标：
 * - 理解CUDA程序的基本组成部分
 * - 学会编写和调用简单的核函数
 * - 掌握基本的CUDA内存操作
 */

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * 核函数：在GPU上执行的函数
 * 
 * 特点：
 * - 使用__global__修饰符
 * - 返回类型必须是void
 * - 在GPU上并行执行
 * - 由主机代码调用
 */
__global__ void helloWorldKernel(char* message, int messageLength) {
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保不越界访问
    if (idx < messageLength) {
        // 简单的字符变换：将小写字母转换为大写
        if (message[idx] >= 'a' && message[idx] <= 'z') {
            message[idx] = message[idx] - 'a' + 'A';
        }
    }
}

/**
 * 更复杂的Hello World核函数
 * 展示线程块和线程的概念
 */
__global__ void helloWorldWithInfo(char* output, int maxLength) {
    // 获取线程信息
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int globalId = blockId * blockDim.x + threadId;
    
    // 创建消息字符串（简化版本）
    if (globalId == 0 && maxLength > 50) {
        // 只有第一个线程写入消息
        const char* msg = "Hello from CUDA! Block 0, Thread 0";
        for (int i = 0; i < 35 && i < maxLength - 1; ++i) {
            output[i] = msg[i];
        }
        output[35] = '\0';
    }
}

int main() {
    std::cout << "=== Hello World CUDA程序 ===" << std::endl;
    std::cout << "这是您的第一个CUDA程序！\n" << std::endl;
    
    // 1. 检查CUDA设备
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "没有找到CUDA设备！" << std::endl;
        return 1;
    }
    
    std::cout << "找到 " << deviceCount << " 个CUDA设备" << std::endl;
    
    // 获取设备信息
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "使用设备: " << deviceProp.name << std::endl;
    std::cout << "计算能力: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    
    // 2. 第一个示例：字符串大小写转换
    std::cout << "\n--- 示例1: 字符串大小写转换 ---" << std::endl;
    
    std::string hostMessage = "hello cuda world!";
    int messageLength = hostMessage.length();
    
    std::cout << "原始消息: \"" << hostMessage << "\"" << std::endl;
    
    // 在设备上分配内存
    char* deviceMessage = nullptr;
    size_t messageSize = messageLength * sizeof(char);
    CUDA_CHECK(cudaMalloc(&deviceMessage, messageSize));
    
    // 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(deviceMessage, hostMessage.c_str(), messageSize, cudaMemcpyHostToDevice));
    
    // 配置执行参数
    int blockSize = 256;  // 每个线程块的线程数
    int gridSize = (messageLength + blockSize - 1) / blockSize;  // 线程块数
    
    std::cout << "执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
    
    // 启动核函数
    helloWorldKernel<<<gridSize, blockSize>>>(deviceMessage, messageLength);
    
    // 检查核函数执行错误
    CUDA_CHECK(cudaGetLastError());
    
    // 等待GPU完成
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 将结果从设备复制回主机
    std::vector<char> result(messageLength + 1, '\0');
    CUDA_CHECK(cudaMemcpy(result.data(), deviceMessage, messageSize, cudaMemcpyDeviceToHost));
    
    std::cout << "转换后消息: \"" << result.data() << "\"" << std::endl;
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(deviceMessage));
    
    // 3. 第二个示例：线程信息展示
    std::cout << "\n--- 示例2: 线程信息展示 ---" << std::endl;
    
    const int outputSize = 100;
    char* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceOutput, outputSize));
    
    // 初始化设备内存
    CUDA_CHECK(cudaMemset(deviceOutput, 0, outputSize));
    
    // 启动核函数（使用单个线程块）
    helloWorldWithInfo<<<1, 1>>>(deviceOutput, outputSize);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果
    std::vector<char> output(outputSize);
    CUDA_CHECK(cudaMemcpy(output.data(), deviceOutput, outputSize, cudaMemcpyDeviceToHost));
    
    std::cout << "GPU消息: \"" << output.data() << "\"" << std::endl;
    
    CUDA_CHECK(cudaFree(deviceOutput));
    
    // 4. 程序总结
    std::cout << "\n--- 程序总结 ---" << std::endl;
    std::cout << "恭喜！您已经成功运行了第一个CUDA程序！" << std::endl;
    std::cout << "您学到了：" << std::endl;
    std::cout << "  • 如何编写CUDA核函数" << std::endl;
    std::cout << "  • 如何分配和管理GPU内存" << std::endl;
    std::cout << "  • 如何在主机和设备间传输数据" << std::endl;
    std::cout << "  • 如何配置和启动核函数" << std::endl;
    std::cout << "  • 如何进行错误检查" << std::endl;
    
    return 0;
}

/*
编译和运行说明：

1. 编译命令：
   nvcc -o hello_world_cuda hello_world_cuda.cu

2. 运行：
   ./hello_world_cuda

3. 预期输出：
   - 显示CUDA设备信息
   - 将小写字符串转换为大写
   - 显示来自GPU的问候消息

4. 常见问题：
   - 如果编译失败，检查CUDA工具链是否正确安装
   - 如果运行时出错，检查是否有可用的CUDA设备
   - 注意核函数中的线程索引计算

5. 扩展练习：
   - 修改消息内容
   - 尝试不同的线程块配置
   - 添加更多的字符处理逻辑
   - 测量程序执行时间
*/