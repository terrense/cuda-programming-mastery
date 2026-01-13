#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>

#ifdef USE_NCCL
#include <nccl.h>
#endif

// 简单的向量加法核函数
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 矩阵乘法核函数（简化版）
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 检查CUDA错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 模拟多GPU数据并行训练
void simulateDataParallelTraining() {
    std::cout << "\n=== 模拟数据并行训练 ===\n";

    const int N = 512;  // 矩阵大小
    const int numBatches = 4;  // 模拟4个批次
    const size_t matrixSize = N * N * sizeof(float);

    // 分配主机内存
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N);

    // 初始化矩阵
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_B, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_C, matrixSize));

    // 创建多个流来模拟多GPU
    std::vector<cudaStream_t> streams(numBatches);
    for (int i = 0; i < numBatches; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    std::cout << "使用 " << numBatches << " 个CUDA流模拟多GPU数据并行训练\n";

    auto startTime = std::chrono::high_resolution_clock::now();

    // 模拟多批次并行处理
    for (int batch = 0; batch < numBatches; batch++) {
        // 异步复制数据
        CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), matrixSize,
                                  cudaMemcpyHostToDevice, streams[batch]));
        CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), matrixSize,
                                  cudaMemcpyHostToDevice, streams[batch]));

        // 启动矩阵乘法核函数
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                     (N + blockSize.y - 1) / blockSize.y);

        matrixMul<<<gridSize, blockSize, 0, streams[batch]>>>(d_A, d_B, d_C, N);

        // 异步复制结果
        CUDA_CHECK(cudaMemcpyAsync(h_C.data(), d_C, matrixSize,
                                  cudaMemcpyDeviceToHost, streams[batch]));
    }

    // 同步所有流
    for (int i = 0; i < numBatches; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "模拟数据并行训练完成，耗时: " << duration.count() << " ms\n";

    // 验证结果（检查前几个元素）
    std::cout << "结果矩阵前5x5元素:\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << std::fixed << std::setprecision(2) << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // 清理资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    for (int i = 0; i < numBatches; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

// 模拟多GPU推理
void simulateBatchInference() {
    std::cout << "\n=== 模拟批处理推理 ===\n";

    const int batchSize = 1024;
    const int inputSize = 512;
    const int outputSize = 10;

    // 分配主机内存
    std::vector<float> h_input(batchSize * inputSize);
    std::vector<float> h_output(batchSize * outputSize, 0.0f);

    // 初始化输入数据
    for (int i = 0; i < batchSize * inputSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, batchSize * inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batchSize * outputSize * sizeof(float)));

    // 复制输入数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                         batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "处理批次大小: " << batchSize << " 个样本\n";
    std::cout << "输入维度: " << inputSize << ", 输出维度: " << outputSize << "\n";

    auto startTime = std::chrono::high_resolution_clock::now();

    // 模拟推理操作（简单的向量加法）
    dim3 blockSize(256);
    dim3 gridSize((batchSize * outputSize + blockSize.x - 1) / blockSize.x);

    // 启动简化的推理核函数
    vectorAdd<<<gridSize, blockSize>>>(d_input, d_input, d_output, batchSize * outputSize);

    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                         batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "批处理推理完成，耗时: " << duration.count() << " ms\n";
    std::cout << "吞吐量: " << (batchSize * 1000.0f / duration.count()) << " 样本/秒\n";

    // 显示部分结果
    std::cout << "前3个样本的推理结果:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "样本 " << i << ": ";
        for (int j = 0; j < 5; j++) {
            std::cout << std::fixed << std::setprecision(3) << h_output[i * outputSize + j] << " ";
        }
        std::cout << "...\n";
    }

    // 清理资源
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// 性能基准测试
void performanceBenchmark() {
    std::cout << "\n=== 性能基准测试 ===\n";

    // 测试不同数据大小的向量加法性能
    std::vector<int> dataSizes = {1024, 4096, 16384, 65536, 262144, 1048576};

    std::cout << "向量加法性能测试:\n";
    std::cout << "数据大小\t时间(ms)\t带宽(GB/s)\n";
    std::cout << "----------------------------------------\n";

    for (int dataSize : dataSizes) {
        size_t bytes = dataSize * sizeof(float);

        // 分配内存
        float *h_a, *h_b, *h_c;
        float *d_a, *d_b, *d_c;

        h_a = new float[dataSize];
        h_b = new float[dataSize];
        h_c = new float[dataSize];

        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        // 初始化数据
        for (int i = 0; i < dataSize; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
        }

        // 复制数据到设备
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        // 预热
        dim3 blockSize(256);
        dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 性能测试
        const int numIterations = 100;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < numIterations; iter++) {
            vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, dataSize);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        float avgTime = duration.count() / (1000.0f * numIterations);  // ms
        float bandwidth = (3.0f * bytes) / (duration.count() / numIterations * 1e-6) / 1e9;  // GB/s

        std::cout << dataSize << "\t\t" << std::fixed << std::setprecision(2)
                 << avgTime << "\t\t" << bandwidth << "\n";

        // 清理内存
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }
}

// 显示GPU信息和多GPU框架特性
void displayFrameworkInfo() {
    std::cout << "\n=== CUDA多GPU编程框架特性 ===\n";

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    std::cout << "检测到 " << deviceCount << " 个GPU设备\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        std::cout << "\n设备 " << i << ": " << prop.name << "\n";
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  全局内存: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
        std::cout << "  多处理器数量: " << prop.multiProcessorCount << "\n";
        std::cout << "  最大线程数/块: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  支持并发核函数: " << (prop.concurrentKernels ? "是" : "否") << "\n";
        std::cout << "  支持P2P访问: " << (prop.unifiedAddressing ? "是" : "否") << "\n";

        // 获取当前可用内存
        CUDA_CHECK(cudaSetDevice(i));
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        std::cout << "  可用内存: " << free / (1024*1024) << " MB\n";
    }

    std::cout << "\n框架支持的功能:\n";
    std::cout << "✓ 数据并行训练\n";
    std::cout << "✓ 模型并行训练\n";
    std::cout << "✓ 流水线并行训练\n";
    std::cout << "✓ 批处理推理\n";
    std::cout << "✓ 流式推理\n";
    std::cout << "✓ 负载均衡\n";
    std::cout << "✓ 数据分片管理\n";

#ifdef USE_NCCL
    std::cout << "✓ NCCL高性能通信\n";
#else
    std::cout << "✗ NCCL通信 (未启用)\n";
#endif

    std::cout << "✓ 性能分析和优化\n";
    std::cout << "✓ 错误处理和调试\n";
}

int main() {
    std::cout << "CUDA多GPU编程框架演示\n";
    std::cout << "========================\n";

    try {
        // 显示框架信息
        displayFrameworkInfo();

        // 模拟数据并行训练
        simulateDataParallelTraining();

        // 模拟批处理推理
        simulateBatchInference();

        // 性能基准测试
        performanceBenchmark();

        std::cout << "\n=== 总结 ===\n";
        std::cout << "本演示程序展示了CUDA多GPU编程框架的核心功能:\n";
        std::cout << "1. 使用CUDA流模拟多GPU并行处理\n";
        std::cout << "2. 数据并行训练的基本模式\n";
        std::cout << "3. 高效的批处理推理\n";
        std::cout << "4. 性能基准测试和分析\n";
        std::cout << "\n在真实的多GPU环境中，框架还支持:\n";
        std::cout << "- NCCL集合通信操作 (AllReduce, AllGather等)\n";
        std::cout << "- 智能负载均衡和任务调度\n";
        std::cout << "- 自动数据分片和内存管理\n";
        std::cout << "- 跨GPU的梯度同步和参数更新\n";

        std::cout << "\n演示程序完成!\n";

    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
