#include "../../src/core/multi_gpu_framework.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace cuda_learning;

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

// 演示数据并行训练
void demonstrateDataParallelTraining() {
    std::cout << "\n=== 数据并行训练演示 ===\n";

    // 初始化多GPU上下文
    MultiGPUContext context;
    std::vector<int> deviceIds = multi_gpu_utils::getOptimalDeviceSet(2);

    if (deviceIds.empty()) {
        std::cout << "没有找到可用的GPU设备\n";
        return;
    }

    if (!context.initialize(deviceIds)) {
        std::cout << "多GPU上下文初始化失败: " << context.getLastError() << "\n";
        return;
    }

    std::cout << "成功初始化 " << deviceIds.size() << " 个GPU设备\n";

    // 创建训练数据
    const int dataSize = 1024 * 1024; // 1M floats
    const int batchSize = 256;
    std::vector<float> trainingData(dataSize);

    // 初始化随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (auto& val : trainingData) {
        val = dis(gen);
    }

    // 创建多GPU训练器
    MultiGPUTrainer trainer(&context);

    // 定义训练步骤
    auto trainStep = [](void* data, size_t size, int batch) {
        // 模拟训练操作
        float* floatData = static_cast<float*>(data);
        int numElements = size / sizeof(float);

        // 启动向量加法核函数作为训练操作的模拟
        dim3 blockSize(256);
        dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);

        // 这里简化为自加操作
        vectorAdd<<<gridSize, blockSize>>>(floatData, floatData, floatData, numElements);
        cudaDeviceSynchronize();
    };

    // 执行数据并行训练
    auto startTime = std::chrono::high_resolution_clock::now();

    bool success = trainer.trainDataParallel(
        trainingData.data(),
        trainingData.size() * sizeof(float),
        batchSize,
        trainStep
    );

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (success) {
        std::cout << "数据并行训练成功完成\n";
        std::cout << "训练时间: " << duration.count() << " ms\n";
    } else {
        std::cout << "数据并行训练失败\n";
    }

    // 生成性能报告
    std::cout << multi_gpu_utils::generateMultiGPUReport(context);

    context.finalize();
}

// 演示批处理推理
void demonstrateBatchInference() {
    std::cout << "\n=== 批处理推理演示 ===\n";

    // 初始化多GPU上下文
    MultiGPUContext context;
    std::vector<int> deviceIds = multi_gpu_utils::getOptimalDeviceSet(2);

    if (deviceIds.empty()) {
        std::cout << "没有找到可用的GPU设备\n";
        return;
    }

    if (!context.initialize(deviceIds)) {
        std::cout << "多GPU上下文初始化失败: " << context.getLastError() << "\n";
        return;
    }

    std::cout << "成功初始化 " << deviceIds.size() << " 个GPU设备进行推理\n";

    // 创建输入和输出数据
    const int batchSize = 1024;
    const int inputSize = batchSize * 512; // 每个样本512个float
    const int outputSize = batchSize * 10;  // 每个样本10个输出

    std::vector<float> inputData(inputSize);
    std::vector<float> outputData(outputSize, 0.0f);

    // 初始化输入数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (auto& val : inputData) {
        val = dis(gen);
    }

    // 创建多GPU推理器
    MultiGPUInference inference(&context);

    // 定义推理步骤
    auto inferenceStep = [](void* input, void* output, int deviceId) {
        float* inputPtr = static_cast<float*>(input);
        float* outputPtr = static_cast<float*>(output);

        // 模拟推理操作：简单的线性变换
        int inputElements = 512;  // 每个样本的输入大小
        int outputElements = 10;  // 每个样本的输出大小

        dim3 blockSize(256);
        dim3 gridSize((outputElements + blockSize.x - 1) / blockSize.x);

        // 简化的推理核函数（实际应该是更复杂的神经网络操作）
        // 这里只是将输入的前10个元素复制到输出
        cudaMemcpy(outputPtr, inputPtr, outputElements * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    };

    // 执行批处理推理
    auto startTime = std::chrono::high_resolution_clock::now();

    bool success = inference.inferBatch(
        inputData.data(),
        inputData.size() * sizeof(float),
        batchSize,
        outputData.data(),
        outputData.size() * sizeof(float),
        inferenceStep
    );

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (success) {
        std::cout << "批处理推理成功完成\n";
        std::cout << "推理时间: " << duration.count() << " ms\n";
        std::cout << "处理了 " << batchSize << " 个样本\n";

        // 显示部分结果
        std::cout << "前5个样本的输出结果:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "样本 " << i << ": ";
            for (int j = 0; j < 5; ++j) {
                std::cout << outputData[i * 10 + j] << " ";
            }
            std::cout << "...\n";
        }
    } else {
        std::cout << "批处理推理失败\n";
    }

    context.finalize();
}

// 演示NCCL通信操作
void demonstrateNCCLCommunication() {
    std::cout << "\n=== NCCL通信操作演示 ===\n";

    // 获取可用设备
    std::vector<int> deviceIds = multi_gpu_utils::getOptimalDeviceSet(2);

    if (deviceIds.size() < 2) {
        std::cout << "需要至少2个GPU设备进行通信演示\n";
        return;
    }

    // 初始化NCCL通信器
    NCCLCommunicator communicator;
    if (!communicator.initialize(deviceIds)) {
        std::cout << "NCCL通信器初始化失败\n";
        return;
    }

    std::cout << "成功初始化NCCL通信器，设备数量: " << deviceIds.size() << "\n";

    // 创建测试数据
    const int dataSize = 1024;
    std::vector<float> hostData(dataSize);

    // 初始化数据
    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = static_cast<float>(i);
    }

    // 在每个设备上分配内存并复制数据
    std::vector<float*> deviceBuffers(deviceIds.size());

    for (size_t i = 0; i < deviceIds.size(); ++i) {
        cudaSetDevice(deviceIds[i]);
        cudaMalloc(&deviceBuffers[i], dataSize * sizeof(float));

        // 每个设备的数据稍有不同
        std::vector<float> deviceData = hostData;
        for (auto& val : deviceData) {
            val += static_cast<float>(i); // 设备偏移
        }

        cudaMemcpy(deviceBuffers[i], deviceData.data(),
                  dataSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::cout << "在各设备上准备了测试数据\n";

    // 演示AllReduce操作
    std::cout << "执行AllReduce操作...\n";
    auto startTime = std::chrono::high_resolution_clock::now();

    bool success = communicator.allReduce(
        deviceBuffers[0], deviceBuffers[0], dataSize,
        ncclFloat, ncclSum, deviceIds
    );

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    if (success) {
        std::cout << "AllReduce操作成功完成\n";
        std::cout << "通信时间: " << duration.count() << " μs\n";

        // 验证结果
        std::vector<float> result(dataSize);
        cudaSetDevice(deviceIds[0]);
        cudaMemcpy(result.data(), deviceBuffers[0],
                  dataSize * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "AllReduce结果前10个元素: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << "\n";

        // 估算通信带宽
        float dataGB = (dataSize * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float bandwidth = dataGB / (duration.count() / 1000000.0f);
        std::cout << "估算通信带宽: " << bandwidth << " GB/s\n";

    } else {
        std::cout << "AllReduce操作失败\n";
    }

    // 清理设备内存
    for (size_t i = 0; i < deviceIds.size(); ++i) {
        cudaSetDevice(deviceIds[i]);
        cudaFree(deviceBuffers[i]);
    }

    communicator.finalize();
}

// 演示负载均衡
void demonstrateLoadBalancing() {
    std::cout << "\n=== 负载均衡演示 ===\n";

    // 获取可用设备
    std::vector<int> deviceIds = multi_gpu_utils::getOptimalDeviceSet(4);

    if (deviceIds.empty()) {
        std::cout << "没有找到可用的GPU设备\n";
        return;
    }

    // 创建GPU设备信息
    std::vector<GPUDevice> devices;
    for (int deviceId : deviceIds) {
        GPUDevice device;
        device.deviceId = deviceId;

        cudaSetDevice(deviceId);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);

        device.name = prop.name;
        device.totalMemory = prop.totalGlobalMem;
        device.computeCapability = prop.major * 10 + prop.minor;

        size_t free, total;
        cudaMemGetInfo(&free, &total);
        device.freeMemory = free;
        device.isActive = true;

        devices.push_back(device);
    }

    // 测试不同的负载均衡策略
    std::vector<LoadBalanceStrategy> strategies = {
        LoadBalanceStrategy::ROUND_ROBIN,
        LoadBalanceStrategy::MEMORY_AWARE,
        LoadBalanceStrategy::COMPUTE_AWARE,
        LoadBalanceStrategy::DYNAMIC_ADAPTIVE
    };

    std::vector<std::string> strategyNames = {
        "轮询策略",
        "内存感知策略",
        "计算感知策略",
        "动态自适应策略"
    };

    for (size_t i = 0; i < strategies.size(); ++i) {
        std::cout << "\n测试 " << strategyNames[i] << ":\n";

        LoadBalancer balancer(strategies[i]);
        balancer.setAvailableDevices(devices);

        // 模拟不同的设备负载
        balancer.updateDeviceLoad(deviceIds[0], 0.8f);  // 高负载
        balancer.updateDeviceLoad(deviceIds[1], 0.3f);  // 低负载
        if (deviceIds.size() > 2) {
            balancer.updateDeviceLoad(deviceIds[2], 0.6f);  // 中等负载
        }
        if (deviceIds.size() > 3) {
            balancer.updateDeviceLoad(deviceIds[3], 0.2f);  // 很低负载
        }

        // 创建模拟任务
        MultiGPUTask dummyTask("test_task", [](int deviceId, const DataShard& shard) {
            // 空任务
        });

        // 测试设备选择
        std::map<int, int> selectionCount;
        const int numSelections = 100;

        for (int j = 0; j < numSelections; ++j) {
            int selectedDevice = balancer.selectDevice(dummyTask);
            selectionCount[selectedDevice]++;
        }

        // 显示选择统计
        std::cout << "设备选择统计 (100次选择):\n";
        for (const auto& pair : selectionCount) {
            float percentage = (pair.second * 100.0f) / numSelections;
            std::cout << "  设备 " << pair.first << ": " << pair.second
                     << " 次 (" << percentage << "%)\n";
        }

        // 计算负载均衡效率
        auto loads = balancer.getDeviceLoads();
        float efficiency = multi_gpu_utils::calculateLoadBalanceEfficiency(loads);
        std::cout << "负载均衡效率: " << (efficiency * 100.0f) << "%\n";
    }
}

int main() {
    std::cout << "CUDA多GPU编程框架演示程序\n";
    std::cout << "========================================\n";

    // 检查CUDA环境
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "错误: 没有找到CUDA设备\n";
        return -1;
    }

    std::cout << "检测到 " << deviceCount << " 个CUDA设备\n";

    // 显示设备信息
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "设备 " << i << ": " << prop.name
                 << " (计算能力: " << prop.major << "." << prop.minor << ")\n";
    }

    try {
        // 运行各种演示
        demonstrateLoadBalancing();

        if (deviceCount >= 2) {
            demonstrateNCCLCommunication();
            demonstrateDataParallelTraining();
            demonstrateBatchInference();
        } else {
            std::cout << "\n注意: 需要至少2个GPU设备才能演示多GPU通信和并行功能\n";
        }

    } catch (const std::exception& e) {
        std::cout << "演示过程中发生异常: " << e.what() << "\n";
        return -1;
    }

    std::cout << "\n多GPU框架演示完成!\n";
    return 0;
}
