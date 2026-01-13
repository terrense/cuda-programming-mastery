#include "../../src/core/multi_gpu_framework.h"
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <random>

using namespace cuda_learning;

class MultiGPUFrameworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 获取可用设备
        cudaGetDeviceCount(&deviceCount_);

        if (deviceCount_ > 0) {
            // 最多使用2个设备进行测试
            int numDevices = std::min(deviceCount_, 2);
            for (int i = 0; i < numDevices; ++i) {
                deviceIds_.push_back(i);
            }
        }
    }

    void TearDown() override {
        // 清理资源
    }

    int deviceCount_ = 0;
    std::vector<int> deviceIds_;
};

// 测试NCCL通信器初始化
TEST_F(MultiGPUFrameworkTest, NCCLCommunicatorInitialization) {
    if (deviceIds_.size() < 2) {
        GTEST_SKIP() << "需要至少2个GPU设备进行NCCL测试";
    }

    NCCLCommunicator communicator;

    // 测试初始化
    EXPECT_TRUE(communicator.initialize(deviceIds_));

    // 测试获取通信器
    for (int deviceId : deviceIds_) {
        ncclComm_t comm = communicator.getComm(deviceId);
        EXPECT_NE(comm, nullptr);
    }

    // 测试清理
    communicator.finalize();
}

// 测试AllReduce操作
TEST_F(MultiGPUFrameworkTest, NCCLAllReduce) {
    if (deviceIds_.size() < 2) {
        GTEST_SKIP() << "需要至少2个GPU设备进行AllReduce测试";
    }

    NCCLCommunicator communicator;
    ASSERT_TRUE(communicator.initialize(deviceIds_));

    const int dataSize = 1024;
    std::vector<float*> deviceBuffers(deviceIds_.size());

    // 在每个设备上分配内存并初始化数据
    for (size_t i = 0; i < deviceIds_.size(); ++i) {
        cudaSetDevice(deviceIds_[i]);
        cudaMalloc(&deviceBuffers[i], dataSize * sizeof(float));

        // 初始化数据为设备ID
        std::vector<float> hostData(dataSize, static_cast<float>(deviceIds_[i]));
        cudaMemcpy(deviceBuffers[i], hostData.data(),
                  dataSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    // 执行AllReduce
    bool success = communicator.allReduce(
        deviceBuffers[0], deviceBuffers[0], dataSize,
        ncclFloat, ncclSum, deviceIds_
    );

    EXPECT_TRUE(success);

    // 验证结果
    std::vector<float> result(dataSize);
    cudaSetDevice(deviceIds_[0]);
    cudaMemcpy(result.data(), deviceBuffers[0],
              dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 期望的结果是所有设备ID的和
    float expectedSum = 0.0f;
    for (int deviceId : deviceIds_) {
        expectedSum += static_cast<float>(deviceId);
    }

    for (int i = 0; i < 10; ++i) { // 检查前10个元素
        EXPECT_FLOAT_EQ(result[i], expectedSum);
    }

    // 清理内存
    for (size_t i = 0; i < deviceIds_.size(); ++i) {
        cudaSetDevice(deviceIds_[i]);
        cudaFree(deviceBuffers[i]);
    }

    communicator.finalize();
}

// 测试数据分片管理器
TEST_F(MultiGPUFrameworkTest, DataShardManager) {
    if (deviceIds_.empty()) {
        GTEST_SKIP() << "需要至少1个GPU设备进行数据分片测试";
    }

    DataShardManager shardManager;

    // 创建测试数据
    const size_t dataSize = 1024 * sizeof(float);
    std::vector<float> hostData(1024);

    // 初始化数据
    for (size_t i = 0; i < hostData.size(); ++i) {
        hostData[i] = static_cast<float>(i);
    }

    // 创建分片
    auto shards = shardManager.createShards(
        hostData.data(), dataSize, deviceIds_,
        ShardingStrategy::DATA_PARALLEL
    );

    EXPECT_EQ(shards.size(), deviceIds_.size());

    // 验证分片属性
    size_t totalShardSize = 0;
    for (const auto& shard : shards) {
        EXPECT_GE(shard.shardId, 0);
        EXPECT_NE(shard.devicePtr, nullptr);
        EXPECT_GT(shard.size, 0);
        totalShardSize += shard.size;
    }

    // 总分片大小应该大于等于原始数据大小
    EXPECT_GE(totalShardSize, dataSize);

    // 清理分片
    shardManager.cleanupShards(shards);
}

// 测试负载均衡器
TEST_F(MultiGPUFrameworkTest, LoadBalancer) {
    if (deviceIds_.empty()) {
        GTEST_SKIP() << "需要至少1个GPU设备进行负载均衡测试";
    }

    // 创建GPU设备信息
    std::vector<GPUDevice> devices;
    for (int deviceId : deviceIds_) {
        GPUDevice device;
        device.deviceId = deviceId;
        device.name = "Test GPU " + std::to_string(deviceId);
        device.totalMemory = 1024 * 1024 * 1024; // 1GB
        device.freeMemory = 512 * 1024 * 1024;   // 512MB
        device.isActive = true;
        devices.push_back(device);
    }

    // 测试轮询策略
    LoadBalancer balancer(LoadBalanceStrategy::ROUND_ROBIN);
    balancer.setAvailableDevices(devices);

    MultiGPUTask dummyTask("test_task", [](int, const DataShard&) {});

    // 测试设备选择
    std::set<int> selectedDevices;
    for (int i = 0; i < static_cast<int>(deviceIds_.size()) * 2; ++i) {
        int selectedDevice = balancer.selectDevice(dummyTask);
        selectedDevices.insert(selectedDevice);
    }

    // 应该选择了所有可用设备
    EXPECT_EQ(selectedDevices.size(), deviceIds_.size());

    // 测试负载更新
    balancer.updateDeviceLoad(deviceIds_[0], 0.8f);
    auto loads = balancer.getDeviceLoads();
    EXPECT_FLOAT_EQ(loads[deviceIds_[0]], 0.8f);

    // 测试平均负载计算
    float avgLoad = balancer.getAverageLoad();
    EXPECT_GE(avgLoad, 0.0f);
    EXPECT_LE(avgLoad, 1.0f);
}

// 测试多GPU上下文
TEST_F(MultiGPUFrameworkTest, MultiGPUContext) {
    if (deviceIds_.empty()) {
        GTEST_SKIP() << "需要至少1个GPU设备进行上下文测试";
    }

    MultiGPUContext context;

    // 测试初始化
    EXPECT_TRUE(context.initialize(deviceIds_));

    // 测试设备访问
    const auto& devices = context.getDevices();
    EXPECT_EQ(devices.size(), deviceIds_.size());

    for (int deviceId : deviceIds_) {
        EXPECT_TRUE(context.isDeviceActive(deviceId));

        GPUDevice* device = context.getDevice(deviceId);
        EXPECT_NE(device, nullptr);
        EXPECT_EQ(device->deviceId, deviceId);

        cudaStream_t stream = context.getStream(deviceId);
        EXPECT_NE(stream, nullptr);
    }

    // 测试内存分配
    const size_t allocSize = 1024 * sizeof(float);
    void* devicePtr = context.allocateOnDevice(deviceIds_[0], allocSize);
    EXPECT_NE(devicePtr, nullptr);

    // 测试内存释放
    context.freeOnDevice(deviceIds_[0], devicePtr);

    // 测试组件访问
    EXPECT_NE(context.getCommunicator(), nullptr);
    EXPECT_NE(context.getShardManager(), nullptr);
    EXPECT_NE(context.getLoadBalancer(), nullptr);
    EXPECT_NE(context.getExecutor(), nullptr);

    // 测试清理
    context.finalize();
}

// 测试多GPU执行器
TEST_F(MultiGPUFrameworkTest, MultiGPUExecutor) {
    if (deviceIds_.empty()) {
        GTEST_SKIP() << "需要至少1个GPU设备进行执行器测试";
    }

    MultiGPUExecutor executor;

    // 测试初始化
    EXPECT_TRUE(executor.initialize(deviceIds_));

    // 创建测试任务
    std::atomic<int> taskCounter(0);
    MultiGPUTask testTask("test_task",
        [&taskCounter](int deviceId, const DataShard& shard) {
            taskCounter++;
            // 简单的设备同步
            cudaSetDevice(deviceId);
            cudaDeviceSynchronize();
        });

    // 创建测试分片
    std::vector<DataShard> shards;
    for (size_t i = 0; i < deviceIds_.size(); ++i) {
        DataShard shard;
        shard.shardId = static_cast<int>(i);
        shard.deviceId = deviceIds_[i];
        shard.size = 1024;

        // 分配设备内存
        cudaSetDevice(deviceIds_[i]);
        cudaMalloc(&shard.devicePtr, shard.size);

        shards.push_back(shard);
    }

    // 执行任务
    bool success = executor.executeTask(testTask, shards);
    EXPECT_TRUE(success);

    // 验证任务执行次数
    EXPECT_EQ(taskCounter.load(), static_cast<int>(deviceIds_.size()));

    // 清理分片内存
    for (const auto& shard : shards) {
        cudaSetDevice(shard.deviceId);
        cudaFree(shard.devicePtr);
    }

    executor.finalize();
}

// 测试多GPU训练器
TEST_F(MultiGPUFrameworkTest, MultiGPUTrainer) {
    if (deviceIds_.size() < 2) {
        GTEST_SKIP() << "需要至少2个GPU设备进行训练器测试";
    }

    MultiGPUContext context;
    ASSERT_TRUE(context.initialize(deviceIds_));

    MultiGPUTrainer trainer(&context);

    // 创建训练数据
    const int dataSize = 1024;
    std::vector<float> trainingData(dataSize, 1.0f);

    // 定义训练步骤
    std::atomic<int> trainStepCounter(0);
    auto trainStep = [&trainStepCounter](void* data, size_t size, int batch) {
        trainStepCounter++;
        // 模拟训练操作
        cudaDeviceSynchronize();
    };

    // 执行数据并行训练
    bool success = trainer.trainDataParallel(
        trainingData.data(),
        trainingData.size() * sizeof(float),
        32, // batch size
        trainStep
    );

    EXPECT_TRUE(success);
    EXPECT_EQ(trainStepCounter.load(), static_cast<int>(deviceIds_.size()));

    context.finalize();
}

// 测试多GPU推理器
TEST_F(MultiGPUFrameworkTest, MultiGPUInference) {
    if (deviceIds_.size() < 2) {
        GTEST_SKIP() << "需要至少2个GPU设备进行推理器测试";
    }

    MultiGPUContext context;
    ASSERT_TRUE(context.initialize(deviceIds_));

    MultiGPUInference inference(&context);

    // 创建输入和输出数据
    const int batchSize = 64;
    const int inputSize = batchSize * 128;
    const int outputSize = batchSize * 10;

    std::vector<float> inputData(inputSize, 2.0f);
    std::vector<float> outputData(outputSize, 0.0f);

    // 定义推理步骤
    std::atomic<int> inferStepCounter(0);
    auto inferenceStep = [&inferStepCounter](void* input, void* output, int deviceId) {
        inferStepCounter++;
        // 模拟推理操作：简单复制
        cudaMemcpy(output, input, 10 * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    };

    // 执行批处理推理
    bool success = inference.inferBatch(
        inputData.data(),
        inputData.size() * sizeof(float),
        batchSize,
        outputData.data(),
        outputData.size() * sizeof(float),
        inferenceStep
    );

    EXPECT_TRUE(success);
    EXPECT_EQ(inferStepCounter.load(), static_cast<int>(deviceIds_.size()));

    // 验证输出数据
    bool hasNonZeroOutput = false;
    for (float val : outputData) {
        if (val != 0.0f) {
            hasNonZeroOutput = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZeroOutput);

    context.finalize();
}

// 测试工具函数
TEST_F(MultiGPUFrameworkTest, UtilityFunctions) {
    // 测试获取最优设备集合
    auto optimalDevices = multi_gpu_utils::getOptimalDeviceSet(2);
    EXPECT_LE(optimalDevices.size(), 2);

    // 测试通信开销估算
    float cost = multi_gpu_utils::estimateCommunicationCost(
        CommType::ALL_REDUCE, 1024 * 1024, 2);
    EXPECT_GE(cost, 0.0f);

    // 测试负载均衡效率计算
    std::map<int, float> loads = {{0, 0.5f}, {1, 0.6f}, {2, 0.4f}};
    float efficiency = multi_gpu_utils::calculateLoadBalanceEfficiency(loads);
    EXPECT_GE(efficiency, 0.0f);
    EXPECT_LE(efficiency, 1.0f);
}

// 性能基准测试
TEST_F(MultiGPUFrameworkTest, PerformanceBenchmark) {
    if (deviceIds_.size() < 2) {
        GTEST_SKIP() << "需要至少2个GPU设备进行性能测试";
    }

    NCCLCommunicator communicator;
    ASSERT_TRUE(communicator.initialize(deviceIds_));

    // 测试不同数据大小的通信性能
    std::vector<size_t> dataSizes = {1024, 4096, 16384, 65536};

    for (size_t dataSize : dataSizes) {
        std::vector<float*> deviceBuffers(deviceIds_.size());

        // 分配内存
        for (size_t i = 0; i < deviceIds_.size(); ++i) {
            cudaSetDevice(deviceIds_[i]);
            cudaMalloc(&deviceBuffers[i], dataSize * sizeof(float));

            // 初始化数据
            std::vector<float> hostData(dataSize, 1.0f);
            cudaMemcpy(deviceBuffers[i], hostData.data(),
                      dataSize * sizeof(float), cudaMemcpyHostToDevice);
        }

        // 测量AllReduce性能
        auto startTime = std::chrono::high_resolution_clock::now();

        bool success = communicator.allReduce(
            deviceBuffers[0], deviceBuffers[0], dataSize,
            ncclFloat, ncclSum, deviceIds_
        );

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - startTime);

        EXPECT_TRUE(success);

        // 计算带宽
        float dataGB = (dataSize * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float bandwidth = dataGB / (duration.count() / 1000000.0f);

        std::cout << "数据大小: " << dataSize << " floats, "
                 << "时间: " << duration.count() << " μs, "
                 << "带宽: " << bandwidth << " GB/s" << std::endl;

        // 清理内存
        for (size_t i = 0; i < deviceIds_.size(); ++i) {
            cudaSetDevice(deviceIds_[i]);
            cudaFree(deviceBuffers[i]);
        }
    }

    communicator.finalize();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // 检查CUDA环境
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "警告: 没有找到CUDA设备，跳过所有测试" << std::endl;
        return 0;
    }

    std::cout << "检测到 " << deviceCount << " 个CUDA设备" << std::endl;

    return RUN_ALL_TESTS();
}
