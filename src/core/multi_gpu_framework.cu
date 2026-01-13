#include "multi_gpu_framework.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <cmath>

namespace cuda_learning {

// ============================================================================
// MultiGPUTask Implementation
// ============================================================================

void MultiGPUTask::execute(int deviceId, const DataShard& shard) {
    if (taskFunc_) {
        taskFunc_(deviceId, shard);
        completed_ = true;
    }
}

// ============================================================================
// NCCLCommunicator Implementation
// ============================================================================

NCCLCommunicator::NCCLCommunicator() : initialized_(false) {}

NCCLCommunicator::~NCCLCommunicator() {
    finalize();
}

bool NCCLCommunicator::initialize(const std::vector<int>& deviceIds) {
    if (initialized_) {
        return true;
    }

    try {
        // 生成NCCL唯一ID
        ncclResult_t result = ncclGetUniqueId(&ncclId_);
        if (!multi_gpu_utils::checkNCCLError(result, "ncclGetUniqueId")) {
            return false;
        }

        // 为每个设备创建通信器和流
        std::vector<ncclComm_t> comms(deviceIds.size());

        // 初始化NCCL通信器
        result = ncclCommInitAll(comms.data(), deviceIds.size(), deviceIds.data());
        if (!multi_gpu_utils::checkNCCLError(result, "ncclCommInitAll")) {
            return false;
        }

        // 为每个设备创建CUDA流并存储通信器
        for (size_t i = 0; i < deviceIds.size(); ++i) {
            int deviceId = deviceIds[i];

            // 设置设备
            cudaSetDevice(deviceId);

            // 创建CUDA流
            cudaStream_t stream;
            cudaError_t cudaResult = cudaStreamCreate(&stream);
            if (cudaResult != cudaSuccess) {
                std::cerr << "Failed to create CUDA stream for device " << deviceId << std::endl;
                cleanup();
                return false;
            }

            comms_[deviceId] = comms[i];
            streams_[deviceId] = stream;
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in NCCLCommunicator::initialize: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void NCCLCommunicator::finalize() {
    cleanup();
    initialized_ = false;
}

bool NCCLCommunicator::allReduce(void* sendBuff, void* recvBuff, size_t count,
                                 ncclDataType_t dataType, ncclRedOp_t op,
                                 const std::vector<int>& deviceIds) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    // 启动所有设备的allReduce操作
    for (int deviceId : deviceIds) {
        cudaSetDevice(deviceId);

        auto commIt = comms_.find(deviceId);
        auto streamIt = streams_.find(deviceId);

        if (commIt == comms_.end() || streamIt == streams_.end()) {
            std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
            return false;
        }

        ncclResult_t result = ncclAllReduce(sendBuff, recvBuff, count, dataType, op,
                                           commIt->second, streamIt->second);
        if (!multi_gpu_utils::checkNCCLError(result, "ncclAllReduce")) {
            return false;
        }
    }

    return synchronize(deviceIds);
}

bool NCCLCommunicator::allGather(void* sendBuff, void* recvBuff, size_t sendCount,
                                 ncclDataType_t dataType, const std::vector<int>& deviceIds) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    for (int deviceId : deviceIds) {
        cudaSetDevice(deviceId);

        auto commIt = comms_.find(deviceId);
        auto streamIt = streams_.find(deviceId);

        if (commIt == comms_.end() || streamIt == streams_.end()) {
            std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
            return false;
        }

        ncclResult_t result = ncclAllGather(sendBuff, recvBuff, sendCount, dataType,
                                           commIt->second, streamIt->second);
        if (!multi_gpu_utils::checkNCCLError(result, "ncclAllGather")) {
            return false;
        }
    }

    return synchronize(deviceIds);
}

bool NCCLCommunicator::reduceScatter(void* sendBuff, void* recvBuff, size_t recvCount,
                                     ncclDataType_t dataType, ncclRedOp_t op,
                                     const std::vector<int>& deviceIds) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    for (int deviceId : deviceIds) {
        cudaSetDevice(deviceId);

        auto commIt = comms_.find(deviceId);
        auto streamIt = streams_.find(deviceId);

        if (commIt == comms_.end() || streamIt == streams_.end()) {
            std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
            return false;
        }

        ncclResult_t result = ncclReduceScatter(sendBuff, recvBuff, recvCount, dataType, op,
                                               commIt->second, streamIt->second);
        if (!multi_gpu_utils::checkNCCLError(result, "ncclReduceScatter")) {
            return false;
        }
    }

    return synchronize(deviceIds);
}

bool NCCLCommunicator::broadcast(void* buff, size_t count, ncclDataType_t dataType,
                                int root, const std::vector<int>& deviceIds) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    for (int deviceId : deviceIds) {
        cudaSetDevice(deviceId);

        auto commIt = comms_.find(deviceId);
        auto streamIt = streams_.find(deviceId);

        if (commIt == comms_.end() || streamIt == streams_.end()) {
            std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
            return false;
        }

        ncclResult_t result = ncclBcast(buff, count, dataType, root,
                                       commIt->second, streamIt->second);
        if (!multi_gpu_utils::checkNCCLError(result, "ncclBcast")) {
            return false;
        }
    }

    return synchronize(deviceIds);
}

bool NCCLCommunicator::reduce(void* sendBuff, void* recvBuff, size_t count,
                             ncclDataType_t dataType, ncclRedOp_t op, int root,
                             const std::vector<int>& deviceIds) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    for (int deviceId : deviceIds) {
        cudaSetDevice(deviceId);

        auto commIt = comms_.find(deviceId);
        auto streamIt = streams_.find(deviceId);

        if (commIt == comms_.end() || streamIt == streams_.end()) {
            std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
            return false;
        }

        ncclResult_t result = ncclReduce(sendBuff, recvBuff, count, dataType, op, root,
                                        commIt->second, streamIt->second);
        if (!multi_gpu_utils::checkNCCLError(result, "ncclReduce")) {
            return false;
        }
    }

    return synchronize(deviceIds);
}

bool NCCLCommunicator::send(void* sendBuff, size_t count, ncclDataType_t dataType,
                           int peer, int deviceId) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    cudaSetDevice(deviceId);

    auto commIt = comms_.find(deviceId);
    auto streamIt = streams_.find(deviceId);

    if (commIt == comms_.end() || streamIt == streams_.end()) {
        std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
        return false;
    }

    ncclResult_t result = ncclSend(sendBuff, count, dataType, peer,
                                  commIt->second, streamIt->second);
    return multi_gpu_utils::checkNCCLError(result, "ncclSend");
}

bool NCCLCommunicator::recv(void* recvBuff, size_t count, ncclDataType_t dataType,
                           int peer, int deviceId) {
    if (!initialized_) {
        std::cerr << "NCCL communicator not initialized" << std::endl;
        return false;
    }

    cudaSetDevice(deviceId);

    auto commIt = comms_.find(deviceId);
    auto streamIt = streams_.find(deviceId);

    if (commIt == comms_.end() || streamIt == streams_.end()) {
        std::cerr << "Device " << deviceId << " not found in communicator" << std::endl;
        return false;
    }

    ncclResult_t result = ncclRecv(recvBuff, count, dataType, peer,
                                  commIt->second, streamIt->second);
    return multi_gpu_utils::checkNCCLError(result, "ncclRecv");
}

bool NCCLCommunicator::synchronize(const std::vector<int>& deviceIds) {
    for (int deviceId : deviceIds) {
        auto streamIt = streams_.find(deviceId);
        if (streamIt != streams_.end()) {
            cudaSetDevice(deviceId);
            cudaError_t result = cudaStreamSynchronize(streamIt->second);
            if (result != cudaSuccess) {
                std::cerr << "Failed to synchronize stream for device " << deviceId
                         << ": " << cudaGetErrorString(result) << std::endl;
                return false;
            }
        }
    }
    return true;
}

ncclComm_t NCCLCommunicator::getComm(int deviceId) const {
    auto it = comms_.find(deviceId);
    return (it != comms_.end()) ? it->second : nullptr;
}

void NCCLCommunicator::cleanup() {
    // 销毁CUDA流
    for (auto& pair : streams_) {
        cudaSetDevice(pair.first);
        cudaStreamDestroy(pair.second);
    }
    streams_.clear();

    // 销毁NCCL通信器
    for (auto& pair : comms_) {
        ncclCommDestroy(pair.second);
    }
    comms_.clear();
}

// ============================================================================
// DataShardManager Implementation
// ============================================================================

DataShardManager::DataShardManager() : nextShardId_(0) {}

DataShardManager::~DataShardManager() {
    // 清理所有分片
    std::vector<DataShard> allShards;
    for (const auto& pair : shards_) {
        allShards.push_back(pair.second);
    }
    cleanupShards(allShards);
}

std::vector<DataShard> DataShardManager::createShards(void* data, size_t totalSize,
                                                     const std::vector<int>& deviceIds,
                                                     ShardingStrategy strategy) {
    std::vector<DataShard> shards;
    int numDevices = deviceIds.size();

    for (int i = 0; i < numDevices; ++i) {
        DataShard shard;
        shard.shardId = nextShardId_++;
        shard.deviceId = deviceIds[i];
        shard.strategy = strategy;
        shard.size = calculateShardSize(totalSize, numDevices, i, strategy);
        shard.startOffset = calculateShardOffset(totalSize, numDevices, i, strategy);

        // 在目标设备上分配内存
        cudaSetDevice(shard.deviceId);
        cudaError_t result = cudaMalloc(&shard.devicePtr, shard.size);
        if (result != cudaSuccess) {
            std::cerr << "Failed to allocate memory on device " << shard.deviceId
                     << ": " << cudaGetErrorString(result) << std::endl;
            // 清理已分配的分片
            cleanupShards(shards);
            return {};
        }

        shards.push_back(shard);
        shards_[shard.shardId] = shard;
    }

    return shards;
}

bool DataShardManager::distributeData(const std::vector<DataShard>& shards) {
    for (const auto& shard : shards) {
        cudaSetDevice(shard.deviceId);

        // 计算源数据指针
        char* srcPtr = static_cast<char*>(shards[0].devicePtr) + shard.startOffset;

        // 复制数据到设备
        cudaError_t result = cudaMemcpy(shard.devicePtr, srcPtr, shard.size, cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            std::cerr << "Failed to copy data to device " << shard.deviceId
                     << ": " << cudaGetErrorString(result) << std::endl;
            return false;
        }
    }

    return true;
}

bool DataShardManager::gatherData(const std::vector<DataShard>& shards, void* outputBuffer) {
    for (const auto& shard : shards) {
        cudaSetDevice(shard.deviceId);

        // 计算目标缓冲区指针
        char* dstPtr = static_cast<char*>(outputBuffer) + shard.startOffset;

        // 从设备复制数据
        cudaError_t result = cudaMemcpy(dstPtr, shard.devicePtr, shard.size, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            std::cerr << "Failed to copy data from device " << shard.deviceId
                     << ": " << cudaGetErrorString(result) << std::endl;
            return false;
        }
    }

    return true;
}

std::vector<DataShard> DataShardManager::reshardData(const std::vector<DataShard>& currentShards,
                                                    const std::vector<int>& newDeviceIds,
                                                    ShardingStrategy newStrategy) {
    // 首先收集当前数据
    size_t totalSize = 0;
    for (const auto& shard : currentShards) {
        totalSize += shard.size;
    }

    // 分配临时主机内存
    void* tempBuffer = malloc(totalSize);
    if (!tempBuffer) {
        std::cerr << "Failed to allocate temporary buffer for resharding" << std::endl;
        return {};
    }

    // 收集数据到主机
    if (!gatherData(currentShards, tempBuffer)) {
        free(tempBuffer);
        return {};
    }

    // 创建新的分片
    std::vector<DataShard> newShards = createShards(tempBuffer, totalSize, newDeviceIds, newStrategy);

    // 分发数据到新分片
    if (!newShards.empty()) {
        distributeData(newShards);
    }

    free(tempBuffer);
    return newShards;
}

DataShard DataShardManager::getShardInfo(int shardId) const {
    auto it = shards_.find(shardId);
    return (it != shards_.end()) ? it->second : DataShard();
}

std::vector<DataShard> DataShardManager::getAllShards() const {
    std::vector<DataShard> result;
    for (const auto& pair : shards_) {
        result.push_back(pair.second);
    }
    return result;
}

void DataShardManager::cleanupShards(const std::vector<DataShard>& shards) {
    for (const auto& shard : shards) {
        if (shard.devicePtr) {
            cudaSetDevice(shard.deviceId);
            cudaFree(shard.devicePtr);
        }
        shards_.erase(shard.shardId);
    }
}

size_t DataShardManager::calculateShardSize(size_t totalSize, int numDevices,
                                           int deviceIndex, ShardingStrategy strategy) {
    switch (strategy) {
        case ShardingStrategy::BATCH_PARALLEL:
        case ShardingStrategy::DATA_PARALLEL:
            // 均匀分割
            return (totalSize + numDevices - 1) / numDevices;

        case ShardingStrategy::MODEL_PARALLEL:
            // 根据模型层分割（这里简化为均匀分割）
            return (totalSize + numDevices - 1) / numDevices;

        case ShardingStrategy::PIPELINE_PARALLEL:
            // 流水线分割（每个设备处理不同阶段）
            return totalSize; // 每个设备都需要完整数据

        case ShardingStrategy::HYBRID_PARALLEL:
            // 混合策略（这里简化为均匀分割）
            return (totalSize + numDevices - 1) / numDevices;

        default:
            return (totalSize + numDevices - 1) / numDevices;
    }
}

size_t DataShardManager::calculateShardOffset(size_t totalSize, int numDevices,
                                             int deviceIndex, ShardingStrategy strategy) {
    switch (strategy) {
        case ShardingStrategy::BATCH_PARALLEL:
        case ShardingStrategy::DATA_PARALLEL:
            // 连续分割
            return (totalSize / numDevices) * deviceIndex;

        case ShardingStrategy::MODEL_PARALLEL:
            // 模型层分割
            return (totalSize / numDevices) * deviceIndex;

        case ShardingStrategy::PIPELINE_PARALLEL:
            // 流水线不需要偏移
            return 0;

        case ShardingStrategy::HYBRID_PARALLEL:
            // 混合策略
            return (totalSize / numDevices) * deviceIndex;

        default:
            return (totalSize / numDevices) * deviceIndex;
    }
}

// ============================================================================
// LoadBalancer Implementation
// ============================================================================

LoadBalancer::LoadBalancer(LoadBalanceStrategy strategy)
    : strategy_(strategy), roundRobinIndex_(0) {}

void LoadBalancer::setAvailableDevices(const std::vector<GPUDevice>& devices) {
    devices_ = devices;

    // 初始化负载统计
    for (const auto& device : devices) {
        deviceLoads_[device.deviceId] = 0.0f;
        deviceMemoryUsage_[device.deviceId] = 0;
    }
}

int LoadBalancer::selectDevice(const MultiGPUTask& task) {
    if (devices_.empty()) {
        return -1;
    }

    switch (strategy_) {
        case LoadBalanceStrategy::ROUND_ROBIN:
            return selectByRoundRobin();

        case LoadBalanceStrategy::MEMORY_AWARE:
            return selectByMemoryAware();

        case LoadBalanceStrategy::COMPUTE_AWARE:
            return selectByComputeAware();

        case LoadBalanceStrategy::DYNAMIC_ADAPTIVE:
            return selectByDynamicAdaptive();

        default:
            return selectByRoundRobin();
    }
}

std::vector<int> LoadBalancer::selectDevices(int numDevices) {
    std::vector<int> selectedDevices;

    for (int i = 0; i < numDevices && i < static_cast<int>(devices_.size()); ++i) {
        // 这里简化为选择前N个设备
        selectedDevices.push_back(devices_[i].deviceId);
    }

    return selectedDevices;
}

void LoadBalancer::updateDeviceLoad(int deviceId, float loadFactor) {
    deviceLoads_[deviceId] = loadFactor;
}

void LoadBalancer::updateDeviceMemory(int deviceId, size_t freeMemory) {
    deviceMemoryUsage_[deviceId] = freeMemory;
}

std::map<int, float> LoadBalancer::getDeviceLoads() const {
    return deviceLoads_;
}

float LoadBalancer::getAverageLoad() const {
    if (deviceLoads_.empty()) {
        return 0.0f;
    }

    float totalLoad = 0.0f;
    for (const auto& pair : deviceLoads_) {
        totalLoad += pair.second;
    }

    return totalLoad / deviceLoads_.size();
}

std::vector<std::pair<int, int>> LoadBalancer::rebalanceLoad() {
    std::vector<std::pair<int, int>> migrations;

    // 简化的重平衡策略：将高负载设备的任务迁移到低负载设备
    float avgLoad = getAverageLoad();

    std::vector<int> highLoadDevices, lowLoadDevices;

    for (const auto& pair : deviceLoads_) {
        if (pair.second > avgLoad * 1.2f) {
            highLoadDevices.push_back(pair.first);
        } else if (pair.second < avgLoad * 0.8f) {
            lowLoadDevices.push_back(pair.first);
        }
    }

    // 创建迁移对
    size_t minSize = std::min(highLoadDevices.size(), lowLoadDevices.size());
    for (size_t i = 0; i < minSize; ++i) {
        migrations.emplace_back(highLoadDevices[i], lowLoadDevices[i]);
    }

    return migrations;
}

int LoadBalancer::selectByRoundRobin() {
    if (devices_.empty()) {
        return -1;
    }

    int deviceId = devices_[roundRobinIndex_].deviceId;
    roundRobinIndex_ = (roundRobinIndex_ + 1) % devices_.size();
    return deviceId;
}

int LoadBalancer::selectByMemoryAware() {
    if (devices_.empty()) {
        return -1;
    }

    // 选择可用内存最多的设备
    int bestDevice = devices_[0].deviceId;
    size_t maxFreeMemory = deviceMemoryUsage_[bestDevice];

    for (const auto& device : devices_) {
        size_t freeMemory = deviceMemoryUsage_[device.deviceId];
        if (freeMemory > maxFreeMemory) {
            maxFreeMemory = freeMemory;
            bestDevice = device.deviceId;
        }
    }

    return bestDevice;
}

int LoadBalancer::selectByComputeAware() {
    if (devices_.empty()) {
        return -1;
    }

    // 选择负载最低的设备
    int bestDevice = devices_[0].deviceId;
    float minLoad = deviceLoads_[bestDevice];

    for (const auto& device : devices_) {
        float load = deviceLoads_[device.deviceId];
        if (load < minLoad) {
            minLoad = load;
            bestDevice = device.deviceId;
        }
    }

    return bestDevice;
}

int LoadBalancer::selectByDynamicAdaptive() {
    if (devices_.empty()) {
        return -1;
    }

    // 综合考虑负载和内存使用情况
    int bestDevice = devices_[0].deviceId;
    float bestScore = 0.0f;

    for (const auto& device : devices_) {
        int deviceId = device.deviceId;
        float load = deviceLoads_[deviceId];
        size_t freeMemory = deviceMemoryUsage_[deviceId];

        // 计算综合得分（负载越低、可用内存越多得分越高）
        float memoryScore = static_cast<float>(freeMemory) / device.totalMemory;
        float loadScore = 1.0f - load;
        float totalScore = 0.6f * loadScore + 0.4f * memoryScore;

        if (totalScore > bestScore) {
            bestScore = totalScore;
            bestDevice = deviceId;
        }
    }

    return bestDevice;
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

namespace multi_gpu_utils {

bool checkNCCLError(ncclResult_t result, const std::string& operation) {
    if (result != ncclSuccess) {
        std::cerr << "NCCL error in " << operation << ": " << ncclGetErrorString(result) << std::endl;
        return false;
    }
    return true;
}

std::vector<int> getOptimalDeviceSet(int numDevices) {
    std::vector<int> devices;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    // 选择前N个可用设备
    for (int i = 0; i < std::min(numDevices, deviceCount); ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // 检查设备是否支持CUDA
        if (prop.major >= 3) { // 至少需要计算能力3.0
            devices.push_back(i);
        }
    }

    return devices;
}

float estimateCommunicationCost(CommType type, size_t dataSize, int numDevices) {
    // 简化的通信开销估算（实际应该基于硬件特性）
    float baseCost = static_cast<float>(dataSize) / (1024.0f * 1024.0f * 1024.0f); // GB

    switch (type) {
        case CommType::ALL_REDUCE:
            return baseCost * 2.0f * (numDevices - 1) / numDevices;

        case CommType::ALL_GATHER:
            return baseCost * (numDevices - 1);

        case CommType::REDUCE_SCATTER:
            return baseCost * (numDevices - 1) / numDevices;

        case CommType::BROADCAST:
            return baseCost;

        case CommType::REDUCE:
            return baseCost * (numDevices - 1) / numDevices;

        case CommType::POINT_TO_POINT:
            return baseCost;

        default:
            return baseCost;
    }
}

float calculateLoadBalanceEfficiency(const std::map<int, float>& deviceLoads) {
    if (deviceLoads.empty()) {
        return 0.0f;
    }

    // 计算负载的标准差
    float mean = 0.0f;
    for (const auto& pair : deviceLoads) {
        mean += pair.second;
    }
    mean /= deviceLoads.size();

    float variance = 0.0f;
    for (const auto& pair : deviceLoads) {
        float diff = pair.second - mean;
        variance += diff * diff;
    }
    variance /= deviceLoads.size();

    float stddev = std::sqrt(variance);

    // 效率 = 1 - (标准差 / 平均值)，值越接近1表示负载越均衡
    return (mean > 0.0f) ? std::max(0.0f, 1.0f - stddev / mean) : 0.0f;
}

std::string generateMultiGPUReport(const MultiGPUContext& context) {
    std::stringstream ss;

    ss << "=== Multi-GPU Performance Report ===\n";

    const auto& devices = context.getDevices();
    ss << "Active Devices: " << devices.size() << "\n";

    for (const auto& device : devices) {
        ss << "Device " << device.deviceId << ": " << device.name << "\n";
        ss << "  Memory: " << device.freeMemory / (1024*1024) << " MB free / "
           << device.totalMemory / (1024*1024) << " MB total\n";
        ss << "  Compute Capability: " << device.computeCapability << "\n";
        ss << "  Active: " << (device.isActive ? "Yes" : "No") << "\n";
    }

    // 添加负载均衡信息
    if (context.getLoadBalancer()) {
        auto loads = context.getLoadBalancer()->getDeviceLoads();
        float efficiency = calculateLoadBalanceEfficiency(loads);

        ss << "\nLoad Balance Efficiency: " << (efficiency * 100.0f) << "%\n";
        ss << "Device Loads:\n";
        for (const auto& pair : loads) {
            ss << "  Device " << pair.first << ": " << (pair.second * 100.0f) << "%\n";
        }
    }

    ss << "=====================================\n";

    return ss.str();
}

} // namespace multi_gpu_utils

} // namespace cuda_learning
// ============================================================================
// MultiGPUExecutor Implementation
// ============================================================================

MultiGPUExecutor::MultiGPUExecutor() : shutdown_(false) {}

MultiGPUExecutor::~MultiGPUExecutor() {
    finalize();
}

bool MultiGPUExecutor::initialize(const std::vector<int>& deviceIds) {
    devices_.clear();

    // 初始化设备信息
    for (int deviceId : deviceIds) {
        GPUDevice device;
        device.deviceId = deviceId;

        cudaSetDevice(deviceId);

        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        device.name = prop.name;
        device.totalMemory = prop.totalGlobalMem;
        device.computeCapability = prop.major * 10 + prop.minor;

        // 获取可用内存
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        device.freeMemory = free;

        // 创建CUDA流
        cudaStreamCreate(&device.stream);
        device.isActive = true;

        devices_.push_back(device);
    }

    // 初始化通信器和负载均衡器
    communicator_ = std::make_unique<NCCLCommunicator>();
    if (!communicator_->initialize(deviceIds)) {
        return false;
    }

    loadBalancer_ = std::make_unique<LoadBalancer>(LoadBalanceStrategy::DYNAMIC_ADAPTIVE);
    loadBalancer_->setAvailableDevices(devices_);

    // 启动工作线程
    shutdown_ = false;
    for (size_t i = 0; i < devices_.size(); ++i) {
        workerThreads_.emplace_back(&MultiGPUExecutor::workerThreadFunction, this, devices_[i].deviceId);
    }

    return true;
}

void MultiGPUExecutor::finalize() {
    // 停止工作线程
    shutdown_ = true;
    taskCondition_.notify_all();

    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    workerThreads_.clear();

    // 清理设备资源
    for (auto& device : devices_) {
        if (device.stream) {
            cudaSetDevice(device.deviceId);
            cudaStreamDestroy(device.stream);
        }
    }
    devices_.clear();

    // 清理通信器
    if (communicator_) {
        communicator_->finalize();
        communicator_.reset();
    }

    loadBalancer_.reset();
}

bool MultiGPUExecutor::executeTask(const MultiGPUTask& task, const std::vector<DataShard>& shards) {
    if (shards.size() != devices_.size()) {
        std::cerr << "Number of shards must match number of devices" << std::endl;
        return false;
    }

    // 在每个设备上执行任务
    std::vector<std::thread> taskThreads;
    std::atomic<bool> success(true);

    for (size_t i = 0; i < devices_.size(); ++i) {
        taskThreads.emplace_back([&, i]() {
            try {
                executeTaskOnDevice(task, shards[i], devices_[i].deviceId);
            } catch (const std::exception& e) {
                std::cerr << "Task execution failed on device " << devices_[i].deviceId
                         << ": " << e.what() << std::endl;
                success = false;
            }
        });
    }

    // 等待所有任务完成
    for (auto& thread : taskThreads) {
        thread.join();
    }

    return success;
}

bool MultiGPUExecutor::executeTasks(const std::vector<MultiGPUTask>& tasks,
                                   const std::vector<DataShard>& shards) {
    for (const auto& task : tasks) {
        if (!executeTask(task, shards)) {
            return false;
        }
    }
    return true;
}

bool MultiGPUExecutor::executeParallel(const MultiGPUTask& task,
                                      const std::vector<DataShard>& shards) {
    return executeTask(task, shards); // 已经是并行执行
}

bool MultiGPUExecutor::executePipeline(const std::vector<MultiGPUTask>& tasks,
                                      const std::vector<DataShard>& shards) {
    // 流水线执行：每个设备执行不同的任务阶段
    if (tasks.size() != devices_.size()) {
        std::cerr << "Number of tasks must match number of devices for pipeline execution" << std::endl;
        return false;
    }

    std::vector<std::thread> pipelineThreads;
    std::atomic<bool> success(true);

    for (size_t i = 0; i < devices_.size(); ++i) {
        pipelineThreads.emplace_back([&, i]() {
            try {
                // 每个设备执行对应的任务阶段
                DataShard shard = (i < shards.size()) ? shards[i] : DataShard();
                executeTaskOnDevice(tasks[i], shard, devices_[i].deviceId);
            } catch (const std::exception& e) {
                std::cerr << "Pipeline task execution failed on device " << devices_[i].deviceId
                         << ": " << e.what() << std::endl;
                success = false;
            }
        });
    }

    // 等待所有阶段完成
    for (auto& thread : pipelineThreads) {
        thread.join();
    }

    return success;
}

void MultiGPUExecutor::synchronize() {
    for (const auto& device : devices_) {
        cudaSetDevice(device.deviceId);
        cudaStreamSynchronize(device.stream);
    }
}

void MultiGPUExecutor::synchronizeDevice(int deviceId) {
    for (const auto& device : devices_) {
        if (device.deviceId == deviceId) {
            cudaSetDevice(deviceId);
            cudaStreamSynchronize(device.stream);
            break;
        }
    }
}

std::map<int, float> MultiGPUExecutor::getExecutionTimes() const {
    // 这里返回模拟的执行时间，实际实现需要记录真实时间
    std::map<int, float> times;
    for (const auto& device : devices_) {
        times[device.deviceId] = 0.0f; // 需要实际测量
    }
    return times;
}

float MultiGPUExecutor::getTotalExecutionTime() const {
    // 返回模拟的总执行时间
    return 0.0f; // 需要实际测量
}

void MultiGPUExecutor::workerThreadFunction(int deviceId) {
    cudaSetDevice(deviceId);

    while (!shutdown_) {
        std::unique_lock<std::mutex> lock(taskMutex_);
        taskCondition_.wait(lock, [this] { return shutdown_; });

        if (shutdown_) {
            break;
        }

        // 工作线程逻辑（这里简化）
    }
}

void MultiGPUExecutor::executeTaskOnDevice(const MultiGPUTask& task,
                                          const DataShard& shard, int deviceId) {
    cudaSetDevice(deviceId);

    // 执行任务
    MultiGPUTask taskCopy = task;
    taskCopy.execute(deviceId, shard);

    // 同步设备
    cudaDeviceSynchronize();
}

// ============================================================================
// MultiGPUContext Implementation
// ============================================================================

MultiGPUContext::MultiGPUContext() : initialized_(false) {}

MultiGPUContext::~MultiGPUContext() {
    finalize();
}

bool MultiGPUContext::initialize(const std::vector<int>& deviceIds,
                                LoadBalanceStrategy loadStrategy) {
    if (initialized_) {
        return true;
    }

    try {
        // 初始化设备
        if (!initializeDevices(deviceIds)) {
            setError("Failed to initialize devices");
            return false;
        }

        // 初始化通信器
        communicator_ = std::make_unique<NCCLCommunicator>();
        if (!communicator_->initialize(deviceIds)) {
            setError("Failed to initialize NCCL communicator");
            return false;
        }

        // 初始化数据分片管理器
        shardManager_ = std::make_unique<DataShardManager>();

        // 初始化负载均衡器
        loadBalancer_ = std::make_unique<LoadBalancer>(loadStrategy);
        loadBalancer_->setAvailableDevices(devices_);

        // 初始化执行器
        executor_ = std::make_unique<MultiGPUExecutor>();
        if (!executor_->initialize(deviceIds)) {
            setError("Failed to initialize multi-GPU executor");
            return false;
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        setError("Exception during initialization: " + std::string(e.what()));
        return false;
    }
}

void MultiGPUContext::finalize() {
    if (!initialized_) {
        return;
    }

    // 清理组件
    if (executor_) {
        executor_->finalize();
        executor_.reset();
    }

    if (communicator_) {
        communicator_->finalize();
        communicator_.reset();
    }

    shardManager_.reset();
    loadBalancer_.reset();

    // 清理设备
    cleanupDevices();

    initialized_ = false;
}

GPUDevice* MultiGPUContext::getDevice(int deviceId) {
    for (auto& device : devices_) {
        if (device.deviceId == deviceId) {
            return &device;
        }
    }
    return nullptr;
}

bool MultiGPUContext::isDeviceActive(int deviceId) const {
    for (const auto& device : devices_) {
        if (device.deviceId == deviceId) {
            return device.isActive;
        }
    }
    return false;
}

void* MultiGPUContext::allocateOnDevice(int deviceId, size_t size) {
    cudaSetDevice(deviceId);

    void* ptr = nullptr;
    cudaError_t result = cudaMalloc(&ptr, size);

    if (result != cudaSuccess) {
        setError("Failed to allocate memory on device " + std::to_string(deviceId) +
                ": " + cudaGetErrorString(result));
        return nullptr;
    }

    return ptr;
}

void MultiGPUContext::freeOnDevice(int deviceId, void* ptr) {
    if (ptr) {
        cudaSetDevice(deviceId);
        cudaFree(ptr);
    }
}

bool MultiGPUContext::copyBetweenDevices(int srcDevice, int dstDevice,
                                        void* src, void* dst, size_t size) {
    cudaError_t result = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, size);

    if (result != cudaSuccess) {
        setError("Failed to copy between devices: " + std::string(cudaGetErrorString(result)));
        return false;
    }

    return true;
}

cudaStream_t MultiGPUContext::getStream(int deviceId) {
    GPUDevice* device = getDevice(deviceId);
    return device ? device->stream : nullptr;
}

void MultiGPUContext::synchronizeStream(int deviceId) {
    cudaStream_t stream = getStream(deviceId);
    if (stream) {
        cudaSetDevice(deviceId);
        cudaStreamSynchronize(stream);
    }
}

void MultiGPUContext::synchronizeAllStreams() {
    for (const auto& device : devices_) {
        cudaSetDevice(device.deviceId);
        cudaStreamSynchronize(device.stream);
    }
}

bool MultiGPUContext::initializeDevices(const std::vector<int>& deviceIds) {
    devices_.clear();

    for (int deviceId : deviceIds) {
        GPUDevice device;
        device.deviceId = deviceId;

        // 设置设备
        cudaError_t result = cudaSetDevice(deviceId);
        if (result != cudaSuccess) {
            std::cerr << "Failed to set device " << deviceId << ": "
                     << cudaGetErrorString(result) << std::endl;
            return false;
        }

        // 获取设备属性
        cudaDeviceProp prop;
        result = cudaGetDeviceProperties(&prop, deviceId);
        if (result != cudaSuccess) {
            std::cerr << "Failed to get device properties for device " << deviceId
                     << ": " << cudaGetErrorString(result) << std::endl;
            return false;
        }

        device.name = prop.name;
        device.totalMemory = prop.totalGlobalMem;
        device.computeCapability = prop.major * 10 + prop.minor;

        // 获取可用内存
        size_t free, total;
        result = cudaMemGetInfo(&free, &total);
        if (result != cudaSuccess) {
            std::cerr << "Failed to get memory info for device " << deviceId
                     << ": " << cudaGetErrorString(result) << std::endl;
            return false;
        }
        device.freeMemory = free;

        // 创建CUDA流
        result = cudaStreamCreate(&device.stream);
        if (result != cudaSuccess) {
            std::cerr << "Failed to create stream for device " << deviceId
                     << ": " << cudaGetErrorString(result) << std::endl;
            return false;
        }

        device.isActive = true;
        devices_.push_back(device);
    }

    return true;
}

void MultiGPUContext::cleanupDevices() {
    for (auto& device : devices_) {
        if (device.stream) {
            cudaSetDevice(device.deviceId);
            cudaStreamDestroy(device.stream);
        }
    }
    devices_.clear();
}

void MultiGPUContext::setError(const std::string& error) {
    lastError_ = error;
    std::cerr << "MultiGPUContext Error: " << error << std::endl;
}

// ============================================================================
// MultiGPUTrainer Implementation
// ============================================================================

MultiGPUTrainer::MultiGPUTrainer(MultiGPUContext* context) : context_(context) {}

bool MultiGPUTrainer::trainDataParallel(void* data, size_t dataSize, int batchSize,
                                       std::function<void(void*, size_t, int)> trainStep) {
    if (!context_ || !context_->getShardManager()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    std::vector<int> deviceIds;
    for (const auto& device : devices) {
        deviceIds.push_back(device.deviceId);
    }

    // 创建数据分片
    auto shards = context_->getShardManager()->createShards(
        data, dataSize, deviceIds, ShardingStrategy::DATA_PARALLEL);

    if (shards.empty()) {
        return false;
    }

    // 分发数据到各设备
    if (!context_->getShardManager()->distributeData(shards)) {
        return false;
    }

    // 创建训练任务
    MultiGPUTask trainingTask("data_parallel_training",
        [trainStep, batchSize](int deviceId, const DataShard& shard) {
            trainStep(shard.devicePtr, shard.size, batchSize);
        });

    // 执行训练
    bool success = context_->getExecutor()->executeTask(trainingTask, shards);

    // 清理分片
    context_->getShardManager()->cleanupShards(shards);

    return success;
}

bool MultiGPUTrainer::trainModelParallel(void* model, size_t modelSize,
                                        std::function<void(void*, int)> trainStep) {
    if (!context_ || !context_->getShardManager()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    std::vector<int> deviceIds;
    for (const auto& device : devices) {
        deviceIds.push_back(device.deviceId);
    }

    // 创建模型分片
    auto shards = context_->getShardManager()->createShards(
        model, modelSize, deviceIds, ShardingStrategy::MODEL_PARALLEL);

    if (shards.empty()) {
        return false;
    }

    // 分发模型到各设备
    if (!context_->getShardManager()->distributeData(shards)) {
        return false;
    }

    // 创建训练任务
    MultiGPUTask trainingTask("model_parallel_training",
        [trainStep](int deviceId, const DataShard& shard) {
            trainStep(shard.devicePtr, deviceId);
        });

    // 执行训练
    bool success = context_->getExecutor()->executeTask(trainingTask, shards);

    // 清理分片
    context_->getShardManager()->cleanupShards(shards);

    return success;
}

bool MultiGPUTrainer::trainPipelineParallel(const std::vector<void*>& modelStages,
                                           const std::vector<size_t>& stageSizes,
                                           std::function<void(void*, int, int)> trainStep) {
    if (!context_ || !context_->getExecutor()) {
        return false;
    }

    if (modelStages.size() != stageSizes.size()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    if (devices.size() < modelStages.size()) {
        return false; // 设备数量不足
    }

    // 为每个阶段创建任务
    std::vector<MultiGPUTask> pipelineTasks;
    std::vector<DataShard> stageShards;

    for (size_t i = 0; i < modelStages.size(); ++i) {
        // 创建阶段任务
        MultiGPUTask stageTask("pipeline_stage_" + std::to_string(i),
            [trainStep, i](int deviceId, const DataShard& shard) {
                trainStep(shard.devicePtr, deviceId, static_cast<int>(i));
            });
        pipelineTasks.push_back(stageTask);

        // 创建阶段数据分片
        DataShard shard;
        shard.shardId = static_cast<int>(i);
        shard.deviceId = devices[i].deviceId;
        shard.size = stageSizes[i];
        shard.strategy = ShardingStrategy::PIPELINE_PARALLEL;

        // 在设备上分配内存并复制数据
        cudaSetDevice(shard.deviceId);
        cudaMalloc(&shard.devicePtr, shard.size);
        cudaMemcpy(shard.devicePtr, modelStages[i], shard.size, cudaMemcpyHostToDevice);

        stageShards.push_back(shard);
    }

    // 执行流水线训练
    bool success = context_->getExecutor()->executePipeline(pipelineTasks, stageShards);

    // 清理阶段分片
    for (const auto& shard : stageShards) {
        cudaSetDevice(shard.deviceId);
        cudaFree(shard.devicePtr);
    }

    return success;
}

bool MultiGPUTrainer::synchronizeGradients(void* gradients, size_t gradSize) {
    if (!context_ || !context_->getCommunicator()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    std::vector<int> deviceIds;
    for (const auto& device : devices) {
        deviceIds.push_back(device.deviceId);
    }

    // 执行梯度全归约
    size_t count = gradSize / sizeof(float); // 假设是float类型
    return context_->getCommunicator()->allReduce(
        gradients, gradients, count, ncclFloat, ncclSum, deviceIds);
}

bool MultiGPUTrainer::updateParameters(void* parameters, size_t paramSize) {
    if (!context_ || !context_->getCommunicator()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    std::vector<int> deviceIds;
    for (const auto& device : devices) {
        deviceIds.push_back(device.deviceId);
    }

    // 广播更新后的参数到所有设备
    size_t count = paramSize / sizeof(float); // 假设是float类型
    return context_->getCommunicator()->broadcast(
        parameters, count, ncclFloat, 0, deviceIds);
}

// ============================================================================
// MultiGPUInference Implementation
// ============================================================================

MultiGPUInference::MultiGPUInference(MultiGPUContext* context) : context_(context) {}

bool MultiGPUInference::inferBatch(void* inputData, size_t inputSize, int batchSize,
                                  void* outputData, size_t outputSize,
                                  std::function<void(void*, void*, int)> inferenceStep) {
    if (!context_ || !context_->getShardManager()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    std::vector<int> deviceIds;
    for (const auto& device : devices) {
        deviceIds.push_back(device.deviceId);
    }

    // 创建输入数据分片
    auto inputShards = context_->getShardManager()->createShards(
        inputData, inputSize, deviceIds, ShardingStrategy::BATCH_PARALLEL);

    if (inputShards.empty()) {
        return false;
    }

    // 创建输出数据分片
    auto outputShards = context_->getShardManager()->createShards(
        outputData, outputSize, deviceIds, ShardingStrategy::BATCH_PARALLEL);

    if (outputShards.empty()) {
        context_->getShardManager()->cleanupShards(inputShards);
        return false;
    }

    // 分发输入数据
    if (!context_->getShardManager()->distributeData(inputShards)) {
        context_->getShardManager()->cleanupShards(inputShards);
        context_->getShardManager()->cleanupShards(outputShards);
        return false;
    }

    // 创建推理任务
    MultiGPUTask inferenceTask("batch_inference",
        [inferenceStep, &outputShards](int deviceId, const DataShard& inputShard) {
            // 找到对应的输出分片
            DataShard* outputShard = nullptr;
            for (auto& shard : outputShards) {
                if (shard.deviceId == deviceId) {
                    outputShard = &shard;
                    break;
                }
            }

            if (outputShard) {
                inferenceStep(inputShard.devicePtr, outputShard->devicePtr, deviceId);
            }
        });

    // 执行推理
    bool success = context_->getExecutor()->executeTask(inferenceTask, inputShards);

    // 收集输出数据
    if (success) {
        success = context_->getShardManager()->gatherData(outputShards, outputData);
    }

    // 清理分片
    context_->getShardManager()->cleanupShards(inputShards);
    context_->getShardManager()->cleanupShards(outputShards);

    return success;
}

bool MultiGPUInference::inferStream(void* inputData, size_t inputSize,
                                   void* outputData, size_t outputSize,
                                   std::function<void(void*, void*, int)> inferenceStep) {
    // 流式推理的简化实现，实际应该支持流水线处理
    return inferBatch(inputData, inputSize, 1, outputData, outputSize, inferenceStep);
}

bool MultiGPUInference::inferModelParallel(void* inputData, size_t inputSize,
                                          const std::vector<void*>& modelParts,
                                          void* outputData, size_t outputSize) {
    if (!context_ || modelParts.empty()) {
        return false;
    }

    // 获取可用设备
    const auto& devices = context_->getDevices();
    if (devices.size() < modelParts.size()) {
        return false; // 设备数量不足
    }

    // 为每个模型部分创建推理任务
    std::vector<MultiGPUTask> inferTasks;
    std::vector<DataShard> modelShards;

    for (size_t i = 0; i < modelParts.size(); ++i) {
        // 创建模型部分任务
        MultiGPUTask partTask("model_part_" + std::to_string(i),
            [inputData, outputData, i](int deviceId, const DataShard& modelShard) {
                // 简化的模型并行推理逻辑
                // 实际实现需要根据具体模型结构进行
            });
        inferTasks.push_back(partTask);

        // 创建模型分片（这里简化处理）
        DataShard shard;
        shard.shardId = static_cast<int>(i);
        shard.deviceId = devices[i].deviceId;
        shard.devicePtr = modelParts[i]; // 假设已经在设备上
        shard.strategy = ShardingStrategy::MODEL_PARALLEL;

        modelShards.push_back(shard);
    }

    // 执行模型并行推理
    return context_->getExecutor()->executeTasks(inferTasks, modelShards);
}

} // namespace cuda_learning
