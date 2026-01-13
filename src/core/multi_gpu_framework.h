#pragma once

#include <cuda_runtime.h>
#ifdef USE_NCCL
#include <nccl.h>
#else
// NCCL类型的简单定义，当NCCL不可用时使用
typedef void* ncclComm_t;
typedef struct { char internal[128]; } ncclUniqueId;
typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
#define ncclSuccess 0
#define ncclFloat 0
#define ncclSum 0
inline const char* ncclGetErrorString(ncclResult_t) { return "NCCL not available"; }
inline ncclResult_t ncclGetUniqueId(ncclUniqueId*) { return 0; }
inline ncclResult_t ncclCommInitAll(ncclComm_t*, int, int*) { return 0; }
inline ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclBcast(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclSend(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclRecv(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) { return 0; }
inline ncclResult_t ncclCommDestroy(ncclComm_t) { return 0; }
#endif
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace cuda_learning {

// 前向声明
class MultiGPUContext;
class LoadBalancer;
class DataShard;

// GPU设备信息
struct GPUDevice {
    int deviceId;
    std::string name;
    size_t totalMemory;
    size_t freeMemory;
    int computeCapability;
    cudaStream_t stream;
#ifdef USE_NCCL
    ncclComm_t ncclComm;
#endif
    bool isActive;

    GPUDevice() : deviceId(-1), totalMemory(0), freeMemory(0),
                  computeCapability(0), stream(nullptr),
#ifdef USE_NCCL
                  ncclComm(nullptr),
#endif
                  isActive(false) {}
};

// 通信操作类型
enum class CommType {
    ALL_REDUCE,     // 全归约
    ALL_GATHER,     // 全收集
    REDUCE_SCATTER, // 归约散射
    BROADCAST,      // 广播
    REDUCE,         // 归约
    POINT_TO_POINT  // 点对点
};

// 数据分片策略
enum class ShardingStrategy {
    BATCH_PARALLEL,    // 批次并行
    DATA_PARALLEL,     // 数据并行
    MODEL_PARALLEL,    // 模型并行
    PIPELINE_PARALLEL, // 流水线并行
    HYBRID_PARALLEL    // 混合并行
};

// 负载均衡策略
enum class LoadBalanceStrategy {
    ROUND_ROBIN,       // 轮询
    MEMORY_AWARE,      // 内存感知
    COMPUTE_AWARE,     // 计算感知
    DYNAMIC_ADAPTIVE   // 动态自适应
};

// 数据分片信息
struct DataShard {
    int shardId;
    int deviceId;
    size_t startOffset;
    size_t size;
    void* devicePtr;
    ShardingStrategy strategy;

    DataShard() : shardId(-1), deviceId(-1), startOffset(0),
                  size(0), devicePtr(nullptr),
                  strategy(ShardingStrategy::BATCH_PARALLEL) {}
};

// 通信操作描述
struct CommOperation {
    CommType type;
    std::vector<int> deviceIds;
    void* sendBuffer;
    void* recvBuffer;
    size_t count;
    ncclDataType_t dataType;
    ncclRedOp_t reduceOp;
    int root;  // 用于broadcast和reduce

    CommOperation() : type(CommType::ALL_REDUCE), sendBuffer(nullptr),
                      recvBuffer(nullptr), count(0),
                      dataType(ncclFloat), reduceOp(ncclSum), root(0) {}
};

// 多GPU任务
class MultiGPUTask {
public:
    using TaskFunction = std::function<void(int deviceId, const DataShard& shard)>;

    MultiGPUTask(const std::string& name, TaskFunction func)
        : name_(name), taskFunc_(func), completed_(false) {}

    void execute(int deviceId, const DataShard& shard);
    bool isCompleted() const { return completed_; }
    const std::string& getName() const { return name_; }

private:
    std::string name_;
    TaskFunction taskFunc_;
    std::atomic<bool> completed_;
};

// NCCL通信管理器
class NCCLCommunicator {
public:
    NCCLCommunicator();
    ~NCCLCommunicator();

    // 初始化NCCL通信
    bool initialize(const std::vector<int>& deviceIds);
    void finalize();

    // 通信操作
    bool allReduce(void* sendBuff, void* recvBuff, size_t count,
                   ncclDataType_t dataType, ncclRedOp_t op,
                   const std::vector<int>& deviceIds);

    bool allGather(void* sendBuff, void* recvBuff, size_t sendCount,
                   ncclDataType_t dataType, const std::vector<int>& deviceIds);

    bool reduceScatter(void* sendBuff, void* recvBuff, size_t recvCount,
                       ncclDataType_t dataType, ncclRedOp_t op,
                       const std::vector<int>& deviceIds);

    bool broadcast(void* buff, size_t count, ncclDataType_t dataType,
                   int root, const std::vector<int>& deviceIds);

    bool reduce(void* sendBuff, void* recvBuff, size_t count,
                ncclDataType_t dataType, ncclRedOp_t op, int root,
                const std::vector<int>& deviceIds);

    // 点对点通信
    bool send(void* sendBuff, size_t count, ncclDataType_t dataType,
              int peer, int deviceId);

    bool recv(void* recvBuff, size_t count, ncclDataType_t dataType,
              int peer, int deviceId);

    // 同步操作
    bool synchronize(const std::vector<int>& deviceIds);

    // 获取通信器
    ncclComm_t getComm(int deviceId) const;

private:
    std::map<int, ncclComm_t> comms_;
    std::map<int, cudaStream_t> streams_;
    ncclUniqueId ncclId_;
    bool initialized_;

    void cleanup();
};

// 数据分片管理器
class DataShardManager {
public:
    DataShardManager();
    ~DataShardManager();

    // 创建数据分片
    std::vector<DataShard> createShards(void* data, size_t totalSize,
                                       const std::vector<int>& deviceIds,
                                       ShardingStrategy strategy);

    // 分发数据到各GPU
    bool distributeData(const std::vector<DataShard>& shards);

    // 收集数据从各GPU
    bool gatherData(const std::vector<DataShard>& shards, void* outputBuffer);

    // 重新分片
    std::vector<DataShard> reshardData(const std::vector<DataShard>& currentShards,
                                      const std::vector<int>& newDeviceIds,
                                      ShardingStrategy newStrategy);

    // 获取分片信息
    DataShard getShardInfo(int shardId) const;
    std::vector<DataShard> getAllShards() const;

    // 清理分片
    void cleanupShards(const std::vector<DataShard>& shards);

private:
    std::map<int, DataShard> shards_;
    int nextShardId_;

    size_t calculateShardSize(size_t totalSize, int numDevices,
                             int deviceIndex, ShardingStrategy strategy);
    size_t calculateShardOffset(size_t totalSize, int numDevices,
                               int deviceIndex, ShardingStrategy strategy);
};

// 负载均衡器
class LoadBalancer {
public:
    LoadBalancer(LoadBalanceStrategy strategy = LoadBalanceStrategy::ROUND_ROBIN);

    // 设置可用设备
    void setAvailableDevices(const std::vector<GPUDevice>& devices);

    // 选择设备执行任务
    int selectDevice(const MultiGPUTask& task);
    std::vector<int> selectDevices(int numDevices);

    // 更新设备状态
    void updateDeviceLoad(int deviceId, float loadFactor);
    void updateDeviceMemory(int deviceId, size_t freeMemory);

    // 获取负载统计
    std::map<int, float> getDeviceLoads() const;
    float getAverageLoad() const;

    // 重新平衡负载
    std::vector<std::pair<int, int>> rebalanceLoad();

private:
    LoadBalanceStrategy strategy_;
    std::vector<GPUDevice> devices_;
    std::map<int, float> deviceLoads_;
    std::map<int, size_t> deviceMemoryUsage_;
    int roundRobinIndex_;

    int selectByRoundRobin();
    int selectByMemoryAware();
    int selectByComputeAware();
    int selectByDynamicAdaptive();
};

// 多GPU执行器
class MultiGPUExecutor {
public:
    MultiGPUExecutor();
    ~MultiGPUExecutor();

    // 初始化
    bool initialize(const std::vector<int>& deviceIds);
    void finalize();

    // 执行任务
    bool executeTask(const MultiGPUTask& task, const std::vector<DataShard>& shards);
    bool executeTasks(const std::vector<MultiGPUTask>& tasks,
                     const std::vector<DataShard>& shards);

    // 并行执行
    bool executeParallel(const MultiGPUTask& task,
                        const std::vector<DataShard>& shards);

    // 流水线执行
    bool executePipeline(const std::vector<MultiGPUTask>& tasks,
                        const std::vector<DataShard>& shards);

    // 同步等待
    void synchronize();
    void synchronizeDevice(int deviceId);

    // 获取执行统计
    std::map<int, float> getExecutionTimes() const;
    float getTotalExecutionTime() const;

private:
    std::vector<GPUDevice> devices_;
    std::unique_ptr<NCCLCommunicator> communicator_;
    std::unique_ptr<LoadBalancer> loadBalancer_;
    std::vector<std::thread> workerThreads_;
    std::mutex taskMutex_;
    std::condition_variable taskCondition_;
    std::atomic<bool> shutdown_;

    void workerThreadFunction(int deviceId);
    void executeTaskOnDevice(const MultiGPUTask& task,
                           const DataShard& shard, int deviceId);
};

// 多GPU上下文管理器
class MultiGPUContext {
public:
    MultiGPUContext();
    ~MultiGPUContext();

    // 初始化多GPU环境
    bool initialize(const std::vector<int>& deviceIds,
                   LoadBalanceStrategy loadStrategy = LoadBalanceStrategy::ROUND_ROBIN);
    void finalize();

    // 获取组件
    NCCLCommunicator* getCommunicator() { return communicator_.get(); }
    DataShardManager* getShardManager() { return shardManager_.get(); }
    LoadBalancer* getLoadBalancer() { return loadBalancer_.get(); }
    MultiGPUExecutor* getExecutor() { return executor_.get(); }

    // 设备管理
    const std::vector<GPUDevice>& getDevices() const { return devices_; }
    GPUDevice* getDevice(int deviceId);
    bool isDeviceActive(int deviceId) const;

    // 内存管理
    void* allocateOnDevice(int deviceId, size_t size);
    void freeOnDevice(int deviceId, void* ptr);
    bool copyBetweenDevices(int srcDevice, int dstDevice,
                           void* src, void* dst, size_t size);

    // 流管理
    cudaStream_t getStream(int deviceId);
    void synchronizeStream(int deviceId);
    void synchronizeAllStreams();

    // 错误处理
    std::string getLastError() const { return lastError_; }
    void clearError() { lastError_.clear(); }

private:
    std::vector<GPUDevice> devices_;
    std::unique_ptr<NCCLCommunicator> communicator_;
    std::unique_ptr<DataShardManager> shardManager_;
    std::unique_ptr<LoadBalancer> loadBalancer_;
    std::unique_ptr<MultiGPUExecutor> executor_;
    std::string lastError_;
    bool initialized_;

    bool initializeDevices(const std::vector<int>& deviceIds);
    void cleanupDevices();
    void setError(const std::string& error);
};

// 多GPU训练框架
class MultiGPUTrainer {
public:
    MultiGPUTrainer(MultiGPUContext* context);

    // 数据并行训练
    bool trainDataParallel(void* data, size_t dataSize, int batchSize,
                          std::function<void(void*, size_t, int)> trainStep);

    // 模型并行训练
    bool trainModelParallel(void* model, size_t modelSize,
                           std::function<void(void*, int)> trainStep);

    // 流水线并行训练
    bool trainPipelineParallel(const std::vector<void*>& modelStages,
                              const std::vector<size_t>& stageSizes,
                              std::function<void(void*, int, int)> trainStep);

    // 梯度同步
    bool synchronizeGradients(void* gradients, size_t gradSize);

    // 参数更新
    bool updateParameters(void* parameters, size_t paramSize);

private:
    MultiGPUContext* context_;
};

// 多GPU推理框架
class MultiGPUInference {
public:
    MultiGPUInference(MultiGPUContext* context);

    // 批处理推理
    bool inferBatch(void* inputData, size_t inputSize, int batchSize,
                   void* outputData, size_t outputSize,
                   std::function<void(void*, void*, int)> inferenceStep);

    // 流式推理
    bool inferStream(void* inputData, size_t inputSize,
                    void* outputData, size_t outputSize,
                    std::function<void(void*, void*, int)> inferenceStep);

    // 模型并行推理
    bool inferModelParallel(void* inputData, size_t inputSize,
                           const std::vector<void*>& modelParts,
                           void* outputData, size_t outputSize);

private:
    MultiGPUContext* context_;
};

// 工具函数
namespace multi_gpu_utils {

// 检查NCCL错误
bool checkNCCLError(ncclResult_t result, const std::string& operation);

// 获取最优设备组合
std::vector<int> getOptimalDeviceSet(int numDevices);

// 计算通信开销
float estimateCommunicationCost(CommType type, size_t dataSize, int numDevices);

// 计算负载均衡效率
float calculateLoadBalanceEfficiency(const std::map<int, float>& deviceLoads);

// 生成性能报告
std::string generateMultiGPUReport(const MultiGPUContext& context);

} // namespace multi_gpu_utils

} // namespace cuda_learning
