#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <chrono>

namespace cuda_learning {

// 内存类型枚举
enum class MemoryType {
    GLOBAL,
    SHARED,
    CONSTANT,
    TEXTURE,
    UNIFIED
};

// 内存访问模式枚举
enum class AccessPattern {
    COALESCED,      // 合并访问
    STRIDED,        // 跨步访问
    RANDOM,         // 随机访问
    BROADCAST       // 广播访问
};

// 内存性能测试结果
struct MemoryBenchmarkResult {
    MemoryType memoryType;
    AccessPattern accessPattern;
    size_t dataSize;
    float bandwidth_GB_s;
    float executionTime_ms;
    int blockSize;
    int gridSize;
    std::string description;

    MemoryBenchmarkResult();
    std::string toString() const;
};

// 内存管理教学系统主类
class MemoryManagementTeacher {
public:
    // 全局内存示例
    static void demonstrateGlobalMemory();
    static void showCoalescedVsUncoalescedAccess();
    static void demonstrateMemoryAlignment();

    // 共享内存示例
    static void demonstrateSharedMemory();
    static void showBankConflicts();
    static void demonstrateSharedMemoryOptimization();

    // 常量内存示例
    static void demonstrateConstantMemory();
    static void showConstantMemoryBroadcast();

    // 纹理内存示例（如果支持）
    static void demonstrateTextureMemory();

    // 统一内存示例
    static void demonstrateUnifiedMemory();

    // 内存带宽测试
    static MemoryBenchmarkResult benchmarkMemoryBandwidth(
        MemoryType memType,
        AccessPattern pattern,
        size_t dataSize,
        int iterations = 100
    );

    // 比较不同内存类型的性能
    static std::vector<MemoryBenchmarkResult> compareMemoryTypes(size_t dataSize);

    // 生成内存优化报告
    static std::string generateOptimizationReport(const std::vector<MemoryBenchmarkResult>& results);
};

// 全局内存操作核函数
namespace GlobalMemoryKernels {
    // 合并访问示例
    __global__ void coalescedAccess(float* input, float* output, int n);

    // 非合并访问示例
    __global__ void uncoalescedAccess(float* input, float* output, int n, int stride);

    // 内存对齐示例
    __global__ void alignedAccess(float* input, float* output, int n);
    __global__ void misalignedAccess(float* input, float* output, int n, int offset);
}

// 共享内存操作核函数
namespace SharedMemoryKernels {
    // 基本共享内存使用
    __global__ void basicSharedMemory(float* input, float* output, int n);

    // 银行冲突演示
    __global__ void bankConflictDemo(float* output, int n);
    __global__ void noBankConflictDemo(float* output, int n);

    // 共享内存矩阵转置
    __global__ void sharedMemoryTranspose(float* input, float* output, int width, int height);

    // 共享内存归约
    __global__ void sharedMemoryReduction(float* input, float* output, int n);
}

// 常量内存操作核函数
namespace ConstantMemoryKernels {
    // 常量内存声明（在.cu文件中定义）
    extern __constant__ float const_coefficients[256];

    // 使用常量内存的卷积
    __global__ void constantMemoryConvolution(float* input, float* output,
                                            int width, int height, int kernelSize);

    // 常量内存广播示例
    __global__ void constantMemoryBroadcast(float* input, float* output, int n);
}

// 内存带宽测试工具
class MemoryBandwidthTester {
public:
    // 测试全局内存带宽
    static float testGlobalMemoryBandwidth(size_t dataSize, AccessPattern pattern);

    // 测试共享内存带宽
    static float testSharedMemoryBandwidth(size_t dataSize);

    // 测试常量内存带宽
    static float testConstantMemoryBandwidth(size_t dataSize);

    // 测试纹理内存带宽
    static float testTextureMemoryBandwidth(size_t dataSize);

    // 获取理论峰值带宽
    static float getTheoreticalBandwidth(int deviceId = 0);

    // 计算带宽效率
    static float calculateBandwidthEfficiency(float measured, float theoretical);

private:
    // 内部测试核函数
    static float runBandwidthTest(void* kernel, void** args,
                                size_t dataSize, int iterations);
};

// 内存访问模式分析器
class MemoryAccessAnalyzer {
public:
    // 分析访问模式
    static std::string analyzeAccessPattern(const std::vector<int>& accessIndices);

    // 检测合并访问
    static bool isCoalescedAccess(const std::vector<int>& accessIndices, int warpSize = 32);

    // 计算银行冲突数量
    static int calculateBankConflicts(const std::vector<int>& sharedMemIndices, int numBanks = 32);

    // 生成访问模式可视化
    static std::string visualizeAccessPattern(const std::vector<int>& accessIndices, int warpSize = 32);

    // 提供优化建议
    static std::vector<std::string> getOptimizationSuggestions(
        MemoryType memType,
        AccessPattern pattern,
        const MemoryBenchmarkResult& result
    );
};

// 内存优化工具
class MemoryOptimizer {
public:
    // 计算最优的数据布局
    static std::vector<int> optimizeDataLayout(const std::vector<int>& originalLayout,
                                             AccessPattern targetPattern);

    // 建议最优的块大小
    static int suggestOptimalBlockSize(MemoryType memType, size_t dataSize);

    // 计算共享内存使用量
    static size_t calculateSharedMemoryUsage(int blockSize, size_t elementSize,
                                           int elementsPerThread);

    // 检查内存对齐
    static bool checkMemoryAlignment(void* ptr, size_t alignment = 128);

    // 建议内存填充
    static size_t suggestMemoryPadding(size_t originalSize, size_t alignment = 128);
};

// 内存教学演示类
class MemoryDemo {
public:
    // 运行完整的内存管理演示
    static void runCompleteDemo();

    // 交互式内存类型比较
    static void interactiveMemoryComparison();

    // 内存优化案例研究
    static void memoryOptimizationCaseStudy();

    // 生成学习报告
    static std::string generateLearningReport(const std::vector<MemoryBenchmarkResult>& results);

private:
    // 辅助函数
    static void printSectionHeader(const std::string& title);
    static void printBenchmarkResults(const std::vector<MemoryBenchmarkResult>& results);
    static void waitForUserInput();
};

} // namespace cuda_learning

#endif // MEMORY_MANAGEMENT_H
