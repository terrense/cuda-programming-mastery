#ifndef PERFORMANCE_ANALYZER_H
#define PERFORMANCE_ANALYZER_H

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>

namespace cuda_learning {

// 性能指标结构
struct PerformanceMetrics {
    float executionTime_ms;
    float bandwidth_GB_s;
    float occupancy_percent;
    int activeWarps;
    int maxWarps;
    int registersPerThread;
    size_t sharedMemoryUsed;
    size_t sharedMemoryAvailable;
    float computeUtilization_percent;
    float memoryUtilization_percent;

    PerformanceMetrics();
    std::string toString() const;
};

// 性能瓶颈类型
enum class BottleneckType {
    MEMORY_BOUND,           // 内存受限
    COMPUTE_BOUND,          // 计算受限
    OCCUPANCY_LIMITED,      // 占用率受限
    REGISTER_LIMITED,       // 寄存器受限
    SHARED_MEMORY_LIMITED,  // 共享内存受限
    INSTRUCTION_BOUND,      // 指令受限
    DIVERGENCE_BOUND,       // 分支发散受限
    NONE                    // 无明显瓶颈
};

// 瓶颈分析结果
struct BottleneckAnalysis {
    BottleneckType primaryBottleneck;
    BottleneckType secondaryBottleneck;
    float severity;  // 严重程度 0-1
    std::string description;
    std::vector<std::string> suggestions;

    BottleneckAnalysis();
    std::string toString() const;
};

// 优化建议类型
enum class OptimizationType {
    MEMORY_COALESCING,      // 内存合并
    SHARED_MEMORY_USAGE,    // 共享内存使用
    REGISTER_OPTIMIZATION,  // 寄存器优化
    OCCUPANCY_IMPROVEMENT,  // 占用率提升
    ALGORITHM_CHANGE,       // 算法改进
    BLOCK_SIZE_TUNING,      // 块大小调优
    GRID_SIZE_TUNING,       // 网格大小调优
    INSTRUCTION_OPTIMIZATION // 指令优化
};

// 优化建议
struct OptimizationSuggestion {
    OptimizationType type;
    std::string title;
    std::string description;
    std::string codeExample;
    float expectedImprovement; // 预期性能提升百分比
    int priority; // 优先级 1-5

    OptimizationSuggestion();
    std::string toString() const;
};

// CUDA事件计时器
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool timing_active;

public:
    CudaTimer();
    ~CudaTimer();

    void start();
    void stop();
    float getElapsedTime(); // 返回毫秒

    // 静态方法用于简单计时
    template<typename Func>
    static float timeFunction(Func func, int iterations = 1);

    // 计时核函数执行
    template<typename... Args>
    static float timeKernel(void(*kernel)(Args...), dim3 grid, dim3 block,
                           size_t sharedMem, cudaStream_t stream, Args... args);
};

// 占用率计算器
class OccupancyCalculator {
public:
    // 计算理论占用率
    static float calculateTheoreticalOccupancy(int blockSize, int registersPerThread,
                                             size_t sharedMemPerBlock, int deviceId = 0);

    // 计算实际占用率
    static float calculateActualOccupancy(const void* kernel, int blockSize,
                                        size_t sharedMemPerBlock, int deviceId = 0);

    // 获取最优块大小
    static int getOptimalBlockSize(const void* kernel, int deviceId = 0);

    // 分析占用率限制因素
    static std::string analyzeOccupancyLimiters(int blockSize, int registersPerThread,
                                              size_t sharedMemPerBlock, int deviceId = 0);

    // 计算不同块大小的占用率
    static std::map<int, float> calculateOccupancyForBlockSizes(
        const void* kernel, const std::vector<int>& blockSizes, int deviceId = 0);

private:
    static cudaDeviceProp getDeviceProperties(int deviceId);
};

// 性能分析器主类
class PerformanceAnalyzer {
public:
    // 分析核函数性能
    template<typename... Args>
    static PerformanceMetrics analyzeKernel(void(*kernel)(Args...), dim3 grid, dim3 block,
                                           size_t sharedMem, cudaStream_t stream,
                                           Args... args, int iterations = 100);

    // 检测性能瓶颈
    static BottleneckAnalysis detectBottlenecks(const PerformanceMetrics& metrics,
                                              int deviceId = 0);

    // 生成优化建议
    static std::vector<OptimizationSuggestion> generateOptimizationSuggestions(
        const PerformanceMetrics& metrics, const BottleneckAnalysis& bottlenecks);

    // 比较不同配置的性能
    template<typename... Args>
    static std::vector<PerformanceMetrics> compareConfigurations(
        void(*kernel)(Args...), const std::vector<dim3>& grids,
        const std::vector<dim3>& blocks, Args... args);

    // 自动调优块大小
    template<typename... Args>
    static std::pair<dim3, PerformanceMetrics> autoTuneBlockSize(
        void(*kernel)(Args...), dim3 grid, int maxBlockSize, Args... args);

    // 生成性能报告
    static std::string generatePerformanceReport(const PerformanceMetrics& metrics,
                                               const BottleneckAnalysis& bottlenecks,
                                               const std::vector<OptimizationSuggestion>& suggestions);
};

// 内存带宽分析器
class MemoryBandwidthAnalyzer {
public:
    // 测试内存带宽
    static float measureBandwidth(size_t dataSize, int iterations = 100);

    // 分析内存访问效率
    static float analyzeMemoryEfficiency(float measuredBandwidth, int deviceId = 0);

    // 检测内存访问模式
    static std::string detectAccessPattern(const PerformanceMetrics& metrics);

    // 建议内存优化
    static std::vector<std::string> suggestMemoryOptimizations(float efficiency);
};

// 计算强度分析器
class ComputeIntensityAnalyzer {
public:
    // 计算算术强度 (FLOP/Byte)
    static float calculateArithmeticIntensity(long long flops, long long memoryBytes);

    // 分析计算受限 vs 内存受限
    static std::string analyzeComputeMemoryBound(float arithmeticIntensity, int deviceId = 0);

    // 建议优化策略
    static std::vector<std::string> suggestOptimizationStrategy(float arithmeticIntensity);
};

// 自动性能调优器
class AutoTuner {
public:
    struct TuningConfig {
        std::vector<int> blockSizes;
        std::vector<int> gridSizes;
        std::vector<size_t> sharedMemSizes;
        int maxIterations;
        float convergenceThreshold;

        TuningConfig();
    };

    struct TuningResult {
        dim3 optimalGrid;
        dim3 optimalBlock;
        size_t optimalSharedMem;
        PerformanceMetrics bestMetrics;
        std::vector<PerformanceMetrics> allResults;

        TuningResult();
        std::string toString() const;
    };

    // 自动调优核函数
    template<typename... Args>
    static TuningResult autoTune(void(*kernel)(Args...), const TuningConfig& config,
                               Args... args);

    // 网格搜索调优
    template<typename... Args>
    static TuningResult gridSearchTune(void(*kernel)(Args...), const TuningConfig& config,
                                     Args... args);

    // 遗传算法调优
    template<typename... Args>
    static TuningResult geneticAlgorithmTune(void(*kernel)(Args...), const TuningConfig& config,
                                           Args... args);
};

// 性能基准测试套件
class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        std::string testName;
        PerformanceMetrics metrics;
        std::string deviceInfo;
        std::chrono::system_clock::time_point timestamp;

        BenchmarkResult();
        std::string toString() const;
    };

    // 运行标准基准测试
    static std::vector<BenchmarkResult> runStandardBenchmarks(int deviceId = 0);

    // 内存带宽基准测试
    static BenchmarkResult benchmarkMemoryBandwidth(int deviceId = 0);

    // 计算性能基准测试
    static BenchmarkResult benchmarkComputePerformance(int deviceId = 0);

    // 占用率基准测试
    static BenchmarkResult benchmarkOccupancy(int deviceId = 0);

    // 生成基准测试报告
    static std::string generateBenchmarkReport(const std::vector<BenchmarkResult>& results);

    // 比较不同设备性能
    static std::string compareDevicePerformance(const std::vector<int>& deviceIds);
};

// 性能监控器
class PerformanceMonitor {
private:
    std::vector<PerformanceMetrics> history;
    bool monitoring;
    std::chrono::steady_clock::time_point startTime;

public:
    PerformanceMonitor();

    void startMonitoring();
    void stopMonitoring();
    void recordMetrics(const PerformanceMetrics& metrics);

    std::vector<PerformanceMetrics> getHistory() const;
    PerformanceMetrics getAverageMetrics() const;
    std::string generateMonitoringReport() const;

    // 检测性能回归
    bool detectPerformanceRegression(float threshold = 0.1f) const;

    // 清除历史记录
    void clearHistory();
};

// 性能可视化工具
class PerformanceVisualizer {
public:
    // 生成性能图表数据
    static std::string generateChartData(const std::vector<PerformanceMetrics>& metrics);

    // 生成占用率热力图
    static std::string generateOccupancyHeatmap(const std::map<int, float>& occupancyData);

    // 生成性能趋势图
    static std::string generateTrendChart(const std::vector<PerformanceMetrics>& history);

    // 生成瓶颈分析图
    static std::string generateBottleneckChart(const BottleneckAnalysis& analysis);
};

} // namespace cuda_learning

#endif // PERFORMANCE_ANALYZER_H
