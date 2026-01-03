#include "performance_analyzer.h"
#include "error_handler.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>

namespace cuda_learning {

// PerformanceMetrics 实现
PerformanceMetrics::PerformanceMetrics()
    : executionTime_ms(0.0f), bandwidth_GB_s(0.0f), occupancy_percent(0.0f),
      activeWarps(0), maxWarps(0), registersPerThread(0),
      sharedMemoryUsed(0), sharedMemoryAvailable(0),
      computeUtilization_percent(0.0f), memoryUtilization_percent(0.0f) {}

std::string PerformanceMetrics::toString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "执行时间: " << executionTime_ms << " ms\n";
    ss << "内存带宽: " << bandwidth_GB_s << " GB/s\n";
    ss << "占用率: " << occupancy_percent << "%\n";
    ss << "活跃Warp数: " << activeWarps << "/" << maxWarps << "\n";
    ss << "每线程寄存器: " << registersPerThread << "\n";
    ss << "共享内存使用: " << sharedMemoryUsed << "/" << sharedMemoryAvailable << " 字节\n";
    ss << "计算利用率: " << computeUtilization_percent << "%\n";
    ss << "内存利用率: " << memoryUtilization_percent << "%";
    return ss.str();
}

// BottleneckAnalysis 实现
BottleneckAnalysis::BottleneckAnalysis()
    : primaryBottleneck(BottleneckType::NONE), secondaryBottleneck(BottleneckType::NONE),
      severity(0.0f) {}

std::string BottleneckAnalysis::toString() const {
    std::stringstream ss;

    auto bottleneckToString = [](BottleneckType type) -> std::string {
        switch (type) {
            case BottleneckType::MEMORY_BOUND: return "内存受限";
            case BottleneckType::COMPUTE_BOUND: return "计算受限";
            case BottleneckType::OCCUPANCY_LIMITED: return "占用率受限";
            case BottleneckType::REGISTER_LIMITED: return "寄存器受限";
            case BottleneckType::SHARED_MEMORY_LIMITED: return "共享内存受限";
            case BottleneckType::INSTRUCTION_BOUND: return "指令受限";
            case BottleneckType::DIVERGENCE_BOUND: return "分支发散受限";
            case BottleneckType::NONE: return "无明显瓶颈";
            default: return "未知";
        }
    };

    ss << "主要瓶颈: " << bottleneckToString(primaryBottleneck) << "\n";
    if (secondaryBottleneck != BottleneckType::NONE) {
        ss << "次要瓶颈: " << bottleneckToString(secondaryBottleneck) << "\n";
    }
    ss << "严重程度: " << std::fixed << std::setprecision(1) << (severity * 100) << "%\n";
    ss << "描述: " << description << "\n";

    if (!suggestions.empty()) {
        ss << "优化建议:\n";
        for (size_t i = 0; i < suggestions.size(); i++) {
            ss << "  " << (i + 1) << ". " << suggestions[i] << "\n";
        }
    }

    return ss.str();
}

// OptimizationSuggestion 实现
OptimizationSuggestion::OptimizationSuggestion()
    : type(OptimizationType::MEMORY_COALESCING), expectedImprovement(0.0f), priority(3) {}

std::string OptimizationSuggestion::toString() const {
    std::stringstream ss;
    ss << "标题: " << title << "\n";
    ss << "描述: " << description << "\n";
    ss << "预期提升: " << std::fixed << std::setprecision(1) << expectedImprovement << "%\n";
    ss << "优先级: " << priority << "/5\n";
    if (!codeExample.empty()) {
        ss << "代码示例:\n" << codeExample << "\n";
    }
    return ss.str();
}

// CudaTimer 实现
CudaTimer::CudaTimer() : timing_active(false) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void CudaTimer::start() {
    cudaEventRecord(start_event);
    timing_active = true;
}

void CudaTimer::stop() {
    if (timing_active) {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        timing_active = false;
    }
}

float CudaTimer::getElapsedTime() {
    if (timing_active) {
        stop();
    }

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    return elapsed_time;
}

template<typename Func>
float CudaTimer::timeFunction(Func func, int iterations) {
    CudaTimer timer;
    timer.start();

    for (int i = 0; i < iterations; i++) {
        func();
    }

    timer.stop();
    return timer.getElapsedTime() / iterations;
}

// OccupancyCalculator 实现
float OccupancyCalculator::calculateTheoreticalOccupancy(int blockSize, int registersPerThread,
                                                       size_t sharedMemPerBlock, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);

    // 计算每个SM可以容纳的块数
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;

    // 寄存器限制
    int maxBlocksByRegisters = prop.regsPerMultiprocessor / (blockSize * registersPerThread);

    // 共享内存限制
    int maxBlocksBySharedMem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;

    // 取最小值
    int actualBlocksPerSM = std::min({maxBlocksPerSM, maxBlocksByRegisters, maxBlocksBySharedMem});

    // 计算占用率
    float occupancy = (float)(actualBlocksPerSM * blockSize) / prop.maxThreadsPerMultiProcessor;
    return std::min(occupancy, 1.0f) * 100.0f;
}

float OccupancyCalculator::calculateActualOccupancy(const void* kernel, int blockSize,
                                                  size_t sharedMemPerBlock, int deviceId) {
    int minGridSize, optimalBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, kernel,
                                      sharedMemPerBlock, blockSize);

    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, kernel,
                                                 blockSize, sharedMemPerBlock);

    cudaDeviceProp prop = getDeviceProperties(deviceId);
    float occupancy = (float)(maxActiveBlocks * blockSize) / prop.maxThreadsPerMultiProcessor;
    return std::min(occupancy, 1.0f) * 100.0f;
}

int OccupancyCalculator::getOptimalBlockSize(const void* kernel, int deviceId) {
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
    return blockSize;
}

std::string OccupancyCalculator::analyzeOccupancyLimiters(int blockSize, int registersPerThread,
                                                        size_t sharedMemPerBlock, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);
    std::stringstream ss;

    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
    int maxBlocksByRegisters = prop.regsPerMultiprocessor / (blockSize * registersPerThread);
    int maxBlocksBySharedMem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;

    ss << "占用率限制因素分析:\n";
    ss << "  线程数限制: " << maxBlocksPerSM << " 块/SM\n";
    ss << "  寄存器限制: " << maxBlocksByRegisters << " 块/SM\n";
    ss << "  共享内存限制: " << maxBlocksBySharedMem << " 块/SM\n";

    int actualBlocks = std::min({maxBlocksPerSM, maxBlocksByRegisters, maxBlocksBySharedMem});

    if (actualBlocks == maxBlocksByRegisters) {
        ss << "主要限制: 寄存器使用过多\n";
        ss << "建议: 减少寄存器使用或增加块大小\n";
    } else if (actualBlocks == maxBlocksBySharedMem) {
        ss << "主要限制: 共享内存使用过多\n";
        ss << "建议: 减少共享内存使用或优化数据布局\n";
    } else {
        ss << "主要限制: 线程数配置\n";
        ss << "建议: 调整块大小以提高占用率\n";
    }

    return ss.str();
}

cudaDeviceProp OccupancyCalculator::getDeviceProperties(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    return prop;
}

// PerformanceAnalyzer 核心实现
BottleneckAnalysis PerformanceAnalyzer::detectBottlenecks(const PerformanceMetrics& metrics, int deviceId) {
    BottleneckAnalysis analysis;

    // 分析占用率
    if (metrics.occupancy_percent < 50.0f) {
        analysis.primaryBottleneck = BottleneckType::OCCUPANCY_LIMITED;
        analysis.severity = (50.0f - metrics.occupancy_percent) / 50.0f;
        analysis.description = "占用率过低，GPU资源未充分利用";
        analysis.suggestions.push_back("增加块大小或减少资源使用");
        analysis.suggestions.push_back("检查寄存器和共享内存使用");
    }
    // 分析内存利用率
    else if (metrics.memoryUtilization_percent > 80.0f && metrics.computeUtilization_percent < 50.0f) {
        analysis.primaryBottleneck = BottleneckType::MEMORY_BOUND;
        analysis.severity = metrics.memoryUtilization_percent / 100.0f;
        analysis.description = "内存带宽成为瓶颈，计算单元等待数据";
        analysis.suggestions.push_back("优化内存访问模式，实现合并访问");
        analysis.suggestions.push_back("使用共享内存缓存频繁访问的数据");
        analysis.suggestions.push_back("考虑使用纹理内存或常量内存");
    }
    // 分析计算利用率
    else if (metrics.computeUtilization_percent > 80.0f && metrics.memoryUtilization_percent < 50.0f) {
        analysis.primaryBottleneck = BottleneckType::COMPUTE_BOUND;
        analysis.severity = metrics.computeUtilization_percent / 100.0f;
        analysis.description = "计算资源成为瓶颈，可能需要算法优化";
        analysis.suggestions.push_back("优化算法复杂度");
        analysis.suggestions.push_back("使用更高效的数学函数");
        analysis.suggestions.push_back("考虑使用Tensor Core或其他专用单元");
    }
    // 分析寄存器使用
    else if (metrics.registersPerThread > 32) {
        analysis.primaryBottleneck = BottleneckType::REGISTER_LIMITED;
        analysis.severity = std::min(1.0f, (metrics.registersPerThread - 32) / 32.0f);
        analysis.description = "寄存器使用过多，限制了占用率";
        analysis.suggestions.push_back("减少局部变量使用");
        analysis.suggestions.push_back("使用共享内存替代部分寄存器");
        analysis.suggestions.push_back("考虑函数内联优化");
    }
    else {
        analysis.primaryBottleneck = BottleneckType::NONE;
        analysis.description = "性能表现良好，无明显瓶颈";
        analysis.suggestions.push_back("继续监控性能变化");
        analysis.suggestions.push_back("考虑进一步的微调优化");
    }

    return analysis;
}

std::vector<OptimizationSuggestion> PerformanceAnalyzer::generateOptimizationSuggestions(
    const PerformanceMetrics& metrics, const BottleneckAnalysis& bottlenecks) {

    std::vector<OptimizationSuggestion> suggestions;

    // 基于瓶颈类型生成建议
    switch (bottlenecks.primaryBottleneck) {
        case BottleneckType::MEMORY_BOUND: {
            OptimizationSuggestion memCoalescing;
            memCoalescing.type = OptimizationType::MEMORY_COALESCING;
            memCoalescing.title = "优化内存合并访问";
            memCoalescing.description = "重新组织数据访问模式以实现内存合并，提高带宽利用率";
            memCoalescing.expectedImprovement = 30.0f;
            memCoalescing.priority = 5;
            memCoalescing.codeExample = R"(
// 优化前：跨步访问
__global__ void uncoalesced(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx * stride] = data[idx * stride] * 2.0f;
}

// 优化后：合并访问
__global__ void coalesced(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}
)";
            suggestions.push_back(memCoalescing);

            OptimizationSuggestion sharedMem;
            sharedMem.type = OptimizationType::SHARED_MEMORY_USAGE;
            sharedMem.title = "使用共享内存缓存";
            sharedMem.description = "使用共享内存缓存频繁访问的全局内存数据";
            sharedMem.expectedImprovement = 25.0f;
            sharedMem.priority = 4;
            suggestions.push_back(sharedMem);
            break;
        }

        case BottleneckType::OCCUPANCY_LIMITED: {
            OptimizationSuggestion occupancy;
            occupancy.type = OptimizationType::OCCUPANCY_IMPROVEMENT;
            occupancy.title = "提高占用率";
            occupancy.description = "调整块大小和资源使用以提高GPU占用率";
            occupancy.expectedImprovement = 40.0f;
            occupancy.priority = 5;
            occupancy.codeExample = R"(
// 使用占用率计算器找到最优块大小
int optimalBlockSize = OccupancyCalculator::getOptimalBlockSize(myKernel);
dim3 blockSize(optimalBlockSize);
dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
)";
            suggestions.push_back(occupancy);
            break;
        }

        case BottleneckType::REGISTER_LIMITED: {
            OptimizationSuggestion registers;
            registers.type = OptimizationType::REGISTER_OPTIMIZATION;
            registers.title = "优化寄存器使用";
            registers.description = "减少寄存器使用以提高占用率";
            registers.expectedImprovement = 20.0f;
            registers.priority = 4;
            registers.codeExample = R"(
// 优化前：过多局部变量
__global__ void registerHeavy() {
    float a, b, c, d, e, f, g, h; // 过多寄存器
    // ... 复杂计算
}

// 优化后：使用共享内存
__global__ void registerOptimized() {
    __shared__ float temp[256];
    // 使用共享内存替代部分寄存器
}
)";
            suggestions.push_back(registers);
            break;
        }

        default:
            break;
    }

    // 通用优化建议
    if (metrics.occupancy_percent < 75.0f) {
        OptimizationSuggestion blockTuning;
        blockTuning.type = OptimizationType::BLOCK_SIZE_TUNING;
        blockTuning.title = "调优块大小";
        blockTuning.description = "使用自动调优工具找到最佳块大小配置";
        blockTuning.expectedImprovement = 15.0f;
        blockTuning.priority = 3;
        suggestions.push_back(blockTuning);
    }

    return suggestions;
}

std::string PerformanceAnalyzer::generatePerformanceReport(const PerformanceMetrics& metrics,
                                                         const BottleneckAnalysis& bottlenecks,
                                                         const std::vector<OptimizationSuggestion>& suggestions) {
    std::stringstream ss;

    ss << "=== CUDA性能分析报告 ===\n\n";

    ss << "1. 性能指标:\n";
    ss << metrics.toString() << "\n\n";

    ss << "2. 瓶颈分析:\n";
    ss << bottlenecks.toString() << "\n";

    ss << "3. 优化建议:\n";
    for (size_t i = 0; i < suggestions.size(); i++) {
        ss << "建议 " << (i + 1) << ":\n";
        ss << suggestions[i].toString() << "\n";
    }

    ss << "4. 总结:\n";
    if (bottlenecks.primaryBottleneck == BottleneckType::NONE) {
        ss << "性能表现良好，建议继续监控和微调。\n";
    } else {
        ss << "发现性能瓶颈，建议按优先级实施优化措施。\n";
        ss << "预期总体性能提升: ";
        float totalImprovement = 0.0f;
        for (const auto& suggestion : suggestions) {
            totalImprovement += suggestion.expectedImprovement * (suggestion.priority / 5.0f);
        }
        ss << std::fixed << std::setprecision(1) << totalImprovement << "%\n";
    }

    return ss.str();
}

// MemoryBandwidthAnalyzer 实现
float MemoryBandwidthAnalyzer::measureBandwidth(size_t dataSize, int iterations) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);

    // 简单的内存复制核函数
    auto copyKernel = [](float* input, float* output, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = input[idx];
        }
    };

    int blockSize = 256;
    int gridSize = (dataSize / sizeof(float) + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.start();

    for (int i = 0; i < iterations; i++) {
        // 这里需要实际的核函数调用，简化处理
        cudaMemcpy(d_output, d_input, dataSize, cudaMemcpyDeviceToDevice);
    }

    timer.stop();
    float elapsedTime = timer.getElapsedTime();

    // 计算带宽 (GB/s)
    float bandwidth = (2.0f * dataSize * iterations) / (elapsedTime * 1e-3) / (1024*1024*1024);

    cudaFree(d_input);
    cudaFree(d_output);

    return bandwidth;
}

float MemoryBandwidthAnalyzer::analyzeMemoryEfficiency(float measuredBandwidth, int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    // 估算理论峰值带宽
    float theoreticalBandwidth = prop.memoryBusWidth * prop.memoryClockRate * 2.0f / (8 * 1e6);

    return (measuredBandwidth / theoreticalBandwidth) * 100.0f;
}

// AutoTuner 实现
AutoTuner::TuningConfig::TuningConfig()
    : blockSizes({64, 128, 256, 512, 1024}),
      gridSizes({32, 64, 128, 256}),
      sharedMemSizes({0, 1024, 2048, 4096}),
      maxIterations(100),
      convergenceThreshold(0.01f) {}

AutoTuner::TuningResult::TuningResult()
    : optimalGrid(1), optimalBlock(256), optimalSharedMem(0) {}

std::string AutoTuner::TuningResult::toString() const {
    std::stringstream ss;
    ss << "最优配置:\n";
    ss << "  网格大小: (" << optimalGrid.x << ", " << optimalGrid.y << ", " << optimalGrid.z << ")\n";
    ss << "  块大小: (" << optimalBlock.x << ", " << optimalBlock.y << ", " << optimalBlock.z << ")\n";
    ss << "  共享内存: " << optimalSharedMem << " 字节\n";
    ss << "最佳性能:\n" << bestMetrics.toString();
    return ss.str();
}

// PerformanceBenchmark 实现
PerformanceBenchmark::BenchmarkResult::BenchmarkResult()
    : timestamp(std::chrono::system_clock::now()) {}

std::string PerformanceBenchmark::BenchmarkResult::toString() const {
    std::stringstream ss;
    ss << "测试名称: " << testName << "\n";
    ss << "设备信息: " << deviceInfo << "\n";
    ss << "性能指标:\n" << metrics.toString() << "\n";
    return ss.str();
}

std::vector<PerformanceBenchmark::BenchmarkResult> PerformanceBenchmark::runStandardBenchmarks(int deviceId) {
    std::vector<BenchmarkResult> results;

    // 内存带宽测试
    results.push_back(benchmarkMemoryBandwidth(deviceId));

    // 计算性能测试
    results.push_back(benchmarkComputePerformance(deviceId));

    // 占用率测试
    results.push_back(benchmarkOccupancy(deviceId));

    return results;
}

PerformanceBenchmark::BenchmarkResult PerformanceBenchmark::benchmarkMemoryBandwidth(int deviceId) {
    BenchmarkResult result;
    result.testName = "内存带宽基准测试";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    result.deviceInfo = std::string(prop.name);

    // 测试内存带宽
    size_t dataSize = 64 * 1024 * 1024; // 64MB
    result.metrics.bandwidth_GB_s = MemoryBandwidthAnalyzer::measureBandwidth(dataSize);
    result.metrics.memoryUtilization_percent = MemoryBandwidthAnalyzer::analyzeMemoryEfficiency(
        result.metrics.bandwidth_GB_s, deviceId);

    return result;
}

PerformanceBenchmark::BenchmarkResult PerformanceBenchmark::benchmarkComputePerformance(int deviceId) {
    BenchmarkResult result;
    result.testName = "计算性能基准测试";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    result.deviceInfo = std::string(prop.name);

    // 这里应该实现具体的计算性能测试
    // 简化处理，设置一些示例值
    result.metrics.computeUtilization_percent = 75.0f;
    result.metrics.executionTime_ms = 10.5f;

    return result;
}

PerformanceBenchmark::BenchmarkResult PerformanceBenchmark::benchmarkOccupancy(int deviceId) {
    BenchmarkResult result;
    result.testName = "占用率基准测试";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    result.deviceInfo = std::string(prop.name);

    // 测试不同块大小的占用率
    result.metrics.occupancy_percent = OccupancyCalculator::calculateTheoreticalOccupancy(
        256, 16, 1024, deviceId);

    return result;
}

} // namespace cuda_learning
