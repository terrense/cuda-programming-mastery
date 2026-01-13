#ifndef ADVANCED_MEMORY_OPTIMIZER_H
#define ADVANCED_MEMORY_OPTIMIZER_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace cuda_learning {

// 内存合并分析结果
struct CoalescingAnalysis {
    size_t data_size;
    int access_pattern;
    float coalesced_time_ms;
    float strided_time_ms;
    float efficiency;           // 合并访问时间 / 跨步访问时间
    float bandwidth_gb_s;       // 内存带宽 GB/s
    std::vector<std::string> optimization_suggestions;
};

// 银行冲突分析结果
struct BankConflictAnalysis {
    int shared_memory_size;
    int access_pattern;
    float conflict_time_ms;
    float conflict_free_time_ms;
    float conflict_ratio;       // 冲突时间 / 无冲突时间
    int estimated_conflicts;    // 估算的冲突数量
    std::vector<std::string> optimization_suggestions;
};

// 纹理内存演示结果
struct TextureMemoryDemo {
    int width;
    int height;
    float execution_time_ms;
    float cache_hit_rate;
    std::vector<std::string> usage_recommendations;
};

// 表面内存演示结果
struct SurfaceMemoryDemo {
    int width;
    int height;
    float execution_time_ms;
    float write_bandwidth_gb_s;
    std::vector<std::string> usage_recommendations;
};

// 高级内存优化器主类
class AdvancedMemoryOptimizer {
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

public:
    AdvancedMemoryOptimizer();
    ~AdvancedMemoryOptimizer();

    // 分析内存合并模式
    CoalescingAnalysis analyzeMemoryCoalescing(size_t data_size, int access_pattern = 2);

    // 分析银行冲突
    BankConflictAnalysis analyzeBankConflicts(int shared_memory_size, int access_pattern = 2);

    // 演示纹理内存使用
    TextureMemoryDemo demonstrateTextureMemory(int width, int height);

    // 演示表面内存使用
    SurfaceMemoryDemo demonstrateSurfaceMemory(int width, int height);

    // 生成优化报告
    std::string generateOptimizationReport(const CoalescingAnalysis& coalescing,
                                         const BankConflictAnalysis& bank_conflict);
};

} // namespace cuda_learning

#endif // ADVANCED_MEMORY_OPTIMIZER_H
