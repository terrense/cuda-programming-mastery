#include "advanced_memory_optimizer.h"
#include "error_handler.h"
#include <iostream>
#include <iomanip>

using namespace cuda_learning;

int main() {
    std::cout << "=== CUDA高级内存优化技术演示 ===\n\n";

    try {
        AdvancedMemoryOptimizer optimizer;

        // 1. 内存合并分析演示
        std::cout << "1. 内存合并模式分析:\n";
        std::cout << "   测试不同访问模式的性能差异...\n\n";

        // 测试不同数据大小和访问模式
        std::vector<size_t> test_sizes = {1024, 4096, 16384, 65536};
        std::vector<int> access_patterns = {1, 2, 4, 8, 16};

        for (size_t data_size : test_sizes) {
            std::cout << "   数据大小: " << data_size << " 元素\n";

            for (int pattern : access_patterns) {
                auto analysis = optimizer.analyzeMemoryCoalescing(data_size, pattern);

                std::cout << "     跨步=" << pattern
                         << ": 效率=" << std::fixed << std::setprecision(2)
                         << (analysis.efficiency * 100) << "%, "
                         << "带宽=" << std::setprecision(1) << analysis.bandwidth_gb_s << " GB/s\n";
            }
            std::cout << "\n";
        }

        // 2. 银行冲突分析演示
        std::cout << "2. 共享内存银行冲突分析:\n";
        std::cout << "   测试不同访问模式的银行冲突情况...\n\n";

        std::vector<int> shared_sizes = {256, 512, 1024};
        std::vector<int> conflict_patterns = {1, 2, 4, 8};

        for (int shared_size : shared_sizes) {
            std::cout << "   共享内存大小: " << shared_size << " 元素\n";

            for (int pattern : conflict_patterns) {
                auto analysis = optimizer.analyzeBankConflicts(shared_size, pattern);

                std::cout << "     模式=" << pattern
                         << ": 冲突比率=" << std::fixed << std::setprecision(2)
                         << analysis.conflict_ratio << "x, "
                         << "估算冲突=" << analysis.estimated_conflicts << "\n";
            }
            std::cout << "\n";
        }

        // 3. 纹理内存演示
        std::cout << "3. 纹理内存使用演示:\n";
        std::cout << "   测试纹理内存的性能特性...\n\n";

        std::vector<std::pair<int, int>> texture_sizes = {{64, 64}, {128, 128}, {256, 256}};

        for (auto [width, height] : texture_sizes) {
            auto demo = optimizer.demonstrateTextureMemory(width, height);

            std::cout << "   尺寸: " << width << "x" << height
                     << ", 执行时间: " << std::fixed << std::setprecision(3)
                     << demo.execution_time_ms << " ms"
                     << ", 缓存命中率: " << std::setprecision(1)
                     << (demo.cache_hit_rate * 100) << "%\n";
        }
        std::cout << "\n";

        // 4. 表面内存演示
        std::cout << "4. 表面内存使用演示:\n";
        std::cout << "   测试表面内存的写入性能...\n\n";

        for (auto [width, height] : texture_sizes) {
            auto demo = optimizer.demonstrateSurfaceMemory(width, height);

            std::cout << "   尺寸: " << width << "x" << height
                     << ", 执行时间: " << std::fixed << std::setprecision(3)
                     << demo.execution_time_ms << " ms"
                     << ", 写入带宽: " << std::setprecision(1)
                     << demo.write_bandwidth_gb_s << " GB/s\n";
        }
        std::cout << "\n";

        // 5. 生成综合优化报告
        std::cout << "5. 综合优化分析报告:\n\n";

        // 使用中等大小的数据进行详细分析
        auto coalescing_analysis = optimizer.analyzeMemoryCoalescing(16384, 4);
        auto bank_conflict_analysis = optimizer.analyzeBankConflicts(512, 2);

        std::string report = optimizer.generateOptimizationReport(coalescing_analysis, bank_conflict_analysis);
        std::cout << report << std::endl;

        // 6. 优化建议总结
        std::cout << "6. 关键优化技术总结:\n";
        std::cout << "   - 内存合并: 确保连续线程访问连续内存地址\n";
        std::cout << "   - 银行冲突避免: 使用填充或改变索引模式\n";
        std::cout << "   - 纹理内存: 适用于具有空间局部性的随机访问\n";
        std::cout << "   - 表面内存: 支持读写操作的2D/3D数据处理\n";
        std::cout << "   - 性能分析: 使用CUDA事件和分析工具测量优化效果\n\n";

        std::cout << "演示完成！通过这些技术可以显著提升CUDA程序的内存访问效率。\n";

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
