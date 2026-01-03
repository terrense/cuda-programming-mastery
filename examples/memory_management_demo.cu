#include "../src/core/memory_management.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cuda_learning;

void demonstrateMemoryCoalescing() {
    std::cout << "\n=== 内存合并访问演示 ===" << std::endl;

    const size_t dataSize = 32 * 1024 * 1024; // 32MB

    std::cout << "测试数据大小: " << dataSize / (1024*1024) << " MB" << std::endl;

    // 测试合并访问
    float coalescedBW = MemoryBandwidthTester::testGlobalMemoryBandwidth(dataSize, AccessPattern::COALESCED);
    std::cout << "合并访问带宽: " << std::fixed << std::setprecision(2) << coalescedBW << " GB/s" << std::endl;

    // 测试跨步访问
    float stridedBW = MemoryBandwidthTester::testGlobalMemoryBandwidth(dataSize, AccessPattern::STRIDED);
    std::cout << "跨步访问带宽: " << std::fixed << std::setprecision(2) << stridedBW << " GB/s" << std::endl;

    // 计算效率
    float efficiency = stridedBW / coalescedBW;
    std::cout << "跨步访问效率: " << (efficiency * 100) << "%" << std::endl;

    // 获取理论带宽
    float theoreticalBW = MemoryBandwidthTester::getTheoreticalBandwidth(0);
    std::cout << "理论峰值带宽: " << theoreticalBW << " GB/s" << std::endl;
    std::cout << "合并访问效率: " << (coalescedBW / theoreticalBW * 100) << "%" << std::endl;
}

void demonstrateSharedMemoryBankConflicts() {
    std::cout << "\n=== 共享内存银行冲突演示 ===" << std::endl;

    // 模拟不同的访问模式
    std::vector<int> conflictPattern = {0, 2, 4, 6, 8, 10, 12, 14}; // 2-way conflict
    std::vector<int> noConflictPattern = {0, 1, 2, 3, 4, 5, 6, 7};   // no conflict

    MemoryAccessAnalyzer analyzer;

    // 分析银行冲突
    int conflicts1 = analyzer.calculateBankConflicts(conflictPattern);
    int conflicts2 = analyzer.calculateBankConflicts(noConflictPattern);

    std::cout << "冲突访问模式银行冲突数: " << conflicts1 << std::endl;
    std::cout << "无冲突访问模式银行冲突数: " << conflicts2 << std::endl;

    // 可视化访问模式
    std::cout << "\n访问模式可视化:" << std::endl;
    std::cout << "冲突模式: " << analyzer.visualizeAccessPattern(conflictPattern) << std::endl;
    std::cout << "无冲突模式: " << analyzer.visualizeAccessPattern(noConflictPattern) << std::endl;
}

void demonstrateMemoryOptimization() {
    std::cout << "\n=== 内存优化工具演示 ===" << std::endl;

    MemoryOptimizer optimizer;

    // 测试内存对齐
    float* aligned_ptr;
    cudaMalloc(&aligned_ptr, 1024 * sizeof(float));

    bool isAligned = optimizer.checkMemoryAlignment(aligned_ptr, 128);
    std::cout << "内存对齐检查: " << (isAligned ? "已对齐" : "未对齐") << std::endl;

    // 建议最优块大小
    int optimalBlockSize = optimizer.suggestOptimalBlockSize(MemoryType::GLOBAL, 1024*1024);
    std::cout << "建议的最优块大小: " << optimalBlockSize << std::endl;

    // 计算共享内存使用量
    size_t sharedMemUsage = optimizer.calculateSharedMemoryUsage(256, sizeof(float), 4);
    std::cout << "共享内存使用量: " << sharedMemUsage << " 字节" << std::endl;

    // 建议内存填充
    size_t padding = optimizer.suggestMemoryPadding(1000, 128);
    std::cout << "建议的内存填充: " << padding << " 字节" << std::endl;

    cudaFree(aligned_ptr);
}

void runInteractiveMemoryComparison() {
    std::cout << "\n=== 交互式内存类型比较 ===" << std::endl;

    std::vector<size_t> testSizes = {
        1 * 1024 * 1024,   // 1MB
        16 * 1024 * 1024,  // 16MB
        64 * 1024 * 1024,  // 64MB
        256 * 1024 * 1024  // 256MB
    };

    std::cout << std::setw(12) << "数据大小"
              << std::setw(15) << "合并访问"
              << std::setw(15) << "跨步访问"
              << std::setw(12) << "效率比" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    for (size_t size : testSizes) {
        float coalescedBW = MemoryBandwidthTester::testGlobalMemoryBandwidth(size, AccessPattern::COALESCED);
        float stridedBW = MemoryBandwidthTester::testGlobalMemoryBandwidth(size, AccessPattern::STRIDED);
        float ratio = stridedBW / coalescedBW;

        std::cout << std::setw(10) << (size / (1024*1024)) << "MB"
                  << std::setw(12) << std::fixed << std::setprecision(2) << coalescedBW << " GB/s"
                  << std::setw(12) << stridedBW << " GB/s"
                  << std::setw(10) << (ratio * 100) << "%" << std::endl;
    }
}

void generateLearningReport() {
    std::cout << "\n=== 学习报告生成 ===" << std::endl;

    // 收集测试结果
    std::vector<MemoryBenchmarkResult> results = MemoryManagementTeacher::compareMemoryTypes(64 * 1024 * 1024);

    // 生成报告
    std::string report = MemoryManagementTeacher::generateOptimizationReport(results);
    std::cout << report << std::endl;

    // 添加学习要点
    std::cout << "\n=== 关键学习要点 ===" << std::endl;
    std::cout << "1. 内存合并访问是GPU性能优化的关键" << std::endl;
    std::cout << "2. 共享内存可以显著减少全局内存访问" << std::endl;
    std::cout << "3. 银行冲突会严重影响共享内存性能" << std::endl;
    std::cout << "4. 常量内存适合广播访问模式" << std::endl;
    std::cout << "5. 内存对齐可以提高访问效率" << std::endl;

    std::cout << "\n=== 实践建议 ===" << std::endl;
    std::cout << "1. 始终优先考虑内存访问模式" << std::endl;
    std::cout << "2. 使用CUDA Profiler分析内存瓶颈" << std::endl;
    std::cout << "3. 根据访问模式选择合适的内存类型" << std::endl;
    std::cout << "4. 测试不同的数据布局和块配置" << std::endl;
    std::cout << "5. 关注内存带宽利用率指标" << std::endl;
}

int main() {
    // 初始化CUDA
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "CUDA初始化失败: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "=== CUDA内存管理教学系统演示 ===" << std::endl;

    try {
        // 1. 基础内存类型演示
        MemoryManagementTeacher::demonstrateGlobalMemory();
        MemoryManagementTeacher::demonstrateSharedMemory();
        MemoryManagementTeacher::demonstrateConstantMemory();

        // 2. 内存合并访问演示
        demonstrateMemoryCoalescing();

        // 3. 共享内存银行冲突演示
        demonstrateSharedMemoryBankConflicts();

        // 4. 内存优化工具演示
        demonstrateMemoryOptimization();

        // 5. 交互式内存比较
        runInteractiveMemoryComparison();

        // 6. 生成学习报告
        generateLearningReport();

        // 7. 运行完整演示
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "运行完整内存管理演示..." << std::endl;
        MemoryDemo::runCompleteDemo();

    } catch (const std::exception& e) {
        std::cerr << "演示过程中发生错误: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n内存管理教学系统演示完成!" << std::endl;
    std::cout << "建议：使用CUDA Profiler (如Nsight Compute) 进行更详细的性能分析" << std::endl;

    return 0;
}
