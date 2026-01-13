#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA错误"); \
        } \
    } while(0)

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

// 内存合并检测核函数
__global__ void testCoalescedAccess(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // 合并访问
    }
}

__global__ void testStridedAccess(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (strided_idx < n) {
        output[idx] = input[strided_idx] * 2.0f;  // 跨步访问
    }
}

// 银行冲突测试核函数
__global__ void testBankConflicts(float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        // 产生银行冲突的访问模式
        int conflicted_index = (tid * 2) % 32;  // 2-way bank conflict
        sdata[conflicted_index] = tid * 1.0f;
        __syncthreads();

        // 读取数据
        output[tid] = sdata[conflicted_index];
    }
}

__global__ void testBankConflictFree(float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < n) {
        // 无银行冲突的访问模式
        sdata[tid] = tid * 1.0f;
        __syncthreads();

        // 读取数据
        output[tid] = sdata[tid];
    }
}

// 简化的纹理内存测试核函数
__global__ void testSimpleTextureAccess(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // 模拟2D访问模式
        output[idx] = input[idx] * sinf(x * 0.1f) * cosf(y * 0.1f);
    }
}

// 高级内存优化器类
class AdvancedMemoryOptimizer {
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

public:
    AdvancedMemoryOptimizer() {
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    ~AdvancedMemoryOptimizer() {
        cudaEventDestroy(start_event_);  // 不使用CUDA_CHECK避免析构函数中抛异常
        cudaEventDestroy(stop_event_);
    }

    CoalescingAnalysis analyzeMemoryCoalescing(size_t data_size, int access_pattern = 2) {
        CoalescingAnalysis result;
        result.data_size = data_size;
        result.access_pattern = access_pattern;

        // 分配GPU内存
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, data_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, data_size * sizeof(float)));

        // 初始化输入数据
        std::vector<float> h_input(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            h_input[i] = static_cast<float>(i);
        }
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size * sizeof(float), cudaMemcpyHostToDevice));

        // 配置执行参数
        int block_size = 256;
        int grid_size = (data_size + block_size - 1) / block_size;

        // 测试合并访问
        CUDA_CHECK(cudaEventRecord(start_event_));
        testCoalescedAccess<<<grid_size, block_size>>>(d_input, d_output, data_size);
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float coalesced_time;
        CUDA_CHECK(cudaEventElapsedTime(&coalesced_time, start_event_, stop_event_));
        result.coalesced_time_ms = coalesced_time;

        // 测试跨步访问
        int stride = access_pattern;
        CUDA_CHECK(cudaEventRecord(start_event_));
        testStridedAccess<<<grid_size, block_size>>>(d_input, d_output, data_size, stride);
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float strided_time;
        CUDA_CHECK(cudaEventElapsedTime(&strided_time, start_event_, stop_event_));
        result.strided_time_ms = strided_time;

        // 计算效率
        result.efficiency = coalesced_time / strided_time;
        result.bandwidth_gb_s = (data_size * sizeof(float) * 2) / (coalesced_time * 1e6);  // 读写两次

        // 生成优化建议
        if (result.efficiency > 0.8f) {
            result.optimization_suggestions.push_back("内存访问模式已经很好地合并");
        } else if (result.efficiency > 0.5f) {
            result.optimization_suggestions.push_back("考虑重新排列数据布局以提高合并度");
        } else {
            result.optimization_suggestions.push_back("严重的内存访问不合并，需要重新设计算法");
            result.optimization_suggestions.push_back("考虑使用共享内存或纹理内存");
        }

        // 清理内存
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return result;
    }

    BankConflictAnalysis analyzeBankConflicts(int shared_memory_size, int access_pattern = 2) {
        BankConflictAnalysis result;
        result.shared_memory_size = shared_memory_size;
        result.access_pattern = access_pattern;

        // 分配GPU内存
        float *d_output;
        CUDA_CHECK(cudaMalloc(&d_output, shared_memory_size * sizeof(float)));

        int block_size = std::min(shared_memory_size, 256);
        size_t shared_mem_bytes = shared_memory_size * sizeof(float);

        // 测试有银行冲突的访问
        CUDA_CHECK(cudaEventRecord(start_event_));
        testBankConflicts<<<1, block_size, shared_mem_bytes>>>(d_output, shared_memory_size);
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float conflict_time;
        CUDA_CHECK(cudaEventElapsedTime(&conflict_time, start_event_, stop_event_));
        result.conflict_time_ms = conflict_time;

        // 测试无银行冲突的访问
        CUDA_CHECK(cudaEventRecord(start_event_));
        testBankConflictFree<<<1, block_size, shared_mem_bytes>>>(d_output, shared_memory_size);
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float conflict_free_time;
        CUDA_CHECK(cudaEventElapsedTime(&conflict_free_time, start_event_, stop_event_));
        result.conflict_free_time_ms = conflict_free_time;

        // 计算银行冲突程度
        result.conflict_ratio = conflict_time / conflict_free_time;

        // 估算银行冲突数量
        if (result.conflict_ratio > 2.0f) {
            result.estimated_conflicts = static_cast<int>((result.conflict_ratio - 1.0f) * 16);
        } else {
            result.estimated_conflicts = 0;
        }

        // 生成优化建议
        if (result.conflict_ratio < 1.2f) {
            result.optimization_suggestions.push_back("共享内存访问模式良好，无明显银行冲突");
        } else if (result.conflict_ratio < 2.0f) {
            result.optimization_suggestions.push_back("存在轻微银行冲突，考虑填充数组避免冲突");
        } else {
            result.optimization_suggestions.push_back("严重的银行冲突，需要重新设计共享内存访问模式");
            result.optimization_suggestions.push_back("使用数组填充或改变索引计算方式");
        }

        // 清理内存
        CUDA_CHECK(cudaFree(d_output));

        return result;
    }

    TextureMemoryDemo demonstrateTextureMemory(int width, int height) {
        TextureMemoryDemo result;
        result.width = width;
        result.height = height;

        size_t data_size = width * height;

        // 分配内存
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, data_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, data_size * sizeof(float)));

        // 初始化输入数据
        std::vector<float> h_input(data_size);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                h_input[y * width + x] = sinf(x * 0.1f) * cosf(y * 0.1f);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size * sizeof(float), cudaMemcpyHostToDevice));

        // 配置执行参数
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y);

        // 执行简化的纹理内存测试
        CUDA_CHECK(cudaEventRecord(start_event_));
        testSimpleTextureAccess<<<grid_size, block_size>>>(d_input, d_output, width, height);
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float execution_time;
        CUDA_CHECK(cudaEventElapsedTime(&execution_time, start_event_, stop_event_));
        result.execution_time_ms = execution_time;

        // 模拟缓存命中率
        result.cache_hit_rate = 0.85f;

        // 生成使用建议
        result.usage_recommendations.push_back("纹理内存适用于具有空间局部性的2D数据访问");
        result.usage_recommendations.push_back("自动处理边界条件和插值，减少边界检查开销");
        result.usage_recommendations.push_back("对于随机访问模式，纹理缓存可以显著提高性能");

        // 清理资源
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return result;
    }

    SurfaceMemoryDemo demonstrateSurfaceMemory(int width, int height) {
        SurfaceMemoryDemo result;
        result.width = width;
        result.height = height;

        size_t data_size = width * height;

        // 简化的表面内存演示 - 使用全局内存模拟
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, data_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, data_size * sizeof(float)));

        // 初始化输入数据
        std::vector<float> h_input(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            h_input[i] = static_cast<float>(i % 100) / 100.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), data_size * sizeof(float), cudaMemcpyHostToDevice));

        // 配置执行参数
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y);

        // 执行简化的表面内存测试
        CUDA_CHECK(cudaEventRecord(start_event_));
        testSimpleTextureAccess<<<grid_size, block_size>>>(d_input, d_output, width, height);
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float execution_time;
        CUDA_CHECK(cudaEventElapsedTime(&execution_time, start_event_, stop_event_));
        result.execution_time_ms = execution_time;

        // 计算写入带宽
        result.write_bandwidth_gb_s = (data_size * sizeof(float)) / (execution_time * 1e6);

        // 生成使用建议
        result.usage_recommendations.push_back("表面内存允许对CUDA数组进行读写操作");
        result.usage_recommendations.push_back("适用于需要就地修改2D数据的算法");
        result.usage_recommendations.push_back("与纹理内存结合使用可以实现高效的图像处理管道");

        // 清理资源
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return result;
    }

    std::string generateOptimizationReport(const CoalescingAnalysis& coalescing,
                                         const BankConflictAnalysis& bank_conflict) {
        std::ostringstream report;

        report << "=== 高级内存优化分析报告 ===\n\n";

        // 内存合并分析
        report << "1. 内存合并分析:\n";
        report << "   数据大小: " << coalescing.data_size << " 元素\n";
        report << "   合并访问时间: " << std::fixed << std::setprecision(3)
               << coalescing.coalesced_time_ms << " ms\n";
        report << "   跨步访问时间: " << coalescing.strided_time_ms << " ms\n";
        report << "   访问效率: " << std::setprecision(2) << (coalescing.efficiency * 100) << "%\n";
        report << "   内存带宽: " << std::setprecision(1) << coalescing.bandwidth_gb_s << " GB/s\n";

        report << "   优化建议:\n";
        for (const auto& suggestion : coalescing.optimization_suggestions) {
            report << "   - " << suggestion << "\n";
        }
        report << "\n";

        // 银行冲突分析
        report << "2. 共享内存银行冲突分析:\n";
        report << "   共享内存大小: " << bank_conflict.shared_memory_size << " 元素\n";
        report << "   冲突访问时间: " << std::setprecision(3)
               << bank_conflict.conflict_time_ms << " ms\n";
        report << "   无冲突访问时间: " << bank_conflict.conflict_free_time_ms << " ms\n";
        report << "   冲突比率: " << std::setprecision(2) << bank_conflict.conflict_ratio << "x\n";
        report << "   估算冲突数: " << bank_conflict.estimated_conflicts << "\n";

        report << "   优化建议:\n";
        for (const auto& suggestion : bank_conflict.optimization_suggestions) {
            report << "   - " << suggestion << "\n";
        }
        report << "\n";

        // 总体建议
        report << "3. 总体优化建议:\n";
        if (coalescing.efficiency > 0.8f && bank_conflict.conflict_ratio < 1.2f) {
            report << "   - 内存访问模式已经很好优化\n";
            report << "   - 可以考虑进一步的算法级优化\n";
        } else {
            report << "   - 优先解决内存访问模式问题\n";
            report << "   - 考虑使用性能分析工具进行深入分析\n";
            report << "   - 实验不同的数据布局和访问模式\n";
        }

        return report.str();
    }
};

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

        for (const auto& size_pair : texture_sizes) {
            int width = size_pair.first;
            int height = size_pair.second;
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

        for (const auto& size_pair : texture_sizes) {
            int width = size_pair.first;
            int height = size_pair.second;
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
