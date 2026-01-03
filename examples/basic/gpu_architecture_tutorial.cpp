#include "cuda_environment.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

/**
 * GPU架构原理交互式教程
 *
 * 本程序演示GPU架构的核心概念：
 * 1. GPU vs CPU架构差异
 * 2. 流式多处理器(SM)结构
 * 3. 内存层次结构
 * 4. 计算能力和硬件限制
 */

namespace cuda_learning {

class GPUArchitectureTutorial {
public:
    GPUArchitectureTutorial() : m_cudaEnv() {}

    void runTutorial() {
        std::cout << "=== GPU架构原理交互式教程 ===" << std::endl;
        std::cout << "本教程将带您深入了解GPU架构的核心概念\n" << std::endl;

        // 1. GPU vs CPU 架构对比
        demonstrateGPUvsCPU();

        // 2. 流式多处理器(SM)结构分析
        analyzeSMArchitecture();

        // 3. 内存层次结构详解
        exploreMemoryHierarchy();

        // 4. 计算能力和硬件限制
        analyzeComputeCapability();

        std::cout << "\n=== 教程完成 ===" << std::endl;
        std::cout << "您已经了解了GPU架构的核心概念！" << std::endl;
    }

private:
    cuda_learning::CudaEnvironment m_cudaEnv;

    void demonstrateGPUvsCPU() {
        std::cout << "1. GPU vs CPU 架构对比" << std::endl;
        std::cout << "========================" << std::endl;

        std::cout << "CPU特点：" << std::endl;
        std::cout << "  • 少量核心(通常2-16个)，每个核心功能强大" << std::endl;
        std::cout << "  • 大容量缓存(L1/L2/L3)" << std::endl;
        std::cout << "  • 复杂的分支预测和乱序执行" << std::endl;
        std::cout << "  • 适合复杂的串行任务" << std::endl;

        std::cout << "\nGPU特点：" << std::endl;
        std::cout << "  • 大量简单核心(数百到数千个)" << std::endl;
        std::cout << "  • 小容量缓存，更多晶体管用于计算" << std::endl;
        std::cout << "  • SIMT(单指令多线程)执行模式" << std::endl;
        std::cout << "  • 适合大规模并行计算" << std::endl;

        // 获取当前GPU信息进行对比
        if (m_cudaEnv.isCudaAvailable()) {
            auto bestGPU = m_cudaEnv.getBestGPU();
            std::cout << "\n当前GPU信息：" << std::endl;
            std::cout << "  • GPU型号: " << bestGPU.name << std::endl;
            std::cout << "  • 流式多处理器数量: " << bestGPU.multiProcessorCount << std::endl;
            std::cout << "  • 每个SM最大线程数: " << bestGPU.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "  • 总计算核心数(估算): " << bestGPU.multiProcessorCount * 64 << " (假设每SM 64核)" << std::endl;
        }

        waitForUser();
    }

    void analyzeSMArchitecture() {
        std::cout << "\n2. 流式多处理器(SM)结构分析" << std::endl;
        std::cout << "==============================" << std::endl;

        std::cout << "SM是GPU的基本计算单元，包含：" << std::endl;
        std::cout << "  • CUDA核心：执行浮点和整数运算" << std::endl;
        std::cout << "  • 特殊功能单元(SFU)：执行超越函数" << std::endl;
        std::cout << "  • 共享内存：SM内线程间快速数据交换" << std::endl;
        std::cout << "  • 寄存器文件：每个线程的私有存储" << std::endl;
        std::cout << "  • 线程束调度器：管理32个线程为一组的执行" << std::endl;

        if (m_cudaEnv.isCudaAvailable()) {
            auto bestGPU = m_cudaEnv.getBestGPU();

            // 获取更详细的设备属性
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, bestGPU.deviceId);

            std::cout << "\n当前GPU的SM详细信息：" << std::endl;
            std::cout << "  • SM数量: " << prop.multiProcessorCount << std::endl;
            std::cout << "  • 每个SM最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "  • 每个SM最大线程块数: " << prop.maxBlocksPerMultiProcessor << std::endl;
            std::cout << "  • 共享内存大小: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
            std::cout << "  • 寄存器数量: " << prop.regsPerMultiprocessor << std::endl;
            std::cout << "  • 线程束大小: " << prop.warpSize << std::endl;

            // 计算理论峰值性能
            int coresPerSM = getCoresPerSM(prop.major, prop.minor);
            if (coresPerSM > 0) {
                std::cout << "  • 每个SM的CUDA核心数: " << coresPerSM << std::endl;
                std::cout << "  • 总CUDA核心数: " << prop.multiProcessorCount * coresPerSM << std::endl;
            }
        }

        waitForUser();
    }

    void exploreMemoryHierarchy() {
        std::cout << "\n3. 内存层次结构详解" << std::endl;
        std::cout << "=====================" << std::endl;

        std::cout << "GPU内存层次结构(从快到慢)：" << std::endl;
        std::cout << "  1. 寄存器 (Registers)" << std::endl;
        std::cout << "     • 每个线程私有，访问速度最快" << std::endl;
        std::cout << "     • 数量有限，过多使用会降低占用率" << std::endl;

        std::cout << "  2. 共享内存 (Shared Memory)" << std::endl;
        std::cout << "     • 同一线程块内线程共享" << std::endl;
        std::cout << "     • 可编程管理的缓存" << std::endl;
        std::cout << "     • 需要注意银行冲突" << std::endl;

        std::cout << "  3. L1缓存 (L1 Cache)" << std::endl;
        std::cout << "     • 硬件管理，缓存全局内存访问" << std::endl;
        std::cout << "     • 与共享内存共享同一片存储空间" << std::endl;

        std::cout << "  4. L2缓存 (L2 Cache)" << std::endl;
        std::cout << "     • 所有SM共享的二级缓存" << std::endl;
        std::cout << "     • 缓存全局内存和纹理内存" << std::endl;

        std::cout << "  5. 全局内存 (Global Memory)" << std::endl;
        std::cout << "     • 容量最大但延迟最高" << std::endl;
        std::cout << "     • 所有线程都可访问" << std::endl;
        std::cout << "     • 需要合并访问以获得最佳性能" << std::endl;

        if (m_cudaEnv.isCudaAvailable()) {
            auto bestGPU = m_cudaEnv.getBestGPU();
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, bestGPU.deviceId);

            std::cout << "\n当前GPU内存信息：" << std::endl;
            std::cout << "  • 全局内存总量: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
            std::cout << "  • 共享内存(每个SM): " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
            std::cout << "  • 共享内存(每个块): " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
            std::cout << "  • L2缓存大小: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
            std::cout << "  • 内存总线宽度: " << prop.memoryBusWidth << " bits" << std::endl;
            std::cout << "  • 内存时钟频率: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;

            // 计算理论内存带宽
            float memBandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
            std::cout << "  • 理论内存带宽: " << std::fixed << std::setprecision(1)
                      << memBandwidth << " GB/s" << std::endl;
        }

        waitForUser();
    }

    void analyzeComputeCapability() {
        std::cout << "\n4. 计算能力和硬件限制" << std::endl;
        std::cout << "========================" << std::endl;

        std::cout << "计算能力(Compute Capability)定义了GPU支持的功能：" << std::endl;
        std::cout << "  • 3.x: Kepler架构，支持动态并行" << std::endl;
        std::cout << "  • 5.x: Maxwell架构，改进的SM设计" << std::endl;
        std::cout << "  • 6.x: Pascal架构，统一内存和NVLink" << std::endl;
        std::cout << "  • 7.x: Volta/Turing架构，Tensor核心" << std::endl;
        std::cout << "  • 8.x: Ampere架构，第三代Tensor核心" << std::endl;

        if (m_cudaEnv.isCudaAvailable()) {
            auto bestGPU = m_cudaEnv.getBestGPU();
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, bestGPU.deviceId);

            std::cout << "\n当前GPU计算能力分析：" << std::endl;
            std::cout << "  • 计算能力: " << prop.major << "." << prop.minor << std::endl;

            // 根据计算能力提供架构信息
            std::string architecture = getArchitectureName(prop.major, prop.minor);
            std::cout << "  • 架构: " << architecture << std::endl;

            std::cout << "\n硬件限制：" << std::endl;
            std::cout << "  • 每个块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
            std::cout << "  • 每个块最大维度: (" << prop.maxThreadsDim[0]
                      << ", " << prop.maxThreadsDim[1]
                      << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
            std::cout << "  • 网格最大维度: (" << prop.maxGridSize[0]
                      << ", " << prop.maxGridSize[1]
                      << ", " << prop.maxGridSize[2] << ")" << std::endl;
            std::cout << "  • 每个块最大寄存器数: " << prop.regsPerBlock << std::endl;

            // 分析性能特征
            analyzePerformanceCharacteristics(prop);
        }

        waitForUser();
    }

    void analyzePerformanceCharacteristics(const cudaDeviceProp& prop) {
        std::cout << "\n性能特征分析：" << std::endl;

        // 计算理论峰值性能
        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        if (coresPerSM > 0) {
            int totalCores = prop.multiProcessorCount * coresPerSM;
            float peakGFLOPS = totalCores * prop.clockRate * 2 / 1.0e6f; // 假设FMA操作
            std::cout << "  • 理论峰值性能: " << std::fixed << std::setprecision(1)
                      << peakGFLOPS << " GFLOPS (单精度)" << std::endl;
        }

        // 分析占用率限制因素
        std::cout << "  • 占用率限制因素：" << std::endl;
        std::cout << "    - 寄存器使用量" << std::endl;
        std::cout << "    - 共享内存使用量" << std::endl;
        std::cout << "    - 线程块大小" << std::endl;
        std::cout << "    - 每个SM的最大线程块数: " << prop.maxBlocksPerMultiProcessor << std::endl;

        // 给出优化建议
        std::cout << "\n优化建议：" << std::endl;
        std::cout << "  • 选择合适的线程块大小(通常是32的倍数)" << std::endl;
        std::cout << "  • 最小化寄存器使用以提高占用率" << std::endl;
        std::cout << "  • 合理使用共享内存进行数据重用" << std::endl;
        std::cout << "  • 确保内存访问模式的合并" << std::endl;
    }

    int getCoresPerSM(int major, int minor) {
        // 根据计算能力返回每个SM的CUDA核心数
        if (major == 2) return 32;  // Fermi
        if (major == 3) return 192; // Kepler
        if (major == 5) return 128; // Maxwell
        if (major == 6) return (minor == 0) ? 64 : 128; // Pascal
        if (major == 7) return 64;  // Volta/Turing
        if (major == 8) return (minor == 0) ? 64 : 128; // Ampere
        return -1; // 未知架构
    }

    std::string getArchitectureName(int major, int minor) {
        if (major == 2) return "Fermi";
        if (major == 3) return "Kepler";
        if (major == 5) return "Maxwell";
        if (major == 6) return "Pascal";
        if (major == 7) return (minor == 0) ? "Volta" : "Turing";
        if (major == 8) return "Ampere";
        return "Unknown";
    }

    void waitForUser() {
        std::cout << "\n按回车键继续..." << std::endl;
        std::cin.get();
    }
};

} // namespace cuda_learning

int main() {
    // 初始化错误处理
    cuda_learning::ErrorHandler::initialize("gpu_architecture_tutorial.log");

    try {
        cuda_learning::GPUArchitectureTutorial tutorial;
        tutorial.runTutorial();
    } catch (const std::exception& e) {
        std::cerr << "教程执行出错: " << e.what() << std::endl;
        cuda_learning::ErrorHandler::shutdown();
        return 1;
    }

    cuda_learning::ErrorHandler::shutdown();
    return 0;
}
