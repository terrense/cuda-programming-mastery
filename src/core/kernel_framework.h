#ifndef KERNEL_FRAMEWORK_H
#define KERNEL_FRAMEWORK_H

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

namespace cuda_learning {

// 线程索引计算辅助结构
struct ThreadIndex {
    int globalX, globalY, globalZ;
    int localX, localY, localZ;
    int blockX, blockY, blockZ;
    int gridX, gridY, gridZ;

    ThreadIndex();
    void calculate();
    bool isValidThread(int maxX, int maxY = 1, int maxZ = 1) const;
};

// 网格和块配置结构
struct GridBlockConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMemSize;
    cudaStream_t stream;

    GridBlockConfig();
    GridBlockConfig(dim3 grid, dim3 block, size_t sharedMem = 0, cudaStream_t s = 0);

    // 计算总线程数
    int getTotalThreads() const;

    // 验证配置有效性
    bool isValid() const;

    // 获取配置信息字符串
    std::string toString() const;
};

// 核函数模板类型枚举
enum class KernelTemplate {
    VECTOR_ADD,
    MATRIX_MULTIPLY,
    REDUCTION,
    CONVOLUTION_2D,
    ELEMENT_WISE,
    CUSTOM
};

// 核函数模板生成器
class KernelTemplateGenerator {
public:
    // 生成指定类型的核函数模板
    static std::string generateTemplate(KernelTemplate type, const std::string& kernelName);

    // 生成向量加法模板
    static std::string generateVectorAddTemplate(const std::string& kernelName);

    // 生成矩阵乘法模板
    static std::string generateMatrixMultiplyTemplate(const std::string& kernelName);

    // 生成归约操作模板
    static std::string generateReductionTemplate(const std::string& kernelName);

    // 生成2D卷积模板
    static std::string generateConvolution2DTemplate(const std::string& kernelName);

    // 生成元素级操作模板
    static std::string generateElementWiseTemplate(const std::string& kernelName);

    // 生成自定义模板框架
    static std::string generateCustomTemplate(const std::string& kernelName);
};

// 网格和块配置优化器
class GridBlockOptimizer {
public:
    // 为1D数据计算最优配置
    static GridBlockConfig optimize1D(int dataSize, int deviceId = 0);

    // 为2D数据计算最优配置
    static GridBlockConfig optimize2D(int width, int height, int deviceId = 0);

    // 为3D数据计算最优配置
    static GridBlockConfig optimize3D(int width, int height, int depth, int deviceId = 0);

    // 为矩阵乘法计算最优配置
    static GridBlockConfig optimizeMatMul(int M, int N, int K, int deviceId = 0);

    // 为归约操作计算最优配置
    static GridBlockConfig optimizeReduction(int dataSize, int deviceId = 0);

    // 获取设备属性
    static cudaDeviceProp getDeviceProperties(int deviceId = 0);

    // 计算最大占用率的块大小
    static int calculateOptimalBlockSize(const void* kernel, int deviceId = 0);

private:
    // 内部辅助函数
    static int roundUp(int value, int multiple);
    static bool isPowerOfTwo(int value);
    static int nextPowerOfTwo(int value);
};

// 线程索引计算辅助函数
namespace ThreadUtils {
    // 获取当前线程的全局索引
    __device__ inline int getGlobalThreadId1D() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ inline int getGlobalThreadId2D(int width) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        return y * width + x;
    }

    __device__ inline int getGlobalThreadId3D(int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        return z * width * height + y * width + x;
    }

    // 检查线程是否在有效范围内
    __device__ inline bool isValidThread1D(int maxSize) {
        return getGlobalThreadId1D() < maxSize;
    }

    __device__ inline bool isValidThread2D(int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        return x < width && y < height;
    }

    __device__ inline bool isValidThread3D(int width, int height, int depth) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        return x < width && y < height && z < depth;
    }
}

// 核函数启动辅助类
class KernelLauncher {
public:
    template<typename... Args>
    static cudaError_t launch(void(*kernel)(Args...),
                             const GridBlockConfig& config,
                             Args... args) {
        kernel<<<config.gridDim, config.blockDim, config.sharedMemSize, config.stream>>>(args...);
        return cudaGetLastError();
    }

    // 带错误检查的启动
    template<typename... Args>
    static bool launchWithCheck(void(*kernel)(Args...),
                               const GridBlockConfig& config,
                               Args... args) {
        cudaError_t error = launch(kernel, config, args...);
        if (error != cudaSuccess) {
            return false;
        }

        error = cudaDeviceSynchronize();
        return error == cudaSuccess;
    }
};

} // namespace cuda_learning

#endif // KERNEL_FRAMEWORK_H
