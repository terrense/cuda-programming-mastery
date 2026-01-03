#include "kernel_framework.h"
#include "error_handler.h"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace cuda_learning {

// ThreadIndex 实现
ThreadIndex::ThreadIndex() : globalX(0), globalY(0), globalZ(0),
                            localX(0), localY(0), localZ(0),
                            blockX(0), blockY(0), blockZ(0),
                            gridX(0), gridY(0), gridZ(0) {}

void ThreadIndex::calculate() {
    // 注意：这个函数只能在主机端调用，用于教学演示
    // 实际的线程索引计算应该在设备端进行
    localX = threadIdx.x;
    localY = threadIdx.y;
    localZ = threadIdx.z;

    blockX = blockIdx.x;
    blockY = blockIdx.y;
    blockZ = blockIdx.z;

    gridX = gridDim.x;
    gridY = gridDim.y;
    gridZ = gridDim.z;

    globalX = blockIdx.x * blockDim.x + threadIdx.x;
    globalY = blockIdx.y * blockDim.y + threadIdx.y;
    globalZ = blockIdx.z * blockDim.z + threadIdx.z;
}

bool ThreadIndex::isValidThread(int maxX, int maxY, int maxZ) const {
    return globalX < maxX && globalY < maxY && globalZ < maxZ;
}

// GridBlockConfig 实现
GridBlockConfig::GridBlockConfig() : gridDim(1, 1, 1), blockDim(1, 1, 1),
                                    sharedMemSize(0), stream(0) {}

GridBlockConfig::GridBlockConfig(dim3 grid, dim3 block, size_t sharedMem, cudaStream_t s)
    : gridDim(grid), blockDim(block), sharedMemSize(sharedMem), stream(s) {}

int GridBlockConfig::getTotalThreads() const {
    return gridDim.x * gridDim.y * gridDim.z *
           blockDim.x * blockDim.y * blockDim.z;
}

bool GridBlockConfig::isValid() const {
    // 检查块维度限制
    if (blockDim.x > 1024 || blockDim.y > 1024 || blockDim.z > 64) {
        return false;
    }

    // 检查总线程数限制
    int totalBlockThreads = blockDim.x * blockDim.y * blockDim.z;
    if (totalBlockThreads > 1024) {
        return false;
    }

    // 检查网格维度限制
    if (gridDim.x > 65535 || gridDim.y > 65535 || gridDim.z > 65535) {
        return false;
    }

    return true;
}

std::string GridBlockConfig::toString() const {
    std::stringstream ss;
    ss << "Grid: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << "), ";
    ss << "Block: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << "), ";
    ss << "SharedMem: " << sharedMemSize << " bytes, ";
    ss << "TotalThreads: " << getTotalThreads();
    return ss.str();
}

// KernelTemplateGenerator 实现
std::string KernelTemplateGenerator::generateTemplate(KernelTemplate type, const std::string& kernelName) {
    switch (type) {
        case KernelTemplate::VECTOR_ADD:
            return generateVectorAddTemplate(kernelName);
        case KernelTemplate::MATRIX_MULTIPLY:
            return generateMatrixMultiplyTemplate(kernelName);
        case KernelTemplate::REDUCTION:
            return generateReductionTemplate(kernelName);
        case KernelTemplate::CONVOLUTION_2D:
            return generateConvolution2DTemplate(kernelName);
        case KernelTemplate::ELEMENT_WISE:
            return generateElementWiseTemplate(kernelName);
        case KernelTemplate::CUSTOM:
            return generateCustomTemplate(kernelName);
        default:
            return generateCustomTemplate(kernelName);
    }
}

std::string KernelTemplateGenerator::generateVectorAddTemplate(const std::string& kernelName) {
    std::stringstream ss;
    ss << "// 向量加法核函数模板\n";
    ss << "__global__ void " << kernelName << "(float* a, float* b, float* c, int n) {\n";
    ss << "    // 计算全局线程索引\n";
    ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    \n";
    ss << "    // 边界检查\n";
    ss << "    if (idx < n) {\n";
    ss << "        c[idx] = a[idx] + b[idx];\n";
    ss << "    }\n";
    ss << "}\n\n";
    ss << "// 主机端调用示例\n";
    ss << "void launch_" << kernelName << "(float* d_a, float* d_b, float* d_c, int n) {\n";
    ss << "    int blockSize = 256;\n";
    ss << "    int gridSize = (n + blockSize - 1) / blockSize;\n";
    ss << "    " << kernelName << "<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);\n";
    ss << "    cudaDeviceSynchronize();\n";
    ss << "}\n";
    return ss.str();
}

std::string KernelTemplateGenerator::generateMatrixMultiplyTemplate(const std::string& kernelName) {
    std::stringstream ss;
    ss << "// 矩阵乘法核函数模板\n";
    ss << "__global__ void " << kernelName << "(float* A, float* B, float* C, int M, int N, int K) {\n";
    ss << "    // 计算输出矩阵的行和列索引\n";
    ss << "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n";
    ss << "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    \n";
    ss << "    // 边界检查\n";
    ss << "    if (row < M && col < N) {\n";
    ss << "        float sum = 0.0f;\n";
    ss << "        \n";
    ss << "        // 计算点积\n";
    ss << "        for (int k = 0; k < K; k++) {\n";
    ss << "            sum += A[row * K + k] * B[k * N + col];\n";
    ss << "        }\n";
    ss << "        \n";
    ss << "        C[row * N + col] = sum;\n";
    ss << "    }\n";
    ss << "}\n\n";
    ss << "// 主机端调用示例\n";
    ss << "void launch_" << kernelName << "(float* d_A, float* d_B, float* d_C, int M, int N, int K) {\n";
    ss << "    dim3 blockSize(16, 16);\n";
    ss << "    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);\n";
    ss << "    " << kernelName << "<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);\n";
    ss << "    cudaDeviceSynchronize();\n";
    ss << "}\n";
    return ss.str();
}

std::string KernelTemplateGenerator::generateReductionTemplate(const std::string& kernelName) {
    std::stringstream ss;
    ss << "// 归约操作核函数模板（求和）\n";
    ss << "__global__ void " << kernelName << "(float* input, float* output, int n) {\n";
    ss << "    extern __shared__ float sdata[];\n";
    ss << "    \n";
    ss << "    // 计算线程索引\n";
    ss << "    int tid = threadIdx.x;\n";
    ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    \n";
    ss << "    // 加载数据到共享内存\n";
    ss << "    sdata[tid] = (idx < n) ? input[idx] : 0.0f;\n";
    ss << "    __syncthreads();\n";
    ss << "    \n";
    ss << "    // 树形归约\n";
    ss << "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n";
    ss << "        if (tid < s) {\n";
    ss << "            sdata[tid] += sdata[tid + s];\n";
    ss << "        }\n";
    ss << "        __syncthreads();\n";
    ss << "    }\n";
    ss << "    \n";
    ss << "    // 写回结果\n";
    ss << "    if (tid == 0) {\n";
    ss << "        output[blockIdx.x] = sdata[0];\n";
    ss << "    }\n";
    ss << "}\n\n";
    ss << "// 主机端调用示例\n";
    ss << "void launch_" << kernelName << "(float* d_input, float* d_output, int n) {\n";
    ss << "    int blockSize = 256;\n";
    ss << "    int gridSize = (n + blockSize - 1) / blockSize;\n";
    ss << "    size_t sharedMemSize = blockSize * sizeof(float);\n";
    ss << "    " << kernelName << "<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);\n";
    ss << "    cudaDeviceSynchronize();\n";
    ss << "}\n";
    return ss.str();
}

std::string KernelTemplateGenerator::generateConvolution2DTemplate(const std::string& kernelName) {
    std::stringstream ss;
    ss << "// 2D卷积核函数模板\n";
    ss << "__global__ void " << kernelName << "(float* input, float* kernel, float* output,\n";
    ss << "                                    int inputWidth, int inputHeight,\n";
    ss << "                                    int kernelSize, int outputWidth, int outputHeight) {\n";
    ss << "    // 计算输出位置\n";
    ss << "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n";
    ss << "    \n";
    ss << "    // 边界检查\n";
    ss << "    if (col < outputWidth && row < outputHeight) {\n";
    ss << "        float sum = 0.0f;\n";
    ss << "        int halfKernel = kernelSize / 2;\n";
    ss << "        \n";
    ss << "        // 卷积计算\n";
    ss << "        for (int ky = 0; ky < kernelSize; ky++) {\n";
    ss << "            for (int kx = 0; kx < kernelSize; kx++) {\n";
    ss << "                int inputRow = row + ky - halfKernel;\n";
    ss << "                int inputCol = col + kx - halfKernel;\n";
    ss << "                \n";
    ss << "                // 边界处理（零填充）\n";
    ss << "                if (inputRow >= 0 && inputRow < inputHeight &&\n";
    ss << "                    inputCol >= 0 && inputCol < inputWidth) {\n";
    ss << "                    float inputVal = input[inputRow * inputWidth + inputCol];\n";
    ss << "                    float kernelVal = kernel[ky * kernelSize + kx];\n";
    ss << "                    sum += inputVal * kernelVal;\n";
    ss << "                }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "        \n";
    ss << "        output[row * outputWidth + col] = sum;\n";
    ss << "    }\n";
    ss << "}\n\n";
    ss << "// 主机端调用示例\n";
    ss << "void launch_" << kernelName << "(float* d_input, float* d_kernel, float* d_output,\n";
    ss << "                                int inputWidth, int inputHeight, int kernelSize,\n";
    ss << "                                int outputWidth, int outputHeight) {\n";
    ss << "    dim3 blockSize(16, 16);\n";
    ss << "    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,\n";
    ss << "                  (outputHeight + blockSize.y - 1) / blockSize.y);\n";
    ss << "    " << kernelName << "<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,\n";
    ss << "                                                  inputWidth, inputHeight, kernelSize,\n";
    ss << "                                                  outputWidth, outputHeight);\n";
    ss << "    cudaDeviceSynchronize();\n";
    ss << "}\n";
    return ss.str();
}

std::string KernelTemplateGenerator::generateElementWiseTemplate(const std::string& kernelName) {
    std::stringstream ss;
    ss << "// 元素级操作核函数模板\n";
    ss << "__global__ void " << kernelName << "(float* input, float* output, int n) {\n";
    ss << "    // 计算全局线程索引\n";
    ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    \n";
    ss << "    // 边界检查\n";
    ss << "    if (idx < n) {\n";
    ss << "        // TODO: 在此处实现具体的元素级操作\n";
    ss << "        // 示例：ReLU激活函数\n";
    ss << "        output[idx] = fmaxf(0.0f, input[idx]);\n";
    ss << "    }\n";
    ss << "}\n\n";
    ss << "// 主机端调用示例\n";
    ss << "void launch_" << kernelName << "(float* d_input, float* d_output, int n) {\n";
    ss << "    int blockSize = 256;\n";
    ss << "    int gridSize = (n + blockSize - 1) / blockSize;\n";
    ss << "    " << kernelName << "<<<gridSize, blockSize>>>(d_input, d_output, n);\n";
    ss << "    cudaDeviceSynchronize();\n";
    ss << "}\n";
    return ss.str();
}

std::string KernelTemplateGenerator::generateCustomTemplate(const std::string& kernelName) {
    std::stringstream ss;
    ss << "// 自定义核函数模板\n";
    ss << "__global__ void " << kernelName << "(/* 添加您的参数 */) {\n";
    ss << "    // 计算线程索引\n";
    ss << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    int idy = blockIdx.y * blockDim.y + threadIdx.y;\n";
    ss << "    int idz = blockIdx.z * blockDim.z + threadIdx.z;\n";
    ss << "    \n";
    ss << "    // 局部线程索引\n";
    ss << "    int tid_x = threadIdx.x;\n";
    ss << "    int tid_y = threadIdx.y;\n";
    ss << "    int tid_z = threadIdx.z;\n";
    ss << "    \n";
    ss << "    // TODO: 添加边界检查\n";
    ss << "    // if (idx < maxX && idy < maxY && idz < maxZ) {\n";
    ss << "    \n";
    ss << "    // TODO: 实现您的核函数逻辑\n";
    ss << "    \n";
    ss << "    // }\n";
    ss << "}\n\n";
    ss << "// 主机端调用示例\n";
    ss << "void launch_" << kernelName << "(/* 添加您的参数 */) {\n";
    ss << "    // TODO: 配置网格和块维度\n";
    ss << "    dim3 blockSize(/* 设置块大小 */);\n";
    ss << "    dim3 gridSize(/* 设置网格大小 */);\n";
    ss << "    \n";
    ss << "    // TODO: 启动核函数\n";
    ss << "    " << kernelName << "<<<gridSize, blockSize>>>(/* 传递参数 */);\n";
    ss << "    \n";
    ss << "    // 同步等待完成\n";
    ss << "    cudaDeviceSynchronize();\n";
    ss << "}\n";
    return ss.str();
}

// GridBlockOptimizer 实现
GridBlockConfig GridBlockOptimizer::optimize1D(int dataSize, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);

    // 选择合适的块大小（通常是32的倍数，最大1024）
    int blockSize = std::min(1024, std::max(32, roundUp(dataSize / prop.multiProcessorCount, 32)));
    blockSize = std::min(blockSize, prop.maxThreadsPerBlock);

    // 计算网格大小
    int gridSize = (dataSize + blockSize - 1) / blockSize;
    gridSize = std::min(gridSize, prop.maxGridSize[0]);

    return GridBlockConfig(dim3(gridSize), dim3(blockSize));
}

GridBlockConfig GridBlockOptimizer::optimize2D(int width, int height, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);

    // 选择16x16或32x32的块大小
    int blockX = 16, blockY = 16;
    if (width >= 32 && height >= 32) {
        blockX = blockY = 32;
    }

    // 确保不超过设备限制
    while (blockX * blockY > prop.maxThreadsPerBlock) {
        if (blockX > blockY) blockX /= 2;
        else blockY /= 2;
    }

    // 计算网格大小
    int gridX = (width + blockX - 1) / blockX;
    int gridY = (height + blockY - 1) / blockY;

    gridX = std::min(gridX, prop.maxGridSize[0]);
    gridY = std::min(gridY, prop.maxGridSize[1]);

    return GridBlockConfig(dim3(gridX, gridY), dim3(blockX, blockY));
}

GridBlockConfig GridBlockOptimizer::optimize3D(int width, int height, int depth, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);

    // 选择8x8x8的块大小作为起点
    int blockX = 8, blockY = 8, blockZ = 8;

    // 调整以适应设备限制
    while (blockX * blockY * blockZ > prop.maxThreadsPerBlock) {
        if (blockZ > 1) blockZ /= 2;
        else if (blockY > blockX) blockY /= 2;
        else blockX /= 2;
    }

    // 计算网格大小
    int gridX = (width + blockX - 1) / blockX;
    int gridY = (height + blockY - 1) / blockY;
    int gridZ = (depth + blockZ - 1) / blockZ;

    gridX = std::min(gridX, prop.maxGridSize[0]);
    gridY = std::min(gridY, prop.maxGridSize[1]);
    gridZ = std::min(gridZ, prop.maxGridSize[2]);

    return GridBlockConfig(dim3(gridX, gridY, gridZ), dim3(blockX, blockY, blockZ));
}

GridBlockConfig GridBlockOptimizer::optimizeMatMul(int M, int N, int K, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);

    // 矩阵乘法通常使用16x16或32x32的块
    int blockSize = 16;
    if (M >= 32 && N >= 32 && prop.maxThreadsPerBlock >= 1024) {
        blockSize = 32;
    }

    int gridX = (N + blockSize - 1) / blockSize;
    int gridY = (M + blockSize - 1) / blockSize;

    gridX = std::min(gridX, prop.maxGridSize[0]);
    gridY = std::min(gridY, prop.maxGridSize[1]);

    return GridBlockConfig(dim3(gridX, gridY), dim3(blockSize, blockSize));
}

GridBlockConfig GridBlockOptimizer::optimizeReduction(int dataSize, int deviceId) {
    cudaDeviceProp prop = getDeviceProperties(deviceId);

    // 归约操作通常使用较大的块大小
    int blockSize = std::min(1024, nextPowerOfTwo(dataSize / prop.multiProcessorCount));
    blockSize = std::max(blockSize, 64); // 最小64个线程

    int gridSize = (dataSize + blockSize - 1) / blockSize;
    gridSize = std::min(gridSize, prop.maxGridSize[0]);

    size_t sharedMemSize = blockSize * sizeof(float);

    return GridBlockConfig(dim3(gridSize), dim3(blockSize), sharedMemSize);
}

cudaDeviceProp GridBlockOptimizer::getDeviceProperties(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    return prop;
}

int GridBlockOptimizer::calculateOptimalBlockSize(const void* kernel, int deviceId) {
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
    return blockSize;
}

int GridBlockOptimizer::roundUp(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

bool GridBlockOptimizer::isPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

int GridBlockOptimizer::nextPowerOfTwo(int value) {
    if (value <= 1) return 1;

    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

} // namespace cuda_learning
