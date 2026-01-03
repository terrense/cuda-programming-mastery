/**
 * 简单矩阵运算的CUDA实现
 * 
 * 本程序展示基础的矩阵运算，包括：
 * 1. 矩阵加法
 * 2. 矩阵乘法（朴素实现）
 * 3. 矩阵转置
 * 4. 矩阵-向量乘法
 * 
 * 学习目标：
 * - 理解2D线程索引在矩阵运算中的应用
 * - 掌握矩阵运算的并行化策略
 * - 学会处理2D数据结构
 * - 了解内存访问模式对性能的影响
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * 矩阵加法核函数
 * C = A + B
 */
__global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int cols) {
    // 计算2D线程索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查边界
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * 矩阵乘法核函数（朴素实现）
 * C = A * B
 * A: M x K, B: K x N, C: M x N
 */
__global__ void matrixMulNaive(const float* A, const float* B, float* C, 
                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * 矩阵转置核函数
 * B = A^T
 */
__global__ void matrixTranspose(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // A[row][col] -> B[col][row]
        B[col * rows + row] = A[row * cols + col];
    }
}

/**
 * 优化的矩阵转置（使用共享内存）
 */
#define TILE_SIZE 16
__global__ void matrixTransposeOptimized(const float* A, float* B, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1避免银行冲突
    
    int blockRow = blockIdx.y * TILE_SIZE;
    int blockCol = blockIdx.x * TILE_SIZE;
    
    int row = blockRow + threadIdx.y;
    int col = blockCol + threadIdx.x;
    
    // 加载数据到共享内存
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = A[row * cols + col];
    }
    
    __syncthreads();
    
    // 写入转置后的位置
    row = blockCol + threadIdx.y;
    col = blockRow + threadIdx.x;
    
    if (row < cols && col < rows) {
        B[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * 矩阵-向量乘法核函数
 * y = A * x
 * A: M x N, x: N x 1, y: M x 1
 */
__global__ void matrixVectorMul(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

class MatrixOperationsDemo {
public:
    MatrixOperationsDemo(int rows, int cols) : m_rows(rows), m_cols(cols) {
        size_t matrixSize = rows * cols * sizeof(float);
        size_t vectorSize = std::max(rows, cols) * sizeof(float);
        
        // 分配主机内存
        h_A.resize(rows * cols);
        h_B.resize(rows * cols);
        h_C.resize(rows * cols);
        h_x.resize(cols);
        h_y.resize(rows);
        
        // 初始化数据
        initializeData();
        
        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_A, matrixSize));
        CUDA_CHECK(cudaMalloc(&d_B, matrixSize));
        CUDA_CHECK(cudaMalloc(&d_C, matrixSize));
        CUDA_CHECK(cudaMalloc(&d_x, vectorSize));
        CUDA_CHECK(cudaMalloc(&d_y, vectorSize));
        
        // 复制数据到设备
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrixSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrixSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    ~MatrixOperationsDemo() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
    }
    
    void runAllTests() {
        std::cout << "=== 矩阵运算CUDA实现演示 ===" << std::endl;
        std::cout << "矩阵大小: " << m_rows << " x " << m_cols << std::endl;
        std::cout << "内存使用: " << (2 * m_rows * m_cols * sizeof(float) / 1024.0 / 1024.0) 
                  << " MB\n" << std::endl;
        
        // 测试各种矩阵运算
        testMatrixAddition();
        testMatrixMultiplication();
        testMatrixTranspose();
        testMatrixVectorMultiplication();
        
        // 性能分析
        performanceAnalysis();
    }

private:
    int m_rows, m_cols;
    std::vector<float> h_A, h_B, h_C, h_x, h_y;
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *d_x = nullptr, *d_y = nullptr;
    
    void initializeData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // 初始化矩阵A和B
        for (int i = 0; i < m_rows * m_cols; ++i) {
            h_A[i] = dis(gen);
            h_B[i] = dis(gen);
        }
        
        // 初始化向量x
        for (int i = 0; i < m_cols; ++i) {
            h_x[i] = dis(gen);
        }
    }
    
    void testMatrixAddition() {
        std::cout << "--- 测试矩阵加法 ---" << std::endl;
        
        // 配置执行参数
        dim3 blockSize(16, 16);
        dim3 gridSize((m_cols + blockSize.x - 1) / blockSize.x,
                      (m_rows + blockSize.y - 1) / blockSize.y);
        
        std::cout << "执行配置: <<<(" << gridSize.x << "," << gridSize.y 
                  << "), (" << blockSize.x << "," << blockSize.y << ")>>>" << std::endl;
        
        // 创建计时事件
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // 执行核函数
        CUDA_CHECK(cudaEventRecord(start));
        matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, m_rows, m_cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        // 计算执行时间
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // 验证结果
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m_rows * m_cols * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        bool correct = verifyMatrixAddition();
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        std::cout << "结果正确性: " << (correct ? "✓ 正确" : "✗ 错误") << std::endl;
        
        // 显示部分结果
        if (m_rows <= 8 && m_cols <= 8) {
            printMatrix("结果矩阵C (A + B):", h_C.data(), m_rows, m_cols);
        }
        
        std::cout << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void testMatrixMultiplication() {
        std::cout << "--- 测试矩阵乘法 (A * A^T) ---" << std::endl;
        
        // 为了演示，我们计算 A * A^T (结果是 m_rows x m_rows)
        int resultRows = m_rows;
        int resultCols = m_rows;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((resultCols + blockSize.x - 1) / blockSize.x,
                      (resultRows + blockSize.y - 1) / blockSize.y);
        
        std::cout << "计算 A(" << m_rows << "x" << m_cols << ") * A^T(" 
                  << m_cols << "x" << m_rows << ")" << std::endl;
        std::cout << "执行配置: <<<(" << gridSize.x << "," << gridSize.y 
                  << "), (" << blockSize.x << "," << blockSize.y << ")>>>" << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // 首先计算A的转置
        matrixTranspose<<<gridSize, blockSize>>>(d_A, d_B, m_rows, m_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 然后计算A * A^T
        CUDA_CHECK(cudaEventRecord(start));
        matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, m_rows, m_cols, m_rows);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        
        // 计算FLOPS
        long long flops = 2LL * m_rows * m_rows * m_cols; // 每个元素需要m_cols次乘加
        float gflops = (flops / 1e9f) / (milliseconds / 1000.0f);
        std::cout << "性能: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
        
        // 显示部分结果
        if (m_rows <= 6) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, resultRows * resultCols * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
            printMatrix("结果矩阵C (A * A^T):", h_C.data(), resultRows, resultCols);
        }
        
        std::cout << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void testMatrixTranspose() {
        std::cout << "--- 测试矩阵转置 ---" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((m_cols + blockSize.x - 1) / blockSize.x,
                      (m_rows + blockSize.y - 1) / blockSize.y);
        
        std::cout << "朴素版本:" << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // 测试朴素版本
        CUDA_CHECK(cudaEventRecord(start));
        matrixTranspose<<<gridSize, blockSize>>>(d_A, d_C, m_rows, m_cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds1 = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds1, start, stop));
        
        std::cout << "  执行时间: " << milliseconds1 << " ms" << std::endl;
        
        // 测试优化版本（如果矩阵足够大）
        if (m_rows >= TILE_SIZE && m_cols >= TILE_SIZE) {
            std::cout << "优化版本 (共享内存):" << std::endl;
            
            dim3 optimizedBlockSize(TILE_SIZE, TILE_SIZE);
            dim3 optimizedGridSize((m_cols + TILE_SIZE - 1) / TILE_SIZE,
                                   (m_rows + TILE_SIZE - 1) / TILE_SIZE);
            
            CUDA_CHECK(cudaEventRecord(start));
            matrixTransposeOptimized<<<optimizedGridSize, optimizedBlockSize>>>(d_A, d_B, m_rows, m_cols);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            
            float milliseconds2 = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds2, start, stop));
            
            std::cout << "  执行时间: " << milliseconds2 << " ms" << std::endl;
            std::cout << "  加速比: " << std::fixed << std::setprecision(2) 
                      << milliseconds1 / milliseconds2 << "x" << std::endl;
        }
        
        // 验证结果
        if (m_rows <= 8 && m_cols <= 8) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, m_rows * m_cols * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
            printMatrix("原矩阵A:", h_A.data(), m_rows, m_cols);
            printMatrix("转置矩阵A^T:", h_C.data(), m_cols, m_rows);
        }
        
        std::cout << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void testMatrixVectorMultiplication() {
        std::cout << "--- 测试矩阵-向量乘法 ---" << std::endl;
        
        int blockSize = 256;
        int gridSize = (m_rows + blockSize - 1) / blockSize;
        
        std::cout << "计算 A(" << m_rows << "x" << m_cols << ") * x(" << m_cols << "x1)" << std::endl;
        std::cout << "执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        matrixVectorMul<<<gridSize, blockSize>>>(d_A, d_x, d_y, m_rows, m_cols);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
        
        // 复制结果并显示
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, m_rows * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        if (m_rows <= 10 && m_cols <= 10) {
            printMatrix("矩阵A:", h_A.data(), m_rows, m_cols);
            printVector("向量x:", h_x.data(), m_cols);
            printVector("结果y = A*x:", h_y.data(), m_rows);
        }
        
        std::cout << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void performanceAnalysis() {
        std::cout << "--- 性能分析和优化建议 ---" << std::endl;
        
        std::cout << "矩阵运算性能特点：" << std::endl;
        std::cout << "  • 矩阵加法：内存带宽受限" << std::endl;
        std::cout << "  • 矩阵乘法：计算密集型，可以通过分块优化" << std::endl;
        std::cout << "  • 矩阵转置：内存访问模式敏感" << std::endl;
        std::cout << "  • 矩阵-向量乘法：适合行并行化" << std::endl;
        
        std::cout << "\n优化策略：" << std::endl;
        std::cout << "  • 使用合适的线程块大小(16x16对2D问题效果好)" << std::endl;
        std::cout << "  • 利用共享内存减少全局内存访问" << std::endl;
        std::cout << "  • 注意内存合并访问模式" << std::endl;
        std::cout << "  • 对于大矩阵，考虑分块算法" << std::endl;
        std::cout << "  • 使用cuBLAS库获得最佳性能" << std::endl;
    }
    
    bool verifyMatrixAddition() {
        const float epsilon = 1e-5f;
        for (int i = 0; i < m_rows * m_cols; ++i) {
            float expected = h_A[i] + h_B[i];
            if (std::abs(h_C[i] - expected) > epsilon) {
                return false;
            }
        }
        return true;
    }
    
    void printMatrix(const std::string& name, const float* matrix, int rows, int cols) {
        std::cout << name << std::endl;
        for (int i = 0; i < rows; ++i) {
            std::cout << "  ";
            for (int j = 0; j < cols; ++j) {
                std::cout << std::fixed << std::setprecision(2) << std::setw(7) 
                          << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    void printVector(const std::string& name, const float* vector, int size) {
        std::cout << name << std::endl;
        std::cout << "  [";
        for (int i = 0; i < size; ++i) {
            std::cout << std::fixed << std::setprecision(2) << vector[i];
            if (i < size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl << std::endl;
    }
};

int main() {
    // 检查CUDA设备
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "没有找到CUDA设备！" << std::endl;
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    
    // 测试不同大小的矩阵
    std::vector<std::pair<int, int>> testSizes = {
        {4, 4},      // 小矩阵，便于查看结果
        {64, 64},    // 中等矩阵
        {512, 512}   // 大矩阵，测试性能
    };
    
    for (auto& size : testSizes) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        MatrixOperationsDemo demo(size.first, size.second);
        demo.runAllTests();
    }
    
    std::cout << "\n=== 总结 ===" << std::endl;
    std::cout << "您已经学会了：" << std::endl;
    std::cout << "  • 2D线程索引在矩阵运算中的应用" << std::endl;
    std::cout << "  • 基本矩阵运算的CUDA实现" << std::endl;
    std::cout << "  • 共享内存优化技巧" << std::endl;
    std::cout << "  • 性能测量和分析方法" << std::endl;
    std::cout << "  • 不同运算的性能特征" << std::endl;
    
    return 0;
}

/*
编译和运行说明：

1. 编译命令：
   nvcc -O3 -o matrix_operations matrix_operations.cu

2. 运行：
   ./matrix_operations

3. 预期输出：
   - 不同大小矩阵的运算结果
   - 各种运算的性能数据
   - 优化版本的性能对比

4. 学习要点：
   - 理解2D线程索引的计算方法
   - 掌握矩阵运算的并行化策略
   - 学会使用共享内存进行优化
   - 了解不同运算的性能特征

5. 扩展练习：
   - 实现更高效的矩阵乘法（分块算法）
   - 添加其他矩阵运算（求逆、特征值等）
   - 比较与cuBLAS库的性能差异
   - 实现不同数据类型的版本
*/