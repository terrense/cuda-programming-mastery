#include <cuda_runtime.h>
#include <iostream>

// 矩阵加法核函数
__global__ void matrixAdd(float* a, float* b, float* c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int ROWS = 16;
    const int COLS = 16;
    const int N = ROWS * COLS;
    const size_t size = N * sizeof(float);
    
    std::cout << "=== CUDA矩阵加法演示 ===" << std::endl;
    std::cout << "矩阵大小: " << ROWS << " x " << COLS << std::endl;
    
    // 分配主机内存
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i % 10;
        h_b[i] = (i * 2) % 10;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 复制数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 配置2D执行参数
    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x,
                  (ROWS + blockSize.y - 1) / blockSize.y);
    
    std::cout << "执行配置: <<<(" << gridSize.x << "," << gridSize.y 
              << "), (" << blockSize.x << "," << blockSize.y << ")>>>" << std::endl;
    
    // 启动核函数
    matrixAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, ROWS, COLS);
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 显示结果（4x4子矩阵）
    std::cout << "\n矩阵A (左上角4x4):" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_a[i * COLS + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n矩阵B (左上角4x4):" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_b[i * COLS + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n矩阵C = A + B (左上角4x4):" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << h_c[i * COLS + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "\n🎉 矩阵加法计算正确！" << std::endl;
    }
    
    // 清理内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
