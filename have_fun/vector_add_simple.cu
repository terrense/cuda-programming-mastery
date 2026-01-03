#include <cuda_runtime.h>
#include <iostream>

// 向量加法核函数
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    std::cout << "=== CUDA向量加法演示 ===" << std::endl;
    std::cout << "向量大小: " << N << " 个元素" << std::endl;

    // 分配主机内存
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 复制数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 配置执行参数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::cout << "执行配置: <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;

    // 启动核函数
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // 等待GPU完成
    cudaDeviceSynchronize();

    // 复制结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证结果（显示前10个）
    std::cout << "\n结果验证（前10个元素）:" << std::endl;
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i];
        if (h_c[i] == expected) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗ (期望: " << expected << ")" << std::endl;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "\n🎉 向量加法计算正确！" << std::endl;
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
