#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// CUDA向量加法
__global__ void vectorAddGPU(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU向量加法
void vectorAddCPU(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 10240 * 10240;  // 1M个元素
    const size_t size = N * sizeof(float);

    std::cout << "=== CUDA vs CPU 性能对比 ===" << std::endl;
    std::cout << "向量大小: " << N << " 个元素 (" << size / (1024*1024) << " MB)" << std::endl;

    // 分配主机内存
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c_cpu = new float[N];
    float* h_c_gpu = new float[N];

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // === CPU测试 ===
    std::cout << "\n--- CPU测试 ---" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    vectorAddCPU(h_a, h_b, h_c_cpu, N);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);

    std::cout << "CPU执行时间: " << duration_cpu.count() / 1000.0 << " ms" << std::endl;

    // === GPU测试 ===
    std::cout << "\n--- GPU测试 ---" << std::endl;

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 复制数据到设备
    auto start_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 配置执行参数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动核函数
    vectorAddGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // 复制结果回主机
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);

    std::cout << "GPU执行时间 (包含数据传输): " << duration_gpu.count() / 1000.0 << " ms" << std::endl;

    // === 仅GPU计算时间 ===
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_kernel);
    vectorAddGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

    std::cout << "GPU纯计算时间: " << kernel_time << " ms" << std::endl;

    // === 结果验证 ===
    bool correct = true;
    for (int i = 0; i < 100; i++) {  // 检查前100个元素
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "\n--- 性能分析 ---" << std::endl;
    if (correct) {
        std::cout << "✓ 结果验证正确" << std::endl;

        float speedup_total = (float)duration_cpu.count() / duration_gpu.count();
        float speedup_compute = (duration_cpu.count() / 1000.0) / kernel_time;

        std::cout << "总体加速比: " << speedup_total << "x" << std::endl;
        std::cout << "纯计算加速比: " << speedup_compute << "x" << std::endl;

        // 计算带宽
        float bandwidth = (3.0f * size) / (kernel_time / 1000.0f) / (1024*1024*1024);
        std::cout << "GPU内存带宽: " << bandwidth << " GB/s" << std::endl;
    } else {
        std::cout << "✗ 结果验证失败" << std::endl;
    }

    // 清理内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}
