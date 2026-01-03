// gpu_more_impressive.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                 \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// A heavier-but-still-simple compute kernel that is easy to scale.
// Each iteration does several FLOPs and some memory traffic.
// We purposely do K iterations to amortize H2D/D2H.
__global__ void saxpy_chain(float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = a[idx];
    float y = b[idx];
    float z = c[idx];

    // Do "iters" rounds of math.
    // Keep values in registers; write back once at the end.
    // This massively increases compute per byte transferred.
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // A mix of FMAs and a cheap nonlinearity.
        // (Avoid sin/cos to keep performance more representative and portable.)
        x = x * 1.000001f + y * 0.999999f;
        z = z + x * 0.000001f;
        // A tiny branchless clamp-ish operation
        float t = z - 1.0f;
        z = z - t * (t > 0.0f);
    }

    a[idx] = x;
    c[idx] = z;
}

// CPU reference doing the same math to compare end-to-end.
static void cpu_compute(float* a, const float* b, float* c, int n, int iters) {
    for (int idx = 0; idx < n; idx++) {
        float x = a[idx];
        float y = b[idx];
        float z = c[idx];
        for (int i = 0; i < iters; i++) {
            x = x * 1.000001f + y * 0.999999f;
            z = z + x * 0.000001f;
            float t = z - 1.0f;
            z = z - t * (t > 0.0f);
        }
        a[idx] = x;
        c[idx] = z;
    }
}

int main(int argc, char** argv) {
    // Scale similar to your previous test.
    // N defaults to ~1e8 floats (about 400MB per array).
    // K defaults to 200 iterations on GPU to make compute dominate.
    long long N = 10240LL * 10240LL; // 104,857,600
    int K = 200;

    if (argc >= 2) N = std::atoll(argv[1]);   // allow override
    if (argc >= 3) K = std::atoi(argv[2]);    // allow override

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::cout << "=== GPU 'More Impressive' Demo (Amortize Transfers) ===\n";
    std::cout << "N = " << N << " floats  (" << std::fixed << std::setprecision(1)
              << (double)bytes / (1024.0 * 1024.0) << " MB per array)\n";
    std::cout << "K = " << K << " iterations per element\n\n";

    // Device info
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA device found.\n";
        return 1;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")\n";
    std::cout << "Global Mem: " << (prop.totalGlobalMem / (1024.0*1024.0*1024.0)) << " GB\n\n";

    // Allocate pinned host memory (key improvement)
    float *h_a=nullptr, *h_b=nullptr, *h_c=nullptr;
    float *h_a_cpu=nullptr, *h_c_cpu=nullptr;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));
    CUDA_CHECK(cudaMallocHost(&h_a_cpu, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c_cpu, bytes));

    // Init
    for (long long i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>((i * 2) % 1000) * 0.001f;
        h_c[i] = static_cast<float>((i * 3) % 1000) * 0.001f;
    }
    // Copy for CPU baseline
    std::memcpy(h_a_cpu, h_a, bytes);
    std::memcpy(h_c_cpu, h_c, bytes);

    // Allocate device memory
    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Create stream + events
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaEvent_t e_start, e_h2d_done, e_kernel_done, e_d2h_done;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_h2d_done));
    CUDA_CHECK(cudaEventCreate(&e_kernel_done));
    CUDA_CHECK(cudaEventCreate(&e_d2h_done));

    // Warm-up to avoid first-use overhead in timing
    CUDA_CHECK(cudaFree(0));

    // ---------------- CPU Timing ----------------
    std::cout << "--- CPU (same K loops) ---\n";
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpu_compute(h_a_cpu, h_b, h_c_cpu, (int)N, K);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_t1 - cpu_t0).count();
    double cpu_ms = cpu_us / 1000.0;
    std::cout << "CPU time: " << cpu_ms << " ms\n\n";

    // ---------------- GPU Timing (async + pinned) ----------------
    std::cout << "--- GPU (H2D + K loops on device + D2H) ---\n";

    int block = 256;
    int grid = (int)((N + block - 1) / block);

    CUDA_CHECK(cudaEventRecord(e_start, stream));

    // Async H2D
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_c, h_c, bytes, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaEventRecord(e_h2d_done, stream));

    // Kernel (K iterations) - compute-heavy relative to transfer
    saxpy_chain<<<grid, block, 0, stream>>>(d_a, d_b, d_c, (int)N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(e_kernel_done, stream));

    // Async D2H (only pull back the arrays we care about verifying)
    CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaEventRecord(e_d2h_done, stream));
    CUDA_CHECK(cudaEventSynchronize(e_d2h_done));

    float ms_total=0, ms_h2d=0, ms_kernel=0, ms_d2h=0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, e_start, e_d2h_done));
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, e_start, e_h2d_done));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, e_h2d_done, e_kernel_done));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, e_kernel_done, e_d2h_done));

    std::cout << "H2D time:    " << ms_h2d   << " ms\n";
    std::cout << "Kernel time: " << ms_kernel<< " ms\n";
    std::cout << "D2H time:    " << ms_d2h   << " ms\n";
    std::cout << "Total time:  " << ms_total << " ms\n\n";

    // ---------------- Verify (sample) ----------------
    bool ok = true;
    for (int i = 0; i < 200; i++) {
        int idx = i * 9973; // spread samples a bit
        if (idx >= (int)N) break;
        float da = std::fabs(h_a[idx] - h_a_cpu[idx]);
        float dc = std::fabs(h_c[idx] - h_c_cpu[idx]);
        if (da > 1e-3f || dc > 1e-3f) { // tolerate small FP differences
            ok = false;
            std::cout << "Mismatch at " << idx
                      << " | a GPU=" << h_a[idx] << " CPU=" << h_a_cpu[idx]
                      << " | c GPU=" << h_c[idx] << " CPU=" << h_c_cpu[idx]
                      << "\n";
            break;
        }
    }

    std::cout << "--- Result ---\n";
    std::cout << (ok ? "✓ Verification PASSED\n" : "✗ Verification FAILED\n");

    // Speedup
    double gpu_ms = ms_total;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Speedup (Total):  " << (cpu_ms / gpu_ms) << "x\n";
    std::cout << "Speedup (Kernel): " << (cpu_ms / ms_kernel) << "x  (CPU vs GPU compute only)\n\n";

    // Rough effective bandwidth: total bytes moved over PCIe path here (3*H2D + 2*D2H arrays)
    // H2D: a,b,c = 3*bytes ; D2H: a,c = 2*bytes -> total transfer = 5*bytes
    double transfer_gb = (5.0 * (double)bytes) / (1024.0*1024.0*1024.0);
    double transfer_s = (ms_h2d + ms_d2h) / 1000.0;
    std::cout << "Approx transfer throughput: "
              << (transfer_gb / transfer_s) << " GB/s (H2D+D2H)\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(e_start));
    CUDA_CHECK(cudaEventDestroy(e_h2d_done));
    CUDA_CHECK(cudaEventDestroy(e_kernel_done));
    CUDA_CHECK(cudaEventDestroy(e_d2h_done));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFreeHost(h_a_cpu));
    CUDA_CHECK(cudaFreeHost(h_c_cpu));

    return 0;
}
