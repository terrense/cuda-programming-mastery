#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace cuda_learning {
namespace yolo {

// 设备函数：激活函数
__device__ __forceinline__ float relu_activation(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float leaky_relu_activation(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

__device__ __forceinline__ float swish_activation(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

// 融合的Conv+BN+ReLU核函数
__global__ void fused_conv_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float bn_eps) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_h * output_w;

    if (tid >= total_output_elements) return;

    // 计算输出位置
    int n = tid / (out_channels * output_h * output_w);
    int remaining = tid % (out_channels * output_h * output_w);
    int c_out = remaining / (output_h * output_w);
    remaining = remaining % (output_h * output_w);
    int h_out = remaining / output_w;
    int w_out = remaining % output_w;

    // 计算卷积
    float conv_result = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = n * in_channels * input_h * input_w +
                                   c_in * input_h * input_w +
                                   h_in * input_w + w_in;

                    int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                    c_in * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // 添加偏置（如果有）
    if (bias != nullptr) {
        conv_result += bias[c_out];
    }

    // 批归一化
    float bn_scale = bn_weight[c_out] / sqrtf(bn_var[c_out] + bn_eps);
    float bn_shift = bn_bias[c_out] - bn_mean[c_out] * bn_scale;
    float bn_result = conv_result * bn_scale + bn_shift;

    // ReLU激活
    float final_result = relu_activation(bn_result);

    output[tid] = final_result;
}

// 融合的Conv+BN+LeakyReLU核函数
__global__ void fused_conv_bn_leaky_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float bn_eps, float leaky_alpha) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_h * output_w;

    if (tid >= total_output_elements) return;

    // 计算输出位置
    int n = tid / (out_channels * output_h * output_w);
    int remaining = tid % (out_channels * output_h * output_w);
    int c_out = remaining / (output_h * output_w);
    remaining = remaining % (output_h * output_w);
    int h_out = remaining / output_w;
    int w_out = remaining % output_w;

    // 计算卷积
    float conv_result = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = n * in_channels * input_h * input_w +
                                   c_in * input_h * input_w +
                                   h_in * input_w + w_in;

                    int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                    c_in * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // 添加偏置（如果有）
    if (bias != nullptr) {
        conv_result += bias[c_out];
    }

    // 批归一化
    float bn_scale = bn_weight[c_out] / sqrtf(bn_var[c_out] + bn_eps);
    float bn_shift = bn_bias[c_out] - bn_mean[c_out] * bn_scale;
    float bn_result = conv_result * bn_scale + bn_shift;

    // LeakyReLU激活
    float final_result = leaky_relu_activation(bn_result, leaky_alpha);

    output[tid] = final_result;
}

// 融合的Conv+BN+Swish核函数
__global__ void fused_conv_bn_swish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float bn_eps) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_h * output_w;

    if (tid >= total_output_elements) return;

    // 计算输出位置
    int n = tid / (out_channels * output_h * output_w);
    int remaining = tid % (out_channels * output_h * output_w);
    int c_out = remaining / (output_h * output_w);
    remaining = remaining % (output_h * output_w);
    int h_out = remaining / output_w;
    int w_out = remaining % output_w;

    // 计算卷积
    float conv_result = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = n * in_channels * input_h * input_w +
                                   c_in * input_h * input_w +
                                   h_in * input_w + w_in;

                    int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                    c_in * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    conv_result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // 添加偏置（如果有）
    if (bias != nullptr) {
        conv_result += bias[c_out];
    }

    // 批归一化
    float bn_scale = bn_weight[c_out] / sqrtf(bn_var[c_out] + bn_eps);
    float bn_shift = bn_bias[c_out] - bn_mean[c_out] * bn_scale;
    float bn_result = conv_result * bn_scale + bn_shift;

    // Swish激活
    float final_result = swish_activation(bn_result);

    output[tid] = final_result;
}

} // namespace yolo
} // namespace cuda_learning

// C接口函数
extern "C" {

void launch_fused_conv_bn_relu_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias,
    const float* bn_mean, const float* bn_var,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float bn_eps,
    cudaStream_t stream) {

    int total_output_elements = batch_size * out_channels * output_h * output_w;
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;

    cuda_learning::yolo::fused_conv_bn_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, bn_weight, bn_bias, bn_mean, bn_var, output,
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, bn_eps);
}

void launch_fused_conv_bn_leaky_relu_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias,
    const float* bn_mean, const float* bn_var,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float bn_eps, float leaky_alpha,
    cudaStream_t stream) {

    int total_output_elements = batch_size * out_channels * output_h * output_w;
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;

    cuda_learning::yolo::fused_conv_bn_leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, bn_weight, bn_bias, bn_mean, bn_var, output,
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, bn_eps, leaky_alpha);
}

void launch_fused_conv_bn_swish_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias,
    const float* bn_mean, const float* bn_var,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int output_h, int output_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, float bn_eps,
    cudaStream_t stream) {

    int total_output_elements = batch_size * out_channels * output_h * output_w;
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;

    cuda_learning::yolo::fused_conv_bn_swish_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, bn_weight, bn_bias, bn_mean, bn_var, output,
        batch_size, in_channels, out_channels,
        input_h, input_w, output_h, output_w,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, bn_eps);
}

}
