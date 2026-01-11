#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace yolo {
namespace tensorrt {

/**
 * @brief YOLO检测层CUDA核函数声明
 */
void launchYOLODetectionKernel(
    const float* input,
    float* output,
    float* workspace,
    int batch_size,
    int height,
    int width,
    int num_classes,
    int num_anchors,
    float conf_threshold,
    float nms_threshold,
    int grid_x,
    int grid_y,
    int block_x,
    int block_y,
    cudaStream_t stream);

/**
 * @brief YOLO检测层CUDA核函数 (FP16版本)
 */
void launchYOLODetectionKernelHalf(
    const __half* input,
    __half* output,
    __half* workspace,
    int batch_size,
    int height,
    int width,
    int num_classes,
    int num_anchors,
    float conf_threshold,
    float nms_threshold,
    int grid_x,
    int grid_y,
    int block_x,
    int block_y,
    cudaStream_t stream);

/**
 * @brief Focus层CUDA核函数声明
 */
void launchFocusKernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

/**
 * @brief Focus层CUDA核函数 (FP16版本)
 */
void launchFocusKernelHalf(
    const __half* input,
    __half* output,
    int batch_size,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

} // namespace tensorrt
} // namespace yolo
