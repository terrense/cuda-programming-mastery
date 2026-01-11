#include "plugin_kernels.cuh"
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

namespace yolo {
namespace tensorrt {

// YOLO检测层核函数实现
__global__ void yolo_detection_kernel(
    const float* input,
    float* output,
    float* workspace,
    int batch_size,
    int height,
    int width,
    int num_classes,
    int num_anchors,
    float conf_threshold,
    float nms_threshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (batch_idx >= batch_size) return;

    int total_elements = height * width * num_anchors;
    if (idx >= total_elements) return;

    // 计算网格位置和锚点索引
    int anchor_idx = idx % num_anchors;
    int spatial_idx = idx / num_anchors;
    int grid_y = spatial_idx / width;
    int grid_x = spatial_idx % width;

    // 输入数据布局: [batch, anchors, grid_h, grid_w, 5 + num_classes]
    int input_stride = num_anchors * height * width * (5 + num_classes);
    int anchor_stride = height * width * (5 + num_classes);
    int spatial_stride = 5 + num_classes;

    const float* anchor_data = input + batch_idx * input_stride +
                              anchor_idx * anchor_stride +
                              spatial_idx * spatial_stride;

    // 解析边界框信息
    float box_x = anchor_data[0];
    float box_y = anchor_data[1];
    float box_w = anchor_data[2];
    float box_h = anchor_data[3];
    float objectness = anchor_data[4];

    // 应用sigmoid激活
    objectness = 1.0f / (1.0f + expf(-objectness));

    // 检查置信度阈值
    if (objectness < conf_threshold) return;

    // 计算类别概率
    float max_class_prob = 0.0f;
    int max_class_idx = 0;

    for (int c = 0; c < num_classes; ++c) {
        float class_prob = 1.0f / (1.0f + expf(-anchor_data[5 + c]));
        if (class_prob > max_class_prob) {
            max_class_prob = class_prob;
            max_class_idx = c;
        }
    }

    float final_confidence = objectness * max_class_prob;
    if (final_confidence < conf_threshold) return;

    // 转换边界框坐标 (相对于网格 -> 绝对坐标)
    box_x = (1.0f / (1.0f + expf(-box_x)) + grid_x) / width;
    box_y = (1.0f / (1.0f + expf(-box_y)) + grid_y) / height;
    box_w = expf(box_w) / width;
    box_h = expf(box_h) / height;

    // 写入输出 (使用原子操作避免竞争)
    int output_idx = atomicAdd(&workspace[batch_idx * 1000], 1); // 计数器
    if (output_idx < 1000) { // 最大检测数量限制
        int output_offset = batch_idx * 1000 * 6 + output_idx * 6;
        output[output_offset + 0] = box_x;
        output[output_offset + 1] = box_y;
        output[output_offset + 2] = box_w;
        output[output_offset + 3] = box_h;
        output[output_offset + 4] = final_confidence;
        output[output_offset + 5] = static_cast<float>(max_class_idx);
    }
}

__global__ void yolo_detection_kernel_half(
    const __half* input,
    __half* output,
    __half* workspace,
    int batch_size,
    int height,
    int width,
    int num_classes,
    int num_anchors,
    float conf_threshold,
    float nms_threshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (batch_idx >= batch_size) return;

    int total_elements = height * width * num_anchors;
    if (idx >= total_elements) return;

    // 计算网格位置和锚点索引
    int anchor_idx = idx % num_anchors;
    int spatial_idx = idx / num_anchors;
    int grid_y = spatial_idx / width;
    int grid_x = spatial_idx % width;

    // 输入数据布局: [batch, anchors, grid_h, grid_w, 5 + num_classes]
    int input_stride = num_anchors * height * width * (5 + num_classes);
    int anchor_stride = height * width * (5 + num_classes);
    int spatial_stride = 5 + num_classes;

    const __half* anchor_data = input + batch_idx * input_stride +
                               anchor_idx * anchor_stride +
                               spatial_idx * spatial_stride;

    // 解析边界框信息 (转换为float进行计算)
    float box_x = __half2float(anchor_data[0]);
    float box_y = __half2float(anchor_data[1]);
    float box_w = __half2float(anchor_data[2]);
    float box_h = __half2float(anchor_data[3]);
    float objectness = __half2float(anchor_data[4]);

    // 应用sigmoid激活
    objectness = 1.0f / (1.0f + expf(-objectness));

    // 检查置信度阈值
    if (objectness < conf_threshold) return;

    // 计算类别概率
    float max_class_prob = 0.0f;
    int max_class_idx = 0;

    for (int c = 0; c < num_classes; ++c) {
        float class_prob = 1.0f / (1.0f + expf(-__half2float(anchor_data[5 + c])));
        if (class_prob > max_class_prob) {
            max_class_prob = class_prob;
            max_class_idx = c;
        }
    }

    float final_confidence = objectness * max_class_prob;
    if (final_confidence < conf_threshold) return;

    // 转换边界框坐标
    box_x = (1.0f / (1.0f + expf(-box_x)) + grid_x) / width;
    box_y = (1.0f / (1.0f + expf(-box_y)) + grid_y) / height;
    box_w = expf(box_w) / width;
    box_h = expf(box_h) / height;

    // 写入输出 (转换回half)
    int output_idx = atomicAdd(reinterpret_cast<int*>(&workspace[batch_idx * 1000]), 1);
    if (output_idx < 1000) {
        int output_offset = batch_idx * 1000 * 6 + output_idx * 6;
        output[output_offset + 0] = __float2half(box_x);
        output[output_offset + 1] = __float2half(box_y);
        output[output_offset + 2] = __float2half(box_w);
        output[output_offset + 3] = __float2half(box_h);
        output[output_offset + 4] = __float2half(final_confidence);
        output[output_offset + 5] = __float2half(static_cast<float>(max_class_idx));
    }
}

// Focus层核函数实现
__global__ void focus_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (batch_idx >= batch_size) return;

    int output_height = height / 2;
    int output_width = width / 2;
    int total_output_elements = channels * 4 * output_height * output_width;

    if (idx >= total_output_elements) return;

    // 计算输出位置
    int output_c = idx / (output_height * output_width);
    int spatial_idx = idx % (output_height * output_width);
    int output_y = spatial_idx / output_width;
    int output_x = spatial_idx % output_width;

    // 计算对应的输入通道和位置
    int input_c = output_c / 4;
    int sub_pixel = output_c % 4;

    // 子像素位置映射
    int offset_y = sub_pixel / 2;
    int offset_x = sub_pixel % 2;

    int input_y = output_y * 2 + offset_y;
    int input_x = output_x * 2 + offset_x;

    // 计算输入和输出索引
    int input_idx = batch_idx * channels * height * width +
                   input_c * height * width +
                   input_y * width + input_x;

    int output_idx = batch_idx * channels * 4 * output_height * output_width +
                    output_c * output_height * output_width +
                    output_y * output_width + output_x;

    // 复制数据
    output[output_idx] = input[input_idx];
}

__global__ void focus_kernel_half(
    const __half* input,
    __half* output,
    int batch_size,
    int channels,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (batch_idx >= batch_size) return;

    int output_height = height / 2;
    int output_width = width / 2;
    int total_output_elements = channels * 4 * output_height * output_width;

    if (idx >= total_output_elements) return;

    // 计算输出位置
    int output_c = idx / (output_height * output_width);
    int spatial_idx = idx % (output_height * output_width);
    int output_y = spatial_idx / output_width;
    int output_x = spatial_idx % output_width;

    // 计算对应的输入通道和位置
    int input_c = output_c / 4;
    int sub_pixel = output_c % 4;

    // 子像素位置映射
    int offset_y = sub_pixel / 2;
    int offset_x = sub_pixel % 2;

    int input_y = output_y * 2 + offset_y;
    int input_x = output_x * 2 + offset_x;

    // 计算输入和输出索引
    int input_idx = batch_idx * channels * height * width +
                   input_c * height * width +
                   input_y * width + input_x;

    int output_idx = batch_idx * channels * 4 * output_height * output_width +
                    output_c * output_height * output_width +
                    output_y * output_width + output_x;

    // 复制数据
    output[output_idx] = input[input_idx];
}

// 核函数启动器实现
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
    cudaStream_t stream) {

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    yolo_detection_kernel<<<grid, block, 0, stream>>>(
        input, output, workspace,
        batch_size, height, width,
        num_classes, num_anchors,
        conf_threshold, nms_threshold);
}

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
    cudaStream_t stream) {

    dim3 grid(grid_x, grid_y);
    dim3 block(block_x, block_y);

    yolo_detection_kernel_half<<<grid, block, 0, stream>>>(
        input, output, workspace,
        batch_size, height, width,
        num_classes, num_anchors,
        conf_threshold, nms_threshold);
}

void launchFocusKernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    cudaStream_t stream) {

    int output_height = height / 2;
    int output_width = width / 2;
    int total_elements = channels * 4 * output_height * output_width;

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    dim3 grid(grid_size, batch_size);
    dim3 block(block_size);

    focus_kernel<<<grid, block, 0, stream>>>(
        input, output, batch_size, channels, height, width);
}

void launchFocusKernelHalf(
    const __half* input,
    __half* output,
    int batch_size,
    int channels,
    int height,
    int width,
    cudaStream_t stream) {

    int output_height = height / 2;
    int output_width = width / 2;
    int total_elements = channels * 4 * output_height * output_width;

    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    dim3 grid(grid_size, batch_size);
    dim3 block(block_size);

    focus_kernel_half<<<grid, block, 0, stream>>>(
        input, output, batch_size, channels, height, width);
}

} // namespace tensorrt
} // namespace yolo
