#include "src/operators/conv_pool_operators.h"
#include "src/operators/operators.h"
#include <iostream>
#include <cassert>

using namespace cuda_learning::operators;

int main() {
    std::cout << "Testing Convolution and Pooling Operators..." << std::endl;

    try {
        // 初始化算子系统
        initializeOperatorSystem();

        // 注册卷积和池化算子
        registerConvPoolOperators();

        // 测试Conv2D算子
        std::cout << "\n=== Testing Conv2D Operator ===" << std::endl;

        // 创建输入张量 [N=1, C=3, H=8, W=8]
        FloatTensor input({1, 3, 8, 8});
        input.fill(1.0f);

        // 创建卷积核 [K=16, C=3, R=3, S=3]
        FloatTensor weight({16, 3, 3, 3});
        weight.fill(0.1f);

        // 设置卷积参数
        OperatorContext context;
        context.setParam("stride_h", 1);
        context.setParam("stride_w", 1);
        context.setParam("pad_h", 1);
        context.setParam("pad_w", 1);

        // 创建卷积算子
        Conv2DOperator conv_op;

        // 推断输出形状
        auto output_shapes = conv_op.inferOutputShapes({input.shape(), weight.shape()}, context);
        FloatTensor output(output_shapes[0].dims());

        std::cout << "Input shape: [" << input.shape().dim(0) << ", " << input.shape().dim(1)
                  << ", " << input.shape().dim(2) << ", " << input.shape().dim(3) << "]" << std::endl;
        std::cout << "Weight shape: [" << weight.shape().dim(0) << ", " << weight.shape().dim(1)
                  << ", " << weight.shape().dim(2) << ", " << weight.shape().dim(3) << "]" << std::endl;
        std::cout << "Output shape: [" << output.shape().dim(0) << ", " << output.shape().dim(1)
                  << ", " << output.shape().dim(2) << ", " << output.shape().dim(3) << "]" << std::endl;

        // 执行卷积
        conv_op.forward({input, weight}, {output}, context);

        // 验证输出
        auto output_host = output.toHost();
        bool has_nan = false;
        for (float val : output_host) {
            if (std::isnan(val)) {
                has_nan = true;
                break;
            }
        }

        std::cout << "Conv2D test: " << (has_nan ? "FAILED" : "PASSED") << std::endl;

        // 测试MaxPool2D算子
        std::cout << "\n=== Testing MaxPool2D Operator ===" << std::endl;

        // 使用卷积输出作为池化输入
        OperatorContext pool_context;
        pool_context.setParam("kernel_h", 2);
        pool_context.setParam("kernel_w", 2);
        pool_context.setParam("stride_h", 2);
        pool_context.setParam("stride_w", 2);
        pool_context.setParam("pad_h", 0);
        pool_context.setParam("pad_w", 0);

        MaxPool2DOperator pool_op;

        auto pool_output_shapes = pool_op.inferOutputShapes({output.shape()}, pool_context);
        FloatTensor pool_output(pool_output_shapes[0].dims());

        std::cout << "Pool input shape: [" << output.shape().dim(0) << ", " << output.shape().dim(1)
                  << ", " << output.shape().dim(2) << ", " << output.shape().dim(3) << "]" << std::endl;
        std::cout << "Pool output shape: [" << pool_output.shape().dim(0) << ", " << pool_output.shape().dim(1)
                  << ", " << pool_output.shape().dim(2) << ", " << pool_output.shape().dim(3) << "]" << std::endl;

        // 执行池化
        pool_op.forward({output}, {pool_output}, pool_context);

        // 验证输出
        auto pool_output_host = pool_output.toHost();
        has_nan = false;
        for (float val : pool_output_host) {
            if (std::isnan(val)) {
                has_nan = true;
                break;
            }
        }

        std::cout << "MaxPool2D test: " << (has_nan ? "FAILED" : "PASSED") << std::endl;

        // 测试BatchNorm2D算子
        std::cout << "\n=== Testing BatchNorm2D Operator ===" << std::endl;

        // 创建批量归一化参数
        int channels = pool_output.shape().dim(1);
        FloatTensor gamma({channels});
        FloatTensor beta({channels});
        gamma.ones();
        beta.zero();

        OperatorContext bn_context;
        bn_context.setParam("eps", 1e-5f);
        bn_context.setParam("training", true);

        BatchNorm2DOperator bn_op;

        auto bn_output_shapes = bn_op.inferOutputShapes({pool_output.shape()}, bn_context);
        FloatTensor bn_output(bn_output_shapes[0].dims());

        std::cout << "BN input shape: [" << pool_output.shape().dim(0) << ", " << pool_output.shape().dim(1)
                  << ", " << pool_output.shape().dim(2) << ", " << pool_output.shape().dim(3) << "]" << std::endl;
        std::cout << "Gamma/Beta shape: [" << gamma.shape().dim(0) << "]" << std::endl;

        // 执行批量归一化
        bn_op.forward({pool_output, gamma, beta}, {bn_output}, bn_context);

        // 验证输出
        auto bn_output_host = bn_output.toHost();
        has_nan = false;
        for (float val : bn_output_host) {
            if (std::isnan(val)) {
                has_nan = true;
                break;
            }
        }

        std::cout << "BatchNorm2D test: " << (has_nan ? "FAILED" : "PASSED") << std::endl;

        std::cout << "\n=== All Tests Completed ===" << std::endl;

        // 清理算子系统
        shutdownOperatorSystem();

    } catch (const std::exception& e) {
        std::cout << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
