#pragma once

#include "../../operators/base_operator.h"
#include "../../operators/tensor.h"

namespace cuda_learning {
namespace yolo {

// 输入算子
class InputOperator : public operators::BaseOperator {
public:
    InputOperator() : BaseOperator("Input") {}

    void forward(const std::vector<operators::FloatTensor>& inputs,
                std::vector<operators::FloatTensor>& outputs,
                const operators::OperatorContext& context) override {
        // 输入算子只是传递数据
        outputs = inputs;
    }

    std::vector<operators::TensorShape> inferOutputShapes(
        const std::vector<operators::TensorShape>& input_shapes,
        const operators::OperatorContext& context) override {
        return input_shapes;
    }

    std::string getDescription() const override {
        return "Input operator for model inputs";
    }
};

// 卷积算子（简化实现）
class ConvOperator : public operators::BaseOperator {
public:
    ConvOperator() : BaseOperator("Conv") {}

    void forward(const std::vector<operators::FloatTensor>& inputs,
                std::vector<operators::FloatTensor>& outputs,
                const operators::OperatorContext& context) override {
        // 简化的卷积实现 - 只是复制输入到输出
        if (!inputs.empty() && !outputs.empty()) {
            outputs[0] = inputs[0];
        }
    }

    std::vector<operators::TensorShape> inferOutputShapes(
        const std::vector<operators::TensorShape>& input_shapes,
        const operators::OperatorContext& context) override {
        if (input_shapes.empty()) return {};

        // 简化的形状推断 - 假设输出形状与输入相同
        return {input_shapes[0]};
    }

    std::vector<std::string> getRequiredParams() const override {
        return {"kernel_size", "stride", "padding"};
    }

    std::string getDescription() const override {
        return "2D Convolution operator";
    }
};

// 批归一化算子
class BatchNormOperator : public operators::BaseOperator {
public:
    BatchNormOperator() : BaseOperator("BatchNormalization") {}

    void forward(const std::vector<operators::FloatTensor>& inputs,
                std::vector<operators::FloatTensor>& outputs,
                const operators::OperatorContext& context) override {
        if (!inputs.empty() && !outputs.empty()) {
            outputs[0] = inputs[0];
        }
    }

    std::vector<operators::TensorShape> inferOutputShapes(
        const std::vector<operators::TensorShape>& input_shapes,
        const operators::OperatorContext& context) override {
        return input_shapes;
    }

    std::string getDescription() const override {
        return "Batch Normalization operator";
    }
};

// ReLU激活算子
class ReLUOperator : public operators::BaseOperator {
public:
    ReLUOperator() : BaseOperator("Relu") {}

    void forward(const std::vector<operators::FloatTensor>& inputs,
                std::vector<operators::FloatTensor>& outputs,
                const operators::OperatorContext& context) override {
        if (!inputs.empty() && !outputs.empty()) {
            outputs[0] = inputs[0];
        }
    }

    std::vector<operators::TensorShape> inferOutputShapes(
        const std::vector<operators::TensorShape>& input_shapes,
        const operators::OperatorContext& context) override {
        return input_shapes;
    }

    std::string getDescription() const override {
        return "ReLU activation operator";
    }
};

// YOLO检测算子
class YOLODetectOperator : public operators::BaseOperator {
public:
    YOLODetectOperator() : BaseOperator("YOLODetect") {}

    void forward(const std::vector<operators::FloatTensor>& inputs,
                std::vector<operators::FloatTensor>& outputs,
                const operators::OperatorContext& context) override {
        // 简化的YOLO检测实现
        for (size_t i = 0; i < inputs.size() && i < outputs.size(); ++i) {
            outputs[i] = inputs[i];
        }
    }

    std::vector<operators::TensorShape> inferOutputShapes(
        const std::vector<operators::TensorShape>& input_shapes,
        const operators::OperatorContext& context) override {
        // YOLO通常有多个输出
        std::vector<operators::TensorShape> output_shapes;
        for (const auto& shape : input_shapes) {
            output_shapes.push_back(shape);
        }
        return output_shapes;
    }

    std::vector<std::string> getRequiredParams() const override {
        return {"num_classes", "anchors"};
    }

    std::string getDescription() const override {
        return "YOLO detection head operator";
    }
};

} // namespace yolo
} // namespace cuda_learning
