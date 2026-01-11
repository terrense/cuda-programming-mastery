#include "model_graph.h"
#include "../operators/yolo_operators.h"
#include "../../core/error_handler.h"
#include <iostream>
#include <algorithm>
#include <queue>

namespace cuda_learning {
namespace yolo {

bool ModelGraph::addNode(const std::string& name, const std::string& op_type) {
    if (hasNode(name)) {
        std::cerr << "Node already exists: " << name << std::endl;
        return false;
    }

    // 检查算子是否支持
    if (!CudaOperatorMapper::isOperatorSupported(op_type)) {
        std::cerr << "Unsupported operator type: " << op_type << std::endl;
        return false;
    }

    auto node = std::make_unique<GraphNode>(name, op_type);

    // 创建算子实例
    node->op = CudaOperatorMapper::createOperator(op_type);
    if (!node->op) {
        std::cerr << "Failed to create operator: " << op_type << std::endl;
        return false;
    }

    nodes_[name] = std::move(node);
    std::cout << "Added node: " << name << " (type: " << op_type << ")" << std::endl;
    return true;
}

bool ModelGraph::addEdge(const std::string& tensor_name, const std::string& producer,
                        const std::vector<std::string>& consumers) {
    if (hasEdge(tensor_name)) {
        std::cerr << "Edge already exists: " << tensor_name << std::endl;
        return false;
    }

    // 验证生产者节点存在
    if (!producer.empty() && !hasNode(producer)) {
        std::cerr << "Producer node not found: " << producer << std::endl;
        return false;
    }

    // 验证消费者节点存在
    for (const auto& consumer : consumers) {
        if (!hasNode(consumer)) {
            std::cerr << "Consumer node not found: " << consumer << std::endl;
            return false;
        }
    }

    auto edge = std::make_unique<GraphEdge>(tensor_name);
    edge->producer_node = producer;
    edge->consumer_nodes = consumers;

    edges_[tensor_name] = std::move(edge);
    std::cout << "Added edge: " << tensor_name << " (producer: " << producer << ")" << std::endl;
    return true;
}

bool ModelGraph::topologicalSort() {
    execution_order_.clear();
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> rec_stack;

    // 检查是否有环
    for (const auto& pair : nodes_) {
        if (visited.find(pair.first) == visited.end()) {
            if (hasCycle()) {
                std::cerr << "Graph contains cycle, cannot perform topological sort" << std::endl;
                return false;
            }
        }
    }

    // 执行拓扑排序
    visited.clear();
    for (const auto& pair : nodes_) {
        if (visited.find(pair.first) == visited.end()) {
            dfsVisit(pair.first, visited, rec_stack, execution_order_);
        }
    }

    // 反转顺序
    std::reverse(execution_order_.begin(), execution_order_.end());

    std::cout << "Topological sort completed. Execution order:" << std::endl;
    for (const auto& node_name : execution_order_) {
        std::cout << "  " << node_name << std::endl;
    }

    return true;
}

bool ModelGraph::inferShapes(const std::map<std::string, operators::TensorShape>& input_shapes) {
    std::cout << "Starting shape inference..." << std::endl;

    // 设置输入张量形状
    for (const auto& pair : input_shapes) {
        auto edge = getEdge(pair.first);
        if (edge) {
            edge->shape = pair.second;
            std::cout << "Input shape set: " << pair.first << " -> " << pair.second.toString() << std::endl;
        }
    }

    // 按执行顺序推断形状
    for (const auto& node_name : execution_order_) {
        auto node = getNode(node_name);
        if (!node) continue;

        // 收集输入形状
        std::vector<operators::TensorShape> input_tensor_shapes;
        for (const auto& input_name : node->input_names) {
            auto edge = getEdge(input_name);
            if (edge) {
                input_tensor_shapes.push_back(edge->shape);
            }
        }

        // 推断输出形状
        try {
            auto output_shapes = node->op->inferOutputShapes(input_tensor_shapes, node->context);

            // 设置输出张量形状
            for (size_t i = 0; i < node->output_names.size() && i < output_shapes.size(); ++i) {
                auto edge = getEdge(node->output_names[i]);
                if (edge) {
                    edge->shape = output_shapes[i];
                    std::cout << "Output shape inferred: " << node->output_names[i]
                              << " -> " << output_shapes[i].toString() << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Shape inference failed for node " << node_name << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::cout << "Shape inference completed successfully" << std::endl;
    return true;
}

bool ModelGraph::execute(const std::map<std::string, operators::FloatTensor>& inputs,
                        std::map<std::string, operators::FloatTensor>& outputs) {
    std::cout << "Starting graph execution..." << std::endl;

    // 清空之前的执行状态
    for (auto& pair : nodes_) {
        pair.second->executed = false;
    }
    tensors_.clear();

    // 设置输入张量
    for (const auto& pair : inputs) {
        tensors_[pair.first] = pair.second;
        std::cout << "Input tensor set: " << pair.first << std::endl;
    }

    // 按执行顺序执行节点
    for (const auto& node_name : execution_order_) {
        auto node = getNode(node_name);
        if (!node) {
            std::cerr << "Node not found: " << node_name << std::endl;
            return false;
        }

        std::cout << "Executing node: " << node_name << " (" << node->op_type << ")" << std::endl;

        // 收集输入张量
        std::vector<operators::FloatTensor> input_tensors;
        for (const auto& input_name : node->input_names) {
            auto it = tensors_.find(input_name);
            if (it == tensors_.end()) {
                std::cerr << "Input tensor not found: " << input_name << std::endl;
                return false;
            }
            input_tensors.push_back(it->second);
        }

        // 准备输出张量
        std::vector<operators::FloatTensor> output_tensors;
        for (const auto& output_name : node->output_names) {
            auto edge = getEdge(output_name);
            if (edge) {
                operators::FloatTensor tensor(edge->shape.dims());
                output_tensors.push_back(tensor);
            }
        }

        try {
            // 执行算子
            node->op->forward(input_tensors, output_tensors, node->context);

            // 存储输出张量
            for (size_t i = 0; i < node->output_names.size() && i < output_tensors.size(); ++i) {
                tensors_[node->output_names[i]] = output_tensors[i];
            }

            node->executed = true;
            std::cout << "Node executed successfully: " << node_name << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Execution failed for node " << node_name << ": " << e.what() << std::endl;
            return false;
        }
    }

    // 收集输出张量
    for (const auto& output_name : output_names_) {
        auto it = tensors_.find(output_name);
        if (it != tensors_.end()) {
            outputs[output_name] = it->second;
            std::cout << "Output tensor collected: " << output_name << std::endl;
        }
    }

    std::cout << "Graph execution completed successfully" << std::endl;
    return true;
}

bool ModelGraph::optimize() {
    std::cout << "Starting graph optimization..." << std::endl;

    // 这里可以实现各种图优化策略：
    // 1. 算子融合
    // 2. 常量折叠
    // 3. 死代码消除
    // 4. 内存优化

    // 简单的示例：移除未使用的节点
    std::unordered_set<std::string> used_nodes;

    // 从输出开始反向标记使用的节点
    std::queue<std::string> to_visit;
    for (const auto& output : output_names_) {
        auto edge = getEdge(output);
        if (edge && !edge->producer_node.empty()) {
            to_visit.push(edge->producer_node);
        }
    }

    while (!to_visit.empty()) {
        std::string node_name = to_visit.front();
        to_visit.pop();

        if (used_nodes.find(node_name) != used_nodes.end()) {
            continue;
        }

        used_nodes.insert(node_name);
        auto node = getNode(node_name);
        if (node) {
            for (const auto& input_name : node->input_names) {
                auto edge = getEdge(input_name);
                if (edge && !edge->producer_node.empty()) {
                    to_visit.push(edge->producer_node);
                }
            }
        }
    }

    // 移除未使用的节点
    auto it = nodes_.begin();
    while (it != nodes_.end()) {
        if (used_nodes.find(it->first) == used_nodes.end()) {
            std::cout << "Removing unused node: " << it->first << std::endl;
            it = nodes_.erase(it);
        } else {
            ++it;
        }
    }

    std::cout << "Graph optimization completed" << std::endl;
    return true;
}

GraphNode* ModelGraph::getNode(const std::string& name) {
    auto it = nodes_.find(name);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

const GraphNode* ModelGraph::getNode(const std::string& name) const {
    auto it = nodes_.find(name);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

GraphEdge* ModelGraph::getEdge(const std::string& name) {
    auto it = edges_.find(name);
    return (it != edges_.end()) ? it->second.get() : nullptr;
}

const GraphEdge* ModelGraph::getEdge(const std::string& name) const {
    auto it = edges_.find(name);
    return (it != edges_.end()) ? it->second.get() : nullptr;
}

operators::FloatTensor* ModelGraph::getTensor(const std::string& name) {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

const operators::FloatTensor* ModelGraph::getTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

void ModelGraph::printGraph() const {
    std::cout << "\n=== Model Graph Information ===" << std::endl;
    std::cout << "Nodes: " << nodes_.size() << std::endl;
    std::cout << "Edges: " << edges_.size() << std::endl;

    std::cout << "\nInputs:" << std::endl;
    for (const auto& input : input_names_) {
        std::cout << "  " << input << std::endl;
    }

    std::cout << "\nOutputs:" << std::endl;
    for (const auto& output : output_names_) {
        std::cout << "  " << output << std::endl;
    }

    std::cout << "\nNodes:" << std::endl;
    for (const auto& pair : nodes_) {
        const auto& node = pair.second;
        std::cout << "  " << node->name << " (" << node->op_type << ")" << std::endl;
        std::cout << "    Inputs: ";
        for (const auto& input : node->input_names) {
            std::cout << input << " ";
        }
        std::cout << std::endl;
        std::cout << "    Outputs: ";
        for (const auto& output : node->output_names) {
            std::cout << output << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "==============================\n" << std::endl;
}

bool ModelGraph::validate() const {
    // 验证输入输出节点存在
    for (const auto& input : input_names_) {
        if (!hasEdge(input)) {
            std::cerr << "Input edge not found: " << input << std::endl;
            return false;
        }
    }

    for (const auto& output : output_names_) {
        if (!hasEdge(output)) {
            std::cerr << "Output edge not found: " << output << std::endl;
            return false;
        }
    }

    // 验证节点连接
    for (const auto& pair : nodes_) {
        const auto& node = pair.second;

        for (const auto& input_name : node->input_names) {
            if (!hasEdge(input_name)) {
                std::cerr << "Input edge not found for node " << node->name << ": " << input_name << std::endl;
                return false;
            }
        }

        for (const auto& output_name : node->output_names) {
            if (!hasEdge(output_name)) {
                std::cerr << "Output edge not found for node " << node->name << ": " << output_name << std::endl;
                return false;
            }
        }
    }

    return true;
}

void ModelGraph::clear() {
    nodes_.clear();
    edges_.clear();
    tensors_.clear();
    input_names_.clear();
    output_names_.clear();
    execution_order_.clear();
}

bool ModelGraph::hasNode(const std::string& name) const {
    return nodes_.find(name) != nodes_.end();
}

bool ModelGraph::hasEdge(const std::string& name) const {
    return edges_.find(name) != edges_.end();
}

bool ModelGraph::hasCycle() const {
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> rec_stack;

    for (const auto& pair : nodes_) {
        if (visited.find(pair.first) == visited.end()) {
            std::vector<std::string> dummy_order;
            dfsVisit(pair.first, visited, rec_stack, dummy_order);
            if (!rec_stack.empty()) {
                return true;
            }
        }
    }

    return false;
}

void ModelGraph::dfsVisit(const std::string& node_name, std::unordered_set<std::string>& visited,
                         std::unordered_set<std::string>& rec_stack, std::vector<std::string>& order) const {
    visited.insert(node_name);
    rec_stack.insert(node_name);

    auto node = getNode(node_name);
    if (node) {
        for (const auto& output_name : node->output_names) {
            auto edge = getEdge(output_name);
            if (edge) {
                for (const auto& consumer : edge->consumer_nodes) {
                    if (visited.find(consumer) == visited.end()) {
                        dfsVisit(consumer, visited, rec_stack, order);
                    }
                }
            }
        }
    }

    rec_stack.erase(node_name);
    order.push_back(node_name);
}

// CudaOperatorMapper 实现
std::map<std::string, std::string> CudaOperatorMapper::operator_mapping_;

std::string CudaOperatorMapper::mapOperatorType(const std::string& original_type) {
    if (operator_mapping_.empty()) {
        initializeMapping();
    }

    auto it = operator_mapping_.find(original_type);
    return (it != operator_mapping_.end()) ? it->second : original_type;
}

bool CudaOperatorMapper::isOperatorSupported(const std::string& op_type) {
    auto supported_ops = getSupportedOperators();
    return std::find(supported_ops.begin(), supported_ops.end(), op_type) != supported_ops.end();
}

std::vector<std::string> CudaOperatorMapper::getSupportedOperators() {
    return {
        "Input", "Conv", "BatchNormalization", "Relu", "Add", "Mul", "MatMul",
        "MaxPool", "AveragePool", "Reshape", "Transpose", "Concat",
        "YOLODetect", "Sigmoid", "Softmax"
    };
}

std::unique_ptr<operators::BaseOperator> CudaOperatorMapper::createOperator(const std::string& op_type) {
    std::cout << "Creating operator: " << op_type << std::endl;

    if (op_type == "Input") {
        return std::make_unique<InputOperator>();
    } else if (op_type == "Conv") {
        return std::make_unique<ConvOperator>();
    } else if (op_type == "BatchNormalization") {
        return std::make_unique<BatchNormOperator>();
    } else if (op_type == "Relu") {
        return std::make_unique<ReLUOperator>();
    } else if (op_type == "YOLODetect") {
        return std::make_unique<YOLODetectOperator>();
    }

    std::cerr << "Unsupported operator type: " << op_type << std::endl;
    return nullptr;
}

void CudaOperatorMapper::initializeMapping() {
    operator_mapping_["Conv"] = "Conv2D";
    operator_mapping_["BatchNormalization"] = "BatchNorm";
    operator_mapping_["Relu"] = "ReLU";
    operator_mapping_["Add"] = "ElementwiseAdd";
    operator_mapping_["Mul"] = "ElementwiseMul";
    operator_mapping_["MatMul"] = "MatrixMultiply";
    operator_mapping_["MaxPool"] = "MaxPool2D";
    operator_mapping_["AveragePool"] = "AvgPool2D";
}

} // namespace yolo
} // namespace cuda_learning
