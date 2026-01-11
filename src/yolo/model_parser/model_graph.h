#pragma once

#include "../../operators/tensor.h"
#include "../../operators/base_operator.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_set>

namespace cuda_learning {
namespace yolo {

// 前向声明
class ModelGraph;

// 图节点（表示一个算子）
class GraphNode {
public:
    std::string name;
    std::string op_type;
    std::unique_ptr<operators::BaseOperator> op;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    operators::OperatorContext context;

    // 执行状态
    bool executed = false;

    GraphNode(const std::string& node_name, const std::string& operator_type)
        : name(node_name), op_type(operator_type) {}
};

// 图边（表示张量连接）
class GraphEdge {
public:
    std::string tensor_name;
    std::string producer_node;
    std::vector<std::string> consumer_nodes;
    operators::TensorShape shape;
    operators::DataType dtype;

    GraphEdge(const std::string& name) : tensor_name(name) {}
};

// 模型计算图
class ModelGraph {
private:
    std::map<std::string, std::unique_ptr<GraphNode>> nodes_;
    std::map<std::string, std::unique_ptr<GraphEdge>> edges_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::string> execution_order_;

    // 张量存储
    std::map<std::string, operators::FloatTensor> tensors_;

public:
    ModelGraph() = default;
    ~ModelGraph() = default;

    // 添加节点
    bool addNode(const std::string& name, const std::string& op_type);

    // 添加边
    bool addEdge(const std::string& tensor_name, const std::string& producer,
                const std::vector<std::string>& consumers);

    // 设置输入输出
    void setInputs(const std::vector<std::string>& inputs) { input_names_ = inputs; }
    void setOutputs(const std::vector<std::string>& outputs) { output_names_ = outputs; }

    // 获取输入输出
    const std::vector<std::string>& getInputs() const { return input_names_; }
    const std::vector<std::string>& getOutputs() const { return output_names_; }

    // 拓扑排序
    bool topologicalSort();

    // 形状推断
    bool inferShapes(const std::map<std::string, operators::TensorShape>& input_shapes);

    // 执行图
    bool execute(const std::map<std::string, operators::FloatTensor>& inputs,
                std::map<std::string, operators::FloatTensor>& outputs);

    // 优化图
    bool optimize();

    // 获取节点
    GraphNode* getNode(const std::string& name);
    const GraphNode* getNode(const std::string& name) const;

    // 获取边
    GraphEdge* getEdge(const std::string& name);
    const GraphEdge* getEdge(const std::string& name) const;

    // 获取张量
    operators::FloatTensor* getTensor(const std::string& name);
    const operators::FloatTensor* getTensor(const std::string& name) const;

    // 图信息
    size_t getNodeCount() const { return nodes_.size(); }
    size_t getEdgeCount() const { return edges_.size(); }

    // 打印图信息
    void printGraph() const;

    // 验证图的有效性
    bool validate() const;

    // 清空图
    void clear();

private:
    // 辅助函数
    bool hasNode(const std::string& name) const;
    bool hasEdge(const std::string& name) const;
    std::vector<std::string> getNodeInputs(const std::string& node_name) const;
    std::vector<std::string> getNodeOutputs(const std::string& node_name) const;
    bool hasCycle() const;
    void dfsVisit(const std::string& node_name, std::unordered_set<std::string>& visited,
                  std::unordered_set<std::string>& rec_stack, std::vector<std::string>& order) const;
};

// CUDA算子映射器
class CudaOperatorMapper {
public:
    // 将ONNX/PyTorch算子映射到CUDA算子
    static std::string mapOperatorType(const std::string& original_type);

    // 检查算子是否支持
    static bool isOperatorSupported(const std::string& op_type);

    // 获取支持的算子列表
    static std::vector<std::string> getSupportedOperators();

    // 创建算子实例
    static std::unique_ptr<operators::BaseOperator> createOperator(const std::string& op_type);

private:
    static std::map<std::string, std::string> operator_mapping_;
    static void initializeMapping();
};

} // namespace yolo
} // namespace cuda_learning
