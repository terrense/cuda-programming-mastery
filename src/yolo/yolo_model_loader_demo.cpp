#include "model_parser/onnx_parser.h"
#include "model_parser/pytorch_loader.h"
#include "model_parser/model_graph.h"
#include "utils/gpu_memory_manager.h"
#include "../core/error_handler.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace cuda_learning::yolo;

void demonstrateONNXParser() {
    std::cout << "\n=== ONNX Parser Demonstration ===" << std::endl;

    ONNXParser parser;

    // 加载模型（使用虚拟路径，会创建演示模型）
    if (!parser.loadModel("yolo_model.onnx")) {
        std::cerr << "Failed to load ONNX model" << std::endl;
        return;
    }

    // 解析模型
    if (!parser.parseModel()) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return;
    }

    // 打印模型信息
    parser.printModelInfo();

    // 获取输入输出信息
    auto inputs = parser.getInputs();
    auto outputs = parser.getOutputs();

    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Inputs: " << inputs.size() << ", Outputs: " << outputs.size() << std::endl;

    // 测试权重获取
    void* weight_data;
    size_t weight_size;
    if (parser.getWeights("conv1_weight", &weight_data, &weight_size)) {
        std::cout << "Retrieved weight data: " << weight_size << " bytes" << std::endl;
    }
}

void demonstratePyTorchLoader() {
    std::cout << "\n=== PyTorch Loader Demonstration ===" << std::endl;

    PyTorchLoader loader;

    // 加载模型
    if (!loader.loadModel("yolo_model.pt")) {
        std::cerr << "Failed to load PyTorch model" << std::endl;
        return;
    }

    // 打印模型信息
    loader.printModelInfo();

    // 设置输入数据
    std::vector<float> input_data(1 * 3 * 640 * 640, 0.5f); // 填充虚拟数据
    std::vector<int> input_shape = {1, 3, 640, 640};

    if (!loader.setInput("input", input_data, input_shape)) {
        std::cerr << "Failed to set input data" << std::endl;
        return;
    }

    // 执行推理
    auto start_time = std::chrono::high_resolution_clock::now();
    auto outputs = loader.forward();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Generated " << outputs.size() << " output tensors" << std::endl;

    // 导出为ONNX
    if (loader.exportToONNX("exported_yolo.onnx", input_shape)) {
        std::cout << "Model exported to ONNX successfully" << std::endl;
    }
}

void demonstrateModelGraph() {
    std::cout << "\n=== Model Graph Demonstration ===" << std::endl;

    ModelGraph graph;

    // 构建一个简单的YOLO图结构
    std::cout << "Building model graph..." << std::endl;

    // 添加节点（使用支持的算子类型）
    graph.addNode("input", "Input");
    graph.addNode("conv1", "Conv");
    graph.addNode("bn1", "BatchNormalization");
    graph.addNode("relu1", "Relu");
    graph.addNode("conv2", "Conv");
    graph.addNode("detect", "YOLODetect");

    // 添加边（张量连接）
    graph.addEdge("input_tensor", "", {"conv1"});
    graph.addEdge("conv1_output", "conv1", {"bn1"});
    graph.addEdge("bn1_output", "bn1", {"relu1"});
    graph.addEdge("relu1_output", "relu1", {"conv2"});
    graph.addEdge("conv2_output", "conv2", {"detect"});
    graph.addEdge("detection_output", "detect", {});

    // 设置输入输出
    graph.setInputs({"input_tensor"});
    graph.setOutputs({"detection_output"});

    // 打印图信息
    graph.printGraph();

    // 验证图
    if (graph.validate()) {
        std::cout << "Graph validation passed" << std::endl;
    } else {
        std::cout << "Graph validation failed" << std::endl;
    }

    // 拓扑排序
    if (graph.topologicalSort()) {
        std::cout << "Topological sort completed" << std::endl;
    }

    // 形状推断
    std::map<std::string, cuda_learning::operators::TensorShape> input_shapes;
    input_shapes["input_tensor"] = cuda_learning::operators::TensorShape({1, 3, 640, 640});

    if (graph.inferShapes(input_shapes)) {
        std::cout << "Shape inference completed" << std::endl;
    }

    // 优化图
    if (graph.optimize()) {
        std::cout << "Graph optimization completed" << std::endl;
    }
}

void demonstrateMemoryManagement() {
    std::cout << "\n=== GPU Memory Management Demonstration ===" << std::endl;

    // 初始化全局内存管理器
    if (!GlobalMemoryManager::initialize(2ULL * 1024 * 1024 * 1024, // 2GB GPU内存限制
                                        1024 * 1024 * 1024)) {        // 1GB张量池
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return;
    }

    auto* gpu_mem_mgr = GlobalMemoryManager::getGPUMemoryManager();
    auto* weight_mgr = GlobalMemoryManager::getWeightDataManager();
    auto* tensor_pool = GlobalMemoryManager::getTensorMemoryPool();

    // 演示GPU内存分配
    std::cout << "Allocating GPU memory..." << std::endl;
    void* ptr1 = gpu_mem_mgr->allocate(1024 * 1024, "test_buffer_1"); // 1MB
    void* ptr2 = gpu_mem_mgr->allocate(2 * 1024 * 1024, "test_buffer_2"); // 2MB
    void* ptr3 = gpu_mem_mgr->allocate(512 * 1024, "test_buffer_3"); // 512KB

    gpu_mem_mgr->printMemoryInfo();

    // 演示权重数据管理
    std::cout << "Loading weight data..." << std::endl;
    std::vector<float> conv1_weights(3 * 64 * 3 * 3, 0.1f); // 卷积层权重
    std::vector<float> conv2_weights(64 * 128 * 3 * 3, 0.2f);

    weight_mgr->loadWeights("conv1.weight", conv1_weights.data(),
                           conv1_weights.size() * sizeof(float));
    weight_mgr->loadWeights("conv2.weight", conv2_weights.data(),
                           conv2_weights.size() * sizeof(float));

    weight_mgr->printWeightInfo();

    // 演示张量内存池
    std::cout << "Allocating tensor memory..." << std::endl;
    void* tensor1 = tensor_pool->allocateTensor({1, 3, 640, 640}, sizeof(float), "input_tensor");
    void* tensor2 = tensor_pool->allocateTensor({1, 64, 320, 320}, sizeof(float), "feature_map1");
    void* tensor3 = tensor_pool->allocateTensor({1, 128, 160, 160}, sizeof(float), "feature_map2");

    tensor_pool->printPoolInfo();

    // 清理内存
    std::cout << "Cleaning up memory..." << std::endl;
    gpu_mem_mgr->deallocate(ptr1);
    gpu_mem_mgr->deallocate(ptr2);
    gpu_mem_mgr->deallocate(ptr3);

    tensor_pool->deallocateTensor(tensor1);
    tensor_pool->deallocateTensor(tensor2);
    tensor_pool->deallocateTensor(tensor3);

    // 检查内存泄漏
    gpu_mem_mgr->checkMemoryLeaks();

    // 打印最终统计
    GlobalMemoryManager::printGlobalMemoryInfo();

    // 清理全局管理器
    GlobalMemoryManager::cleanup();
}

void demonstrateOperatorMapping() {
    std::cout << "\n=== CUDA Operator Mapping Demonstration ===" << std::endl;

    // 测试算子映射
    std::vector<std::string> onnx_ops = {"Conv", "BatchNormalization", "Relu", "Add", "MaxPool"};

    std::cout << "ONNX to CUDA operator mapping:" << std::endl;
    for (const auto& op : onnx_ops) {
        std::string cuda_op = CudaOperatorMapper::mapOperatorType(op);
        bool supported = CudaOperatorMapper::isOperatorSupported(op);

        std::cout << "  " << op << " -> " << cuda_op
                  << " (supported: " << (supported ? "yes" : "no") << ")" << std::endl;
    }

    // 获取支持的算子列表
    auto supported_ops = CudaOperatorMapper::getSupportedOperators();
    std::cout << "\nSupported operators (" << supported_ops.size() << "):" << std::endl;
    for (const auto& op : supported_ops) {
        std::cout << "  " << op << std::endl;
    }
}

int main() {
    std::cout << "YOLO Model Loader Demo Starting..." << std::endl;

    try {
        // 演示各个组件
        demonstrateONNXParser();
        demonstratePyTorchLoader();
        demonstrateModelGraph();
        demonstrateMemoryManagement();
        demonstrateOperatorMapping();

        std::cout << "\n=== Demo Completed Successfully ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
