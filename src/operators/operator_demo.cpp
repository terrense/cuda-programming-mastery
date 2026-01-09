#include "operators.h"
#include <iostream>
#include <vector>

using namespace cuda_learning::operators;

int main() {
    std::cout << "=== CUDA Operator System Demo ===\n\n";

    try {
        // 初始化算子系统
        initializeOperatorSystem();

        // 显示系统信息
        std::cout << getOperatorSystemInfo() << "\n";

        // 列出可用算子
        listAvailableOperators();

        // 运行系统测试
        std::cout << "=== Running System Tests ===\n";
        bool test_passed = runOperatorSystemTests();

        if (test_passed) {
            std::cout << "\n✓ All tests passed successfully!\n";
        } else {
            std::cout << "\n✗ Some tests failed!\n";
        }

        // 演示高级用法
        std::cout << "\n=== Advanced Usage Demo ===\n";

        // 创建更复杂的张量
        std::cout << "Creating larger tensors...\n";
        FloatTensor large_a({1000, 1000}, DeviceType::GPU);
        FloatTensor large_b({1000, 1000}, DeviceType::GPU);

        large_a.uniform(0.0f, 1.0f);
        large_b.uniform(0.0f, 1.0f);

        std::cout << "Tensor A: " << large_a.toString() << "\n";
        std::cout << "Tensor B: " << large_b.toString() << "\n";

        // 使用调度器执行算子
        std::cout << "Executing addition using dispatcher...\n";

        OperatorContext context;
        context.setDevice(0); // 使用第一个GPU

        std::vector<FloatTensor> inputs = {large_a, large_b};
        std::vector<FloatTensor> outputs;

        // 推断输出形状
        std::vector<TensorShape> input_shapes = {large_a.shape(), large_b.shape()};
        auto output_shapes = OperatorDispatcher::inferOutputShapes("add", input_shapes, context);

        // 预分配输出张量
        outputs.resize(1);
        outputs[0] = FloatTensor(output_shapes[0], DeviceType::GPU);

        // 执行算子
        auto start = std::chrono::high_resolution_clock::now();
        OperatorDispatcher::execute("add", inputs, outputs, context);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Addition completed in " << duration.count() << " microseconds\n";

        // 验证结果的一小部分
        auto result_sample = outputs[0].slice(0, 0, 10).toHost();
        std::cout << "Sample results (first 10 elements): ";
        for (size_t i = 0; i < std::min(result_sample.size(), size_t(10)); ++i) {
            std::cout << std::fixed << std::setprecision(3) << result_sample[i] << " ";
        }
        std::cout << "\n";

        // 内存使用估计
        size_t memory_usage = OperatorDispatcher::estimateMemoryUsage("add", input_shapes, context);
        std::cout << "Estimated memory usage: " << (memory_usage / (1024*1024)) << " MB\n";

        std::cout << "\n=== Demo completed successfully! ===\n";

    } catch (const std::exception& e) {
        std::cerr << "Demo failed with error: " << e.what() << std::endl;
        return 1;
    }

    // 清理系统
    shutdownOperatorSystem();

    return 0;
}
