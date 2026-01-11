#include "tensorrt_yolo_accelerator.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace yolo::tensorrt;

/**
 * @brief TensorRT YOLO加速演示程序
 *
 * 演示TensorRT引擎构建、自定义插件和动态形状处理的完整功能
 */
class TensorRTDemo {
public:
    void runDemo() {
        std::cout << "=== TensorRT YOLO加速演示程序 ===" << std::endl;

        // 1. 演示引擎构建
        demonstrateEngineBuilding();

        // 2. 演示自定义插件
        demonstrateCustomPlugins();

        // 3. 演示动态形状处理
        demonstrateDynamicShapes();

        // 4. 演示完整推理流程
        demonstrateInferencePipeline();

        // 5. 演示性能基准测试
        demonstratePerformanceBenchmark();

        std::cout << "=== 演示程序完成 ===" << std::endl;
    }

private:
    void demonstrateEngineBuilding() {
        std::cout << "\n--- 1. TensorRT引擎构建演示 ---" << std::endl;

        try {
            // 创建引擎构建器
            TensorRTEngineBuilder builder;

            // 配置构建参数
            TensorRTEngineBuilder::BuildConfig config;
            config.max_batch_size = 8;
            config.enable_fp16 = true;
            config.enable_dynamic_shapes = true;
            config.max_workspace_size = 1ULL << 30; // 1GB

            // 添加动态形状配置
            TensorRTEngineBuilder::BuildConfig::DynamicShapeProfile profile;
            profile.tensor_name = "images";
            profile.min_shape = {1, 3, 320, 320};
            profile.opt_shape = {4, 3, 640, 640};
            profile.max_shape = {8, 3, 1280, 1280};
            config.dynamic_shapes.push_back(profile);

            std::cout << "✓ 引擎构建配置完成" << std::endl;
            std::cout << "  - 最大批次大小: " << config.max_batch_size << std::endl;
            std::cout << "  - FP16精度: " << (config.enable_fp16 ? "启用" : "禁用") << std::endl;
            std::cout << "  - 动态形状: " << (config.enable_dynamic_shapes ? "启用" : "禁用") << std::endl;
            std::cout << "  - 工作空间大小: " << config.max_workspace_size / (1024*1024) << " MB" << std::endl;

            // 注意: 实际的ONNX文件构建需要有效的模型文件
            std::cout << "✓ 引擎构建器创建成功" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "✗ 引擎构建演示失败: " << e.what() << std::endl;
        }
    }

    void demonstrateCustomPlugins() {
        std::cout << "\n--- 2. 自定义插件演示 ---" << std::endl;

        try {
            // 获取插件管理器
            auto& plugin_manager = PluginManager::getInstance();

            // 注册YOLO插件
            plugin_manager.registerYOLOPlugins();

            // 列出已注册的插件
            auto registered_plugins = plugin_manager.listRegisteredPlugins();
            std::cout << "✓ 已注册的自定义插件:" << std::endl;
            for (const auto& plugin_name : registered_plugins) {
                std::cout << "  - " << plugin_name << std::endl;
            }

            // 获取特定插件创建器
            auto yolo_plugin_creator = plugin_manager.getPluginCreator("YOLODetection");
            if (yolo_plugin_creator) {
                std::cout << "✓ YOLO检测插件创建器获取成功" << std::endl;
                std::cout << "  - 插件名称: " << yolo_plugin_creator->getPluginName() << std::endl;
                std::cout << "  - 插件版本: " << yolo_plugin_creator->getPluginVersion() << std::endl;
            }

            auto focus_plugin_creator = plugin_manager.getPluginCreator("Focus");
            if (focus_plugin_creator) {
                std::cout << "✓ Focus层插件创建器获取成功" << std::endl;
                std::cout << "  - 插件名称: " << focus_plugin_creator->getPluginName() << std::endl;
                std::cout << "  - 插件版本: " << focus_plugin_creator->getPluginVersion() << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "✗ 自定义插件演示失败: " << e.what() << std::endl;
        }
    }

    void demonstrateDynamicShapes() {
        std::cout << "\n--- 3. 动态形状处理演示 ---" << std::endl;

        try {
            // 创建模拟引擎 (实际应用中需要真实的TensorRT引擎)
            std::cout << "注意: 此演示需要有效的TensorRT引擎" << std::endl;

            // 演示形状配置
            DynamicShapeHandler::ShapeConfig shape_config;
            shape_config.tensor_name = "images";
            shape_config.min_shape = {1, 3, 320, 320};
            shape_config.opt_shape = {4, 3, 640, 640};
            shape_config.max_shape = {8, 3, 1280, 1280};

            std::cout << "✓ 动态形状配置:" << std::endl;
            std::cout << "  - 张量名称: " << shape_config.tensor_name << std::endl;
            std::cout << "  - 最小形状: [";
            for (size_t i = 0; i < shape_config.min_shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << shape_config.min_shape[i];
            }
            std::cout << "]" << std::endl;

            std::cout << "  - 优化形状: [";
            for (size_t i = 0; i < shape_config.opt_shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << shape_config.opt_shape[i];
            }
            std::cout << "]" << std::endl;

            std::cout << "  - 最大形状: [";
            for (size_t i = 0; i < shape_config.max_shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << shape_config.max_shape[i];
            }
            std::cout << "]" << std::endl;

            // 验证形状配置
            if (shape_config.isValid()) {
                std::cout << "✓ 形状配置验证通过" << std::endl;
            } else {
                std::cout << "✗ 形状配置验证失败" << std::endl;
            }

            // 测试形状范围检查
            std::vector<int> test_shape = {2, 3, 640, 640};
            if (shape_config.isInRange(test_shape)) {
                std::cout << "✓ 测试形状 [2, 3, 640, 640] 在有效范围内" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "✗ 动态形状处理演示失败: " << e.what() << std::endl;
        }
    }

    void demonstrateInferencePipeline() {
        std::cout << "\n--- 4. 推理流程演示 ---" << std::endl;

        try {
            // 创建加速器配置
            TensorRTYOLOAccelerator::AcceleratorConfig config;
            config.build_config.max_batch_size = 4;
            config.build_config.enable_fp16 = true;
            config.build_config.enable_dynamic_shapes = true;

            // 动态形状配置
            DynamicShapeHandler::ShapeConfig shape_config;
            shape_config.tensor_name = "images";
            shape_config.min_shape = {1, 3, 320, 320};
            shape_config.opt_shape = {2, 3, 640, 640};
            shape_config.max_shape = {4, 3, 1280, 1280};
            config.shape_configs.push_back(shape_config);

            // 批处理配置
            config.batch_config.enable_auto_batching = true;
            config.batch_config.max_batch_size = 4;
            config.batch_config.batch_timeout_ms = 10.0f;

            std::cout << "✓ 加速器配置完成" << std::endl;

            // 创建加速器实例
            auto accelerator = std::make_unique<TensorRTYOLOAccelerator>(config);

            std::cout << "✓ TensorRT YOLO加速器创建成功" << std::endl;

            // 演示配置设置
            accelerator->setConfidenceThreshold(0.5f);
            accelerator->setNMSThreshold(0.45f);
            accelerator->enableProfiling(true);

            std::cout << "✓ 推理参数配置完成" << std::endl;
            std::cout << "  - 置信度阈值: 0.5" << std::endl;
            std::cout << "  - NMS阈值: 0.45" << std::endl;
            std::cout << "  - 性能分析: 启用" << std::endl;

            // 模拟推理数据
            int width = 640, height = 640, channels = 3;
            std::vector<float> dummy_image(width * height * channels, 0.5f);

            std::cout << "✓ 模拟输入数据准备完成" << std::endl;
            std::cout << "  - 图像尺寸: " << width << "x" << height << "x" << channels << std::endl;

            // 注意: 实际推理需要初始化的引擎
            std::cout << "注意: 完整推理演示需要有效的ONNX模型文件" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "✗ 推理流程演示失败: " << e.what() << std::endl;
        }
    }

    void demonstratePerformanceBenchmark() {
        std::cout << "\n--- 5. 性能基准测试演示 ---" << std::endl;

        try {
            // 模拟性能测试数据
            std::vector<float> inference_times;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(5.0f, 15.0f);

            // 生成模拟推理时间
            for (int i = 0; i < 100; ++i) {
                inference_times.push_back(dis(gen));
            }

            // 计算统计信息
            float total_time = 0.0f;
            float min_time = inference_times[0];
            float max_time = inference_times[0];

            for (float time : inference_times) {
                total_time += time;
                min_time = std::min(min_time, time);
                max_time = std::max(max_time, time);
            }

            float avg_time = total_time / inference_times.size();
            float fps = 1000.0f / avg_time;

            std::cout << "✓ 性能基准测试结果 (模拟数据):" << std::endl;
            std::cout << "  - 测试次数: " << inference_times.size() << std::endl;
            std::cout << "  - 平均推理时间: " << avg_time << " ms" << std::endl;
            std::cout << "  - 最小推理时间: " << min_time << " ms" << std::endl;
            std::cout << "  - 最大推理时间: " << max_time << " ms" << std::endl;
            std::cout << "  - 平均FPS: " << fps << std::endl;

            // 批处理性能对比
            std::cout << "\n✓ 批处理性能对比 (模拟数据):" << std::endl;
            for (int batch_size = 1; batch_size <= 8; batch_size *= 2) {
                float batch_time = avg_time * batch_size * 0.7f; // 模拟批处理效率
                float batch_fps = batch_size * 1000.0f / batch_time;
                std::cout << "  - 批次大小 " << batch_size << ": "
                          << batch_time << " ms, " << batch_fps << " FPS" << std::endl;
            }

            // 精度对比
            std::cout << "\n✓ 精度模式性能对比 (模拟数据):" << std::endl;
            std::cout << "  - FP32: " << avg_time << " ms" << std::endl;
            std::cout << "  - FP16: " << avg_time * 0.6f << " ms (提升 "
                      << (1.0f - 0.6f) * 100 << "%)" << std::endl;
            std::cout << "  - INT8: " << avg_time * 0.4f << " ms (提升 "
                      << (1.0f - 0.4f) * 100 << "%)" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "✗ 性能基准测试演示失败: " << e.what() << std::endl;
        }
    }
};

int main() {
    try {
        TensorRTDemo demo;
        demo.runDemo();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "演示程序异常: " << e.what() << std::endl;
        return 1;
    }
}
