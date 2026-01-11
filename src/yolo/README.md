# YOLO GPU加速系统

本目录包含YOLO模型GPU加速的完整实现，包括：

## 组件结构
- `model_parser/` - 模型解析器（ONNX/PyTorch）
- `inference/` - 推理管道和优化
- `tensorrt/` - TensorRT集成和自定义插件
- `monitoring/` - 性能监控和调优工具
- `utils/` - 工具函数和辅助类

## 主要功能
1. **模型解析和加载**: 支持ONNX和PyTorch模型格式
2. **GPU推理管道**: 高效的批处理推理系统
3. **TensorRT优化**: 自动优化和自定义插件
4. **性能监控**: 实时性能分析和自动调优
5. **内存管理**: GPU内存池和优化策略

## 使用示例
```cpp
#include "yolo/yolo_accelerator.h"

// 创建YOLO加速器
YOLOAccelerator accelerator;

// 加载模型
accelerator.loadModel("yolo_model.onnx");

// 执行推理
auto detections = accelerator.inference(image);
```
