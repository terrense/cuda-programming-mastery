# 第1课：CUDA开发环境搭建和基础框架

## 🎯 学习目标

通过本课学习，你将掌握：
- CUDA开发环境的完整搭建过程
- GPU硬件检测和配置方法
- CUDA程序的基本结构和编译流程
- 错误处理和调试的最佳实践

## 📚 理论基础

### 什么是CUDA？

**CUDA (Compute Unified Device Architecture)** 是NVIDIA开发的并行计算平台和编程模型。它允许开发者使用GPU进行通用计算，而不仅仅是图形渲染。

### CUDA的核心概念

1. **主机(Host)**: CPU及其内存
2. **设备(Device)**: GPU及其内存
3. **核函数(Kernel)**: 在GPU上并行执行的函数
4. **线程层次结构**: Grid → Block → Thread

### GPU vs CPU架构对比

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数量 | 少(2-32个) | 多(数百到数千个) |
| 核心复杂度 | 复杂，功能强大 | 简单，专门用于计算 |
| 缓存 | 大容量多级缓存 | 小容量缓存 |
| 适用场景 | 复杂串行任务 | 大规模并行计算 |
| 内存带宽 | 相对较低 | 非常高 |

## 🛠️ 实践操作

### 1. 环境检测

你已经运行过的环境检测程序展示了：

```bash
./build/examples/basic/cuda_env_test
```

**输出解析**：
```
✓ CUDA is available                    # CUDA运行时可用
✓ CUDA installation is valid           # CUDA安装有效
CUDA Runtime Version: 11.5             # CUDA运行时版本
CUDA Driver Version: 13.0              # CUDA驱动版本
Number of GPUs: 1                      # GPU数量

GPU 0: NVIDIA GeForce GTX 1650         # GPU型号
  Compute Capability: 7.5               # 计算能力
  Total Memory: 4095 MB                 # 显存大小
  Multiprocessors: 16                   # 流式多处理器数量
  Max Threads per Block: 1024           # 每个线程块最大线程数
```

### 2. 计算能力详解

你的GTX 1650的计算能力是**7.5**，这意味着：

- **架构**: Turing (图灵)
- **支持特性**:
  - 统一内存
  - 动态并行
  - 表面内存和纹理内存
  - 协作组
  - 独立线程调度

### 3. 硬件规格分析

**你的GPU规格**：
- **流式多处理器(SM)**: 16个
- **CUDA核心**: 约1024个 (每个SM约64个核心)
- **显存**: 4GB GDDR6
- **内存带宽**: 约128 GB/s
- **最大线程数**: 16,384个并发线程

## 💡 关键概念详解

### 1. 流式多处理器(SM)

每个SM包含：
- **CUDA核心**: 执行浮点和整数运算
- **特殊功能单元(SFU)**: 执行sin、cos、sqrt等函数
- **共享内存**: 64KB，SM内线程间共享
- **寄存器文件**: 65536个32位寄存器
- **线程束调度器**: 管理32个线程为一组的执行

### 2. 内存层次结构

```
线程私有：
├── 寄存器 (Registers)          # 最快，每个线程私有
│
线程块共享：
├── 共享内存 (Shared Memory)     # 很快，线程块内共享
├── L1缓存 (L1 Cache)           # 快，硬件管理
│
全局共享：
├── L2缓存 (L2 Cache)           # 中等，所有SM共享
├── 全局内存 (Global Memory)     # 慢但容量大
├── 常量内存 (Constant Memory)   # 只读，有缓存
└── 纹理内存 (Texture Memory)    # 只读，空间局部性优化
```

### 3. 线程层次结构

```
Grid (网格)
├── Block 0 (线程块0)
│   ├── Thread 0 (线程0)
│   ├── Thread 1 (线程1)
│   └── ...
├── Block 1 (线程块1)
│   ├── Thread 0 (线程0)
│   ├── Thread 1 (线程1)
│   └── ...
└── ...
```

## 🔧 开发环境组件

### 1. CUDA工具链

- **nvcc**: CUDA编译器
- **cuda-gdb**: CUDA调试器
- **nvprof**: 性能分析器
- **nvidia-smi**: GPU监控工具

### 2. 项目结构

```
CUDA_programmming/
├── src/core/                   # 核心组件
│   ├── cuda_environment.*     # 环境检测和配置
│   └── error_handler.*        # 错误处理和日志
├── examples/basic/             # 基础示例
├── tests/                      # 测试套件
└── docs/                       # 文档
```

## 🚨 错误处理最佳实践

### CUDA错误检查宏

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)
```

### 常见错误类型

1. **编译时错误**:
   - 语法错误
   - 架构不兼容
   - 头文件缺失

2. **运行时错误**:
   - 内存不足
   - 无效的内存访问
   - 核函数启动失败

3. **逻辑错误**:
   - 线程索引计算错误
   - 数据竞争
   - 同步问题

## 🎓 学习检查点

完成本课后，你应该能够：

- [ ] 解释GPU和CPU的架构差异
- [ ] 识别你的GPU的关键规格参数
- [ ] 理解CUDA的内存层次结构
- [ ] 使用CUDA错误检查宏
- [ ] 配置CUDA开发环境

## 🔗 相关代码文件

- `src/core/cuda_environment.h/cpp` - 环境检测实现
- `src/core/error_handler.h/cpp` - 错误处理实现
- `examples/basic/cuda_env_test.cpp` - 环境测试示例

## 📝 练习题

1. **理论题**: 解释为什么GPU适合并行计算而CPU适合串行计算？
2. **实践题**: 修改环境检测程序，添加内存带宽的理论计算
3. **分析题**: 比较你的GPU和其他GPU型号的规格差异

## 🚀 下一步

完成本课学习后，继续学习：
- **第2课**: GPU架构和CUDA编程模型
- **第3课**: 第一个CUDA程序开发
