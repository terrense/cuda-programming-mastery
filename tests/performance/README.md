# 核函数模块性能测试

本目录包含CUDA核函数模块的性能测试，用于验证各种内存访问模式和线程配置优化的效果。

## 测试文件说明

### 1. memory_access_benchmarks.cu
**内存访问模式基准测试**

测试内容：
- 合并内存访问 vs 非合并访问性能对比
- 跨步内存访问对性能的影响
- 随机内存访问模式分析
- 内存对齐对性能的影响
- 共享内存优化效果验证

主要功能：
- `benchmarkCoalescedAccess()` - 合并访问基准测试
- `benchmarkStridedAccess()` - 跨步访问基准测试
- `benchmarkRandomAccess()` - 随机访问基准测试
- `benchmarkMemoryAlignment()` - 内存对齐影响测试
- `benchmarkSharedMemoryOptimization()` - 共享内存优化测试

### 2. thread_config_optimization.cu
**线程配置优化验证测试**

测试内容：
- 不同块大小对性能的影响
- 网格大小优化验证
- 2D线程配置优化
- 共享内存使用对线程配置的影响
- 计算密集型 vs 内存密集型核函数的最优配置
- 占用率与性能关系验证

主要功能：
- `testBlockSizeOptimization()` - 块大小优化测试
- `testGridSizeOptimization()` - 网格大小优化测试
- `test2DThreadConfiguration()` - 2D线程配置测试
- `testSharedMemoryImpactOnThreadConfig()` - 共享内存影响测试
- `testComputeVsMemoryBoundOptimization()` - 计算/内存密集型优化测试
- `testOccupancyPerformanceRelationship()` - 占用率性能关系测试

### 3. kernel_performance_test_suite.cu
**综合性能测试套件**

提供统一的测试入口，包含：
- 完整的测试流程管理
- 设备信息检测和显示
- 综合性能分析
- 测试报告生成
- 性能优化建议

## 编译和运行

### 使用CMake编译

```bash
# 在项目根目录下
mkdir build && cd build
cmake ..
make

# 运行单独的测试
./memory_access_benchmarks
./thread_config_optimization

# 运行完整测试套件
./kernel_performance_test_suite
```

### 使用nvcc直接编译

```bash
# 编译内存访问基准测试
nvcc -o memory_access_benchmarks tests/performance/memory_access_benchmarks.cu -I src/core

# 编译线程配置优化测试
nvcc -o thread_config_optimization tests/performance/thread_config_optimization.cu -I src/core

# 编译完整测试套件
nvcc -o kernel_performance_test_suite tests/performance/kernel_performance_test_suite.cu -I src/core
```

## 测试输出说明

### 内存访问模式测试输出
```
=== 合并内存访问基准测试 ===
    块大小    执行时间(ms)      带宽(GB/s)   占用率%
--------------------------------------------------
        64           2.145          156.32      75.0
       128           1.892          184.67      87.5
       256           1.756          198.45     100.0
       512           1.823          191.23      93.8
      1024           2.034          171.56      81.3
```

### 线程配置优化测试输出
```
=== 块大小优化测试 ===
    块大小    执行时间(ms)   占用率%    有效带宽    效率指标
------------------------------------------------------------
        64           2.145      75.0        156.32      117.24
       128           1.892      87.5        184.67      161.59
       256           1.756     100.0        198.45      198.45
       512           1.823      93.8        191.23      179.37
      1024           2.034      81.3        171.56      139.46

最佳块大小: 256, 最佳时间: 1.756 ms
优化器建议: Grid(4096, 1, 1), Block(256, 1, 1), SharedMem: 0
```

## 性能指标说明

### 关键指标
- **执行时间**: 核函数平均执行时间（毫秒）
- **带宽**: 有效内存带宽（GB/s）
- **占用率**: GPU占用率百分比
- **效率指标**: 综合性能指标（占用率 × 带宽）

### 性能分析
- **合并访问 vs 跨步访问**: 合并访问通常比跨步访问快2-10倍
- **最优块大小**: 通常在128-512之间，取决于具体核函数
- **占用率影响**: 高占用率不一定意味着高性能，需要综合考虑
- **共享内存效果**: 对于重复访问的数据，共享内存可以显著提升性能

## 测试要求

### 硬件要求
- CUDA兼容的GPU（计算能力3.0+）
- 至少1GB GPU内存

### 软件要求
- CUDA Toolkit 10.0+
- C++11兼容编译器
- CMake 3.10+（如果使用CMake编译）

## 故障排除

### 常见问题
1. **编译错误**: 检查CUDA路径和头文件包含
2. **运行时错误**: 确保GPU有足够内存
3. **性能异常**: 检查GPU是否被其他进程占用

### 调试建议
- 使用`nvidia-smi`监控GPU状态
- 检查CUDA错误码
- 验证内存分配和释放
- 使用Nsight工具进行详细分析

## 扩展测试

可以通过修改以下参数来扩展测试：
- `DEFAULT_DATA_SIZE`: 调整测试数据大小
- `DEFAULT_ITERATIONS`: 调整测试迭代次数
- 添加新的核函数进行测试
- 修改块大小和网格大小范围

## 参考资料

- CUDA C++ Programming Guide
- CUDA Best Practices Guide
- GPU性能优化相关论文和文档
