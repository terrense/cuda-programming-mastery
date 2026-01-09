#pragma once

// 算子框架核心组件
#include "tensor.h"
#include "base_operator.h"
#include "example_operators.h"
#include "math_operators.h"
#include "conv_pool_operators.h"

namespace cuda_learning {
namespace operators {

// 初始化算子系统
void initializeOperatorSystem();

// 清理算子系统
void shutdownOperatorSystem();

// 获取系统信息
std::string getOperatorSystemInfo();

// 列出所有可用算子
void listAvailableOperators();

// 运行算子系统测试
bool runOperatorSystemTests();

} // namespace operators
} // namespace cuda_learning
