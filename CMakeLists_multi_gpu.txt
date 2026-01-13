# 简化的多GPU框架CMakeLists.txt (不依赖NCCL)

cmake_minimum_required(VERSION 3.18)
project(MultiGPUFramework LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构 (GTX 1650 支持计算能力7.5)
set(CMAKE_CUDA_ARCHITECTURES "75")

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core)
include_directories(${CUDA_INCLUDE_DIRS})

# 源文件 (简化版，不包含NCCL相关功能)
set(MULTI_GPU_SOURCES
    src/core/cuda_environment.cpp
    src/core/error_handler.cpp
)

# 创建简化的多GPU演示程序
add_executable(simple_multi_gpu_demo
    examples/advanced/simple_multi_gpu_demo.cu
    ${MULTI_GPU_SOURCES}
)

# 设置目标属性
set_target_properties(simple_multi_gpu_demo PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 链接库
target_link_libraries(simple_multi_gpu_demo
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# 打印配置信息
message(STATUS "简化多GPU框架配置:")
message(STATUS "  CUDA版本: ${CUDA_VERSION}")
message(STATUS "  CUDA架构: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  构建类型: ${CMAKE_BUILD_TYPE}")
