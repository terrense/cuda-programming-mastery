# YOLO监控系统 CMake配置
cmake_minimum_required(VERSION 3.18)
project(YOLOMonitoring LANGUAGES CXX CUDA)

# 查找CUDA
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 简单测试程序
add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test
    CUDA::cudart
    Threads::Threads
)

# 简化的瓶颈分析器测试
add_executable(simple_bottleneck_test simple_bottleneck_test.cu)
set_target_properties(simple_bottleneck_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)
target_link_libraries(simple_bottleneck_test
    CUDA::cudart
    Threads::Threads
)

message(STATUS "YOLO Monitoring Module Configuration:")
message(STATUS "  Simple test enabled")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
