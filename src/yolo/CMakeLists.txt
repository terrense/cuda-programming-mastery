# YOLO GPU加速系统 CMake配置

cmake_minimum_required(VERSION 3.18)
project(YOLOAccelerator LANGUAGES CXX CUDA)

# 查找CUDA
find_package(CUDAToolkit REQUIRED)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加必要的operators和core源文件
set(OPERATORS_SOURCES
    ../operators/tensor.cu
    ../operators/base_operator.cpp
)

set(CORE_SOURCES
    ../core/error_handler.cpp
)

# YOLO模型解析器库
set(YOLO_PARSER_SOURCES
    model_parser/onnx_parser.cpp
    model_parser/pytorch_loader.cpp
    model_parser/model_graph.cpp
)

add_library(yolo_parser STATIC ${YOLO_PARSER_SOURCES} ${OPERATORS_SOURCES} ${CORE_SOURCES})
target_link_libraries(yolo_parser CUDA::cudart)

# GPU内存管理库
set(YOLO_MEMORY_SOURCES
    utils/gpu_memory_manager.cpp
)

add_library(yolo_memory STATIC ${YOLO_MEMORY_SOURCES})
target_link_libraries(yolo_memory CUDA::cudart)

# 添加推理优化子目录
add_subdirectory(inference)

# 添加监控系统子目录
add_subdirectory(monitoring)

# 简化的YOLO演示程序
add_executable(simple_yolo_demo simple_yolo_demo.cpp)
target_link_libraries(simple_yolo_demo
    yolo_parser
    yolo_memory
    CUDA::cudart
)

# 完整的YOLO模型加载器演示程序
add_executable(yolo_model_loader_demo yolo_model_loader_demo.cpp)
target_link_libraries(yolo_model_loader_demo
    yolo_parser
    yolo_memory
    CUDA::cudart
)

# 查找线程库
find_package(Threads REQUIRED)
target_link_libraries(simple_yolo_demo Threads::Threads)
target_link_libraries(yolo_model_loader_demo Threads::Threads)

# 打印配置信息
message(STATUS "YOLO Module Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Inference Optimization: Enabled")
