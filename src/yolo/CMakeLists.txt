# YOLO GPU加速系统 CMake配置

cmake_minimum_required(VERSION 3.18)
project(YOLOAccelerator)

# 查找CUDA
find_package(CUDA REQUIRED)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CUDA_INCLUDE_DIRS})

# YOLO模型解析器库
set(YOLO_PARSER_SOURCES
    model_parser/onnx_parser.cpp
    model_parser/pytorch_loader.cpp
    model_parser/model_graph.cpp
)

add_library(yolo_parser STATIC ${YOLO_PARSER_SOURCES})

# GPU内存管理库
set(YOLO_MEMORY_SOURCES
    utils/gpu_memory_manager.cpp
)

add_library(yolo_memory STATIC ${YOLO_MEMORY_SOURCES})
target_link_libraries(yolo_memory ${CUDA_LIBRARIES})

# 简化的YOLO演示程序
add_executable(simple_yolo_demo simple_yolo_demo.cpp)
target_link_libraries(simple_yolo_demo
    yolo_parser
    yolo_memory
    ${CUDA_LIBRARIES}
)

# 完整的YOLO模型加载器演示程序
add_executable(yolo_model_loader_demo yolo_model_loader_demo.cpp)
target_link_libraries(yolo_model_loader_demo
    yolo_parser
    yolo_memory
    ${CUDA_LIBRARIES}
)

# 打印配置信息
message(STATUS "YOLO Module Configuration:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
