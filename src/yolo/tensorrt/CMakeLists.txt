# TensorRT集成模块 CMake配置

cmake_minimum_required(VERSION 3.18)
project(YOLOTensorRT LANGUAGES CXX CUDA)

# 查找必要的依赖
find_package(CUDAToolkit REQUIRED)

# 查找TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_PARSER nvonnxparser
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# 检查TensorRT是否找到
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER)
    message(WARNING "TensorRT not found. TensorRT integration will be disabled.")
    return()
endif()

# 设置包含目录
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# TensorRT引擎构建器库
set(TENSORRT_ENGINE_SOURCES
    tensorrt_engine_builder.cpp
    tensorrt_engine_builder.h
)

add_library(tensorrt_engine STATIC ${TENSORRT_ENGINE_SOURCES})
target_link_libraries(tensorrt_engine
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_PARSER}
    CUDA::cudart
)

# 自定义插件库
set(TENSORRT_PLUGIN_SOURCES
    custom_plugins.cpp
    custom_plugins.h
    plugin_kernels.cu
    plugin_kernels.cuh
)

add_library(tensorrt_plugins STATIC ${TENSORRT_PLUGIN_SOURCES})
target_link_libraries(tensorrt_plugins
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    CUDA::cudart
    CUDA::cub
)

# 动态形状处理库
set(DYNAMIC_SHAPE_SOURCES
    dynamic_shape_handler.cpp
    dynamic_shape_handler.h
)

add_library(dynamic_shape STATIC ${DYNAMIC_SHAPE_SOURCES})
target_link_libraries(dynamic_shape
    ${TENSORRT_LIBRARY_INFER}
    CUDA::cudart
)

# 主TensorRT加速器库
set(TENSORRT_ACCELERATOR_SOURCES
    tensorrt_yolo_accelerator.cpp
    tensorrt_yolo_accelerator.h
)

add_library(tensorrt_yolo_accelerator STATIC ${TENSORRT_ACCELERATOR_SOURCES})
target_link_libraries(tensorrt_yolo_accelerator
    tensorrt_engine
    tensorrt_plugins
    dynamic_shape
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${TENSORRT_LIBRARY_PARSER}
    CUDA::cudart
)

# 添加必要的依赖库
find_package(Threads REQUIRED)
target_link_libraries(tensorrt_yolo_accelerator Threads::Threads)

# 链接YOLO工具库
if(TARGET yolo_memory)
    target_link_libraries(tensorrt_yolo_accelerator yolo_memory)
endif()

# 设置编译选项
target_compile_options(tensorrt_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

target_compile_options(tensorrt_plugins PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

target_compile_options(dynamic_shape PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_compile_options(tensorrt_yolo_accelerator PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# 设置CUDA编译选项
set_property(TARGET tensorrt_plugins PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 创建TensorRT演示程序
add_executable(tensorrt_yolo_demo tensorrt_demo.cpp)
target_link_libraries(tensorrt_yolo_demo
    tensorrt_yolo_accelerator
    tensorrt_engine
    tensorrt_plugins
    dynamic_shape
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${TENSORRT_LIBRARY_PARSER}
    CUDA::cudart
    Threads::Threads
)

# 安装目标
install(TARGETS tensorrt_engine tensorrt_plugins dynamic_shape tensorrt_yolo_accelerator
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    tensorrt_engine_builder.h
    custom_plugins.h
    dynamic_shape_handler.h
    tensorrt_yolo_accelerator.h
    DESTINATION include/yolo/tensorrt
)

# 打印配置信息
message(STATUS "TensorRT Module Configuration:")
message(STATUS "  TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  TensorRT Infer: ${TENSORRT_LIBRARY_INFER}")
message(STATUS "  TensorRT Plugin: ${TENSORRT_LIBRARY_INFER_PLUGIN}")
message(STATUS "  TensorRT Parser: ${TENSORRT_LIBRARY_PARSER}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")

# 添加编译定义
target_compile_definitions(tensorrt_yolo_accelerator PRIVATE
    TENSORRT_VERSION_MAJOR=8
    CUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
)

# 设置输出目录
set_target_properties(tensorrt_engine tensorrt_plugins dynamic_shape tensorrt_yolo_accelerator
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
