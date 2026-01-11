cmake_minimum_required(VERSION 3.18)

# 设置项目名称
set(PROJECT_NAME yolo_inference_optimization)

# 查找CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# 查找cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib64 lib)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    set(CUDNN_FOUND TRUE)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
else()
    set(CUDNN_FOUND FALSE)
    message(WARNING "cuDNN not found, some features may be disabled")
endif()

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${CUDA_INCLUDE_DIRS})

if(CUDNN_FOUND)
    include_directories(${CUDNN_INCLUDE_DIR})
    add_definitions(-DUSE_CUDNN)
endif()

# 源文件
set(INFERENCE_SOURCES
    batch_inference_manager.cpp
    operator_fusion.cpp
    multi_stream_inference.cpp
    fusion_kernels.cu
)

set(DEMO_SOURCES
    yolo_inference_pipeline_demo.cpp
)

# 依赖的库文件
set(DEPENDENCY_SOURCES
    ../utils/gpu_memory_manager.cpp
    ../../core/error_handler.cpp
    ../../operators/tensor.cu
    ../../operators/base_operator.cpp
)

# 创建推理优化库
add_library(yolo_inference_opt STATIC ${INFERENCE_SOURCES} ${DEPENDENCY_SOURCES})

# 设置库的属性
set_target_properties(yolo_inference_opt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 链接库
target_link_libraries(yolo_inference_opt
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
)

if(CUDNN_FOUND)
    target_link_libraries(yolo_inference_opt ${CUDNN_LIBRARY})
endif()

# 创建演示程序
add_executable(yolo_inference_demo ${DEMO_SOURCES})

# 设置可执行文件属性
set_target_properties(yolo_inference_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 链接演示程序
target_link_libraries(yolo_inference_demo
    yolo_inference_opt
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
)

if(CUDNN_FOUND)
    target_link_libraries(yolo_inference_demo ${CUDNN_LIBRARY})
endif()

# 添加线程支持
find_package(Threads REQUIRED)
target_link_libraries(yolo_inference_opt Threads::Threads)
target_link_libraries(yolo_inference_demo Threads::Threads)

# 编译选项
target_compile_options(yolo_inference_opt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)

target_compile_options(yolo_inference_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)

# 安装规则
install(TARGETS yolo_inference_opt yolo_inference_demo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(FILES
    batch_inference_manager.h
    operator_fusion.h
    multi_stream_inference.h
    DESTINATION include/yolo/inference
)

# 创建测试目标（可选）
option(BUILD_TESTS "Build inference optimization tests" OFF)

if(BUILD_TESTS)
    # 这里可以添加测试相关的代码
    message(STATUS "Building inference optimization tests")
endif()

# 打印配置信息
message(STATUS "=== YOLO Inference Optimization Configuration ===")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "cuDNN Found: ${CUDNN_FOUND}")
if(CUDNN_FOUND)
    message(STATUS "cuDNN Include: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN Library: ${CUDNN_LIBRARY}")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "===============================================")
