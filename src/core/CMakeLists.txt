# Multi-GPU Framework CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

# 查找CUDA
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

# 查找NCCL库
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT}/include)
find_library(NCCL_LIBRARY nccl HINTS ${NCCL_ROOT}/lib ${NCCL_ROOT}/lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(WARNING "NCCL not found. Multi-GPU functionality will be limited.")
    set(NCCL_FOUND FALSE)
else()
    set(NCCL_FOUND TRUE)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
endif()

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

if(NCCL_FOUND)
    include_directories(${NCCL_INCLUDE_DIR})
    add_definitions(-DUSE_NCCL)
endif()

# 源文件
set(MULTI_GPU_SOURCES
    multi_gpu_framework.cu
    cuda_environment.cpp
    error_handler.cpp
    performance_analyzer.cu
    memory_management.cu
)

# 头文件
set(MULTI_GPU_HEADERS
    multi_gpu_framework.h
    cuda_environment.h
    error_handler.h
    performance_analyzer.h
    memory_management.h
)

# 创建多GPU框架库
add_library(multi_gpu_framework STATIC ${MULTI_GPU_SOURCES})

# 设置目标属性
set_target_properties(multi_gpu_framework PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 链接库
target_link_libraries(multi_gpu_framework
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
)

if(NCCL_FOUND)
    target_link_libraries(multi_gpu_framework ${NCCL_LIBRARY})
endif()

# 安装规则
install(TARGETS multi_gpu_framework
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${MULTI_GPU_HEADERS}
    DESTINATION include/cuda_learning/core
)

# 创建演示程序
if(BUILD_EXAMPLES)
    # 多GPU演示程序
    add_executable(multi_gpu_demo
        ../../examples/advanced/multi_gpu_demo.cu
    )

    set_target_properties(multi_gpu_demo PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    target_link_libraries(multi_gpu_demo
        multi_gpu_framework
        ${CUDA_LIBRARIES}
    )

    if(NCCL_FOUND)
        target_link_libraries(multi_gpu_demo ${NCCL_LIBRARY})
    endif()

    # 安装演示程序
    install(TARGETS multi_gpu_demo
        RUNTIME DESTINATION bin
    )
endif()

# 创建测试程序
if(BUILD_TESTS)
    enable_testing()

    # 多GPU单元测试
    add_executable(test_multi_gpu
        ../../tests/unit/test_multi_gpu_framework.cpp
    )

    target_link_libraries(test_multi_gpu
        multi_gpu_framework
        ${CUDA_LIBRARIES}
    )

    if(NCCL_FOUND)
        target_link_libraries(test_multi_gpu ${NCCL_LIBRARY})
    endif()

    add_test(NAME MultiGPUFrameworkTest COMMAND test_multi_gpu)
endif()

# 打印配置信息
message(STATUS "Multi-GPU Framework Configuration:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  NCCL Found: ${NCCL_FOUND}")
if(NCCL_FOUND)
    message(STATUS "  NCCL Include: ${NCCL_INCLUDE_DIR}")
    message(STATUS "  NCCL Library: ${NCCL_LIBRARY}")
endif()
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
