# CUDA Operators Module CMakeLists.txt

# 设置最低CMake版本
cmake_minimum_required(VERSION 3.18)

# 查找CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 查找cuBLAS和cuDNN（可选）
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
find_library(CUDNN_LIBRARIES cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# 算子库源文件
set(OPERATOR_SOURCES
    tensor.cu
    base_operator.cpp
    example_operators.cu
    math_operators.cu
    conv_pool_operators.cu
    operators.cpp
)

# 算子库头文件
set(OPERATOR_HEADERS
    tensor.h
    base_operator.h
    example_operators.h
    math_operators.h
    conv_pool_operators.h
    operators.h
)

# 创建算子库
add_library(cuda_operators STATIC ${OPERATOR_SOURCES} ${OPERATOR_HEADERS})

# 设置目标属性
set_target_properties(cuda_operators PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 包含目录
target_include_directories(cuda_operators PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CUDA_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(cuda_operators PUBLIC
    CUDA::cudart
    CUDA::curand
    ${CUBLAS_LIBRARIES}
)

# 如果找到cuDNN，则链接
if(CUDNN_LIBRARIES)
    target_link_libraries(cuda_operators PUBLIC ${CUDNN_LIBRARIES})
    target_compile_definitions(cuda_operators PUBLIC HAVE_CUDNN)
endif()

# 编译选项
target_compile_options(cuda_operators PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
    >
)

# 创建演示程序
add_executable(operator_demo operator_demo.cpp)
add_executable(math_operators_demo math_operators_demo.cpp)
add_executable(conv_pool_demo conv_pool_demo.cpp)

# 链接演示程序
target_link_libraries(operator_demo
    cuda_operators
    cuda_core  # 假设核心库名为cuda_core
)

target_link_libraries(math_operators_demo
    cuda_operators
    cuda_core  # 假设核心库名为cuda_core
)

target_link_libraries(conv_pool_demo
    cuda_operators
    cuda_core  # 假设核心库名为cuda_core
)

# 设置演示程序属性
set_target_properties(operator_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(math_operators_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(conv_pool_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 安装目标
install(TARGETS cuda_operators operator_demo math_operators_demo conv_pool_demo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(FILES ${OPERATOR_HEADERS}
    DESTINATION include/cuda_learning/operators
)
