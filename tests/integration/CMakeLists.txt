# YOLO Integration Tests CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# YOLO system source files
set(YOLO_SOURCES
    ../../src/yolo/model_parser/onnx_parser.cpp
    ../../src/yolo/model_parser/pytorch_loader.cpp
    ../../src/yolo/model_parser/model_graph.cpp
    ../../src/yolo/inference/batch_inference_manager.cpp
    ../../src/yolo/inference/multi_stream_inference.cpp
    ../../src/yolo/inference/operator_fusion.cpp
    ../../src/yolo/inference/fusion_kernels.cu
    ../../src/yolo/tensorrt/tensorrt_engine.cpp
    ../../src/yolo/monitoring/performance_monitor.cpp
    ../../src/yolo/utils/gpu_memory_manager.cpp
    ../../src/core/error_handler.cpp
    ../../src/operators/tensor.cpp
)

# YOLO Integration Tests executable
add_executable(yolo_integration_tests
    yolo_integration_tests.cpp
    ${YOLO_SOURCES}
)

target_link_libraries(yolo_integration_tests
    ${GTEST_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    pthread
)

# YOLO Performance Benchmarks executable
add_executable(yolo_performance_benchmarks
    yolo_performance_benchmarks.cpp
    ${YOLO_SOURCES}
)

target_link_libraries(yolo_performance_benchmarks
    ${GTEST_LIBRARIES}
    benchmark::benchmark
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    pthread
)

# Set CUDA architectures
set_property(TARGET yolo_integration_tests PROPERTY CUDA_ARCHITECTURES 75 80 86)
set_property(TARGET yolo_performance_benchmarks PROPERTY CUDA_ARCHITECTURES 75 80 86)

# Test discovery
include(GoogleTest)
gtest_discover_tests(yolo_integration_tests)

# Custom test targets
add_custom_target(run_integration_tests
    COMMAND yolo_integration_tests --gtest_output=xml:integration_test_results.xml
    DEPENDS yolo_integration_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running YOLO integration tests"
)

add_custom_target(run_performance_benchmarks
    COMMAND yolo_performance_benchmarks --gtest_output=xml:benchmark_test_results.xml
    DEPENDS yolo_performance_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running YOLO performance benchmarks"
)

add_custom_target(run_detailed_benchmarks
    COMMAND yolo_performance_benchmarks --benchmark --benchmark_format=json --benchmark_out=detailed_benchmarks.json
    DEPENDS yolo_performance_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running detailed YOLO benchmarks with Google Benchmark"
)

# Regression testing target
add_custom_target(run_regression_tests
    COMMAND yolo_integration_tests --gtest_filter="*Regression*" --gtest_output=xml:regression_test_results.xml
    DEPENDS yolo_integration_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running YOLO regression tests"
)

# Combined test target
add_custom_target(run_all_yolo_tests
    DEPENDS run_integration_tests run_performance_benchmarks
    COMMENT "Running all YOLO integration and performance tests"
)

# Install test executables
install(TARGETS yolo_integration_tests yolo_performance_benchmarks
    RUNTIME DESTINATION bin/tests
)

# Install test data and fixtures
install(DIRECTORY fixtures/
    DESTINATION share/tests/fixtures
    OPTIONAL
)
