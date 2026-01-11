# Tests CMakeLists.txt

# Find testing framework (using simple approach for now)
# In a real project, you might want to use Google Test or similar

# Unit tests
file(GLOB_RECURSE TEST_SOURCES "unit/*.cpp" "unit/*.cu")
if(TEST_SOURCES)
    add_executable(unit_tests ${TEST_SOURCES})
    target_link_libraries(unit_tests cuda_learning_core)
    target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/core)

    # Set CUDA properties for tests that use CUDA
    set_target_properties(unit_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Enable CUDA language for the test target
    set_property(TARGET unit_tests PROPERTY CUDA_STANDARD 17)

    add_test(NAME UnitTests COMMAND unit_tests)
endif()

# Integration tests
file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "integration/*.cpp" "integration/*.cu")
if(INTEGRATION_TEST_SOURCES)
    add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})
    target_link_libraries(integration_tests cuda_learning_core)
    target_include_directories(integration_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/core)

    set_target_properties(integration_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    add_test(NAME IntegrationTests COMMAND integration_tests)
endif()

# Performance tests
file(GLOB_RECURSE PERF_TEST_SOURCES "performance/*.cpp" "performance/*.cu")
if(PERF_TEST_SOURCES)
    add_executable(performance_tests ${PERF_TEST_SOURCES})
    target_link_libraries(performance_tests cuda_learning_core)
    target_include_directories(performance_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/core)

    set_target_properties(performance_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    add_test(NAME PerformanceTests COMMAND performance_tests)
endif()

# Individual performance test executables for easier debugging
add_executable(memory_access_benchmarks tests/performance/memory_access_benchmarks.cu)
target_link_libraries(memory_access_benchmarks cuda_learning_core)
target_include_directories(memory_access_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
set_target_properties(memory_access_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(thread_config_optimization tests/performance/thread_config_optimization.cu)
target_link_libraries(thread_config_optimization cuda_learning_core)
target_include_directories(thread_config_optimization PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
set_target_properties(thread_config_optimization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Add individual performance tests
add_test(NAME MemoryAccessBenchmarks COMMAND memory_access_benchmarks)
add_test(NAME ThreadConfigOptimization COMMAND thread_config_optimization)

# Kernel performance test suite
add_executable(kernel_performance_test_suite tests/performance/kernel_performance_test_suite.cu)
target_link_libraries(kernel_performance_test_suite cuda_learning_core)
target_include_directories(kernel_performance_test_suite PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
set_target_properties(kernel_performance_test_suite PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_test(NAME KernelPerformanceTestSuite COMMAND kernel_performance_test_suite)

# Individual test executables for easier debugging
add_executable(test_cuda_environment unit/test_cuda_environment.cpp)
target_link_libraries(test_cuda_environment cuda_learning_core)
target_include_directories(test_cuda_environment PRIVATE ${CMAKE_SOURCE_DIR}/src/core)

add_executable(test_basic_examples unit/test_basic_examples.cpp)
target_link_libraries(test_basic_examples cuda_learning_core)
target_include_directories(test_basic_examples PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
set_target_properties(test_basic_examples PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(test_example_compilation unit/test_example_compilation.cpp)
target_link_libraries(test_example_compilation cuda_learning_core)
target_include_directories(test_example_compilation PRIVATE ${CMAKE_SOURCE_DIR}/src/core)

add_executable(test_example_execution unit/test_example_execution.cpp)
target_link_libraries(test_example_execution cuda_learning_core)
target_include_directories(test_example_execution PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
set_target_properties(test_example_execution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Custom operator unit tests
add_executable(test_math_operators unit/test_math_operators.cu)
target_link_libraries(test_math_operators cuda_learning_core)
target_include_directories(test_math_operators PRIVATE ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/src/operators)
set_target_properties(test_math_operators PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(test_conv_pool_operators unit/test_conv_pool_operators.cu)
target_link_libraries(test_conv_pool_operators cuda_learning_core)
target_include_directories(test_conv_pool_operators PRIVATE ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/src/operators)
set_target_properties(test_conv_pool_operators PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Operator performance tests
add_executable(operator_performance_tests performance/operator_performance_tests.cu)
target_link_libraries(operator_performance_tests cuda_learning_core)
target_include_directories(operator_performance_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_SOURCE_DIR}/src/operators)
set_target_properties(operator_performance_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Kernel performance tests
add_executable(kernel_performance_tests kernel_performance_tests.cu)
target_link_libraries(kernel_performance_tests cuda_learning_core)
target_include_directories(kernel_performance_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
set_target_properties(kernel_performance_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Try to find Google Test
find_package(GTest QUIET)
if(GTest_FOUND)
    target_link_libraries(kernel_performance_tests GTest::gtest GTest::gtest_main)
    add_test(NAME KernelPerformanceTests COMMAND kernel_performance_tests)
else()
    # If Google Test is not found, create a simple test runner
    message(STATUS "Google Test not found. Creating simple test runner for kernel performance tests.")

    # Create a simple main function wrapper
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/simple_test_runner.cpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/simple_test_runner.cpp"
        @ONLY
    )

    add_executable(kernel_performance_tests_simple
        "${CMAKE_CURRENT_BINARY_DIR}/simple_test_runner.cpp"
        kernel_performance_tests.cu)
    target_link_libraries(kernel_performance_tests_simple cuda_learning_core)
    target_include_directories(kernel_performance_tests_simple PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
    set_target_properties(kernel_performance_tests_simple PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    add_test(NAME KernelPerformanceTestsSimple COMMAND kernel_performance_tests_simple)
endif()

# Add individual tests
add_test(NAME CudaEnvironmentTest COMMAND test_cuda_environment)
add_test(NAME BasicExamplesTest COMMAND test_basic_examples)
add_test(NAME ExampleCompilationTest COMMAND test_example_compilation)
add_test(NAME ExampleExecutionTest COMMAND test_example_execution)
add_test(NAME MathOperatorsTest COMMAND test_math_operators)
add_test(NAME ConvPoolOperatorsTest COMMAND test_conv_pool_operators)
add_test(NAME OperatorPerformanceTest COMMAND operator_performance_tests)
