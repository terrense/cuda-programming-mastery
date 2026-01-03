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
