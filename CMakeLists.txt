cmake_minimum_required(VERSION 3.18)
project(CudaProgrammingMastery LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV QUIET)

# Set CUDA architectures (support multiple GPU generations)
# For WSL2, focus on modern architectures commonly used in development
# Removed compute_89 and compute_90 for compatibility with older GPUs like GTX 1650
set(CMAKE_CUDA_ARCHITECTURES "75")

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# WSL2 specific optimizations
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Enable compile_commands.json for IntelliSense
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # WSL2 specific CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    # Optimize for WSL2 environment
    if(EXISTS "/proc/version")
        file(READ "/proc/version" PROC_VERSION)
        if(PROC_VERSION MATCHES "microsoft|WSL")
            message(STATUS "Detected WSL2 environment")
            # WSL2 specific optimizations
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
        endif()
    endif()
endif()

# CUDA compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(src)

# Core library
file(GLOB_RECURSE CORE_SOURCES
    "src/core/*.cpp"
    "src/core/*.cu"
)

add_library(cuda_learning_core ${CORE_SOURCES})
target_include_directories(cuda_learning_core PUBLIC src/core)
set_target_properties(cuda_learning_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Examples
add_subdirectory(examples)

# Tests
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    endif()
endif()

# Installation
install(TARGETS cuda_learning_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/core/
    DESTINATION include/cuda_learning
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "=== CUDA Programming Mastery Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build docs: ${BUILD_DOCS}")
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "OpenCV: Not found (optional)")
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_learning_core PRIVATE CUDA::cudart)


set_source_files_properties(
  examples/basic/thread_hierarchy_examples.cpp
  examples/basic/cuda_programming_model_demo.cpp
  PROPERTIES LANGUAGE CUDA
)

set_target_properties(cuda_learning_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
