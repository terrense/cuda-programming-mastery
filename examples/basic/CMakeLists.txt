# Basic CUDA Examples

# Environment test example
add_executable(cuda_env_test cuda_env_test.cpp)
target_link_libraries(cuda_env_test cuda_learning_core ${CUDA_LIBRARIES})
set_target_properties(cuda_env_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# GPU Architecture Tutorial
add_executable(gpu_architecture_tutorial gpu_architecture_tutorial.cpp)
target_link_libraries(gpu_architecture_tutorial cuda_learning_core ${CUDA_LIBRARIES})
set_target_properties(gpu_architecture_tutorial PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# CUDA Programming Model Demo
add_executable(cuda_programming_model_demo cuda_programming_model_demo.cpp)
target_link_libraries(cuda_programming_model_demo cuda_learning_core ${CUDA_LIBRARIES})
set_target_properties(cuda_programming_model_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Thread Hierarchy Examples
add_executable(thread_hierarchy_examples thread_hierarchy_examples.cpp)
target_link_libraries(thread_hierarchy_examples cuda_learning_core ${CUDA_LIBRARIES})
set_target_properties(thread_hierarchy_examples PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Hello World CUDA
add_executable(hello_world_cuda hello_world_cuda.cu)
target_link_libraries(hello_world_cuda ${CUDA_LIBRARIES})
set_target_properties(hello_world_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Vector Addition Examples
add_executable(vector_addition vector_addition.cu)
target_link_libraries(vector_addition ${CUDA_LIBRARIES})
set_target_properties(vector_addition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Matrix Operations Examples
add_executable(matrix_operations matrix_operations.cu)
target_link_libraries(matrix_operations ${CUDA_LIBRARIES})
set_target_properties(matrix_operations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
