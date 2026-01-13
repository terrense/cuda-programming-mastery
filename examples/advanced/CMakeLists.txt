# Advanced CUDA Examples

# Advanced Memory Optimization Demo
add_executable(advanced_memory_optimization_demo advanced_memory_optimization_demo.cu)
target_link_libraries(advanced_memory_optimization_demo cuda_learning_core)
set_target_properties(advanced_memory_optimization_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
