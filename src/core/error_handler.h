#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA错误 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA错误"); \
        } \
    } while(0)

#endif // ERROR_HANDLER_H
