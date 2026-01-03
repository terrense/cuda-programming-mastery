#include "cuda_environment.h"
#include "error_handler.h"
#include <iostream>

int main() {
    // Initialize error handler
    cuda_learning::ErrorHandler::initialize("cuda_env_test.log");
    
    std::cout << "CUDA Programming Mastery - Environment Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Create CUDA environment instance
    cuda_learning::CudaEnvironment cudaEnv;
    
    // Test CUDA availability
    if (!cudaEnv.isCudaAvailable()) {
        std::cerr << "CUDA is not available on this system!" << std::endl;
        cuda_learning::ErrorHandler::shutdown();
        return 1;
    }
    
    std::cout << "✓ CUDA is available" << std::endl;
    
    // Validate CUDA installation
    if (!cudaEnv.validateCudaInstallation()) {
        std::cerr << "CUDA installation validation failed!" << std::endl;
        cuda_learning::ErrorHandler::shutdown();
        return 1;
    }
    
    std::cout << "✓ CUDA installation is valid" << std::endl;
    
    // Print system information
    cudaEnv.printSystemInfo();
    
    // Configure development environment
    if (cudaEnv.configureDevelopmentEnvironment()) {
        std::cout << "✓ Development environment configured successfully" << std::endl;
        
        // Get best GPU info
        auto bestGPU = cudaEnv.getBestGPU();
        std::cout << "Best GPU selected: " << bestGPU.name 
                  << " (Device " << bestGPU.deviceId << ")" << std::endl;
    } else {
        std::cerr << "Failed to configure development environment" << std::endl;
        cuda_learning::ErrorHandler::shutdown();
        return 1;
    }
    
    std::cout << "\n✓ Environment test completed successfully!" << std::endl;
    
    // Cleanup
    cuda_learning::ErrorHandler::shutdown();
    return 0;
}