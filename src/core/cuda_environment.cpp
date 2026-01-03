#include "cuda_environment.h"
#include "error_handler.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace cuda_learning {

CudaEnvironment::CudaEnvironment() 
    : m_currentDevice(0), m_initialized(false) {
    initializeGPUList();
}

CudaEnvironment::~CudaEnvironment() {
    // Cleanup if needed
}

bool CudaEnvironment::detectGPU() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        ErrorHandler::logError("Failed to detect GPU devices: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    
    if (deviceCount == 0) {
        ErrorHandler::logError("No CUDA-capable devices found");
        return false;
    }
    
    m_initialized = true;
    return true;
}

bool CudaEnvironment::isCudaAvailable() {
    return detectGPU() && !m_gpuList.empty();
}

std::vector<GPUInfo> CudaEnvironment::getAvailableGPUs() {
    if (!m_initialized) {
        initializeGPUList();
    }
    return m_gpuList;
}

GPUInfo CudaEnvironment::getBestGPU() {
    if (m_gpuList.empty()) {
        return GPUInfo{};
    }
    
    // Find GPU with highest compute capability and most memory
    auto bestGPU = std::max_element(m_gpuList.begin(), m_gpuList.end(),
        [](const GPUInfo& a, const GPUInfo& b) {
            if (a.computeCapabilityMajor != b.computeCapabilityMajor) {
                return a.computeCapabilityMajor < b.computeCapabilityMajor;
            }
            if (a.computeCapabilityMinor != b.computeCapabilityMinor) {
                return a.computeCapabilityMinor < b.computeCapabilityMinor;
            }
            return a.totalMemory < b.totalMemory;
        });
    
    return *bestGPU;
}

bool CudaEnvironment::configureDevelopmentEnvironment() {
    if (!detectGPU()) {
        return false;
    }
    
    // Set the best available GPU as default
    GPUInfo bestGPU = getBestGPU();
    if (bestGPU.isSupported) {
        return setDevice(bestGPU.deviceId);
    }
    
    return false;
}

bool CudaEnvironment::validateCudaInstallation() {
    // Check CUDA runtime
    int runtimeVersion = 0;
    cudaError_t error = cudaRuntimeGetVersion(&runtimeVersion);
    if (error != cudaSuccess) {
        ErrorHandler::logError("CUDA runtime not available");
        return false;
    }
    
    // Check driver
    int driverVersion = 0;
    error = cudaDriverGetVersion(&driverVersion);
    if (error != cudaSuccess) {
        ErrorHandler::logError("CUDA driver not available");
        return false;
    }
    
    // Verify compatibility
    if (driverVersion < runtimeVersion) {
        ErrorHandler::logWarning("Driver version is older than runtime version");
    }
    
    return true;
}

std::string CudaEnvironment::getCudaVersion() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    int major = version / 1000;
    int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string CudaEnvironment::getDriverVersion() {
    int version = 0;
    cudaDriverGetVersion(&version);
    int major = version / 1000;
    int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

void CudaEnvironment::printSystemInfo() {
    std::cout << "=== CUDA System Information ===" << std::endl;
    std::cout << "CUDA Runtime Version: " << getCudaVersion() << std::endl;
    std::cout << "CUDA Driver Version: " << getDriverVersion() << std::endl;
    std::cout << "Number of GPUs: " << m_gpuList.size() << std::endl;
    
    for (const auto& gpu : m_gpuList) {
        std::cout << "\nGPU " << gpu.deviceId << ": " << gpu.name << std::endl;
        std::cout << "  Compute Capability: " << gpu.computeCapabilityMajor 
                  << "." << gpu.computeCapabilityMinor << std::endl;
        std::cout << "  Total Memory: " << gpu.totalMemory / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << gpu.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << gpu.maxThreadsPerBlock << std::endl;
    }
}

bool CudaEnvironment::setDevice(int deviceId) {
    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) {
        ErrorHandler::logError("Failed to set device " + std::to_string(deviceId));
        return false;
    }
    m_currentDevice = deviceId;
    return true;
}

int CudaEnvironment::getCurrentDevice() {
    return m_currentDevice;
}

void CudaEnvironment::initializeGPUList() {
    m_gpuList.clear();
    
    if (!detectGPU()) {
        return;
    }
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        GPUInfo info = queryGPUInfo(i);
        if (info.isSupported) {
            m_gpuList.push_back(info);
        }
    }
}

GPUInfo CudaEnvironment::queryGPUInfo(int deviceId) {
    GPUInfo info = {};
    info.deviceId = deviceId;
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    
    if (error != cudaSuccess) {
        info.isSupported = false;
        return info;
    }
    
    info.name = prop.name;
    info.totalMemory = prop.totalGlobalMem;
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    
    // Check if compute capability is sufficient (>= 3.5)
    info.isSupported = (prop.major > 3) || (prop.major == 3 && prop.minor >= 5);
    
    // Get current memory info
    size_t free, total;
    cudaSetDevice(deviceId);
    cudaMemGetInfo(&free, &total);
    info.freeMemory = free;
    
    return info;
}

} // namespace cuda_learning