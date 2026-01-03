#pragma once

#include <vector>
#include <string>
#include <memory>

namespace cuda_learning {

struct GPUInfo {
    int deviceId;
    std::string name;
    size_t totalMemory;
    size_t freeMemory;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    bool isSupported;
};

class CudaEnvironment {
public:
    CudaEnvironment();
    ~CudaEnvironment();

    // Environment detection
    bool detectGPU();
    bool isCudaAvailable();
    std::vector<GPUInfo> getAvailableGPUs();
    GPUInfo getBestGPU();
    
    // Environment configuration
    bool configureDevelopmentEnvironment();
    bool validateCudaInstallation();
    std::string getCudaVersion();
    std::string getDriverVersion();
    
    // Utility functions
    void printSystemInfo();
    bool setDevice(int deviceId);
    int getCurrentDevice();
    
private:
    std::vector<GPUInfo> m_gpuList;
    int m_currentDevice;
    bool m_initialized;
    
    void initializeGPUList();
    GPUInfo queryGPUInfo(int deviceId);
};

} // namespace cuda_learning