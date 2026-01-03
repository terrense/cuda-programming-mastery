#include "cuda_environment.h"
#include "error_handler.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <iomanip>

// Simple test framework
class TestRunner {
public:
    static void run_test(const std::string& test_name, bool (*test_func)()) {
        std::cout << "Running " << test_name << "... ";
        try {
            if (test_func()) {
                std::cout << "PASSED" << std::endl;
                s_passed++;
            } else {
                std::cout << "FAILED" << std::endl;
                s_failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (Exception: " << e.what() << ")" << std::endl;
            s_failed++;
        } catch (...) {
            std::cout << "FAILED (Unknown exception)" << std::endl;
            s_failed++;
        }
        s_total++;
    }

    static void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total: " << s_total << std::endl;
        std::cout << "Passed: " << s_passed << std::endl;
        std::cout << "Failed: " << s_failed << std::endl;
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1)
                  << (s_total > 0 ? (100.0 * s_passed / s_total) : 0.0) << "%" << std::endl;
    }

    static int get_exit_code() {
        return s_failed > 0 ? 1 : 0;
    }

private:
    static int s_total;
    static int s_passed;
    static int s_failed;
};

int TestRunner::s_total = 0;
int TestRunner::s_passed = 0;
int TestRunner::s_failed = 0;

// Test functions
bool test_cuda_environment_creation() {
    try {
        cuda_learning::CudaEnvironment env;
        return true; // If we can create it without crashing, test passes
    } catch (...) {
        return false;
    }
}

bool test_cuda_detection() {
    try {
        cuda_learning::CudaEnvironment env;
        // This test will pass even if no CUDA is available
        // It just tests that the detection doesn't crash
        bool result = env.detectGPU();
        return true; // Always pass - we're testing the function doesn't crash
    } catch (...) {
        return false;
    }
}

bool test_gpu_info_retrieval() {
    try {
        cuda_learning::CudaEnvironment env;
        auto gpus = env.getAvailableGPUs();
        // Test passes if we can retrieve GPU list (even if empty)
        return true;
    } catch (...) {
        return false;
    }
}

bool test_cuda_availability_check() {
    try {
        cuda_learning::CudaEnvironment env;
        bool available = env.isCudaAvailable();

        // Check consistency with device count
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        bool expectedAvailable = (error == cudaSuccess && deviceCount > 0);

        return available == expectedAvailable;
    } catch (...) {
        return false;
    }
}

bool test_best_gpu_selection() {
    try {
        cuda_learning::CudaEnvironment env;
        auto gpus = env.getAvailableGPUs();

        if (gpus.empty()) {
            // If no GPUs, getBestGPU should return empty GPUInfo
            auto bestGPU = env.getBestGPU();
            return bestGPU.name.empty();
        } else {
            // If GPUs exist, getBestGPU should return a valid GPU
            auto bestGPU = env.getBestGPU();
            return !bestGPU.name.empty() && bestGPU.isSupported;
        }
    } catch (...) {
        return false;
    }
}

bool test_cuda_version_info() {
    try {
        cuda_learning::CudaEnvironment env;
        std::string cudaVersion = env.getCudaVersion();
        std::string driverVersion = env.getDriverVersion();

        // Versions should not be empty if CUDA is available
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);

        if (error == cudaSuccess) {
            return !cudaVersion.empty() && !driverVersion.empty();
        } else {
            // If CUDA is not available, versions might be empty or "0.0"
            return true;
        }
    } catch (...) {
        return false;
    }
}

bool test_device_setting() {
    try {
        cuda_learning::CudaEnvironment env;
        auto gpus = env.getAvailableGPUs();

        if (gpus.empty()) {
            // If no GPUs, setting device should fail
            return !env.setDevice(0);
        } else {
            // If GPUs exist, setting to first device should succeed
            bool success = env.setDevice(gpus[0].deviceId);
            if (success) {
                return env.getCurrentDevice() == gpus[0].deviceId;
            }
            return false;
        }
    } catch (...) {
        return false;
    }
}

bool test_cuda_installation_validation() {
    try {
        cuda_learning::CudaEnvironment env;
        bool valid = env.validateCudaInstallation();

        // Check consistency with CUDA availability
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        bool cudaWorking = (error == cudaSuccess);

        // Installation should be valid if CUDA runtime works
        return valid == cudaWorking;
    } catch (...) {
        return false;
    }
}

bool test_development_environment_configuration() {
    try {
        cuda_learning::CudaEnvironment env;
        bool configured = env.configureDevelopmentEnvironment();

        // Should succeed if CUDA is available and there are supported GPUs
        auto gpus = env.getAvailableGPUs();
        bool hasValidGPU = false;
        for (const auto& gpu : gpus) {
            if (gpu.isSupported) {
                hasValidGPU = true;
                break;
            }
        }

        return configured == hasValidGPU;
    } catch (...) {
        return false;
    }
}

bool test_system_info_printing() {
    try {
        cuda_learning::CudaEnvironment env;
        // This should not crash
        env.printSystemInfo();
        return true;
    } catch (...) {
        return false;
    }
}

bool test_error_handler_initialization() {
    try {
        cuda_learning::ErrorHandler::initialize("test_env.log");
        cuda_learning::ErrorHandler::logInfo("Test message");
        cuda_learning::ErrorHandler::shutdown();
        return true;
    } catch (...) {
        return false;
    }
}

bool test_error_handler_logging() {
    try {
        cuda_learning::ErrorHandler::initialize("test_env.log");

        // Test different log levels
        cuda_learning::ErrorHandler::logDebug("Debug message");
        cuda_learning::ErrorHandler::logInfo("Info message");
        cuda_learning::ErrorHandler::logWarning("Warning message");
        cuda_learning::ErrorHandler::logError("Error message");

        cuda_learning::ErrorHandler::shutdown();
        return true;
    } catch (...) {
        return false;
    }
}

bool test_error_handler_configuration() {
    try {
        cuda_learning::ErrorHandler::initialize("test_env.log");

        // Test configuration changes
        cuda_learning::ErrorHandler::setLogLevel(cuda_learning::LogLevel::WARNING);
        cuda_learning::ErrorHandler::enableConsoleOutput(false);
        cuda_learning::ErrorHandler::enableFileOutput(true);

        // These should not crash
        cuda_learning::ErrorHandler::logDebug("Should not appear");
        cuda_learning::ErrorHandler::logWarning("Should appear");

        cuda_learning::ErrorHandler::shutdown();
        return true;
    } catch (...) {
        return false;
    }
}

bool test_cuda_error_checking() {
    try {
        cuda_learning::ErrorHandler::initialize("test_env.log");

        // Test successful CUDA call
        cudaError_t success = cudaSuccess;
        bool result1 = cuda_learning::ErrorHandler::checkCudaError(success, __FILE__, __LINE__);

        // Test failed CUDA call
        cudaError_t error = cudaErrorInvalidValue;
        bool result2 = cuda_learning::ErrorHandler::checkCudaError(error, __FILE__, __LINE__);

        cuda_learning::ErrorHandler::shutdown();

        return result1 && !result2; // Success should return true, error should return false
    } catch (...) {
        return false;
    }
}

int main() {
    std::cout << "CUDA Environment Unit Tests" << std::endl;
    std::cout << "===========================" << std::endl;

    // Run tests
    TestRunner::run_test("CudaEnvironment Creation", test_cuda_environment_creation);
    TestRunner::run_test("CUDA Detection", test_cuda_detection);
    TestRunner::run_test("GPU Info Retrieval", test_gpu_info_retrieval);
    TestRunner::run_test("CUDA Availability Check", test_cuda_availability_check);
    TestRunner::run_test("Best GPU Selection", test_best_gpu_selection);
    TestRunner::run_test("CUDA Version Info", test_cuda_version_info);
    TestRunner::run_test("Device Setting", test_device_setting);
    TestRunner::run_test("CUDA Installation Validation", test_cuda_installation_validation);
    TestRunner::run_test("Development Environment Configuration", test_development_environment_configuration);
    TestRunner::run_test("System Info Printing", test_system_info_printing);
    TestRunner::run_test("ErrorHandler Initialization", test_error_handler_initialization);
    TestRunner::run_test("ErrorHandler Logging", test_error_handler_logging);
    TestRunner::run_test("ErrorHandler Configuration", test_error_handler_configuration);
    TestRunner::run_test("CUDA Error Checking", test_cuda_error_checking);

    // Print summary
    TestRunner::print_summary();

    return TestRunner::get_exit_code();
}
