#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <cuda_runtime.h>

namespace cuda_learning {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

class ErrorHandler {
public:
    static void initialize(const std::string& logFile = "cuda_learning.log");
    static void shutdown();
    
    // Logging functions
    static void logDebug(const std::string& message);
    static void logInfo(const std::string& message);
    static void logWarning(const std::string& message);
    static void logError(const std::string& message);
    static void logFatal(const std::string& message);
    
    // CUDA error handling
    static bool checkCudaError(cudaError_t error, const char* file, int line);
    static void handleCudaError(cudaError_t error, const char* file, int line);
    
    // Configuration
    static void setLogLevel(LogLevel level);
    static void enableConsoleOutput(bool enable);
    static void enableFileOutput(bool enable);
    
private:
    static std::unique_ptr<std::ofstream> s_logFile;
    static LogLevel s_logLevel;
    static bool s_consoleOutput;
    static bool s_fileOutput;
    static bool s_initialized;
    
    static void writeLog(LogLevel level, const std::string& message);
    static std::string getCurrentTimestamp();
    static std::string logLevelToString(LogLevel level);
};

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (!ErrorHandler::checkCudaError(error, __FILE__, __LINE__)) { \
            return false; \
        } \
    } while(0)

// Macro for CUDA error checking with custom error handling
#define CUDA_CHECK_THROW(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            ErrorHandler::handleCudaError(error, __FILE__, __LINE__); \
        } \
    } while(0)

} // namespace cuda_learning