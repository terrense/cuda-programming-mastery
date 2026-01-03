#include "error_handler.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cuda_learning {

// Static member definitions
std::unique_ptr<std::ofstream> ErrorHandler::s_logFile = nullptr;
LogLevel ErrorHandler::s_logLevel = LogLevel::INFO;
bool ErrorHandler::s_consoleOutput = true;
bool ErrorHandler::s_fileOutput = true;
bool ErrorHandler::s_initialized = false;

void ErrorHandler::initialize(const std::string& logFile) {
    if (s_initialized) {
        return;
    }
    
    if (s_fileOutput) {
        s_logFile = std::make_unique<std::ofstream>(logFile, std::ios::app);
        if (!s_logFile->is_open()) {
            std::cerr << "Warning: Could not open log file: " << logFile << std::endl;
            s_fileOutput = false;
        }
    }
    
    s_initialized = true;
    logInfo("ErrorHandler initialized");
}

void ErrorHandler::shutdown() {
    if (!s_initialized) {
        return;
    }
    
    logInfo("ErrorHandler shutting down");
    
    if (s_logFile && s_logFile->is_open()) {
        s_logFile->close();
    }
    s_logFile.reset();
    s_initialized = false;
}

void ErrorHandler::logDebug(const std::string& message) {
    writeLog(LogLevel::DEBUG, message);
}

void ErrorHandler::logInfo(const std::string& message) {
    writeLog(LogLevel::INFO, message);
}

void ErrorHandler::logWarning(const std::string& message) {
    writeLog(LogLevel::WARNING, message);
}

void ErrorHandler::logError(const std::string& message) {
    writeLog(LogLevel::ERROR, message);
}

void ErrorHandler::logFatal(const std::string& message) {
    writeLog(LogLevel::FATAL, message);
}

bool ErrorHandler::checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::string errorMsg = "CUDA Error at " + std::string(file) + ":" + 
                              std::to_string(line) + " - " + cudaGetErrorString(error);
        logError(errorMsg);
        return false;
    }
    return true;
}

void ErrorHandler::handleCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::string errorMsg = "CUDA Error at " + std::string(file) + ":" + 
                              std::to_string(line) + " - " + cudaGetErrorString(error);
        logFatal(errorMsg);
        throw std::runtime_error(errorMsg);
    }
}

void ErrorHandler::setLogLevel(LogLevel level) {
    s_logLevel = level;
}

void ErrorHandler::enableConsoleOutput(bool enable) {
    s_consoleOutput = enable;
}

void ErrorHandler::enableFileOutput(bool enable) {
    s_fileOutput = enable;
}

void ErrorHandler::writeLog(LogLevel level, const std::string& message) {
    if (!s_initialized) {
        initialize();
    }
    
    if (level < s_logLevel) {
        return;
    }
    
    std::string timestamp = getCurrentTimestamp();
    std::string levelStr = logLevelToString(level);
    std::string logMessage = "[" + timestamp + "] [" + levelStr + "] " + message;
    
    // Console output
    if (s_consoleOutput) {
        if (level >= LogLevel::ERROR) {
            std::cerr << logMessage << std::endl;
        } else {
            std::cout << logMessage << std::endl;
        }
    }
    
    // File output
    if (s_fileOutput && s_logFile && s_logFile->is_open()) {
        *s_logFile << logMessage << std::endl;
        s_logFile->flush();
    }
}

std::string ErrorHandler::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string ErrorHandler::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNOWN";
    }
}

} // namespace cuda_learning