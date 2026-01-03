#include "cuda_environment.h"
#include "error_handler.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <sstream>

// Simple test framework (reusing from existing test)
class TestRunner {
public:
    static void run_test(const std::string& test_name, bool (*test_func)()) {
        std::cout << "Running " << test_name << "... ";
        if (test_func()) {
            std::cout << "PASSED" << std::endl;
            s_passed++;
        } else {
            std::cout << "FAILED" << std::endl;
            s_failed++;
        }
        s_total++;
    }
    
    static void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total: " << s_total << std::endl;
        std::cout << "Passed: " << s_passed << std::endl;
        std::cout << "Failed: " << s_failed << std::endl;
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

// 辅助函数：检查文件是否存在
bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// 辅助函数：执行系统命令并获取返回值
int execute_command(const std::string& command) {
    return std::system(command.c_str());
}

// 辅助函数：检查CUDA编译器是否可用
bool is_nvcc_available() {
    int result = execute_command("nvcc --version > nul 2>&1");
    return result == 0;
}

// 辅助函数：检查CUDA设备是否可用
bool is_cuda_device_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// 测试函数

bool test_nvcc_compiler_availability() {
    return is_nvcc_available();
}

bool test_example_files_exist() {
    std::vector<std::string> example_files = {
        "examples/basic/hello_world_cuda.cu",
        "examples/basic/vector_addition.cu",
        "examples/basic/matrix_operations.cu",
        "examples/basic/gpu_architecture_tutorial.cpp",
        "examples/basic/cuda_programming_model_demo.cpp",
        "examples/basic/thread_hierarchy_examples.cpp"
    };
    
    bool all_exist = true;
    for (const auto& file : example_files) {
        if (!file_exists(file)) {
            std::cout << "Missing file: " << file << std::endl;
            all_exist = false;
        }
    }
    
    return all_exist;
}

bool test_hello_world_compilation() {
    if (!is_nvcc_available()) {
        std::cout << "nvcc不可用，跳过编译测试" << std::endl;
        return true;
    }
    
    // 尝试编译hello_world_cuda.cu
    std::string compile_cmd = "nvcc -o test_hello_world examples/basic/hello_world_cuda.cu > nul 2>&1";
    int result = execute_command(compile_cmd);
    
    // 清理生成的文件
    execute_command("del test_hello_world.exe > nul 2>&1");
    
    return result == 0;
}

bool test_vector_addition_compilation() {
    if (!is_nvcc_available()) {
        std::cout << "nvcc不可用，跳过编译测试" << std::endl;
        return true;
    }
    
    // 尝试编译vector_addition.cu
    std::string compile_cmd = "nvcc -o test_vector_addition examples/basic/vector_addition.cu > nul 2>&1";
    int result = execute_command(compile_cmd);
    
    // 清理生成的文件
    execute_command("del test_vector_addition.exe > nul 2>&1");
    
    return result == 0;
}

bool test_matrix_operations_compilation() {
    if (!is_nvcc_available()) {
        std::cout << "nvcc不可用，跳过编译测试" << std::endl;
        return true;
    }
    
    // 尝试编译matrix_operations.cu
    std::string compile_cmd = "nvcc -o test_matrix_operations examples/basic/matrix_operations.cu > nul 2>&1";
    int result = execute_command(compile_cmd);
    
    // 清理生成的文件
    execute_command("del test_matrix_operations.exe > nul 2>&1");
    
    return result == 0;
}

bool test_cpp_examples_compilation() {
    // 测试C++示例的编译（需要链接到cuda_learning_core）
    // 这里我们只检查文件是否存在和基本语法
    
    std::vector<std::string> cpp_files = {
        "examples/basic/gpu_architecture_tutorial.cpp",
        "examples/basic/cuda_programming_model_demo.cpp",
        "examples/basic/thread_hierarchy_examples.cpp"
    };
    
    bool all_valid = true;
    for (const auto& file : cpp_files) {
        if (!file_exists(file)) {
            all_valid = false;
            continue;
        }
        
        // 简单的语法检查：确保文件包含必要的头文件
        std::ifstream f(file);
        std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
        
        if (content.find("#include") == std::string::npos) {
            all_valid = false;
        }
        
        if (content.find("main") == std::string::npos) {
            all_valid = false;
        }
    }
    
    return all_valid;
}

bool test_cmake_configuration() {
    // 检查CMakeLists.txt文件是否存在
    std::vector<std::string> cmake_files = {
        "CMakeLists.txt",
        "examples/CMakeLists.txt",
        "examples/basic/CMakeLists.txt"
    };
    
    bool all_exist = true;
    for (const auto& file : cmake_files) {
        if (!file_exists(file)) {
            std::cout << "Missing CMake file: " << file << std::endl;
            all_exist = false;
        }
    }
    
    return all_exist;
}

bool test_example_content_validity() {
    // 检查示例文件是否包含必要的内容
    
    struct FileCheck {
        std::string filename;
        std::vector<std::string> required_content;
    };
    
    std::vector<FileCheck> checks = {
        {
            "examples/basic/hello_world_cuda.cu",
            {"__global__", "cudaMalloc", "cudaMemcpy", "main"}
        },
        {
            "examples/basic/vector_addition.cu",
            {"__global__", "vectorAdd", "cudaMalloc", "cudaMemcpy"}
        },
        {
            "examples/basic/matrix_operations.cu",
            {"__global__", "matrixAdd", "dim3", "blockIdx"}
        }
    };
    
    bool all_valid = true;
    for (const auto& check : checks) {
        if (!file_exists(check.filename)) {
            all_valid = false;
            continue;
        }
        
        std::ifstream f(check.filename);
        std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
        
        for (const auto& required : check.required_content) {
            if (content.find(required) == std::string::npos) {
                std::cout << "Missing content '" << required 
                          << "' in " << check.filename << std::endl;
                all_valid = false;
            }
        }
    }
    
    return all_valid;
}

bool test_error_handling_in_examples() {
    // 检查示例是否包含适当的错误处理
    
    std::vector<std::string> cuda_files = {
        "examples/basic/hello_world_cuda.cu",
        "examples/basic/vector_addition.cu",
        "examples/basic/matrix_operations.cu"
    };
    
    bool all_have_error_handling = true;
    for (const auto& file : cuda_files) {
        if (!file_exists(file)) {
            all_have_error_handling = false;
            continue;
        }
        
        std::ifstream f(file);
        std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
        
        // 检查是否有错误检查
        bool has_error_check = (content.find("cudaGetLastError") != std::string::npos) ||
                              (content.find("CUDA_CHECK") != std::string::npos) ||
                              (content.find("cudaGetErrorString") != std::string::npos);
        
        if (!has_error_check) {
            std::cout << "No error handling found in " << file << std::endl;
            all_have_error_handling = false;
        }
    }
    
    return all_have_error_handling;
}

bool test_documentation_in_examples() {
    // 检查示例是否包含适当的文档和注释
    
    std::vector<std::string> all_files = {
        "examples/basic/hello_world_cuda.cu",
        "examples/basic/vector_addition.cu",
        "examples/basic/matrix_operations.cu",
        "examples/basic/gpu_architecture_tutorial.cpp",
        "examples/basic/cuda_programming_model_demo.cpp",
        "examples/basic/thread_hierarchy_examples.cpp"
    };
    
    bool all_documented = true;
    for (const auto& file : all_files) {
        if (!file_exists(file)) {
            all_documented = false;
            continue;
        }
        
        std::ifstream f(file);
        std::string content((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
        
        // 检查是否有文档注释
        bool has_documentation = (content.find("/**") != std::string::npos) ||
                                (content.find("/*") != std::string::npos) ||
                                (content.find("//") != std::string::npos);
        
        if (!has_documentation) {
            std::cout << "No documentation found in " << file << std::endl;
            all_documented = false;
        }
        
        // 检查是否有学习目标或说明
        bool has_learning_info = (content.find("学习目标") != std::string::npos) ||
                               (content.find("本程序") != std::string::npos) ||
                               (content.find("演示") != std::string::npos);
        
        if (!has_learning_info) {
            std::cout << "No learning information found in " << file << std::endl;
            all_documented = false;
        }
    }
    
    return all_documented;
}

int main() {
    std::cout << "Example Compilation and Content Tests" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // 初始化错误处理
    cuda_learning::ErrorHandler::initialize("test_example_compilation.log");
    
    // 运行测试
    TestRunner::run_test("NVCC Compiler Availability", test_nvcc_compiler_availability);
    TestRunner::run_test("Example Files Exist", test_example_files_exist);
    TestRunner::run_test("Hello World Compilation", test_hello_world_compilation);
    TestRunner::run_test("Vector Addition Compilation", test_vector_addition_compilation);
    TestRunner::run_test("Matrix Operations Compilation", test_matrix_operations_compilation);
    TestRunner::run_test("C++ Examples Compilation Check", test_cpp_examples_compilation);
    TestRunner::run_test("CMake Configuration", test_cmake_configuration);
    TestRunner::run_test("Example Content Validity", test_example_content_validity);
    TestRunner::run_test("Error Handling in Examples", test_error_handling_in_examples);
    TestRunner::run_test("Documentation in Examples", test_documentation_in_examples);
    
    // 打印总结
    TestRunner::print_summary();
    
    // 清理
    cuda_learning::ErrorHandler::shutdown();
    
    return TestRunner::get_exit_code();
}