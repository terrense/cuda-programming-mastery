# CUDA Programming Mastery Learning System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.5%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-blue.svg)](https://docs.microsoft.com/en-us/windows/wsl/)

A comprehensive learning system designed to help developers master CUDA programming from basics to advanced optimization techniques. This project provides a structured approach to learning GPU programming with hands-on examples, performance analysis tools, and real-world applications.

## 🌟 Features

- **Progressive Learning Path**: From GPU architecture basics to advanced optimization
- **Hands-on Examples**: Practical CUDA code examples and exercises
- **Custom Operator Development**: Framework for building optimized CUDA operators
- **YOLO Acceleration**: Real-world deep learning model optimization
- **Performance Analysis**: Built-in profiling and optimization tools
- **WSL2 Optimized**: Full support for Windows Subsystem for Linux 2
- **VS Code Integration**: IntelliSense, debugging, and task automation

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with Compute Capability 6.0 or higher (recommended)
- **Software**:
  - WSL2 Ubuntu (recommended) or native Linux
  - CUDA Toolkit 11.5+ for WSL2
  - CMake 3.18+
  - GCC 7+ or Clang 6+

### Installation

**WSL2 Ubuntu (Recommended):**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cuda-programming-mastery.git
cd cuda-programming-mastery

# Run automated WSL2 setup
chmod +x setup_wsl2.sh
./setup_wsl2.sh

# Test your environment
chmod +x test_wsl2_cuda.sh
./test_wsl2_cuda.sh

# Build and run
./build.sh
./build/examples/basic/hello_world_cuda
```

**Manual Setup:**
```bash
git clone https://github.com/YOUR_USERNAME/cuda-programming-mastery.git
cd cuda-programming-mastery
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
make -j$(nproc)
```

## 📚 Learning Path

### 1. Foundation (Weeks 1-2)
- GPU architecture understanding
- CUDA programming model
- Environment setup and first programs

### 2. Core Skills (Weeks 3-4)
- Kernel function development
- Memory management optimization
- Performance analysis

### 3. Advanced Techniques (Weeks 5-6)
- Custom operator development
- Multi-GPU programming
- Advanced optimization patterns

### 4. Real-world Application (Weeks 7-8)
- YOLO model acceleration
- Production deployment
- Performance tuning

## 🏗️ Project Structure

```
├── src/                    # Source code
│   ├── core/              # Core system components
│   ├── examples/          # CUDA examples by difficulty
│   ├── operators/         # Custom CUDA operators
│   └── yolo/              # YOLO acceleration implementation
├── examples/              # Runnable examples
│   ├── basic/             # Basic CUDA examples
│   ├── intermediate/      # Intermediate examples
│   ├── advanced/          # Advanced examples
│   └── projects/          # Complete projects
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation and tutorials
└── .kiro/                 # Kiro IDE specifications
```

## 🧪 Testing

The project includes comprehensive unit tests and integration tests to ensure code quality and functionality.

**Automated Test Suite:**
```bash
# WSL2/Linux
chmod +x run_tests.sh
./run_tests.sh

# Windows
run_tests.bat
```

**Manual Testing:**
```bash
# Build with tests enabled
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)

# Run all tests
ctest --output-on-failure

# Run individual test suites
./test_cuda_environment      # Core environment tests
./test_basic_examples        # Basic CUDA example tests
./test_example_compilation   # Compilation verification tests
./test_example_execution     # Runtime execution tests
```

## 📖 Examples

### Hello World CUDA
```cuda
__global__ void helloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d!\n", idx);
}

int main() {
    helloKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Vector Addition
```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite: `./run_tests.sh`
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for the CUDA toolkit and documentation
- The CUDA community for examples and best practices
- Contributors and testers who help improve this project

## 📞 Support

If you encounter any issues or have questions:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/terrense/cuda-programming-mastery/issues)
3. Create a new issue if needed
4. My personal E-mail : deeplearningman0723@gmail.com

## 🗺️ Roadmap

- [ ] Advanced memory optimization techniques
- [ ] Multi-GPU programming examples
- [ ] TensorRT integration
- [ ] Deep learning operator implementations
- [ ] Performance profiling tools
- [ ] Interactive Jupyter notebooks
