# CUDA Programming Mastery Learning System

A comprehensive learning system designed to help developers master CUDA programming from basics to advanced optimization techniques, with optimized support for WSL2 Ubuntu development.

## 🚀 Quick Start for WSL2 Ubuntu

**Automated Setup (Recommended):**
```bash
# Clone the repository
git clone <repository-url>
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
See [SETUP.md](SETUP.md) for detailed installation instructions.

## Features

- **Progressive Learning Path**: From GPU architecture basics to advanced optimization
- **Hands-on Examples**: Practical CUDA code examples and exercises
- **Custom Operator Development**: Framework for building optimized CUDA operators
- **YOLO Acceleration**: Real-world deep learning model optimization
- **Performance Analysis**: Built-in profiling and optimization tools
- **WSL2 Optimized**: Full support for Windows Subsystem for Linux 2
- **VS Code Integration**: IntelliSense, debugging, and task automation

## Prerequisites

- **Hardware**: NVIDIA GPU with Compute Capability 6.0 or higher (recommended)
- **Software**:
  - WSL2 Ubuntu (recommended) or native Linux
  - CUDA Toolkit 12.0+ for WSL2
  - CMake 3.18+
  - GCC 7+ or Clang 6+

## Development Environment

**Recommended Setup:**
- WSL2 Ubuntu 20.04/22.04
- VS Code with Remote-WSL extension
- NVIDIA drivers on Windows host
- CUDA Toolkit in WSL2 (not Windows)

## Quick Start

### WSL2 Ubuntu (Recommended)
```bash
# Automated setup
./setup_wsl2.sh

# Test environment
./test_wsl2_cuda.sh

# Build project
./build.sh

# Run tests
chmod +x run_tests.sh
./run_tests.sh

# Run examples
./build/examples/basic/cuda_env_test
./build/examples/basic/hello_world_cuda
```

### Windows
```cmd
# Build project
build.bat

# Run tests
run_tests.bat

# Run examples
build\examples\basic\Debug\hello_world_cuda.exe
```

### Manual Build
```bash
git clone <repository-url>
cd cuda-programming-mastery
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure
# or run individual tests
./test_cuda_environment
./test_basic_examples
```

## Project Structure

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
└── CMakeLists.txt         # Build configuration
```

## Learning Path

1. **Foundation** (Weeks 1-2)
   - GPU architecture understanding
   - CUDA programming model
   - Environment setup and first programs

2. **Core Skills** (Weeks 3-4)
   - Kernel function development
   - Memory management optimization
   - Performance analysis

3. **Advanced Techniques** (Weeks 5-6)
   - Custom operator development
   - Multi-GPU programming
   - Advanced optimization patterns

4. **Real-world Application** (Weeks 7-8)
   - YOLO model acceleration
   - Production deployment
   - Performance tuning

## Testing

The project includes comprehensive unit tests and integration tests to ensure code quality and functionality.

### Running Tests

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

### Test Categories

- **Environment Tests**: Verify CUDA installation and GPU detection
- **Example Tests**: Validate basic CUDA programming examples
- **Compilation Tests**: Ensure all examples compile correctly
- **Execution Tests**: Verify runtime correctness and performance
- **Integration Tests**: End-to-end workflow validation

### Test Requirements

- CUDA-capable GPU (recommended but not required)
- CUDA Toolkit installation
- CMake 3.18+
- C++17 compatible compiler

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for improvements.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite: `./run_tests.sh`
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
