# Contributing to CUDA Programming Mastery Learning System

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- WSL2 Ubuntu or native Linux
- CUDA Toolkit 11.5+
- CMake 3.18+
- Git

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cuda-programming-mastery.git
   cd cuda-programming-mastery
   ```
3. Set up the development environment:
   ```bash
   ./setup_wsl2.sh
   ```
4. Build and test:
   ```bash
   ./build.sh
   ./run_tests.sh
   ```

## 📝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/YOUR_USERNAME/cuda-programming-mastery/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, CUDA version, GPU model)
   - Error messages or logs

### Suggesting Features

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and motivation
   - Possible implementation approach

### Code Contributions

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow coding standards**:
   - Use meaningful variable and function names
   - Add comments for complex CUDA kernels
   - Follow existing code style
   - Include error checking for CUDA calls

3. **Write tests**:
   - Add unit tests for new functionality
   - Ensure all existing tests pass
   - Test on different GPU architectures if possible

4. **Update documentation**:
   - Update README.md if needed
   - Add inline code documentation
   - Update examples if applicable

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./build/test_cuda_environment
./build/test_basic_examples
./build/test_example_execution
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Use descriptive test names
- Test both success and failure cases
- Include performance regression tests for optimizations

### Test Requirements

- All tests must pass on CI
- New features require corresponding tests
- Performance changes should include benchmarks

## 📚 Documentation

### Code Documentation

- Use clear, descriptive comments
- Document CUDA kernel parameters and behavior
- Explain complex algorithms and optimizations
- Include performance characteristics

### Examples

- Provide complete, runnable examples
- Include expected output
- Add learning objectives
- Explain key concepts

## 🎯 Project Areas

### High Priority

- Basic CUDA examples and tutorials
- Performance optimization techniques
- Error handling and debugging tools
- Cross-platform compatibility

### Medium Priority

- Advanced CUDA features
- Multi-GPU programming
- Integration with deep learning frameworks
- Interactive tutorials

### Future Goals

- Web-based learning interface
- Automated performance analysis
- Cloud deployment examples
- Mobile GPU support

## 🔧 Development Guidelines

### CUDA Code Standards

```cuda
// Good: Clear kernel with error checking
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Always check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
```

### Performance Guidelines

- Profile before optimizing
- Document performance characteristics
- Include benchmarks for optimizations
- Test on multiple GPU architectures

### Memory Management

- Always pair malloc/free, cudaMalloc/cudaFree
- Use RAII patterns in C++
- Check for memory leaks
- Handle out-of-memory conditions

## 🤝 Community

### Communication

- Be respectful and inclusive
- Help newcomers learn CUDA
- Share knowledge and best practices
- Provide constructive feedback

### Code Review

- Review code for correctness and performance
- Suggest improvements kindly
- Test changes when possible
- Approve when ready

## 📋 Checklist for Contributors

Before submitting a PR, ensure:

- [ ] Code follows project standards
- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Performance impact is considered

## 🆘 Getting Help

- Check the [documentation](docs/)
- Ask questions in [Discussions](https://github.com/YOUR_USERNAME/cuda-programming-mastery/discussions)
- Join our community channels
- Reach out to maintainers

Thank you for contributing to CUDA Programming Mastery! 🚀
