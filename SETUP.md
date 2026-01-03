# CUDA Programming Mastery - Setup Guide

This guide will help you set up the CUDA Programming Mastery learning environment on your system, with special focus on WSL2 Ubuntu.

## WSL2 Ubuntu Quick Setup (Recommended)

If you're using WSL2 Ubuntu, use our automated setup script:

```bash
# Make the setup script executable
chmod +x setup_wsl2.sh

# Run the automated setup
./setup_wsl2.sh
```

This script will:
- Install CUDA Toolkit for WSL2
- Set up build tools and dependencies
- Configure development environment
- Create test programs
- Set up VS Code integration

After running the script, restart your terminal and proceed to [Build and Test](#build-and-test).

## Manual Setup Instructions

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with Compute Capability 3.5 or higher
- At least 4GB GPU memory (8GB+ recommended)
- 8GB+ system RAM

### Software Requirements
- **CUDA Toolkit 11.0+**: Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit)
- **Compatible C++ Compiler**:
  - Windows: Visual Studio 2019+ or MinGW-w64
  - Linux: GCC 7+ or Clang 6+
  - macOS: Xcode 10+ (CUDA support limited)
- **CMake 3.18+**: Download from [CMake.org](https://cmake.org/download/)

### Optional Dependencies
- **OpenCV**: For computer vision examples
- **cuDNN**: For deep learning optimizations
- **TensorRT**: For inference optimization

## Installation Steps

### 1. Install CUDA Toolkit

#### Windows
1. Download CUDA Toolkit from NVIDIA Developer website
2. Run the installer and follow the setup wizard
3. Add CUDA to your PATH (usually done automatically)
4. Verify installation: `nvcc --version`

#### Linux (Ubuntu/Debian) - Including WSL2
```bash
# For WSL2 Ubuntu, use the CUDA WSL2 repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# For native Ubuntu (not WSL2), use:
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install -y cuda-toolkit-12-3

# Add to PATH (add these lines to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### 2. Install CMake

#### Windows
1. Download CMake installer from cmake.org
2. Run installer and add CMake to PATH
3. Verify: `cmake --version`

#### Linux
```bash
sudo apt-get install cmake
# Or for newer version:
pip install cmake
```

### 3. Clone and Build Project

```bash
# Clone the repository
git clone <repository-url>
cd cuda-programming-mastery

# Option 1: Using automated build script (Recommended for WSL2)
chmod +x build.sh
./build.sh

# Option 2: Using CMake manually
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j$(nproc)  # Linux/macOS
# OR
cmake --build . --config Release  # Cross-platform

# Option 3: Using Makefile (Linux/macOS only)
make all
```

## Build and Test

### 4. Verify Installation

```bash
# Test CUDA environment
./build/examples/cuda_env_test  # Linux/macOS
build\examples\cuda_env_test.exe  # Windows

# Run unit tests
./build/tests/unit_tests  # Linux/macOS
build\tests\unit_tests.exe  # Windows
```

## Troubleshooting

### Common Issues

#### "CUDA not found" Error
- Ensure CUDA Toolkit is properly installed
- Check that `nvcc` is in your PATH
- Verify GPU drivers are up to date

#### WSL2 Specific Issues

**CUDA not working in WSL2:**
- Ensure you have Windows 11 or Windows 10 version 21H2 or later
- Install NVIDIA drivers on Windows host (not in WSL2)
- Use CUDA Toolkit for WSL2 (not the Windows version)
- Verify with: `nvidia-smi` (should show GPU info)

**Build fails with "compute capability" errors:**
- Update CMAKE_CUDA_ARCHITECTURES in CMakeLists.txt
- Check your GPU's compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Remove unsupported architectures from the build configuration

**IntelliSense not working:**
- Ensure compile_commands.json is generated: `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
- Copy compile_commands.json to project root
- Install C/C++ extension in VS Code
- Use Remote-WSL extension for VS Code

#### Compilation Errors
- Check C++ compiler compatibility with CUDA version
- Ensure CMake version is 3.18 or later
- Try building with different CUDA architectures

#### Runtime Errors
- Update GPU drivers to latest version
- Check GPU memory availability
- Ensure proper CUDA runtime libraries are installed

### GPU Compatibility Check

Run this command to check your GPU compatibility:
```bash
nvidia-smi
```

Look for:
- GPU model and compute capability
- Available memory
- Driver version

### Build Configuration Options

You can customize the build with these CMake options:

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Disable tests
cmake .. -DBUILD_TESTS=OFF

# Enable documentation
cmake .. -DBUILD_DOCS=ON

# Specify CUDA architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="60;70;75"
```

## Development Environment Setup

### Visual Studio Code (Recommended)
1. Install C/C++ extension
2. Install CUDA syntax highlighting extension
3. Configure IntelliSense for CUDA files

### Visual Studio (Windows)
1. Install CUDA integration for Visual Studio
2. Create new CUDA project or import existing

### CLion
1. Enable CUDA support in settings
2. Configure CMake toolchain

## Next Steps

Once setup is complete:

1. **Start with basics**: Run `examples/basic/cuda_env_test`
2. **Follow tutorials**: Check `docs/tutorials/` directory
3. **Practice coding**: Work through examples in order
4. **Build projects**: Try the complete projects in `examples/projects/`

## Getting Help

- Check the troubleshooting section above
- Review CUDA documentation: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Ask questions in project issues or discussions
- Join CUDA developer forums

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 750 Ti (CC 5.0) | RTX 3060+ (CC 8.6+) |
| GPU Memory | 2GB | 8GB+ |
| System RAM | 8GB | 16GB+ |
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.20+ |