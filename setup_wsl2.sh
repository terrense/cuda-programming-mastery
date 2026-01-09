#!/bin/bash

echo "CUDA Programming Mastery - WSL2 Ubuntu Setup"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running in WSL2
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    print_error "This script is designed for WSL2. Please run on WSL2 Ubuntu."
    exit 1
fi

print_status "Detected WSL2 environment"

# Update package list
print_info "Updating package list..."
sudo apt update

# Install essential build tools
print_info "Installing essential build tools..."
sudo apt install -y build-essential cmake git wget curl

# Check if CUDA is already installed
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_status "CUDA $CUDA_VERSION is already installed"
else
    print_info "CUDA Toolkit not found. Installing CUDA for WSL2..."
    
    # Download and install CUDA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    
    # Update and install CUDA
    sudo apt update
    sudo apt install -y cuda-toolkit-12-3
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    print_status "CUDA Toolkit installed. Please restart your terminal or run 'source ~/.bashrc'"
fi

# Install additional development tools
print_info "Installing additional development tools..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    libeigen3-dev \
    libopencv-dev \
    pkg-config

# Install Python packages for development
print_info "Installing Python packages..."
pip3 install --user numpy matplotlib jupyter

# Create development directories
print_info "Setting up development environment..."
mkdir -p ~/cuda-projects
mkdir -p ~/.vscode-server/extensions

# Configure git (if not already configured)
if ! git config --global user.name &> /dev/null; then
    print_info "Git configuration needed. Please enter your details:"
    read -p "Git username: " git_username
    read -p "Git email: " git_email
    git config --global user.name "$git_username"
    git config --global user.email "$git_email"
    print_status "Git configured"
fi

# Check VS Code Server
if command -v code &> /dev/null; then
    print_status "VS Code CLI available"
    
    # Recommend extensions
    print_info "Recommended VS Code extensions for CUDA development:"
    echo "  - ms-vscode.cpptools (C/C++ IntelliSense)"
    echo "  - nvidia.nsight-vscode-edition (CUDA debugging)"
    echo "  - ms-python.python (Python support)"
    echo "  - twxs.cmake (CMake support)"
    echo ""
    echo "Install with: code --install-extension <extension-id>"
else
    print_warning "VS Code CLI not found. Install VS Code with Remote-WSL extension for best experience."
fi

# Test CUDA installation
print_info "Testing CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    print_status "CUDA compiler test passed"
    
    # Test nvidia-smi if available
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU information:"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    else
        print_warning "nvidia-smi not available (normal in some WSL2 setups)"
    fi
else
    print_error "CUDA installation test failed"
fi

# Create a simple test program
print_info "Creating CUDA test program..."
cat > ~/cuda-projects/test_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Testing CUDA..." << std::endl;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount > 0) {
        hello_kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
        std::cout << "CUDA test completed successfully!" << std::endl;
    }
    
    return 0;
}
EOF

# Compile and test the program
cd ~/cuda-projects
if nvcc -o test_cuda test_cuda.cu 2>/dev/null; then
    print_status "Test program compiled successfully"
    if ./test_cuda; then
        print_status "CUDA test program executed successfully"
    else
        print_warning "CUDA test program compiled but execution failed"
    fi
else
    print_error "Failed to compile test program"
fi

# Final setup summary
echo ""
echo "============================================="
echo "WSL2 CUDA Development Environment Setup Complete!"
echo "============================================="
echo ""
print_status "Essential tools installed:"
echo "  - Build tools (gcc, cmake, git)"
echo "  - CUDA Toolkit"
echo "  - Development libraries"
echo "  - Python development environment"
echo ""
print_info "Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Clone your CUDA projects to ~/cuda-projects/"
echo "3. Use VS Code with Remote-WSL extension"
echo "4. Build projects with: ./build.sh"
echo ""
print_info "Useful commands:"
echo "  - Check CUDA: nvcc --version"
echo "  - Check GPU: nvidia-smi"
echo "  - Build project: ./build.sh"
echo "  - Run tests: ./build/tests/unit_tests"
echo ""
if [ -f ~/.bashrc ]; then
    print_warning "Please restart your terminal or run 'source ~/.bashrc' to update PATH"
fi