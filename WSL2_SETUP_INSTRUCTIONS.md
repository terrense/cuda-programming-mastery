# WSL2 Ubuntu CUDA 开发环境设置说明

## 🎯 快速开始

在WSL2 Ubuntu终端中运行以下命令：

```bash
# 1. 给脚本添加执行权限
chmod +x build.sh setup_wsl2.sh test_wsl2_cuda.sh

# 2. 运行自动化设置（推荐）
./setup_wsl2.sh

# 3. 测试环境
./test_wsl2_cuda.sh

# 4. 构建项目
./build.sh

# 5. 运行示例
./build/examples/basic/hello_world_cuda
```

## 🔧 环境要求

### Windows 主机要求
- Windows 11 或 Windows 10 版本 21H2+
- 安装最新的 NVIDIA 驱动程序（支持WSL2）
- 启用 WSL2 功能

### WSL2 Ubuntu 要求
- Ubuntu 20.04 或 22.04
- 安装 CUDA Toolkit for WSL2（不是Windows版本）
- 基本开发工具

## 📋 手动设置步骤

如果自动化脚本失败，可以手动执行：

### 1. 安装基础工具
```bash
sudo apt update
sudo apt install -y build-essential cmake git wget curl
```

### 2. 安装 CUDA Toolkit for WSL2
```bash
# 下载并安装 CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# 更新并安装 CUDA
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# 添加到 PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. 验证安装
```bash
# 检查 CUDA 编译器
nvcc --version

# 检查 GPU（如果可用）
nvidia-smi
```

### 4. 构建项目
```bash
# 使用 CMake
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j$(nproc)

# 或使用构建脚本
./build.sh
```

## 🚀 VS Code 集成

### 推荐扩展
在 VS Code 中安装以下扩展：
- Remote - WSL
- C/C++
- CMake Tools
- NVIDIA Nsight Visual Studio Code Edition

### 配置步骤
1. 在 Windows 中打开 VS Code
2. 安装 Remote-WSL 扩展
3. 按 `Ctrl+Shift+P`，选择 "Remote-WSL: New WSL Window"
4. 在 WSL2 中打开项目文件夹
5. VS Code 会自动检测配置文件

## 🔍 故障排除

### CUDA 不可用
```bash
# 检查 WSL2 版本
wsl --version

# 检查 Windows NVIDIA 驱动
# 在 Windows PowerShell 中运行：
nvidia-smi
```

### 编译错误
```bash
# 检查编译器版本
gcc --version
nvcc --version
cmake --version

# 清理并重新构建
rm -rf build
./build.sh
```

### GPU 不可见
- 确保 Windows 主机安装了支持 WSL2 的 NVIDIA 驱动
- 重启 WSL2：`wsl --shutdown` 然后重新打开
- 检查 Windows 驱动版本是否支持 WSL2

## 📚 学习路径

构建成功后，按以下顺序学习：

1. **环境测试**: `./build/examples/basic/cuda_env_test`
2. **Hello World**: `./build/examples/basic/hello_world_cuda`
3. **基础示例**: 查看 `examples/basic/` 目录
4. **进阶示例**: 查看 `examples/intermediate/` 目录
5. **项目实战**: 查看 `examples/projects/` 目录

## 🎯 性能优化

WSL2 环境下的优化建议：

1. **内存管理**: GPU 内存与 Windows 主机共享
2. **文件系统**: 项目文件放在 WSL2 文件系统中（不是 /mnt/c/）
3. **编译缓存**: 使用 `ccache` 加速重复编译
4. **并行构建**: 使用 `make -j$(nproc)` 充分利用CPU核心

## 📞 获取帮助

如果遇到问题：

1. 运行诊断脚本：`./test_wsl2_cuda.sh`
2. 查看日志文件：`cuda_learning.log`
3. 检查 CUDA 示例是否能编译运行
4. 参考 NVIDIA WSL2 官方文档

---

**提示**: 这个环境配置完成后，你就可以开始系统化的 CUDA 编程学习之旅了！🚀
