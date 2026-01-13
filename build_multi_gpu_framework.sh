#!/bin/bash

# CUDA多GPU编程框架构建脚本
# 作者: CUDA学习系统
# 版本: 1.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 命令未找到，请先安装"
        return 1
    fi
    return 0
}

# 检查CUDA环境
check_cuda_environment() {
    print_info "检查CUDA环境..."

    # 检查nvidia-smi
    if ! check_command nvidia-smi; then
        print_error "NVIDIA驱动未安装或未正确配置"
        return 1
    fi

    # 检查nvcc
    if ! check_command nvcc; then
        print_error "CUDA工具包未安装或未正确配置"
        return 1
    fi

    # 获取CUDA版本
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "检测到CUDA版本: $CUDA_VERSION"

    # 检查GPU设备
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    print_success "检测到 $GPU_COUNT 个GPU设备"

    if [ $GPU_COUNT -eq 0 ]; then
        print_error "未检测到GPU设备"
        return 1
    fi

    return 0
}

# 检查NCCL库
check_nccl() {
    print_info "检查NCCL库..."

    # 常见的NCCL安装路径
    NCCL_PATHS=(
        "/usr/local/nccl"
        "/usr/local/cuda/include"
        "/opt/nccl"
        "$HOME/nccl"
    )

    NCCL_FOUND=false
    for path in "${NCCL_PATHS[@]}"; do
        if [ -f "$path/include/nccl.h" ] || [ -f "$path/nccl.h" ]; then
            export NCCL_ROOT="$path"
            NCCL_FOUND=true
            print_success "找到NCCL库: $path"
            break
        fi
    done

    if [ "$NCCL_FOUND" = false ]; then
        print_warning "未找到NCCL库，多GPU通信功能将受限"
        print_info "请从以下地址下载安装NCCL: https://developer.nvidia.com/nccl"
        return 1
    fi

    return 0
}

# 检查依赖
check_dependencies() {
    print_info "检查构建依赖..."

    # 检查CMake
    if ! check_command cmake; then
        print_error "请安装CMake 3.18或更高版本"
        return 1
    fi

    CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "检测到CMake版本: $CMAKE_VERSION"

    # 检查编译器
    if ! check_command g++; then
        print_error "请安装g++编译器"
        return 1
    fi

    GCC_VERSION=$(g++ --version | head -n1 | sed 's/g++ (.*) \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "检测到GCC版本: $GCC_VERSION"

    return 0
}

# 创建构建目录
setup_build_directory() {
    print_info "设置构建目录..."

    BUILD_DIR="build_multi_gpu"

    if [ -d "$BUILD_DIR" ]; then
        print_warning "构建目录已存在，清理中..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    print_success "构建目录创建完成: $BUILD_DIR"
}

# 配置CMake
configure_cmake() {
    print_info "配置CMake..."

    CMAKE_ARGS=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_EXAMPLES=ON"
        "-DBUILD_TESTS=ON"
        "-DCMAKE_CUDA_ARCHITECTURES=60;70;75;80;86"
    )

    # 如果找到NCCL，添加路径
    if [ -n "$NCCL_ROOT" ]; then
        CMAKE_ARGS+=("-DNCCL_ROOT=$NCCL_ROOT")
    fi

    # 执行CMake配置
    if cmake .. "${CMAKE_ARGS[@]}"; then
        print_success "CMake配置成功"
    else
        print_error "CMake配置失败"
        return 1
    fi

    return 0
}

# 编译项目
build_project() {
    print_info "开始编译项目..."

    # 获取CPU核心数
    NPROC=$(nproc)
    print_info "使用 $NPROC 个并行编译进程"

    # 编译
    if make -j$NPROC; then
        print_success "项目编译成功"
    else
        print_error "项目编译失败"
        return 1
    fi

    return 0
}

# 运行测试
run_tests() {
    print_info "运行测试..."

    if [ -f "test_multi_gpu" ]; then
        print_info "运行多GPU框架测试..."
        if ./test_multi_gpu; then
            print_success "所有测试通过"
        else
            print_warning "部分测试失败，请检查日志"
        fi
    else
        print_warning "测试程序未找到，跳过测试"
    fi
}

# 运行演示程序
run_demo() {
    print_info "运行演示程序..."

    if [ -f "multi_gpu_demo" ]; then
        print_info "运行多GPU演示程序..."
        if ./multi_gpu_demo; then
            print_success "演示程序运行成功"
        else
            print_warning "演示程序运行失败，请检查GPU环境"
        fi
    else
        print_warning "演示程序未找到"
    fi
}

# 安装项目
install_project() {
    print_info "安装项目..."

    if [ "$EUID" -eq 0 ]; then
        # 以root权限运行
        if make install; then
            print_success "项目安装成功"
        else
            print_error "项目安装失败"
            return 1
        fi
    else
        # 普通用户，安装到用户目录
        print_info "以普通用户身份安装到 $HOME/.local"
        if make install DESTDIR="$HOME/.local"; then
            print_success "项目安装到用户目录成功"
        else
            print_error "项目安装失败"
            return 1
        fi
    fi

    return 0
}

# 生成性能报告
generate_performance_report() {
    print_info "生成性能报告..."

    REPORT_FILE="../multi_gpu_performance_report.txt"

    {
        echo "CUDA多GPU编程框架性能报告"
        echo "生成时间: $(date)"
        echo "========================================"
        echo

        echo "系统信息:"
        echo "--------"
        uname -a
        echo

        echo "GPU信息:"
        echo "--------"
        nvidia-smi -L
        echo

        echo "CUDA版本:"
        echo "---------"
        nvcc --version
        echo

        if [ -n "$NCCL_ROOT" ]; then
            echo "NCCL配置:"
            echo "---------"
            echo "NCCL路径: $NCCL_ROOT"
            echo
        fi

        echo "编译配置:"
        echo "---------"
        echo "构建类型: Release"
        echo "CUDA架构: 60;70;75;80;86"
        echo "并行编译进程: $(nproc)"
        echo

        if [ -f "multi_gpu_demo" ]; then
            echo "性能测试结果:"
            echo "-------------"
            ./multi_gpu_demo 2>&1 | grep -E "(时间|带宽|效率|GPU|设备)"
        fi

    } > "$REPORT_FILE"

    print_success "性能报告已生成: $REPORT_FILE"
}

# 清理函数
cleanup() {
    print_info "清理临时文件..."
    cd ..
    # 可选择是否删除构建目录
    # rm -rf "$BUILD_DIR"
}

# 显示使用帮助
show_help() {
    echo "CUDA多GPU编程框架构建脚本"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -c, --clean         清理构建目录"
    echo "  -t, --test-only     仅运行测试"
    echo "  -d, --demo-only     仅运行演示"
    echo "  -i, --install       编译后安装"
    echo "  -r, --report        生成性能报告"
    echo "  --skip-tests        跳过测试"
    echo "  --skip-demo         跳过演示"
    echo
    echo "示例:"
    echo "  $0                  完整构建流程"
    echo "  $0 -i               构建并安装"
    echo "  $0 -t               仅运行测试"
    echo "  $0 -r               生成性能报告"
}

# 主函数
main() {
    print_info "开始构建CUDA多GPU编程框架"
    print_info "========================================"

    # 解析命令行参数
    CLEAN_ONLY=false
    TEST_ONLY=false
    DEMO_ONLY=false
    INSTALL_PROJECT=false
    GENERATE_REPORT=false
    SKIP_TESTS=false
    SKIP_DEMO=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                CLEAN_ONLY=true
                shift
                ;;
            -t|--test-only)
                TEST_ONLY=true
                shift
                ;;
            -d|--demo-only)
                DEMO_ONLY=true
                shift
                ;;
            -i|--install)
                INSTALL_PROJECT=true
                shift
                ;;
            -r|--report)
                GENERATE_REPORT=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-demo)
                SKIP_DEMO=true
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 清理模式
    if [ "$CLEAN_ONLY" = true ]; then
        print_info "清理构建目录..."
        rm -rf build_multi_gpu
        print_success "清理完成"
        exit 0
    fi

    # 检查环境
    if ! check_cuda_environment; then
        exit 1
    fi

    if ! check_dependencies; then
        exit 1
    fi

    check_nccl  # NCCL是可选的，失败不退出

    # 仅测试模式
    if [ "$TEST_ONLY" = true ]; then
        if [ -d "build_multi_gpu" ]; then
            cd build_multi_gpu
            run_tests
        else
            print_error "构建目录不存在，请先运行完整构建"
            exit 1
        fi
        exit 0
    fi

    # 仅演示模式
    if [ "$DEMO_ONLY" = true ]; then
        if [ -d "build_multi_gpu" ]; then
            cd build_multi_gpu
            run_demo
        else
            print_error "构建目录不存在，请先运行完整构建"
            exit 1
        fi
        exit 0
    fi

    # 完整构建流程
    setup_build_directory

    if ! configure_cmake; then
        cleanup
        exit 1
    fi

    if ! build_project; then
        cleanup
        exit 1
    fi

    # 运行测试
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    fi

    # 运行演示
    if [ "$SKIP_DEMO" = false ]; then
        run_demo
    fi

    # 安装项目
    if [ "$INSTALL_PROJECT" = true ]; then
        install_project
    fi

    # 生成性能报告
    if [ "$GENERATE_REPORT" = true ]; then
        generate_performance_report
    fi

    print_success "构建流程完成!"
    print_info "========================================"
    print_info "可执行文件位置:"
    print_info "  多GPU演示程序: build_multi_gpu/multi_gpu_demo"
    print_info "  单元测试程序: build_multi_gpu/test_multi_gpu"
    print_info ""
    print_info "使用方法:"
    print_info "  cd build_multi_gpu"
    print_info "  ./multi_gpu_demo    # 运行演示程序"
    print_info "  ./test_multi_gpu    # 运行单元测试"

    cleanup
}

# 捕获中断信号
trap cleanup EXIT INT TERM

# 运行主函数
main "$@"
