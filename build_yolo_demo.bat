@echo off
REM YOLO模型加载器演示构建脚本 (Windows版本)

echo === YOLO Model Loader Demo Build Script ===

REM 检查CUDA环境
echo Checking CUDA environment...
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: CUDA compiler (nvcc) not found. Please install CUDA toolkit.
    exit /b 1
)

REM 获取CUDA版本
for /f "tokens=*" %%i in ('nvcc --version ^| findstr "release"') do set CUDA_INFO=%%i
echo CUDA Info: %CUDA_INFO%

REM 检查GPU
echo Checking GPU availability...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
) else (
    echo Warning: nvidia-smi not found. GPU detection skipped.
)

REM 创建构建目录
set BUILD_DIR=build_yolo
echo Creating build directory: %BUILD_DIR%
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM 配置CMake
echo Configuring CMake...
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86" -DBUILD_TESTING=ON ../src/yolo
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

REM 编译
echo Building YOLO module...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

REM 检查编译结果
if exist "Release\yolo_model_loader_demo.exe" (
    echo Build successful! Executable created: Release\yolo_model_loader_demo.exe
) else if exist "yolo_model_loader_demo.exe" (
    echo Build successful! Executable created: yolo_model_loader_demo.exe
) else (
    echo Build failed! Executable not found.
    exit /b 1
)

REM 运行演示程序
echo Running YOLO model loader demo...
echo ==================================
if exist "Release\yolo_model_loader_demo.exe" (
    Release\yolo_model_loader_demo.exe
) else (
    yolo_model_loader_demo.exe
)

REM 检查运行结果
if %errorlevel% equ 0 (
    echo ==================================
    echo Demo completed successfully!
) else (
    echo Demo failed with error code: %errorlevel%
    exit /b 1
)

REM 运行测试（如果启用）
if exist "CTestTestfile.cmake" (
    echo Running tests...
    ctest --output-on-failure
)

echo === Build and Demo Completed ===
pause
