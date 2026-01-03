@echo off
echo CUDA Programming Mastery Build Script
echo =====================================

echo Checking for CUDA installation...
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: NVCC not found. Please install CUDA Toolkit.
    pause
    exit /b 1
)

echo Checking for CMake...
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake not found. Please install CMake 3.18 or later.
    pause
    exit /b 1
)

echo Creating build directory...
if not exist build mkdir build

echo Configuring project...
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed.
    pause
    exit /b 1
)

echo Building project...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo ERROR: Build failed.
    pause
    exit /b 1
)

echo Build completed successfully!
echo Run examples\basic\cuda_env_test.exe to test your environment.
pause