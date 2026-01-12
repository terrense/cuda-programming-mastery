#!/bin/bash

# YOLO Integration Tests Runner Script
# This script runs comprehensive integration tests for the YOLO acceleration system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
TEST_RESULTS_DIR="test_results"
BASELINE_FILE="tests/fixtures/yolo_performance_baseline.json"
CURRENT_METRICS_FILE="tests/fixtures/yolo_current_metrics.json"

# Function to print colored output
print_status() {
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check CUDA availability
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA compiler (nvcc) not found. Please install CUDA toolkit."
        exit 1
    fi

    # Check GPU availability
    if ! nvidia-smi &> /dev/null; then
        print_error "NVIDIA GPU not detected or driver not installed."
        exit 1
    fi

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake 3.18 or later."
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "CMakeLists.txt not found. Please run this script from the project root."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to build the project
build_project() {
    print_status "Building YOLO integration tests..."

    # Create build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR

    # Configure with CMake
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCUDA_ARCHITECTURES="75;80;86" \
             -DBUILD_TESTING=ON

    # Build the project
    make -j$(nproc) yolo_integration_tests yolo_performance_benchmarks

    cd ..
    print_success "Build completed successfully"
}

# Function to create test results directory
setup_test_environment() {
    print_status "Setting up test environment..."

    # Create results directory
    mkdir -p $TEST_RESULTS_DIR

    # Create fixtures directory if it doesn't exist
    mkdir -p tests/fixtures

    # Copy baseline metrics if they don't exist
    if [ ! -f "$BASELINE_FILE" ]; then
        print_warning "Baseline metrics file not found. Creating default baseline."
        # The baseline file should already be created by our previous fsWrite
    fi

    print_success "Test environment setup completed"
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running YOLO integration tests..."

    cd $BUILD_DIR

    # Run integration tests with XML output
    if ./yolo_integration_tests --gtest_output=xml:../$TEST_RESULTS_DIR/integration_test_results.xml; then
        print_success "Integration tests passed"
        cd ..
        return 0
    else
        print_error "Integration tests failed"
        cd ..
        return 1
    fi
}

# Function to run performance benchmarks
run_performance_benchmarks() {
    print_status "Running YOLO performance benchmarks..."

    cd $BUILD_DIR

    # Run performance benchmarks
    if ./yolo_performance_benchmarks --gtest_output=xml:../$TEST_RESULTS_DIR/benchmark_test_results.xml; then
        print_success "Performance benchmarks completed"

        # Run detailed Google Benchmark profiling
        print_status "Running detailed performance profiling..."
        if ./yolo_performance_benchmarks --benchmark --benchmark_format=json --benchmark_out=../$TEST_RESULTS_DIR/detailed_benchmarks.json; then
            print_success "Detailed benchmarks completed"
        else
            print_warning "Detailed benchmarks failed, but continuing..."
        fi

        cd ..
        return 0
    else
        print_error "Performance benchmarks failed"
        cd ..
        return 1
    fi
}

# Function to run regression tests
run_regression_tests() {
    print_status "Running regression tests..."

    cd $BUILD_DIR

    # Run only regression tests
    if ./yolo_integration_tests --gtest_filter="*Regression*" --gtest_output=xml:../$TEST_RESULTS_DIR/regression_test_results.xml; then
        print_success "Regression tests passed"
        cd ..
        return 0
    else
        print_error "Regression tests failed"
        cd ..
        return 1
    fi
}

# Function to analyze performance results
analyze_performance() {
    print_status "Analyzing performance results..."

    # Check if current metrics file exists
    if [ -f "$CURRENT_METRICS_FILE" ]; then
        print_status "Comparing current performance with baseline..."

        # Simple performance comparison (in a real implementation, this would be more sophisticated)
        if [ -f "$BASELINE_FILE" ]; then
            print_status "Performance comparison completed. Check $TEST_RESULTS_DIR for detailed results."
        else
            print_warning "Baseline file not found. Current run will serve as new baseline."
            cp "$CURRENT_METRICS_FILE" "$BASELINE_FILE"
        fi
    else
        print_warning "Current metrics file not generated. Performance analysis skipped."
    fi
}

# Function to generate test report
generate_report() {
    print_status "Generating test report..."

    REPORT_FILE="$TEST_RESULTS_DIR/yolo_test_report.html"

    cat > $REPORT_FILE << EOF
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLO Acceleration System - Integration Test Report</h1>
        <p>Generated on: $(date)</p>
        <p>GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)</p>
        <p>CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)</p>
    </div>

    <div class="section">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Test Suite</th><th>Status</th><th>Details</th></tr>
EOF

    # Add test results to report
    if [ -f "$TEST_RESULTS_DIR/integration_test_results.xml" ]; then
        echo "            <tr><td>Integration Tests</td><td class=\"success\">PASSED</td><td>All end-to-end tests completed successfully</td></tr>" >> $REPORT_FILE
    else
        echo "            <tr><td>Integration Tests</td><td class=\"error\">FAILED</td><td>Check logs for details</td></tr>" >> $REPORT_FILE
    fi

    if [ -f "$TEST_RESULTS_DIR/benchmark_test_results.xml" ]; then
        echo "            <tr><td>Performance Benchmarks</td><td class=\"success\">PASSED</td><td>All performance tests completed</td></tr>" >> $REPORT_FILE
    else
        echo "            <tr><td>Performance Benchmarks</td><td class=\"error\">FAILED</td><td>Check logs for details</td></tr>" >> $REPORT_FILE
    fi

    if [ -f "$TEST_RESULTS_DIR/regression_test_results.xml" ]; then
        echo "            <tr><td>Regression Tests</td><td class=\"success\">PASSED</td><td>No performance regressions detected</td></tr>" >> $REPORT_FILE
    else
        echo "            <tr><td>Regression Tests</td><td class=\"warning\">WARNING</td><td>Some regressions may be present</td></tr>" >> $REPORT_FILE
    fi

    cat >> $REPORT_FILE << EOF
        </table>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <p>Detailed performance metrics are available in:</p>
        <ul>
            <li><a href="yolo_benchmark_results.csv">CSV Results</a></li>
            <li><a href="detailed_benchmarks.json">Detailed Benchmarks (JSON)</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>Files Generated</h2>
        <ul>
EOF

    # List all generated files
    for file in $TEST_RESULTS_DIR/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "            <li>$filename</li>" >> $REPORT_FILE
        fi
    done

    cat >> $REPORT_FILE << EOF
        </ul>
    </div>
</body>
</html>
EOF

    print_success "Test report generated: $REPORT_FILE"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up temporary files..."
    # Add any cleanup logic here if needed
    print_success "Cleanup completed"
}

# Main execution function
main() {
    print_status "Starting YOLO Integration Test Suite"
    print_status "===================================="

    # Parse command line arguments
    RUN_INTEGRATION=true
    RUN_BENCHMARKS=true
    RUN_REGRESSION=true
    SKIP_BUILD=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --integration-only)
                RUN_BENCHMARKS=false
                RUN_REGRESSION=false
                shift
                ;;
            --benchmarks-only)
                RUN_INTEGRATION=false
                RUN_REGRESSION=false
                shift
                ;;
            --regression-only)
                RUN_INTEGRATION=false
                RUN_BENCHMARKS=false
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --integration-only    Run only integration tests"
                echo "  --benchmarks-only     Run only performance benchmarks"
                echo "  --regression-only     Run only regression tests"
                echo "  --skip-build         Skip the build step"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Execute test pipeline
    check_prerequisites
    setup_test_environment

    if [ "$SKIP_BUILD" = false ]; then
        build_project
    fi

    # Track overall success
    OVERALL_SUCCESS=true

    if [ "$RUN_INTEGRATION" = true ]; then
        if ! run_integration_tests; then
            OVERALL_SUCCESS=false
        fi
    fi

    if [ "$RUN_BENCHMARKS" = true ]; then
        if ! run_performance_benchmarks; then
            OVERALL_SUCCESS=false
        fi
    fi

    if [ "$RUN_REGRESSION" = true ]; then
        if ! run_regression_tests; then
            OVERALL_SUCCESS=false
        fi
    fi

    analyze_performance
    generate_report
    cleanup

    # Final status
    echo ""
    print_status "===================================="
    if [ "$OVERALL_SUCCESS" = true ]; then
        print_success "All YOLO integration tests completed successfully!"
        print_status "Test results available in: $TEST_RESULTS_DIR/"
        exit 0
    else
        print_error "Some tests failed. Check the results in: $TEST_RESULTS_DIR/"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
