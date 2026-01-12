# YOLO Acceleration System Integration Tests

This directory contains comprehensive integration tests for the YOLO acceleration system, covering end-to-end inference accuracy and performance benchmarking.

## Overview

The integration tests validate the complete YOLO acceleration pipeline including:

- **End-to-End Inference Accuracy**: Validates that the entire inference pipeline produces correct results
- **Performance Benchmarking**: Measures and validates system performance metrics
- **Regression Testing**: Ensures performance doesn't degrade over time
- **Memory Usage Validation**: Monitors GPU memory consumption and efficiency
- **Multi-Component Integration**: Tests interaction between all system components

## Test Structure

### Core Test Files

- `yolo_integration_tests.cpp` - Main integration test suite with accuracy and functionality tests
- `yolo_performance_benchmarks.cpp` - Comprehensive performance benchmarking suite
- `CMakeLists.txt` - Build configuration for integration tests
- `run_yolo_tests.sh` - Automated test runner script

### Test Categories

#### 1. End-to-End Accuracy Tests (`YOLOIntegrationTest`)

- **ONNX Model Inference Accuracy**: Tests complete ONNX model loading and inference pipeline
- **PyTorch Model Inference Accuracy**: Validates PyTorch model integration and batch processing
- **Batch Inference Manager Performance**: Tests dynamic batching with accuracy validation
- **Multi-Stream Inference Performance**: Validates concurrent inference with load balancing
- **Operator Fusion Performance**: Tests fused operations maintain accuracy while improving performance
- **TensorRT Integration Performance**: Validates TensorRT optimization integration

#### 2. Performance Benchmarks (`YOLOPerformanceBenchmark`)

- **Single Image Inference Latency**: Measures inference time across different image sizes
- **Batch Inference Throughput**: Tests throughput scaling with different batch sizes
- **Multi-Stream Concurrent Performance**: Evaluates concurrent processing capabilities
- **Operator Fusion Performance Impact**: Quantifies fusion optimization benefits
- **Memory Usage Scaling**: Analyzes memory consumption patterns
- **Load Balance Strategy Comparison**: Compares different load balancing approaches

#### 3. Regression Tests (`YOLOPerformanceRegressionTest`)

- **Inference Time Regression**: Detects performance degradation in inference speed
- **Throughput Regression**: Monitors throughput performance over time
- **Memory Usage Regression**: Tracks memory consumption increases

## Performance Metrics and Thresholds

### Key Performance Indicators

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Single Inference Latency (640x640) | < 25ms | Maximum acceptable inference time |
| Batch Throughput (BS=8) | > 200 QPS | Minimum batch processing throughput |
| Multi-Stream Throughput | > 150 QPS | Minimum concurrent processing rate |
| GPU Utilization | > 80% | Minimum GPU resource utilization |
| Memory Efficiency | < 100MB/request | Maximum memory per inference request |
| Fusion Speedup | > 1.2x | Minimum operator fusion improvement |

### Baseline Metrics

The system maintains baseline performance metrics in `tests/fixtures/yolo_performance_baseline.json`:

```json
{
  "single_inference": {
    "640x640": {
      "avg_latency_ms": 22.0,
      "throughput_qps": 45.5,
      "memory_usage_mb": 1200
    }
  },
  "batch_inference": {
    "batch_size_8": {
      "throughput_qps": 493.8,
      "memory_usage_mb": 5800
    }
  }
}
```

## Running the Tests

### Prerequisites

- CUDA 11.0+ with compatible GPU
- CMake 3.18+
- Google Test framework
- Google Benchmark library
- OpenCV 4.0+
- C++17 compatible compiler

### Quick Start

```bash
# Run all tests
./tests/integration/run_yolo_tests.sh

# Run specific test categories
./tests/integration/run_yolo_tests.sh --integration-only
./tests/integration/run_yolo_tests.sh --benchmarks-only
./tests/integration/run_yolo_tests.sh --regression-only

# Skip build step (if already built)
./tests/integration/run_yolo_tests.sh --skip-build
```

### Manual Execution

```bash
# Build tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make yolo_integration_tests yolo_performance_benchmarks

# Run integration tests
./yolo_integration_tests --gtest_output=xml:integration_results.xml

# Run performance benchmarks
./yolo_performance_benchmarks --gtest_output=xml:benchmark_results.xml

# Run detailed Google Benchmark profiling
./yolo_performance_benchmarks --benchmark --benchmark_format=json --benchmark_out=detailed_benchmarks.json
```

## Test Results and Reporting

### Generated Files

After running tests, the following files are generated in `test_results/`:

- `integration_test_results.xml` - Integration test results in JUnit XML format
- `benchmark_test_results.xml` - Performance benchmark results
- `regression_test_results.xml` - Regression test results
- `yolo_benchmark_results.csv` - Detailed performance metrics in CSV format
- `detailed_benchmarks.json` - Google Benchmark detailed profiling data
- `yolo_test_report.html` - Comprehensive HTML test report

### Performance Analysis

The test suite automatically compares current performance against baseline metrics and flags regressions:

- **Green**: Performance within acceptable range
- **Yellow**: Minor performance degradation (< 5%)
- **Red**: Significant regression requiring investigation (> 5%)

### Continuous Integration

The tests are designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run YOLO Integration Tests
  run: |
    ./tests/integration/run_yolo_tests.sh

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_results/
```

## Test Configuration

### Customizing Test Parameters

Key test parameters can be modified in the test files:

```cpp
// In yolo_integration_tests.cpp
static constexpr int num_test_images_ = 10;
static constexpr float max_inference_time_ms_ = 50.0f;
static constexpr float min_throughput_qps_ = 20.0f;

// In yolo_performance_benchmarks.cpp
static constexpr int images_per_size_ = 20;
std::vector<int> batch_sizes = {1, 2, 4, 8, 16};
std::vector<int> stream_counts = {1, 2, 4, 8};
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES` - Specify which GPU to use for testing
- `YOLO_TEST_MODEL_PATH` - Custom path to YOLO model files
- `YOLO_TEST_ITERATIONS` - Override default benchmark iteration count

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch sizes or concurrent request counts
   - Check GPU memory availability with `nvidia-smi`

2. **Model Loading Failures**
   - Ensure YOLO model files are available
   - Check file permissions and paths

3. **Performance Regression False Positives**
   - Verify system is not under load during testing
   - Check for thermal throttling
   - Ensure consistent GPU clock speeds

### Debug Mode

Enable debug output for detailed test information:

```bash
# Run with verbose output
./yolo_integration_tests --gtest_verbose

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
./yolo_integration_tests
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate performance thresholds and assertions
3. Update baseline metrics if introducing new test cases
4. Ensure tests are deterministic and reproducible
5. Add documentation for new test categories

### Test Development Guidelines

- **Accuracy Tests**: Focus on correctness validation
- **Performance Tests**: Measure specific metrics with clear thresholds
- **Regression Tests**: Compare against historical baselines
- **Integration Tests**: Test component interactions
- **Stress Tests**: Validate system limits and error handling

## References

- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [CUDA Testing Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [YOLO Model Documentation](../src/yolo/README.md)
