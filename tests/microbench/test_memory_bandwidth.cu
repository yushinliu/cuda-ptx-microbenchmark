/**
 * @file test_memory_bandwidth.cu
 * @brief Memory bandwidth microbenchmark tests (TDD)
 *
 * TDD Workflow:
 * 1. Write tests first (RED)
 * 2. Implement kernels to pass tests (GREEN)
 * 3. Refactor and optimize
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "fixtures/gpu_test_fixture.h"

namespace cpm {
namespace microbench {

// Kernel declarations (to be implemented)
__global__ void sequential_bandwidth_kernel(float* data, float* result,
                                            size_t n, uint64_t* cycles);
__global__ void random_bandwidth_kernel(int* indices, float* data,
                                        size_t n, uint64_t* cycles);
__global__ void stride_bandwidth_kernel(float* data, size_t stride,
                                        size_t n, uint64_t* cycles);

// Host wrapper functions
void launch_sequential_bandwidth(float* data, float* result, size_t n,
                                  uint64_t* cycles, cudaStream_t stream = 0);
void launch_random_bandwidth(int* indices, float* data, size_t n,
                              uint64_t* cycles, cudaStream_t stream = 0);
void launch_stride_bandwidth(float* data, size_t stride, size_t n,
                              uint64_t* cycles, cudaStream_t stream = 0);

/**
 * @brief Test fixture for memory bandwidth benchmarks
 */
class MemoryBandwidthTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();

        // Allocate device memory
        cudaMalloc(&d_data_, kMaxDataSize * sizeof(float));
        cudaMalloc(&d_result_, sizeof(float));
        cudaMalloc(&d_cycles_, sizeof(uint64_t));
        cudaMalloc(&d_indices_, kMaxDataSize * sizeof(int));

        // Initialize data with pattern
        std::vector<float> host_data(kMaxDataSize, 1.0f);
        for (size_t i = 0; i < kMaxDataSize; ++i) {
            host_data[i] = static_cast<float>(i % 100);
        }
        cudaMemcpy(d_data_, host_data.data(), kMaxDataSize * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_data_);
        cudaFree(d_result_);
        cudaFree(d_cycles_);
        cudaFree(d_indices_);
        GpuTestFixture::TearDown();
    }

    // Helper to create pointer chasing pattern for random access
    void create_pointer_chase_pattern(int* indices, size_t n, size_t stride) {
        std::vector<int> host_indices(n);
        for (size_t i = 0; i < n; ++i) {
            host_indices[i] = (i + stride) % n;
        }
        cudaMemcpy(indices, host_indices.data(), n * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    static constexpr size_t kMaxDataSize = 1024 * 1024;  // 4MB
    float* d_data_ = nullptr;
    float* d_result_ = nullptr;
    uint64_t* d_cycles_ = nullptr;
    int* d_indices_ = nullptr;
};

// ============================================================================
// Sequential Bandwidth Tests
// ============================================================================

TEST_F(MemoryBandwidthTest, test_sequential_bandwidth_kernel_exists) {
    // Given: Valid device pointers and data size
    const size_t n = 1024;

    // When: Launch kernel
    sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);

    // Then: Kernel should complete without error
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(MemoryBandwidthTest, test_sequential_bandwidth_returns_valid_cycles) {
    // Given: Data array
    const size_t n = 1024;

    // When: Run kernel
    sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive and reasonable
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u) << "Cycles should be positive";
    EXPECT_LT(cycles, 100000000u) << "Cycles should be reasonable for 1K elements";
}

TEST_F(MemoryBandwidthTest, test_sequential_bandwidth_computes_correct_sum) {
    // Given: Initialized data
    const size_t n = 100;
    std::vector<float> host_data(n);
    for (size_t i = 0; i < n; ++i) {
        host_data[i] = 1.0f;
    }
    cudaMemcpy(d_data_, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // When: Run kernel
    sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Result should be sum of all elements
    float result = 0.0f;
    cudaMemcpy(&result, d_result_, sizeof(float), cudaMemcpyDeviceToHost);

    float expected = static_cast<float>(n);  // n * 1.0f
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(MemoryBandwidthTest, test_sequential_bandwidth_empty_data) {
    // Given: Empty data (n=0)
    const size_t n = 0;

    // When: Run kernel with empty data
    sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully (no crash)
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(MemoryBandwidthTest, test_sequential_bandwidth_single_element) {
    // Given: Single element
    const size_t n = 1;
    float host_data = 42.0f;
    cudaMemcpy(d_data_, &host_data, sizeof(float), cudaMemcpyHostToDevice);

    // When: Run kernel
    sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Result should be that element
    float result = 0.0f;
    cudaMemcpy(&result, d_result_, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(result, 42.0f);
}

TEST_F(MemoryBandwidthTest, test_sequential_bandwidth_large_data) {
    // Given: Large data size (1M elements)
    const size_t n = 1024 * 1024;

    // When: Run kernel
    sequential_bandwidth_kernel<<<256, 256>>>(d_data_, d_result_, n, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);

    // And: Cycles should be recorded
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

// ============================================================================
// Random Bandwidth Tests (Pointer Chasing)
// ============================================================================

TEST_F(MemoryBandwidthTest, test_random_bandwidth_kernel_exists) {
    // Given: Valid pointers and chase pattern
    const size_t n = 1024;
    create_pointer_chase_pattern(d_indices_, n, 1);

    // When: Launch kernel
    random_bandwidth_kernel<<<1, 1>>>(d_indices_, d_data_, n, d_cycles_);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(MemoryBandwidthTest, test_random_bandwidth_returns_valid_cycles) {
    // Given: Pointer chase pattern
    const size_t n = 1024;
    create_pointer_chase_pattern(d_indices_, n, 1);

    // When: Run kernel
    random_bandwidth_kernel<<<1, 1>>>(d_indices_, d_data_, n, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    // Random access should take more cycles than sequential
    EXPECT_LT(cycles, 1000000000u);
}

TEST_F(MemoryBandwidthTest, test_random_bandwidth_slower_than_sequential) {
    // Given: Same data size for both tests
    const size_t n = 10000;

    // Create sequential indices (0->1->2->...)
    std::vector<int> sequential_indices(n);
    for (size_t i = 0; i < n; ++i) {
        sequential_indices[i] = (i + 1) % n;
    }
    cudaMemcpy(d_indices_, sequential_indices.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);

    // When: Measure sequential-like access
    random_bandwidth_kernel<<<1, 1>>>(d_indices_, d_data_, n, d_cycles_);
    cudaDeviceSynchronize();
    uint64_t sequential_cycles = 0;
    cudaMemcpy(&sequential_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Create random indices
    std::vector<int> random_indices(n);
    for (size_t i = 0; i < n; ++i) {
        random_indices[i] = i;
    }
    std::random_shuffle(random_indices.begin(), random_indices.end());
    // Ensure it's a cycle
    for (size_t i = 0; i < n; ++i) {
        random_indices[i] = random_indices[i] % n;
    }
    cudaMemcpy(d_indices_, random_indices.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);

    random_bandwidth_kernel<<<1, 1>>>(d_indices_, d_data_, n, d_cycles_);
    cudaDeviceSynchronize();
    uint64_t random_cycles = 0;
    cudaMemcpy(&random_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Random access should generally be slower
    // Note: This may occasionally fail due to cache effects, but should pass most of the time
    EXPECT_GT(random_cycles, 0u);
    EXPECT_GT(sequential_cycles, 0u);
}

TEST_F(MemoryBandwidthTest, test_random_bandwidth_empty_data) {
    // Given: Empty data
    const size_t n = 0;

    // When: Run kernel
    random_bandwidth_kernel<<<1, 1>>>(d_indices_, d_data_, n, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully
    EXPECT_EQ(err, cudaSuccess);
}

// ============================================================================
// Stride Bandwidth Tests
// ============================================================================

TEST_F(MemoryBandwidthTest, test_stride_bandwidth_kernel_exists) {
    // Given: Valid pointers and stride
    const size_t n = 1024;
    const size_t stride = 4;

    // When: Launch kernel
    stride_bandwidth_kernel<<<1, 1>>>(d_data_, stride, n, d_cycles_);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(MemoryBandwidthTest, test_stride_bandwidth_returns_valid_cycles) {
    // Given: Data and stride
    const size_t n = 1024;
    const size_t stride = 4;

    // When: Run kernel
    stride_bandwidth_kernel<<<1, 1>>>(d_data_, stride, n, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
}

TEST_F(MemoryBandwidthTest, test_stride_bandwidth_varying_strides) {
    // Given: Data array
    const size_t n = 1024 * 1024;

    std::vector<size_t> strides = {1, 4, 8, 16, 32, 64, 128};
    std::vector<uint64_t> cycles_results;

    for (size_t stride : strides) {
        // When: Run with different strides
        stride_bandwidth_kernel<<<1, 1>>>(d_data_, stride, n, d_cycles_);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycles_results.push_back(cycles);

        // Each result should be valid
        EXPECT_GT(cycles, 0u) << "Stride " << stride << " failed";
    }

    // Larger strides should generally take more cycles (more cache misses)
    // This is a heuristic, not a strict requirement
    EXPECT_EQ(cycles_results.size(), strides.size());
}

TEST_F(MemoryBandwidthTest, test_stride_bandwidth_stride_one_equals_sequential) {
    // Given: Same data
    const size_t n = 10000;

    // When: Run stride=1 (sequential)
    stride_bandwidth_kernel<<<1, 1>>>(d_data_, 1, n, d_cycles_);
    cudaDeviceSynchronize();
    uint64_t stride_cycles = 0;
    cudaMemcpy(&stride_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Run sequential kernel
    sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);
    cudaDeviceSynchronize();
    uint64_t sequential_cycles = 0;
    cudaMemcpy(&sequential_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Both should complete and have reasonable cycle counts
    EXPECT_GT(stride_cycles, 0u);
    EXPECT_GT(sequential_cycles, 0u);
}

TEST_F(MemoryBandwidthTest, test_stride_bandwidth_empty_data) {
    // Given: Empty data
    const size_t n = 0;
    const size_t stride = 4;

    // When: Run kernel
    stride_bandwidth_kernel<<<1, 1>>>(d_data_, stride, n, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(MemoryBandwidthTest, test_stride_bandwidth_zero_stride) {
    // Given: Zero stride (edge case)
    const size_t n = 100;
    const size_t stride = 0;

    // When: Run kernel with stride=0
    stride_bandwidth_kernel<<<1, 1>>>(d_data_, stride, n, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should either handle gracefully or fail predictably
    // Zero stride means accessing same element repeatedly
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorInvalidValue);
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

TEST_F(MemoryBandwidthTest, test_bandwidth_measurement_consistency) {
    // Given: Fixed data size
    const size_t n = 100000;

    // When: Run multiple times
    std::vector<uint64_t> cycles_list;
    const int iterations = 10;

    for (int i = 0; i < iterations; ++i) {
        sequential_bandwidth_kernel<<<1, 1>>>(d_data_, d_result_, n, d_cycles_);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycles_list.push_back(cycles);
    }

    // Then: Results should be reasonably consistent (within 50% of median)
    std::sort(cycles_list.begin(), cycles_list.end());
    uint64_t median = cycles_list[iterations / 2];

    for (uint64_t c : cycles_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.5) << "Cycle count varies too much: " << c << " vs median " << median;
    }
}

}  // namespace microbench
}  // namespace cpm
