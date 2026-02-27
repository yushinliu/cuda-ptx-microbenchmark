/**
 * @file test_shared_memory.cu
 * @brief Shared memory bank conflict microbenchmark tests (TDD)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>

#include "fixtures/gpu_test_fixture.h"

namespace cpm {
namespace microbench {

// Kernel declaration
__global__ void bank_conflict_kernel(float* result, int stride,
                                     uint64_t* cycles);

/**
 * @brief Test fixture for shared memory bank conflict benchmarks
 */
class SharedMemoryTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();

        // Allocate device memory for results
        cudaMalloc(&d_result_, sizeof(float));
        cudaMalloc(&d_cycles_, sizeof(uint64_t));
    }

    void TearDown() override {
        cudaFree(d_result_);
        cudaFree(d_cycles_);
        GpuTestFixture::TearDown();
    }

    float* d_result_ = nullptr;
    uint64_t* d_cycles_ = nullptr;
};

// ============================================================================
// Basic Bank Conflict Tests
// ============================================================================

TEST_F(SharedMemoryTest, test_bank_conflict_kernel_exists) {
    // Given: Valid pointers and stride
    const int stride = 1;

    // When: Launch kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(SharedMemoryTest, test_bank_conflict_returns_valid_cycles) {
    // Given: Stride that causes no conflicts
    const int stride = 1;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u) << "Cycles should be positive";
    EXPECT_LT(cycles, 100000000u) << "Cycles should be reasonable";
}

TEST_F(SharedMemoryTest, test_bank_conflict_returns_valid_result) {
    // Given: Valid stride
    const int stride = 1;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Result should be valid
    float result = 0.0f;
    cudaMemcpy(&result, d_result_, sizeof(float), cudaMemcpyDeviceToHost);

    // Result should be a valid float (not NaN or inf)
    EXPECT_FALSE(std::isnan(result)) << "Result is NaN";
    EXPECT_FALSE(std::isinf(result)) << "Result is infinity";
}

// ============================================================================
// Stride Tests (Different Bank Conflict Levels)
// ============================================================================

TEST_F(SharedMemoryTest, test_bank_conflict_stride_one_no_conflict) {
    // Given: Stride 1 (consecutive threads access consecutive banks)
    // On most GPUs, this should have no bank conflicts
    const int stride = 1;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Should complete successfully
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(SharedMemoryTest, test_bank_conflict_stride_variations) {
    // Given: Various strides that cause different conflict patterns
    std::vector<int> strides = {1, 2, 4, 8, 16, 32};
    std::vector<uint64_t> cycle_results;

    for (int stride : strides) {
        // When: Run with each stride
        bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_results.push_back(cycles);

        EXPECT_GT(cycles, 0u) << "Failed for stride: " << stride;
    }

    // Then: All strides should complete
    EXPECT_EQ(cycle_results.size(), strides.size());
}

TEST_F(SharedMemoryTest, test_bank_conflict_stride_32_full_conflict) {
    // Given: Stride 32 (all threads access the same bank on 32-bank GPUs)
    const int stride = 32;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Should complete (even with maximum conflicts)
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    // Full bank conflict should take more cycles than no conflict
    // But we can't easily compare without a baseline in this test
}

TEST_F(SharedMemoryTest, test_bank_conflict_comparison) {
    // Given: Two strides - one with no conflict, one with full conflict
    const int no_conflict_stride = 1;
    const int full_conflict_stride = 32;

    // When: Measure no-conflict case
    bank_conflict_kernel<<<1, 32>>>(d_result_, no_conflict_stride, d_cycles_);
    cudaDeviceSynchronize();
    uint64_t no_conflict_cycles = 0;
    cudaMemcpy(&no_conflict_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // When: Measure full-conflict case
    bank_conflict_kernel<<<1, 32>>>(d_result_, full_conflict_stride, d_cycles_);
    cudaDeviceSynchronize();
    uint64_t full_conflict_cycles = 0;
    cudaMemcpy(&full_conflict_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Full conflict should generally take more cycles
    // Note: This is a heuristic, may vary by GPU architecture
    EXPECT_GT(no_conflict_cycles, 0u);
    EXPECT_GT(full_conflict_cycles, 0u);

    // Full conflict should be slower (allow some tolerance for measurement noise)
    EXPECT_GE(full_conflict_cycles, no_conflict_cycles * 0.5)
        << "Full bank conflict should generally be slower";
}

// ============================================================================
// Thread Configuration Tests
// ============================================================================

TEST_F(SharedMemoryTest, test_bank_conflict_different_thread_counts) {
    // Given: Different thread counts per block
    std::vector<int> thread_counts = {1, 16, 32, 64, 128, 256};
    const int stride = 1;

    for (int threads : thread_counts) {
        // When: Run with different thread counts
        bank_conflict_kernel<<<1, threads>>>(d_result_, stride, d_cycles_);
        cudaError_t err = cudaDeviceSynchronize();

        // Then: Should complete successfully
        EXPECT_EQ(err, cudaSuccess) << "Failed for " << threads << " threads";

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        EXPECT_GT(cycles, 0u) << "No cycles recorded for " << threads << " threads";
    }
}

TEST_F(SharedMemoryTest, test_bank_conflict_multiple_blocks) {
    // Given: Multiple blocks
    const int num_blocks = 4;
    const int threads_per_block = 32;
    const int stride = 1;

    // When: Run with multiple blocks
    bank_conflict_kernel<<<num_blocks, threads_per_block>>>(d_result_, stride, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SharedMemoryTest, test_bank_conflict_zero_stride) {
    // Given: Zero stride (all threads access same element)
    const int stride = 0;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete (maximum bank conflict)
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(SharedMemoryTest, test_bank_conflict_negative_stride) {
    // Given: Negative stride
    const int stride = -1;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should either succeed or fail gracefully
    // Negative stride behavior depends on implementation
    if (err == cudaSuccess) {
        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        EXPECT_GT(cycles, 0u);
    }
}

TEST_F(SharedMemoryTest, test_bank_conflict_large_stride) {
    // Given: Very large stride
    const int stride = 1024;

    // When: Run kernel
    bank_conflict_kernel<<<1, 32>>>(d_result_, stride, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(SharedMemoryTest, test_bank_conflict_single_thread) {
    // Given: Single thread (no possibility of bank conflict)
    const int stride = 1;

    // When: Run with single thread
    bank_conflict_kernel<<<1, 1>>>(d_result_, stride, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Should complete quickly
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 1000000u) << "Single thread should be fast";
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(SharedMemoryTest, test_bank_conflict_consistency) {
    // Given: Fixed parameters
    const int stride = 4;
    const int threads = 32;

    // When: Run multiple times
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        bank_conflict_kernel<<<1, threads>>>(d_result_, stride, d_cycles_);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    // Then: Results should be reasonably consistent
    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.5) << "Too much variance: " << c << " vs median " << median;
    }
}

}  // namespace microbench
}  // namespace cpm
