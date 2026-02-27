/**
 * @file test_warp_scheduler.cu
 * @brief Warp scheduler microbenchmark tests (TDD)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "fixtures/gpu_test_fixture.h"

namespace cpm {
namespace microbench {

// Kernel declaration and implementation
__global__ void warp_scheduler_latency(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Simple computation to simulate warp scheduling work
    volatile int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += i;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    if (threadIdx.x == 0 && cycles != nullptr) {
        *cycles = end_cycle - start_cycle;
    }
}

/**
 * @brief Test fixture for warp scheduler benchmarks
 */
class WarpSchedulerTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_cycles_, sizeof(uint64_t));
    }

    void TearDown() override {
        cudaFree(d_cycles_);
        GpuTestFixture::TearDown();
    }

    uint64_t* d_cycles_ = nullptr;
};

// ============================================================================
// Basic Warp Scheduler Tests
// ============================================================================

TEST_F(WarpSchedulerTest, test_warp_scheduler_kernel_exists) {
    // Given: Valid pointer and iterations
    const int iterations = 100;

    // When: Launch kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_returns_valid_cycles) {
    // Given: Iteration count
    const int iterations = 1000;

    // When: Run kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u) << "Cycles should be positive";
    EXPECT_LT(cycles, 100000000u) << "Cycles should be reasonable";
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_increases_with_iterations) {
    // Given: Different iteration counts
    std::vector<int> iteration_counts = {100, 500, 1000, 5000};
    std::vector<uint64_t> cycle_counts;

    for (int iters : iteration_counts) {
        // When: Run with each iteration count
        warp_scheduler_latency<<<1, 1>>>(d_cycles_, iters);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_counts.push_back(cycles);

        EXPECT_GT(cycles, 0u) << "Failed for iterations: " << iters;
    }

    // Then: More iterations should generally take more cycles
    for (size_t i = 1; i < cycle_counts.size(); ++i) {
        EXPECT_GE(cycle_counts[i], cycle_counts[i-1] * 0.5)
            << "More iterations should generally take more cycles";
    }
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_zero_iterations) {
    // Given: Zero iterations
    const int iterations = 0;

    // When: Run kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u) << "Zero iterations should result in zero cycles";
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_single_iteration) {
    // Given: Single iteration
    const int iterations = 1;

    // When: Run kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Should complete with valid cycles
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 10000u) << "Single iteration should be very fast";
}

// ============================================================================
// Multi-Thread Tests
// ============================================================================

TEST_F(WarpSchedulerTest, test_warp_scheduler_multiple_threads) {
    // Given: Multiple threads
    const int iterations = 1000;
    const int num_threads = 32;

    // When: Run with warp-sized thread count
    warp_scheduler_latency<<<1, num_threads>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_full_warp) {
    // Given: Full warp (32 threads)
    const int iterations = 1000;
    const int warp_size = 32;

    // When: Run with full warp
    warp_scheduler_latency<<<1, warp_size>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Should complete with valid cycles
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_multiple_warps) {
    // Given: Multiple warps per block
    const int iterations = 1000;
    const int num_warps = 4;
    const int threads_per_block = num_warps * 32;

    // When: Run with multiple warps
    warp_scheduler_latency<<<1, threads_per_block>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_multiple_blocks) {
    // Given: Multiple blocks
    const int iterations = 1000;
    const int num_blocks = 4;
    const int threads_per_block = 32;

    // When: Run with multiple blocks
    warp_scheduler_latency<<<num_blocks, threads_per_block>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

// ============================================================================
// Scheduler Latency Characteristics Tests
// ============================================================================

TEST_F(WarpSchedulerTest, test_warp_scheduler_latency_per_iteration) {
    // Given: Large iteration count for accurate measurement
    const int iterations = 10000;

    // When: Run kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Calculate cycles per iteration
    double cpi = static_cast<double>(total_cycles) / iterations;

    // Warp scheduler latency should be small (1-2 cycles typically)
    EXPECT_GE(cpi, 0.5) << "Scheduler latency too low";
    EXPECT_LE(cpi, 10.0) << "Scheduler latency too high";
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_consistency) {
    // Given: Fixed parameters
    const int iterations = 5000;

    // When: Run multiple times
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
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
        EXPECT_LT(deviation, 0.3) << "Too much variance: " << c << " vs median " << median;
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(WarpSchedulerTest, test_warp_scheduler_null_cycles_pointer) {
    // Given: Null pointer for cycles
    const int iterations = 100;

    // When: Run kernel with null pointer
    warp_scheduler_latency<<<1, 1>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should fail gracefully
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_negative_iterations) {
    // Given: Negative iterations
    const int iterations = -1;

    // When: Run kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully or timeout
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_large_iterations) {
    // Given: Large iteration count
    const int iterations = 1000000;

    // When: Run kernel
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete (may take a while)
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(WarpSchedulerTest, test_warp_scheduler_single_thread) {
    // Given: Single thread
    const int iterations = 1000;

    // When: Run with single thread
    warp_scheduler_latency<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Should complete quickly
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
}

}  // namespace microbench
}  // namespace cpm
