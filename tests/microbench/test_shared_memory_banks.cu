/**
 * @file test_shared_memory_banks.cu
 * @brief Shared memory bank conflict microbenchmark tests (TDD)
 *
 * Tests for shared memory bank conflict measurement on RTX 4070.
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

// Kernel declarations
__global__ void shared_memory_no_conflict_kernel(uint64_t* cycles, int iterations);
__global__ void shared_memory_2way_conflict_kernel(uint64_t* cycles, int iterations);
__global__ void shared_memory_4way_conflict_kernel(uint64_t* cycles, int iterations);
__global__ void shared_memory_8way_conflict_kernel(uint64_t* cycles, int iterations);
__global__ void shared_memory_32way_conflict_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for shared memory bank conflict benchmarks
 */
class SharedMemoryBankTest : public GpuTestFixture {
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

    double cycles_per_iteration(uint64_t total_cycles, int iterations) {
        return static_cast<double>(total_cycles) / iterations;
    }
};

// ============================================================================
// No Conflict Tests
// ============================================================================

TEST_F(SharedMemoryBankTest, test_no_conflict_kernel_exists) {
    const int iterations = 100;
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(SharedMemoryBankTest, test_no_conflict_returns_valid_cycles) {
    const int iterations = 1000;
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(SharedMemoryBankTest, test_no_conflict_per_iteration_reasonable) {
    const int iterations = 10000;
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // Shared memory latency without conflicts is typically 20-40 cycles
    EXPECT_GE(cpi, 10.0) << "No-conflict latency too low";
    EXPECT_LE(cpi, 100.0) << "No-conflict latency too high";
}

// ============================================================================
// 2-Way Conflict Tests
// ============================================================================

TEST_F(SharedMemoryBankTest, test_2way_conflict_kernel_exists) {
    const int iterations = 100;
    shared_memory_2way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(SharedMemoryBankTest, test_2way_conflict_returns_valid_cycles) {
    const int iterations = 1000;
    shared_memory_2way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// 4-Way Conflict Tests
// ============================================================================

TEST_F(SharedMemoryBankTest, test_4way_conflict_kernel_exists) {
    const int iterations = 100;
    shared_memory_4way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(SharedMemoryBankTest, test_4way_conflict_returns_valid_cycles) {
    const int iterations = 1000;
    shared_memory_4way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// 8-Way Conflict Tests
// ============================================================================

TEST_F(SharedMemoryBankTest, test_8way_conflict_kernel_exists) {
    const int iterations = 100;
    shared_memory_8way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(SharedMemoryBankTest, test_8way_conflict_returns_valid_cycles) {
    const int iterations = 1000;
    shared_memory_8way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// 32-Way Conflict Tests
// ============================================================================

TEST_F(SharedMemoryBankTest, test_32way_conflict_kernel_exists) {
    const int iterations = 100;
    shared_memory_32way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(SharedMemoryBankTest, test_32way_conflict_returns_valid_cycles) {
    const int iterations = 1000;
    shared_memory_32way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// Conflict Comparison Tests
// ============================================================================

TEST_F(SharedMemoryBankTest, test_conflict_increases_latency) {
    const int iterations = 10000;

    // Measure no conflict
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t no_conflict_cycles = 0;
    cudaMemcpy(&no_conflict_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Measure 2-way conflict
    shared_memory_2way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t conflict2_cycles = 0;
    cudaMemcpy(&conflict2_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Measure 4-way conflict
    shared_memory_4way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t conflict4_cycles = 0;
    cudaMemcpy(&conflict4_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double no_conflict_cpi = cycles_per_iteration(no_conflict_cycles, iterations);
    double conflict2_cpi = cycles_per_iteration(conflict2_cycles, iterations);
    double conflict4_cpi = cycles_per_iteration(conflict4_cycles, iterations);

    EXPECT_GT(no_conflict_cpi, 0.0);
    EXPECT_GT(conflict2_cpi, 0.0);
    EXPECT_GT(conflict4_cpi, 0.0);

    // Higher conflict should generally mean higher latency
    EXPECT_GE(conflict2_cpi, no_conflict_cpi * 0.8)
        << "2-way conflict should not be significantly faster than no conflict";
    EXPECT_GE(conflict4_cpi, conflict2_cpi * 0.8)
        << "4-way conflict should not be significantly faster than 2-way";
}

TEST_F(SharedMemoryBankTest, test_32way_conflict_slowest) {
    const int iterations = 10000;

    // Measure all conflict levels
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t no_conflict_cycles = 0;
    cudaMemcpy(&no_conflict_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    shared_memory_32way_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t conflict32_cycles = 0;
    cudaMemcpy(&conflict32_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double no_conflict_cpi = cycles_per_iteration(no_conflict_cycles, iterations);
    double conflict32_cpi = cycles_per_iteration(conflict32_cycles, iterations);

    EXPECT_GT(no_conflict_cpi, 0.0);
    EXPECT_GT(conflict32_cpi, 0.0);

    // 32-way conflict should be the slowest
    EXPECT_GE(conflict32_cpi, no_conflict_cpi * 0.9)
        << "32-way conflict should be slower than no conflict";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SharedMemoryBankTest, test_shared_memory_null_cycles_pointer) {
    const int iterations = 100;
    shared_memory_no_conflict_kernel<<<1, 32>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(SharedMemoryBankTest, test_shared_memory_zero_iterations) {
    const int iterations = 0;
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u);
}

TEST_F(SharedMemoryBankTest, test_shared_memory_negative_iterations) {
    const int iterations = -1;
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(SharedMemoryBankTest, test_shared_memory_large_iterations) {
    const int iterations = 1000000;
    shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(SharedMemoryBankTest, test_shared_memory_consistency) {
    const int iterations = 5000;
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        shared_memory_no_conflict_kernel<<<1, 32>>>(d_cycles_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.3) << "Too much variance in shared memory latency";
    }
}

TEST_F(SharedMemoryBankTest, test_shared_memory_varying_thread_counts) {
    std::vector<int> thread_counts = {1, 16, 32, 64, 128, 256};
    const int iterations = 1000;

    for (int threads : thread_counts) {
        shared_memory_no_conflict_kernel<<<1, threads>>>(d_cycles_, iterations);
        cudaError_t err = cudaDeviceSynchronize();

        EXPECT_EQ(err, cudaSuccess) << "Failed for " << threads << " threads";

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        EXPECT_GT(cycles, 0u) << "No cycles recorded for " << threads << " threads";
    }
}

}  // namespace microbench
}  // namespace cpm
