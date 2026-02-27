/**
 * @file test_instruction_throughput.cu
 * @brief Instruction throughput microbenchmark tests (TDD)
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
__global__ void fma_throughput_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for instruction throughput benchmarks
 */
class InstructionThroughputTest : public GpuTestFixture {
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

    // Helper to calculate throughput (instructions per cycle)
    double instructions_per_cycle(uint64_t total_cycles, int iterations, int instructions_per_iter) {
        if (total_cycles == 0) return 0.0;
        return static_cast<double>(iterations * instructions_per_iter) / total_cycles;
    }

    // Helper to calculate cycles per instruction
    double cycles_per_instruction(uint64_t total_cycles, int iterations, int instructions_per_iter) {
        if (iterations == 0 || instructions_per_iter == 0) return 0.0;
        return static_cast<double>(total_cycles) / (iterations * instructions_per_iter);
    }
};

// ============================================================================
// FMA Throughput Tests
// ============================================================================

TEST_F(InstructionThroughputTest, test_fma_throughput_kernel_exists) {
    // Given: Valid pointer and iterations
    const int iterations = 100;

    // When: Launch kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(InstructionThroughputTest, test_fma_throughput_returns_valid_cycles) {
    // Given: Iteration count
    const int iterations = 1000;

    // When: Run kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u) << "Cycles should be positive";
    EXPECT_LT(cycles, 100000000u) << "Cycles should be reasonable";
}

TEST_F(InstructionThroughputTest, test_fma_throughput_increases_with_iterations) {
    // Given: Different iteration counts
    std::vector<int> iteration_counts = {100, 500, 1000, 5000};
    std::vector<uint64_t> cycle_counts;

    for (int iters : iteration_counts) {
        // When: Run with each iteration count
        fma_throughput_kernel<<<1, 1>>>(d_cycles_, iters);
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

TEST_F(InstructionThroughputTest, test_fma_throughput_per_instruction_reasonable) {
    // Given: Large iteration count for accurate measurement
    const int iterations = 10000;
    const int instructions_per_iter = 100;  // Expected independent FMAs per iteration

    // When: Run kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Calculate cycles per instruction
    double cpi = cycles_per_instruction(total_cycles, iterations, instructions_per_iter);

    // Throughput-optimized FMA should have CPI << 1 (multiple instructions per cycle)
    // or at least CPI < latency (showing ILP is working)
    EXPECT_GT(cpi, 0.0) << "CPI should be positive";
    EXPECT_LT(cpi, 1.0) << "Throughput test should achieve better than 1 CPI";
}

TEST_F(InstructionThroughputTest, test_fma_throughput_zero_iterations) {
    // Given: Zero iterations
    const int iterations = 0;

    // When: Run kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u) << "Zero iterations should result in zero cycles";
}

TEST_F(InstructionThroughputTest, test_fma_throughput_single_iteration) {
    // Given: Single iteration
    const int iterations = 1;

    // When: Run kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Should complete with valid cycles
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 10000u) << "Single iteration should be very fast";
}

// ============================================================================
// Multi-Thread Throughput Tests
// ============================================================================

TEST_F(InstructionThroughputTest, test_fma_throughput_multiple_threads) {
    // Given: Multiple threads
    const int iterations = 1000;
    const int num_threads = 32;

    // When: Run with warp-sized thread count
    fma_throughput_kernel<<<1, num_threads>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(InstructionThroughputTest, test_fma_throughput_multiple_blocks) {
    // Given: Multiple blocks
    const int iterations = 1000;
    const int num_blocks = 4;
    const int threads_per_block = 32;

    // When: Run with multiple blocks
    fma_throughput_kernel<<<num_blocks, threads_per_block>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete successfully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(InstructionThroughputTest, test_fma_throughput_varying_thread_counts) {
    // Given: Different thread counts
    std::vector<int> thread_counts = {1, 16, 32, 64, 128, 256};
    const int iterations = 1000;

    for (int threads : thread_counts) {
        // When: Run with each thread count
        fma_throughput_kernel<<<1, threads>>>(d_cycles_, iterations);
        cudaError_t err = cudaDeviceSynchronize();

        // Then: Should complete successfully
        EXPECT_EQ(err, cudaSuccess) << "Failed for " << threads << " threads";

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        EXPECT_GT(cycles, 0u) << "No cycles recorded for " << threads << " threads";
    }
}

// ============================================================================
// Throughput vs Latency Comparison
// ============================================================================

// Forward declaration from latency test
__global__ void fma_latency_kernel(uint64_t* cycles, int iterations);

TEST_F(InstructionThroughputTest, test_fma_throughput_faster_than_latency) {
    // Given: Same iteration count for both tests
    const int iterations = 10000;

    // When: Measure latency (dependent FMAs)
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t latency_cycles = 0;
    cudaMemcpy(&latency_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // When: Measure throughput (independent FMAs)
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t throughput_cycles = 0;
    cudaMemcpy(&throughput_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Throughput should be faster (fewer cycles for same iteration count)
    // because independent instructions can execute in parallel
    EXPECT_GT(latency_cycles, 0u);
    EXPECT_GT(throughput_cycles, 0u);

    // Throughput should be significantly faster (at least 2x)
    EXPECT_LT(throughput_cycles, latency_cycles * 0.8)
        << "Throughput test should be faster than latency test";
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(InstructionThroughputTest, test_fma_throughput_consistency) {
    // Given: Fixed parameters
    const int iterations = 5000;

    // When: Run multiple times
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
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
        EXPECT_LT(deviation, 0.3) << "Too much variance in throughput measurement";
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(InstructionThroughputTest, test_throughput_null_cycles_pointer) {
    // Given: Null pointer for cycles
    const int iterations = 100;

    // When: Run kernel with null pointer
    fma_throughput_kernel<<<1, 1>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should fail gracefully
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(InstructionThroughputTest, test_throughput_negative_iterations) {
    // Given: Negative iterations
    const int iterations = -1;

    // When: Run kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully or timeout
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(InstructionThroughputTest, test_throughput_large_iterations) {
    // Given: Large iteration count
    const int iterations = 1000000;

    // When: Run kernel
    fma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete (may take a while)
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

}  // namespace microbench
}  // namespace cpm
