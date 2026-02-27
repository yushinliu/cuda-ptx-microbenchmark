/**
 * @file test_instruction_latency.cu
 * @brief Instruction latency microbenchmark tests (TDD)
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
__global__ void fma_latency_kernel(uint64_t* cycles, int iterations);
__global__ void add_latency_kernel(uint64_t* cycles, int iterations);
__global__ void mul_latency_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for instruction latency benchmarks
 */
class InstructionLatencyTest : public GpuTestFixture {
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

    // Helper to calculate cycles per iteration
    double cycles_per_iteration(uint64_t total_cycles, int iterations) {
        return static_cast<double>(total_cycles) / iterations;
    }
};

// ============================================================================
// FMA Latency Tests
// ============================================================================

TEST_F(InstructionLatencyTest, test_fma_latency_kernel_exists) {
    // Given: Valid pointer and iterations
    const int iterations = 100;

    // When: Launch kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(InstructionLatencyTest, test_fma_latency_returns_valid_cycles) {
    // Given: Iteration count
    const int iterations = 1000;

    // When: Run kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u) << "Cycles should be positive";
    EXPECT_LT(cycles, 100000000u) << "Cycles should be reasonable";
}

TEST_F(InstructionLatencyTest, test_fma_latency_increases_with_iterations) {
    // Given: Different iteration counts
    std::vector<int> iteration_counts = {100, 500, 1000, 5000};
    std::vector<uint64_t> cycle_counts;

    for (int iters : iteration_counts) {
        // When: Run with each iteration count
        fma_latency_kernel<<<1, 1>>>(d_cycles_, iters);
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

TEST_F(InstructionLatencyTest, test_fma_latency_per_iteration_reasonable) {
    // Given: Large iteration count for accurate measurement
    const int iterations = 10000;

    // When: Run kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Calculate cycles per FMA
    double cpi = cycles_per_iteration(total_cycles, iterations);

    // FMA latency on modern GPUs is typically 4-6 cycles
    EXPECT_GE(cpi, 1.0) << "FMA latency too low (dependency chain may be broken)";
    EXPECT_LE(cpi, 20.0) << "FMA latency too high";
}

TEST_F(InstructionLatencyTest, test_fma_latency_zero_iterations) {
    // Given: Zero iterations
    const int iterations = 0;

    // When: Run kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u) << "Zero iterations should result in zero cycles";
}

TEST_F(InstructionLatencyTest, test_fma_latency_single_iteration) {
    // Given: Single iteration
    const int iterations = 1;

    // When: Run kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Should complete with valid cycles
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 10000u) << "Single iteration should be very fast";
}

// ============================================================================
// ADD Latency Tests
// ============================================================================

TEST_F(InstructionLatencyTest, test_add_latency_kernel_exists) {
    // Given: Valid pointer and iterations
    const int iterations = 100;

    // When: Launch kernel
    add_latency_kernel<<<1, 1>>>(d_cycles_, iterations);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(InstructionLatencyTest, test_add_latency_returns_valid_cycles) {
    // Given: Iteration count
    const int iterations = 1000;

    // When: Run kernel
    add_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(InstructionLatencyTest, test_add_latency_per_iteration_reasonable) {
    // Given: Large iteration count
    const int iterations = 10000;

    // When: Run kernel
    add_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Calculate cycles per ADD
    double cpi = cycles_per_iteration(total_cycles, iterations);

    // ADD latency on modern GPUs is typically 2-4 cycles
    EXPECT_GE(cpi, 1.0) << "ADD latency too low";
    EXPECT_LE(cpi, 10.0) << "ADD latency too high";
}

// ============================================================================
// MUL Latency Tests
// ============================================================================

TEST_F(InstructionLatencyTest, test_mul_latency_kernel_exists) {
    // Given: Valid pointer and iterations
    const int iterations = 100;

    // When: Launch kernel
    mul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(InstructionLatencyTest, test_mul_latency_returns_valid_cycles) {
    // Given: Iteration count
    const int iterations = 1000;

    // When: Run kernel
    mul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(InstructionLatencyTest, test_mul_latency_per_iteration_reasonable) {
    // Given: Large iteration count
    const int iterations = 10000;

    // When: Run kernel
    mul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Calculate cycles per MUL
    double cpi = cycles_per_iteration(total_cycles, iterations);

    // MUL latency on modern GPUs is typically 4-6 cycles
    EXPECT_GE(cpi, 1.0) << "MUL latency too low";
    EXPECT_LE(cpi, 15.0) << "MUL latency too high";
}

// ============================================================================
// Instruction Comparison Tests
// ============================================================================

TEST_F(InstructionLatencyTest, test_add_faster_than_fma) {
    // Given: Same iteration count
    const int iterations = 10000;

    // When: Measure ADD latency
    add_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t add_cycles = 0;
    cudaMemcpy(&add_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // When: Measure FMA latency
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t fma_cycles = 0;
    cudaMemcpy(&fma_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: ADD should generally be faster or similar to FMA
    double add_cpi = cycles_per_iteration(add_cycles, iterations);
    double fma_cpi = cycles_per_iteration(fma_cycles, iterations);

    EXPECT_GT(add_cpi, 0.0);
    EXPECT_GT(fma_cpi, 0.0);

    // ADD is typically simpler than FMA, so should be similar or faster
    EXPECT_LE(add_cpi, fma_cpi * 1.5) << "ADD should not be significantly slower than FMA";
}

TEST_F(InstructionLatencyTest, test_fma_mul_similar_latency) {
    // Given: Same iteration count
    const int iterations = 10000;

    // When: Measure FMA latency
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t fma_cycles = 0;
    cudaMemcpy(&fma_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // When: Measure MUL latency
    mul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t mul_cycles = 0;
    cudaMemcpy(&mul_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: FMA and MUL should have similar latency (both use FP multiplier)
    double fma_cpi = cycles_per_iteration(fma_cycles, iterations);
    double mul_cpi = cycles_per_iteration(mul_cycles, iterations);

    EXPECT_GT(fma_cpi, 0.0);
    EXPECT_GT(mul_cpi, 0.0);

    // They should be within 2x of each other
    double ratio = fma_cpi / mul_cpi;
    EXPECT_GE(ratio, 0.5) << "FMA latency much lower than MUL";
    EXPECT_LE(ratio, 2.0) << "FMA latency much higher than MUL";
}

// ============================================================================
// Consistency Tests
// ============================================================================

TEST_F(InstructionLatencyTest, test_fma_latency_consistency) {
    // Given: Fixed parameters
    const int iterations = 5000;

    // When: Run multiple times
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    // Then: Results should be consistent
    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.2) << "Too much variance in FMA latency";
    }
}

TEST_F(InstructionLatencyTest, test_add_latency_consistency) {
    // Given: Fixed parameters
    const int iterations = 5000;

    // When: Run multiple times
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        add_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    // Then: Results should be consistent
    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.2) << "Too much variance in ADD latency";
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(InstructionLatencyTest, test_latency_null_cycles_pointer) {
    // Given: Null pointer for cycles
    const int iterations = 100;

    // When: Run kernel with null pointer
    fma_latency_kernel<<<1, 1>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should fail gracefully
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(InstructionLatencyTest, test_latency_negative_iterations) {
    // Given: Negative iterations
    const int iterations = -1;

    // When: Run kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully or timeout
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(InstructionLatencyTest, test_latency_large_iterations) {
    // Given: Large iteration count
    const int iterations = 1000000;

    // When: Run kernel
    fma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete (may take a while)
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

}  // namespace microbench
}  // namespace cpm
