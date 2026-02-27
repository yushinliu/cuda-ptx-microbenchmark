/**
 * @file test_integer_instructions.cu
 * @brief Integer instruction microbenchmark tests (TDD)
 *
 * Tests for IADD3, LOP3, SEL, and SHFL instructions on RTX 4070 (Ada Lovelace).
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
__global__ void iadd3_latency_kernel(uint64_t* cycles, int iterations);
__global__ void iadd3_throughput_kernel(uint64_t* cycles, int iterations);
__global__ void lop3_latency_kernel(uint64_t* cycles, int iterations);
__global__ void lop3_throughput_kernel(uint64_t* cycles, int iterations);
__global__ void sel_latency_kernel(uint64_t* cycles, int iterations);
__global__ void sel_throughput_kernel(uint64_t* cycles, int iterations);
__global__ void shfl_latency_kernel(uint64_t* cycles, int iterations);
__global__ void shfl_throughput_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for integer instruction benchmarks
 */
class IntegerInstructionTest : public GpuTestFixture {
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
// IADD3 Latency Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_iadd3_latency_kernel_exists) {
    const int iterations = 100;
    iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_iadd3_latency_returns_valid_cycles) {
    const int iterations = 1000;
    iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(IntegerInstructionTest, test_iadd3_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // IADD3 latency on Ada Lovelace is typically 4 cycles
    EXPECT_GE(cpi, 1.0) << "IADD3 latency too low";
    EXPECT_LE(cpi, 15.0) << "IADD3 latency too high";
}

TEST_F(IntegerInstructionTest, test_iadd3_latency_zero_iterations) {
    const int iterations = 0;
    iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u);
}

// ============================================================================
// IADD3 Throughput Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_iadd3_throughput_kernel_exists) {
    const int iterations = 100;
    iadd3_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_iadd3_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    iadd3_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(IntegerInstructionTest, test_iadd3_throughput_faster_than_latency) {
    const int iterations = 10000;

    iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t latency_cycles = 0;
    cudaMemcpy(&latency_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    iadd3_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t throughput_cycles = 0;
    cudaMemcpy(&throughput_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(latency_cycles, 0u);
    EXPECT_GT(throughput_cycles, 0u);
    EXPECT_LT(throughput_cycles, latency_cycles * 0.8)
        << "Throughput should be faster than latency";
}

// ============================================================================
// LOP3 Latency Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_lop3_latency_kernel_exists) {
    const int iterations = 100;
    lop3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_lop3_latency_returns_valid_cycles) {
    const int iterations = 1000;
    lop3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(IntegerInstructionTest, test_lop3_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    lop3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // LOP3 latency on Ada Lovelace is typically 4 cycles
    EXPECT_GE(cpi, 1.0) << "LOP3 latency too low";
    EXPECT_LE(cpi, 15.0) << "LOP3 latency too high";
}

// ============================================================================
// LOP3 Throughput Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_lop3_throughput_kernel_exists) {
    const int iterations = 100;
    lop3_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_lop3_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    lop3_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// SEL Latency Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_sel_latency_kernel_exists) {
    const int iterations = 100;
    sel_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_sel_latency_returns_valid_cycles) {
    const int iterations = 1000;
    sel_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(IntegerInstructionTest, test_sel_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    sel_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // SEL latency on Ada Lovelace is typically 2-4 cycles
    EXPECT_GE(cpi, 1.0) << "SEL latency too low";
    EXPECT_LE(cpi, 10.0) << "SEL latency too high";
}

// ============================================================================
// SEL Throughput Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_sel_throughput_kernel_exists) {
    const int iterations = 100;
    sel_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_sel_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    sel_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// SHFL Latency Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_shfl_latency_kernel_exists) {
    const int iterations = 100;
    shfl_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_shfl_latency_returns_valid_cycles) {
    const int iterations = 1000;
    shfl_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(IntegerInstructionTest, test_shfl_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    shfl_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // SHFL latency on Ada Lovelace is typically 10-20 cycles
    EXPECT_GE(cpi, 5.0) << "SHFL latency too low";
    EXPECT_LE(cpi, 50.0) << "SHFL latency too high";
}

// ============================================================================
// SHFL Throughput Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_shfl_throughput_kernel_exists) {
    const int iterations = 100;
    shfl_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(IntegerInstructionTest, test_shfl_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    shfl_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(IntegerInstructionTest, test_iadd3_faster_than_lop3) {
    const int iterations = 10000;

    iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t iadd3_cycles = 0;
    cudaMemcpy(&iadd3_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    lop3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t lop3_cycles = 0;
    cudaMemcpy(&lop3_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double iadd3_cpi = cycles_per_iteration(iadd3_cycles, iterations);
    double lop3_cpi = cycles_per_iteration(lop3_cycles, iterations);

    EXPECT_GT(iadd3_cpi, 0.0);
    EXPECT_GT(lop3_cpi, 0.0);

    // IADD3 and LOP3 should have similar latency
    double ratio = iadd3_cpi / lop3_cpi;
    EXPECT_GE(ratio, 0.5) << "IADD3 much faster than LOP3";
    EXPECT_LE(ratio, 2.0) << "IADD3 much slower than LOP3";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(IntegerInstructionTest, test_integer_null_cycles_pointer) {
    const int iterations = 100;
    iadd3_latency_kernel<<<1, 1>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(IntegerInstructionTest, test_shfl_requires_warp_size) {
    // SHFL requires at least warp-sized thread count
    const int iterations = 100;

    // Should work with warp size
    shfl_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "SHFL should work with warp-sized threads";
}

TEST_F(IntegerInstructionTest, test_integer_consistency) {
    const int iterations = 5000;
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        iadd3_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.2) << "Too much variance in IADD3 latency";
    }
}

}  // namespace microbench
}  // namespace cpm
