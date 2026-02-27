/**
 * @file test_double_precision.cu
 * @brief Double precision instruction microbenchmark tests (TDD)
 *
 * Tests for DADD, DMUL, and DFMA instructions on RTX 4070 (Ada Lovelace).
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
__global__ void dadd_latency_kernel(uint64_t* cycles, int iterations);
__global__ void dadd_throughput_kernel(uint64_t* cycles, int iterations);
__global__ void dmul_latency_kernel(uint64_t* cycles, int iterations);
__global__ void dmul_throughput_kernel(uint64_t* cycles, int iterations);
__global__ void dfma_latency_kernel(uint64_t* cycles, int iterations);
__global__ void dfma_throughput_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for double precision instruction benchmarks
 */
class DoublePrecisionTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_cycles_, sizeof(uint64_t));

        // Check if device supports double precision
        int major = device_props_.major;
        int minor = device_props_.minor;
        double_precision_supported_ = (major >= 2);  // All GPUs since Fermi support FP64
    }

    void TearDown() override {
        cudaFree(d_cycles_);
        GpuTestFixture::TearDown();
    }

    uint64_t* d_cycles_ = nullptr;
    bool double_precision_supported_ = false;

    double cycles_per_iteration(uint64_t total_cycles, int iterations) {
        return static_cast<double>(total_cycles) / iterations;
    }
};

// ============================================================================
// DADD Latency Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dadd_latency_kernel_exists) {
    const int iterations = 100;
    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(DoublePrecisionTest, test_dadd_latency_returns_valid_cycles) {
    const int iterations = 1000;
    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(DoublePrecisionTest, test_dadd_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // DADD latency on Ada Lovelace is typically 5-8 cycles
    EXPECT_GE(cpi, 1.0) << "DADD latency too low";
    EXPECT_LE(cpi, 20.0) << "DADD latency too high";
}

TEST_F(DoublePrecisionTest, test_dadd_latency_zero_iterations) {
    const int iterations = 0;
    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u);
}

// ============================================================================
// DADD Throughput Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dadd_throughput_kernel_exists) {
    const int iterations = 100;
    dadd_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(DoublePrecisionTest, test_dadd_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    dadd_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(DoublePrecisionTest, test_dadd_throughput_faster_than_latency) {
    const int iterations = 10000;

    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t latency_cycles = 0;
    cudaMemcpy(&latency_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    dadd_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t throughput_cycles = 0;
    cudaMemcpy(&throughput_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(latency_cycles, 0u);
    EXPECT_GT(throughput_cycles, 0u);
    EXPECT_LT(throughput_cycles, latency_cycles * 0.8)
        << "Throughput should be faster than latency";
}

// ============================================================================
// DMUL Latency Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dmul_latency_kernel_exists) {
    const int iterations = 100;
    dmul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(DoublePrecisionTest, test_dmul_latency_returns_valid_cycles) {
    const int iterations = 1000;
    dmul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(DoublePrecisionTest, test_dmul_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    dmul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // DMUL latency on Ada Lovelace is typically 5-8 cycles
    EXPECT_GE(cpi, 1.0) << "DMUL latency too low";
    EXPECT_LE(cpi, 20.0) << "DMUL latency too high";
}

// ============================================================================
// DMUL Throughput Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dmul_throughput_kernel_exists) {
    const int iterations = 100;
    dmul_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(DoublePrecisionTest, test_dmul_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    dmul_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// DFMA Latency Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dfma_latency_kernel_exists) {
    const int iterations = 100;
    dfma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(DoublePrecisionTest, test_dfma_latency_returns_valid_cycles) {
    const int iterations = 1000;
    dfma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(DoublePrecisionTest, test_dfma_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    dfma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // DFMA latency on Ada Lovelace is typically 5-8 cycles
    EXPECT_GE(cpi, 1.0) << "DFMA latency too low";
    EXPECT_LE(cpi, 20.0) << "DFMA latency too high";
}

// ============================================================================
// DFMA Throughput Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dfma_throughput_kernel_exists) {
    const int iterations = 100;
    dfma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(DoublePrecisionTest, test_dfma_throughput_returns_valid_cycles) {
    const int iterations = 1000;
    dfma_throughput_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(DoublePrecisionTest, test_dfma_similar_latency_to_dadd) {
    const int iterations = 10000;

    dfma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t dfma_cycles = 0;
    cudaMemcpy(&dfma_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t dadd_cycles = 0;
    cudaMemcpy(&dadd_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double dfma_cpi = cycles_per_iteration(dfma_cycles, iterations);
    double dadd_cpi = cycles_per_iteration(dadd_cycles, iterations);

    EXPECT_GT(dfma_cpi, 0.0);
    EXPECT_GT(dadd_cpi, 0.0);

    // DFMA and DADD should have similar latency
    double ratio = dfma_cpi / dadd_cpi;
    EXPECT_GE(ratio, 0.5) << "DFMA much faster than DADD";
    EXPECT_LE(ratio, 2.0) << "DFMA much slower than DADD";
}

TEST_F(DoublePrecisionTest, test_dfma_similar_latency_to_dmul) {
    const int iterations = 10000;

    dfma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t dfma_cycles = 0;
    cudaMemcpy(&dfma_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    dmul_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t dmul_cycles = 0;
    cudaMemcpy(&dmul_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double dfma_cpi = cycles_per_iteration(dfma_cycles, iterations);
    double dmul_cpi = cycles_per_iteration(dmul_cycles, iterations);

    EXPECT_GT(dfma_cpi, 0.0);
    EXPECT_GT(dmul_cpi, 0.0);

    // DFMA and DMUL should have similar latency (both use multiplier)
    double ratio = dfma_cpi / dmul_cpi;
    EXPECT_GE(ratio, 0.5) << "DFMA much faster than DMUL";
    EXPECT_LE(ratio, 2.0) << "DFMA much slower than DMUL";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DoublePrecisionTest, test_double_precision_null_cycles_pointer) {
    const int iterations = 100;
    dadd_latency_kernel<<<1, 1>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(DoublePrecisionTest, test_double_precision_negative_iterations) {
    const int iterations = -1;
    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(DoublePrecisionTest, test_double_precision_large_iterations) {
    const int iterations = 1000000;
    dadd_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(DoublePrecisionTest, test_double_precision_consistency) {
    const int iterations = 5000;
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        dfma_latency_kernel<<<1, 1>>>(d_cycles_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.2) << "Too much variance in DFMA latency";
    }
}

}  // namespace microbench
}  // namespace cpm
