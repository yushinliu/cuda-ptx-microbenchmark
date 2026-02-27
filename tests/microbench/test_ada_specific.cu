/**
 * @file test_ada_specific.cu
 * @brief Ada Lovelace specific microbenchmark tests (TDD)
 *
 * Tests for CP.ASYNC and LDMATRIX instructions on RTX 4070 (sm_89).
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
__global__ void cp_async_latency_kernel(uint64_t* cycles, int iterations, const int* gmem_buffer);
__global__ void cp_async_throughput_kernel(uint64_t* cycles, int iterations, const int* gmem_buffer);
__global__ void ldmatrix_latency_kernel(uint64_t* cycles, int iterations);
__global__ void ldmatrix_throughput_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for Ada Lovelace specific benchmarks
 */
class AdaSpecificTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_cycles_, sizeof(uint64_t));
        cudaMalloc(&d_gmem_buffer_, 256 * sizeof(int));

        // Check for sm_89 (Ada Lovelace) or higher
        int major = device_props_.major;
        int minor = device_props_.minor;
        ada_supported_ = (major > 8) || (major == 8 && minor >= 9);
    }

    void TearDown() override {
        cudaFree(d_cycles_);
        cudaFree(d_gmem_buffer_);
        GpuTestFixture::TearDown();
    }

    uint64_t* d_cycles_ = nullptr;
    int* d_gmem_buffer_ = nullptr;
    bool ada_supported_ = false;

    double cycles_per_iteration(uint64_t total_cycles, int iterations) {
        return static_cast<double>(total_cycles) / iterations;
    }
};

// ============================================================================
// CP.ASYNC Latency Tests
// ============================================================================

TEST_F(AdaSpecificTest, test_cp_async_latency_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 100;
    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(AdaSpecificTest, test_cp_async_latency_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 1000;
    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(AdaSpecificTest, test_cp_async_latency_per_iteration_reasonable) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 10000;
    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // CP.ASYNC latency on Ada Lovelace is typically 10-50 cycles
    EXPECT_GE(cpi, 1.0) << "CP.ASYNC latency too low";
    EXPECT_LE(cpi, 100.0) << "CP.ASYNC latency too high";
}

TEST_F(AdaSpecificTest, test_cp_async_latency_zero_iterations) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 0;
    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u);
}

// ============================================================================
// CP.ASYNC Throughput Tests
// ============================================================================

TEST_F(AdaSpecificTest, test_cp_async_throughput_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 100;
    cp_async_throughput_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(AdaSpecificTest, test_cp_async_throughput_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 1000;
    cp_async_throughput_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(AdaSpecificTest, test_cp_async_throughput_faster_than_latency) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 10000;

    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaDeviceSynchronize();
    uint64_t latency_cycles = 0;
    cudaMemcpy(&latency_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cp_async_throughput_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaDeviceSynchronize();
    uint64_t throughput_cycles = 0;
    cudaMemcpy(&throughput_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(latency_cycles, 0u);
    EXPECT_GT(throughput_cycles, 0u);
    EXPECT_LT(throughput_cycles, latency_cycles * 0.8)
        << "Throughput should be faster than latency";
}

// ============================================================================
// LDMATRIX Latency Tests
// ============================================================================

TEST_F(AdaSpecificTest, test_ldmatrix_latency_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);  // Tensor Cores available since Volta

    const int iterations = 100;
    ldmatrix_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(AdaSpecificTest, test_ldmatrix_latency_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 1000;
    ldmatrix_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(AdaSpecificTest, test_ldmatrix_latency_per_iteration_reasonable) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 10000;
    ldmatrix_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // LDMATRIX latency on Ada Lovelace is typically 10-30 cycles
    EXPECT_GE(cpi, 1.0) << "LDMATRIX latency too low";
    EXPECT_LE(cpi, 100.0) << "LDMATRIX latency too high";
}

// ============================================================================
// LDMATRIX Throughput Tests
// ============================================================================

TEST_F(AdaSpecificTest, test_ldmatrix_throughput_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 100;
    ldmatrix_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(AdaSpecificTest, test_ldmatrix_throughput_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 1000;
    ldmatrix_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(AdaSpecificTest, test_cp_async_faster_than_ldmatrix) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 10000;

    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaDeviceSynchronize();
    uint64_t cp_async_cycles = 0;
    cudaMemcpy(&cp_async_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    ldmatrix_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t ldmatrix_cycles = 0;
    cudaMemcpy(&ldmatrix_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cp_async_cpi = cycles_per_iteration(cp_async_cycles, iterations);
    double ldmatrix_cpi = cycles_per_iteration(ldmatrix_cycles, iterations);

    EXPECT_GT(cp_async_cpi, 0.0);
    EXPECT_GT(ldmatrix_cpi, 0.0);

    // Both should be in similar range (10-100 cycles)
    double ratio = cp_async_cpi / ldmatrix_cpi;
    EXPECT_GE(ratio, 0.2) << "CP.ASYNC much faster than LDMATRIX";
    EXPECT_LE(ratio, 5.0) << "CP.ASYNC much slower than LDMATRIX";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AdaSpecificTest, test_ada_specific_null_cycles_pointer) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 100;
    cp_async_latency_kernel<<<1, 32>>>(nullptr, iterations, d_gmem_buffer_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(AdaSpecificTest, test_ada_specific_negative_iterations) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = -1;
    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(AdaSpecificTest, test_ada_specific_large_iterations) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 1000000;
    cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(AdaSpecificTest, test_ada_specific_consistency) {
    SKIP_IF_COMPUTE_LESS_THAN(8, 0);

    const int iterations = 5000;
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        cp_async_latency_kernel<<<1, 32>>>(d_cycles_, iterations, d_gmem_buffer_);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.3) << "Too much variance in CP.ASYNC latency";
    }
}

}  // namespace microbench
}  // namespace cpm
