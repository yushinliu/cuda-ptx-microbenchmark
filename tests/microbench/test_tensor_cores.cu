/**
 * @file test_tensor_cores.cu
 * @brief Tensor Core microbenchmark tests (TDD)
 *
 * Tests for HMMA (half-precision MMA) and IMMA (integer MMA) on RTX 4070.
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
__global__ void hmma_latency_kernel(uint64_t* cycles, int iterations);
__global__ void hmma_throughput_kernel(uint64_t* cycles, int iterations);
__global__ void imma_latency_kernel(uint64_t* cycles, int iterations);
__global__ void imma_throughput_kernel(uint64_t* cycles, int iterations);

/**
 * @brief Test fixture for Tensor Core benchmarks
 */
class TensorCoreTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_cycles_, sizeof(uint64_t));

        // Check for Tensor Core support (sm_70+ for HMMA, sm_72+ for IMMA)
        int major = device_props_.major;
        int minor = device_props_.minor;
        tensor_cores_supported_ = (major >= 7);
        int8_mma_supported_ = (major > 7) || (major == 7 && minor >= 2);
    }

    void TearDown() override {
        cudaFree(d_cycles_);
        GpuTestFixture::TearDown();
    }

    uint64_t* d_cycles_ = nullptr;
    bool tensor_cores_supported_ = false;
    bool int8_mma_supported_ = false;

    double cycles_per_iteration(uint64_t total_cycles, int iterations) {
        return static_cast<double>(total_cycles) / iterations;
    }
};

// ============================================================================
// HMMA Latency Tests
// ============================================================================

TEST_F(TensorCoreTest, test_hmma_latency_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 100;
    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(TensorCoreTest, test_hmma_latency_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 1000;
    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(TensorCoreTest, test_hmma_latency_per_iteration_reasonable) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 10000;
    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // HMMA latency on Ada Lovelace is typically 4-8 cycles
    EXPECT_GE(cpi, 1.0) << "HMMA latency too low";
    EXPECT_LE(cpi, 20.0) << "HMMA latency too high";
}

TEST_F(TensorCoreTest, test_hmma_latency_zero_iterations) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 0;
    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u);
}

// ============================================================================
// HMMA Throughput Tests
// ============================================================================

TEST_F(TensorCoreTest, test_hmma_throughput_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 100;
    hmma_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(TensorCoreTest, test_hmma_throughput_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 1000;
    hmma_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(TensorCoreTest, test_hmma_throughput_faster_than_latency) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 10000;

    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t latency_cycles = 0;
    cudaMemcpy(&latency_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    hmma_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t throughput_cycles = 0;
    cudaMemcpy(&throughput_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(latency_cycles, 0u);
    EXPECT_GT(throughput_cycles, 0u);
    EXPECT_LT(throughput_cycles, latency_cycles * 0.8)
        << "Throughput should be faster than latency";
}

// ============================================================================
// IMMA Latency Tests
// ============================================================================

TEST_F(TensorCoreTest, test_imma_latency_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 2);

    const int iterations = 100;
    imma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(TensorCoreTest, test_imma_latency_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 2);

    const int iterations = 1000;
    imma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

TEST_F(TensorCoreTest, test_imma_latency_per_iteration_reasonable) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 2);

    const int iterations = 10000;
    imma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = cycles_per_iteration(total_cycles, iterations);

    // IMMA latency on Ada Lovelace is typically 4-8 cycles
    EXPECT_GE(cpi, 1.0) << "IMMA latency too low";
    EXPECT_LE(cpi, 20.0) << "IMMA latency too high";
}

// ============================================================================
// IMMA Throughput Tests
// ============================================================================

TEST_F(TensorCoreTest, test_imma_throughput_kernel_exists) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 2);

    const int iterations = 100;
    imma_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(TensorCoreTest, test_imma_throughput_returns_valid_cycles) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 2);

    const int iterations = 1000;
    imma_throughput_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 100000000u);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(TensorCoreTest, test_hmma_imma_similar_latency) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 2);

    const int iterations = 10000;

    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t hmma_cycles = 0;
    cudaMemcpy(&hmma_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    imma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaDeviceSynchronize();
    uint64_t imma_cycles = 0;
    cudaMemcpy(&imma_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double hmma_cpi = cycles_per_iteration(hmma_cycles, iterations);
    double imma_cpi = cycles_per_iteration(imma_cycles, iterations);

    EXPECT_GT(hmma_cpi, 0.0);
    EXPECT_GT(imma_cpi, 0.0);

    // HMMA and IMMA should have similar latency
    double ratio = hmma_cpi / imma_cpi;
    EXPECT_GE(ratio, 0.5) << "HMMA much faster than IMMA";
    EXPECT_LE(ratio, 2.0) << "HMMA much slower than IMMA";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TensorCoreTest, test_tensor_core_null_cycles_pointer) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 100;
    hmma_latency_kernel<<<1, 32>>>(nullptr, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(TensorCoreTest, test_tensor_core_negative_iterations) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = -1;
    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

TEST_F(TensorCoreTest, test_tensor_core_large_iterations) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 1000000;
    hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(TensorCoreTest, test_tensor_core_consistency) {
    SKIP_IF_COMPUTE_LESS_THAN(7, 0);

    const int iterations = 5000;
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        hmma_latency_kernel<<<1, 32>>>(d_cycles_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.3) << "Too much variance in HMMA latency";
    }
}

}  // namespace microbench
}  // namespace cpm
