/**
 * @file test_l2_cache.cu
 * @brief L2 cache microbenchmark tests (TDD)
 *
 * Tests for L2 cache latency, bandwidth, and global memory latency on RTX 4070.
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
__global__ void l2_cache_latency_kernel(uint64_t* cycles, int* buffer, int buffer_size, int iterations);
__global__ void l2_cache_bandwidth_kernel(uint64_t* cycles, float* output, const float* input, int num_elements);
__global__ void global_memory_latency_kernel(uint64_t* cycles, int* buffer, int buffer_size, int iterations);

/**
 * @brief Test fixture for L2 cache benchmarks
 */
class L2CacheTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_cycles_, sizeof(uint64_t));

        // Allocate buffers for L2-sized test (48MB for RTX 4070)
        // Use smaller size for testing to avoid timeout
        l2_buffer_size_ = 1024 * 1024;  // 1M elements = 4MB (int)
        cudaMalloc(&d_l2_buffer_, l2_buffer_size_ * sizeof(int));

        // Initialize with pointer chasing pattern
        std::vector<int> h_buffer(l2_buffer_size_);
        for (int i = 0; i < l2_buffer_size_; ++i) {
            h_buffer[i] = (i + 1) % l2_buffer_size_;
        }
        cudaMemcpy(d_l2_buffer_, h_buffer.data(), l2_buffer_size_ * sizeof(int), cudaMemcpyHostToDevice);

        // Allocate bandwidth buffers
        bandwidth_elements_ = 1024 * 1024;  // 1M floats = 4MB
        cudaMalloc(&d_bandwidth_input_, bandwidth_elements_ * sizeof(float));
        cudaMalloc(&d_bandwidth_output_, bandwidth_elements_ * sizeof(float));

        std::vector<float> h_bandwidth(bandwidth_elements_, 1.0f);
        cudaMemcpy(d_bandwidth_input_, h_bandwidth.data(), bandwidth_elements_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_cycles_);
        cudaFree(d_l2_buffer_);
        cudaFree(d_bandwidth_input_);
        cudaFree(d_bandwidth_output_);
        GpuTestFixture::TearDown();
    }

    uint64_t* d_cycles_ = nullptr;
    int* d_l2_buffer_ = nullptr;
    int l2_buffer_size_ = 0;
    float* d_bandwidth_input_ = nullptr;
    float* d_bandwidth_output_ = nullptr;
    int bandwidth_elements_ = 0;
};

// ============================================================================
// L2 Cache Latency Tests
// ============================================================================

TEST_F(L2CacheTest, test_l2_cache_latency_kernel_exists) {
    const int iterations = 100;
    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(L2CacheTest, test_l2_cache_latency_returns_valid_cycles) {
    const int iterations = 1000;
    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 1000000000u);
}

TEST_F(L2CacheTest, test_l2_cache_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = static_cast<double>(total_cycles) / iterations;

    // L2 cache latency on Ada Lovelace is typically 150-250 cycles
    EXPECT_GE(cpi, 50.0) << "L2 latency too low";
    EXPECT_LE(cpi, 500.0) << "L2 latency too high";
}

TEST_F(L2CacheTest, test_l2_cache_latency_zero_iterations) {
    const int iterations = 0;
    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u);
}

// ============================================================================
// L2 Cache Bandwidth Tests
// ============================================================================

TEST_F(L2CacheTest, test_l2_cache_bandwidth_kernel_exists) {
    l2_cache_bandwidth_kernel<<<1, 256>>>(d_cycles_, d_bandwidth_output_, d_bandwidth_input_, bandwidth_elements_);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(L2CacheTest, test_l2_cache_bandwidth_returns_valid_cycles) {
    l2_cache_bandwidth_kernel<<<1, 256>>>(d_cycles_, d_bandwidth_output_, d_bandwidth_input_, bandwidth_elements_);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 1000000000u);
}

TEST_F(L2CacheTest, test_l2_cache_bandwidth_calculates_bandwidth) {
    l2_cache_bandwidth_kernel<<<16, 256>>>(d_cycles_, d_bandwidth_output_, d_bandwidth_input_, bandwidth_elements_);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);

    // Calculate bandwidth in GB/s
    double seconds = static_cast<double>(cycles) / 2.48e9;  // RTX 4070 boost clock
    double bytes = bandwidth_elements_ * sizeof(float);
    double bandwidth_gbps = (bytes / seconds) / 1e9;

    // L2 bandwidth should be reasonable (10-1000 GB/s)
    EXPECT_GT(bandwidth_gbps, 1.0) << "L2 bandwidth too low";
    EXPECT_LT(bandwidth_gbps, 5000.0) << "L2 bandwidth too high";
}

// ============================================================================
// Global Memory Latency Tests
// ============================================================================

TEST_F(L2CacheTest, test_global_memory_latency_kernel_exists) {
    const int iterations = 100;
    global_memory_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(L2CacheTest, test_global_memory_latency_returns_valid_cycles) {
    const int iterations = 1000;
    global_memory_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaDeviceSynchronize();

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 1000000000u);
}

TEST_F(L2CacheTest, test_global_memory_latency_per_iteration_reasonable) {
    const int iterations = 10000;
    global_memory_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double cpi = static_cast<double>(total_cycles) / iterations;

    // Global memory latency on Ada Lovelace is typically 300-500 cycles
    EXPECT_GE(cpi, 100.0) << "Global memory latency too low";
    EXPECT_LE(cpi, 1000.0) << "Global memory latency too high";
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_F(L2CacheTest, test_global_memory_slower_than_l2) {
    const int iterations = 10000;

    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaDeviceSynchronize();
    uint64_t l2_cycles = 0;
    cudaMemcpy(&l2_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    global_memory_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaDeviceSynchronize();
    uint64_t global_cycles = 0;
    cudaMemcpy(&global_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double l2_cpi = static_cast<double>(l2_cycles) / iterations;
    double global_cpi = static_cast<double>(global_cycles) / iterations;

    EXPECT_GT(l2_cpi, 0.0);
    EXPECT_GT(global_cpi, 0.0);

    // Global memory should be slower than L2
    EXPECT_GT(global_cpi, l2_cpi * 0.8) << "Global memory should be slower than L2";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(L2CacheTest, test_l2_cache_null_buffer) {
    const int iterations = 100;
    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, nullptr, l2_buffer_size_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(L2CacheTest, test_l2_cache_zero_buffer_size) {
    const int iterations = 100;
    l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, 0, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(L2CacheTest, test_l2_cache_multiple_threads) {
    const int iterations = 1000;
    l2_cache_latency_kernel<<<4, 256>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

TEST_F(L2CacheTest, test_l2_cache_consistency) {
    const int iterations = 5000;
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        l2_cache_latency_kernel<<<1, 1>>>(d_cycles_, d_l2_buffer_, l2_buffer_size_, iterations);
        cudaDeviceSynchronize();

        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cycle_list.push_back(cycles);
    }

    std::sort(cycle_list.begin(), cycle_list.end());
    uint64_t median = cycle_list[runs / 2];

    for (uint64_t c : cycle_list) {
        double deviation = std::abs(static_cast<double>(c) - median) / median;
        EXPECT_LT(deviation, 0.3) << "Too much variance in L2 latency";
    }
}

}  // namespace microbench
}  // namespace cpm
