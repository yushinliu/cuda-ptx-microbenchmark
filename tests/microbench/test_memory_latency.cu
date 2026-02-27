/**
 * @file test_memory_latency.cu
 * @brief Memory latency microbenchmark tests (TDD)
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
__global__ void memory_latency_kernel(float** pointers, int iterations,
                                      uint64_t* cycles);

/**
 * @brief Test fixture for memory latency benchmarks
 */
class MemoryLatencyTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();

        // Allocate device memory
        cudaMalloc(&d_cycles_, sizeof(uint64_t));
        cudaMalloc(&d_pointers_, kMaxChainLength * sizeof(float*));
        cudaMalloc(&d_buffer_, kMaxChainLength * sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_cycles_);
        cudaFree(d_pointers_);
        cudaFree(d_buffer_);
        GpuTestFixture::TearDown();
    }

    // Create a dependency chain: ptr[0] points to buffer[0], etc.
    void create_dependency_chain(float** pointers, float* buffer, size_t n) {
        std::vector<float*> host_pointers(n);
        for (size_t i = 0; i < n; ++i) {
            host_pointers[i] = &buffer[i];
        }
        cudaMemcpy(pointers, host_pointers.data(), n * sizeof(float*),
                   cudaMemcpyHostToDevice);
    }

    // Create a circular dependency chain for chasing
    void create_circular_chain(float** pointers, float* buffer, size_t n) {
        std::vector<float*> host_pointers(n);
        for (size_t i = 0; i < n; ++i) {
            host_pointers[i] = &buffer[(i + 1) % n];
        }
        cudaMemcpy(pointers, host_pointers.data(), n * sizeof(float*),
                   cudaMemcpyHostToDevice);
    }

    static constexpr size_t kMaxChainLength = 1024 * 1024;
    uint64_t* d_cycles_ = nullptr;
    float** d_pointers_ = nullptr;
    float* d_buffer_ = nullptr;
};

// ============================================================================
// Basic Latency Tests
// ============================================================================

TEST_F(MemoryLatencyTest, test_memory_latency_kernel_exists) {
    // Given: Valid pointers and iterations
    const int iterations = 100;
    create_dependency_chain(d_pointers_, d_buffer_, 100);

    // When: Launch kernel
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);

    // Then: Kernel should complete
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(MemoryLatencyTest, test_memory_latency_returns_valid_cycles) {
    // Given: Dependency chain
    const int iterations = 1000;
    create_circular_chain(d_pointers_, d_buffer_, 100);

    // When: Run kernel
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Cycles should be positive
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u) << "Cycles should be positive";
    EXPECT_LT(cycles, 10000000000u) << "Cycles should be reasonable";
}

TEST_F(MemoryLatencyTest, test_memory_latency_increases_with_iterations) {
    // Given: Fixed chain length
    const size_t chain_length = 100;
    create_circular_chain(d_pointers_, d_buffer_, chain_length);

    // When: Run with different iteration counts
    std::vector<int> iteration_counts = {100, 500, 1000};
    std::vector<uint64_t> cycle_counts;

    for (int iters : iteration_counts) {
        memory_latency_kernel<<<1, 1>>>(d_pointers_, iters, d_cycles_);
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

TEST_F(MemoryLatencyTest, test_memory_latency_empty_chain) {
    // Given: Zero iterations
    const int iterations = 0;

    // When: Run kernel with no iterations
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should handle gracefully
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cycles, 0u) << "Zero iterations should result in zero cycles";
}

TEST_F(MemoryLatencyTest, test_memory_latency_single_iteration) {
    // Given: Single iteration
    const int iterations = 1;
    create_circular_chain(d_pointers_, d_buffer_, 10);

    // When: Run kernel
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
    cudaDeviceSynchronize();

    // Then: Should complete with valid cycles
    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_GT(cycles, 0u);
    EXPECT_LT(cycles, 1000000u) << "Single iteration should be fast";
}

// ============================================================================
// Chain Length Tests
// ============================================================================

TEST_F(MemoryLatencyTest, test_memory_latency_varying_chain_lengths) {
    // Given: Different chain lengths
    std::vector<size_t> chain_lengths = {10, 100, 1000, 10000};
    const int iterations = 100;

    for (size_t length : chain_lengths) {
        // When: Create chain and run
        create_circular_chain(d_pointers_, d_buffer_, length);
        memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
        cudaDeviceSynchronize();

        // Then: Should complete successfully
        uint64_t cycles = 0;
        cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        EXPECT_GT(cycles, 0u) << "Failed for chain length: " << length;
    }
}

TEST_F(MemoryLatencyTest, test_memory_latency_large_chain) {
    // Given: Large chain that exceeds cache
    const size_t chain_length = 512 * 1024;  // 2MB of float pointers
    const int iterations = 10;

    // Reallocate for large chain if needed
    cudaFree(d_pointers_);
    cudaFree(d_buffer_);
    cudaMalloc(&d_pointers_, chain_length * sizeof(float*));
    cudaMalloc(&d_buffer_, chain_length * sizeof(float));

    create_circular_chain(d_pointers_, d_buffer_, chain_length);

    // When: Run kernel
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should complete
    EXPECT_EQ(err, cudaSuccess);

    uint64_t cycles = 0;
    cudaMemcpy(&cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    EXPECT_GT(cycles, 0u);
}

// ============================================================================
// Latency Measurement Accuracy Tests
// ============================================================================

TEST_F(MemoryLatencyTest, test_memory_latency_consistency) {
    // Given: Fixed parameters
    const int iterations = 1000;
    const size_t chain_length = 1000;
    create_circular_chain(d_pointers_, d_buffer_, chain_length);

    // When: Run multiple times
    std::vector<uint64_t> cycle_list;
    const int runs = 10;

    for (int i = 0; i < runs; ++i) {
        memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
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
        EXPECT_LT(deviation, 0.3) << "Too much variance in latency measurement";
    }
}

TEST_F(MemoryLatencyTest, test_memory_latency_per_iteration) {
    // Given: Chain that forces memory access
    const size_t chain_length = 10000;  // Large enough to not fit in registers
    const int iterations = 10000;
    create_circular_chain(d_pointers_, d_buffer_, chain_length);

    // When: Run kernel
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
    cudaDeviceSynchronize();

    uint64_t total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles_, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Then: Calculate cycles per iteration
    double cycles_per_iter = static_cast<double>(total_cycles) / iterations;

    // Memory latency should be in reasonable range (tens to hundreds of cycles)
    EXPECT_GT(cycles_per_iter, 1.0) << "Cycles per iteration too low";
    EXPECT_LT(cycles_per_iter, 1000.0) << "Cycles per iteration too high";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(MemoryLatencyTest, test_memory_latency_null_pointers) {
    // Given: Null pointer array (edge case)
    const int iterations = 100;

    // When: Run with null pointers
    memory_latency_kernel<<<1, 1>>>(nullptr, iterations, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should fail gracefully or handle it
    // Note: This may crash or return error, both are acceptable
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorIllegalAddress);
}

TEST_F(MemoryLatencyTest, test_memory_latency_negative_iterations) {
    // Given: Negative iterations (treated as large positive due to unsigned)
    const int iterations = -1;  // Will be treated as very large
    create_circular_chain(d_pointers_, d_buffer_, 100);

    // When: Run kernel
    memory_latency_kernel<<<1, 1>>>(d_pointers_, iterations, d_cycles_);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Should either timeout or handle it
    // This is a stress test - we mainly care it doesn't crash the driver
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorLaunchTimeout);
}

}  // namespace microbench
}  // namespace cpm
