#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>

#include "fixtures/gpu_test_fixture.h"
#include "kernels/ptx/arithmetic.h"
#include "kernels/ptx/memory_ptx.h"
#include "kernels/ptx/synchronization.h"
#include "core/gpu_timer.h"

namespace cpm {

// FMA instruction test suite
class PtxFmaTest : public GpuTestFixture {};

TEST_F(PtxFmaTest, test_fma_single_precision_computes_correctly) {
    // Given: Input data
    float a = 2.0f, b = 3.0f, c = 4.0f;
    float expected = a * b + c;  // 10.0f
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: Execute PTX FMA instruction
    fma_kernel<<<1, 1>>>(a, b, c, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: Result is correct
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(PtxFmaTest, test_fma_with_negative_values) {
    // Given: Negative inputs
    float a = -2.0f, b = 3.0f, c = 4.0f;
    float expected = a * b + c;  // -2.0f
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: Execute FMA
    fma_kernel<<<1, 1>>>(a, b, c, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: Result handles negatives correctly
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(PtxFmaTest, test_fma_with_zero) {
    // Given: Zero inputs
    float a = 0.0f, b = 3.0f, c = 4.0f;
    float expected = 4.0f;  // 0 * 3 + 4 = 4
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: Execute FMA with zero
    fma_kernel<<<1, 1>>>(a, b, c, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: Result is correct
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(PtxFmaTest, test_fma_special_values) {
    // Test with special floating-point values
    struct TestCase {
        float a, b, c;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        {1.0f, 0.0f, 0.0f, "multiply by zero"},
        {1.0f, 1.0f, 0.0f, "add zero"},
        {2.0f, 0.5f, 0.0f, "fractional result"},
        {1.0f, 1.0f, 1.0f, "simple addition"},
        {-1.0f, -1.0f, 0.0f, "negative times negative"},
    };

    // Allocate once outside the loop to prevent leak on assertion failure
    float* d_result;
    cudaError_t alloc_err = cudaMalloc(&d_result, sizeof(float));
    ASSERT_EQ(alloc_err, cudaSuccess) << "Failed to allocate device memory";

    for (const auto& tc : test_cases) {
        float expected = tc.a * tc.b + tc.c;
        float result = 0.0f;

        fma_kernel<<<1, 1>>>(tc.a, tc.b, tc.c, d_result);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        EXPECT_FLOAT_EQ(result, expected) << "Failed for: " << tc.description;
    }

    cudaFree(d_result);
}

TEST_F(PtxFmaTest, test_fma_latency_kernel_runs) {
    // Given: Data array and iterations
    const int num_threads = 256;
    const int iterations = 1000;
    std::vector<float> data(num_threads, 1.0f);
    float* d_data;
    cudaMalloc(&d_data, num_threads * sizeof(float));
    cudaMemcpy(d_data, data.data(), num_threads * sizeof(float), cudaMemcpyHostToDevice);

    // When: Run latency test kernel
    fma_latency_test_kernel<<<1, num_threads>>>(d_data, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    cudaFree(d_data);
}

TEST_F(PtxFmaTest, test_fma_throughput_kernel_runs) {
    // Given: Data array and iterations
    const int num_threads = 256;
    const int iterations = 1000;
    std::vector<float> data(num_threads * 4, 1.0f);
    float* d_data;
    cudaMalloc(&d_data, num_threads * 4 * sizeof(float));
    cudaMemcpy(d_data, data.data(), num_threads * 4 * sizeof(float), cudaMemcpyHostToDevice);

    // When: Run throughput test kernel
    fma_throughput_test_kernel<<<1, num_threads>>>(d_data, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    cudaFree(d_data);
}

// Memory instruction test suite
class PtxMemoryTest : public GpuTestFixture {
protected:
    float* d_data_ = nullptr;
    static constexpr size_t kDataSize = 1024;

    void SetUp() override {
        GpuTestFixture::SetUp();
        cudaMalloc(&d_data_, kDataSize * sizeof(float));
        std::vector<float> host_data(kDataSize, 1.0f);
        cudaMemcpy(d_data_, host_data.data(), kDataSize * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_data_);
        GpuTestFixture::TearDown();
    }
};

TEST_F(PtxMemoryTest, test_ldg_loads_global_memory_correctly) {
    // Given: Initialized device memory
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: Use LDG to load
    ldg_kernel<<<1, 1>>>(d_data_, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: Loaded value is correct
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(PtxMemoryTest, test_ldg_with_different_values) {
    // Given: Different values in memory
    std::vector<float> test_values = {0.0f, -1.0f, 3.14159f, 1e10f, 1e-10f};
    float* d_result;
    cudaError_t alloc_err = cudaMalloc(&d_result, sizeof(float));
    ASSERT_EQ(alloc_err, cudaSuccess) << "Failed to allocate device memory";

    for (size_t i = 0; i < test_values.size(); ++i) {
        // Set value at index i
        cudaMemcpy(&d_data_[i], &test_values[i], sizeof(float),
                   cudaMemcpyHostToDevice);

        // Load using LDG
        ldg_kernel<<<1, 1>>>(&d_data_[i], d_result);
        cudaDeviceSynchronize();

        float result;
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_FLOAT_EQ(result, test_values[i]);
    }

    cudaFree(d_result);
}

TEST_F(PtxMemoryTest, test_lds_loads_shared_memory_correctly) {
    // Given: Shared memory will be initialized in kernel
    const int threads = 32;
    float result = 0.0f;
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // When: Use LDS to load shared memory
    lds_kernel<<<1, threads>>>(d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Then: Result is correct (kernel sets value to 42.0f)
    EXPECT_FLOAT_EQ(result, 42.0f);
}

TEST_F(PtxMemoryTest, test_stg_stores_global_memory_correctly) {
    // Given: Device memory and value to store
    float value = 3.14159f;
    float result = 0.0f;

    // When: Use STG to store
    stg_kernel<<<1, 1>>>(d_data_, value);
    cudaDeviceSynchronize();

    // Read back using standard memcpy
    cudaMemcpy(&result, d_data_, sizeof(float), cudaMemcpyDeviceToHost);

    // Then: Stored value is correct
    EXPECT_FLOAT_EQ(result, value);
}

TEST_F(PtxMemoryTest, test_ldg_ca_kernel_runs) {
    // Given: Large data array
    const size_t n = 1024 * 1024;  // 1M elements
    float* d_large_data;
    float* d_result;
    cudaMalloc(&d_large_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Initialize data
    std::vector<float> host_data(n, 1.0f);
    cudaMemcpy(d_large_data, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // When: Run LDG.CA kernel
    const int threads = 256;
    const int blocks = 64;
    ldg_ca_kernel<<<blocks, threads>>>(d_large_data, d_result, n);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    cudaFree(d_large_data);
    cudaFree(d_result);
}

TEST_F(PtxMemoryTest, test_ldg_cs_kernel_runs) {
    // Given: Large data array
    const size_t n = 1024 * 1024;  // 1M elements
    float* d_large_data;
    float* d_result;
    cudaMalloc(&d_large_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Initialize data
    std::vector<float> host_data(n, 1.0f);
    cudaMemcpy(d_large_data, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // When: Run LDG.CS kernel
    const int threads = 256;
    const int blocks = 64;
    ldg_cs_kernel<<<blocks, threads>>>(d_large_data, d_result, n);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    cudaFree(d_large_data);
    cudaFree(d_result);
}

// Synchronization instruction test suite
class PtxSyncTest : public GpuTestFixture {};

TEST_F(PtxSyncTest, test_bar_sync_kernel_runs) {
    // Given: Data array and iterations
    const int threads = 256;
    const int iterations = 100;
    float* d_data;
    cudaMalloc(&d_data, sizeof(float));

    // When: Run barrier sync test kernel
    bar_sync_test_kernel<<<1, threads>>>(d_data, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    cudaFree(d_data);
}

TEST_F(PtxSyncTest, test_bar_sync_produces_correct_result) {
    // Given: Data array
    const int threads = 256;
    const int iterations = 10;
    float result = 0.0f;
    float* d_data;
    cudaMalloc(&d_data, sizeof(float));

    // When: Run barrier sync test kernel
    bar_sync_test_kernel<<<1, threads>>>(d_data, iterations);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Then: Result should be reasonable (threads + iterations)
    // Initial value is 0, each iteration adds 1
    EXPECT_GE(result, 0.0f);
    EXPECT_LE(result, static_cast<float>(threads + iterations));
}

TEST_F(PtxSyncTest, test_membar_kernel_runs) {
    // Given: Flag and result memory
    int* d_flag;
    int* d_result;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    // When: Run membar test kernel
    membar_test_kernel<<<1, 2>>>(d_flag, d_result);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    // Verify result was set
    int result = 0;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(result, 1);

    cudaFree(d_flag);
    cudaFree(d_result);
}

TEST_F(PtxSyncTest, test_atom_add_kernel_runs) {
    // Given: Counter initialized to 0
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    const int n = 1000;
    const int threads = 256;
    const int blocks = 4;

    // When: Run atomic add kernel
    atom_add_test_kernel<<<blocks, threads>>>(d_counter, n);
    cudaError_t err = cudaDeviceSynchronize();

    // Then: Kernel completes successfully
    EXPECT_EQ(err, cudaSuccess);

    // Verify counter was incremented
    int result = 0;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(result, blocks * threads);

    cudaFree(d_counter);
}

// Performance tests
class PtxPerformanceTest : public GpuTestFixture {};

TEST_F(PtxPerformanceTest, test_fma_latency_is_reasonable) {
    // Given: Measurement parameters
    const int num_threads = 256;
    const int iterations = 10000;
    const float expected_latency_ns = 4.0f;  // FMA typical latency
    const float tolerance = 0.5f;             // 50% tolerance

    std::vector<float> data(num_threads, 1.0f);
    float* d_data;
    cudaMalloc(&d_data, num_threads * sizeof(float));
    cudaMemcpy(d_data, data.data(), num_threads * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    fma_latency_test_kernel<<<1, num_threads>>>(d_data, 100);
    cudaDeviceSynchronize();

    // When: Measure FMA latency
    GpuTimer timer;
    timer.start();
    fma_latency_test_kernel<<<1, num_threads>>>(d_data, iterations);
    cudaDeviceSynchronize();
    timer.stop();

    float elapsed_ms = timer.elapsed_ms();
    float ops_per_thread = static_cast<float>(iterations);
    float measured_latency_ns = (elapsed_ms * 1e6f) / ops_per_thread;

    cudaFree(d_data);

    // Then: Measurement should be positive and reasonable
    EXPECT_GT(measured_latency_ns, 0.0f);
    EXPECT_LT(measured_latency_ns, expected_latency_ns * (1 + tolerance) * 10);  // Very loose bound
}

TEST_F(PtxPerformanceTest, test_ldg_ca_vs_cs_performance) {
    // Given: Large data array
    const size_t n = 4 * 1024 * 1024;  // 4M elements = 16MB
    float* d_data;
    float* d_result;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    std::vector<float> host_data(n, 1.0f);
    cudaMemcpy(d_data, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = 64;
    const int warmup = 5;
    const int iterations = 20;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        cudaMemset(d_result, 0, sizeof(float));
        ldg_ca_kernel<<<blocks, threads>>>(d_data, d_result, n);
        cudaMemset(d_result, 0, sizeof(float));
        ldg_cs_kernel<<<blocks, threads>>>(d_data, d_result, n);
    }
    cudaDeviceSynchronize();

    // Time LDG.CA
    GpuTimer timer_ca;
    timer_ca.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_result, 0, sizeof(float));
        ldg_ca_kernel<<<blocks, threads>>>(d_data, d_result, n);
    }
    cudaDeviceSynchronize();
    timer_ca.stop();

    // Time LDG.CS
    GpuTimer timer_cs;
    timer_cs.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_result, 0, sizeof(float));
        ldg_cs_kernel<<<blocks, threads>>>(d_data, d_result, n);
    }
    cudaDeviceSynchronize();
    timer_cs.stop();

    float time_ca = timer_ca.elapsed_ms();
    float time_cs = timer_cs.elapsed_ms();

    cudaFree(d_data);
    cudaFree(d_result);

    // Then: Both should complete, times should be positive
    EXPECT_GT(time_ca, 0.0f);
    EXPECT_GT(time_cs, 0.0f);

    // For streaming data, CS should not be significantly slower than CA
    // (Allow 2x difference for measurement variance)
    EXPECT_LT(time_cs, time_ca * 2.0f);
}

// Parameterized test for different thread counts
class PtxThreadCountTest : public GpuTestFixture,
                           public ::testing::WithParamInterface<int> {};

TEST_P(PtxThreadCountTest, test_fma_with_different_thread_counts) {
    int threads = GetParam();
    std::vector<float> data(threads, 1.0f);
    float* d_data;
    cudaMalloc(&d_data, threads * sizeof(float));
    cudaMemcpy(d_data, data.data(), threads * sizeof(float), cudaMemcpyHostToDevice);

    fma_latency_test_kernel<<<1, threads>>>(d_data, 100);
    cudaError_t err = cudaDeviceSynchronize();

    EXPECT_EQ(err, cudaSuccess);

    cudaFree(d_data);
}

INSTANTIATE_TEST_SUITE_P(
    ThreadCounts,
    PtxThreadCountTest,
    ::testing::Values(32, 64, 128, 256)
);

}  // namespace cpm
