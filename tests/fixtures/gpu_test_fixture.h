#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>

namespace cpm {

/**
 * @brief Base test fixture for all GPU-related tests
 *
 * Provides CUDA device setup/teardown and error checking.
 * All GPU tests should inherit from this class.
 */
class GpuTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA device availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        ASSERT_EQ(err, cudaSuccess)
            << "Failed to get CUDA device count: " << cudaGetErrorString(err);
        ASSERT_GT(device_count, 0) << "No CUDA devices found";

        // Get current device properties
        cudaGetDevice(&device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);

        // Reset device state for clean test environment
        cudaDeviceReset();
    }

    void TearDown() override {
        // Check for residual CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error at teardown: "
                      << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all operations complete
        cudaDeviceSynchronize();
    }

    int device_id_ = 0;
    cudaDeviceProp device_props_;
};

/**
 * @brief Skip test if compute capability is less than required
 */
#define SKIP_IF_COMPUTE_LESS_THAN(major, minor) \
    do { \
        if (device_props_.major < major || \
            (device_props_.major == major && device_props_.minor < minor)) { \
            GTEST_SKIP() << "Requires compute capability " << major << "." << minor; \
        } \
    } while(0)

/**
 * @brief Check CUDA error and return assertion result
 */
#define ASSERT_CUDA_SUCCESS(err) \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)

#define EXPECT_CUDA_SUCCESS(err) \
    EXPECT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)

}  // namespace cpm
