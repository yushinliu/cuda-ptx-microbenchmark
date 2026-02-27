#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <algorithm>

#include "fixtures/gpu_test_fixture.h"

// Forward declarations for kernels
__global__ void dummy_kernel() {
    // Empty kernel for timing tests
}

__global__ void nanosleep_kernel(int nanoseconds) {
    __nanosleep(nanoseconds);
}

namespace cpm {

// Minimal GpuTimer implementation for testing
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    void start() {
        cudaEventRecord(start_event_);
    }

    void stop() {
        cudaEventRecord(stop_event_);
        cudaEventSynchronize(stop_event_);
    }

    float elapsed_ms() const {
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start_event_, stop_event_);
        return elapsed;
    }

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

// Test fixture
class GpuTimerTest : public GpuTestFixture {
protected:
    void SetUp() override {
        GpuTestFixture::SetUp();
        timer_ = std::make_unique<GpuTimer>();
    }

    void TearDown() override {
        timer_.reset();
        GpuTestFixture::TearDown();
    }

    std::unique_ptr<GpuTimer> timer_;
};

// RED: First write a failing test
TEST_F(GpuTimerTest, test_start_stop_returns_positive_elapsed_time) {
    // Given: Timer is created
    ASSERT_NE(timer_, nullptr);

    // When: Start and immediately stop
    timer_->start();
    timer_->stop();

    // Then: Elapsed time should be positive
    float elapsed = timer_->elapsed_ms();
    EXPECT_GT(elapsed, 0.0f);
    EXPECT_LT(elapsed, 1.0f);  // Should be less than 1ms for empty operation
}

TEST_F(GpuTimerTest, test_nanosleep_measures_accurate_duration) {
    // Given: Known sleep duration
    const int sleep_ns = 1000000;  // 1ms
    const float tolerance = 0.2f;   // 20% tolerance for sleep variance

    // When: Measure sleep duration
    timer_->start();
    nanosleep_kernel<<<1, 1>>>(sleep_ns);
    cudaDeviceSynchronize();
    timer_->stop();

    // Then: Measurement should be within expected range
    float elapsed_ms = timer_->elapsed_ms();
    float expected_ms = sleep_ns / 1e6f;
    EXPECT_NEAR(elapsed_ms, expected_ms, expected_ms * tolerance);
}

TEST_F(GpuTimerTest, test_multiple_kernel_launches_accumulate_time) {
    // Given: Multiple kernel launches
    const int iterations = 100;

    // When: Time multiple launches
    timer_->start();
    for (int i = 0; i < iterations; ++i) {
        dummy_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();
    timer_->stop();

    float total_time = timer_->elapsed_ms();

    // Then: Total time should be positive
    EXPECT_GT(total_time, 0.0f);

    // Time single launch for comparison
    timer_->start();
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    timer_->stop();
    float single_time = timer_->elapsed_ms();

    // Multiple launches should take more time than single
    EXPECT_GT(total_time, single_time);
}

TEST_F(GpuTimerTest, test_timer_reusable_after_stop) {
    // Given: Timer has been used once
    timer_->start();
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    timer_->stop();
    float first_elapsed = timer_->elapsed_ms();

    // When: Reuse timer for second measurement
    timer_->start();
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    timer_->stop();
    float second_elapsed = timer_->elapsed_ms();

    // Then: Both measurements should be valid
    EXPECT_GT(first_elapsed, 0.0f);
    EXPECT_GT(second_elapsed, 0.0f);
}

// Parameterized test for different iteration counts
class GpuTimerIterationsTest : public GpuTimerTest,
                               public ::testing::WithParamInterface<int> {};

TEST_P(GpuTimerIterationsTest, test_scaling_with_iterations) {
    int iterations = GetParam();

    timer_->start();
    for (int i = 0; i < iterations; ++i) {
        dummy_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();
    timer_->stop();

    float total_time = timer_->elapsed_ms();
    EXPECT_GT(total_time, 0.0f);

    // More iterations should generally take more time
    // (though kernel overhead dominates for small iterations)
}

INSTANTIATE_TEST_SUITE_P(
    IterationCounts,
    GpuTimerIterationsTest,
    ::testing::Values(1, 10, 100, 1000)
);

}  // namespace cpm
