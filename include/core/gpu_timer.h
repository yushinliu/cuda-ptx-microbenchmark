#pragma once

#include <cuda_runtime.h>

namespace cpm {

/**
 * @brief High-precision GPU timer using CUDA events
 *
 * Provides accurate timing of GPU operations with microsecond precision.
 */
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();

    // Disable copy
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Enable move
    GpuTimer(GpuTimer&& other) noexcept;
    GpuTimer& operator=(GpuTimer&& other) noexcept;

    /**
     * @brief Start the timer
     */
    void start();

    /**
     * @brief Stop the timer
     */
    void stop();

    /**
     * @brief Get elapsed time in milliseconds
     */
    float elapsed_ms() const;

    /**
     * @brief Get elapsed time in microseconds
     */
    float elapsed_us() const;

    /**
     * @brief Reset the timer
     */
    void reset();

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool running_;
};

}  // namespace cpm
