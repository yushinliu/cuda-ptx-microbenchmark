#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace cpm {

/**
 * @brief Status codes for memory bandwidth benchmark operations
 */
enum class BandwidthStatus {
    kSuccess = 0,
    kSkipped = 1,
    kError = 2,
    kInvalidSize = 3
};

/**
 * @brief Result structure for memory bandwidth benchmark
 */
struct BandwidthResult {
    float read_bandwidth_gbps;
    float write_bandwidth_gbps;
    float copy_bandwidth_gbps;
    BandwidthStatus status;
    float elapsed_ms;
    size_t bytes_processed;
    int iterations;
};

/**
 * @brief Global Memory bandwidth benchmark class for RTX 4070
 *
 * Tests global memory (GDDR6X) performance.
 * RTX 4070 memory bandwidth: 504 GB/s
 */
class MemoryBandwidthBenchmark {
public:
    /**
     * @brief Construct memory bandwidth benchmark
     * @param data_size Size of data to test
     * @param alignment Memory alignment for data buffer (default 128 bytes)
     */
    MemoryBandwidthBenchmark(size_t data_size, size_t alignment = 128);

    /**
     * @brief Destructor
     */
    ~MemoryBandwidthBenchmark();

    // Disable copy
    MemoryBandwidthBenchmark(const MemoryBandwidthBenchmark&) = delete;
    MemoryBandwidthBenchmark& operator=(const MemoryBandwidthBenchmark&) = delete;

    // Enable move
    MemoryBandwidthBenchmark(MemoryBandwidthBenchmark&& other) noexcept;
    MemoryBandwidthBenchmark& operator=(MemoryBandwidthBenchmark&& other) noexcept;

    /**
     * @brief Run read bandwidth test
     * @return Benchmark result with read bandwidth
     */
    BandwidthResult run_read_benchmark();

    /**
     * @brief Run write bandwidth test
     * @return Benchmark result with write bandwidth
     */
    BandwidthResult run_write_benchmark();

    /**
     * @brief Run copy bandwidth test
     * @return Benchmark result with copy bandwidth
     */
    BandwidthResult run_copy_benchmark();

    /**
     * @brief Get the configured data size
     */
    size_t get_data_size() const { return data_size_; }

    /**
     * @brief Get the configured alignment
     */
    size_t get_alignment() const { return alignment_; }

private:
    size_t data_size_;
    size_t alignment_;

    // Device memory
    float* d_src_data_ = nullptr;
    float* d_dst_data_ = nullptr;
    float* d_output_ = nullptr;

    // Internal initialization
    bool initialize();
    void cleanup();

    // Calculate bandwidth from elapsed time
    float calculate_bandwidth_gbps(size_t bytes, float elapsed_ms) const;
};

// CUDA kernel declarations
__global__ void global_read_kernel(const float* input, float* output, size_t n);
__global__ void global_write_kernel(float* output, float value, size_t n);
__global__ void global_copy_kernel(const float* input, float* output, size_t n);

}  // namespace cpm
