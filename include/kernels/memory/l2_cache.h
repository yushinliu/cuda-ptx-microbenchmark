#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace cpm {

/**
 * @brief Status codes for L2 cache benchmark operations
 */
enum class L2CacheStatus {
    kSuccess = 0,
    kSkipped = 1,
    kError = 2,
    kInvalidSize = 3,
    kInvalidStride = 4
};

/**
 * @brief Result structure for L2 cache benchmark
 */
struct L2CacheResult {
    float bandwidth_gbps;
    float hit_rate;
    L2CacheStatus status;
    float elapsed_ms;
    size_t bytes_processed;
    int iterations;
    size_t stride;
};

/**
 * @brief L2 Cache benchmark class for RTX 4070
 *
 * Tests L2 cache performance with various access patterns.
 * L2 cache size on RTX 4070: 36MB
 */
class L2CacheBenchmark {
public:
    /**
     * @brief Construct L2 cache benchmark
     * @param data_size Size of data to test
     * @param alignment Memory alignment for data buffer (default 128 bytes)
     */
    L2CacheBenchmark(size_t data_size, size_t alignment = 128);

    /**
     * @brief Destructor
     */
    ~L2CacheBenchmark();

    // Disable copy
    L2CacheBenchmark(const L2CacheBenchmark&) = delete;
    L2CacheBenchmark& operator=(const L2CacheBenchmark&) = delete;

    // Enable move
    L2CacheBenchmark(L2CacheBenchmark&& other) noexcept;
    L2CacheBenchmark& operator=(L2CacheBenchmark&& other) noexcept;

    /**
     * @brief Run sequential access test
     * @return Benchmark result with bandwidth and hit rate
     */
    L2CacheResult run_sequential_access();

    /**
     * @brief Run random access test
     * @return Benchmark result with bandwidth and hit rate
     */
    L2CacheResult run_random_access();

    /**
     * @brief Run strided access test
     * @param stride_elements Number of elements to stride
     * @return Benchmark result with bandwidth and hit rate
     */
    L2CacheResult run_stride_access(size_t stride_elements);

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
    float* d_data_ = nullptr;
    float* d_output_ = nullptr;
    int* d_indices_ = nullptr;

    // Host memory for indices
    std::vector<int> h_indices_;

    // Internal initialization
    bool initialize();
    void cleanup();

    // Generate random access pattern
    void generate_random_indices();

    // Calculate bandwidth from elapsed time
    float calculate_bandwidth_gbps(size_t bytes, float elapsed_ms) const;

    // Estimate hit rate based on access pattern and data size
    float estimate_hit_rate_sequential() const;
    float estimate_hit_rate_random() const;
    float estimate_hit_rate_stride(size_t stride) const;
};

// CUDA kernel declarations
__global__ void l2_sequential_read_kernel(const float* input, float* output, size_t n);
__global__ void l2_stride_read_kernel(const float* input, float* output,
                                      size_t n, size_t stride_elements);

}  // namespace cpm
