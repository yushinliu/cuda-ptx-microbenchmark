#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace cpm {

/**
 * @brief Status codes for L1 cache benchmark operations
 */
enum class L1CacheStatus {
    kSuccess = 0,
    kSkipped = 1,
    kError = 2,
    kInvalidSize = 3
};

/**
 * @brief Result structure for L1 cache benchmark
 */
struct L1CacheResult {
    float bandwidth_gbps;
    float hit_rate;
    L1CacheStatus status;
    float elapsed_ms;
    size_t bytes_processed;
    int iterations;
};

/**
 * @brief L1 Cache benchmark class for RTX 4070
 *
 * Tests L1 cache performance with various access patterns.
 * L1 cache size on RTX 4070: 128KB per SM
 */
class L1CacheBenchmark {
public:
    /**
     * @brief Construct L1 cache benchmark
     * @param data_size Size of data to test (should be <= 128KB for L1 hits)
     * @param alignment Memory alignment for data buffer (default 128 bytes)
     */
    L1CacheBenchmark(size_t data_size, size_t alignment = 128);

    /**
     * @brief Destructor
     */
    ~L1CacheBenchmark();

    // Disable copy
    L1CacheBenchmark(const L1CacheBenchmark&) = delete;
    L1CacheBenchmark& operator=(const L1CacheBenchmark&) = delete;

    // Enable move
    L1CacheBenchmark(L1CacheBenchmark&& other) noexcept;
    L1CacheBenchmark& operator=(L1CacheBenchmark&& other) noexcept;

    /**
     * @brief Run sequential access test (high hit rate expected)
     * @return Benchmark result with bandwidth and hit rate
     */
    L1CacheResult run_sequential_access();

    /**
     * @brief Run random access test (lower hit rate expected)
     * @return Benchmark result with bandwidth and hit rate
     */
    L1CacheResult run_random_access();

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
};

// CUDA kernel declarations
__global__ void l1_sequential_read_kernel(const float* input, float* output, size_t n);
__global__ void l1_random_read_kernel(const float* input, float* output,
                                      const int* indices, size_t n);

}  // namespace cpm
