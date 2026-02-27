#include "kernels/memory/l1_cache.h"
#include "core/gpu_timer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <random>
#include <cmath>

namespace cpm {

// CUDA kernel implementations
__global__ void l1_sequential_read_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Use warp shuffle for partial reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, sum);
    }
}

__global__ void l1_random_read_kernel(const float* input, float* output,
                                      const int* indices, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        int index = indices[i];
        sum += input[index];
    }

    // Use warp shuffle for partial reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, sum);
    }
}

// L1CacheBenchmark implementation
L1CacheBenchmark::L1CacheBenchmark(size_t data_size, size_t alignment)
    : data_size_(data_size), alignment_(alignment),
      d_data_(nullptr), d_output_(nullptr), d_indices_(nullptr) {
}

L1CacheBenchmark::~L1CacheBenchmark() {
    cleanup();
}

L1CacheBenchmark::L1CacheBenchmark(L1CacheBenchmark&& other) noexcept
    : data_size_(other.data_size_),
      alignment_(other.alignment_),
      d_data_(other.d_data_),
      d_output_(other.d_output_),
      d_indices_(other.d_indices_),
      h_indices_(std::move(other.h_indices_)) {
    other.d_data_ = nullptr;
    other.d_output_ = nullptr;
    other.d_indices_ = nullptr;
}

L1CacheBenchmark& L1CacheBenchmark::operator=(L1CacheBenchmark&& other) noexcept {
    if (this != &other) {
        cleanup();
        data_size_ = other.data_size_;
        alignment_ = other.alignment_;
        d_data_ = other.d_data_;
        d_output_ = other.d_output_;
        d_indices_ = other.d_indices_;
        h_indices_ = std::move(other.h_indices_);
        other.d_data_ = nullptr;
        other.d_output_ = nullptr;
        other.d_indices_ = nullptr;
    }
    return *this;
}

bool L1CacheBenchmark::initialize() {
    if (data_size_ == 0) {
        return false;
    }

    // Calculate padded size for alignment
    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);
    size_t padded_bytes = ((num_elements * sizeof(float) + alignment_ - 1) / alignment_) * alignment_;

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_data_, padded_bytes);
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMalloc(&d_output_, sizeof(float));
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    // Initialize data with pattern
    std::vector<float> host_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = static_cast<float>(i % 1000) * 0.001f;
    }

    err = cudaMemcpy(d_data_, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    return true;
}

void L1CacheBenchmark::cleanup() {
    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    if (d_indices_) {
        cudaFree(d_indices_);
        d_indices_ = nullptr;
    }
}

void L1CacheBenchmark::generate_random_indices() {
    if (data_size_ == 0) return;

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);
    h_indices_.resize(num_elements);

    // Create random permutation
    std::iota(h_indices_.begin(), h_indices_.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(h_indices_.begin(), h_indices_.end(), gen);

    // Allocate and copy to device
    if (d_indices_) {
        cudaFree(d_indices_);
    }
    cudaMalloc(&d_indices_, num_elements * sizeof(int));
    cudaMemcpy(d_indices_, h_indices_.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice);
}

float L1CacheBenchmark::calculate_bandwidth_gbps(size_t bytes, float elapsed_ms) const {
    if (elapsed_ms <= 0.0f) return 0.0f;
    // Bandwidth = bytes / time, convert to GB/s
    float seconds = elapsed_ms / 1000.0f;
    float bytes_per_second = static_cast<float>(bytes) / seconds;
    return bytes_per_second / (1024.0f * 1024.0f * 1024.0f);
}

float L1CacheBenchmark::estimate_hit_rate_sequential() const {
    // Sequential access within L1 size should have near-perfect hit rate
    constexpr size_t kL1Size = 128 * 1024;  // 128KB
    if (data_size_ <= kL1Size) {
        return 0.98f;  // Near-perfect hit rate for L1-resident data
    }
    // Streaming access for larger data
    float l1_lines = static_cast<float>(kL1Size) / 128.0f;  // 128B cache lines
    float data_lines = static_cast<float>(data_size_) / 128.0f;
    return l1_lines / data_lines;  // Simple capacity model
}

float L1CacheBenchmark::estimate_hit_rate_random() const {
    // Random access hit rate depends on working set vs cache size
    constexpr size_t kL1Size = 128 * 1024;  // 128KB
    float working_set = static_cast<float>(std::min(data_size_, kL1Size));
    float total_data = static_cast<float>(data_size_);

    if (data_size_ <= kL1Size) {
        // With sufficient iterations, random access can still achieve good hit rate
        // due to temporal locality from repeated accesses
        return 0.85f;
    }

    // For larger data, hit rate is roughly cache_size / data_size
    return std::min(0.95f, working_set / total_data);
}

L1CacheResult L1CacheBenchmark::run_sequential_access() {
    L1CacheResult result = {};
    result.status = L1CacheStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = L1CacheStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = L1CacheStatus::kError;
        return result;
    }

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 32;  // Multiple blocks for occupancy

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l1_sequential_read_kernel<<<blocks, threads_per_block>>>(d_data_, d_output_, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 100;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l1_sequential_read_kernel<<<blocks, threads_per_block>>>(d_data_, d_output_, num_elements);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float total_elapsed = timer.elapsed_ms();
    result.elapsed_ms = total_elapsed / iterations;
    result.iterations = iterations;

    // Calculate bandwidth
    size_t bytes_per_iteration = num_elements * sizeof(float);
    result.bytes_processed = bytes_per_iteration * iterations;
    result.bandwidth_gbps = calculate_bandwidth_gbps(result.bytes_processed, total_elapsed);

    // Estimate hit rate
    result.hit_rate = estimate_hit_rate_sequential();

    return result;
}

L1CacheResult L1CacheBenchmark::run_random_access() {
    L1CacheResult result = {};
    result.status = L1CacheStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = L1CacheStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = L1CacheStatus::kError;
        return result;
    }

    // Generate random indices
    generate_random_indices();

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 32;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l1_random_read_kernel<<<blocks, threads_per_block>>>(
            d_data_, d_output_, d_indices_, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 100;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l1_random_read_kernel<<<blocks, threads_per_block>>>(
            d_data_, d_output_, d_indices_, num_elements);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float total_elapsed = timer.elapsed_ms();
    result.elapsed_ms = total_elapsed / iterations;
    result.iterations = iterations;

    // Calculate bandwidth
    size_t bytes_per_iteration = num_elements * sizeof(float);
    result.bytes_processed = bytes_per_iteration * iterations;
    result.bandwidth_gbps = calculate_bandwidth_gbps(result.bytes_processed, total_elapsed);

    // Estimate hit rate
    result.hit_rate = estimate_hit_rate_random();

    return result;
}

}  // namespace cpm
