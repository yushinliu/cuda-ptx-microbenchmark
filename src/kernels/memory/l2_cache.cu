#include "kernels/memory/l2_cache.h"
#include "core/gpu_timer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <random>
#include <cmath>

namespace cpm {

// CUDA kernel implementations
__global__ void l2_sequential_read_kernel(const float* input, float* output, size_t n) {
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

__global__ void l2_stride_read_kernel(const float* input, float* output,
                                      size_t n, size_t stride_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        size_t index = (i * stride_elements) % n;
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

__global__ void l2_random_read_kernel(const float* input, float* output,
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

// L2CacheBenchmark implementation
L2CacheBenchmark::L2CacheBenchmark(size_t data_size, size_t alignment)
    : data_size_(data_size), alignment_(alignment),
      d_data_(nullptr), d_output_(nullptr), d_indices_(nullptr) {
}

L2CacheBenchmark::~L2CacheBenchmark() {
    cleanup();
}

L2CacheBenchmark::L2CacheBenchmark(L2CacheBenchmark&& other) noexcept
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

L2CacheBenchmark& L2CacheBenchmark::operator=(L2CacheBenchmark&& other) noexcept {
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

bool L2CacheBenchmark::initialize() {
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

void L2CacheBenchmark::cleanup() {
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

void L2CacheBenchmark::generate_random_indices() {
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

float L2CacheBenchmark::calculate_bandwidth_gbps(size_t bytes, float elapsed_ms) const {
    if (elapsed_ms <= 0.0f) return 0.0f;
    // Bandwidth = bytes / time, convert to GB/s
    float seconds = elapsed_ms / 1000.0f;
    float bytes_per_second = static_cast<float>(bytes) / seconds;
    return bytes_per_second / (1024.0f * 1024.0f * 1024.0f);
}

float L2CacheBenchmark::estimate_hit_rate_sequential() const {
    // Sequential access patterns
    constexpr size_t kL2Size = 36 * 1024 * 1024;  // 36MB

    if (data_size_ <= kL2Size) {
        // Data fits in L2 - streaming prefetch works well
        return 0.95f;
    }

    // Data exceeds L2 - streaming has no temporal locality
    // But prefetching still helps with spatial locality
    return 0.30f;  // Mostly streaming, few hits
}

float L2CacheBenchmark::estimate_hit_rate_random() const {
    // Random access hit rate
    constexpr size_t kL2Size = 36 * 1024 * 1024;  // 36MB

    if (data_size_ <= kL2Size) {
        // Working set fits in L2
        return 0.90f;
    }

    // Working set exceeds L2
    float working_set = static_cast<float>(std::min(data_size_, kL2Size));
    float total_data = static_cast<float>(data_size_);
    return std::min(0.95f, working_set / total_data);
}

float L2CacheBenchmark::estimate_hit_rate_stride(size_t stride) const {
    // Strided access hit rate depends on stride size vs cache line
    constexpr size_t kCacheLine = 128;  // bytes
    constexpr size_t kElementsPerLine = kCacheLine / sizeof(float);  // 32 floats

    if (stride == 0) {
        return 0.0f;  // Invalid
    }

    if (stride <= kElementsPerLine) {
        // Stride within cache line - good spatial locality
        return 0.85f;
    } else if (stride <= kElementsPerLine * 4) {
        // Moderate stride - some locality
        return 0.60f;
    } else {
        // Large stride - poor locality
        return 0.25f;
    }
}

L2CacheResult L2CacheBenchmark::run_sequential_access() {
    L2CacheResult result = {};
    result.status = L2CacheStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = L2CacheStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = L2CacheStatus::kError;
        return result;
    }

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 64;  // More blocks for larger data

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l2_sequential_read_kernel<<<blocks, threads_per_block>>>(d_data_, d_output_, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 50;  // Fewer iterations for larger data

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l2_sequential_read_kernel<<<blocks, threads_per_block>>>(d_data_, d_output_, num_elements);
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
    result.stride = 1;

    return result;
}

L2CacheResult L2CacheBenchmark::run_random_access() {
    L2CacheResult result = {};
    result.status = L2CacheStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = L2CacheStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = L2CacheStatus::kError;
        return result;
    }

    // Generate random indices
    generate_random_indices();

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 64;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l2_random_read_kernel<<<blocks, threads_per_block>>>(
            d_data_, d_output_, d_indices_, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 50;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l2_random_read_kernel<<<blocks, threads_per_block>>>(
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
    result.stride = 0;

    return result;
}

L2CacheResult L2CacheBenchmark::run_stride_access(size_t stride_elements) {
    L2CacheResult result = {};
    result.status = L2CacheStatus::kSuccess;

    // Handle invalid parameters
    if (data_size_ == 0) {
        result.status = L2CacheStatus::kSkipped;
        return result;
    }

    if (stride_elements == 0) {
        result.status = L2CacheStatus::kInvalidStride;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = L2CacheStatus::kError;
        return result;
    }

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 64;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l2_stride_read_kernel<<<blocks, threads_per_block>>>(
            d_data_, d_output_, num_elements, stride_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 50;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        l2_stride_read_kernel<<<blocks, threads_per_block>>>(
            d_data_, d_output_, num_elements, stride_elements);
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
    result.hit_rate = estimate_hit_rate_stride(stride_elements);
    result.stride = stride_elements;

    return result;
}

}  // namespace cpm
