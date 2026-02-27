#include "kernels/memory/global_memory.h"
#include "core/gpu_timer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>

namespace cpm {

// CUDA kernel implementations
__global__ void global_read_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    #pragma unroll 16
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, sum);
    }
}

__global__ void global_write_kernel(float* output, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    #pragma unroll 16
    for (size_t i = idx; i < n; i += stride) {
        output[i] = value;
    }
}

__global__ void global_copy_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    #pragma unroll 16
    for (size_t i = idx; i < n; i += stride) {
        output[i] = input[i];
    }
}

// MemoryBandwidthBenchmark implementation
MemoryBandwidthBenchmark::MemoryBandwidthBenchmark(size_t data_size, size_t alignment)
    : data_size_(data_size), alignment_(alignment),
      d_src_data_(nullptr), d_dst_data_(nullptr), d_output_(nullptr) {
}

MemoryBandwidthBenchmark::~MemoryBandwidthBenchmark() {
    cleanup();
}

MemoryBandwidthBenchmark::MemoryBandwidthBenchmark(MemoryBandwidthBenchmark&& other) noexcept
    : data_size_(other.data_size_),
      alignment_(other.alignment_),
      d_src_data_(other.d_src_data_),
      d_dst_data_(other.d_dst_data_),
      d_output_(other.d_output_) {
    other.d_src_data_ = nullptr;
    other.d_dst_data_ = nullptr;
    other.d_output_ = nullptr;
}

MemoryBandwidthBenchmark& MemoryBandwidthBenchmark::operator=(MemoryBandwidthBenchmark&& other) noexcept {
    if (this != &other) {
        cleanup();
        data_size_ = other.data_size_;
        alignment_ = other.alignment_;
        d_src_data_ = other.d_src_data_;
        d_dst_data_ = other.d_dst_data_;
        d_output_ = other.d_output_;
        other.d_src_data_ = nullptr;
        other.d_dst_data_ = nullptr;
        other.d_output_ = nullptr;
    }
    return *this;
}

bool MemoryBandwidthBenchmark::initialize() {
    if (data_size_ == 0) {
        return false;
    }

    // Calculate padded size for alignment
    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);
    size_t padded_bytes = ((num_elements * sizeof(float) + alignment_ - 1) / alignment_) * alignment_;

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_src_data_, padded_bytes);
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMalloc(&d_dst_data_, padded_bytes);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    err = cudaMalloc(&d_output_, sizeof(float));
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    // Initialize source data with pattern
    std::vector<float> host_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = static_cast<float>(i % 1000) * 0.001f;
    }

    err = cudaMemcpy(d_src_data_, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        return false;
    }

    return true;
}

void MemoryBandwidthBenchmark::cleanup() {
    if (d_src_data_) {
        cudaFree(d_src_data_);
        d_src_data_ = nullptr;
    }
    if (d_dst_data_) {
        cudaFree(d_dst_data_);
        d_dst_data_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
}

float MemoryBandwidthBenchmark::calculate_bandwidth_gbps(size_t bytes, float elapsed_ms) const {
    if (elapsed_ms <= 0.0f) return 0.0f;
    // Bandwidth = bytes / time, convert to GB/s
    float seconds = elapsed_ms / 1000.0f;
    float bytes_per_second = static_cast<float>(bytes) / seconds;
    return bytes_per_second / (1024.0f * 1024.0f * 1024.0f);
}

BandwidthResult MemoryBandwidthBenchmark::run_read_benchmark() {
    BandwidthResult result = {};
    result.status = BandwidthStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = BandwidthStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = BandwidthStatus::kError;
        return result;
    }

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 128;  // Many blocks for memory latency hiding

    // Warmup
    for (int i = 0; i < 3; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        global_read_kernel<<<blocks, threads_per_block>>>(d_src_data_, d_output_, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 20;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_output_, 0, sizeof(float));
        global_read_kernel<<<blocks, threads_per_block>>>(d_src_data_, d_output_, num_elements);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float total_elapsed = timer.elapsed_ms();
    result.elapsed_ms = total_elapsed / iterations;
    result.iterations = iterations;

    // Calculate bandwidth
    size_t bytes_per_iteration = num_elements * sizeof(float);
    result.bytes_processed = bytes_per_iteration * iterations;
    result.read_bandwidth_gbps = calculate_bandwidth_gbps(result.bytes_processed, total_elapsed);

    return result;
}

BandwidthResult MemoryBandwidthBenchmark::run_write_benchmark() {
    BandwidthResult result = {};
    result.status = BandwidthStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = BandwidthStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = BandwidthStatus::kError;
        return result;
    }

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 128;

    float value = 1.0f;

    // Warmup
    for (int i = 0; i < 3; ++i) {
        global_write_kernel<<<blocks, threads_per_block>>>(d_dst_data_, value, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 20;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        global_write_kernel<<<blocks, threads_per_block>>>(d_dst_data_, value, num_elements);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float total_elapsed = timer.elapsed_ms();
    result.elapsed_ms = total_elapsed / iterations;
    result.iterations = iterations;

    // Calculate bandwidth
    size_t bytes_per_iteration = num_elements * sizeof(float);
    result.bytes_processed = bytes_per_iteration * iterations;
    result.write_bandwidth_gbps = calculate_bandwidth_gbps(result.bytes_processed, total_elapsed);

    return result;
}

BandwidthResult MemoryBandwidthBenchmark::run_copy_benchmark() {
    BandwidthResult result = {};
    result.status = BandwidthStatus::kSuccess;

    // Handle empty data
    if (data_size_ == 0) {
        result.status = BandwidthStatus::kSkipped;
        return result;
    }

    // Initialize
    if (!initialize()) {
        result.status = BandwidthStatus::kError;
        return result;
    }

    size_t num_elements = (data_size_ + sizeof(float) - 1) / sizeof(float);

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = 128;

    // Warmup
    for (int i = 0; i < 3; ++i) {
        global_copy_kernel<<<blocks, threads_per_block>>>(d_src_data_, d_dst_data_, num_elements);
        cudaDeviceSynchronize();
    }

    // Timed runs
    GpuTimer timer;
    const int iterations = 20;

    timer.start();
    for (int i = 0; i < iterations; ++i) {
        global_copy_kernel<<<blocks, threads_per_block>>>(d_src_data_, d_dst_data_, num_elements);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float total_elapsed = timer.elapsed_ms();
    result.elapsed_ms = total_elapsed / iterations;
    result.iterations = iterations;

    // Calculate bandwidth (2x bytes for read + write)
    size_t bytes_per_iteration = num_elements * sizeof(float) * 2;
    result.bytes_processed = bytes_per_iteration * iterations;
    result.copy_bandwidth_gbps = calculate_bandwidth_gbps(result.bytes_processed, total_elapsed);

    return result;
}

}  // namespace cpm
