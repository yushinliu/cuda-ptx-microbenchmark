/**
 * @file memory_bandwidth.cu
 * @brief Memory bandwidth microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * - Sequential access: Measure peak bandwidth with coalesced access
 * - Random access: Pointer chasing to measure latency-bound bandwidth
 * - Strided access: Measure cache line and memory controller behavior
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// Sequential Bandwidth Kernel
// ============================================================================

/**
 * @brief Sequential memory bandwidth test using coalesced access
 *
 * Each thread reads consecutive elements from global memory.
 * Uses PTX inline assembly for precise timing.
 */
__global__ void sequential_bandwidth_kernel(float* data, float* result,
                                            size_t n, uint64_t* cycles) {
    uint64_t start_cycle, end_cycle;
    float sum = 0.0f;

    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Read cycle counter before computation
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Sequential access pattern - each thread reads strided elements
    for (size_t i = tid; i < n; i += stride) {
        float value;
        // Use PTX ld.global for explicit global memory load
        asm volatile ("ld.global.f32 %0, [%1];"
                      : "=f"(value)
                      : "l"(data + i));
        sum += value;
    }

    // Read cycle counter after computation
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (to prevent optimization)
    if (tid == 0) {
        *result = sum;
        *cycles = end_cycle - start_cycle;
    }
}

// ============================================================================
// Random Bandwidth Kernel (Pointer Chasing)
// ============================================================================

/**
 * @brief Random memory bandwidth test using pointer chasing
 *
 * Uses indices array to create a linked-list traversal pattern.
 * This measures latency-bound performance rather than peak bandwidth.
 */
__global__ void random_bandwidth_kernel(int* indices, float* data,
                                        size_t n, uint64_t* cycles) {
    uint64_t start_cycle, end_cycle;
    float sum = 0.0f;
    int idx = 0;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Pointer chasing pattern
    // Each iteration depends on the previous (creates dependency chain)
    for (size_t i = 0; i < n; ++i) {
        // Load next index
        int next_idx;
        asm volatile ("ld.global.s32 %0, [%1];"
                      : "=r"(next_idx)
                      : "l"(indices + idx));

        // Load data at current index
        float value;
        asm volatile ("ld.global.f32 %0, [%1];"
                      : "=f"(value)
                      : "l"(data + idx));

        sum += value;
        idx = next_idx;

        // Prevent infinite loop on invalid index
        if (idx < 0 || idx >= static_cast<int>(n)) {
            idx = 0;
        }
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store results
    *cycles = end_cycle - start_cycle;
    // Use sum to prevent optimization
    if (sum != sum) {  // NaN check (always false for valid data)
        *cycles = 0;
    }
}

// ============================================================================
// Strided Bandwidth Kernel
// ============================================================================

/**
 * @brief Strided memory bandwidth test
 *
 * Accesses memory with configurable stride to measure:
 * - Cache line utilization
 * - Memory controller behavior
 * - Bank conflict patterns
 */
__global__ void stride_bandwidth_kernel(float* data, size_t stride,
                                        size_t n, uint64_t* cycles) {
    uint64_t start_cycle, end_cycle;
    float sum = 0.0f;

    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Strided access pattern
    // Each thread accesses elements separated by 'stride'
    size_t num_accesses = n / stride;
    for (size_t i = tid; i < num_accesses; i += total_threads) {
        size_t idx = i * stride;
        if (idx < n) {
            float value;
            asm volatile ("ld.global.f32 %0, [%1];"
                          : "=f"(value)
                          : "l"(data + idx));
            sum += value;
        }
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store results
    if (tid == 0) {
        *cycles = end_cycle - start_cycle;
        // Prevent optimization
        if (sum != sum) {
            *cycles = 0;
        }
    }
}

}  // namespace microbench
}  // namespace cpm

// ============================================================================
// C API for test linkage
// ============================================================================

using namespace cpm::microbench;

void launch_sequential_bandwidth(float* data, float* result, size_t n,
                                  uint64_t* cycles, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (static_cast<int>(n) + block_size - 1) / block_size;
    grid_size = min(grid_size, 256);  // Limit grid size

    sequential_bandwidth_kernel<<<grid_size, block_size, 0, stream>>>(
        data, result, n, cycles);
}

void launch_random_bandwidth(int* indices, float* data, size_t n,
                              uint64_t* cycles, cudaStream_t stream) {
    random_bandwidth_kernel<<<1, 1, 0, stream>>>(indices, data, n, cycles);
}

void launch_stride_bandwidth(float* data, size_t stride, size_t n,
                              uint64_t* cycles, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = 64;

    stride_bandwidth_kernel<<<grid_size, block_size, 0, stream>>>(
        data, stride, n, cycles);
}
