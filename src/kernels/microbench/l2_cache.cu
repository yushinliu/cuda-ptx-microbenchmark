/**
 * @file l2_cache.cu
 * @brief L2 cache and global memory microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * - Pointer chasing for latency measurement
 * - Sequential access with ld.global.ca for L2 bandwidth
 * - Uncached global memory access for global latency
 *
 * RTX 4070 (Ada Lovelace, sm_89) memory hierarchy:
 * - L1 cache: 128KB per SM
 * - L2 cache: 48MB shared
 * - Global memory: GDDR6X
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// L2 Cache Latency Kernel
// ============================================================================

/**
 * @brief L2 cache latency measurement kernel using pointer chasing
 *
 * Creates a dependency chain of memory loads that hit in L2 cache.
 * Uses a pre-initialized linked list pattern within L2-sized buffer.
 *
 * @param cycles Output: total cycles elapsed
 * @param buffer Pointer to buffer with pointer-chasing pattern
 * @param buffer_size Size of buffer in elements
 * @param iterations Number of chase iterations
 */
__global__ void l2_cache_latency_kernel(uint64_t* cycles, int* buffer, int buffer_size, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize pointer chase index
    int idx = 0;

    // Warm up L2 cache by traversing the linked list once
    #pragma unroll 1
    for (int i = 0; i < buffer_size; ++i) {
        idx = buffer[idx];
    }

    // Reset index for measurement
    idx = 0;

    // Read cycle counter before the chase
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Pointer chase through L2 cache
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // Load with .ca (cache at all levels) to ensure L2 hits
        asm volatile (
            "ld.global.ca.s32 %0, [%1];"
            : "=r"(idx)
            : "l"(&buffer[idx])
        );
    }

    // Read cycle counter after the chase
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (idx < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// L2 Cache Bandwidth Kernel
// ============================================================================

/**
 * @brief L2 cache bandwidth measurement kernel
 *
 * Measures sequential read bandwidth from L2 cache using ld.global.ca.
 *
 * @param cycles Output: total cycles elapsed
 * @param output Output buffer for writes
 * @param input Input buffer for reads
 * @param num_elements Number of elements to process
 */
__global__ void l2_cache_bandwidth_kernel(uint64_t* cycles, float* output, const float* input, int num_elements) {
    uint64_t start_cycle, end_cycle;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Read cycle counter before bandwidth test
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Sequential read from L2 cache
    #pragma unroll 1
    for (int i = tid; i < num_elements; i += stride) {
        float val;
        // Load with .ca (cache at all levels)
        asm volatile (
            "ld.global.ca.f32 %0, [%1];"
            : "=f"(val)
            : "l"(&input[i])
        );
        sum += val;
    }

    // Read cycle counter after bandwidth test
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Write result to prevent optimization
    if (tid < num_elements) {
        output[tid] = sum;
    }

    // Store timing result (only from thread 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }
}

// ============================================================================
// Global Memory Latency Kernel
// ============================================================================

/**
 * @brief Global memory latency measurement kernel
 *
 * Creates a dependency chain of uncached memory loads.
 * Uses ld.global.cg (cache global) to bypass L1 and potentially L2.
 *
 * @param cycles Output: total cycles elapsed
 * @param buffer Pointer to buffer with pointer-chasing pattern
 * @param buffer_size Size of buffer in elements
 * @param iterations Number of chase iterations
 */
__global__ void global_memory_latency_kernel(uint64_t* cycles, int* buffer, int buffer_size, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize pointer chase index
    int idx = 0;

    // Read cycle counter before the chase
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Pointer chase through global memory (uncached)
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // Load with .cg (cache global) to bypass L1
        // This forces global memory access
        asm volatile (
            "ld.global.cg.s32 %0, [%1];"
            : "=r"(idx)
            : "l"(&buffer[idx])
        );
    }

    // Read cycle counter after the chase
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (idx < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

}  // namespace microbench
}  // namespace cpm
