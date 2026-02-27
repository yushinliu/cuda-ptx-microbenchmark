/**
 * @file shared_memory.cu
 * @brief Shared memory bank conflict microbenchmark kernels using PTX
 *
 * Based on Volta microbenchmark methodology:
 * Measures shared memory access patterns with varying degrees of bank conflicts.
 * Uses PTX inline assembly for precise control over memory instructions.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

/**
 * @brief Shared memory bank conflict test kernel
 *
 * Tests shared memory bandwidth with configurable stride to create
 * different levels of bank conflicts.
 *
 * Bank conflict explanation:
 * - Stride 1: Consecutive threads access consecutive banks (no conflict)
 * - Stride 2: Even threads access even banks, odd threads access odd banks
 * - Stride 32: All threads access the same bank (maximum conflict on 32-bank GPUs)
 *
 * @param result Output: sum of accessed values
 * @param stride Access stride between threads
 * @param cycles Output: total cycles elapsed
 */
__global__ void bank_conflict_kernel(float* result, int stride,
                                     uint64_t* cycles) {
    // Shared memory buffer - sized to hold multiple banks worth of data
    // 4096 floats = 16KB, enough for bank conflict testing
    __shared__ float shared_buffer[4096];

    uint64_t start_cycle, end_cycle;
    float sum = 0.0f;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Initialize shared memory (coalesced, no conflicts)
    for (int i = tid; i < 4096; i += block_size) {
        shared_buffer[i] = static_cast<float>(i);
    }

    // Ensure all threads have initialized before testing
    __syncthreads();

    // Read cycle counter before bank conflict test
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Access shared memory with specified stride
    // This creates bank conflicts based on stride value
    #pragma unroll 1
    for (int iter = 0; iter < 100; ++iter) {
        int idx = (tid * stride + iter) % 4096;

        // Use PTX ld.shared for explicit shared memory load
        float value;
        int offset = idx * sizeof(float);  // Byte offset as int
        asm volatile (
            "ld.shared.f32 %0, [%1];"
            : "=f"(value)
            : "r"(offset)
        );

        sum += value;
    }

    // Read cycle counter after test
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store results (only thread 0 to avoid races)
    if (tid == 0) {
        *result = sum;
        *cycles = end_cycle - start_cycle;
    }
}

/**
 * @brief Shared memory bandwidth test with explicit bank addressing
 *
 * This version uses explicit bank addressing to measure:
 * - Peak bandwidth (no conflicts)
 * - 2-way, 4-way, 16-way, 32-way conflicts
 */
__global__ void shared_memory_bandwidth_kernel(float* result, int access_pattern,
                                               uint64_t* cycles) {
    // 16KB shared memory buffer
    __shared__ float shared_buffer[4096];

    uint64_t start_cycle, end_cycle;
    float sum = 0.0f;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Initialize shared memory
    for (int i = tid; i < 4096; i += block_size) {
        shared_buffer[i] = static_cast<float>(i);
    }
    __syncthreads();

    // Calculate access index based on pattern
    // Pattern 0: Linear (no conflict)
    // Pattern 1: Modulo 16 (16-way conflict)
    // Pattern 2: Modulo 32 (32-way conflict)
    int bank_id;
    switch (access_pattern) {
        case 0:  // Linear - no conflict
            bank_id = tid;
            break;
        case 1:  // 16-way conflict
            bank_id = tid % 16;
            break;
        case 2:  // 32-way conflict
            bank_id = tid % 32;
            break;
        default:
            bank_id = tid;
    }

    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    #pragma unroll 1
    for (int iter = 0; iter < 100; ++iter) {
        int idx = (bank_id + iter * 32) % 4096;

        float value;
        int offset = idx * sizeof(float);
        asm volatile (
            "ld.shared.f32 %0, [%1];"
            : "=f"(value)
            : "r"(offset)
        );

        sum += value;
    }

    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    if (tid == 0) {
        *result = sum;
        *cycles = end_cycle - start_cycle;
    }
}

}  // namespace microbench
}  // namespace cpm
