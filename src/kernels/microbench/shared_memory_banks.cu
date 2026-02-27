/**
 * @file shared_memory_banks.cu
 * @brief Shared memory bank conflict microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * Measures latency with varying degrees of bank conflicts.
 *
 * RTX 4070 (Ada Lovelace, sm_89) shared memory:
 * - 32 banks (standard configuration)
 * - Bank width: 4 bytes
 * - Bank conflict occurs when multiple threads in a warp access same bank
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// Shared Memory No Conflict Kernel
// ============================================================================

/**
 * @brief Shared memory latency measurement with no bank conflicts
 *
 * Each thread in a warp accesses a different bank (stride = 1).
 * This is the optimal access pattern.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of load operations
 */
__global__ void shared_memory_no_conflict_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Shared memory buffer - 32 banks * 4 bytes = 128 bytes per row
    __shared__ int smem_buffer[1024];

    // Initialize shared memory
    if (threadIdx.x < 1024) {
        smem_buffer[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int result = 0;

    // Calculate index for no conflict: each thread accesses different bank
    // stride-1 access: thread i accesses bank i
    int idx = tid;

    // Read cycle counter before shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Shared memory loads with no conflict
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // Load from shared memory
        asm volatile (
            "ld.shared.b32 %0, [%1];"
            : "=r"(result)
            : "r"(idx * 4)
        );
        // Update index to create dependency chain
        idx = result % 1024;
    }

    // Read cycle counter after shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (result < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// Shared Memory 2-Way Conflict Kernel
// ============================================================================

/**
 * @brief Shared memory latency measurement with 2-way bank conflicts
 *
 * Pairs of threads access the same bank (stride = 16).
 * With 32 threads and stride 16, we get 2-way conflicts.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of load operations
 */
__global__ void shared_memory_2way_conflict_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Shared memory buffer
    __shared__ int smem_buffer[1024];

    // Initialize shared memory
    if (threadIdx.x < 1024) {
        smem_buffer[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int result = 0;

    // Calculate index for 2-way conflict: stride 16
    // Threads 0,16 -> bank 0; threads 1,17 -> bank 1; etc.
    int idx = (tid % 16) * 16 + (tid / 16);

    // Read cycle counter before shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Shared memory loads with 2-way conflict
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile (
            "ld.shared.b32 %0, [%1];"
            : "=r"(result)
            : "r"(idx * 4)
        );
        idx = result % 1024;
    }

    // Read cycle counter after shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (result < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// Shared Memory 4-Way Conflict Kernel
// ============================================================================

/**
 * @brief Shared memory latency measurement with 4-way bank conflicts
 *
 * Groups of 4 threads access the same bank (stride = 8).
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of load operations
 */
__global__ void shared_memory_4way_conflict_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Shared memory buffer
    __shared__ int smem_buffer[1024];

    // Initialize shared memory
    if (threadIdx.x < 1024) {
        smem_buffer[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int result = 0;

    // Calculate index for 4-way conflict: stride 8
    int idx = (tid % 8) * 8 + (tid / 8);

    // Read cycle counter before shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Shared memory loads with 4-way conflict
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile (
            "ld.shared.b32 %0, [%1];"
            : "=r"(result)
            : "r"(idx * 4)
        );
        idx = result % 1024;
    }

    // Read cycle counter after shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (result < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// Shared Memory 8-Way Conflict Kernel
// ============================================================================

/**
 * @brief Shared memory latency measurement with 8-way bank conflicts
 *
 * Groups of 8 threads access the same bank (stride = 4).
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of load operations
 */
__global__ void shared_memory_8way_conflict_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Shared memory buffer
    __shared__ int smem_buffer[1024];

    // Initialize shared memory
    if (threadIdx.x < 1024) {
        smem_buffer[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int result = 0;

    // Calculate index for 8-way conflict: stride 4
    int idx = (tid % 4) * 4 + (tid / 4);

    // Read cycle counter before shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Shared memory loads with 8-way conflict
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile (
            "ld.shared.b32 %0, [%1];"
            : "=r"(result)
            : "r"(idx * 4)
        );
        idx = result % 1024;
    }

    // Read cycle counter after shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (result < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// Shared Memory 32-Way Conflict Kernel
// ============================================================================

/**
 * @brief Shared memory latency measurement with 32-way bank conflicts
 *
 * All threads in a warp access the same bank (stride = 0).
 * This is the worst-case access pattern.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of load operations
 */
__global__ void shared_memory_32way_conflict_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Shared memory buffer
    __shared__ int smem_buffer[1024];

    // Initialize shared memory
    if (threadIdx.x < 1024) {
        smem_buffer[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    int tid = threadIdx.x;
    int result = 0;

    // Calculate index for 32-way conflict: all threads access same location
    int idx = 0;

    // Read cycle counter before shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Shared memory loads with 32-way conflict
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile (
            "ld.shared.b32 %0, [%1];"
            : "=r"(result)
            : "r"(idx * 4)
        );
        idx = result % 1024;
    }

    // Read cycle counter after shared memory access
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    if (result < 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

}  // namespace microbench
}  // namespace cpm
