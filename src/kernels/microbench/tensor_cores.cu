/**
 * @file tensor_cores.cu
 * @brief Tensor Core microbenchmark kernels using PTX inline assembly
 *
 * RTX 4070 (Ada Lovelace, sm_89) Tensor Core instructions:
 * - HMMA: Half-precision matrix multiply-accumulate using mma.sync PTX
 * - IMMA: Integer matrix multiply-accumulate
 *
 * Uses mma.sync.aligned.m16n8k8 PTX instruction for half-precision MMA.
 * Requires sm_70+ (Volta and later).
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// HMMA Latency Kernel using mma.sync PTX
// ============================================================================

/**
 * @brief HMMA (Half-precision MMA) latency measurement kernel
 *
 * Measures latency of half-precision matrix multiply-accumulate using
 * mma.sync.aligned.m16n8k8 PTX instruction.
 * Available on sm_70+ (Volta and later).
 *
 * m16n8k8 shape: A(16x8) * B(8x8) + C(16x8)
 * - A: 2 registers (4 half2 elements)
 * - B: 1 register (2 half2 elements)
 * - C: 2 registers (4 half2 elements)
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of HMMA operations
 */
__global__ void hmma_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Matrix fragments for m16n8k8 MMA
    // A matrix (16x8): 4 half2 elements = 2 x 32-bit registers
    unsigned int a0 = 0x3C003C00;  // Two FP16 1.0 values
    unsigned int a1 = 0x3C003C00;

    // B matrix (8x8): 2 half2 elements = 1 x 32-bit register
    unsigned int b0 = 0x3C003C00;

    // C matrix (16x8): 4 half2 elements = 2 x 32-bit registers (accumulator)
    unsigned int c0 = 0;
    unsigned int c1 = 0;

    // Read cycle counter before HMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // HMMA operations using mma.sync PTX
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
        // D = A * B + C
        // Output: c0, c1 (D matrix, also accumulator)
        // Input A: a0, a1 (row-major)
        // Input B: b0 (col-major)
        // Input C: c0, c1 (accumulator)
        asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3}, {%4}, {%5, %6};"
            : "=r"(c0), "=r"(c1)
            : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1)
        );
    }

    // Read cycle counter after HMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    unsigned int sum = c0 + c1;
    if (sum == 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// HMMA Throughput Kernel
// ============================================================================

/**
 * @brief HMMA throughput measurement kernel
 *
 * Uses independent HMMA operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void hmma_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent matrix fragments for throughput test
    unsigned int a0_0 = 0x3C003C00;
    unsigned int a1_0 = 0x3C003C00;
    unsigned int b0_0 = 0x3C003C00;
    unsigned int c0_0 = 0;
    unsigned int c1_0 = 0;

    unsigned int a0_1 = 0x3C003C00;
    unsigned int a1_1 = 0x3C003C00;
    unsigned int b0_1 = 0x3C003C00;
    unsigned int c0_1 = 0;
    unsigned int c1_1 = 0;

    // Read cycle counter before HMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent HMMA operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // First HMMA
        asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3}, {%4}, {%5, %6};"
            : "=r"(c0_0), "=r"(c1_0)
            : "r"(a0_0), "r"(a1_0), "r"(b0_0), "r"(c0_0), "r"(c1_0)
        );
        // Second HMMA (independent)
        asm volatile (
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3}, {%4}, {%5, %6};"
            : "=r"(c0_1), "=r"(c1_1)
            : "r"(a0_1), "r"(a1_1), "r"(b0_1), "r"(c0_1), "r"(c1_1)
        );
    }

    // Read cycle counter after HMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    unsigned int sum = c0_0 + c1_0 + c0_1 + c1_1;
    if (sum == 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// IMMA Latency Kernel using mma.sync PTX
// ============================================================================

/**
 * @brief IMMA (Integer MMA) latency measurement kernel
 *
 * Measures latency of integer matrix multiply-accumulate using
 * mma.sync.aligned.m16n8k16 PTX instruction with s32.s8.s8.s32.
 * Available on sm_75+ (Turing and later).
 *
 * m16n8k16 shape: A(16x16) * B(16x8) + C(16x8)
 * - A: 2 registers (8 x s8 elements, 4 per register)
 * - B: 1 register (4 x s8 elements)
 * - C: 4 registers (4 x s32 elements, 1 per register)
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of IMMA operations
 */
__global__ void imma_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Matrix fragments for m16n8k16 IMMA
    // A matrix (16x16, s8): 8 elements per thread = 2 x 32-bit registers
    // (4 x s8 packed per 32-bit register)
    unsigned int a0 = 0x01010101;  // Four INT8 1 values
    unsigned int a1 = 0x01010101;  // Four more INT8 1 values

    // B matrix (16x8, s8): 4 elements per thread = 1 x 32-bit register
    unsigned int b0 = 0x01010101;

    // C matrix (16x8, s32): 4 elements per thread = 4 x 32-bit registers
    unsigned int c0 = 0;
    unsigned int c1 = 0;
    unsigned int c2 = 0;
    unsigned int c3 = 0;

    // Read cycle counter before IMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // IMMA operations using mma.sync PTX
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
        // D = A * B + C
        // A and B are signed 8-bit integers
        // C and D are signed 32-bit integers
        asm volatile (
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
            : "r"(a0), "r"(a1), "r"(b0),
              "r"(c0), "r"(c1), "r"(c2), "r"(c3)
        );
    }

    // Read cycle counter after IMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    unsigned int sum = c0 + c1 + c2 + c3;
    if (sum == 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

// ============================================================================
// IMMA Throughput Kernel
// ============================================================================

/**
 * @brief IMMA throughput measurement kernel
 *
 * Uses independent IMMA operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void imma_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent matrix fragments for throughput test
    unsigned int a0_0 = 0x01010101;
    unsigned int a1_0 = 0x01010101;
    unsigned int b0_0 = 0x01010101;
    unsigned int c0_0 = 0;
    unsigned int c1_0 = 0;
    unsigned int c2_0 = 0;
    unsigned int c3_0 = 0;

    unsigned int a0_1 = 0x01010101;
    unsigned int a1_1 = 0x01010101;
    unsigned int b0_1 = 0x01010101;
    unsigned int c0_1 = 0;
    unsigned int c1_1 = 0;
    unsigned int c2_1 = 0;
    unsigned int c3_1 = 0;

    // Read cycle counter before IMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent IMMA operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // First IMMA
        asm volatile (
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=r"(c0_0), "=r"(c1_0), "=r"(c2_0), "=r"(c3_0)
            : "r"(a0_0), "r"(a1_0), "r"(b0_0),
              "r"(c0_0), "r"(c1_0), "r"(c2_0), "r"(c3_0)
        );
        // Second IMMA (independent)
        asm volatile (
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=r"(c0_1), "=r"(c1_1), "=r"(c2_1), "=r"(c3_1)
            : "r"(a0_1), "r"(a1_1), "r"(b0_1),
              "r"(c0_1), "r"(c1_1), "r"(c2_1), "r"(c3_1)
        );
    }

    // Read cycle counter after IMMA
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0)
    if (threadIdx.x == 0) {
        *cycles = end_cycle - start_cycle;
    }

    // Prevent optimization
    unsigned int sum = c0_0 + c1_0 + c2_0 + c3_0 + c0_1 + c1_1 + c2_1 + c3_1;
    if (sum == 0) {
        if (threadIdx.x == 0) {
            *cycles = 0;
        }
    }
}

}  // namespace microbench
}  // namespace cpm
