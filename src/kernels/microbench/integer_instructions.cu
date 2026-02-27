/**
 * @file integer_instructions.cu
 * @brief Integer instruction microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * Uses dependency chains for latency and independent streams for throughput.
 *
 * RTX 4070 (Ada Lovelace, sm_89) specific instructions:
 * - IADD3: Integer add with 3 operands
 * - LOP3: Logic operations (AND/OR/XOR) with 3 operands
 * - SEL: Select instruction
 * - SHFL: Warp shuffle
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// IADD3 Latency Kernel
// ============================================================================

/**
 * @brief IADD3 (Integer Add 3 operands) latency measurement kernel
 *
 * Creates a dependency chain of IADD3 operations:
 *   result = result + a + b
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of IADD3 operations in the chain
 */
__global__ void iadd3_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize with non-trivial values
    int result = 1;
    const int a = 1;
    const int b = 1;

    // Read cycle counter before the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain simulating IADD3: result = result + a + b
    // Using two add instructions since iadd3 is not a valid PTX instruction
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile (
            "add.s32 %0, %0, %1;\n\t"
            "add.s32 %0, %0, %2;"
            : "+r"(result)
            : "r"(a), "r"(b)
        );
    }

    // Read cycle counter after the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0) {
        *cycles = 0;
    }
}

// ============================================================================
// IADD3 Throughput Kernel
// ============================================================================

/**
 * @brief IADD3 throughput measurement kernel
 *
 * Uses independent IADD3 operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void iadd3_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    int r0 = 0, r1 = 1, r2 = 2, r3 = 3;
    int r4 = 4, r5 = 5, r6 = 6, r7 = 7;

    const int a = 1;
    const int b = 1;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent operations simulating IADD3
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r0) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r1) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r2) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r3) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r4) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r5) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r6) : "r"(a), "r"(b));
        asm volatile ("add.s32 %0, %0, %1; add.s32 %0, %0, %2;" : "+r"(r7) : "r"(a), "r"(b));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    int final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0) {
        *cycles = 0;
    }
}

// ============================================================================
// LOP3 Latency Kernel
// ============================================================================

/**
 * @brief LOP3 (Logic Operation 3 operands) latency measurement kernel
 *
 * Creates a dependency chain of LOP3 operations.
 * Uses LUT mode 0xF8 which is (a & b) ^ c
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of LOP3 operations in the chain
 */
__global__ void lop3_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize
    int result = 0xFFFFFFFF;
    const int a = 0xAAAAAAAA;
    const int b = 0x55555555;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of LOP3 operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX LOP3: result = (result & a) ^ b (LUT mode 0x96)
        asm volatile (
            "lop3.b32 %0, %0, %1, %2, 0x96;"
            : "+r"(result)
            : "r"(a), "r"(b)
        );
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result == 0) {
        *cycles = 0;
    }
}

// ============================================================================
// LOP3 Throughput Kernel
// ============================================================================

/**
 * @brief LOP3 throughput measurement kernel
 *
 * Uses independent LOP3 operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void lop3_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    int r0 = 0, r1 = 1, r2 = 2, r3 = 3;
    int r4 = 4, r5 = 5, r6 = 6, r7 = 7;

    const int a = 0xAAAAAAAA;
    const int b = 0x55555555;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent LOP3 operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r0) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r1) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r2) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r3) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r4) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r5) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r6) : "r"(a), "r"(b));
        asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r7) : "r"(a), "r"(b));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    int final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum == 0) {
        *cycles = 0;
    }
}

// ============================================================================
// SEL Latency Kernel
// ============================================================================

/**
 * @brief SEL (Select) latency measurement kernel
 *
 * Creates a dependency chain of SEL operations.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of SEL operations in the chain
 */
__global__ void sel_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize
    int result = 0;
    const int a = 1;
    const int b = 2;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of SEL operations
    // Using C conditional operator which compiles to selp
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // SEL: result = (result != 0) ? a : b
        result = (result != 0) ? a : b;
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0) {
        *cycles = 0;
    }
}

// ============================================================================
// SEL Throughput Kernel
// ============================================================================

/**
 * @brief SEL throughput measurement kernel
 *
 * Uses independent SEL operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void sel_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    int r0 = 0, r1 = 1, r2 = 0, r3 = 1;
    int r4 = 0, r5 = 1, r6 = 0, r7 = 1;

    const int a = 1;
    const int b = 2;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent SEL operations using C conditional operator
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        r0 = (r0 != 0) ? a : b;
        r1 = (r1 != 0) ? a : b;
        r2 = (r2 != 0) ? a : b;
        r3 = (r3 != 0) ? a : b;
        r4 = (r4 != 0) ? a : b;
        r5 = (r5 != 0) ? a : b;
        r6 = (r6 != 0) ? a : b;
        r7 = (r7 != 0) ? a : b;
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    int final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0) {
        *cycles = 0;
    }
}

// ============================================================================
// SHFL Latency Kernel
// ============================================================================

/**
 * @brief SHFL (Warp Shuffle) latency measurement kernel
 *
 * Creates a dependency chain of SHFL operations within a warp.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of SHFL operations in the chain
 */
__global__ void shfl_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize - each thread gets its lane ID
    int result = threadIdx.x % 32;
    const int mask = 0xFFFFFFFF;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of SHFL operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX SHFL: result = shfl(result, 0, mask)
        // Shuffle from lane 0 to all lanes
        asm volatile (
            "shfl.sync.idx.b32 %0, %0, 0, 0x1F, %1;"
            : "+r"(result)
            : "r"(mask)
        );
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result (only from thread 0 to avoid races)
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
// SHFL Throughput Kernel
// ============================================================================

/**
 * @brief SHFL throughput measurement kernel
 *
 * Uses independent SHFL operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void shfl_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent values
    int r0 = threadIdx.x % 32;
    int r1 = (threadIdx.x + 1) % 32;
    int r2 = (threadIdx.x + 2) % 32;
    int r3 = (threadIdx.x + 3) % 32;

    const int mask = 0xFFFFFFFF;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent SHFL operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("shfl.sync.idx.b32 %0, %0, 0, 0x1F, %1;" : "+r"(r0) : "r"(mask));
        asm volatile ("shfl.sync.idx.b32 %0, %0, 1, 0x1F, %1;" : "+r"(r1) : "r"(mask));
        asm volatile ("shfl.sync.idx.b32 %0, %0, 2, 0x1F, %1;" : "+r"(r2) : "r"(mask));
        asm volatile ("shfl.sync.idx.b32 %0, %0, 3, 0x1F, %1;" : "+r"(r3) : "r"(mask));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results (only from thread 0)
    if (threadIdx.x == 0) {
        int final_sum = r0 + r1 + r2 + r3;
        *cycles = end_cycle - start_cycle;

        if (final_sum < 0) {
            *cycles = 0;
        }
    }
}

}  // namespace microbench
}  // namespace cpm
