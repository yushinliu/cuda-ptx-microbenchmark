/**
 * @file instruction_throughput.cu
 * @brief Instruction throughput microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * Uses independent instructions to measure peak throughput.
 * Multiple independent operations can execute in parallel.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

/**
 * @brief FMA (Fused Multiply-Add) throughput measurement kernel
 *
 * Uses independent FMA operations to measure peak throughput.
 * Unlike latency test, these operations have no dependencies.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations (each with multiple independent FMAs)
 */
__global__ void fma_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Use multiple independent accumulators to allow ILP
    // Modern GPUs can issue multiple FMAs per cycle
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;

    const float a = 1.000001f;
    const float b = 0.000001f;

    // Read cycle counter before computation
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent FMA operations - no dependencies between them
    // This allows the GPU to exploit instruction-level parallelism
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // 8 independent FMAs per iteration
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r0) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r1) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r2) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r3) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r4) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r5) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r6) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r7) : "f"(a), "f"(b));

        // Additional independent FMAs to increase ILP
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r0) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r1) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r2) : "f"(a), "f"(b));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r3) : "f"(a), "f"(b));
    }

    // Read cycle counter after computation
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results to prevent optimization
    float final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (final_sum < 0.0f) {
        *cycles = 0;
    }
}

/**
 * @brief ADD throughput measurement kernel
 *
 * Uses independent ADD operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void add_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    float r0 = 0.0f, r1 = 1.0f, r2 = 2.0f, r3 = 3.0f;
    float r4 = 4.0f, r5 = 5.0f, r6 = 6.0f, r7 = 7.0f;

    const float inc = 1.0f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent ADD operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r0) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r1) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r2) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r3) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r4) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r5) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r6) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r7) : "f"(inc));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    float final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0.0f) {
        *cycles = 0;
    }
}

/**
 * @brief MUL throughput measurement kernel
 *
 * Uses independent MUL operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void mul_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    float r0 = 1.001f, r1 = 1.002f, r2 = 1.003f, r3 = 1.004f;
    float r4 = 1.005f, r5 = 1.006f, r6 = 1.007f, r7 = 1.008f;

    const float mult = 1.0001f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent MUL operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r0) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r1) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r2) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r3) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r4) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r5) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r6) : "f"(mult));
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r7) : "f"(mult));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    float final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0.0f) {
        *cycles = 0;
    }
}

/**
 * @brief Mixed instruction throughput measurement kernel
 *
 * Measures throughput of mixed FMA, ADD, and MUL operations.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void mixed_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    const float a = 1.0001f;
    const float b = 0.0001f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Mixed independent operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // FMA
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r0) : "f"(a), "f"(b));
        // ADD
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(r1) : "f"(a));
        // MUL
        asm volatile ("mul.f32 %0, %0, %1;" : "+f"(r2) : "f"(a));
        // FMA again
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r3) : "f"(a), "f"(b));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    float final_sum = r0 + r1 + r2 + r3;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0.0f) {
        *cycles = 0;
    }
}

}  // namespace microbench
}  // namespace cpm
