/**
 * @file double_precision.cu
 * @brief Double precision instruction microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * Uses dependency chains for latency and independent streams for throughput.
 *
 * RTX 4070 (Ada Lovelace, sm_89) double precision instructions:
 * - DADD: Double precision add
 * - DMUL: Double precision multiply
 * - DFMA: Double precision fused multiply-add
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// DADD Latency Kernel
// ============================================================================

/**
 * @brief DADD (Double Add) latency measurement kernel
 *
 * Creates a dependency chain of DADD operations:
 *   result = result + increment
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of DADD operations in the chain
 */
__global__ void dadd_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize with non-trivial values
    double result = 1.0;
    const double increment = 0.000001;

    // Read cycle counter before the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of DADD operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX DADD: result = result + increment
        asm volatile (
            "add.f64 %0, %0, %1;"
            : "+d"(result)
            : "d"(increment)
        );
    }

    // Read cycle counter after the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0.0) {
        *cycles = 0;
    }
}

// ============================================================================
// DADD Throughput Kernel
// ============================================================================

/**
 * @brief DADD throughput measurement kernel
 *
 * Uses independent DADD operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void dadd_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    double r0 = 0.0, r1 = 1.0, r2 = 2.0, r3 = 3.0;
    double r4 = 4.0, r5 = 5.0, r6 = 6.0, r7 = 7.0;

    const double inc = 0.000001;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent DADD operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r0) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r1) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r2) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r3) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r4) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r5) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r6) : "d"(inc));
        asm volatile ("add.f64 %0, %0, %1;" : "+d"(r7) : "d"(inc));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    double final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0.0) {
        *cycles = 0;
    }
}

// ============================================================================
// DMUL Latency Kernel
// ============================================================================

/**
 * @brief DMUL (Double Multiply) latency measurement kernel
 *
 * Creates a dependency chain of DMUL operations.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of DMUL operations in the chain
 */
__global__ void dmul_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize with value close to 1 to prevent overflow/underflow
    double result = 1.000001;
    const double multiplier = 1.000001;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of DMUL operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX DMUL: result = result * multiplier
        asm volatile (
            "mul.f64 %0, %0, %1;"
            : "+d"(result)
            : "d"(multiplier)
        );
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0.0) {
        *cycles = 0;
    }
}

// ============================================================================
// DMUL Throughput Kernel
// ============================================================================

/**
 * @brief DMUL throughput measurement kernel
 *
 * Uses independent DMUL operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void dmul_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    double r0 = 1.001, r1 = 1.002, r2 = 1.003, r3 = 1.004;
    double r4 = 1.005, r5 = 1.006, r6 = 1.007, r7 = 1.008;

    const double mult = 1.0001;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent DMUL operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r0) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r1) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r2) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r3) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r4) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r5) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r6) : "d"(mult));
        asm volatile ("mul.f64 %0, %0, %1;" : "+d"(r7) : "d"(mult));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    double final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0.0) {
        *cycles = 0;
    }
}

// ============================================================================
// DFMA Latency Kernel
// ============================================================================

/**
 * @brief DFMA (Double Fused Multiply-Add) latency measurement kernel
 *
 * Creates a dependency chain of DFMA operations:
 *   result = result * a + b
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of DFMA operations in the chain
 */
__global__ void dfma_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize with non-trivial values
    double result = 1.0;
    const double a = 1.000001;
    const double b = 0.000001;

    // Read cycle counter before the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of DFMA operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX DFMA: result = result * a + b
        asm volatile (
            "fma.rn.f64 %0, %0, %1, %2;"
            : "+d"(result)
            : "d"(a), "d"(b)
        );
    }

    // Read cycle counter after the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0.0) {
        *cycles = 0;
    }
}

// ============================================================================
// DFMA Throughput Kernel
// ============================================================================

/**
 * @brief DFMA throughput measurement kernel
 *
 * Uses independent DFMA operations to measure peak throughput.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of iterations
 */
__global__ void dfma_throughput_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Multiple independent accumulators
    double r0 = 1.0, r1 = 2.0, r2 = 3.0, r3 = 4.0;
    double r4 = 5.0, r5 = 6.0, r6 = 7.0, r7 = 8.0;

    const double a = 1.000001;
    const double b = 0.000001;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Independent DFMA operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r0) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r1) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r2) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r3) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r4) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r5) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r6) : "d"(a), "d"(b));
        asm volatile ("fma.rn.f64 %0, %0, %1, %2;" : "+d"(r7) : "d"(a), "d"(b));
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Combine results
    double final_sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;

    *cycles = end_cycle - start_cycle;

    if (final_sum < 0.0) {
        *cycles = 0;
    }
}

}  // namespace microbench
}  // namespace cpm
