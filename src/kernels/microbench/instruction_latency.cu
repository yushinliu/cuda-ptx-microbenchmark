/**
 * @file instruction_latency.cu
 * @brief Instruction latency microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * Uses dependency chains to measure instruction latency.
 * Each operation depends on the previous, preventing parallel execution.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

// ============================================================================
// FMA Latency Kernel
// ============================================================================

/**
 * @brief FMA (Fused Multiply-Add) latency measurement kernel
 *
 * Creates a dependency chain of FMA operations:
 *   result = result * a + b
 *
 * Each FMA depends on the previous result, so no ILP is possible.
 * This measures the raw latency of the FMA instruction.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of FMA operations in the chain
 */
__global__ void fma_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize with non-trivial values to prevent optimization
    float result = 1.0f;
    const float a = 1.000001f;  // Slightly more than 1
    const float b = 0.000001f;

    // Read cycle counter before the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of FMA operations
    // Each iteration depends on the previous result
    #pragma unroll 1  // Prevent unrolling to maintain dependency
    for (int i = 0; i < iterations; ++i) {
        // PTX FMA: result = result * a + b
        asm volatile (
            "fma.rn.f32 %0, %0, %1, %2;"
            : "+f"(result)
            : "f"(a), "f"(b)
        );
    }

    // Read cycle counter after the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent compiler from optimizing away the loop
    // Use result in a way that can't be computed at compile time
    if (result < 0.0f) {
        *cycles = 0;
    }
}

// ============================================================================
// ADD Latency Kernel
// ============================================================================

/**
 * @brief ADD instruction latency measurement kernel
 *
 * Creates a dependency chain of ADD operations.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of ADD operations in the chain
 */
__global__ void add_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize
    float result = 0.0f;
    const float increment = 1.0f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of ADD operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX ADD: result = result + increment
        asm volatile (
            "add.f32 %0, %0, %1;"
            : "+f"(result)
            : "f"(increment)
        );
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0.0f) {
        *cycles = 0;
    }
}

// ============================================================================
// MUL Latency Kernel
// ============================================================================

/**
 * @brief MUL instruction latency measurement kernel
 *
 * Creates a dependency chain of MUL operations.
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of MUL operations in the chain
 */
__global__ void mul_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize with value close to 1 to prevent overflow/underflow
    float result = 1.000001f;
    const float multiplier = 1.000001f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of MUL operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX MUL: result = result * multiplier
        asm volatile (
            "mul.f32 %0, %0, %1;"
            : "+f"(result)
            : "f"(multiplier)
        );
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0.0f) {
        *cycles = 0;
    }
}

// ============================================================================
// SFU Instruction Latency Kernels
// ============================================================================

/**
 * @brief SQRT (Square Root) instruction latency measurement kernel
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of SQRT operations in the chain
 */
__global__ void sqrt_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize
    float result = 2.0f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of SQRT operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX SQRT: result = sqrt(result)
        asm volatile (
            "sqrt.approx.f32 %0, %0;"
            : "+f"(result)
        );
        // Add small value to prevent convergence to 1
        result += 1.0f;
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < 0.0f) {
        *cycles = 0;
    }
}

/**
 * @brief SIN instruction latency measurement kernel
 *
 * @param cycles Output: total cycles elapsed
 * @param iterations Number of SIN operations in the chain
 */
__global__ void sin_latency_kernel(uint64_t* cycles, int iterations) {
    uint64_t start_cycle, end_cycle;

    // Initialize
    float result = 0.1f;

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Dependency chain of SIN operations
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // PTX SIN: result = sin(result)
        asm volatile (
            "sin.approx.f32 %0, %0;"
            : "+f"(result)
        );
        // Add small increment to keep changing input
        result += 0.01f;
    }

    // Read cycle counter
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Prevent optimization
    if (result < -1.0f || result > 1.0f) {
        *cycles = 0;
    }
}

}  // namespace microbench
}  // namespace cpm
