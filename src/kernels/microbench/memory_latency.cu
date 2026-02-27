/**
 * @file memory_latency.cu
 * @brief Memory latency microbenchmark kernels using PTX inline assembly
 *
 * Based on Volta microbenchmark methodology:
 * Uses pointer chasing with dependency chains to measure memory latency.
 * Each load depends on the previous, preventing instruction-level parallelism.
 */

#include <cuda_runtime.h>
#include <stdint.h>

namespace cpm {
namespace microbench {

/**
 * @brief Memory latency measurement kernel using pointer chasing
 *
 * Creates a dependency chain where each memory access depends on the previous.
 * This prevents the GPU from hiding latency through parallel execution.
 *
 * @param pointers Array of pointers forming a circular linked list
 * @param iterations Number of times to traverse the chain
 * @param cycles Output: total cycles elapsed
 */
__global__ void memory_latency_kernel(float** pointers, int iterations,
                                      uint64_t* cycles) {
    uint64_t start_cycle, end_cycle;

    // Initialize pointer chase
    float* current = pointers[0];
    float sum = 0.0f;

    // Read cycle counter before the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(start_cycle));

    // Pointer chasing with dependency chain
    // Each load depends on the previous (no ILP possible)
    #pragma unroll 1  // Prevent unrolling to maintain dependency chain
    for (int i = 0; i < iterations; ++i) {
        // Load next pointer from current location
        // This creates a true dependency: we can't proceed until load completes
        float* next;
        asm volatile (
            "ld.global.u64 %0, [%1];"
            : "=l"(next)
            : "l"(current)
        );

        // Also load the float value (to make it a realistic memory access)
        float value;
        asm volatile (
            "ld.global.f32 %0, [%1];"
            : "=f"(value)
            : "l"(current)
        );

        sum += value;
        current = next;

        // Prevent null pointer dereference
        if (current == nullptr) {
            current = pointers[0];
        }
    }

    // Read cycle counter after the dependency chain
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(end_cycle));

    // Store result
    *cycles = end_cycle - start_cycle;

    // Use sum to prevent compiler optimization
    // NaN check that always evaluates to false for valid data
    if (sum != sum) {
        *cycles = 0;
    }
}

}  // namespace microbench
}  // namespace cpm
