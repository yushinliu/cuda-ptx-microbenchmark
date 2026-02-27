#include "kernels/ptx/synchronization.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cpm {

// BAR.SYNC - Barrier synchronization using PTX
// Tests barrier synchronization within a thread block
__global__ void bar_sync_test_kernel(float* data, int iterations) {
    __shared__ float shared[256];

    int tid = threadIdx.x;

    // Initialize shared memory with thread ID
    shared[tid] = static_cast<float>(tid);

    // Synchronize to ensure all threads have written
    __syncthreads();

    // Perform iterations with barrier synchronization
    for (int i = 0; i < iterations; ++i) {
        // PTX: bar.sync 0 - synchronize all threads in block
        asm volatile ("bar.sync 0;");

        // Read neighbor's value (with wrap-around)
        int neighbor = (tid + 1) % blockDim.x;
        float val = shared[neighbor];

        // bar.sync again before writing
        asm volatile ("bar.sync 0;");

        // Write to own position
        shared[tid] = val + 1.0f;
    }

    // Final synchronization
    __syncthreads();

    // Thread 0 writes result
    if (tid == 0) {
        *data = shared[0];
    }
}

// MEMBAR.GL - Memory barrier for global memory
// Ensures global memory operations are visible across the device
__global__ void membar_test_kernel(int* flag, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        // Set flag
        *flag = 1;

        // PTX: membar.gl - global memory barrier
        // Ensures the write to flag is visible to other threads
        asm volatile ("membar.gl;");
    }

    // Synchronize all threads
    __syncthreads();

    if (tid == 1) {
        // PTX: membar.gl before reading
        // Ensures we see the most recent value
        asm volatile ("membar.gl;");

        // Read flag
        *result = *flag;
    }
}

// ATOM.ADD - Atomic add using PTX
// Performs atomic addition on global memory
__global__ void atom_add_test_kernel(int* counter, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        int old;
        // PTX: atom.global.add.s32 - atomic add to global memory
        // Using .relaxed memory ordering for performance while maintaining atomicity
        // Added "memory" clobber to prevent compiler reordering
        asm volatile ("atom.global.add.relaxed.s32 %0, [%1], 1;"
                      : "=r"(old)
                      : "l"(counter)
                      : "memory");
    }
}

}  // namespace cpm
