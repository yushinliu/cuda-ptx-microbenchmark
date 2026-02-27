#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace cpm {

// PTX synchronization instruction kernels

/**
 * @brief BAR.SYNC - Barrier synchronization using PTX
 * @param data Shared memory data for testing
 * @param iterations Number of barrier iterations
 */
__global__ void bar_sync_test_kernel(float* data, int iterations);

/**
 * @brief MEMBAR.GL - Memory barrier for global memory
 * @param flag Flag for synchronization
 * @param result Result of synchronized read
 */
__global__ void membar_test_kernel(int* flag, int* result);

/**
 * @brief ATOM.ADD - Atomic add using PTX
 * @param counter Counter to increment
 * @param n Number of increments per thread
 */
__global__ void atom_add_test_kernel(int* counter, int n);

}  // namespace cpm
