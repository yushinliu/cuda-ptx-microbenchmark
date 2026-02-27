#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace cpm {

// PTX memory instruction kernels

/**
 * @brief LDG - Load Global memory using PTX
 * @param data Source address in global memory
 * @param result Destination for loaded value
 */
__global__ void ldg_kernel(const float* data, float* result);

/**
 * @brief LDS - Load Shared memory using PTX
 * @param result Destination for loaded value
 * @note Shared memory is initialized in kernel
 */
__global__ void lds_kernel(float* result);

/**
 * @brief STG - Store Global memory using PTX
 * @param data Destination address in global memory
 * @param value Value to store
 */
__global__ void stg_kernel(float* data, float value);

/**
 * @brief LDG.CA - Load Global with cache-all
 * @param data Source array
 * @param result Output sum
 * @param n Number of elements
 */
__global__ void ldg_ca_kernel(const float* data, float* result, size_t n);

/**
 * @brief LDG.CS - Load Global with cache-streaming
 * @param data Source array
 * @param result Output sum
 * @param n Number of elements
 */
__global__ void ldg_cs_kernel(const float* data, float* result, size_t n);

}  // namespace cpm
