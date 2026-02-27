#pragma once

#include <cuda_runtime.h>

namespace cpm {

// FMA (Fused Multiply-Add) instruction kernels

/**
 * @brief Single FMA operation using inline PTX
 * @param a First operand
 * @param b Second operand
 * @param c Third operand (added)
 * @param result Output result (a * b + c)
 */
__global__ void fma_kernel(float a, float b, float c, float* result);

/**
 * @brief FMA latency test kernel - chains FMA operations to measure latency
 * @param data Input/output data array
 * @param iterations Number of FMA iterations per thread
 */
__global__ void fma_latency_test_kernel(float* data, int iterations);

/**
 * @brief FMA throughput test kernel - independent FMAs for throughput measurement
 * @param data Input/output data array
 * @param iterations Number of FMA iterations per thread
 */
__global__ void fma_throughput_test_kernel(float* data, int iterations);

// Additional arithmetic instruction kernels

/**
 * @brief ADD.F32 using inline PTX
 */
__global__ void add_f32_kernel(const float* a, const float* b, float* result, size_t n);

/**
 * @brief MUL.F32 using inline PTX
 */
__global__ void mul_f32_kernel(const float* a, const float* b, float* result, size_t n);

}  // namespace cpm
