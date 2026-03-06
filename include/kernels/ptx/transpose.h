#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cpm {

cudaError_t launch_transpose_ptx_naive(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_transpose_ptx_opt(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_transpose_ptx_vector(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_transpose_ptx_swizzle(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_transpose_ptx_cpasync(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_transpose_ptx_vswizzle(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

cudaError_t launch_transpose_ptx_swizzle16(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = nullptr);

}  // namespace cpm
