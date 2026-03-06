#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

__device__ __forceinline__ int swizzle_index(int row, int col) {
    const int swizzled_col = col ^ row;
    return row * kTileDim + swizzled_col;
}

__global__ void transpose_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) {
        return;
    }

    const int64_t in_idx = y * cols + x;
    const int64_t out_idx = x * rows + y;
    float val;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(input + in_idx));
    asm volatile("st.global.f32 [%0], %1;" : : "l"(output + out_idx), "f"(val) : "memory");
}

__global__ void transpose_swizzle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    __shared__ float tile[kTileDim * kTileDim];

    int64_t x = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.x;
    int64_t y = static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row_in_tile = threadIdx.y + j;
        if (x < cols && (y + j) < rows) {
            const int64_t in_idx = (y + j) * cols + x;
            float val;
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(input + in_idx));
            tile[swizzle_index(row_in_tile, threadIdx.x)] = val;
        }
    }

    __syncthreads();

    x = static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.x;
    y = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int col_in_tile = threadIdx.y + j;
        if (x < rows && (y + j) < cols) {
            const float val = tile[swizzle_index(threadIdx.x, col_in_tile)];
            const int64_t out_idx = (y + j) * rows + x;
            asm volatile("st.global.f32 [%0], %1;" : : "l"(output + out_idx), "f"(val) : "memory");
        }
    }
}

}  // namespace

extern "C" cudaError_t launch_transpose_ptx_naive(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((cols + block.x - 1) / block.x),
        static_cast<unsigned int>((rows + block.y - 1) / block.y));
    transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_transpose_ptx_opt(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    dim3 block(kTileDim, kBlockRows);
    dim3 grid(
        static_cast<unsigned int>((cols + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((rows + kTileDim - 1) / kTileDim));
    transpose_swizzle_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}
