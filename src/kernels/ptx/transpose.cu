#include "kernels/ptx/transpose.h"

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;
constexpr int kBlockRows16 = 16;
constexpr int kVecWidth = 4;
constexpr int kVecThreadsX = kTileDim / kVecWidth;

__device__ __forceinline__ bool is_vec4_aligned(int64_t element_index) {
    return (element_index & (kVecWidth - 1)) == 0;
}

__device__ __forceinline__ float4 load_vec4_ptx(const float* ptr) {
    float4 vec;
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w)
        : "l"(ptr));
    return vec;
}

__device__ __forceinline__ void store_vec4_ptx(float* ptr, const float4& vec) {
    asm volatile(
        "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(ptr), "f"(vec.x), "f"(vec.y), "f"(vec.z), "f"(vec.w)
        : "memory");
}

__device__ __forceinline__ void cp_async_vec4(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr));
#else
    const float4 vec = *reinterpret_cast<const float4*>(gmem_ptr);
    *reinterpret_cast<float4*>(smem_ptr) = vec;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

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

__global__ void transpose_vector_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    __shared__ float tile[kTileDim][kTileDim + 1];

    const int64_t tile_x = static_cast<int64_t>(blockIdx.x) * kTileDim;
    const int64_t tile_y = static_cast<int64_t>(blockIdx.y) * kTileDim;
    const int col_base = threadIdx.x * kVecWidth;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row_in_tile = threadIdx.y + j;
        const int64_t global_x = tile_x + col_base;
        const int64_t global_y = tile_y + row_in_tile;
        if (global_y < rows) {
            const int64_t base_idx = global_y * cols + global_x;
            if ((global_x + kVecWidth - 1) < cols && is_vec4_aligned(base_idx)) {
                const float4 vec = load_vec4_ptx(input + base_idx);
                tile[row_in_tile][col_base + 0] = vec.x;
                tile[row_in_tile][col_base + 1] = vec.y;
                tile[row_in_tile][col_base + 2] = vec.z;
                tile[row_in_tile][col_base + 3] = vec.w;
            } else {
                #pragma unroll
                for (int k = 0; k < kVecWidth; ++k) {
                    if ((global_x + k) < cols) {
                        const int64_t idx = global_y * cols + global_x + k;
                        float val;
                        asm volatile("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(input + idx));
                        tile[row_in_tile][col_base + k] = val;
                    }
                }
            }
        }
    }

    __syncthreads();

    const int64_t out_x = static_cast<int64_t>(blockIdx.y) * kTileDim + col_base;
    const int64_t out_y = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if ((out_y + j) < cols) {
            const int logical_col = threadIdx.y + j;
            const int64_t out_idx = (out_y + j) * rows + out_x;
            if ((out_x + kVecWidth - 1) < rows && is_vec4_aligned(out_idx)) {
                const float4 vec = make_float4(
                    tile[col_base + 0][logical_col],
                    tile[col_base + 1][logical_col],
                    tile[col_base + 2][logical_col],
                    tile[col_base + 3][logical_col]);
                store_vec4_ptx(output + out_idx, vec);
            } else {
                #pragma unroll
                for (int k = 0; k < kVecWidth; ++k) {
                    if ((out_x + k) < rows) {
                        const float val = tile[col_base + k][logical_col];
                        asm volatile(
                            "st.global.f32 [%0], %1;"
                            :
                            : "l"(output + out_idx + k), "f"(val)
                            : "memory");
                    }
                }
            }
        }
    }
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
        if (x < rows && (y + j) < cols) {
            const int row_in_tile = threadIdx.x;
            const int col_in_tile = threadIdx.y + j;
            const float val = tile[swizzle_index(row_in_tile, col_in_tile)];
            const int64_t out_idx = (y + j) * rows + x;
            asm volatile("st.global.f32 [%0], %1;" : : "l"(output + out_idx), "f"(val) : "memory");
        }
    }
}

__global__ void transpose_swizzle16_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    __shared__ float tile[kTileDim * kTileDim];

    int64_t x = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.x;
    int64_t y = static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows16) {
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
    for (int j = 0; j < kTileDim; j += kBlockRows16) {
        const int col_in_tile = threadIdx.y + j;
        if (x < rows && (y + j) < cols) {
            const float val = tile[swizzle_index(threadIdx.x, col_in_tile)];
            const int64_t out_idx = (y + j) * rows + x;
            asm volatile("st.global.f32 [%0], %1;" : : "l"(output + out_idx), "f"(val) : "memory");
        }
    }
}

__global__ void transpose_cpasync_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    __shared__ float tile[kTileDim][kTileDim];

    const int64_t tile_x = static_cast<int64_t>(blockIdx.x) * kTileDim;
    const int64_t tile_y = static_cast<int64_t>(blockIdx.y) * kTileDim;
    const int col_base = threadIdx.x * kVecWidth;
    bool issued_async = false;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row_in_tile = threadIdx.y + j;
        const int64_t global_x = tile_x + col_base;
        const int64_t global_y = tile_y + row_in_tile;
        if (global_y < rows) {
            const int64_t base_idx = global_y * cols + global_x;
            if ((global_x + kVecWidth - 1) < cols && is_vec4_aligned(base_idx)) {
                cp_async_vec4(&tile[row_in_tile][col_base], input + base_idx);
                issued_async = true;
            } else {
                #pragma unroll
                for (int k = 0; k < kVecWidth; ++k) {
                    if ((global_x + k) < cols) {
                        tile[row_in_tile][col_base + k] = input[base_idx + k];
                    }
                }
            }
        }
    }

    if (issued_async) {
        cp_async_commit();
        cp_async_wait();
    }
    __syncthreads();

    const int64_t out_x = static_cast<int64_t>(blockIdx.y) * kTileDim + col_base;
    const int64_t out_y = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if ((out_y + j) < cols) {
            const int logical_col = threadIdx.y + j;
            const int64_t out_idx = (out_y + j) * rows + out_x;
            if ((out_x + kVecWidth - 1) < rows && is_vec4_aligned(out_idx)) {
                const float4 vec = make_float4(
                    tile[col_base + 0][logical_col],
                    tile[col_base + 1][logical_col],
                    tile[col_base + 2][logical_col],
                    tile[col_base + 3][logical_col]);
                store_vec4_ptx(output + out_idx, vec);
            } else {
                #pragma unroll
                for (int k = 0; k < kVecWidth; ++k) {
                    if ((out_x + k) < rows) {
                        output[out_idx + k] = tile[col_base + k][logical_col];
                    }
                }
            }
        }
    }
}

__global__ void transpose_vswizzle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    __shared__ float tile[kTileDim * kTileDim];

    const int64_t tile_x = static_cast<int64_t>(blockIdx.x) * kTileDim;
    const int64_t tile_y = static_cast<int64_t>(blockIdx.y) * kTileDim;
    const int col_base = threadIdx.x * kVecWidth;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        const int row_in_tile = threadIdx.y + j;
        const int64_t global_x = tile_x + col_base;
        const int64_t global_y = tile_y + row_in_tile;
        if (global_y < rows) {
            const int64_t base_idx = global_y * cols + global_x;
            if ((global_x + kVecWidth - 1) < cols && is_vec4_aligned(base_idx)) {
                const float4 vec = load_vec4_ptx(input + base_idx);
                tile[swizzle_index(row_in_tile, col_base + 0)] = vec.x;
                tile[swizzle_index(row_in_tile, col_base + 1)] = vec.y;
                tile[swizzle_index(row_in_tile, col_base + 2)] = vec.z;
                tile[swizzle_index(row_in_tile, col_base + 3)] = vec.w;
            } else {
                #pragma unroll
                for (int k = 0; k < kVecWidth; ++k) {
                    if ((global_x + k) < cols) {
                        const int64_t idx = global_y * cols + global_x + k;
                        float val;
                        asm volatile("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(input + idx));
                        tile[swizzle_index(row_in_tile, col_base + k)] = val;
                    }
                }
            }
        }
    }

    __syncthreads();

    const int64_t out_x = static_cast<int64_t>(blockIdx.y) * kTileDim + col_base;
    const int64_t out_y = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < kTileDim; j += kBlockRows) {
        if ((out_y + j) < cols) {
            const int logical_col = threadIdx.y + j;
            const int64_t out_idx = (out_y + j) * rows + out_x;
            if ((out_x + kVecWidth - 1) < rows && is_vec4_aligned(out_idx)) {
                const float4 vec = make_float4(
                    tile[swizzle_index(col_base + 0, logical_col)],
                    tile[swizzle_index(col_base + 1, logical_col)],
                    tile[swizzle_index(col_base + 2, logical_col)],
                    tile[swizzle_index(col_base + 3, logical_col)]);
                store_vec4_ptx(output + out_idx, vec);
            } else {
                #pragma unroll
                for (int k = 0; k < kVecWidth; ++k) {
                    if ((out_x + k) < rows) {
                        const float val = tile[swizzle_index(col_base + k, logical_col)];
                        asm volatile(
                            "st.global.f32 [%0], %1;"
                            :
                            : "l"(output + out_idx + k), "f"(val)
                            : "memory");
                    }
                }
            }
        }
    }
}

}  // namespace

namespace cpm {

cudaError_t launch_transpose_ptx_naive(
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

cudaError_t launch_transpose_ptx_opt(
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

cudaError_t launch_transpose_ptx_vector(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    dim3 block(kVecThreadsX, kBlockRows);
    dim3 grid(
        static_cast<unsigned int>((cols + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((rows + kTileDim - 1) / kTileDim));
    transpose_vector_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}

cudaError_t launch_transpose_ptx_swizzle(
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

cudaError_t launch_transpose_ptx_cpasync(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    dim3 block(kVecThreadsX, kBlockRows);
    dim3 grid(
        static_cast<unsigned int>((cols + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((rows + kTileDim - 1) / kTileDim));
    transpose_cpasync_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}

cudaError_t launch_transpose_ptx_vswizzle(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    dim3 block(kVecThreadsX, kBlockRows);
    dim3 grid(
        static_cast<unsigned int>((cols + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((rows + kTileDim - 1) / kTileDim));
    transpose_vswizzle_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}

cudaError_t launch_transpose_ptx_swizzle16(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream) {
    dim3 block(kTileDim, kBlockRows16);
    dim3 grid(
        static_cast<unsigned int>((cols + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((rows + kTileDim - 1) / kTileDim));
    transpose_swizzle16_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}

}  // namespace cpm
