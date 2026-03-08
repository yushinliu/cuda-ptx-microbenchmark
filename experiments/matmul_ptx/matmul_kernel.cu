#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor_op_multiplicand_sm75.h>
#include <cutlass/tensor_ref.h>
#include <mma.h>

#include <cstdint>

namespace {

using namespace nvcuda;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kBlockM = 64;
constexpr int kBlockN = 128;
constexpr int kBlockK = 16;
constexpr int kWarpRows = 2;
constexpr int kWarpCols = 4;
constexpr int kWarpsPerBlock = kWarpRows * kWarpCols;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kStages = 3;
constexpr int kABytesPerStage = kBlockM * kBlockK * static_cast<int>(sizeof(half));
constexpr int kBBytesPerStage = kBlockK * kBlockN * static_cast<int>(sizeof(half));
constexpr int kVecBytes = 16;
constexpr int kAChunks = kABytesPerStage / kVecBytes;
constexpr int kBChunks = kBBytesPerStage / kVecBytes;
constexpr int kAChunksPerRow = (kBlockK * static_cast<int>(sizeof(half))) / kVecBytes;
constexpr int kBChunksPerRow = (kBlockN * static_cast<int>(sizeof(half))) / kVecBytes;

constexpr int kSmallBlockM = 64;
constexpr int kSmallBlockN = 64;
constexpr int kSmallBlockK = 16;
constexpr int kSmallWarpRows = 2;
constexpr int kSmallWarpCols = 2;
constexpr int kSmallWarpsPerBlock = kSmallWarpRows * kSmallWarpCols;
constexpr int kSmallThreadsPerBlock = kSmallWarpsPerBlock * 32;
constexpr int kSmallStages = 2;
constexpr int kSmallABytesPerStage = kSmallBlockM * kSmallBlockK * static_cast<int>(sizeof(half));
constexpr int kSmallBBytesPerStage = kSmallBlockK * kSmallBlockN * static_cast<int>(sizeof(half));
constexpr int kSmallAChunks = kSmallABytesPerStage / kVecBytes;
constexpr int kSmallBChunks = kSmallBBytesPerStage / kVecBytes;
constexpr int kSmallAChunksPerRow = (kSmallBlockK * static_cast<int>(sizeof(half))) / kVecBytes;
constexpr int kSmallBChunksPerRow = (kSmallBlockN * static_cast<int>(sizeof(half))) / kVecBytes;
constexpr int kLdmatrixTileM = 16;
constexpr int kLdmatrixTileN = 8;
constexpr int kLdmatrixTileK = 16;
constexpr int kLdmatrixThreadsPerBlock = 32;

using CutlassHalf = cutlass::half_t;
using LdmatrixWarpShape = cutlass::gemm::GemmShape<kLdmatrixTileM, kLdmatrixTileN, kLdmatrixTileK>;
using LdmatrixPolicy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
        LdmatrixWarpShape,
        32,
        CutlassHalf,
        cutlass::layout::RowMajor,
        CutlassHalf,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>,
    cutlass::MatrixShape<1, 1>>;
using LdmatrixSmemLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
    cutlass::sizeof_bits<CutlassHalf>::value,
    kLdmatrixTileK>;
using LdmatrixSmemLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
    cutlass::sizeof_bits<CutlassHalf>::value,
    kLdmatrixTileK>;
using LdmatrixWarpMma = cutlass::gemm::warp::MmaTensorOp<
    LdmatrixWarpShape,
    CutlassHalf,
    LdmatrixSmemLayoutA,
    CutlassHalf,
    LdmatrixSmemLayoutB,
    float,
    cutlass::layout::RowMajor,
    LdmatrixPolicy>;

using AccumulatorFragment = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
using MatrixAFragment = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;
using MatrixBFragment = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;

static_assert(MatrixAFragment::num_elements == 16, "Unexpected matrix_a fragment size");
static_assert(MatrixBFragment::num_elements == 16, "Unexpected matrix_b fragment size");
static_assert(AccumulatorFragment::num_elements == 8, "Unexpected accumulator fragment size");

template <typename T>
__device__ __forceinline__ T* ptr_add_bytes(T* ptr, int bytes) {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + bytes);
}

__device__ __forceinline__ void mma_sync_16x16(
    AccumulatorFragment& acc,
    const MatrixAFragment& a_frag,
    const MatrixBFragment& b_frag) {
    wmma::mma_sync(acc, a_frag, b_frag, acc);
}

__device__ __forceinline__ void cp_async_copy_16(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr));
#else
    *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" : : "n"(PendingGroups));
#endif
}

__device__ __forceinline__ void zero_vec4(void* dst) {
    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

__global__ void matmul_mma_ldmatrix_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int64_t m,
    int64_t n,
    int64_t k) {
    __shared__ alignas(16) CutlassHalf smem_a[kLdmatrixTileM * kLdmatrixTileK];
    __shared__ alignas(16) CutlassHalf smem_b[kLdmatrixTileK * kLdmatrixTileN];

    const int lane_id = threadIdx.x & 31;
    const int tile_row = static_cast<int>(blockIdx.y) * kLdmatrixTileM;
    const int tile_col = static_cast<int>(blockIdx.x) * kLdmatrixTileN;

    using FragmentA = typename LdmatrixWarpMma::FragmentA;
    using FragmentB = typename LdmatrixWarpMma::FragmentB;
    using FragmentC = typename LdmatrixWarpMma::FragmentC;
    using TransformedFragmentA = typename LdmatrixWarpMma::TransformedFragmentA;
    using TransformedFragmentB = typename LdmatrixWarpMma::TransformedFragmentB;
    using IteratorA = typename LdmatrixWarpMma::IteratorA;
    using IteratorB = typename LdmatrixWarpMma::IteratorB;
    using IteratorC = typename LdmatrixWarpMma::IteratorC;
    using TensorRefA = typename IteratorA::TensorRef;
    using TensorRefB = typename IteratorB::TensorRef;
    using TensorRefC = typename IteratorC::TensorRef;

    FragmentC accum;
    #pragma unroll
    for (int i = 0; i < FragmentC::kElements; ++i) {
        accum[i] = 0.0f;
    }

    const auto smem_layout_a = LdmatrixSmemLayoutA::packed({kLdmatrixTileM, kLdmatrixTileK});
    const auto smem_layout_b = LdmatrixSmemLayoutB::packed({kLdmatrixTileK, kLdmatrixTileN});
    TensorRefA ref_a(smem_a, smem_layout_a);
    TensorRefB ref_b(smem_b, smem_layout_b);
    LdmatrixWarpMma mma;

    const CutlassHalf* a_cutlass = reinterpret_cast<const CutlassHalf*>(a);
    const CutlassHalf* b_cutlass = reinterpret_cast<const CutlassHalf*>(b);

    for (int k0 = 0; k0 < k; k0 += kLdmatrixTileK) {
        for (int idx = lane_id; idx < kLdmatrixTileM * kLdmatrixTileK; idx += blockDim.x) {
            const int row = idx / kLdmatrixTileK;
            const int col = idx % kLdmatrixTileK;
            ref_a.at({row, col}) = a_cutlass[(tile_row + row) * k + (k0 + col)];
        }

        for (int idx = lane_id; idx < kLdmatrixTileK * kLdmatrixTileN; idx += blockDim.x) {
            const int row = idx / kLdmatrixTileN;
            const int col = idx % kLdmatrixTileN;
            ref_b.at({row, col}) = b_cutlass[(k0 + row) * n + (tile_col + col)];
        }

        __syncthreads();

        FragmentA frag_a;
        FragmentB frag_b;
        TransformedFragmentA frag_a_mma;
        TransformedFragmentB frag_b_mma;
        IteratorA iter_a(ref_a, lane_id);
        IteratorB iter_b(ref_b, lane_id);
        iter_a.load(frag_a);
        iter_b.load(frag_b);
        mma.transform(frag_a_mma, frag_b_mma, frag_a, frag_b);
        mma(accum, frag_a_mma, frag_b_mma, accum);

        __syncthreads();
    }

    TensorRefC ref_c(c + tile_row * n + tile_col, n);
    IteratorC iter_c(ref_c, lane_id);
    iter_c.store(accum);
}

__global__ void matmul_mma_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int64_t m,
    int64_t n,
    int64_t k) {
    __shared__ half smem_a[kBlockM * kBlockK];
    __shared__ half smem_b[kBlockK * kBlockN];

    const int warp_id = threadIdx.x >> 5;
    const int warp_m = warp_id / kWarpCols;
    const int warp_n = warp_id % kWarpCols;

    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;

    AccumulatorFragment acc00;
    AccumulatorFragment acc01;
    AccumulatorFragment acc10;
    AccumulatorFragment acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    for (int k0 = 0; k0 < k; k0 += kBlockK) {
        for (int idx = threadIdx.x; idx < kBlockM * kBlockK; idx += blockDim.x) {
            const int row = idx / kBlockK;
            const int col = idx % kBlockK;
            const int global_row = block_row + row;
            const int global_col = k0 + col;
            if (global_row < m && global_col < k) {
                smem_a[idx] = a[global_row * k + global_col];
            } else {
                smem_a[idx] = __float2half(0.0f);
            }
        }

        for (int idx = threadIdx.x; idx < kBlockK * kBlockN; idx += blockDim.x) {
            const int row = idx / kBlockN;
            const int col = idx % kBlockN;
            const int global_row = k0 + row;
            const int global_col = block_col + col;
            if (global_row < k && global_col < n) {
                smem_b[idx] = b[global_row * n + global_col];
            } else {
                smem_b[idx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        const int row_base = warp_m * (2 * kWmmaM);
        const int col_base = warp_n * (2 * kWmmaN);
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += kWmmaK) {
            MatrixAFragment a_frag0;
            MatrixAFragment a_frag1;
            MatrixBFragment b_frag0;
            MatrixBFragment b_frag1;
            const half* a_tile0 = &smem_a[(row_base + 0) * kBlockK + kk];
            const half* a_tile1 = &smem_a[(row_base + kWmmaM) * kBlockK + kk];
            const half* b_tile0 = &smem_b[kk * kBlockN + col_base + 0];
            const half* b_tile1 = &smem_b[kk * kBlockN + col_base + kWmmaN];
            wmma::load_matrix_sync(a_frag0, a_tile0, kBlockK);
            wmma::load_matrix_sync(a_frag1, a_tile1, kBlockK);
            wmma::load_matrix_sync(b_frag0, b_tile0, kBlockN);
            wmma::load_matrix_sync(b_frag1, b_tile1, kBlockN);
            mma_sync_16x16(acc00, a_frag0, b_frag0);
            mma_sync_16x16(acc01, a_frag0, b_frag1);
            mma_sync_16x16(acc10, a_frag1, b_frag0);
            mma_sync_16x16(acc11, a_frag1, b_frag1);
        }

        __syncthreads();
    }

    const int warp_row0 = block_row + warp_m * (2 * kWmmaM);
    const int warp_row1 = warp_row0 + kWmmaM;
    const int warp_col0 = block_col + warp_n * (2 * kWmmaN);
    const int warp_col1 = warp_col0 + kWmmaN;
    if (warp_row0 < m && warp_col0 < n) {
        wmma::store_matrix_sync(c + warp_row0 * n + warp_col0, acc00, n, wmma::mem_row_major);
    }
    if (warp_row0 < m && warp_col1 < n) {
        wmma::store_matrix_sync(c + warp_row0 * n + warp_col1, acc01, n, wmma::mem_row_major);
    }
    if (warp_row1 < m && warp_col0 < n) {
        wmma::store_matrix_sync(c + warp_row1 * n + warp_col0, acc10, n, wmma::mem_row_major);
    }
    if (warp_row1 < m && warp_col1 < n) {
        wmma::store_matrix_sync(c + warp_row1 * n + warp_col1, acc11, n, wmma::mem_row_major);
    }
}

__global__ void matmul_mma_cpasync_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int64_t m,
    int64_t n,
    int64_t k) {
    __shared__ uint4 smem_a_raw[kStages][kAChunks];
    __shared__ uint4 smem_b_raw[kStages][kBChunks];
    const int warp_id = threadIdx.x >> 5;
    const int warp_m = warp_id / kWarpCols;
    const int warp_n = warp_id % kWarpCols;

    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    const int stage_count = static_cast<int>((k + kBlockK - 1) / kBlockK);

    auto load_stage = [&](int stage_idx, int k0) {
        for (int chunk = threadIdx.x; chunk < kAChunks; chunk += blockDim.x) {
            const int row = chunk / kAChunksPerRow;
            const int col = (chunk % kAChunksPerRow) * (kVecBytes / static_cast<int>(sizeof(half)));
            void* dst = &smem_a_raw[stage_idx][chunk];
            if ((block_row + row) < m && (k0 + col) < k) {
                const half* src = a + (block_row + row) * k + k0 + col;
                cp_async_copy_16(dst, src);
            } else {
                zero_vec4(dst);
            }
        }

        for (int chunk = threadIdx.x; chunk < kBChunks; chunk += blockDim.x) {
            const int row = chunk / kBChunksPerRow;
            const int col = (chunk % kBChunksPerRow) * (kVecBytes / static_cast<int>(sizeof(half)));
            void* dst = &smem_b_raw[stage_idx][chunk];
            if ((k0 + row) < k && (block_col + col) < n) {
                const half* src = b + (k0 + row) * n + block_col + col;
                cp_async_copy_16(dst, src);
            } else {
                zero_vec4(dst);
            }
        }
        cp_async_commit();
    };

    AccumulatorFragment acc00;
    AccumulatorFragment acc01;
    AccumulatorFragment acc10;
    AccumulatorFragment acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    const int preload_count = stage_count < (kStages - 1) ? stage_count : (kStages - 1);
    for (int preload = 0; preload < preload_count; ++preload) {
        load_stage(preload, preload * kBlockK);
    }
    if (preload_count > 0) {
        cp_async_wait<0>();
        __syncthreads();
    }

    for (int stage = 0; stage < stage_count; ++stage) {
        const int curr = stage % kStages;
        const int preload_stage = stage + preload_count;
        if (preload_stage < stage_count) {
            load_stage(preload_stage % kStages, preload_stage * kBlockK);
        }

        const int row_base = warp_m * (2 * kWmmaM);
        const int col_base = warp_n * (2 * kWmmaN);
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += kWmmaK) {
            MatrixAFragment a_frag0;
            MatrixAFragment a_frag1;
            MatrixBFragment b_frag0;
            MatrixBFragment b_frag1;
            const half* a_tile0 = ptr_add_bytes(reinterpret_cast<half*>(&smem_a_raw[curr][0]), ((row_base + 0) * kBlockK + kk) * static_cast<int>(sizeof(half)));
            const half* a_tile1 = ptr_add_bytes(reinterpret_cast<half*>(&smem_a_raw[curr][0]), ((row_base + kWmmaM) * kBlockK + kk) * static_cast<int>(sizeof(half)));
            const half* b_tile0 = ptr_add_bytes(reinterpret_cast<half*>(&smem_b_raw[curr][0]), (kk * kBlockN + col_base + 0) * static_cast<int>(sizeof(half)));
            const half* b_tile1 = ptr_add_bytes(reinterpret_cast<half*>(&smem_b_raw[curr][0]), (kk * kBlockN + col_base + kWmmaN) * static_cast<int>(sizeof(half)));
            wmma::load_matrix_sync(a_frag0, a_tile0, kBlockK);
            wmma::load_matrix_sync(a_frag1, a_tile1, kBlockK);
            wmma::load_matrix_sync(b_frag0, b_tile0, kBlockN);
            wmma::load_matrix_sync(b_frag1, b_tile1, kBlockN);
            mma_sync_16x16(acc00, a_frag0, b_frag0);
            mma_sync_16x16(acc01, a_frag0, b_frag1);
            mma_sync_16x16(acc10, a_frag1, b_frag0);
            mma_sync_16x16(acc11, a_frag1, b_frag1);
        }

        if (preload_stage < stage_count) {
            cp_async_wait<0>();
            __syncthreads();
        }
    }

    const int warp_row0 = block_row + warp_m * (2 * kWmmaM);
    const int warp_row1 = warp_row0 + kWmmaM;
    const int warp_col0 = block_col + warp_n * (2 * kWmmaN);
    const int warp_col1 = warp_col0 + kWmmaN;
    if (warp_row0 < m && warp_col0 < n) {
        wmma::store_matrix_sync(c + warp_row0 * n + warp_col0, acc00, n, wmma::mem_row_major);
    }
    if (warp_row0 < m && warp_col1 < n) {
        wmma::store_matrix_sync(c + warp_row0 * n + warp_col1, acc01, n, wmma::mem_row_major);
    }
    if (warp_row1 < m && warp_col0 < n) {
        wmma::store_matrix_sync(c + warp_row1 * n + warp_col0, acc10, n, wmma::mem_row_major);
    }
    if (warp_row1 < m && warp_col1 < n) {
        wmma::store_matrix_sync(c + warp_row1 * n + warp_col1, acc11, n, wmma::mem_row_major);
    }
}

__global__ void matmul_mma_cpasync_fast_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int64_t m,
    int64_t n,
    int64_t k) {
    __shared__ uint4 smem_a_raw[kStages][kAChunks];
    __shared__ uint4 smem_b_raw[kStages][kBChunks];
    const int warp_id = threadIdx.x >> 5;
    const int warp_m = warp_id / kWarpCols;
    const int warp_n = warp_id % kWarpCols;

    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    const int stage_count = static_cast<int>(k / kBlockK);

    auto load_stage = [&](int stage_idx, int k0) {
        for (int chunk = threadIdx.x; chunk < kAChunks; chunk += blockDim.x) {
            const int row = chunk / kAChunksPerRow;
            const int col = (chunk % kAChunksPerRow) * (kVecBytes / static_cast<int>(sizeof(half)));
            void* dst = &smem_a_raw[stage_idx][chunk];
            const half* src = a + (block_row + row) * k + k0 + col;
            cp_async_copy_16(dst, src);
        }

        for (int chunk = threadIdx.x; chunk < kBChunks; chunk += blockDim.x) {
            const int row = chunk / kBChunksPerRow;
            const int col = (chunk % kBChunksPerRow) * (kVecBytes / static_cast<int>(sizeof(half)));
            void* dst = &smem_b_raw[stage_idx][chunk];
            const half* src = b + (k0 + row) * n + block_col + col;
            cp_async_copy_16(dst, src);
        }
        cp_async_commit();
    };

    AccumulatorFragment acc00;
    AccumulatorFragment acc01;
    AccumulatorFragment acc10;
    AccumulatorFragment acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    const int preload_count = stage_count < (kStages - 1) ? stage_count : (kStages - 1);
    for (int preload = 0; preload < preload_count; ++preload) {
        load_stage(preload, preload * kBlockK);
    }
    if (preload_count > 0) {
        cp_async_wait<0>();
        __syncthreads();
    }

    for (int stage = 0; stage < stage_count; ++stage) {
        const int curr = stage % kStages;
        const int preload_stage = stage + preload_count;
        if (preload_stage < stage_count) {
            load_stage(preload_stage % kStages, preload_stage * kBlockK);
        }

        const int row_base = warp_m * (2 * kWmmaM);
        const int col_base = warp_n * (2 * kWmmaN);
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += kWmmaK) {
            MatrixAFragment a_frag0;
            MatrixAFragment a_frag1;
            MatrixBFragment b_frag0;
            MatrixBFragment b_frag1;
            const half* a_tile0 = ptr_add_bytes(reinterpret_cast<half*>(&smem_a_raw[curr][0]), ((row_base + 0) * kBlockK + kk) * static_cast<int>(sizeof(half)));
            const half* a_tile1 = ptr_add_bytes(reinterpret_cast<half*>(&smem_a_raw[curr][0]), ((row_base + kWmmaM) * kBlockK + kk) * static_cast<int>(sizeof(half)));
            const half* b_tile0 = ptr_add_bytes(reinterpret_cast<half*>(&smem_b_raw[curr][0]), (kk * kBlockN + col_base + 0) * static_cast<int>(sizeof(half)));
            const half* b_tile1 = ptr_add_bytes(reinterpret_cast<half*>(&smem_b_raw[curr][0]), (kk * kBlockN + col_base + kWmmaN) * static_cast<int>(sizeof(half)));
            wmma::load_matrix_sync(a_frag0, a_tile0, kBlockK);
            wmma::load_matrix_sync(a_frag1, a_tile1, kBlockK);
            wmma::load_matrix_sync(b_frag0, b_tile0, kBlockN);
            wmma::load_matrix_sync(b_frag1, b_tile1, kBlockN);
            mma_sync_16x16(acc00, a_frag0, b_frag0);
            mma_sync_16x16(acc01, a_frag0, b_frag1);
            mma_sync_16x16(acc10, a_frag1, b_frag0);
            mma_sync_16x16(acc11, a_frag1, b_frag1);
        }

        if (preload_stage < stage_count) {
            cp_async_wait<0>();
            __syncthreads();
        }
    }

    const int warp_row0 = block_row + warp_m * (2 * kWmmaM);
    const int warp_row1 = warp_row0 + kWmmaM;
    const int warp_col0 = block_col + warp_n * (2 * kWmmaN);
    const int warp_col1 = warp_col0 + kWmmaN;
    wmma::store_matrix_sync(c + warp_row0 * n + warp_col0, acc00, n, wmma::mem_row_major);
    wmma::store_matrix_sync(c + warp_row0 * n + warp_col1, acc01, n, wmma::mem_row_major);
    wmma::store_matrix_sync(c + warp_row1 * n + warp_col0, acc10, n, wmma::mem_row_major);
    wmma::store_matrix_sync(c + warp_row1 * n + warp_col1, acc11, n, wmma::mem_row_major);
}

__global__ void matmul_mma_cpasync_small_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int64_t m,
    int64_t n,
    int64_t k) {
    __shared__ uint4 smem_a_raw[kSmallStages][kSmallAChunks];
    __shared__ uint4 smem_b_raw[kSmallStages][kSmallBChunks];
    const int warp_id = threadIdx.x >> 5;
    const int warp_m = warp_id / kSmallWarpCols;
    const int warp_n = warp_id % kSmallWarpCols;

    const int block_row = static_cast<int>(blockIdx.y) * kSmallBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kSmallBlockN;
    const int stage_count = static_cast<int>((k + kSmallBlockK - 1) / kSmallBlockK);

    auto load_stage = [&](int stage_idx, int k0) {
        for (int chunk = threadIdx.x; chunk < kSmallAChunks; chunk += blockDim.x) {
            const int row = chunk / kSmallAChunksPerRow;
            const int col = (chunk % kSmallAChunksPerRow) * (kVecBytes / static_cast<int>(sizeof(half)));
            void* dst = &smem_a_raw[stage_idx][chunk];
            if ((block_row + row) < m && (k0 + col) < k) {
                const half* src = a + (block_row + row) * k + k0 + col;
                cp_async_copy_16(dst, src);
            } else {
                zero_vec4(dst);
            }
        }

        for (int chunk = threadIdx.x; chunk < kSmallBChunks; chunk += blockDim.x) {
            const int row = chunk / kSmallBChunksPerRow;
            const int col = (chunk % kSmallBChunksPerRow) * (kVecBytes / static_cast<int>(sizeof(half)));
            void* dst = &smem_b_raw[stage_idx][chunk];
            if ((k0 + row) < k && (block_col + col) < n) {
                const half* src = b + (k0 + row) * n + block_col + col;
                cp_async_copy_16(dst, src);
            } else {
                zero_vec4(dst);
            }
        }
        cp_async_commit();
    };

    AccumulatorFragment acc00;
    AccumulatorFragment acc01;
    AccumulatorFragment acc10;
    AccumulatorFragment acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    const int preload_count = stage_count < (kSmallStages - 1) ? stage_count : (kSmallStages - 1);
    for (int preload = 0; preload < preload_count; ++preload) {
        load_stage(preload, preload * kSmallBlockK);
    }
    if (preload_count > 0) {
        cp_async_wait<0>();
        __syncthreads();
    }

    for (int stage = 0; stage < stage_count; ++stage) {
        const int curr = stage % kSmallStages;
        const int preload_stage = stage + preload_count;
        if (preload_stage < stage_count) {
            load_stage(preload_stage % kSmallStages, preload_stage * kSmallBlockK);
        }

        const int row_base = warp_m * (2 * kWmmaM);
        const int col_base = warp_n * (2 * kWmmaN);
        #pragma unroll
        for (int kk = 0; kk < kSmallBlockK; kk += kWmmaK) {
            MatrixAFragment a_frag0;
            MatrixAFragment a_frag1;
            MatrixBFragment b_frag0;
            MatrixBFragment b_frag1;
            const half* a_tile0 = ptr_add_bytes(reinterpret_cast<half*>(&smem_a_raw[curr][0]), ((row_base + 0) * kSmallBlockK + kk) * static_cast<int>(sizeof(half)));
            const half* a_tile1 = ptr_add_bytes(reinterpret_cast<half*>(&smem_a_raw[curr][0]), ((row_base + kWmmaM) * kSmallBlockK + kk) * static_cast<int>(sizeof(half)));
            const half* b_tile0 = ptr_add_bytes(reinterpret_cast<half*>(&smem_b_raw[curr][0]), (kk * kSmallBlockN + col_base + 0) * static_cast<int>(sizeof(half)));
            const half* b_tile1 = ptr_add_bytes(reinterpret_cast<half*>(&smem_b_raw[curr][0]), (kk * kSmallBlockN + col_base + kWmmaN) * static_cast<int>(sizeof(half)));
            wmma::load_matrix_sync(a_frag0, a_tile0, kSmallBlockK);
            wmma::load_matrix_sync(a_frag1, a_tile1, kSmallBlockK);
            wmma::load_matrix_sync(b_frag0, b_tile0, kSmallBlockN);
            wmma::load_matrix_sync(b_frag1, b_tile1, kSmallBlockN);
            mma_sync_16x16(acc00, a_frag0, b_frag0);
            mma_sync_16x16(acc01, a_frag0, b_frag1);
            mma_sync_16x16(acc10, a_frag1, b_frag0);
            mma_sync_16x16(acc11, a_frag1, b_frag1);
        }

        if (preload_stage < stage_count) {
            cp_async_wait<0>();
            __syncthreads();
        }
    }

    const int warp_row0 = block_row + warp_m * (2 * kWmmaM);
    const int warp_row1 = warp_row0 + kWmmaM;
    const int warp_col0 = block_col + warp_n * (2 * kWmmaN);
    const int warp_col1 = warp_col0 + kWmmaN;
    if (warp_row0 < m && warp_col0 < n) {
        wmma::store_matrix_sync(c + warp_row0 * n + warp_col0, acc00, n, wmma::mem_row_major);
    }
    if (warp_row0 < m && warp_col1 < n) {
        wmma::store_matrix_sync(c + warp_row0 * n + warp_col1, acc01, n, wmma::mem_row_major);
    }
    if (warp_row1 < m && warp_col0 < n) {
        wmma::store_matrix_sync(c + warp_row1 * n + warp_col0, acc10, n, wmma::mem_row_major);
    }
    if (warp_row1 < m && warp_col1 < n) {
        wmma::store_matrix_sync(c + warp_row1 * n + warp_col1, acc11, n, wmma::mem_row_major);
    }
}

}  // namespace

extern "C" cudaError_t launch_matmul_ptx_mma(
    const half* a,
    const half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
    dim3 block(kThreadsPerBlock);
    dim3 grid(
        static_cast<unsigned int>((n + kBlockN - 1) / kBlockN),
        static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    matmul_mma_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_matmul_ptx_mma_cpasync(
    const half* a,
    const half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
    dim3 block(kThreadsPerBlock);
    dim3 grid(
        static_cast<unsigned int>((n + kBlockN - 1) / kBlockN),
        static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    if ((m % kBlockM) == 0 && (n % kBlockN) == 0) {
        matmul_mma_cpasync_fast_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    } else {
        matmul_mma_cpasync_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t launch_matmul_ptx_mma_ldmatrix(
    const half* a,
    const half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
    dim3 block(kLdmatrixThreadsPerBlock);
    dim3 grid(
        static_cast<unsigned int>((n + kLdmatrixTileN - 1) / kLdmatrixTileN),
        static_cast<unsigned int>((m + kLdmatrixTileM - 1) / kLdmatrixTileM));
    matmul_mma_ldmatrix_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    return cudaGetLastError();
}
