#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

#include "fixtures/gpu_test_fixture.h"
#include "kernels/ptx/transpose.h"

namespace cpm {
namespace {

__global__ void reference_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols) {
    const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

__global__ void count_mismatches_kernel(
    const float* lhs,
    const float* rhs,
    int64_t n,
    int* mismatches) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n && lhs[idx] != rhs[idx]) {
        atomicAdd(mismatches, 1);
    }
}

cudaError_t launch_reference_transpose(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols) {
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((cols + block.x - 1) / block.x),
        static_cast<unsigned int>((rows + block.y - 1) / block.y));
    reference_transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    return cudaGetLastError();
}

int count_mismatches(const float* lhs, const float* rhs, int64_t n) {
    int* d_mismatches = nullptr;
    cudaError_t err = cudaMalloc(&d_mismatches, sizeof(int));
    if (err != cudaSuccess) {
        return -1;
    }

    err = cudaMemset(d_mismatches, 0, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_mismatches);
        return -1;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    count_mismatches_kernel<<<blocks, threads>>>(lhs, rhs, n, d_mismatches);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_mismatches);
        return -1;
    }

    int host_mismatches = 0;
    err = cudaMemcpy(&host_mismatches, d_mismatches, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_mismatches);
    if (err != cudaSuccess) {
        return -1;
    }

    return host_mismatches;
}

}  // namespace

class PtxTransposeTest : public GpuTestFixture {};

using TransposeLauncher = cudaError_t (*)(const float*, float*, int64_t, int64_t, cudaStream_t);

void expect_matches_reference(
    TransposeLauncher launcher,
    int64_t rows,
    int64_t cols,
    int pattern_mod) {
    const int64_t elements = rows * cols;
    std::vector<float> host_input(static_cast<size_t>(elements));
    for (int64_t i = 0; i < elements; ++i) {
        host_input[static_cast<size_t>(i)] =
            static_cast<float>((i % pattern_mod) - (pattern_mod / 2)) * 0.125f;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_reference = nullptr;

    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_input, elements * sizeof(float)));
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_output, elements * sizeof(float)));
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_reference, elements * sizeof(float)));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        d_input, host_input.data(), elements * sizeof(float), cudaMemcpyHostToDevice));

    ASSERT_CUDA_SUCCESS(launch_reference_transpose(d_input, d_reference, rows, cols));
    ASSERT_CUDA_SUCCESS(launcher(d_input, d_output, rows, cols, nullptr));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    EXPECT_EQ(count_mismatches(d_output, d_reference, elements), 0);

    cudaFree(d_reference);
    cudaFree(d_output);
    cudaFree(d_input);
}

TEST_F(PtxTransposeTest, test_transpose_naive_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_naive, 257, 509, 97);
}

TEST_F(PtxTransposeTest, test_transpose_opt_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_opt, 3000, 5000, 251);
}

TEST_F(PtxTransposeTest, test_transpose_opt_handles_small_square_matrix) {
    expect_matches_reference(launch_transpose_ptx_opt, 32, 32, 17);
}

TEST_F(PtxTransposeTest, test_transpose_vector_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_vector, 2048, 2048, 131);
}

TEST_F(PtxTransposeTest, test_transpose_swizzle_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_swizzle, 3000, 5000, 251);
}

TEST_F(PtxTransposeTest, test_transpose_cpasync_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_cpasync, 2048, 4096, 193);
}

TEST_F(PtxTransposeTest, test_transpose_vswizzle_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_vswizzle, 2048, 4096, 193);
}

TEST_F(PtxTransposeTest, test_transpose_swizzle16_matches_reference_on_gpu) {
    expect_matches_reference(launch_transpose_ptx_swizzle16, 3000, 5000, 251);
}

}  // namespace cpm
