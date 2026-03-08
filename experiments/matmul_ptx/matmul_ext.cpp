#include <torch/extension.h>

using cudaStream_t = void*;
constexpr int kCudaSuccess = 0;

extern "C" int launch_matmul_ptx_mma(
    const at::Half* a,
    const at::Half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream);
extern "C" int launch_matmul_ptx_mma_cpasync(
    const at::Half* a,
    const at::Half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream);
extern "C" int launch_matmul_ptx_mma_cpasync_k32(
    const at::Half* a,
    const at::Half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream);
extern "C" int launch_matmul_ptx_mma_ldmatrix(
    const at::Half* a,
    const at::Half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream);
extern "C" int launch_matmul_ptx_mma_ldmatrix_block(
    const at::Half* a,
    const at::Half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream);
extern "C" int launch_matmul_ptx_mma_ldmatrix_block_cpasync_a(
    const at::Half* a,
    const at::Half* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream);

namespace {

void check_input(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must be float16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must be float16");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions must match");
    TORCH_CHECK(a.size(0) % 16 == 0, "m must be divisible by 16");
    TORCH_CHECK(a.size(1) % 16 == 0, "k must be divisible by 16");
    TORCH_CHECK(b.size(1) % 16 == 0, "n must be divisible by 16");
}

torch::Tensor run_launcher(
    torch::Tensor a,
    torch::Tensor b,
    int (*launcher)(const at::Half*, const at::Half*, float*, int64_t, int64_t, int64_t, cudaStream_t)) {
    check_input(a, b);
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    auto output = torch::empty({m, n}, a.options().dtype(torch::kFloat32));
    int err = launcher(
        a.data_ptr<at::Half>(),
        b.data_ptr<at::Half>(),
        output.data_ptr<float>(),
        m,
        n,
        k,
        nullptr);
    TORCH_CHECK(err == kCudaSuccess, "CUDA kernel launch failed with code ", err);
    return output;
}

}  // namespace

torch::Tensor matmul_mma(torch::Tensor a, torch::Tensor b) {
    return run_launcher(a, b, launch_matmul_ptx_mma);
}

torch::Tensor matmul_mma_cpasync(torch::Tensor a, torch::Tensor b) {
    return run_launcher(a, b, launch_matmul_ptx_mma_cpasync);
}

torch::Tensor matmul_mma_cpasync_k32(torch::Tensor a, torch::Tensor b) {
    return run_launcher(a, b, launch_matmul_ptx_mma_cpasync_k32);
}

torch::Tensor matmul_mma_ldmatrix(torch::Tensor a, torch::Tensor b) {
    return run_launcher(a, b, launch_matmul_ptx_mma_ldmatrix);
}

torch::Tensor matmul_mma_ldmatrix_block(torch::Tensor a, torch::Tensor b) {
    return run_launcher(a, b, launch_matmul_ptx_mma_ldmatrix_block);
}

torch::Tensor matmul_mma_ldmatrix_block_cpasync_a(torch::Tensor a, torch::Tensor b) {
    return run_launcher(a, b, launch_matmul_ptx_mma_ldmatrix_block_cpasync_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_mma", &matmul_mma, "PTX MMA matmul (CUDA)");
    m.def("matmul_mma_cpasync", &matmul_mma_cpasync, "PTX MMA cp.async matmul (CUDA)");
    m.def("matmul_mma_cpasync_k32", &matmul_mma_cpasync_k32, "PTX MMA cp.async matmul with blockK=32 (CUDA)");
    m.def("matmul_mma_ldmatrix", &matmul_mma_ldmatrix, "PTX ldmatrix + mma.sync matmul (CUDA)");
    m.def("matmul_mma_ldmatrix_block", &matmul_mma_ldmatrix_block, "PTX block-level ldmatrix + mma.sync matmul (CUDA)");
    m.def("matmul_mma_ldmatrix_block_cpasync_a", &matmul_mma_ldmatrix_block_cpasync_a, "PTX block-level ldmatrix + mma.sync matmul with cp.async A staging (CUDA)");
}
