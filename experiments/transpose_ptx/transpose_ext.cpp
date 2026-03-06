#include <torch/extension.h>

using cudaStream_t = void*;
constexpr int kCudaSuccess = 0;

extern "C" int launch_transpose_ptx_naive(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream);
extern "C" int launch_transpose_ptx_opt(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream);

namespace {

void check_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "only float32 is supported");
}

torch::Tensor run_launcher(
    torch::Tensor input,
    int (*launcher)(const float*, float*, int64_t, int64_t, cudaStream_t)) {
    check_input(input);
    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    auto output = torch::empty({cols, rows}, input.options());
    int err = launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        nullptr);
    TORCH_CHECK(err == kCudaSuccess, "CUDA kernel launch failed with code ", err);
    return output;
}

}  // namespace

torch::Tensor transpose_ptx_naive(torch::Tensor input) {
    return run_launcher(input, launch_transpose_ptx_naive);
}

torch::Tensor transpose_ptx_opt(torch::Tensor input) {
    return run_launcher(input, launch_transpose_ptx_opt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_ptx_naive", &transpose_ptx_naive, "PTX transpose naive (CUDA)");
    m.def("transpose_ptx_opt", &transpose_ptx_opt, "PTX transpose optimized (CUDA)");
}
