#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/gpu_timer.h"
#include "kernels/ptx/transpose.h"

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

struct BenchmarkStats {
    std::string name;
    float avg_ms;
    float min_ms;
    float max_ms;
    float gbps;
};

BenchmarkStats run_benchmark(
    const std::string& name,
    const std::function<cudaError_t(cudaStream_t)>& launcher,
    int warmup,
    int iters,
    double bytes_moved) {
    for (int i = 0; i < warmup; ++i) {
        cudaError_t err = launcher(nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cpm::GpuTimer timer;
        timer.start();
        cudaError_t err = launcher(nullptr);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        timer.stop();
        samples.push_back(timer.elapsed_ms());
    }

    const float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
    const float avg_ms = sum / static_cast<float>(samples.size());
    const auto minmax = std::minmax_element(samples.begin(), samples.end());
    const float gbps = static_cast<float>(bytes_moved / (static_cast<double>(avg_ms) * 1.0e6));
    return BenchmarkStats{name, avg_ms, *minmax.first, *minmax.second, gbps};
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [rows] [cols] [warmup] [iters]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int64_t rows = 4096;
    int64_t cols = 4096;
    int warmup = 20;
    int iters = 80;

    if (argc > 1) {
        rows = std::atoll(argv[1]);
    }
    if (argc > 2) {
        cols = std::atoll(argv[2]);
    }
    if (argc > 3) {
        warmup = std::atoi(argv[3]);
    }
    if (argc > 4) {
        iters = std::atoi(argv[4]);
    }
    if (argc > 5 || rows <= 0 || cols <= 0 || warmup < 0 || iters <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    const int64_t elements = rows * cols;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
    std::vector<float> host_input(static_cast<size_t>(elements));
    for (int64_t i = 0; i < elements; ++i) {
        host_input[static_cast<size_t>(i)] = static_cast<float>((i % 251) - 125) * 0.125f;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaError_t err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc input failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc output failed: " << cudaGetErrorString(err) << '\n';
        cudaFree(d_input);
        return 1;
    }
    err = cudaMemcpy(d_input, host_input.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << '\n';
        cudaFree(d_output);
        cudaFree(d_input);
        return 1;
    }

    const double bytes_moved = 2.0 * static_cast<double>(bytes);
    std::vector<BenchmarkStats> results;
    try {
        results.push_back(run_benchmark(
            "reference_cuda",
            [&](cudaStream_t) { return launch_reference_transpose(d_input, d_output, rows, cols); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_naive",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_naive(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_opt",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_opt(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_vector",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_vector(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_swizzle",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_swizzle(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_cpasync",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_cpasync(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_vswizzle",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_vswizzle(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
        results.push_back(run_benchmark(
            "ptx_swizzle16",
            [&](cudaStream_t stream) { return cpm::launch_transpose_ptx_swizzle16(d_input, d_output, rows, cols, stream); },
            warmup,
            iters,
            bytes_moved));
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << '\n';
        cudaFree(d_output);
        cudaFree(d_input);
        return 1;
    }

    std::cout << "shape=(" << rows << ", " << cols << "), warmup=" << warmup
              << ", iters=" << iters << '\n';
    std::cout << std::left << std::setw(18) << "impl"
              << std::right << std::setw(12) << "avg_ms"
              << std::setw(12) << "min_ms"
              << std::setw(12) << "max_ms"
              << std::setw(12) << "GB/s" << '\n';
    for (const auto& result : results) {
        std::cout << std::left << std::setw(18) << result.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << result.avg_ms
                  << std::setw(12) << result.min_ms
                  << std::setw(12) << result.max_ms
                  << std::setw(12) << std::setprecision(2) << result.gbps << '\n';
    }

    cudaFree(d_output);
    cudaFree(d_input);
    return 0;
}
