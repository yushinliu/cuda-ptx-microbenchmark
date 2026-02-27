#include <iostream>
#include <cuda_runtime.h>

#include "core/gpu_timer.h"
#include "core/benchmark_runner.h"
#include "core/result_collector.h"

void print_device_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    std::cout << "CUDA Devices: " << device_count << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        std::cout << "\nDevice " << i << ": " << props.name << std::endl;
        std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global Memory: " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  L2 Cache Size: " << props.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Memory Bus Width: " << props.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Memory Clock Rate: " << props.memoryClockRate / 1000 << " MHz" << std::endl;
    }
}

int main(int /* argc */, char** /* argv */) {
    std::cout << "CUDA+PTX Microbenchmark" << std::endl;
    std::cout << "=======================" << std::endl;

    print_device_info();

    // TODO: Run benchmark suite based on command line arguments

    return 0;
}
