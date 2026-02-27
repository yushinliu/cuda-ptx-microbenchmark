#include "kernels/ptx/arithmetic.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cpm {

// FMA (Fused Multiply-Add) kernel using inline PTX
// Computes: result = a * b + c
__global__ void fma_kernel(float a, float b, float c, float* result) {
    float r;
    // PTX: fma.rn.f32 - round-to-nearest, single precision FMA
    asm volatile ("fma.rn.f32 %0, %1, %2, %3;"
                  : "=f"(r)
                  : "f"(a), "f"(b), "f"(c));
    *result = r;
}

// FMA latency test - chains dependent FMAs to measure latency
__global__ void fma_latency_test_kernel(float* data, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = data[idx];
    float b = 1.0001f;
    float c = 0.0001f;

    // Unroll disabled to prevent compiler optimization of the chain
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // Each FMA depends on the previous result (a is both input and output)
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;"
                      : "+f"(a)
                      : "f"(b), "f"(c));
    }

    data[idx] = a;
}

// FMA throughput test - independent FMAs for throughput measurement
__global__ void fma_throughput_test_kernel(float* data, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use multiple registers to create independent instruction streams
    float r0 = data[idx];
    float r1 = data[idx + 1];
    float r2 = data[idx + 2];
    float r3 = data[idx + 3];
    float b = 1.0001f;
    float c = 0.0001f;

    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        // Independent FMAs - can be executed in parallel
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;"
                      : "+f"(r0)
                      : "f"(b), "f"(c));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;"
                      : "+f"(r1)
                      : "f"(b), "f"(c));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;"
                      : "+f"(r2)
                      : "f"(b), "f"(c));
        asm volatile ("fma.rn.f32 %0, %0, %1, %2;"
                      : "+f"(r3)
                      : "f"(b), "f"(c));
    }

    // Combine results
    data[idx] = r0 + r1 + r2 + r3;
}

// ADD.F32 using inline PTX
__global__ void add_f32_kernel(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r;
        asm volatile ("add.f32 %0, %1, %2;"
                      : "=f"(r)
                      : "f"(a[idx]), "f"(b[idx]));
        result[idx] = r;
    }
}

// MUL.F32 using inline PTX
__global__ void mul_f32_kernel(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r;
        asm volatile ("mul.f32 %0, %1, %2;"
                      : "=f"(r)
                      : "f"(a[idx]), "f"(b[idx]));
        result[idx] = r;
    }
}

}  // namespace cpm
