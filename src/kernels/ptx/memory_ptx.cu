#include "kernels/ptx/memory_ptx.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cpm {

// LDG - Load Global using PTX
// Uses ld.global.f32 to load from global memory
__global__ void ldg_kernel(const float* data, float* result) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float r;
    // PTX: ld.global.f32 - load from global memory
    asm volatile ("ld.global.f32 %0, [%1];"
                  : "=f"(r)
                  : "l"(data + idx));
    *result = r;
}

// LDS - Load Shared using PTX
// Initializes shared memory, then loads from it using PTX
__global__ void lds_kernel(float* result) {
    __shared__ float shared_data[256];

    int tid = threadIdx.x;

    // Initialize shared memory
    shared_data[tid] = 42.0f;

    // Synchronize to ensure all writes complete
    __syncthreads();

    // Load from shared memory using PTX
    float r;
    asm volatile ("ld.shared.f32 %0, [%1];"
                  : "=f"(r)
                  : "l"(&shared_data[tid]));

    // Only thread 0 writes result
    if (tid == 0) {
        *result = r;
    }
}

// STG - Store Global using PTX
// Uses st.global.f32 to store to global memory
__global__ void stg_kernel(float* data, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // PTX: st.global.f32 - store to global memory
    // Added "memory" clobber to prevent compiler reordering
    asm volatile ("st.global.f32 [%0], %1;"
                  :
                  : "l"(data + idx), "f"(value)
                  : "memory");
}

// LDG.CA - Load Global with cache-all
// Uses ld.global.ca.f32 to load with L1 caching
__global__ void ldg_ca_kernel(const float* data, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        float r;
        // PTX: ld.global.ca.f32 - cache at all levels (L1/L2)
        asm volatile ("ld.global.ca.f32 %0, [%1];"
                      : "=f"(r)
                      : "l"(data + i));
        sum += r;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, sum);
    }
}

// LDG.CS - Load Global with cache-streaming
// Uses ld.global.cs.f32 to load with streaming (no L1 cache)
__global__ void ldg_cs_kernel(const float* data, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    #pragma unroll 4
    for (size_t i = idx; i < n; i += stride) {
        float r;
        // PTX: ld.global.cs.f32 - cache streaming (L2 only)
        asm volatile ("ld.global.cs.f32 %0, [%1];"
                      : "=f"(r)
                      : "l"(data + i));
        sum += r;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(result, sum);
    }
}

}  // namespace cpm
