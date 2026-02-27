# CUDA PTX Microbenchmark

[![CUDA Build and Test](https://github.com/yushinliu/cuda-ptx-microbenchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/yushinliu/cuda-ptx-microbenchmark/actions/workflows/ci.yml)

A comprehensive microbenchmark suite for NVIDIA GPUs using CUDA and PTX assembly, specifically optimized for RTX 4070 (Compute Capability 8.9).

## Features

- **L1/L2 Cache Benchmarking**: Sequential and random access patterns with hit rate estimation
- **PTX Instruction Testing**: Direct assembly testing for FMA, LDG, LDS, BAR.SYNC instructions
- **Memory Bandwidth**: Global memory read/write/copy performance measurement
- **Integer Instruction Benchmarks**: IADD3, LOP3, SEL, SHFL latency and throughput (RTX 4070)
- **Double Precision Benchmarks**: DADD, DMUL, DFMA latency and throughput
- **Tensor Core Benchmarks**: HMMA (FP16), IMMA (INT8) matrix operations
- **Shared Memory Bank Conflicts**: 2-way, 4-way, 8-way, 32-way conflict measurement
- **Ada Lovelace Specific**: CP.ASYNC, LDMATRIX instruction benchmarks
- **Google Test Integration**: Full TDD workflow with 100+ tests, 80%+ coverage

## Requirements

- CUDA Toolkit 12.4+
- CMake 3.18+
- Google Test
- NVIDIA GPU with Compute Capability 8.0+ (optimized for sm_89 / RTX 4070)

## Quick Start

### WSL2 Setup

```bash
# Use conda environment CUDA
export PATH="/home/$USER/miniconda3/envs/cuda_env/bin:$PATH"
export CUDACXX=/home/$USER/miniconda3/envs/cuda_env/bin/nvcc

# Or use Triton cached CUDA
export PATH="/home/$USER/.triton/nvidia/nvcc/cuda_nvcc-linux-x86_64-12.8.93-archive/bin:$PATH"
```

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=$CUDACXX -DCMAKE_CUDA_ARCHITECTURES=89 -DENABLE_TESTING=ON
make -j$(nproc)
```

### Run Tests

```bash
# Run all tests
ctest --output-on-failure

# Run specific test suites
./tests/unit_tests
./tests/integration_tests --gtest_filter="*L1Cache*"
./tests/e2e_tests

# Run new microbenchmark tests
./tests/microbench_tests
./tests/microbench_tests --gtest_filter="*Integer*:*Double*"
./tests/microbench_tests --gtest_filter="*Tensor*:*SharedMemoryBank*"
```

### Run Benchmark

```bash
./cpm
```

## Project Structure

```
.
├── src/                           # Source code
│   ├── core/                     # Timer, result collector, benchmark runner
│   ├── kernels/memory/           # L1/L2 cache, global memory benchmarks
│   ├── kernels/ptx/              # PTX instruction kernels
│   └── kernels/microbench/       # RTX 4070 microbenchmarks
│       ├── integer_instructions.cu   # IADD3, LOP3, SEL, SHFL
│       ├── double_precision.cu       # DADD, DMUL, DFMA
│       ├── tensor_cores.cu           # HMMA, IMMA
│       ├── shared_memory_banks.cu    # Bank conflict measurement
│       ├── l2_cache.cu               # L2 latency/bandwidth
│       └── ada_specific.cu           # CP.ASYNC, LDMATRIX
├── include/                       # Header files
├── tests/                         # Google Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Kernel integration tests
│   ├── e2e/                      # End-to-end tests
│   └── microbench/               # RTX 4070 microbenchmark TDD tests
│       ├── test_integer_instructions.cu
│       ├── test_double_precision.cu
│       ├── test_tensor_cores.cu
│       ├── test_shared_memory_banks.cu
│       ├── test_l2_cache.cu
│       └── test_ada_specific.cu
├── docs/                          # Documentation
└── scripts/                      # Helper scripts
```

## PTX Quick Reference

### Memory Instructions
```cuda
// Load with cache-all (L1 + L2)
asm volatile ("ld.global.ca.f32 %0, [%1];" : "=f"(val) : "l"(ptr));

// Load with cache-streaming (L2 only)
asm volatile ("ld.global.cs.f32 %0, [%1];" : "=f"(val) : "l"(ptr));

// Store with memory barrier
asm volatile ("st.global.f32 [%0], %1;" : : "l"(ptr), "f"(val) : "memory");
```

### Synchronization
```cuda
// Barrier sync
asm volatile ("bar.sync 0;");

// Memory barrier
asm volatile ("membar.gl;" ::: "memory");

// Atomic add with relaxed ordering
asm volatile ("atom.global.add.relaxed.s32 %0, [%1], 1;"
              : "=r"(old) : "l"(ptr) : "memory");
```

## RTX 4070 (Ada Lovelace, sm_89) Benchmark Results

Based on Volta microbenchmark methodology from [arXiv:1804.06826](https://arxiv.org/abs/1804.06826):

### Instruction Latency (cycles)

| Instruction | Type | Latency | Throughput (ops/cycle/SM) | Notes |
|-------------|------|---------|---------------------------|-------|
| FMA | FP32 | ~4 | 128 | Fused multiply-add |
| ADD | FP32 | ~4 | 128 | Simple addition |
| MUL | FP32 | ~4 | 128 | Multiplication |
| IADD3 | INT32 | ~4 | 128 | Simulated with 2x ADD |
| LOP3 | INT32 | ~4 | 128 | Logic operations |
| SEL | INT32 | ~4 | 128 | Select/predication |
| SHFL | - | ~10 | 32 | Warp shuffle |
| DADD | FP64 | ~5 | 2 | Double precision |
| DMUL | FP64 | ~5 | 2 | Double precision |
| DFMA | FP64 | ~5 | 2 | Double precision FMA |

### Memory Hierarchy Latency

| Memory Level | Size | Latency (cycles) | Notes |
|--------------|------|------------------|-------|
| Shared/L1 | 128KB per SM | ~30 | Configurable partition |
| L2 Cache | 36-48MB | ~200 | GPU-wide |
| Global (cached) | - | ~300 | Hitting L2 |
| Global (uncached) | - | ~400 | Bypassing L2 |

### Ada Lovelace Specific Features

| Feature | Latency | Notes |
|---------|---------|-------|
| CP.ASYNC | ~10-50 | Async global to shared copy |
| LDMATRIX | ~20 | Matrix load for Tensor Cores |
| HMMA (FP16) | ~8 | Tensor Core matrix multiply |
| IMMA (INT8) | ~8 | Integer matrix multiply |

*Note: These are expected values based on architecture specifications. Actual measurements may vary.*

## Hardware Support

| GPU | Compute Capability | L1 Cache | L2 Cache | Status |
|-----|-------------------|----------|----------|--------|
| RTX 4070/4070 Super | 8.9 | 128 KB | 36-48 MB | ✅ Optimized |
| RTX 4080 | 8.9 | 128 KB | 64 MB | ✅ Supported |
| RTX 4090 | 8.9 | 128 KB | 72 MB | ✅ Supported |
| RTX 3090 | 8.6 | 128 KB | 24 MB | ⚠️ Test needed |

## Contributing

This project follows TDD workflow:
1. Write failing test (RED)
2. Implement minimal code (GREEN)
3. Refactor (IMPROVE)
4. Ensure 80%+ coverage

See [docs/TDD_WORKFLOW.md](docs/TDD_WORKFLOW.md) for details.

## License

MIT License - See LICENSE file

## References

- Jia, Z., Maggioni, M., Staiger, B., & Scarpazza, D. P. (2018). [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826). arXiv:1804.06826.

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) and TDD methodology.

This project uses the microbenchmarking methodology from the Volta paper to characterize RTX 4070 (Ada Lovelace) GPU architecture.
