# CUDA PTX Microbenchmark

[![CUDA Build and Test](https://github.com/yushinliu/cuda-ptx-microbenchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/yushinliu/cuda-ptx-microbenchmark/actions/workflows/ci.yml)

A comprehensive microbenchmark suite for NVIDIA GPUs using CUDA and PTX assembly, specifically optimized for RTX 4070 (Compute Capability 8.9).

## Features

- **L1/L2 Cache Benchmarking**: Sequential and random access patterns with hit rate estimation
- **PTX Instruction Testing**: Direct assembly testing for FMA, LDG, LDS, BAR.SYNC instructions
- **Memory Bandwidth**: Global memory read/write/copy performance measurement
- **Google Test Integration**: Full TDD workflow with unit, integration, and E2E tests

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
```

### Run Benchmark

```bash
./cpm
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── core/              # Timer, result collector, benchmark runner
│   ├── kernels/memory/    # L1/L2 cache, global memory benchmarks
│   └── kernels/ptx/       # PTX instruction kernels
├── include/               # Header files
├── tests/                 # Google Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Kernel integration tests
│   └── e2e/              # End-to-end tests
├── docs/                  # Documentation
└── scripts/              # Helper scripts
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

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) and TDD methodology.
