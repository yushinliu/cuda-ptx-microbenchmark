# Matmul PTX Optimization Report

## Environment

- GPU: NVIDIA GeForce RTX 4070 SUPER
- Validation policy: GPU only, no CPU fallback, no simulation
- Build constraint: `MAX_JOBS=8`
- Nsight Compute:
  - `2025.3.1` failed in this WSL environment with driver compatibility errors
  - working version:
    `/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu`

## Test And Benchmark Method

- Correctness:
  - `/home/yuliu/miniconda3/bin/python -m pytest -q /mnt/d/yuliu/cuda-ptx-microbenchmark/experiments/matmul_ptx/test_matmul_ptx.py`
  - current best kernel state: `6 passed`
  - correctness now also covers the experimental `matmul_mma_ldmatrix` path
- Benchmark:
  - `/home/yuliu/miniconda3/bin/python benchmark_matmul.py --m <M> --n <N> --k <K> --warmup 20 --iters 80`
  - current script uses CUDA events for timing, not host wall clock
  - `TFLOP/s` is computed from `median_ms`
- Profile:
  - `/home/yuliu/miniconda3/pkgs/nsight-compute-2025.2.1.3-0/nsight-compute-2025.2.1/ncu --target-processes all --kernel-name-base demangled -o <report> /home/yuliu/miniconda3/bin/python profile_target.py --impl <impl> --m <M> --n <N> --k <K> --warmup 3 --iters 10`

## Kernel Evolution

### 1. Baseline PTX/WMMA Kernel

- Tensor Core path confirmed in SASS: `HMMA.16816.F32`
- Async copy path confirmed in SASS: `LDGSTS.E.128`
- Early versions used a shared-memory transpose for `B`, which added measurable overhead

### 2. Row-Major B Feed

- Changed `matrix_b` feed to row-major
- Removed the per-stage shared-memory transpose of `B`
- Result: clear improvement for the `cp.async` kernel

### 3. Double Buffer To Deeper Pipeline

- Initial `cp.async` path used a 2-stage ping-pong double buffer
- Current best kernel uses a 3-stage pipeline
- The pipeline preloads future tiles before compute and reuses stage buffers in a ring
- This improved large-shape throughput significantly while keeping correctness intact

### 4. Negative Experiments That Were Reverted

- shape-specialized dispatch to a `64x64` small-tile kernel was tested but did not improve the default path reliably
- changing the large kernel to `cp.async.wait_group 1` also regressed latency
- these variants were not kept in the final default kernel

### 5. Full-Tile Fast Path

- Added a dedicated `cp.async` fast path for shapes that exactly match the threadblock tiling
- This removes bounds checks and zero-fill handling from the steady-state path
- Current dispatch uses the fast path when `m % 64 == 0` and `n % 128 == 0`
- This matched the main benchmark shapes and produced the best medium and large shape results so far

### 6. Experimental `ldmatrix + mma.sync` Warp Kernel

- Added a separate experimental kernel: `matmul_mma_ldmatrix_kernel`
- This path does not replace the current best `cp.async` kernel
- The implementation uses a single warp per output tile (`16x8x16` instruction tile) and relies on CUTLASS warp iterators to drive:
  - `ldmatrix`
  - `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
  - accumulator store-back
- GPU correctness passed against PyTorch, so the fragment load/store mapping is now validated on hardware
- SASS confirms the expected low-level instructions are present:
  - `LDSM.16.M88.*`
  - `LDSM.16.MT88.*`
  - `HMMA.16816.F32`

## Current Best Benchmark

Source: `bench_fastpath_*.json`

| Shape | PyTorch med ms | PTX cp.async med ms | PTX cp.async TFLOP/s | Result |
| --- | ---: | ---: | ---: | --- |
| 512x512x512 | 0.0398 | 0.0787 | 3.41 | PTX slower |
| 1024x1024x1024 | 0.0642 | 0.0737 | 29.13 | PTX slightly slower |
| 2048x2048x2048 | 0.3201 | 0.4049 | 42.43 | PTX slower but materially improved |

Important note:

- Switching from host wall-clock timing to CUDA event timing materially reduced measurement noise
- On 2048, the current best fast path reduced PTX `cp.async` from the earlier generic 3-stage `0.5152 ms` average-style measurement and from about `0.4168 ms` median in the earlier event-timed generic path down to `0.4049 ms` median
- On 1024, the fast path brought PTX `cp.async` down to `0.0737 ms` median, close to PyTorch `0.0642 ms`
- The best general configuration remained the `64x128x16` tile plus the 3-stage pipeline

## Experimental `ldmatrix` Benchmark

Source: `bench_ldmatrix_*.json`

| Shape | PyTorch med ms | PTX cp.async med ms | PTX ldmatrix med ms | Result |
| --- | ---: | ---: | ---: | --- |
| 512x512x512 | 0.0475 | 0.0205 | 0.1249 | `ldmatrix` much slower |
| 1024x1024x1024 | 0.0553 | 0.0737 | 0.8299 | `ldmatrix` much slower |

Interpretation:

- The new path is functionally correct and truly uses `ldmatrix + mma.sync`
- However, the current design is only a correctness and lowering milestone, not a performance win
- One warp per block and one instruction tile per output tile leaves too much threadblock-level reuse and pipeline overlap on the table

## Nsight Compute Findings

### PyTorch 2048

- Kernel: `ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1_nn`
- Representative metrics:
  - `gpu__time_duration.sum`: about `341 us`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: about `44.3%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: about `20.0%`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`: about `15.3%`
  - `launch__registers_per_thread`: `234`
  - `launch__shared_mem_per_block_static`: `32.77 KB`
  - `launch__waves_per_multiprocessor`: `2.29`

### PTX cp.async 3-stage 2048

- Kernel: `<unnamed>::matmul_mma_cpasync_kernel(const __half *, const __half *, float *, long, long, long)`
- Representative metrics:
  - `gpu__time_duration.sum`: about `585 us`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: about `25.8%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: about `13.1%`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`: about `59.7%`
  - `launch__registers_per_thread`: `55`
  - `launch__shared_mem_per_block_static`: `18.43 KB`
  - `launch__waves_per_multiprocessor`: `2.29`

### PTX cp.async Fast Path 2048

- Kernel: `<unnamed>::matmul_mma_cpasync_fast_kernel(const __half *, const __half *, float *, long, long, long)`
- Representative metrics:
  - `gpu__time_duration.sum`: about `571 us`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: about `26.4%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: about `13.5%`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`: about `59.9%`
  - `launch__registers_per_thread`: `56`
  - `launch__shared_mem_per_block_static`: `18.43 KB`
  - `launch__waves_per_multiprocessor`: `2.29`

### Experimental PTX `ldmatrix` 512

- Kernel: `<unnamed>::matmul_mma_ldmatrix_kernel(const __half *, const __half *, float *, long, long, long)`
- Representative metrics:
  - `gpu__time_duration.sum`: about `214.8 us`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: about `50.6%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: about `2.29%`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`: about `39.7%`
  - `launch__registers_per_thread`: `26`
  - `launch__shared_mem_per_block_static`: `0.768 KB`
  - `launch__waves_per_multiprocessor`: `1.52`

## Interpretation

- `ncu` is fully working and was used on both PyTorch and the custom PTX kernel.
- The current PTX kernel already uses Tensor Core instructions and `cp.async`.
- Moving from 2-stage double buffer to a deeper 3-stage pipeline produced a real gain, especially on `2048`.
- CUDA event timing showed the custom kernel is closer to PyTorch than the earlier host-side timing suggested.
- The full-tile fast path produced another measurable gain on medium and large shapes by removing steady-state boundary handling.
- The remaining gap to PyTorch is not explained by occupancy alone.
- The PTX kernel actually shows much higher active warps than PyTorch, but lower SM throughput.
- This indicates the bottleneck has shifted toward warp-level instruction efficiency, fragment load/store overhead, and kernel structure rather than simple under-occupancy.
- The experimental `ldmatrix` kernel confirms that simply dropping below `wmma` is not enough by itself.
- Even with direct `LDSM/HMMA` instructions, the kernel is still slow because the surrounding execution structure is too small and cannot amortize loads or overlap work effectively.

## Remaining Headroom

There is still clear optimization headroom, but it likely requires a more aggressive rewrite than parameter tuning alone:

- replace more `wmma` wrapper usage with inline PTX `ldmatrix` + `mma.sync`
- reduce fragment load/store overhead and shared-memory traffic around `wmma::load_matrix_sync`
- consider warp-specialized copy/compute roles so `cp.async` overlap is more explicit
- add size-based kernel dispatch if small and medium shapes prefer different tile shapes
- tune shared-memory layout to reduce L1/shared pressure further

## Status Against Goal

- Goal met:
  - GPU-only correctness validation
  - `ncu` profiling on the 4070
  - PTX kernel with `mma` and `cp.async`
  - correctness preserved after optimization
  - measurable speedup versus earlier PTX baselines
- Goal not yet met:
  - current PTX kernel is still slower than PyTorch on the main medium and large shapes, though the gap is narrower after the fast-path optimization
  - the experimental `ldmatrix` rewrite is correct but currently much slower than both PyTorch and the tuned `cp.async` kernel

## Artifacts

- NCU reports:
  - `results/ncu_pytorch_512_seq.ncu-rep`
  - `results/ncu_mma_512_seq.ncu-rep`
  - `results/ncu_cpasync_512_seq.ncu-rep`
  - `results/ncu_pytorch_1024_rowmajorb.ncu-rep`
  - `results/ncu_cpasync_1024_rowmajorb.ncu-rep`
  - `results/ncu_pytorch_2048_stage3.ncu-rep`
  - `results/ncu_cpasync_2048_stage3.ncu-rep`
  - `results/ncu_cpasync_2048_fastpath.ncu-rep`
  - `results/ncu_ldmatrix_512.ncu-rep`
- Benchmarks:
  - `results/bench_rowmajorb_*.json`
  - `results/bench_tile64x64_*.json`
  - `results/bench_stage3_*.json`
  - `results/bench_events_*.json`
  - `results/bench_fastpath_*.json`
  - `results/bench_ldmatrix_512.json`
  - `results/bench_ldmatrix_1024.json`
