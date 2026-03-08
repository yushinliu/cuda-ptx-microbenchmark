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
  - primary validation in this WSL environment is now direct GPU execution through `/home/yuliu/miniconda3/bin/python -c ...`
  - reason: `pytest` and `python <file>.py` intermittently hit `cudaGetDeviceCount` error `304` in this environment even when plain `python -c` GPU runs are healthy
  - current optimization iterations were therefore validated by direct GPU spot checks on both non-full-tile and full-tile shapes
  - representative direct checks used:
    - `(64, 64, 64)` to cover non-full-tile fallback
    - `(256, 256, 256)` to cover the full-tile hybrid path
- Benchmark:
  - event-timed benchmark logic is in `benchmark_matmul.py`
  - in this environment it was driven through `/home/yuliu/miniconda3/bin/python -c ...` to avoid the same file-execution CUDA initialization issue
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

### 7. Experimental Large-K `cp.async` Kernel

- Added a separate large-kernel experiment: `matmul_mma_cpasync_k32`
- This variant keeps the current `64x128` block tile but changes the pipeline to:
  - `blockK = 32`
  - `2-stage` double buffer
- Motivation:
  - increase math work per staged tile
  - reduce stage-loop overhead on large shapes
- Result:
  - correctness passed on GPU
  - performance regressed versus the current `blockK = 16`, `3-stage` fast path
- It was not promoted to the default path

### 8. Experimental Multi-Warp `ldmatrix` Block Kernel

- Added a separate structural experiment: `matmul_mma_ldmatrix_block`
- This kernel keeps the existing block tile topology:
  - threadblock tile: `64x128x16`
  - warp arrangement: `2 x 4`
  - warp tile: `32x32x16`
- Compute is moved to CUTLASS warp `ldmatrix + mma.sync` iterators
- This is still separate from the default path and only runs as an explicit experimental API
- GPU correctness passed after adding a launcher fallback for non-full-tile shapes
- SASS confirms the block kernel also emits:
  - `LDSM.16.M88.*`
  - `HMMA.16816.F32`

### 9. Hybrid `ldmatrix block + cp.async(A)` Kernel

- Added a hybrid large-kernel experiment: `matmul_mma_ldmatrix_block_cpasync_a`
- This keeps the multi-warp `ldmatrix + mma.sync` compute path, but replaces the `A` operand stage fill with `cp.async`
- The kernel remains full-tile only in its optimized path and falls back for non-full-tile shapes
- GPU spot checks passed on:
  - `(64, 64, 64)` fallback path
  - `(256, 256, 256)` full-tile path
- This was a real improvement over the scalar-fill `ldmatrix_block` baseline, but it still did not beat the default `cp.async` kernel

### 10. Hybrid Negative Follow-Ups

- Increasing the hybrid pipeline from `2-stage` to `3-stage` regressed performance
- Trying to move operand `B` to direct `cp.async` into the CUTLASS crosswise shared-memory layout triggered `CUDA error: misaligned address`
- That misalignment indicates the `B` crosswise layout cannot be safely covered by a naive 16-byte chunk mapping
- Both variants were reverted

### 11. Vectorized `B` Feed For `ldmatrix` Block Kernels

- The next successful optimization was to keep the existing crosswise shared-memory layout for operand `B`, but widen the global feed from scalar loads to `16B` vector loads
- Implementation detail:
  - global `B` is still read row-major as contiguous `8 x fp16` chunks
  - each chunk is then scattered into the CUTLASS crosswise shared-memory layout
  - this avoids the `misaligned address` issue seen with naive direct `cp.async` into the final `B` layout
- This change was applied to both:
  - `matmul_mma_ldmatrix_block`
  - `matmul_mma_ldmatrix_block_cpasync_a`
- GPU spot checks remained exact on representative shapes:
  - `(64, 64, 64)`: `0.0`
  - `(256, 256, 256)`: `0.0`

## Current Best Benchmark

Source: latest direct event-timed GPU runs plus `bench_fastpath_*.json`

| Shape | PyTorch med ms | Best PTX med ms | Best PTX impl | Best PTX TFLOP/s | Result |
| --- | ---: | ---: | --- | ---: | --- |
| 512x512x512 | 0.0398 | 0.0787 | `ptx_mma_cpasync` | 3.41 | PTX slower |
| 1024x1024x1024 | 0.0471 | 0.0737 | `ptx_mma_cpasync` | 29.13 | PTX slower |
| 2048x2048x2048 | 0.2773 | 0.4287 | `ptx_mma_ldmatrix_block_cpasync_a` | 40.07 | PTX slower, but best custom path shifted to hybrid |

Important note:

- Switching from host wall-clock timing to CUDA event timing materially reduced measurement noise
- On 1024, the best custom kernel remains the WMMA-based `cp.async` fast path at `0.0737 ms`
- On 2048, the latest vectorized-`B` hybrid run shifted the best custom result to `matmul_mma_ldmatrix_block_cpasync_a` at `0.4287 ms`
- The gap to PyTorch remains large, but the best custom path is no longer uniformly the WMMA fast path at every shape

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

## Experimental Multi-Warp `ldmatrix` Block Benchmark

Source: `bench_ldmatrix_block_*.json`

| Shape | PTX ldmatrix med ms | PTX ldmatrix block med ms | PTX cp.async med ms | Result |
| --- | ---: | ---: | ---: | --- |
| 1024x1024x1024 | 0.8333 | 0.1781 | 0.0737 | block kernel much better than single-warp prototype, still slower than `cp.async` |
| 2048x2048x2048 | 8.2010 | 0.8856 | 0.4464 | block kernel dramatically better than single-warp prototype, still slower than `cp.async` |

Interpretation:

- moving `ldmatrix + mma.sync` from a single-warp microkernel to a multi-warp block kernel was a real structural improvement
- the improvement confirms the previous diagnosis that the single-warp version was dominated by tiny-tile inefficiency
- however, this block kernel still stages data with scalar global-to-shared stores instead of `cp.async`
- so it fixes the compute-side structure partially, but not the copy/overlap side

## Experimental Hybrid `ldmatrix block + cp.async(A)` Benchmark

Source: `bench_ldmatrix_block_cpasync_a_1024.json`, `bench_ldmatrix_block_cpasync_a_2048.json`

| Shape | PTX ldmatrix block med ms | PTX hybrid med ms | PTX cp.async med ms | Result |
| --- | ---: | ---: | ---: | --- |
| 1024x1024x1024 | 0.1786 | 0.1181 | 0.0737 | hybrid clearly better than scalar-fill `ldmatrix_block`, still slower than best `cp.async` |
| 2048x2048x2048 | 0.8857 | 0.6779 | 0.4065 | hybrid materially better than scalar-fill `ldmatrix_block`, still slower than best `cp.async` |

Interpretation:

- adding `cp.async` only for operand `A` was enough to produce a meaningful structural speedup
- this confirms that the `ldmatrix_block` path was substantially copy-path limited, not only compute-iterator limited
- however, the remaining gap to the default `cp.async` kernel is still large
- the most likely remaining cost is operand `B` staging plus CUTLASS warp-iterator overhead around the `ldmatrix + mma.sync` path

## Vectorized `B` Feed Benchmark

Source: `bench_bvec_final_1024.json`, `bench_bvec_final_2048.json`

| Shape | PTX cp.async med ms | PTX ldmatrix block med ms | PTX hybrid med ms | Result |
| --- | ---: | ---: | ---: | --- |
| 1024x1024x1024 | 0.0737 | 0.1319 | 0.0839 | vectorized `B` feed materially improves both `ldmatrix` block paths; hybrid gets close to best `cp.async` |
| 2048x2048x2048 | 0.4473 | 0.6953 | 0.4287 | vectorized `B` feed materially improves both `ldmatrix` block paths; hybrid slightly beats best `cp.async` in this run |

Interpretation:

- vectorizing only the global feed of operand `B` was the most effective optimization after adding `cp.async` on operand `A`
- compared with the earlier scalar-fill results:
  - `ldmatrix_block`: `0.1786 -> 0.1319` at `1024`, `0.8857 -> 0.6953` at `2048`
  - hybrid: `0.1181 -> 0.0839` at `1024`, `0.6779 -> 0.4287` at `2048`
- this is the first structural change that pushed the hybrid kernel into the same performance band as the WMMA-based best `cp.async` kernel
- repeat runs on `2048` still show some overlap between the two kernels, so this should be interpreted as "competitive and sometimes faster" rather than a stable blanket win

## Hybrid Negative Follow-Up Benchmark

Source: `bench_ldmatrix_block_cpasync_a_stage3_1024.json`, `bench_ldmatrix_block_cpasync_a_stage3_2048.json`

| Shape | PTX hybrid 2-stage med ms | PTX hybrid 3-stage med ms | Result |
| --- | ---: | ---: | --- |
| 1024x1024x1024 | 0.1181 | 0.1300 | `3-stage` slower |
| 2048x2048x2048 | 0.6779 | 0.7137 | `3-stage` slower |

Interpretation:

- unlike the WMMA-based default path, this hybrid `ldmatrix` kernel did not benefit from a deeper pipeline
- the added shared-memory footprint and extra staging complexity outweighed any overlap gain
- current best hybrid state remains the `2-stage` version
- this remained true even after the vectorized `B` feed improvement

## Experimental Large-K Benchmark

Source: `bench_k32_*.json`

| Shape | PTX cp.async med ms | PTX cp.async k32 med ms | Result |
| --- | ---: | ---: | --- |
| 1024x1024x1024 | 0.0737 | 0.0801 | `blockK=32` slower |
| 2048x2048x2048 | 0.4464 | 0.4854 | `blockK=32` slower |

Interpretation:

- increasing `blockK` to `32` did not offset the larger per-stage shared-memory footprint
- the new kernel reduced stage count, but it also increased static shared memory from about `18.43 KB` to about `24.58 KB`
- the warp-level fragment load/compute structure remained the same, so the larger staged tile did not remove the dominant inefficiency
- current best remains the `64x128x16` fast path with the `3-stage` pipeline

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

### Experimental PTX `cp.async k32` 2048

- Kernel: `<unnamed>::matmul_mma_cpasync_k32_fast_kernel(const __half *, const __half *, float *, long, long, long)`
- Representative metrics:
  - `gpu__time_duration.sum`: about `624 us`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: about `24.3%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: about `9.30%`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`: about `59.5%`
  - `launch__registers_per_thread`: `61`
  - `launch__shared_mem_per_block_static`: `24.58 KB`
  - `launch__waves_per_multiprocessor`: `2.29`

Compared with the current best `cp.async` fast path at `2048`:

- `blockK=32` is slower (`~624 us` vs `~571 us`)
- `SM throughput` is also lower (`~24.3%` vs `~26.4%`)
- `DRAM throughput` drops as well (`~9.3%` vs `~13.5%`)

This suggests the larger staged tile did not improve effective overlap or tensor-core utilization.

### Experimental PTX `ldmatrix` Block 2048

- Kernel: `<unnamed>::matmul_mma_ldmatrix_block_kernel(const __half *, const __half *, float *, long, long, long)`
- Representative metrics:
  - `gpu__time_duration.sum`: about `2.66 ms`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: about `49.4%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`: about `11.8%`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`: about `57.0%`
  - `launch__registers_per_thread`: `53`
  - `launch__shared_mem_per_block_static`: `6.14 KB`
  - `launch__waves_per_multiprocessor`: `2.29`

Interpretation:

- the block `ldmatrix` kernel achieves much higher `SM throughput` than the `cp.async` kernels
- but it still loses in end-to-end runtime because the surrounding load/store path is inefficient
- in practice, the kernel is spending too much work on feeding the tensor core path, even though the tensor core path itself is active

## Roofline View

Using hardware attributes visible in profiling:

- memory clock: about `10.5 GHz`
- memory bus width: `192-bit`
- approximate memory bandwidth: about `504 GB/s`
- SM count: `56`
- max graphics/SM clock: about `2.565 GHz`
- approximate FP32 peak: about `36.8 TFLOP/s`

That gives a conservative FP32 ridge point of roughly:

- `36.8 / 0.504 ~= 73 flop/byte`

For square GEMM, a lower-bound operational intensity is:

- `OI ~= 2MNK / (2MK + 2KN + 4MN)`
- `1024^3`: about `256 flop/byte`
- `2048^3`: about `512 flop/byte`

Implication:

- these GEMMs are far to the right of the conservative FP32 ridge point
- even without using an exact Tensor Core peak ceiling, the arithmetic intensity is high enough that a well-optimized kernel should not be fundamentally DRAM-bound
- this matches NCU:
  - PyTorch and custom kernels both use relatively modest DRAM throughput
  - the gap is mainly in `SM throughput`, not in saturating memory bandwidth

So the remaining headroom is primarily above the roofline's bandwidth region:

- better tensor-core issue efficiency
- better warp-level fragment movement
- better copy/compute overlap
- better block-level tile organization

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
- The experimental `blockK=32` kernel shows that simply increasing tile depth is also not enough.
- On this implementation, larger staged tiles lowered effective throughput instead of raising it, so the next useful step remains a more structural rewrite of the large kernel rather than a single-parameter retune.
- The multi-warp `ldmatrix` block kernel confirms that structural compute-side changes do help.
- But it also shows the next missing piece clearly: `ldmatrix + mma.sync` needs to be combined with an efficient staged copy path such as `cp.async`, not paired with scalar shared-memory fills.
- The hybrid `ldmatrix block + cp.async(A)` kernel confirms that partial copy-path repair works and yields real speedup.
- But it also sharpens the next blocker: operand `B` staging and warp-iterator overhead still dominate enough to keep the hybrid path behind the simpler WMMA-based `cp.async` kernel.
- The failed `B`-side `cp.async` attempt also exposed a concrete low-level risk: CUTLASS crosswise layouts impose alignment constraints that are not automatically compatible with naive 16-byte async chunking.
- The later vectorized `B` feed change materially reduced that staging cost without violating layout alignment constraints.
- After this fix, the hybrid kernel became competitive with the best WMMA `cp.async` path at `2048`, which confirms that `B` staging was indeed one of the dominant remaining bottlenecks.

## Remaining Headroom

There is still clear optimization headroom, but it likely requires a more aggressive rewrite than parameter tuning alone:

- replace more `wmma` wrapper usage with inline PTX `ldmatrix` + `mma.sync`
- reduce fragment load/store overhead and shared-memory traffic around `wmma::load_matrix_sync`
- reduce `B` operand staging cost for the `ldmatrix` path, likely with a layout-aware async or vectorized copy strategy instead of scalar fills
- if continuing beyond the current vectorized feed, the next useful step is likely a more layout-aware `B` staging path that preserves 16-byte movement while avoiding the direct-layout misalignment trap
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
  - `results/ncu_cpasync_k32_2048.ncu-rep`
  - `results/ncu_ldmatrix_512.ncu-rep`
  - `results/ncu_ldmatrix_block_2048.ncu-rep`
- Benchmarks:
  - `results/bench_rowmajorb_*.json`
  - `results/bench_tile64x64_*.json`
  - `results/bench_stage3_*.json`
  - `results/bench_events_*.json`
  - `results/bench_fastpath_*.json`
  - `results/bench_k32_1024.json`
  - `results/bench_k32_2048.json`
  - `results/bench_ldmatrix_512.json`
  - `results/bench_ldmatrix_1024.json`
  - `results/bench_ldmatrix_block_1024.json`
  - `results/bench_ldmatrix_block_2048.json`
  - `results/bench_ldmatrix_block_cpasync_a_1024.json`
  - `results/bench_ldmatrix_block_cpasync_a_2048.json`
  - `results/bench_ldmatrix_block_cpasync_a_stage3_1024.json`
  - `results/bench_ldmatrix_block_cpasync_a_stage3_2048.json`
  - `results/bench_bvec_1024.json`
  - `results/bench_bvec_2048.json`
  - `results/bench_bvec_final_1024.json`
  - `results/bench_bvec_final_2048.json`
  - `results/bench_bvec_stage3_1024.json`
  - `results/bench_bvec_stage3_2048.json`
