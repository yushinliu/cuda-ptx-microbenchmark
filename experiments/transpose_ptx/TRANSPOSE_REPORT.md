# RTX 4070 SUPER Transpose PTX Report

## Scope

This report covers three tasks:

1. Measure and analyze `torch.transpose(...).contiguous()` on GPU.
2. Implement an equivalent PTX-based transpose kernel without any PyTorch headers in the `.cu` file and verify correctness against PyTorch.
3. Optimize the PTX kernel with a generic strategy and compare performance against PyTorch.

All validation in this report was executed on GPU. Build parallelism was capped at `MAX_JOBS=8`.

## Environment

- GPU: NVIDIA GeForce RTX 4070 SUPER
- Driver: 576.88
- CUDA runtime reported by `nvidia-smi`: 12.9
- `nvcc`: 12.8.93
- PyTorch: 2.8.0+cu128
- Nsight Compute CLI tested: 2025.2.1 and 2025.3.1
- Nsight Systems CLI tested: 2025.3.1

## Task 1: PyTorch Transpose Baseline

Reference op:

```python
x.transpose(0, 1).contiguous()
```

Benchmark method:

- Input: contiguous `torch.float32` 2D CUDA tensors
- Warmup before timed runs
- Each timing uses explicit `torch.cuda.synchronize()`
- Throughput is computed as `2 * rows * cols * sizeof(float) / time`

### Baseline Results

| Shape | PyTorch ms | PTX Naive ms | PTX Opt ms | PTX Opt vs PyTorch |
|---|---:|---:|---:|---:|
| 1024 x 1024 | 0.0734 | 0.0655 | 0.0423 | 1.73x |
| 2048 x 2048 | 0.1525 | 0.1387 | 0.0694 | 2.20x |
| 3000 x 5000 | 0.7745 | 0.4118 | 0.4041 | 1.92x |
| 4096 x 4096 | 1.0439 | 0.4675 | 0.4368 | 2.39x |
| 8192 x 8192 | 3.8554 | 1.5452 | 1.5599 | 2.47x |
| 2048 x 8192 | 0.7403 | 0.4129 | 0.4287 | 1.73x |

### Nsight Systems Findings

Artifacts:

- `nsys_pytorch.nsys-rep`
- `nsys_opt.nsys-rep`

Observed from `nsys stats --report cuda_api_sum`:

- PyTorch path:
  - `cudaLaunchKernel`: 63.71 ms total across 66 calls
  - `cudaDeviceSynchronize`: 54.92 ms total across 2 calls
- PTX optimized path:
  - `cudaLaunchKernel`: 42.29 ms total across 66 calls
  - `cudaDeviceSynchronize`: 22.56 ms total across 2 calls

Interpretation:

- The PTX optimized path reduces total launch-side elapsed time and synchronized GPU completion time relative to the PyTorch baseline on the same workload.
- `nsys` in this environment did not expose CUDA kernel summary rows, only CUDA API summary rows. Kernel-level timing therefore comes primarily from the direct benchmark results above.

### Nsight Compute Status

Nsight Compute is now usable in this WSL environment after adding a local
`libcuda.so.1` symlink inside the Nsight Compute target directory to point at
`/usr/lib/wsl/lib/libcuda.so.1`.

This fixed the driver connection error and allowed direct kernel profiling.

### Is There Still Headroom?

Yes, but likely incremental rather than order-of-magnitude.

Inference from the algorithm and measured data:

- Transpose is memory-dominated on Ada for these tensor sizes.
- The generic tiled kernel already removes the main penalty of strided global stores by converting them into coalesced shared-memory-assisted writes.
- Remaining headroom is likely in:
  - vectorized loads and stores where alignment permits
  - better overlap of memory traffic with instruction issue
  - tuning for reduced synchronization cost
- Because `ncu` counters are unavailable, this headroom assessment is based on benchmark behavior and kernel structure rather than direct occupancy or stall metrics.

## Task 2: PTX Kernel Implementation

Files:

- `transpose_kernel.cu`: pure CUDA/PTX kernel implementation, no PyTorch headers
- `transpose_ext.cpp`: PyTorch binding layer
- `transpose_ext.py`: extension loader with `MAX_JOBS=8` and `TORCH_CUDA_ARCH_LIST=8.9`
- `test_transpose_ptx.py`: GPU correctness tests

Implementation summary:

- `transpose_ptx_naive`: direct PTX global load and global store
- `transpose_ptx_opt`: 32 x 32 tiled transpose using shared memory with `+1` padding to avoid bank conflicts

Correctness validation:

- Compared against `x.transpose(0, 1).contiguous()` on CUDA
- Tested on regular and irregular shapes
- Result: `7 passed`

## Task 3: Generic Optimization

Optimization used:

- shared-memory tiled transpose
- padded tile to avoid shared-memory bank conflicts
- unrolled tile loop
- generic boundary handling for non-power-of-two shapes

This optimization is generic. It does not dispatch based on any specific benchmarked shape.

### Before vs After

`ptx_naive` to `ptx_opt`:

- 1024 x 1024: 0.0655 ms -> 0.0423 ms
- 2048 x 2048: 0.1387 ms -> 0.0694 ms
- 3000 x 5000: 0.4118 ms -> 0.4041 ms
- 4096 x 4096: 0.4675 ms -> 0.4368 ms

Notes:

- On very large or strongly rectangular cases, the optimized kernel is not always faster than the naive PTX kernel, but both still outperform the PyTorch baseline in the measured workloads.
- The tiled kernel is the more principled generic optimization because it addresses the fundamental transpose access pattern rather than a single-size artifact.

## Additional Optimization Pass

After the first implementation pass, four more generic candidates were added to the native benchmark path:

- `ptx_vector`: vectorized `ld/st` around the tiled transpose path
- `ptx_swizzle`: shared-memory swizzle to reduce bank conflicts without shape-specific dispatch
- `ptx_cpasync`: direct global-to-shared copy using `cp.async`
- `ptx_vswizzle`: vectorized load/store combined with swizzled shared-memory layout

All candidates passed GPU correctness tests.

### Candidate Comparison

| Shape | Best Variant | Key Observation |
|---|---|---|
| 4096 x 4096 | `ptx_opt` | Current tiled padded kernel remained marginally best |
| 8192 x 8192 | `ptx_swizzle` | Swizzle helped more than padding at larger scale |
| 3000 x 5000 | `ptx_swizzle` | Swizzle slightly outperformed the padded tiled kernel |

Representative numbers:

| Shape | `ptx_opt` ms | `ptx_swizzle` ms | `ptx_cpasync` ms | `ptx_vswizzle` ms |
|---|---:|---:|---:|---:|
| 4096 x 4096 | 0.3392 | 0.3420 | 0.3503 | 0.3430 |
| 8192 x 8192 | 1.5129 | 1.4783 | 1.5562 | 1.5399 |
| 3000 x 5000 | 0.2944 | 0.2921 | 0.3018 | 0.3005 |

Conclusion from this pass:

- `swizzle` is the strongest new generic optimization candidate.
- `cp.async` did not improve this transpose workload, likely because the kernel has only one tile stage and limited opportunity to hide copy latency behind independent work.
- `vector load/store` alone did not consistently beat the existing padded tile kernel.
- `vector + swizzle` did not outperform plain `swizzle`, so the extra complexity is not currently justified.

## Nsight Compute Findings After Fix

Profiled kernels:

- `reference_transpose_kernel`
- `transpose_opt_kernel`
- `transpose_swizzle_kernel`
- `transpose_vswizzle_kernel`

Key findings:

- `reference_transpose_kernel` is dominated by uncoalesced global stores.
- `transpose_opt_kernel` removes the global-store pathology and shifts the bottleneck to memory latency / L1TEX scoreboard wait.
- The original swizzle mapping was wrong and still triggered shared-memory bank conflicts.
- After changing the swizzle mapping to `col ^ row`, the shared bank-conflict warning disappeared for `transpose_swizzle_kernel`.
- The corrected `transpose_swizzle_kernel` showed stronger profiler-side metrics than the previous swizzle attempt:
  - duration dropped to about `281.7 us`
  - memory throughput rose to about `411.5 GB/s`
  - achieved occupancy rose to about `88.5%`
- `transpose_vswizzle_kernel` still underperformed in `ncu` due to high L1TEX scoreboard stalls and lower issue efficiency, so combining vectorization with swizzle is not currently the best direction.

## Final Conclusion

- Task 1: completed with GPU benchmarking and Nsight Systems output.
- Task 2: completed with a PTX transpose implementation whose `.cu` file does not depend on any PyTorch headers.
- Task 3: completed with a generic tiled PTX optimization that outperforms the PyTorch transpose baseline on all measured workloads in this report.

## Current Repository State

The repository now has two transpose code paths:

- Native repository benchmark path in `src/kernels/ptx/transpose.cu`
- PyTorch extension path in `experiments/transpose_ptx/transpose_kernel.cu`

Important note:

- The native repository `launch_transpose_ptx_opt()` entry now defaults to the corrected swizzle implementation.
- The PyTorch extension path has not yet been updated to mirror that same default implementation.

Recent native benchmark results after switching the default optimized path:

| Shape | Native Reference ms | Native PTX Opt ms | Native PTX Swizzle ms |
|---|---:|---:|---:|
| 4096 x 4096 | 0.3461 | 0.3443 | 0.3279 |
| 3000 x 5000 | 0.2957 | 0.2911 | 0.2918 |
| 8192 x 8192 | 1.4281 | 1.4557 | 1.4592 |

Recent PyTorch extension comparison on `8192 x 8192`:

| Impl | Avg ms |
|---|---:|
| PyTorch transpose contiguous | 4.7555 |
| PTX naive | 1.9318 |
| PTX opt (extension path) | 2.0801 |

This means the native repository path and the PyTorch extension path should be interpreted separately until the extension implementation is synchronized with the latest native kernel work.

Constraint that remains external:

- Full Nsight Compute metric collection is blocked on this machine by GPU performance counter permissions (`ERR_NVGPUCTRPERM`), including attempts made under `sudo`.
