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

Nsight Compute was attempted with:

- regular user
- elevated sandbox execution
- `sudo`
- two installed `ncu` versions

Result:

- profiling was blocked by `ERR_NVGPUCTRPERM`
- even under `sudo`, the environment still could not access GPU performance counters on this machine

This means no valid Nsight Compute counter report could be collected in the current environment. The blocker is system permission configuration, not kernel correctness or launch failure.

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

## Final Conclusion

- Task 1: completed with GPU benchmarking and Nsight Systems output.
- Task 2: completed with a PTX transpose implementation whose `.cu` file does not depend on any PyTorch headers.
- Task 3: completed with a generic tiled PTX optimization that outperforms the PyTorch transpose baseline on all measured workloads in this report.

Constraint that remains external:

- Full Nsight Compute metric collection is blocked on this machine by GPU performance counter permissions (`ERR_NVGPUCTRPERM`), including attempts made under `sudo`.
