from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch

from matmul_ext import (
    matmul_mma,
    matmul_mma_cpasync,
    matmul_mma_cpasync_k32,
    matmul_mma_ldmatrix,
    matmul_mma_ldmatrix_block,
    matmul_mma_ldmatrix_block_cpasync_a,
)


@dataclass
class BenchResult:
    impl: str
    shape: tuple[int, int, int]
    input_dtype: str
    output_dtype: str
    avg_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    tflops: float


def torch_mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b, out_dtype=torch.float32)


def run_bench(fn, a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int) -> tuple[float, float, float, float]:
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()

    times_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_event.record()
        fn(a, b)
        end_event.record()
        end_event.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    return (
        sum(times_ms) / len(times_ms),
        statistics.median(times_ms),
        min(times_ms),
        max(times_ms),
    )


def validate(a: torch.Tensor, b: torch.Tensor):
    ref = torch_mm_fp32(a, b)
    y_mma = matmul_mma(a, b)
    y_cpasync = matmul_mma_cpasync(a, b)
    y_cpasync_k32 = matmul_mma_cpasync_k32(a, b)
    y_ldmatrix = matmul_mma_ldmatrix(a, b)
    y_ldmatrix_block = matmul_mma_ldmatrix_block(a, b)
    y_ldmatrix_block_cpasync_a = matmul_mma_ldmatrix_block_cpasync_a(a, b)
    err_mma = (y_mma - ref).abs().max().item()
    err_cpasync = (y_cpasync - ref).abs().max().item()
    err_cpasync_k32 = (y_cpasync_k32 - ref).abs().max().item()
    err_ldmatrix = (y_ldmatrix - ref).abs().max().item()
    err_ldmatrix_block = (y_ldmatrix_block - ref).abs().max().item()
    err_ldmatrix_block_cpasync_a = (y_ldmatrix_block_cpasync_a - ref).abs().max().item()
    if (
        err_mma > 1e-2
        or err_cpasync > 1e-2
        or err_cpasync_k32 > 1e-2
        or err_ldmatrix > 1e-2
        or err_ldmatrix_block > 1e-2
        or err_ldmatrix_block_cpasync_a > 1e-2
    ):
        raise RuntimeError(
            f"correctness failed: mma={err_mma}, "
            f"cpasync={err_cpasync}, cpasync_k32={err_cpasync_k32}, "
            f"ldmatrix={err_ldmatrix}, ldmatrix_block={err_ldmatrix_block}, "
            f"ldmatrix_block_cpasync_a={err_ldmatrix_block_cpasync_a}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch mm vs PTX MMA kernels on GPU")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.float16).contiguous()
    validate(a, b)

    impls = {
        "pytorch_mm_out_fp32": torch_mm_fp32,
        "ptx_mma": matmul_mma,
        "ptx_mma_cpasync": matmul_mma_cpasync,
        "ptx_mma_cpasync_k32": matmul_mma_cpasync_k32,
        "ptx_mma_ldmatrix": matmul_mma_ldmatrix,
        "ptx_mma_ldmatrix_block": matmul_mma_ldmatrix_block,
        "ptx_mma_ldmatrix_block_cpasync_a": matmul_mma_ldmatrix_block_cpasync_a,
    }

    flops = 2.0 * args.m * args.n * args.k
    results: list[BenchResult] = []
    for name, fn in impls.items():
        avg_ms, median_ms, min_ms, max_ms = run_bench(fn, a, b, args.warmup, args.iters)
        tflops = flops / (median_ms * 1e-3) / 1e12
        y = fn(a, b)
        results.append(
            BenchResult(
                impl=name,
                shape=(args.m, args.n, args.k),
                input_dtype=str(a.dtype),
                output_dtype=str(y.dtype),
                avg_ms=avg_ms,
                median_ms=median_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                tflops=tflops,
            )
        )

    print(f"shape=({args.m}, {args.n}, {args.k}), dtype={a.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"{'impl':24s} {'avg_ms':>10s} {'med_ms':>10s} {'min_ms':>10s} {'max_ms':>10s} {'TFLOP/s':>10s}")
    for r in results:
        print(f"{r.impl:24s} {r.avg_ms:10.4f} {r.median_ms:10.4f} {r.min_ms:10.4f} {r.max_ms:10.4f} {r.tflops:10.2f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
