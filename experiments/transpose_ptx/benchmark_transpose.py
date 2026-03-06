from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from transpose_ext import transpose_naive, transpose_opt


@dataclass
class BenchResult:
    impl: str
    shape: tuple[int, int]
    dtype: str
    avg_ms: float
    min_ms: float
    max_ms: float
    gbps: float


def run_bench(fn, x: torch.Tensor, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        y = fn(x)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return sum(times_ms) / len(times_ms), min(times_ms), max(times_ms)


def pytorch_impl(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).contiguous()


def validate(x: torch.Tensor):
    ref = pytorch_impl(x)
    y_naive = transpose_naive(x)
    y_opt = transpose_opt(x)
    err_naive = (y_naive - ref).abs().max().item()
    err_opt = (y_opt - ref).abs().max().item()
    if err_naive != 0.0 or err_opt != 0.0:
        raise RuntimeError(f"correctness failed: naive={err_naive}, opt={err_opt}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch transpose vs PTX kernels on GPU")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    x = torch.randn(args.rows, args.cols, device="cuda", dtype=torch.float32).contiguous()
    validate(x)

    impls = {
        "pytorch_transpose_contiguous": pytorch_impl,
        "ptx_naive": transpose_naive,
        "ptx_opt": transpose_opt,
    }

    element_size = x.element_size()
    bytes_moved = 2.0 * args.rows * args.cols * element_size

    results: list[BenchResult] = []
    for name, fn in impls.items():
        avg_ms, min_ms, max_ms = run_bench(fn, x, args.warmup, args.iters)
        gbps = bytes_moved / (avg_ms * 1e-3) / 1e9
        results.append(
            BenchResult(
                impl=name,
                shape=(args.rows, args.cols),
                dtype=str(x.dtype),
                avg_ms=avg_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                gbps=gbps,
            )
        )

    print(f"shape=({args.rows}, {args.cols}), dtype={x.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"{'impl':34s} {'avg_ms':>10s} {'min_ms':>10s} {'max_ms':>10s} {'GB/s':>10s}")
    for r in results:
        print(f"{r.impl:34s} {r.avg_ms:10.4f} {r.min_ms:10.4f} {r.max_ms:10.4f} {r.gbps:10.2f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
