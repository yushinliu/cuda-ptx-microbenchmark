from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

import benchmark_matmul as bm


DEFAULT_SIZES = (256, 512, 1024, 2048, 4096, 8192)


def sweep_config(size: int) -> tuple[int, int]:
    if size <= 512:
        return (20, 120)
    if size <= 1024:
        return (20, 80)
    if size <= 2048:
        return (10, 40)
    if size <= 4096:
        return (5, 20)
    return (3, 10)


def validate_selected(a: torch.Tensor, b: torch.Tensor) -> None:
    ref = bm.torch_mm_fp32(a, b)
    for name, fn in (
        ("ptx_mma_cpasync", bm.matmul_mma_cpasync),
        ("ptx_mma_ldmatrix_block_cpasync_a", bm.matmul_mma_ldmatrix_block_cpasync_a),
    ):
        y = fn(a, b)
        err = (y - ref).abs().max().item()
        if err > 1e-2:
            raise RuntimeError(f"{name} correctness failed with max abs err {err}")


def run_sweep(
    sizes: list[int],
    output_json: Path,
) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.cuda.init()
    torch.cuda.get_device_properties(0)

    impls = {
        "pytorch_mm_out_fp32": bm.torch_mm_fp32,
        "ptx_mma_cpasync": bm.matmul_mma_cpasync,
        "ptx_mma_ldmatrix_block_cpasync_a": bm.matmul_mma_ldmatrix_block_cpasync_a,
    }

    all_results: list[dict] = []
    for size in sizes:
        warmup, iters = sweep_config(size)
        a = torch.randn(size, size, device="cuda", dtype=torch.float16).contiguous()
        b = torch.randn(size, size, device="cuda", dtype=torch.float16).contiguous()
        validate_selected(a, b)

        flops = 2.0 * size * size * size
        print(f"size={size}, warmup={warmup}, iters={iters}, dtype={a.dtype}")
        for name, fn in impls.items():
            avg_ms, median_ms, min_ms, max_ms = bm.run_bench(fn, a, b, warmup, iters)
            y = fn(a, b)
            record = asdict(
                bm.BenchResult(
                    impl=name,
                    shape=(size, size, size),
                    input_dtype=str(a.dtype),
                    output_dtype=str(y.dtype),
                    avg_ms=avg_ms,
                    median_ms=median_ms,
                    min_ms=min_ms,
                    max_ms=max_ms,
                    tflops=flops / (median_ms * 1e-3) / 1e12,
                )
            )
            record["warmup"] = warmup
            record["iters"] = iters
            all_results.append(record)
            print(
                f"  {name:28s} med_ms={median_ms:10.4f} "
                f"tflops={record['tflops']:10.2f}"
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"saved {output_json}")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run power-of-two GPU matmul sweep")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Square GEMM sizes to benchmark",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/pow2_sweep.json"),
    )
    args = parser.parse_args()
    run_sweep(args.sizes, args.output_json)


if __name__ == "__main__":
    main()
