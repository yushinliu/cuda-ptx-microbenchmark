from __future__ import annotations

import argparse

import torch

from matmul_ext import (
    matmul_mma,
    matmul_mma_cpasync,
    matmul_mma_cpasync_k32,
    matmul_mma_ldmatrix,
    matmul_mma_ldmatrix_block,
    matmul_mma_ldmatrix_block_cpasync_a,
)


def torch_mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b, out_dtype=torch.float32)


def run_once(impl: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if impl == "pytorch":
        return torch_mm_fp32(a, b)
    if impl == "mma":
        return matmul_mma(a, b)
    if impl == "cpasync":
        return matmul_mma_cpasync(a, b)
    if impl == "cpasync_k32":
        return matmul_mma_cpasync_k32(a, b)
    if impl == "ldmatrix":
        return matmul_mma_ldmatrix(a, b)
    if impl == "ldmatrix_block":
        return matmul_mma_ldmatrix_block(a, b)
    if impl == "ldmatrix_block_cpasync_a":
        return matmul_mma_ldmatrix_block_cpasync_a(a, b)
    raise ValueError(f"unknown impl: {impl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        choices=["pytorch", "mma", "cpasync", "cpasync_k32", "ldmatrix", "ldmatrix_block", "ldmatrix_block_cpasync_a"],
        required=True,
    )
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.float16).contiguous()
    ref = torch_mm_fp32(a, b)
    y = run_once(args.impl, a, b)
    err = (y - ref).abs().max().item()
    if err > 1e-2:
        raise RuntimeError(f"correctness failed for {args.impl}: {err}")

    for _ in range(args.warmup):
        y = run_once(args.impl, a, b)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        y = run_once(args.impl, a, b)
    torch.cuda.synchronize()

    print(f"done impl={args.impl}, shape=({args.m}, {args.n}, {args.k}), iters={args.iters}")


if __name__ == "__main__":
    main()
