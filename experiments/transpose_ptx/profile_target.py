from __future__ import annotations

import argparse

import torch

from transpose_ext import transpose_naive, transpose_opt


def run_once(impl: str, x: torch.Tensor) -> torch.Tensor:
    if impl == "pytorch":
        return x.transpose(0, 1).contiguous()
    if impl == "naive":
        return transpose_naive(x)
    if impl == "opt":
        return transpose_opt(x)
    raise ValueError(f"unknown impl: {impl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["pytorch", "naive", "opt"], required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    x = torch.randn(args.rows, args.cols, device="cuda", dtype=torch.float32).contiguous()
    ref = x.transpose(0, 1).contiguous()
    y = run_once(args.impl, x)
    err = (y - ref).abs().max().item()
    if err != 0.0:
        raise RuntimeError(f"correctness failed for {args.impl}: {err}")

    for _ in range(args.warmup):
        y = run_once(args.impl, x)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        y = run_once(args.impl, x)
    torch.cuda.synchronize()

    print(f"done impl={args.impl}, shape=({args.rows}, {args.cols}), iters={args.iters}")


if __name__ == "__main__":
    main()
