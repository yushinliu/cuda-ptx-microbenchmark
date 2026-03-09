from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load
import tilelang


_EXT_MOD: Any = None


def load_ext(verbose: bool = False):
    global _EXT_MOD
    if _EXT_MOD is not None:
        return _EXT_MOD

    os.environ["MAX_JOBS"] = "8"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    this_dir = Path(__file__).resolve().parent
    cutlass_include = Path(tilelang.__file__).resolve().parent / "3rdparty" / "cutlass" / "include"
    _EXT_MOD = load(
        name="matmul_ptx_ext",
        sources=[
            str(this_dir / "matmul_ext.cpp"),
            str(this_dir / "matmul_kernel.cu"),
        ],
        extra_include_paths=[str(cutlass_include)],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
        ],
        extra_cflags=["-O3"],
        verbose=verbose,
        with_cuda=True,
    )
    return _EXT_MOD


def matmul_mma(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma(x, w)


def matmul_mma_cpasync(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_cpasync(x, w)


def matmul_mma_cpasync_k32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_cpasync_k32(x, w)


def matmul_mma_cpasync_128(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_cpasync_128(x, w)


def matmul_mma_cpasync_128_k32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_cpasync_128_k32(x, w)


def matmul_mma_ldmatrix(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_ldmatrix(x, w)


def matmul_mma_ldmatrix_block(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_ldmatrix_block(x, w)


def matmul_mma_ldmatrix_block_cpasync_a(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul_mma_ldmatrix_block_cpasync_a(x, w)
