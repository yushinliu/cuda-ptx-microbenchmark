from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_EXT_MOD: Any = None


def load_ext(verbose: bool = False):
    global _EXT_MOD
    if _EXT_MOD is not None:
        return _EXT_MOD

    os.environ.setdefault("MAX_JOBS", "8")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    this_dir = Path(__file__).resolve().parent
    _EXT_MOD = load(
        name="transpose_ptx_ext",
        sources=[
            str(this_dir / "transpose_ext.cpp"),
            str(this_dir / "transpose_kernel.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        extra_cflags=["-O3"],
        verbose=verbose,
        with_cuda=True,
    )
    return _EXT_MOD


def transpose_naive(x: torch.Tensor) -> torch.Tensor:
    return load_ext().transpose_ptx_naive(x)


def transpose_opt(x: torch.Tensor) -> torch.Tensor:
    return load_ext().transpose_ptx_opt(x)
