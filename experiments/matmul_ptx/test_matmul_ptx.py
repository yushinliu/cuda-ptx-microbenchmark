import pytest
import torch

from matmul_ext import load_ext, matmul_mma, matmul_mma_cpasync, matmul_mma_ldmatrix


def torch_mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b, out_dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_extension_loads():
    mod = load_ext(verbose=False)
    assert mod is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 64, 64), (128, 128, 128), (256, 256, 256)])
def test_matmul_correctness_against_pytorch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(k, n, device="cuda", dtype=torch.float16).contiguous()
    ref = torch_mm_fp32(a, b)

    y_mma = matmul_mma(a, b)
    y_cpasync = matmul_mma_cpasync(a, b)
    y_ldmatrix = matmul_mma_ldmatrix(a, b)

    assert y_mma.dtype == torch.float32
    assert y_cpasync.dtype == torch.float32
    assert y_ldmatrix.dtype == torch.float32

    max_abs_err_mma = (y_mma - ref).abs().max().item()
    max_abs_err_cpasync = (y_cpasync - ref).abs().max().item()
    max_abs_err_ldmatrix = (y_ldmatrix - ref).abs().max().item()
    assert max_abs_err_mma <= 1e-2
    assert max_abs_err_cpasync <= 1e-2
    assert max_abs_err_ldmatrix <= 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_output_shape_matches_mm():
    a = torch.randn(128, 64, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(64, 96, device="cuda", dtype=torch.float16).contiguous()
    y = matmul_mma(a, b)
    assert tuple(y.shape) == (128, 96)
