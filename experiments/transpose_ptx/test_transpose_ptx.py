import pytest
import torch

from transpose_ext import load_ext, transpose_naive, transpose_opt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_extension_loads():
    mod = load_ext(verbose=False)
    assert mod is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
@pytest.mark.parametrize("shape", [(257, 509), (512, 512), (1024, 1536), (2048, 1024), (3000, 5000)])
def test_transpose_correctness_against_pytorch(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    ref = x.transpose(0, 1).contiguous()

    y_naive = transpose_naive(x)
    y_opt = transpose_opt(x)

    max_abs_err_naive = (y_naive - ref).abs().max().item()
    max_abs_err_opt = (y_opt - ref).abs().max().item()
    assert max_abs_err_naive == 0.0
    assert max_abs_err_opt == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_output_shape_matches_transpose():
    x = torch.randn(321, 123, device="cuda", dtype=torch.float32).contiguous()
    y = transpose_naive(x)
    assert tuple(y.shape) == (123, 321)
