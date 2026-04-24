"""FWHT correctness tests."""
from __future__ import annotations

import pytest
import torch

from fastfood.fwht import fwht, fwht_ortho, is_power_of_two, next_power_of_two


def _hadamard_matrix(d: int) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix of size d (power of two)."""
    assert is_power_of_two(d)
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat(
            (
                torch.cat((H, H), dim=1),
                torch.cat((H, -H), dim=1),
            ),
            dim=0,
        )
    return H


@pytest.mark.parametrize("d", [1, 2, 4, 8, 16, 64, 256])
def test_fwht_matches_explicit_hadamard(d):
    torch.manual_seed(d)
    x = torch.randn(5, d)
    H = _hadamard_matrix(d)
    expected = x @ H.T
    got = fwht(x)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d", [2, 8, 64, 1024])
def test_fwht_involution(d):
    torch.manual_seed(0)
    x = torch.randn(3, d)
    y = fwht(fwht(x))
    assert torch.allclose(y, d * x, atol=1e-4 * d, rtol=1e-5)


def test_fwht_ortho_is_orthonormal():
    torch.manual_seed(0)
    d = 32
    x = torch.randn(4, d)
    y = fwht_ortho(fwht_ortho(x))
    assert torch.allclose(y, x, atol=1e-5)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fwht(torch.randn(5, 6))


def test_fwht_preserves_leading_dims():
    x = torch.randn(2, 3, 4, 8)
    y = fwht(x)
    assert y.shape == x.shape


def test_fwht_does_not_mutate_input():
    x = torch.randn(10, 16)
    x0 = x.clone()
    _ = fwht(x)
    assert torch.equal(x, x0)


def test_next_power_of_two():
    assert next_power_of_two(1) == 1
    assert next_power_of_two(2) == 2
    assert next_power_of_two(3) == 4
    assert next_power_of_two(5) == 8
    assert next_power_of_two(17) == 32
    assert next_power_of_two(1024) == 1024
    assert next_power_of_two(1025) == 2048


def test_fwht_preserves_dtype_and_device():
    for dt in (torch.float32, torch.float64):
        x = torch.randn(8, dtype=dt)
        assert fwht(x).dtype == dt


# ---------------------------------------------------------------------------
# Autograd: the transform is self-adjoint (H symmetric, H @ H = d I), so
# backward is fwht(grad_output). Regression guard for a bug where the
# Triton backend silently detached the autograd graph on CUDA.
# ---------------------------------------------------------------------------


def _assert_fwht_backward(device: str) -> None:
    torch.manual_seed(0)
    d = 64
    x = torch.randn(3, d, device=device, requires_grad=True)
    y = fwht(x)
    assert y.requires_grad, "fwht output should require grad"
    assert y.grad_fn is not None, "fwht output should have a grad_fn"

    # Analytic backward: grad_x = grad_y @ H = fwht(grad_y).
    grad_out = torch.randn_like(y)
    (grad_x,) = torch.autograd.grad(y, x, grad_out)

    # Reference on CPU via the dense Hadamard matrix.
    H = _hadamard_matrix(d).to(device=device, dtype=grad_out.dtype)
    expected = grad_out @ H  # H is symmetric, so H.T == H.
    assert torch.allclose(grad_x, expected, atol=1e-4, rtol=1e-5)


def test_fwht_backward_cpu():
    _assert_fwht_backward("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_fwht_backward_cuda():
    _assert_fwht_backward("cuda")


def test_fwht_gradcheck_cpu():
    torch.manual_seed(0)
    d = 16
    x = torch.randn(2, d, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fwht, (x,), eps=1e-6, atol=1e-5)
