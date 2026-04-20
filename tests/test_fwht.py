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
