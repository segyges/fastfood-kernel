"""Fast Walsh-Hadamard Transform.

Iterative butterfly over the last dimension. Works on arbitrary leading
dimensions (batching is free). The transform is the unnormalized Hadamard:
H H^T = d I. Divide by sqrt(d) for the orthonormal form.

A custom CUDA kernel can later be swapped in by replacing :func:`fwht`.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["fwht", "fwht_ortho", "is_power_of_two", "next_power_of_two"]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fwht(x: Tensor) -> Tensor:
    """Unnormalized FWHT along the last axis.

    The last dimension of ``x`` must be a power of two. Returns a new tensor;
    ``x`` is not modified. Supports arbitrary leading batch dimensions.
    """
    d = x.shape[-1]
    if not is_power_of_two(d):
        raise ValueError(f"FWHT length must be a power of two, got {d}")
    if d == 1:
        return x.clone()

    lead = x.shape[:-1]
    y = x.reshape(-1, d).clone()
    batch = y.shape[0]

    h = 1
    while h < d:
        y = y.view(batch, d // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(batch, d)
        h *= 2

    return y.view(*lead, d)


def fwht_ortho(x: Tensor) -> Tensor:
    """Orthonormal FWHT: ``fwht(x) / sqrt(d)``."""
    d = x.shape[-1]
    return fwht(x) / (d ** 0.5)
