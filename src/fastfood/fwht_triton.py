"""Triton FWHT backend.

In-register butterfly using reshape/permute/split/join. One program per row,
BLOCK_SIZE = d. Suitable for d up to a few thousand on current GPUs; for
larger d we'd need a multi-stage kernel with shared-memory scratch.

Swap in via::

    from fastfood import fwht as fwht_mod
    from fastfood.fwht_triton import fwht as triton_fwht
    fwht_mod.fwht = triton_fwht
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

from .fwht import is_power_of_two

__all__ = ["fwht", "fwht_ortho"]


@triton.jit
def _fwht_stage(v, D: tl.constexpr, H: tl.constexpr):
    BLOCKS: tl.constexpr = D // (2 * H)
    v2 = tl.reshape(v, (BLOCKS, 2, H))
    v2 = tl.permute(v2, (0, 2, 1))
    a, b = tl.split(v2)
    v2 = tl.join(a + b, a - b)
    v2 = tl.permute(v2, (0, 2, 1))
    return tl.reshape(v2, (D,))


@triton.jit
def _fwht_kernel(
    x_ptr, y_ptr,
    stride_row,
    D: tl.constexpr,
    LOG_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, D)
    v = tl.load(x_ptr + pid * stride_row + offs)

    for stage in tl.static_range(LOG_D):
        v = _fwht_stage(v, D, 1 << stage)

    tl.store(y_ptr + pid * stride_row + offs, v)


def _log2_exact(n: int) -> int:
    if not is_power_of_two(n):
        raise ValueError(f"FWHT length must be a power of two, got {n}")
    return (n - 1).bit_length()


def fwht(x: Tensor) -> Tensor:
    """Unnormalized FWHT along the last axis (Triton backend)."""
    d = x.shape[-1]
    log_d = _log2_exact(d)
    if d == 1:
        return x.clone()

    lead = x.shape[:-1]
    x_flat = x.reshape(-1, d).contiguous()
    y = torch.empty_like(x_flat)
    n = x_flat.shape[0]
    if n == 0:
        return y.view(*lead, d)

    grid = (n,)
    _fwht_kernel[grid](
        x_flat, y,
        x_flat.stride(0),
        D=d, LOG_D=log_d,
        num_warps=min(max(d // 32, 1), 16),
    )
    return y.view(*lead, d)


def fwht_ortho(x: Tensor) -> Tensor:
    d = x.shape[-1]
    return fwht(x) / (d ** 0.5)
