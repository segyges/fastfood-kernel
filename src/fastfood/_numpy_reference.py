"""Pure NumPy reference implementation used to validate the torch version.

Single-precision, deterministic given a seed, and structured to mirror
:mod:`fastfood.transform`. Not exported from the top-level package -- this
is a correctness oracle, not a user-facing API.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["fwht_np", "FastfoodNumpy"]


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fwht_np(x: np.ndarray) -> np.ndarray:
    """Unnormalized FWHT along the last axis (iterative butterfly)."""
    d = x.shape[-1]
    if d & (d - 1) != 0:
        raise ValueError(f"FWHT length must be power of two, got {d}")
    y = x.copy()
    h = 1
    while h < d:
        lead = y.shape[:-1]
        y = y.reshape(*lead, d // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = np.stack((a + b, a - b), axis=-2).reshape(*lead, d)
        h *= 2
    return y


class FastfoodNumpy:
    """Numpy Fastfood, constructed to exactly match the torch reference.

    Given identical seeds and dtypes, buffers are identical to
    :class:`fastfood.transform.Fastfood` buffers at the bit level (up to
    the float conversion of ``B``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        *,
        seed: int | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = float(sigma)
        self.dtype = np.dtype(dtype)

        d = _next_power_of_two(in_features)
        n_blocks = math.ceil(out_features / d)
        self.d = d
        self.n_blocks = n_blocks

        rng = np.random.default_rng(seed)
        self.B = (rng.integers(0, 2, size=(n_blocks, d)) * 2 - 1).astype(self.dtype)
        self.Pi = np.stack(
            [rng.permutation(d) for _ in range(n_blocks)], axis=0
        ).astype(np.int64)
        self.G = rng.standard_normal((n_blocks, d)).astype(self.dtype)
        # chi_d via sqrt(sum of d squared normals)
        s_chi = np.sqrt(
            (rng.standard_normal((n_blocks, d, d)) ** 2).sum(axis=-1)
        ).astype(self.dtype)
        g_norm = np.linalg.norm(self.G, axis=-1, keepdims=True)
        self.S = (s_chi / g_norm).astype(self.dtype)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dim {self.in_features}, got {x.shape[-1]}"
            )
        x = x.astype(self.dtype, copy=False)
        lead = x.shape[:-1]
        d, n_blocks = self.d, self.n_blocks

        if self.in_features < d:
            pad = np.zeros((*lead, d - self.in_features), dtype=self.dtype)
            x = np.concatenate((x, pad), axis=-1)

        x = np.broadcast_to(x[..., None, :], (*lead, n_blocks, d)).copy()
        x = x * self.B
        x = fwht_np(x)
        pi = np.broadcast_to(self.Pi, (*lead, n_blocks, d))
        x = np.take_along_axis(x, pi, axis=-1)
        x = x * self.G
        x = fwht_np(x)
        x = x * self.S
        x = x / (self.sigma * math.sqrt(d))
        x = x.reshape(*lead, n_blocks * d)
        if x.shape[-1] != self.out_features:
            x = x[..., : self.out_features]
        return x
