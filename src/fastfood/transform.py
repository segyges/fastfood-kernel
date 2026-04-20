"""Fastfood random projection.

Implements the factorization

    V = (1 / (sigma * sqrt(d))) * S @ H @ G @ Pi @ H @ B

from Le, Sarlos, Smola, "Fastfood - Approximating Kernel Expansions in
Loglinear Time" (ICML 2013). Applied to a vector, each step costs O(d) or
O(d log d) (the two FWHTs), yielding an O(n log d) projection to n features.

Parameter shapes are ``(n_blocks, d)`` where ``d`` is the next power of two at
or above ``in_features`` and ``n_blocks = ceil(out_features / d)``. Output is
sliced to ``out_features`` along the last axis.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from . import fwht as _fwht_module
from .fwht import next_power_of_two

__all__ = ["Fastfood"]


def _sample_chi(shape: tuple[int, ...], d: int, generator: torch.Generator | None,
                device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sample chi-distributed values with d degrees of freedom.

    Uses the defining identity: if Z_1, ..., Z_d ~ N(0, 1) iid then
    sqrt(sum Z_i^2) ~ chi_d. This keeps the sampling bound to the provided
    generator (unlike ``torch.distributions.Chi2``).
    """
    normals = torch.empty((*shape, d), device=device, dtype=dtype)
    normals.normal_(generator=generator)
    return normals.pow(2).sum(dim=-1).sqrt()


class Fastfood(nn.Module):
    """Fastfood linear projection from ``in_features`` to ``out_features``.

    Parameters are stored as non-trainable buffers by default; set
    ``trainable=True`` to expose ``G`` and ``S`` as ``nn.Parameter`` (B, Pi
    stay fixed since they are discrete).

    Args:
        in_features: Input dimension. Inputs are zero-padded to the next
            power of two internally.
        out_features: Output dimension.
        sigma: Kernel bandwidth. Row variance of V is ``1 / sigma^2``.
        seed: If not None, used to seed a ``torch.Generator`` for
            reproducible parameter sampling.
        trainable: If True, ``G`` and ``S`` become trainable parameters.
        device, dtype: Where to place the buffers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        *,
        seed: int | None = None,
        trainable: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        compile: bool | dict = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        dtype = dtype or torch.get_default_dtype()
        device = torch.device(device) if device is not None else torch.device("cpu")

        d = next_power_of_two(in_features)
        n_blocks = math.ceil(out_features / d)

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = float(sigma)
        self.d = d
        self.n_blocks = n_blocks

        generator: torch.Generator | None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        else:
            generator = None

        # B: ±1 Rademacher on CPU (integer ops on CPU, move later)
        b_raw = torch.empty((n_blocks, d), dtype=torch.int64)
        b_raw.random_(0, 2, generator=generator)
        B = (b_raw * 2 - 1).to(dtype=dtype, device=device)

        # Pi: random permutation per block
        Pi = torch.empty((n_blocks, d), dtype=torch.long)
        for k in range(n_blocks):
            Pi[k] = torch.randperm(d, generator=generator)
        Pi = Pi.to(device=device)

        # G: iid standard normal
        G = torch.empty((n_blocks, d), dtype=dtype)
        G.normal_(generator=generator)

        # S: s_i / ||G_block||_2, with s_i ~ chi_d
        s_chi = _sample_chi((n_blocks, d), d, generator, device=torch.device("cpu"), dtype=dtype)
        g_norm = G.norm(dim=-1, keepdim=True)
        S = (s_chi / g_norm).to(device=device)
        G = G.to(device=device)

        self.register_buffer("B", B)
        self.register_buffer("Pi", Pi)
        if trainable:
            self.G = nn.Parameter(G)
            self.S = nn.Parameter(S)
        else:
            self.register_buffer("G", G)
            self.register_buffer("S", S)

        self._compiled_core = None
        if compile:
            opts = compile if isinstance(compile, dict) else {}
            self._compiled_core = torch.compile(self._core, **opts)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sigma={self.sigma}, d={self.d}, n_blocks={self.n_blocks}"
        )

    def _core(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        d, n_blocks = self.d, self.n_blocks

        if self.in_features < d:
            pad = x.new_zeros(*lead, d - self.in_features)
            x = torch.cat((x, pad), dim=-1)

        x = x.unsqueeze(-2).expand(*lead, n_blocks, d)

        x = x * self.B
        x = _fwht_module.fwht(x)

        pi = self.Pi
        for _ in range(len(lead)):
            pi = pi.unsqueeze(0)
        pi = pi.expand(*lead, n_blocks, d)
        x = torch.gather(x, -1, pi)

        x = x * self.G
        x = _fwht_module.fwht(x)
        x = x * self.S
        x = x / (self.sigma * math.sqrt(d))

        x = x.reshape(*lead, n_blocks * d)
        if x.shape[-1] != self.out_features:
            x = x[..., : self.out_features]
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dim {self.in_features}, got {x.shape[-1]}"
            )
        if self._compiled_core is not None:
            return self._compiled_core(x)
        return self._core(x)

    def weight_matrix(self) -> Tensor:
        """Return the explicit ``(out_features, in_features)`` projection.

        Useful for testing. Cost is O(out_features * d log d + out_features^2).
        """
        eye = torch.eye(self.in_features, device=self.B.device, dtype=self.B.dtype)
        return self.forward(eye).T.contiguous()
