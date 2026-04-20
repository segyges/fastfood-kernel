"""RBF-kernel random feature maps backed by :class:`Fastfood`.

For the Gaussian kernel ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))`` the
Rahimi-Recht construction is

    phi(x) = sqrt(2 / n) * cos(V x + b),

with ``V`` rows drawn iid from ``N(0, sigma^{-2} I)`` and ``b`` uniform on
``[0, 2 pi)``. Fastfood supplies ``V`` in O(n log d) time. The sin/cos
variant (no bias) avoids the extra variance from the random phase at the
cost of doubling the feature count.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .transform import Fastfood

__all__ = ["RBFSampler", "RBFSinCosSampler"]


class RBFSampler(nn.Module):
    """RBF random features with random-phase cosines.

    ``out_features`` cosine features approximating the Gaussian kernel with
    bandwidth ``sigma``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.projection = Fastfood(
            in_features, out_features, sigma=sigma, seed=seed, device=device, dtype=dtype
        )
        dtype = dtype or torch.get_default_dtype()
        device = torch.device(device) if device is not None else torch.device("cpu")

        gen: torch.Generator | None
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) + 1)
        else:
            gen = None
        bias = torch.empty(out_features, dtype=dtype)
        bias.uniform_(0.0, 2.0 * math.pi, generator=gen)
        self.register_buffer("bias", bias.to(device=device))

        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        z = self.projection(x) + self.bias
        return math.sqrt(2.0 / self.out_features) * torch.cos(z)


class RBFSinCosSampler(nn.Module):
    """RBF random features using concatenated sin/cos pairs.

    Produces ``2 * out_features`` features. This is unbiased and has lower
    variance than :class:`RBFSampler` at equal feature budget.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.projection = Fastfood(
            in_features, out_features, sigma=sigma, seed=seed, device=device, dtype=dtype
        )
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        z = self.projection(x)
        scale = 1.0 / math.sqrt(self.out_features)
        return scale * torch.cat((torch.cos(z), torch.sin(z)), dim=-1)
