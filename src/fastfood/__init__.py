"""Fastfood: random-feature kernel approximation in PyTorch."""
from . import fwht as fwht  # noqa: PLC0414 -- expose the submodule
from .features import RBFSampler, RBFSinCosSampler
from .transform import Fastfood

__all__ = [
    "Fastfood",
    "RBFSampler",
    "RBFSinCosSampler",
    "fwht",
]

__version__ = "0.1.0"
