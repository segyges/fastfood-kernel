"""Device transfer, dtype, and custom-FWHT hook."""
from __future__ import annotations

import pytest
import torch

import fastfood.fwht as fwht_mod
from fastfood import Fastfood, RBFSampler


def test_fastfood_to_device_and_back():
    ff = Fastfood(16, 64, seed=0)
    x = torch.randn(4, 16)
    y_cpu = ff(x)

    ff_moved = Fastfood(16, 64, seed=0)
    ff_moved.to("cpu")  # no-op but exercises the path
    assert torch.equal(y_cpu, ff_moved(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_fastfood_cuda_matches_cpu():
    ff = Fastfood(32, 256, sigma=1.1, seed=7)
    x = torch.randn(5, 32)
    y_cpu = ff(x)

    ff.to("cuda")
    y_cuda = ff(x.to("cuda")).cpu()
    assert torch.allclose(y_cpu, y_cuda, atol=1e-4)


def test_fastfood_float64_dtype():
    ff = Fastfood(16, 32, seed=0, dtype=torch.float64)
    x = torch.randn(3, 16, dtype=torch.float64)
    y = ff(x)
    assert y.dtype == torch.float64
    assert y.shape == (3, 32)


def test_rbf_sampler_follows_fastfood_dtype():
    rbf = RBFSampler(8, 16, seed=0, dtype=torch.float64)
    x = torch.randn(4, 8, dtype=torch.float64)
    y = rbf(x)
    assert y.dtype == torch.float64


def test_fwht_hook_is_live_replaceable():
    """Monkey-patching fastfood.fwht.fwht should affect Fastfood.forward.

    This is the escape hatch for swapping in a custom CUDA butterfly kernel.
    """
    original = fwht_mod.fwht
    called = {"n": 0}

    def wrapped(x):
        called["n"] += 1
        return original(x)

    fwht_mod.fwht = wrapped
    try:
        ff = Fastfood(8, 16, seed=0)
        _ = ff(torch.randn(3, 8))
        # Fastfood applies FWHT twice.
        assert called["n"] >= 2
    finally:
        fwht_mod.fwht = original
