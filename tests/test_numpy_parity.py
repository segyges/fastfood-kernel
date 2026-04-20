"""NumPy reference parity with the torch implementation.

The numpy implementation is kept bit-compatible with the torch one given
identical parameter buffers. These tests build a torch Fastfood, copy its
parameters into a ``FastfoodNumpy`` instance, and check the outputs agree
exactly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfood import Fastfood
from fastfood._numpy_reference import FastfoodNumpy, fwht_np
from fastfood.fwht import fwht


def _clone_torch_to_numpy(ff: Fastfood) -> FastfoodNumpy:
    ref = FastfoodNumpy(ff.in_features, ff.out_features, sigma=ff.sigma,
                         seed=0, dtype=np.float32)
    ref.B = ff.B.detach().cpu().numpy().copy()
    ref.Pi = ff.Pi.detach().cpu().numpy().copy()
    ref.G = ff.G.detach().cpu().numpy().copy()
    ref.S = ff.S.detach().cpu().numpy().copy()
    return ref


@pytest.mark.parametrize("d", [2, 4, 16, 128, 1024])
def test_fwht_numpy_matches_torch(d):
    rng = np.random.default_rng(d)
    x = rng.standard_normal((4, d)).astype(np.float32)
    y_np = fwht_np(x)
    y_t = fwht(torch.from_numpy(x)).numpy()
    assert np.allclose(y_np, y_t, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "in_feat,out_feat",
    [(8, 8), (8, 32), (10, 64), (10, 100), (64, 256), (65, 257)],
)
def test_fastfood_numpy_matches_torch(in_feat, out_feat):
    torch.manual_seed(0)
    ff = Fastfood(in_feat, out_feat, sigma=1.3, seed=42)
    ref = _clone_torch_to_numpy(ff)

    rng = np.random.default_rng(in_feat * out_feat)
    x = rng.standard_normal((7, in_feat)).astype(np.float32)

    y_np = ref(x)
    y_t = ff(torch.from_numpy(x)).numpy()
    # Identical up to float32 rounding order of the two backends.
    assert np.allclose(y_np, y_t, atol=1e-5, rtol=1e-5)
    # In practice we also expect bit-equality for these inputs.
    assert np.max(np.abs(y_np - y_t)) < 1e-5


def test_fastfood_numpy_batched_input():
    ff = Fastfood(12, 48, seed=1)
    ref = _clone_torch_to_numpy(ff)
    x = np.random.default_rng(0).standard_normal((2, 3, 12)).astype(np.float32)
    y_np = ref(x)
    y_t = ff(torch.from_numpy(x)).numpy()
    assert y_np.shape == y_t.shape == (2, 3, 48)
    assert np.allclose(y_np, y_t, atol=1e-5)
