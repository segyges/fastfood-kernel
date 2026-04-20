"""Shape, determinism, and basic statistical correctness of Fastfood."""
from __future__ import annotations

import math

import pytest
import torch

from fastfood import Fastfood, RBFSampler, RBFSinCosSampler


@pytest.mark.parametrize(
    "in_feat,out_feat",
    [(8, 8), (8, 32), (10, 64), (10, 100), (128, 512), (65, 257)],
)
def test_fastfood_shape(in_feat, out_feat):
    ff = Fastfood(in_feat, out_feat, seed=0)
    x = torch.randn(13, in_feat)
    y = ff(x)
    assert y.shape == (13, out_feat)


def test_fastfood_batch_dims():
    ff = Fastfood(12, 48, seed=0)
    x = torch.randn(2, 3, 4, 12)
    y = ff(x)
    assert y.shape == (2, 3, 4, 48)


def test_fastfood_determinism():
    x = torch.randn(5, 20)
    ff1 = Fastfood(20, 128, seed=123)
    ff2 = Fastfood(20, 128, seed=123)
    assert torch.equal(ff1(x), ff2(x))


def test_fastfood_different_seeds_differ():
    x = torch.randn(5, 20)
    y1 = Fastfood(20, 128, seed=1)(x)
    y2 = Fastfood(20, 128, seed=2)(x)
    assert not torch.allclose(y1, y2)


def test_fastfood_rejects_wrong_input_dim():
    ff = Fastfood(16, 32, seed=0)
    with pytest.raises(ValueError):
        ff(torch.randn(5, 15))


def test_fastfood_validates_args():
    with pytest.raises(ValueError):
        Fastfood(0, 10)
    with pytest.raises(ValueError):
        Fastfood(10, 0)
    with pytest.raises(ValueError):
        Fastfood(10, 10, sigma=0.0)


def test_fastfood_entry_variance():
    """Entries of V should have variance ~1/sigma^2 (Gaussian scaling)."""
    torch.manual_seed(0)
    sigma = 1.7
    ff = Fastfood(64, 4096, sigma=sigma, seed=0)
    W = ff.weight_matrix()
    var = W.var().item()
    expected = 1.0 / sigma**2
    assert 0.8 * expected < var < 1.2 * expected, (var, expected)


def test_fastfood_linearity():
    ff = Fastfood(32, 64, seed=0)
    x = torch.randn(5, 32)
    y = torch.randn(5, 32)
    a, b = 0.7, -1.3
    out = ff(a * x + b * y)
    expected = a * ff(x) + b * ff(y)
    assert torch.allclose(out, expected, atol=1e-5)


def test_fastfood_weight_matrix_agrees_with_forward():
    ff = Fastfood(10, 40, seed=5)
    W = ff.weight_matrix()
    x = torch.randn(6, 10)
    assert torch.allclose(x @ W.T, ff(x), atol=1e-5)


def test_fastfood_trainable_flag():
    ff = Fastfood(8, 16, seed=0, trainable=True)
    assert isinstance(ff.G, torch.nn.Parameter)
    assert isinstance(ff.S, torch.nn.Parameter)
    ff_fixed = Fastfood(8, 16, seed=0)
    assert not isinstance(ff_fixed.G, torch.nn.Parameter)


def test_fastfood_state_dict_roundtrip():
    ff = Fastfood(16, 64, seed=42)
    state = ff.state_dict()
    ff2 = Fastfood(16, 64, seed=999)
    ff2.load_state_dict(state)
    x = torch.randn(3, 16)
    assert torch.equal(ff(x), ff2(x))


def test_rbf_kernel_diagonal_is_near_one():
    torch.manual_seed(0)
    rbf = RBFSampler(32, 2048, sigma=1.5, seed=0)
    x = torch.randn(50, 32)
    feat = rbf(x)
    diag = (feat * feat).sum(dim=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=0.1)


@pytest.mark.parametrize("cls", [RBFSampler, RBFSinCosSampler])
def test_rbf_kernel_converges(cls):
    torch.manual_seed(0)
    sigma = 1.2
    d = 16
    x = torch.randn(30, d)
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    K_true = torch.exp(-(diff**2).sum(-1) / (2 * sigma**2))

    errs = []
    for n in (256, 1024, 4096):
        s = cls(d, n, sigma=sigma, seed=n)
        K_a = s(x) @ s(x).T
        errs.append((K_a - K_true).abs().mean().item())
    # Error should shrink roughly sqrt(n): last should be < half the first.
    assert errs[-1] < 0.5 * errs[0], errs
    # Absolute sanity: 4096 features should give MAE < 0.05 on this scale.
    assert errs[-1] < 0.05, errs


def test_rbf_sincos_doubles_feature_count():
    rbf = RBFSinCosSampler(8, 32, seed=0)
    x = torch.randn(3, 8)
    y = rbf(x)
    assert y.shape == (3, 64)
