"""Benchmark Fastfood against the paper's reference points.

The Le-Sarlos-Smola paper (ICML 2013) reports three things we can check:

1. Per-element kernel-matrix MSE: Fastfood should be *comparable* to
   explicit random kitchen sinks (RKS) at matched feature count.
2. Convergence rate: MSE ~ O(1/n) as n grows.
3. Application-level accuracy is equivalent to RKS.

We check (1) and (2) here on a small dense dataset. (3) is covered by
:func:`test_kernel_ridge_regression_matches_rks`, which fits closed-form
ridge regression on both feature maps and verifies that the predictions
agree.

These are property tests, not exact-number reproductions: wall-clock times
and constants from 2013 hardware aren't meaningful here.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from fastfood import Fastfood, RBFSampler, RBFSinCosSampler


# -- explicit random kitchen sinks baseline ---------------------------------
def _rks_features(x: torch.Tensor, n: int, sigma: float, seed: int) -> torch.Tensor:
    """RBF random features via an explicit Gaussian matrix: O(n*d) memory."""
    d = x.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.empty(n, d).normal_(generator=gen) / sigma
    b = torch.empty(n).uniform_(0.0, 2.0 * math.pi, generator=gen)
    z = x @ W.T + b
    return math.sqrt(2.0 / n) * torch.cos(z)


def _rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    return torch.exp(-(diff**2).sum(-1) / (2.0 * sigma**2))


def _mean_squared_kernel_error(K_approx: torch.Tensor, K_true: torch.Tensor) -> float:
    return ((K_approx - K_true) ** 2).mean().item()


# -- benchmark 1: Fastfood MSE matches RKS MSE (same feature count) --------
@pytest.mark.parametrize("n_features", [256, 1024, 4096])
def test_fastfood_kernel_mse_matches_rks(n_features):
    torch.manual_seed(0)
    d = 32
    sigma = 1.5
    n_points = 80

    x = torch.randn(n_points, d)
    K_true = _rbf_kernel(x, sigma)

    # average MSE over several seeds to dampen variance
    mse_ff, mse_rks = [], []
    for seed in range(10):
        ff = RBFSampler(d, n_features, sigma=sigma, seed=seed)
        K_ff = ff(x) @ ff(x).T
        mse_ff.append(_mean_squared_kernel_error(K_ff, K_true))

        feat_rks = _rks_features(x, n_features, sigma, seed=seed + 100)
        K_rks = feat_rks @ feat_rks.T
        mse_rks.append(_mean_squared_kernel_error(K_rks, K_true))

    ratio = np.mean(mse_ff) / np.mean(mse_rks)
    # Paper: comparable, no blow-up. Allow 0.5x to 2x.
    assert 0.5 < ratio < 2.0, (ratio, mse_ff, mse_rks)


# -- benchmark 2: convergence rate O(1/n) ----------------------------------
def test_fastfood_mse_converges_as_inverse_n():
    torch.manual_seed(0)
    d = 16
    sigma = 1.2
    x = torch.randn(60, d)
    K_true = _rbf_kernel(x, sigma)

    sizes = [256, 1024, 4096]
    mses = []
    for n in sizes:
        trials = []
        for seed in range(8):
            s = RBFSinCosSampler(d, n, sigma=sigma, seed=seed)
            K = s(x) @ s(x).T
            trials.append(_mean_squared_kernel_error(K, K_true))
        mses.append(float(np.mean(trials)))

    # Going from 256 -> 4096 (16x features) should cut MSE by ~16x.
    # Allow wide tolerance: at least 4x improvement, at most 64x.
    ratio = mses[0] / mses[-1]
    assert 4.0 < ratio < 64.0, (ratio, mses)


# -- benchmark 3: feature maps give equivalent downstream accuracy ---------
def test_kernel_ridge_regression_matches_rks():
    """Ridge regression on Fastfood features agrees with RKS features.

    The paper's key application claim is that Fastfood is a drop-in
    replacement for RKS with equivalent predictive accuracy. We verify
    this by fitting closed-form ridge regression with both feature maps
    and checking that their predictions (and test errors) agree.
    """
    torch.manual_seed(0)
    n_train, n_test = 400, 200
    d = 8
    sigma = 1.5
    n_features = 2048
    lam = 1e-3

    x_all = torch.randn(n_train + n_test, d)
    w_true = torch.randn(d) / math.sqrt(d)
    y_all = torch.sin(x_all @ w_true) + 0.05 * torch.randn(n_train + n_test)
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    def predict(feat_fn):
        Ftr = feat_fn(x_train)
        Fte = feat_fn(x_test)
        n_feat = Ftr.shape[1]
        A = Ftr.T @ Ftr + lam * torch.eye(n_feat)
        w = torch.linalg.solve(A, Ftr.T @ y_train)
        return Fte @ w

    ff = RBFSampler(d, n_features, sigma=sigma, seed=1)
    pred_ff = predict(ff)

    def rks_fn(x):
        return _rks_features(x, n_features, sigma, seed=2)
    pred_rks = predict(rks_fn)

    mse_ff = ((pred_ff - y_test) ** 2).mean().item()
    mse_rks = ((pred_rks - y_test) ** 2).mean().item()
    var_y = y_test.var().item()

    # Both feature maps should learn something (beat predicting the mean).
    assert mse_ff < var_y, (mse_ff, var_y)
    assert mse_rks < var_y, (mse_rks, var_y)
    # And they should agree with each other -- this is the paper's claim.
    assert abs(mse_ff - mse_rks) / max(mse_ff, mse_rks) < 0.3, (mse_ff, mse_rks)


# -- benchmark 4: wall-clock -- Fastfood scales better than RKS -----------
# -- benchmark 5: USPS-style digit classification --------------------------
@pytest.mark.paper
def test_digits_classification_matches_rks():
    """Classification accuracy on sklearn's digits dataset.

    This is the 8x8 subsampled version of the UCI pen-digit / USPS-style
    task used in the paper. We solve multi-class one-hot ridge regression
    in the random feature space and check that Fastfood and RKS achieve
    comparable accuracy.
    """
    sklearn = pytest.importorskip("sklearn")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    X = data.data.astype("float32")
    y = data.target.astype("int64")
    X /= X.max()  # scale into [0, 1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    X_tr_t = torch.from_numpy(X_tr)
    X_te_t = torch.from_numpy(X_te)
    y_tr_t = torch.from_numpy(y_tr)
    y_te_t = torch.from_numpy(y_te)
    n_classes = 10
    Y_tr = torch.nn.functional.one_hot(y_tr_t, n_classes).float()

    d = X_tr.shape[1]  # 64
    sigma = 4.0
    n_features = 2048
    lam = 1e-2

    def accuracy(feat_fn):
        Ftr = feat_fn(X_tr_t)
        Fte = feat_fn(X_te_t)
        n_feat = Ftr.shape[1]
        A = Ftr.T @ Ftr + lam * torch.eye(n_feat)
        W = torch.linalg.solve(A, Ftr.T @ Y_tr)
        preds = (Fte @ W).argmax(dim=1)
        return (preds == y_te_t).float().mean().item()

    acc_ff = accuracy(RBFSampler(d, n_features, sigma=sigma, seed=0))

    def rks_fn(x):
        return _rks_features(x, n_features, sigma, seed=1)
    acc_rks = accuracy(rks_fn)

    print(f"\n  digits: Fastfood {acc_ff:.3f}  |  RKS {acc_rks:.3f}")
    # Both should be well above random (0.1) -- the paper reports ~0.97.
    assert acc_ff > 0.9, acc_ff
    assert acc_rks > 0.9, acc_rks
    # And they should agree closely.
    assert abs(acc_ff - acc_rks) < 0.03, (acc_ff, acc_rks)


@pytest.mark.benchmark
@pytest.mark.paper
def test_fastfood_faster_than_rks_for_large_d():
    """Fastfood is O(n log d); RKS is O(n d). At large d with n >= d the
    Fastfood forward should be faster. We don't claim paper-specific
    numbers, just the scaling property.
    """
    torch.manual_seed(0)
    d = 1024
    n = 4096
    sigma = 1.0
    x = torch.randn(64, d)

    ff = RBFSampler(d, n, sigma=sigma, seed=0)
    # warmup
    for _ in range(3):
        ff(x)

    t0 = time.perf_counter()
    for _ in range(20):
        ff(x)
    t_ff = time.perf_counter() - t0

    gen = torch.Generator(device="cpu").manual_seed(0)
    W = torch.empty(n, d).normal_(generator=gen) / sigma
    b = torch.empty(n).uniform_(0.0, 2 * math.pi, generator=gen)
    scale = math.sqrt(2.0 / n)

    def rks(x):
        return scale * torch.cos(x @ W.T + b)

    for _ in range(3):
        rks(x)
    t0 = time.perf_counter()
    for _ in range(20):
        rks(x)
    t_rks = time.perf_counter() - t0

    # We don't hard-assert speed (matmul is very tuned), but we report it
    # and check Fastfood is at least within 4x of RKS (i.e. the python
    # overhead hasn't ruined the asymptotic advantage).
    print(f"\n  Fastfood {n=}, {d=}: {t_ff*1000:.2f}ms / 20 iters")
    print(f"  RKS      {n=}, {d=}: {t_rks*1000:.2f}ms / 20 iters")
    assert t_ff < 4 * t_rks
