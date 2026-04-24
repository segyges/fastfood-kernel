"""End-to-end training on Fashion-MNIST through a Fastfood layer.

The Le-Sarlos-Smola paper only uses Fastfood inside closed-form kernel
ridge regression, so none of the paper's experiments ever call
``.backward()`` through the transform. That left the CUDA/Triton path's
missing autograd rule (fixed in 2bde461) completely uncovered.

This script is the smallest realistic thing that does exercise the
backward path: a single cheap conv layer feeds into a trainable Fastfood
projection, then a linear head, and the whole stack is trained with
cross-entropy + Adam.

    Conv2d(1->8, 3x3, stride 2)  ->  ReLU  ->  flatten   (1568 feats)
    Fastfood(1568 -> 4096, trainable=True)    (G, S are parameters)
    ReLU  ->  Linear(4096 -> 10)  ->  CE loss

Before the autograd fix the CUDA run would either raise "element 0 of
tensors does not require grad" at ``loss.backward()``, or (depending on
torch version) silently leave ``conv.weight.grad = None`` -- i.e. the
Fastfood layer would act as a detached random projection and the conv
below it would never update. We assert non-zero grads on both the conv
filters and the Fastfood ``G``/``S`` parameters at the end as a
regression guard.

Usage:
    uv run --extra benchmarks python scripts/train_fashion_mnist.py
    uv run --extra benchmarks python scripts/train_fashion_mnist.py --epochs 2 --device cuda
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn

from fastfood import Fastfood


# -- model -----------------------------------------------------------------
class ConvFastfoodNet(nn.Module):
    """Tiny conv -> Fastfood -> linear classifier.

    Deliberately cheap: the conv has 8 * 1 * 3 * 3 = 72 weights, the
    linear head dominates parameter count. The point is to have a
    differentiable layer *below* Fastfood so that a broken backward pass
    through the transform shows up as a dead conv.
    """

    def __init__(
        self,
        n_classes: int = 10,
        conv_channels: int = 8,
        n_features: int = 4096,
        sigma: float = 4.0,
        seed: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1)
        # 28x28 -> 14x14 after stride-2 conv with padding=1.
        flat_dim = conv_channels * 14 * 14
        self.ff = Fastfood(
            in_features=flat_dim,
            out_features=n_features,
            sigma=sigma,
            seed=seed,
            trainable=True,
            compile=False,  # training loop is small; skip the JIT tax
        )
        self.head = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(x))
        h = h.flatten(1)
        h = F.relu(self.ff(h))
        return self.head(h)


# -- data ------------------------------------------------------------------
def load_fashion_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch Fashion-MNIST via sklearn's OpenML cache.

    Returns (X_train, y_train, X_test, y_test) with X in [0, 1] and
    shape (N, 1, 28, 28), y of dtype long. The openml dataset is one
    flat table of 70 000 rows; we use the canonical 60 000 / 10 000
    split that the original Fashion-MNIST release defines.
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        raise SystemExit(
            "This script needs sklearn. Run it via:\n"
            "    uv run --extra benchmarks python scripts/train_fashion_mnist.py"
        ) from e

    print("loading Fashion-MNIST (cached on disk after first run)...")
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype("float32") / 255.0  # (70000, 784)
    y = ds.target.astype("int64")  # string labels '0'..'9' -> int via astype
    X = torch.from_numpy(X).view(-1, 1, 28, 28)
    y = torch.from_numpy(y)
    # Fashion-MNIST is shipped with the first 60k as train, last 10k as test.
    return X[:60_000], y[:60_000], X[60_000:], y[60_000:]


# -- train / eval ----------------------------------------------------------
@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch: int, device: str) -> float:
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], batch):
        xb = X[i : i + batch].to(device, non_blocking=True)
        yb = y[i : i + batch].to(device, non_blocking=True)
        logits = model(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return correct / X.shape[0]


def train(
    model: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    *,
    epochs: int,
    batch: int,
    lr: float,
    device: str,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_tr.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        t0 = time.perf_counter()
        running_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = X_tr[idx].to(device, non_blocking=True)
            yb = y_tr[idx].to(device, non_blocking=True)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        dt = time.perf_counter() - t0
        tr_acc = accuracy(model, X_tr[:5000], y_tr[:5000], batch, device)
        te_acc = accuracy(model, X_te, y_te, batch, device)
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"loss={running_loss / n_batches:.4f}  "
            f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  "
            f"({dt:.1f}s)"
        )


# -- regression guard: gradients actually reached every trainable tensor ---
def assert_grads_flowed(model: ConvFastfoodNet) -> None:
    """After a training step, every trainable param should have a grad.

    This is the explicit regression guard for the CUDA-fwht-backward
    bug: if autograd were detached at the Fastfood layer, the conv
    weights below it would still show ``grad=None`` (or all-zero) after
    ``loss.backward()``.
    """
    checks = {
        "conv.weight": model.conv.weight,
        "conv.bias": model.conv.bias,
        "ff.G": model.ff.G,
        "ff.S": model.ff.S,
        "head.weight": model.head.weight,
    }
    for name, p in checks.items():
        assert p.grad is not None, f"{name} has no gradient -- autograd was detached"
        assert torch.isfinite(p.grad).all(), f"{name}.grad has non-finite entries"
        assert p.grad.abs().sum().item() > 0, f"{name}.grad is all zero"
    print("grad-flow check passed: conv, fastfood (G, S), and head all received gradients.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-features", type=int, default=4096)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"device: {args.device}")

    X_tr, y_tr, X_te, y_te = load_fashion_mnist()
    print(f"train: {tuple(X_tr.shape)}  test: {tuple(X_te.shape)}")

    model = ConvFastfoodNet(
        n_features=args.n_features, sigma=args.sigma, seed=args.seed
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    train(
        model, X_tr, y_tr, X_te, y_te,
        epochs=args.epochs, batch=args.batch, lr=args.lr, device=args.device,
    )
    assert_grads_flowed(model)


if __name__ == "__main__":
    main()
