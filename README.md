# fastfood

PyTorch implementation of the Fastfood transform from Le, Sarlós & Smola,
*"Fastfood — Approximating Kernel Expansions in Loglinear Time"* (ICML 2013).

Fastfood approximates a dense `n × d` Gaussian random matrix with the structured
product

```
V = (1 / (σ √d)) · S · H · G · Π · H · B
```

where `H` is the Hadamard matrix, `B` is random ±1 diagonal, `Π` is a random
permutation, and `G`, `S` are random diagonals drawn from Gaussian and
chi distributions respectively. Applying `V` to a vector costs
`O(d log d)` time and `O(d)` memory instead of the naive `O(n · d)` — which
lets kernel random features scale to very high feature counts.

## Install

```bash
uv sync
```

For GPU you need a CUDA-matched `torch` wheel; `uv sync` will install CPU
torch by default.

## Quickstart

```python
import torch
from fastfood import Fastfood, RBFSampler

# Raw Fastfood projection: 64-dim input -> 4096 random features.
ff = Fastfood(in_features=64, out_features=4096, sigma=1.0, seed=0)
x = torch.randn(32, 64)
y = ff(x)            # (32, 4096)

# RBF kernel features: exp(-||x-y||² / (2σ²)) ≈ ⟨φ(x), φ(y)⟩.
rbf = RBFSampler(in_features=64, out_features=4096, sigma=1.0, seed=0)
features = rbf(x)    # (32, 4096)
K_approx = features @ features.T  # approximates the Gaussian kernel matrix
```

`Fastfood` is a standard `nn.Module` — buffers move with `.to(device)`,
`state_dict()` round-trips, and `trainable=True` exposes `G` and `S` as
parameters for end-to-end training.

## API

- `fastfood.Fastfood(in_features, out_features, sigma=1.0, *, seed=None, trainable=False, device=None, dtype=None)` — the raw Fastfood projection.
- `fastfood.RBFSampler(...)` — Rahimi-Recht random-phase cosine features.
- `fastfood.RBFSinCosSampler(...)` — concatenated sin/cos features (no bias; lower variance at equal parameter budget, but 2× the output dim).
- `fastfood.fwht.fwht(x)` / `fastfood.fwht.fwht_ortho(x)` — batched Fast Walsh-Hadamard Transform along the last axis. The `Fastfood` module calls into `fastfood.fwht.fwht` by name, so replacing the attribute (e.g. `fastfood.fwht.fwht = my_cuda_fwht`) is sufficient to swap in a custom CUDA kernel globally.

Input dimensions that aren't a power of two are zero-padded internally to the
next power of two.

## Testing

```bash
uv run pytest
```

The tests cover four areas:

1. **FWHT correctness** (`test_fwht.py`) — matches an explicit Hadamard
   matrix, is its own inverse up to scale, handles arbitrary batch shapes.
2. **Fastfood shape / determinism / statistics** (`test_transform.py`) —
   `V` entries have variance `1/σ²`, linearity holds, seeding is
   reproducible, `state_dict` round-trips.
3. **NumPy parity** (`test_numpy_parity.py`) — the torch forward matches
   a pure-NumPy oracle to float32 precision when the parameter buffers are
   identical.
4. **Paper reproduction** (`test_benchmarks.py`) — per-entry kernel MSE
   matches explicit random kitchen sinks, convergence is `O(1/n)`, ridge
   regression predictions agree between Fastfood and RKS, and digit
   classification on `sklearn.datasets.load_digits` gets ~98 % accuracy
   (the paper reports ~97 % on USPS).

Run only the paper-reproduction tests with `uv run pytest -m paper`.

## Performance

Current FWHT is a pure-PyTorch butterfly, which is GPU-compatible but not
optimal. Because `torch.matmul` is extremely well-tuned, dense RKS can beat
Fastfood on CPU at moderate sizes. The scaling advantage appears at very
large `d` (and on GPU where matmul memory bandwidth dominates).

The FWHT is isolated in `fastfood/fwht.py` and replaceable — a custom CUDA
kernel can be dropped in there without touching the rest of the library.

## References

Quoc Le, Tamás Sarlós, Alex Smola. *Fastfood — Approximating Kernel
Expansions in Loglinear Time.* ICML 2013.
