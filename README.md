# fastfood

PyTorch implementation of the Fastfood transform from Le, Sarlós & Smola, *"Fastfood — Approximating Kernel Expansions in Loglinear Time"* (ICML 2013).

Fastfood approximates a dense `n × d` Gaussian random matrix with the structured product

```
V = (1 / (σ √d)) · S · H · G · Π · H · B
```

where `H` is the Hadamard matrix, `B` is random ±1 diagonal, `Π` is a random permutation, and `G`, `S` are random diagonals drawn from Gaussian and chi distributions respectively. Applying `V` to a vector costs `O(d log d)` time and `O(d)` memory instead of the naive `O(n · d)` — which lets kernel random features scale to very high feature counts.

## Install

```bash
uv sync
```

For GPU you need a CUDA-matched `torch` wheel; `uv sync` will install CPU torch by default.

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

`Fastfood` is a standard `nn.Module` — buffers move with `.to(device)`, `state_dict()` round-trips, and `trainable=True` exposes `G` and `S` as parameters for end-to-end training.

## API

- `fastfood.Fastfood(in_features, out_features, sigma=1.0, *, seed=None, trainable=False, device=None, dtype=None, compile=True)` — the raw Fastfood projection. `compile=True` (default) wraps the forward in `torch.compile`; pass `compile=False` to skip JIT, or pass a dict of `torch.compile` options.
- `fastfood.RBFSampler(...)` — Rahimi-Recht random-phase cosine features.
- `fastfood.RBFSinCosSampler(...)` — concatenated sin/cos features (no bias; lower variance at equal parameter budget, but 2× the output dim).
- `fastfood.fwht.fwht(x)` / `fastfood.fwht.fwht_ortho(x)` — batched Fast Walsh-Hadamard Transform along the last axis. Auto-dispatches to a Triton kernel on CUDA (when Triton is importable) and the pure-PyTorch butterfly otherwise. `fastfood.fwht.fwht_eager(x)` forces the PyTorch path. The `Fastfood` module calls into `fastfood.fwht.fwht` by name, so replacing the attribute (e.g. `fastfood.fwht.fwht = my_cuda_fwht`) is sufficient to swap in a custom kernel globally — do this before the first forward if you also have `compile=True`, since the compiled graph captures the callee at trace time.

Input dimensions that aren't a power of two are zero-padded internally to the next power of two.

## Testing

```bash
uv run pytest
```

The tests cover four areas:

1. **FWHT correctness** (`test_fwht.py`) — matches an explicit Hadamard matrix, is its own inverse up to scale, handles arbitrary batch shapes.
2. **Fastfood shape / determinism / statistics** (`test_transform.py`) — `V` entries have variance `1/σ²`, linearity holds, seeding is reproducible, `state_dict` round-trips.
3. **NumPy parity** (`test_numpy_parity.py`) — the torch forward matches a pure-NumPy oracle to float32 precision when the parameter buffers are identical.
4. **Paper reproduction** (`test_benchmarks.py`) — per-entry kernel MSE matches explicit random kitchen sinks, convergence is `O(1/n)`, ridge regression predictions agree between Fastfood and RKS, and digit classification on `sklearn.datasets.load_digits` gets ~98 % accuracy (the paper reports ~97 % on USPS).

Run only the paper-reproduction tests with `uv run pytest -m paper`.

## Performance

Two opt-in accelerations ship on by default:

1. **Triton FWHT** (`fastfood/fwht_triton.py`) — a one-program-per-row butterfly in registers using `reshape/permute/split/join`. Used automatically on CUDA whenever `triton` can be imported (Triton ships with modern `torch`, so this is usually transparent).
2. **`torch.compile`** — the whole `Fastfood.forward` is wrapped in `torch.compile`, fusing the B/G/S multiplies, permutation, and FWHTs.

On an RTX 3080, batch 256, float32 (`uv run python scripts/bench.py`):

| size (in → out) | eager | +compile | +triton | compile+triton |
|---|---|---|---|---|
| 64 → 4096 | 0.572 ms | 0.207 ms (2.77×) | 0.135 ms (4.23×) | **0.101 ms (5.66×)** |
| 256 → 16384 | 2.676 ms | 0.727 ms (3.68×) | 0.375 ms (7.13×) | **0.288 ms (9.29×)** |
| 1024 → 65536 | 12.625 ms | 3.244 ms (3.89×) | 1.394 ms (9.06×) | **0.997 ms (12.66×)** |

**JIT cost** (paid once per unique shape, then cached):

- First `torch.compile` call in a fresh Python process: ~5–8 s (Inductor + Triton runtime warm-up).
- Subsequent `torch.compile` calls in the same process: ~200–800 ms.
- Triton FWHT kernel compile: ~60–220 ms per unique `d`, cached on disk in `~/.triton/cache`.

Pass `compile=False` to `Fastfood(...)` to skip JIT if you only run a few forwards. The Triton backend still kicks in on CUDA; disable it by setting `fastfood.fwht.fwht = fastfood.fwht.fwht_eager` before constructing the module.

The FWHT is isolated in `fastfood/fwht.py` and replaceable — a custom CUDA kernel can be dropped in there without touching the rest of the library.

## References

Quoc Le, Tamás Sarlós, Alex Smola. *Fastfood — Approximating Kernel Expansions in Loglinear Time.* ICML 2013.
