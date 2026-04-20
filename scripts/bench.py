"""Benchmark Fastfood forward across backends.

Backends:
    eager            - no compile, pure-PyTorch FWHT (baseline)
    compile          - torch.compile, pure-PyTorch FWHT
    triton           - no compile, Triton FWHT
    compile+triton   - torch.compile, Triton FWHT (library default on CUDA)

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --sizes 64:4096 256:16384
    uv run python scripts/bench.py --backends eager compile+triton
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from fastfood import Fastfood, fwht as fwht_mod


@dataclass
class SizeSpec:
    in_features: int
    out_features: int
    batch: int = 256

    def __str__(self) -> str:
        return f"in={self.in_features} out={self.out_features} batch={self.batch}"


def parse_sizes(specs: list[str]) -> list[SizeSpec]:
    out: list[SizeSpec] = []
    for s in specs:
        parts = s.split(":")
        if len(parts) == 2:
            out.append(SizeSpec(int(parts[0]), int(parts[1])))
        elif len(parts) == 3:
            out.append(SizeSpec(int(parts[0]), int(parts[1]), int(parts[2])))
        else:
            raise ValueError(f"bad size spec: {s}")
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_forward(fn, x: torch.Tensor, device: torch.device,
                 trials: int = 50, warmup: int = 10) -> tuple[float, float]:
    """Return (first_call_seconds, median_steady_seconds)."""
    sync(device)
    t0 = time.perf_counter()
    fn(x)
    sync(device)
    first = time.perf_counter() - t0

    for _ in range(warmup):
        fn(x)
    sync(device)

    times: list[float] = []
    for _ in range(trials):
        sync(device)
        t0 = time.perf_counter()
        fn(x)
        sync(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return first, times[len(times) // 2]


@contextmanager
def swap_fwht(use_triton: bool):
    original = fwht_mod.fwht
    if use_triton:
        fwht_mod.fwht = fwht_mod._fwht_triton  # type: ignore[attr-defined]
    else:
        fwht_mod.fwht = fwht_mod.fwht_eager
    try:
        yield
    finally:
        fwht_mod.fwht = original


BACKENDS = {
    "eager":          {"compile": False, "triton": False},
    "compile":        {"compile": True,  "triton": False},
    "triton":         {"compile": False, "triton": True},
    "compile+triton": {"compile": True,  "triton": True},
}


def bench_backend(backend: str, spec: SizeSpec, device: torch.device,
                  trials: int, warmup: int) -> dict:
    cfg = BACKENDS[backend]
    if cfg["triton"] and not getattr(fwht_mod, "_TRITON_AVAILABLE", False):
        raise RuntimeError("triton backend requested but Triton unavailable")
    with swap_fwht(cfg["triton"]):
        t0 = time.perf_counter()
        ff = Fastfood(
            spec.in_features, spec.out_features, sigma=1.0,
            seed=0, device=device, compile=cfg["compile"],
        )
        sync(device)
        build = time.perf_counter() - t0
        x = torch.randn(spec.batch, spec.in_features, device=device)
        first, med = time_forward(ff, x, device, trials=trials, warmup=warmup)
    return {"build_s": build, "first_call_s": first, "median_s": med}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=[
        "64:4096", "256:16384", "1024:65536",
    ])
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS.keys()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    device = torch.device(args.device)
    sizes = parse_sizes(args.sizes)

    print(f"device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print()

    header = f"{'size':<30} {'backend':<18} {'build(s)':>10} {'first(s)':>10} {'median(ms)':>12} {'jit(ms)':>10}"
    print(header)
    print("-" * len(header))

    for spec in sizes:
        eager_med = None
        for backend in args.backends:
            r = bench_backend(backend, spec, device, args.trials, args.warmup)
            jit_ms = max(0.0, r["first_call_s"] - r["median_s"]) * 1e3
            med_ms = r["median_s"] * 1e3
            speedup = ""
            if backend == "eager":
                eager_med = r["median_s"]
            elif eager_med is not None:
                speedup = f"  ({eager_med / r['median_s']:.2f}x)"
            print(f"{str(spec):<30} {backend:<18} "
                  f"{r['build_s']:>10.3f} {r['first_call_s']:>10.3f} "
                  f"{med_ms:>12.3f} {jit_ms:>10.1f}{speedup}")
        print()


if __name__ == "__main__":
    main()
