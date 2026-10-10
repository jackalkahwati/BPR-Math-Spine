#!/usr/bin/env python3
"""Report-only benchmark for the Max-Cut phase-oscillator heuristic (bpr/optimization.py).

Compares the heuristic with named baselines on a few instances and prints cut values and ratios to a
certified upper bound. It asserts nothing; tests/test_optimization_maxcut.py holds the checks.

Baselines
---------
- random cut: the best of 20 uniformly random partitions (expected value |E|/2);
- 1-flip local search from a random partition (guarantees at least half the total weight);
- GW rounding via Burer-Monteiro: a rank-r factorization of the Goemans-Williamson SDP, solved by projected
  gradient, then the best of 100 random hyperplanes. The SDP is not solved to certified optimality, so this
  baseline is labelled uncertified;
- upper bound: Mohar-Poljak, max cut <= (n/4) lambda_max(L), with L the graph Laplacian (certified).

The heuristic is run as the module ships (default parameters), with and without 1-flip polish.

Usage: python3 scripts/benchmark_maxcut.py [--quick] [--runs K]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpr.optimization import MaxCutBPR, local_search_polish, random_graph


def cut_value(W, x):
    return 0.5 * float(np.sum(W * np.abs(x[:, None] - x[None, :])))


def random_regular_graph(n, d, seed):
    """Random d-regular simple graph by the configuration model with rejection (n d even)."""
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        stubs = rng.permutation(np.repeat(np.arange(n), d)).reshape(-1, 2)
        a, b = stubs[:, 0], stubs[:, 1]
        if np.any(a == b):
            continue
        W = np.zeros((n, n))
        W[a, b] += 1
        W[b, a] += 1
        if W.max() == 1:
            return W
    raise RuntimeError("no simple graph found")


def mohar_poljak_bound(W):
    L = np.diag(W.sum(axis=1)) - W
    return W.shape[0] / 4.0 * float(np.linalg.eigvalsh(L)[-1])


def gw_burer_monteiro(W, rank, seed, iters=500, hyperplanes=100):
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    V = rng.standard_normal((n, rank))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    step = 0.5 / max(1.0, float(np.abs(W).sum(axis=1).max()))
    for _ in range(iters):
        V -= step * (W @ V)          # gradient of sum_ij W_ij v_i . v_j (up to a factor 2)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    best, best_x = -1.0, None
    for _ in range(hyperplanes):
        x = (V @ rng.standard_normal(rank) > 0).astype(int)
        c = cut_value(W, x)
        if c > best:
            best, best_x = c, x
    return best, best_x


def run_instance(name, W, runs):
    n = W.shape[0]
    rng = np.random.default_rng(0)
    t0 = time.time()
    rand = max(cut_value(W, rng.integers(0, 2, n)) for _ in range(20))
    ls = [cut_value(W, local_search_polish(W, rng.integers(0, 2, n))) for _ in range(runs)]
    raw, pol = [], []
    for s in range(runs):
        mc = MaxCutBPR(W)
        x = mc.extract_cut(mc.solve(seed=s))
        raw.append(mc.cut_value(x))
        pol.append(mc.cut_value(local_search_polish(W, x)))
    gw, _ = gw_burer_monteiro(W, rank=int(np.ceil(np.sqrt(2 * n))) + 1, seed=0)
    ub = mohar_poljak_bound(W)
    print(f"\n{name}: n = {n}, |E| = {int(W.sum() / 2)}, total weight / 2 = {W.sum() / 4:.1f}")
    rows = [
        ("random cut (best of 20)", [rand]),
        ("1-flip local search", ls),
        ("phase oscillator, raw", raw),
        ("phase oscillator + 1-flip", pol),
        ("GW via Burer-Monteiro (uncertified)", [gw]),
    ]
    for label, vals in rows:
        v = np.array(vals)
        print(f"  {label:38s} mean {v.mean():9.1f}  min {v.min():9.1f}  max {v.max():9.1f}"
              f"  mean/UB {v.mean() / ub:.4f}")
    print(f"  {'Mohar-Poljak upper bound (certified)':38s} {ub:9.1f}   ({time.time() - t0:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="small instances only")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    sizes = [100] if args.quick else [100, 500]
    for n in sizes:
        run_instance(f"G({n}, 1/2), seed 0", random_graph(n, 0.5, seed=0), args.runs)
        run_instance(f"random 3-regular, n = {n}, seed 0", random_regular_graph(n, 3, seed=0), args.runs)
    print("\nRatios are to a certified upper bound, not to the optimum. No approximation guarantee is implied.")


if __name__ == "__main__":
    main()
