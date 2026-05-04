"""
Lottery Ticket Variance Slope Test
====================================

BPR's Topological Cramér–Rao bound (`bpr/info_geometry.py`):

    Var(θ̂) ≥ 1 / (N · F_max · |W|²)

predicts that systems with higher topological winding W have estimator
variance falling as 1/|W|².

Reading each iterative-magnitude-pruning (IMP) round as a unit increment
in topological refinement (each round commits one more bit about which
sub-circuit survives), seed-to-seed test-accuracy variance at pruning
round k should follow

    Var(k) ∝ 1 / k²

i.e. a log-log slope of -2 versus round number. This script measures
that slope on a CSV of IMP results and reports a verdict.

Verdicts
--------
    slope ∈ [-2.5, -1.5]   HIT      (consistent with BPR)
    slope ∈ [-1.5,  0.0]   WEAK     (some falloff but not Heisenberg-like)
    slope ≥ 0              FALSIFY  (no relationship; story is decorative)

Usage
-----
    # Smoke tests:
    python experiments/lottery_ticket_variance.py --demo bpr
    python experiments/lottery_ticket_variance.py --demo null

    # Real data (CSV with columns: round, seed, accuracy):
    python experiments/lottery_ticket_variance.py path/to/imp_results.csv

Data format
-----------
CSV with at minimum these columns (case-insensitive, common synonyms
accepted):
    round     IMP pruning round (1 = first prune; 0 also accepted but dropped
              before the log fit)
    seed      random seed / replicate id
    accuracy  test accuracy at convergence (0..1 or 0..100; units cancel)

Where to get data
-----------------
Frankle & Carbin's OpenLTH releases code, not pre-baked CSVs. Options:
    * Run OpenLTH yourself (CIFAR-10 + ResNet-20, ~5 seeds × 12 rounds).
    * Pull from any follow-up paper that published per-seed accuracies
      (e.g. "Stabilizing the Lottery Ticket Hypothesis", Frankle 2019,
      and the linear-mode-connectivity paper, Frankle 2020).
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import linregress


COL_ALIASES = {
    "round": ("round", "pruning_round", "iteration", "level", "k"),
    "seed": ("seed", "run", "replicate", "trial"),
    "accuracy": ("accuracy", "test_acc", "test_accuracy", "top1", "acc"),
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for canonical, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in cols:
                out[canonical] = df[cols[alias]]
                break
        else:
            raise SystemExit(
                f"missing column for '{canonical}' (tried {aliases}); "
                f"got columns {list(df.columns)}"
            )
    return pd.DataFrame(out)


def synthetic(kind: str, n_rounds: int = 12, n_seeds: int = 50,
              rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    rows = []
    base_acc = 0.92
    for k in range(1, n_rounds + 1):
        if kind == "bpr":
            sigma = 0.02 / k                  # Var ∝ 1/k²  ⇒  std ∝ 1/k
        elif kind == "null":
            sigma = 0.01                      # flat across rounds
        elif kind == "weak":
            sigma = 0.02 / np.sqrt(k)         # Var ∝ 1/k  (slope = -1)
        else:
            raise SystemExit(f"unknown demo {kind!r}")
        for s in range(n_seeds):
            rows.append({
                "round": k,
                "seed": s,
                "accuracy": base_acc + rng.normal(0.0, sigma),
            })
    return pd.DataFrame(rows)


def fit_slope(df: pd.DataFrame) -> Dict:
    df = df[df["round"] > 0]                  # log undefined at 0
    sigma = df.groupby("round")["accuracy"].std(ddof=1)
    sigma = sigma[sigma > 0]
    if len(sigma) < 3:
        raise SystemExit("need ≥3 pruning rounds with multiple seeds each")
    rounds = sigma.index.astype(float).to_numpy()
    var = sigma.to_numpy() ** 2
    fit = linregress(np.log(rounds), np.log(var))
    return {
        "rounds": rounds,
        "variance": var,
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue ** 2),
        "stderr": float(fit.stderr),
        "p": float(fit.pvalue),
    }


def verdict(slope: float, stderr: float) -> str:
    if -2.5 <= slope <= -1.5:
        return f"HIT      slope {slope:+.2f} ± {stderr:.2f} — consistent with BPR's -2"
    if slope < 0:
        return f"WEAK     slope {slope:+.2f} ± {stderr:.2f} — negative but not Heisenberg-like"
    return f"FALSIFY  slope {slope:+.2f} ± {stderr:.2f} — inconsistent with BPR's prediction"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", help="CSV of IMP results")
    ap.add_argument("--demo", choices=["bpr", "null", "weak"],
                    help="run on synthetic data of given shape")
    args = ap.parse_args(argv)

    if args.demo:
        df = synthetic(args.demo)
        print(f"# synthetic [{args.demo}]: {len(df)} rows")
    elif args.path:
        df = normalize_columns(pd.read_csv(args.path))
        print(f"# loaded {args.path}: {len(df)} rows")
    else:
        ap.error("supply --demo {bpr,null,weak} or a CSV path")

    res = fit_slope(df)

    print(f"rounds analysed: {len(res['rounds'])}")
    print("  round   variance")
    for k, v in zip(res["rounds"], res["variance"]):
        print(f"  {int(k):>5d}   {v:.3e}")
    print(
        f"\nlog-log fit:  slope = {res['slope']:+.3f} ± {res['stderr']:.3f}"
        f"   R² = {res['r2']:.3f}   p = {res['p']:.2e}"
    )
    print(verdict(res["slope"], res["stderr"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
