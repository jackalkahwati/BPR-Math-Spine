"""
Lottery Ticket Variance Slope Test
====================================

BPR's Topological Cramér–Rao bound (`bpr/info_geometry.py`):

    Var(θ̂) ≥ 1 / (N · F_max · |W|²)

predicts that systems with higher topological winding W have estimator
variance falling as 1/|W|².

If iterative-magnitude-pruning (IMP) is read as a procedure that
incrementally raises an effective topological order, seed-to-seed
variance of the trained model's metric should fall as a power law in
whatever proxy for |W| is plotted on the x-axis. BPR's prediction is
slope = -2.

This script measures that slope on a CSV of IMP results and reports a
verdict.  Three scientific guards are built in:

  1. Multiple x-axes.  --x can be `round` (procedural index), `density`
     (fraction of weights remaining; the natural physical x-axis), or
     `sparsity` (1 - density). They are not log-linearly related, so
     the slopes mean different things and should be reported separately.

  2. Multiple metrics.  --metric can be `accuracy` (default) or `loss`.
     Accuracy is bounded above and so its variance compresses near a
     ceiling; loss is not. A slope that holds in both metrics is much
     harder to dismiss as a bounded-metric artefact.

  3. Condition contrast.  If the CSV has a `condition` column the
     script fits a slope per condition and the verdict only HITs when
     the winning-ticket arm has BPR-shaped slope and the baselines do
     not. Useful columns to populate: `winning_ticket`, `random_mask`,
     `rerandomized_init`. With a single arm the script falls back to
     the unconditional verdict.

Verdicts (per arm)
------------------
    slope ∈ [-2.5, -1.5]   HIT      (consistent with BPR)
    slope ∈ [-1.5,  0.0]   WEAK     (some falloff but not Heisenberg-like)
    slope ≥ 0              FALSIFY  (no relationship; story is decorative)

Usage
-----
    # Smoke tests:
    python experiments/lottery_ticket_variance.py --demo bpr
    python experiments/lottery_ticket_variance.py --demo null
    python experiments/lottery_ticket_variance.py --demo contrast    # winner vs baseline

    # Real data:
    python experiments/lottery_ticket_variance.py imp_results.csv
    python experiments/lottery_ticket_variance.py imp_results.csv --x density --metric loss

Data format
-----------
CSV columns (case-insensitive, common synonyms accepted):
    round       IMP pruning round (1 = first prune; 0 dropped before log fit)
    seed        random seed / replicate id
    accuracy    test accuracy at convergence              (used if --metric accuracy)
    loss        test loss at convergence  (optional)      (used if --metric loss)
    density     fraction of weights remaining (optional)  (used if --x density)
    sparsity    1 - density                   (optional)  (used if --x sparsity)
    condition   arm label (optional; e.g. winning_ticket / random_mask)

Where to get data
-----------------
Frankle & Carbin's OpenLTH releases code, not pre-baked CSVs. Options:
    * Run OpenLTH yourself (CIFAR-10 + ResNet-20, 30+ seeds × 12 rounds,
      with a random-mask control arm at matched sparsity).
    * Pull from any follow-up paper that published per-seed accuracies
      (Frankle 2019 "Stabilizing the LTH"; Frankle 2020 LMC paper).
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress


COL_ALIASES = {
    "round":     ("round", "pruning_round", "iteration", "level", "k"),
    "seed":      ("seed", "run", "replicate", "trial"),
    "accuracy":  ("accuracy", "test_acc", "test_accuracy", "top1", "acc"),
    "loss":      ("loss", "test_loss", "val_loss"),
    "density":   ("density", "remaining_fraction", "weights_remaining"),
    "sparsity":  ("sparsity", "pruned_fraction"),
    "condition": ("condition", "arm", "treatment"),
}
REQUIRED = ("round", "seed")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out: Dict[str, pd.Series] = {}
    for canonical, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in cols:
                out[canonical] = df[cols[alias]]
                break
    missing = [c for c in REQUIRED if c not in out]
    if missing:
        raise SystemExit(
            f"missing required column(s) {missing}; got {list(df.columns)}"
        )
    return pd.DataFrame(out)


def synthetic(kind: str, n_rounds: int = 12, n_seeds: int = 50,
              rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    rows: List[dict] = []
    base_acc = 0.92
    prune_rate = 0.2  # 20% of remaining weights pruned each round (OpenLTH default)

    def emit(condition: str, sigma_fn):
        for k in range(1, n_rounds + 1):
            sigma = sigma_fn(k)
            density = (1 - prune_rate) ** k
            for s in range(n_seeds):
                noise = rng.normal(0.0, sigma)
                rows.append({
                    "round": k, "seed": s,
                    "accuracy": base_acc + noise,
                    "loss": 0.30 - noise,
                    "density": density,
                    "sparsity": 1.0 - density,
                    "condition": condition,
                })

    if kind == "bpr":
        emit("winning_ticket", lambda k: 0.02 / k)
    elif kind == "weak":
        emit("winning_ticket", lambda k: 0.02 / np.sqrt(k))
    elif kind == "null":
        emit("winning_ticket", lambda k: 0.01)
    elif kind == "contrast":
        emit("winning_ticket", lambda k: 0.02 / k)         # BPR-shape
        emit("random_mask",    lambda k: 0.012)            # flat baseline
    else:
        raise SystemExit(f"unknown demo {kind!r}")
    return pd.DataFrame(rows)


def fit_slope(df: pd.DataFrame, x_col: str, metric: str) -> Dict:
    if x_col not in df.columns:
        raise SystemExit(f"x-axis '{x_col}' not in CSV (have {list(df.columns)})")
    if metric not in df.columns:
        raise SystemExit(f"metric '{metric}' not in CSV (have {list(df.columns)})")
    df = df[df[x_col] > 0].copy()
    sigma = df.groupby(x_col)[metric].std(ddof=1)
    sigma = sigma[sigma > 0]
    if len(sigma) < 3:
        raise SystemExit(f"need ≥3 unique {x_col} values with multiple seeds")
    x = sigma.index.astype(float).to_numpy()
    var = sigma.to_numpy() ** 2
    fit = linregress(np.log(x), np.log(var))
    return {
        "x_col": x_col, "metric": metric,
        "x": x, "variance": var,
        "slope": float(fit.slope),
        "stderr": float(fit.stderr),
        "r2": float(fit.rvalue ** 2),
        "p": float(fit.pvalue),
        "n_points": int(len(x)),
    }


def classify(slope: float) -> str:
    if -2.5 <= slope <= -1.5:
        return "HIT"
    if slope < 0:
        return "WEAK"
    return "FALSIFY"


def report(res: Dict, label: str = "") -> str:
    tag = classify(res["slope"])
    head = f"[{label}] " if label else ""
    return (
        f"{head}{tag:<7}  {res['metric']} vs {res['x_col']}: "
        f"slope = {res['slope']:+.3f} ± {res['stderr']:.3f}, "
        f"R² = {res['r2']:.3f}, n = {res['n_points']}, p = {res['p']:.2e}"
    )


def contrast_verdict(by_condition: Dict[str, Dict]) -> str:
    """A real HIT requires winning-ticket arm BPR-shaped AND baselines not."""
    winner_keys = [k for k in by_condition if "winning" in k.lower() or "ticket" in k.lower()]
    other_keys  = [k for k in by_condition if k not in winner_keys]
    if not winner_keys or not other_keys:
        return "(single arm — no contrast available)"
    winner = by_condition[winner_keys[0]]
    if classify(winner["slope"]) != "HIT":
        return "CONTRAST FALSIFY  winning-ticket arm itself is not BPR-shaped"
    bad = [k for k in other_keys if classify(by_condition[k]["slope"]) == "HIT"]
    if bad:
        return (
            f"CONTRAST FALSIFY  baseline arm(s) {bad} also show BPR-shape; "
            "effect is not specific to winning tickets"
        )
    return "CONTRAST HIT  winning-ticket arm BPR-shaped, baselines are not"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", help="CSV of IMP results")
    ap.add_argument("--demo", choices=["bpr", "weak", "null", "contrast"],
                    help="run on synthetic data of given shape")
    ap.add_argument("--x", default="round", choices=["round", "density", "sparsity"],
                    help="x-axis for the slope fit (default: round)")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "loss"],
                    help="metric whose seed-variance is fit (default: accuracy)")
    args = ap.parse_args(argv)

    if args.demo:
        df = synthetic(args.demo)
        print(f"# synthetic [{args.demo}]: {len(df)} rows")
    elif args.path:
        df = normalize_columns(pd.read_csv(args.path))
        print(f"# loaded {args.path}: {len(df)} rows")
    else:
        ap.error("supply --demo {bpr,weak,null,contrast} or a CSV path")

    if "condition" in df.columns and df["condition"].nunique() > 1:
        results = {
            cond: fit_slope(group, args.x, args.metric)
            for cond, group in df.groupby("condition")
        }
        for cond, res in results.items():
            print(report(res, label=cond))
        print()
        print(contrast_verdict(results))
    else:
        res = fit_slope(df, args.x, args.metric)
        print(report(res))

    return 0


if __name__ == "__main__":
    sys.exit(main())
