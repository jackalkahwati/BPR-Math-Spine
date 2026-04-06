"""
BPR Viable Prime Corridor — Full Characterization
===================================================

Enumerates every prime in the z=6 viable corridor [103935, 110634],
computes per-prime metrics, classifies each prime, and answers
whether p=104729 is special within the family.

Outputs:
  analysis/results/corridor/
    all_primes.csv           — full 580-prime table
    ranked.csv               — sorted by core score
    classifications.csv      — category assignments
    corridor_summary.json    — key statistics and answers

  analysis/plots/
    corridor_score.png           — score vs p across corridor
    corridor_tradeoff.png        — 1/alpha vs v_EW error (Pareto)
    corridor_robustness.png      — robustness score vs p
    corridor_classification.png  — coloured by classification

Author: Claude Code audit, 2026-04-06
"""

from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "analysis" / "results" / "corridor"
PLOT_DIR  = REPO_ROOT / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
EULER_GAMMA  = 0.5772156649015329
LAMBDA_QCD   = 0.332          # bpr/gauge_unification.py line 27
EXP_ALPHA    = 137.035999084  # CODATA 2018
EXP_VEW      = 246.22         # PDG 2024
EXP_NS       = 0.9649         # Planck 2018
EXP_DCP      = 1.196          # PDG 2024
SIGMA_NS     = 0.0042
SIGMA_DCP    = 0.045

Z            = 6
BASELINE_P   = 104729
CORRIDOR_LO  = 103935   # continuous 1%-threshold lower bound
CORRIDOR_HI  = 110634   # continuous 1%-threshold upper bound

# ── Primality ───────────────────────────────────────────────────────────────
_MR = (2,3,5,7,11,13,17,19,23,29,31,37)
def is_prime(n: int) -> bool:
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23):
        if n == p: return True
        if n % p == 0: return False
    d, r = n-1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in _MR:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for _ in range(r-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True

# ── Physics ─────────────────────────────────────────────────────────────────
def inv_alpha(p: int) -> float:
    return math.log(p)**2 + Z/2.0 + EULER_GAMMA - 1.0/(2*math.pi)

def v_ew(p: int) -> float:
    return LAMBDA_QCD * p**(1/3) * (math.log(p) + Z - 2)

def n_s(p: int) -> float:
    return 1.0 - 2.0 / (p**(1/3) * 4/3)

def frac_err(pred: float, exp: float) -> float:
    return abs(pred - exp) / abs(exp)

def core_score(p: int) -> float:
    ae = frac_err(inv_alpha(p), EXP_ALPHA)
    ve = frac_err(v_ew(p),      EXP_VEW)
    return math.sqrt((ae**2 + ve**2) / 2)

# ── Curvature / robustness ─────────────────────────────────────────────────
def local_curvature(p: int, primes: List[int]) -> float:
    """d²(score)/dp² estimated from nearest prime neighbours."""
    idx = primes.index(p)
    if idx < 1 or idx >= len(primes) - 1:
        return float("nan")
    s_m = core_score(primes[idx-1])
    s_0 = core_score(p)
    s_p = core_score(primes[idx+1])
    h1  = primes[idx]   - primes[idx-1]
    h2  = primes[idx+1] - primes[idx]
    # Second-order finite difference for unequal spacing
    return 2*(s_p/h2 - s_0*(1/h1+1/h2) + s_m/h1) / (h1+h2)

def robustness_score(p: int, primes: List[int], delta: int = 200) -> float:
    """
    Mean core score of primes within ±delta of p.
    Lower is more robust (good score is stable over a range).
    """
    window = [q for q in primes if abs(q - p) <= delta]
    if not window:
        return float("nan")
    return float(np.mean([core_score(q) for q in window]))

def distance_from_edges(p: int) -> Tuple[float, float]:
    """Fractional distance from lower and upper continuous corridor edges."""
    width = CORRIDOR_HI - CORRIDOR_LO
    d_lo = (p - CORRIDOR_LO) / width
    d_hi = (CORRIDOR_HI - p) / width
    return d_lo, d_hi

# ── Enumerate ───────────────────────────────────────────────────────────────
print("Finding all primes in corridor...")
primes = [n for n in range(CORRIDOR_LO, CORRIDOR_HI + 1) if is_prime(n)]
print(f"  {len(primes)} primes in [{CORRIDOR_LO}, {CORRIDOR_HI}]")

# ── Compute all metrics ──────────────────────────────────────────────────────
print("Computing metrics...")
rows: List[Dict[str, Any]] = []
for p in primes:
    ia   = inv_alpha(p)
    vw   = v_ew(p)
    ns   = n_s(p)
    ae   = frac_err(ia, EXP_ALPHA)
    ve   = frac_err(vw, EXP_VEW)
    sc   = math.sqrt((ae**2 + ve**2) / 2)
    ns_s = abs(ns - EXP_NS) / SIGMA_NS
    d_lo, d_hi = distance_from_edges(p)
    rows.append({
        "p":                p,
        "inv_alpha_pred":   round(ia, 8),
        "inv_alpha_err_ppm": round(ae * 1e6, 3),
        "v_EW_pred":        round(vw, 6),
        "v_EW_err_pct":     round(ve * 100, 6),
        "core_score_pct":   round(sc * 100, 6),
        "n_s_pred":         round(ns, 8),
        "n_s_sigma":        round(ns_s, 4),
        "dist_from_lo":     round(d_lo, 5),
        "dist_from_hi":     round(d_hi, 5),
        "is_baseline":      int(p == BASELINE_P),
    })

# Add curvature and robustness (needs full prime list)
print("Computing curvature and robustness...")
for row in rows:
    p = row["p"]
    row["curvature"]    = round(local_curvature(p, primes), 12)
    row["robustness"]   = round(robustness_score(p, primes, delta=200), 8)

# ── Classifications ─────────────────────────────────────────────────────────
scores    = [r["core_score_pct"]     for r in rows]
ae_vals   = [r["inv_alpha_err_ppm"]  for r in rows]
ve_vals   = [r["v_EW_err_pct"]       for r in rows]
rob_vals  = [r["robustness"]         for r in rows]

min_score  = min(scores)
min_ae     = min(ae_vals)
min_ve     = min(ve_vals)
min_rob    = min(r for r in rob_vals if not math.isnan(r))

# Thresholds for "near-optimal" (within 10% of best value for that metric)
NEAR_OPT_FACTOR = 1.10

for row in rows:
    cats = []

    # alpha-optimal: within 10× of minimum 1/alpha ppm error
    if row["inv_alpha_err_ppm"] <= min_ae * 10:
        cats.append("alpha-optimal")

    # v_EW-optimal: within 10× of minimum v_EW error
    if row["v_EW_err_pct"] <= min_ve * 10:
        cats.append("vEW-optimal")

    # RMS-optimal: within 10% of minimum core score
    if row["core_score_pct"] <= min_score * NEAR_OPT_FACTOR:
        cats.append("RMS-optimal")

    # edge: within 5% of corridor width from either edge
    if row["dist_from_lo"] < 0.05 or row["dist_from_hi"] < 0.05:
        cats.append("fragile-edge")
    elif row["dist_from_lo"] > 0.20 and row["dist_from_hi"] > 0.20:
        cats.append("robust-interior")

    # robustness
    if not math.isnan(row["robustness"]) and row["robustness"] <= min_rob * NEAR_OPT_FACTOR:
        cats.append("robust")

    row["classification"] = "|".join(cats) if cats else "mid-range"

# ── Save CSVs ───────────────────────────────────────────────────────────────
def save_csv(data, path):
    if not data: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
    print(f"  Saved {path.name} ({len(data)} rows)")

save_csv(rows, OUT_DIR / "all_primes.csv")
ranked = sorted(rows, key=lambda r: r["core_score_pct"])
save_csv(ranked, OUT_DIR / "ranked.csv")
save_csv(rows,   OUT_DIR / "classifications.csv")

# ── Summary statistics ──────────────────────────────────────────────────────
baseline_row = next(r for r in rows if r["p"] == BASELINE_P)
baseline_rank = ranked.index(baseline_row) + 1

rms_opt   = ranked[0]
alpha_opt = min(rows, key=lambda r: r["inv_alpha_err_ppm"])
vew_opt   = min(rows, key=lambda r: r["v_EW_err_pct"])
rob_opt   = min((r for r in rows if not math.isnan(r["robustness"])),
                key=lambda r: r["robustness"])

n_rms_near = sum(1 for r in rows if r["core_score_pct"] <= min_score * 1.10)
n_alpha_opt = sum(1 for r in rows if r["inv_alpha_err_ppm"] <= min_ae * 10)
n_vew_opt   = sum(1 for r in rows if r["v_EW_err_pct"]     <= min_ve * 10)

summary = {
    "corridor": {"lo": CORRIDOR_LO, "hi": CORRIDOR_HI, "width": CORRIDOR_HI - CORRIDOR_LO},
    "total_primes": len(primes),
    "baseline": {
        "p": BASELINE_P,
        "rank": baseline_rank,
        "core_score_pct": baseline_row["core_score_pct"],
        "inv_alpha_err_ppm": baseline_row["inv_alpha_err_ppm"],
        "v_EW_err_pct": baseline_row["v_EW_err_pct"],
        "dist_from_lo": baseline_row["dist_from_lo"],
        "dist_from_hi": baseline_row["dist_from_hi"],
        "classification": baseline_row["classification"],
        "robustness": baseline_row["robustness"],
    },
    "rms_optimal": {"p": rms_opt["p"], "score": rms_opt["core_score_pct"],
                    "inv_alpha_err_ppm": rms_opt["inv_alpha_err_ppm"],
                    "v_EW_err_pct": rms_opt["v_EW_err_pct"]},
    "alpha_optimal": {"p": alpha_opt["p"], "inv_alpha_err_ppm": alpha_opt["inv_alpha_err_ppm"],
                      "score": alpha_opt["core_score_pct"]},
    "vEW_optimal": {"p": vew_opt["p"], "v_EW_err_pct": vew_opt["v_EW_err_pct"],
                    "score": vew_opt["core_score_pct"]},
    "robustness_optimal": {"p": rob_opt["p"], "robustness": rob_opt["robustness"],
                           "score": rob_opt["core_score_pct"]},
    "near_rms_optimal_count": n_rms_near,
    "near_alpha_optimal_count": n_alpha_opt,
    "near_vew_optimal_count": n_vew_opt,
    "score_stats": {
        "min":  round(min(scores),  6),
        "max":  round(max(scores),  6),
        "mean": round(float(np.mean(scores)), 6),
        "std":  round(float(np.std(scores)),  6),
    },
}
with open(OUT_DIR / "corridor_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved corridor_summary.json")

# ── Console report ──────────────────────────────────────────────────────────
print()
print("=" * 90)
print(f"CORRIDOR REPORT  [{CORRIDOR_LO}–{CORRIDOR_HI}]  z=6")
print("=" * 90)
print(f"  Total viable primes (<=1% core score): {len(primes)}")
print(f"  Score range:  {min(scores):.4f}% – {max(scores):.4f}%   mean={np.mean(scores):.4f}%  std={np.std(scores):.4f}%")
print()
print(f"  RMS-optimal:    p={rms_opt['p']:>7}   score={rms_opt['core_score_pct']:.4f}%   1/a={rms_opt['inv_alpha_err_ppm']:.1f}ppm  vEW={rms_opt['v_EW_err_pct']:.4f}%")
print(f"  alpha-optimal:  p={alpha_opt['p']:>7}   score={alpha_opt['core_score_pct']:.4f}%   1/a={alpha_opt['inv_alpha_err_ppm']:.3f}ppm  vEW={alpha_opt['v_EW_err_pct']:.4f}%")
print(f"  vEW-optimal:    p={vew_opt['p']:>7}   score={vew_opt['core_score_pct']:.4f}%   1/a={vew_opt['inv_alpha_err_ppm']:.1f}ppm  vEW={vew_opt['v_EW_err_pct']:.4f}%")
print(f"  Rob-optimal:    p={rob_opt['p']:>7}   score={rob_opt['core_score_pct']:.4f}%   robustness={rob_opt['robustness']:.4f}%")
print()
print(f"  Baseline p={BASELINE_P}:")
print(f"    rank          = {baseline_rank} / {len(primes)}")
print(f"    core score    = {baseline_row['core_score_pct']:.4f}%")
print(f"    1/alpha error = {baseline_row['inv_alpha_err_ppm']:.3f} ppm  ({baseline_row['inv_alpha_err_ppm']/min_ae:.1f}× above minimum)")
print(f"    v_EW error    = {baseline_row['v_EW_err_pct']:.4f}%  ({baseline_row['v_EW_err_pct']/min_ve:.1f}× above minimum)")
print(f"    dist from lo  = {baseline_row['dist_from_lo']:.3f}  ({baseline_row['dist_from_lo']*100:.1f}% into corridor)")
print(f"    dist from hi  = {baseline_row['dist_from_hi']:.3f}  ({baseline_row['dist_from_hi']*100:.1f}% from upper edge)")
print(f"    classification= {baseline_row['classification']}")
print(f"    robustness    = {baseline_row['robustness']:.4f}%")
print()
print("  Top 20 by core score:")
print(f"  {'rank':>4}  {'p':>8}  {'score':>9}  {'1/a ppm':>9}  {'vEW %':>8}  {'class'}")
print("  " + "-" * 75)
for i, r in enumerate(ranked[:20]):
    bl = " <-- BASELINE" if r["p"] == BASELINE_P else ""
    print(f"  {i+1:>4}  {r['p']:>8}  {r['core_score_pct']:>8.4f}%  {r['inv_alpha_err_ppm']:>9.1f}  {r['v_EW_err_pct']:>8.4f}%  {r['classification']}{bl}")
print(f"  ...")
# Show where baseline appears
if baseline_rank > 20:
    print(f"  {baseline_rank:>4}  {BASELINE_P:>8}  {baseline_row['core_score_pct']:>8.4f}%  {baseline_row['inv_alpha_err_ppm']:>9.1f}  {baseline_row['v_EW_err_pct']:>8.4f}%  {baseline_row['classification']}  <-- BASELINE")

if __name__ == "__main__":
    # Trigger plotting
    import subprocess, sys
    print("\nRunning plots...")
    subprocess.run([sys.executable, str(Path(__file__).parent / "corridor_plots.py")], check=True)
    print("Done.")
