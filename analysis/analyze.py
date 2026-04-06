"""
BPR Parameter Uniqueness Audit — Analysis & Plots
==================================================

Reads sweep results from analysis/results/ and produces:
  plots/score_vs_prime.png       — score vs prime (prime sweep)
  plots/score_vs_z.png           — score vs z (z ablation)
  plots/heatmap_joint.png        — joint score heatmap (prime × z)
  plots/alpha_vs_prime.png       — 1/α error vs prime
  summary.txt                    — sensitivity summary and red-team memo

Usage:
  cd /home/user/BPR-Math-Spine
  python analysis/analyze.py

Author: Claude Code audit, 2026-04-06
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXP_INV_ALPHA = 137.035999084
EXP_V_EW = 246.22
BASELINE_P = 104729
BASELINE_Z = 6


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading results ...")
prime_rows = load_csv(RESULTS_DIR / "prime_sweep.csv")
z_rows = load_csv(RESULTS_DIR / "z_ablation.csv")
joint_rows = load_csv(RESULTS_DIR / "joint_sweep.csv")

prime_data = [(int(r["p"]), to_float(r["composite_score"]),
               to_float(r["inv_alpha_0_ppm"]),
               to_float(r["v_EW_GeV_frac_err"]) * 100)
              for r in prime_rows]

z_data = [(int(r["z"]), to_float(r["composite_score"]),
           to_float(r["inv_alpha_0_pred"]),
           to_float(r["v_EW_GeV_pred"]))
          for r in z_rows]

joint_data = {(int(r["p"]), int(r["z"])): to_float(r["composite_score"])
              for r in joint_rows}

# ---------------------------------------------------------------------------
# 1. Score vs Prime
# ---------------------------------------------------------------------------

print("Plotting score vs prime ...")
primes = [d[0] for d in prime_data]
scores = [d[1] for d in prime_data]
alpha_ppms = [d[2] for d in prime_data]

baseline_idx = primes.index(BASELINE_P) if BASELINE_P in primes else None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(primes, scores, "b-o", ms=4, lw=1.5, label="Composite score")
if baseline_idx is not None:
    ax1.axvline(BASELINE_P, color="red", ls="--", lw=1.5, alpha=0.8, label=f"p={BASELINE_P} (baseline)")
    ax1.plot(BASELINE_P, scores[baseline_idx], "r*", ms=12)
ax1.set_ylabel("Composite score\n(RMS fractional error, lower=better)")
ax1.set_title("BPR Parameter Audit: Score vs Substrate Prime (z=6 fixed)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))

ax2.plot(primes, alpha_ppms, "g-s", ms=4, lw=1.5, label="1/α error (ppm)")
if baseline_idx is not None:
    ax2.axvline(BASELINE_P, color="red", ls="--", lw=1.5, alpha=0.8)
    ax2.plot(BASELINE_P, alpha_ppms[baseline_idx], "r*", ms=12)
ax2.set_xlabel("Substrate prime p")
ax2.set_ylabel("1/α(0) error (ppm)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "score_vs_prime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR / 'score_vs_prime.png'}")

# ---------------------------------------------------------------------------
# 2. Score vs z
# ---------------------------------------------------------------------------

print("Plotting score vs z ...")
z_vals = [d[0] for d in z_data]
z_scores = [d[1] for d in z_data]
z_alphas = [d[2] for d in z_data]
z_vews = [d[3] for d in z_data]

fig, axes = plt.subplots(3, 1, figsize=(8, 9))

axes[0].bar(z_vals, z_scores, color=["red" if z == BASELINE_Z else "steelblue" for z in z_vals],
            edgecolor="black", linewidth=0.8)
axes[0].set_ylabel("Composite score")
axes[0].set_title(f"BPR Parameter Audit: Score vs Coordination Number z (p={BASELINE_P} fixed)")
for i, (z, s) in enumerate(zip(z_vals, z_scores)):
    axes[0].text(z, s + 0.001, f"{s:.4f}", ha="center", fontsize=9)
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(z_vals, z_alphas, color=["red" if z == BASELINE_Z else "steelblue" for z in z_vals],
            edgecolor="black", linewidth=0.8)
axes[1].axhline(EXP_INV_ALPHA, color="black", ls="--", lw=1.5, label=f"Exp: {EXP_INV_ALPHA:.3f}")
axes[1].set_ylabel("Predicted 1/α(0)")
axes[1].legend()
axes[1].grid(True, axis="y", alpha=0.3)

axes[2].bar(z_vals, z_vews, color=["red" if z == BASELINE_Z else "steelblue" for z in z_vals],
            edgecolor="black", linewidth=0.8)
axes[2].axhline(EXP_V_EW, color="black", ls="--", lw=1.5, label=f"Exp: {EXP_V_EW} GeV")
axes[2].set_xlabel("Coordination number z")
axes[2].set_ylabel("Predicted v_EW (GeV)")
axes[2].legend()
axes[2].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "score_vs_z.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR / 'score_vs_z.png'}")

# ---------------------------------------------------------------------------
# 3. Joint heatmap
# ---------------------------------------------------------------------------

print("Plotting joint heatmap ...")
all_primes = sorted(set(k[0] for k in joint_data.keys()))
all_zs = sorted(set(k[1] for k in joint_data.keys()))

heatmap = np.full((len(all_zs), len(all_primes)), float("nan"))
for i, z in enumerate(all_zs):
    for j, p in enumerate(all_primes):
        heatmap[i, j] = joint_data.get((p, z), float("nan"))

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn_r", origin="lower",
               vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
plt.colorbar(im, ax=ax, label="Composite score (lower = better)")

ax.set_xticks(range(len(all_primes)))
ax.set_xticklabels([str(p) for p in all_primes], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(all_zs)))
ax.set_yticklabels([str(z) for z in all_zs])
ax.set_xlabel("Substrate prime p")
ax.set_ylabel("Coordination number z")
ax.set_title("BPR Joint Score Heatmap: prime × z\n(red=worse, green=better)")

# Mark baseline
if BASELINE_P in all_primes and BASELINE_Z in all_zs:
    bi = all_primes.index(BASELINE_P)
    bj = all_zs.index(BASELINE_Z)
    ax.add_patch(plt.Rectangle((bi - 0.5, bj - 0.5), 1, 1,
                                fill=False, edgecolor="blue", linewidth=3))
    ax.text(bi, bj, "★", ha="center", va="center", fontsize=14, color="blue")

# Annotate each cell
for i, z in enumerate(all_zs):
    for j, p in enumerate(all_primes):
        val = heatmap[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=6.5,
                    color="black")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "heatmap_joint.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR / 'heatmap_joint.png'}")

# ---------------------------------------------------------------------------
# 4. Alpha error zoomed
# ---------------------------------------------------------------------------

print("Plotting 1/α error detail ...")
fig, ax = plt.subplots(figsize=(11, 5))

# Re-read the actual predicted values from CSV
alpha_preds = {int(r["p"]): to_float(r["inv_alpha_0_pred"]) for r in prime_rows}
signed_errors = [(p, alpha_preds.get(p, float("nan")) - EXP_INV_ALPHA) for p in primes]

p_plot = [x[0] for x in signed_errors]
e_plot = [x[1] for x in signed_errors]

ax.plot(p_plot, e_plot, "b-o", ms=4, lw=1.5)
ax.axhline(0, color="black", ls="-", lw=0.8)
ax.axvline(BASELINE_P, color="red", ls="--", lw=1.5, alpha=0.8, label=f"p={BASELINE_P}")
if BASELINE_P in p_plot:
    bi = p_plot.index(BASELINE_P)
    ax.plot(BASELINE_P, e_plot[bi], "r*", ms=12)

ax.fill_between(p_plot, -0.002, 0.002, alpha=0.15, color="green", label="±0.002 band")
ax.set_xlabel("Substrate prime p")
ax.set_ylabel("Predicted 1/α(0) − Experimental\n(137.036 = target)")
ax.set_title("BPR 1/α Signed Error vs Prime (z=6 fixed)\nTarget: 0.000, monotone drift visible")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "alpha_error_vs_prime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {PLOTS_DIR / 'alpha_error_vs_prime.png'}")

# ---------------------------------------------------------------------------
# 5. Sensitivity summary text
# ---------------------------------------------------------------------------

baseline_score = next((d[1] for d in prime_data if d[0] == BASELINE_P), float("nan"))
best_prime_in_sweep = min(prime_data, key=lambda d: d[1])
n_better_primes = sum(1 for d in prime_data if d[1] < baseline_score)
best_z_score = min(d[1] for d in z_data)
baseline_z_score = next((d[1] for d in z_data if d[0] == BASELINE_Z), float("nan"))
n_better_z = sum(1 for d in z_data if d[1] < baseline_z_score)

alpha_at_baseline = alpha_preds.get(BASELINE_P, float("nan"))
alpha_err_baseline = alpha_at_baseline - EXP_INV_ALPHA
# Slope d(1/alpha)/dp at baseline
if len(p_plot) > 1:
    dp = np.diff(p_plot)
    de = np.diff(e_plot)
    local_deriv = np.mean(de / dp)  # avg slope over sweep
else:
    local_deriv = float("nan")

summary = f"""
BPR PARAMETER UNIQUENESS AUDIT — SENSITIVITY SUMMARY
=====================================================
Generated: 2026-04-06
Sweep: 51 primes [104479..105019], z in [4,5,6,7,8]
Composite score = RMS(fractional errors) over independent observables:
  - inv_alpha_0  (1/α at q²=0)
  - v_EW_GeV     (electroweak scale / Higgs VEV)

BASELINE: p=104729, z=6
  Composite score:  {baseline_score:.6f}
  1/α(0) predicted: {alpha_at_baseline:.6f}  (exp: {EXP_INV_ALPHA:.6f})
  1/α(0) error:     {alpha_err_baseline:+.6f}  ({alpha_err_baseline/EXP_INV_ALPHA*1e6:.1f} ppm)

─────────────────────────────────────────────────────────────────────
Q1: IS p = 104729 A SHARP OPTIMUM OR PART OF A PLATEAU?
─────────────────────────────────────────────────────────────────────
Short answer: NEITHER — it is on a monotone slope. It is not a local minimum.

Among the {len(prime_data)} primes tested (25 below, 25 above):
  Primes with BETTER score than baseline: {n_better_primes}/{len(prime_data)}
  Best prime in sweep: p={best_prime_in_sweep[0]} (score={best_prime_in_sweep[1]:.6f})
  Baseline rank: #{len(prime_data) - n_better_primes} out of {len(prime_data)}

The score improves monotonically as p increases. Every one of the 25
primes ABOVE 104729 (all tested) scores BETTER than p=104729.
This means p=104729 is not special relative to nearby alternatives.

Physical reason: 1/α = [ln(p)]² + z/2 + γ − 1/(2π)
The term [ln(p)]² = {np.log(BASELINE_P)**2:.4f} is the dominant contribution
(≈133.6 out of 137.0). d(1/α)/dp = 2·ln(p)/p ≈ {2*np.log(BASELINE_P)/BASELINE_P:.2e} per unit of p.
The experimental target is {EXP_INV_ALPHA:.6f}; baseline prediction is {alpha_at_baseline:.6f}.
The residual {alpha_err_baseline:+.6f} can be improved by increasing p by
≈ {-alpha_err_baseline / (2*np.log(BASELINE_P)/BASELINE_P):.0f} units — well within the
spacing of consecutive primes (avg ~{np.mean(np.diff(sorted(primes))):.1f} apart near 104729).

Primes near 104729 where 1/α would match experiment EXACTLY would be
around p ≈ {int(BASELINE_P - alpha_err_baseline / (2*np.log(BASELINE_P)/BASELINE_P))}.
(Not necessarily prime, but the function is continuous and smooth.)

─────────────────────────────────────────────────────────────────────
Q2: IS z = 6 UNIQUELY GOOD OR JUST SLIGHTLY BETTER?
─────────────────────────────────────────────────────────────────────
z values tested: {[d[0] for d in z_data]}
Scores:          {[round(d[1],4) for d in z_data]}
Baseline (z=6):  {baseline_z_score:.6f}
z values with better score: {n_better_z}

z=6 IS meaningfully better than z=4,5,7,8. The z-ablation shows a
clear minimum near z=6. This looks like genuine signal.

HOWEVER, this is almost entirely driven by the linear z/2 term in
1/α = [ln(p)]² + z/2 + γ − 1/(2π). With p fixed at 104729:
  z=4: 1/α ≈ {136.031:.3f}  (error ≈ {abs(136.031-EXP_INV_ALPHA):.3f})
  z=5: 1/α ≈ {136.531:.3f}  (error ≈ {abs(136.531-EXP_INV_ALPHA):.3f})
  z=6: 1/α ≈ {137.031:.3f}  (error ≈ {abs(137.031-EXP_INV_ALPHA):.3f})
  z=7: 1/α ≈ {137.531:.3f}  (error ≈ {abs(137.531-EXP_INV_ALPHA):.3f})
  z=8: 1/α ≈ {138.031:.3f}  (error ≈ {abs(138.031-EXP_INV_ALPHA):.3f})

The score is minimized near z=6 because z/2 = 3 is the integer value of
z/2 that brings [ln(p)]² + z/2 closest to 137.036 given the chosen p.
If p were shifted, the optimal z would shift accordingly.
This is a two-parameter fit to one number (1/α), not an independent prediction.

v_EW = Λ_QCD × p^(1/3) × (ln(p) + z − 2) also depends on z, and z=6
gives v_EW = 243.5 GeV vs exp 246.2 GeV (1.1% off). Changing z to 7
would give ≈259 GeV (5.3% off), so z=6 is better for v_EW too. But this
is again a two-parameter (p, z) fit to two numbers (1/α and v_EW), with
no overdetermination.

─────────────────────────────────────────────────────────────────────
Q3: DO NEARBY ALTERNATIVES PRESERVE MOST OF THE FIT?
─────────────────────────────────────────────────────────────────────
YES. The best nearby prime (p={best_prime_in_sweep[0]}) scores {best_prime_in_sweep[1]:.6f}
vs the baseline {baseline_score:.6f} — a {abs(best_prime_in_sweep[1]-baseline_score)/baseline_score*100:.1f}% improvement.
The score variation across all 51 tested primes (at z=6) ranges from
{min(d[1] for d in prime_data):.6f} to {max(d[1] for d in prime_data):.6f},
a factor of {max(d[1] for d in prime_data)/min(d[1] for d in prime_data):.2f}×.
The entire range of scores is within the "reasonable fit" zone — none
are wildly better or worse than the baseline. The model shows NO sharp
optimum at p=104729.

─────────────────────────────────────────────────────────────────────
Q4: DOES THE FRAMEWORK LOOK STRUCTURALLY CONSTRAINED OR UNDERDETERMINED?
─────────────────────────────────────────────────────────────────────
UNDERDETERMINED. Two free parameters (p, z) are being fit to (at most)
two genuinely parameter-dependent observables (1/α and v_EW). That is
not overdetermined. A model with 2 free parameters and 2 observables
has zero degrees of freedom — no predictive power test is possible.

The remaining "predictions" in the BPR framework fall into three
categories that INFLATE the apparent evidence:
  (a) NON-INDEPENDENT: observables that don't vary with p or z at all
      (lepton masses, sin²θ_W via the circular GUT-unification routine)
  (b) TRIVIALLY DERIVED: 1/α(M_Z) = 1/α(0) − 9.084 (not independent)
  (c) BROKEN: Ω_Λ calculation uses cosmological p_cosmo (~2.7e61), not
      the substrate p, so it cannot distinguish p=104729 from any other p
      (and gives ~1e-104 for all values, wrong by 100 orders of magnitude)

"""

# Count how many observables are genuinely p/z dependent
independent_obs_count = 2  # inv_alpha_0 and v_EW
total_claimed_obs = 6  # in the sweep

summary += f"""
─────────────────────────────────────────────────────────────────────
RED-TEAM ANALYSIS
─────────────────────────────────────────────────────────────────────

1. HIDDEN FREE PARAMETERS
   The formula 1/α = [ln(p)]² + z/2 + γ − 1/(2π) is a 2-parameter
   fit. Given the experimental value 137.036, infinitely many (p, z)
   pairs satisfy it (z = 2×(137.036 - [ln(p)]² - γ + 1/(2π))). The
   choice of integer z=6 and prime p=104729 reduces this to a discrete
   search, but the "uniqueness" derives from the discreteness, not from
   deep physics.

2. CIRCULAR PREDICTIONS
   GaugeCouplingRunning.weinberg_angle_at_MZ returns exactly 0.23122
   for ALL values of p. The method constructs threshold corrections
   specifically to force gauge unification, then runs back down to
   recover the starting coupling — it is mathematically guaranteed to
   return the input values. This is not a prediction; it is an identity.

3. NON-PARAMETER-DEPENDENT PREDICTIONS
   ChargedLeptonSpectrum does not accept p or z. The lepton mass
   predictions (e.g., m_e/m_τ from l_modes = (1, 14, 59)) are
   independent of the substrate parameters and would be identical
   for p=3, z=2 or p=999983, z=12. They cannot be used to argue
   for or against the uniqueness of p=104729.

4. DOWNSTREAM DEPENDENCE
   Several claimed "independent" predictions are actually algebraically
   downstream of the 1/α prediction:
   • 1/α(M_Z) = 1/α(0) − 9.084 (trivially derived)
   • sin²θ_W Route 3 = α_EM(M_Z) / α₂_MZ (uses BPR's 1/α(0))
   • α_GUT = π / (p^(1/3) × z) (another formula in p and z, not
     validated against independent experiment)

5. SMOOTHNESS OF THE LANDSCAPE
   The score varies smoothly and monotonically over the prime sweep.
   If BPR's p were uniquely selected by some physical mechanism, one
   would expect a sharp minimum (like a resonance or quantization
   condition). Instead, the landscape is featureless: p=104729 ranks
   #26 out of 51 tested primes. Every prime above it performs better.

6. POST-HOC FITTING INDICATORS
   The VALIDATION_STATUS.md file shows predictions labeled
   "FRAMEWORK" (formula + some experimental inputs) and "CONSISTENT"
   (matches data but not BPR-unique). The l_modes (1, 14, 59) for
   lepton generations appear chosen to match the known mass ratios
   (the comment in charged_leptons.py states "l = √210 DERIVED from
   boundary-Higgs mixing: degenerate" — but 59² / 14² = 3481/196 = 17.76
   while m_τ/m_μ = 16.82, a 5.6% discrepancy that matches within the
   "5.3% off" claim in the code). The selection of l_modes is not
   derived from the substrate parameters p or z; it appears to be
   reverse-engineered from the mass hierarchy.

7. Ω_Λ IS BROKEN
   The dark energy prediction in cross_predictions.py uses
   p_cosmo = R_Hubble / L_Planck ≈ 2.7 × 10^61, not p=104729.
   For any value of the local p argument, it returns ≈ 1 × 10^-104
   for Ω_Λ (compared to exp 0.689). This is wrong by 100 orders of
   magnitude. The claimed ~0.685 result in VALIDATION_STATUS.md comes
   from a different code path (possibly hardcoded in constants.py:
   OMEGA_LAMBDA = 0.685, which matches Planck 2018 exactly).

─────────────────────────────────────────────────────────────────────
NEXT EXPERIMENT THAT WOULD MOST DECISIVELY CONFIRM/REFUTE UNIQUENESS
─────────────────────────────────────────────────────────────────────
The strongest test would be: find a third observable O₃ that depends
on p and z through a DIFFERENT functional form than 1/α and v_EW,
then check whether p=104729, z=6 simultaneously minimizes the error
on O₃ without adjusting p or z.

Candidate: the proton/electron mass ratio m_p/m_e ≈ 1836.15.
If BPR can derive this from (p, z) without using it as input, and
the value uniquely selects p=104729 with z=6, that would be strong
evidence. Conversely, if the derivation again reduces to a smooth
formula with no sharp minimum, that would be strong evidence against
uniqueness.

A second useful test: extend the prime sweep to ±100 primes (or to
the full range where 1/α error < 1%), and fit a polynomial to the
score landscape. If there is a true minimum with a second derivative
consistent with a physical mechanism, it would be visible. The current
monotone slope strongly suggests no physical minimum exists in this
neighborhood.
"""

# Save summary
summary_path = REPO_ROOT / "analysis" / "summary.txt"
with open(summary_path, "w") as f:
    f.write(summary)
print(f"\n  Saved: {summary_path}")
print(summary)

print("\nAnalysis complete. Outputs:")
print(f"  {PLOTS_DIR / 'score_vs_prime.png'}")
print(f"  {PLOTS_DIR / 'score_vs_z.png'}")
print(f"  {PLOTS_DIR / 'heatmap_joint.png'}")
print(f"  {PLOTS_DIR / 'alpha_error_vs_prime.png'}")
print(f"  {REPO_ROOT / 'analysis' / 'summary.txt'}")
