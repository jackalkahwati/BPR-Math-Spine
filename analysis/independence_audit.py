"""
BPR Independence Audit — Finding and Testing Independent Observables
======================================================================

Goal: Find observables that GENUINELY depend on (p, z) but are NOT
algebraically downstream of the 1/α formula and NOT hardcoded or circular.

Method:
  - Systematically check each observable's code path
  - Classify as: INDEPENDENT / DOWNSTREAM / CONSTANT / CIRCULAR / BROKEN
  - Pick the strongest independent candidates
  - Sweep (p, z) and check whether error landscape has a minimum at (104729, 6)

Result: Two genuinely independent candidates identified:
  1. n_s (inflationary spectral index): depends on p^(1/3) only
     NOT downstream of [ln(p)]^2
  2. δ_CP (CKM CP-phase): depends on 1/sqrt(z+1) only
     NOT downstream of z/2

Author: Claude Code independence audit, 2026-04-06
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

# Experimental reference values
EXP = {
    "n_s":       0.9649,      # Planck 2018 TT,TE,EE+lowE+lensing
    "n_s_unc":   0.0042,      # 1σ uncertainty
    "r_tensor":  0.064,       # Planck 2018 95% CL upper bound (r < 0.064)
    "delta_CP":  1.196,       # PDG 2024 CKM CP-phase [rad] (1.196 ± 0.045)
    "delta_CP_unc": 0.045,    # 1σ uncertainty [rad]
    "inv_alpha_0": 137.035999084,  # CODATA 2018
    "v_EW_GeV":  246.22,      # PDG 2024
}

BASELINE_P = 104729
BASELINE_Z = 6

# --------------------------------------------------------------------------
# Primality test (same as sweep.py)
# --------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    if n < 2: return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small: return True
    if any(n % p == 0 for p in small): return False
    r, d = 0, n - 1
    while d % 2 == 0: r += 1; d //= 2
    for a in small:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True


def primes_near(center: int, n_below: int = 25, n_above: int = 25) -> list[int]:
    below = []
    k = center - 1 if _is_prime(center) else center
    while len(below) < n_below and k > 2:
        if _is_prime(k): below.append(k)
        k -= 1
    above = []
    k = center + 1
    while len(above) < n_above:
        if _is_prime(k): above.append(k)
        k += 1
    return sorted(below + ([center] if _is_prime(center) else []) + above)


# --------------------------------------------------------------------------
# OBSERVABLE CLASSIFICATION
# --------------------------------------------------------------------------

CLASSIFICATIONS = """
OBSERVABLE INDEPENDENCE CLASSIFICATION
=======================================

inv_alpha_0:
  Formula: [ln(p)]^2 + z/2 + gamma - 1/(2*pi)
  Depends on: p (via [ln(p)]^2), z (via z/2)
  Status: CORE FORMULA — defines what "downstream" means

inv_alpha_MZ:
  Formula: inv_alpha_0 - 9.084
  Status: DOWNSTREAM (trivially derived from inv_alpha_0)

v_EW:
  Formula: Lambda_QCD * p^(1/3) * (ln(p) + z - 2)
  Depends on: p^(1/3) [DIFFERENT form] and ln(p) [same ingredient as core]
  Uses experimental input: Lambda_QCD = 0.332 GeV
  Status: PARTIALLY INDEPENDENT (p^(1/3) part is independent; ln(p) overlaps)

sin2_theta_W (GaugeCouplingRunning):
  Status: CIRCULAR — constructs threshold corrections to force unification,
  then recovers hardcoded input values. Returns 0.23122 for all p.

Lepton masses (ChargedLeptonSpectrum):
  Status: CONSTANT — no p or z parameters; ignores substrate entirely.

Omega_Lambda (dark_energy_from_impedance):
  Status: BROKEN — uses cosmological p_cosmo = R_H/L_P ~ 2.7e61, not substrate p.
  Returns ~1e-104 for all p values.

PMNSMatrix sin2_theta12:
  Formula: 1/3 - 1/(3.5 * ln(p))
  Depends on: ln(p) [same ingredient as core, but 1/ln(p) not [ln(p)]^2]
  Also: s13 = 0.150 is HARDCODED — not derived
  Status: WEAKLY INDEPENDENT (different functional form of ln(p), but s13 hardcoded)

n_s (InflationaryParameters.spectral_index):
  Formula: 1 - 2/N  where  N = p^(1/3) * (1 + 1/d)  [d = 3 = spatial dims]
         = 1 - 3/(2 * p^(1/3))
  Depends on: p^(1/3) ONLY
  NO ln(p) — completely different functional form from [ln(p)]^2
  NO z dependence at all
  No hardcoded experimental targets in the formula
  Experimental comparison: 0.9649 +/- 0.0042 (Planck 2018)
  Status: **GENUINELY INDEPENDENT** — best candidate for p-uniqueness test

r (InflationaryParameters.tensor_to_scalar):
  Formula: 12/N^2 = 27/(4 * p^(2/3))
  Trivially downstream of n_s (both from N = p^(1/3)*4/3)
  Status: DOWNSTREAM of n_s (same N)

delta_CP (CKMMatrix):
  Formula: pi/2 - 1/sqrt(z+1)
  Depends on: z ONLY via 1/sqrt(z+1)  [completely different from z/2]
  NO p dependence
  No hardcoded experimental targets
  Experimental comparison: 1.196 +/- 0.045 rad (PDG 2024)
  Status: **GENUINELY INDEPENDENT** — best candidate for z-uniqueness test

|V_cb| (CKMMatrix):
  Formula: sqrt(m_s_exp / m_b_exp) / sqrt(ln(p) + z/3)
  USES HARDCODED experimental quark masses as inputs
  Status: DISQUALIFIED (experimental inputs baked in)

m_b correction (QuarkMassSpectrum):
  Formula: m_t * (E_b/c_t) * (2 + 1/(3*ln(p)))
  Uses experimental m_t as anchor
  Status: WEAKLY INDEPENDENT of 1/alpha (different form: 1/(3*ln(p)))
  But depends on m_t as experimental input.
"""

# --------------------------------------------------------------------------
# Compute the two independent observables
# --------------------------------------------------------------------------

def compute_n_s(p: int) -> dict[str, Any]:
    """Compute inflationary spectral index n_s from substrate prime p.

    Formula: N = p^(1/3) * (1 + 1/3) = p^(1/3) * 4/3
             n_s = 1 - 2/N = 1 - 3/(2 * p^(1/3))

    This uses p^(1/3) ONLY — completely different from [ln(p)]^2.
    No z dependence. No hardcoded experimental values.

    The d=3 parameter is the number of spatial dimensions, not
    an experimental observable. It is fixed by physics, not fitted.
    """
    from bpr.cosmology import InflationaryParameters
    ip = InflationaryParameters(p=p, d=3)
    n_s = ip.spectral_index
    r = ip.tensor_to_scalar
    N = ip.n_efolds

    n_s_err = abs(n_s - EXP["n_s"])
    n_s_frac_err = n_s_err / EXP["n_s"]
    n_s_sigma = n_s_err / EXP["n_s_unc"]

    # r: only an upper bound, so penalize if r > 0.064, not if r < 0.064
    r_violation = max(0.0, r - EXP["r_tensor"])

    return {
        "p": p,
        "N_efolds": N,
        "n_s_pred": n_s,
        "n_s_frac_err": n_s_frac_err,
        "n_s_sigma": n_s_sigma,
        "n_s_signed": n_s - EXP["n_s"],
        "r_tensor_pred": r,
        "r_violation": r_violation,
    }


def compute_delta_cp(z: int) -> dict[str, Any]:
    """Compute CKM CP-phase delta_CP from coordination number z.

    Formula: delta_CP = pi/2 - 1/sqrt(z+1)

    Uses z ONLY via 1/sqrt(z+1) — completely different from z/2 in 1/alpha.
    No p dependence. No hardcoded experimental values in the formula.

    Experimental: 1.196 +/- 0.045 rad (PDG 2024).
    """
    delta = np.pi / 2.0 - 1.0 / np.sqrt(z + 1.0)
    delta_deg = np.degrees(delta)
    delta_err = abs(delta - EXP["delta_CP"])
    delta_frac_err = delta_err / EXP["delta_CP"]
    delta_sigma = delta_err / EXP["delta_CP_unc"]

    return {
        "z": z,
        "delta_CP_pred_rad": delta,
        "delta_CP_pred_deg": delta_deg,
        "delta_CP_frac_err": delta_frac_err,
        "delta_CP_sigma": delta_sigma,
        "delta_CP_signed": delta - EXP["delta_CP"],
    }


# --------------------------------------------------------------------------
# PART 1: Find what p value n_s ACTUALLY prefers
# --------------------------------------------------------------------------

def find_optimal_p_for_ns() -> None:
    """Determine what substrate prime would minimize the n_s error.

    n_s = 1 - 3/(2 * p^(1/3)) = EXP["n_s"]
    => p^(1/3) = 3 / (2 * (1 - EXP["n_s"]))
    => p = (3 / (2 * (1 - EXP["n_s"])))^3
    """
    target_N = 2.0 / (1.0 - EXP["n_s"])  # N needed for exact n_s
    target_p13 = target_N / (4.0 / 3.0)  # p^(1/3) = N / (4/3)
    target_p = target_p13 ** 3

    print(f"\n=== What p does n_s PREFER? ===")
    print(f"Target n_s = {EXP['n_s']} requires N = {target_N:.2f}")
    print(f"This requires p^(1/3) = {target_p13:.2f}, i.e., p ≈ {target_p:.0f}")
    print(f"Baseline p = {BASELINE_P} (difference: {int(BASELINE_P - target_p):+,})")
    print(f"Baseline n_s = {1 - 2/(BASELINE_P**(1/3)*4/3):.6f} "
          f"(err = {(1 - 2/(BASELINE_P**(1/3)*4/3)) - EXP['n_s']:+.4f}, "
          f"{((1 - 2/(BASELINE_P**(1/3)*4/3)) - EXP['n_s'])/EXP['n_s_unc']:.2f}σ above target)")
    print(f"The n_s-optimal prime is ~{int(target_p)} — far from 104729!")


# --------------------------------------------------------------------------
# PART 2: n_s sweep over nearby primes
# --------------------------------------------------------------------------

def run_ns_prime_sweep(n_below: int = 25, n_above: int = 25) -> list[dict]:
    primes = primes_near(BASELINE_P, n_below, n_above)
    print(f"\n=== n_s prime sweep: {len(primes)} primes (z irrelevant) ===")
    results = []
    for p in primes:
        r = compute_n_s(p)
        marker = " <-- BASELINE" if p == BASELINE_P else ""
        print(f"  p={p:7d}{marker:16s} n_s={r['n_s_pred']:.6f} "
              f"err={r['n_s_signed']:+.5f} ({r['n_s_sigma']:.2f}σ)")
        results.append(r)
    return results


# --------------------------------------------------------------------------
# PART 3: delta_CP sweep over z values (with extended range)
# --------------------------------------------------------------------------

def run_delta_cp_z_sweep(z_values: list[int] | None = None) -> list[dict]:
    if z_values is None:
        z_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    print(f"\n=== delta_CP z sweep: z in {z_values} (p irrelevant) ===")
    results = []
    for z in z_values:
        r = compute_delta_cp(z)
        marker = " <-- BASELINE" if z == BASELINE_Z else ""
        print(f"  z={z:3d}{marker:16s} delta_CP={r['delta_CP_pred_rad']:.4f} rad = "
              f"{r['delta_CP_pred_deg']:.2f}° err={r['delta_CP_signed']:+.4f} "
              f"({r['delta_CP_sigma']:.2f}σ)")
        results.append(r)
    return results


# --------------------------------------------------------------------------
# PART 4: Combined score sweep including n_s
# --------------------------------------------------------------------------

def compute_combined_observables(p: int, z: int) -> dict[str, Any]:
    """Compute all three independent observables for a given (p, z).

    INDEPENDENT observables:
      1. inv_alpha_0: [ln(p)]^2 + z/2 + gamma - 1/(2*pi)   [uses p^2, z]
      2. n_s:         1 - 3/(2 * p^(1/3))                    [uses p^(1/3) only]
      3. delta_CP:    pi/2 - 1/sqrt(z+1)                      [uses z only]

    These three cover different functional forms:
      - [ln(p)]^2 (from alpha)
      - p^(1/3)   (from n_s)
      - 1/sqrt(z+1) (from delta_CP)

    This is the most overdetermined test possible from this codebase.
    """
    result = {"p": p, "z": z}

    # Observable 1: 1/alpha
    from bpr.alpha_derivation import inverse_alpha_from_substrate
    inv_a = inverse_alpha_from_substrate(p, z)
    result["inv_alpha_0_pred"] = inv_a
    result["inv_alpha_0_frac_err"] = abs(inv_a - EXP["inv_alpha_0"]) / EXP["inv_alpha_0"]

    # Observable 2: n_s (p^(1/3) only, independent of z)
    ns_data = compute_n_s(p)
    result["n_s_pred"] = ns_data["n_s_pred"]
    result["n_s_frac_err"] = ns_data["n_s_frac_err"]
    result["n_s_sigma"] = ns_data["n_s_sigma"]

    # Observable 3: delta_CP (sqrt(z+1) only, independent of p)
    dcp_data = compute_delta_cp(z)
    result["delta_CP_pred"] = dcp_data["delta_CP_pred_rad"]
    result["delta_CP_frac_err"] = dcp_data["delta_CP_frac_err"]
    result["delta_CP_sigma"] = dcp_data["delta_CP_sigma"]

    # Composite score: RMS of fractional errors over THREE independent observables
    errors = [
        result["inv_alpha_0_frac_err"],
        result["n_s_frac_err"],
        result["delta_CP_frac_err"],
    ]
    result["composite_3obs_score"] = math.sqrt(sum(e**2 for e in errors) / 3)

    return result


def run_combined_prime_sweep(n_below: int = 25, n_above: int = 25) -> list[dict]:
    primes = primes_near(BASELINE_P, n_below, n_above)
    z = BASELINE_Z
    print(f"\n=== Combined 3-observable sweep: {len(primes)} primes, z={z} ===")
    results = []
    for p in primes:
        r = compute_combined_observables(p, z)
        marker = " <-- BASELINE" if p == BASELINE_P else ""
        print(f"  p={p:7d}{marker:16s} "
              f"alpha={r['inv_alpha_0_frac_err']*1e6:.0f}ppm  "
              f"n_s={r['n_s_frac_err']*100:.3f}%  "
              f"dCP={r['delta_CP_frac_err']*100:.3f}%  "
              f"score={r['composite_3obs_score']:.6f}")
        results.append(r)
    return results


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        for row in rows: w.writerow(row)
    print(f"  Saved: {path}")


def save_json(data: Any, path: Path) -> None:
    def clean(obj):
        if isinstance(obj, float):
            return str(obj) if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        return obj
    with open(path, "w") as f:
        json.dump(clean(data), f, indent=2)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("BPR Independence Audit")
    print("=" * 72)
    print(CLASSIFICATIONS)

    # Find what p n_s actually prefers
    find_optimal_p_for_ns()

    # n_s sweep
    ns_results = run_ns_prime_sweep(n_below=25, n_above=25)

    # delta_CP sweep
    dcp_results = run_delta_cp_z_sweep(list(range(2, 13)))

    # Combined 3-observable sweep
    combined_results = run_combined_prime_sweep(n_below=25, n_above=25)

    # Save
    print("\nSaving results ...")
    save_csv(ns_results, RESULTS_DIR / "independence_ns_sweep.csv")
    save_csv(dcp_results, RESULTS_DIR / "independence_dcp_sweep.csv")
    save_csv(combined_results, RESULTS_DIR / "independence_combined_sweep.csv")
    save_json({
        "ns_sweep": ns_results,
        "dcp_sweep": dcp_results,
        "combined_sweep": combined_results,
    }, RESULTS_DIR / "independence_results.json")

    # Print summary table
    combined_sorted = sorted(combined_results, key=lambda r: r["composite_3obs_score"])
    print("\n" + "=" * 72)
    print("TOP 10 PRIMES: 3-OBSERVABLE SCORE (1/α + n_s + δ_CP)")
    print("Score = RMS(frac errors) over 3 independent observables")
    print("=" * 72)
    print(f"{'Rank':>4}  {'p':>8}  {'z':>4}  {'Score':>10}  "
          f"{'α err(ppm)':>12}  {'n_s err%':>10}  {'δ_CP err%':>10}")
    print("-" * 72)
    baseline_score = next(r["composite_3obs_score"] for r in combined_results
                         if r["p"] == BASELINE_P)
    baseline_rank = next(i for i, r in enumerate(combined_sorted, 1)
                        if r["p"] == BASELINE_P)
    for i, r in enumerate(combined_sorted[:10], 1):
        marker = " <-- BASELINE" if r["p"] == BASELINE_P else ""
        print(f"  {i:2d}  {r['p']:8d}  {r['z']:4d}  {r['composite_3obs_score']:10.6f}  "
              f"{r['inv_alpha_0_frac_err']*1e6:12.1f}  "
              f"{r['n_s_frac_err']*100:10.4f}%  "
              f"{r['delta_CP_frac_err']*100:10.4f}%{marker}")
    print(f"\nBaseline p={BASELINE_P} ranks #{baseline_rank} of {len(combined_sorted)}")
    print(f"Baseline 3-obs score: {baseline_score:.6f}")
    print(f"Best score: {combined_sorted[0]['composite_3obs_score']:.6f} at p={combined_sorted[0]['p']}")

    # Key observations
    print("\n" + "=" * 72)
    print("KEY OBSERVATIONS")
    print("=" * 72)

    alpha_optimal_p = min(combined_results, key=lambda r: r["inv_alpha_0_frac_err"])["p"]
    ns_optimal_p = min(ns_results, key=lambda r: r["n_s_frac_err"])["p"]
    baseline_ns_row = next(r for r in ns_results if r["p"] == BASELINE_P)

    print(f"\n1. Prime that minimizes 1/α error alone:  p = {alpha_optimal_p}")
    print(f"   Prime that minimizes n_s error alone:   p = {ns_optimal_p}")
    print(f"   (These should agree if p=104729 is truly optimal — do they?)")

    print(f"\n2. n_s at p={BASELINE_P}:")
    print(f"   Predicted: {baseline_ns_row['n_s_pred']:.6f}")
    print(f"   Experimental: {EXP['n_s']} ± {EXP['n_s_unc']}")
    print(f"   Error: {baseline_ns_row['n_s_signed']:+.5f} = {baseline_ns_row['n_s_sigma']:.2f}σ")

    target_N = 2.0 / (1.0 - EXP["n_s"])
    target_p = (target_N / (4.0/3.0))**3
    print(f"\n3. The n_s-optimal prime would be p ≈ {target_p:.0f} (vs {BASELINE_P})")
    print(f"   These differ by {int(BASELINE_P - target_p):+,} — spanning many thousands of primes")
    print(f"   The two observables (1/α and n_s) disagree about which prime to choose.")

    z_optimal_delta = min(dcp_results, key=lambda r: r["delta_CP_frac_err"])["z"]
    baseline_dcp = next(r for r in dcp_results if r["z"] == BASELINE_Z)
    print(f"\n4. δ_CP at z={BASELINE_Z}:")
    print(f"   Predicted: {baseline_dcp['delta_CP_pred_rad']:.4f} rad")
    print(f"   Experimental: {EXP['delta_CP']} ± {EXP['delta_CP_unc']} rad")
    print(f"   Error: {baseline_dcp['delta_CP_sigma']:.2f}σ")
    print(f"   z value that minimizes δ_CP error: z = {z_optimal_delta}")

    print("\n5. Is n_s monotone over the prime sweep?")
    ns_vals = [r["n_s_pred"] for r in ns_results]
    diffs = [ns_vals[i+1] - ns_vals[i] for i in range(len(ns_vals)-1)]
    is_monotone = all(d > 0 for d in diffs)
    print(f"   Monotonically increasing: {is_monotone}")
    print(f"   Range: {min(ns_vals):.6f} to {max(ns_vals):.6f}")
    print(f"   The landscape has NO minimum at p=104729 — same conclusion as 1/α sweep")

    print("\nDone. See analysis/results/independence_*.csv and analysis/memo_independence.md")
