"""
BPR Parameter Uniqueness Audit — Sweep Harness
================================================

Systematically tests whether p = 104729 and z = 6 are structurally unique
by sweeping over nearby primes and z values, collecting all observables, and
computing a composite score.

Outputs (written to analysis/results/):
  sweep_results.csv        — one row per (p, z) pair
  sweep_results.json       — same, structured
  prime_sweep.csv          — prime-only sweep at z = 6
  z_ablation.csv           — z-only sweep at p = 104729
  joint_sweep.csv          — joint grid (nearby primes × z values)

Usage:
  cd /home/user/BPR-Math-Spine
  python analysis/sweep.py

Design decisions (documented up front, not post-hoc):
  - Composite score = RMS of fractional errors across INDEPENDENT observables
  - Only observables that actually vary with (p, z) are included
  - Non-independent / circular observables are noted but excluded from score
  - Normalization: fractional error = |predicted - expected| / |expected|
  - No free weights; all observables equally weighted in RMS

Author: Claude Code audit, 2026-04-06
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add repo root to path so bpr modules are importable
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Experimental reference values (CODATA 2018 / PDG 2024)
# ---------------------------------------------------------------------------

EXP = {
    "inv_alpha_0":   137.035999084,   # CODATA 2018
    "inv_alpha_MZ":  127.952,         # PDG 2024
    "v_EW_GeV":      246.22,          # PDG 2024 (Higgs VEV)
    "sin2_theta_W":  0.23122,         # PDG 2024 MS-bar
    "Omega_Lambda":  0.6889,          # Planck 2018
    "m_e_MeV":       0.51099895,      # CODATA 2018
    "m_mu_MeV":      105.6583755,     # PDG 2024
    "m_tau_MeV":     1776.86,         # PDG 2024 (anchor)
}

# Independent observables used in composite score.
# EXCLUDED from score (with reason noted in comments):
#   inv_alpha_MZ  — trivially = inv_alpha_0 - 9.084; not independent
#   sin2_theta_W  — GaugeCouplingRunning returns exactly 0.23122 for all p
#                   (circular: threshold corrections constructed to force unification)
#   Omega_Lambda  — dark_energy_from_impedance uses cosmological p_cosmo (~2.7e61),
#                   not the local substrate p; returns ~1e-104 for all p
#   m_e, m_mu     — ChargedLeptonSpectrum takes no p or z arguments; constant
#   m_tau         — anchor mass (experimental input), not a prediction

SCORE_OBSERVABLES = ["inv_alpha_0", "v_EW_GeV"]

# ---------------------------------------------------------------------------
# Primality test (Miller-Rabin, deterministic for n < 3.3e24)
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Witnesses sufficient for n < 3,215,031,751 and up to 3.3e24
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_near(center: int, n_below: int = 25, n_above: int = 25) -> list[int]:
    """Return up to n_below primes below center and n_above primes above center.

    center itself is included if prime.
    """
    below = []
    k = center - 1 if _is_prime(center) else center
    while len(below) < n_below and k > 2:
        if _is_prime(k):
            below.append(k)
        k -= 1

    above = []
    k = center + 1
    while len(above) < n_above:
        if _is_prime(k):
            above.append(k)
        k += 1

    primes = sorted(below + ([center] if _is_prime(center) else []) + above)
    return primes

# ---------------------------------------------------------------------------
# Observable computation
# ---------------------------------------------------------------------------

def compute_observables(p: int, z: int) -> dict[str, Any]:
    """Compute all BPR observables for a given (p, z) pair.

    Returns a dict with predicted values, errors, and metadata.
    Some observables are flagged as NON_INDEPENDENT because they do not
    actually vary with p or z.
    """
    result: dict[str, Any] = {
        "p": p,
        "z": z,
        "is_prime": _is_prime(p),
    }

    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Observable 1: 1/α(q² = 0)
    # Formula: [ln(p)]² + z/2 + γ − 1/(2π)
    # Genuinely depends on BOTH p and z.
    # -------------------------------------------------------------------------
    try:
        from bpr.alpha_derivation import inverse_alpha_from_substrate
        inv_alpha_0 = inverse_alpha_from_substrate(p, z)
        result["inv_alpha_0_pred"] = inv_alpha_0
        result["inv_alpha_0_frac_err"] = abs(inv_alpha_0 - EXP["inv_alpha_0"]) / EXP["inv_alpha_0"]
        result["inv_alpha_0_ppm"] = result["inv_alpha_0_frac_err"] * 1e6
        result["inv_alpha_0_status"] = "ok"
    except Exception as e:
        result["inv_alpha_0_pred"] = float("nan")
        result["inv_alpha_0_frac_err"] = float("nan")
        result["inv_alpha_0_ppm"] = float("nan")
        result["inv_alpha_0_status"] = f"error: {e}"

    # -------------------------------------------------------------------------
    # Observable 2: 1/α(M_Z)
    # = 1/α(0) - Δ(1/α)_QED where Δ = 9.084 (fixed SM running)
    # NOTE: NOT independent from inv_alpha_0; excluded from composite score.
    # -------------------------------------------------------------------------
    try:
        from bpr.alpha_derivation import inverse_alpha_at_MZ
        inv_alpha_MZ = inverse_alpha_at_MZ(p, z)
        result["inv_alpha_MZ_pred"] = inv_alpha_MZ
        result["inv_alpha_MZ_frac_err"] = abs(inv_alpha_MZ - EXP["inv_alpha_MZ"]) / EXP["inv_alpha_MZ"]
        result["inv_alpha_MZ_status"] = "ok [non-independent of inv_alpha_0]"
    except Exception as e:
        result["inv_alpha_MZ_pred"] = float("nan")
        result["inv_alpha_MZ_frac_err"] = float("nan")
        result["inv_alpha_MZ_status"] = f"error: {e}"

    # -------------------------------------------------------------------------
    # Observable 3: Electroweak scale (Higgs VEV)
    # Formula: Λ_QCD × p^(1/3) × (ln(p) + z − 2)
    # Genuinely depends on BOTH p and z.
    # -------------------------------------------------------------------------
    try:
        from bpr.gauge_unification import electroweak_scale_GeV
        v_EW = electroweak_scale_GeV(p, z)
        result["v_EW_GeV_pred"] = v_EW
        result["v_EW_GeV_frac_err"] = abs(v_EW - EXP["v_EW_GeV"]) / EXP["v_EW_GeV"]
        result["v_EW_GeV_status"] = "ok"
    except Exception as e:
        result["v_EW_GeV_pred"] = float("nan")
        result["v_EW_GeV_frac_err"] = float("nan")
        result["v_EW_GeV_status"] = f"error: {e}"

    # -------------------------------------------------------------------------
    # Observable 4: sin²θ_W (weak mixing angle)
    # NOTE: GaugeCouplingRunning.weinberg_angle_at_MZ constructs threshold
    # corrections that force unification, then reconstructs starting values.
    # It returns ≈ 0.23122 for ALL p values (the hardcoded experimental input).
    # This is a mathematical tautology; flagged NON_INDEPENDENT.
    # -------------------------------------------------------------------------
    try:
        from bpr.gauge_unification import GaugeCouplingRunning
        gcr = GaugeCouplingRunning(p=p)
        sin2_TW = gcr.weinberg_angle_at_MZ
        result["sin2_theta_W_pred"] = sin2_TW
        result["sin2_theta_W_frac_err"] = abs(sin2_TW - EXP["sin2_theta_W"]) / EXP["sin2_theta_W"]
        result["sin2_theta_W_status"] = "ok [CIRCULAR: returns hardcoded exp value]"
    except Exception as e:
        result["sin2_theta_W_pred"] = float("nan")
        result["sin2_theta_W_frac_err"] = float("nan")
        result["sin2_theta_W_status"] = f"error: {e}"

    # -------------------------------------------------------------------------
    # Observable 5: Lepton masses
    # NOTE: ChargedLeptonSpectrum takes no p or z parameters.
    # Masses are CONSTANT across all sweeps.
    # m_tau is the anchor (1 experimental input); m_e, m_mu are derived
    # from fixed l_modes = (1, 14, 59) ratios.
    # Flagged NON_INDEPENDENT.
    # -------------------------------------------------------------------------
    try:
        from bpr.charged_leptons import ChargedLeptonSpectrum
        ls = ChargedLeptonSpectrum()
        m_e, m_mu, m_tau = ls.masses_MeV
        result["m_e_MeV_pred"] = m_e
        result["m_mu_MeV_pred"] = m_mu
        result["m_tau_MeV_pred"] = m_tau
        result["m_e_frac_err"] = abs(m_e - EXP["m_e_MeV"]) / EXP["m_e_MeV"]
        result["m_mu_frac_err"] = abs(m_mu - EXP["m_mu_MeV"]) / EXP["m_mu_MeV"]
        result["m_e_status"] = "ok [NON-INDEPENDENT: no p/z parameters]"
        result["m_mu_status"] = "ok [NON-INDEPENDENT: no p/z parameters]"
    except Exception as e:
        result["m_e_MeV_pred"] = float("nan")
        result["m_mu_MeV_pred"] = float("nan")
        result["m_tau_MeV_pred"] = float("nan")
        result["m_e_frac_err"] = float("nan")
        result["m_mu_frac_err"] = float("nan")
        result["m_e_status"] = f"error: {e}"
        result["m_mu_status"] = f"error: {e}"

    # -------------------------------------------------------------------------
    # Observable 6: Ω_Λ (dark energy fraction)
    # NOTE: dark_energy_from_impedance uses p_cosmo = R_H / L_Planck ≈ 2.7e61,
    # NOT the local substrate p. The local p argument is stored but unused in
    # the main formula. Returns ~1e-104 for all p values (wildly wrong).
    # Flagged NON_INDEPENDENT.
    # -------------------------------------------------------------------------
    try:
        from bpr.cross_predictions import dark_energy_from_impedance
        de = dark_energy_from_impedance(p)
        result["Omega_Lambda_pred"] = de["Omega_Lambda"]
        result["Omega_Lambda_frac_err"] = abs(de["Omega_Lambda"] - EXP["Omega_Lambda"]) / EXP["Omega_Lambda"]
        result["Omega_Lambda_status"] = "ok [NON-INDEPENDENT: uses cosmological p_cosmo, not substrate p]"
    except Exception as e:
        result["Omega_Lambda_pred"] = float("nan")
        result["Omega_Lambda_frac_err"] = float("nan")
        result["Omega_Lambda_status"] = f"error: {e}"

    # -------------------------------------------------------------------------
    # Derived: GUT scale (varies with p only via M_GUT = M_Pl / p^(1/4))
    # -------------------------------------------------------------------------
    try:
        from bpr.alpha_derivation import gut_scale_GeV
        result["M_GUT_GeV"] = gut_scale_GeV(p)
    except Exception as e:
        result["M_GUT_GeV"] = float("nan")

    # -------------------------------------------------------------------------
    # Derived: alpha_GUT from lattice (depends on both p and z)
    # -------------------------------------------------------------------------
    try:
        from bpr.alpha_derivation import alpha_gut_from_lattice
        result["alpha_GUT"] = alpha_gut_from_lattice(p, z)
    except Exception as e:
        result["alpha_GUT"] = float("nan")

    # -------------------------------------------------------------------------
    # Composite score: RMS of fractional errors over INDEPENDENT observables
    # Score = sqrt(mean of (frac_err_i)^2) for i in SCORE_OBSERVABLES
    # Lower is better.
    # -------------------------------------------------------------------------
    errors = []
    for obs in SCORE_OBSERVABLES:
        key = f"{obs}_frac_err"
        v = result.get(key, float("nan"))
        if not math.isnan(v):
            errors.append(v)

    if errors:
        result["composite_score"] = math.sqrt(sum(e**2 for e in errors) / len(errors))
        result["n_observables_scored"] = len(errors)
    else:
        result["composite_score"] = float("nan")
        result["n_observables_scored"] = 0

    result["runtime_s"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def run_baseline() -> dict[str, Any]:
    """Run the baseline at p = 104729, z = 6."""
    print("Running baseline: p=104729, z=6 ...")
    r = compute_observables(104729, 6)
    print(f"  1/α(0):          {r.get('inv_alpha_0_pred', 'ERR'):.6f}  (exp: {EXP['inv_alpha_0']:.6f})")
    print(f"  1/α(M_Z):        {r.get('inv_alpha_MZ_pred', 'ERR'):.6f}  (exp: {EXP['inv_alpha_MZ']:.3f})")
    print(f"  v_EW (GeV):      {r.get('v_EW_GeV_pred', 'ERR'):.4f}  (exp: {EXP['v_EW_GeV']:.2f})")
    print(f"  sin²θ_W:         {r.get('sin2_theta_W_pred', 'ERR'):.5f}  (exp: {EXP['sin2_theta_W']:.5f})")
    print(f"  Composite score: {r.get('composite_score', 'ERR'):.6f}")
    return r


def run_prime_sweep(z: int = 6, n_below: int = 25, n_above: int = 25) -> list[dict[str, Any]]:
    """Sweep over nearby primes with fixed z."""
    center = 104729
    primes = primes_near(center, n_below, n_above)
    print(f"\nPrime sweep: {len(primes)} primes around {center}, z={z} ...")
    results = []
    for i, p in enumerate(primes):
        marker = " <-- BASELINE" if p == center else ""
        r = compute_observables(p, z)
        score = r.get("composite_score", float("nan"))
        print(f"  [{i+1:3d}/{len(primes)}] p={p:7d}{marker:16s}  score={score:.6f}")
        results.append(r)
    return results


def run_z_ablation(p: int = 104729, z_values: list[int] | None = None) -> list[dict[str, Any]]:
    """Sweep over z values with fixed p."""
    if z_values is None:
        z_values = [4, 5, 6, 7, 8]
    print(f"\nZ ablation: p={p}, z in {z_values} ...")
    results = []
    for z in z_values:
        marker = " <-- BASELINE" if z == 6 else ""
        r = compute_observables(p, z)
        score = r.get("composite_score", float("nan"))
        inv_a = r.get("inv_alpha_0_pred", float("nan"))
        v_ew = r.get("v_EW_GeV_pred", float("nan"))
        print(f"  z={z}{marker:16s}  1/α={inv_a:.4f}  v_EW={v_ew:.2f} GeV  score={score:.6f}")
        results.append(r)
    return results


def run_joint_sweep(
    n_primes: int = 9,
    z_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Joint sweep over a grid of nearby primes × z values."""
    if z_values is None:
        z_values = [4, 5, 6, 7, 8]
    center = 104729
    n_side = n_primes // 2
    primes = primes_near(center, n_below=n_side, n_above=n_side)
    if len(primes) > n_primes:
        primes = primes[:n_primes]

    total = len(primes) * len(z_values)
    print(f"\nJoint sweep: {len(primes)} primes × {len(z_values)} z values = {total} runs ...")
    results = []
    i = 0
    for p in primes:
        for z in z_values:
            i += 1
            r = compute_observables(p, z)
            score = r.get("composite_score", float("nan"))
            marker = " <-- BASELINE" if (p == center and z == 6) else ""
            print(f"  [{i:3d}/{total}] p={p:7d}  z={z}{marker:16s}  score={score:.6f}")
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save list of dicts to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved: {path}")


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON with float cleaning."""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("BPR Parameter Uniqueness Audit — Sweep Harness")
    print("=" * 70)
    print()
    print("Independent observables in composite score:", SCORE_OBSERVABLES)
    print("Score = RMS(fractional errors), lower is better")
    print()
    print("NOTE: The following observables are flagged as NON-INDEPENDENT")
    print("  and EXCLUDED from the composite score:")
    print("  - inv_alpha_MZ  : trivially derived from inv_alpha_0 (subtract 9.084)")
    print("  - sin2_theta_W  : GaugeCouplingRunning is circular; returns 0.23122 for all p")
    print("  - Omega_Lambda  : uses cosmological p_cosmo, not substrate p; returns ~1e-104")
    print("  - m_e, m_mu     : ChargedLeptonSpectrum has no p/z parameters; constant")
    print()

    all_results: dict[str, Any] = {}

    # 1. Baseline
    baseline = run_baseline()
    all_results["baseline"] = baseline

    # 2. Prime sweep (25 below, 25 above)
    prime_results = run_prime_sweep(z=6, n_below=25, n_above=25)
    all_results["prime_sweep"] = prime_results

    # 3. Z ablation
    z_results = run_z_ablation(p=104729, z_values=[4, 5, 6, 7, 8])
    all_results["z_ablation"] = z_results

    # 4. Joint sweep (9 primes × 5 z values = 45 runs)
    joint_results = run_joint_sweep(n_primes=9, z_values=[4, 5, 6, 7, 8])
    all_results["joint_sweep"] = joint_results

    # Combine all for master CSV
    all_rows = prime_results + z_results + joint_results
    # Deduplicate by (p, z) key
    seen = set()
    unique_rows = []
    for r in all_rows:
        key = (r["p"], r["z"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)
    unique_rows.sort(key=lambda r: (r["composite_score"] if not math.isnan(r.get("composite_score", float("nan"))) else 1e9))

    print("\nSaving results ...")
    save_csv(prime_results, RESULTS_DIR / "prime_sweep.csv")
    save_csv(z_results, RESULTS_DIR / "z_ablation.csv")
    save_csv(joint_results, RESULTS_DIR / "joint_sweep.csv")
    save_csv(unique_rows, RESULTS_DIR / "sweep_results.csv")
    save_json(all_results, RESULTS_DIR / "sweep_results.json")

    # Print ranked summary
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER PAIRS BY COMPOSITE SCORE (lower = better fit)")
    print("=" * 70)
    print(f"{'Rank':>4}  {'p':>8}  {'z':>4}  {'Score':>10}  {'1/α err(ppm)':>14}  {'v_EW err%':>10}")
    print("-" * 70)
    for i, r in enumerate(unique_rows[:10], 1):
        score = r.get("composite_score", float("nan"))
        alpha_ppm = r.get("inv_alpha_0_ppm", float("nan"))
        vew_err = r.get("v_EW_GeV_frac_err", float("nan")) * 100
        baseline_marker = " <-- BASELINE" if (r["p"] == 104729 and r["z"] == 6) else ""
        print(f"  {i:2d}  {r['p']:8d}  {r['z']:4d}  {score:10.6f}  {alpha_ppm:14.1f}  {vew_err:10.4f}%{baseline_marker}")

    print("\nDone. Run analysis/analyze.py to generate plots and summary.")
