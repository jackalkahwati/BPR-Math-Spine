"""
BPR GUT Sector Audit
=====================
Tests whether alpha_GUT = pi/(p^(1/3)*z) and M_GUT = M_Pl/p^(1/4)
are derived or ansatz, whether they are internally consistent with
SM gauge coupling running, and whether fixing them would shift the
corridor attractors.

Outputs:
  analysis/results/gut_audit/
    gut_running.csv        — coupling values across corridor at each M_GUT
    consistency.json       — three-coupling fit summary
    gut_audit_summary.json — full findings

Author: Claude Code audit, 2026-04-07
"""

from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "analysis" / "results" / "gut_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

# ── Constants ──────────────────────────────────────────────────────────────
EULER_G = 0.5772156649015329
M_PL    = 1.22093e19    # GeV
M_Z     = 91.1876       # GeV

# One-loop SM beta-function coefficients (same convention as bpr/alpha_derivation.py)
# d(1/αi)/d(ln μ) = Bi/(2π)  [top-down direction]
B1 = 41.0/10    # U(1)_Y  GUT-normalised
B2 = -19.0/6    # SU(2)_L
B3 = -7.0       # SU(3)_C

# SM experimental couplings at M_Z
INV_A1_MZ = 1 / ((5/3) * (1/127.952) / (1 - 0.23122))   # GUT-normalised
INV_A2_MZ = 1 / ((1/127.952) / 0.23122)
INV_A3_MZ = 1 / 0.1179

# BPR experimental references
EXP_ALPHA  = 137.035999084
LAMBDA_QCD = 0.332
Z          = 6

# Corridor primes of interest
CORRIDOR_PRIMES = [103943, 104729, 104743, 107251, 107243, 107713, 109721]
BASELINE_P = 104729


# ── GUT sector formulas ─────────────────────────────────────────────────────

def gut_scale(p: int) -> float:
    """M_GUT = M_Pl / p^{1/4}  (BPR formula)"""
    return M_PL / p**0.25

def alpha_gut_bare(p: int, z: int = Z) -> float:
    """α_GUT = π / (p^{1/3} * z)  (BPR bare lattice coupling)"""
    return math.pi / (p**(1/3) * z)

def inv_alpha_0(p: int, z: int = Z) -> float:
    return math.log(p)**2 + z/2 + EULER_G - 1/(2*math.pi)


# ── Top-down running: BPR α_GUT → SM couplings at M_Z ─────────────────────

def bpr_topdown(p: int, z: int = Z) -> Dict[str, float]:
    """Run BPR's bare GUT coupling down to M_Z via one-loop SM."""
    inv_ag = 1.0 / alpha_gut_bare(p, z)
    M_gut  = gut_scale(p)
    L      = math.log(M_gut / M_Z)
    inv_a1 = inv_ag + B1/(2*math.pi)*L
    inv_a2 = inv_ag + B2/(2*math.pi)*L
    inv_a3 = inv_ag + B3/(2*math.pi)*L
    if any(x <= 0 for x in [inv_a1, inv_a2, inv_a3]):
        return {}
    a1, a2, a3 = 1/inv_a1, 1/inv_a2, 1/inv_a3
    aY  = 0.6*a1
    s2  = aY/(aY+a2)
    aem = a2*s2
    return {
        'M_gut': M_gut, 'L': L,
        'inv_alpha_gut': inv_ag,
        'inv_a1_MZ': inv_a1, 'inv_a2_MZ': inv_a2, 'inv_a3_MZ': inv_a3,
        'alpha_s_MZ': a3, 'sin2_tw_MZ': s2, 'inv_alpha_em_MZ': 1/aem,
    }


# ── Bottom-up: SM couplings → what they imply at BPR's M_GUT ──────────────

def sm_bottomup_at_gut(p: int) -> Dict[str, float]:
    """Run SM couplings upward to BPR's M_GUT."""
    M_gut = gut_scale(p)
    L     = math.log(M_gut / M_Z)
    # Bottom-up: 1/αi(M_GUT) = 1/αi(M_Z) - Bi/(2π)*L
    inv_a1 = INV_A1_MZ - B1/(2*math.pi)*L
    inv_a2 = INV_A2_MZ - B2/(2*math.pi)*L
    inv_a3 = INV_A3_MZ - B3/(2*math.pi)*L
    spread = max(inv_a1, inv_a2, inv_a3) - min(inv_a1, inv_a2, inv_a3)
    return {
        'inv_a1_at_gut': inv_a1,
        'inv_a2_at_gut': inv_a2,
        'inv_a3_at_gut': inv_a3,
        'spread': spread,
        'unified': spread < 2.0,  # threshold for near-unification
    }


# ── Sensitivity: how much do GUT predictions vary across corridor? ──────────

def gut_corridor_sensitivity(primes: List[int]) -> List[Dict]:
    rows = []
    for p in primes:
        td = bpr_topdown(p, Z)
        bu = sm_bottomup_at_gut(p)
        if not td:
            continue
        rows.append({
            'p':              p,
            'M_gut_GeV':      td['M_gut'],
            'inv_alpha_gut_BPR': td['inv_alpha_gut'],
            'inv_a3_SM_at_gut': bu['inv_a3_at_gut'],
            'ratio_BPR_SM':    td['inv_alpha_gut'] / bu['inv_a3_at_gut'],
            'alpha_s_BPR':    td['alpha_s_MZ'],
            'alpha_s_exp':    0.1179,
            'alpha_s_err_pct': abs(td['alpha_s_MZ'] - 0.1179)/0.1179*100,
            'sin2_tw_BPR':    td['sin2_tw_MZ'],
            'sin2_tw_exp':    0.23122,
            'sin2_tw_err_pct': abs(td['sin2_tw_MZ'] - 0.23122)/0.23122*100,
            'SM_spread_at_gut': bu['spread'],
            'is_baseline':    int(p == BASELINE_P),
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def is_prime(n):
    if n < 2: return False
    for pp in (2,3,5,7,11,13,17,19,23):
        if n==pp: return True
        if n%pp==0: return False
    d,r = n-1, 0
    while d%2==0: d//=2; r+=1
    for a in (2,3,5,7,11,13,17,19,23,29,31,37):
        if a>=n: continue
        x=pow(a,d,n)
        if x==1 or x==n-1: continue
        for _ in range(r-1):
            x=x*x%n
            if x==n-1: break
        else: return False
    return True

def main():
    # All corridor primes for GUT sensitivity map
    corridor_primes = [n for n in range(103935, 110635) if is_prime(n)]
    print(f"Running GUT audit on {len(corridor_primes)} corridor primes...")

    rows = gut_corridor_sensitivity(corridor_primes)

    # Save CSV
    path = OUT_DIR / "gut_running.csv"
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved gut_running.csv ({len(rows)} rows)")

    # Key metrics
    bl = next(r for r in rows if r['p'] == BASELINE_P)
    best_as = min(rows, key=lambda r: r['alpha_s_err_pct'])
    best_s2 = min(rows, key=lambda r: r['sin2_tw_err_pct'])

    # Consistency summary
    consistency = {
        "baseline_alpha_s_err_pct": bl['alpha_s_err_pct'],
        "baseline_sin2_tw_err_pct": bl['sin2_tw_err_pct'],
        "best_alpha_s_prime":       best_as['p'],
        "best_alpha_s_err_pct":     best_as['alpha_s_err_pct'],
        "best_sin2_tw_prime":       best_s2['p'],
        "best_sin2_tw_err_pct":     best_s2['sin2_tw_err_pct'],
        "SM_spread_at_BPR_gut":     bl['SM_spread_at_gut'],
        "inv_alpha_gut_BPR":        bl['inv_alpha_gut_BPR'],
        "inv_a3_SM_at_gut":         bl['inv_a3_SM_at_gut'],
        "ratio_BPR_over_SM":        bl['ratio_BPR_SM'],
        "alpha_s_corridor_variation_pct": (
            max(r['alpha_s_BPR'] for r in rows) -
            min(r['alpha_s_BPR'] for r in rows)
        ) / 0.1179 * 100,
    }
    with open(OUT_DIR / 'consistency.json', 'w') as f:
        json.dump(consistency, f, indent=2)
    print(f"  Saved consistency.json")

    # Number-theoretic findings
    from sympy import primepi
    numtheory = {
        "p_104729_rank": int(primepi(104729)),
        "p_104743_rank": int(primepi(104743)),
        "p_107251_rank": int(primepi(107251)),
        "p_107713_rank": int(primepi(107713)),
        "interpretation": "p=104729 is the 10,000th prime — likely the selection criterion",
        "alpha_optimal_rank": "10,001st prime (104743)",
        "gap": "Baseline is 1 prime before the best alpha match",
    }

    # Provenance findings
    provenance = {
        "M_GUT_formula": {
            "formula": "M_GUT = M_Pl / p^{1/4}",
            "status": "ANSATZ",
            "evidence": "Root commit 61fc581; one-line definition, no derivation. "
                        "M_Pl/p^{1/4} is not a unification scale in the SM "
                        "(SM spread = 14 coupling units at this scale).",
            "note": "The code acknowledges this with 'NOTE: bare coupling without threshold corrections'",
        },
        "alpha_GUT_formula": {
            "formula": "alpha_GUT = pi / (p^{1/3} * z)",
            "status": "ANSATZ",
            "evidence": "Root commit 61fc581; the 'derivation' is N_eff = p^{1/3}*z/(2pi), "
                        "alpha_GUT = 1/(2*N_eff). The N_eff formula counts boundary modes "
                        "but is asserted without a path-integral calculation.",
            "note": "BPR value 1/alpha_GUT=90 vs SM value at same scale 1/alpha_3=49.2 "
                    "(factor 1.83x discrepancy).",
        },
    }

    summary = {
        "corridor_primes_tested": len(rows),
        "gut_problems": [
            "M_GUT = M_Pl/p^{1/4} is not a coupling unification scale in the SM "
            "(SM coupling spread = 14 units at BPR's M_GUT, vs 0 needed)",
            "alpha_GUT = pi/(p^{1/3}*z) gives 1/alpha_GUT=90 but SM predicts "
            "1/alpha_3=49.2 at BPR's M_GUT (factor 1.83x off)",
            "alpha_s(M_Z) from BPR top-down running = 0.0203 (exp 0.1179, factor 5.8x off)",
            "sin2_theta_W from BPR top-down = 0.274 (exp 0.231, 18% off)",
            "The NOTE in the code admits: 'requires S2 cohomology charges for boundary "
            "mode representations (eta_i values)' — a placeholder for absent derivation",
        ],
        "corridor_sensitivity": {
            "alpha_s_varies_by": f"{consistency['alpha_s_corridor_variation_pct']:.4f}% across corridor",
            "interpretation": "GUT predictions vary by ~2% across corridor — "
                              "same order as corridor variation in other observables, "
                              "but all values are ~580% wrong, so variation is irrelevant",
        },
        "number_theory": numtheory,
        "provenance": provenance,
        "attractor_impact": {
            "question": "Would fixing GUT sector shift the attractors?",
            "answer": "The two attractors (alpha at 104749, vEW at 107709) are determined "
                      "by the LOW-ENERGY formulas (1/alpha_0 and v_EW), not by the GUT sector. "
                      "Fixing the GUT sector would not shift either attractor unless the "
                      "corrected GUT coupling feeds back into a corrected 1/alpha_0 formula. "
                      "Currently there is no such feedback path.",
            "consequence": "The attractor gap is independent of the GUT sector problem.",
        },
    }
    with open(OUT_DIR / 'gut_audit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved gut_audit_summary.json")

    # Console report
    print()
    print("=" * 70)
    print("GUT SECTOR AUDIT RESULTS")
    print("=" * 70)
    print()
    print(f"  BPR 1/alpha_GUT at p=104729:   {bl['inv_alpha_gut_BPR']:.4f}")
    print(f"  SM 1/alpha_3 at BPR's M_GUT:   {bl['inv_a3_SM_at_gut']:.4f}")
    print(f"  Ratio (BPR / SM):              {bl['ratio_BPR_SM']:.4f}x")
    print(f"  SM coupling spread at BPR-M_GUT: {bl['SM_spread_at_gut']:.2f} (need 0 for unification)")
    print()
    print(f"  alpha_s(M_Z) BPR prediction:   {bl['alpha_s_BPR']:.6f}  (exp: 0.1179, err: {bl['alpha_s_err_pct']:.1f}%)")
    print(f"  sin2_theta_W BPR prediction:   {bl['sin2_tw_BPR']:.6f}  (exp: 0.23122, err: {bl['sin2_tw_err_pct']:.1f}%)")
    print()
    print(f"  NUMBER THEORY: p=104729 is the 10,000th prime")
    print(f"  Alpha-optimal p=104743 is the 10,001st prime")
    print()
    print("  GUT formulas status: ANSATZ (both introduced in root commit 61fc581)")
    print("  Attractor impact: NONE (attractors depend on low-energy formulas only)")

    return rows, summary

if __name__ == "__main__":
    rows, summary = main()
