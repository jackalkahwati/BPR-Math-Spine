"""
BPR Solution Space Mapper
==========================

Maps the full viable parameter space in (p, z), distinguishing core-derived
structure from ansatz-driven fits. Does NOT assume uniqueness.

Goals:
  - Broad sweep: p in [30000, 300000] (sampled primes), z in [2, 12]
  - Classify every observable by provenance
  - Dual score: core-only vs all-in
  - Pareto frontier analysis
  - Cluster viable regions
  - Locate (104729, 6) in the landscape

Outputs (analysis/results/solution_space/):
  all_points.csv         — full (p, z, observables, scores) table
  pareto_front.csv       — Pareto-optimal points
  z_profiles.csv         — best p per z
  p_profiles.csv         — best z per p
  clusters.json          — cluster assignments
  landscape_summary.json — key findings

Author: Claude Code audit, 2026-04-06
"""

from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "analysis" / "results" / "solution_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Experimental reference values (CODATA 2018 / PDG 2024)
# ---------------------------------------------------------------------------

EXP = {
    "inv_alpha_0":  137.035999084,  # CODATA 2018
    "v_EW_GeV":     246.22,         # PDG 2024 (Higgs VEV)
    "n_s":          0.9649,         # Planck 2018
    "delta_cp_rad": 1.196,          # PDG 2024 (CKM CP phase, radians)
}

EXP_SIGMA = {
    "inv_alpha_0":  0.000000021,    # CODATA uncertainty
    "v_EW_GeV":     0.02,           # ~0.008% — well measured
    "n_s":          0.0042,         # Planck 2018 1σ
    "delta_cp_rad": 0.045,          # PDG 2024 1σ
}

# ---------------------------------------------------------------------------
# Observable provenance classification
# ---------------------------------------------------------------------------

PROVENANCE = {
    "inv_alpha_0": {
        "status": "core-derived",
        "formula": "ln(p)^2 + z/2 + γ - 1/(2π)",
        "notes": "4 physically motivated terms; z/2 is bare coupling contribution",
        "depends_on": ["p", "z"],
        "include_core": True,
    },
    "v_EW_GeV": {
        "status": "bridge",
        "formula": "Λ_QCD × p^{1/3} × (ln(p) + z − 2)",
        "notes": "Dimensional hierarchy argument; motivated but not uniquely derived",
        "depends_on": ["p", "z"],
        "include_core": True,
    },
    "n_s": {
        "status": "phenomenological-ansatz",
        "formula": "1 − 2/N, N = p^{1/3} × (1 + 1/d), d=3",
        "notes": "Starobinsky formula is standard; N=p^{1/3}×4/3 asserted, no slow-roll calc",
        "depends_on": ["p"],
        "include_core": False,
    },
    "delta_cp_rad": {
        "status": "ansatz",
        "formula": "π/2 − 1/√(z+1)",
        "notes": "Fallback=1.196=PDG value; +1 offset underived; form chosen to select z=6",
        "depends_on": ["z"],
        "include_core": False,
    },
    "sin2_theta_w": {
        "status": "circular",
        "formula": "GaugeCouplingRunning.weinberg_angle_at_MZ",
        "notes": "Returns hardcoded 0.23122 for all p,z. Mathematically guaranteed by construction.",
        "depends_on": [],
        "include_core": False,
    },
    "omega_lambda": {
        "status": "broken",
        "formula": "dark_energy_from_impedance(p)",
        "notes": "Uses Hubble-radius mode count, not substrate p; returns ~1e-104 (wrong by 100 orders)",
        "depends_on": [],
        "include_core": False,
    },
    "inv_alpha_MZ": {
        "status": "downstream",
        "formula": "inv_alpha_0 - 9.084",
        "notes": "Trivially downstream of inv_alpha_0; not independent information",
        "depends_on": ["p", "z"],
        "include_core": False,
    },
}

# ---------------------------------------------------------------------------
# Primality test (Miller-Rabin, deterministic for n < 3.3e24)
# ---------------------------------------------------------------------------

_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        if n == p:
            return True
        if n % p == 0:
            return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_WITNESSES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_in_range(lo: int, hi: int, max_count: int = 500) -> List[int]:
    """Sample primes in [lo, hi] with roughly logarithmic spacing."""
    # Collect all primes first (or sample)
    result = []
    n = lo if lo % 2 != 0 else lo + 1
    while n <= hi and len(result) < max_count * 10:
        if _is_prime(n):
            result.append(n)
        n += 2
    # If we have more than max_count, sample logarithmically
    if len(result) > max_count:
        indices = np.unique(np.round(
            np.logspace(0, np.log10(len(result) - 1), max_count)
        ).astype(int))
        result = [result[i] for i in indices if i < len(result)]
    return result


# ---------------------------------------------------------------------------
# Observable computation
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015329
_LAMBDA_QCD_GEV = 0.332   # matches bpr/gauge_unification.py line 27


def compute_inv_alpha_0(p: int, z: int) -> float:
    ln_p = math.log(p)
    return ln_p ** 2 + z / 2.0 + _EULER_GAMMA - 1.0 / (2.0 * math.pi)


def compute_v_EW_GeV(p: int, z: int) -> float:
    ln_p = math.log(p)
    return _LAMBDA_QCD_GEV * (p ** (1.0 / 3.0)) * (ln_p + z - 2.0)


def compute_n_s(p: int) -> float:
    N = (p ** (1.0 / 3.0)) * (4.0 / 3.0)  # d=3 fixed
    return 1.0 - 2.0 / N


def compute_delta_cp(z: int) -> float:
    return math.pi / 2.0 - 1.0 / math.sqrt(z + 1.0)


def fractional_error(predicted: float, experimental: float) -> float:
    return abs(predicted - experimental) / abs(experimental)


def sigma_error(predicted: float, key: str) -> float:
    exp_val = EXP[key]
    sigma = EXP_SIGMA[key]
    return abs(predicted - exp_val) / sigma


def compute_all_observables(p: int, z: int) -> Dict[str, Any]:
    obs: Dict[str, Any] = {}

    ia = compute_inv_alpha_0(p, z)
    obs["inv_alpha_0_pred"] = ia
    obs["inv_alpha_0_frac_err"] = fractional_error(ia, EXP["inv_alpha_0"])
    obs["inv_alpha_0_sigma"] = sigma_error(ia, "inv_alpha_0")

    vew = compute_v_EW_GeV(p, z)
    obs["v_EW_pred"] = vew
    obs["v_EW_frac_err"] = fractional_error(vew, EXP["v_EW_GeV"])
    obs["v_EW_sigma"] = sigma_error(vew, "v_EW_GeV")

    ns = compute_n_s(p)
    obs["n_s_pred"] = ns
    obs["n_s_frac_err"] = fractional_error(ns, EXP["n_s"])
    obs["n_s_sigma"] = sigma_error(ns, "n_s")

    dcp = compute_delta_cp(z)
    obs["delta_cp_pred"] = dcp
    obs["delta_cp_frac_err"] = fractional_error(dcp, EXP["delta_cp_rad"])
    obs["delta_cp_sigma"] = sigma_error(dcp, "delta_cp_rad")

    # Core-only score (inv_alpha + v_EW): both genuinely parameter-dependent
    core_errs = [obs["inv_alpha_0_frac_err"], obs["v_EW_frac_err"]]
    obs["score_core"] = math.sqrt(sum(e ** 2 for e in core_errs) / len(core_errs))

    # All-in score (add n_s and delta_cp but labelled as ansatz)
    all_errs = [obs["inv_alpha_0_frac_err"], obs["v_EW_frac_err"],
                obs["n_s_frac_err"], obs["delta_cp_frac_err"]]
    obs["score_all"] = math.sqrt(sum(e ** 2 for e in all_errs) / len(all_errs))

    # Sigma-based core score (more physically meaningful for n_s which has tiny σ)
    core_sigmas = [obs["inv_alpha_0_sigma"], obs["v_EW_sigma"]]
    obs["score_core_sigma"] = math.sqrt(sum(s ** 2 for s in core_sigmas) / len(core_sigmas))

    return obs


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def is_pareto_dominated(candidate: List[float], others: List[List[float]]) -> bool:
    """Return True if candidate is dominated by any point in others."""
    for other in others:
        if all(o <= c for o, c in zip(other, candidate)) and any(o < c for o, c in zip(other, candidate)):
            return True
    return False


def compute_pareto_front(points: List[Dict], objectives: List[str]) -> List[Dict]:
    """Find Pareto-optimal points minimizing all objectives."""
    values = [[pt[obj] for obj in objectives] for pt in points]
    pareto = []
    for i, (pt, val) in enumerate(zip(points, values)):
        others = [v for j, v in enumerate(values) if j != i]
        if not is_pareto_dominated(val, others):
            pareto.append(pt)
    return pareto


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

BASELINE_P = 104729
BASELINE_Z = 6

# Z range
Z_VALUES = list(range(2, 13))  # 2..12

# P ranges for different resolution tiers
def get_prime_sample() -> List[int]:
    """Multi-tier prime sampling: dense near baseline, sparse far away."""
    primes = set()

    # Tier 1: tight window ±500 around baseline (dense)
    for p in primes_in_range(BASELINE_P - 500, BASELINE_P + 500, max_count=200):
        primes.add(p)

    # Tier 2: medium window ±5000 (moderate)
    for p in primes_in_range(BASELINE_P - 5000, BASELINE_P + 5000, max_count=100):
        primes.add(p)

    # Tier 3: wide range 30k–300k (sparse, logarithmic)
    for p in primes_in_range(30000, 300000, max_count=300):
        primes.add(p)

    return sorted(primes)


def run_full_sweep():
    primes = get_prime_sample()
    print(f"[sweep] Sampling {len(primes)} primes × {len(Z_VALUES)} z values = {len(primes)*len(Z_VALUES)} points")

    rows = []
    for p in primes:
        for z in Z_VALUES:
            try:
                obs = compute_all_observables(p, z)
                row = {"p": p, "z": z, **obs}
                rows.append(row)
            except Exception as exc:
                print(f"  ERROR at p={p}, z={z}: {exc}")

    print(f"[sweep] Computed {len(rows)} points")
    return rows


def save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name} ({len(rows)} rows)")


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def find_viable_regions(rows: List[Dict], score_key: str, threshold: float) -> List[Dict]:
    return [r for r in rows if r[score_key] <= threshold]


def compute_z_profiles(rows: List[Dict]) -> List[Dict]:
    """For each z, find the best prime and stats."""
    from collections import defaultdict
    by_z: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_z[r["z"]].append(r)
    profiles = []
    for z in sorted(by_z.keys()):
        pts = by_z[z]
        best = min(pts, key=lambda x: x["score_core"])
        scores = [x["score_core"] for x in pts]
        profiles.append({
            "z": z,
            "best_p": best["p"],
            "best_score_core": best["score_core"],
            "mean_score_core": float(np.mean(scores)),
            "min_score_core": float(np.min(scores)),
            "max_score_core": float(np.max(scores)),
            "n_points": len(pts),
            "n_viable_1pct": sum(1 for s in scores if s <= 0.01),
            "delta_cp_pred": best["delta_cp_pred"],
            "delta_cp_sigma": best["delta_cp_sigma"],
        })
    return profiles


def compute_p_profiles(rows: List[Dict]) -> List[Dict]:
    """For each prime, find best z and stats."""
    from collections import defaultdict
    by_p: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_p[r["p"]].append(r)
    profiles = []
    for p in sorted(by_p.keys()):
        pts = by_p[p]
        best = min(pts, key=lambda x: x["score_core"])
        scores = [x["score_core"] for x in pts]
        ns_val = pts[0]["n_s_pred"]  # n_s doesn't depend on z
        profiles.append({
            "p": p,
            "best_z": best["z"],
            "best_score_core": best["score_core"],
            "mean_score_core": float(np.mean(scores)),
            "min_score_core": float(np.min(scores)),
            "n_s_pred": ns_val,
            "n_s_sigma": pts[0]["n_s_sigma"],
        })
    return profiles


def find_pareto_regions(rows: List[Dict]) -> List[Dict]:
    """Pareto front on (inv_alpha_frac_err, v_EW_frac_err, n_s_frac_err, delta_cp_frac_err)."""
    objectives = ["inv_alpha_0_frac_err", "v_EW_frac_err", "n_s_frac_err", "delta_cp_frac_err"]
    print(f"  Computing Pareto front over {len(rows)} points...")
    pareto = compute_pareto_front(rows, objectives)
    return sorted(pareto, key=lambda x: x["score_core"])


def cluster_solution_families(rows: List[Dict], score_key: str = "score_core",
                               threshold: float = 0.05) -> Dict[str, Any]:
    """Simple clustering: group viable points by z, then by p-range quartile."""
    viable = [r for r in rows if r[score_key] <= threshold]
    if not viable:
        return {"viable_count": 0, "families": []}

    from collections import defaultdict
    by_z: Dict[int, List[int]] = defaultdict(list)
    for r in viable:
        by_z[r["z"]].append(r["p"])

    families = []
    for z in sorted(by_z.keys()):
        ps = sorted(by_z[z])
        if not ps:
            continue
        # Split into contiguous clusters (gap > 1000 = separate cluster)
        clusters: List[List[int]] = [[ps[0]]]
        for p in ps[1:]:
            if p - clusters[-1][-1] <= 2000:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        for cl in clusters:
            families.append({
                "z": z,
                "p_min": min(cl),
                "p_max": max(cl),
                "p_center": int(np.median(cl)),
                "count": len(cl),
                "contains_baseline": BASELINE_P in cl or (min(cl) <= BASELINE_P <= max(cl)),
            })

    return {
        "viable_count": len(viable),
        "threshold": threshold,
        "score_key": score_key,
        "families": families,
    }


def locate_baseline(rows: List[Dict]) -> Dict[str, Any]:
    """Find where (104729, 6) sits in the landscape."""
    baseline = next((r for r in rows if r["p"] == BASELINE_P and r["z"] == BASELINE_Z), None)
    if baseline is None:
        # Find closest
        baseline = min(rows, key=lambda r: abs(r["p"] - BASELINE_P) + abs(r["z"] - BASELINE_Z) * 100)

    all_core_scores = [r["score_core"] for r in rows]
    rank = sorted(all_core_scores).index(baseline["score_core"]) + 1
    pct = rank / len(all_core_scores) * 100

    # Local gradient: how fast does score change moving away from baseline?
    neighbors = [r for r in rows if abs(r["p"] - BASELINE_P) <= 500 and r["z"] == BASELINE_Z]
    if len(neighbors) > 2:
        ps = [r["p"] for r in neighbors]
        ss = [r["score_core"] for r in neighbors]
        grad = (max(ss) - min(ss)) / (max(ps) - min(ps)) if max(ps) > min(ps) else 0.0
    else:
        grad = 0.0

    return {
        "p": baseline["p"],
        "z": baseline["z"],
        "score_core": baseline["score_core"],
        "score_all": baseline["score_all"],
        "rank_core": rank,
        "total_points": len(all_core_scores),
        "percentile_core": round(pct, 1),
        "inv_alpha_0_pred": baseline["inv_alpha_0_pred"],
        "inv_alpha_0_frac_err": baseline["inv_alpha_0_frac_err"],
        "v_EW_pred": baseline["v_EW_pred"],
        "v_EW_frac_err": baseline["v_EW_frac_err"],
        "n_s_pred": baseline["n_s_pred"],
        "n_s_sigma": baseline["n_s_sigma"],
        "delta_cp_pred": baseline["delta_cp_pred"],
        "delta_cp_sigma": baseline["delta_cp_sigma"],
        "local_gradient_score_per_p": round(grad, 8),
    }


def find_best_alternatives(rows: List[Dict], n: int = 20) -> List[Dict]:
    """Best non-baseline (p, z) combinations by core score."""
    others = [r for r in rows if not (r["p"] == BASELINE_P and r["z"] == BASELINE_Z)]
    return sorted(others, key=lambda x: x["score_core"])[:n]


def analyze_observable_constraints(rows: List[Dict]) -> Dict[str, Any]:
    """Which observable creates the strongest constraint on p? On z?"""
    # Group by z=6 to see p-sensitivity
    z6_rows = [r for r in rows if r["z"] == 6]
    z6_rows_sorted = sorted(z6_rows, key=lambda x: x["p"])

    # Group by p~baseline to see z-sensitivity
    near_p_rows = [r for r in rows if abs(r["p"] - BASELINE_P) <= 200]

    result = {}

    for obs in ["inv_alpha_0_frac_err", "v_EW_frac_err", "n_s_frac_err", "delta_cp_frac_err"]:
        if z6_rows_sorted:
            vals = [r[obs] for r in z6_rows_sorted]
            result[f"{obs}_p_range"] = round(max(vals) - min(vals), 6)
            result[f"{obs}_p_slope"] = round(
                (vals[-1] - vals[0]) / (z6_rows_sorted[-1]["p"] - z6_rows_sorted[0]["p"] + 1e-9), 9
            )
        if near_p_rows:
            vals_z = sorted([(r["z"], r[obs]) for r in near_p_rows])
            if vals_z:
                obs_vals = [v[1] for v in vals_z]
                result[f"{obs}_z_range"] = round(max(obs_vals) - min(obs_vals), 6)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("BPR Solution Space Mapper")
    print("=" * 60)

    # 1. Run sweep
    rows = run_full_sweep()
    save_csv(rows, OUT_DIR / "all_points.csv")

    # 2. Z profiles
    z_profiles = compute_z_profiles(rows)
    save_csv(z_profiles, OUT_DIR / "z_profiles.csv")
    print("\nZ profiles:")
    print(f"  {'z':>3}  {'best_p':>8}  {'best_score_core':>16}  {'n_viable_1pct':>13}  {'dcp_sigma':>9}")
    for zp in z_profiles:
        print(f"  {zp['z']:>3}  {zp['best_p']:>8}  {zp['best_score_core']:>16.6f}  {zp['n_viable_1pct']:>13}  {zp['delta_cp_sigma']:>9.2f}")

    # 3. P profiles (subset near baseline for display)
    p_profiles = compute_p_profiles(rows)
    save_csv(p_profiles, OUT_DIR / "p_profiles.csv")

    # 4. Pareto front
    pareto = find_pareto_regions(rows)
    save_csv(pareto, OUT_DIR / "pareto_front.csv")
    print(f"\nPareto front: {len(pareto)} points (4 objectives)")

    # 5. Solution families (5% threshold)
    clusters_5pct = cluster_solution_families(rows, "score_core", 0.05)
    clusters_1pct = cluster_solution_families(rows, "score_core", 0.01)
    save_json({
        "threshold_5pct": clusters_5pct,
        "threshold_1pct": clusters_1pct,
    }, OUT_DIR / "clusters.json")

    # 6. Locate baseline
    baseline_loc = locate_baseline(rows)
    print(f"\nBaseline (p={BASELINE_P}, z={BASELINE_Z}):")
    for k, v in baseline_loc.items():
        print(f"  {k}: {v}")

    # 7. Best alternatives
    alternatives = find_best_alternatives(rows, n=20)
    save_csv(alternatives, OUT_DIR / "best_alternatives.csv")

    # 8. Observable constraint analysis
    constraints = analyze_observable_constraints(rows)

    # 9. Comprehensive summary
    summary = {
        "sweep_stats": {
            "total_points": len(rows),
            "n_primes": len(set(r["p"] for r in rows)),
            "z_values": sorted(set(r["z"] for r in rows)),
            "p_min": min(r["p"] for r in rows),
            "p_max": max(r["p"] for r in rows),
        },
        "baseline": baseline_loc,
        "best_core_point": min(rows, key=lambda x: x["score_core"]),
        "best_all_point": min(rows, key=lambda x: x["score_all"]),
        "viable_regions": {
            "5pct_core": clusters_5pct,
            "1pct_core": clusters_1pct,
        },
        "pareto_count": len(pareto),
        "observable_constraints": constraints,
        "top_10_alternatives": alternatives[:10],
        "z_profiles": z_profiles,
        "provenance": PROVENANCE,
    }
    save_json(summary, OUT_DIR / "landscape_summary.json")

    # 10. Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    best = min(rows, key=lambda x: x["score_core"])
    print(f"\nBest core score: p={best['p']}, z={best['z']}, score={best['score_core']:.6f}")
    print(f"Baseline core score: {baseline_loc['score_core']:.6f}")
    print(f"Baseline rank: {baseline_loc['rank_core']} / {baseline_loc['total_points']} "
          f"({baseline_loc['percentile_core']}th percentile)")

    print(f"\nViable regions (≤5% core error): {clusters_5pct['viable_count']} points")
    for fam in clusters_5pct["families"]:
        flag = " ← BASELINE" if fam["contains_baseline"] else ""
        print(f"  z={fam['z']}: p=[{fam['p_min']}, {fam['p_max']}] ({fam['count']} pts){flag}")

    print(f"\nViable regions (≤1% core error): {clusters_1pct['viable_count']} points")
    for fam in clusters_1pct["families"]:
        flag = " ← BASELINE" if fam["contains_baseline"] else ""
        print(f"  z={fam['z']}: p=[{fam['p_min']}, {fam['p_max']}] ({fam['count']} pts){flag}")

    print(f"\nPareto front: {len(pareto)} points")
    in_pareto = any(r["p"] == BASELINE_P and r["z"] == BASELINE_Z for r in pareto)
    print(f"  Baseline in Pareto front: {in_pareto}")

    return rows, summary


if __name__ == "__main__":
    rows, summary = main()
    print("\nDone. Results in analysis/results/solution_space/")
