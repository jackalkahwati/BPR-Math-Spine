#!/usr/bin/env python3
"""Calibration check for the legacy substrate prime p. A diagnostic, not a derivation.

What this is
------------
A legacy-phenomenology diagnostic. p = 104,761 is an α calibration: it is
the nearest prime ≡ 1 (mod 4) to the real root of the legacy 1/α formula.
This script measures how much the two legacy p-dependent outputs that are
compared with data actually constrain p (and z). It derives nothing, and a
good match here is not evidence for the framework. The claims it checks
("p is over-determined", "p is uniquely selected", "z = 6 is derived") are
withdrawn; see the banner of doc/derivations/p_selection_principle.md and
doc/CLOSED_AND_DEPRECATED.md.

Model outputs (current tree, called directly)
---------------------------------------------
    1/α(p, z)  = [ln p]^2 + z/2 + γ − 1/(2π)
                 bpr.alpha_derivation.inverse_alpha_from_substrate   (eq. A)
    v_EW(p, z) = Λ_QCD p^(1/3) (ln p + z − 2)
                 bpr.gauge_unification.electroweak_scale_GeV         (eq. C)

Λ_QCD is an input of v_EW. Its central value is read from the default of
electroweak_scale_GeV (0.332 GeV). The ±0.017 GeV band is illustrative:
the 2026-10 review took it from a lattice determination found by search,
not from the PDG table. Override both with --lambda-qcd and --lambda-unc.

Data: 1/α = 137.035999084(21) (CODATA 2018) and v_EW = 246.22 GeV
(bpr.constants.V_HIGGS).

Score
-----
Each prime gets the unweighted RMS of the two fractional errors,
    score(p) = sqrt((e_α^2 + e_v^2) / 2),   e = (model − data) / data,
the core score of the legacy audit. A prime "falls within X" when
score ≤ X, for X = 0.5%, 1% and 2%. The stricter count, both |e_α| ≤ X
and |e_v| ≤ X, is reported alongside. Unweighted scoring ignores the
measurement errors. With them, every prime misses 1/α by tens of
thousands of σ (reported below), so the tolerance, not the data, sets the
counts.

Related legacy flavor outputs (m_t, m_b from bpr.qcd_flavor.QuarkMassSpectrum
driven by the framework's own v_EW(p)) are printed for reference primes.
They scale with v_EW (m_t = v_EW/√2), carry no independent information on
p, and are not scored. The ansatz observables n_s and δ_CP are not scored
either.

What it reports
---------------
1. Tolerance counts at z, Λ_QCD central, and the continuous corridor edges.
2. Continuous roots of the α and v_EW equations, and how tightly α pins p.
3. Sensitivity of the v_EW root to Λ_QCD ± its uncertainty.
4. The continuous joint root (p*, z*), with z real, if it is unique.
5. A full prime scan for every integer z in 2..12 at three Λ_QCD values.

Ranges and runtime: every prime in [20,000, 400,000] (31,598 primes) is
scanned. The script checks that each tolerance corridor lies strictly
inside that range, so the counts at the default settings are complete.
The joint root is searched for z in [0.5, 40]. Runtime is a few seconds.

Provenance: the findings re-checked here come from the 2026-10 review of
unmerged branches (the parameter audit on claude/audit-bpr-parameters-6fSrQ).
This is a fresh implementation against the current tree, not a port of
that branch's analysis scripts, whose prime sampler stopped at 109,721.

Usage:
    python3 scripts/p_calibration_check.py [--json] [--z 6]
        [--lambda-qcd 0.332] [--lambda-unc 0.017]
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bpr.alpha_derivation import inverse_alpha_from_substrate  # noqa: E402
from bpr.gauge_unification import electroweak_scale_GeV  # noqa: E402
from bpr.qcd_flavor import QuarkMassSpectrum  # noqa: E402

# ── Data (comparison targets, not inputs to the model outputs) ──────────────
INV_ALPHA_DATA = 137.035999084        # CODATA 2018, q^2 = 0
INV_ALPHA_SIGMA = 0.000000021         # CODATA 2018 standard uncertainty
V_EW_DATA_GEV = 246.22                # bpr.constants.V_HIGGS

# ── Settings ───────────────────────────────────────────────────────────────
P_CALIBRATED = 104761
Z_DEFAULT = 6
LAMBDA_QCD_DEFAULT_GEV = float(
    inspect.signature(electroweak_scale_GeV).parameters["Lambda_QCD_GeV"].default
)
LAMBDA_QCD_UNC_GEV = 0.017            # illustrative; see module docstring
TOLERANCES = (0.005, 0.01, 0.02)
PRIME_RANGE = (20_000, 400_000)
Z_SCAN = tuple(range(2, 13))
JOINT_Z_RANGE = (0.5, 40.0)

DISCLAIMER = ("Legacy-phenomenology diagnostic. p is an alpha calibration; "
              "this is not a derivation and not a test of the framework.")


# ── Helpers ────────────────────────────────────────────────────────────────

def primes_in(lo: int, hi: int) -> np.ndarray:
    """All primes in [lo, hi] (sieve of Eratosthenes)."""
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, math.isqrt(hi) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    found = np.nonzero(sieve)[0]
    return found[found >= lo]


def _bisect(f, a: float, b: float, rel_tol: float = 1e-14, max_iter: int = 300) -> float:
    """Root of f on [a, b]; f(a) and f(b) must differ in sign."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError(f"no sign change on [{a}, {b}]")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if b - a <= rel_tol * max(1.0, abs(m)):
            break
    return 0.5 * (a + b)


def errors(p, z=Z_DEFAULT, lam=LAMBDA_QCD_DEFAULT_GEV):
    """Fractional errors (e_alpha, e_v) of the two legacy outputs."""
    p = np.asarray(p, dtype=float)
    e_alpha = (inverse_alpha_from_substrate(p, z) - INV_ALPHA_DATA) / INV_ALPHA_DATA
    e_v = (electroweak_scale_GeV(p, z, lam) - V_EW_DATA_GEV) / V_EW_DATA_GEV
    return e_alpha, e_v


def score(p, z=Z_DEFAULT, lam=LAMBDA_QCD_DEFAULT_GEV):
    """Unweighted RMS of the fractional errors in 1/α and v_EW."""
    e_alpha, e_v = errors(p, z, lam)
    return np.sqrt((e_alpha ** 2 + e_v ** 2) / 2.0)


def alpha_root(z: float = Z_DEFAULT) -> float:
    """Real p with inverse_alpha_from_substrate(p, z) = 1/α (data)."""
    f = lambda x: float(inverse_alpha_from_substrate(math.exp(x), z)) - INV_ALPHA_DATA
    return math.exp(_bisect(f, 1.0, 30.0))


def vew_root(z: float = Z_DEFAULT, lam: float = LAMBDA_QCD_DEFAULT_GEV) -> float:
    """Real p with electroweak_scale_GeV(p, z, lam) = v_EW (data)."""
    f = lambda x: float(electroweak_scale_GeV(math.exp(x), z, lam)) - V_EW_DATA_GEV
    return math.exp(_bisect(f, 1.0, 30.0))


# ── Sections ───────────────────────────────────────────────────────────────

def tolerance_counts(primes, z, lam, tolerances=TOLERANCES, p_ref=P_CALIBRATED):
    """Primes within each tolerance, by RMS score and by both errors separately."""
    e_alpha, e_v = errors(primes, z, lam)
    s = np.sqrt((e_alpha ** 2 + e_v ** 2) / 2.0)
    order = np.argsort(s, kind="stable")
    best = int(order[0])
    ref_matches = np.nonzero(primes == p_ref)[0]
    ref_rank = int(np.nonzero(order == ref_matches[0])[0][0]) + 1 if ref_matches.size else None
    ea_ref, ev_ref = (float(x) for x in errors(p_ref, z, lam))
    s_ref = math.sqrt((ea_ref ** 2 + ev_ref ** 2) / 2.0)
    rows = []
    for tol in tolerances:
        in_rms = s <= tol
        in_both = (np.abs(e_alpha) <= tol) & (np.abs(e_v) <= tol)
        rows.append({
            "tolerance": tol,
            "n_rms": int(in_rms.sum()),
            "rms_range": [int(primes[in_rms].min()), int(primes[in_rms].max())] if in_rms.any() else None,
            "n_both": int(in_both.sum()),
            "both_range": [int(primes[in_both].min()), int(primes[in_both].max())] if in_both.any() else None,
            "p_ref_within_rms": bool(s_ref <= tol),
            "p_ref_within_both": bool(abs(ea_ref) <= tol and abs(ev_ref) <= tol),
            "edges": corridor_edges(tol, z, lam, float(primes[best])),
        })
    return {
        "z": z,
        "lambda_qcd_GeV": lam,
        "n_primes_scanned": int(primes.size),
        "prime_range": [int(primes.min()), int(primes.max())],
        "best_prime": int(primes[best]),
        "best_score": float(s[best]),
        "best_e_alpha": float(e_alpha[best]),
        "best_e_v": float(e_v[best]),
        "p_ref": p_ref,
        "p_ref_score": s_ref,
        "p_ref_rank": ref_rank,
        "p_ref_e_alpha": ea_ref,
        "p_ref_e_v": ev_ref,
        "rows": rows,
    }


def corridor_edges(tol, z, lam, split):
    """Continuous p where the score crosses tol, on either side of split."""
    f = lambda x: float(score(x, z, lam)) - tol
    if f(split) >= 0:
        return None
    lo = _bisect(f, split / 4.0, split)
    hi = _bisect(f, split, split * 4.0)
    out = {"lo": lo, "hi": hi}
    for name, x in (("lo", lo), ("hi", hi)):
        ea, ev = (float(e) for e in errors(x, z, lam))
        out[f"{name}_e_alpha"] = ea
        out[f"{name}_e_v"] = ev
    return out


def alpha_pinning(primes, z, p_ref=P_CALIBRATED):
    """How tightly the α equation alone pins p."""
    root = alpha_root(z)
    below = primes[primes <= root]
    above = primes[primes > root]
    e_alpha = (inverse_alpha_from_substrate(primes.astype(float), z) - INV_ALPHA_DATA) / INV_ALPHA_DATA
    abs_e = np.abs(e_alpha)
    e_ref = (float(inverse_alpha_from_substrate(p_ref, z)) - INV_ALPHA_DATA) / INV_ALPHA_DATA
    best = int(np.argmin(abs_e))
    one_mod_4 = primes[primes % 4 == 1]
    nearest_1mod4 = int(one_mod_4[np.argmin(np.abs(one_mod_4 - root))])
    as_good = sorted(int(p) for p in primes[abs_e <= abs(e_ref)])
    return {
        "root": root,
        "bracket": [int(below.max()), int(above.min())],
        "bracket_gap": int(above.min() - below.max()),
        "mean_prime_gap": math.log(root),
        "nearest_prime": int(primes[np.argmin(np.abs(primes - root))]),
        "nearest_prime_1mod4": nearest_1mod4,
        "p_ref": p_ref,
        "p_ref_distance": p_ref - root,
        "p_ref_inv_alpha": float(inverse_alpha_from_substrate(p_ref, z)),
        "p_ref_ppm": e_ref * 1e6,
        "best_prime": int(primes[best]),
        "best_ppm": float(e_alpha[best]) * 1e6,
        "primes_at_least_as_good_as_ref": as_good,
        "min_sigma_miss": float(abs_e.min()) * INV_ALPHA_DATA / INV_ALPHA_SIGMA,
    }


def lambda_sensitivity(z, lam, unc, p_ref=P_CALIBRATED):
    """v_EW root and the v_EW miss at p_ref across Λ_QCD ± unc."""
    rows = []
    for value in (lam - unc, lam, lam + unc):
        v_ref = float(electroweak_scale_GeV(p_ref, z, value))
        rows.append({
            "lambda_qcd_GeV": value,
            "vew_root": vew_root(z, value),
            "p_ref_v_EW_GeV": v_ref,
            "p_ref_e_v": (v_ref - V_EW_DATA_GEV) / V_EW_DATA_GEV,
        })
    per_unit = float(electroweak_scale_GeV(p_ref, z, 1.0))
    return {
        "z": z,
        "lambda_unc_GeV": unc,
        "rows": rows,
        "lambda_for_exact_v_at_p_ref": V_EW_DATA_GEV / per_unit,
    }


def joint_root(lam, z_range=JOINT_Z_RANGE, n_grid=400):
    """Real (p*, z*) solving both equations; None unless the root is unique.

    For each real z the α equation fixes p_α(z); the joint root is where
    v_EW(p_α(z), z) equals the data. Sign changes are counted on a grid.
    """
    f = lambda z: float(electroweak_scale_GeV(alpha_root(z), z, lam)) - V_EW_DATA_GEV
    grid = np.linspace(z_range[0], z_range[1], n_grid)
    values = [f(z) for z in grid]
    brackets = [(grid[i], grid[i + 1]) for i in range(n_grid - 1)
                if values[i] * values[i + 1] < 0]
    out = {"lambda_qcd_GeV": lam, "z_range": list(z_range), "n_roots": len(brackets)}
    if len(brackets) != 1:
        out.update(p=None, z=None, well_defined=False)
        return out
    z_star = _bisect(f, float(brackets[0][0]), float(brackets[0][1]))
    out.update(p=alpha_root(z_star), z=z_star, well_defined=True)
    return out


def z_scan(primes, lams, z_values=Z_SCAN, tol=0.01):
    """Best prime and count within tol for every integer z and each Λ_QCD."""
    lo, hi = int(primes.min()), int(primes.max())
    table = []
    for lam in lams:
        rows = []
        for z in z_values:
            s = score(primes, z, lam)
            i = int(np.argmin(s))
            rows.append({
                "z": z,
                "best_prime": int(primes[i]),
                "best_score": float(s[i]),
                "n_within_tol": int((s <= tol).sum()),
                "at_scan_edge": int(primes[i]) in (lo, hi),
            })
        best = min(rows, key=lambda r: r["best_score"])
        table.append({"lambda_qcd_GeV": lam, "tolerance": tol, "rows": rows, "best_z": best["z"]})
    return table


def flavor_outputs(ps, z, lam):
    """Legacy m_t, m_b at the framework's own v_EW(p). Reported, not scored."""
    ref = QuarkMassSpectrum()
    m_t_data, m_b_data = ref.anchor_mass_up_MeV, ref.anchor_mass_down_MeV
    rows = []
    for p in ps:
        v = float(electroweak_scale_GeV(p, z, lam))
        masses = QuarkMassSpectrum(v_EW_GeV=v, p=int(p), z=float(z)).all_masses_MeV
        rows.append({
            "p": int(p),
            "v_EW_GeV": v,
            "m_t_GeV": masses["t"] / 1000.0,
            "m_t_e": (masses["t"] - m_t_data) / m_t_data,
            "m_b_GeV": masses["b"] / 1000.0,
            "m_b_e": (masses["b"] - m_b_data) / m_b_data,
        })
    return {"m_t_data_GeV": m_t_data / 1000.0, "m_b_data_GeV": m_b_data / 1000.0, "rows": rows}


def run(z=Z_DEFAULT, lam=LAMBDA_QCD_DEFAULT_GEV, unc=LAMBDA_QCD_UNC_GEV,
        prime_range=PRIME_RANGE):
    """Compute every section; returns a JSON-serialisable dict."""
    primes = primes_in(*prime_range)
    counts = tolerance_counts(primes, z, lam)
    pin = alpha_pinning(primes, z)
    lam_values = (lam - unc, lam, lam + unc)
    # A corridor that exists must close inside the scanned range for its
    # count to be complete; no corridor (best score above tol) is complete.
    complete = all(r["edges"] is None or (prime_range[0] < r["edges"]["lo"]
                   and r["edges"]["hi"] < prime_range[1]) for r in counts["rows"])
    reference = sorted({P_CALIBRATED, pin["best_prime"], counts["best_prime"]})
    return {
        "disclaimer": DISCLAIMER,
        "data": {"inv_alpha": INV_ALPHA_DATA, "inv_alpha_sigma": INV_ALPHA_SIGMA,
                 "v_EW_GeV": V_EW_DATA_GEV},
        "score": "sqrt((e_alpha^2 + e_v^2)/2), unweighted fractional errors",
        "counts": counts,
        "counts_complete": complete,
        "roots": {"alpha": pin["root"], "v_EW": vew_root(z, lam)},
        "alpha_pinning": pin,
        "lambda_sensitivity": lambda_sensitivity(z, lam, unc),
        "joint": [joint_root(value) for value in lam_values],
        "z_scan": z_scan(primes, lam_values),
        "flavor": flavor_outputs(reference, z, lam),
    }


# ── Report ─────────────────────────────────────────────────────────────────

def format_report(r) -> str:
    c, pin, ls = r["counts"], r["alpha_pinning"], r["lambda_sensitivity"]
    out = [
        "p calibration check (legacy phenomenology)",
        r["disclaimer"],
        "",
        f"Observables: 1/alpha (data {r['data']['inv_alpha']}) and v_EW "
        f"(data {r['data']['v_EW_GeV']} GeV); z = {c['z']}, "
        f"Lambda_QCD = {c['lambda_qcd_GeV']:.3f} GeV.",
        f"Score: {r['score']}.",
        f"Primes scanned: {c['n_primes_scanned']} in [{c['prime_range'][0]}, {c['prime_range'][1]}].",
        "",
        "1. Tolerance counts",
        "   tol     RMS<=tol  range              both<=tol  continuous RMS corridor      "
        f"p={c['p_ref']} (RMS / both)",
    ]
    for row in c["rows"]:
        rng = "{}-{}".format(*row["rms_range"]) if row["rms_range"] else "-"
        edges = row["edges"]
        corridor = f"{edges['lo']:.2f} - {edges['hi']:.2f}" if edges else "-"
        out.append(
            f"   {row['tolerance'] * 100:4.1f}%  {row['n_rms']:8d}  {rng:<17}  {row['n_both']:9d}  "
            f"{corridor:<27}  {'yes' if row['p_ref_within_rms'] else 'no'} / "
            f"{'yes' if row['p_ref_within_both'] else 'no'}")
    out += [
        f"   Best prime {c['best_prime']}: score {c['best_score'] * 100:.4f}% "
        f"(1/alpha {c['best_e_alpha'] * 1e6:+.1f} ppm, v_EW {c['best_e_v'] * 100:+.3f}%). "
        "No prime scores below this floor.",
        f"   p={c['p_ref']}: score {c['p_ref_score'] * 100:.4f}%, rank {c['p_ref_rank']} of "
        f"{c['n_primes_scanned']} (1/alpha {c['p_ref_e_alpha'] * 1e6:+.1f} ppm, "
        f"v_EW {c['p_ref_e_v'] * 100:+.3f}%).",
    ]
    for row in c["rows"]:
        edges = row["edges"]
        if edges is None:
            continue
        sides = []
        for side in ("lo", "hi"):
            ea, ev = edges[f"{side}_e_alpha"], edges[f"{side}_e_v"]
            sides.append(f"{'lower' if side == 'lo' else 'upper'} edge 1/alpha {ea * 100:+.3f}%, "
                         f"v_EW {ev * 100:+.3f}% ({'v_EW' if abs(ev) > abs(ea) else '1/alpha'} dominates)")
        out.append(f"   {row['tolerance'] * 100:.1f}% corridor: " + "; ".join(sides) + ".")
    out.append("   Corridors lie inside the scanned range: "
               + ("yes, counts are complete." if r["counts_complete"] else "NO, counts are truncated."))
    out += [
        "",
        "2. Continuous roots",
        f"   alpha root p = {r['roots']['alpha']:.2f}; v_EW root p = {r['roots']['v_EW']:.2f} "
        f"({(r['roots']['v_EW'] / r['roots']['alpha'] - 1) * 100:+.2f}% apart).",
        f"   The alpha root lies between consecutive primes {pin['bracket'][0]} and {pin['bracket'][1]} "
        f"(gap {pin['bracket_gap']}; mean gap ln p = {pin['mean_prime_gap']:.2f}).",
        f"   Nearest prime {pin['nearest_prime']}; nearest prime with p mod 4 = 1: {pin['nearest_prime_1mod4']}.",
        f"   p={pin['p_ref']}: {pin['p_ref_distance']:+.2f} from the root, 1/alpha = "
        f"{pin['p_ref_inv_alpha']:.6f} ({pin['p_ref_ppm']:+.1f} ppm).",
        f"   Best prime for alpha alone: {pin['best_prime']} ({pin['best_ppm']:+.1f} ppm). Primes fitting "
        f"alpha at least as well as {pin['p_ref']}: {pin['primes_at_least_as_good_as_ref']}.",
        f"   Against the CODATA uncertainty every prime misses 1/alpha by at least "
        f"{pin['min_sigma_miss']:.2e} sigma: alpha pins p only to within the prime spacing.",
        "",
        f"3. Lambda_QCD sensitivity of the v_EW root (illustrative +-{ls['lambda_unc_GeV'] * 1000:.0f} MeV)",
    ]
    for row in ls["rows"]:
        out.append(
            f"   Lambda_QCD = {row['lambda_qcd_GeV']:.3f} GeV: v_EW root p = {row['vew_root']:.0f}; "
            f"v_EW({c['p_ref']}) = {row['p_ref_v_EW_GeV']:.2f} GeV ({row['p_ref_e_v'] * 100:+.2f}%)")
    out.append(f"   Lambda_QCD giving v_EW = data exactly at p={c['p_ref']}: "
               f"{ls['lambda_for_exact_v_at_p_ref']:.4f} GeV.")
    out += ["", "4. Continuous joint root (p*, z*), z real"]
    for j in r["joint"]:
        if j["well_defined"]:
            out.append(f"   Lambda_QCD = {j['lambda_qcd_GeV']:.3f} GeV: z* = {j['z']:.3f}, p* = {j['p']:.0f} "
                       f"(unique for z in [{j['z_range'][0]}, {j['z_range'][1]}])")
        else:
            out.append(f"   Lambda_QCD = {j['lambda_qcd_GeV']:.3f} GeV: not well defined "
                       f"({j['n_roots']} roots for z in {j['z_range']})")
    out += ["", "5. Full prime scan by integer z (best RMS score; count within 1%)"]
    for block in r["z_scan"]:
        cells = ", ".join(f"z={row['z']}: {row['best_score'] * 100:.2f}% ({row['n_within_tol']})"
                          + (" [edge]" if row["at_scan_edge"] else "") for row in block["rows"])
        out.append(f"   Lambda_QCD = {block['lambda_qcd_GeV']:.3f} GeV, best z = {block['best_z']}: {cells}")
    fl = r["flavor"]
    out += ["", f"6. Related legacy flavor outputs at z = {c['z']} (QuarkMassSpectrum with the framework's "
                f"v_EW(p); not scored, they scale with v_EW)"]
    for row in fl["rows"]:
        out.append(f"   p={row['p']}: v_EW = {row['v_EW_GeV']:.2f} GeV, m_t = {row['m_t_GeV']:.2f} GeV "
                   f"({row['m_t_e'] * 100:+.2f}% vs {fl['m_t_data_GeV']:.2f}), m_b = {row['m_b_GeV']:.3f} GeV "
                   f"({row['m_b_e'] * 100:+.2f}% vs {fl['m_b_data_GeV']:.2f})")
    n_min = min(row["n_rms"] for row in c["rows"])
    best_z = sorted({block["best_z"] for block in r["z_scan"]})
    out += ["", "Reading:"]
    if n_min > 1:
        out.append(f"   At least {n_min} primes meet each tolerance, so these two outputs do not "
                   "select p; p is fixed by calibrating the alpha formula.")
    else:
        out.append("   At these settings some tolerance holds at most one prime; that reflects the "
                   "tolerance chosen, not a selection of p.")
    if len(best_z) > 1:
        out.append(f"   The best integer z changes across the Lambda_QCD band ({best_z}), "
                   "so the z preference is not robust.")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Legacy-phenomenology calibration check for p (not a derivation).")
    parser.add_argument("--json", action="store_true", help="print strict JSON instead of text")
    parser.add_argument("--z", type=int, default=Z_DEFAULT, help="integer coordination number (default 6)")
    parser.add_argument("--lambda-qcd", type=float, default=LAMBDA_QCD_DEFAULT_GEV,
                        help="central Lambda_QCD in GeV (default: electroweak_scale_GeV's default)")
    parser.add_argument("--lambda-unc", type=float, default=LAMBDA_QCD_UNC_GEV,
                        help="illustrative Lambda_QCD uncertainty in GeV (default 0.017)")
    args = parser.parse_args(argv)
    result = run(z=args.z, lam=args.lambda_qcd, unc=args.lambda_unc)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
