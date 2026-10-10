"""Tests for scripts/p_calibration_check.py (legacy p calibration diagnostic).

These lock the structural findings of the 2026-10 review of unmerged
branches (parameter audit on claude/audit-bpr-parameters-6fSrQ), recomputed
on the current tree: p is an alpha calibration, many primes meet any of the
tested tolerances, alpha pins p only to within the prime spacing, the v_EW
root moves with the Lambda_QCD input, and the z preference is not robust.
They check orderings and wide margins rather than exact floating counts.
"""
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bpr.alpha_derivation import inverse_alpha_from_substrate
from bpr.gauge_unification import electroweak_scale_GeV

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p_calibration_check.py"
P = 104761


def _load():
    spec = importlib.util.spec_from_file_location("p_calibration_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def result(mod):
    return mod.run()


def _row(result, tol):
    (row,) = [r for r in result["counts"]["rows"] if r["tolerance"] == tol]
    return row


def test_docstring_states_diagnostic_status(mod):
    doc = mod.__doc__
    assert "not a derivation" in doc
    assert "α calibration" in doc
    assert "legacy-phenomenology diagnostic" in doc.lower()
    assert "claude/audit-bpr-parameters-6fSrQ" in doc
    assert "openai" not in doc.lower()


def test_uses_current_tree_functions(mod):
    e_alpha, e_v = mod.errors(P)
    assert float(e_alpha) == pytest.approx(
        (inverse_alpha_from_substrate(P, 6) - mod.INV_ALPHA_DATA) / mod.INV_ALPHA_DATA, rel=1e-12)
    assert float(e_v) == pytest.approx(
        (electroweak_scale_GeV(P, 6) - mod.V_EW_DATA_GEV) / mod.V_EW_DATA_GEV, rel=1e-12)
    # Lambda_QCD central value follows the tree's default rather than a copy.
    assert mod.LAMBDA_QCD_DEFAULT_GEV == pytest.approx(electroweak_scale_GeV(P, 6) / electroweak_scale_GeV(P, 6, 1.0))


def test_primes_sieve(mod):
    sympy = pytest.importorskip("sympy")
    assert [int(p) for p in mod.primes_in(104000, 106000)] == list(sympy.primerange(104000, 106001))
    assert len(mod.primes_in(*mod.PRIME_RANGE)) == len(list(sympy.primerange(*mod.PRIME_RANGE)))


def test_counts_complete_and_monotone_in_tolerance(result):
    assert result["counts_complete"]
    n = [_row(result, t)["n_rms"] for t in (0.005, 0.01, 0.02)]
    assert n[0] < n[1] < n[2]
    # The count roughly doubles each time the tolerance doubles: it tracks
    # the tolerance, not the data.
    assert n[1] > 1.5 * n[0] and n[2] > 1.5 * n[1]


def test_p_not_uniquely_selected_at_one_percent(result):
    row = _row(result, 0.01)
    assert row["n_rms"] > 100
    assert row["n_both"] > 100
    assert row["p_ref_within_rms"]
    counts = result["counts"]
    assert counts["best_prime"] != P
    assert counts["p_ref_rank"] > 100


def test_count_equals_primes_between_continuous_edges(mod, result):
    for tol in (0.005, 0.01, 0.02):
        row = _row(result, tol)
        lo, hi = row["edges"]["lo"], row["edges"]["hi"]
        between = mod.primes_in(math.ceil(lo), math.floor(hi))
        assert row["n_rms"] == len(between)


def test_score_floor_from_alpha_vew_tension(result):
    counts = result["counts"]
    # No prime reaches 0.3%: the two equations prefer different p.
    assert 0.003 < counts["best_score"] < 0.004
    # v_EW, not alpha, sets the lower edge of the 1% corridor.
    edges = _row(result, 0.01)["edges"]
    assert abs(edges["lo_e_v"]) > 5 * abs(edges["lo_e_alpha"])


def test_p_ref_misses_vew_by_about_one_percent(result):
    counts = result["counts"]
    assert -0.012 < counts["p_ref_e_v"] < -0.010
    assert not _row(result, 0.01)["p_ref_within_both"]
    assert _row(result, 0.02)["p_ref_within_both"]


def test_alpha_root_within_prime_gap_of_p(result):
    pin = result["alpha_pinning"]
    root = pin["root"]
    assert 104749 < root < 104750
    assert abs(P - root) < 2 * math.log(root)
    assert pin["bracket"] == [104743, 104759]
    assert pin["nearest_prime"] == 104743
    assert pin["nearest_prime_1mod4"] == P
    assert 15 < pin["p_ref_ppm"] < 25
    assert abs(pin["best_ppm"]) < pin["p_ref_ppm"]


def test_other_primes_fit_alpha_as_well(mod, result):
    pin = result["alpha_pinning"]
    assert {104743, 104759, P} <= set(pin["primes_at_least_as_good_as_ref"])
    # alpha alone admits many primes even at a 0.1% tolerance
    primes = mod.primes_in(*mod.PRIME_RANGE)
    e_alpha, _ = mod.errors(primes)
    assert int((abs(e_alpha) <= 0.001).sum()) > 50
    # and against the CODATA error, every prime misses by > 1e4 sigma
    assert pin["min_sigma_miss"] > 1e4


def test_roots_differ(result):
    roots = result["roots"]
    assert 107700 < roots["v_EW"] < 107720
    assert roots["v_EW"] / roots["alpha"] - 1 > 0.02


def test_vew_root_moves_with_lambda_qcd(result):
    ls = result["lambda_sensitivity"]
    lo, mid, hi = (r["vew_root"] for r in ls["rows"])
    assert lo > mid > hi
    assert hi < P < lo
    assert lo - hi > 20000
    # The Lambda_QCD that makes v_EW exact at p = 104761 lies inside the band.
    lam = ls["lambda_for_exact_v_at_p_ref"]
    assert ls["rows"][0]["lambda_qcd_GeV"] < lam < ls["rows"][2]["lambda_qcd_GeV"]


def test_joint_root_unique_and_not_integer(result):
    joint = result["joint"]
    assert all(j["well_defined"] and j["n_roots"] == 1 for j in joint)
    z_lo_lam, z_mid, z_hi_lam = (j["z"] for j in joint)
    assert round(z_mid) == 6 and abs(z_mid - 6) > 0.1
    assert round(z_lo_lam) == 7 and round(z_hi_lam) == 5


def test_integer_z_preference_moves_with_lambda(result):
    best = [block["best_z"] for block in result["z_scan"]]
    assert best == [7, 6, 5]
    for block in result["z_scan"]:
        assert not any(row["at_scan_edge"] for row in block["rows"])
    central = {row["z"]: row for row in result["z_scan"][1]["rows"]}
    assert central[6]["n_within_tol"] > 100
    assert central[5]["n_within_tol"] == 0 and central[7]["n_within_tol"] == 0


def test_flavor_outputs_track_vew(result):
    for row in result["flavor"]["rows"]:
        assert row["m_t_GeV"] == pytest.approx(row["v_EW_GeV"] / math.sqrt(2.0), rel=1e-12)
    (ref,) = [r for r in result["flavor"]["rows"] if r["p"] == P]
    assert 171.5 < ref["m_t_GeV"] < 173.0


def test_cli_json_and_runtime(result):
    start = time.time()
    proc = subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True,
                          text=True, timeout=120, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert time.time() - start < 60
    data = json.loads(proc.stdout)
    assert "not a derivation" in data["disclaimer"]
    assert data["counts"]["rows"][1]["n_rms"] == result["counts"]["rows"][1]["n_rms"]


def test_cli_text_report():
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                          text=True, timeout=120, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "not a derivation" in proc.stdout
    assert "counts are complete" in proc.stdout
