"""Glueball Benchmark v1 — lock the sealed targets, Gate 1 results, and blindness.

The targets are locked HERE and in the protocol doc, never in the solver. One
test greps the solver source to enforce that mechanically: if a later edit
leaks a target number (or the string 'X(2370)') into the solver, it fails.
"""
import inspect

import numpy as np
import pytest

import bpr.glueball_benchmark as gb
from bpr.glueball_benchmark import (
    derrick_analysis,
    site_centered_state,
    bond_centered_state,
    state_energy,
    localization_ratio,
    fluctuation_spectrum,
    gate1_report,
    benchmark_status,
    report,
)

# ---------------------------------------------------------------------------
# Sealed envelope: locked comparison targets (Morningstar & Peardon 1999).
# These constants exist ONLY here and in doc/GLUEBALL_BENCHMARK_V1.md.
# ---------------------------------------------------------------------------
LATTICE_0PP_MEV = 1730.0
LATTICE_2PP_MEV = 2400.0
LATTICE_0MP_MEV = 2590.0
RATIO_2PP = LATTICE_2PP_MEV / LATTICE_0PP_MEV     # 1.387
RATIO_0MP = LATTICE_0MP_MEV / LATTICE_0PP_MEV     # 1.497
# Gate-4 pass bands fixed in the protocol doc:
GATE4_BAND_2PP = (1.25, 1.55)
GATE4_BAND_0MP = (1.35, 1.65)


def test_targets_locked():
    """The sealed ratios and their pass bands are what the protocol doc states."""
    assert RATIO_2PP == pytest.approx(1.387, abs=0.001)
    assert RATIO_0MP == pytest.approx(1.497, abs=0.001)
    assert GATE4_BAND_2PP[0] < RATIO_2PP < GATE4_BAND_2PP[1]
    assert GATE4_BAND_0MP[0] < RATIO_0MP < GATE4_BAND_0MP[1]


def test_solver_is_blind_to_targets():
    """MECHANICAL BLINDNESS GUARD: the solver module source must not contain the
    target masses, ratios, or the experimental candidate's name."""
    src = inspect.getsource(gb)
    for leaked in ("1730", "2400", "2590", "1.387", "1.497", "2370", "2395"):
        assert leaked not in src, f"solver leaked sealed target: {leaked}"


# ---------------------------------------------------------------------------
# Gate 1: discrete localized states of the frozen equations
# ---------------------------------------------------------------------------

def test_derrick_excludes_static_continuum_solitons():
    d = derrick_analysis()
    assert "EXCLUDED" in d["static_continuum_solitons_in_3d"]
    assert any("discreteness" in e for e in d["escapes"])
    assert any("U(1)" in e for e in d["escapes"])
    # line defects must be excluded as particle candidates, not counted as a pass
    assert any("excluded" in e for e in d["escapes"] if "defect" in e)


def test_two_distinct_bound_states_converge():
    s1, s2 = site_centered_state(), bond_centered_state()
    assert s1["residual"] < 1e-10
    assert s2["residual"] < 1e-10
    # distinct states: different energies and different profiles
    e1, e2 = state_energy(s1["phi"]), state_energy(s2["phi"])
    assert abs(e1 - e2) > 1e-3
    assert not np.allclose(np.sort(np.abs(s1["phi"])), np.sort(np.abs(s2["phi"])))


def test_states_are_localized():
    s1 = site_centered_state()
    assert localization_ratio(s1["phi"], distance=20) > 1e3


def test_u1_zero_mode_is_exact():
    """L- phi = 0 (phase zero mode) must hold to numerical precision."""
    s1 = site_centered_state()
    fl = fluctuation_spectrum(s1["phi"])
    assert fl["zero_mode_residual"] < 1e-9


def test_discrete_internal_modes_below_band():
    """The Bogoliubov spectrum has >=1 discrete internal mode below the
    continuum band edge — the discreteness Gate 1 asks for."""
    s1 = site_centered_state()
    fl = fluctuation_spectrum(s1["phi"])
    assert fl["n_internal_modes"] >= 1
    assert all(0 < w < fl["band_edge_omega2"] for w in fl["internal_omega2"])


def test_gate1_passes():
    assert gate1_report()["gate1"] == "PASS"


# ---------------------------------------------------------------------------
# Honesty guards: gates 2-4 sealed, no J^PC claim
# ---------------------------------------------------------------------------

def test_gates_2_to_4_reflect_opened_envelope():
    """Gates 2-4 were computed in bpr/glueball_gates.py; the statuses here must
    reflect the opened envelope's verdicts, not claim success."""
    st = benchmark_status()
    assert "PARTIAL FAIL" in st["gate2_jpc_families"]
    assert st["gate3_ordering"].startswith("FAIL")
    assert st["gate4_mass_ratios"].startswith("FAIL")


def test_no_jpc_or_glueball_claim():
    """The report must state the ring result carries no J^PC content and must
    not claim any glueball identification."""
    txt = report()
    assert "no J^PC content" in txt
    assert "OPENED" in txt
    assert "no glueball identification" in txt
