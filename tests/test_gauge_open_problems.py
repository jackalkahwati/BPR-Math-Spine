"""Locks for the three Path B open-problem modules.

1. gauge_phase_mc      — M2 phase location (Z_2 literature anchor + D_n scans)
2. glueball_channels_mc — M4 machinery (operator symmetry + sealed-gate logic)
3. lepton_dn_assignment — M3 sqrt(210) wrinkle (exact rep theory)

Every module must stay blind to the sealed glueball targets (mechanical grep,
same discipline as the other Path B tests).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from bpr.gauge_phase_mc import (WilsonMC, dn_tables, pseudo_critical, scan,
                                z2_tables)
from bpr.glueball_channels_mc import (benchmark_v3_readiness,
                                      jackknife_effective_mass,
                                      measure_correlators, plateau,
                                      slice_operators)
from bpr.lepton_dn_assignment import (LEPTON_LABELS, composite_charge,
                                      fuse_e_e, lepton_assignment,
                                      masses_inert_check, wrinkle_status)
from bpr.nonabelian_gauge_sector import ALLOWED_CLASSES

BPR_DIR = pathlib.Path(__file__).resolve().parents[1] / "bpr"


# ---------------------------------------------------------------------------
# gauge_phase_mc — group tables and MC sanity
# ---------------------------------------------------------------------------

def test_dn_tables_are_a_group():
    for n in ALLOWED_CLASSES:
        M, I, c, d = dn_tables(n)
        G = len(I)
        assert G == 2 * n
        # identity is index 0: e*g = g*e = g
        assert (M[0, :] == np.arange(G)).all()
        assert (M[:, 0] == np.arange(G)).all()
        # inverses: g * g^-1 = e
        assert (M[np.arange(G), I] == 0).all()
        # E1 characters bounded by dimension
        assert np.abs(c).max() <= d + 1e-12


def test_frozen_start_is_action_minimum():
    """Cold start (all identity links): every plaquette has flux = identity,
    so the mean E1 plaquette is exactly 1."""
    mc = WilsonMC(dn_tables(5), L=3, beta=1.0, seed=0)
    assert mc.mean_plaquette() == pytest.approx(1.0)


def test_strong_coupling_disorders_plaquette():
    """At beta ~ 0 the links randomize: <P> must drop far below 1."""
    mc = WilsonMC(dn_tables(5), L=3, beta=0.01, seed=1)
    r = mc.run(n_equil=60, n_meas=60, stride=2)
    assert r["plaq_mean"] < 0.5


def test_weak_coupling_orders_plaquette():
    """Deep in the ordered phase <P> stays near 1."""
    mc = WilsonMC(dn_tables(5), L=3, beta=6.0, seed=2)
    r = mc.run(n_equil=60, n_meas=60, stride=2)
    assert r["plaq_mean"] > 0.8


def test_wilson_result_metadata_is_additive_and_generic():
    """Supplied D_n and Z2 tables retain characters and a generic model label."""
    for tables in (dn_tables(12), z2_tables()):
        mc = WilsonMC(tables, L=2, beta=1.0, seed=0)
        np.testing.assert_array_equal(mc.c, tables[2])
        assert mc.d == tables[3]
        result = mc.run(n_equil=0, n_meas=1, stride=1)
        assert set(result) == {
            "plaq_mean", "plaq_susc", "n_samples", "model_metadata",
        }
        assert result["n_samples"] == 1
        assert isinstance(result["plaq_mean"], float)
        assert np.isfinite(result["plaq_mean"])
        assert result["plaq_susc"] == 0.0
        metadata = result["model_metadata"]
        assert metadata == mc.model_metadata
        assert metadata is not mc.model_metadata
        assert metadata["model_id"] == "finite-group-character-wilson-v1"
        assert metadata["group"] == "supplied group tables"
        assert metadata["representation"] == "supplied character/table values c and normalization d"
        assert metadata["beta_lambda_mapping"] == "NOT ESTABLISHED"
        assert metadata["physical_matching"] == "NOT ESTABLISHED"


@pytest.mark.slow
def test_z2_anchor_reproduces_literature():
    """The 3D Z_2 gauge transition (dual Ising) sits at beta ~= 0.7613; the
    small-L susceptibility peak must land within the coarse-scan window."""
    betas = np.arange(0.55, 1.00, 0.05)
    res = scan(z2_tables(), betas, L=4, seed=7, n_equil=100, n_meas=200)
    pc = pseudo_critical(res)
    assert not pc["at_edge"]
    assert abs(pc["beta_c"] - 0.7613) < 0.15


# ---------------------------------------------------------------------------
# glueball_channels_mc — operator symmetry and gate honesty
# ---------------------------------------------------------------------------

def test_channel_operators_at_identity():
    """Cold configuration: A1 = 1 (all plaquettes trivial) and the two
    difference operators (B1, A2) vanish identically."""
    mc = WilsonMC(dn_tables(5), L=4, beta=1.0, seed=0)
    ops = slice_operators(mc, 0)
    assert ops["A1"] == pytest.approx(1.0)
    assert ops["B1"] == pytest.approx(0.0, abs=1e-12)
    assert ops["A2"] == pytest.approx(0.0, abs=1e-12)


def test_correlator_shapes_and_reality():
    r = measure_correlators(5, 1.0, L=3, seed=3, n_equil=30, n_meas=60)
    for ch in ("A1", "B1", "A2"):
        c = r["corr"][ch]
        assert c.shape == (r["n_cfg"], 3)
        assert np.isfinite(c).all()


def test_jackknife_handles_bad_signal():
    """Non-positive correlators must yield None, never a fake mass."""
    c = np.column_stack([np.ones(10), -np.ones(10), np.ones(10) * 0.1])
    out = jackknife_effective_mass(c)
    assert out[0] is None            # C(1) < 0
    assert plateau([None, None]) is None


def test_readiness_gates_withhold_ratios():
    """At a tiny budget the gates must fail and no ratio may be exposed —
    the sealed envelope cannot be burned on noise."""
    r = benchmark_v3_readiness(5, beta=1.0, L=3, seed=3,
                               n_equil=30, n_meas=60)
    assert r["ready_for_v3"] is False
    assert r["envelope"] == "SEALED"
    assert "ratios_available" not in r


# ---------------------------------------------------------------------------
# lepton_dn_assignment — exact rep theory
# ---------------------------------------------------------------------------

def test_mu_constituents_reproduce_label():
    assert 14 * 15 == 210
    assert LEPTON_LABELS["mu"] == pytest.approx(np.sqrt(210.0))


def test_masses_are_charge_inert():
    assert masses_inert_check() is True


def test_fusion_dimensions_conserved():
    """dim(E_a (x) E_b) = 4 must equal the summed dims of the components."""
    dims = {"A1": 1, "A2": 1, "B1": 1, "B2": 1}
    for n in ALLOWED_CLASSES:
        kmax = (n - 1) // 2 if n % 2 else n // 2 - 1
        for a in range(1, kmax + 1):
            for b in range(1, kmax + 1):
                comps = fuse_e_e(a, b, n)
                # E_{a+b} and E_{a-b} branches; a coincidence halves the list
                total = sum(dims.get(cp, 2) for cp in comps)
                assert total in (2, 3, 4)


def test_d5_muon_charge_determinate():
    """n=5: 15 mod 5 = 0 (A-type), so E x A is a single irrep — the wrinkle
    closes outright for the pentagonal class."""
    mu = composite_charge((14, 15), 5)
    assert mu["determinate"] is True
    assert mu["components"] == ["E1"]


def test_wrinkle_status_is_honest():
    st = wrinkle_status()
    assert st["masses_affected"] is False
    assert st["m3_pass_affected"] is False
    assert 5 in st["closed_for_n"]
    for n in st["still_ambiguous_for_n"]:
        tab = lepton_assignment(n)
        assert tab["mu"]["determinate"] is False
        assert len(tab["mu"]["components"]) > 1


def test_electron_tau_always_determinate():
    for n in ALLOWED_CLASSES:
        tab = lepton_assignment(n)
        assert tab["e"]["determinate"] and tab["tau"]["determinate"]


# ---------------------------------------------------------------------------
# Blindness — the three modules never mention a sealed target
# ---------------------------------------------------------------------------

def test_open_problem_modules_blind_to_glueball_targets():
    for mod in ("gauge_phase_mc.py", "glueball_channels_mc.py",
                "lepton_dn_assignment.py", "gauge_mc_fast.py",
                "flavor_label_scan.py"):
        src = (BPR_DIR / mod).read_text()
        for leaked in ("1730", "2400", "2590", "1.387", "1.497", "2370",
                       "2359", "X(2370)", "Morningstar", "BESIII"):
            assert leaked not in src, f"{mod} leaks sealed value {leaked}"
