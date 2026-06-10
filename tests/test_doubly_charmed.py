"""Tests for doubly-charmed baryon predictions (§12.7–12.9 of qcd_flavor)."""

from __future__ import annotations

import pytest

from bpr.qcd_flavor import (
    QuarkMassSpectrum,
    derive_kappa_HQET_from_substrate,
    doubly_charmed_hyperfine_splitting,
    doubly_charmed_isospin_splitting,
    doubly_charmed_splitting_kappa_scan,
    omega_cc_hyperfine_splitting,
)


# ---------------------------------------------------------------------------
# SU(3) flavor splitting (Omega_cc - Xi_cc)
# ---------------------------------------------------------------------------

def test_su3_splitting_central_matches_lhcb_within_2sigma():
    qms = QuarkMassSpectrum(v_EW_GeV=246.0, p=104761)
    m = qms.all_masses_MeV
    r = doubly_charmed_isospin_splitting(m_s_MeV=m["s"], m_d_MeV=m["d"])
    assert abs(r["residual_sigma"]) < 2.0
    assert r["status"] == "NON-DISCRIMINATING"


def test_su3_splitting_lo_hqet_is_in_tension():
    """LO HQET (kappa=1) is discriminated against at >3 sigma."""
    qms = QuarkMassSpectrum(v_EW_GeV=246.0, p=104761)
    m = qms.all_masses_MeV
    r = doubly_charmed_isospin_splitting(
        m_s_MeV=m["s"], m_d_MeV=m["d"], kappa_HQET=1.0
    )
    assert r["residual_sigma"] < -3.0


def test_kappa_scan_brackets_lhcb():
    scan = doubly_charmed_splitting_kappa_scan()
    residuals = [row["residual_MeV"] for row in scan["scan"]]
    assert min(residuals) < 0 < max(residuals)


def test_kappa_derivation_two_ansatze_converge():
    """Mode-ratio and mass-ratio ansätze agree to better than 1%."""
    r = derive_kappa_HQET_from_substrate()
    k1 = r["kappa_values"]["mode_ratio"]
    k2 = r["kappa_values"]["mass_ratio"]
    assert abs(k1 - k2) / k1 < 0.01
    assert r["kappa_values"]["LO"] == 1.0


# ---------------------------------------------------------------------------
# Hyperfine splitting (Xi_cc* - Xi_cc) — registered prediction
# ---------------------------------------------------------------------------

def test_hyperfine_spin_factor_is_exact_3_over_2():
    r = doubly_charmed_hyperfine_splitting()
    assert r["spin_factor_exact"] == 1.5


def test_hyperfine_registered_prediction_value():
    """Registered: 64.45 * (1 + 2/6) = 85.93 MeV. Locked until LHCb measures."""
    r = doubly_charmed_hyperfine_splitting()
    assert r["registered_prediction_MeV"] == pytest.approx(64.45 * 4.0 / 3.0, rel=1e-12)
    assert r["registered_prediction_MeV"] == pytest.approx(85.93, abs=0.01)


def test_hyperfine_prediction_within_lattice_band():
    r = doubly_charmed_hyperfine_splitting()
    lo, hi = r["lattice_band_MeV"]
    assert lo <= r["registered_prediction_MeV"] <= hi


def test_hyperfine_ansatze_ordering():
    """LO < one-shell < two-shell; no ansatz exceeds lattice upper band."""
    r = doubly_charmed_hyperfine_splitting()
    p = r["predictions_MeV"]
    assert p["LO_equal_wavefunction"] < p["one_shell_pull"] < p["two_shell_pull"]
    assert p["two_shell_pull"] < r["lattice_band_MeV"][1]


def test_omega_cc_hyperfine_registered():
    o = omega_cc_hyperfine_splitting()
    assert o["registered_prediction_MeV"] == pytest.approx(70.7 * 4.0 / 3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# §12.10 Registered pre-dictions: doubly-bottom + mixed + triply-charmed
# ---------------------------------------------------------------------------

def test_xi_bb_hyperfine_registered():
    """Ξbb* − Ξbb hyperfine = 21.2 × 4/3 = 28.27 MeV. Locked until LHCb measures."""
    from bpr.qcd_flavor import doubly_bottom_hyperfine_splitting
    r = doubly_bottom_hyperfine_splitting()
    assert r["registered_prediction_MeV"] == pytest.approx(21.2 * 4.0 / 3.0, rel=1e-12)
    lo, hi = r["lattice_band_MeV"]
    assert lo <= r["registered_prediction_MeV"] <= hi


def test_omega_bb_hyperfine_registered():
    from bpr.qcd_flavor import omega_bb_hyperfine_splitting
    r = omega_bb_hyperfine_splitting()
    assert r["registered_prediction_MeV"] == pytest.approx(25.0 * 4.0 / 3.0, rel=1e-12)


def test_omega_bb_xi_bb_su3_splitting():
    from bpr.qcd_flavor import omega_bb_minus_xi_bb_splitting
    r = omega_bb_minus_xi_bb_splitting()
    assert r["registered_prediction_MeV"] == pytest.approx(1.125 * (93.88 - 4.73), rel=1e-4)


def test_omega_ccc_within_lattice_band():
    from bpr.qcd_flavor import triply_charmed_omega_ccc_mass_predictions
    r = triply_charmed_omega_ccc_mass_predictions()
    lo, hi = r["lattice_band_MeV"]
    assert lo <= r["registered_prediction_MeV"] <= hi
