"""Checks for doc/derivations/predictions_2026-09-26.md.

Oracles: the round-4 Green-Schwarz factorization (charges parsed from it), the
U(1)_F-SU(3)^2 anomaly of the 16 content, gauge invariance of theta-bar under
U(1)_F and hypercharge, the nullspace of the charge matrix, the standard QCD
axion relations, and the flux-note compactification relation.
"""

import json

import numpy as np
import pytest
import sympy as sp

import bpr.green_schwarz_quantization as gs
import bpr.predictions as p


def test_stueckelberg_charges_are_parsed_from_the_round4_factorization():
    sol = gs.hyperbolic_solution(gs.minimal_fields(3))
    assert sp.simplify(sp.sympify(sol["Y_g"]) - sp.sympify("3*lambda_V + 9*x2 - lambda_T")) == 0
    assert p.gs_stueckelberg_charges() == {"k_b": 12, "k_a": 18, "qcd_coefficient_of_b": 3}
    assert p.gs_stueckelberg_charges(flux=2)["k_b"] == 24


def test_gs_qcd_coefficient_matches_the_u1f_su3_anomaly():
    # Per 16 the coloured Weyl fermions (Q twice, u^c, d^c) have sum T = 2; three families of charge 3 give 18.
    assert p.f_su3_anomaly() == 18
    th = p.theta_bar_structure()
    assert th["gs_coefficient_matches_anomaly"]


def test_theta_bar_is_gauge_invariant_and_does_not_contain_a():
    th = p.theta_bar_structure()
    assert th["f_invariant"] and th["y_invariant"]
    assert th["coefficients"]["a"] == 0 and th["coefficients"]["b"] == 3
    # Dropping the Higgs phases breaks U(1)_F invariance: the Higgs is necessarily Peccei-Quinn charged.
    c, q = th["coefficients"], th["f_charges"]
    assert c["b"] * q["b"] + c["a"] * q["a"] == 36 != 0


def test_two_physical_phases_theta_bar_and_zeta():
    out = p.gauge_invariant_phases()
    assert out["dimension"] == 2
    basis = sp.Matrix([[sp.Rational(x) for x in v] for v in out["basis"]])
    theta_bar = sp.Matrix([[3, 0, 3, 3]])
    zeta = sp.Matrix([[3, -2, 0, 0]])
    assert basis.rank() == sp.Matrix.vstack(basis, theta_bar, zeta).rank() == 2
    assert p.gauge_invariant_phases(with_singlet_charge=12)["dimension"] == 3


def test_minimal_higgs_gives_an_excluded_pqww_axion():
    f = p.pqww_decay_constant()
    assert f == pytest.approx(p.HIGGS_VEV_GEV / np.sqrt(2) / np.sqrt(18), rel=1e-3)
    assert f < 50  # ~41 GeV at tan beta = 1: a visible PQWW axion, long excluded
    assert p.pqww_decay_constant(tan_beta=10) < f


def test_heavy_singlet_makes_the_axion_invisible():
    assert p.dfsz_like_decay_constant(1e16, 1e12, 6) == pytest.approx(1e12 / 6, rel=1e-3)
    assert p.dfsz_like_decay_constant(1e16, 1e20, 6) == pytest.approx(1e16 / 3, rel=1e-3)


def test_quality_needs_a_large_action():
    assert 185 < p.quality_required_action() < 195
    lo, hi = p.wrapped_string_action()
    assert hi < p.quality_required_action()


def test_axion_mass_relation_band_and_cosmology():
    assert p.axion_mass_ev(1e12) == pytest.approx(5.70e-6)
    band = p.axion_band()
    lo, hi = band["f_gev"]
    assert 1e14 < lo < hi < 1e18
    assert band["ma_ev"][0] < band["ma_ev"][1] < 1e-7
    assert 1e3 < band["frequency_hz"][0] < band["frequency_hz"][1] < 1e7
    assert all(t < 0.1 for t in band["theta_i_for_dm"])
    # Superradiance disfavours the top of the band; isocurvature forces low-scale inflation if the axion is the DM.
    assert band["superradiance_excluded_f"][0] < hi
    assert all(x["H_inflation_max_gev"] < 1e10 and x["tensor_to_scalar_max"] < 1e-8 for x in band["isocurvature_if_dm"])


def test_misalignment_is_continuous_at_the_crossover():
    fc = 1.5e17
    assert p.misalignment_theta_for_dm(fc * (1 - 1e-9)) == pytest.approx(p.misalignment_theta_for_dm(fc * (1 + 1e-9)))
    assert p.misalignment_theta_for_dm(1e16) == pytest.approx(4.4e-3, rel=0.05)


def test_compactification_band_matches_the_flux_note():
    lo, hi = p.compactification_band()
    assert lo == pytest.approx(2 * 0.02 * p.REDUCED_PLANCK_GEV / 3)
    assert hi == pytest.approx(2 * 0.4 * p.REDUCED_PLANCK_GEV / 3)


def test_family_number():
    fam = p.family_number_statements()
    assert fam["consistent"] and 4 in fam["excludes"] and 6 in fam["allowed"]


def test_report():
    report = p.demonstration_report()
    json.loads(json.dumps(report))
    assert report["empirical_validation"] is False and report["limitations"] == p.LIMITATIONS
