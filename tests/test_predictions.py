"""Checks for doc/derivations/predictions_2026-09-26.md.

Oracles: the Stueckelberg rotation (the physical axion is orthogonal to the eaten
one), the round-4 Green-Schwarz coefficients, the standard QCD axion mass
relation, and the flux-note compactification relation.
"""

import json

import numpy as np
import pytest

import bpr.green_schwarz_quantization as gs
import bpr.predictions as p


def test_stueckelberg_charges_come_from_the_round4_couplings():
    sol = gs.hyperbolic_solution(gs.minimal_fields(3))
    assert sol["Y_e"] == "6*x2" and "3*lambda_V" in sol["Y_g"] and "9*x2" in sol["Y_g"]
    ch = p.gs_stueckelberg_charges()
    assert ch == {"k_b": 12, "k_a": 18, "qcd_coefficient_of_b": 3}


@pytest.mark.parametrize("ratio", [1e-3, 0.1, 1.0, 10.0, 1e3])
def test_the_physical_axion_always_couples_to_qcd(ratio):
    out = p.stueckelberg_axions(12, 18, 1.0, ratio)
    eaten, phys = np.array(out["eaten"]), np.array(out["physical"])
    assert abs(eaten @ phys) < 1e-12 and abs(np.linalg.norm(phys) - 1) < 1e-12
    assert abs(out["qcd_component"]) > 0  # nonzero whenever k_a != 0


def test_the_axion_decouples_only_if_the_bf_term_vanishes():
    assert p.stueckelberg_axions(12, 0, 1.0, 1.0)["qcd_component"] == pytest.approx(0.0, abs=1e-15)


def test_axion_mass_relation_and_band():
    assert p.axion_mass_ev(1e12) == pytest.approx(5.70e-6)
    band = p.axion_band()
    lo, hi = band["fa_gev"]
    assert 1e14 < lo < hi < 1e18
    assert band["ma_ev"][0] < band["ma_ev"][1] < 1e-7
    assert all(t < 0.1 for t in band["theta_i_for_dm"])  # dark-matter misalignment needs a small angle


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
