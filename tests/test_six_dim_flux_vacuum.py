"""Checks for doc/derivations/flux_compactification_2026-09-26.md.

Independent oracles: hand-derived product-space curvature, the hand-reduced
radion potential, numerical minimization, and the standard n=2 breathing-mode
normalization. Exact symbolic checks of a supplied action, not physics validation.
"""

import json
import math

import numpy as np
import pytest
import sympy as sp

import bpr.six_dim_flux_vacuum as v


def test_curvature_matches_product_space_formulas():
    cond = v.einstein_conditions()
    assert sp.simplify(cond["ricci_scalar"] - 2 / v.r ** 2) == 0  # flat M4 + round S^2 of radius r
    assert sp.simplify(cond["F_squared"] - 2 * v.B ** 2) == 0
    assert cond["off_diagonal_vanish"] and cond["four_d_isotropic"] and cond["sphere_isotropic"]
    # Hand derivation: 4D block  -M^4/r^2 = -B^2/2 - Lambda ; sphere block 0 = B^2/2 - Lambda.
    assert sp.simplify(cond["four_d_equation"] - (v.Lam + v.B ** 2 / 2 - v.M ** 4 / v.r ** 2)) == 0
    assert sp.simplify(cond["sphere_equation"] - (v.Lam - v.B ** 2 / 2)) == 0


def test_vacuum_solution_and_flux_quantization():
    sol = v.vacuum_solution()
    assert sp.simplify(sol["radius"] - v.m / (2 * v.M ** 2 * v.e)) == 0
    assert sp.simplify(sol["Lambda"] - 2 * v.M ** 8 * v.e ** 2 / v.m ** 2) == 0
    # Dirac quantization with unit charge: e * B * 4 pi r^2 = 2 pi m.
    assert sp.simplify(v.e * sol["B"] * 4 * sp.pi * sol["radius"] ** 2 - 2 * sp.pi * v.m) == 0


def test_reduced_potential_matches_hand_reduction():
    hand = 4 * sp.pi * v.r0 ** 4 * (-v.M ** 4 / v.r ** 4 + v.Lam / v.r ** 2
                                    + v.m ** 2 / (8 * v.e ** 2 * v.r ** 6))
    assert sp.simplify(v.reduced_potential() - hand) == 0


def test_breathing_mode_normalization_is_standard():
    kin = v.radion_kinetic_coefficient()
    K0 = sp.simplify(kin["K"].subs(v.psi, 0))
    # For n = 2 internal dimensions, L_kin = -(n(n+2)/4) M_Pl^2 (d psi)^2 = -2 M_Pl^2 (d psi)^2.
    assert sp.simplify(K0 / kin["M_Pl_squared"] - 4) == 0


def test_radion_is_stable_with_mass_one_over_radius():
    stab = v.radion_stability()
    assert stab["V_at_vacuum"] == 0 and stab["dV"] == 0
    assert sp.simplify(stab["d2V"] - 16 * sp.pi * v.M ** 4) == 0
    assert stab["radion_mass_squared_times_r0_squared"] == 1


@pytest.mark.parametrize("flux", [1, 2, 3, 5])
def test_numerical_minimum_of_the_potential(flux):
    Mv, ev = 1.3, 0.7
    sol = v.vacuum_solution()
    subs = {v.M: Mv, v.e: ev, v.m: flux}
    r_star = float(sol["radius"].subs(subs))
    lam = float(sol["Lambda"].subs(subs))
    V = sp.lambdify(v.r, v.reduced_potential().subs({**subs, v.Lam: lam, v.r0: r_star}))
    grid = np.linspace(0.5 * r_star, 2.0 * r_star, 20001)
    values = V(grid)
    assert abs(grid[np.argmin(values)] - r_star) < 1e-3 * r_star
    assert abs(V(r_star)) < 1e-12
    # Detuning Lambda upward makes the minimum positive (de Sitter-like); downward, negative.
    V_up = sp.lambdify(v.r, v.reduced_potential().subs({**subs, v.Lam: 1.01 * lam, v.r0: r_star}))
    V_dn = sp.lambdify(v.r, v.reduced_potential().subs({**subs, v.Lam: 0.99 * lam, v.r0: r_star}))
    assert np.min(V_up(grid)) > 0 > np.min(V_dn(grid))


def test_four_dimensional_scale_relations():
    rel = v.four_d_relations()
    assert sp.simplify(rel["M_Pl_squared"] - sp.pi * v.m ** 2 / v.e ** 2) == 0
    assert rel["inverse_radius_in_terms_of_g4"] == 0
    assert rel["inverse_radius_over_M_in_terms_of_g4"] == 0
    rows = v.illustrative_scales(3, (0.5,))
    assert rows[0]["inverse_radius_GeV"] == pytest.approx(2 * 0.5 * v.REDUCED_PLANCK_GEV / 3)
    assert rows[0]["radion_mass_GeV"] == pytest.approx(rows[0]["inverse_radius_GeV"])


def test_classical_control_needs_a_weak_flux_coupling():
    rel = v.four_d_relations()
    bound = float(rel["control_bound_on_g4"].subs(v.m, 3))
    assert bound == pytest.approx(3 / (4 * math.sqrt(math.pi)))
    # Numerically, 1/r crosses M6 exactly at the bound (independent of the symbolic relation).
    below, above = v.illustrative_scales(3, (0.99 * bound, 1.01 * bound))
    assert below["inverse_radius_over_M6"] < 1 < above["inverse_radius_over_M6"]


def test_green_schwarz_background_is_unsourced():
    # X is a 2-form on S^2 only, so X wedge X is a 4-form on a 2D space: zero.
    X = sp.Symbol("B0")
    two_form = sp.zeros(6, 6)
    two_form[4, 5], two_form[5, 4] = X, -X
    # Any 4-form built from two copies needs four distinct indices among {4, 5}: impossible.
    nonzero = [(a, b) for a in range(6) for b in range(a + 1, 6) if two_form[a, b] != 0]
    assert all(set(p) <= {4, 5} for p in nonzero)
    assert v.green_schwarz_background() == {"X_wedge_X": 0, "S2": 0, "p1": 0, "B_field_source": 0}


def test_report_is_strict_json():
    report = v.demonstration_report()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["empirical_validation"] is False
    assert report["limitations"] == v.LIMITATIONS


@pytest.mark.parametrize("reference", [3, 7])
def test_flux_landscape_matches_grid_minimization(reference):
    land = v.flux_landscape(reference, tuple(range(1, reference + 3)))
    lam = 2 / reference ** 2
    V0 = v.reduced_potential().subs({v.M: 1, v.e: 1, v.Lam: lam, v.r0: 1})
    grid = np.linspace(0.05, 40.0, 400001)
    for row in land["sectors"]:
        values = sp.lambdify(v.r, V0.subs(v.m, row["flux"]))(grid)
        interior = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1
        if row["vacuum"] == "none":
            assert len(interior) == 0
            continue
        assert len(interior) == 1
        idx = interior[0]
        assert grid[idx] == pytest.approx(row["radius"], rel=1e-3)
        expected = {"Minkowski": 0, "de Sitter": 1, "anti-de Sitter": -1}[row["vacuum"]]
        if expected == 0:
            assert abs(values[idx]) < 1e-9
        else:
            assert np.sign(values[idx]) == expected
    # Only the reference sector is flat; lower flux is AdS; m^2 <= (4/3) reference^2.
    kinds = [row["vacuum"] for row in land["sectors"]]
    assert kinds.count("Minkowski") == 1 and kinds[reference - 1] == "Minkowski"
    assert all(kind == "anti-de Sitter" for kind in kinds[:reference - 1])
    for row in land["sectors"]:
        assert (row["vacuum"] != "none") == (row["flux"] ** 2 <= 4 * reference ** 2 / 3)
