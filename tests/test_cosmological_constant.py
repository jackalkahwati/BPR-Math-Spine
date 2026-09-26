"""Checks for doc/derivations/cosmological_constant_2026-09-26.md.

Independent oracle: the 6D Einstein equations evaluated directly on the
football metric (d theta^2 + alpha^2 sin^2 theta d phi^2) with de Sitter 4D
slicing and flux quantization on the reduced area, using the curvature routine
already checked against hand formulas in test_six_dim_flux_vacuum.
"""

import json

import pytest
import sympy as sp

import bpr.cosmological_constant as c
import bpr.six_dim_flux_vacuum as v


def _football_equations(alpha_value, flux, values):
    H = sp.Symbol("H", positive=True)
    r = v.r
    g = sp.diag(-1, sp.exp(2 * H * v.t), sp.exp(2 * H * v.t), sp.exp(2 * H * v.t),
                r ** 2, alpha_value ** 2 * r ** 2 * sp.sin(v.theta) ** 2)
    B = flux / (2 * values[c.e] * alpha_value * r ** 2)  # e B (4 pi alpha r^2) = 2 pi m
    F = sp.zeros(6, 6)
    F[4, 5] = B * alpha_value * r ** 2 * sp.sin(v.theta)
    F[5, 4] = -F[4, 5]
    E, _, _ = v.einstein_residual(g, F)
    subs = {v.M: values[c.M], v.Lam: values[c.Lam]}
    return H, [sp.simplify((E[1, 1] / g[1, 1]).subs(subs)), sp.simplify((E[4, 4] / g[4, 4]).subs(subs))]


def test_large_tension_has_no_compactified_vacuum_in_either_computation():
    # alpha = 1/2: effective flux m/alpha = 6 exceeds sqrt(4/3) * 5 (the flat flux is 5): no stationary point.
    values = {c.M: 1, c.e: sp.Rational(1, 2), c.m: 3, c.Lam: sp.Rational(1, 50)}
    H, eqs = _football_equations(sp.Rational(1, 2), 3, values)
    h = sp.Symbol("h", real=True)
    eqs_h = [sp.numer(sp.together(eq.subs(H, sp.sqrt(h)))) for eq in eqs]
    sols = [s for s in sp.solve(eqs_h, [h, v.r], dict=True) if s[v.r].is_real and s[v.r] > 0]
    assert sols == []
    assert not c.stationary_radius_squared().subs(values).subs(c.alpha, sp.Rational(1, 2)).is_real


@pytest.mark.parametrize("alpha_value", [sp.Rational(3, 5), sp.Rational(13, 20), sp.Rational(7, 10)])
def test_hubble_rate_matches_direct_six_dimensional_equations(alpha_value):
    values = {c.M: 1, c.e: sp.Rational(1, 2), c.m: 3, c.Lam: sp.Rational(1, 50)}
    H, eqs = _football_equations(alpha_value, 3, values)
    h = sp.Symbol("h", real=True)
    eqs_h = [sp.numer(sp.together(eq.subs(H, sp.sqrt(h)))) for eq in eqs]
    sols = [s for s in sp.solve(eqs_h, [h, v.r], dict=True) if s[v.r].is_real and s[v.r] > 0]
    smallest = min(sols, key=lambda s: float(s[v.r]))
    module = c.hubble_squared().subs(values).subs(c.alpha, alpha_value)
    assert float(smallest[h]) == pytest.approx(float(module), rel=1e-9, abs=1e-12)


def test_no_self_tuning_at_the_flat_point():
    test = c.self_tuning_test()
    assert test["H2_at_flat"] == 0
    assert test["dH2_dT"] != 0 and not test["self_tuning"]
    assert test["dH2_dT"] > 0  # more brane vacuum energy -> de Sitter


def test_flat_tensions_are_discrete_one_per_flux_quantum():
    rows = c.discrete_flat_tensions()
    alphas = [row["alpha"] for row in rows]
    assert alphas == sorted(alphas) and len(set(alphas)) == len(alphas)
    diffs = {round(b - a, 12) for a, b in zip(alphas, alphas[1:])}
    assert len(diffs) == 1  # alpha_flat is linear in m: equally spaced, isolated values


def test_tuning_magnitudes():
    mags = c.tuning_magnitudes()
    assert 1e-122 < mags["rho_obs_over_MPl4"] < 1e-119
    assert mags["rho_obs_over_TeV4"] < 1e-58


def test_report():
    report = c.demonstration_report()
    json.loads(json.dumps(report))
    assert report["self_tuning_test"]["self_tuning"] == "False"
    assert report["limitations"] == c.LIMITATIONS
