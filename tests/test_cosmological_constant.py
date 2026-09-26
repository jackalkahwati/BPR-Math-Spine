"""Checks for doc/derivations/cosmological_constant_2026-09-26.md.

Independent oracle: the 6D Einstein equations evaluated directly on the
football metric (d theta^2 + alpha^2 sin^2 theta d phi^2) with de Sitter 4D
slicing and flux quantization imposed by integrating F over the football, using
the curvature routine already checked against hand formulas in
test_six_dim_flux_vacuum; and a regularized cone for the deficit angle.
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
    # Uniform field strength B on the football; fix B by Dirac quantization, e * integral F = 2 pi m.
    Bs = sp.Symbol("B_s", positive=True)
    F_thetaphi = Bs * alpha_value * r ** 2 * sp.sin(v.theta)
    total = sp.integrate(sp.integrate(F_thetaphi, (v.theta, 0, sp.pi)), (v.phi, 0, 2 * sp.pi))
    B = sp.solve(sp.Eq(values[c.e] * total, 2 * sp.pi * flux), Bs)[0]
    F = sp.zeros(6, 6)
    F[4, 5] = F_thetaphi.subs(Bs, B)
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


@pytest.mark.parametrize("alpha_value,sign", [(sp.Rational(53, 100), 1), (sp.Rational(11, 20), 1),
                                              (sp.Rational(3, 5), 0), (sp.Rational(13, 20), -1),
                                              (sp.Rational(7, 10), -1)])
def test_hubble_rate_matches_direct_six_dimensional_equations(alpha_value, sign):
    # alpha < 3/5 (more tension) is de Sitter, 3/5 is flat, alpha > 3/5 is anti-de Sitter.
    values = {c.M: 1, c.e: sp.Rational(1, 2), c.m: 3, c.Lam: sp.Rational(1, 50)}
    H, eqs = _football_equations(alpha_value, 3, values)
    h = sp.Symbol("h", real=True)
    eqs_h = [sp.numer(sp.together(eq.subs(H, sp.sqrt(h)))) for eq in eqs]
    sols = [s for s in sp.solve(eqs_h, [h, v.r], dict=True) if s[v.r].is_real and s[v.r] > 0]
    smallest = min(sols, key=lambda s: float(s[v.r]))
    module = c.hubble_squared().subs(values).subs(c.alpha, alpha_value)
    assert float(smallest[h]) == pytest.approx(float(module), rel=1e-9, abs=1e-12)
    assert (float(module) > 1e-12) - (float(module) < -1e-12) == sign


def test_regularized_cone_deficit_equals_tension_over_M4():
    # Smooth cap f(rho) with f'(0) = 1, f'(inf) = alpha. Brane energy density rho = M^4 G_tt (Lambda = 0, F = 0);
    # its integral is T = 2 pi M^4 (1 - alpha), i.e. deficit delta = 2 pi (1 - alpha) = T / M^4.
    a, eps = sp.Rational(3, 5), sp.Rational(1, 10)
    rho = v.theta  # radial coordinate
    f = a * rho + (1 - a) * eps * sp.atan(rho / eps)
    g = sp.diag(-1, 1, 1, 1, 1, f ** 2)
    E, _, _ = v.einstein_residual(g, sp.zeros(6, 6))
    density = sp.simplify(E[0, 0].subs(v.Lam, 0))  # = M^4 G_tt = T_tt
    integrand = sp.simplify(density * f)  # sqrt(g_2) = f
    T = 2 * sp.pi * sp.integrate(integrand, (rho, 0, sp.oo))
    assert sp.simplify(T - 2 * sp.pi * v.M ** 4 * (1 - a)) == 0
    assert sp.simplify(c.tension_of(a).subs(c.M, v.M) - T) == 0


def test_no_self_tuning_at_the_flat_point():
    test = c.self_tuning_test()
    assert test["H2_at_flat"] == 0
    assert test["dH2_dT"] != 0 and not test["self_tuning"]
    assert test["dH2_dT"] > 0  # more brane vacuum energy -> de Sitter
    assert test["dH2_dT"] == 1 / (90 * sp.pi)


def test_flat_point_derivative_in_closed_form():
    expected = sp.sqrt(2) * c.e * sp.sqrt(c.Lam) / (3 * sp.pi * c.m * c.M ** 4)
    assert sp.simplify(c.flat_point_derivative_closed_form() - expected) == 0


def test_flatness_pins_alpha_on_either_root():
    flat = c.flatness_conditions()
    assert flat["u"] == c.M ** 4 / (2 * c.Lam) and flat["B2"] == 2 * c.Lam
    assert len(flat["alpha"]) == 1 and sp.simplify(flat["alpha"][0] - c.flat_alpha()) == 0


def test_radion_stability_of_the_smaller_root():
    st = c.radion_stability()
    assert st["stationarity_is_constraint"] and st["smaller_root_stable"]
    # Numerical check of V'' at both roots for three tensions.
    values = {c.M: 1, c.e: sp.Rational(1, 2), c.m: 3, c.Lam: sp.Rational(1, 50)}
    V = c.radion_potential().subs(values)
    for a in (sp.Rational(11, 20), sp.Rational(3, 5), sp.Rational(7, 10)):
        roots = sorted(float(x) for x in sp.solve(sp.diff(V.subs(c.alpha, a), c.u), c.u) if x.is_real and x > 0)
        second = [float(sp.diff(V.subs(c.alpha, a), c.u, 2).subs(c.u, x)) for x in roots]
        assert len(roots) == 2 and second[0] > 0 > second[1]


def test_flat_tensions_are_discrete_one_per_flux_quantum():
    values = {c.M: 1, c.e: sp.Rational(1, 2), c.Lam: sp.Rational(1, 50)}
    rows = c.discrete_flat_tensions(values)
    assert [row["flux"] for row in rows] == [1, 2, 3, 4, 5]  # alpha_flat = m/5 must not exceed 1
    for row in rows:
        H2 = c.hubble_squared().subs(values).subs(c.m, row["flux"]).subs(c.alpha, sp.nsimplify(row["alpha"]))
        assert abs(float(H2)) < 1e-12  # each listed tension really is flat
        H2_off = c.hubble_squared().subs(values).subs(c.m, row["flux"]).subs(c.alpha, sp.nsimplify(row["alpha"]) * 0.99)
        assert float(H2_off) > 1e-8  # and a nearby tension is not


def test_tuning_magnitudes():
    mags = c.tuning_magnitudes()
    assert 1e-122 < mags["rho_obs_over_MPl4"] < 1e-119  # reduced Planck mass
    assert 1e-124 < mags["rho_obs_over_unreduced_MPl4"] < 1e-122
    assert mags["rho_obs_over_TeV4"] < 1e-58


def test_report():
    report = c.demonstration_report()
    json.loads(json.dumps(report))
    assert report["self_tuning_test"]["self_tuning"] == "False"
    assert report["limitations"] == c.LIMITATIONS
