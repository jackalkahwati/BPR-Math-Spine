"""Checks for doc/derivations/green_schwarz_quantization_2026-09-26.md.

Independent oracles: direct substitution into the symbolic anomaly polynomial
of chiral_parent_completion, brute-force integer searches for Green-Schwarz
vectors in the even lattice U and the odd lattice I_{1,1} (by linear algebra,
not factorization), hand rational arithmetic for the x^4 coefficient, explicit
Gram matrices, a positive control for the odd-lattice search, and grid
minimization of the radion potential. Exact characteristic-class algebra, not
physics validation.
"""

import json
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import sympy as sp

import bpr.chiral_parent_completion as cpc
import bpr.green_schwarz_quantization as g

MONOMIALS = [g.lamV ** 2, g.lamV * g.x2, g.lamV * g.lamT, g.x2 ** 2, g.x2 * g.lamT, g.lamT ** 2]


def target_vector(poly):
    p = sp.Poly(poly, g.lamV, g.x2, g.lamT)
    return [Fraction(str(sp.nsimplify(p.coeff_monomial(mono)))) for mono in MONOMIALS]


def product_vector(e, f):
    """Coefficients of (e . gen)(f . gen) on MONOMIALS, gen = (lambda_V, x2, lambda_T)."""
    return [e[0] * f[0], e[0] * f[1] + e[1] * f[0], e[0] * f[2] + e[2] * f[0],
            e[1] * f[1], e[1] * f[2] + e[2] * f[1], e[2] * f[2]]


def solve_partner(e, target):
    """Integer f with product_vector(e, f) == target, found by least squares then exact check."""
    A = np.array([[e[0], 0, 0], [e[1], e[0], 0], [e[2], 0, e[0]],
                  [0, e[1], 0], [0, e[2], e[1]], [0, 0, e[2]]], dtype=float)
    b = np.array([float(t) for t in target])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    f = [int(round(v)) for v in sol]
    return f if product_vector(e, f) == list(target) else None


def brute_force_U(poly, box=10):
    target = target_vector(poly)
    found = []
    for e in product(range(-box, box + 1), repeat=3):
        if e == (0, 0, 0):
            continue
        f = solve_partner(e, target)
        if f is not None:
            found.append((e, tuple(f)))
    return found


def brute_force_I11(poly, box=16):
    """P R = 2 I8 with P, R integral; equal parity in lambda_V, x2; opposite parity in lambda_T."""
    target = [2 * t for t in target_vector(poly)]
    if any(t.denominator != 1 for t in target):
        return []
    found = []
    for P in product(range(-box, box + 1), repeat=3):
        if P == (0, 0, 0):
            continue
        R = solve_partner(P, target)
        if R is not None and (P[0] - R[0]) % 2 == 0 and (P[1] - R[1]) % 2 == 0 and (P[2] - R[2]) % 2 == 1:
            found.append((P, tuple(R)))
    return found


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_integral_polynomial_matches_direct_substitution(q):
    fields = g.minimal_fields(q)
    direct = cpc.total_polynomial(fields).subs({cpc.S2: 2 * g.lamV, cpc.p1: 2 * g.lamT})
    direct = sp.expand(direct.subs(cpc.X, sp.sqrt(g.x2)))
    assert sp.expand(direct - g.integral_polynomial(fields)) == 0


def test_x4_coefficient_forces_three_to_divide_the_parent_charge():
    # Any lattice: the x^4 coefficient (1/2) b_X.b_X lies in (1/2)Z. For 16_+(q): 16 q^4 / 24.
    for q in range(1, 31):
        coeff = Fraction(16 * q ** 4, 24)
        assert ((2 * coeff).denominator == 1) == (q % 3 == 0)
        assert g.necessary_conditions(g.minimal_fields(q))["pass"] == (q % 3 == 0)


def test_parent_charge_one_fails_every_lattice():
    fields = g.minimal_fields(1)
    nec = g.necessary_conditions(fields)
    assert not nec["pass"] and nec["gram"]["bX.bX"] == "4/3"
    assert not g.hyperbolic_solution(fields)["exists"]
    assert brute_force_U(g.integral_polynomial(fields)) == []
    assert not g.odd_lattice_solution(fields)["exists"]


def test_parent_charge_three_has_an_explicit_U_lattice_solution():
    fields = g.minimal_fields(3)
    poly = g.integral_polynomial(fields)
    sol = g.hyperbolic_solution(fields)
    assert sol["exists"]
    assert sp.expand(sp.sympify(sol["Y_e"]) * sp.sympify(sol["Y_g"]) - poly) == 0
    # Independent brute force finds the same factorization class, e.g. (6 x2)(3 lambda_V + 9 x2 - lambda_T).
    found = brute_force_U(poly)
    assert ((0, 6, 0), (3, 9, -1)) in found
    # Explicit lattice vectors in U reproduce every required Gram entry.
    vecs = g.lattice_vectors_in_U(sol)
    required = {k: int(sp.Rational(v)) for k, v in g.necessary_conditions(fields)["gram"].items()}
    assert vecs["gram"] == required
    assert all(v % 2 == 0 for v in vecs["a"])  # characteristic in the even lattice U


def test_odd_lattice_is_obstructed_but_the_search_can_succeed():
    for q in (1, 3, 6):
        poly = g.integral_polynomial(g.minimal_fields(q))
        assert not g.odd_lattice_solution(g.minimal_fields(q))["exists"]
        assert brute_force_I11(poly) == []
    # Positive control: pure singlets 1_+(0) 1_+(1) 1_-(3) 1_-(4) admit an I_{1,1} solution.
    control = [(1, "1", 0), (1, "1", 1), (-1, "1", 3), (-1, "1", 4)]
    assert g.odd_lattice_solution(control)["exists"]
    assert brute_force_I11(g.integral_polynomial(control)) != []


def test_scan_and_family_number():
    rows = g.minimal_completion_scan(12)
    assert [r["parent_charge"] for r in rows if r["U"]] == [3, 6, 9, 12]
    assert all(r["necessary"] == r["U"] for r in rows)
    assert not any(r["I11"] for r in rows)
    stmt = g.family_number_statement()
    assert stmt["all_multiples_of_three"] and stmt["every_multiple_of_three_passes_U"]
    assert stmt["minimal_family_number"] == 3
    # Index theorem: n_gen = q |m|, so flux 1 with q = 3 gives exactly three chiral 16s.
    modes = cpc.zero_modes(g.minimal_fields(3), 1)
    assert modes == [{"representation": "16", "charge": 3, "multiplicity": 3}]


def test_vectorlike_charge_one_states_do_not_change_the_anomaly():
    base = g.integral_polynomial(g.minimal_fields(3))
    for Q in (1, 2):
        extended = g.minimal_fields(3) + [(1, "1", Q), (-1, "1", Q)]
        assert sp.expand(g.integral_polynomial(extended) - base) == 0


def test_near_minimal_alternatives():
    rows = {(r["parent_charge"], r["Qa"], r["Qb"]): r for r in g.near_minimal_scan()}
    # At q = 1 only (Qa, Qb) = (0, 4) and (3, 1) pass, both adding four massless singlets per flux unit.
    passing_q1 = sorted((qa, qb) for (q, qa, qb), r in rows.items() if q == 1 and r["U"])
    assert passing_q1 == [(0, 4), (3, 1)]
    assert rows[(1, 0, 4)]["massless_singlets_per_unit_flux"] == 4
    assert all(r["necessary"] == r["U"] for r in rows.values())


def test_unit_flux_vacuum_is_the_only_compactified_minimum():
    land = g.unit_flux_landscape()
    kinds = {row["flux"]: row["vacuum"] for row in land["sectors"]}
    assert kinds == {1: "Minkowski", 2: "none", 3: "none"}
    assert not land["zero_flux_has_minimum"]
    # Independent grid check for m = 0: V has a single interior maximum and no minimum.
    import bpr.six_dim_flux_vacuum as v
    V0 = sp.lambdify(v.r, v.reduced_potential().subs({v.M: 1, v.e: 1, v.Lam: 2, v.r0: 1, v.m: 0}))
    grid = np.linspace(0.05, 20, 200001)
    vals = V0(grid)
    interior_min = np.where((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:]))[0]
    interior_max = np.where((vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:]))[0]
    assert len(interior_min) == 0 and len(interior_max) == 1


def test_report_is_strict_json():
    report = g.demonstration_report()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["empirical_validation"] is False
    assert report["limitations"] == g.LIMITATIONS
