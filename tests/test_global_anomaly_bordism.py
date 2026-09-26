"""Checks for doc/derivations/global_anomalies_2026-09-26.md.

Oracles independent of the BPR-6D verdict: A(1) associativity, the Adem
relations on every constructed module (a check of the Wu-formula input),
hand-computed Wu relations, the ko_* chart of a point, Witten's SU(2) anomaly
Omega_5^Spin(BSU(2)) = Z2, the absence of 4D Witten and cubic anomalies for
Spin(10), the cubic anomaly of SU(3), the known Spin bordism of BU(1), and
Lee-Tachikawa's Omega_7^Spin(BSU(2)) = Omega_7^Spin(BSU(3)) = 0.
"""

import json

import pytest

import bpr.global_anomaly_bordism as g


def chart(module, s_max=12, T=20):
    return g.ext_chart(module, s_max, T)


def tower_from(entries, start, s_max):
    return entries == [(s, 1) for s in range(start, s_max + 1)]


def test_a1_is_an_eight_dimensional_associative_algebra():
    assert g.a1_checks() == {"dimension": 8, "associative": True, "top_degree": 6}


@pytest.mark.parametrize("n", [4, 6, 10])
def test_wu_formula_satisfies_adem_relations_on_bso(n):
    assert g.module_from_poly(g.bso_algebra(n, 12)).check_adem()


def test_bspin10_steenrod_squares_by_hand():
    alg = g.bspin10_algebra(12)
    gen = {nm: alg.gen(i) for i, nm in enumerate(alg.names)}
    # Wu formula with w1 = w2 = w3 = w5 = w9 = 0: Sq1 w4 = 0, Sq2 w4 = w6, Sq1 w6 = w7, Sq2 w6 = 0,
    # Sq2 w7 = w9 = 0, Sq1 w8 = w9 = 0, Sq2 w8 = w10.
    assert alg.sq(1, gen["w4"]) == frozenset()
    assert alg.sq(2, gen["w4"]) == frozenset([gen["w6"]])
    assert alg.sq(1, gen["w6"]) == frozenset([gen["w7"]])
    assert alg.sq(2, gen["w6"]) == frozenset()
    assert alg.sq(2, gen["w7"]) == frozenset()
    assert alg.sq(1, gen["w8"]) == frozenset()
    assert alg.sq(2, gen["w8"]) == frozenset([gen["w10"]])


def test_every_module_satisfies_the_adem_relations():
    for M in g.bpr6d_modules(14).values():
        assert M.check_adem()
    for n in (2, 3):
        assert g.module_from_poly(g.bsu_algebra(n, 14)).check_adem()


def test_ko_chart_of_a_point():
    c = chart(g.point_module(), 10, 18)
    assert tower_from(c[0], 0, 10)
    assert c[1] == [(1, 1)] and c[2] == [(2, 1)]
    assert tower_from(c[4], 3, 10) and tower_from(c[8], 4, 10)
    assert all(k not in c for k in (3, 5, 6, 7))


def test_witten_su2_anomaly_and_lee_tachikawa():
    c = chart(g.module_from_poly(g.bsu_algebra(2, 14)))
    assert c[5] == [(1, 1)]  # Omega_5^Spin(BSU(2)) = Z2: Witten's anomaly
    assert tower_from(c[4], 0, 12)
    assert 7 not in c  # Omega_7^Spin(BSU(2)) = 0


def test_su3_has_a_cubic_anomaly_but_no_witten_or_7d_anomaly():
    c = chart(g.module_from_poly(g.bsu_algebra(3, 14)))
    assert 5 not in c and 7 not in c
    assert tower_from(c[6], 1, 12)  # free part in degree 6: the 4D cubic anomaly


def test_bu1_spin_bordism():
    c = chart(g.module_from_poly(g.cp_infinity_algebra(14)))
    assert all(k not in c for k in (1, 3, 5, 7))
    assert tower_from(c[2], 0, 12) and tower_from(c[4], 1, 12)
    assert c[6][:2] == [(0, 1), (1, 1)] and all(dim == 2 for s, dim in c[6][2:])  # Z^2


def test_spin10_has_no_4d_witten_or_cubic_anomaly():
    c = chart(g.bpr6d_modules(14)["BSpin(10)"])
    assert 5 not in c and 6 not in c  # pi_4(Spin(10)) = 0; SO(10) has no cubic Casimir
    assert tower_from(c[4], 0, 12)


@pytest.mark.parametrize("name", ["BSpin(10)", "CP^infty", "BSpin(10)^CP^infty"])
def test_omega7_vanishes_for_every_summand(name):
    M = g.bpr6d_modules(14)[name]
    c = g.ext_chart(M, 20, 28)
    assert 7 not in c  # E2 empty in stem 7 for s <= 20: no differentials needed
    assert all(g.sq1_homology(M, d) == 0 for d in range(1, 14, 2))  # no h0-towers in odd stems


def test_truncation_independence():
    for name in ("BSpin(10)", "BSpin(10)^CP^infty"):
        low = g.ext_chart(g.bpr6d_modules(12)[name], 12, 20)
        high = g.ext_chart(g.bpr6d_modules(14)[name], 12, 20)
        for stem in range(0, 10):
            assert low.get(stem) == high.get(stem)


def test_report():
    report = g.demonstration_report()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["verdict"] == {"omega7_vanishes": True, "global_anomaly": "none"}
    assert report["empirical_validation"] is False


def _wu_by_splitting_principle(i, j, n):
    """Sq^i w_j from Sq(e) = e + e^2 on formal roots, re-expressed in elementary symmetric polynomials."""
    import sympy as sp
    from sympy.polys.polyfuncs import symmetrize
    es = sp.symbols("e1:{}".format(n + 1))
    from itertools import combinations
    sigma = sum(sp.prod([es[k] + es[k] ** 2 for k in c]) for c in combinations(range(n), j))
    poly = sp.Poly(sp.expand(sigma), *es)
    part = sum(coeff * sp.prod([e ** p for e, p in zip(es, mono)])
               for mono, coeff in poly.terms() if sum(mono) == j + i)
    sym, rest, names = symmetrize(sp.expand(part), *es, formal=True)
    assert rest == 0
    out = set()
    for term, coeff in sp.Poly(sym, *[nm for nm, _ in names]).terms():
        if coeff % 2 == 0:
            continue
        idx = []
        for (nm, _), power in zip(names, term):
            idx += [int(str(nm)[1:])] * power
        idx = sorted(idx)
        if len(idx) == 1:
            out ^= {(0, idx[0])}
        elif len(idx) == 2:
            out ^= {tuple(idx)}
        else:
            out ^= {tuple(idx)}
    return out


@pytest.mark.parametrize("i,j", [(1, 2), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (2, 6)])
def test_wu_formula_matches_the_splitting_principle(i, j):
    n = 6
    expected = _wu_by_splitting_principle(i, j, n)
    got = {tuple(sorted(pair)) for pair in g.wu(i, j, n)}
    assert got == expected
    assert g.wu(2, 1, n) == []  # Sq^i w_j = 0 for i > j


def test_twisted_structure_bordism_groups():
    data = g.twisted_structure_analysis()
    # Spin x_Z2 Spin(10) x U(1): Omega_7 E2 is a single class at s = 0 (detected by c1 w2 w3).
    assert data["omega7_spin10_u1_E2"] == [[0, 1]]
    # Omega_5^{Spin x_Z2 Spin(10)} = Z2 (the w2 w3 class of Wang-Wen-Witten).
    assert data["omega5_spin10_E2"] == [[0, 1]]
    # Omega_5^{Spin x_Z2 Spin(5)} has order at most 4.
    assert data["omega5_spin5_order_bound"] == 4
    assert data["branching"] == {"is_4_times_4": True, "dimension": 16}
    for n in (5, 10):
        assert g.thom_module_bso(n, 12).check_adem()


def test_twisted_anomaly_is_trivial_on_the_generator():
    arg = g.twisted_anomaly_argument()
    assert arg["flux_index_16_plus"] == 3 and arg["flux_index_16_minus"] == 0
    assert arg["fourth_power_trivial"] and arg["anomaly_on_generator"] == "trivial"
