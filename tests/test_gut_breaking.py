"""Checks for doc/derivations/gut_breaking_2026-09-26.md.

Oracles: Standard-Model content and anomaly sums of the 16, explicit root
systems (SU(5) has 20 roots, SU(3) x SU(2) has 8), exhaustive flux scans, the
round-4 Green-Schwarz coefficient, and Wigner rotation matrices checked against
the SO(3) character formula.
"""

import json
import random
from fractions import Fraction as F
from itertools import product

import pytest

import bpr.gut_breaking as g
import bpr.green_schwarz_quantization as gs


def test_sixteen_has_standard_model_content_and_is_anomaly_free():
    assert g.sm_multiplet_dimensions() == {"Q": 6, "u^c": 3, "d^c": 3, "L": 2, "e^c": 1, "nu^c": 1}
    ws = g.weights16()
    assert sum(g.dot(g.Y, w) for w in ws) == 0 and sum(g.dot(g.Y, w) ** 3 for w in ws) == 0
    assert g.dot(g.Y, g.X) == 0  # Y and X are orthogonal in the trace form


def test_any_spin10_flux_is_chirality_neutral():
    rng = random.Random(1)
    for _ in range(50):
        h = tuple(F(rng.randint(-3, 3)) for _ in range(5))
        if not g.flux_quantized(h):
            continue
        for row in g.zero_mode_indices(h):
            assert row["net"] == 3  # 16_+ and 16_- are the same Spin(10) rep with opposite 6D chirality


def test_flux_breaking_never_gives_the_standard_model_with_massless_hypercharge():
    theorem = g.flux_plane_theorem()
    assert theorem["generic_in_plane_is_sm"] and not theorem["generic_in_plane_Y_massless"]
    assert theorem["X_direction_Y_massless"] and theorem["X_direction_centralizer_roots"] == 20
    hits = {"sm": 0, "sm_massless_Y": 0}
    for h in product(range(-3, 4), repeat=5):
        if any(h) and g.centralizer_is_exactly_sm(h):
            hits["sm"] += 1
            hits["sm_massless_Y"] += int(g.hypercharge_massless(h))
    assert hits["sm"] > 0 and hits["sm_massless_Y"] == 0


def test_the_stueckelberg_coefficient_is_nonzero():
    # Y_g = 3 lambda_V + 9 x^2 - lambda_T at parent charge 3: the lambda_V coefficient drives the 4D BF term
    # B ^ tr(<F> F), proportional to 3 tr(h h), which is positive for every nonzero Cartan flux.
    sol = gs.hyperbolic_solution(gs.minimal_fields(3))
    import sympy as sp
    Yg = sp.sympify(sol["Y_g"])
    assert sp.Poly(Yg, gs.lamV, gs.x2, gs.lamT).coeff_monomial(gs.lamV) == 3
    for h in [(1, 1, 1, 3, 3), (0, 0, 0, 1, 1), (1, 0, 0, 0, 0)]:
        assert g.dot(h, h) > 0


@pytest.mark.parametrize("j", range(1, 11))
def test_d2_multiplicities_match_characters_and_are_never_uniform(j):
    assert g.d2_multiplicities(j) == g.d2_character_formula(j)
    assert g.orbifold_family_counts(j)["uniform_choices"] == []


def test_orbifold_classes_follow_pati_salam_and_su5():
    classes = g.orbifold_classes()
    assert {lab for lab, c in classes.items() if c[0] == 1} == {"Q", "L"}  # (4,2,1)
    assert {lab for lab, c in classes.items() if c[1] == 1} == {"Q", "u^c", "e^c"}  # the SU(5) 10
    counts = g.orbifold_family_counts(1)["counts"]
    for choice in counts.values():
        assert min(choice.values()) == 0 and max(choice.values()) == 1


def test_report():
    report = g.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["orbifold_uniform_j_up_to_10"] == []
    assert report["flux_scan"]["sm_with_massless_Y"] == 0
    assert report["limitations"] == g.LIMITATIONS
