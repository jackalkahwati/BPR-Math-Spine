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


def _anomalies_of_zero_modes(h):
    """SU(3)^3, Y^3, SU(3)^2 Y, SU(2)^2 Y and grav-Y of the full zero-mode spectrum (both 16s, both chiralities)."""
    out = {"Y3": F(0), "grav_Y": F(0), "su3_sq_Y": F(0), "su2_sq_Y": F(0), "su3_cubed": F(0), "su2_doublets": F(0)}
    for row in g.zero_mode_indices(h):
        w = row["weight"]
        n = row["from_16plus"] + row["from_16minus"]  # signed chiral multiplicity
        y = g.dot(g.Y, w)
        out["Y3"] += n * y ** 3
        out["grav_Y"] += n * y
        colour = [c for c in w[:3]]
        is_triplet = len({c for c in colour}) == 2  # colour weight of a 3 or 3bar
        if is_triplet:
            out["su3_sq_Y"] += n * y
            out["su3_cubed"] += n * (1 if sum(1 for c in colour if c > 0) == 1 else -1)  # 3 vs 3bar weights
        if g.dot(g.T3L, w) != 0:
            out["su2_sq_Y"] += n * y
            out["su2_doublets"] += n * (1 if g.dot(g.T3L, w) > 0 else 0)
    return out


def test_zero_mode_spectrum_is_anomaly_free_for_every_quantized_flux():
    rng = random.Random(1)
    checked = 0
    for _ in range(200):
        h = tuple(F(rng.randint(-3, 3)) for _ in range(5))
        if not g.flux_quantized(h):
            continue
        an = _anomalies_of_zero_modes(h)
        assert an["Y3"] == 0 and an["grav_Y"] == 0 and an["su3_sq_Y"] == 0 and an["su2_sq_Y"] == 0
        assert an["su3_cubed"] == 0 and an["su2_doublets"] % 2 == 0  # SU(3)^3 and Witten's SU(2) anomaly
        checked += 1
    assert checked > 20


def test_every_quantized_spin10_flux_is_unstable():
    # Round 3, Prop. 5b: W bosons with monopole number |n| >= 2 are tachyonic; pi_1(Spin(10)) = 0, so the flux relaxes.
    scan = g.stability_scan(3)
    assert scan["quantized_nonzero"] > 8000 and scan["stable"] == 0
    assert g.max_root_monopole_number((F(-2), F(-2), F(-2), F(-3), F(-3))) == 6


def test_flux_breaking_never_gives_the_standard_model_with_massless_hypercharge():
    theorem = g.flux_plane_theorem()
    assert theorem["generic_quantized"] and not theorem["generic_stable"]
    assert theorem["flipped_massless_direction_roots"] == 20 and not theorem["flipped_generic_massless"]
    assert theorem["generic_in_plane_is_sm"] and not theorem["generic_in_plane_Y_massless"]
    assert theorem["X_direction_Y_massless"] and theorem["X_direction_centralizer_roots"] == 20
    hits = {"sm": 0, "sm_massless_Y": 0}
    for h in product(range(-3, 4), repeat=5):
        if any(h) and g.centralizer_is_exactly_sm(h):
            hits["sm"] += 1
            hits["sm_massless_Y"] += int(g.hypercharge_massless(h))
    assert hits["sm"] > 0 and hits["sm_massless_Y"] == 0


def test_bf_vector_is_derived_from_the_green_schwarz_class():
    # Y_g = 3 lambda_V + 9 x^2 - lambda_T at parent charge 3 (round 4); expanding around the background gives the
    # 4D BF vector (18 m, 3 h): the massive Spin(10) direction is h itself.
    sol = gs.hyperbolic_solution(gs.minimal_fields(3))
    import sympy as sp
    Yg = sp.sympify(sol["Y_g"])
    assert sp.Poly(Yg, gs.lamV, gs.x2, gs.lamT).coeff_monomial(gs.lamV) == 3
    for h in [(2, 2, 2, 1, 1), (0, 0, 0, 1, 1), (2, 0, 0, 0, 0)]:
        vec = g.bf_vector(h)
        assert vec[0] == 18 and vec[1:] == [3 * x for x in h]
        # Y is massless only if its (trace-form) projection on the massive direction vanishes.
        assert (g.dot(g.Y, h) == 0) == g.hypercharge_massless(tuple(F(x) for x in h))


@pytest.mark.parametrize("j", range(1, 11))
def test_d2_multiplicities_match_characters_and_are_never_uniform(j):
    assert g.d2_multiplicities(j) == g.d2_character_formula(j)
    assert g.orbifold_family_counts(j)["uniform_choices"] == []


def test_rotation_quotients_keep_at_most_two_families_per_multiplet():
    bounds = g.max_families_per_multiplet_under_rotations(24)
    assert bounds[2] == 2 and all(bounds[N] == 1 for N in range(3, 25)) and bounds["Z2xZ2"] == 1


def test_orbifold_classes_follow_pati_salam_and_su5():
    classes = g.orbifold_classes()
    assert {lab for lab, c in classes.items() if c[0] == 1} == {"Q", "L"}  # (4,2,1)
    assert {lab for lab, c in classes.items() if c[1] == 1} == {"Q", "u^c", "e^c"}  # the SU(5) 10
    counts = g.orbifold_family_counts(1)["counts"]
    zeros = []
    for choice in counts.values():
        assert min(choice.values()) == 0 and max(choice.values()) == 1
        zeros.append(sum(1 for v in choice.values() if v == 0))
    assert sorted(zeros) == [1, 1, 2, 2]  # one or two multiplets lose every family


def test_report():
    report = g.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["orbifold_uniform_j_up_to_10"] == []
    assert report["flux_scan"]["sm_with_massless_Y"] == 0
    assert report["limitations"] == g.LIMITATIONS
