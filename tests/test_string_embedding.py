"""Checks for doc/derivations/string_embedding_2026-09-26.md.

Oracles: the known SU(2) spectra on T = 0 (genus formula, 22 fundamentals on a
line, 54 + 1 adjoint on a cubic), SO(10) on a -4 curve with two vectors, the
Park-Taylor U(1) normalization anchored in green_schwarz_quantization, and
direct recomputation of the anomaly equations for the example spectra.
"""

import json
from fractions import Fraction

import pytest

import bpr.string_embedding as s


@pytest.mark.parametrize("b,genus,fund", [(1, 0, 22), (2, 0, 40), (3, 1, 54), (4, 3, 64)])
def test_su2_t0_reproduces_plane_curve_genus(b, genus, fund):
    row = s.su2_t0_spectrum(b)
    assert row["equations_hold"]
    assert row["genus"] == Fraction((b - 1) * (b - 2), 2) == genus
    assert row["fundamentals"] == fund


def test_so10_nonabelian_conditions():
    # Standard 6D result: SO(10) on a -4 curve carries exactly two vectors and no spinors.
    minus4 = s.so10_on_curve(-4)
    assert minus4["consistent"] and minus4["n16"] == 0 and minus4["n10"] == 2
    assert minus4["a.b"] == 2 and minus4["b.b"] == -4
    # On P^2 (T = 0, a = -3) without adjoint hypers: only b = 1 (5 x 16, 7 x 10) and b = 2 (8 x 16, 10 x 10).
    assert s.so10_t0_solutions(20) == [{"b": 1, "n16": 5, "n10": 7}, {"b": 2, "n16": 8, "n10": 10}]
    # Known P^2 spectra: 5 x 16 + 7 x 10 on a line, 8 x 16 + 10 x 10 on a conic. Adjoint hypers add b = 3, 4.
    with_adj = s.so10_t0_solutions(20, max_adjoints=6)
    assert {"b": 3, "n16": 9, "n10": 9, "adjoints": 1} in with_adj
    assert {"b": 4, "n16": 8, "n10": 4, "adjoints": 3} in with_adj
    assert all(r["n10"] > 0 for r in with_adj)  # on T = 0, every solution still has 10s
    for n in range(-4, 3):
        row = s.so10_on_curve(n)
        assert row["consistent"] and row["n10"] == row["n16"] + 2  # the quartic condition


def test_bpr6d_premise_is_a_chirality_convention():
    mt = s.mt_quantities(s.minimal_fields(3))
    assert mt["minus_a_dot_btilde"] == 24  # = (1/6) * 16 * 9: Park-Taylor with b~ = 2 b_X
    assert mt["premise_holds_in_hyperino_convention"]
    assert mt["charge_gcd"] == 3 and not mt["massless_charges_generate_lattice"]
    swapped = s.mt_quantities([(-1, "16", 3), (1, "16", 0)])
    assert swapped["minus_a_dot_btilde"] == -24 and not swapped["premise_holds_in_hyperino_convention"]


def test_vectorlike_charge_one_pair_resolves_it_without_changing_anything_else():
    ext = s.vectorlike_charge_one_extension()
    assert ext["I8_unchanged"] and ext["quantization_pass"]
    assert ext["families_at_unit_flux"] == 3
    assert ext["net_singlet_chirality"] == 0
    assert ext["mt"]["massless_charges_generate_lattice"]


def test_supersymmetric_examples_and_gcd_reduction():
    ex = s.susy_examples()
    assert ex["three_net"]["all_ok"] and ex["three_net"]["charge_gcd"] == 3
    assert ex["rescaled"]["all_ok"] and ex["rescaled"]["charge_gcd"] == 1
    # 3 | n_gen is allowed with gcd 1: three net 16s per unit flux and charge-1 singlets.
    assert ex["gcd_one_three_net"]["all_ok"] and ex["gcd_one_three_net"]["charge_gcd"] == 1
    assert ex["gcd_one_three_net"]["net_16_per_unit_flux"] == 3
    # Negative neutral-hyper counts are rejected.
    assert not s.so10_u1_t0_check(1, [3, 3, 3, -3, -3], [0] * 7, [6] * 70, -1)["all_ok"]
    # Independent recomputation for the rescaled spectrum: beta = 10.
    q16, s1 = [1, 1, 1, -1, -1], [2] * 70
    beta = Fraction(2 * (2 * sum(q * q for q in q16)), 2 * 1)  # b.b~ = lambda sum A_R q^2, lambda = 2, b = 1
    assert beta == 10
    assert Fraction(-3 * 2 * beta) == Fraction(-(16 * sum(q * q for q in q16) + sum(x * x for x in s1)), 6)
    assert 4 * beta * beta == Fraction(16 * sum(q ** 4 for q in q16) + sum(x ** 4 for x in s1), 3)
    red = s.gcd_reduction_always_consistent(1, [3, 3, 3, -3, -3], [0] * 7, [6] * 70, 99)
    assert red["applicable"] and red["reduced_all_ok"]


def test_gcd_reduction_is_always_consistent_on_t0():
    lemma = s.reduction_lemma()
    assert lemma["gcd_one_normalization_always_consistent_on_T0"]
    # Direct scan at b = 2 (8 x 16, 10 x 10): every anomaly-free spectrum with charge gcd > 1 found by
    # a two-charge singlet solver reduces consistently.
    import itertools

    def singlets_for(S2, S4, cap):
        for a in range(1, 13):
            for b in range(a + 1, 13):
                det = a * a * b ** 4 - b * b * a ** 4
                na = Fraction(S2 * b ** 4 - S4 * b * b, det)
                nb = Fraction(a * a * S4 - a ** 4 * S2, det)
                if na.denominator == 1 and nb.denominator == 1 and na >= 0 and nb >= 0 and na + nb <= cap:
                    return [a] * int(na) + [b] * int(nb)
        return None

    checked = 0
    for q16 in itertools.combinations_with_replacement([-4, -2, 0, 2, 4], 8):
        p10 = [0] * 10
        beta = sum(q * q for q in q16)
        S2 = 36 * beta - 16 * sum(q * q for q in q16)
        S4 = 12 * beta * beta - 16 * sum(q ** 4 for q in q16)
        if S2 <= 0 or S4 <= 0:
            continue
        sing = singlets_for(S2, S4, 91)
        if sing is None:
            continue
        neutral = 319 - 128 - 100 - len(sing)
        full = s.so10_u1_t0_check(2, list(q16), p10, sing, neutral)
        if full["all_ok"] and full["charge_gcd"] > 1:
            assert s.gcd_reduction_always_consistent(2, list(q16), p10, sing, neutral)["reduced_all_ok"]
            checked += 1
    assert checked > 0


def test_t1_supersymmetric_theory_can_force_charges_into_3z():
    # T = 1 (lattice U, a = (-2, -2)) is the supersymmetric counterpart of one non-chiral 2-form.
    three = s.t1_pure_u1(3)
    assert three["consistent"] and three["btilde"] == ["72", "24"] and three["minus_a_dot_btilde"] == "192"
    # Independent check of the equations with the U pairing.
    x, y = 72, 24
    assert -2 * x - 2 * y == -Fraction(128 * 9, 6) and 2 * x * y == Fraction(128 * 81, 3)
    assert 128 + 117 - 1 + 29 == 273
    one = s.t1_pure_u1(1)
    assert not one["consistent"]  # a.b~ = -128/6 is not an integer


def test_t0_integrality_lemma():
    assert s.t0_integrality_lemma()["btilde_integral_whenever_3btilde2_integral"]


def test_report_is_strict_json():
    report = s.demonstration_report()
    text = json.dumps(report, default=str, allow_nan=False)
    assert json.loads(text)["empirical_validation"] is False
    assert report["limitations"] == s.LIMITATIONS
