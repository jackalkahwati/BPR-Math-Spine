"""Regression tests for the 2026-09-10 flavor-foundation repair.

Checks the conditional mathematics in bpr.flavor_foundations and that the
retained phenomenological numbers are unchanged while their status labels
now say CONJECTURAL / EMPIRICAL_INPUT rather than DERIVED.
"""
from fractions import Fraction

import numpy as np
import pytest

from bpr.flavor_foundations import (
    compact_boson_weights,
    sphere_line_zero_modes,
    su_bundle_twisted_index,
)


class TestCompactBoson:
    def test_conformal_spin_is_integer_mn(self):
        for m in range(-3, 4):
            for n in range(-3, 4):
                h, hb = compact_boson_weights(m, n)
                assert h - hb == m * n
                assert h >= 0 and hb >= 0

    def test_r2_3_table(self):
        assert compact_boson_weights(1, 0) == (Fraction(1, 12), Fraction(1, 12))
        assert compact_boson_weights(0, 1) == (Fraction(3, 4), Fraction(3, 4))
        assert compact_boson_weights(1, 1) == (Fraction(4, 3), Fraction(1, 3))
        assert compact_boson_weights(2, 2) == (Fraction(16, 3), Fraction(4, 3))

    def test_no_half_integer_spin_for_any_radius(self):
        for r2 in (Fraction(1), Fraction(2), Fraction(3), Fraction(7, 2)):
            for m in range(-2, 3):
                for n in range(-2, 3):
                    h, hb = compact_boson_weights(m, n, r2)
                    assert (h - hb).denominator == 1

    def test_old_arithmetic_slip(self):
        assert Fraction(10, 3) + Fraction(1, 16) == Fraction(163, 48)

    def test_rejects_bad_inputs(self):
        with pytest.raises(TypeError):
            compact_boson_weights(1.0, 1)
        with pytest.raises(ValueError):
            compact_boson_weights(1, 1, Fraction(0))


class TestDiracZeroModes:
    def test_index_equals_degree(self):
        for q in range(-5, 6):
            zm = sphere_line_zero_modes(q)
            assert zm.index == q
            assert zm.positive * zm.negative == 0

    def test_degree_three_gives_three_but_so_does_four(self):
        assert sphere_line_zero_modes(3).positive == 3
        assert sphere_line_zero_modes(4).positive == 4

    def test_untwisted_has_no_zero_modes(self):
        zm = sphere_line_zero_modes(0)
        assert (zm.positive, zm.negative) == (0, 0)


class TestSUBundleIndex:
    def test_pure_su_index_vanishes(self):
        for r in (2, 3, 5):
            assert su_bundle_twisted_index(r) == 0

    def test_line_twist_scales_with_rank(self):
        assert su_bundle_twisted_index(3, 1) == 3
        assert su_bundle_twisted_index(3, 3) == 9

    def test_rank_one_rejected(self):
        with pytest.raises(ValueError):
            su_bundle_twisted_index(1, 3)


class TestRetainedPhenomenologyStatus:
    def test_legacy_numbers_unchanged(self):
        from bpr.qcd_flavor import derive_l_modes
        m = derive_l_modes(6, 3)
        assert m["l_up"] == (1, 24, 283)
        assert m["l_down"] == (1, 4, 30)
        assert m["l_lep"][0] == 1 and m["l_lep"][2] == 59
        assert np.isclose(m["l_lep"][1] ** 2, 210)

    def test_status_labels_are_not_derived(self):
        from bpr.qcd_flavor import derive_l_modes
        m = derive_l_modes(6, 3)
        assert set(m["derivation_status"].values()) == {"CONJECTURAL"}
        assert m["input_status"]["n_gen"] == "EMPIRICAL_INPUT"

    def test_number_of_generations_legacy_ansatz(self):
        from bpr.neutrino import number_of_generations
        assert number_of_generations("sphere") == 3
        assert "ansatz" in number_of_generations.__doc__.lower()

    def test_first_principles_flags_input(self):
        import bpr.first_principles as fp
        src = open(fp.__file__).read()
        assert 'P5.10_number_of_generations_status"] = "EMPIRICAL_INPUT"' in src
