"""Lock in the phason-coupling derivation attempt: what it pins and what it can't.

These guard the honest boundary of the derivation — the FORM and ORDER are
derived, the exact exponent is not — so a later edit can't quietly upgrade the
partial result into a clean (1/p)^7 prediction.
"""
import pytest

from bpr.phason_coupling_derivation import (
    d_perp,
    exponent_corners,
    epsilon_from_exponent,
    required_exponent,
    class_bracket,
    derivation_report,
)


def test_required_exponent_is_about_seven():
    """ε_required corresponds to k_req ≈ 7.0 powers of 1/p."""
    assert required_exponent() == pytest.approx(7.0, abs=0.2)


def test_exponent_form_is_small_integer():
    """Derived FORM: k = c·(d_⊥ − τ) is a small integer, never a 40-order span."""
    for n in (5, 8, 9, 12):
        for corner in exponent_corners(n):
            assert isinstance(corner.k, int)
            assert 0 < corner.k <= 8


def test_nine_fold_energy_bracket_contains_seven():
    """The propulsion-relevant 9-fold class: energy corners (6,8) bracket k_req."""
    b = class_bracket(9)
    assert b["d_perp"] == 4
    assert b["energy_bracket"] == (6, 8)
    assert b["required_inside_energy_bracket"] is True


def test_seven_is_odd_between_even_corners():
    """KEY HONEST FINDING: clean energy-convention counting gives EVEN k (6, 8);
    k_req ≈ 7 is odd and sits between them, so (1/p)^7 is NOT reproduced exactly
    by clean cut-and-project counting."""
    b = class_bracket(9)
    even_corners = b["energy_bracket"]
    assert all(k % 2 == 0 for k in even_corners)
    # k_req is not within 0.05 of an even integer → genuinely between corners
    k = b["k_required"]
    assert min(abs(k - 6), abs(k - 8)) > 0.5


def test_only_nine_fold_brackets_required_exponent():
    """STRUCTURAL DISCRIMINATOR: among the allowed classes, only the rank-6
    (9-fold) class lands in the marginal regime that brackets k_req. The rank-4
    classes (5/8/12-fold) give energy-bracket (2,4) — they do NOT bracket 7."""
    bracketing = [n for n in (5, 8, 9, 12)
                  if class_bracket(n)["required_inside_energy_bracket"]]
    assert bracketing == [9]


def test_rank4_classes_have_smaller_exponents():
    """Rank-4 classes (d_⊥=2) give energy corners (2,4), well below 7."""
    for n in (5, 8, 12):
        b = class_bracket(n)
        assert b["d_perp"] == 2
        assert b["energy_bracket"] == (2, 4)


def test_epsilon_at_seven_matches_required_order():
    """(1/p)^7 reproduces ε_required to within an order of magnitude."""
    eps7 = epsilon_from_exponent(7)
    from bpr.phason_sector import required_coupling_efficiency
    ratio = eps7 / required_coupling_efficiency()
    assert 0.1 < ratio < 10.0


def test_report_states_partial_not_pinned():
    """The report must declare the result PARTIAL and flag the odd-exponent
    obstruction — never claim a clean (1/p)^7 prediction."""
    txt = derivation_report()
    assert "PARTIAL" in txt
    assert "NOT pinned" in txt or "not pinned" in txt
    assert "ODD" in txt or "odd" in txt
    assert "coincidence" in txt.lower()
