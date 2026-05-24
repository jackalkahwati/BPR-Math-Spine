"""Tests for the Casimir constraint on the δ=2 amplitude (speculative branch)."""
import math
import pytest

from bpr.casimir_constraint import (
    DATA, R_F_NM, CasimirPoint, epsilon_bound, best_bound,
    required_precision_to_detect, lift_required_epsilon,
)


def test_epsilon_bound_formula():
    # ε < p·(R/R_f)^2
    p = CasimirPoint(100, 0.01, "x")
    assert epsilon_bound(p) == pytest.approx(0.01 * (100 / R_F_NM) ** 2)


def test_smaller_separation_gives_tighter_bound():
    # the deviation is amplified at small R, so small-R data bounds ε harder
    near = epsilon_bound(CasimirPoint(62, 0.0175, "x"))
    far = epsilon_bound(CasimirPoint(750, 0.005, "x"))
    assert near < far


def test_best_bound_is_order_1e4_to_1e5():
    b, _ = best_bound()
    assert 1e-6 < b < 1e-3       # current data bounds ε to ~1e-4–1e-5


def test_open_window_is_enormous():
    b, _ = best_bound()
    lift = lift_required_epsilon()
    assert lift < b                                   # lift is allowed
    window = math.log10(b / lift)
    assert window > 25                                # ~30-order open window


def test_lift_floor_is_unreachable_by_casimir():
    # detecting the lift-relevant epsilon needs absurd precision
    lift = lift_required_epsilon()
    need = required_precision_to_detect(lift, 62)
    assert need < 1e-25                               # ~1e-33 — impossible
