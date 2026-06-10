"""Tests for the phason lift-budget decisive verdict."""
import pytest
from bpr.phason_sector import phason_defect_lift_budget


def test_lift_budget_undecidable_for_car():
    r = phason_defect_lift_budget(mass_kg=1500.0)
    assert r["verdict"] == "UNDECIDABLE"
    assert r["orders_of_magnitude_gap"] > 20  # ~31 orders


def test_lift_budget_excluded_for_absurd_mass():
    """A mass so large it needs eps > experimental bound must be EXCLUDED."""
    # eps scales linearly with mass; bound ~5e-5, base ~6e-36/1500kg.
    # need mass ~ 1500 * (5e-5 / 6e-36) ~ 1e34 kg
    r = phason_defect_lift_budget(mass_kg=1e35)
    assert r["verdict"] == "EXCLUDED"


def test_lift_budget_keys():
    r = phason_defect_lift_budget()
    assert set(r) >= {"eps_required", "eps_bound", "verdict", "orders_of_magnitude_gap"}
