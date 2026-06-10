"""Tests for the c=1 compact-boson heavy winding spectrum."""
import pytest
from bpr.gut_winding_spectrum import (
    winding_mode_mass_squared,
    enumerate_light_modes,
    gut_winding_report,
)


def test_lightest_mode_is_momentum():
    """At R^2=3, lightest mode is pure momentum (1,0) with M2 = 1/3."""
    modes = enumerate_light_modes()
    assert modes[0]["M2"] == pytest.approx(1.0 / 3.0)


def test_mode_mass_formula():
    # (n=1, w=0) at R^2=3: M2 = 1/3
    assert winding_mode_mass_squared(1, 0) == pytest.approx(1.0 / 3.0)
    # (n=0, w=1): M2 = (R/2)^2 = 3/4
    assert winding_mode_mass_squared(0, 1) == pytest.approx(0.75)


def test_modes_sorted_and_nontrivial():
    modes = enumerate_light_modes()
    m2 = [m["M2"] for m in modes]
    assert m2 == sorted(m2)
    assert all(x > 0 for x in m2)  # vacuum excluded


def test_threshold_right_order_of_magnitude():
    """Threshold magnitude should be comparable to the ~0.19 residual."""
    r = gut_winding_report()
    assert abs(r["threshold_delta_inv_alpha_b1"]) > r["residual_to_close_approx"]


def test_status_flags_missing_rep_assignment():
    r = gut_winding_report()
    assert "rep ASSIGNMENT" in r["status"]
    assert "NOT done" in r["status"]
