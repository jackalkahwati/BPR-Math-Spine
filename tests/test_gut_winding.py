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


# ---------------------------------------------------------------------------
# SU(5) X,Y rep assignment + threshold closure
# ---------------------------------------------------------------------------

def test_su5_xy_uses_12_states():
    """X,Y multiplet (3,2,±5/6) has exactly 12 states."""
    from bpr.gut_winding_spectrum import gut_threshold_su5_assigned
    r = gut_threshold_su5_assigned()
    assert r["n_xy_states_assigned"] == 12


def test_su5_xy_asymmetric_pattern():
    """|Δ(1/α₁)| > |Δ(1/α₂)| > |Δ(1/α₃)| -- the GUT-asymmetry signature."""
    from bpr.gut_winding_spectrum import gut_threshold_su5_assigned
    r = gut_threshold_su5_assigned()
    d = r["delta_inv_alpha"]
    assert abs(d["U1_Y_GUT_norm"]) > abs(d["SU2_L"]) > abs(d["SU3_c"])


def test_su5_xy_closes_qualitatively():
    """Magnitude reaches the residual scale (1.5% in 1/alpha units)."""
    from bpr.gut_winding_spectrum import gut_threshold_su5_assigned
    r = gut_threshold_su5_assigned()
    assert r["closes_qualitatively"]
    # max contribution comfortably exceeds the residual
    assert r["max_abs_contribution"] > r["residual_to_close_approx"]


def test_su5_status_flags_ansatz():
    """Status string is explicit about being ANSATZ-BASED."""
    from bpr.gut_winding_spectrum import gut_threshold_su5_assigned
    r = gut_threshold_su5_assigned()
    assert "ANSATZ" in r["status"]
    assert "boundary CFT" in r["status"]
