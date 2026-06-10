"""Tests for two-gap superconductivity (MgB2)."""
import pytest
from bpr.superconductivity import TwoGapSuperconductor, mgb2_two_gap_tc


def test_mgb2_tc_within_1pct():
    r = mgb2_two_gap_tc()
    assert abs(r["Tc_residual_pct"]) < 1.0


def test_mgb2_combined_fermi_energy():
    sc = TwoGapSuperconductor()
    assert sc.E_fermi_combined_eV == pytest.approx(16.4)


def test_coordination_hypotheses_rejected():
    """The 2/z and 1/(2z) factors must overshoot (documents the honest result)."""
    r = mgb2_two_gap_tc()
    assert r["rejected_hypotheses"]["f=1+2/z"]["Tc_K"] > 60.0
    assert r["rejected_hypotheses"]["f=1+1/(2z)"]["Tc_K"] > 44.0


def test_status_says_framework():
    r = mgb2_two_gap_tc()
    assert "FRAMEWORK" in r["status"]
