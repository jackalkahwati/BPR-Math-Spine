"""Tests for the BPR black-hole Page curve (unitarity demonstration)."""
import numpy as np
import pytest
from bpr.page_curve import page_entropy, BlackHolePageCurve


def test_page_entropy_symmetric():
    """S_A = S_B: entropy is symmetric under swapping subsystems."""
    assert page_entropy(3.0, 7.0) == pytest.approx(page_entropy(7.0, 3.0))


def test_page_curve_is_unitary():
    r = BlackHolePageCurve(n_cells=100).curve()
    assert r["is_unitary"]
    assert r["initial_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert r["final_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_page_time_at_half():
    r = BlackHolePageCurve(n_cells=100).curve()
    assert r["page_time_fraction"] == pytest.approx(0.5, abs=0.02)


def test_curve_rises_then_falls():
    S = BlackHolePageCurve(n_cells=80).curve()["radiation_entropy_nats"]
    mid = len(S) // 2
    assert np.all(np.diff(S[:mid]) > 0)   # strictly rising before Page time
    assert np.all(np.diff(S[mid + 1:]) < 0)  # strictly falling after


def test_peak_entropy_scales_with_p():
    """Larger p => more states per cell => higher peak entropy."""
    lo = BlackHolePageCurve(n_cells=50, p=101).curve()["peak_entropy_nats"]
    hi = BlackHolePageCurve(n_cells=50, p=104761).curve()["peak_entropy_nats"]
    assert hi > lo
