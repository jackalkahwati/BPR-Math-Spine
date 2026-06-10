"""Tests for substrate bounce cosmology."""
import pytest
from bpr.bounce_cosmology import BounceCosmology


def test_sub_planckian_bounce():
    """a_boundary > Planck length => bounce below Planck density."""
    r = BounceCosmology(p=104761).report()
    assert r["a_boundary_over_planck"] > 1.0
    assert r["rho_crit_over_planck"] < 1.0


def test_bounce_density_scales_with_p():
    """Larger p => larger a_boundary => lower bounce density."""
    lo = BounceCosmology(p=10007).report()
    hi = BounceCosmology(p=1000003).report()
    assert hi["rho_crit_kg_m3"] < lo["rho_crit_kg_m3"]


def test_friedmann_vanishes_at_bounce():
    bc = BounceCosmology()
    assert bc.friedmann_correction(1.0) == pytest.approx(0.0)
    assert bc.friedmann_correction(0.0) == pytest.approx(1.0)


def test_gw_cutoff_positive():
    r = BounceCosmology().report()
    assert r["gw_cutoff_frequency_Hz"] > 0
