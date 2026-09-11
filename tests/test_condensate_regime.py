"""Lock the condensate-regime results: theorem, metastability, RJ curve, T*.

Guards all four findings and the honesty boundary (the flagged assumption and
the undetermined substrate temperature stay on record).
"""
import numpy as np
import pytest

from bpr.condensate_regime import (
    energy,
    gradient_flow,
    condensate_fraction,
    ground_state_check,
    rj_condensation_curve,
    crossover_energy,
    t_star_ring,
    t_star_sphere,
    regime_verdict,
    report,
)


# --- (1) ground-state theorem ------------------------------------------------

def test_uniform_state_is_the_ground_state():
    """Winding-free flow converges to the uniform condensate exactly (n0 = 1,
    H = H(uniform)); and the uniform state beats random states."""
    gs = ground_state_check()
    assert gs["windingfree_n0"] > 0.9999
    assert gs["windingfree_H"] == pytest.approx(gs["H_uniform"], abs=1e-6)
    rng = np.random.default_rng(1)
    p = 101
    uni = np.ones(p, complex)
    for _ in range(20):
        z = rng.normal(size=p) + 1j * rng.normal(size=p)
        z *= np.sqrt(p / np.sum(np.abs(z) ** 2))
        assert energy(z) > energy(uni)


def test_jensen_amgm_equalities():
    """The analytic ingredients: at fixed norm, the uniform state minimizes
    both the hopping bound and the quartic (spot numerical check)."""
    p = 64
    uni = np.ones(p)
    rng = np.random.default_rng(2)
    z = np.abs(rng.normal(size=p)) + 0.1
    z *= np.sqrt(p / np.sum(z ** 2))
    assert np.sum(z ** 4) >= np.sum(uni ** 4) - 1e-12          # Jensen
    hop_z = np.sum(z * np.roll(z, 1))
    assert hop_z <= p + 1e-12                                   # AM-GM bound


# --- (2) winding metastability ----------------------------------------------

def test_random_phase_flow_lands_in_winding_sector_above_uniform():
    gs = ground_state_check()
    assert gs["wound_dominant_winding"] != 0
    assert gs["wound_above_uniform"] is True


# --- (3) Rayleigh-Jeans curve ------------------------------------------------

def test_rj_curve_is_monotone_and_spans():
    curve = rj_condensation_curve(101)
    fr = [r["condensate_fraction"] for r in curve]
    en = [r["E_over_N"] for r in curve]
    # energy increases along the parametrization, fraction decreases
    assert all(en[i] <= en[i + 1] + 1e-12 for i in range(len(en) - 1))
    assert all(fr[i] >= fr[i + 1] - 1e-12 for i in range(len(fr) - 1))
    assert fr[0] > 0.99          # deep condensate at E -> 0
    assert fr[-1] < 0.05         # no condensate at high E


def test_crossover_energy_small():
    """The 50% crossover sits at low E/N (~0.07 for p=101, C=1) — condensation
    is a LOW-energy phenomenon, matching the dynamical result."""
    ce = crossover_energy()
    assert 0.02 < ce < 0.2


# --- (4) characteristic temperatures ----------------------------------------

def test_ring_t_star_scales_as_12_over_p():
    for p in (101, 401, 1009):
        assert t_star_ring(p) * p == pytest.approx(12.0, rel=1e-3)


def test_sphere_t_star_scales_as_one_over_log_p():
    for p in (10 ** 4, 104761, 10 ** 6):
        assert t_star_sphere(p) * np.log(p) == pytest.approx(1.0, rel=0.02)


def test_bpr_sphere_condition_is_mild():
    """At BPR's p the 2D coherence condition is T below ~8.6% of the natural
    scale — mild, not fine-tuned. The 1D toy would be ~1e-4 (fragile)."""
    ts2 = t_star_sphere(104761)
    assert 0.07 < ts2 < 0.10
    assert t_star_ring(4001) < 0.005      # 1D fragility (scaling demo)


# --- honesty guards ----------------------------------------------------------

def test_verdict_flags_assumption_and_open_temperature():
    v = regime_verdict()
    assert "EXACT" in v["vacuum_is_condensate"]
    assert "classical" in v["vacuum_is_condensate"]
    assert "quantum vacuum NOT computed" in v["vacuum_is_condensate"]
    assert "winding" in v["relaxation_caveat"]
    assert "temperature" in v["undetermined"]
    assert "assumption" in v["undetermined"]
    assert v["status"].startswith("CONDITIONAL")


def test_report_is_conditional_not_guaranteed():
    """2026-09-10 review: the classical fixed-norm theorem does not establish
    the quantum vacuum, and T* is conditional on assumed dispersion, cutoff
    and density. The report must not overclaim."""
    txt = report()
    assert "NET: CONDITIONAL" in txt
    assert "GUARANTEED" not in txt
    assert "ROBUST" not in txt
    assert "not established" in txt
    assert "quantum vacuum" in txt.lower()
    assert "UNDETERMINED" in txt
