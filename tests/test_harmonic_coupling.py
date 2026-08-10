"""Lock in the coupled-vs-uncoupled result: coherence is emergent from coupling.

Guards both the physics (the transition and robustness are real) and the honesty
(the module must not claim BPR-specificity or artifact evidence).
"""
import numpy as np
import pytest

from bpr.harmonic_coupling import (
    order_parameter,
    simulate,
    critical_coupling,
    coupling_sweep,
    perturbation_recovery,
    report,
)


def test_order_parameter_bounds():
    """r ∈ [0,1]; identical phases → r=1, uniform phases → r≈0."""
    assert order_parameter(np.zeros(100)) == pytest.approx(1.0)
    assert order_parameter(np.linspace(0, 2 * np.pi, 500, endpoint=False)) < 0.05


def test_uncoupled_field_is_incoherent():
    """K=0 (linear superposition) stays incoherent: r ~ 1/√N, small."""
    rng = np.random.default_rng(0)
    N = 300
    omega = rng.normal(0, 1, N)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    r = simulate(0.0, omega, theta0)["r_final"]
    assert r < 0.15


def test_strong_coupling_produces_coherence():
    """K well above K_c drives near-complete phase-locking (r → 1)."""
    rng = np.random.default_rng(0)
    N = 300
    omega = rng.normal(0, 1, N)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    r = simulate(3.0, omega, theta0)["r_final"]
    assert r > 0.85


def test_transition_near_theoretical_kc():
    """Coherence turns on near the Kuramoto K_c = σ√(8/π) ≈ 1.60, not before."""
    sweep = coupling_sweep()
    assert critical_coupling(1.0) == pytest.approx(1.5958, abs=1e-3)
    below = [row["r"] for row in sweep["scan"] if row["K"] <= 1.25]
    above = [row["r"] for row in sweep["scan"] if row["K"] >= 2.25]
    assert max(below) < 0.2       # incoherent below K_c
    assert min(above) > 0.6       # coherent above K_c


def test_coupled_field_recovers_from_perturbation():
    """A coupled field has a coherent attractor: kick it, it returns."""
    rec = perturbation_recovery(K=3.0)
    assert rec["r_just_after"] < 0.5 * rec["r_before"]   # kick really disrupted it
    assert rec["r_recovered"] > 0.85                      # and it came back
    assert rec["recovers"] is True


def test_linear_field_does_not_recover():
    """K=0 has no restoring force — no attractor, no recovery."""
    rec = perturbation_recovery(K=0.0)
    assert rec["recovers"] is False


def test_report_disclaims_bpr_and_artifacts():
    """HONESTY GUARD: the result is generic nonlinear dynamics, not BPR-specific
    and not artifact evidence; the parameter cost must be stated."""
    txt = report()
    assert "not BPR-specific" in txt
    assert "not" in txt.lower() and "evidence for any artifact" in txt
    assert "adds O(N²) parameters" in txt or "adds O(N^2) parameters" in txt
