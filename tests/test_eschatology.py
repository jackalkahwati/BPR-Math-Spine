"""
Tests for Theory XXII: Invariant Structure, Boundary Dynamics, and Symbolic Meaning.

Tests cover all core mathematical constructs from the eschatology module:
    - Stain dynamics (Eq 6)
    - Heart gain function and coherence evolution (Eq 12-13)
    - Judgment functional (Def 5.1)
    - Deception classifier (Def 6.1, Eq 14)
    - Collapse-reset dynamics (Eq 15)
    - Death trichotomy (Thm 8.1)
    - Symbolic projection operator (Def 3.1)
    - Cross-traditional mapping (Table 1)

Run with:  pytest tests/test_eschatology.py -v
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Stain Dynamics  (Eq 6)
# ═══════════════════════════════════════════════════════════════════════════

class TestStainDynamics:
    """ds/dt = alpha * u_minus * (1 - s) - beta * u_plus * s - gamma * s"""

    def test_initial_stain_in_range(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(s0=0.0)
        assert sd.s0 == 0.0
        sd = StainDynamics(s0=0.5)
        assert sd.s0 == 0.5

    def test_initial_stain_out_of_range_raises(self):
        from bpr.eschatology import StainDynamics
        with pytest.raises(ValueError):
            StainDynamics(s0=-0.1)
        with pytest.raises(ValueError):
            StainDynamics(s0=1.5)

    def test_ds_dt_zero_at_equilibrium(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=1.0, beta=0.5, gamma=0.01)
        s_star = sd.steady_state(u_plus=1.0, u_minus=1.0)
        rate = sd.ds_dt(s_star, u_plus=1.0, u_minus=1.0)
        assert abs(rate) < 1e-10, "Rate should vanish at steady state"

    def test_stain_increases_under_pure_noise(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=1.0, beta=0.5, gamma=0.01, s0=0.0)
        rate = sd.ds_dt(0.0, u_plus=0.0, u_minus=1.0)
        assert rate > 0, "Stain should increase when only noise is present"

    def test_stain_decreases_under_pure_restoration(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=1.0, beta=1.0, gamma=0.1, s0=0.8)
        rate = sd.ds_dt(0.8, u_plus=1.0, u_minus=0.0)
        assert rate < 0, "Stain should decrease when only restoration is present"

    def test_steady_state_no_noise_is_zero(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=1.0, beta=0.5, gamma=0.01)
        s_star = sd.steady_state(u_plus=1.0, u_minus=0.0)
        assert s_star == pytest.approx(0.0, abs=1e-12)

    def test_steady_state_pure_noise(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=1.0, beta=0.5, gamma=0.01)
        s_star = sd.steady_state(u_plus=0.0, u_minus=1.0)
        # s* = alpha * u_minus / (alpha * u_minus + gamma)
        expected = 1.0 / (1.0 + 0.01)
        assert s_star == pytest.approx(expected, rel=1e-6)

    def test_evolve_returns_valid_shape(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(s0=0.1)
        t, s = sd.evolve(
            t_span=(0, 10),
            u_plus=lambda t: 0.5,
            u_minus=lambda t: 0.5,
            n_points=100,
        )
        assert len(t) == 100
        assert len(s) == 100
        assert np.all(s >= 0.0) and np.all(s <= 1.0)

    def test_evolve_converges_to_steady_state(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=1.0, beta=0.5, gamma=0.1, s0=0.0)
        u_p, u_m = 1.0, 0.5
        t, s = sd.evolve(
            t_span=(0, 200),
            u_plus=lambda t: u_p,
            u_minus=lambda t: u_m,
            n_points=500,
        )
        s_star = sd.steady_state(u_p, u_m)
        assert s[-1] == pytest.approx(s_star, abs=0.02)

    def test_stain_bounded_0_1(self):
        from bpr.eschatology import StainDynamics
        sd = StainDynamics(alpha=5.0, beta=0.1, gamma=0.01, s0=0.0)
        t, s = sd.evolve(
            t_span=(0, 50),
            u_plus=lambda t: 0.0,
            u_minus=lambda t: 10.0,
            n_points=200,
        )
        assert np.all(s >= 0.0) and np.all(s <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Heart Gain Function  (Eq 12-13)
# ═══════════════════════════════════════════════════════════════════════════

class TestHeartGainFunction:
    """G(s) = exp(-kappa_s * s - 0.5 * sigma_s(s)^2)"""

    def test_gain_at_zero_stain_is_one(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction()
        assert hg.G(0.0) == pytest.approx(1.0)

    def test_gain_decreases_with_stain(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction()
        assert hg.G(0.5) < hg.G(0.0)
        assert hg.G(1.0) < hg.G(0.5)

    def test_gain_positive_for_all_stain(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction()
        for s in np.linspace(0, 1, 50):
            assert hg.G(s) > 0

    def test_asymptotic_coherence_at_zero_stain(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction(K_bar=1.0, nu=0.1)
        K_star = hg.asymptotic_coherence(0.0)
        # G(0) = 1, so K* = 1.0 / (1.0 + 0.1) = 10/11
        expected = 1.0 / (1.0 + 0.1)
        assert K_star == pytest.approx(expected, rel=1e-6)

    def test_asymptotic_coherence_at_max_stain_is_low(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction(K_bar=1.0, nu=0.1, kappa_s=5.0)
        K_star = hg.asymptotic_coherence(1.0)
        assert K_star < 0.1, "High stain should drive asymptotic coherence near zero"

    def test_dK_dt_positive_when_K_low_s_low(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction()
        rate = hg.dK_dt(K=0.0, s=0.0)
        assert rate > 0, "Coherence should increase from zero when stain is zero"

    def test_dK_dt_negative_when_K_high_s_high(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction(kappa_s=5.0, nu=1.0)
        rate = hg.dK_dt(K=0.9, s=0.99)
        assert rate < 0, "Coherence should decrease when stain is high"

    def test_evolve_coherence_shape(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction()
        t, K = hg.evolve_coherence(
            t_span=(0, 20),
            K0=0.5,
            s_trajectory=lambda t: 0.3,
            n_points=100,
        )
        assert len(t) == 100
        assert len(K) == 100
        assert np.all(K >= 0.0) and np.all(K <= 1.0)

    def test_evolve_coherence_converges(self):
        from bpr.eschatology import HeartGainFunction
        hg = HeartGainFunction(K_bar=1.0, nu=0.1, kappa_s=2.0)
        s_fixed = 0.3
        t, K = hg.evolve_coherence(
            t_span=(0, 200),
            K0=0.1,
            s_trajectory=lambda t: s_fixed,
            n_points=500,
        )
        K_star = hg.asymptotic_coherence(s_fixed)
        assert K[-1] == pytest.approx(K_star, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# Judgment Functional  (Def 5.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestJudgmentFunctional:
    """J(H_i) = lim_{t->inf} K(S_t(H_i))"""

    def test_high_restoration_gives_high_judgment(self):
        from bpr.eschatology import JudgmentFunctional
        jf = JudgmentFunctional()
        result = jf.evaluate(
            u_plus=lambda t: 5.0,
            u_minus=lambda t: 0.1,
            K0=0.5,
            t_final=100.0,
        )
        assert result["J"] > 0.7, "Strong restoration should yield high judgment"

    def test_high_noise_gives_low_judgment(self):
        from bpr.eschatology import JudgmentFunctional, HeartGainFunction, StainDynamics
        hg = HeartGainFunction(kappa_s=5.0)  # stronger stain coupling
        sd = StainDynamics(alpha=2.0, beta=0.1, gamma=0.01)
        jf = JudgmentFunctional(heart_gain=hg, stain_dynamics=sd)
        result = jf.evaluate(
            u_plus=lambda t: 0.1,
            u_minus=lambda t: 5.0,
            K0=0.5,
            t_final=100.0,
        )
        assert result["J"] < 0.3, "Strong noise should yield low judgment"

    def test_analytic_matches_numeric(self):
        from bpr.eschatology import JudgmentFunctional
        jf = JudgmentFunctional()
        u_p, u_m = 1.0, 0.5
        J_analytic = jf.evaluate_analytic(u_p, u_m)
        result = jf.evaluate(
            u_plus=lambda t: u_p,
            u_minus=lambda t: u_m,
            K0=0.5,
            t_final=200.0,
            n_points=1000,
        )
        assert result["J"] == pytest.approx(J_analytic, abs=0.05)

    def test_result_keys_present(self):
        from bpr.eschatology import JudgmentFunctional
        jf = JudgmentFunctional()
        result = jf.evaluate(
            u_plus=lambda t: 1.0,
            u_minus=lambda t: 1.0,
        )
        assert "J" in result
        assert "K_final" in result
        assert "s_final" in result
        assert "K_star" in result
        assert "s_star" in result

    def test_stain_determines_asymptotic_fate(self):
        """Proposition 5.1: high stain drives K* -> 0."""
        from bpr.eschatology import JudgmentFunctional, HeartGainFunction, StainDynamics
        hg = HeartGainFunction(kappa_s=5.0)
        sd = StainDynamics(alpha=2.0, beta=0.01, gamma=0.001, s0=0.0)
        jf = JudgmentFunctional(heart_gain=hg, stain_dynamics=sd)
        result = jf.evaluate(
            u_plus=lambda t: 0.01,
            u_minus=lambda t: 5.0,
            t_final=100.0,
        )
        assert result["s_final"] > 0.8, "Stain should be high"
        assert result["J"] < 0.15, "High stain should yield near-zero coherence"


# ═══════════════════════════════════════════════════════════════════════════
# Deception Classifier  (Def 6.1, Eq 14)
# ═══════════════════════════════════════════════════════════════════════════

class TestDeceptionClassifier:
    """Deception: K_local > K_c AND K_global < K_c"""

    def test_deceptive_state_detected(self):
        from bpr.eschatology import DeceptionClassifier
        dc = DeceptionClassifier(K_c=0.5)
        assert dc.is_deceptive(K_local=0.8, K_global=0.2) is True

    def test_truthful_state_not_deceptive(self):
        from bpr.eschatology import DeceptionClassifier
        dc = DeceptionClassifier(K_c=0.5)
        assert dc.is_deceptive(K_local=0.8, K_global=0.8) is False

    def test_disordered_state_not_deceptive(self):
        from bpr.eschatology import DeceptionClassifier
        dc = DeceptionClassifier(K_c=0.5)
        assert dc.is_deceptive(K_local=0.3, K_global=0.2) is False

    def test_deception_degree_positive_when_deceptive(self):
        from bpr.eschatology import DeceptionClassifier
        dc = DeceptionClassifier(K_c=0.5)
        d = dc.deception_degree(K_local=0.9, K_global=0.1)
        assert d == pytest.approx(0.8, abs=1e-10)

    def test_deception_degree_zero_when_truthful(self):
        from bpr.eschatology import DeceptionClassifier
        dc = DeceptionClassifier(K_c=0.5)
        d = dc.deception_degree(K_local=0.8, K_global=0.8)
        assert d == 0.0

    def test_kramers_escape_finite(self):
        from bpr.eschatology import DeceptionClassifier
        tau = DeceptionClassifier.kramers_escape_time(delta_V=1.0, epsilon=0.5)
        assert np.isfinite(tau)
        assert tau > 0
        assert tau == pytest.approx(np.exp(2.0))

    def test_kramers_escape_infinite_without_noise(self):
        from bpr.eschatology import DeceptionClassifier
        tau = DeceptionClassifier.kramers_escape_time(delta_V=1.0, epsilon=0.0)
        assert tau == np.inf

    def test_kramers_escape_zero_without_barrier(self):
        from bpr.eschatology import DeceptionClassifier
        tau = DeceptionClassifier.kramers_escape_time(delta_V=0.0, epsilon=0.5)
        assert tau == 0.0

    def test_kramers_escape_increases_with_barrier(self):
        from bpr.eschatology import DeceptionClassifier
        tau1 = DeceptionClassifier.kramers_escape_time(delta_V=1.0, epsilon=0.5)
        tau2 = DeceptionClassifier.kramers_escape_time(delta_V=2.0, epsilon=0.5)
        assert tau2 > tau1

    def test_classify_attractor_landscape(self):
        from bpr.eschatology import DeceptionClassifier
        dc = DeceptionClassifier(K_c=0.5)
        K_local = np.array([0.8, 0.9, 0.2, 0.3])
        K_global = np.array([0.8, 0.2, 0.3, 0.7])
        classes = dc.classify_attractor_landscape(K_local, K_global)
        assert classes["truthful"][0] is np.True_
        assert classes["deceptive"][1] is np.True_
        assert classes["disordered"][2] is np.True_
        assert classes["hidden"][3] is np.True_

    def test_deceptive_attractor_finite_lifetime(self):
        """Theorem 6.1: deceptive attractors have finite lifetime for any epsilon > 0."""
        from bpr.eschatology import DeceptionClassifier
        # Use epsilon values large enough that exp(delta_V/eps) fits in float64
        for eps in [0.01, 0.1, 1.0, 10.0]:
            tau = DeceptionClassifier.kramers_escape_time(delta_V=5.0, epsilon=eps)
            assert np.isfinite(tau), f"Lifetime must be finite for epsilon={eps}"
            assert tau > 0, f"Lifetime must be positive for epsilon={eps}"


# ═══════════════════════════════════════════════════════════════════════════
# Collapse-Reset Dynamics  (Eq 15)
# ═══════════════════════════════════════════════════════════════════════════

class TestCollapseResetDynamics:
    """Q_eff(t+dt) = Q_eff - alpha_Q * Q_eff if Q > Q_c, else Q_0"""

    def test_single_step_degrades(self):
        from bpr.eschatology import CollapseResetDynamics
        cr = CollapseResetDynamics(Q_0=1.0, Q_c=0.1, alpha_Q=0.02)
        Q_next = cr.step(1.0)
        assert Q_next < 1.0
        assert Q_next == pytest.approx(0.98)

    def test_reset_at_threshold(self):
        from bpr.eschatology import CollapseResetDynamics
        cr = CollapseResetDynamics(Q_0=1.0, Q_c=0.1, alpha_Q=0.5)
        # Start at Q_c, next step should go below and reset
        Q_next = cr.step(0.15)
        # 0.15 - 0.5*0.15 = 0.075 < 0.1, so resets to 1.0
        assert Q_next == pytest.approx(1.0)

    def test_evolve_sawtooth_pattern(self):
        from bpr.eschatology import CollapseResetDynamics
        cr = CollapseResetDynamics(Q_0=1.0, Q_c=0.1, alpha_Q=0.02)
        Q = cr.evolve(500)
        assert len(Q) == 501
        # Should show at least one collapse-reset cycle
        resets = cr.collapse_times(500)
        assert len(resets) >= 1, "Should have at least one collapse-reset event"

    def test_quality_bounded_above(self):
        from bpr.eschatology import CollapseResetDynamics
        cr = CollapseResetDynamics(Q_0=1.0, Q_c=0.1, alpha_Q=0.02)
        Q = cr.evolve(1000)
        assert np.max(Q) <= cr.Q_0 + 1e-10

    def test_collapse_period_analytic(self):
        from bpr.eschatology import CollapseResetDynamics
        cr = CollapseResetDynamics(Q_0=1.0, Q_c=0.1, alpha_Q=0.02)
        T_analytic = cr.collapse_period()
        assert T_analytic > 0
        assert np.isfinite(T_analytic)
        # Verify: (1 - 0.02)^T = 0.1 => T = ln(0.1)/ln(0.98)
        expected = np.log(0.1) / np.log(0.98)
        assert T_analytic == pytest.approx(expected, rel=1e-6)

    def test_multiple_collapse_cycles(self):
        from bpr.eschatology import CollapseResetDynamics
        cr = CollapseResetDynamics(Q_0=1.0, Q_c=0.1, alpha_Q=0.05)
        Q = cr.evolve(500)
        resets = cr.collapse_times(500)
        assert len(resets) >= 2, "Should have multiple collapse-reset cycles"


# ═══════════════════════════════════════════════════════════════════════════
# Death Trichotomy  (Theorem 8.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestDeathTrichotomy:
    """W != 0 => exactly three topologically allowed fates."""

    def test_nonzero_winding_has_three_fates(self):
        from bpr.eschatology import DeathTrichotomy, WindingFate
        dt = DeathTrichotomy(W=3)
        fates = dt.allowed_fates()
        assert len(fates) == 3
        assert WindingFate.DISSOLUTION in fates
        assert WindingFate.MIGRATION in fates
        assert WindingFate.REINCORPORATION in fates

    def test_zero_winding_only_dissolution(self):
        from bpr.eschatology import DeathTrichotomy, WindingFate
        dt = DeathTrichotomy(W=0)
        fates = dt.allowed_fates()
        assert len(fates) == 1
        assert fates[0] == WindingFate.DISSOLUTION

    def test_dissolution_requires_anti_winding(self):
        from bpr.eschatology import DeathTrichotomy
        dt = DeathTrichotomy(W=5)
        assert dt.dissolution_requires() == -5

    def test_classify_dissolution(self):
        from bpr.eschatology import DeathTrichotomy, WindingFate
        dt = DeathTrichotomy(W=3)
        fate = dt.classify_fate(W_final=0, substrate_coupled=False, mode_transferred=False)
        assert fate == WindingFate.DISSOLUTION

    def test_classify_migration(self):
        from bpr.eschatology import DeathTrichotomy, WindingFate
        dt = DeathTrichotomy(W=3)
        fate = dt.classify_fate(W_final=3, substrate_coupled=False, mode_transferred=True)
        assert fate == WindingFate.MIGRATION

    def test_classify_reincorporation(self):
        from bpr.eschatology import DeathTrichotomy, WindingFate
        dt = DeathTrichotomy(W=3)
        fate = dt.classify_fate(W_final=3, substrate_coupled=True, mode_transferred=False)
        assert fate == WindingFate.REINCORPORATION

    def test_winding_conservation(self):
        from bpr.eschatology import DeathTrichotomy
        assert DeathTrichotomy.verify_conservation(5, [3, 2]) is True
        assert DeathTrichotomy.verify_conservation(5, [3, 1]) is False

    def test_negative_winding(self):
        from bpr.eschatology import DeathTrichotomy, WindingFate
        dt = DeathTrichotomy(W=-2)
        fates = dt.allowed_fates()
        assert len(fates) == 3
        assert dt.dissolution_requires() == 2


# ═══════════════════════════════════════════════════════════════════════════
# Symbolic Projection  (Def 3.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestSymbolicProjection:
    """pi: S -> Sigma surjective, invariant-preserving."""

    def test_build_default_projection(self):
        from bpr.eschatology import build_default_projection
        proj = build_default_projection()
        # Should have 9 registered invariants (Table 1)
        truth_elems = proj.project("invariant_truth")
        assert len(truth_elems) == 4  # physics + 3 traditions

    def test_projection_round_trip(self):
        from bpr.eschatology import build_default_projection
        proj = build_default_projection()
        elems = proj.project("information_conservation")
        for elem in elems:
            inv = proj.inverse_image(elem.name, elem.domain)
            assert inv is not None
            assert inv.name == "information_conservation"

    def test_source_attractors_exist(self):
        from bpr.eschatology import build_default_projection, AttractorType
        proj = build_default_projection()
        sources = proj.source_attractors()
        assert len(sources) >= 9, "Should have all 9 source attractors from Table 1"

    def test_local_attractors_per_tradition(self):
        from bpr.eschatology import build_default_projection
        proj = build_default_projection()
        for tradition in ["Judaism", "Christianity", "Islam"]:
            locals_ = proj.local_attractors(tradition)
            assert len(locals_) >= 9, f"{tradition} should have at least 9 local attractors"

    def test_physics_elements_are_source_type(self):
        from bpr.eschatology import build_default_projection, AttractorType
        proj = build_default_projection()
        truth_elems = proj.project("invariant_truth")
        physics_elem = [e for e in truth_elems if e.domain == "physics"]
        assert len(physics_elem) == 1
        assert physics_elem[0].attractor_type == AttractorType.SOURCE

    def test_tradition_elements_are_local_type(self):
        from bpr.eschatology import build_default_projection, AttractorType
        proj = build_default_projection()
        truth_elems = proj.project("invariant_truth")
        tradition_elems = [e for e in truth_elems if e.domain != "physics"]
        assert all(e.attractor_type == AttractorType.LOCAL for e in tradition_elems)

    def test_inverse_image_missing_returns_none(self):
        from bpr.eschatology import build_default_projection
        proj = build_default_projection()
        assert proj.inverse_image("nonexistent", "physics") is None

    def test_project_missing_returns_empty(self):
        from bpr.eschatology import build_default_projection
        proj = build_default_projection()
        assert proj.project("nonexistent") == []


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Traditional Map  (Table 1)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossTraditionalMap:
    """Verify the translation dictionary structure."""

    def test_map_has_nine_entries(self):
        from bpr.eschatology import CROSS_TRADITIONAL_MAP
        assert len(CROSS_TRADITIONAL_MAP) == 9

    def test_all_entries_have_required_keys(self):
        from bpr.eschatology import CROSS_TRADITIONAL_MAP
        required_keys = {"dynamical_concept", "mathematical_form", "Judaism", "Christianity", "Islam"}
        for entry in CROSS_TRADITIONAL_MAP:
            assert required_keys.issubset(entry.keys()), f"Missing keys in {entry}"

    def test_first_entry_is_invariant_truth(self):
        from bpr.eschatology import CROSS_TRADITIONAL_MAP
        assert CROSS_TRADITIONAL_MAP[0]["dynamical_concept"] == "Invariant truth"

    def test_last_entry_is_topological_trichotomy(self):
        from bpr.eschatology import CROSS_TRADITIONAL_MAP
        assert CROSS_TRADITIONAL_MAP[-1]["dynamical_concept"] == "Topological trichotomy"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: coupled stain-coherence-judgment pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end tests combining multiple constructs."""

    def test_restoration_beats_noise(self):
        """When restoration dominates noise, judgment should be high."""
        from bpr.eschatology import JudgmentFunctional
        jf = JudgmentFunctional()
        J = jf.evaluate_analytic(u_plus_const=5.0, u_minus_const=0.1)
        assert J > 0.8

    def test_noise_beats_restoration(self):
        """When noise dominates restoration, judgment should be low."""
        from bpr.eschatology import JudgmentFunctional, HeartGainFunction, StainDynamics
        hg = HeartGainFunction(kappa_s=5.0)
        sd = StainDynamics(alpha=2.0, beta=0.1, gamma=0.01)
        jf = JudgmentFunctional(heart_gain=hg, stain_dynamics=sd)
        J = jf.evaluate_analytic(u_plus_const=0.1, u_minus_const=5.0)
        assert J < 0.3

    def test_deceptive_state_collapses_under_judgment(self):
        """Deceptive states should not survive judgment."""
        from bpr.eschatology import DeceptionClassifier, JudgmentFunctional
        dc = DeceptionClassifier(K_c=0.5)
        jf = JudgmentFunctional()

        # Deceptive state: high local coherence maintained by self-deception
        # But globally, noise is high
        result = jf.evaluate(
            u_plus=lambda t: 0.2,
            u_minus=lambda t: 3.0,
            K0=0.9,
            t_final=100.0,
        )
        # Under judgment, the initially high coherence should degrade
        assert result["J"] < 0.5

    def test_collapse_reset_preserves_winding(self):
        """Phase transitions should not violate winding conservation."""
        from bpr.eschatology import DeathTrichotomy
        # Before collapse: W = 5
        # After collapse: must distribute W across components
        assert DeathTrichotomy.verify_conservation(5, [5]) is True
        assert DeathTrichotomy.verify_conservation(5, [3, 2]) is True
        assert DeathTrichotomy.verify_conservation(5, [2, 2]) is False

    def test_full_pipeline_monotonicity(self):
        """More restoration relative to noise should always improve judgment."""
        from bpr.eschatology import JudgmentFunctional
        jf = JudgmentFunctional()
        ratios = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        judgments = []
        for ratio in ratios:
            J = jf.evaluate_analytic(u_plus_const=ratio, u_minus_const=1.0)
            judgments.append(J)
        # Judgment should be monotonically increasing with restoration ratio
        for i in range(len(judgments) - 1):
            assert judgments[i + 1] >= judgments[i], (
                f"Judgment should increase with restoration: {judgments}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Terminal Coherence Surge  (Section 10.1.2)
# ═══════════════════════════════════════════════════════════════════════════

class TestTerminalCoherenceSurge:
    """Gamma burst as boundary-decoupling phase transition."""

    def test_surge_exhibits_peak_before_tc(self):
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge(t_c=30.0)
        t = np.linspace(0, 60, 1000)
        K = tcs.coherence_profile(t)
        peak_idx = np.argmax(K)
        assert t[peak_idx] < tcs.t_c, "Peak should occur before critical time"

    def test_surge_decays_after_tc(self):
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge(t_c=30.0)
        t_post = np.linspace(30.0, 60.0, 100)
        K_post = tcs.coherence_profile(t_post)
        # Coherence should monotonically decrease after t_c
        assert np.all(np.diff(K_post) <= 1e-10)

    def test_peak_coherence_exceeds_background(self):
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge(K_bg=0.05, A=0.8)
        assert tcs.peak_coherence() > tcs.K_bg

    def test_coherence_bounded_by_one(self):
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge(A=10.0)  # large amplitude
        t = np.linspace(0, 60, 1000)
        K = tcs.coherence_profile(t)
        assert np.all(K <= 1.0)

    def test_bpr_consistency_check(self):
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge()
        assert tcs.is_consistent_with_bpr(0.75) is True             # within [0.5, 1.5]
        assert tcs.is_consistent_with_bpr(0.0, tolerance=0) is False  # neural rundown (below BPR range)
        assert tcs.is_consistent_with_bpr(3.0) is False              # outside range
        assert tcs.is_consistent_with_bpr(0.5, tolerance=0) is True  # lower boundary inclusive (Bugbot fix)
        assert tcs.is_consistent_with_bpr(-0.1) is False             # negative

    def test_neural_rundown_is_monotone(self):
        from bpr.eschatology import TerminalCoherenceSurge
        t = np.linspace(0, 60, 500)
        K_rundown = TerminalCoherenceSurge.neural_rundown_profile(t)
        # Monotonically decreasing
        assert np.all(np.diff(K_rundown) <= 0)

    def test_bpr_distinguishable_from_rundown(self):
        """BPR surge should have higher peak than monotone rundown."""
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge(K_bg=0.05, A=0.8, t_c=30.0)
        t = np.linspace(0, 60, 1000)
        K_bpr = tcs.coherence_profile(t)
        K_rundown = TerminalCoherenceSurge.neural_rundown_profile(t, K0=0.5)
        assert np.max(K_bpr) > np.max(K_rundown)

    def test_frequency_range_set(self):
        from bpr.eschatology import TerminalCoherenceSurge
        tcs = TerminalCoherenceSurge()
        assert tcs.freq_range == (25.0, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# Superlinear Collective Coherence Scaling  (Section 10.2.2)
# ═══════════════════════════════════════════════════════════════════════════

class TestCollectiveCoherenceScaling:
    """chi_group ~ N^{1 + delta} with delta > 0."""

    def test_superlinear_exceeds_linear(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        ccs = CollectiveCoherenceScaling(delta=0.15)
        for N in [10, 50, 100, 1000]:
            assert ccs.group_coherence(N) > ccs.linear_coherence(N)

    def test_linear_at_delta_zero(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        ccs = CollectiveCoherenceScaling(delta=0.0)
        for N in [10, 100]:
            assert ccs.group_coherence(N) == pytest.approx(ccs.linear_coherence(N))

    def test_superlinear_ratio_increases_with_N(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        ccs = CollectiveCoherenceScaling(delta=0.15)
        ratios = [ccs.superlinear_ratio(N) for N in [10, 100, 1000]]
        for i in range(len(ratios) - 1):
            assert ratios[i + 1] > ratios[i]

    def test_fit_exponent_recovers_delta(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        true_delta = 0.2
        ccs = CollectiveCoherenceScaling(delta=true_delta, chi_1=1.5)
        N_values = np.array([5, 10, 20, 50, 100, 200, 500])
        chi_values = np.array([ccs.group_coherence(N) for N in N_values])
        fitted = ccs.fit_exponent(N_values, chi_values)
        assert fitted == pytest.approx(true_delta, abs=0.01)

    def test_is_superlinear_true(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        ccs = CollectiveCoherenceScaling(delta=0.2)
        N = np.array([10, 50, 100, 500])
        chi = np.array([ccs.group_coherence(n) for n in N])
        assert ccs.is_superlinear(N, chi) == True

    def test_is_superlinear_false_for_linear(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        ccs = CollectiveCoherenceScaling(delta=0.0)
        N = np.array([10, 50, 100, 500])
        chi = np.array([ccs.linear_coherence(n) for n in N])
        # Use significance threshold to handle float precision on perfectly linear data
        assert ccs.is_superlinear(N, chi, significance=0.01) == False

    def test_zero_agents_returns_zero(self):
        from bpr.eschatology import CollectiveCoherenceScaling
        ccs = CollectiveCoherenceScaling()
        assert ccs.group_coherence(0) == 0.0
        assert ccs.linear_coherence(0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Duty-Cycle Optimization  (Section 10.2.3)
# ═══════════════════════════════════════════════════════════════════════════

class TestDutyCycleOptimizer:
    """D* ~ 6/7 for matched active/rest quality ratio."""

    def test_sabbath_duty_cycle(self):
        from bpr.eschatology import DutyCycleOptimizer
        assert DutyCycleOptimizer.sabbath_duty_cycle() == pytest.approx(6.0 / 7.0)

    def test_optimal_duty_cycle_analytic(self):
        from bpr.eschatology import DutyCycleOptimizer
        dco = DutyCycleOptimizer(Q_active=6.0, Q_rest=1.0)
        assert dco.optimal_duty_cycle == pytest.approx(6.0 / 7.0)

    def test_equal_q_gives_half(self):
        from bpr.eschatology import DutyCycleOptimizer
        dco = DutyCycleOptimizer(Q_active=1.0, Q_rest=1.0)
        assert dco.optimal_duty_cycle == pytest.approx(0.5)

    def test_sustained_output_zero_at_extremes(self):
        from bpr.eschatology import DutyCycleOptimizer
        dco = DutyCycleOptimizer()
        assert dco.sustained_output(0.0) == 0.0
        assert dco.sustained_output(1.0) == 0.0

    def test_sustained_output_positive_at_moderate_duty(self):
        from bpr.eschatology import DutyCycleOptimizer
        dco = DutyCycleOptimizer()
        assert dco.sustained_output(0.5) > 0

    def test_scan_returns_valid_arrays(self):
        from bpr.eschatology import DutyCycleOptimizer
        dco = DutyCycleOptimizer()
        D, out = dco.scan_duty_cycles(n_points=50, n_cycles=50)
        assert len(D) == 50
        assert len(out) == 50
        assert np.all(D > 0) and np.all(D < 1)

    def test_optimal_is_interior(self):
        """Optimal duty cycle should not be at the extremes."""
        from bpr.eschatology import DutyCycleOptimizer
        dco = DutyCycleOptimizer(Q_active=6.0, Q_rest=1.0)
        D_opt = dco.find_optimal(n_points=100, n_cycles=100)
        assert 0.3 < D_opt < 0.99


# ═══════════════════════════════════════════════════════════════════════════
# Falsification Criteria  (Section 10.4)
# ═══════════════════════════════════════════════════════════════════════════

class TestFalsificationCriteria:
    """Explicit tests for the four falsification criteria."""

    def test_memory_kernel_oscillatory_passes(self):
        """Criterion 1: oscillatory correlations should not falsify."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        tau = np.linspace(0, 10, 500)
        # Oscillatory decay (BPR prediction) — need enough sign changes for 10% threshold
        # cos(20*tau) has ~63 zero crossings in [0,10], giving fraction ~63/499 > 0.1
        C = np.exp(-tau / 2.0) * np.cos(20.0 * tau)
        result = fc.test_memory_kernel_present(C, tau)
        assert result["has_oscillation"] == True
        assert result["falsified"] == False

    def test_memory_kernel_monotone_falsifies(self):
        """Criterion 1: pure exponential decay falsifies."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        tau = np.linspace(0, 10, 200)
        # Pure exponential (no oscillation)
        C = np.exp(-tau / 2.0)
        result = fc.test_memory_kernel_present(C, tau)
        assert result["has_oscillation"] == False
        assert result["falsified"] == True

    def test_trichotomy_three_fates_passes(self):
        """Criterion 2: three known fates should not falsify."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        result = fc.test_trichotomy_complete(
            ["dissolution", "migration", "reincorporation"]
        )
        assert result["falsified"] is False
        assert len(result["unknown_fates"]) == 0

    def test_trichotomy_fourth_fate_falsifies(self):
        """Criterion 2: a fourth fate should falsify."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        result = fc.test_trichotomy_complete(
            ["dissolution", "migration", "teleportation"]
        )
        assert result["falsified"] is True
        assert "teleportation" in result["unknown_fates"]

    def test_universality_high_overlap_passes(self):
        """Criterion 3: shared source attractors should not falsify."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        traditions = {
            "Judaism": ["truth", "record", "judgment", "deception"],
            "Christianity": ["truth", "record", "judgment", "deception"],
            "Islam": ["truth", "record", "judgment", "deception"],
        }
        result = fc.test_source_attractor_universality(traditions)
        assert result["falsified"] is False
        assert result["overlap_fraction"] == pytest.approx(1.0)

    def test_universality_no_overlap_falsifies(self):
        """Criterion 3: no shared attractors should falsify."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        traditions = {
            "A": ["truth", "record"],
            "B": ["beauty", "harmony"],
        }
        result = fc.test_source_attractor_universality(traditions)
        assert result["falsified"] is True
        assert result["overlap_fraction"] == 0.0

    def test_deceptive_transience_with_noise_passes(self):
        """Criterion 4: finite escape time with noise should not falsify."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        result = fc.test_deceptive_attractor_transience(delta_V=2.0, epsilon=0.5)
        assert result["is_finite"] == True
        assert result["falsified"] == False

    def test_deceptive_transience_no_noise_not_falsified(self):
        """Criterion 4: infinite lifetime without noise is physically expected."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        result = fc.test_deceptive_attractor_transience(delta_V=2.0, epsilon=0.0)
        # epsilon=0 means no noise, so infinite lifetime is expected, not falsifying
        assert result["falsified"] is False

    def test_run_all_returns_all_criteria(self):
        """run_all should return results for all provided criteria."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        tau = np.linspace(0, 10, 100)
        C = np.exp(-tau) * np.cos(2 * tau)
        results = fc.run_all(
            correlation_data=C,
            tau_values=tau,
            observed_fates=["dissolution", "migration"],
            traditions={"A": ["truth"], "B": ["truth"]},
            delta_V=1.0,
            epsilon=0.1,
        )
        assert "memory_kernel" in results
        assert "trichotomy" in results
        assert "universality" in results
        assert "deceptive_transience" in results

    def test_run_all_skips_missing_data(self):
        """run_all should skip criteria without data."""
        from bpr.eschatology import FalsificationCriteria
        fc = FalsificationCriteria()
        results = fc.run_all()
        assert "memory_kernel" not in results
        assert "trichotomy" not in results
        assert "deceptive_transience" in results  # always runs
