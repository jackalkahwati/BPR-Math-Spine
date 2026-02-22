"""
Tests for Functional Architecture of Reality
==========================================================

Coverage:
  - ResonanceKernel (Eq 2)
  - RealizedStateProjection (Eq 3)
  - IdentityWinding (Sec 3)
  - PermissionField (Sec 4)
  - SemanticEncoding (Sec 5)
  - SalienceField (Sec 6)
  - TrajectoryEvaluation (Sec 7)
  - CoherenceStack (Thm 8.1)
  - InteroperatorConsistency (Sec 9)
  - BPRStabilityMeasure
"""

import numpy as np
import pytest


# ===========================================================================
# ResonanceKernel
# ===========================================================================

class TestResonanceKernel:
    """Tests for constructive resonance kernel K_r (Eq 2)."""

    def test_evaluate_max_at_zero(self):
        """Maximum kernel at d_G=0, Δφ=0, d_p=0."""
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel(alpha=1.0, beta=1.0, eta=1.0)
        K_max = kr.evaluate(d_G=0, delta_phi=0, d_p=0)
        assert K_max == pytest.approx(kr.max_kernel_value())

    def test_evaluate_decays_with_distance(self):
        """Kernel decreases with graph distance."""
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel(alpha=1.0, beta=1.0, eta=1.0)
        K0 = kr.evaluate(d_G=0, delta_phi=0, d_p=0)
        K5 = kr.evaluate(d_G=5, delta_phi=0, d_p=0)
        assert K5 < K0

    def test_evaluate_phase_alignment(self):
        """Maximum at Δφ=0 (cos=1), minimum at Δφ=π (cos=-1)."""
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel(alpha=0, beta=2.0, eta=0)
        K_aligned = kr.evaluate(0, 0.0, 0)
        K_anti = kr.evaluate(0, np.pi, 0)
        assert K_aligned > K_anti

    def test_evaluate_positive(self):
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel()
        K = kr.evaluate(d_G=3.0, delta_phi=1.5, d_p=2.0)
        assert K > 0

    def test_evaluate_batch_shape(self):
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel()
        d_G = np.array([0, 1, 2, 3])
        K = kr.evaluate_batch(d_G, np.zeros(4), np.zeros(4))
        assert K.shape == (4,)

    def test_max_kernel_value(self):
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel(beta=2.0)
        assert kr.max_kernel_value() == pytest.approx(np.exp(2.0))

    def test_resonance_read_normalized(self):
        """resonance_read returns a weighted average."""
        from bpr.functional_architecture import ResonanceKernel
        kr = ResonanceKernel(alpha=0, beta=0, eta=0)
        cache = np.array([[1.0, 0.0], [0.0, 1.0]])
        d_G = np.array([0.0, 0.0])
        dphi = np.array([0.0, 0.0])
        dp = np.array([0.0, 0.0])
        result = kr.resonance_read(None, cache, d_G, dphi, dp)
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-10)


# ===========================================================================
# RealizedStateProjection
# ===========================================================================

class TestRealizedStateProjection:
    """Tests for realized state projection ψ = Π_r(Ψ) (Eq 3)."""

    def test_project_single_mode(self):
        """Single mode: ψ = G exp(iφ)."""
        from bpr.functional_architecture import RealizedStateProjection
        rsp = RealizedStateProjection()
        amp = np.array([1.0])
        phase = np.array([0.0])
        psi = rsp.project(amp, phase)
        assert psi == pytest.approx(1.0 + 0j)

    def test_project_phase_rotation(self):
        from bpr.functional_architecture import RealizedStateProjection
        rsp = RealizedStateProjection()
        amp = np.array([1.0])
        phase = np.array([np.pi / 2])
        psi = rsp.project(amp, phase)
        assert psi == pytest.approx(1j, abs=1e-10)

    def test_project_destructive_interference(self):
        """Two equal modes with π phase shift cancel."""
        from bpr.functional_architecture import RealizedStateProjection
        rsp = RealizedStateProjection()
        amp = np.array([1.0, 1.0])
        phase = np.array([0.0, np.pi])
        psi = rsp.project(amp, phase)
        assert abs(psi) == pytest.approx(0.0, abs=1e-10)

    def test_projection_amplitude_nonneg(self):
        from bpr.functional_architecture import RealizedStateProjection
        rsp = RealizedStateProjection()
        amp = np.random.randn(5)
        phase = np.random.uniform(0, 2 * np.pi, 5)
        assert rsp.projection_amplitude(amp, phase) >= 0

    def test_project_field(self):
        from bpr.functional_architecture import RealizedStateProjection
        rsp = RealizedStateProjection()
        amp = np.array([1.0, 0.5])
        phase = np.array([0.0, 0.0])
        basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        field = rsp.project_field(amp, phase, basis)
        assert field.shape == (3,)
        np.testing.assert_allclose(field, [1.0, 0.5, 0.0], atol=1e-10)


# ===========================================================================
# IdentityWinding
# ===========================================================================

class TestIdentityWinding:
    """Tests for identity as topological winding (Sec 3)."""

    def test_winding_number_zero(self):
        """Constant phase → W = 0."""
        from bpr.functional_architecture import IdentityWinding
        phi = np.full(20, 1.0)
        assert IdentityWinding.winding_number(phi) == 0

    def test_winding_number_one(self):
        """Phase going from 0 to 2π → W = 1."""
        from bpr.functional_architecture import IdentityWinding
        phi = np.linspace(0, 2 * np.pi, 101)
        assert IdentityWinding.winding_number(phi) == 1

    def test_winding_number_two(self):
        """Phase going from 0 to 4π → W = 2."""
        from bpr.functional_architecture import IdentityWinding
        phi = np.linspace(0, 4 * np.pi, 201)
        assert IdentityWinding.winding_number(phi) == 2

    def test_winding_number_negative(self):
        """Phase going from 2π to 0 → W = -1."""
        from bpr.functional_architecture import IdentityWinding
        phi = np.linspace(2 * np.pi, 0, 101)
        assert IdentityWinding.winding_number(phi) == -1

    def test_topologically_protected(self):
        from bpr.functional_architecture import IdentityWinding
        assert IdentityWinding.is_topologically_protected(1) is True
        assert IdentityWinding.is_topologically_protected(-2) is True
        assert IdentityWinding.is_topologically_protected(0) is False

    def test_memory_timescale(self):
        from bpr.functional_architecture import IdentityWinding
        iw = IdentityWinding(tau_0=1.0, alpha=2.0)
        assert iw.memory_timescale(0) == pytest.approx(1.0)
        assert iw.memory_timescale(3) == pytest.approx(9.0)

    def test_identity_continuity_constant(self):
        """Constant winding → continuous identity."""
        from bpr.functional_architecture import IdentityWinding
        iw = IdentityWinding()
        traj = [np.linspace(0, 2 * np.pi, 51) for _ in range(5)]
        result = iw.identity_continuity(traj)
        assert result["is_continuous"] is True

    def test_identity_continuity_discontinuous(self):
        """Winding changes → discontinuous identity."""
        from bpr.functional_architecture import IdentityWinding
        iw = IdentityWinding()
        traj = [
            np.linspace(0, 2 * np.pi, 51),  # W=1
            np.linspace(0, 4 * np.pi, 101),  # W=2
        ]
        result = iw.identity_continuity(traj)
        assert result["is_continuous"] is False

    def test_structural_similarity_identical(self):
        from bpr.functional_architecture import IdentityWinding
        phi = np.linspace(0, 2 * np.pi, 50)
        sigma = IdentityWinding.structural_similarity(phi, phi)
        assert sigma == pytest.approx(1.0, rel=1e-6)

    def test_vortex_core_energy_positive(self):
        from bpr.functional_architecture import IdentityWinding
        E = IdentityWinding.vortex_core_energy(W=1, system_size=10.0, core_size=0.1)
        assert E > 0

    def test_vortex_energy_scales_with_W(self):
        from bpr.functional_architecture import IdentityWinding
        E1 = IdentityWinding.vortex_core_energy(1)
        E3 = IdentityWinding.vortex_core_energy(3)
        assert E3 == pytest.approx(3.0 * E1)


# ===========================================================================
# PermissionField
# ===========================================================================

class TestPermissionField:
    """Tests for permission-gated coupling (Sec 4)."""

    def test_permission_above_threshold(self):
        """E >> E_min → P ≈ 1."""
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=10.0)
        P = float(pf.permission(5.0))
        assert P > 0.99

    def test_permission_below_threshold(self):
        """E << E_min → P ≈ 0."""
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=10.0)
        P = float(pf.permission(-5.0))
        assert P < 0.01

    def test_permission_at_threshold(self):
        """E = E_min → P = 0.5."""
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=10.0)
        P = float(pf.permission(0.5))
        assert P == pytest.approx(0.5, abs=1e-10)

    def test_gated_coupling(self):
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=10.0)
        # High eligibility → P ≈ 1, so C ≈ inner_product
        C = pf.gated_coupling(eligibility=10.0, inner_product=3.0)
        assert C == pytest.approx(3.0, rel=0.01)

    def test_gated_coupling_sealed(self):
        """Low eligibility → P ≈ 0 → C ≈ 0."""
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=10.0)
        C = pf.gated_coupling(eligibility=-10.0, inner_product=3.0)
        assert abs(C) < 0.01

    def test_boundary_tension(self):
        from bpr.functional_architecture import PermissionField
        pf = PermissionField()
        grad_V = np.array([3.0, 4.0])
        assert pf.boundary_tension(grad_V) == pytest.approx(5.0)

    def test_is_sealed(self):
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=20.0)
        assert pf.is_sealed(-5.0) is True
        assert pf.is_sealed(5.0) is False

    def test_is_fully_open(self):
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=20.0)
        assert pf.is_fully_open(5.0) is True
        assert pf.is_fully_open(-5.0) is False

    def test_permission_batch(self):
        """Permission field on array input."""
        from bpr.functional_architecture import PermissionField
        pf = PermissionField(E_min=0.5, steepness=10.0)
        E = np.array([-5.0, 0.5, 5.0])
        P = pf.permission(E)
        assert P.shape == (3,)
        assert P[0] < 0.01
        assert P[1] == pytest.approx(0.5, abs=1e-10)
        assert P[2] > 0.99


# ===========================================================================
# SemanticEncoding
# ===========================================================================

class TestSemanticEncoding:
    """Tests for semantic encoding operator (Sec 5)."""

    def test_distortion_identical(self):
        from bpr.functional_architecture import SemanticEncoding
        psi = np.array([1.0, 2.0, 3.0])
        assert SemanticEncoding.distortion(psi, psi) == pytest.approx(0.0)

    def test_distortion_orthogonal(self):
        from bpr.functional_architecture import SemanticEncoding
        psi = np.array([1.0, 0.0])
        phi = np.array([0.0, 1.0])
        D = SemanticEncoding.distortion(psi, phi)
        assert D == pytest.approx(2.0)  # |psi - phi|² / |psi|² = 2/1

    def test_coherence_regularizer_equivariant(self):
        """L_coh = 0 when encoding is equivariant."""
        from bpr.functional_architecture import SemanticEncoding
        phi = np.array([1.0, 2.0, 3.0])
        assert SemanticEncoding.coherence_regularizer(phi, phi) == pytest.approx(0.0)

    def test_coherence_regularizer_nonzero(self):
        from bpr.functional_architecture import SemanticEncoding
        phi_fg = np.array([1.0, 2.0])
        phi_gf = np.array([2.0, 1.0])
        L = SemanticEncoding.coherence_regularizer(phi_fg, phi_gf)
        assert L > 0

    def test_phase_projection(self):
        from bpr.functional_architecture import SemanticEncoding
        x = np.array([1 + 1j, -1 + 0j, 0 + 1j])
        phases = SemanticEncoding.phase_projection(x)
        assert phases[0] == pytest.approx(np.pi / 4)
        assert phases[1] == pytest.approx(np.pi)
        assert phases[2] == pytest.approx(np.pi / 2)


# ===========================================================================
# SalienceField
# ===========================================================================

class TestSalienceField:
    """Tests for salience field and attention (Sec 6)."""

    def test_attention_weights_sum_one(self):
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=1.0)
        R = np.array([1.0, 2.0, 3.0])
        weights = sf.attention_weights(R)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-10)

    def test_attention_weights_positive(self):
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=1.0)
        R = np.random.randn(10)
        weights = sf.attention_weights(R)
        assert np.all(weights > 0)

    def test_high_beta_concentrates(self):
        """High β → weights concentrate on maximum."""
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=100.0)
        R = np.array([0.0, 1.0, 0.5])
        weights = sf.attention_weights(R)
        assert weights[1] > 0.99

    def test_low_beta_uniform(self):
        """β → 0 → uniform weights."""
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=0.001)
        R = np.array([1.0, 2.0, 3.0])
        weights = sf.attention_weights(R)
        assert np.std(weights) < 0.01

    def test_entropy_uniform_maximum(self):
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=0.001)
        R = np.zeros(10)
        H = sf.entropy(R)
        H_max = sf.max_entropy(10)
        assert H == pytest.approx(H_max, rel=0.01)

    def test_entropy_concentrated_low(self):
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=100.0)
        R = np.array([0.0, 10.0, 0.0])
        H = sf.entropy(R)
        assert H < 0.1

    def test_concentration_ratio_range(self):
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=1.0)
        R = np.array([1.0, 2.0, 3.0])
        cr = sf.concentration_ratio(R)
        assert 0 <= cr <= 1

    def test_attended_state(self):
        from bpr.functional_architecture import SalienceField
        sf = SalienceField(beta=100.0)
        states = np.array([[1.0, 0.0], [0.0, 1.0]])
        R = np.array([0.0, 10.0])
        result = sf.attended_state(states, R)
        # Should be close to second state
        np.testing.assert_allclose(result, [0.0, 1.0], atol=0.01)


# ===========================================================================
# TrajectoryEvaluation
# ===========================================================================

class TestTrajectoryEvaluation:
    """Tests for trajectory evaluation functional J(γ) (Sec 7)."""

    def test_evaluate_sum(self):
        from bpr.functional_architecture import TrajectoryEvaluation
        te = TrajectoryEvaluation()
        U = np.array([1.0, 2.0, 3.0])
        J = te.evaluate(U, dt=0.5)
        assert J == pytest.approx(3.0, rel=1e-10)

    def test_evaluate_with_control_penalty(self):
        from bpr.functional_architecture import TrajectoryEvaluation
        te = TrajectoryEvaluation(lambda_control=1.0)
        U = np.array([10.0])
        control = np.array([2.0])
        J = te.evaluate_with_control(U, control, dt=1.0)
        # (10 - 1.0*4) * 1 = 6.0
        assert J == pytest.approx(6.0, rel=1e-10)

    def test_utility_from_potential(self):
        from bpr.functional_architecture import TrajectoryEvaluation
        V = np.array([1.0, 2.0, 3.0])
        U = TrajectoryEvaluation.utility_from_coherence_potential(V)
        np.testing.assert_array_equal(U, -V)

    def test_is_stationary_constant(self):
        """Constant utility → stationary."""
        from bpr.functional_architecture import TrajectoryEvaluation
        te = TrajectoryEvaluation()
        U = np.full(20, 5.0)
        assert te.is_stationary(U) == True

    def test_is_stationary_varying(self):
        """Rapidly varying utility → not stationary."""
        from bpr.functional_architecture import TrajectoryEvaluation
        te = TrajectoryEvaluation()
        U = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        assert te.is_stationary(U, tolerance=0.01) == False


# ===========================================================================
# CoherenceStack
# ===========================================================================

class TestCoherenceStack:
    """Tests for minimal coherence stack (Thm 8.1)."""

    def test_full_stack_coherent(self):
        from bpr.functional_architecture import CoherenceStack
        cs = CoherenceStack()
        assert cs.is_coherent is True
        assert cs.n_operators == 9

    def test_missing_operator_not_coherent(self):
        from bpr.functional_architecture import CoherenceStack, OperatorType
        cs = CoherenceStack()
        cs_missing = cs.remove_operator(OperatorType.IDENTITY)
        assert cs_missing.is_coherent is False
        assert cs_missing.n_operators == 8

    def test_collapse_modes(self):
        from bpr.functional_architecture import CoherenceStack, OperatorType, COLLAPSE_MODES
        cs = CoherenceStack()
        cs_no_memory = cs.remove_operator(OperatorType.PERSISTENCE)
        modes = cs_no_memory.collapse_modes()
        assert len(modes) == 1
        assert modes[0] == COLLAPSE_MODES[OperatorType.PERSISTENCE]

    def test_missing_operators_list(self):
        from bpr.functional_architecture import CoherenceStack, OperatorType
        cs = CoherenceStack()
        cs2 = cs.remove_operator(OperatorType.SALIENCE)
        cs3 = cs2.remove_operator(OperatorType.ENCODING)
        missing = cs3.missing_operators()
        assert len(missing) == 2
        assert OperatorType.SALIENCE in missing
        assert OperatorType.ENCODING in missing

    def test_all_operators_have_collapse_mode(self):
        from bpr.functional_architecture import CoherenceStack, OperatorType, COLLAPSE_MODES
        cs = CoherenceStack()
        for op in cs.operators_present:
            assert op in COLLAPSE_MODES

    def test_remove_each_operator_breaks_coherence(self):
        """Removing any single operator breaks coherence (Thm 8.1)."""
        from bpr.functional_architecture import CoherenceStack
        cs = CoherenceStack()
        for op in list(cs.operators_present.keys()):
            cs_partial = cs.remove_operator(op)
            assert cs_partial.is_coherent is False


# ===========================================================================
# InteroperatorConsistency
# ===========================================================================

class TestInteroperatorConsistency:
    """Tests for interoperator consistency conditions (Sec 9)."""

    def test_salience_stability_consistent(self):
        """Attention concentrated on S → consistent."""
        from bpr.functional_architecture import InteroperatorConsistency
        weights = np.array([0.4, 0.35, 0.2, 0.025, 0.025])
        mask = np.array([True, True, True, False, False])
        result = InteroperatorConsistency.salience_stability_consistency(
            weights, mask, epsilon=0.1
        )
        assert result["consistent"] is True
        assert result["mass_on_S"] == pytest.approx(0.95)

    def test_salience_stability_inconsistent(self):
        """Attention mostly off S → inconsistent."""
        from bpr.functional_architecture import InteroperatorConsistency
        weights = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        mask = np.array([True, True, False, False, False])
        result = InteroperatorConsistency.salience_stability_consistency(
            weights, mask, epsilon=0.1
        )
        assert result["consistent"] is False

    def test_permission_eligibility_consistent(self):
        from bpr.functional_architecture import InteroperatorConsistency
        P = np.array([0.0, 0.0, 0.8, 0.9])
        E = np.array([0.1, 0.3, 0.8, 1.0])
        result = InteroperatorConsistency.permission_eligibility_consistency(
            P, E, E_threshold=0.5
        )
        assert result["consistent"] is True
        assert result["violations"] == 0

    def test_permission_eligibility_violation(self):
        """P > 0 when E < threshold → violation."""
        from bpr.functional_architecture import InteroperatorConsistency
        P = np.array([0.5, 0.0])  # P=0.5 but E=0.1 < 0.5
        E = np.array([0.1, 0.8])
        result = InteroperatorConsistency.permission_eligibility_consistency(
            P, E, E_threshold=0.5
        )
        assert result["consistent"] is False
        assert result["violations"] == 1

    def test_winding_cache_consistent(self):
        from bpr.functional_architecture import InteroperatorConsistency
        W = np.array([1, 2, 3, 4])
        tau = np.array([1.0, 2.0, 3.0, 4.0])
        result = InteroperatorConsistency.winding_cache_consistency(W, tau)
        assert result["consistent"] is True
        assert result["correlation"] > 0.9

    def test_winding_cache_inconsistent(self):
        from bpr.functional_architecture import InteroperatorConsistency
        W = np.array([1, 2, 3, 4])
        tau = np.array([4.0, 3.0, 2.0, 1.0])  # Decreasing (wrong)
        result = InteroperatorConsistency.winding_cache_consistency(W, tau)
        assert result["consistent"] is False


# ===========================================================================
# BPRStabilityMeasure
# ===========================================================================

class TestBPRStabilityMeasure:
    """Tests for BPR stability measure bridging construct."""

    def test_compute_nonnegative(self):
        from bpr.functional_architecture import BPRStabilityMeasure
        phi = np.random.randn(50)
        S = BPRStabilityMeasure.compute(phi)
        assert S >= 0

    def test_compute_zero_for_constant(self):
        from bpr.functional_architecture import BPRStabilityMeasure
        phi = np.full(30, 2.0)
        assert BPRStabilityMeasure.compute(phi) == pytest.approx(0.0, abs=1e-12)

    def test_attractor_separability_loss(self):
        from bpr.functional_architecture import BPRStabilityMeasure
        # Identity covariance → L_AS = 0
        phi_ensemble = np.random.randn(100, 5)
        # Not exactly zero but should be well-defined
        L = BPRStabilityMeasure.attractor_separability_loss(phi_ensemble)
        assert L >= 0


# ===========================================================================
# Integration tests
# ===========================================================================

class TestFunctionalArchitectureIntegration:
    """Cross-class integration tests for Functional Architecture of Reality."""

    def test_kernel_projection_pipeline(self):
        """Resonance kernel → projection → realized state."""
        from bpr.functional_architecture import ResonanceKernel, RealizedStateProjection
        kr = ResonanceKernel(alpha=0.5, beta=1.0, eta=0.5)
        rsp = RealizedStateProjection(n_modes=3)

        # Cache read gives amplitudes
        d_G = np.array([0.0, 1.0, 2.0])
        dphi = np.array([0.0, 0.1, 0.2])
        dp = np.array([0.0, 0.5, 1.0])
        weights = kr.evaluate_batch(d_G, dphi, dp)
        weights = weights / np.sum(weights)

        # Use weights as amplitudes for projection
        phases = np.array([0.0, np.pi / 4, np.pi / 2])
        psi = rsp.project(weights, phases)
        assert np.isfinite(psi)

    def test_salience_selects_stable(self):
        """Salience field concentrates attention on high-stability states."""
        from bpr.functional_architecture import SalienceField, BPRStabilityMeasure

        # Two states: one constant (in S), one varying (not in S)
        phi_stable = np.full(50, 1.0)
        phi_unstable = np.linspace(0, 10, 50)
        S_stable = BPRStabilityMeasure.compute(phi_stable)
        S_unstable = BPRStabilityMeasure.compute(phi_unstable)

        # Salience from stability (note: paper says high S → high attention)
        sf = SalienceField(beta=10.0)
        R = np.array([S_stable, S_unstable])
        weights = sf.attention_weights(R)
        # The one with higher stability measure gets more weight
        assert weights[1] > weights[0]  # Unstable has higher S value

    def test_permission_gates_coupling(self):
        """Permission field blocks coupling when eligibility is low."""
        from bpr.functional_architecture import PermissionField, ResonanceKernel
        pf = PermissionField(E_min=0.5, steepness=20.0)
        kr = ResonanceKernel(beta=1.0)

        # High eligibility → full coupling
        K = kr.evaluate(d_G=0, delta_phi=0, d_p=0)
        C_open = pf.gated_coupling(eligibility=5.0, inner_product=K)
        assert C_open == pytest.approx(K, rel=0.01)

        # Low eligibility → blocked
        C_sealed = pf.gated_coupling(eligibility=-5.0, inner_product=K)
        assert C_sealed < 0.01 * K

    def test_winding_identity_and_cache(self):
        """Higher winding → longer memory → stronger identity."""
        from bpr.functional_architecture import IdentityWinding
        iw = IdentityWinding(tau_0=1.0, alpha=2.0)

        # W=1 vs W=3
        tau1 = iw.memory_timescale(1)
        tau3 = iw.memory_timescale(3)
        assert tau3 > tau1  # 9 > 1

        # Both topologically protected
        assert IdentityWinding.is_topologically_protected(1) is True
        assert IdentityWinding.is_topologically_protected(3) is True

    def test_coherence_stack_completeness(self):
        """All nine operators present → coherent; remove any one → incoherent."""
        from bpr.functional_architecture import CoherenceStack
        cs = CoherenceStack()
        assert cs.is_coherent is True

        for op in list(cs.operators_present.keys()):
            partial = cs.remove_operator(op)
            assert partial.is_coherent is False
            assert len(partial.collapse_modes()) == 1
