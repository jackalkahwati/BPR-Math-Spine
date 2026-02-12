"""
Tests for Theory XXIII: Meta-Boundary Dynamics with Exogenous Decree.

Run with: pytest -v tests/test_meta_boundary.py
"""

import numpy as np
import pytest


# =====================================================================
# Core meta-functional and constraint potential
# =====================================================================


class TestConstraintPotential:
    def test_double_well_minima(self):
        from bpr.meta_boundary import constraint_potential_double_well
        eta = 1.0
        V_plus = constraint_potential_double_well(np.array([eta]), eta=eta)
        V_minus = constraint_potential_double_well(np.array([-eta]), eta=eta)
        V_max = constraint_potential_double_well(np.array([0.0]), eta=eta)
        assert np.isclose(V_plus, 0.0)
        assert np.isclose(V_minus, 0.0)
        assert V_max > 0

    def test_double_well_derivative_at_minima(self):
        from bpr.meta_boundary import constraint_potential_derivative
        eta = 1.0
        Vp_plus = constraint_potential_derivative(np.array([eta]), eta=eta)
        Vp_minus = constraint_potential_derivative(np.array([-eta]), eta=eta)
        assert np.isclose(Vp_plus, 0.0)
        assert np.isclose(Vp_minus, 0.0)


class TestMetaFunctional:
    def test_meta_functional_positive_for_stable_config(self):
        from bpr.meta_boundary import meta_functional_value, MetaBoundaryParams
        kappa = np.ones(10) * 0.5
        grad_kappa = np.zeros((10, 1))
        sigma_frust = np.zeros(10)
        J = meta_functional_value(kappa, grad_kappa, sigma_frust, params=MetaBoundaryParams())
        assert J > 0

    def test_meta_functional_increases_with_gradient(self):
        from bpr.meta_boundary import meta_functional_value, MetaBoundaryParams
        kappa = np.ones(10) * 0.5
        sigma_frust = np.zeros(10)
        grad_small = np.zeros((10, 1))
        grad_large = np.ones((10, 1)) * 2.0
        J_small = meta_functional_value(kappa, grad_small, sigma_frust, params=MetaBoundaryParams())
        J_large = meta_functional_value(kappa, grad_large, sigma_frust, params=MetaBoundaryParams())
        assert J_large > J_small


class TestMetaBoundaryRHS:
    def test_rhs_shape(self):
        from bpr.meta_boundary import meta_boundary_rhs, MetaBoundaryParams
        kappa = np.ones(20)
        sigma_frust = np.zeros(20)
        lap_kappa = np.zeros(20)
        rhs = meta_boundary_rhs(kappa, sigma_frust, lap_kappa, params=MetaBoundaryParams())
        assert rhs.shape == kappa.shape

    def test_rhs_with_decree(self):
        from bpr.meta_boundary import meta_boundary_rhs, MetaBoundaryParams
        kappa = np.ones(20) * 0.5
        sigma_frust = np.zeros(20)
        lap_kappa = np.zeros(20)
        decree = np.ones(20) * 0.1
        rhs = meta_boundary_rhs(kappa, sigma_frust, lap_kappa, decree=decree, params=MetaBoundaryParams())
        assert rhs.shape == kappa.shape

    def test_rhs_zero_at_equilibrium(self):
        from bpr.meta_boundary import meta_boundary_rhs, MetaBoundaryParams
        params = MetaBoundaryParams(eta=1.0)
        kappa = np.ones(20) * params.eta  # at minimum
        sigma_frust = np.zeros(20)
        lap_kappa = np.zeros(20)
        rhs = meta_boundary_rhs(kappa, sigma_frust, lap_kappa, params=params)
        assert np.allclose(rhs, 0.0, atol=1e-10)


# =====================================================================
# Decree classes (A) and (B)
# =====================================================================


class TestDecreeStochastic:
    def test_stochastic_zero_mean(self):
        from bpr.meta_boundary import decree_stochastic_sample, DecreeStochasticParams
        np.random.seed(42)
        params = DecreeStochasticParams(var_D=0.1, rng=np.random.default_rng(42))
        x = np.linspace(0, 1, 100)
        samples = [decree_stochastic_sample(x, t, params) for t in range(10)]
        mean = np.mean(samples)
        assert abs(mean) < 0.5  # approximate zero mean

    def test_stochastic_bounded_variance(self):
        from bpr.meta_boundary import decree_stochastic_sample, DecreeStochasticParams
        params = DecreeStochasticParams(var_D=1.0, rng=np.random.default_rng(43))
        x = np.linspace(0, 1, 50)
        D = decree_stochastic_sample(x, 0.0, params)
        var = np.var(D)
        assert var < 10.0  # reasonable bound


class TestDecreeImpulse:
    def test_impulse_events_generated(self):
        from bpr.meta_boundary import generate_impulse_events, DecreeImpulseParams
        params = DecreeImpulseParams(rate=1.0, rng=np.random.default_rng(44))
        t_ev, x_ev, amp = generate_impulse_events(10.0, params)
        assert len(t_ev) == len(x_ev) == len(amp)

    def test_impulse_sample_shape(self):
        from bpr.meta_boundary import decree_impulse_sample
        x = np.linspace(-2, 2, 50)
        t_events = np.array([0.5])
        x_events = np.array([0.0])
        amplitudes = np.array([1.0])
        D = decree_impulse_sample(x, 0.5, t_events, x_events, amplitudes)
        assert D.shape == x.shape


# =====================================================================
# Meta-eligibility E2
# =====================================================================


class TestMetaEligibility:
    def test_E2_positive_when_eligible(self):
        from bpr.meta_boundary import MetaEligibility
        elig = MetaEligibility(S_acc=10.0, S_D=2.0, C_barrier=5.0)
        assert elig.E2 == 7.0
        assert elig.meta_rewrite_eligible is True

    def test_E2_negative_when_not_eligible(self):
        from bpr.meta_boundary import MetaEligibility
        elig = MetaEligibility(S_acc=1.0, S_D=0.0, C_barrier=5.0)
        assert elig.E2 == -4.0
        assert elig.meta_rewrite_eligible is False

    def test_S_D_computation(self):
        from bpr.meta_boundary import compute_S_D
        kappa = np.array([1.0, 2.0, 3.0])
        decree = np.array([0.5, 0.5, 0.5])
        S_D = compute_S_D(kappa, decree, dx=1.0)
        assert np.isclose(S_D, 1.0 * 0.5 + 2.0 * 0.5 + 3.0 * 0.5)


# =====================================================================
# Detectability signatures
# =====================================================================


class TestDetectability:
    def test_trivial_transition_zero_signatures(self):
        from bpr.meta_boundary import detectability_signatures
        result = detectability_signatures(1.0, 1.0)  # κ_a = κ_b
        assert result.domain_wall_energy == 0
        assert result.spectral_drift == 0
        assert result.any_detectable is False

    def test_nontrivial_transition_has_signatures(self):
        from bpr.meta_boundary import detectability_signatures
        result = detectability_signatures(-1.0, 1.0)  # κ_a ≠ κ_b
        assert result.domain_wall_energy > 0
        assert result.spectral_drift > 0
        assert result.any_detectable

    def test_domain_wall_tension_positive(self):
        from bpr.meta_boundary import domain_wall_tension
        sigma = domain_wall_tension(eta=1.0, alpha=1.0, beta=1.0, lambda_kappa=1.0)
        assert sigma > 0


# =====================================================================
# Front propagation and predictions
# =====================================================================


class TestFrontVelocity:
    def test_velocity_bound_positive(self):
        from bpr.meta_boundary import front_velocity_bound
        v_max = front_velocity_bound()
        assert v_max > 0


class TestPredictions:
    def test_predictions_dict_structure(self):
        from bpr.meta_boundary import meta_boundary_predictions
        preds = meta_boundary_predictions()
        assert "P23.1_front_velocity_bound" in preds
        assert "P23.2_domain_wall_tension" in preds
        assert "P23.3_barrier_cost" in preds
        assert "P23.4_tau_kappa" in preds

    def test_predictions_values_sensible(self):
        from bpr.meta_boundary import meta_boundary_predictions
        preds = meta_boundary_predictions()
        assert preds["P23.1_front_velocity_bound"] > 0
        assert preds["P23.2_domain_wall_tension"] > 0
        assert preds["P23.3_barrier_cost"] > 0


# =====================================================================
# Derivations
# =====================================================================


class TestAgentFieldDerivation:
    def test_D_equals_gA(self):
        from bpr.meta_boundary_derivations import derive_decree_from_agent
        A = np.array([1.0, 2.0, 3.0])
        D = derive_decree_from_agent(A, g=2.0)
        assert np.allclose(D, 2.0 * A)

    def test_verify_agent_derivation(self):
        from bpr.meta_boundary_derivations import verify_agent_derivation
        A = np.array([0.5, 1.0])
        out = verify_agent_derivation(A, g=1.0)
        assert "D" in out
        assert "consistency" in out


class TestCoarseGrainDerivation:
    def test_D_mem_from_zeta_mean(self):
        from bpr.meta_boundary_derivations import coarse_grain_effective_dynamics
        zeta_mean = np.array([0.1, 0.2, 0.3])
        kappa = np.zeros(3)
        D_mem = coarse_grain_effective_dynamics(kappa, zeta_mean, coupling=1.0)
        assert np.allclose(D_mem, zeta_mean)

    def test_effective_decree_from_memory(self):
        from bpr.meta_boundary_derivations import effective_decree_from_memory
        kappa_hist = np.array([0.5, 0.6, 0.7])
        t_hist = np.array([0.0, 0.1, 0.2])
        D = effective_decree_from_memory(kappa_hist, t_hist, tau_mem=0.5, coupling=1.0)
        assert isinstance(D, (float, np.floating))


# =====================================================================
# First-principles integration
# =====================================================================


class TestFirstPrinciplesIntegration:
    def test_meta_boundary_in_first_principles(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate(p=104729, N=10000)
        params = sdt.meta_boundary_params()
        assert params is not None
        assert hasattr(params, "tau_kappa")
        assert hasattr(params, "eta")

    def test_predictions_include_P23(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate(p=104729, N=10000)
        preds = sdt.predictions()
        p23_keys = [k for k in preds if k.startswith("P23.")]
        assert len(p23_keys) >= 5
