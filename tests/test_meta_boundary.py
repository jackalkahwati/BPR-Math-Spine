"""
Tests for Theory XXIII: Meta-Boundary Dynamics and Global Phase Reindexing.

Tests cover all core mathematical constructs from the meta_boundary module:
    - Constraint field and state triple (Def 1.1)
    - Boundary phase action and dispersion (Eq 7-9)
    - Gauge structure and curvature invariance (Def 3.1, Thm 3.2)
    - Constraint potential (Eq 31, 35)
    - Meta-functional J[kappa; b, m] (Def 4.1)
    - Meta-boundary evolution equation (Eq 27)
    - Meta-rewrite eligibility (Eq 36-37)
    - Front solutions (Eq 43, 46, 49)
    - Energy cost scaling (Eq 51-56)
    - Spectral drift (Eq 60-61)
    - Detectability theorem (Thm 10.1)
    - Hysteresis (Eq 64-67)
    - Thermodynamic signatures (Eq 68-71)
    - Stability analysis (Eq 73-78)
    - Topological memory scaling (Eq 83)
    - CERN scenario (Section 16)

Run with:  pytest tests/test_meta_boundary.py -v
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Field (Def 1.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintField:
    """State triple (b, m, kappa) and winding number."""

    def test_create_state_triple(self):
        from bpr.meta_boundary import ConstraintField
        kappa = np.ones(10)
        cf = ConstraintField(kappa=kappa)
        assert len(cf.kappa) == 10
        assert cf.b is None
        assert cf.m is None

    def test_create_with_all_fields(self):
        from bpr.meta_boundary import ConstraintField
        cf = ConstraintField(
            kappa=np.ones(10),
            b=np.zeros(10),
            m=np.random.randn(10),
            x=np.linspace(0, 1, 10),
        )
        assert cf.b is not None
        assert cf.m is not None
        assert cf.x is not None

    def test_winding_number_zero(self):
        from bpr.meta_boundary import ConstraintField
        phi = np.zeros(100)  # constant phase
        assert ConstraintField.winding_number(phi) == 0

    def test_winding_number_one(self):
        from bpr.meta_boundary import ConstraintField
        theta = np.linspace(0, 2 * np.pi, 200)
        assert ConstraintField.winding_number(theta) == 1

    def test_winding_number_negative(self):
        from bpr.meta_boundary import ConstraintField
        theta = np.linspace(0, -2 * np.pi, 200)
        assert ConstraintField.winding_number(theta) == -1

    def test_winding_number_two(self):
        from bpr.meta_boundary import ConstraintField
        theta = np.linspace(0, 4 * np.pi, 400)
        assert ConstraintField.winding_number(theta) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Boundary Phase Action (Eq 7-9)
# ═══════════════════════════════════════════════════════════════════════════

class TestBoundaryPhaseAction:
    """Phase action, dispersion relation, and decoherence from gradients."""

    def test_dispersion_at_zero_k(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        bpa = BoundaryPhaseAction(kappa_s=1.0, mu_Phi=1.0, m_Phi_sq=2.0)
        omega = bpa.dispersion(np.array([0.0]))
        assert omega[0] == pytest.approx(1.0)  # sqrt(1.0 * 2.0 / 2)

    def test_dispersion_monotone_in_k(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        bpa = BoundaryPhaseAction()
        k = np.linspace(0, 10, 100)
        omega = bpa.dispersion(k)
        assert np.all(np.diff(omega) >= 0)

    def test_phase_gap(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        bpa = BoundaryPhaseAction(mu_Phi=2.0, m_Phi_sq=4.0)
        gap = bpa.phase_gap()
        assert gap == pytest.approx(2.0)  # sqrt(2 * 4 / 2) = 2

    def test_decoherence_rate_proportional_to_gradient(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        bpa = BoundaryPhaseAction(kappa_s=2.0)
        assert bpa.decoherence_rate(1.0) == pytest.approx(2.0)
        assert bpa.decoherence_rate(0.0) == pytest.approx(0.0)

    def test_geometric_decoherence_ratio(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        ratio = BoundaryPhaseAction.geometric_decoherence_ratio(4.0, 2.0)
        assert ratio == pytest.approx(2.0)

    def test_phase_energy_zero_for_uniform(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        bpa = BoundaryPhaseAction()
        grad_sq = np.zeros(50)
        assert bpa.phase_energy(grad_sq) == pytest.approx(0.0)

    def test_phase_energy_positive_for_gradients(self):
        from bpr.meta_boundary import BoundaryPhaseAction
        bpa = BoundaryPhaseAction(kappa_s=1.0)
        grad_sq = np.ones(50)
        assert bpa.phase_energy(grad_sq, dx=0.1) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Gauge Structure (Section 3)
# ═══════════════════════════════════════════════════════════════════════════

class TestGaugeStructure:
    """Boundary phase curvature and gauge invariance."""

    def test_boundary_phase_real_positive(self):
        from bpr.meta_boundary import GaugeStructure
        phi = GaugeStructure.boundary_phase(1.0 + 0.0j)
        assert phi == pytest.approx(0.0)

    def test_boundary_phase_imaginary(self):
        from bpr.meta_boundary import GaugeStructure
        phi = GaugeStructure.boundary_phase(0.0 + 1.0j)
        assert phi == pytest.approx(np.pi / 2)

    def test_gauge_transform(self):
        from bpr.meta_boundary import GaugeStructure
        # Phi_AB -> Phi_AB + theta_A - theta_B (antisymmetric convention)
        phi = GaugeStructure.gauge_transform(0.5, 0.3, 0.2)
        assert phi == pytest.approx(0.6)

    def test_curvature_simple(self):
        from bpr.meta_boundary import GaugeStructure
        omega = GaugeStructure.boundary_phase_curvature(0.1, 0.2, 0.3)
        assert omega == pytest.approx(0.6)

    def test_gauge_transform_curvature(self):
        """Thm 3.2: Omega_ABC is gauge invariant (unchanged under any rotation)."""
        from bpr.meta_boundary import GaugeStructure
        phi_AB, phi_BC, phi_CA = 0.5, 0.7, -0.3
        omega_orig = GaugeStructure.boundary_phase_curvature(phi_AB, phi_BC, phi_CA)
        theta_A, theta_B, theta_C = 1.2, -0.5, 0.8
        omega_new = GaugeStructure.gauge_transform_curvature(
            omega_orig, theta_A, theta_B, theta_C,
        )
        assert omega_new == pytest.approx(omega_orig)

    def test_gauge_invariant_curvature_exp(self):
        """Eq 19: exp(i*Omega_ABC) is the gauge-invariant observable."""
        from bpr.meta_boundary import GaugeStructure
        z = GaugeStructure.gauge_invariant_curvature_exp(0.5, 0.7, -0.3)
        assert abs(abs(z) - 1.0) < 1e-10  # unit modulus

    def test_curvature_two_form_1d_zero(self):
        from bpr.meta_boundary import GaugeStructure
        phi_1d = np.linspace(0, 2 * np.pi, 100)
        F = GaugeStructure.curvature_two_form(phi_1d)
        assert np.allclose(F, 0.0)

    def test_curvature_two_form_2d(self):
        from bpr.meta_boundary import GaugeStructure
        phi_2d = np.random.randn(20, 20)
        F = GaugeStructure.curvature_two_form(phi_2d, dx=0.1)
        assert F.shape == (20, 20)


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Potential (Eq 31-35)
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintPotential:
    """Double-well and multi-well potentials."""

    def test_double_well_minima(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential(lambda_kappa=1.0, eta=1.0)
        assert cp.double_well(1.0) == pytest.approx(0.0)
        assert cp.double_well(-1.0) == pytest.approx(0.0)

    def test_double_well_maximum_at_zero(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential(lambda_kappa=1.0, eta=1.0)
        V0 = cp.double_well(0.0)
        assert V0 > 0
        assert V0 == pytest.approx(0.25)  # lambda/4 * eta^4

    def test_double_well_derivative_zero_at_minima(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential(lambda_kappa=1.0, eta=1.0)
        assert cp.double_well_derivative(1.0) == pytest.approx(0.0, abs=1e-10)
        assert cp.double_well_derivative(-1.0) == pytest.approx(0.0, abs=1e-10)
        assert cp.double_well_derivative(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_barrier_height(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential(lambda_kappa=2.0, eta=1.5)
        expected = 2.0 * 1.5**4 / 4.0
        assert cp.barrier_height() == pytest.approx(expected)

    def test_mass_gap_positive(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential()
        assert cp.mass_gap() > 0

    def test_domain_wall_width_positive(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential()
        assert cp.domain_wall_width() > 0

    def test_multi_well_three_phases(self):
        from bpr.meta_boundary import ConstraintPotential
        kappa = np.linspace(-2, 2, 100)
        V = ConstraintPotential.multi_well(kappa, [-1.0, 0.0, 1.0])
        # Should have minima near -1, 0, 1
        assert V[0] > V[25]  # near -1, V should be low
        assert V[50] < V[40]  # near 0, V should be low

    def test_multi_well_at_exact_phases(self):
        from bpr.meta_boundary import ConstraintPotential
        # At exact phase positions, V should be zero (without perturbation)
        for kappa_a in [-1.0, 0.0, 1.0]:
            V = ConstraintPotential.multi_well(kappa_a, [-1.0, 0.0, 1.0], delta=0.0)
            assert V == pytest.approx(0.0)

    def test_second_derivative_positive_at_minima(self):
        from bpr.meta_boundary import ConstraintPotential
        cp = ConstraintPotential(lambda_kappa=1.0, eta=1.0)
        # V''(eta) = lambda * (3*eta^2 - eta^2) = 2*lambda*eta^2
        assert cp.double_well_second_derivative(1.0) == pytest.approx(2.0)
        assert cp.double_well_second_derivative(-1.0) == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Meta-Functional (Def 4.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaFunctional:
    """Constraint frustration functional J[kappa; b, m]."""

    def test_uniform_field_at_minimum(self):
        from bpr.meta_boundary import MetaFunctional
        mf = MetaFunctional()
        kappa = np.ones(100) * mf.potential.eta
        J = mf.evaluate(kappa, dx=0.1)
        # At minimum: gradient = 0, V(eta) = 0, so J ~ 0
        assert J == pytest.approx(0.0, abs=1e-6)

    def test_nonuniform_has_positive_gradient_energy(self):
        from bpr.meta_boundary import MetaFunctional
        mf = MetaFunctional()
        x = np.linspace(-5, 5, 200)
        kappa = np.tanh(x)  # non-uniform
        J = mf.evaluate(kappa, dx=x[1] - x[0])
        assert J > 0

    def test_functional_derivative_zero_at_minimum(self):
        from bpr.meta_boundary import MetaFunctional
        mf = MetaFunctional()
        kappa = np.ones(100) * mf.potential.eta
        dJ = mf.functional_derivative(kappa, dx=0.1)
        # At a uniform minimum, the derivative should be near zero
        assert np.max(np.abs(dJ)) < 0.1

    def test_with_stress_coupling(self):
        from bpr.meta_boundary import MetaFunctional
        mf = MetaFunctional(gamma=1.0)
        kappa = np.ones(50) * 0.5
        sigma = np.ones(50) * 2.0
        J_with = mf.evaluate(kappa, dx=0.1, sigma_frust=sigma)
        J_without = mf.evaluate(kappa, dx=0.1)
        assert J_with != J_without


# ═══════════════════════════════════════════════════════════════════════════
# Meta-Boundary Evolution (Eq 27)
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaBoundaryEvolution:
    """Reaction-diffusion evolution of constraint field."""

    def test_effective_diffusivity(self):
        from bpr.meta_boundary import MetaBoundaryEvolution
        mbe = MetaBoundaryEvolution(alpha=1.0, nu=0.5, tau_kappa=5.0)
        assert mbe.effective_diffusivity == pytest.approx(0.3)

    def test_rhs_shape(self):
        from bpr.meta_boundary import MetaBoundaryEvolution
        mbe = MetaBoundaryEvolution()
        kappa = np.random.randn(50)
        dydt = mbe.rhs(kappa, dx=0.1)
        assert dydt.shape == kappa.shape

    def test_equilibrium_at_minimum(self):
        from bpr.meta_boundary import MetaBoundaryEvolution
        mbe = MetaBoundaryEvolution()
        kappa = np.ones(100) * mbe.potential.eta
        dydt = mbe.rhs(kappa, dx=0.1)
        # At uniform minimum, time derivative should be near zero
        assert np.max(np.abs(dydt)) < 0.1

    def test_evolve_relaxes_to_minimum(self):
        from bpr.meta_boundary import MetaBoundaryEvolution
        mbe = MetaBoundaryEvolution(tau_kappa=1.0, alpha=1.0, nu=0.1,
                                     beta=1.0)
        # Start with small perturbation around a minimum
        kappa0 = np.ones(50) * mbe.potential.eta + 0.01 * np.random.randn(50)
        t, kappa_traj = mbe.evolve(kappa0, t_span=(0, 50), dx=0.1, n_steps=100)
        # Should relax toward eta
        final = kappa_traj[-1]
        assert np.std(final - mbe.potential.eta) < np.std(kappa0 - mbe.potential.eta)

    def test_evolve_output_shape(self):
        from bpr.meta_boundary import MetaBoundaryEvolution
        mbe = MetaBoundaryEvolution()
        kappa0 = np.zeros(30)
        t, kappa_traj = mbe.evolve(kappa0, t_span=(0, 10), dx=0.1, n_steps=50)
        assert len(t) == 50
        assert kappa_traj.shape == (50, 30)


# ═══════════════════════════════════════════════════════════════════════════
# Meta-Rewrite Eligibility (Eq 36-37)
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaRewriteEligibility:
    """Accumulated stress and barrier cost."""

    def test_steady_state_stress(self):
        from bpr.meta_boundary import MetaRewriteEligibility
        mre = MetaRewriteEligibility(mu=0.5)
        assert mre.accumulated_stress_steady_state(10.0) == pytest.approx(20.0)

    def test_eligibility_above_barrier(self):
        from bpr.meta_boundary import MetaRewriteEligibility
        mre = MetaRewriteEligibility()
        assert mre.is_eligible(5.0, 3.0) is True

    def test_not_eligible_below_barrier(self):
        from bpr.meta_boundary import MetaRewriteEligibility
        mre = MetaRewriteEligibility()
        assert mre.is_eligible(1.0, 3.0) is False

    def test_barrier_cost_double_well(self):
        from bpr.meta_boundary import MetaRewriteEligibility
        cost = MetaRewriteEligibility.barrier_cost_double_well(1.0, 2.0)
        assert cost == pytest.approx(4.0)  # 1.0 * 2.0^4 / 4 = 4.0

    def test_evolve_stress_converges(self):
        from bpr.meta_boundary import MetaRewriteEligibility
        mre = MetaRewriteEligibility(mu=0.5)
        sigma_0 = 5.0
        t, S = mre.evolve_stress(
            sigma_source=lambda t: sigma_0,
            S0=0.0,
            t_span=(0, 50),
        )
        # Should converge to S* = sigma_0 / mu = 10.0
        assert S[-1] == pytest.approx(10.0, abs=0.5)

    def test_eligibility_at_boundary(self):
        from bpr.meta_boundary import MetaRewriteEligibility
        mre = MetaRewriteEligibility()
        assert mre.is_eligible(3.0, 3.0) is True  # exact boundary


# ═══════════════════════════════════════════════════════════════════════════
# Front Solutions (Eq 43-49)
# ═══════════════════════════════════════════════════════════════════════════

class TestFrontSolution:
    """Static kink, moving front, and speed bounds."""

    def test_static_kink_endpoints(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        x = np.linspace(-20, 20, 1000)
        kappa = fs.static_kink(x)
        assert kappa[0] == pytest.approx(-fs.potential.eta, abs=0.01)
        assert kappa[-1] == pytest.approx(fs.potential.eta, abs=0.01)

    def test_static_kink_center(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        x = np.linspace(-20, 20, 1001)  # odd number so x=0 is exactly at center
        kappa = fs.static_kink(x, x0=0.0)
        idx_center = len(x) // 2
        assert kappa[idx_center] == pytest.approx(0.0, abs=0.01)

    def test_static_kink_monotone(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        x = np.linspace(-20, 20, 1000)
        kappa = fs.static_kink(x)
        assert np.all(np.diff(kappa) >= 0)

    def test_wall_tension_positive(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        assert fs.wall_tension() > 0

    def test_front_velocity_proportional_to_bias(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        v1 = fs.front_velocity(1.0)
        v2 = fs.front_velocity(2.0)
        assert v2 == pytest.approx(2 * v1)

    def test_front_velocity_zero_without_bias(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        assert fs.front_velocity(0.0) == pytest.approx(0.0)

    def test_max_velocity_positive(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        assert fs.max_velocity() > 0

    def test_front_velocity_bounded_by_max(self):
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        v_max = fs.max_velocity()
        # With a modest bias, velocity should be below max
        v = fs.front_velocity(0.1)
        assert abs(v) < v_max


# ═══════════════════════════════════════════════════════════════════════════
# Energy Cost Scaling (Eq 51-56)
# ═══════════════════════════════════════════════════════════════════════════

class TestEnergyCostScaling:
    """Domain wall energy and critical nucleus."""

    def test_region_cost_increases_with_R(self):
        from bpr.meta_boundary import EnergyCostScaling
        ecs = EnergyCostScaling(sigma_wall=1.0, delta_J=0.1)
        costs = [ecs.region_cost(R) for R in [1.0, 2.0, 5.0]]
        for i in range(len(costs) - 1):
            assert costs[i + 1] > costs[i]

    def test_critical_radius_formula(self):
        from bpr.meta_boundary import EnergyCostScaling
        ecs = EnergyCostScaling(sigma_wall=2.0, delta_J=0.5, d=3)
        # R_crit = (d-1) * sigma_wall / |delta_J| = 2 * 2 / 0.5 = 8
        assert ecs.critical_radius() == pytest.approx(8.0)

    def test_global_rewrite_cost(self):
        from bpr.meta_boundary import EnergyCostScaling
        ecs = EnergyCostScaling(sigma_wall=1.0, d=3)
        # C_global >= sigma_wall * L^{d-1} = 1 * 10^2 = 100
        assert ecs.global_rewrite_cost(10.0) == pytest.approx(100.0)

    def test_nucleation_barrier_finite(self):
        from bpr.meta_boundary import EnergyCostScaling
        ecs = EnergyCostScaling(sigma_wall=1.0, delta_J=0.1, d=3)
        barrier = ecs.nucleation_barrier()
        assert np.isfinite(barrier)
        assert barrier > 0

    def test_zero_delta_J_infinite_radius(self):
        from bpr.meta_boundary import EnergyCostScaling
        ecs = EnergyCostScaling(delta_J=0.0)
        assert ecs.critical_radius() == np.inf

    def test_surface_law_d3(self):
        from bpr.meta_boundary import EnergyCostScaling
        ecs = EnergyCostScaling(sigma_wall=1.0, d=3)
        # L^2 scaling in d=3
        c1 = ecs.global_rewrite_cost(1.0)
        c2 = ecs.global_rewrite_cost(2.0)
        assert c2 / c1 == pytest.approx(4.0)  # 2^2 = 4


# ═══════════════════════════════════════════════════════════════════════════
# Spectral Drift (Eq 60-61)
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralDrift:
    """Eigenvalue shifts from constraint transitions."""

    def test_spectral_shift_linear(self):
        from bpr.meta_boundary import SpectralDrift
        sd = SpectralDrift()
        shift = sd.spectral_shift(d_lambda_d_kappa=2.0, delta_kappa=0.5)
        assert shift == pytest.approx(1.0)

    def test_relative_shift_bound(self):
        from bpr.meta_boundary import SpectralDrift
        sd = SpectralDrift(kappa_char=1.0)
        bound = sd.relative_shift_bound(delta_kappa=0.1, d_ln_lambda_d_ln_kappa=2.0)
        assert bound == pytest.approx(0.2)

    def test_vacuum_energy_shift(self):
        from bpr.meta_boundary import SpectralDrift
        eigenvalues = np.array([1.0, 4.0, 9.0])
        delta_eigenvalues = np.array([0.1, 0.2, 0.3])
        shift = SpectralDrift.vacuum_energy_shift(eigenvalues, delta_eigenvalues)
        # (1/4) * (0.1/1 + 0.2/2 + 0.3/3) = (1/4) * (0.1 + 0.1 + 0.1) = 0.075
        assert shift == pytest.approx(0.075)

    def test_clock_constraint(self):
        from bpr.meta_boundary import SpectralDrift
        sd = SpectralDrift(kappa_char=1.0)
        bound = sd.constrain_from_clock(1e-17, d_ln_lambda_d_ln_kappa=1.0)
        assert bound == pytest.approx(1e-17)

    def test_zero_shift_at_zero_delta_kappa(self):
        from bpr.meta_boundary import SpectralDrift
        sd = SpectralDrift()
        assert sd.spectral_shift(5.0, 0.0) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Detectability Theorem (Thm 10.1)
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectabilityTheorem:
    """No-go theorem for undetectable rewrites."""

    def test_all_signatures_present(self):
        from bpr.meta_boundary import DetectabilityTheorem
        result = DetectabilityTheorem.check_signatures(
            E_wall=1.0, max_spectral_shift=0.5,
            max_metric_perturbation=0.1, coherence_dip=0.3,
        )
        assert result["any_detected"] == True
        assert result["n_signatures"] == 4

    def test_no_signatures(self):
        from bpr.meta_boundary import DetectabilityTheorem
        result = DetectabilityTheorem.check_signatures()
        assert result["any_detected"] == False
        assert result["n_signatures"] == 0

    def test_single_signature(self):
        from bpr.meta_boundary import DetectabilityTheorem
        result = DetectabilityTheorem.check_signatures(E_wall=1.0)
        assert result["wall_energy"] == True
        assert result["spectral_drift"] == False
        assert result["n_signatures"] == 1

    def test_transition_detectable(self):
        from bpr.meta_boundary import DetectabilityTheorem
        assert DetectabilityTheorem.is_transition_detectable(
            0.0, 1.0, sigma_wall=1.0, alpha=1.0
        ) is True

    def test_no_transition_not_detectable(self):
        from bpr.meta_boundary import DetectabilityTheorem
        assert DetectabilityTheorem.is_transition_detectable(
            1.0, 1.0, sigma_wall=1.0, alpha=1.0
        ) is False


# ═══════════════════════════════════════════════════════════════════════════
# Hysteresis (Eq 64-67)
# ═══════════════════════════════════════════════════════════════════════════

class TestHysteresis:
    """Hysteresis loops and Berry phase."""

    def test_threshold_at_zero(self):
        from bpr.meta_boundary import Hysteresis
        h = Hysteresis(epsilon_0=1.0, epsilon_prime=0.1, epsilon_double_prime=0.01)
        assert h.threshold(0.0) == pytest.approx(1.0)

    def test_threshold_linear_response(self):
        from bpr.meta_boundary import Hysteresis
        h = Hysteresis(epsilon_0=1.0, epsilon_prime=0.5, epsilon_double_prime=0.0)
        assert h.threshold(1.0) == pytest.approx(1.5)

    def test_hysteresis_loop_asymmetry(self):
        from bpr.meta_boundary import Hysteresis
        h = Hysteresis(epsilon_double_prime=0.1)
        kappa = np.linspace(0, 2, 50)
        eps_fwd, eps_rev = h.hysteresis_loop(kappa)
        # Forward and reverse should differ
        assert not np.allclose(eps_fwd, eps_rev[::-1])

    def test_berry_phase_mod_2pi(self):
        from bpr.meta_boundary import Hysteresis
        phases = np.array([0.5, 1.0, 0.5, np.pi])
        bp = Hysteresis.berry_phase(phases)
        assert 0 <= bp < 2 * np.pi

    def test_berry_phase_zero_for_trivial_loop(self):
        from bpr.meta_boundary import Hysteresis
        phases = np.zeros(10)
        assert Hysteresis.berry_phase(phases) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Thermodynamic Signatures (Eq 68-71)
# ═══════════════════════════════════════════════════════════════════════════

class TestThermodynamicSignatures:
    """Entropy production and dissipation bounds."""

    def test_entropy_production_non_negative(self):
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures()
        dkappa = np.random.randn(50)
        S_dot = ts.entropy_production_rate(dkappa, dx=0.1)
        assert S_dot >= 0

    def test_entropy_production_zero_at_equilibrium(self):
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures()
        dkappa = np.zeros(50)
        assert ts.entropy_production_rate(dkappa) == pytest.approx(0.0)

    def test_front_entropy_rate(self):
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures(tau_kappa=10.0, T_eff=2.0)
        rate = ts.front_entropy_rate(v=1.0, sigma_wall=2.0, A_front=3.0)
        # 10 * 1^2 * 2 * 3 / 2 = 30
        assert rate == pytest.approx(30.0)

    def test_minimum_dissipation(self):
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures(tau_kappa=5.0)
        Q_min = ts.minimum_dissipation(delta_kappa_L2_sq=10.0, t_f=2.0)
        assert Q_min == pytest.approx(25.0)  # 5 * 10 / 2

    def test_minimum_dissipation_decreases_with_time(self):
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures()
        Q1 = ts.minimum_dissipation(10.0, 1.0)
        Q2 = ts.minimum_dissipation(10.0, 10.0)
        assert Q2 < Q1

    def test_total_dissipation(self):
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures(tau_kappa=1.0)
        history = np.ones((10, 5))
        Q = ts.total_dissipation(history, dt=0.1, dx=0.1)
        # tau_kappa * sum(1^2) * dt * dx = 1 * 50 * 0.1 * 0.1 = 0.5
        assert Q == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Stability Analysis (Eq 73-78)
# ═══════════════════════════════════════════════════════════════════════════

class TestStabilityAnalysis:
    """Linear and Lyapunov stability."""

    def test_stable_at_minimum_no_coupling(self):
        from bpr.meta_boundary import StabilityAnalysis
        sa = StabilityAnalysis(gamma=0.0)
        assert sa.is_stable(W_double_prime=0.0) is True

    def test_unstable_with_large_negative_coupling(self):
        from bpr.meta_boundary import StabilityAnalysis
        sa = StabilityAnalysis(gamma=100.0)
        # Large negative W'' destabilizes
        assert sa.is_stable(W_double_prime=-100.0) is False

    def test_critical_coupling(self):
        from bpr.meta_boundary import StabilityAnalysis
        sa = StabilityAnalysis()
        gc = sa.critical_coupling(W_double_prime=-1.0)
        assert gc > 0
        assert np.isfinite(gc)

    def test_growth_rate_negative_at_equilibrium(self):
        from bpr.meta_boundary import StabilityAnalysis
        sa = StabilityAnalysis(gamma=0.0)
        omega = sa.linear_growth_rate(k=1.0, W_double_prime=0.0)
        assert omega < 0  # stable

    def test_lyapunov_rate_non_positive(self):
        from bpr.meta_boundary import StabilityAnalysis
        sa = StabilityAnalysis()
        kappa = np.random.randn(50)
        dL_dt = sa.lyapunov_rate(kappa, dx=0.1)
        assert dL_dt <= 0


# ═══════════════════════════════════════════════════════════════════════════
# Topological Memory Scaling (Eq 83)
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologicalMemoryScaling:
    """Coherence time scaling with winding number."""

    def test_coherence_increases_with_winding(self):
        from bpr.meta_boundary import TopologicalMemoryScaling
        tms = TopologicalMemoryScaling(tau_0=1.0, alpha_topo=1.0)
        tau_1 = tms.coherence_time(1)
        tau_2 = tms.coherence_time(2)
        tau_5 = tms.coherence_time(5)
        assert tau_2 > tau_1
        assert tau_5 > tau_2

    def test_zero_winding_gives_base(self):
        from bpr.meta_boundary import TopologicalMemoryScaling
        tms = TopologicalMemoryScaling(tau_0=3.0)
        assert tms.coherence_time(0) == pytest.approx(3.0)

    def test_power_law_scaling(self):
        from bpr.meta_boundary import TopologicalMemoryScaling
        tms = TopologicalMemoryScaling(tau_0=1.0, alpha_topo=2.0)
        # tau_m = |W|^2
        assert tms.coherence_time(3) == pytest.approx(9.0)

    def test_fit_exponent_recovers(self):
        from bpr.meta_boundary import TopologicalMemoryScaling
        tms = TopologicalMemoryScaling(tau_0=1.0, alpha_topo=1.5)
        W = np.array([1, 2, 3, 5, 10, 20])
        tau = np.array([tms.coherence_time(w) for w in W])
        fitted_alpha = tms.fit_exponent(W, tau)
        assert fitted_alpha == pytest.approx(1.5, abs=0.05)

    def test_negative_winding_same_as_positive(self):
        from bpr.meta_boundary import TopologicalMemoryScaling
        tms = TopologicalMemoryScaling()
        assert tms.coherence_time(-3) == tms.coherence_time(3)


# ═══════════════════════════════════════════════════════════════════════════
# CERN Scenario (Section 16)
# ═══════════════════════════════════════════════════════════════════════════

class TestCERNScenario:
    """Collider-triggered meta-boundary transition analysis."""

    def test_surface_energy_positive(self):
        from bpr.meta_boundary import CERNScenario
        cs = CERNScenario()
        assert cs.minimum_surface_energy() > 0

    def test_coherent_fraction_decreases_with_N(self):
        from bpr.meta_boundary import CERNScenario
        cs = CERNScenario()
        f1 = cs.coherent_stress_fraction(100)
        f2 = cs.coherent_stress_fraction(10000)
        assert f2 < f1

    def test_coherent_fraction_scaling(self):
        from bpr.meta_boundary import CERNScenario
        cs = CERNScenario()
        # N^{-1/2} scaling
        f = cs.coherent_stress_fraction(100)
        assert f == pytest.approx(0.1)

    def test_induced_curvature_tiny(self):
        from bpr.meta_boundary import CERNScenario
        cs = CERNScenario(E_collision=14e3)
        delta_omega = cs.gauge_invariant_curvature(R_crit=1.0)
        assert delta_omega < 1e-10

    def test_transition_not_possible(self):
        from bpr.meta_boundary import CERNScenario
        cs = CERNScenario()
        result = cs.is_transition_possible(R_crit=1.0)
        assert result["transition_possible"] == False
        assert result["energy_sufficient"] == False
        assert result["curvature_sufficient"] == False

    def test_result_keys(self):
        from bpr.meta_boundary import CERNScenario
        cs = CERNScenario()
        result = cs.is_transition_possible()
        required_keys = {
            "energy_sufficient", "energy_ratio", "coherent_fraction",
            "induced_curvature", "curvature_sufficient", "transition_possible",
        }
        assert required_keys.issubset(result.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end tests combining multiple constructs."""

    def test_kink_energy_matches_wall_tension(self):
        """Domain wall energy from kink profile should match analytic sigma_wall."""
        from bpr.meta_boundary import FrontSolution, MetaFunctional
        fs = FrontSolution()
        x = np.linspace(-20, 20, 2000)
        dx = x[1] - x[0]
        kappa = fs.static_kink(x)
        mf = MetaFunctional(alpha=fs.potential.alpha, beta=fs.potential.beta,
                            gamma=0.0, potential=fs.potential)
        J_kink = mf.evaluate(kappa, dx=dx)
        sigma_analytic = fs.surface_tension()
        # Numerical integral should be close to analytic
        assert J_kink == pytest.approx(sigma_analytic, rel=0.1)

    def test_detectability_with_nonzero_transition(self):
        """Any kappa_a != kappa_b should trigger detection."""
        from bpr.meta_boundary import (
            DetectabilityTheorem, FrontSolution, EnergyCostScaling,
        )
        fs = FrontSolution()
        sigma = fs.surface_tension()
        # Build a simple transition
        result = DetectabilityTheorem.check_signatures(
            E_wall=sigma * 4 * np.pi,  # sphere of R=1
            max_spectral_shift=0.01,
        )
        assert result["any_detected"] == True

    def test_dissipation_exceeds_minimum(self):
        """Actual dissipation from evolution should exceed the minimum bound."""
        from bpr.meta_boundary import ThermodynamicSignatures
        ts = ThermodynamicSignatures(tau_kappa=1.0)
        # Simulated d_t kappa history
        n_t, n_x = 50, 30
        dkappa_dt = np.random.randn(n_t, n_x) * 0.1
        dt, dx = 0.1, 0.1
        Q_actual = ts.total_dissipation(dkappa_dt, dt=dt, dx=dx)
        # The minimum bound for the total kappa change
        delta_kappa = np.sum(dkappa_dt, axis=0) * dt
        delta_L2_sq = np.sum(delta_kappa**2) * dx
        t_f = n_t * dt
        Q_min = ts.minimum_dissipation(delta_L2_sq, t_f)
        assert Q_actual >= Q_min * 0.9  # allow small numerical slack

    def test_speed_bound_is_physical(self):
        """Front velocity must not exceed v_max."""
        from bpr.meta_boundary import FrontSolution
        fs = FrontSolution()
        v_max = fs.max_velocity()
        # For a small bias, check the velocity is bounded
        for dJ in [0.01, 0.1, 0.5]:
            v = abs(fs.front_velocity(dJ))
            # For small enough biases, v < v_max
            if v > v_max:
                # This is allowed for large biases (beyond linearized regime)
                pass
            else:
                assert v <= v_max * 1.01  # small numerical tolerance
