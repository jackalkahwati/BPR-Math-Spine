"""
Tests for RPST Stability Manifolds
================================================

Coverage:
  - discrete_sawtooth / sawtooth_derivative utilities
  - FixedPointAnalysis (Eq 4-5)
  - DynamicalMatrix (Eq 8-9)
  - TransferMatrixSpectrum (Thm 2.1, Eq 10)
  - SpectralStabilityDeligne (Thm 3.1)
  - BoundaryPhaseEnergy (Eq 12)
  - StabilityMeasure (Eq 15)
  - LyapunovConvergence (Eq 16-17)
  - AttractorThreshold (Thm 7.1, Eq 19-20)
  - ResonanceBand (Eq 18)
  - PhaseGradientDecoherence (Eq 21)
  - CacheTimescaleDerivation (Cor 8.1)
  - PhaseTransitionThreshold (Thm 9.1)
  - StabilityManifold (Thm 10.1)
"""

import numpy as np
import pytest


# ===========================================================================
# Utility functions
# ===========================================================================

class TestDiscreteSawtooth:
    """Tests for discrete_sawtooth and sawtooth_derivative."""

    def test_sawtooth_zero(self):
        from bpr.stability_manifolds import discrete_sawtooth
        assert discrete_sawtooth(0, 7) == 0

    def test_sawtooth_small_positive(self):
        from bpr.stability_manifolds import discrete_sawtooth
        assert discrete_sawtooth(2, 7) == 2

    def test_sawtooth_wrap_around(self):
        from bpr.stability_manifolds import discrete_sawtooth
        # 5 mod 7 > 7//2 = 3, so 5 - 7 = -2
        assert discrete_sawtooth(5, 7) == -2

    def test_sawtooth_oddness(self):
        """D(-x) = -D(x)."""
        from bpr.stability_manifolds import discrete_sawtooth
        p = 11
        for x in range(1, p):
            assert discrete_sawtooth(x, p) == -discrete_sawtooth(p - x, p)

    def test_sawtooth_derivative_interior(self):
        from bpr.stability_manifolds import sawtooth_derivative
        assert sawtooth_derivative(1, 7) == 1.0

    def test_sawtooth_derivative_zero(self):
        from bpr.stability_manifolds import sawtooth_derivative
        assert sawtooth_derivative(0, 7) == 1.0


# ===========================================================================
# FixedPointAnalysis
# ===========================================================================

class TestFixedPointAnalysis:
    """Tests for fixed-point conditions (Eq 4-5)."""

    def test_constant_config_is_fixed(self):
        """Constant q with zero momentum is a fixed point."""
        from bpr.stability_manifolds import FixedPointAnalysis
        fp = FixedPointAnalysis(p=7)
        N = 4
        q = np.full(N, 3)
        pi_field = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        J[0, 1] = J[1, 0] = 1
        J[1, 2] = J[2, 1] = 1
        J[2, 3] = J[3, 2] = 1
        result = fp.is_fixed_point(q, pi_field, J)
        assert result["is_fixed"] is True
        assert result["momentum_zero"] is True
        assert result["harmonic"] is True

    def test_nonzero_momentum_not_fixed(self):
        from bpr.stability_manifolds import FixedPointAnalysis
        fp = FixedPointAnalysis(p=7)
        q = np.array([3, 3, 3])
        pi_field = np.array([0, 1, 0])
        J = np.eye(3)
        result = fp.is_fixed_point(q, pi_field, J)
        assert result["is_fixed"] is False
        assert result["momentum_zero"] is False

    def test_find_constant_fixed_points(self):
        from bpr.stability_manifolds import FixedPointAnalysis
        fp = FixedPointAnalysis(p=5)
        configs = fp.find_constant_fixed_points(3)
        assert configs.shape == (5, 3)
        # Each row is constant
        for row in configs:
            assert np.all(row == row[0])


# ===========================================================================
# DynamicalMatrix
# ===========================================================================

class TestDynamicalMatrix:
    """Tests for dynamical matrix and graph Laplacian (Eq 8-9)."""

    def test_dynamical_matrix_shape(self):
        from bpr.stability_manifolds import DynamicalMatrix
        dm = DynamicalMatrix(p=7)
        q = np.array([0, 0, 0, 0])
        J = np.zeros((4, 4))
        J[0, 1] = J[1, 0] = 1
        J[1, 2] = J[2, 1] = 1
        J[2, 3] = J[3, 2] = 1
        J[3, 0] = J[0, 3] = 1
        M = dm.dynamical_matrix(q, J)
        assert M.shape == (4, 4)

    def test_graph_laplacian_row_sum_zero(self):
        """Laplacian rows should sum to zero."""
        from bpr.stability_manifolds import DynamicalMatrix
        dm = DynamicalMatrix(p=7)
        q = np.array([0, 0, 0, 0])
        J = np.zeros((4, 4))
        for i in range(4):
            J[i, (i + 1) % 4] = 1
            J[i, (i - 1) % 4] = 1
        L = dm.graph_laplacian(q, J)
        row_sums = np.sum(L, axis=1)
        np.testing.assert_allclose(row_sums, np.zeros(4), atol=1e-12)

    def test_graph_laplacian_positive_semidefinite(self):
        """Laplacian of an undirected graph is positive semi-definite."""
        from bpr.stability_manifolds import DynamicalMatrix
        dm = DynamicalMatrix(p=101)
        q = np.zeros(6, dtype=int)
        J = np.zeros((6, 6))
        for i in range(6):
            J[i, (i + 1) % 6] = 1
            J[i, (i - 1) % 6] = 1
        L = dm.graph_laplacian(q, J)
        eigs = np.real(np.linalg.eigvals(L))
        assert np.all(eigs >= -1e-10)

    def test_transfer_matrix_shape(self):
        from bpr.stability_manifolds import DynamicalMatrix
        dm = DynamicalMatrix(p=7)
        N = 4
        q = np.zeros(N, dtype=int)
        J = np.eye(N)
        T = dm.transfer_matrix(q, J)
        assert T.shape == (2 * N, 2 * N)

    def test_transfer_matrix_det_one(self):
        """Symplectic transfer matrix has det = 1."""
        from bpr.stability_manifolds import DynamicalMatrix
        dm = DynamicalMatrix(p=101)
        N = 3
        q = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        J[0, 1] = J[1, 0] = 0.5
        J[1, 2] = J[2, 1] = 0.5
        T = dm.transfer_matrix(q, J)
        det = np.linalg.det(T)
        assert det == pytest.approx(1.0, abs=1e-8)


# ===========================================================================
# TransferMatrixSpectrum
# ===========================================================================

class TestTransferMatrixSpectrum:
    """Tests for transfer matrix eigenvalue structure (Thm 2.1, Eq 10)."""

    def test_unit_circle_in_band(self):
        """For λ_k ∈ [0, 4], |μ±_k| = 1."""
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        for lk in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
            mu_plus, mu_minus = tms.eigenvalues_from_laplacian(lk)
            assert abs(mu_plus) == pytest.approx(1.0, abs=1e-10)
            assert abs(mu_minus) == pytest.approx(1.0, abs=1e-10)

    def test_conjugate_pair(self):
        """In-band eigenvalues are complex conjugates."""
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        mu_p, mu_m = tms.eigenvalues_from_laplacian(2.0)
        assert mu_p == pytest.approx(mu_m.conjugate(), abs=1e-10)

    def test_out_of_band_exceeds_unit_circle(self):
        """For λ_k > 4, at least one |μ| > 1."""
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        mu_p, mu_m = tms.eigenvalues_from_laplacian(5.0)
        magnitudes = sorted([abs(mu_p), abs(mu_m)])
        assert magnitudes[1] > 1.0

    def test_marginal_stability_all_in_band(self):
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        eigs = np.array([0.0, 1.0, 2.0, 3.0])
        assert tms.is_marginally_stable(eigs) is True

    def test_marginal_stability_out_of_band(self):
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        eigs = np.array([0.0, 1.0, 5.0])
        assert tms.is_marginally_stable(eigs) is False

    def test_resonance_frequencies_range(self):
        """ω_k ∈ [0, π] for λ_k ∈ [0, 4]."""
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        lk = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega = tms.resonance_frequencies(lk)
        assert np.all(omega >= -1e-10)
        assert np.all(omega <= np.pi + 1e-10)

    def test_resonance_frequency_at_lambda_0(self):
        """ω(0) = arccos(1) = 0."""
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        omega = tms.resonance_frequencies(np.array([0.0]))
        assert omega[0] == pytest.approx(0.0, abs=1e-10)

    def test_resonance_frequency_at_lambda_4(self):
        """ω(4) = arccos(-1) = π."""
        from bpr.stability_manifolds import TransferMatrixSpectrum
        tms = TransferMatrixSpectrum()
        omega = tms.resonance_frequencies(np.array([4.0]))
        assert omega[0] == pytest.approx(np.pi, abs=1e-10)


# ===========================================================================
# SpectralStabilityDeligne
# ===========================================================================

class TestSpectralStabilityDeligne:
    """Tests for spectral stability via Deligne (Thm 3.1)."""

    def test_katz_sarnak_density_normalization(self):
        """KS density integrates to 1 over [0, π]."""
        from bpr.stability_manifolds import SpectralStabilityDeligne
        theta = np.linspace(0, np.pi, 10000)
        density = SpectralStabilityDeligne.katz_sarnak_density(theta)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        integral = _trapz(density, theta)
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_katz_sarnak_density_nonnegative(self):
        from bpr.stability_manifolds import SpectralStabilityDeligne
        theta = np.linspace(0, np.pi, 100)
        density = SpectralStabilityDeligne.katz_sarnak_density(theta)
        assert np.all(density >= 0)

    def test_verify_unit_circle_exact(self):
        """Eigenvalues exactly on unit circle pass."""
        from bpr.stability_manifolds import SpectralStabilityDeligne
        ssd = SpectralStabilityDeligne(p=7)
        sqrt7 = np.sqrt(7)
        eigs = sqrt7 * np.exp(1j * np.array([0, 0.5, 1.0, 2.0]))
        result = ssd.verify_unit_circle(eigs)
        assert result["all_on_unit_circle"] is True
        assert result["max_deviation"] < 1e-10

    def test_verify_unit_circle_off(self):
        """Eigenvalues off the circle fail."""
        from bpr.stability_manifolds import SpectralStabilityDeligne
        ssd = SpectralStabilityDeligne(p=7)
        eigs = np.array([3.0 + 0j, 2.0 + 0j])  # |3|/√7 ≈ 1.13
        result = ssd.verify_unit_circle(eigs)
        assert result["max_deviation"] > 0.05

    def test_frobenius_angles_range(self):
        from bpr.stability_manifolds import SpectralStabilityDeligne
        ssd = SpectralStabilityDeligne(p=7)
        eigs = np.sqrt(7) * np.exp(1j * np.linspace(0, np.pi, 5))
        angles = ssd.frobenius_angles(eigs)
        assert np.all(angles >= -1e-10)
        assert np.all(angles <= np.pi + 1e-10)


# ===========================================================================
# BoundaryPhaseEnergy
# ===========================================================================

class TestBoundaryPhaseEnergy:
    """Tests for boundary phase energy E_Φ (Eq 12)."""

    def test_energy_nonnegative(self):
        from bpr.stability_manifolds import BoundaryPhaseEnergy
        bpe = BoundaryPhaseEnergy(kappa_s=1.0)
        phi = np.random.randn(100)
        assert bpe.energy(phi) >= 0

    def test_constant_field_zero_energy(self):
        """Constant φ → ∇Φ = 0 → E = 0."""
        from bpr.stability_manifolds import BoundaryPhaseEnergy
        bpe = BoundaryPhaseEnergy(kappa_s=2.0)
        phi = np.full(50, 1.5)
        assert bpe.energy(phi) == pytest.approx(0.0, abs=1e-12)

    def test_on_manifold_constant(self):
        from bpr.stability_manifolds import BoundaryPhaseEnergy
        bpe = BoundaryPhaseEnergy()
        phi = np.full(20, 3.0)
        assert bpe.is_on_manifold(phi) is True

    def test_not_on_manifold_gradient(self):
        from bpr.stability_manifolds import BoundaryPhaseEnergy
        bpe = BoundaryPhaseEnergy()
        phi = np.linspace(0, 10, 50)
        assert bpe.is_on_manifold(phi) is False

    def test_energy_scales_with_kappa(self):
        from bpr.stability_manifolds import BoundaryPhaseEnergy
        phi = np.linspace(0, 1, 20)
        E1 = BoundaryPhaseEnergy(kappa_s=1.0).energy(phi)
        E2 = BoundaryPhaseEnergy(kappa_s=3.0).energy(phi)
        assert E2 == pytest.approx(3.0 * E1, rel=1e-6)

    def test_energy_2d(self):
        from bpr.stability_manifolds import BoundaryPhaseEnergy
        bpe = BoundaryPhaseEnergy(kappa_s=1.0)
        phi_2d = np.zeros((10, 10))
        assert bpe.energy_2d(phi_2d) == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# StabilityMeasure
# ===========================================================================

class TestStabilityMeasure:
    """Tests for stability measure S(γ) (Eq 15)."""

    def test_nonnegative(self):
        from bpr.stability_manifolds import StabilityMeasure
        sm = StabilityMeasure()
        phi = np.random.randn(50)
        assert sm.compute(phi) >= 0

    def test_zero_for_constant(self):
        from bpr.stability_manifolds import StabilityMeasure
        sm = StabilityMeasure()
        phi = np.full(30, 2.0)
        assert sm.compute(phi) == pytest.approx(0.0, abs=1e-12)

    def test_in_stability_manifold(self):
        from bpr.stability_manifolds import StabilityMeasure
        sm = StabilityMeasure()
        phi = np.full(30, 2.0)
        assert sm.is_in_stability_manifold(phi) is True

    def test_not_in_stability_manifold(self):
        from bpr.stability_manifolds import StabilityMeasure
        sm = StabilityMeasure()
        phi = np.linspace(0, 10, 30)
        assert sm.is_in_stability_manifold(phi) is False

    def test_convergence_trajectory(self):
        from bpr.stability_manifolds import StabilityMeasure
        sm = StabilityMeasure()
        # Trajectory converging to constant
        traj = [np.linspace(0, 10 * np.exp(-t), 20) for t in range(5)]
        S_values = sm.convergence_trajectory(traj)
        assert len(S_values) == 5
        # Should decrease
        assert S_values[-1] < S_values[0]


# ===========================================================================
# LyapunovConvergence
# ===========================================================================

class TestLyapunovConvergence:
    """Tests for Lyapunov convergence (Eq 16-17)."""

    def test_convergence_rate(self):
        from bpr.stability_manifolds import LyapunovConvergence
        lc = LyapunovConvergence(alpha=2.0, mu=0.5)
        assert lc.convergence_rate == pytest.approx(2.0)

    def test_lyapunov_bound_decay(self):
        from bpr.stability_manifolds import LyapunovConvergence
        lc = LyapunovConvergence(alpha=1.0, mu=1.0)
        V0 = 10.0
        t = np.linspace(0, 5, 50)
        bound = lc.lyapunov_bound(V0, t)
        assert bound[0] == pytest.approx(V0)
        assert np.all(np.diff(bound) < 0)

    def test_lyapunov_bound_at_t0(self):
        from bpr.stability_manifolds import LyapunovConvergence
        lc = LyapunovConvergence(alpha=1.0, mu=1.0)
        assert lc.lyapunov_bound(5.0, 0.0) == pytest.approx(5.0)

    def test_lyapunov_rate_negative(self):
        """Rate should be negative when S > 0 and ε = 0."""
        from bpr.stability_manifolds import LyapunovConvergence
        lc = LyapunovConvergence(alpha=1.0, epsilon=0.0)
        assert lc.lyapunov_rate(2.0) < 0

    def test_time_to_threshold(self):
        from bpr.stability_manifolds import LyapunovConvergence
        lc = LyapunovConvergence(alpha=1.0, mu=1.0)
        t = lc.time_to_threshold(V0=10.0, V_target=1.0)
        # t = ln(10)/2 ≈ 1.1513
        assert t == pytest.approx(np.log(10) / 2.0, rel=1e-6)


# ===========================================================================
# AttractorThreshold
# ===========================================================================

class TestAttractorThreshold:
    """Tests for attractor threshold ‖L‖₂ ≤ 4 (Thm 7.1)."""

    def test_in_basin_small_eigenvalues(self):
        from bpr.stability_manifolds import AttractorThreshold
        at = AttractorThreshold()
        eigs = np.array([0.0, 1.0, 2.0, 3.5])
        result = at.check_spectral_condition(eigs)
        assert result["in_basin"] is True
        assert result["violating_modes"] == 0

    def test_out_of_basin_large_eigenvalue(self):
        from bpr.stability_manifolds import AttractorThreshold
        at = AttractorThreshold()
        eigs = np.array([0.0, 1.0, 5.0])
        result = at.check_spectral_condition(eigs)
        assert result["in_basin"] is False
        assert result["violating_modes"] == 1

    def test_coupling_threshold_sawtooth(self):
        from bpr.stability_manifolds import AttractorThreshold
        at = AttractorThreshold()
        assert at.coupling_threshold(D_prime_max=1.0) == pytest.approx(1.0)

    def test_stable_coupling_below(self):
        from bpr.stability_manifolds import AttractorThreshold
        at = AttractorThreshold()
        assert at.is_stable_coupling(J=0.8) is True

    def test_unstable_coupling_above(self):
        from bpr.stability_manifolds import AttractorThreshold
        at = AttractorThreshold()
        assert at.is_stable_coupling(J=1.5) is False

    def test_boundary_coupling(self):
        from bpr.stability_manifolds import AttractorThreshold
        at = AttractorThreshold()
        assert at.is_stable_coupling(J=1.0) is True


# ===========================================================================
# ResonanceBand
# ===========================================================================

class TestResonanceBand:
    """Tests for resonance band structure (Eq 18)."""

    def test_frequency_at_zero(self):
        from bpr.stability_manifolds import ResonanceBand
        rb = ResonanceBand()
        assert rb.frequency(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_frequency_at_four(self):
        from bpr.stability_manifolds import ResonanceBand
        rb = ResonanceBand()
        assert rb.frequency(4.0) == pytest.approx(np.pi, abs=1e-10)

    def test_is_in_band(self):
        from bpr.stability_manifolds import ResonanceBand
        rb = ResonanceBand()
        assert rb.is_in_band(2.0) == True
        assert rb.is_in_band(5.0) == False

    def test_band_structure_ring(self):
        """Ring graph with N=6, J=0.5: all Laplacian eigs ∈ [0, 4]."""
        from bpr.stability_manifolds import ResonanceBand, DynamicalMatrix
        dm = DynamicalMatrix(p=101)
        N = 6
        q = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        for i in range(N):
            J[i, (i + 1) % N] = 0.5
            J[i, (i - 1) % N] = 0.5
        L = dm.graph_laplacian(q, J)
        eigs = np.real(np.linalg.eigvals(L))

        rb = ResonanceBand()
        result = rb.band_structure(eigs)
        assert result["n_out_of_band"] == 0
        assert result["n_in_band"] == N


# ===========================================================================
# PhaseGradientDecoherence
# ===========================================================================

class TestPhaseGradientDecoherence:
    """Tests for decoherence from phase gradients (Eq 21)."""

    def test_zero_gradient_zero_rate(self):
        from bpr.stability_manifolds import PhaseGradientDecoherence
        pgd = PhaseGradientDecoherence(kappa_s=1.0, hbar=1.0)
        grad = np.zeros(10)
        assert pgd.decoherence_rate(grad) == pytest.approx(0.0)

    def test_nonzero_gradient_positive_rate(self):
        from bpr.stability_manifolds import PhaseGradientDecoherence
        pgd = PhaseGradientDecoherence(kappa_s=1.0, hbar=1.0)
        grad = np.array([1.0, 0.5, 0.3])
        assert pgd.decoherence_rate(grad) > 0

    def test_rate_formula(self):
        """Γ_dec = (κ_s/ℏ) ‖∇Φ‖²."""
        from bpr.stability_manifolds import PhaseGradientDecoherence
        pgd = PhaseGradientDecoherence(kappa_s=2.0, hbar=0.5)
        grad = np.array([1.0, 2.0])
        expected = 2.0 / 0.5 * (1.0 + 4.0)
        assert pgd.decoherence_rate(grad) == pytest.approx(expected)

    def test_coherence_time_zero_gradient(self):
        from bpr.stability_manifolds import PhaseGradientDecoherence
        pgd = PhaseGradientDecoherence()
        assert pgd.coherence_time(np.zeros(5)) == np.inf

    def test_coherence_time_inverse(self):
        from bpr.stability_manifolds import PhaseGradientDecoherence
        pgd = PhaseGradientDecoherence(kappa_s=1.0, hbar=1.0)
        grad = np.array([1.0])
        tau = pgd.coherence_time(grad)
        gamma = pgd.decoherence_rate(grad)
        assert tau == pytest.approx(1.0 / gamma)

    def test_from_field_constant(self):
        from bpr.stability_manifolds import PhaseGradientDecoherence
        pgd = PhaseGradientDecoherence()
        phi = np.full(20, 1.0)
        assert pgd.decoherence_rate_from_field(phi) == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# CacheTimescaleDerivation
# ===========================================================================

class TestCacheTimescaleDerivation:
    """Tests for Cache timescale derivation (Cor 8.1)."""

    def test_timescale_zero_winding(self):
        from bpr.stability_manifolds import CacheTimescaleDerivation
        ctd = CacheTimescaleDerivation(tau_0=1.0, alpha=1.0)
        assert ctd.cache_timescale(0) == pytest.approx(1.0)

    def test_timescale_scaling(self):
        """τ_m = τ_0 |W|^α."""
        from bpr.stability_manifolds import CacheTimescaleDerivation
        ctd = CacheTimescaleDerivation(tau_0=2.0, alpha=1.5)
        W = 3
        expected = 2.0 * 3.0 ** 1.5
        assert ctd.cache_timescale(W) == pytest.approx(expected)

    def test_timescale_increases_with_W(self):
        from bpr.stability_manifolds import CacheTimescaleDerivation
        ctd = CacheTimescaleDerivation(tau_0=1.0, alpha=1.0)
        assert ctd.cache_timescale(3) > ctd.cache_timescale(2)
        assert ctd.cache_timescale(2) > ctd.cache_timescale(1)

    def test_gradient_scaling(self):
        """|∇Φ| ~ |W|^{-α/2}."""
        from bpr.stability_manifolds import CacheTimescaleDerivation
        ctd = CacheTimescaleDerivation(tau_0=1.0, alpha=2.0)
        assert ctd.gradient_scaling(4) == pytest.approx(4.0 ** (-1.0))

    def test_decoherence_scaling(self):
        """Γ_dec ~ |W|^{-α}."""
        from bpr.stability_manifolds import CacheTimescaleDerivation
        ctd = CacheTimescaleDerivation(tau_0=1.0, alpha=1.0)
        assert ctd.decoherence_scaling(5) == pytest.approx(1.0 / 5.0)

    def test_inverse_relation(self):
        """τ_m = 1 / Γ_dec (up to constants)."""
        from bpr.stability_manifolds import CacheTimescaleDerivation
        ctd = CacheTimescaleDerivation(tau_0=1.0, alpha=2.0)
        W = 3
        tau = ctd.cache_timescale(W)
        gamma_scale = ctd.decoherence_scaling(W)
        # τ_m * Γ_dec_scaling = τ_0 |W|^α * |W|^{-α} = τ_0
        assert tau * gamma_scale == pytest.approx(ctd.tau_0, rel=1e-10)


# ===========================================================================
# PhaseTransitionThreshold
# ===========================================================================

class TestPhaseTransitionThreshold:
    """Tests for phase transition threshold (Thm 9.1)."""

    def test_stable_positive_V_pp(self):
        from bpr.stability_manifolds import PhaseTransitionThreshold
        ptt = PhaseTransitionThreshold(m_kappa=1.0, alpha_damp=1.0)
        assert ptt.is_linearly_stable(gamma_couple=0.5, V_double_prime=1.0) is True

    def test_unstable_large_negative_V_pp(self):
        from bpr.stability_manifolds import PhaseTransitionThreshold
        ptt = PhaseTransitionThreshold(m_kappa=1.0, alpha_damp=1.0)
        assert ptt.is_linearly_stable(gamma_couple=2.0, V_double_prime=-1.0) is False

    def test_critical_coupling(self):
        """γ_crit = m²_κ α_damp / |V''|."""
        from bpr.stability_manifolds import PhaseTransitionThreshold
        ptt = PhaseTransitionThreshold(m_kappa=2.0, alpha_damp=0.5)
        gamma_c = ptt.critical_coupling(V_double_prime=-1.0)
        # m²α/|V''| = 4*0.5/1 = 2.0
        assert gamma_c == pytest.approx(2.0)

    def test_growth_rate_stable(self):
        from bpr.stability_manifolds import PhaseTransitionThreshold
        ptt = PhaseTransitionThreshold(m_kappa=1.0, alpha_damp=1.0)
        sigma = ptt.growth_rate(gamma_couple=0.5, V_double_prime=1.0, tau_kappa=1.0)
        assert sigma < 0  # Stable


# ===========================================================================
# StabilityManifold
# ===========================================================================

class TestStabilityManifold:
    """Tests for the main result S (Thm 10.1)."""

    def test_constant_config_in_manifold(self):
        from bpr.stability_manifolds import StabilityManifold
        sm = StabilityManifold(p=7)
        N = 4
        q = np.full(N, 3)
        pi_field = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        for i in range(N):
            J[i, (i + 1) % N] = 0.5
            J[i, (i - 1) % N] = 0.5
        result = sm.check_membership(q, pi_field, J)
        assert result["in_manifold"] is True

    def test_nonzero_momentum_not_in_manifold(self):
        from bpr.stability_manifolds import StabilityManifold
        sm = StabilityManifold(p=7)
        q = np.array([3, 3, 3])
        pi_field = np.array([1, 0, 0])
        J = np.eye(3) * 0.5
        result = sm.check_membership(q, pi_field, J)
        assert result["in_manifold"] is False

    def test_continuum_constant_field(self):
        from bpr.stability_manifolds import StabilityManifold
        sm = StabilityManifold(kappa_s=1.0, hbar=1.0)
        phi = np.full(50, 2.0)
        result = sm.continuum_check(phi)
        assert result["in_manifold"] is True
        assert result["E_phi"] == pytest.approx(0.0, abs=1e-12)
        assert result["S_gamma"] == pytest.approx(0.0, abs=1e-12)
        assert result["Gamma_dec"] == pytest.approx(0.0, abs=1e-12)

    def test_continuum_gradient_field_not_in_manifold(self):
        from bpr.stability_manifolds import StabilityManifold
        sm = StabilityManifold()
        phi = np.linspace(0, 5, 50)
        result = sm.continuum_check(phi)
        assert result["in_manifold"] is False
        assert result["E_phi"] > 0
        assert result["Gamma_dec"] > 0

    def test_simulate_returns_trajectory(self):
        from bpr.stability_manifolds import StabilityManifold
        sm = StabilityManifold(p=7)
        N = 4
        q0 = np.array([3, 3, 3, 3])
        pi0 = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        for i in range(N):
            J[i, (i + 1) % N] = 0.5
            J[i, (i - 1) % N] = 0.5
        result = sm.simulate(q0, pi0, J, n_steps=10)
        assert result["q_trajectory"].shape == (11, N)
        assert result["pi_trajectory"].shape == (11, N)
        assert result["S_trajectory"].shape == (11,)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestStabilityManifoldsIntegration:
    """Cross-class integration tests for RPST Stability Manifolds."""

    def test_laplacian_spectrum_determines_stability(self):
        """Ring with small J → all λ ≤ 4 → stable."""
        from bpr.stability_manifolds import (
            DynamicalMatrix, AttractorThreshold, TransferMatrixSpectrum
        )
        N = 6
        q = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        for i in range(N):
            J[i, (i + 1) % N] = 0.3
            J[i, (i - 1) % N] = 0.3
        dm = DynamicalMatrix(p=101)
        L = dm.graph_laplacian(q, J)
        eigs = np.real(np.linalg.eigvals(L))

        at = AttractorThreshold()
        assert at.check_spectral_condition(eigs)["in_basin"] is True

        tms = TransferMatrixSpectrum()
        assert tms.is_marginally_stable(eigs) is True

    def test_energy_and_stability_measure_agree(self):
        """E_Φ = 0 iff S(γ) = 0 for 1D fields."""
        from bpr.stability_manifolds import BoundaryPhaseEnergy, StabilityMeasure
        bpe = BoundaryPhaseEnergy(kappa_s=1.0)
        sm = StabilityMeasure()

        # Constant field: both zero
        phi_const = np.full(30, 1.0)
        assert bpe.is_on_manifold(phi_const) is True
        assert sm.is_in_stability_manifold(phi_const) is True

        # Gradient field: both nonzero
        phi_grad = np.linspace(0, 5, 30)
        assert bpe.is_on_manifold(phi_grad) is False
        assert sm.is_in_stability_manifold(phi_grad) is False

    def test_decoherence_cache_consistency(self):
        """Higher W → lower decoherence → longer cache timescale."""
        from bpr.stability_manifolds import (
            CacheTimescaleDerivation, PhaseGradientDecoherence
        )
        ctd = CacheTimescaleDerivation(tau_0=1.0, alpha=1.0)
        tau1 = ctd.cache_timescale(1)
        tau5 = ctd.cache_timescale(5)
        assert tau5 > tau1

        # Decoherence scaling decreases with W
        gamma1 = ctd.decoherence_scaling(1)
        gamma5 = ctd.decoherence_scaling(5)
        assert gamma5 < gamma1

    def test_fixed_point_implies_manifold_membership(self):
        """A fixed point with small coupling is in S."""
        from bpr.stability_manifolds import StabilityManifold
        sm = StabilityManifold(p=101)
        N = 4
        q = np.full(N, 50)  # Constant
        pi = np.zeros(N, dtype=int)
        J = np.zeros((N, N))
        for i in range(N):
            J[i, (i + 1) % N] = 0.2
            J[i, (i - 1) % N] = 0.2
        result = sm.check_membership(q, pi, J)
        assert result["fixed_point"]["is_fixed"] is True
        assert result["spectral_condition"]["in_basin"] is True
        assert result["in_manifold"] is True
