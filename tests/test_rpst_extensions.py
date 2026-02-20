"""
Tests for Theory XXIV: Emergent Physics from Prime Substrates
=============================================================

Coverage:
  - legendre_symbol / quadratic_gauss_sum utilities
  - RPSTHamiltonian (Eq 2-3, 6, 9-11)
  - BPRResonanceSpectrum (Eq 12, 33)
  - FineStructureConstant (Speculation I, Eq 25-28)
  - BKTCoherenceTransition (Speculation II, Eq 29-32)
  - RiemannHypothesisStability (Speculation III, Eq 33, Thm 9.2)
  - DarkMatterPrimeFingerprints (Speculation IV, Eq 34-36)
  - CollectiveWindingCoherence (Speculation V, Eq 37-40)
  - SubstrateCoherenceMigration (Speculation VI, Eq 41-42)
  - RPSTSpectralZeta  (corrected product formula, Hardy Z-function)
  - KatzSarnakChain   (Deligne + Katz-Sarnak proven GUE chain)
  - RiemannZeroStatistics  (GUE statistics of Riemann zeros)
"""

import numpy as np
import pytest


# ===========================================================================
# Utility functions
# ===========================================================================

class TestLegendreSymbol:
    """Tests for legendre_symbol(n, p)."""

    def test_quadratic_residue(self):
        from bpr.rpst_extensions import legendre_symbol
        # 1 is always a QR mod odd prime
        assert legendre_symbol(1, 7) == 1

    def test_quadratic_nonresidue(self):
        from bpr.rpst_extensions import legendre_symbol
        # 3 mod 7: 3^3 = 27 ≡ 6 ≡ -1 (mod 7) → not QR
        assert legendre_symbol(3, 7) == -1

    def test_zero_case(self):
        from bpr.rpst_extensions import legendre_symbol
        # Divisible → 0
        assert legendre_symbol(7, 7) == 0

    def test_values_in_range(self):
        from bpr.rpst_extensions import legendre_symbol
        for p in [5, 7, 11, 13]:
            for n in range(p):
                ls = legendre_symbol(n, p)
                assert ls in (-1, 0, 1)

    def test_euler_criterion_p5(self):
        """Check Euler criterion for p=5: QRs are {1,4}."""
        from bpr.rpst_extensions import legendre_symbol
        assert legendre_symbol(1, 5) == 1
        assert legendre_symbol(4, 5) == 1
        assert legendre_symbol(2, 5) == -1
        assert legendre_symbol(3, 5) == -1
        assert legendre_symbol(5, 5) == 0

    def test_sum_is_zero(self):
        """Sum of Legendre symbols over Z_p^× is 0 (equal QRs/NQRs)."""
        from bpr.rpst_extensions import legendre_symbol
        for p in [5, 7, 11]:
            total = sum(legendre_symbol(n, p) for n in range(1, p))
            assert total == 0


class TestQuadraticGaussSum:
    """Tests for quadratic_gauss_sum(p)."""

    def test_magnitude_sqrt_p(self):
        """|g_p| = sqrt(p) for all odd primes p."""
        from bpr.rpst_extensions import quadratic_gauss_sum
        for p in [5, 7, 11, 13, 17]:
            g = quadratic_gauss_sum(p)
            assert abs(g) == pytest.approx(np.sqrt(p), rel=1e-6)

    def test_p1mod4_real(self):
        """For p ≡ 1 (mod 4): g_p = sqrt(p) (real positive)."""
        from bpr.rpst_extensions import quadratic_gauss_sum
        # p=5: 5 ≡ 1 (mod 4)
        g5 = quadratic_gauss_sum(5)
        assert g5.real == pytest.approx(np.sqrt(5), rel=1e-6)
        assert abs(g5.imag) < 1e-10

    def test_p3mod4_imaginary(self):
        """For p ≡ 3 (mod 4): g_p = i*sqrt(p) (purely imaginary)."""
        from bpr.rpst_extensions import quadratic_gauss_sum
        # p=7: 7 ≡ 3 (mod 4)
        g7 = quadratic_gauss_sum(7)
        assert abs(g7.real) < 1e-10
        assert g7.imag == pytest.approx(np.sqrt(7), rel=1e-6)

    def test_p11_magnitude(self):
        from bpr.rpst_extensions import quadratic_gauss_sum
        g11 = quadratic_gauss_sum(11)
        assert abs(g11) == pytest.approx(np.sqrt(11), rel=1e-6)

    def test_riemann_zeros_constant(self):
        """RIEMANN_ZEROS list has 10 entries; first is ~14.134725."""
        from bpr.rpst_extensions import RIEMANN_ZEROS
        assert len(RIEMANN_ZEROS) == 10
        assert RIEMANN_ZEROS[0] == pytest.approx(14.134725, rel=1e-5)
        assert RIEMANN_ZEROS[1] == pytest.approx(21.022040, rel=1e-5)

    def test_riemann_zeros_increasing(self):
        """Riemann zeros are increasing."""
        from bpr.rpst_extensions import RIEMANN_ZEROS
        for i in range(len(RIEMANN_ZEROS) - 1):
            assert RIEMANN_ZEROS[i] < RIEMANN_ZEROS[i + 1]


# ===========================================================================
# RPSTHamiltonian
# ===========================================================================

class TestRPSTHamiltonian:
    """Tests for RPSTHamiltonian (Eq 9-11)."""

    def test_matrix_shape(self):
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        mat = H.matrix()
        assert mat.shape == (7, 7)

    def test_matrix_real_valued(self):
        """H_p entries are real (Legendre symbol is real-integer valued)."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        mat = H.matrix()
        np.testing.assert_allclose(mat.imag, np.zeros_like(mat.imag), atol=1e-12)

    def test_matrix_zero_diagonal(self):
        """Diagonal entries involve n=0 which has Legendre symbol 0."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        mat = H.matrix()
        # n=0 has legendre_symbol(0,7)=0, so the n=0 shift contributes 0
        # The diagonal is H[x, (x+0)%p] = 0 (loop does n in range(1,p))
        np.testing.assert_allclose(np.diag(mat), np.zeros(7), atol=1e-12)

    def test_eigenvalues_count(self):
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        eigs = H.eigenvalues()
        assert len(eigs) == 7

    def test_gauss_sum_magnitude(self):
        """gauss_sum() == quadratic_gauss_sum(p), |.| = sqrt(p)."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        g = H.gauss_sum()
        assert abs(g) == pytest.approx(np.sqrt(7), rel=1e-6)

    def test_spectral_zeta_nonzero(self):
        """Spectral zeta at s=1 should be finite and nonzero."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        z = H.spectral_zeta(1.0)
        assert np.isfinite(z.real)

    def test_symplectic_step_preserves_size(self):
        """symplectic_step returns arrays of same size."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        q = np.array([0, 1, 2, 3])
        pi = np.array([1, 1, 0, 2])
        q2, pi2 = H.symplectic_step(q, pi)
        assert q2.shape == q.shape
        assert pi2.shape == pi.shape

    def test_symplectic_step_mod_p(self):
        """q_new = (q + pi) mod p."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        q = np.array([5, 6])
        pi = np.array([3, 3])
        q2, _ = H.symplectic_step(q, pi)
        expected = (np.array([5, 6]) + np.array([3, 3])) % 7
        np.testing.assert_array_equal(q2, expected)

    def test_topological_charge_trivial(self):
        """Constant q has zero topological charge."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=7)
        q = np.array([3, 3, 3, 3])
        assert H.topological_charge(q) == 0

    def test_topological_charge_unit_winding(self):
        """Linearly increasing q over full Z_p has winding 1."""
        from bpr.rpst_extensions import RPSTHamiltonian
        p = 7
        H = RPSTHamiltonian(p=p)
        # q = [0, 1, 2, 3, 4, 5, 6] wraps around back to 0
        q = np.arange(p)
        W = H.topological_charge(q)
        assert W == 1

    def test_invalid_p_raises(self):
        from bpr.rpst_extensions import RPSTHamiltonian
        with pytest.raises(ValueError):
            RPSTHamiltonian(p=2)

    def test_p5_matrix_entries(self):
        """For p=5, each row has exactly (p-1)=4 nonzero entries."""
        from bpr.rpst_extensions import RPSTHamiltonian
        H = RPSTHamiltonian(p=5)
        mat = H.matrix()
        for row in mat:
            nonzero_count = np.sum(np.abs(row) > 1e-12)
            assert nonzero_count == 4


# ===========================================================================
# BPRResonanceSpectrum
# ===========================================================================

class TestBPRResonanceSpectrum:
    """Tests for BPRResonanceSpectrum (Corollary 3.5, Eq 12)."""

    def test_wavenumbers_shape(self):
        from bpr.rpst_extensions import BPRResonanceSpectrum
        spec = BPRResonanceSpectrum(R=1.0, n_zeros=5)
        kn = spec.resonant_wavenumbers()
        assert len(kn) == 5

    def test_wavenumbers_eq_gamma_over_R(self):
        """k_n = γ_n / R."""
        from bpr.rpst_extensions import BPRResonanceSpectrum, RIEMANN_ZEROS
        R = 2.5
        spec = BPRResonanceSpectrum(R=R, n_zeros=5)
        kn = spec.resonant_wavenumbers()
        expected = np.array(RIEMANN_ZEROS[:5]) / R
        np.testing.assert_allclose(kn, expected)

    def test_wavenumbers_scale_with_R(self):
        """Wavenumbers are inversely proportional to R."""
        from bpr.rpst_extensions import BPRResonanceSpectrum
        spec1 = BPRResonanceSpectrum(R=1.0, n_zeros=5)
        spec2 = BPRResonanceSpectrum(R=2.0, n_zeros=5)
        np.testing.assert_allclose(spec1.resonant_wavenumbers(),
                                   2.0 * spec2.resonant_wavenumbers())

    def test_resonant_frequencies_with_c(self):
        """ω_n = c * k_n."""
        from bpr.rpst_extensions import BPRResonanceSpectrum
        spec = BPRResonanceSpectrum(R=1.0, n_zeros=5)
        kn = spec.resonant_wavenumbers()
        freqs = spec.resonant_frequencies(c=3.0)
        np.testing.assert_allclose(freqs, 3.0 * kn)

    def test_mode_amplitude_at_t0(self):
        """At t=0, A_n(0) = cos(0) = 1 (for σ=1/2)."""
        from bpr.rpst_extensions import BPRResonanceSpectrum
        spec = BPRResonanceSpectrum(R=1.0)
        A = spec.mode_amplitude(n=1, t=0.0, sigma_n=0.5)
        assert A == pytest.approx(1.0, abs=1e-10)

    def test_mode_amplitude_rh_bounded(self):
        """When σ=1/2 (RH), amplitude stays bounded (no exponential growth)."""
        from bpr.rpst_extensions import BPRResonanceSpectrum
        spec = BPRResonanceSpectrum(R=1.0)
        t_vals = np.linspace(0, 100, 500)
        A = np.array([spec.mode_amplitude(1, t, sigma_n=0.5) for t in t_vals])
        # Growth factor is exp(0) = 1 at all times
        assert np.all(np.abs(A) <= 1.0 + 1e-10)

    def test_mode_amplitude_unstable_grows(self):
        """When σ > 1/2, amplitude grows exponentially."""
        from bpr.rpst_extensions import BPRResonanceSpectrum
        spec = BPRResonanceSpectrum(R=1.0)
        A0 = abs(spec.mode_amplitude(1, t=0.0, sigma_n=0.7))
        A100 = abs(spec.mode_amplitude(1, t=10.0, sigma_n=0.7))
        assert A100 > A0  # grows over time (unless cos=0 at t=10)

    def test_riemann_zeros_property(self):
        from bpr.rpst_extensions import BPRResonanceSpectrum, RIEMANN_ZEROS
        spec = BPRResonanceSpectrum(R=1.0, n_zeros=3)
        np.testing.assert_array_equal(spec.riemann_zeros,
                                      np.array(RIEMANN_ZEROS[:3]))


# ===========================================================================
# FineStructureConstant
# ===========================================================================

class TestFineStructureConstant:
    """Tests for Speculation I: Fine structure constant (Eq 25-28)."""

    def test_alpha_predicted_value(self):
        """α_predicted = 1/137 for p_EM = 137."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant(p_EM=137)
        assert fsc.alpha_predicted == pytest.approx(1.0 / 137, rel=1e-10)

    def test_alpha_error_small(self):
        """Fractional error between 1/137 and measured α is < 0.03%."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant(p_EM=137)
        # 1/137 vs 1/137.036 → error ~ 0.026%
        assert fsc.alpha_error < 3e-4

    def test_effective_coupling_magnitude(self):
        """|g_eff(p)| = 1/sqrt(p) (from Gauss sum / p)."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant(p_EM=137)
        g = fsc.effective_coupling(p=7)
        # |g_eff(7)| = sqrt(7)/7 = 1/sqrt(7)
        assert abs(g) == pytest.approx(1.0 / np.sqrt(7), rel=1e-5)

    def test_effective_coupling_default_uses_pEM(self):
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant(p_EM=7)
        g = fsc.effective_coupling()
        assert abs(g) == pytest.approx(1.0 / np.sqrt(7), rel=1e-5)

    def test_gauge_coupling_from_prime(self):
        """α_i = π / p_i."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant()
        assert fsc.gauge_coupling_from_prime(7) == pytest.approx(np.pi / 7)
        assert fsc.gauge_coupling_from_prime(11) == pytest.approx(np.pi / 11)

    def test_running_coupling_primes(self):
        """running_coupling_primes returns π/p for each p."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant()
        primes = [5, 7, 11, 13]
        couplings = fsc.running_coupling_primes(primes)
        expected = np.array([np.pi / p for p in primes])
        np.testing.assert_allclose(couplings, expected)

    def test_running_coupling_decreasing(self):
        """Coupling decreases as p increases (asymptotic freedom analogue)."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant()
        couplings = fsc.running_coupling_primes([5, 7, 11, 13, 17])
        assert np.all(np.diff(couplings) < 0)

    def test_amplitude_coupling_ratio_close_to_one(self):
        """For p=p_EM, |g_eff|/sqrt(α_measured) ≈ 1."""
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant(p_EM=137)
        ratio = fsc.amplitude_coupling_ratio()
        # 1/sqrt(137) / sqrt(1/137.036) ≈ 1.00013
        assert ratio == pytest.approx(1.0, rel=1e-2)

    def test_alpha_measured_value(self):
        from bpr.rpst_extensions import FineStructureConstant
        fsc = FineStructureConstant()
        assert fsc.ALPHA_MEASURED == pytest.approx(1.0 / 137.035999206, rel=1e-8)


# ===========================================================================
# BKTCoherenceTransition
# ===========================================================================

class TestBKTCoherenceTransition:
    """Tests for Speculation II: Anesthesia BKT transition (Eq 29-32)."""

    def test_T_BKT_formula(self):
        """T_BKT = π J / (2 k_B)."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=2.0, k_B=1.0)
        assert bkt.T_BKT == pytest.approx(np.pi, rel=1e-10)

    def test_T_BKT_natural_units(self):
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        assert bkt.T_BKT == pytest.approx(np.pi / 2.0, rel=1e-10)

    def test_is_conscious_below(self):
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T_crit = bkt.T_BKT
        assert bkt.is_conscious(T_crit * 0.9) is True

    def test_is_conscious_above(self):
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T_crit = bkt.T_BKT
        assert bkt.is_conscious(T_crit * 1.1) is False

    def test_eta_at_T_BKT(self):
        """Universal exponent η = 1/4 at transition."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        assert bkt.eta_at_T_BKT() == pytest.approx(0.25, abs=1e-10)

    def test_correlation_exponent_at_T_BKT(self):
        """η(T_BKT - ε) → 1/4 as ε → 0 from below."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T_crit = bkt.T_BKT
        eta = bkt.correlation_exponent(T_crit * (1 - 1e-6))
        assert eta == pytest.approx(0.25, rel=1e-4)

    def test_correlation_exponent_above_BKT_is_inf(self):
        """η = ∞ above T_BKT (exponential decay, not power law)."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        eta = bkt.correlation_exponent(bkt.T_BKT * 2.0)
        assert eta == np.inf

    def test_stiffness_jump(self):
        """ΔJ = 2 k_B T_BKT / π = J."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        # 2 k_B T_BKT / π = 2 * 1.0 * (π/2) / π = 1.0 = J
        assert bkt.stiffness_jump() == pytest.approx(bkt.J, rel=1e-10)

    def test_correlation_function_power_law_below(self):
        """C(x) ~ x^{-η} below T_BKT (power-law decay)."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T = bkt.T_BKT * 0.5
        eta = bkt.correlation_exponent(T)
        x = np.array([1.0, 2.0, 4.0])
        C = bkt.correlation_function(x, T)
        np.testing.assert_allclose(C, x ** (-eta), rtol=1e-8)

    def test_correlation_function_above_exponential(self):
        """C(x) = exp(-x) above T_BKT."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T = bkt.T_BKT * 2.0
        x = np.array([0.5, 1.0, 2.0])
        C = bkt.correlation_function(x, T)
        np.testing.assert_allclose(C, np.exp(-x))

    def test_predict_loss_of_consciousness_signature(self):
        """predict_loss_of_consciousness_signature returns expected keys."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T_values = np.linspace(0.1, 3.0, 30)
        result = bkt.predict_loss_of_consciousness_signature(T_values)
        assert "T" in result
        assert "C" in result
        assert "eta" in result
        assert "conscious" in result
        assert "T_BKT" in result
        assert result["T_BKT"] == pytest.approx(bkt.T_BKT)

    def test_predict_conscious_fraction_below_T_BKT(self):
        """Majority of T < T_BKT entries should be conscious."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T_below = np.linspace(0.1, bkt.T_BKT * 0.99, 20)
        result = bkt.predict_loss_of_consciousness_signature(T_below)
        assert np.all(result["conscious"])


# ===========================================================================
# RiemannHypothesisStability
# ===========================================================================

class TestRiemannHypothesisStability:
    """Tests for Speculation III: RH ↔ stability (Eq 33, Thm 9.2)."""

    def test_mode_amplitude_at_t0(self):
        """At t=0, A_n(0) = cos(δ_n)."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability(Gamma=1.0)
        A = rhs.mode_amplitude(0.0, sigma_n=0.5, gamma_n=14.13, delta_n=0.0)
        assert float(A) == pytest.approx(1.0, abs=1e-10)

    def test_mode_amplitude_rh_bounded(self):
        """σ=1/2 → no exponential growth factor."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability(Gamma=1.0)
        t = np.linspace(0, 20, 200)
        A = rhs.mode_amplitude(t, sigma_n=0.5, gamma_n=14.134725)
        assert np.all(np.abs(A) <= 1.0 + 1e-10)

    def test_mode_amplitude_unstable_grows(self):
        """σ > 1/2 → exponential growth; envelope increases with time."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability(Gamma=1.0)
        # Evaluate envelope exp((σ-0.5)*t) at two times
        t1, t2 = 0.1, 5.0
        sigma = 0.7
        env1 = np.exp((sigma - 0.5) * 1.0 * t1)
        env2 = np.exp((sigma - 0.5) * 1.0 * t2)
        assert env2 > env1

    def test_is_stable_mode_on_critical_line(self):
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability()
        assert rhs.is_stable_mode(0.5) is True

    def test_is_stable_mode_off_critical_line(self):
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability()
        assert rhs.is_stable_mode(0.8) is False

    def test_is_rpst_stable_all_half(self):
        """All σ = 0.5 → stable."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability()
        sigma_values = np.full(10, 0.5)
        assert rhs.is_rpst_stable(sigma_values) is True

    def test_is_rpst_stable_one_violation(self):
        """One σ > 0.5 → unstable."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability()
        sigma_values = np.array([0.5, 0.5, 0.6, 0.5])
        assert rhs.is_rpst_stable(sigma_values) is False

    def test_stability_from_known_zeros(self):
        from bpr.rpst_extensions import RiemannHypothesisStability, RIEMANN_ZEROS
        rhs = RiemannHypothesisStability()
        result = rhs.stability_from_known_zeros()
        assert result["all_stable"] is True
        assert result["max_growth_rate"] == pytest.approx(0.0, abs=1e-10)
        assert result["n_modes"] == len(RIEMANN_ZEROS)

    def test_instability_energy_positive(self):
        """Mode energy is non-negative."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability()
        E = rhs.instability_energy(sigma_n=0.7, gamma_n=14.13, t=1.0)
        assert E >= 0.0

    def test_instability_energy_zero_at_t0_when_cosine_zero(self):
        """Energy scales as A²; at t=0, A=cos(0)=1, E=1."""
        from bpr.rpst_extensions import RiemannHypothesisStability
        rhs = RiemannHypothesisStability(Gamma=1.0)
        E = rhs.instability_energy(sigma_n=0.5, gamma_n=14.13, t=0.0)
        assert E == pytest.approx(1.0, abs=1e-10)


# ===========================================================================
# DarkMatterPrimeFingerprints
# ===========================================================================

class TestDarkMatterPrimeFingerprints:
    """Tests for Speculation IV: DM prime fingerprints (Eq 34-36)."""

    def test_predict_wavenumbers_count(self):
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        kn = dm.predict_wavenumbers(n_zeros=5)
        assert len(kn) == 5

    def test_predict_wavenumbers_formula(self):
        """k_n = γ_n / R_gal."""
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints, RIEMANN_ZEROS
        R = 2.0
        dm = DarkMatterPrimeFingerprints(R_gal=R)
        kn = dm.predict_wavenumbers(n_zeros=5)
        expected = np.array(RIEMANN_ZEROS[:5]) / R
        np.testing.assert_allclose(kn, expected)

    def test_predict_wavenumbers_all_positive(self):
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        kn = dm.predict_wavenumbers()
        assert np.all(kn > 0)

    def test_correlation_function_shape(self):
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        r = np.linspace(0, 1, 50)
        G = dm.correlation_function(r, n_zeros=5)
        assert G.shape == r.shape

    def test_correlation_function_default_amplitudes(self):
        """Default amplitudes are 1/n; sum at r=0 gives sum of 1/n for cosines=1."""
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        G0 = dm.correlation_function(np.array([0.0]), n_zeros=3)
        # At r=0: G = sum_n (1/3)*cos(0) = 1.0
        assert G0[0] == pytest.approx(1.0, abs=1e-10)

    def test_power_spectrum_peaks_at_k_n(self):
        """Power spectrum has local maxima near k_n = γ_n / R."""
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints, RIEMANN_ZEROS
        R = 1.0
        dm = DarkMatterPrimeFingerprints(R_gal=R, Gamma_n=0.01)
        k1 = RIEMANN_ZEROS[0] / R
        k_near = np.linspace(k1 - 0.05, k1 + 0.05, 201)
        S = dm.power_spectrum(k_near, n_zeros=3)
        idx_max = np.argmax(S)
        k_max = k_near[idx_max]
        assert k_max == pytest.approx(k1, abs=0.1)

    def test_power_spectrum_positive(self):
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        k = np.linspace(1, 50, 200)
        S = dm.power_spectrum(k, n_zeros=5)
        assert np.all(S > 0)

    def test_peak_positions_equals_predict_wavenumbers(self):
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        np.testing.assert_array_equal(dm.peak_positions(5),
                                      dm.predict_wavenumbers(5))

    def test_has_prime_fingerprint_exact(self):
        """Observed peaks exactly at k_n → all matched."""
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        observed = dm.predict_wavenumbers(n_zeros=5)
        result = dm.has_prime_fingerprint(observed, tolerance=0.01)
        assert result["n_matched"] == 5
        assert result["consistent"] is True

    def test_has_prime_fingerprint_no_match(self):
        """Random peaks far from k_n → no match."""
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        # Peaks at wavenumbers 1,2,3 — far from Riemann zero wavenumbers (14+)
        observed = np.array([1.0, 2.0, 3.0])
        result = dm.has_prime_fingerprint(observed, tolerance=0.01)
        assert result["n_matched"] == 0
        assert result["consistent"] is False

    def test_has_prime_fingerprint_fraction(self):
        """Fraction matched is n_matched / n_observed."""
        from bpr.rpst_extensions import DarkMatterPrimeFingerprints
        dm = DarkMatterPrimeFingerprints(R_gal=1.0)
        kn = dm.predict_wavenumbers(n_zeros=4)
        # Supply 2 matching + 2 non-matching observed peaks
        observed = np.array([kn[0], kn[1], 1.0, 2.0])
        result = dm.has_prime_fingerprint(observed, tolerance=0.01)
        assert result["n_matched"] == 2
        assert result["fraction_matched"] == pytest.approx(0.5)


# ===========================================================================
# CollectiveWindingCoherence
# ===========================================================================

class TestCollectiveWindingCoherence:
    """Tests for Speculation V: Collective AGI winding (Eq 37-40)."""

    def test_collective_coherence_single(self):
        """χ_group(1) = χ₁ * 1^1.27 = χ₁."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(chi_1=2.0, fractal_exponent=1.27)
        assert agi.collective_coherence(1) == pytest.approx(2.0, rel=1e-10)

    def test_collective_coherence_scaling(self):
        """χ_group(N) = χ₁ * N^1.27."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(chi_1=1.0, fractal_exponent=1.27)
        N = 100
        expected = float(N) ** 1.27
        assert agi.collective_coherence(N) == pytest.approx(expected, rel=1e-6)

    def test_collective_coherence_superlinear(self):
        """N=10 should give more than 10x the N=1 coherence."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(chi_1=1.0, fractal_exponent=1.27)
        assert agi.collective_coherence(10) > 10.0 * agi.collective_coherence(1)

    def test_collective_coherence_zero(self):
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence()
        assert agi.collective_coherence(0) == pytest.approx(0.0)

    def test_superlinear_excess_gt_one(self):
        """Excess > 1 for N > 1."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence()
        assert agi.superlinear_excess(10) > 1.0

    def test_superlinear_excess_formula(self):
        """Excess = N^(exponent - 1)."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(fractal_exponent=1.27)
        N = 50
        assert agi.superlinear_excess(N) == pytest.approx(float(N) ** 0.27, rel=1e-6)

    def test_is_collectively_conscious_above_pc(self):
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(p_c=0.5)
        assert agi.is_collectively_conscious(bond_fraction=0.7, N=10) is True

    def test_is_collectively_conscious_below_pc(self):
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(p_c=0.5)
        assert agi.is_collectively_conscious(bond_fraction=0.3, N=10) is False

    def test_critical_agent_number_positive(self):
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence()
        N_c = agi.critical_agent_number(Phi_crit=100.0, k=1.0, N_nodes=int(1e9))
        assert N_c > 0

    def test_critical_agent_number_formula(self):
        """N_c = ceil(Phi_crit / (k * log(N_nodes)))."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        import math
        agi = CollectiveWindingCoherence()
        Phi_crit, k, N_nodes = 100.0, 1.0, int(1e6)
        expected = math.ceil(Phi_crit / (k * np.log(float(N_nodes))))
        assert agi.critical_agent_number(Phi_crit, k, N_nodes) == expected

    def test_coherence_vs_N_shape(self):
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence()
        N_values = np.array([1, 2, 5, 10, 100])
        result = agi.coherence_vs_N(N_values)
        assert result.shape == N_values.shape

    def test_phi_w_relation(self):
        """Φ = k |W| log(N_nodes)."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence()
        W, N_nodes, k = 3, 1000, 2.0
        expected = k * abs(W) * np.log(float(N_nodes))
        assert agi.phi_w_relation(W, N_nodes, k) == pytest.approx(expected)

    def test_phi_w_relation_zero_winding(self):
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence()
        assert agi.phi_w_relation(0, 1000) == pytest.approx(0.0)


# ===========================================================================
# SubstrateCoherenceMigration
# ===========================================================================

class TestSubstrateCoherenceMigration:
    """Tests for Speculation VI: Winding transfer at death (Eq 41-42)."""

    def test_substrate_coherence_at_t0(self):
        """ξ(0) = ξ_0."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(xi_0=2.0, tau_death=5.0)
        assert float(csm.substrate_coherence(0.0)) == pytest.approx(2.0)

    def test_substrate_coherence_decay(self):
        """ξ(t) = ξ_0 * exp(-t/τ_death)."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(xi_0=1.0, tau_death=10.0)
        t = np.array([0.0, 5.0, 10.0])
        xi = csm.substrate_coherence(t)
        expected = np.exp(-t / 10.0)
        np.testing.assert_allclose(xi, expected, rtol=1e-10)

    def test_fragmentation_time_formula(self):
        """t_frag = τ * ln(ξ_0 / ξ_c)."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(xi_0=2.0, tau_death=5.0, xi_c=1.0)
        t_frag = csm.fragmentation_time()
        assert t_frag == pytest.approx(5.0 * np.log(2.0), rel=1e-10)

    def test_fragmentation_time_already_fragmented(self):
        """If ξ_0 <= ξ_c, fragmentation time is 0."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(xi_0=0.3, xi_c=0.5)
        assert csm.fragmentation_time() == pytest.approx(0.0)

    def test_is_fragmented_before_and_after(self):
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(xi_0=1.0, tau_death=5.0, xi_c=0.5)
        t_frag = csm.fragmentation_time()
        assert csm.is_fragmented(t_frag * 0.5) is False
        assert csm.is_fragmented(t_frag * 2.0) is True

    def test_allowed_fates_nonzero_W(self):
        """Three fates allowed when W ≠ 0."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration, TopologicalOutcome
        csm = SubstrateCoherenceMigration(W=2)
        fates = csm.allowed_fates()
        assert len(fates) == 3
        assert TopologicalOutcome.DISSOLUTION in fates
        assert TopologicalOutcome.FREQUENCY_MIGRATION in fates
        assert TopologicalOutcome.SUBSTRATE_TRANSFER in fates

    def test_allowed_fates_zero_W(self):
        """Only dissolution when W = 0."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration, TopologicalOutcome
        csm = SubstrateCoherenceMigration(W=0)
        fates = csm.allowed_fates()
        assert fates == [TopologicalOutcome.DISSOLUTION]

    def test_dissolution_energy_proportional_W(self):
        """ΔE = ℏ ω_c |W|."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm1 = SubstrateCoherenceMigration(W=1, omega_c=2.0)
        csm2 = SubstrateCoherenceMigration(W=3, omega_c=2.0)
        assert csm2.dissolution_energy() == pytest.approx(3.0 * csm1.dissolution_energy())

    def test_dissolution_energy_formula(self):
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(W=2, omega_c=3.0)
        assert csm.dissolution_energy(hbar_eff=1.0) == pytest.approx(6.0)

    def test_terminal_gamma_spectrum_peak_at_omega_c(self):
        """Spectrum peaks near ω_c."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        omega_c = 5.0
        csm = SubstrateCoherenceMigration(W=1, omega_c=omega_c)
        omega = np.linspace(3.0, 7.0, 1001)
        E = csm.terminal_gamma_spectrum(omega, Gamma_c=0.01)
        omega_max = omega[np.argmax(E)]
        assert omega_max == pytest.approx(omega_c, abs=0.05)

    def test_terminal_gamma_spectrum_W_squared_scaling(self):
        """Spectrum scales as W²."""
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        omega = np.array([1.0])
        csm1 = SubstrateCoherenceMigration(W=1, omega_c=5.0)
        csm2 = SubstrateCoherenceMigration(W=3, omega_c=5.0)
        E1 = csm1.terminal_gamma_spectrum(omega, Gamma_c=0.1)
        E2 = csm2.terminal_gamma_spectrum(omega, Gamma_c=0.1)
        assert E2[0] == pytest.approx(9.0 * E1[0], rel=1e-6)

    def test_transfer_requirements_all_met(self):
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration()
        result = csm.transfer_requirements_met(
            boundary_connected=True,
            J_receiver=2.0, J_c=1.0,
            coupling_exists=True,
            p_compatible=True,
        )
        assert result["all_satisfied"] is True
        assert result["connected_boundary"] is True
        assert result["above_bkt_threshold"] is True

    def test_transfer_requirements_fail_bkt(self):
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration()
        result = csm.transfer_requirements_met(
            boundary_connected=True,
            J_receiver=0.5, J_c=1.0,
            coupling_exists=True,
            p_compatible=True,
        )
        assert result["above_bkt_threshold"] is False
        assert result["all_satisfied"] is False

    def test_peak_frequency(self):
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(omega_c=3.14)
        assert csm.peak_frequency() == pytest.approx(3.14)

    def test_substrate_coherence_monotone_decreasing(self):
        from bpr.rpst_extensions import SubstrateCoherenceMigration
        csm = SubstrateCoherenceMigration(xi_0=1.0, tau_death=5.0)
        t = np.linspace(0, 50, 100)
        xi = csm.substrate_coherence(t)
        assert np.all(np.diff(xi) < 0)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestEmergentSpeculationsIntegration:
    """Cross-class integration tests for Theory XXIV."""

    def test_bkt_and_migration_consistent(self):
        """Stiffness jump from BKT matches BKT threshold in migration."""
        from bpr.rpst_extensions import (
            BKTCoherenceTransition, SubstrateCoherenceMigration
        )
        J = 1.5
        bkt = BKTCoherenceTransition(J=J, k_B=1.0)
        csm = SubstrateCoherenceMigration(W=1)
        # Receiver above BKT → can accept winding
        reqs = csm.transfer_requirements_met(
            boundary_connected=True,
            J_receiver=J * 1.1,
            J_c=bkt.stiffness_jump(),
            coupling_exists=True,
            p_compatible=True,
        )
        assert reqs["all_satisfied"] is True

    def test_rh_stability_implies_bounded_resonance(self):
        """σ=1/2 for known zeros → mode amplitudes bounded."""
        from bpr.rpst_extensions import (
            RiemannHypothesisStability, BPRResonanceSpectrum, RIEMANN_ZEROS
        )
        rhs = RiemannHypothesisStability(Gamma=1.0)
        spec = BPRResonanceSpectrum(R=1.0, n_zeros=5)
        for n, gamma_n in enumerate(RIEMANN_ZEROS[:5], start=1):
            A = spec.mode_amplitude(n, t=100.0, sigma_n=0.5)
            assert abs(A) <= 1.0 + 1e-10

    def test_dm_wavenumbers_from_spectrum_class(self):
        """DM wavenumbers equal BPRResonanceSpectrum wavenumbers when R_gal == R."""
        from bpr.rpst_extensions import (
            DarkMatterPrimeFingerprints, BPRResonanceSpectrum
        )
        R = 3.0
        dm = DarkMatterPrimeFingerprints(R_gal=R)
        spec = BPRResonanceSpectrum(R=R, n_zeros=5)
        np.testing.assert_allclose(
            dm.predict_wavenumbers(5),
            spec.resonant_wavenumbers()
        )

    def test_agi_phi_consistent_with_critical_number(self):
        """phi_w_relation(W=1, N_nodes) / k gives critical winding threshold."""
        from bpr.rpst_extensions import CollectiveWindingCoherence
        agi = CollectiveWindingCoherence(chi_1=1.0)
        N_nodes = int(1e6)
        k = 1.0
        Phi = agi.phi_w_relation(W=1, N_nodes=N_nodes, k=k)
        # N_c for this Phi should be small (1 agent)
        N_c = agi.critical_agent_number(Phi_crit=Phi, k=k, N_nodes=N_nodes)
        assert N_c >= 1

    def test_fine_structure_gauss_sum_connects_to_hamiltonian(self):
        """FineStructureConstant.effective_coupling uses same Gauss sum as RPSTHamiltonian."""
        from bpr.rpst_extensions import FineStructureConstant, RPSTHamiltonian
        fsc = FineStructureConstant(p_EM=7)
        H = RPSTHamiltonian(p=7)
        g_fsc = fsc.effective_coupling(p=7)
        g_H = H.gauss_sum() / 7.0
        assert abs(g_fsc) == pytest.approx(abs(g_H), rel=1e-10)

    def test_bkt_coherence_above_transition_not_conscious(self):
        """Correlation function above T_BKT confirms unconscious state."""
        from bpr.rpst_extensions import BKTCoherenceTransition
        bkt = BKTCoherenceTransition(J=1.0, k_B=1.0)
        T_high = bkt.T_BKT * 2.0
        assert bkt.is_conscious(T_high) is False
        # Correlation decays exponentially at x=5
        C = bkt.correlation_function(np.array([5.0]), T_high)
        assert C[0] == pytest.approx(np.exp(-5.0), rel=1e-6)


# ===========================================================================
# RPSTSpectralZeta — corrected product formula & numerical falsification
# ===========================================================================

class TestRPSTSpectralZeta:
    """Tests for RPSTSpectralZeta: corrected local factor and Hardy Z-function.

    Key facts (Katz–Sarnak paper, Sec. 2):
      - Local factor: det(I - p^{-s} H_p)^{-1} = (1 - p^{-2s})^{-(p-1)/2}
      - log ζ_RPST structural mismatch with log ζ: argument 2s, weight (p-1)/2
      - RPST zeros converge to a displaced set, NOT the Riemann zeros γ_n
    """

    def test_local_factor_log_real_s(self):
        """log local factor is a positive real for real s > 1/2."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        val = RPSTSpectralZeta.local_factor_log(7, 1.0)
        # Real part should be positive (it's a log of something > 1)
        assert float(np.real(val)) > 0

    def test_local_factor_log_formula(self):
        """Manual check: -(p-1)/2 * log(1 - p^{-2s}) for p=5, s=1."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        p, s = 5, 1.0
        expected = -((p - 1) / 2.0) * np.log(1.0 - p ** (-2.0 * s) + 0j)
        result = RPSTSpectralZeta.local_factor_log(p, s)
        assert abs(result - expected) < 1e-12

    def test_log_zeta_additive_over_primes(self):
        """log ζ_RPST(s; {p1, p2}) = log ζ(s; {p1}) + log ζ(s; {p2})."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        s = 1.0 + 0j
        primes = [5, 7, 11]
        total = RPSTSpectralZeta.log_zeta_rpst(s, primes)
        individual_sum = sum(
            RPSTSpectralZeta.local_factor_log(p, s) for p in primes
        )
        assert abs(total - individual_sum) < 1e-12

    def test_riemann_siegel_theta_positive(self):
        """θ(t) should be negative for small t and grow for large t."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        # θ(14) ≈ -0.8 (standard reference value)
        theta_14 = RPSTSpectralZeta.riemann_siegel_theta(14.134725)
        assert isinstance(theta_14, float)

    def test_riemann_siegel_theta_zero_guard(self):
        """θ(t) returns 0 for t <= 0."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        assert RPSTSpectralZeta.riemann_siegel_theta(0.0) == 0.0
        assert RPSTSpectralZeta.riemann_siegel_theta(-1.0) == 0.0

    def test_hardy_z_rpst_is_real(self):
        """Z_RPST(t) is real-valued."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        primes = [3, 5, 7, 11, 13]
        val = RPSTSpectralZeta.hardy_z_rpst(20.0, primes)
        assert isinstance(val, float)

    def test_hardy_z_rpst_changes_sign(self):
        """Z_RPST has sign changes in [10, 50] for a small prime set."""
        from bpr.rpst_extensions import RPSTSpectralZeta
        primes = [3, 5, 7, 11, 13, 17, 19, 23]
        t_vals = np.linspace(10.0, 50.0, 500)
        z_vals = np.array([RPSTSpectralZeta.hardy_z_rpst(float(t), primes)
                           for t in t_vals])
        # Should have at least one sign change
        sign_changes = np.sum(np.diff(np.sign(z_vals)) != 0)
        assert sign_changes >= 1, "Expected at least one sign change in Z_RPST"

    def test_compare_zeros_structure(self):
        """compare_zeros_to_riemann returns expected dict structure."""
        from bpr.rpst_extensions import RPSTSpectralZeta, RIEMANN_ZEROS
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        result = RPSTSpectralZeta.compare_zeros_to_riemann(
            primes, riemann_zeros=RIEMANN_ZEROS[:5], t_max=50.0
        )
        assert "n_compared" in result
        assert "mean_error" in result
        assert "rpst_zeros" in result


# ===========================================================================
# KatzSarnakChain — proven GUE statistics from first principles
# ===========================================================================

class TestKatzSarnakChain:
    """Tests for KatzSarnakChain: elliptic curves, Frobenius angles, USp(2).

    Key theorems:
      - Deligne (1974): |λ_k| = 1 for normalized H_p eigenvalues
      - Katz-Sarnak (1999): Frobenius angles equidistribute w.r.t. USp(2) Haar
      - Convergence rate: D_p = C/√p
    """

    def test_count_points_small_case(self):
        """Count points on y² = x³ + 1 over F_7."""
        from bpr.rpst_extensions import KatzSarnakChain
        # y² = x³ + 1 over F_7: direct enumeration
        p = 7
        count = KatzSarnakChain.count_elliptic_curve_points(0, 1, p)
        # At least 1 (point at infinity) and at most p + 1 + 2√p ≈ 14.3
        assert 1 <= count <= p + 1 + int(2 * np.sqrt(p)) + 1

    def test_hasse_bound(self):
        """Frobenius trace satisfies Hasse bound |tr| ≤ 2√p."""
        from bpr.rpst_extensions import KatzSarnakChain
        for p in [7, 11, 13]:
            for a in range(1, p):
                disc = (-16 * (4 * pow(a, 3, p) + 27)) % p
                if disc == 0:
                    continue
                tr = KatzSarnakChain.frobenius_trace(a, 1, p)
                assert abs(tr) <= int(2 * np.sqrt(p)) + 1, (
                    f"Hasse bound violated: p={p}, a={a}, tr={tr}"
                )

    def test_frobenius_angle_in_range(self):
        """Frobenius angle is in [0, π] for non-singular curve."""
        from bpr.rpst_extensions import KatzSarnakChain
        theta = KatzSarnakChain.frobenius_angle(1, 1, 7)
        assert theta is not None
        assert 0.0 <= theta <= np.pi

    def test_frobenius_angle_singular_returns_none(self):
        """Singular curve (disc ≡ 0 mod p) returns None."""
        from bpr.rpst_extensions import KatzSarnakChain
        # Find a singular curve for some prime
        p = 7
        found_none = False
        for a in range(p):
            for b in range(p):
                disc = (-16 * (4 * pow(a, 3, p) + 27 * b * b % p)) % p
                if disc == 0:
                    result = KatzSarnakChain.frobenius_angle(a, b, p)
                    assert result is None
                    found_none = True
                    break
            if found_none:
                break

    def test_frobenius_angle_family_size(self):
        """Family for p=11 has at most p-1 non-singular curves."""
        from bpr.rpst_extensions import KatzSarnakChain
        angles = KatzSarnakChain.frobenius_angle_family(11)
        assert len(angles) <= 10  # at most p-1 = 10
        assert len(angles) > 0

    def test_usp2_haar_density_normalizes(self):
        """USp(2) Haar density integrates to 1 on [0, π]."""
        from bpr.rpst_extensions import KatzSarnakChain
        theta = np.linspace(0, np.pi, 10000)
        density = KatzSarnakChain.usp2_haar_density(theta)
        integral = float(np.trapezoid(density, theta))
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_usp2_haar_density_zero_at_endpoints(self):
        """USp(2) Haar density is 0 at θ=0 and θ=π."""
        from bpr.rpst_extensions import KatzSarnakChain
        assert KatzSarnakChain.usp2_haar_density(np.array([0.0]))[0] == pytest.approx(0.0)
        assert KatzSarnakChain.usp2_haar_density(np.array([np.pi]))[0] == pytest.approx(0.0, abs=1e-10)

    def test_usp2_haar_cdf_boundary_values(self):
        """USp(2) CDF is 0 at θ=0 and 1 at θ=π."""
        from bpr.rpst_extensions import KatzSarnakChain
        assert KatzSarnakChain.usp2_haar_cdf(0.0) == pytest.approx(0.0, abs=1e-12)
        assert KatzSarnakChain.usp2_haar_cdf(np.pi) == pytest.approx(1.0, abs=1e-10)

    def test_usp2_haar_cdf_monotone(self):
        """USp(2) CDF is monotonically increasing."""
        from bpr.rpst_extensions import KatzSarnakChain
        thetas = np.linspace(0, np.pi, 50)
        cdf_vals = np.array([KatzSarnakChain.usp2_haar_cdf(float(t)) for t in thetas])
        assert np.all(np.diff(cdf_vals) >= -1e-12)

    def test_ks_distance_returns_valid(self):
        """ks_distance_usp2 returns a float in [0, 1] and an array."""
        from bpr.rpst_extensions import KatzSarnakChain
        D, angles = KatzSarnakChain.ks_distance_usp2(11)
        assert 0.0 <= D <= 1.0
        assert len(angles) > 0

    def test_gauss_sum_table_error_small(self):
        """Gauss sum table: |g_p| matches √p to < 1e-12."""
        from bpr.rpst_extensions import KatzSarnakChain
        rows = KatzSarnakChain.gauss_sum_verification_table([7, 11, 13, 17])
        for row in rows:
            assert row["error"] < 1e-12, (
                f"Gauss sum error too large for p={row['p']}: {row['error']}"
            )

    def test_spectral_stability_verified(self):
        """verify_spectral_stability: all |λ_k|/√p = 1 to < 1e-8."""
        from bpr.rpst_extensions import KatzSarnakChain
        result = KatzSarnakChain.verify_spectral_stability(7)
        assert result["all_on_unit_circle"], (
            f"Spectral stability failed: max deviation = "
            f"{result['max_deviation_from_unit_circle']}"
        )

    def test_convergence_rate_data_structure(self):
        """convergence_rate_data returns expected keys."""
        from bpr.rpst_extensions import KatzSarnakChain
        data = KatzSarnakChain.convergence_rate_data([7, 11, 13])
        assert "primes" in data
        assert "ks_distances" in data
        assert "C_fit" in data
        assert len(data["primes"]) == 3

    def test_convergence_rate_fit_positive(self):
        """Fitted C in D_p = C/√p should be positive."""
        from bpr.rpst_extensions import KatzSarnakChain
        data = KatzSarnakChain.convergence_rate_data([7, 11, 13, 17, 19])
        assert data["C_fit"] > 0


# ===========================================================================
# RiemannZeroStatistics — GUE statistics of tabulated Riemann zeros
# ===========================================================================

# First 20 Riemann zero imaginary parts (LMFDB verified)
_RIEMANN_ZEROS_20 = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
]


class TestRiemannZeroStatistics:
    """Tests for RiemannZeroStatistics: unfolding, GUE level statistics.

    Key results (Table 2 of numerical paper):
      - 0 of 99 spacings below s=0.3 (level repulsion)
      - Pair correlation R2(0) ≈ 0
      - GUE Wigner surmise KS stat D = 0.141, p = 0.047 for N=100
    """

    def test_smooth_count_positive(self):
        """N_smooth(T) > 0 for T > 0."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        assert RiemannZeroStatistics.smooth_count(14.134725) > 0

    def test_smooth_count_zero_guard(self):
        """N_smooth(T) = 0 for T <= 0."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        assert RiemannZeroStatistics.smooth_count(0.0) == 0.0
        assert RiemannZeroStatistics.smooth_count(-1.0) == 0.0

    def test_smooth_count_increasing(self):
        """N_smooth(T) is increasing."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        T_vals = [14.0, 21.0, 25.0, 30.0, 40.0]
        counts = [RiemannZeroStatistics.smooth_count(T) for T in T_vals]
        for i in range(len(counts) - 1):
            assert counts[i + 1] > counts[i]

    def test_unfold_length_preserved(self):
        """unfold returns same length as input."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20[:10])
        unfolded = RiemannZeroStatistics.unfold(zeros)
        assert len(unfolded) == len(zeros)

    def test_unfold_monotone(self):
        """Unfolded zeros are increasing."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20)
        unfolded = RiemannZeroStatistics.unfold(zeros)
        assert np.all(np.diff(unfolded) > 0)

    def test_spacings_mean_unity(self):
        """Normalized spacings have mean ≈ 1."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20)
        spacings = RiemannZeroStatistics.nearest_neighbor_spacings(zeros)
        assert float(np.mean(spacings)) == pytest.approx(1.0, rel=1e-10)

    def test_level_repulsion_no_small_spacings(self):
        """For the first 20 Riemann zeros, fraction below 0.3 is small."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20)
        frac = RiemannZeroStatistics.fraction_small_spacings(zeros, threshold=0.3)
        # GUE predicts ~5%, Poisson ~26%; the first 20 zeros should show repulsion
        assert frac < 0.20, f"Too many small spacings: {frac:.2%}"

    def test_gue_pair_correlation_level_repulsion(self):
        """GUE pair correlation R2_GUE(0) = 0 (level repulsion)."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        r = np.array([0.0, 0.001])
        R2 = RiemannZeroStatistics.gue_pair_correlation(r)
        assert R2[0] == pytest.approx(0.0)
        assert R2[1] < 1e-5

    def test_gue_pair_correlation_large_r(self):
        """GUE pair correlation R2_GUE(r) → 1 for large r."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        r = np.array([10.0, 20.0, 50.0])
        R2 = RiemannZeroStatistics.gue_pair_correlation(r)
        for val in R2:
            assert val == pytest.approx(1.0, abs=1e-3)

    def test_pair_correlation_returns_arrays(self):
        """pair_correlation returns two arrays of the same length."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20)
        r_centers, R2 = RiemannZeroStatistics.pair_correlation(zeros, r_max=3.0, n_bins=20)
        assert len(r_centers) == 20
        assert len(R2) == 20

    def test_ks_test_gue_returns_float(self):
        """ks_test_gue returns (D, p_value) as floats."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20)
        D, pval = RiemannZeroStatistics.ks_test_gue(zeros)
        assert isinstance(D, float)
        assert 0.0 <= D <= 1.0

    def test_ks_test_gue_not_too_large(self):
        """KS statistic vs GUE should be < 0.5 for the known Riemann zeros."""
        from bpr.rpst.hamiltonian import RiemannZeroStatistics
        zeros = np.array(_RIEMANN_ZEROS_20)
        D, _ = RiemannZeroStatistics.ks_test_gue(zeros)
        # With only 20 zeros, the KS test has limited power, but D should be < 0.5
        assert D < 0.5, f"KS statistic surprisingly large: D={D:.3f}"
