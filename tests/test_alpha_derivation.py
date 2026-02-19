"""
Tests for Theory XXII: Fine Structure Constant from Substrate.

Run with:  pytest -v tests/test_alpha_derivation.py
"""

import numpy as np
import pytest


# =====================================================================
# §22.1  Core formula: 1/α from substrate
# =====================================================================

class TestInverseAlpha:
    """Verify the core formula 1/α = [ln(p)]² + z/2 + γ − 1/(2π)."""

    def test_default_prediction_close_to_137(self):
        from bpr.alpha_derivation import inverse_alpha_from_substrate
        inv_a = inverse_alpha_from_substrate()
        # Must be within 0.1% of 137.036
        assert abs(inv_a - 137.036) / 137.036 < 0.001

    def test_deviation_under_40_ppm(self):
        from bpr.alpha_derivation import alpha_breakdown
        bd = alpha_breakdown()
        assert bd.deviation_ppm < 40.0

    def test_four_terms_sum_correctly(self):
        from bpr.alpha_derivation import alpha_breakdown
        bd = alpha_breakdown()
        total = bd.screening + bd.bare + bd.lattice_correction + bd.scheme_correction
        assert abs(total - bd.inv_alpha_predicted) < 1e-12

    def test_screening_is_dominant_term(self):
        from bpr.alpha_derivation import alpha_breakdown
        bd = alpha_breakdown()
        # [ln(p)]² should be ~97% of the total
        assert bd.screening / bd.inv_alpha_predicted > 0.95

    def test_bare_coupling_equals_kappa(self):
        from bpr.alpha_derivation import alpha_breakdown
        bd = alpha_breakdown(z=6)
        assert bd.bare == pytest.approx(3.0)
        bd4 = alpha_breakdown(z=4)
        assert bd4.bare == pytest.approx(2.0)

    def test_scheme_correction_negative(self):
        from bpr.alpha_derivation import alpha_breakdown
        bd = alpha_breakdown()
        assert bd.scheme_correction < 0
        assert abs(bd.scheme_correction + 1.0 / (2 * np.pi)) < 1e-15


class TestAlphaEM:
    """Verify the derived α_EM values."""

    def test_alpha_0_close_to_experimental(self):
        from bpr.alpha_derivation import alpha_em_from_substrate
        alpha = alpha_em_from_substrate()
        alpha_exp = 1.0 / 137.036
        assert abs(alpha - alpha_exp) / alpha_exp < 0.001

    def test_alpha_MZ_close_to_experimental(self):
        from bpr.alpha_derivation import alpha_em_at_MZ
        alpha_mz = alpha_em_at_MZ()
        alpha_mz_exp = 1.0 / 127.952
        assert abs(alpha_mz - alpha_mz_exp) / alpha_mz_exp < 0.005

    def test_alpha_MZ_larger_than_alpha_0(self):
        from bpr.alpha_derivation import alpha_em_from_substrate, alpha_em_at_MZ
        a0 = alpha_em_from_substrate()
        amz = alpha_em_at_MZ()
        # α runs UP with energy (QED screening)
        assert amz > a0


# =====================================================================
# §22.2  Parameter dependence
# =====================================================================

class TestParameterDependence:
    """Verify physically sensible parameter dependence."""

    def test_alpha_decreases_with_larger_p(self):
        from bpr.alpha_derivation import alpha_em_from_substrate
        # Larger p → more screening → weaker coupling
        alpha_small = alpha_em_from_substrate(p=997)
        alpha_large = alpha_em_from_substrate(p=104729)
        assert alpha_large < alpha_small

    def test_alpha_decreases_with_larger_z(self):
        from bpr.alpha_derivation import alpha_em_from_substrate
        # Larger z → larger κ → weaker coupling
        alpha_z4 = alpha_em_from_substrate(z=4)
        alpha_z6 = alpha_em_from_substrate(z=6)
        assert alpha_z6 < alpha_z4

    def test_trivial_p_gives_strong_coupling(self):
        from bpr.alpha_derivation import inverse_alpha_from_substrate
        # p = 2 (smallest prime): ln(2)² + 3 + γ − 1/(2π) ≈ 4.06
        inv_a = inverse_alpha_from_substrate(p=2, z=6)
        assert inv_a < 5  # strong coupling regime

    def test_large_p_gives_weak_coupling(self):
        from bpr.alpha_derivation import inverse_alpha_from_substrate
        # p = 10^9 + 7: very weak coupling
        inv_a = inverse_alpha_from_substrate(p=1000000007, z=6)
        assert inv_a > 400


# =====================================================================
# §22.3  GUT-scale quantities
# =====================================================================

class TestGUTScale:
    """Verify GUT-scale derived quantities."""

    def test_gut_scale_order_of_magnitude(self):
        from bpr.alpha_derivation import gut_scale_GeV
        m_gut = gut_scale_GeV()
        # Should be ~10^17 GeV
        assert 1e16 < m_gut < 1e19

    def test_alpha_gut_perturbative(self):
        from bpr.alpha_derivation import alpha_gut_from_lattice
        a_gut = alpha_gut_from_lattice()
        # α_GUT should be perturbative (< 1)
        assert 0 < a_gut < 1
        # And in the right ballpark (1/20 to 1/100)
        assert 0.01 < a_gut < 0.05


# =====================================================================
# §22.4  Sensitivity analysis
# =====================================================================

class TestSensitivity:
    """Verify sensitivity analysis functions."""

    def test_exact_p_close_to_framework_p(self):
        from bpr.alpha_derivation import prime_for_exact_alpha
        p_exact = prime_for_exact_alpha()
        # p* should be within a few percent of 104729
        assert abs(p_exact - 104729) / 104729 < 0.05

    def test_sensitivity_dict_complete(self):
        from bpr.alpha_derivation import sensitivity_to_prime
        sens = sensitivity_to_prime()
        assert "d_inv_alpha_d_ln_p" in sens
        assert "d_inv_alpha_d_p" in sens
        assert "p_for_exact_alpha" in sens

    def test_sensitivity_positive(self):
        from bpr.alpha_derivation import sensitivity_to_prime
        sens = sensitivity_to_prime()
        # d(1/α)/d(ln p) = 2 ln(p) > 0
        assert sens["d_inv_alpha_d_ln_p"] > 0


# =====================================================================
# §22.5  Full derivation chain
# =====================================================================

class TestFullDerivation:
    """End-to-end derivation tests."""

    def test_derive_alpha_returns_result(self):
        from bpr.alpha_derivation import derive_alpha
        result = derive_alpha()
        assert result.p == 104729
        assert result.z == 6
        assert 136 < result.inv_alpha_0 < 138
        assert 127 < result.inv_alpha_MZ < 129

    def test_derivation_deviation_under_half_percent(self):
        from bpr.alpha_derivation import derive_alpha
        result = derive_alpha()
        assert result.deviation_0_percent < 0.5
        assert result.deviation_MZ_percent < 0.5

    def test_summary_runs(self):
        from bpr.alpha_derivation import summary
        s = summary()
        assert "137" in s
        assert "ppm" in s.lower() or "screening" in s.lower()

    def test_breakdown_summary_runs(self):
        from bpr.alpha_derivation import alpha_breakdown
        bd = alpha_breakdown()
        s = bd.summary()
        assert "screening" in s.lower() or "Z_p" in s


# =====================================================================
# §22.6  Cross-check with GUT running
# =====================================================================

class TestGUTCrossCheck:
    """Cross-check the formula against GUT running (§22.7)."""

    def test_gut_running_returns_valid_dict(self):
        from bpr.alpha_derivation import alpha_em_from_gut_running
        result = alpha_em_from_gut_running()
        assert "inv_alpha_gut" in result
        assert "inv_alpha_1_MZ" in result
        assert result["M_gut_GeV"] > 1e15

    def test_gut_running_couplings_ordered(self):
        from bpr.alpha_derivation import alpha_em_from_gut_running
        result = alpha_em_from_gut_running()
        # At M_Z: 1/α₁ > 1/α₂ > 1/α₃ (standard ordering)
        if not np.isnan(result["inv_alpha_1_MZ"]):
            assert result["inv_alpha_1_MZ"] > result["inv_alpha_2_MZ"]
            assert result["inv_alpha_2_MZ"] > result["inv_alpha_3_MZ"]


# =====================================================================
# §22.7  Numerical precision
# =====================================================================

class TestPrecision:
    """Verify numerical precision of the computation."""

    def test_euler_gamma_correct(self):
        from bpr.alpha_derivation import _EULER_GAMMA
        # Euler-Mascheroni to 16 digits
        assert abs(_EULER_GAMMA - 0.5772156649015329) < 1e-16

    def test_formula_reproducible(self):
        from bpr.alpha_derivation import inverse_alpha_from_substrate
        # Same inputs should give identical results
        a = inverse_alpha_from_substrate(104729, 6)
        b = inverse_alpha_from_substrate(104729, 6)
        assert a == b

    def test_known_numerical_value(self):
        from bpr.alpha_derivation import inverse_alpha_from_substrate
        inv_a = inverse_alpha_from_substrate(104729, 6)
        # Pre-computed: should be approximately 137.031
        assert abs(inv_a - 137.031) < 0.01
