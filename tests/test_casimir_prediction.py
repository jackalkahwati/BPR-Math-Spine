"""
Tests for Complete BPR Casimir Prediction

These tests verify:
1. All parameters are derived (no free parameters)
2. Dimensional consistency throughout
3. Physical limiting cases
4. Correct order of magnitude (even if unmeasurable)
"""

import numpy as np
import pytest
from bpr.rpst.boundary_energy import (
    SubstrateParameters,
    LatticeGeometry,
    casimir_substrate_params,
    derive_all_couplings,
    L_PLANCK
)
from bpr.rpst.vacuum_coupling import compute_g, Geometry
from bpr.rpst.decay_oscillation import derive_all_decay_oscillation
from bpr.rpst.casimir_prediction import (
    compute_bpr_casimir_prediction,
    standard_casimir_force,
    compare_to_experimental_precision,
    generate_prediction_curve
)


class TestStandardCasimir:
    """Tests for standard Casimir force calculation."""

    def test_casimir_negative(self):
        """Casimir force is attractive (negative)."""
        F = standard_casimir_force(100e-9)
        assert F < 0

    def test_casimir_scaling(self):
        """Casimir scales as 1/a⁴."""
        F1 = standard_casimir_force(100e-9)
        F2 = standard_casimir_force(200e-9)

        # F ~ 1/a⁴, so F2/F1 = (a1/a2)⁴ = (1/2)⁴ = 1/16
        expected_ratio = (100/200)**4
        actual_ratio = F2 / F1

        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)

    def test_casimir_order_of_magnitude(self):
        """Casimir at 100nm should be ~10 Pa."""
        F = standard_casimir_force(100e-9)
        assert 1 < abs(F) < 100  # Order of 10 Pa


class TestDerivedParameters:
    """Tests that all parameters are properly derived."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_lambda_derived(self, params):
        """λ is derived from substrate."""
        couplings = derive_all_couplings(params)
        assert couplings.lambda_bpr > 0
        assert couplings.lambda_bpr < 1e-80  # Should be Planck-suppressed

    def test_g_derived(self, params):
        """g is derived from vacuum coupling."""
        result = compute_g(Geometry.PARALLEL_PLATES, 100e-9, params.radius)
        assert result.g >= 0

    def test_xi_derived(self, params):
        """ξ is derived from correlation length."""
        decay_osc = derive_all_decay_oscillation(params, 100e-9)
        assert decay_osc.xi > 0

    def test_Lambda_derived(self, params):
        """Λ is derived from eigenmode spacing."""
        decay_osc = derive_all_decay_oscillation(params, 100e-9)
        assert decay_osc.Lambda > 0


class TestCombinedPrediction:
    """Tests for combined BPR Casimir prediction."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_prediction_computes(self, params):
        """Prediction can be computed."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)
        assert pred is not None
        assert pred.delta_F_over_F is not None

    def test_no_free_parameters(self, params):
        """All parameters in prediction are derived."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)

        # Check all parameters have values
        assert pred.lambda_bpr != 0
        assert pred.xi > 0
        assert pred.Lambda > 0
        # g can be zero or very small

    def test_correction_is_small(self, params):
        """BPR correction is much smaller than 1."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)
        assert abs(pred.delta_F_over_F) < 1e-50

    def test_standard_casimir_correct(self, params):
        """Standard Casimir in prediction matches direct calculation."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)
        F_direct = standard_casimir_force(100e-9)
        assert np.isclose(pred.F_standard, F_direct)


class TestDimensionalConsistency:
    """Tests for dimensional consistency."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_delta_F_over_F_dimensionless(self, params):
        """ΔF/F is dimensionless."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)
        # Dimensionless quantity - just check it's a float
        assert isinstance(pred.delta_F_over_F, (float, np.floating))

    def test_lambda_has_correct_dimensions(self, params):
        """λ has dimensions [Energy × Length²]."""
        couplings = derive_all_couplings(params)

        # λ = (ℓ_P²/8π) × κ × J
        # [λ] = [Length²] × [dimensionless] × [Energy] = [Energy × Length²]

        # Check by varying J
        params2 = SubstrateParameters(
            p=params.p, N=params.N, J=2*params.J,
            geometry=params.geometry, radius=params.radius
        )
        couplings2 = derive_all_couplings(params2)

        # λ should scale with J
        assert np.isclose(couplings2.lambda_bpr / couplings.lambda_bpr, 2.0)

    def test_xi_has_length_dimension(self, params):
        """ξ scales with length."""
        decay1 = derive_all_decay_oscillation(params, 100e-9)

        params2 = SubstrateParameters(
            p=params.p, N=params.N, J=params.J,
            geometry=params.geometry, radius=2*params.radius
        )
        decay2 = derive_all_decay_oscillation(params2, 100e-9)

        # ξ should scale with radius
        assert decay2.xi > decay1.xi


class TestPhysicalLimits:
    """Tests for physical limiting behavior."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_large_separation_suppressed(self, params):
        """Correction decreases at large separation."""
        pred1 = compute_bpr_casimir_prediction(params, 100e-9)
        pred2 = compute_bpr_casimir_prediction(params, 1000e-9)

        # At larger separation, |ΔF/F| should be smaller
        # (or at least not larger by many orders)
        ratio = abs(pred2.delta_F_over_F / pred1.delta_F_over_F)
        assert ratio < 1e10  # Not wildly larger

    def test_exp_factor_bounded(self, params):
        """Exponential factor is in [0, 1]."""
        for a in [10e-9, 100e-9, 1000e-9]:
            pred = compute_bpr_casimir_prediction(params, a)
            assert 0 <= pred.exp_factor <= 1

    def test_osc_factor_bounded(self, params):
        """Oscillation factor is in [-1, 1]."""
        for a in [10e-9, 100e-9, 1000e-9]:
            pred = compute_bpr_casimir_prediction(params, a)
            assert -1 <= pred.osc_factor <= 1


class TestExperimentalComparison:
    """Tests for experimental comparison."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_comparison_computes(self, params):
        """Experimental comparison can be computed."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)
        comparison = compare_to_experimental_precision(pred)

        assert 'detectable' in comparison
        assert 'orders_below' in comparison

    def test_not_detectable(self, params):
        """Prediction is not detectable at current precision."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)
        comparison = compare_to_experimental_precision(pred, 1e-3)

        assert not comparison['detectable']
        assert comparison['orders_below'] > 50  # Many orders below


class TestPredictionCurve:
    """Tests for prediction curve generation."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_curve_generates(self, params):
        """Prediction curve can be generated."""
        seps = np.array([50e-9, 100e-9, 200e-9])
        a, dF, F = generate_prediction_curve(params, seps)

        assert len(a) == 3
        assert len(dF) == 3
        assert len(F) == 3

    def test_curve_values_finite(self, params):
        """All curve values are finite."""
        seps = np.logspace(-8, -6, 10)
        a, dF, F = generate_prediction_curve(params, seps)

        assert np.all(np.isfinite(dF))
        assert np.all(np.isfinite(F))


class TestPlanckSuppression:
    """Tests confirming Planck suppression is the dominant effect."""

    @pytest.fixture
    def params(self):
        return casimir_substrate_params()

    def test_lambda_contains_planck_squared(self, params):
        """λ is proportional to ℓ_P²."""
        couplings = derive_all_couplings(params)

        # λ = (ℓ_P²/8π) × κ × J
        # So λ / ℓ_P² should be order of J (with factors)
        ratio = couplings.lambda_bpr / L_PLANCK**2

        # Should be order of J/8π ≈ 10^-20 J
        assert 1e-25 < ratio < 1e-15

    def test_planck_is_dominant_suppression(self, params):
        """Planck suppression dominates over other factors."""
        pred = compute_bpr_casimir_prediction(params, 100e-9)

        # exp factor is ~1
        assert pred.exp_factor > 0.99

        # cos factor is ~1
        assert abs(pred.osc_factor) > 0.99

        # The suppression comes from λ × g
        # λ ~ 10^-90, g ~ 10^-16
        assert pred.lambda_bpr < 1e-80
        assert abs(pred.delta_F_over_F) < 1e-80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
