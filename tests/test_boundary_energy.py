"""
Tests for Boundary Energy Derivation

These tests verify:
1. Dimensional consistency of derived couplings
2. Correct scaling with substrate parameters
3. Physical limits (free field, tight coupling)

SPRINT: Week 1, Task 1.1
"""

import numpy as np
import pytest
from bpr.rpst.boundary_energy import (
    SubstrateParameters,
    LatticeGeometry,
    DerivedCouplings,
    derive_kappa,
    derive_kappa_dimensional,
    derive_correlation_length,
    derive_lambda_bpr,
    derive_all_couplings,
    verify_dimensional_consistency,
    casimir_substrate_params,
    L_PLANCK,
)


class TestSubstrateParameters:
    """Tests for SubstrateParameters dataclass."""

    def test_valid_creation(self):
        """Valid parameters create successfully."""
        params = SubstrateParameters(
            p=101,
            N=100,
            J=1.0,
            geometry=LatticeGeometry.SPHERE,
            radius=0.01
        )
        assert params.p == 101
        assert params.N == 100

    def test_requires_prime(self):
        """p must be prime."""
        with pytest.raises(ValueError, match="must be prime"):
            SubstrateParameters(
                p=100,  # Not prime!
                N=100,
                J=1.0,
                geometry=LatticeGeometry.SPHERE,
                radius=0.01
            )

    def test_requires_positive_J(self):
        """J must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            SubstrateParameters(
                p=101,
                N=100,
                J=-1.0,  # Negative!
                geometry=LatticeGeometry.SPHERE,
                radius=0.01
            )

    def test_lattice_spacing_ring(self):
        """Ring lattice spacing is 2πR/N."""
        params = SubstrateParameters(
            p=101,
            N=100,
            J=1.0,
            geometry=LatticeGeometry.RING,
            radius=1.0
        )
        expected = 2 * np.pi * 1.0 / 100
        assert np.isclose(params.lattice_spacing, expected)

    def test_lattice_spacing_sphere(self):
        """Sphere lattice spacing is R√(4π/N)."""
        params = SubstrateParameters(
            p=101,
            N=100,
            J=1.0,
            geometry=LatticeGeometry.SPHERE,
            radius=1.0
        )
        expected = 1.0 * np.sqrt(4 * np.pi / 100)
        assert np.isclose(params.lattice_spacing, expected)

    def test_coordination_numbers(self):
        """Coordination numbers are correct."""
        for geom, expected_z in [
            (LatticeGeometry.RING, 2),
            (LatticeGeometry.SQUARE, 4),
            (LatticeGeometry.SPHERE, 6),
        ]:
            params = SubstrateParameters(
                p=101, N=100, J=1.0, geometry=geom, radius=1.0
            )
            assert params.coordination_number == expected_z

    def test_boundary_area_sphere(self):
        """Sphere area is 4πR²."""
        R = 2.0
        params = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE,
            radius=R
        )
        assert np.isclose(params.boundary_area, 4 * np.pi * R**2)


class TestDeriveKappa:
    """Tests for κ derivation."""

    def test_kappa_positive(self):
        """κ should be positive."""
        params = casimir_substrate_params()
        kappa = derive_kappa(params)
        assert kappa > 0

    def test_kappa_depends_on_geometry(self):
        """κ depends on coordination number."""
        params_ring = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.RING, radius=1.0
        )
        params_sphere = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )

        kappa_ring = derive_kappa(params_ring)
        kappa_sphere = derive_kappa(params_sphere)

        # Sphere has higher coordination → higher κ
        assert kappa_sphere > kappa_ring
        assert kappa_ring == 1.0  # z=2 → κ=1
        assert kappa_sphere == 3.0  # z=6 → κ=3

    def test_kappa_independent_of_p(self):
        """κ doesn't depend on prime modulus."""
        params_small = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SQUARE, radius=1.0
        )
        params_large = SubstrateParameters(
            p=104729, N=100, J=1.0,
            geometry=LatticeGeometry.SQUARE, radius=1.0
        )

        assert derive_kappa(params_small) == derive_kappa(params_large)

    def test_kappa_independent_of_N(self):
        """κ doesn't depend on number of nodes."""
        params_sparse = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )
        params_dense = SubstrateParameters(
            p=101, N=10000, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )

        assert derive_kappa(params_sparse) == derive_kappa(params_dense)


class TestDeriveKappaDimensional:
    """Tests for dimensional κ derivation."""

    def test_scales_with_J(self):
        """κ_dim scales linearly with J."""
        params_1 = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )
        params_2 = SubstrateParameters(
            p=101, N=100, J=2.0,  # 2× coupling
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )

        kappa_1 = derive_kappa_dimensional(params_1)
        kappa_2 = derive_kappa_dimensional(params_2)

        assert np.isclose(kappa_2 / kappa_1, 2.0)

    def test_independent_of_radius(self):
        """κ_dim doesn't depend on boundary size."""
        params_small = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=0.01
        )
        params_large = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )

        kappa_small = derive_kappa_dimensional(params_small)
        kappa_large = derive_kappa_dimensional(params_large)

        assert np.isclose(kappa_small, kappa_large)


class TestDeriveCorrelationLength:
    """Tests for correlation length derivation."""

    def test_xi_positive(self):
        """ξ should be positive."""
        params = casimir_substrate_params()
        xi = derive_correlation_length(params)
        assert xi > 0

    def test_xi_scales_with_radius(self):
        """ξ scales linearly with radius (via lattice spacing)."""
        params_1 = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )
        params_2 = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=2.0
        )

        xi_1 = derive_correlation_length(params_1)
        xi_2 = derive_correlation_length(params_2)

        assert np.isclose(xi_2 / xi_1, 2.0, rtol=0.01)

    def test_xi_increases_with_p(self):
        """ξ increases with ln(p)."""
        params_small = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )
        params_large = SubstrateParameters(
            p=104729, N=100, J=1.0,  # Much larger prime
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )

        xi_small = derive_correlation_length(params_small)
        xi_large = derive_correlation_length(params_large)

        # ξ ~ √ln(p), so larger p → larger ξ
        assert xi_large > xi_small

        # Check scaling: ξ ~ √ln(p)
        expected_ratio = np.sqrt(np.log(104729) / np.log(101))
        actual_ratio = xi_large / xi_small
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)


class TestDeriveLambdaBPR:
    """Tests for λ_BPR derivation."""

    def test_lambda_positive(self):
        """λ_BPR should be positive."""
        params = casimir_substrate_params()
        lambda_bpr = derive_lambda_bpr(params)
        assert lambda_bpr > 0

    def test_lambda_involves_planck_scale(self):
        """λ_BPR should be proportional to ℓ_P²."""
        params = casimir_substrate_params()
        lambda_bpr = derive_lambda_bpr(params)

        # λ_BPR = (ℓ_P²/8π) × κ_dim
        kappa_dim = derive_kappa_dimensional(params)
        expected = (L_PLANCK**2 / (8 * np.pi)) * kappa_dim

        assert np.isclose(lambda_bpr, expected)

    def test_lambda_scales_with_J(self):
        """λ_BPR scales with coupling J."""
        params_1 = SubstrateParameters(
            p=101, N=100, J=1.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )
        params_2 = SubstrateParameters(
            p=101, N=100, J=2.0,
            geometry=LatticeGeometry.SPHERE, radius=1.0
        )

        lambda_1 = derive_lambda_bpr(params_1)
        lambda_2 = derive_lambda_bpr(params_2)

        assert np.isclose(lambda_2 / lambda_1, 2.0)


class TestDimensionalConsistency:
    """Tests for dimensional consistency verification."""

    def test_consistency_passes(self):
        """Dimensional consistency check should pass."""
        params = casimir_substrate_params()
        result = verify_dimensional_consistency(params)

        assert result['all_consistent'], \
            f"Dimensional consistency failed: {result}"

    def test_kappa_scale_invariant(self):
        """κ (dimensionless) should not change with R."""
        params = casimir_substrate_params()
        result = verify_dimensional_consistency(params)

        assert np.isclose(result['kappa_scaling'], 1.0, rtol=0.01)

    def test_xi_scales_linearly(self):
        """ξ should scale linearly with R."""
        params = casimir_substrate_params()
        result = verify_dimensional_consistency(params)

        assert np.isclose(result['xi_scaling'], 2.0, rtol=0.01)


class TestDeriveAllCouplings:
    """Tests for combined derivation."""

    def test_returns_all_couplings(self):
        """derive_all_couplings returns all fields."""
        params = casimir_substrate_params()
        couplings = derive_all_couplings(params)

        assert isinstance(couplings, DerivedCouplings)
        assert couplings.kappa > 0
        assert couplings.kappa_dimensional > 0
        assert couplings.xi > 0
        assert couplings.lambda_bpr > 0

    def test_consistency_between_methods(self):
        """Individual and combined derivations agree."""
        params = casimir_substrate_params()
        couplings = derive_all_couplings(params)

        assert couplings.kappa == derive_kappa(params)
        assert couplings.kappa_dimensional == derive_kappa_dimensional(params)
        assert couplings.xi == derive_correlation_length(params)
        assert couplings.lambda_bpr == derive_lambda_bpr(params)


class TestPhysicalLimits:
    """Tests for physical limiting cases."""

    def test_large_p_limit(self):
        """For large p, results should stabilize."""
        results = []
        for p in [101, 1009, 10007, 100003]:
            params = SubstrateParameters(
                p=p, N=100, J=1.0,
                geometry=LatticeGeometry.SPHERE, radius=1.0
            )
            results.append(derive_kappa(params))

        # κ should be same for all p (depends only on geometry)
        assert all(r == results[0] for r in results)

    def test_large_N_limit(self):
        """For large N (fine lattice), κ should stabilize."""
        results = []
        for N in [100, 1000, 10000]:
            params = SubstrateParameters(
                p=101, N=N, J=1.0,
                geometry=LatticeGeometry.SPHERE, radius=1.0
            )
            results.append(derive_kappa(params))

        # κ should be same for all N
        assert all(r == results[0] for r in results)


class TestCasimirSubstrateParams:
    """Tests for convenience function."""

    def test_default_params(self):
        """Default parameters are reasonable."""
        params = casimir_substrate_params()

        assert params.radius == 0.01  # 1 cm
        assert params.N == 10000
        assert params.geometry == LatticeGeometry.SQUARE

    def test_custom_params(self):
        """Custom parameters work."""
        params = casimir_substrate_params(
            plate_radius=0.02,
            N=1000,
            J_eV=0.5
        )

        assert params.radius == 0.02
        assert params.N == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
