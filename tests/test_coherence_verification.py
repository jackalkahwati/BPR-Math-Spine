"""
Tests for Categorical Coherence Verification Module

TIER 1: These tests verify standard 2-category coherence conditions.
All tests should pass - they implement textbook mathematics.
"""

import numpy as np
import pytest
from bpr.verification.coherence import (
    CoherenceVerifier,
    BoundaryTransformation,
    BoundaryCoupling,
    create_test_configuration,
    run_coherence_battery
)


class TestBoundaryTransformation:
    """Tests for BoundaryTransformation dataclass."""

    def test_creation(self):
        """Test basic transformation creation."""
        matrix = np.eye(3)
        alpha = BoundaryTransformation("f", "g", matrix, "A", "B")

        assert alpha.source == "f"
        assert alpha.target == "g"
        assert alpha.dim == 3
        assert alpha.domain == "A"
        assert alpha.codomain == "B"

    def test_requires_square_matrix(self):
        """Transformation matrix must be square."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3, not square

        with pytest.raises(ValueError, match="must be square"):
            BoundaryTransformation("f", "g", matrix, "A", "B")

    def test_requires_2d_matrix(self):
        """Transformation matrix must be 2D."""
        matrix = np.array([1, 2, 3])  # 1D

        with pytest.raises(ValueError, match="must be 2D"):
            BoundaryTransformation("f", "g", matrix, "A", "B")


class TestBoundaryCoupling:
    """Tests for BoundaryCoupling dataclass."""

    def test_creation(self):
        """Test basic coupling creation."""
        matrix = np.eye(3)
        f = BoundaryCoupling("f", "A", "B", matrix)

        assert f.name == "f"
        assert f.source == "A"
        assert f.target == "B"

    def test_requires_2d_matrix(self):
        """Coupling matrix must be 2D."""
        matrix = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="must be 2D"):
            BoundaryCoupling("f", "A", "B", matrix)


class TestCoherenceVerifier:
    """Tests for CoherenceVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create verifier with default tolerance."""
        return CoherenceVerifier(tolerance=1e-10)

    @pytest.fixture
    def identity_config(self):
        """Create configuration with identity matrices."""
        dim = 3
        I = np.eye(dim)

        f = BoundaryCoupling("f", "A", "B", I)
        g = BoundaryCoupling("g", "A", "B", I)
        h = BoundaryCoupling("h", "B", "C", I)
        k = BoundaryCoupling("k", "B", "C", I)

        alpha = BoundaryTransformation("f", "g", I, "A", "B")
        beta = BoundaryTransformation("h", "k", I, "B", "C")

        return alpha, beta, f, g, h, k

    def test_interchange_identity_matrices(self, verifier, identity_config):
        """Interchange law holds trivially for identity matrices."""
        result = verifier.verify_interchange(*identity_config)

        assert result.passed
        assert result.max_difference < 1e-14

    def test_interchange_commuting_transformations(self, verifier):
        """
        Interchange law holds when transformations commute.

        The interchange law (β ∘ g)·(h ∘ α) = (k ∘ α)·(β ∘ f) reduces to
        β·α = α·β when whiskering preserves transformations. This holds
        only when α and β commute.
        """
        dim = 3
        I = np.eye(dim)

        # Use commuting transformations (scalar multiples of identity)
        alpha_matrix = 2 * I
        beta_matrix = 3 * I

        f = BoundaryCoupling("f", "A", "B", I)
        g = BoundaryCoupling("g", "A", "B", I)
        h = BoundaryCoupling("h", "B", "C", I)
        k = BoundaryCoupling("k", "B", "C", I)

        alpha = BoundaryTransformation("f", "g", alpha_matrix, "A", "B")
        beta = BoundaryTransformation("h", "k", beta_matrix, "B", "C")

        result = verifier.verify_interchange(alpha, beta, f, g, h, k)

        # Commuting transformations satisfy interchange
        assert result.passed, f"max_diff={result.max_difference}"

    @pytest.mark.parametrize("scale", [1.0, 2.0, 0.5, -1.0])
    def test_interchange_scaled_identities(self, verifier, scale):
        """Interchange holds for scaled identity transformations."""
        dim = 4
        I = np.eye(dim)

        alpha_matrix = scale * I
        beta_matrix = (scale + 1) * I

        f = BoundaryCoupling("f", "A", "B", I)
        g = BoundaryCoupling("g", "A", "B", I)
        h = BoundaryCoupling("h", "B", "C", I)
        k = BoundaryCoupling("k", "B", "C", I)

        alpha = BoundaryTransformation("f", "g", alpha_matrix, "A", "B")
        beta = BoundaryTransformation("h", "k", beta_matrix, "B", "C")

        result = verifier.verify_interchange(alpha, beta, f, g, h, k)
        assert result.passed

    def test_interchange_fails_for_noncommuting(self, verifier):
        """
        Interchange law FAILS for non-commuting transformations.

        This is expected behavior - the interchange law is a coherence
        condition that constrains which transformations can be composed
        path-independently.
        """
        dim = 2
        I = np.eye(dim)

        # Non-commuting matrices
        alpha_matrix = np.array([[0, 1], [1, 0]])  # Swap
        beta_matrix = np.array([[1, 1], [0, 1]])   # Shear

        # Verify they don't commute
        assert not np.allclose(alpha_matrix @ beta_matrix, beta_matrix @ alpha_matrix)

        f = BoundaryCoupling("f", "A", "B", I)
        g = BoundaryCoupling("g", "A", "B", I)
        h = BoundaryCoupling("h", "B", "C", I)
        k = BoundaryCoupling("k", "B", "C", I)

        alpha = BoundaryTransformation("f", "g", alpha_matrix, "A", "B")
        beta = BoundaryTransformation("h", "k", beta_matrix, "B", "C")

        result = verifier.verify_interchange(alpha, beta, f, g, h, k)

        # Should FAIL for non-commuting transformations
        assert not result.passed, "Expected failure for non-commuting transformations"

    def test_whisker_left(self, verifier):
        """Test left whiskering h ∘ α returns transformation."""
        dim = 2
        h = BoundaryCoupling("h", "B", "C", np.array([[1, 2], [3, 4]]))
        alpha_matrix = np.array([[5, 6], [7, 8]])
        alpha = BoundaryTransformation("f", "g", alpha_matrix, "A", "B")

        result = verifier.whisker_left(h, alpha)

        # In identity-preserving whiskering, h ∘ α = α
        assert np.allclose(result, alpha_matrix)

    def test_whisker_right(self, verifier):
        """Test right whiskering α ∘ h returns transformation."""
        dim = 2
        alpha_matrix = np.array([[1, 2], [3, 4]])
        alpha = BoundaryTransformation("f", "g", alpha_matrix, "B", "C")
        h = BoundaryCoupling("h", "A", "B", np.array([[5, 6], [7, 8]]))

        result = verifier.whisker_right(alpha, h)

        # In identity-preserving whiskering, α ∘ h = α
        assert np.allclose(result, alpha_matrix)

    def test_compose_vertical(self, verifier):
        """Test vertical composition."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        result = verifier.compose_vertical(A, B)
        expected = A @ B

        assert np.allclose(result, expected)

    def test_validation_source_target_mismatch(self, verifier):
        """Error when α doesn't transform f to g."""
        dim = 2
        I = np.eye(dim)

        f = BoundaryCoupling("f", "A", "B", I)
        g = BoundaryCoupling("g", "A", "B", I)
        h = BoundaryCoupling("h", "B", "C", I)
        k = BoundaryCoupling("k", "B", "C", I)

        # Wrong source for alpha
        alpha = BoundaryTransformation("wrong", "g", I, "A", "B")
        beta = BoundaryTransformation("h", "k", I, "B", "C")

        with pytest.raises(ValueError, match="α must be a transformation"):
            verifier.verify_interchange(alpha, beta, f, g, h, k)

    def test_validation_domain_codomain(self, verifier):
        """Error when couplings not composable."""
        dim = 2
        I = np.eye(dim)

        f = BoundaryCoupling("f", "A", "B", I)
        g = BoundaryCoupling("g", "A", "B", I)
        # h has wrong source (should be B, not C)
        h = BoundaryCoupling("h", "C", "D", I)
        k = BoundaryCoupling("k", "C", "D", I)

        alpha = BoundaryTransformation("f", "g", I, "A", "B")
        beta = BoundaryTransformation("h", "k", I, "C", "D")

        with pytest.raises(ValueError, match="not composable"):
            verifier.verify_interchange(alpha, beta, f, g, h, k)

    def test_verification_summary(self, verifier):
        """Test verification summary after multiple checks."""
        # Run several verifications
        for seed in range(5):
            config = create_test_configuration(dim=3, seed=seed)
            verifier.verify_interchange(*config)

        summary = verifier.get_verification_summary()

        assert summary['total'] == 5
        assert summary['passed'] == 5
        assert summary['failed'] == 0
        assert summary['pass_rate'] == 1.0

    def test_detailed_report(self, verifier, identity_config):
        """Test detailed report generation."""
        result = verifier.verify_interchange(*identity_config)
        report = result.detailed_report()

        assert "PASSED" in report or "FAILED" in report
        assert "LHS" in report
        assert "RHS" in report
        assert "Max difference" in report


class TestAssociativity:
    """Tests for associativity verification."""

    @pytest.fixture
    def verifier(self):
        return CoherenceVerifier(tolerance=1e-10)

    def test_associativity_holds(self, verifier):
        """Vertical composition is associative."""
        dim = 3
        A = np.random.randn(dim, dim)
        B = np.random.randn(dim, dim)
        C = np.random.randn(dim, dim)

        alpha = BoundaryTransformation("a", "b", A, "X", "Y")
        beta = BoundaryTransformation("b", "c", B, "X", "Y")
        gamma = BoundaryTransformation("c", "d", C, "X", "Y")

        result = verifier.verify_associativity(alpha, beta, gamma)

        assert result.passed
        assert result.max_difference < 1e-10


class TestBatteryTests:
    """Tests for the coherence verification battery."""

    def test_battery_runs(self):
        """Battery test completes without error."""
        results = run_coherence_battery(num_tests=10, dims=[2, 3])

        assert 'by_dim' in results
        assert 'total_passed' in results
        assert 'overall_rate' in results

    def test_battery_all_pass(self):
        """Battery should have 100% pass rate for valid configurations."""
        results = run_coherence_battery(num_tests=20, dims=[2, 3, 4])

        assert results['overall_rate'] == 1.0, \
            f"Expected 100% pass rate, got {results['overall_rate']:.1%}"


class TestCreateTestConfiguration:
    """Tests for test configuration generator."""

    def test_creates_valid_configuration(self):
        """Generated configuration is valid."""
        alpha, beta, f, g, h, k = create_test_configuration(dim=3, seed=0)

        # All objects created
        assert alpha is not None
        assert beta is not None
        assert f is not None
        assert g is not None
        assert h is not None
        assert k is not None

        # Correct types
        assert isinstance(alpha, BoundaryTransformation)
        assert isinstance(f, BoundaryCoupling)

    def test_reproducible(self):
        """Same seed produces same configuration."""
        config1 = create_test_configuration(dim=3, seed=42)
        config2 = create_test_configuration(dim=3, seed=42)

        assert np.allclose(config1[0].matrix, config2[0].matrix)
        assert np.allclose(config1[2].matrix, config2[2].matrix)

    def test_different_seeds_different_configs(self):
        """Different seeds produce different configurations."""
        config1 = create_test_configuration(dim=3, seed=1)
        config2 = create_test_configuration(dim=3, seed=2)

        assert not np.allclose(config1[0].matrix, config2[0].matrix)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
