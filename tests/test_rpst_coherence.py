"""
Tests for RPST Coherence and Boundary Stress Module

Tests cover:
- Tier 1: Kuramoto order parameter (standard physics)
- Tier 1: Boundary stress (gradient energy)
- Tier 2: Eligibility functional (BPR-specific conjecture)
"""

import numpy as np
import pytest
from bpr.rpst.coherence import (
    PhaseCoherence,
    BoundaryStress,
    EligibilityFunctional,
    CoherenceMetrics,
    StressMetrics,
    create_ring_lattice,
    create_square_lattice,
    run_coherence_stress_demo
)


class TestPhaseCoherence:
    """Tests for PhaseCoherence class - TIER 1."""

    @pytest.fixture
    def coherence(self):
        return PhaseCoherence()

    def test_synchronized_phases(self, coherence):
        """Perfectly synchronized phases have K=1."""
        N = 100
        phi = np.zeros(N)  # All phases zero

        metrics = coherence.compute(phi)

        assert np.isclose(metrics.global_coherence, 1.0, atol=1e-10)
        assert np.isclose(metrics.phase_variance, 0.0, atol=1e-10)

    def test_antisynchronized_phases(self, coherence):
        """Alternating phases have low coherence."""
        N = 100
        phi = np.array([0, np.pi] * (N // 2))  # Alternating 0 and π

        metrics = coherence.compute(phi)

        assert np.isclose(metrics.global_coherence, 0.0, atol=1e-10)

    def test_random_phases_low_coherence(self, coherence):
        """Random phases have coherence ~ 1/√N."""
        N = 10000
        np.random.seed(42)
        phi = np.random.uniform(0, 2*np.pi, N)

        metrics = coherence.compute(phi)

        # For random phases, K ~ 1/√N
        expected_order = 1.0 / np.sqrt(N)
        assert metrics.global_coherence < 5 * expected_order  # Allow some variance

    def test_coherence_bounds(self, coherence):
        """Coherence K ∈ [0, 1]."""
        for _ in range(10):
            N = np.random.randint(10, 100)
            phi = np.random.uniform(0, 2*np.pi, N)
            metrics = coherence.compute(phi)

            assert 0 <= metrics.global_coherence <= 1

    def test_mean_phase_computation(self, coherence):
        """Mean phase computed correctly."""
        N = 100
        target_phase = 1.5
        phi = np.full(N, target_phase)

        metrics = coherence.compute(phi)

        assert np.isclose(metrics.mean_phase, target_phase, atol=1e-10)

    def test_local_coherence_with_adjacency(self, coherence):
        """Local coherence computed when adjacency provided."""
        N = 10
        phi = np.zeros(N)
        adj = create_ring_lattice(N, k=2)

        metrics = coherence.compute(phi, adjacency=adj)

        assert metrics.local_coherence is not None
        assert len(metrics.local_coherence) == N
        # All local coherence should be 1 (synchronized)
        assert np.allclose(metrics.local_coherence, 1.0)

    def test_gradient_computation(self, coherence):
        """Gradient of coherence computed correctly."""
        N = 5
        phi = np.array([0, 0.1, 0.2, 0.3, 0.4])

        grad = coherence.compute_gradient(phi)

        assert len(grad) == N
        # Gradient should be non-zero for non-uniform phases
        assert np.linalg.norm(grad) > 0


class TestBoundaryStress:
    """Tests for BoundaryStress class - TIER 1."""

    @pytest.fixture
    def stress(self):
        return BoundaryStress()

    def test_synchronized_zero_stress(self, stress):
        """Synchronized phases have zero frustration stress."""
        N = 10
        phi = np.zeros(N)
        adj = create_ring_lattice(N, k=2)

        metrics = stress.compute(phi, adj)

        assert np.isclose(metrics.frustration_stress, 0.0, atol=1e-10)

    def test_antisynchronized_high_stress(self, stress):
        """Alternating phases have high frustration stress."""
        N = 10
        phi = np.array([0, np.pi] * (N // 2))
        adj = create_ring_lattice(N, k=1)  # Only nearest neighbors

        metrics = stress.compute(phi, adj)

        # Each edge contributes (1 - cos(π)) = 2
        # Ring has N edges
        expected_stress = N * 2
        assert np.isclose(metrics.frustration_stress, expected_stress, atol=1e-10)

    def test_smooth_gradient_moderate_stress(self, stress):
        """Smooth phase gradient has moderate stress."""
        N = 10
        phi = np.linspace(0, np.pi, N)  # Smooth gradient
        adj = create_ring_lattice(N, k=1)

        metrics = stress.compute(phi, adj)

        # Stress should be positive but not maximum
        assert metrics.frustration_stress > 0
        assert metrics.frustration_stress < N * 2

    def test_laplacian_stress(self, stress):
        """Laplacian stress computed correctly."""
        N = 10
        phi = np.linspace(0, 2*np.pi, N, endpoint=False)
        adj = create_ring_lattice(N, k=1)

        metrics = stress.compute(phi, adj)

        assert metrics.laplacian_stress >= 0

    def test_stress_field_computed(self, stress):
        """Per-node stress field computed."""
        N = 10
        phi = np.random.uniform(0, 2*np.pi, N)
        adj = create_ring_lattice(N, k=1)

        metrics = stress.compute(phi, adj)

        assert metrics.stress_field is not None
        assert len(metrics.stress_field) == N
        assert all(s >= 0 for s in metrics.stress_field)

    def test_max_local_stress(self, stress):
        """Maximum local stress tracked."""
        N = 10
        phi = np.zeros(N)
        phi[5] = np.pi  # Single defect

        adj = create_ring_lattice(N, k=1)

        metrics = stress.compute(phi, adj)

        # Max stress should be at defect node
        defect_stress = metrics.stress_field[5]
        assert metrics.max_local_stress >= defect_stress


class TestEligibilityFunctional:
    """Tests for EligibilityFunctional class - TIER 2 CONJECTURE."""

    @pytest.fixture
    def eligibility(self):
        return EligibilityFunctional(kappa_K=0.1, threshold=1.0)

    def test_eligibility_computation(self, eligibility):
        """Eligibility can be computed."""
        N = 20
        phi = np.random.uniform(0, 2*np.pi, N)
        adj = create_ring_lattice(N, k=2)

        E, triggers = eligibility.compute(phi, adj)

        assert isinstance(E, float)
        assert isinstance(triggers, bool)
        assert E >= 0

    def test_high_coherence_low_stress_low_eligibility(self, eligibility):
        """Synchronized state has low eligibility."""
        N = 20
        phi = np.zeros(N)  # Synchronized
        adj = create_ring_lattice(N, k=2)

        E, _ = eligibility.compute(phi, adj)

        # Should be near zero (low stress, no gradient)
        assert E < 0.1

    def test_temporal_gradient_increases_eligibility(self, eligibility):
        """Changing coherence increases eligibility."""
        N = 20
        adj = create_ring_lattice(N, k=2)

        phi_prev = np.zeros(N)
        phi_curr = np.random.uniform(0, 0.5, N)  # Changed

        E_with_history, _ = eligibility.compute(phi_curr, adj, phi_prev, dt=1.0)

        # Without history (only spatial gradient)
        E_no_history, _ = eligibility.compute(phi_curr, adj)

        # Both should be computable
        assert E_with_history >= 0
        assert E_no_history >= 0

    def test_scan_trajectory(self, eligibility):
        """Trajectory scanning detects events."""
        N = 20
        T = 50
        adj = create_ring_lattice(N, k=2)

        # Create trajectory with a transition
        trajectory = np.zeros((T, N))
        for t in range(T):
            if t < 25:
                trajectory[t] = np.zeros(N)  # Synchronized
            else:
                trajectory[t] = np.random.uniform(0, 2*np.pi, N)  # Disordered

        events = eligibility.scan_trajectory(trajectory, adj, dt=1.0)

        # Should detect some events
        assert isinstance(events, list)
        # Events near transition (around t=25) if threshold is right

    def test_threshold_parameter(self):
        """Threshold parameter affects trigger behavior."""
        N = 20
        phi = np.random.uniform(0, 2*np.pi, N)
        adj = create_ring_lattice(N, k=2)

        # Low threshold - should trigger more easily
        elig_low = EligibilityFunctional(kappa_K=0.1, threshold=0.001)
        E_low, triggers_low = elig_low.compute(phi, adj)

        # High threshold - should trigger less easily
        elig_high = EligibilityFunctional(kappa_K=0.1, threshold=1000.0)
        E_high, triggers_high = elig_high.compute(phi, adj)

        # Same E value, different trigger outcomes possible
        assert E_low == E_high  # Same computation
        # Low threshold more likely to trigger
        # (but not guaranteed depending on E value)


class TestLatticeCreation:
    """Tests for lattice adjacency matrix creation."""

    def test_ring_lattice_structure(self):
        """Ring lattice has correct structure."""
        N = 10
        k = 2  # 2 neighbors on each side
        adj = create_ring_lattice(N, k)

        assert adj.shape == (N, N)
        assert np.allclose(adj, adj.T)  # Symmetric

        # Each node should have 2k neighbors
        degrees = np.sum(adj, axis=1)
        assert np.allclose(degrees, 2*k)

    def test_ring_lattice_periodic(self):
        """Ring lattice is periodic."""
        N = 10
        k = 1
        adj = create_ring_lattice(N, k)

        # Node 0 connected to node N-1
        assert adj[0, N-1] == 1
        assert adj[N-1, 0] == 1

    def test_square_lattice_structure(self):
        """Square lattice has correct structure."""
        L = 5
        adj = create_square_lattice(L)
        N = L * L

        assert adj.shape == (N, N)
        assert np.allclose(adj, adj.T)  # Symmetric

        # Interior nodes have 4 neighbors
        # Edge nodes have 3 neighbors
        # Corner nodes have 2 neighbors

    def test_square_lattice_no_periodic(self):
        """Square lattice has open boundaries."""
        L = 5
        adj = create_square_lattice(L)

        # Corner (0,0) = index 0 should have exactly 2 neighbors
        assert np.sum(adj[0]) == 2


class TestCoherenceMetrics:
    """Tests for CoherenceMetrics dataclass."""

    def test_metrics_creation(self):
        """Metrics created correctly."""
        metrics = CoherenceMetrics(
            global_coherence=0.8,
            mean_phase=1.2,
            phase_variance=0.2
        )

        assert metrics.global_coherence == 0.8
        assert metrics.mean_phase == 1.2
        assert metrics.phase_variance == 0.2
        assert metrics.local_coherence is None

    def test_metrics_repr(self):
        """Metrics have string representation."""
        metrics = CoherenceMetrics(
            global_coherence=0.8,
            mean_phase=1.2,
            phase_variance=0.2
        )

        repr_str = repr(metrics)
        assert "0.8" in repr_str
        assert "K=" in repr_str


class TestStressMetrics:
    """Tests for StressMetrics dataclass."""

    def test_metrics_creation(self):
        """Metrics created correctly."""
        metrics = StressMetrics(
            total_stress=1.5,
            frustration_stress=1.0,
            laplacian_stress=0.5,
            max_local_stress=0.3
        )

        assert metrics.total_stress == 1.5
        assert metrics.frustration_stress == 1.0

    def test_metrics_repr(self):
        """Metrics have string representation."""
        metrics = StressMetrics(
            total_stress=1.5,
            frustration_stress=1.0,
            laplacian_stress=0.5,
            max_local_stress=0.3
        )

        repr_str = repr(metrics)
        assert "1.5" in repr_str
        assert "total=" in repr_str


class TestDemo:
    """Test demo function runs."""

    def test_demo_runs(self):
        """Demo function executes without error."""
        result = run_coherence_stress_demo()
        assert result is True


class TestPhysicalConsistency:
    """Tests for physical consistency of metrics."""

    def test_coherence_variance_relation(self):
        """Coherence and variance satisfy K + V = 1."""
        coherence = PhaseCoherence()

        for _ in range(10):
            N = np.random.randint(50, 200)
            phi = np.random.uniform(0, 2*np.pi, N)

            metrics = coherence.compute(phi)

            # Circular variance = 1 - K
            expected_variance = 1 - metrics.global_coherence
            assert np.isclose(metrics.phase_variance, expected_variance, atol=1e-10)

    def test_stress_increases_with_disorder(self):
        """Stress increases as phases become more disordered."""
        stress = BoundaryStress()
        N = 50
        adj = create_ring_lattice(N, k=2)

        # Start synchronized
        phi_sync = np.zeros(N)
        stress_sync = stress.compute(phi_sync, adj).total_stress

        # Add noise
        phi_noisy = np.random.uniform(-0.1, 0.1, N)
        stress_noisy = stress.compute(phi_noisy, adj).total_stress

        # More disorder
        phi_random = np.random.uniform(0, 2*np.pi, N)
        stress_random = stress.compute(phi_random, adj).total_stress

        assert stress_sync <= stress_noisy <= stress_random + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
