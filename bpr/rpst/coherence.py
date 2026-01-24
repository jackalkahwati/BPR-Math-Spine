"""
RPST Coherence and Boundary Stress Definitions

This module provides operational definitions for coherence and boundary
stress in RPST systems, enabling quantitative analysis of phase-ordering
and transition dynamics.

TIER 1: These are standard order parameter and stress definitions.

Definitions from BPR Literature
-------------------------------
1. Coherence K(t): Global phase synchronization measure
2. Boundary Stress σ: Phase frustration at boundaries
3. Eligibility E(ψ): Rewrite trigger functional (Tier 2 conjecture)

The coherence and stress definitions are standard physics (Kuramoto order
parameter, gradient energy). The eligibility functional is a BPR-specific
construct requiring further theoretical justification.

References
----------
[1] Kuramoto, Y. "Chemical Oscillations, Waves, and Turbulence" (1984)
[2] Strogatz, S.H. "From Kuramoto to Crawford" (Physica D, 2000)
[3] BPR Cache Architecture Paper, Sections 8-9
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian


@dataclass
class CoherenceMetrics:
    """Container for coherence analysis results."""
    global_coherence: float  # K ∈ [0,1]
    mean_phase: float        # ψ = arg(Σ exp(iφ))
    phase_variance: float    # Circular variance
    local_coherence: Optional[np.ndarray] = None  # K_i for each node

    def __repr__(self) -> str:
        return (f"CoherenceMetrics(K={self.global_coherence:.4f}, "
                f"ψ={self.mean_phase:.4f}, var={self.phase_variance:.4f})")


@dataclass
class StressMetrics:
    """Container for boundary stress analysis results."""
    total_stress: float           # σ_total
    frustration_stress: float     # σ_frust
    laplacian_stress: float       # σ_lap
    max_local_stress: float       # max_i σ_i
    stress_field: Optional[np.ndarray] = None  # σ(x) at each point

    def __repr__(self) -> str:
        return (f"StressMetrics(total={self.total_stress:.4f}, "
                f"frust={self.frustration_stress:.4f}, "
                f"lap={self.laplacian_stress:.4f})")


class PhaseCoherence:
    """
    Compute phase coherence metrics for RPST systems.

    Implements the Kuramoto order parameter and related measures
    for quantifying global and local phase synchronization.

    Tier 1 Status
    -------------
    The Kuramoto order parameter is standard synchronization theory.
    No BPR-specific claims are made in this class.

    Examples
    --------
    >>> coherence = PhaseCoherence()
    >>> phi = np.random.uniform(0, 2*np.pi, 100)  # Random phases
    >>> metrics = coherence.compute(phi)
    >>> print(f"Global coherence: {metrics.global_coherence:.3f}")
    """

    def __init__(self):
        pass

    def compute(self,
                phases: np.ndarray,
                adjacency: Optional[np.ndarray] = None) -> CoherenceMetrics:
        """
        Compute coherence metrics for a phase configuration.

        Parameters
        ----------
        phases : np.ndarray
            Phase values φᵢ ∈ [0, 2π) for each node
        adjacency : np.ndarray, optional
            Adjacency matrix for local coherence (if provided)

        Returns
        -------
        CoherenceMetrics
            Global and local coherence measures

        Notes
        -----
        Global coherence (Kuramoto order parameter):
            K = |⟨exp(iφ)⟩| = |Σⱼ exp(iφⱼ)| / N

        K ∈ [0,1] where:
            K = 0: Completely incoherent (random phases)
            K = 1: Perfectly synchronized (all phases equal)
        """
        N = len(phases)

        # Complex order parameter
        z = np.mean(np.exp(1j * phases))
        global_coherence = np.abs(z)
        mean_phase = np.angle(z)

        # Circular variance
        phase_variance = 1 - global_coherence

        # Local coherence (if adjacency provided)
        local_coherence = None
        if adjacency is not None:
            local_coherence = self._compute_local_coherence(phases, adjacency)

        return CoherenceMetrics(
            global_coherence=global_coherence,
            mean_phase=mean_phase,
            phase_variance=phase_variance,
            local_coherence=local_coherence
        )

    def _compute_local_coherence(self,
                                  phases: np.ndarray,
                                  adjacency: np.ndarray) -> np.ndarray:
        """
        Compute local coherence at each node.

        K_i = |Σⱼ∈N(i) exp(iφⱼ)| / |N(i)|
        """
        N = len(phases)
        local_K = np.zeros(N)

        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_phases = phases[neighbors]
                z_local = np.mean(np.exp(1j * neighbor_phases))
                local_K[i] = np.abs(z_local)

        return local_K

    def compute_gradient(self, phases: np.ndarray) -> np.ndarray:
        """
        Compute gradient of global coherence w.r.t. phases.

        ∂K/∂φᵢ = (1/N) Im(exp(-iψ) exp(iφᵢ))

        where ψ is the mean phase.

        Returns
        -------
        np.ndarray
            Gradient ∇_φ K
        """
        N = len(phases)
        z = np.mean(np.exp(1j * phases))
        psi = np.angle(z)

        # Gradient: ∂K/∂φᵢ
        grad = np.imag(np.exp(-1j * psi) * np.exp(1j * phases)) / N

        return grad


class BoundaryStress:
    """
    Compute boundary stress metrics for RPST systems.

    Provides two operational definitions:
    1. Frustration stress: Energy in anti-aligned neighbor pairs
    2. Laplacian stress: Squared gradient energy

    Tier 1 Status
    -------------
    These are standard gradient energy definitions used in
    phase-ordering kinetics and XY model analysis.

    Examples
    --------
    >>> stress = BoundaryStress()
    >>> phi = np.array([0, 0.1, 0.2, 3.0, 3.1])  # One defect
    >>> adjacency = np.array([[0,1,0,0,0],
    ...                       [1,0,1,0,0],
    ...                       [0,1,0,1,0],
    ...                       [0,0,1,0,1],
    ...                       [0,0,0,1,0]])
    >>> metrics = stress.compute(phi, adjacency)
    >>> print(f"Total stress: {metrics.total_stress:.3f}")
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize stress calculator.

        Parameters
        ----------
        weights : np.ndarray, optional
            Edge weights wᵢⱼ for frustration calculation
        """
        self.weights = weights

    def compute(self,
                phases: np.ndarray,
                adjacency: np.ndarray,
                boundary_mask: Optional[np.ndarray] = None) -> StressMetrics:
        """
        Compute boundary stress metrics.

        Parameters
        ----------
        phases : np.ndarray
            Phase values φᵢ
        adjacency : np.ndarray
            Adjacency matrix (N×N)
        boundary_mask : np.ndarray, optional
            Boolean mask indicating boundary nodes

        Returns
        -------
        StressMetrics
            Frustration and Laplacian stress measures

        Notes
        -----
        Frustration stress (Eq. stress_frust in BPR literature):
            σ_frust = Σ_{(i,j)∈E} wᵢⱼ (1 - cos(φᵢ - φⱼ))

        Laplacian stress (Eq. stress_lap):
            σ_lap = Σᵢ (Δφᵢ)²

        where Δφᵢ = Σⱼ∈N(i) (φⱼ - φᵢ) is the discrete Laplacian.
        """
        N = len(phases)

        # Compute frustration stress
        sigma_frust = self._frustration_stress(phases, adjacency)

        # Compute Laplacian stress
        sigma_lap = self._laplacian_stress(phases, adjacency)

        # Compute stress field (per-node)
        stress_field = self._local_stress(phases, adjacency)

        # Apply boundary mask if provided
        if boundary_mask is not None:
            boundary_stress_field = stress_field * boundary_mask
            total_stress = np.sum(boundary_stress_field)
        else:
            total_stress = sigma_frust + sigma_lap

        return StressMetrics(
            total_stress=total_stress,
            frustration_stress=sigma_frust,
            laplacian_stress=sigma_lap,
            max_local_stress=np.max(stress_field),
            stress_field=stress_field
        )

    def _frustration_stress(self,
                            phases: np.ndarray,
                            adjacency: np.ndarray) -> float:
        """
        Compute frustration stress.

        σ_frust = Σ_{(i,j)∈E} wᵢⱼ (1 - cos(φᵢ - φⱼ))
        """
        N = len(phases)
        sigma = 0.0

        for i in range(N):
            for j in range(i+1, N):
                if adjacency[i, j] > 0:
                    w_ij = adjacency[i, j]
                    if self.weights is not None:
                        w_ij = self.weights[i, j]
                    sigma += w_ij * (1 - np.cos(phases[i] - phases[j]))

        return sigma

    def _laplacian_stress(self,
                          phases: np.ndarray,
                          adjacency: np.ndarray) -> float:
        """
        Compute Laplacian stress.

        σ_lap = Σᵢ (Δφᵢ)² where Δφᵢ = Σⱼ∈N(i) (φⱼ - φᵢ)
        """
        N = len(phases)
        sigma = 0.0

        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) > 0:
                # Discrete Laplacian
                laplacian_i = np.sum(phases[neighbors] - phases[i])
                sigma += laplacian_i ** 2

        return sigma

    def _local_stress(self,
                      phases: np.ndarray,
                      adjacency: np.ndarray) -> np.ndarray:
        """
        Compute per-node stress field.

        σᵢ = Σⱼ∈N(i) (1 - cos(φᵢ - φⱼ))
        """
        N = len(phases)
        stress = np.zeros(N)

        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]
            for j in neighbors:
                stress[i] += (1 - np.cos(phases[i] - phases[j]))

        return stress


class EligibilityFunctional:
    """
    Compute rewrite eligibility for boundary phase configurations.

    TIER 2 CONJECTURE: This functional is BPR-specific and requires
    further theoretical justification. Use with caution.

    The eligibility functional determines when a discrete "rewrite"
    event should occur based on coherence gradients and boundary stress.

    Form (from BPR literature):
        E(ψ) = ||∇_∂Ω K||₂ · σ(ψ) / (1 + κ_K / K(ψ))

    where:
        K: Global coherence
        σ: Boundary stress
        κ_K: Coherence suppression scale
        ∇_∂Ω K: Boundary gradient of coherence

    A rewrite event is triggered when E(ψ) > ε (threshold).

    Status
    ------
    This is a CONJECTURED functional form. The specific choice of
    terms and their combination lacks first-principles derivation.

    What's justified:
    - Coherence K and stress σ are well-defined (Tier 1)
    - Higher stress + changing coherence = something happening

    What's not justified:
    - Why this particular combination?
    - What sets κ_K and ε?
    - Physical meaning of "rewrite event"
    """

    def __init__(self,
                 kappa_K: float = 0.1,
                 threshold: float = 1.0):
        """
        Initialize eligibility functional.

        Parameters
        ----------
        kappa_K : float
            Coherence suppression scale (default 0.1)
            Higher κ_K suppresses eligibility at low coherence
        threshold : float
            Rewrite threshold ε (default 1.0)

        Notes
        -----
        These parameters are NOT derived from first principles.
        They should be treated as tunable hyperparameters.
        """
        self.kappa_K = kappa_K
        self.threshold = threshold
        self._coherence = PhaseCoherence()
        self._stress = BoundaryStress()

    def compute(self,
                phases: np.ndarray,
                adjacency: np.ndarray,
                phases_prev: Optional[np.ndarray] = None,
                dt: float = 1.0) -> Tuple[float, bool]:
        """
        Compute eligibility and check for rewrite condition.

        Parameters
        ----------
        phases : np.ndarray
            Current phase configuration
        adjacency : np.ndarray
            Adjacency matrix
        phases_prev : np.ndarray, optional
            Previous phase configuration (for gradient)
        dt : float
            Time step (for gradient normalization)

        Returns
        -------
        tuple
            (eligibility_value, triggers_rewrite)

        Notes
        -----
        Formula:
            E(ψ) = ||∇_∂Ω K||₂ · σ(ψ) / (1 + κ_K / K(ψ))

        TIER 2 WARNING: This functional form is conjectured.
        """
        # Compute coherence
        coherence_metrics = self._coherence.compute(phases, adjacency)
        K = coherence_metrics.global_coherence

        # Compute stress
        stress_metrics = self._stress.compute(phases, adjacency)
        sigma = stress_metrics.total_stress

        # Compute coherence gradient (temporal if available)
        if phases_prev is not None:
            K_prev = self._coherence.compute(phases_prev, adjacency).global_coherence
            dK_dt = (K - K_prev) / dt
            grad_K_norm = np.abs(dK_dt)
        else:
            # Use spatial gradient as proxy
            grad_K = self._coherence.compute_gradient(phases)
            grad_K_norm = np.linalg.norm(grad_K)

        # Compute eligibility
        # E = ||∇K|| · σ / (1 + κ_K / K)
        denominator = 1 + self.kappa_K / (K + 1e-10)  # Avoid division by zero
        eligibility = grad_K_norm * sigma / denominator

        # Check threshold (convert to Python bool to avoid numpy.bool_ issues)
        triggers_rewrite = bool(eligibility > self.threshold)

        return float(eligibility), triggers_rewrite

    def scan_trajectory(self,
                        trajectory: np.ndarray,
                        adjacency: np.ndarray,
                        dt: float = 1.0) -> List[Tuple[int, float]]:
        """
        Scan trajectory for rewrite events.

        Parameters
        ----------
        trajectory : np.ndarray
            Phase evolution (T × N array)
        adjacency : np.ndarray
            Adjacency matrix
        dt : float
            Time step

        Returns
        -------
        list
            List of (time_index, eligibility) for triggered events
        """
        T = trajectory.shape[0]
        events = []

        for t in range(1, T):
            eligibility, triggers = self.compute(
                trajectory[t],
                adjacency,
                trajectory[t-1],
                dt
            )
            if triggers:
                events.append((t, eligibility))

        return events


def create_ring_lattice(N: int, k: int = 2) -> np.ndarray:
    """
    Create ring lattice adjacency matrix.

    Parameters
    ----------
    N : int
        Number of nodes
    k : int
        Number of neighbors on each side

    Returns
    -------
    np.ndarray
        Adjacency matrix (N × N)
    """
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            adj[i, (i+j) % N] = 1
            adj[i, (i-j) % N] = 1
    return adj


def create_square_lattice(L: int) -> np.ndarray:
    """
    Create 2D square lattice adjacency matrix.

    Parameters
    ----------
    L : int
        Side length (N = L² nodes)

    Returns
    -------
    np.ndarray
        Adjacency matrix (N × N)
    """
    N = L * L
    adj = np.zeros((N, N))

    for i in range(L):
        for j in range(L):
            idx = i * L + j
            # Right neighbor
            if j < L - 1:
                adj[idx, idx + 1] = 1
                adj[idx + 1, idx] = 1
            # Down neighbor
            if i < L - 1:
                adj[idx, idx + L] = 1
                adj[idx + L, idx] = 1

    return adj


def run_coherence_stress_demo():
    """Demonstration of coherence and stress calculations."""
    print("RPST Coherence and Stress Module Demo")
    print("=" * 50)

    # Create ring lattice
    N = 20
    adj = create_ring_lattice(N, k=2)

    # Test 1: Synchronized phases
    phi_sync = np.zeros(N)
    coherence = PhaseCoherence()
    stress = BoundaryStress()

    metrics_sync = coherence.compute(phi_sync, adj)
    stress_sync = stress.compute(phi_sync, adj)

    print("\n1. Synchronized configuration:")
    print(f"   {metrics_sync}")
    print(f"   {stress_sync}")

    # Test 2: Random phases
    np.random.seed(42)
    phi_random = np.random.uniform(0, 2*np.pi, N)

    metrics_random = coherence.compute(phi_random, adj)
    stress_random = stress.compute(phi_random, adj)

    print("\n2. Random configuration:")
    print(f"   {metrics_random}")
    print(f"   {stress_random}")

    # Test 3: Single defect
    phi_defect = np.linspace(0, 2*np.pi, N)  # Smooth gradient
    phi_defect[N//2] = phi_defect[N//2] + np.pi  # Add defect

    metrics_defect = coherence.compute(phi_defect, adj)
    stress_defect = stress.compute(phi_defect, adj)

    print("\n3. Single defect configuration:")
    print(f"   {metrics_defect}")
    print(f"   {stress_defect}")

    # Test eligibility (Tier 2)
    print("\n4. Eligibility functional (TIER 2 CONJECTURE):")
    eligibility = EligibilityFunctional(kappa_K=0.1, threshold=1.0)
    E, triggers = eligibility.compute(phi_defect, adj)
    print(f"   Eligibility E = {E:.4f}")
    print(f"   Triggers rewrite: {triggers}")

    return True


if __name__ == "__main__":
    run_coherence_stress_demo()
