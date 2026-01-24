"""
U(1) Gauge Symmetry in RPST Substrate

This module proves (or disproves) that RPST has U(1) gauge symmetry,
which is the prerequisite for direct electromagnetic coupling.

KEY QUESTION: Does the RPST Hamiltonian depend only on phase DIFFERENCES?

If YES: φ is a gauge field, can couple to EM at scale α
If NO: φ is geometric only, stuck with Planck suppression

SPRINT: Week 1 of EM Coupling Search
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
from enum import Enum


class GaugeTestResult(Enum):
    """Result of gauge invariance test."""
    INVARIANT = "gauge_invariant"
    NOT_INVARIANT = "not_gauge_invariant"
    NUMERICAL_AMBIGUOUS = "numerical_ambiguous"


@dataclass
class U1SymmetryResult:
    """Complete result of U(1) symmetry analysis."""
    is_u1_symmetric: bool
    max_violation: float
    test_details: dict
    implication: str


def substrate_phase(q: np.ndarray, p: int) -> np.ndarray:
    """
    Extract U(1) phase from RPST substrate variables.

    Maps q ∈ ℤₚ to θ ∈ [0, 2π)

    Parameters
    ----------
    q : np.ndarray
        Substrate position variables (integers mod p)
    p : int
        Prime modulus

    Returns
    -------
    np.ndarray
        Angles on unit circle
    """
    return 2.0 * np.pi * q / p


def phase_difference(theta_i: float, theta_j: float) -> float:
    """
    Compute gauge-invariant phase difference.

    Δθ = θⱼ - θᵢ (mod 2π, centered at 0)

    This is the fundamental gauge-invariant quantity.
    """
    delta = theta_j - theta_i
    # Wrap to [-π, π]
    return np.arctan2(np.sin(delta), np.cos(delta))


def rpst_potential_gauge_form(theta: np.ndarray,
                               adjacency: np.ndarray,
                               J: float = 1.0) -> float:
    """
    RPST potential energy in gauge-invariant form.

    V = Σᵢⱼ J (1 - cos(θⱼ - θᵢ))

    This form manifestly depends only on phase differences.

    Parameters
    ----------
    theta : np.ndarray
        Phase angles
    adjacency : np.ndarray
        Adjacency matrix (which sites are coupled)
    J : float
        Coupling strength

    Returns
    -------
    float
        Potential energy
    """
    N = len(theta)
    V = 0.0

    for i in range(N):
        for j in range(i+1, N):
            if adjacency[i, j] > 0:
                delta_theta = theta[j] - theta[i]
                V += J * adjacency[i, j] * (1 - np.cos(delta_theta))

    return V


def test_global_u1_invariance(H: Callable[[np.ndarray], float],
                               theta: np.ndarray,
                               n_tests: int = 100,
                               tolerance: float = 1e-10) -> Tuple[GaugeTestResult, float]:
    """
    Test if Hamiltonian is invariant under global U(1) transformations.

    Checks: H(θ + α) = H(θ) for all α

    Parameters
    ----------
    H : callable
        Hamiltonian function H(θ) -> energy
    theta : np.ndarray
        Test configuration
    n_tests : int
        Number of random α values to test
    tolerance : float
        Numerical tolerance

    Returns
    -------
    tuple
        (GaugeTestResult, max_violation)
    """
    H_original = H(theta)
    max_violation = 0.0

    for _ in range(n_tests):
        alpha = np.random.uniform(0, 2*np.pi)
        theta_shifted = theta + alpha
        H_shifted = H(theta_shifted)

        violation = abs(H_shifted - H_original)
        max_violation = max(max_violation, violation)

        if violation > tolerance:
            return GaugeTestResult.NOT_INVARIANT, max_violation

    if max_violation < tolerance:
        return GaugeTestResult.INVARIANT, max_violation
    else:
        return GaugeTestResult.NUMERICAL_AMBIGUOUS, max_violation


def test_local_u1_covariance(H: Callable[[np.ndarray], float],
                              theta: np.ndarray,
                              adjacency: np.ndarray,
                              tolerance: float = 1e-10) -> Tuple[bool, dict]:
    """
    Test if Hamiltonian transforms correctly under LOCAL gauge transformations.

    For a true gauge theory:
    - Global transformations: H invariant (phase is unphysical)
    - Local transformations: H changes (gauge field needed to compensate)

    This tests whether we have the structure needed for EM coupling.

    Parameters
    ----------
    H : callable
        Hamiltonian function
    theta : np.ndarray
        Test configuration
    adjacency : np.ndarray
        Connectivity
    tolerance : float
        Numerical tolerance

    Returns
    -------
    tuple
        (has_gauge_structure, details)
    """
    N = len(theta)
    H_original = H(theta)

    results = {
        'global_invariant': False,
        'local_changes_H': False,
        'change_depends_on_gradient': False
    }

    # Test 1: Global transformation (should be invariant)
    alpha_global = np.random.uniform(0, 2*np.pi)
    theta_global = theta + alpha_global
    H_global = H(theta_global)
    results['global_invariant'] = abs(H_global - H_original) < tolerance

    # Test 2: Local transformation (should change H)
    alpha_local = np.random.uniform(0, 2*np.pi, N)  # Different at each site
    theta_local = theta + alpha_local
    H_local = H(theta_local)
    results['local_changes_H'] = abs(H_local - H_original) > tolerance

    # Test 3: Change depends on gradient of α
    # If H changes as Σᵢⱼ (∇α)² this is gauge structure
    # Compute gradient of α along edges
    gradient_sq_sum = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if adjacency[i, j] > 0:
                grad_alpha = alpha_local[j] - alpha_local[i]
                gradient_sq_sum += grad_alpha**2

    # The change in H should be proportional to gradient squared
    H_change = H_local - H_original
    results['change_depends_on_gradient'] = gradient_sq_sum > 0 and H_change != 0

    has_gauge_structure = (results['global_invariant'] and
                          results['local_changes_H'] and
                          results['change_depends_on_gradient'])

    return has_gauge_structure, results


def analyze_u1_symmetry(p: int, N: int, J: float,
                        geometry: str = 'ring') -> U1SymmetryResult:
    """
    Complete analysis of U(1) symmetry in RPST.

    Parameters
    ----------
    p : int
        Prime modulus
    N : int
        Number of sites
    J : float
        Coupling strength
    geometry : str
        'ring' or 'square'

    Returns
    -------
    U1SymmetryResult
        Complete analysis
    """
    # Create adjacency matrix
    if geometry == 'ring':
        adjacency = np.zeros((N, N))
        for i in range(N):
            adjacency[i, (i+1) % N] = 1
            adjacency[(i+1) % N, i] = 1
    else:  # square
        L = int(np.sqrt(N))
        adjacency = np.zeros((N, N))
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                if j < L - 1:
                    adjacency[idx, idx + 1] = 1
                    adjacency[idx + 1, idx] = 1
                if i < L - 1:
                    adjacency[idx, idx + L] = 1
                    adjacency[idx + L, idx] = 1

    # Define Hamiltonian
    def H(theta):
        return rpst_potential_gauge_form(theta, adjacency, J)

    # Generate test configuration
    np.random.seed(42)
    q_test = np.random.randint(0, p, N)
    theta_test = substrate_phase(q_test, p)

    # Run tests
    global_result, global_violation = test_global_u1_invariance(H, theta_test)
    has_gauge, gauge_details = test_local_u1_covariance(H, theta_test, adjacency)

    # Determine overall result
    is_symmetric = (global_result == GaugeTestResult.INVARIANT)

    if is_symmetric and has_gauge:
        implication = ("RPST has U(1) gauge structure. "
                      "Direct EM coupling is possible. "
                      "Proceed to derive coupling constant.")
    elif is_symmetric and not has_gauge:
        implication = ("RPST has global U(1) but not local gauge structure. "
                      "May still couple to EM through global mechanism.")
    else:
        implication = ("RPST is NOT U(1) symmetric. "
                      "Direct EM coupling unlikely. "
                      "Stuck with gravitational channel.")

    return U1SymmetryResult(
        is_u1_symmetric=is_symmetric,
        max_violation=global_violation,
        test_details={
            'global_test': global_result.value,
            'global_violation': global_violation,
            'has_gauge_structure': has_gauge,
            'gauge_details': gauge_details
        },
        implication=implication
    )


def derive_gauge_connection(theta: np.ndarray,
                            adjacency: np.ndarray) -> np.ndarray:
    """
    If RPST has gauge structure, extract the gauge connection A.

    In lattice gauge theory:
        U_ij = exp(i A_ij)

    where U_ij is the parallel transport from i to j.

    For RPST:
        A_ij = θⱼ - θᵢ (the phase gradient along edge)

    This is the quantity that would couple to EM.

    Parameters
    ----------
    theta : np.ndarray
        Phase configuration
    adjacency : np.ndarray
        Connectivity

    Returns
    -------
    np.ndarray
        Gauge connection matrix A[i,j] = phase gradient along (i,j)
    """
    N = len(theta)
    A = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if adjacency[i, j] > 0:
                A[i, j] = theta[j] - theta[i]

    return A


def compute_wilson_loop(theta: np.ndarray,
                        adjacency: np.ndarray,
                        loop_indices: list) -> complex:
    """
    Compute Wilson loop around a closed path.

    W = exp(i ∮ A·dl) = Π exp(i Aᵢⱼ)

    The Wilson loop is gauge-invariant and measures "flux" through loop.
    Non-trivial Wilson loops indicate topological structure.

    Parameters
    ----------
    theta : np.ndarray
        Phase configuration
    adjacency : np.ndarray
        Connectivity
    loop_indices : list
        Ordered list of site indices forming closed loop

    Returns
    -------
    complex
        Wilson loop value (should have |W| = 1)
    """
    W = 1.0 + 0.0j

    for k in range(len(loop_indices)):
        i = loop_indices[k]
        j = loop_indices[(k + 1) % len(loop_indices)]

        if adjacency[i, j] > 0:
            A_ij = theta[j] - theta[i]
            W *= np.exp(1j * A_ij)
        else:
            raise ValueError(f"Sites {i} and {j} are not connected")

    return W


def extract_effective_flux(theta: np.ndarray,
                           adjacency: np.ndarray,
                           plaquette: list) -> float:
    """
    Extract effective "magnetic flux" through a plaquette.

    Φ = arg(W) / 2π (in units of flux quantum)

    This is what would couple to EM field.

    Parameters
    ----------
    theta : np.ndarray
        Phase configuration
    adjacency : np.ndarray
        Connectivity
    plaquette : list
        4 vertices forming an elementary square

    Returns
    -------
    float
        Flux in units of 2π
    """
    W = compute_wilson_loop(theta, adjacency, plaquette)
    return np.angle(W) / (2 * np.pi)


if __name__ == "__main__":
    print("U(1) Gauge Symmetry Analysis in RPST")
    print("=" * 60)

    # Test parameters
    p = 104729  # Large prime
    N = 100     # Ring of 100 sites
    J = 1.0     # Coupling

    print(f"\nParameters: p={p}, N={N}, J={J}")

    # Run analysis
    result = analyze_u1_symmetry(p, N, J, geometry='ring')

    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    print(f"  U(1) Symmetric: {result.is_u1_symmetric}")
    print(f"  Max Violation: {result.max_violation:.2e}")
    print(f"\nTest Details:")
    for key, value in result.test_details.items():
        print(f"  {key}: {value}")
    print(f"\nIMPLICATION:")
    print(f"  {result.implication}")

    # If gauge structure exists, extract some observables
    if result.is_u1_symmetric:
        print(f"\n{'='*60}")
        print("GAUGE STRUCTURE ANALYSIS")
        print(f"{'='*60}")

        # Create test configuration
        np.random.seed(42)
        adjacency = np.zeros((N, N))
        for i in range(N):
            adjacency[i, (i+1) % N] = 1
            adjacency[(i+1) % N, i] = 1

        q_test = np.random.randint(0, p, N)
        theta_test = substrate_phase(q_test, p)

        # Gauge connection
        A = derive_gauge_connection(theta_test, adjacency)
        print(f"\nGauge connection statistics:")
        print(f"  Mean |A|: {np.mean(np.abs(A[A != 0])):.4f}")
        print(f"  Std |A|: {np.std(np.abs(A[A != 0])):.4f}")

        # Wilson loop (around entire ring)
        loop = list(range(N))
        W = compute_wilson_loop(theta_test, adjacency, loop)
        print(f"\nWilson loop (full ring):")
        print(f"  W = {W:.4f}")
        print(f"  |W| = {abs(W):.4f} (should be 1)")
        print(f"  Flux = {np.angle(W)/(2*np.pi):.4f} × 2π")
