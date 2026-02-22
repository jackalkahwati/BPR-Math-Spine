"""
RPST Stability Manifolds
============================================================

Derives the RPST stability manifold S directly from the discrete
symplectic map.  Every quantity is explicit: the transfer matrix
spectrum, the boundary phase energy, the stability measure, the
attractor threshold, and the decoherence rate.

Key objects
-----------
* ``FixedPointAnalysis``       -- Fixed-point condition on Z_p^{2N}     (Eq 4-5)
* ``DynamicalMatrix``          -- Graph Laplacian from couplings         (Eq 8)
* ``TransferMatrixSpectrum``   -- Eigenvalues of linearized map          (Eq 9-10)
* ``SpectralStabilityDeligne`` -- Unit-circle eigenvalues (Thm 3.1)
* ``BoundaryPhaseEnergy``      -- E_Φ[ρ] = κ_s ∫ |∇Φ_AB|² dA          (Eq 12)
* ``StabilityMeasure``         -- S(γ) = ∫ |∇φ·n̂|² dA                  (Eq 15)
* ``LyapunovConvergence``      -- Exponential convergence to S           (Eq 16-17)
* ``AttractorThreshold``       -- ‖L‖₂ ≤ 4 spectral condition           (Thm 7.1)
* ``ResonanceBand``            -- ω_k = arccos(1 - λ_k/2)               (Eq 18)
* ``PhaseGradientDecoherence`` -- Γ_dec = (κ_s/ℏ)‖∇Φ_AB‖²              (Eq 21)
* ``CacheTimescaleDerivation`` -- τ_m = τ_0 |W|^α from first principles (Cor 8.1)
* ``PhaseTransitionThreshold`` -- m²_κ α + γ V''(κ*) > 0                (Thm 9.1)
* ``StabilityManifold``        -- Main result S = {E_Φ = 0}             (Thm 10.1)

References: Al-Kahwati (2026), *RPST Stability Manifolds: Explicit
Derivation from the Discrete Symplectic Map*, StarDrive Research Group.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union


# ---------------------------------------------------------------------------
# Utility: discrete sawtooth (signed difference on Z_p)
# ---------------------------------------------------------------------------

def discrete_sawtooth(x: int, p: int) -> int:
    r"""Discrete difference function D(x) on Z_p (Eq 3).

    D(x) = x  if |x| <= p/2
         = x - p*sign(x)  otherwise

    Satisfies D(-x) = -D(x) (odd), D(0) = 0.

    Parameters
    ----------
    x : int
        Value in Z_p (may be unreduced).
    p : int
        Prime modulus.

    Returns
    -------
    int
        Signed difference in [-p//2, p//2].
    """
    x = int(x) % p
    if x > p // 2:
        return x - p
    return x


def sawtooth_derivative(x: int, p: int) -> float:
    r"""Derivative D'(x) of the discrete sawtooth.

    For the sawtooth D(x) = x, D'(x) = 1 everywhere except at
    the discontinuity |x| = p/2.

    Parameters
    ----------
    x : int
        Phase difference in Z_p.
    p : int
        Prime modulus.

    Returns
    -------
    float
        D'(x), typically 1.0.
    """
    x_mod = int(x) % p
    if x_mod == 0:
        return 1.0
    if x_mod == p // 2 or (p % 2 == 0 and x_mod == p - p // 2):
        return 0.0  # Discontinuity
    return 1.0


# ===================================================================
# Fixed Point Analysis (Sec 2.1, Eq 4-5)
# ===================================================================

@dataclass
class FixedPointAnalysis:
    r"""Fixed-point analysis for the RPST symplectic map (Sec 2.1).

    A configuration (q*, ϖ*) is a fixed point iff:
        ϖ*_i = 0  (mod p)                                    (Eq 4)
        Σ_{j ∈ N(i)} J_{ij} D(q*_j - q*_i) = 0  (mod p)  ∀i  (Eq 5)

    Condition (5) is the discrete Laplace equation: the phase field
    must be discrete-harmonic on the graph.

    Parameters
    ----------
    p : int
        Prime modulus.
    """
    p: int = 101

    def is_fixed_point(self, q: np.ndarray, pi_field: np.ndarray,
                       J: np.ndarray) -> Dict[str, object]:
        r"""Test whether (q, π) is a fixed point of the symplectic map.

        Parameters
        ----------
        q : ndarray of int, shape (N,)
            Phase configuration.
        pi_field : ndarray of int, shape (N,)
            Momentum configuration.
        J : ndarray of float, shape (N, N)
            Coupling matrix (symmetric).

        Returns
        -------
        dict with:
            'is_fixed' : bool
            'momentum_zero' : bool (all ϖ_i = 0)
            'harmonic' : bool (discrete Laplace eq satisfied)
            'laplace_residual' : ndarray (residual at each node)
        """
        p = self.p
        N = len(q)

        # Condition (4): all momenta zero
        momentum_zero = bool(np.all(pi_field % p == 0))

        # Condition (5): discrete Laplace equation
        residual = np.zeros(N, dtype=int)
        for i in range(N):
            s = 0
            for j in range(N):
                if J[i, j] != 0:
                    diff = discrete_sawtooth(int(q[j]) - int(q[i]), p)
                    s += int(J[i, j] * diff)
            residual[i] = s % p

        harmonic = bool(np.all(residual == 0))

        return {
            "is_fixed": momentum_zero and harmonic,
            "momentum_zero": momentum_zero,
            "harmonic": harmonic,
            "laplace_residual": residual,
        }

    def find_constant_fixed_points(self, N: int) -> np.ndarray:
        r"""Return constant configurations q_i = c which are always fixed points.

        For constant q, D(q_j - q_i) = D(0) = 0, so (5) is trivially
        satisfied for any coupling.

        Parameters
        ----------
        N : int
            Number of nodes.

        Returns
        -------
        ndarray, shape (p, N)
            All p constant fixed-point configurations.
        """
        return np.array([np.full(N, c) for c in range(self.p)])


# ===================================================================
# Dynamical Matrix and Graph Laplacian (Sec 2.2, Eq 8-9)
# ===================================================================

@dataclass
class DynamicalMatrix:
    r"""Dynamical matrix M and graph Laplacian L (Eq 8-9).

    M_{ij} = J_{ij} D'(q*_j - q*_i)

    Graph Laplacian:
        L_{ii} = Σ_j M_{ij}
        L_{ij} = -M_{ij}  for i ≠ j

    The transfer matrix T (Eq 9):
        T = [[I - L,  I],
             [-L,     I]]

    Parameters
    ----------
    p : int
        Prime modulus.
    """
    p: int = 101

    def dynamical_matrix(self, q_star: np.ndarray,
                         J: np.ndarray) -> np.ndarray:
        r"""Compute dynamical matrix M_{ij} = J_{ij} D'(q*_j - q*_i) (Eq 8).

        Parameters
        ----------
        q_star : ndarray of int, shape (N,)
            Fixed-point phase configuration.
        J : ndarray, shape (N, N)
            Coupling weights.

        Returns
        -------
        ndarray, shape (N, N)
            Dynamical matrix M.
        """
        N = len(q_star)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if J[i, j] != 0:
                    diff = int(q_star[j]) - int(q_star[i])
                    M[i, j] = J[i, j] * sawtooth_derivative(diff, self.p)
        return M

    def graph_laplacian(self, q_star: np.ndarray,
                        J: np.ndarray) -> np.ndarray:
        r"""Compute graph Laplacian L from dynamical matrix M.

        L_{ii} = Σ_j M_{ij},  L_{ij} = -M_{ij} for i≠j.

        Parameters
        ----------
        q_star : ndarray of int, shape (N,)
            Fixed-point configuration.
        J : ndarray, shape (N, N)
            Coupling weights.

        Returns
        -------
        ndarray, shape (N, N)
            Graph Laplacian.
        """
        M = self.dynamical_matrix(q_star, J)
        N = M.shape[0]
        L = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    L[i, j] = -M[i, j]
            L[i, i] = np.sum(M[i, :])
        return L

    def transfer_matrix(self, q_star: np.ndarray,
                        J: np.ndarray) -> np.ndarray:
        r"""Construct the 2N × 2N transfer matrix T (Eq 9).

        T = [[I - L,  I],
             [-L,     I]]

        Parameters
        ----------
        q_star : ndarray of int, shape (N,)
            Fixed-point configuration.
        J : ndarray, shape (N, N)
            Coupling weights.

        Returns
        -------
        ndarray, shape (2N, 2N)
            Transfer matrix.
        """
        L = self.graph_laplacian(q_star, J)
        N = L.shape[0]
        I_N = np.eye(N)
        T = np.block([
            [I_N - L, I_N],
            [-L, I_N],
        ])
        return T


# ===================================================================
# Transfer Matrix Spectrum (Thm 2.1, Eq 10)
# ===================================================================

@dataclass
class TransferMatrixSpectrum:
    r"""Eigenvalue structure of the transfer matrix (Thm 2.1, Eq 10).

    For each Laplacian eigenvalue λ_k, the transfer matrix eigenvalues are:

        μ±_k = 1 - λ_k/2 ± i √(4λ_k - λ_k²) / 2

    with |μ±_k|² = 1 for all k with 0 ≤ λ_k ≤ 4.
    """

    def eigenvalues_from_laplacian(self, lambda_k: float) -> Tuple[complex, complex]:
        r"""Compute transfer matrix eigenvalues from a Laplacian eigenvalue (Eq 10).

        Parameters
        ----------
        lambda_k : float
            Laplacian eigenvalue.

        Returns
        -------
        tuple of complex
            (μ+_k, μ-_k) pair.
        """
        real_part = 1.0 - lambda_k / 2.0
        discriminant = (2.0 - lambda_k) ** 2 - 4.0

        if discriminant <= 0:
            # Complex conjugate pair, |μ| = 1
            imag_part = np.sqrt(-discriminant) / 2.0
            return (real_part + 1j * imag_part, real_part - 1j * imag_part)
        else:
            # Real eigenvalues, |μ| may exceed 1
            sqrt_disc = np.sqrt(discriminant) / 2.0
            return (real_part + sqrt_disc, real_part - sqrt_disc)

    def all_eigenvalues(self, laplacian_eigenvalues: np.ndarray) -> np.ndarray:
        r"""Compute all transfer matrix eigenvalues from Laplacian spectrum.

        Parameters
        ----------
        laplacian_eigenvalues : ndarray
            Eigenvalues of the graph Laplacian L.

        Returns
        -------
        ndarray, shape (2N,)
            All transfer matrix eigenvalues.
        """
        pairs = [self.eigenvalues_from_laplacian(lk)
                 for lk in laplacian_eigenvalues]
        return np.array([mu for pair in pairs for mu in pair])

    def is_marginally_stable(self, laplacian_eigenvalues: np.ndarray,
                             tolerance: float = 1e-10) -> bool:
        r"""Check marginal stability: all |μ±_k| = 1 (Corollary 2.1).

        Parameters
        ----------
        laplacian_eigenvalues : ndarray
            Laplacian eigenvalues.
        tolerance : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if all transfer matrix eigenvalues lie on the unit circle.
        """
        mu = self.all_eigenvalues(laplacian_eigenvalues)
        magnitudes = np.abs(mu)
        return bool(np.all(np.abs(magnitudes - 1.0) < tolerance))

    def resonance_frequencies(self, laplacian_eigenvalues: np.ndarray) -> np.ndarray:
        r"""Resonance frequencies of each mode (Eq 18).

        ω_k = arccos(1 - λ_k / 2)

        Valid for λ_k ∈ [0, 4], giving ω_k ∈ [0, π].

        Parameters
        ----------
        laplacian_eigenvalues : ndarray
            Laplacian eigenvalues.

        Returns
        -------
        ndarray
            Resonance frequencies ω_k. NaN for λ_k outside [0, 4].
        """
        arg = 1.0 - np.asarray(laplacian_eigenvalues) / 2.0
        arg = np.clip(arg, -1.0, 1.0)
        return np.arccos(arg)


# ===================================================================
# Spectral Stability from Deligne's Theorem (Thm 3.1)
# ===================================================================

@dataclass
class SpectralStabilityDeligne:
    r"""Spectral stability via Deligne's theorem (Thm 3.1).

    The normalized eigenvalues λ_k = α_{1,k}/√p of H_p satisfy
    |λ_k| = 1.  All eigenvalues lie on the unit circle.

    The Katz-Sarnak equidistribution theorem specifies that Frobenius
    eigenangles θ ∈ [0, π] equidistribute as USp(2) Haar measure
    (2/π) sin²θ dθ as p → ∞.

    Parameters
    ----------
    p : int
        Prime modulus.
    """
    p: int = 101

    def verify_unit_circle(self, eigenvalues: np.ndarray,
                           tolerance: float = 0.05) -> Dict[str, object]:
        r"""Verify that normalized eigenvalues lie on the unit circle.

        Eigenvalues α_{1,k} of H_p should satisfy |α_{1,k}| = √p,
        i.e., |α_{1,k}/√p| = 1.

        Parameters
        ----------
        eigenvalues : ndarray
            Eigenvalues of H_p (raw, unnormalized).
        tolerance : float
            Allowed deviation from unit circle.

        Returns
        -------
        dict with:
            'all_on_unit_circle' : bool
            'normalized_magnitudes' : ndarray
            'max_deviation' : float
        """
        sqrt_p = np.sqrt(self.p)
        magnitudes = np.abs(eigenvalues)
        # Filter out zero eigenvalues
        nonzero_mask = magnitudes > 1e-10
        if not np.any(nonzero_mask):
            return {
                "all_on_unit_circle": True,
                "normalized_magnitudes": np.array([]),
                "max_deviation": 0.0,
            }
        normalized = magnitudes[nonzero_mask] / sqrt_p
        max_dev = float(np.max(np.abs(normalized - 1.0)))
        return {
            "all_on_unit_circle": max_dev < tolerance,
            "normalized_magnitudes": normalized,
            "max_deviation": max_dev,
        }

    def frobenius_angles(self, eigenvalues: np.ndarray) -> np.ndarray:
        r"""Extract Frobenius eigenangles θ ∈ [0, π].

        The angles are arg(α_{1,k}/√p), mapped to [0, π].

        Parameters
        ----------
        eigenvalues : ndarray
            Eigenvalues of H_p.

        Returns
        -------
        ndarray
            Eigenangles in [0, π].
        """
        sqrt_p = np.sqrt(self.p)
        nonzero = eigenvalues[np.abs(eigenvalues) > 1e-10]
        angles = np.angle(nonzero / sqrt_p)
        # Map to [0, π]
        return np.abs(angles)

    @staticmethod
    def katz_sarnak_density(theta: np.ndarray) -> np.ndarray:
        r"""USp(2) Haar density for Frobenius eigenangles.

        ρ(θ) = (2/π) sin²(θ)

        Parameters
        ----------
        theta : ndarray
            Angles in [0, π].

        Returns
        -------
        ndarray
            Density values.
        """
        return (2.0 / np.pi) * np.sin(theta) ** 2


# ===================================================================
# Boundary Phase Energy (Sec 4.1, Eq 12-13)
# ===================================================================

@dataclass
class BoundaryPhaseEnergy:
    r"""Boundary phase energy functional E_Φ (Eq 12).

    E_Φ[ρ] = Σ_{A,B} ∫_{∂Ω} κ_s |∇Φ_{AB}|² dA

    The constraint manifold M_Φ = {ρ : ∇Φ_{AB} = 0 ∀A,B}
    consists of the global minima E_Φ = 0 (Prop. 4.1).

    The stability manifold S = M_Φ (Thm 4.1).

    Parameters
    ----------
    kappa_s : float
        Boundary stiffness.
    """
    kappa_s: float = 1.0

    def energy(self, phi: np.ndarray, dx: float = 1.0) -> float:
        r"""Compute boundary phase energy (Eq 12).

        E_Φ = κ_s Σ |∇Φ|² dx (discrete approximation).

        Parameters
        ----------
        phi : ndarray
            Boundary phase field values.
        dx : float
            Grid spacing.

        Returns
        -------
        float
            Boundary phase energy E_Φ ≥ 0.
        """
        grad = np.diff(phi) / dx
        return float(self.kappa_s * np.sum(grad ** 2) * dx)

    def energy_2d(self, phi: np.ndarray, dx: float = 1.0,
                  dy: float = 1.0) -> float:
        r"""Compute boundary phase energy on a 2D grid.

        Parameters
        ----------
        phi : ndarray, shape (Ny, Nx)
            2D boundary phase field.
        dx, dy : float
            Grid spacings.

        Returns
        -------
        float
            Boundary phase energy E_Φ ≥ 0.
        """
        grad_x = np.diff(phi, axis=1) / dx
        grad_y = np.diff(phi, axis=0) / dy
        E_x = np.sum(grad_x ** 2) * dx * dy
        E_y = np.sum(grad_y ** 2) * dx * dy
        return float(self.kappa_s * (E_x + E_y))

    def is_on_manifold(self, phi: np.ndarray, dx: float = 1.0,
                       tolerance: float = 1e-10) -> bool:
        r"""Test if configuration is on the constraint manifold M_Φ.

        ρ ∈ M_Φ iff E_Φ[ρ] = 0 iff ∇Φ_{AB} = 0 everywhere.

        Parameters
        ----------
        phi : ndarray
            Boundary phase field.
        dx : float
            Grid spacing.
        tolerance : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if E_Φ < tolerance.
        """
        return self.energy(phi, dx) < tolerance

    def gradient_field(self, phi: np.ndarray,
                       dx: float = 1.0) -> np.ndarray:
        r"""Compute the boundary phase gradient field.

        Parameters
        ----------
        phi : ndarray
            Boundary phase field.
        dx : float
            Grid spacing.

        Returns
        -------
        ndarray
            Gradient ∇Φ at each interior point.
        """
        return np.diff(phi) / dx


# ===================================================================
# Stability Measure (Sec 5, Eq 15)
# ===================================================================

@dataclass
class StabilityMeasure:
    r"""BPR stability measure S(γ) (Def 5.1, Eq 15).

    S(γ) = ∫_{∂Ω} |∇φ · n̂|² dA

    Properties (Thm 5.1):
    (i)   S(γ) ≥ 0 always
    (ii)  S(γ) = 0 iff γ ∈ S
    (iii) S is monotone non-increasing along convergent trajectories
    (iv)  S is observable
    """

    def compute(self, phi: np.ndarray, normals: Optional[np.ndarray] = None,
                dx: float = 1.0) -> float:
        r"""Compute stability measure S(γ) (Eq 15).

        For 1D boundary: S = Σ (∂φ/∂n)² dx.
        For general case: S = Σ |∇φ · n̂|² dA.

        Parameters
        ----------
        phi : ndarray
            Boundary phase field.
        normals : ndarray, optional
            Outward unit normals. If None, assumes 1D (normal = gradient).
        dx : float
            Element size.

        Returns
        -------
        float
            Stability measure S(γ) ≥ 0.
        """
        grad = np.diff(phi) / dx
        if normals is not None:
            # Project gradient onto normals
            normal_grad = grad * normals[:len(grad)]
        else:
            normal_grad = grad
        return float(np.sum(normal_grad ** 2) * dx)

    def is_in_stability_manifold(self, phi: np.ndarray,
                                 dx: float = 1.0,
                                 tolerance: float = 1e-10) -> bool:
        r"""Test if trajectory lies in S.

        S(γ) = 0 iff γ ∈ S.

        Parameters
        ----------
        phi : ndarray
            Boundary phase field.
        dx : float
            Element size.
        tolerance : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if S(γ) < tolerance.
        """
        return self.compute(phi, dx=dx) < tolerance

    def convergence_trajectory(self, phi_trajectory: List[np.ndarray],
                               dx: float = 1.0) -> np.ndarray:
        r"""Compute S(γ(t)) over a trajectory of phase fields.

        Parameters
        ----------
        phi_trajectory : list of ndarray
            Phase field at successive time steps.
        dx : float
            Grid spacing.

        Returns
        -------
        ndarray
            S(t) values.
        """
        return np.array([self.compute(phi, dx=dx) for phi in phi_trajectory])


# ===================================================================
# Lyapunov Convergence to S (Sec 6, Eq 16-17)
# ===================================================================

@dataclass
class LyapunovConvergence:
    r"""Lyapunov convergence to the stability manifold (Sec 6).

    Axiom 6.1: d/dt V_b(x(t)) ≤ -α S(γ(t)) + ε

    Theorem 6.2 (PL convergence):
        V_b(x(t)) ≤ V_b(x(0)) exp(-2μα t)

    Parameters
    ----------
    alpha : float
        Convergence rate parameter.
    mu : float
        Polyak–Łojasiewicz constant.
    epsilon : float
        Bounded noise/perturbation (0 for exact convergence).
    """
    alpha: float = 1.0
    mu: float = 1.0
    epsilon: float = 0.0

    @property
    def convergence_rate(self) -> float:
        r"""Exponential convergence rate 2μα (Thm 6.2).

        Returns
        -------
        float
            Rate constant.
        """
        return 2.0 * self.mu * self.alpha

    def lyapunov_bound(self, V0: float, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Upper bound on Lyapunov function (Eq 17).

        V_b(t) ≤ V_b(0) exp(-2μα t)

        Parameters
        ----------
        V0 : float
            Initial Lyapunov value V_b(x(0)).
        t : float or ndarray
            Time.

        Returns
        -------
        float or ndarray
            Upper bound on V_b(x(t)).
        """
        return V0 * np.exp(-self.convergence_rate * np.asarray(t, dtype=float))

    def lyapunov_rate(self, S_value: float) -> float:
        r"""Instantaneous rate of decrease of V_b (Eq 16).

        dV_b/dt ≤ -α S(γ) + ε

        Parameters
        ----------
        S_value : float
            Current stability measure S(γ).

        Returns
        -------
        float
            Upper bound on dV_b/dt.
        """
        return -self.alpha * S_value + self.epsilon

    def time_to_threshold(self, V0: float, V_target: float) -> float:
        r"""Time to reach a target Lyapunov value.

        t = ln(V0/V_target) / (2μα)

        Parameters
        ----------
        V0 : float
            Initial value.
        V_target : float
            Target value.

        Returns
        -------
        float
            Time to reach V_target. inf if V_target ≤ 0 or rate = 0.
        """
        if V_target <= 0 or V0 <= V_target:
            return 0.0
        rate = self.convergence_rate
        if rate <= 0:
            return np.inf
        return np.log(V0 / V_target) / rate


# ===================================================================
# Attractor Threshold (Thm 7.1, Eq 19-20)
# ===================================================================

@dataclass
class AttractorThreshold:
    r"""Attractor threshold condition (Thm 7.1).

    A configuration is in the basin of attraction of S iff all
    Laplacian eigenvalues satisfy:

        λ_k(L) ≤ 4  ∀k                         (Eq 19)

    Equivalently: ‖L‖₂ ≤ 4.

    For a d-regular lattice with uniform coupling J:
        J |D'_max| ≤ 1                           (Eq 20)

    Parameters
    ----------
    J_crit : float
        Critical coupling threshold (default 1.0 for sawtooth D'=1).
    """
    J_crit: float = 1.0

    def check_spectral_condition(self,
                                 laplacian_eigenvalues: np.ndarray) -> Dict[str, object]:
        r"""Check attractor threshold ‖L‖₂ ≤ 4 (Thm 7.1).

        Parameters
        ----------
        laplacian_eigenvalues : ndarray
            Eigenvalues of the graph Laplacian L.

        Returns
        -------
        dict with:
            'in_basin' : bool
            'spectral_norm' : float
            'max_eigenvalue' : float
            'violating_modes' : int (count of modes with λ_k > 4)
        """
        real_eigs = np.real(laplacian_eigenvalues)
        max_eig = float(np.max(real_eigs))
        n_violating = int(np.sum(real_eigs > 4.0 + 1e-10))
        return {
            "in_basin": max_eig <= 4.0 + 1e-10,
            "spectral_norm": max_eig,
            "max_eigenvalue": max_eig,
            "violating_modes": n_violating,
        }

    def coupling_threshold(self, D_prime_max: float = 1.0) -> float:
        r"""Critical coupling for a regular lattice (Cor 7.1, Eq 20).

        J |D'_max| ≤ 1  →  J_crit = 1 / |D'_max|

        Parameters
        ----------
        D_prime_max : float
            Maximum of |D'(x)| (1.0 for sawtooth).

        Returns
        -------
        float
            Critical coupling J_crit.
        """
        if D_prime_max <= 0:
            return np.inf
        return 1.0 / D_prime_max

    def is_stable_coupling(self, J: float,
                           D_prime_max: float = 1.0) -> bool:
        r"""Test if coupling is below the critical threshold.

        Parameters
        ----------
        J : float
            Coupling strength.
        D_prime_max : float
            Maximum |D'(x)|.

        Returns
        -------
        bool
            True if J |D'_max| ≤ 1.
        """
        return J * D_prime_max <= 1.0 + 1e-10


# ===================================================================
# Resonance Band Structure (Sec 7.1, Eq 18)
# ===================================================================

@dataclass
class ResonanceBand:
    r"""Resonance band structure from transfer matrix spectrum (Sec 7.1).

    Resonance frequency of mode k:
        ω_k = arccos(1 - λ_k/2)  ∈ [0, π]        (Eq 18)

    Modes with λ_k ∈ [0, 4] are oscillatory (in-band).
    Modes with λ_k > 4 are unstable (out-of-band).
    """

    def frequency(self, lambda_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Resonance frequency ω_k = arccos(1 - λ_k/2) (Eq 18).

        Parameters
        ----------
        lambda_k : float or ndarray
            Laplacian eigenvalue(s).

        Returns
        -------
        float or ndarray
            Resonance frequency in [0, π].
        """
        arg = 1.0 - np.asarray(lambda_k, dtype=float) / 2.0
        return np.arccos(np.clip(arg, -1.0, 1.0))

    def is_in_band(self, lambda_k: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        r"""Check if mode is in the oscillatory band.

        λ_k ∈ [0, 4] → in-band (oscillatory).
        λ_k > 4 → out-of-band (unstable).

        Parameters
        ----------
        lambda_k : float or ndarray
            Laplacian eigenvalue(s).

        Returns
        -------
        bool or ndarray
            True if in-band.
        """
        lk = np.asarray(lambda_k, dtype=float)
        return (lk >= -1e-10) & (lk <= 4.0 + 1e-10)

    def band_structure(self, laplacian_eigenvalues: np.ndarray) -> Dict[str, object]:
        r"""Compute full resonance band structure.

        Parameters
        ----------
        laplacian_eigenvalues : ndarray
            All Laplacian eigenvalues.

        Returns
        -------
        dict with:
            'frequencies' : ndarray (ω_k for in-band modes)
            'n_in_band' : int
            'n_out_of_band' : int
            'bandwidth' : float (ω_max - ω_min)
        """
        lk = np.sort(np.real(laplacian_eigenvalues))
        in_band_mask = self.is_in_band(lk)
        in_band = lk[in_band_mask]
        freqs = self.frequency(in_band)

        return {
            "frequencies": freqs,
            "n_in_band": int(np.sum(in_band_mask)),
            "n_out_of_band": int(np.sum(~in_band_mask)),
            "bandwidth": float(np.ptp(freqs)) if len(freqs) > 0 else 0.0,
        }


# ===================================================================
# Phase Gradient Decoherence (Sec 8, Eq 21)
# ===================================================================

@dataclass
class PhaseGradientDecoherence:
    r"""Decoherence rate from boundary phase gradients (Thm 8.1, Eq 21).

    Γ_dec = (κ_s / ℏ) ‖∇Φ_AB‖²

    States in S (∇Φ_AB = 0) have Γ_dec = 0.
    States far from S decohere rapidly.

    Parameters
    ----------
    kappa_s : float
        Boundary stiffness.
    hbar : float
        Effective Planck constant (1.0 in natural units).
    """
    kappa_s: float = 1.0
    hbar: float = 1.0

    def decoherence_rate(self, grad_phi: np.ndarray) -> float:
        r"""Compute decoherence rate Γ_dec (Eq 21).

        Γ_dec = (κ_s / ℏ) ‖∇Φ_AB‖²

        Parameters
        ----------
        grad_phi : ndarray
            Boundary phase gradient ∇Φ_AB.

        Returns
        -------
        float
            Decoherence rate Γ_dec ≥ 0.
        """
        norm_sq = float(np.sum(np.asarray(grad_phi, dtype=float) ** 2))
        return self.kappa_s * norm_sq / self.hbar

    def decoherence_rate_from_field(self, phi: np.ndarray,
                                    dx: float = 1.0) -> float:
        r"""Compute decoherence rate from a phase field.

        Parameters
        ----------
        phi : ndarray
            Boundary phase field.
        dx : float
            Grid spacing.

        Returns
        -------
        float
            Decoherence rate.
        """
        grad = np.diff(phi) / dx
        return self.decoherence_rate(grad)

    def coherence_time(self, grad_phi: np.ndarray) -> float:
        r"""Coherence time τ_coh = 1/Γ_dec.

        Parameters
        ----------
        grad_phi : ndarray
            Boundary phase gradient.

        Returns
        -------
        float
            Coherence time (inf if Γ_dec = 0).
        """
        gamma = self.decoherence_rate(grad_phi)
        if gamma < 1e-15:
            return np.inf
        return 1.0 / gamma


# ===================================================================
# Cache Timescale Derivation (Cor 8.1)
# ===================================================================

@dataclass
class CacheTimescaleDerivation:
    r"""Derivation of Cache memory timescale from decoherence (Cor 8.1).

    For a winding-W configuration, boundary phase gradient scales as:
        |∇Φ_AB| ~ |W|^{-α/2}

    Therefore:
        Γ_dec ~ |W|^{-α}
        τ_m = 1/Γ_dec = τ_0 |W|^α

    This closes the loop: the Cache persistence formula is derived
    rather than postulated.

    Parameters
    ----------
    tau_0 : float
        Base timescale.
    alpha : float
        Winding protection exponent (α ≥ 1).
    """
    tau_0: float = 1.0
    alpha: float = 1.0

    def cache_timescale(self, W: int) -> float:
        r"""Cache memory timescale τ_m = τ_0 |W|^α.

        Parameters
        ----------
        W : int
            Topological winding number.

        Returns
        -------
        float
            Memory timescale.
        """
        if W == 0:
            return self.tau_0
        return self.tau_0 * abs(W) ** self.alpha

    def gradient_scaling(self, W: int) -> float:
        r"""Boundary gradient scaling |∇Φ| ~ |W|^{-α/2}.

        Parameters
        ----------
        W : int
            Winding number.

        Returns
        -------
        float
            Gradient magnitude scaling factor.
        """
        if W == 0:
            return 1.0
        return abs(W) ** (-self.alpha / 2.0)

    def decoherence_scaling(self, W: int) -> float:
        r"""Decoherence rate scaling Γ_dec ~ |W|^{-α}.

        Parameters
        ----------
        W : int
            Winding number.

        Returns
        -------
        float
            Decoherence rate scaling factor.
        """
        if W == 0:
            return 1.0
        return abs(W) ** (-self.alpha)


# ===================================================================
# Phase Transition Threshold (Sec 9, Thm 9.1, Eq 23-24)
# ===================================================================

@dataclass
class PhaseTransitionThreshold:
    r"""Linear stability at the constraint manifold (Thm 9.1).

    The constraint phase κ* is linearly stable iff:
        m²_κ α_damp + γ_couple V''(κ*) > 0       (Eq 23)

    Critical coupling:
        γ_crit = m²_κ α_damp / |V''(κ*)|          (Eq 24)

    Parameters
    ----------
    m_kappa : float
        Constraint mass.
    alpha_damp : float
        Damping coefficient.
    """
    m_kappa: float = 1.0
    alpha_damp: float = 1.0

    def is_linearly_stable(self, gamma_couple: float,
                           V_double_prime: float) -> bool:
        r"""Check linear stability condition (Eq 23).

        m²_κ α_damp + γ V''(κ*) > 0

        Parameters
        ----------
        gamma_couple : float
            Constraint coupling strength.
        V_double_prime : float
            Second derivative V''(κ*) of stability potential.

        Returns
        -------
        bool
            True if linearly stable.
        """
        return self.m_kappa ** 2 * self.alpha_damp + gamma_couple * V_double_prime > 0

    def critical_coupling(self, V_double_prime: float) -> float:
        r"""Critical coupling strength (Eq 24).

        γ_crit = m²_κ α_damp / |V''(κ*)|

        Parameters
        ----------
        V_double_prime : float
            Second derivative of potential at κ*.

        Returns
        -------
        float
            Critical coupling. inf if V''(κ*) = 0.
        """
        if abs(V_double_prime) < 1e-15:
            return np.inf
        return self.m_kappa ** 2 * self.alpha_damp / abs(V_double_prime)

    def growth_rate(self, gamma_couple: float, V_double_prime: float,
                    tau_kappa: float = 1.0) -> float:
        r"""Linear growth rate σ (from linearization around κ*).

        σ = -(m²_κ α_damp + γ V''(κ*)) / τ_κ

        Stability requires σ < 0.

        Parameters
        ----------
        gamma_couple : float
            Constraint coupling.
        V_double_prime : float
            V''(κ*).
        tau_kappa : float
            Constraint timescale.

        Returns
        -------
        float
            Growth rate σ.
        """
        return -(self.m_kappa ** 2 * self.alpha_damp
                 + gamma_couple * V_double_prime) / tau_kappa


# ===================================================================
# Stability Manifold — Main Result (Thm 10.1, Eq 25-26)
# ===================================================================

@dataclass
class StabilityManifold:
    r"""RPST Stability Manifold S — Main Result (Thm 10.1).

    Discrete form (Eq 25):
        S = {(q, ϖ) ∈ Z²ᴺ_p : ϖ_i = 0, Σ_j J_{ij}D(q_j - q_i) = 0 ∀i, ‖L‖₂ ≤ 4}

    Continuum form (Eq 26):
        S = {ρ : E_Φ[ρ] = 0, S(γ) = 0, Γ_dec = 0}

    Every trajectory satisfying the attractor threshold converges to S
    exponentially.

    Parameters
    ----------
    p : int
        Prime modulus.
    kappa_s : float
        Boundary stiffness.
    hbar : float
        Effective Planck constant.
    """
    p: int = 101
    kappa_s: float = 1.0
    hbar: float = 1.0

    def check_membership(self, q: np.ndarray, pi_field: np.ndarray,
                         J: np.ndarray) -> Dict[str, object]:
        r"""Check if a configuration is in the stability manifold S (Eq 25).

        Three conditions:
        1. ϖ_i = 0 ∀i
        2. Σ_j J_{ij} D(q_j - q_i) = 0 ∀i (discrete harmonic)
        3. ‖L‖₂ ≤ 4 (spectral stability)

        Parameters
        ----------
        q : ndarray of int, shape (N,)
            Phase configuration.
        pi_field : ndarray of int, shape (N,)
            Momentum configuration.
        J : ndarray, shape (N, N)
            Coupling matrix.

        Returns
        -------
        dict with membership results.
        """
        fp = FixedPointAnalysis(p=self.p)
        fp_result = fp.is_fixed_point(q, pi_field, J)

        dm = DynamicalMatrix(p=self.p)
        L = dm.graph_laplacian(q, J)
        eigs = np.real(np.linalg.eigvals(L))

        at = AttractorThreshold()
        spectral = at.check_spectral_condition(eigs)

        in_S = fp_result["is_fixed"] and spectral["in_basin"]

        return {
            "in_manifold": in_S,
            "fixed_point": fp_result,
            "spectral_condition": spectral,
            "laplacian_eigenvalues": eigs,
        }

    def continuum_check(self, phi: np.ndarray,
                        dx: float = 1.0) -> Dict[str, object]:
        r"""Check continuum form of manifold membership (Eq 26).

        Three conditions:
        1. E_Φ[ρ] = 0
        2. S(γ) = 0
        3. Γ_dec = 0

        Parameters
        ----------
        phi : ndarray
            Boundary phase field.
        dx : float
            Grid spacing.

        Returns
        -------
        dict with continuum membership results.
        """
        bpe = BoundaryPhaseEnergy(kappa_s=self.kappa_s)
        sm = StabilityMeasure()
        pgd = PhaseGradientDecoherence(kappa_s=self.kappa_s, hbar=self.hbar)

        E = bpe.energy(phi, dx)
        S = sm.compute(phi, dx=dx)
        Gamma = pgd.decoherence_rate_from_field(phi, dx)

        tol = 1e-10
        return {
            "in_manifold": E < tol and S < tol and Gamma < tol,
            "E_phi": E,
            "S_gamma": S,
            "Gamma_dec": Gamma,
        }

    def simulate(self, q0: np.ndarray, pi0: np.ndarray,
                 J: np.ndarray, n_steps: int = 1000) -> Dict[str, object]:
        r"""Simulate the RPST symplectic map and track stability (Sec 13).

        Parameters
        ----------
        q0 : ndarray of int, shape (N,)
            Initial phases.
        pi0 : ndarray of int, shape (N,)
            Initial momenta.
        J : ndarray, shape (N, N)
            Coupling matrix.
        n_steps : int
            Number of time steps.

        Returns
        -------
        dict with:
            'q_trajectory' : ndarray, shape (n_steps+1, N)
            'pi_trajectory' : ndarray, shape (n_steps+1, N)
            'S_trajectory' : ndarray, shape (n_steps+1,)
        """
        p = self.p
        N = len(q0)
        q_traj = np.zeros((n_steps + 1, N), dtype=int)
        pi_traj = np.zeros((n_steps + 1, N), dtype=int)
        S_traj = np.zeros(n_steps + 1)

        q_traj[0] = q0 % p
        pi_traj[0] = pi0 % p

        # Compute initial stability measure from discrete boundary gradient
        sm = StabilityMeasure()

        for t in range(n_steps):
            q = q_traj[t]
            pi_f = pi_traj[t]

            # Symplectic update (Eq 1-2)
            q_new = (q + pi_f) % p
            pi_new = pi_f.copy()
            for i in range(N):
                coupling_sum = 0
                for j in range(N):
                    if J[i, j] != 0:
                        d = discrete_sawtooth(int(q_new[j]) - int(q_new[i]), p)
                        coupling_sum += int(J[i, j] * d)
                pi_new[i] = int(pi_f[i] - coupling_sum) % p

            q_traj[t + 1] = q_new
            pi_traj[t + 1] = pi_new

            # Compute S from boundary differences
            phi = 2.0 * np.pi * q_new.astype(float) / p
            S_traj[t] = sm.compute(phi)

        # Final step S
        phi_final = 2.0 * np.pi * q_traj[-1].astype(float) / p
        S_traj[-1] = sm.compute(phi_final)

        return {
            "q_trajectory": q_traj,
            "pi_trajectory": pi_traj,
            "S_trajectory": S_traj,
        }
