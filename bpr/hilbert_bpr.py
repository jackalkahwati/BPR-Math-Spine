"""
Hilbert Space Formulation of Boundary Phase Resonance
=====================================================

Formal operator theory for BPR: block boundary phase operator T,
contraction mapping convergence, spectral analysis.

Key structures
--------------
    H_BPR = H_c (+) H_d   (coherent + damped subspaces)
    T = [[A, B], [C, D]]   block operator on H_c (+) H_d
    Convergence: rho(T) < 1
    Energy dissipation: E(Tx) <= alpha * E(x)

References: Al-Kahwati (2026), Hilbert Space Formulation of BPR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Block Boundary Phase Operator
# ---------------------------------------------------------------------------

@dataclass
class BoundaryPhaseOperator:
    """Block operator T = [[A, B], [C, D]] acting on H_c (+) H_d.

    The total Hilbert space dimension is dim(H_c) + dim(H_d).
    A maps H_c -> H_c, B maps H_d -> H_c, etc.

    Parameters
    ----------
    A : ndarray (n_c, n_c)
        Coherent-to-coherent block.
    B : ndarray (n_c, n_d)
        Damped-to-coherent coupling.
    C : ndarray (n_d, n_c)
        Coherent-to-damped coupling.
    D : ndarray (n_d, n_d)
        Damped-to-damped block.
    """
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.C = np.asarray(self.C, dtype=float)
        self.D = np.asarray(self.D, dtype=float)

    @property
    def n_c(self) -> int:
        """Dimension of coherent subspace H_c."""
        return self.A.shape[0]

    @property
    def n_d(self) -> int:
        """Dimension of damped subspace H_d."""
        return self.D.shape[0]

    @property
    def dim(self) -> int:
        """Total Hilbert space dimension."""
        return self.n_c + self.n_d

    def full_matrix(self) -> np.ndarray:
        """Return full (n_c+n_d) x (n_c+n_d) matrix representation of T."""
        top = np.hstack([self.A, self.B])
        bot = np.hstack([self.C, self.D])
        return np.vstack([top, bot])

    def spectral_radius(self) -> float:
        """Compute spectral radius rho(T) = max |eigenvalue|."""
        eigvals = np.linalg.eigvals(self.full_matrix())
        return float(np.max(np.abs(eigvals)))

    def is_contractive(self) -> bool:
        """Check the contraction mapping condition rho(T) < 1."""
        return self.spectral_radius() < 1.0

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply T to a state vector x in H_c (+) H_d.

        Parameters
        ----------
        state : ndarray of length n_c + n_d

        Returns
        -------
        T(state) : ndarray of same length
        """
        state = np.asarray(state, dtype=float)
        return self.full_matrix() @ state

    def iterate(self, state: np.ndarray, n_steps: int) -> np.ndarray:
        """Iterate T^n on state, returning trajectory.

        Returns
        -------
        trajectory : ndarray of shape (n_steps + 1, dim)
        """
        state = np.asarray(state, dtype=float).copy()
        traj = np.empty((n_steps + 1, self.dim))
        traj[0] = state
        T = self.full_matrix()
        for k in range(n_steps):
            state = T @ state
            traj[k + 1] = state
        return traj

    @staticmethod
    def energy(state: np.ndarray) -> float:
        """Hilbert space energy E(x) = ||x||^2."""
        return float(np.dot(state, state))

    def energy_dissipation_rate(self, state: np.ndarray) -> float:
        """Compute E(Tx) / E(x).  Values < 1 indicate dissipation.

        Returns inf if E(x) = 0.
        """
        e_x = self.energy(state)
        if e_x < 1e-30:
            return np.inf
        e_tx = self.energy(self.apply(state))
        return e_tx / e_x

    def attractor_subspace(self, tol: float = 1e-8, max_iter: int = 5000) -> np.ndarray:
        """Find the attractor of T by iterating a random state until convergence.

        Returns
        -------
        attractor : ndarray of length dim
            Fixed point (or limit) satisfying ||T(x) - x|| < tol.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self.dim)
        x /= np.linalg.norm(x)
        T = self.full_matrix()

        for _ in range(max_iter):
            x_new = T @ x
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            # Normalise to prevent collapse to zero for contractive T
            norm = np.linalg.norm(x_new)
            if norm < 1e-30:
                return x_new
            x = x_new / norm

        return x


# ---------------------------------------------------------------------------
# Parameterised BPR family T(lambda)
# ---------------------------------------------------------------------------

def ParameterizedBPR(
    lambda_range: np.ndarray,
    base_A: np.ndarray,
    base_B: np.ndarray,
    base_C: np.ndarray,
    base_D: np.ndarray,
) -> List[Tuple[float, float]]:
    """Sweep coupling parameter lambda and track spectral radius.

    Constructs T(lam) = [[lam*A, B], [C, lam*D]] for each lam in lambda_range.

    Parameters
    ----------
    lambda_range : 1-D array of coupling values
    base_A, base_B, base_C, base_D : base block matrices

    Returns
    -------
    results : list of (lambda, rho(T)) tuples
    """
    results = []
    for lam in np.asarray(lambda_range, dtype=float):
        op = BoundaryPhaseOperator(
            A=lam * base_A,
            B=base_B,
            C=base_C,
            D=lam * base_D,
        )
        results.append((float(lam), op.spectral_radius()))
    return results


# ---------------------------------------------------------------------------
# Hilbert space decomposition
# ---------------------------------------------------------------------------

def decompose_hilbert_space(
    T: BoundaryPhaseOperator,
    tol: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose H into resonant (H_res) and damped (H_damp) subspaces.

    Uses the eigenvectors of T.  Modes with |eigenvalue| > tol are
    classified as resonant; the rest are damped.

    Parameters
    ----------
    T : BoundaryPhaseOperator
    tol : float
        Threshold on |eigenvalue| separating resonant from damped.

    Returns
    -------
    H_res : ndarray, shape (dim, n_res), columns = resonant eigenvectors
    H_damp : ndarray, shape (dim, n_damp), columns = damped eigenvectors
    """
    M = T.full_matrix()
    eigvals, eigvecs = np.linalg.eig(M)

    res_mask = np.abs(eigvals) > tol
    # Use real parts of eigenvectors for real operator
    H_res = np.real(eigvecs[:, res_mask])
    H_damp = np.real(eigvecs[:, ~res_mask])

    # Handle empty subspaces
    if H_res.size == 0:
        H_res = np.empty((T.dim, 0))
    if H_damp.size == 0:
        H_damp = np.empty((T.dim, 0))

    return H_res, H_damp
