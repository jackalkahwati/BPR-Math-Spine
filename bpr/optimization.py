"""
BPR Methods for NP-Hard Optimization
=====================================

Phase oscillator dynamics for combinatorial optimization.
Continuous relaxation of Max-Cut and related problems via
boundary phase resonance.

Key equations
-------------
    phase dynamics:  dphi_i/dt = -beta * sum_j w_ij sin(phi_i - phi_j)
                                 - 2 lambda sin(2 phi_i)
    continuation:    lambda(t) = lambda_max * (t/T)^2
    partition:       x_i = 1 if |phi_i| < pi/2 else 0
    cut value:       C = sum_{(i,j) in cut} w_ij

Predictions: Exact optima on n <= 14, ~5 pct improvement on n = 500.

References: Al-Kahwati (2026), BPR Methods for NP-Hard Optimization
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Continuation schedule
# ---------------------------------------------------------------------------

def continuation_schedule(t: float, T: float, lambda_max: float) -> float:
    """Quadratic continuation schedule for the binarisation penalty.

    lambda(t) = lambda_max * (t / T)^2

    Starts at 0 (free phase relaxation) and ramps to lambda_max,
    gradually forcing phases toward 0 or pi.
    """
    ratio = np.clip(t / T, 0.0, 1.0)
    return lambda_max * ratio ** 2


# ---------------------------------------------------------------------------
# Random graph generator
# ---------------------------------------------------------------------------

def random_graph(n: int, p: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """Erdos-Renyi random graph G(n, p) as a symmetric adjacency matrix.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Edge probability.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    W : ndarray of shape (n, n)
        Symmetric adjacency matrix with 0/1 entries, zero diagonal.
    """
    rng = np.random.default_rng(seed)
    upper = (rng.random((n, n)) < p).astype(float)
    # Zero diagonal and symmetrise
    np.fill_diagonal(upper, 0.0)
    W = np.triu(upper, 1)
    W = W + W.T
    return W


# ---------------------------------------------------------------------------
# Max-Cut BPR solver
# ---------------------------------------------------------------------------

@dataclass
class MaxCutBPR:
    """Phase-oscillator solver for the Max-Cut problem via BPR dynamics.

    The adjacency matrix W encodes the graph.  Each vertex i carries a
    continuous phase phi_i in [-pi, pi].  The dynamics

        dphi_i/dt = -beta sum_j W_ij sin(phi_i - phi_j) - 2 lam sin(2 phi_i)

    relax toward a local minimum of the continuous energy.  A continuation
    schedule ramps lam from 0 to lambda_max, binarising the phases.

    Parameters
    ----------
    adjacency_matrix : ndarray (n, n)
        Symmetric weight matrix of the graph.
    beta : float
        Coupling strength (inverse temperature analogue).
    lambda_max : float
        Terminal binarisation penalty.
    """

    adjacency_matrix: np.ndarray
    beta: float = 1.0
    lambda_max: float = 10.0

    # Populated after solve()
    trajectory: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.n = self.adjacency_matrix.shape[0]

    # ----- dynamics -----

    def phase_dynamics(self, phi: np.ndarray, t: float, lam: float) -> np.ndarray:
        """Right-hand side of the phase oscillator ODE.

        dphi_i/dt = -beta * sum_j W_ij sin(phi_i - phi_j) - 2 lam sin(2 phi_i)
        """
        # Pairwise phase differences: diff[i, j] = phi_i - phi_j
        diff = phi[:, None] - phi[None, :]           # (n, n)
        coupling = -self.beta * np.sum(
            self.adjacency_matrix * np.sin(diff), axis=1
        )
        binarisation = -2.0 * lam * np.sin(2.0 * phi)
        return coupling + binarisation

    # ----- energy -----

    def energy(self, phi: np.ndarray) -> float:
        """Continuous energy functional.

        E = -(beta/2) sum_{i,j} W_ij cos(phi_i - phi_j)

        The ground state of this energy (with binarisation) corresponds
        to the Max-Cut partition.
        """
        diff = phi[:, None] - phi[None, :]
        return -0.5 * self.beta * np.sum(
            self.adjacency_matrix * np.cos(diff)
        )

    # ----- integration -----

    def solve(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        continuation: bool = True,
        phi0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Integrate the phase dynamics with forward Euler.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        dt : float
            Time step size.
        continuation : bool
            If True, ramp lambda from 0 to lambda_max; else hold at lambda_max.
        phi0 : ndarray, optional
            Initial phases.  Random on [-pi, pi] if not provided.

        Returns
        -------
        phi : ndarray (n,)
            Final phase configuration.
        """
        T = n_steps * dt
        if phi0 is not None:
            phi = phi0.copy()
        else:
            phi = np.random.default_rng().uniform(-np.pi, np.pi, self.n)

        trajectory = [phi.copy()]
        for step in range(n_steps):
            t = step * dt
            lam = continuation_schedule(t, T, self.lambda_max) if continuation else self.lambda_max
            dphi = self.phase_dynamics(phi, t, lam)
            phi = phi + dt * dphi
            # Wrap to [-pi, pi]
            phi = (phi + np.pi) % (2 * np.pi) - np.pi
            trajectory.append(phi.copy())

        self.trajectory = np.array(trajectory)
        return phi

    # ----- partition extraction -----

    @staticmethod
    def extract_cut(phi: np.ndarray) -> np.ndarray:
        """Round continuous phases to a binary partition.

        x_i = 1 if |phi_i| < pi/2,  else 0.
        """
        return (np.abs(phi) < np.pi / 2).astype(int)

    def cut_value(self, partition: np.ndarray) -> float:
        """Compute the cut value: sum of weights crossing the partition.

        C = sum_{i<j} W_ij * |x_i - x_j|
        """
        diff = np.abs(partition[:, None] - partition[None, :])
        return 0.5 * np.sum(self.adjacency_matrix * diff)
