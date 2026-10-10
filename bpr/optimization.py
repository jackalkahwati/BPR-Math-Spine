"""
Max-Cut phase-oscillator heuristic (no approximation guarantee)
================================================================

Phase-oscillator dynamics for Max-Cut: a continuous relaxation followed
by rounding.

Key equations
-------------
    energy:          E = +(beta/2) sum_{i,j} W_ij cos(phi_i - phi_j)
                         - lambda sum_i cos(2 phi_i)
                     (MaxCutBPR.energy() returns the lambda = 0 part)
    phase dynamics:  dphi_i/dt = -dE/dphi_i
                               = +beta * sum_j W_ij sin(phi_i - phi_j)
                                 - 2 lambda sin(2 phi_i)
    continuation:    lambda(t) = lambda_max * (t/T)^2
    partition:       x_i = 1 if |phi_i| < pi/2 else 0
    cut value:       C = sum_{(i,j) in cut} w_ij

The coupling is antiferromagnetic: minimizing sum W cos(phi_i - phi_j)
maximizes sum W (1 - cos(phi_i - phi_j))/2, the rank-2 relaxation of
Max-Cut (Burer, Monteiro and Zhang, SIAM J. Optim. 12 (2002) 503).

STATUS: heuristic (rank-2 relaxation + rounding); no approximation
guarantee; not exact for n <= 14 in general.
- Results are empirical and average-case only, against named baselines:
  random cut |E|/2, 1-flip local search, and an uncertified Burer-Monteiro
  proxy for Goemans-Williamson rounding (which beats this heuristic by
  about 9% on random 3-regular graphs with n = 500).
  See tests/test_optimization_maxcut.py and scripts/benchmark_maxcut.py.
  Any "% improvement" is stated only against a named baseline.
- The only worst-case guarantee in this module is that of
  local_search_polish(): a cut of at least half the total weight.
- Until 2026-10 the coupling had the opposite (ferromagnetic) sign, so the
  solver minimized the cut and usually returned 0. The earlier claims
  "exact optima on n <= 14, ~5% improvement on n = 500" are withdrawn;
  see doc/CLOSED_AND_DEPRECATED.md.
- Worst-case context only: approximating Max-Cut within any fixed
  alpha > alpha_GW = 0.878567... on simple unweighted graphs is claimed
  NP-hard in an unrefereed 2026 OpenAI preprint whose main statement is
  Lean-formalized (OAI.OptimalMaxCut.main, via the Unique Games route; not
  re-checked by BPR; index entry omath:102:optimal-maxcut in
  doc/external/openai_math_index.json). Classically: 16/17 is unconditional
  (Hastad 2001; Trevisan, Sorkin, Sudan and Williamson 2000), and alpha_GW
  is optimal under the Unique Games Conjecture (Khot, Kindler, Mossel and
  O'Donnell 2007). This does not bear on the fixed-n or G(n, 1/2) behaviour
  reported here.

References: Goemans and Williamson, J. ACM 42 (1995) 1115;
Burer, Monteiro and Zhang (2002), above.
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
    """Phase-oscillator heuristic for Max-Cut (no approximation guarantee).

    The adjacency matrix W encodes the graph.  Each vertex i carries a
    continuous phase phi_i in [-pi, pi].  The dynamics

        dphi_i/dt = +beta sum_j W_ij sin(phi_i - phi_j) - 2 lam sin(2 phi_i)

    are gradient descent on the antiferromagnetic energy (see energy()) and
    relax toward a local minimum of it.  A continuation schedule ramps lam
    from 0 to lambda_max, binarising the phases.  Local minima need not be
    maximum cuts; polish with local_search_polish() and compare against
    brute_force_max_cut() for small n.

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

        dphi_i/dt = +beta * sum_j W_ij sin(phi_i - phi_j) - 2 lam sin(2 phi_i)
        """
        # Pairwise phase differences: diff[i, j] = phi_i - phi_j
        diff = phi[:, None] - phi[None, :]           # (n, n)
        coupling = self.beta * np.sum(
            self.adjacency_matrix * np.sin(diff), axis=1
        )
        binarisation = -2.0 * lam * np.sin(2.0 * phi)
        return coupling + binarisation

    # ----- energy -----

    def energy(self, phi: np.ndarray) -> float:
        """Continuous energy functional.

        E = +(beta/2) sum_{i,j} W_ij cos(phi_i - phi_j)

        On binary phases (0 or pi), E = (beta/2) * (sum_ij W_ij - 4 C), with
        C the cut value, so the binary ground state is a maximum cut.  The
        dynamics only reach local minima, so the solver does not find it in
        general.
        """
        diff = phi[:, None] - phi[None, :]
        return 0.5 * self.beta * np.sum(
            self.adjacency_matrix * np.cos(diff)
        )

    # ----- integration -----

    def solve(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        continuation: bool = True,
        phi0: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
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
        seed : int, optional
            Seed for the random initial phases (ignored if phi0 is given).

        Returns
        -------
        phi : ndarray (n,)
            Final phase configuration.
        """
        T = n_steps * dt
        if phi0 is not None:
            phi = phi0.copy()
        else:
            phi = np.random.default_rng(seed).uniform(-np.pi, np.pi, self.n)

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


# ---------------------------------------------------------------------------
# Baselines and the classical constant
# ---------------------------------------------------------------------------

def alpha_gw() -> float:
    """Goemans-Williamson constant (Goemans and Williamson 1995).

    alpha_GW = min over -1 <= rho < 1 of 2 arccos(rho) / (pi (1 - rho))
             = 0.8785672...,  attained at rho* ~ -0.68916.

    It is the worst-case expected ratio of SDP + random-hyperplane rounding.
    Nothing in this module attains it; it is a reference value only.
    """
    from scipy.optimize import minimize_scalar

    # In terms of theta = arccos(rho): (2/pi) * theta / (1 - cos theta).
    res = minimize_scalar(
        lambda th: 2.0 * th / (np.pi * (1.0 - np.cos(th))),
        bounds=(2.0, 2.6), method="bounded", options={"xatol": 1e-12},
    )
    return float(res.fun)


def brute_force_max_cut(W: np.ndarray, max_n: int = 20) -> tuple[float, np.ndarray]:
    """Exact Max-Cut by exhaustive search over 2^(n-1) partitions.

    Parameters
    ----------
    W : ndarray (n, n)
        Symmetric weight matrix with zero diagonal.
    max_n : int
        Refuse larger graphs (the cost is O(2^n n^2)).

    Returns
    -------
    (value, partition) : the maximum cut and one 0/1 partition attaining it.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if n > max_n:
        raise ValueError(f"brute force refused for n = {n} > {max_n}")
    if n <= 1:
        return 0.0, np.zeros(n, dtype=int)
    total = W.sum()
    best_value, best_code = -np.inf, 0
    bits = np.arange(n - 1)
    chunk = 1 << 14
    # Vertex n-1 is fixed on side 0, which removes the global flip symmetry.
    for start in range(0, 1 << (n - 1), chunk):
        codes = np.arange(start, min(start + chunk, 1 << (n - 1)))
        x = (codes[:, None] >> bits) & 1
        s = np.hstack([1 - 2 * x, np.ones((len(codes), 1), dtype=int)]).astype(float)
        values = 0.25 * (total - np.einsum("ki,ij,kj->k", s, W, s))
        k = int(np.argmax(values))
        if values[k] > best_value:
            best_value, best_code = float(values[k]), int(codes[k])
    partition = np.append((best_code >> bits) & 1, 0).astype(int)
    return best_value, partition


def local_search_polish(W: np.ndarray, partition: np.ndarray) -> np.ndarray:
    """1-flip local search: move single vertices while the cut increases.

    For non-negative weights the result is a cut of at least half the total
    edge weight, because at a 1-flip local optimum each vertex has at least
    half of its incident weight crossing the cut.  This is the only
    worst-case guarantee available in this module.
    """
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)                # self-loops never cross a cut
    x = np.asarray(partition, dtype=int).copy()
    s = 1.0 - 2.0 * x                       # +1 on side 0, -1 on side 1
    # Each accepted flip raises the cut strictly, so the loop terminates; the
    # cap only guards against round-off cycling.
    for _ in range(max(1, len(s)) ** 2 * 10):
        # gain[i] = (weight to i's own side) - (weight across), i.e. the cut
        # increase from flipping i.
        gain = s * (W @ s)
        i = int(np.argmax(gain))
        if gain[i] <= 1e-12:
            break
        s[i] = -s[i]
    return ((1 - s) / 2).astype(int)
