"""
Conscious Agents as Resonant Nodes in BPR
==========================================

Maps Hoffman's Conscious Agent calculus to BPR boundary couplings.
Derives Markov transition kernels from boundary overlaps and
emergent spacetime geometry from diffusion coordinates.

Key equations
-------------
    boundary coupling: kappa_ij = <phi_i, B_ij phi_j>
    Markov kernel:     T_ij = |kappa_ij|^2 / sum_k |kappa_ik|^2
    diffusion coords:  Phi(i) = (lambda_1^t psi_1(i), lambda_2^t psi_2(i), ...)

The Born rule emerges naturally: probabilities are |kappa|^2-weighted
boundary overlaps.

References: Al-Kahwati (2026), Conscious Agents as Resonant Nodes
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Conscious Agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConsciousAgent:
    """A conscious agent as a resonant BPR node.

    Each agent carries a state in a finite-dimensional Hilbert space,
    plus perception and action fields that mediate boundary couplings.

    Parameters
    ----------
    state_space_dim : int
        Dimension of the agent's internal state space.
    perception_field : ndarray (state_space_dim,), complex
        The agent's perception boundary field phi_i.
    action_field : ndarray (state_space_dim,), complex
        The agent's action boundary field (outgoing coupling).
    """

    state_space_dim: int = 4
    perception_field: Optional[np.ndarray] = None
    action_field: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng()
        if self.perception_field is None:
            # Random unit vector in C^d
            v = rng.standard_normal(self.state_space_dim) + \
                1j * rng.standard_normal(self.state_space_dim)
            self.perception_field = v / np.linalg.norm(v)
        if self.action_field is None:
            v = rng.standard_normal(self.state_space_dim) + \
                1j * rng.standard_normal(self.state_space_dim)
            self.action_field = v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Boundary coupling
# ---------------------------------------------------------------------------

def boundary_coupling(
    phi_i: np.ndarray,
    B_ij: np.ndarray,
    phi_j: np.ndarray,
) -> complex:
    """Boundary coupling between two agents.

    kappa_ij = <phi_i, B_ij phi_j>

    This is the inner product of agent i's perception field with
    the boundary-transformed action field of agent j.

    Parameters
    ----------
    phi_i : ndarray (d,), complex
        Agent i's perception field.
    B_ij : ndarray (d, d), complex
        Boundary coupling operator between agents i and j.
    phi_j : ndarray (d,), complex
        Agent j's action field.

    Returns
    -------
    kappa : complex
        Coupling amplitude.
    """
    # <phi_i | B_ij | phi_j> = phi_i^dagger @ B_ij @ phi_j
    return complex(np.conj(phi_i) @ B_ij @ phi_j)


# ---------------------------------------------------------------------------
# Markov transition kernel
# ---------------------------------------------------------------------------

def markov_transition_kernel(couplings: np.ndarray) -> np.ndarray:
    """Derive Markov transition probabilities from boundary couplings.

    T_ij = |kappa_ij|^2 / sum_k |kappa_ik|^2

    The Born rule emerges: transition probability is the squared
    magnitude of the boundary overlap, normalised.

    Parameters
    ----------
    couplings : ndarray (n, n), complex
        Matrix of boundary couplings kappa_ij.

    Returns
    -------
    T : ndarray (n, n), real
        Row-stochastic Markov transition matrix.
    """
    prob = np.abs(couplings) ** 2
    row_sums = prob.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated agents
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return prob / row_sums


# ---------------------------------------------------------------------------
# Diffusion coordinates
# ---------------------------------------------------------------------------

def diffusion_coordinates(
    T: np.ndarray,
    t: float = 1.0,
    n_dims: int = 3,
) -> np.ndarray:
    """Embed agents in diffusion space from the Markov kernel.

    Phi(i) = (lambda_1^t psi_1(i), lambda_2^t psi_2(i), ...)

    The eigenvalues lambda_k of T are raised to power t (diffusion time),
    weighting short-range vs long-range structure.

    Parameters
    ----------
    T : ndarray (n, n)
        Markov transition matrix (row-stochastic).
    t : float
        Diffusion time (larger t smooths fine structure).
    n_dims : int
        Number of embedding dimensions.

    Returns
    -------
    coords : ndarray (n, n_dims)
        Diffusion coordinates for each agent.
    """
    eigenvalues, eigenvectors = np.linalg.eig(T)

    # Sort by magnitude (descending), skip the trivial eigenvalue = 1
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first (stationary) eigenvector
    n_dims = min(n_dims, len(eigenvalues) - 1)
    lam = eigenvalues[1:n_dims + 1]
    psi = eigenvectors[:, 1:n_dims + 1]

    # Diffusion map embedding
    coords = np.real(psi * (np.abs(lam) ** t)[None, :])
    return coords


# ---------------------------------------------------------------------------
# Emergent metric
# ---------------------------------------------------------------------------

def emergent_metric(diffusion_coords: np.ndarray) -> np.ndarray:
    """Pairwise distances in diffusion space as emergent geometry.

    d(i, j) = ||Phi(i) - Phi(j)||_2

    This metric is the emergent spacetime distance between conscious
    agents, derived purely from boundary couplings.

    Parameters
    ----------
    diffusion_coords : ndarray (n, d)
        Diffusion coordinates.

    Returns
    -------
    D : ndarray (n, n)
        Pairwise distance matrix.
    """
    n = diffusion_coords.shape[0]
    diff = diffusion_coords[:, None, :] - diffusion_coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ---------------------------------------------------------------------------
# Multi-agent network builder
# ---------------------------------------------------------------------------

def agent_network(
    n_agents: int,
    coupling_matrix: Optional[np.ndarray] = None,
    state_dim: int = 4,
    seed: Optional[int] = None,
) -> Tuple[list, np.ndarray]:
    """Build a network of conscious agents with boundary couplings.

    Parameters
    ----------
    n_agents : int
        Number of agents.
    coupling_matrix : ndarray (n, n, d, d), complex, optional
        Boundary operators B_ij.  If None, random unitary couplings.
    state_dim : int
        Dimension of each agent's state space.
    seed : int, optional
        Random seed.

    Returns
    -------
    agents : list of ConsciousAgent
        The agent objects.
    kappa : ndarray (n, n), complex
        Matrix of boundary couplings kappa_ij.
    """
    rng = np.random.default_rng(seed)

    # Create agents
    agents = []
    for _ in range(n_agents):
        p = rng.standard_normal(state_dim) + 1j * rng.standard_normal(state_dim)
        p /= np.linalg.norm(p)
        a = rng.standard_normal(state_dim) + 1j * rng.standard_normal(state_dim)
        a /= np.linalg.norm(a)
        agents.append(ConsciousAgent(
            state_space_dim=state_dim,
            perception_field=p,
            action_field=a,
        ))

    # Build coupling operators if not provided
    if coupling_matrix is None:
        coupling_matrix = np.zeros(
            (n_agents, n_agents, state_dim, state_dim), dtype=complex
        )
        for i in range(n_agents):
            for j in range(n_agents):
                # Random unitary via QR decomposition
                H = rng.standard_normal((state_dim, state_dim)) + \
                    1j * rng.standard_normal((state_dim, state_dim))
                Q, _ = np.linalg.qr(H)
                coupling_matrix[i, j] = Q

    # Compute couplings
    kappa = np.zeros((n_agents, n_agents), dtype=complex)
    for i in range(n_agents):
        for j in range(n_agents):
            kappa[i, j] = boundary_coupling(
                agents[i].perception_field,
                coupling_matrix[i, j],
                agents[j].action_field,
            )

    return agents, kappa


# ---------------------------------------------------------------------------
# Born rule from BPR overlaps
# ---------------------------------------------------------------------------

def born_rule_weights(couplings: np.ndarray) -> np.ndarray:
    """Demonstrate the Born rule as |kappa|^2 probability from overlaps.

    For a single agent i, the probability of transitioning to agent j
    is given by the squared modulus of the boundary coupling:

        p(j|i) = |kappa_ij|^2 / sum_k |kappa_ik|^2

    This is identical to the Markov kernel and shows that the Born rule
    is a natural consequence of boundary phase overlaps.

    Parameters
    ----------
    couplings : ndarray (n,), complex
        Row of boundary couplings from agent i to all others.

    Returns
    -------
    probabilities : ndarray (n,), real
        Normalised Born-rule probabilities.
    """
    weights = np.abs(couplings) ** 2
    total = weights.sum()
    if total < 1e-30:
        return np.ones_like(weights) / len(weights)
    return weights / total


# ---------------------------------------------------------------------------
# Interface invariant
# ---------------------------------------------------------------------------

def interface_invariant(T: np.ndarray) -> Tuple[float, int]:
    """Topological invariant of the Markov transition kernel.

    Computes:
    1. The spectral gap: lambda_1 - lambda_2 (mixing rate).
    2. The number of ergodic classes (eigenvalues with |lambda| = 1).

    These are topological properties of the conscious agent network:
    the spectral gap measures how quickly the network reaches consensus,
    and the ergodic class count measures the number of disconnected
    "realities".

    Parameters
    ----------
    T : ndarray (n, n)
        Markov transition matrix.

    Returns
    -------
    spectral_gap : float
        Difference between the two largest eigenvalue magnitudes.
    n_ergodic : int
        Number of eigenvalues with magnitude within 1e-8 of 1.
    """
    eigenvalues = np.linalg.eigvals(T)
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]

    # Spectral gap
    if len(magnitudes) >= 2:
        spectral_gap = float(magnitudes[0] - magnitudes[1])
    else:
        spectral_gap = 0.0

    # Ergodic classes: eigenvalues with |lambda| ~ 1
    n_ergodic = int(np.sum(np.abs(magnitudes - 1.0) < 1e-8))

    return spectral_gap, n_ergodic
