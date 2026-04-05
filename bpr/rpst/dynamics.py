"""
RPST Dynamics - Eq 0b scaffolding

Implements a reversible, symplectic-like update rule on Z_p.
This is not meant to be a full physical model; it provides a clean, testable
discrete-time evolution that preserves modular phase-space structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .substrate import PrimeField, SubstrateState


def _is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@dataclass
class SymplecticEvolution:
    """
    Discrete-time reversible update on Z_p:

      pi_{t+1} = pi_t - (J q_t)   (mod p)
      q_{t+1}  = q_t  + pi_{t+1}  (mod p)

    Inverse:
      q_t      = q_{t+1} - pi_{t+1}  (mod p)
      pi_t     = pi_{t+1} + (J q_t)  (mod p)
    """

    p: int
    J: Optional[np.ndarray] = None

    def __post_init__(self):
        self.p = int(self.p)
        if not _is_prime(self.p):
            # keep strict: RPST layer assumes prime modulus
            raise ValueError(f"p must be prime, got {self.p}")
        self.field = PrimeField(self.p)

    def _coupling(self, q: np.ndarray) -> np.ndarray:
        if self.J is None:
            return np.zeros_like(q)
        J = np.asarray(self.J, dtype=int)
        qv = np.asarray(q, dtype=int).reshape(-1)
        if J.shape != (qv.size, qv.size):
            raise ValueError("J must be (N,N) for state size N")
        return (J @ qv).reshape(q.shape)

    def step(self, state: SubstrateState) -> SubstrateState:
        q, pi = state.q, state.pi
        force = self._coupling(q)
        pi_next = self.field.sub(pi, force)
        q_next = self.field.add(q, pi_next)
        return SubstrateState(q=q_next, pi=pi_next, p=self.p, t=state.t + 1)

    def step_inverse(self, state: SubstrateState) -> SubstrateState:
        # state is (q_{t+1}, pi_{t+1})
        q_next, pi_next = state.q, state.pi
        q = self.field.sub(q_next, pi_next)
        force = self._coupling(q)
        pi = self.field.add(pi_next, force)
        return SubstrateState(q=q, pi=pi, p=self.p, t=state.t - 1)

    def evolve(self, state: SubstrateState, steps: int) -> List[SubstrateState]:
        s = state
        traj = [s]
        for _ in range(int(steps)):
            s = self.step(s)
            traj.append(s)
        return traj

    def verify_reversibility(self, state: SubstrateState, steps: int = 50) -> bool:
        forward = self.evolve(state, steps=steps)[-1]
        s = forward
        for _ in range(int(steps)):
            s = self.step_inverse(s)
        return np.array_equal(s.q % self.p, state.q % self.p) and np.array_equal(
            s.pi % self.p, state.pi % self.p
        )


# ═══════════════════════════════════════════════════════════════════════
#  Substrate Initialization Dynamics (SID)
# ═══════════════════════════════════════════════════════════════════════

def boundary_transition_operator(phi, E_b, n_grad_phi, alpha=1.0):
    """B[phi] = phi + alpha*E_b*n*grad_phi
    Boundary impulse kicks the phase field, initiating substrate dynamics."""
    return phi + alpha * E_b * n_grad_phi

def prime_index_assignment(phi_fields, primes):
    """p' = argmin Gamma(p) where Gamma(p) = integral |grad_phi_p|^2 dV
    Assign each substrate region to the prime with minimal gradient energy."""
    gamma_values = []
    for phi_p in phi_fields:
        grad = np.gradient(phi_p)
        if isinstance(grad, list):
            gamma = sum(np.sum(g**2) for g in grad)
        else:
            gamma = np.sum(grad**2)
        gamma_values.append(gamma)
    best_idx = int(np.argmin(gamma_values))
    return primes[best_idx], gamma_values

def attractor_selection(H_func, attractors, p_prime):
    """A_k = argmin H(A, p') -- select attractor minimizing Hamiltonian.
    H_func: callable H(A, p) returning energy.
    attractors: list of candidate attractor states."""
    energies = [H_func(A, p_prime) for A in attractors]
    best_idx = int(np.argmin(energies))
    return attractors[best_idx], energies[best_idx]

def substrate_initialization_sequence(phi_0, E_b, n_grad, primes, H_func, attractors, alpha=1.0):
    """Full SID sequence: boundary impulse -> prime assignment -> attractor selection.
    Returns (initialized_phi, assigned_prime, selected_attractor)."""
    # Step 1: Boundary impulse
    phi_kicked = boundary_transition_operator(phi_0, E_b, n_grad, alpha)
    # Step 2: Prime assignment
    phi_fields = [phi_kicked * np.exp(1j * 2*np.pi/p) for p in primes]  # prime-modulated fields
    p_prime, gammas = prime_index_assignment([np.real(f) for f in phi_fields], primes)
    # Step 3: Attractor selection
    attractor, energy = attractor_selection(H_func, attractors, p_prime)
    return phi_kicked, p_prime, attractor


