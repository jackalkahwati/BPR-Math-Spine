"""
Theory VIII: Substrate Complexity Theory
==========================================

Maps computational complexity classes onto BPR substrate properties.
P, NP, BQP gain physical interpretations in terms of winding
configurations, topological parallelism, and oracle boundary injections.

Key objects
-----------
* ``SubstrateComplexity``     – complexity class as substrate property
* ``TopologicalParallelism``  – quantum advantage from winding
* ``ComplexityBound``         – topological lower bound (physical P ≠ NP argument)

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §10
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# §10.2  Complexity classes as substrate properties
# ---------------------------------------------------------------------------

class ComplexityClass(Enum):
    """Standard complexity classes with substrate interpretations."""
    P = "polynomial"           # poly(n) symplectic steps
    NP = "nondeterministic"    # poly verification, exp search over windings
    BQP = "quantum_polynomial" # boundary phase superposition of winding sectors


# ---------------------------------------------------------------------------
# §10.3  Quantum advantage from topological parallelism
# ---------------------------------------------------------------------------

@dataclass
class TopologicalParallelism:
    """Number of simultaneously accessible winding sectors.

    N_parallel = p^W

    where p is the substrate modulus and W is the maintained winding number.
    This exponential parallelism is the resource exploited by quantum algorithms.
    Decoherence (Theory III) limits the effective W.

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    W : float
        Maintained winding number.
    """
    p: int = 7
    W: float = 1.0

    @property
    def n_parallel(self) -> float:
        """Number of parallel winding sectors accessible."""
        return float(self.p) ** self.W

    def quantum_advantage(self, n_classical_steps: int) -> float:
        """Speedup factor over classical computation.

        Speedup ~ N_parallel / n_classical_steps.
        """
        return self.n_parallel / n_classical_steps

    def effective_qubits(self) -> float:
        """Effective number of qubits: log₂(N_parallel)."""
        return self.W * np.log2(self.p)


# ---------------------------------------------------------------------------
# §10.4  Oracle separation from boundary access
# ---------------------------------------------------------------------------

@dataclass
class PolynomialHierarchy:
    """Polynomial hierarchy from alternating boundary injections.

    Σ_k^P corresponds to k alternating boundary injections.

    Each level requires an additional round of boundary condition
    specification by the observer.
    """
    k: int = 0    # level in polynomial hierarchy

    @property
    def description(self) -> str:
        if self.k == 0:
            return "P (no boundary injection)"
        return f"Σ_{self.k}^P ({self.k} alternating boundary injection(s))"

    def oracle_queries_required(self) -> int:
        """Number of observer boundary injections for this level."""
        return self.k


# ---------------------------------------------------------------------------
# §10.5  Topological complexity bound
# ---------------------------------------------------------------------------

@dataclass
class TopologicalComplexityBound:
    """Physical argument for P ≠ NP based on substrate topology.

    No substrate computation can solve an NP-complete problem in poly
    depth without either:
        (a) maintaining exponential winding (exponential energy), or
        (b) accessing boundary oracle injections (observer intervention).

    Distinct winding configurations cannot be connected by local substrate
    updates in polynomial time.
    """
    n: int = 10              # problem input size
    p: int = 7               # substrate modulus
    max_poly_degree: int = 3 # polynomial depth limit

    @property
    def poly_depth_limit(self) -> int:
        """Maximum depth available in polynomial time."""
        return self.n ** self.max_poly_degree

    @property
    def winding_sectors_to_search(self) -> float:
        """Number of winding sectors for an NP-complete search: ~ p^n."""
        return float(self.p) ** self.n

    @property
    def requires_exponential_resource(self) -> bool:
        """True if poly depth is insufficient for NP search."""
        return self.winding_sectors_to_search > self.poly_depth_limit

    def minimum_winding_for_solution(self) -> float:
        """Minimum W needed for quantum solution: W = n (exponential parallelism)."""
        return float(self.n)


# ---------------------------------------------------------------------------
# Error correction overhead  (P8.2)
# ---------------------------------------------------------------------------

def error_correction_overhead(W_target: float, p: int = 7,
                               decoherence_rate: float = 0.01) -> float:
    """Physical qubits per logical qubit, from topological complexity.

    Overhead ~ (decoherence_rate × p^W_target) for maintaining W_target
    coherent winding sectors.
    """
    sectors = float(p) ** W_target
    return decoherence_rate * sectors


# ---------------------------------------------------------------------------
# Adiabatic quantum computation gap  (P8.3)
# ---------------------------------------------------------------------------

def adiabatic_gap(n: int, p: int = 7, gap_prefactor: float = 1.0) -> float:
    """Energy gap to first excited winding sector for adiabatic QC.

    Δ ~ gap_prefactor × p^{-n}   (closes exponentially with problem size).
    """
    return gap_prefactor * float(p) ** (-n)
