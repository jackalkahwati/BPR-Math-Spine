"""
Theory X: Resonant Collective Dynamics
========================================

Maps collective phenomena (flocking, markets, tipping points, cooperation)
onto BPR substrate properties: Kuramoto phase-locking, impedance matching,
and topological winding alignment.

Key objects
-----------
* ``CollectivePhaseField``    – order parameter |Φ_collective|
* ``KuramotoFlocking``        – Vicsek/Kuramoto model on moving substrate
* ``MarketImpedance``         – price from impedance matching
* ``TippingPoint``            – Class A winding transition in social systems
* ``CooperativeWinding``      – cooperation from winding alignment

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §12
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# §12.2  Collective phase field / order parameter
# ---------------------------------------------------------------------------

@dataclass
class CollectivePhaseField:
    """Collective boundary phase for N coupled agents.

    Φ_collective = (1/N) Σ_i exp(i φ_i)

    |Φ| = 0  → disorder
    |Φ| = 1  → perfect alignment

    arg(Φ) = collective direction of action.
    """
    phases: Optional[np.ndarray] = None   # individual phases φ_i

    @property
    def N(self) -> int:
        if self.phases is None:
            return 0
        return len(self.phases)

    @property
    def order_parameter(self) -> complex:
        """Complex order parameter Φ."""
        if self.phases is None or len(self.phases) == 0:
            return 0.0 + 0.0j
        return np.mean(np.exp(1j * self.phases))

    @property
    def coherence(self) -> float:
        """Group coherence |Φ| ∈ [0, 1]."""
        return float(np.abs(self.order_parameter))

    @property
    def collective_direction(self) -> float:
        """Collective direction arg(Φ) (radians)."""
        return float(np.angle(self.order_parameter))


# ---------------------------------------------------------------------------
# §12.3  Flocking / swarming via Kuramoto on moving substrate
# ---------------------------------------------------------------------------

@dataclass
class KuramotoFlocking:
    """Kuramoto model for flocking with nearest-neighbour coupling.

    dφ_i/dt = ω_i + (K/n_i) Σ_{j ∈ N(i)} sin(φ_j − φ_i) + η_i

    Parameters
    ----------
    N : int
        Number of agents.
    K : float
        Alignment coupling (boundary stiffness).
    noise : float
        Noise amplitude η.
    natural_frequencies : ndarray, optional
        ω_i for each agent.
    """
    N: int = 100
    K: float = 1.0
    noise: float = 0.1
    natural_frequencies: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.natural_frequencies is None:
            self.natural_frequencies = np.random.randn(self.N) * 0.1

    def step(self, phases: np.ndarray, dt: float = 0.01,
             adjacency: Optional[np.ndarray] = None) -> np.ndarray:
        """Single Euler step of the Kuramoto dynamics.

        Parameters
        ----------
        phases : ndarray, shape (N,)
        adjacency : ndarray, shape (N, N), optional
            Adjacency matrix.  Default: all-to-all coupling.
        """
        if adjacency is None:
            adjacency = np.ones((self.N, self.N)) - np.eye(self.N)

        dphi = np.copy(self.natural_frequencies)
        for i in range(self.N):
            neighbours = np.where(adjacency[i] > 0)[0]
            if len(neighbours) > 0:
                coupling = np.mean(np.sin(phases[neighbours] - phases[i]))
                dphi[i] += self.K * coupling
        dphi += self.noise * np.random.randn(self.N) / np.sqrt(dt)
        return phases + dphi * dt

    def simulate(self, n_steps: int = 1000, dt: float = 0.01,
                 adjacency: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Run simulation, return (phases_history, coherence_history)."""
        phases = np.random.uniform(0, 2 * np.pi, self.N)
        history = np.zeros((n_steps, self.N))
        coherence = np.zeros(n_steps)
        for t in range(n_steps):
            history[t] = phases
            cpf = CollectivePhaseField(phases=phases)
            coherence[t] = cpf.coherence
            phases = self.step(phases, dt, adjacency)
        return history, coherence

    @property
    def critical_coupling(self) -> float:
        """Critical coupling K_c for the phase transition to flocking.

        K_c ≈ 2 / (π g(0))  for Lorentzian frequency distribution,
        where g(0) is the density at ω = 0.
        """
        sigma = np.std(self.natural_frequencies)
        if sigma < 1e-12:
            return 0.0
        # Lorentzian approx: g(0) = 1/(π σ)
        return 2.0 * sigma


# ---------------------------------------------------------------------------
# §12.4  Market dynamics as impedance matching
# ---------------------------------------------------------------------------

@dataclass
class MarketImpedance:
    """Financial market as impedance-matching system.

    P_market = Σ_i (Z_i⁻¹ V_i) / Σ_i Z_i⁻¹

    Market crashes = impedance resonance (synchronised oscillations).
    Bubbles = phase-locked states where W_market > W_fundamental.

    Parameters
    ----------
    valuations : ndarray
        Agent valuations V_i.
    impedances : ndarray
        Agent impedances Z_i (inverse ∝ aggressiveness).
    """
    valuations: Optional[np.ndarray] = None
    impedances: Optional[np.ndarray] = None

    @property
    def market_price(self) -> float:
        """Equilibrium price from impedance matching."""
        if self.valuations is None or self.impedances is None:
            raise ValueError("Valuations and impedances must be set.")
        inv_Z = 1.0 / self.impedances
        return float(np.sum(inv_Z * self.valuations) / np.sum(inv_Z))

    def impedance_resonance_index(self) -> float:
        """Measure of synchronisation among agent impedances.

        High values → crash risk (impedance resonance).
        """
        if self.impedances is None:
            return 0.0
        # Coefficient of variation (low → synchronised → crash risk)
        mean_Z = np.mean(self.impedances)
        std_Z = np.std(self.impedances)
        if mean_Z < 1e-12:
            return np.inf
        return 1.0 - std_Z / mean_Z   # 1 = perfect synchronisation

    def is_bubble(self, W_market: float, W_fundamental: float) -> bool:
        """Detect bubble: collective winding exceeds fundamental."""
        return W_market > W_fundamental


# ---------------------------------------------------------------------------
# §12.5  Tipping points as Class A winding transitions
# ---------------------------------------------------------------------------

@dataclass
class TippingPoint:
    """Social tipping point as a first-order winding transition.

    W_group: 0 → W_new  when  f_aligned > f_c (percolation threshold).

    f_c depends on network topology:
        Random network:      f_c ~ 1/<k>
        Scale-free network:  f_c → 0 as N → ∞
    """
    mean_degree: float = 10.0     # <k> of the social network
    N: int = 1000                 # population size

    @property
    def critical_fraction(self) -> float:
        """Percolation threshold f_c ~ 1/<k>."""
        return 1.0 / self.mean_degree

    def has_tipped(self, f_aligned: float) -> bool:
        """Check whether the aligned fraction exceeds the tipping threshold."""
        return f_aligned > self.critical_fraction

    @staticmethod
    def scale_free_critical_fraction(N: int, gamma: float = 2.5) -> float:
        """Critical fraction for a scale-free network.

        f_c → 0 as N → ∞ for γ < 3 (Barabási–Albert regime).
        """
        if gamma >= 3:
            return 1.0 / np.log(N)
        return N ** (-(3 - gamma) / (gamma - 1))


# ---------------------------------------------------------------------------
# §12.6  Cooperation from winding alignment
# ---------------------------------------------------------------------------

@dataclass
class CooperativeWinding:
    """Cooperation emerges when individual windings align.

    Aligned collective winding is topologically protected and
    energetically costly to disrupt (defection).

    The Nash equilibrium corresponds to the minimum-energy aligned
    winding configuration.
    """
    individual_windings: Optional[np.ndarray] = None

    @property
    def collective_winding(self) -> float:
        """Total (vector-sum) winding of the group."""
        if self.individual_windings is None:
            return 0.0
        return float(np.sum(self.individual_windings))

    @property
    def alignment(self) -> float:
        """Winding alignment ∈ [0, 1].  1 = all cooperating."""
        if self.individual_windings is None or len(self.individual_windings) == 0:
            return 0.0
        return float(np.abs(np.mean(np.sign(self.individual_windings))))

    def defection_cost(self, agent_idx: int) -> float:
        """Energy cost of agent *i* flipping winding (defecting).

        Cost ∝ change in collective winding magnitude.
        """
        W = self.individual_windings.copy()
        W_before = abs(np.sum(W))
        W[agent_idx] *= -1
        W_after = abs(np.sum(W))
        return W_before - W_after   # positive = costly to defect

    def is_topologically_protected(self, threshold: float = 0.5) -> bool:
        """Protected if alignment exceeds threshold."""
        return self.alignment > threshold
