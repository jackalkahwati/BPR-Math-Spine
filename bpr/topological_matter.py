"""
Theory XIV: Topological Condensed Matter from BPR
==================================================

Derives quantum Hall effect, topological insulators, anyon statistics,
and quantized conductance from boundary winding numbers.

Key results
-----------
* Integer QHE: σ_H = ν e²/h with ν = boundary winding number
* Fractional QHE: filling fractions from fractional winding ν = W/p
* Topological insulator Z₂ index from winding parity
* Anyon exchange phase θ = π W/p from fractional winding
* Quantized conductance G = n × e²/h from winding channel count

References: Al-Kahwati (2026), BPR-Math-Spine extended theories
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

# Physical constants
_E = 1.602176634e-19        # C  (elementary charge)
_H = 6.62607015e-34         # J·s (Planck constant)
_HBAR = _H / (2.0 * np.pi)
_E2_OVER_H = _E ** 2 / _H  # ≈ 3.874 × 10⁻⁵ S (conductance quantum)


# ---------------------------------------------------------------------------
# §14.1  Integer Quantum Hall Effect from boundary winding
# ---------------------------------------------------------------------------

@dataclass
class QuantumHallEffect:
    """Integer Quantum Hall effect from quantised boundary winding.

    The Hall conductance is:
        σ_H = ν × e²/h

    where ν = winding number of the boundary phase around the edge.
    Quantisation is exact because winding numbers are integers.

    Parameters
    ----------
    nu : int – filling factor (= boundary winding number)
    """
    nu: int = 1

    @property
    def hall_conductance(self) -> float:
        """σ_H = ν e²/h [S]."""
        return self.nu * _E2_OVER_H

    @property
    def hall_resistance(self) -> float:
        """R_H = h / (ν e²) [Ω]."""
        if self.nu == 0:
            return float("inf")
        return _H / (self.nu * _E ** 2)

    @property
    def chern_number(self) -> int:
        """Chern number = ν (topological invariant)."""
        return self.nu

    @property
    def edge_mode_count(self) -> int:
        """Number of chiral edge modes = |ν|."""
        return abs(self.nu)


# ---------------------------------------------------------------------------
# §14.2  Fractional Quantum Hall Effect from fractional winding
# ---------------------------------------------------------------------------

@dataclass
class FractionalQHE:
    """Fractional QHE from boundary winding fractions.

    In ℤ_p arithmetic, fractional winding ν = W/p gives
    filling fractions with denominators dividing p.

    The Laughlin sequence ν = 1/(2m+1) corresponds to
    winding W with 2m+1 | p.

    Parameters
    ----------
    W : int – winding numerator
    p : int – substrate prime (denominator)
    """
    W: int = 1
    p: int = 3

    @property
    def filling_fraction(self) -> float:
        """Filling factor ν = W/p."""
        return self.W / self.p

    @property
    def hall_conductance(self) -> float:
        """σ_H = (W/p) × e²/h [S]."""
        return self.filling_fraction * _E2_OVER_H

    @property
    def quasiparticle_charge(self) -> float:
        """Quasiparticle charge e* = e/p [C]."""
        return _E / self.p

    @property
    def is_laughlin_state(self) -> bool:
        """Check if this is a Laughlin state: ν = 1/(2m+1)."""
        return self.W == 1 and self.p % 2 == 1

    def ground_state_degeneracy(self, genus: int = 0) -> int:
        """Ground state degeneracy on genus-g surface: p^g."""
        return self.p ** genus


# ---------------------------------------------------------------------------
# §14.3  Topological Insulators from boundary Z₂ index
# ---------------------------------------------------------------------------

@dataclass
class TopologicalInsulator:
    """Topological insulator from boundary winding parity.

    The Z₂ topological index is the parity of the boundary
    winding number:

        ν₂ = W mod 2

    ν₂ = 1 (odd winding) → topological insulator (conducting edge)
    ν₂ = 0 (even winding) → trivial insulator

    Parameters
    ----------
    W : int – boundary winding number
    gap_eV : float – bulk gap [eV]
    """
    W: int = 1
    gap_eV: float = 0.3

    @property
    def z2_index(self) -> int:
        """Z₂ topological index: W mod 2."""
        return self.W % 2

    @property
    def is_topological(self) -> bool:
        """True if system is a topological insulator."""
        return self.z2_index == 1

    @property
    def edge_state_velocity(self) -> float:
        """Edge state velocity v_edge = Δ × a / ℏ [m/s].

        Typical: Δ ~ 0.3 eV, a ~ 5 Å → v ~ 2.3 × 10⁵ m/s.
        """
        a = 5e-10  # lattice constant [m]
        delta_J = self.gap_eV * _E  # gap in Joules
        return delta_J * a / _HBAR

    @property
    def protection_gap(self) -> float:
        """Topological protection gap [eV]: gap if topological, 0 if not."""
        if self.is_topological:
            return self.gap_eV
        return 0.0

    @property
    def num_dirac_cones(self) -> int:
        """Number of surface Dirac cones: odd for topological."""
        if self.is_topological:
            return 2 * (self.W // 2) + 1  # always odd
        return 0


# ---------------------------------------------------------------------------
# §14.4  Anyon statistics from fractional winding
# ---------------------------------------------------------------------------

@dataclass
class AnyonStatistics:
    """Anyon exchange statistics from fractional boundary winding.

    In 2D, exchanging two particles with fractional winding gives:
        ψ → exp(iθ) ψ,   θ = π W / p

    For W/p = 0 → bosons (θ = 0)
    For W/p = 1/2 → fermions (θ = π/2 ... wait, fermions have θ = π)

    More precisely:
        θ_exchange = π × ν   where ν = W/p is the filling fraction

    Abelian anyons: all exchanges commute.
    Non-abelian anyons: W > 1 (higher winding), exchanges don't commute.

    Parameters
    ----------
    W : int – winding number
    p : int – substrate prime
    """
    W: int = 1
    p: int = 3

    @property
    def exchange_phase(self) -> float:
        """Exchange phase θ = π × W/p [radians]."""
        return np.pi * self.W / self.p

    @property
    def statistical_parameter(self) -> float:
        """Statistical parameter α = W/p ∈ [0, 1]."""
        return (self.W % self.p) / self.p

    @property
    def particle_type(self) -> str:
        """Classify as boson, fermion, or anyon."""
        alpha = self.statistical_parameter
        if np.isclose(alpha, 0.0, atol=1e-10):
            return "boson"
        elif np.isclose(alpha, 0.5, atol=1e-10):
            return "fermion"
        return "anyon"

    @property
    def is_non_abelian(self) -> bool:
        """Non-abelian if W > 1 (multiple winding sectors mix)."""
        return abs(self.W) > 1

    def braiding_matrix(self, n_anyons: int = 2) -> np.ndarray:
        """Braiding matrix for n anyons (abelian case).

        R = exp(iθ) × I  for abelian anyons.
        """
        theta = self.exchange_phase
        return np.exp(1j * theta) * np.eye(n_anyons, dtype=complex)


# ---------------------------------------------------------------------------
# §14.5  Quantized conductance from winding channels
# ---------------------------------------------------------------------------

@dataclass
class QuantizedConductance:
    """Quantized conductance in quantum point contacts.

    Each boundary winding channel carries one conductance quantum:
        G = n × (2e²/h)  (factor 2 for spin)

    or without spin:
        G = n × (e²/h)

    Parameters
    ----------
    n_channels : int – number of open winding channels
    include_spin : bool – include spin degeneracy (factor 2)
    """
    n_channels: int = 1
    include_spin: bool = True

    @property
    def conductance(self) -> float:
        """Quantized conductance G [S]."""
        factor = 2 if self.include_spin else 1
        return self.n_channels * factor * _E2_OVER_H

    @property
    def resistance(self) -> float:
        """Quantized resistance R = 1/G [Ω]."""
        if self.n_channels == 0:
            return float("inf")
        return 1.0 / self.conductance

    @property
    def conductance_quantum(self) -> float:
        """Conductance quantum G₀ = 2e²/h [S]."""
        return 2.0 * _E2_OVER_H


# ---------------------------------------------------------------------------
# §14.6  Majorana zero modes from boundary topology
# ---------------------------------------------------------------------------

def majorana_zero_modes(W: int, geometry: str = "wire") -> int:
    """Number of Majorana zero modes at boundary.

    A topological superconductor wire with winding W hosts
    |W| Majorana modes at each end.

    Parameters
    ----------
    W : int – boundary winding number
    geometry : str – "wire" (1D) or "vortex" (2D)

    Returns
    -------
    int – number of Majorana zero modes per boundary
    """
    if geometry == "wire":
        return abs(W) if W % 2 == 1 else 0
    elif geometry == "vortex":
        return 1 if W % 2 == 1 else 0
    return 0


def topological_protection_temperature(gap_eV: float) -> float:
    """Temperature below which topological protection holds.

    T_top ≈ Δ / (5 k_B)   (conventional estimate: gap/5).

    Returns float – temperature [K].
    """
    return gap_eV * _E / (5.0 * 1.380649e-23)
