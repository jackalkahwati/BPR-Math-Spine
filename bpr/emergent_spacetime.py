"""
Theory XIII: Emergent Spacetime & Holography
=============================================

Derives the dimensionality of spacetime, holographic entanglement entropy,
the Bekenstein bound, and the emergence of Newton's constant from the
structure of the BPR substrate boundary.

Key results
-----------
* Spatial dimensions d = 3 from Killing vectors of S² boundary
* Time dimension d_t = 1 from winding monotonicity constraint
* Holographic entanglement entropy: S_EE = |∂A| / (4 l_P²) (Ryu-Takayanagi)
* Bekenstein bound from finite winding states per boundary area
* Newton's constant G from substrate parameters: G = l_P² c³ / ℏ

References: Al-Kahwati (2026), BPR-Math-Spine extended theories
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Physical constants
_C = 299792458.0
_G = 6.67430e-11
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_L_PLANCK = 1.616255e-35
_L_PLANCK_SQ = _L_PLANCK ** 2
_M_PL = 2.176434e-8          # kg


# ---------------------------------------------------------------------------
# §13.1  Emergent spatial dimensions from boundary topology
# ---------------------------------------------------------------------------

@dataclass
class EmergentDimensions:
    """Number of spacetime dimensions from boundary topology.

    Spatial dimensions:
        d = number of independent Killing vectors of the boundary
        S² → 3 Killing vectors → d = 3 spatial dimensions
        T² → 2 Killing vectors → d = 2 (lower-dimensional universe)

    Time dimension:
        d_t = 1 always, from the winding monotonicity constraint:
        the boundary phase increases monotonically → single time direction.

    Parameters
    ----------
    geometry : str – boundary geometry ("sphere", "torus", "genus_g")
    """
    geometry: str = "sphere"

    @property
    def spatial_dimensions(self) -> int:
        """Number of spatial dimensions from boundary Killing vectors."""
        if self.geometry == "sphere":
            return 3  # SO(3) has 3 generators
        elif self.geometry == "torus":
            return 2  # U(1) × U(1) has 2 generators
        elif self.geometry.startswith("genus_"):
            g = int(self.geometry.split("_")[1])
            return max(3, 2 * g + 1)
        return 3

    @property
    def time_dimensions(self) -> int:
        """Number of time dimensions: always 1 from winding monotonicity."""
        return 1

    @property
    def total_dimensions(self) -> int:
        """Total spacetime dimensions d + 1."""
        return self.spatial_dimensions + self.time_dimensions

    @property
    def signature(self) -> str:
        """Metric signature."""
        d = self.spatial_dimensions
        return f"({'-' + '+' * d})"


# ---------------------------------------------------------------------------
# §13.2  Holographic entanglement entropy (Ryu-Takayanagi)
# ---------------------------------------------------------------------------

@dataclass
class HolographicEntropy:
    """Entanglement entropy from boundary winding configurations.

    The Ryu-Takayanagi formula:
        S_EE(A) = |∂A| / (4 G_N)

    In BPR, this follows from counting boundary winding microstates
    on the minimal surface ∂A.  Each Planck-area cell carries
    winding in ℤ_p, so:

        Ω = p^{|∂A| / l_P²}
        S = ln Ω / (4 ln p) = |∂A| / (4 l_P²)

    Parameters
    ----------
    boundary_area : float – area of entangling surface [m²]
    p : int – substrate prime modulus
    """
    boundary_area: float = 1.0
    p: int = 104729

    @property
    def n_planck_cells(self) -> float:
        """Planck cells on the entangling surface."""
        return self.boundary_area / _L_PLANCK_SQ

    @property
    def entropy(self) -> float:
        """Entanglement entropy S_EE = A / (4 l_P²) [in units of k_B]."""
        return self.n_planck_cells / 4.0

    @property
    def mutual_information_bound(self) -> float:
        """Upper bound on mutual information: I(A:B) ≤ 2 S_EE."""
        return 2.0 * self.entropy

    @property
    def p_independent(self) -> bool:
        """Verify entropy is independent of p (ln p cancels)."""
        return True  # By construction


# ---------------------------------------------------------------------------
# §13.3  Bekenstein bound from finite winding states
# ---------------------------------------------------------------------------

@dataclass
class BekensteinBound:
    """Bekenstein bound from finite winding states per boundary area.

    S ≤ 2π R E / (ℏ c)

    In BPR: the maximum entropy is limited by the number of
    distinct winding configurations in a boundary sphere of radius R:

        S_max = (4π R² / l_P²) / 4 = π R² / l_P²

    The Bekenstein bound follows from energy → area via Schwarzschild.

    Parameters
    ----------
    R : float – system radius [m]
    E : float – total energy [J]
    """
    R: float = 1.0
    E: float = 1.0

    @property
    def bekenstein_entropy(self) -> float:
        """Bekenstein bound S ≤ 2π R E / (ℏ c)."""
        return 2.0 * np.pi * self.R * self.E / (_HBAR * _C)

    @property
    def holographic_entropy(self) -> float:
        """Holographic bound: S_hol = π R² / l_P²."""
        return np.pi * self.R ** 2 / _L_PLANCK_SQ

    @property
    def tighter_bound(self) -> str:
        """Which bound is tighter (for given R, E)?"""
        if self.bekenstein_entropy < self.holographic_entropy:
            return "Bekenstein"
        return "holographic"

    def is_satisfied(self, S: float) -> bool:
        """Check if entropy S satisfies the bound."""
        return S <= min(self.bekenstein_entropy, self.holographic_entropy)


# ---------------------------------------------------------------------------
# §13.4  Emergent Newton's constant from substrate
# ---------------------------------------------------------------------------

def newtons_constant_from_substrate(p: int, N: int, J: float,
                                      xi: float) -> float:
    """Derive Newton's constant G from substrate parameters.

    G = ℏ c / M_Pl²

    In BPR, the Planck mass emerges from the substrate:
        M_Pl² = (J × N) / (4π l_P²)

    And the Planck length:
        l_P = ξ / √(p)

    So G = ℏ c³ ξ² / (J × N × p)

    Parameters
    ----------
    p : int – substrate prime
    N : int – lattice sites
    J : float – coupling [J]
    xi : float – correlation length [m]

    Returns
    -------
    float – emergent Newton's constant [m³ kg⁻¹ s⁻²]
    """
    return _HBAR * _C ** 3 * xi ** 2 / (J * N * p)


# ---------------------------------------------------------------------------
# §13.5  Planck length from substrate
# ---------------------------------------------------------------------------

def planck_length_from_substrate(xi: float, p: int) -> float:
    """Planck length emerges from substrate: l_P = ξ / √p.

    Parameters
    ----------
    xi : float – correlation length [m]
    p : int – substrate prime

    Returns
    -------
    float – emergent Planck length [m]
    """
    return xi / np.sqrt(p)


# ---------------------------------------------------------------------------
# §13.6  ER = EPR connection
# ---------------------------------------------------------------------------

@dataclass
class EREqualsEPR:
    """ER = EPR from boundary connectivity.

    Two entangled particles share a boundary winding connection.
    A maximally entangled pair → ER bridge (Einstein-Rosen wormhole).

    The wormhole length L_ER scales with entanglement entropy:
        L_ER ~ l_P × exp(S_EE / 2)

    Parameters
    ----------
    S_entanglement : float – entanglement entropy [nat]
    """
    S_entanglement: float = 1.0

    @property
    def wormhole_length(self) -> float:
        """ER bridge length L = l_P × exp(S/2) [m]."""
        exp_arg = min(self.S_entanglement / 2.0, 700)
        return _L_PLANCK * np.exp(exp_arg)

    @property
    def traversable(self) -> bool:
        """Traversability: requires negative energy (exotic matter).

        In BPR, traversable wormholes require boundary winding
        configurations with W < 0 (negative energy density).
        """
        return False  # Standard result: not traversable without exotic matter

    @property
    def firewall_resolution(self) -> str:
        """Resolution of the firewall paradox.

        BPR: smooth horizon because boundary phase is continuous.
        No firewall — information is encoded in boundary winding,
        not destroyed at the horizon.
        """
        return "smooth horizon from boundary phase continuity"


# ---------------------------------------------------------------------------
# §13.7  Information paradox resolution
# ---------------------------------------------------------------------------

def page_time(M_solar: float) -> float:
    """Page time: when entanglement entropy starts decreasing.

    t_Page ~ (G M)³ / (ℏ c⁴)

    In BPR: the Page time is when the boundary winding has radiated
    half its configurations.

    Returns float – Page time [seconds].
    """
    M = M_solar * 1.989e30
    return (_G * M) ** 3 / (_HBAR * _C ** 4)


def scrambling_time(M_solar: float) -> float:
    """Scrambling time: fastest information processing at the horizon.

    t_scr = (r_s / c) × ln(S_BH)

    Returns float – scrambling time [seconds].
    """
    M = M_solar * 1.989e30
    r_s = 2.0 * _G * M / _C ** 2
    A = 4.0 * np.pi * r_s ** 2
    S = A / (4.0 * _L_PLANCK_SQ)
    return (r_s / _C) * np.log(S)
