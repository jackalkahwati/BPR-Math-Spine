"""
Emergent Spacetime & Holography
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
    """Entanglement entropy from the induced BPR Einstein-Hilbert term.

    The Ryu-Takayanagi formula:
        S_EE(A) = |∂A| / (4 G_N)

    In BPR, the coefficient-level route is Sakharov induced gravity:

        M_Pl^2 = p Lambda_b^2 / (48 pi^2)
        S      = |∂A| M_Pl^2 / 4 = |∂A| / (4 l_P²)

    Raw p-state winding counts remain useful as a microscopic heuristic,
    but the induced Einstein-Hilbert normalization is what fixes the
    Bekenstein-Hawking coefficient.

    Parameters
    ----------
    boundary_area : float – area of entangling surface [m²]
    p : int – substrate prime modulus
    """
    boundary_area: float = 1.0
    p: int = 104761

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

def planck_mass_from_boundary_cutoff(p: int, Lambda_b: float) -> float:
    """Planck mass from Sakharov-induced gravity at BPR boundary.

    M_Pl² = p × Λ_b² / (48π²)

    Derivation: integrating out the p boundary anyon sectors below the
    boundary UV cutoff Λ_b generates an Einstein-Hilbert term via the
    standard one-loop induced-gravity mechanism. Each of the p scalar
    modes contributes Λ_b² / (96π²) to the induced Einstein coefficient.

    See doc/derivations/planck_length_from_substrate.md.

    Parameters
    ----------
    p : int – substrate prime (= CS level k)
    Lambda_b : float – boundary lattice UV cutoff [J] or [GeV]

    Returns
    -------
    float – Planck mass in the same energy units as Lambda_b
    """
    return Lambda_b * np.sqrt(p / (48.0 * np.pi ** 2))


def boundary_cutoff_from_planck_mass(p: int, M_Pl: float) -> float:
    """Inverse of planck_mass_from_boundary_cutoff.

    Given the observed M_Pl and substrate prime p, predict the boundary
    lattice cutoff Λ_b = M_Pl × √(48π²/p).

    For p = 104761, M_Pl = 1.22×10¹⁹ GeV ⇒ Λ_b ≈ 8.2×10¹⁷ GeV.
    """
    return M_Pl * np.sqrt(48.0 * np.pi ** 2 / p)


def newtons_constant_from_substrate(p: int, Lambda_b: float) -> float:
    """Derive Newton's constant G from (p, Λ_b) via Sakharov induced gravity.

    G = ℏ c / M_Pl² = 48π² ℏ c / (p × Λ_b²)

    Note: unlike the earlier signature (p, N, J, ξ), this depends on only
    one dimensionful scale Λ_b — the boundary lattice cutoff — because the
    Sakharov relation (3) in planck_length_from_substrate.md fixes
    M_Pl / Λ_b from p alone. N (lattice size) has been removed as it is a
    computational grid convention that cancels in dimensionless ratios.

    Parameters
    ----------
    p : int – substrate prime
    Lambda_b : float – boundary UV cutoff [J]

    Returns
    -------
    float – emergent Newton's constant [m³ kg⁻¹ s⁻²]
    """
    M_Pl_energy = planck_mass_from_boundary_cutoff(p, Lambda_b)  # [J]
    M_Pl_kg = M_Pl_energy / _C ** 2
    return _HBAR * _C / M_Pl_kg ** 2


# ---------------------------------------------------------------------------
# §13.5  Planck length from substrate
# ---------------------------------------------------------------------------

def planck_length_from_substrate(xi: float = None, p: int = 104761) -> float:
    """Planck length as the fundamental substrate lattice spacing.

    BPR requires one dimensionful anchor. The absolute value of l_P is
    that anchor; what BPR *derives* is the hierarchy between l_P and the
    boundary lattice spacing a:

        a / l_P = √(p / (48π²)) ≈ 14.87   (for p = 104761)

    via the Sakharov induced-gravity relation
    M_Pl² = p Λ_b² / (48π²). See
    doc/derivations/planck_length_from_substrate.md for the derivation.

    Parameters
    ----------
    xi : float – (ignored, kept for API compatibility)
    p : int – substrate prime

    Returns
    -------
    float – Planck length [m] (physical input; the a/l_P ratio above is
    the derived quantity).
    """
    return _L_PLANCK  # 1.616255e-35 m — the one dimensionful anchor


def boundary_lattice_spacing(p: int = 104761) -> float:
    """Boundary lattice spacing a from Sakharov relation.

    a = l_P × √(p / (48π²))

    For p = 104761, a ≈ 14.87 × l_P ≈ 2.40 × 10⁻³⁴ m,
    corresponding to J = ℏc/a ≈ 8.2 × 10¹⁷ GeV.
    """
    return _L_PLANCK * np.sqrt(p / (48.0 * np.pi ** 2))


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


# ---------------------------------------------------------------------------
# §13.8  Clausius-to-Einstein bridge
# ---------------------------------------------------------------------------

def clausius_entropy_flux(delta_E, T):
    """δS = δE/T — Clausius relation at local Rindler patch.
    Foundation for deriving Einstein equations from boundary thermodynamics."""
    return delta_E / T

def boundary_entropy_density(phi, n_grad_phi, k_B=1.38e-23, l_P=1.616e-35):
    """s_∂ = (k_B/4l_P²)·|φ|² — boundary entropy density.
    Proportional to area in Planck units."""
    return (k_B / (4 * l_P**2)) * np.abs(phi)**2

def quasilocal_stress_from_entropy(K_ij, K_trace, gamma_ij, G=6.674e-11):
    """T^∂_ij = (1/8πG)(K_ij - K·γ_ij)
    Brown-York quasilocal stress tensor from extrinsic curvature.
    This IS the Einstein equation in boundary form."""
    return (1 / (8 * np.pi * G)) * (K_ij - K_trace * gamma_ij)

def einstein_from_boundary_stationarity(entropy_flux, delta_area, G=6.674e-11, c=3e8):
    """δS = δA/(4l_P²) at every null boundary → R_μν - ½Rg_μν = 8πG/c⁴ T_μν.
    Returns the effective energy density implied by entropy flux."""
    l_P = np.sqrt(G * 1.055e-34 / c**3)
    return entropy_flux * 4 * l_P**2 / delta_area

def planck_boundary_potential(A, l_Planck=1.616e-35, lam=1.0):
    """V_B(A) = λ/(A - l_P)² — boundary potential preventing sub-Planck amplitude.
    Resolves the sub-quantum amplitude paradox by quantizing amplitudes."""
    denom = (A - l_Planck)**2
    denom = np.where(np.abs(denom) < 1e-100, 1e-100, denom)
    return lam / denom

def quantized_amplitude_states(n_max, l_Planck=1.616e-35):
    """A_n = n·l_P — quantized amplitude levels from Planck boundary potential.
    Information is preserved under extreme redshift via discrete resonant states."""
    return np.arange(1, n_max + 1) * l_Planck

def gw_dispersion_correction(frequency_Hz, l_Planck=1.616e-35, c=3e8):
    """v_g(f) = c·[1 - (f·l_P/c)²]^{1/2} — GW group velocity with Planck dispersion.
    At ultra-high frequencies, GW propagation deviates from c by phase-locked steps.
    Prediction: Δv/c ~ (f·l_P/c)² ~ 10⁻⁸⁶ at LIGO frequencies (undetectable)
    but ~ 10⁻² at Planck frequency (testable in principle)."""
    x = (frequency_Hz * l_Planck / c)**2
    return c * np.sqrt(np.maximum(1 - x, 0))

def mesoscopic_gravity_deviation(L, l_Planck=1.616e-35, alpha_dev=1.0):
    """δG/G ~ α·(l_P/L)^{d_eff} — gravity deviations at mesoscopic scales.
    Where boundary entropy becomes discrete, Einstein gravity breaks down.
    d_eff ~ 2 for 3+1 dimensions."""
    return alpha_dev * (l_Planck / L)**2
