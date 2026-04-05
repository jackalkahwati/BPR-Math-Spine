"""
Multiscale BPR Lagrangian: Coherence Across All Physical Realms
================================================================

Stacked Lagrangian connecting coherence from sub-Planck to cosmic:
    L_total = sum_n  w_n * L_n[chi_n, Phi_n, zeta_n]

Each scale n has:
    chi_n  -- boundary coherence (0 to 1)
    Phi_n  -- integrated information
    zeta_n -- resonance coupling to adjacent scales

Cross-scale coupling: chi_{n+1} = f(chi_n, zeta_n, Phi_n)

References: Al-Kahwati (2026), BPR Scaling Framework
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Scale hierarchy: sub-Planck to cosmic
# ---------------------------------------------------------------------------

SCALE_HIERARCHY: List[Dict] = [
    {"name": "sub_planck",  "length_m": 1e-35,     "energy_eV": 1.22e28},
    {"name": "planck",      "length_m": 1.616e-35,  "energy_eV": 1.22e28},
    {"name": "nuclear",     "length_m": 1e-15,      "energy_eV": 1e6},
    {"name": "atomic",      "length_m": 1e-10,      "energy_eV": 13.6},
    {"name": "molecular",   "length_m": 1e-9,       "energy_eV": 0.1},
    {"name": "cellular",    "length_m": 1e-5,       "energy_eV": 0.025},
    {"name": "mesoscale",   "length_m": 1e-3,       "energy_eV": 0.025},
    {"name": "organismal",  "length_m": 1.0,        "energy_eV": 0.025},
    {"name": "collective",  "length_m": 1e3,        "energy_eV": 0.025},
    {"name": "planetary",   "length_m": 6.4e6,      "energy_eV": 0.025},
    {"name": "stellar",     "length_m": 7e8,        "energy_eV": 1e3},
    {"name": "galactic",    "length_m": 1e21,       "energy_eV": 1e-4},
    {"name": "cosmic",      "length_m": 8.8e26,     "energy_eV": 2.35e-4},
]

_N_SCALES = len(SCALE_HIERARCHY)

# Critical information threshold for cross-scale coupling
_PHI_CRIT = 1.0


# ===========================================================================
#  Multiscale Lagrangian
# ===========================================================================

@dataclass
class MultiscaleLagrangian:
    """Stacked BPR Lagrangian across all physical scales.

    Parameters
    ----------
    chi_values : np.ndarray, optional
        Boundary coherence at each scale (length N_SCALES, values in [0,1]).
        Defaults to unity at every scale.
    weights : np.ndarray, optional
        Weight w_n for each scale's Lagrangian contribution.
        Defaults to uniform weights 1/N.
    """
    chi_values: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    # Internal storage for Phi and zeta (set via total_lagrangian or directly)
    _Phi: np.ndarray = field(default_factory=lambda: np.ones(_N_SCALES))
    _zeta: np.ndarray = field(default_factory=lambda: np.ones(_N_SCALES))

    def __post_init__(self) -> None:
        if self.chi_values is None:
            self.chi_values = np.ones(_N_SCALES)
        else:
            self.chi_values = np.asarray(self.chi_values, dtype=float)
        if self.weights is None:
            self.weights = np.ones(_N_SCALES) / _N_SCALES
        else:
            self.weights = np.asarray(self.weights, dtype=float)

    # ---- Per-scale Lagrangian -------------------------------------------

    @staticmethod
    def scale_lagrangian(
        n: int,
        chi: float,
        Phi: float,
        zeta: float,
        chi_prev: float = 0.0,
    ) -> float:
        """Single-scale Lagrangian density.

        L_n = chi_n^2 * Phi_n  -  zeta_n * (chi_n - chi_{n-1})^2

        The first term rewards coherence weighted by integrated information;
        the second penalises coherence jumps between adjacent scales
        (elasticity of the boundary stack).

        Parameters
        ----------
        n : int
            Scale index (informational, not used in formula).
        chi : float
            Coherence at scale n.
        Phi : float
            Integrated information at scale n.
        zeta : float
            Resonance coupling to the adjacent (lower) scale.
        chi_prev : float
            Coherence at scale n-1 (zero for the lowest scale).
        """
        return chi ** 2 * Phi - zeta * (chi - chi_prev) ** 2

    # ---- Total Lagrangian -----------------------------------------------

    def total_lagrangian(
        self,
        Phi: Optional[np.ndarray] = None,
        zeta: Optional[np.ndarray] = None,
    ) -> float:
        """Compute L_total = sum_n w_n * L_n.

        Parameters
        ----------
        Phi : np.ndarray, optional
            Integrated information per scale (defaults to stored _Phi).
        zeta : np.ndarray, optional
            Coupling constants per scale (defaults to stored _zeta).
        """
        if Phi is not None:
            self._Phi = np.asarray(Phi, dtype=float)
        if zeta is not None:
            self._zeta = np.asarray(zeta, dtype=float)

        L_total = 0.0
        for n in range(_N_SCALES):
            chi_prev = self.chi_values[n - 1] if n > 0 else 0.0
            L_n = self.scale_lagrangian(
                n,
                self.chi_values[n],
                self._Phi[n],
                self._zeta[n],
                chi_prev,
            )
            L_total += self.weights[n] * L_n
        return L_total

    # ---- Cross-scale coupling -------------------------------------------

    @staticmethod
    def cross_scale_coupling(
        n: int,
        chi_n: float,
        zeta_n: float,
        Phi_n: float,
        Phi_crit: float = _PHI_CRIT,
    ) -> float:
        """Compute coherence at scale n+1 from scale n.

        chi_{n+1} = tanh(zeta_n * chi_n * Phi_n / Phi_crit)

        The tanh saturates coherence to [0, 1) and introduces a critical
        information threshold Phi_crit above which coherence propagates
        efficiently.
        """
        argument = zeta_n * chi_n * Phi_n / Phi_crit
        return float(np.tanh(argument))

    # ---- Coherence propagation ------------------------------------------

    def propagate_coherence(
        self,
        chi_0: float,
        zeta_array: np.ndarray,
        Phi_array: np.ndarray,
        Phi_crit: float = _PHI_CRIT,
    ) -> np.ndarray:
        """Propagate coherence upward through the entire scale hierarchy.

        Starting from chi_0 at the lowest scale, iteratively applies the
        cross-scale coupling to determine chi at every scale.

        Parameters
        ----------
        chi_0 : float
            Seed coherence at the sub-Planck scale.
        zeta_array : np.ndarray
            Coupling constants (length N_SCALES).
        Phi_array : np.ndarray
            Integrated information per scale (length N_SCALES).
        Phi_crit : float
            Critical information threshold.

        Returns
        -------
        np.ndarray
            Coherence chi at each scale (length N_SCALES).
        """
        zeta = np.asarray(zeta_array, dtype=float)
        Phi = np.asarray(Phi_array, dtype=float)
        chi = np.zeros(_N_SCALES)
        chi[0] = chi_0

        for n in range(_N_SCALES - 1):
            chi[n + 1] = self.cross_scale_coupling(
                n, chi[n], zeta[n], Phi[n], Phi_crit
            )

        # Update internal state
        self.chi_values = chi
        self._Phi = Phi
        self._zeta = zeta
        return chi

    # ---- Bottleneck detection -------------------------------------------

    def coherence_bottleneck(self) -> Tuple[int, str, float]:
        """Find the scale where coherence drops most sharply.

        The bottleneck is the weakest link in the cross-scale coherence
        chain -- the scale transition with the largest fractional drop.

        Returns
        -------
        tuple
            (scale_index, scale_name, delta_chi) where delta_chi is the
            drop in coherence from scale n to n+1.
        """
        chi = self.chi_values
        drops = np.diff(chi)  # chi[n+1] - chi[n]
        # Most negative drop = biggest coherence loss
        worst_idx = int(np.argmin(drops))
        worst_drop = float(drops[worst_idx])
        name = SCALE_HIERARCHY[worst_idx]["name"]
        return worst_idx, name, worst_drop


# ===========================================================================
#  Universal BPR polarisation
# ===========================================================================

def universal_bpr_polarization(
    phi_gradient: np.ndarray,
    chi: float,
    scale: str = "lab",
) -> np.ndarray:
    """Compute BPR polarisation P = chi * grad(phi) at any physical scale.

    The boundary polarisation response is the coherence-weighted gradient
    of the scalar potential.  The coherence chi is scale-dependent:
      - lab:     chi ~ 1      (electromechanical piezoelectric regime)
      - nuclear: chi ~ 1e-3   (transmutation coupling, heavily suppressed)
      - cosmic:  chi ~ 1e-6   (astrophysical EM precursor signals)

    Parameters
    ----------
    phi_gradient : np.ndarray
        Gradient of the scalar potential (V/m, MeV/fm, or natural units).
    chi : float
        Boundary coherence (overrides the scale default if provided and
        scale is explicitly given).
    scale : str
        Physical scale label: 'lab', 'nuclear', or 'cosmic'.

    Returns
    -------
    np.ndarray
        Polarisation vector P with the same shape as phi_gradient.
    """
    # Default coherence per scale (can be overridden by explicit chi)
    _scale_defaults = {
        "lab": 1.0,
        "nuclear": 1e-3,
        "cosmic": 1e-6,
    }

    if scale not in _scale_defaults:
        raise ValueError(f"scale must be one of {list(_scale_defaults.keys())}")

    # Use provided chi (allows caller to override defaults)
    effective_chi = chi if chi is not None else _scale_defaults[scale]

    grad = np.asarray(phi_gradient, dtype=float)
    return effective_chi * grad


# ===========================================================================
#  Scale-invariant coherence analysis
# ===========================================================================

def scale_invariant_coherence(
    chi_values: np.ndarray,
    lengths: np.ndarray,
) -> Tuple[float, float, bool]:
    """Test whether coherence chi(L) follows a power law across scales.

    Fits log(chi) vs log(L) by least squares.  A good fit (R^2 > 0.9)
    indicates scale-invariant (self-similar) coherence structure, which
    is the hallmark of a conformal boundary phase.

    Parameters
    ----------
    chi_values : np.ndarray
        Coherence at each scale.
    lengths : np.ndarray
        Characteristic length at each scale (metres).

    Returns
    -------
    tuple
        (exponent, R_squared, is_scale_invariant)
        - exponent: power-law exponent alpha in chi ~ L^alpha
        - R_squared: coefficient of determination
        - is_scale_invariant: True if R^2 > 0.9
    """
    chi = np.asarray(chi_values, dtype=float)
    L = np.asarray(lengths, dtype=float)

    # Filter out zero or negative values (can't take log)
    mask = (chi > 0) & (L > 0)
    if np.sum(mask) < 2:
        return 0.0, 0.0, False

    log_chi = np.log(chi[mask])
    log_L = np.log(L[mask])

    # Linear regression: log(chi) = alpha * log(L) + const
    coeffs = np.polyfit(log_L, log_chi, 1)
    alpha = coeffs[0]

    # R^2 calculation
    log_chi_fit = np.polyval(coeffs, log_L)
    ss_res = np.sum((log_chi - log_chi_fit) ** 2)
    ss_tot = np.sum((log_chi - np.mean(log_chi)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(alpha), float(R2), bool(R2 > 0.9)


# ===========================================================================
#  Mesoscale decoherence syndrome
# ===========================================================================

def mesoscale_decoherence_syndrome(
    chi_mesoscale: float,
    chi_cellular: float,
    threshold: float = 0.5,
) -> Tuple[bool, float, str]:
    """Predict whether mesoscale coherence breakdown occurs.

    The "mesoscale decoherence syndrome" is the prediction that coherence
    can survive from quantum to cellular scales but fail to propagate into
    the mesoscale (~mm) regime.  This would manifest as a sharp drop in
    boundary responsiveness at intermediate scales.

    Parameters
    ----------
    chi_mesoscale : float
        Coherence at the mesoscale (~1 mm).
    chi_cellular : float
        Coherence at the cellular scale (~10 um).
    threshold : float
        Critical ratio below which decoherence syndrome is diagnosed.

    Returns
    -------
    tuple
        (has_syndrome, severity, description)
        - has_syndrome: bool
        - severity: float in [0, 1], where 1 is complete decoherence
        - description: str
    """
    if chi_cellular <= 0:
        return True, 1.0, (
            "Cellular coherence is zero: complete decoherence at all "
            "scales above cellular."
        )

    ratio = chi_mesoscale / chi_cellular
    has_syndrome = ratio < threshold

    # Severity: 0 = no syndrome, 1 = total mesoscale decoherence
    severity = max(0.0, 1.0 - ratio / threshold) if has_syndrome else 0.0

    if has_syndrome:
        desc = (
            f"Mesoscale decoherence syndrome detected: "
            f"chi_meso/chi_cell = {ratio:.4f} < {threshold}. "
            f"Severity = {severity:.3f}. "
            f"Coherence chain breaks at the mm scale."
        )
    else:
        desc = (
            f"No mesoscale decoherence: chi_meso/chi_cell = {ratio:.4f} "
            f">= {threshold}. Coherence propagates through mesoscale."
        )

    return has_syndrome, severity, desc
