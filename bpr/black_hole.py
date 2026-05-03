"""
Black Hole Entropy from the BPR Boundary
========================================

The robust BPR route to the Bekenstein-Hawking coefficient is induced
gravity, not raw ``p``-state counting.  Integrating out the ``p`` boundary
sectors generates an Einstein-Hilbert term:

    M_Pl^2 = p Lambda_b^2 / (48 pi^2)

The Wald/replica entropy of that induced term is then:

    S_BH = A M_Pl^2 / 4 = A / (4 l_P^2)

in the unreduced Planck-mass convention used by this codebase.  The older
``Omega = p^(A/l_P^2)`` winding picture is still useful as a heuristic for
finite boundary information, but it is not the coefficient-level derivation
once the Sakharov boundary cutoff ``a/l_P = sqrt(p/(48 pi^2))`` is imposed.

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §§4, 9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Physical constants
_G = 6.67430e-11        # m³ kg⁻¹ s⁻²
_C = 299792458.0         # m/s
_HBAR = 1.054571817e-34  # J·s
_K_B = 1.380649e-23      # J/K
_L_PLANCK = np.sqrt(_HBAR * _G / _C ** 3)   # ≈ 1.616e-35 m
_L_PLANCK_SQ = _L_PLANCK ** 2


@dataclass
class BlackHoleEntropy:
    """Bekenstein-Hawking entropy from induced boundary gravity.

    The BPR boundary sectors induce the Einstein-Hilbert term through the
    Sakharov relation:

        M_Pl^2 = p Lambda_b^2 / (48 pi^2)

    The horizon entropy is the Wald entropy of that induced term:

        S_BH = A / (4 l_P^2)

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    p : int
        Substrate prime modulus (cancels in final answer).
    """
    M_solar: float = 1.0
    p: int = 104761

    _M_SUN: float = 1.989e30  # kg

    @property
    def mass_kg(self) -> float:
        return self.M_solar * self._M_SUN

    @property
    def schwarzschild_radius(self) -> float:
        """r_s = 2GM/c² (m)."""
        return 2.0 * _G * self.mass_kg / _C ** 2

    @property
    def horizon_area(self) -> float:
        """A = 4π r_s² (m²)."""
        return 4.0 * np.pi * self.schwarzschild_radius ** 2

    @property
    def n_planck_cells(self) -> float:
        """Number of Planck-area cells on the horizon."""
        return self.horizon_area / _L_PLANCK_SQ

    @property
    def n_boundary_cutoff_cells(self) -> float:
        """Number of Sakharov boundary-cutoff cells on the horizon."""
        a_boundary = _L_PLANCK * np.sqrt(self.p / (48.0 * np.pi ** 2))
        return self.horizon_area / a_boundary ** 2

    @property
    def microstates_log(self) -> float:
        """Naive raw winding count: ln(Ω_raw) = (A/a_boundary²) ln(p).

        This is a diagnostic, not the coefficient-level Bekenstein-Hawking
        entropy.  The physical coefficient comes from the induced
        Einstein-Hilbert/Wald entropy.
        """
        return self.n_boundary_cutoff_cells * np.log(self.p)

    @property
    def legacy_planck_cell_microstates_log(self) -> float:
        """Legacy Planck-cell winding heuristic kept for auditability."""
        return self.n_planck_cells * np.log(self.p)

    @property
    def entropy_bpr(self) -> float:
        """BPR entropy: S = A / (4 l_P²) (in units of k_B)."""
        return self.n_planck_cells / 4.0

    @property
    def entropy_bekenstein_hawking(self) -> float:
        """Standard Bekenstein–Hawking: S = A / (4 l_P²)."""
        return self.horizon_area / (4.0 * _L_PLANCK_SQ)

    @property
    def agreement(self) -> bool:
        """True if BPR and Bekenstein–Hawking agree (they always do)."""
        return np.isclose(self.entropy_bpr, self.entropy_bekenstein_hawking,
                          rtol=1e-10)

    @property
    def hawking_temperature(self) -> float:
        """Hawking temperature T_H = ℏ c³ / (8π G M k_B) (K)."""
        return _HBAR * _C ** 3 / (
            8.0 * np.pi * _G * self.mass_kg * _K_B
        )

    @property
    def information_bits(self) -> float:
        """Number of classical bits on the horizon: S / ln(2)."""
        return self.entropy_bpr / np.log(2)


def induced_horizon_entropy(
    horizon_area_m2: float,
    p: int = 104761,
    Lambda_b_J: Optional[float] = None,
) -> float:
    """Horizon entropy from BPR's Sakharov-induced Einstein-Hilbert term.

    In the codebase's unreduced Planck-mass convention:

        M_Pl = Lambda_b * sqrt(p / (48 pi^2))
        S    = A M_Pl^2 / (4 (hbar c)^2)

    If ``Lambda_b_J`` is omitted, the observed Planck length is used as the
    remaining dimensionful anchor and the Sakharov relation fixes
    ``Lambda_b = hbar c / a_boundary``.
    """
    if Lambda_b_J is None:
        a_boundary = _L_PLANCK * np.sqrt(p / (48.0 * np.pi ** 2))
        Lambda_b_J = _HBAR * _C / a_boundary

    return (
        horizon_area_m2
        * p
        * Lambda_b_J ** 2
        / (192.0 * np.pi ** 2 * (_HBAR * _C) ** 2)
    )


def raw_winding_entropy(
    horizon_area_m2: float,
    p: int = 104761,
    Lambda_b_J: Optional[float] = None,
) -> float:
    """Naive entropy from ``p`` labels per boundary cutoff cell.

    This diagnostic intentionally does *not* include the induced-gravity
    normalization.  With the Sakharov boundary spacing it scales with area
    but does not reproduce the Bekenstein-Hawking coefficient.
    """
    if Lambda_b_J is None:
        a_boundary = _L_PLANCK * np.sqrt(p / (48.0 * np.pi ** 2))
    else:
        a_boundary = _HBAR * _C / Lambda_b_J

    return (horizon_area_m2 / a_boundary ** 2) * np.log(p)


def black_hole_entropy(M_solar: float, p: int = 104761) -> float:
    """Convenience: return S_BH = A/(4 l_P²) for a black hole of mass M."""
    return BlackHoleEntropy(M_solar=M_solar, p=p).entropy_bpr
