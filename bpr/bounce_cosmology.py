"""Bounce cosmology from the BPR substrate UV cutoff Λ_b.

Implements the tier-B "Attack C" singularity-resolution target: the substrate
has a finite UV cutoff (boundary lattice spacing a_boundary), so curvature is
bounded and the classical Big-Bang singularity is replaced by a bounce — the
same mechanism as Loop Quantum Cosmology (LQC), but with the critical density
set by BPR's substrate scale rather than the bare Planck density.

Physics
-------
LQC modifies the Friedmann equation with a ρ²/ρ_crit correction:

    H² = (8πG/3) ρ (1 − ρ/ρ_crit).

The bounce occurs at ρ = ρ_crit where H = 0. In standard LQC,
ρ_crit ≈ 0.41 ρ_Planck. In BPR the natural cutoff is the substrate energy
density at the boundary lattice scale a_boundary = ℓ_Pl √(p/48π²):

    ρ_crit^BPR = (ℏ c / a_boundary⁴) × (geometric factor).

Because a_boundary > ℓ_Pl (for p ≫ 48π²), the BPR bounce density is BELOW
the Planck density — the substrate "bounces early," at a sub-Planckian
scale. This shifts the predicted bounce temperature and the primordial
gravitational-wave spectrum cutoff.

Falsifiable prediction
----------------------
The bounce imprints a high-frequency cutoff on the primordial GW spectrum
and a characteristic suppression/oscillation in the scalar power spectrum at
the largest scales (low CMB multipoles). The cutoff scale is set by
ρ_crit^BPR, hence by p. This is testable against CMB-S4 (tensor modes) and
the existing low-ℓ CMB power deficit.

Honest status
-------------
SPECULATIVE EXTENSION. The Friedmann modification is borrowed from LQC; what
BPR contributes is the SPECIFIC value of ρ_crit from the substrate scale.
The bounce dynamics, GW transfer, and CMB transfer functions are standard;
no new GR machinery is derived here. The prediction is a scale, not a full
spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_C = 299792458.0
_HBAR = 1.054571817e-34
_G = 6.67430e-11
_KB = 1.380649e-23
_L_PLANCK = np.sqrt(_HBAR * _G / _C ** 3)
_RHO_PLANCK = _C ** 5 / (_HBAR * _G ** 2)        # ≈ 5.16e96 kg/m³ (mass density)
_T_PLANCK = np.sqrt(_HBAR * _C ** 5 / _G) / _KB  # ≈ 1.42e32 K
_LQC_RHO_CRIT_FRACTION = 0.41                     # ρ_crit/ρ_Pl in standard LQC


@dataclass
class BounceCosmology:
    """BPR substrate bounce: ρ_crit and bounce signatures from p.

    Parameters
    ----------
    p : int
        Substrate structural prime.
    """

    p: int = 104761

    @property
    def a_boundary_m(self) -> float:
        """Boundary lattice spacing a_b = ℓ_Pl √(p/48π²) [m]."""
        return _L_PLANCK * np.sqrt(self.p / (48.0 * np.pi ** 2))

    @property
    def rho_crit_kg_m3(self) -> float:
        """BPR bounce critical density [kg/m³].

        Set by the substrate scale: ρ_crit = (ℏ/c) / a_boundary⁴ × LQC
        fraction. Below ρ_Planck because a_boundary > ℓ_Pl.
        """
        # energy density at the lattice scale, converted to mass density
        rho_energy = _HBAR * _C / self.a_boundary_m ** 4   # J/m³
        rho_mass = rho_energy / _C ** 2                     # kg/m³
        return _LQC_RHO_CRIT_FRACTION * rho_mass

    @property
    def rho_crit_over_planck(self) -> float:
        return self.rho_crit_kg_m3 / _RHO_PLANCK

    @property
    def bounce_temperature_K(self) -> float:
        """Bounce temperature T_b ∝ ρ_crit^(1/4) [K]."""
        # T ∝ (ρ c²)^(1/4) in natural radiation scaling, normalized to
        # T_Planck at ρ_Planck.
        return _T_PLANCK * self.rho_crit_over_planck ** 0.25

    @property
    def gw_cutoff_frequency_Hz(self) -> float:
        """Characteristic primordial-GW cutoff frequency today [Hz].

        Redshifted from the bounce scale. Rough estimate: the comoving
        Hubble scale at the bounce sets the peak; redshifting by the
        bounce-to-today factor (T_b/T_0) gives an observer-frame frequency.
        """
        T_0 = 2.725  # CMB temperature today [K]
        H_bounce = np.sqrt(8.0 * np.pi * _G * self.rho_crit_kg_m3 / 3.0)  # s⁻¹
        f_bounce = H_bounce / (2.0 * np.pi)
        return f_bounce * (T_0 / self.bounce_temperature_K)

    def friedmann_correction(self, rho_over_crit: float) -> float:
        """LQC-modified Friedmann factor (1 − ρ/ρ_crit). Zero at bounce."""
        return 1.0 - rho_over_crit

    def report(self) -> dict:
        return {
            "a_boundary_m": float(self.a_boundary_m),
            "a_boundary_over_planck": float(self.a_boundary_m / _L_PLANCK),
            "rho_crit_kg_m3": float(self.rho_crit_kg_m3),
            "rho_crit_over_planck": float(self.rho_crit_over_planck),
            "bounce_temperature_K": float(self.bounce_temperature_K),
            "bounce_T_over_planck": float(self.bounce_temperature_K / _T_PLANCK),
            "gw_cutoff_frequency_Hz": float(self.gw_cutoff_frequency_Hz),
            "status": (
                "SPECULATIVE: LQC-style Friedmann modification; BPR "
                "contributes ρ_crit from the substrate scale a_boundary. "
                "Sub-Planckian bounce (a_boundary > ℓ_Pl). Prediction is a "
                "scale (bounce T, GW cutoff), not a full spectrum."
            ),
        }
