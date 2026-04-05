"""
Electromechanical Coherence via Boundary Phase Resonance
=========================================================

Unifies flexoelectricity, piezoelectricity, and seismo-EM
phenomena as boundary phase manifestations.

Key equations
-------------
    Flexoelectric:  P ~ chi * nabla phi  (polarization from phase gradient)
    Piezoelectric:  P_i = d_ijk sigma_jk
    BPR coupling:   P_BPR = epsilon_0 * chi_BPR * nabla phi_boundary

Predictions: Pre-seismic EM bursts, strain-induced phase shifts.

References: Al-Kahwati (2026), Emergent Electromechanical Coherence
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# Physical constants
_EPSILON_0 = 8.854187817e-12   # F/m, vacuum permittivity
_C = 299792458.0               # m/s, speed of light


# ---------------------------------------------------------------------------
# Flexoelectric polarization
# ---------------------------------------------------------------------------

def flexoelectric_polarization(
    grad_phi: np.ndarray,
    chi: float,
) -> np.ndarray:
    """Flexoelectric polarization from a phase gradient.

    P = chi * nabla phi

    In BPR, the boundary phase gradient induces a macroscopic polarization
    analogous to the flexoelectric effect in dielectrics.

    Parameters
    ----------
    grad_phi : ndarray
        Phase gradient field (arbitrary shape, e.g. (3,) or (N, 3)).
    chi : float
        Flexoelectric susceptibility (C/m).

    Returns
    -------
    P : ndarray
        Polarization vector(s), same shape as grad_phi.
    """
    return chi * grad_phi


# ---------------------------------------------------------------------------
# Piezoelectric polarization
# ---------------------------------------------------------------------------

def piezoelectric_polarization(
    stress_tensor: np.ndarray,
    d_coefficients: np.ndarray,
) -> np.ndarray:
    """Piezoelectric polarization from mechanical stress.

    P_i = d_ijk sigma_jk

    Contracted using Einstein summation over the stress tensor indices.

    Parameters
    ----------
    stress_tensor : ndarray (3, 3) or (..., 3, 3)
        Mechanical stress tensor.
    d_coefficients : ndarray (3, 3, 3)
        Piezoelectric strain coefficients d_ijk.

    Returns
    -------
    P : ndarray (3,) or (..., 3)
        Polarization vector.
    """
    return np.einsum('ijk,...jk->...i', d_coefficients, stress_tensor)


# ---------------------------------------------------------------------------
# BPR electromechanical coupling
# ---------------------------------------------------------------------------

def bpr_electromechanical_coupling(
    phi_boundary: np.ndarray,
    epsilon_0: float = _EPSILON_0,
    chi_bpr: float = 1.0,
    dx: float = 1.0,
) -> np.ndarray:
    """BPR-induced polarization from boundary phase field.

    P_BPR = epsilon_0 * chi_BPR * nabla phi_boundary

    The boundary phase gradient acts as an effective electric field source.

    Parameters
    ----------
    phi_boundary : ndarray (N,)
        1D boundary phase field.
    epsilon_0 : float
        Vacuum permittivity (F/m).
    chi_bpr : float
        BPR electromechanical susceptibility (dimensionless).
    dx : float
        Grid spacing for numerical gradient.

    Returns
    -------
    P_bpr : ndarray (N,)
        BPR-induced polarization field.
    """
    grad_phi = np.gradient(phi_boundary, dx)
    return epsilon_0 * chi_bpr * grad_phi


# ---------------------------------------------------------------------------
# Seismic EM emission
# ---------------------------------------------------------------------------

def seismic_em_emission(
    strain_rate: float,
    d_eff: float,
    volume: float,
) -> float:
    """Estimate EM power radiated from piezoelectric strain.

    P_em ~ (d_eff * strain_rate)^2 * volume / (epsilon_0 * c)

    This gives the order of magnitude of EM emission from crustal
    piezoelectric minerals under tectonic strain.

    Parameters
    ----------
    strain_rate : float
        Time derivative of strain (1/s).
    d_eff : float
        Effective piezoelectric coefficient (C/N or C/m^2 in Voigt).
    volume : float
        Active volume of piezoelectric material (m^3).

    Returns
    -------
    power : float
        Radiated EM power (W).
    """
    # Polarization current density: dP/dt ~ d_eff * strain_rate
    j_pol = d_eff * strain_rate
    # Larmor-like radiated power: P ~ j^2 V / (eps_0 c)
    power = j_pol ** 2 * volume / (_EPSILON_0 * _C)
    return float(power)


# ---------------------------------------------------------------------------
# Strain-phase shift
# ---------------------------------------------------------------------------

def strain_phase_shift(
    strain: float,
    coupling_constant: float,
) -> float:
    """Phase shift induced by mechanical strain.

    Delta_phi = g * epsilon

    where g is the strain-phase coupling constant (rad per unit strain).

    Parameters
    ----------
    strain : float
        Mechanical strain (dimensionless).
    coupling_constant : float
        Coupling g (rad per unit strain).

    Returns
    -------
    delta_phi : float
        Phase shift (radians).
    """
    return coupling_constant * strain


# ---------------------------------------------------------------------------
# Earthquake light spectrum prediction
# ---------------------------------------------------------------------------

def earthquake_light_spectrum(
    strain_field: np.ndarray,
    material_params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict EM emission frequencies from crustal strain field.

    The BPR framework predicts that pre-seismic EM arises from coherent
    polarization oscillations.  The dominant frequency is set by the
    strain oscillation rate and material resonance.

    Parameters
    ----------
    strain_field : ndarray (N,)
        Spatial strain profile.
    material_params : dict
        'd_eff'          : effective piezo coefficient (C/N)
        'resonance_freq' : material resonance frequency (Hz)
        'Q_factor'       : quality factor of resonance
        'chi_bpr'        : BPR susceptibility

    Returns
    -------
    frequencies : ndarray
        Frequency axis (Hz).
    spectrum : ndarray
        Relative spectral power density.
    """
    d_eff = material_params.get('d_eff', 1e-12)
    f0 = material_params.get('resonance_freq', 1e3)
    Q = material_params.get('Q_factor', 10.0)
    chi = material_params.get('chi_bpr', 1.0)

    N = len(strain_field)
    # Fourier transform of strain profile -> spatial frequencies
    strain_hat = np.fft.rfft(strain_field)
    freqs = np.fft.rfftfreq(N) * f0  # scale to physical frequency

    # Lorentzian response centred at f0 with width f0/Q
    gamma = f0 / (2 * Q)
    response = 1.0 / ((freqs - f0) ** 2 + gamma ** 2)

    # Spectral power ~ |d_eff * chi * strain_hat|^2 * response
    spectrum = (d_eff * chi) ** 2 * np.abs(strain_hat) ** 2 * response
    # Normalise to peak = 1
    peak = np.max(spectrum)
    if peak > 0:
        spectrum /= peak

    return freqs, spectrum


# ---------------------------------------------------------------------------
# Electromechanical BPR dataclass
# ---------------------------------------------------------------------------

@dataclass
class ElectromechanicalBPR:
    """Material-level BPR electromechanical model.

    Bundles material properties and provides convenience methods
    for computing polarization, strain coupling, and EM emission.

    Parameters
    ----------
    chi_flexo : float
        Flexoelectric susceptibility (C/m).
    d_piezo : ndarray (3, 3, 3)
        Piezoelectric tensor d_ijk.
    chi_bpr : float
        BPR boundary-phase susceptibility (dimensionless).
    strain_coupling : float
        Strain-to-phase coupling g (rad per unit strain).
    resonance_freq : float
        Dominant material resonance frequency (Hz).
    Q_factor : float
        Quality factor of the electromechanical resonance.
    """

    chi_flexo: float = 1e-9
    d_piezo: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3, 3))
    )
    chi_bpr: float = 1.0
    strain_coupling: float = 1.0
    resonance_freq: float = 1e3
    Q_factor: float = 10.0

    def polarization(
        self,
        grad_phi: Optional[np.ndarray] = None,
        stress: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Total polarization from flexoelectric and piezoelectric channels.

        P_total = P_flexo + P_piezo
        """
        P = np.zeros(3)
        if grad_phi is not None:
            P += flexoelectric_polarization(grad_phi, self.chi_flexo)
        if stress is not None:
            P += piezoelectric_polarization(stress, self.d_piezo)
        return P

    def phase_shift(self, strain: float) -> float:
        """Phase shift from mechanical strain."""
        return strain_phase_shift(strain, self.strain_coupling)

    def em_power(self, strain_rate: float, volume: float) -> float:
        """EM radiation power estimate from active volume."""
        d_eff = np.max(np.abs(self.d_piezo)) if np.any(self.d_piezo) else 1e-12
        return seismic_em_emission(strain_rate, d_eff, volume)
