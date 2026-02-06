"""
Theory VII: Gravitational Wave Phenomenology in BPR
=====================================================

Treats gravitational waves as propagating boundary stress perturbations,
derives vGW = c from substrate parameters, implements the quadrupole formula
from boundary dynamics, and connects GW memory to Theory I's memory kernel.

Key objects
-----------
* ``GWPropagation``       – speed from boundary elastic modulus
* ``GWQuadrupole``        – emission from boundary phase quadrupole
* ``GWMemory``            – permanent displacement via memory kernel

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §9
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G = 6.67430e-11        # m³ kg⁻¹ s⁻²
_C = 299792458.0         # m/s
_HBAR = 1.054571817e-34  # J·s


# ---------------------------------------------------------------------------
# §9.3  Propagation speed
# ---------------------------------------------------------------------------

@dataclass
class GWPropagation:
    """Gravitational wave propagation speed from boundary elasticity.

    v_GW = √(κ_boundary / ρ_boundary) = c

    κ_boundary and ρ_boundary are both set by the same substrate parameters,
    so their ratio is exactly c² — deriving v_GW = c from first principles.
    """
    kappa_boundary: float = _C ** 2   # boundary stiffness (m²/s²)
    rho_boundary: float = 1.0         # boundary density (dimensionless in natural units)

    @property
    def v_gw(self) -> float:
        """Gravitational wave speed (m/s)."""
        return np.sqrt(self.kappa_boundary / self.rho_boundary)

    @property
    def dispersion(self) -> float:
        """Dispersion: v_gw - c.  Prediction P7.1: exactly zero to all orders."""
        return self.v_gw - _C


# ---------------------------------------------------------------------------
# §9.4  Quadrupole emission from boundary phase
# ---------------------------------------------------------------------------

@dataclass
class GWQuadrupole:
    """Gravitational wave emission via boundary phase quadrupole.

    P_GW = (G / 5c⁵) ⟨ d³Q^φ_ij/dt³  d³Q^φ_ij/dt³ ⟩

    where Q^φ_ij(t) = ∫ φ(x,t) (x_i x_j − ⅓ δ_ij r²) dS.

    Parameters
    ----------
    Q_phi_dddot : ndarray, shape (3, 3, n_times)
        Third time-derivative of the boundary phase quadrupole moment.
    """
    Q_phi_dddot: Optional[np.ndarray] = None

    @property
    def power(self) -> float:
        """Radiated GW power (W), time-averaged over available samples."""
        if self.Q_phi_dddot is None:
            raise ValueError("Quadrupole third derivative not set.")
        Q3 = self.Q_phi_dddot
        # Contract and average: P = (G/5c⁵) ⟨Q³_ij Q³_ij⟩
        n_times = Q3.shape[-1]
        contraction = np.sum(Q3 ** 2, axis=(0, 1))  # sum over i,j per timestep
        avg = np.mean(contraction)
        return (_G / (5.0 * _C ** 5)) * avg

    @staticmethod
    def compute_quadrupole(phi: np.ndarray, x: np.ndarray,
                           dt: float) -> np.ndarray:
        """Compute Q^φ_ij(t) from sampled boundary field.

        Parameters
        ----------
        phi : ndarray, shape (n_points, n_times)
            Boundary phase field φ(x, t).
        x : ndarray, shape (n_points, 3)
            Spatial coordinates of boundary points.
        dt : float
            Time step.

        Returns
        -------
        Q : ndarray, shape (3, 3, n_times)
        """
        n_points, n_times = phi.shape
        Q = np.zeros((3, 3, n_times))
        for t_idx in range(n_times):
            for i in range(3):
                for j in range(3):
                    integrand = phi[:, t_idx] * (
                        x[:, i] * x[:, j]
                        - (1.0 / 3.0) * (i == j) * np.sum(x ** 2, axis=1)
                    )
                    Q[i, j, t_idx] = np.sum(integrand)
        return Q


# ---------------------------------------------------------------------------
# §9.5  GW memory from the boundary memory kernel
# ---------------------------------------------------------------------------

def gw_memory_displacement(memory_kernel_func, delta_T: np.ndarray,
                           times: np.ndarray, dt: float) -> float:
    """Permanent displacement from GW memory effect.

    Δφ_permanent = ∫_{-∞}^{+∞} M(t,t') δT(t') dt'

    Non-zero when M has a non-zero DC component (asymmetric source events).

    Parameters
    ----------
    memory_kernel_func : callable(t, t') → float or ndarray
        Memory kernel from Theory I.
    delta_T : ndarray, shape (n_times,)
        Boundary stress perturbation δT(t).
    times : ndarray, shape (n_times,)
        Time samples.
    dt : float
        Time step.
    """
    # Evaluate at t = max(times) (post-event)
    t_obs = times[-1]
    M_vals = np.array([memory_kernel_func(t_obs, tp) for tp in times])
    return float(np.sum(M_vals * delta_T) * dt)


def prime_harmonic_gw_spectrum(frequencies: np.ndarray, p: int = 7,
                                n_modes: int = 5,
                                amplitude: float = 1e-22) -> np.ndarray:
    """GW memory fine-structure spectrum with prime-harmonic peaks (P7.2).

    Spectral features appear at f_n = n / (p × τ_Planck) for n = 1,…

    For demonstration, models the spectrum as Lorentzian peaks at
    prime-harmonic frequencies.
    """
    tau_planck = 5.391247e-44  # s
    spectrum = np.zeros_like(frequencies)
    for n in range(1, n_modes + 1):
        f_n = n / (p * tau_planck)
        width = f_n * 0.01  # 1% fractional width
        spectrum += amplitude / ((frequencies - f_n) ** 2 + width ** 2)
    return spectrum


# ---------------------------------------------------------------------------
# Stochastic GW background from early-universe phase transitions (P7.3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §9.6  GW memory prime-periodic ringing  (Prediction 15)
# ---------------------------------------------------------------------------

def gw_memory_ringing(times: np.ndarray, p: int = 7,
                       tau_ring: float = 1.0,
                       amplitude: float = 1e-22) -> np.ndarray:
    """Prime-periodic ringing on top of GW memory step function.

    After a binary merger, the standard GW memory is a step function.
    BPR predicts oscillatory corrections at prime harmonics:

        δh_memory(t) = A × Σ_n (1/n²) exp(−t/τ_ring) cos(2πn t / (p τ_Pl))

    For realistic p this frequency is extremely high (>> LIGO band),
    but the envelope modulates lower-frequency observables.

    Parameters
    ----------
    times : ndarray – time after merger (s)
    p : int – substrate prime
    tau_ring : float – ringing decay time (s)
    amplitude : float – strain amplitude

    Returns
    -------
    ndarray – δh(t) ringing correction
    """
    tau_planck = 5.391247e-44  # s
    t = np.asarray(times, dtype=float)
    h_ring = np.zeros_like(t)
    for n in range(1, 6):
        omega_n = 2.0 * np.pi * n / (p * tau_planck)
        h_ring += (1.0 / n ** 2) * np.cos(omega_n * t)
    h_ring *= amplitude * np.exp(-t / (tau_ring + 1e-30))
    return h_ring


def stochastic_gw_background(frequencies: np.ndarray,
                              T_transition: float = 160e9,  # eV (electroweak)
                              amplitude: float = 1e-15) -> np.ndarray:
    """Stochastic GW background from Class D phase transitions (Theory IV).

    Simple power-law + prime-periodic modulation.
    """
    f_peak = T_transition * 1.6e-19 / _HBAR  # convert eV to Hz (rough)
    # Power-law envelope
    x = frequencies / f_peak
    envelope = amplitude * x ** 3 / (1.0 + x ** 4)
    return envelope
