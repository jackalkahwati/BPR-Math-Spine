"""
Plasmoid Confinement via Boundary Phase Resonance
==================================================

Cymatic CFD (CCFD) for predicting stable plasmoid geometries.
Unifies RF plasma confinement with BPR boundary layers.

Key equations
-------------
    Lap(psi) + k^2 psi = 0             (Helmholtz eigenmode)
    F = -grad(P) + rho*nu*Lap(v) + JxB + f_pond
    f_pond = -e^2 / (4 m_e omega^2) grad|E|^2
    Stability: min integral |F|^2 dV

Predictions: Stable radii 5-15 cm at 2.45 GHz

References: Al-Kahwati (2026), Cymatic Resonance & Plasmoid Formation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import jn_zeros, jn
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_E_CHARGE = 1.602176634e-19       # C
_M_E = 9.1093837015e-31           # kg
_MU0 = 4.0e-7 * np.pi             # H/m
_EPSILON0 = 8.854187817e-12       # F/m
_C_LIGHT = 2.99792458e8           # m/s
_K_B = 1.380649e-23               # J/K


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlasmoidConfig:
    """Configuration for a plasmoid confinement scenario.

    Parameters
    ----------
    frequency_hz : float
        Driving RF frequency (Hz).  Default 2.45 GHz (ISM band).
    power_w : float
        Input RF power (W).
    gas_pressure_pa : float
        Background gas pressure (Pa).
    B_field_T : float
        External static magnetic field strength (T).
    """
    frequency_hz: float = 2.45e9
    power_w: float = 1000.0
    gas_pressure_pa: float = 100.0
    B_field_T: float = 0.05


# ---------------------------------------------------------------------------
# Helmholtz eigenmodes in a cylindrical cavity
# ---------------------------------------------------------------------------

def helmholtz_eigenmodes_cylindrical(
    R: float,
    L: float,
    n_modes: int = 5,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Eigenvalues k^2 and radial mode shapes for a cylindrical cavity.

    For TM_mn modes in a cylinder of radius R and length L:
        k_{mn}^2 = (x_{mn}/R)^2 + (p*pi/L)^2

    where x_{mn} is the n-th zero of J_m.  Here we fix m=0, p=1 and
    sweep n = 1..n_modes.

    Parameters
    ----------
    R : float
        Cylinder radius (m).
    L : float
        Cylinder length (m).
    n_modes : int
        Number of radial modes to compute.

    Returns
    -------
    k_squared : ndarray of shape (n_modes,)
        Eigenvalues k^2 for each mode.
    mode_shapes : list of ndarray
        Each entry is the radial profile J_0(x_{0n} * r/R) sampled at 200 points.
    """
    m = 0  # azimuthal order
    p = 1  # axial half-wavelength index
    zeros = jn_zeros(m, n_modes)  # x_{0,1}, x_{0,2}, ...

    k_squared = (zeros / R) ** 2 + (p * np.pi / L) ** 2

    r = np.linspace(0, R, 200)
    mode_shapes = [jn(m, zeros[i] * r / R) for i in range(n_modes)]

    return k_squared, mode_shapes


# ---------------------------------------------------------------------------
# Ponderomotive force
# ---------------------------------------------------------------------------

def ponderomotive_force(
    E_field: np.ndarray,
    omega: float,
    m_e: float = _M_E,
    dx: float = 1e-3,
) -> np.ndarray:
    """Ponderomotive force density f_pond = -e^2/(4 m_e omega^2) grad|E|^2.

    Parameters
    ----------
    E_field : 1-D array
        Electric field amplitude along a radial line (V/m).
    omega : float
        Angular frequency 2*pi*f (rad/s).
    m_e : float
        Electron mass (kg).
    dx : float
        Spatial step for gradient computation (m).

    Returns
    -------
    f_pond : ndarray, same length as E_field
        Ponderomotive force density (N/m^3), inward-pointing for decreasing |E|.
    """
    E_sq = np.abs(E_field) ** 2
    grad_E_sq = np.gradient(E_sq, dx)
    coeff = -_E_CHARGE**2 / (4.0 * m_e * omega**2)
    return coeff * grad_E_sq


# ---------------------------------------------------------------------------
# Pressure balance
# ---------------------------------------------------------------------------

def pressure_balance(
    n_e: float,
    T_e: float,
    T_i: float,
    B: float,
    R: float,
) -> float:
    """Evaluate the radial pressure balance residual.

    Balance: n_e * (T_e + T_i) * k_B  +  B^2 / (2 mu_0) = P_confine(R)
    We model P_confine(R) = B^2/(2 mu_0) * (1 + 1/R) as a simple
    confinement-radius dependence.

    Returns the residual (positive means over-confined).

    Parameters
    ----------
    n_e : float
        Electron number density (m^-3).
    T_e, T_i : float
        Electron and ion temperatures (K).
    B : float
        Magnetic field (T).
    R : float
        Plasmoid radius (m).
    """
    P_kinetic = n_e * (T_e + T_i) * _K_B
    P_magnetic = B**2 / (2.0 * _MU0)
    P_total = P_kinetic + P_magnetic
    P_confine = P_magnetic * (1.0 + 1.0 / R)
    return P_confine - P_total


# ---------------------------------------------------------------------------
# Confinement length
# ---------------------------------------------------------------------------

def confinement_length(v_group: float, delta_f_boundary: float) -> float:
    """Boundary-phase confinement length.

    L_c ~ v_g / (2 pi Delta_f_b)

    Parameters
    ----------
    v_group : float
        Group velocity in the medium (m/s).
    delta_f_boundary : float
        Boundary frequency mismatch (Hz).

    Returns
    -------
    L_c : float
        Confinement length (m).
    """
    return v_group / (2.0 * np.pi * delta_f_boundary)


# ---------------------------------------------------------------------------
# Stability stiffness
# ---------------------------------------------------------------------------

def stability_stiffness(
    L_func,
    R: float,
    dR: float = 1e-5,
) -> float:
    """Effective restoring stiffness k_eff = dL/dR.

    Stable configurations have k_eff > 0 (confinement energy increases
    with radius perturbation).

    Parameters
    ----------
    L_func : callable
        Function L(R) returning a scalar confinement energy metric.
    R : float
        Equilibrium radius (m).
    dR : float
        Finite-difference step for numerical derivative.

    Returns
    -------
    k_eff : float
        Stiffness (positive = stable).
    """
    return (L_func(R + dR) - L_func(R - dR)) / (2.0 * dR)


# ---------------------------------------------------------------------------
# Stable radius prediction
# ---------------------------------------------------------------------------

def stable_radius_prediction(
    config: PlasmoidConfig,
    n_e: float = 1e18,
    T_e: float = 1e4,
    T_i: float = 5e3,
    R_min: float = 0.01,
    R_max: float = 0.30,
) -> Optional[float]:
    """Find the stable plasmoid radius where pressure balance is met.

    Searches for the root of pressure_balance(n_e, T_e, T_i, B, R) = 0
    in the interval [R_min, R_max].

    Returns
    -------
    R_stable : float or None
        Equilibrium radius (m), or None if no root found.
    """
    def _residual(R):
        return pressure_balance(n_e, T_e, T_i, config.B_field_T, R)

    fa = _residual(R_min)
    fb = _residual(R_max)

    # Need a sign change for Brent's method
    if fa * fb > 0:
        return None

    R_eq = brentq(_residual, R_min, R_max, xtol=1e-6)
    return float(R_eq)


# ---------------------------------------------------------------------------
# Boundary phase layer
# ---------------------------------------------------------------------------

def boundary_phase_layer(
    A_modes: np.ndarray,
    phi_boundary: np.ndarray,
    Gamma_b: float = 0.1,
    dt: float = 1e-3,
) -> np.ndarray:
    """Boundary phase layer source term.

    S = -d_t E_b(phi_b) - Gamma_b * Delta(phi_b)

    where E_b(phi) = sum |A_n|^2 cos(phi_n) is the boundary energy and
    Delta(phi) is the discrete Laplacian of the boundary phase field.

    Parameters
    ----------
    A_modes : 1-D array
        Mode amplitudes |A_n|.
    phi_boundary : 1-D array
        Boundary phase values at discrete points.
    Gamma_b : float
        Boundary dissipation coefficient.
    dt : float
        Time step for numerical d_t estimate (internal; returns instantaneous).

    Returns
    -------
    S : 1-D array, same length as phi_boundary
        Source term at each boundary point.
    """
    A_modes = np.asarray(A_modes, dtype=float)
    phi = np.asarray(phi_boundary, dtype=float)

    # Boundary energy derivative: dE_b/dphi ~ -sum |A_n|^2 sin(phi)
    dE_dphi = -np.sum(A_modes**2) * np.sin(phi)

    # Discrete Laplacian of phi (periodic)
    lap_phi = np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi

    S = -dE_dphi - Gamma_b * lap_phi
    return S
