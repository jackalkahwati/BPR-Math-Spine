"""
BPR Interpretation of Fluid Singularities
==========================================

BPR stress tensor in Navier-Stokes; phase-boundary feedback;
interpretation of AI-discovered singularity structures.

Key equations
-------------
    T_BPR = beta * Re(exp(i (phi_b - phi_f))) * grad_u
    G(phi) = 1 / (1 - r exp(i Delta_phi))    (feedback gain)
    viscous damping: d(Delta_phi)/dt ~ -nu k^2

Prediction: Phase condition Delta_phi = n pi + eps near singularity.

References: Al-Kahwati (2026), BPR and AI-Discovered Singularities
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# BPR stress tensor
# ---------------------------------------------------------------------------

def bpr_stress_tensor(
    phi_boundary: float,
    phi_fluid: float,
    grad_u: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """BPR-modified stress tensor contribution.

    T_BPR = beta * Re(exp(i (phi_b - phi_f))) * grad_u

    The boundary-fluid phase mismatch modulates the effective viscous
    stress, amplifying or suppressing it depending on coherence.

    Parameters
    ----------
    phi_boundary : float
        Boundary phase (radians).
    phi_fluid : float
        Fluid interior phase (radians).
    grad_u : ndarray
        Velocity gradient tensor (arbitrary shape).
    beta : float
        BPR coupling strength.

    Returns
    -------
    T : ndarray
        BPR stress tensor, same shape as grad_u.
    """
    delta_phi = phi_boundary - phi_fluid
    modulation = beta * np.cos(delta_phi)  # Re(exp(i delta_phi))
    return modulation * grad_u


# ---------------------------------------------------------------------------
# Feedback gain
# ---------------------------------------------------------------------------

def feedback_gain(r: float, delta_phi: float) -> Tuple[float, float]:
    """Closed-loop feedback gain from boundary-fluid phase coupling.

    G = 1 / (1 - r exp(i Delta_phi))

    Diverges as r -> 1 and Delta_phi -> 0 (mod 2 pi), signalling
    onset of resonant instability.

    Parameters
    ----------
    r : float
        Feedback amplitude (0 < r < 1 for stability).
    delta_phi : float
        Phase difference (radians).

    Returns
    -------
    magnitude : float
        |G|
    phase : float
        arg(G) in radians.
    """
    z = r * np.exp(1j * delta_phi)
    G = 1.0 / (1.0 - z)
    return float(np.abs(G)), float(np.angle(G))


# ---------------------------------------------------------------------------
# Viscous phase damping
# ---------------------------------------------------------------------------

def viscous_phase_damping(delta_phi: float, nu: float, k: float) -> float:
    """Rate of phase-difference decay from viscous dissipation.

    d(Delta_phi)/dt = -nu * k^2

    At high wavenumber k the phase difference is rapidly damped,
    destroying BPR coherence at small scales.

    Parameters
    ----------
    delta_phi : float
        Current phase difference (unused in rate, kept for interface).
    nu : float
        Kinematic viscosity.
    k : float
        Wavenumber.

    Returns
    -------
    rate : float
        Time derivative of Delta_phi.
    """
    return -nu * k ** 2


# ---------------------------------------------------------------------------
# Singularity phase condition
# ---------------------------------------------------------------------------

def singularity_phase_condition(
    delta_phi: float,
    n: int = 1,
    tolerance: float = 0.1,
) -> Tuple[bool, float]:
    """Check whether the phase difference is near the singularity condition.

    Condition:  Delta_phi approx n pi + epsilon

    Near this condition the feedback gain diverges and the BPR stress
    tensor changes sign, corresponding to the AI-discovered singularity
    structures in Navier-Stokes.

    Parameters
    ----------
    delta_phi : float
        Phase difference (radians).
    n : int
        Integer branch index.
    tolerance : float
        Acceptable deviation |epsilon|.

    Returns
    -------
    near : bool
        True if |Delta_phi - n pi| < tolerance.
    epsilon : float
        Deviation from exact resonance.
    """
    target = n * np.pi
    # Wrap difference to [-pi, pi]
    eps = (delta_phi - target + np.pi) % (2 * np.pi) - np.pi
    return bool(np.abs(eps) < tolerance), float(eps)


# ---------------------------------------------------------------------------
# Enstrophy growth from BPR stress
# ---------------------------------------------------------------------------

def enstrophy_growth(
    omega: np.ndarray,
    T_bpr: np.ndarray,
) -> float:
    """Enstrophy production rate from the BPR stress contribution.

    dOmega/dt ~ integral omega . (curl T_BPR)

    Simplified 2D scalar version:  dOmega/dt = sum(omega * T_bpr).

    Parameters
    ----------
    omega : ndarray
        Vorticity field.
    T_bpr : ndarray
        BPR stress field (same shape as omega).

    Returns
    -------
    rate : float
        Enstrophy growth rate contribution.
    """
    return float(np.sum(omega * T_bpr))


# ---------------------------------------------------------------------------
# 2D vorticity-BPR coupled simulation
# ---------------------------------------------------------------------------

def simulate_phase_coupled_flow(
    nx: int = 64,
    ny: int = 64,
    nu: float = 0.01,
    beta: float = 1.0,
    n_steps: int = 100,
    dt: float = 0.001,
) -> dict:
    """Minimal 2D vorticity simulation with BPR phase coupling.

    Solves the vorticity equation on a doubly-periodic domain [0, 2 pi]^2:

        d omega/dt = nu nabla^2 omega + beta Re(exp(i(phi_b - phi_f))) nabla^2 omega

    The boundary phase phi_b is taken as the domain-averaged vorticity
    phase; the fluid phase phi_f is the local vorticity phase.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    nu : float
        Kinematic viscosity.
    beta : float
        BPR coupling strength.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.

    Returns
    -------
    result : dict
        'omega'  : final vorticity field (nx, ny)
        'energy' : list of kinetic energy at each step
        'enstrophy' : list of enstrophy at each step
    """
    # Wavenumber grids
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0  # avoid division by zero

    # Initial condition: Taylor-Green-like vortex
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    y = np.linspace(0, 2 * np.pi, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    omega = np.sin(X) * np.cos(Y)

    energies = []
    enstrophies = []

    for step in range(n_steps):
        omega_hat = np.fft.fft2(omega)

        # Diagnostics: energy = -sum(omega_hat * conj(psi_hat)) / N^2
        psi_hat = -omega_hat / K2
        energy = 0.5 * np.real(np.sum(omega_hat * np.conj(psi_hat))) / (nx * ny)
        enstrophy = 0.5 * np.sum(omega ** 2) / (nx * ny)
        energies.append(float(energy))
        enstrophies.append(float(enstrophy))

        # Boundary phase = arg of mean vorticity phasor
        phi_b = np.angle(np.mean(np.exp(1j * omega)))
        # Effective BPR modulation (scalar field)
        modulation = np.cos(phi_b - omega)  # Re(exp(i(phi_b - phi_f)))

        # Spectral viscous step with BPR-enhanced dissipation
        effective_nu = nu + beta * np.mean(modulation)
        decay = np.exp(-effective_nu * K2 * dt)
        omega_hat *= decay
        omega = np.real(np.fft.ifft2(omega_hat))

    # Final diagnostics
    omega_hat = np.fft.fft2(omega)
    psi_hat = -omega_hat / K2
    energies.append(float(0.5 * np.real(np.sum(omega_hat * np.conj(psi_hat))) / (nx * ny)))
    enstrophies.append(float(0.5 * np.sum(omega ** 2) / (nx * ny)))

    return {
        'omega': omega,
        'energy': energies,
        'enstrophy': enstrophies,
    }
