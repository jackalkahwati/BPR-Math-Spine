"""
Time-Dependent Ginzburg-Landau BPR Solver
==========================================

Implements the foundational BPR simulation: boundary-induced pattern
entrainment during phase transitions via TDGL dynamics.

Key equations
-------------
    dPsi/dt = -(alpha*Psi + beta*|Psi|^2*Psi - kappa*Lap(Psi) - lam*V(x)) + xi(t)
    F[Psi] = integral[alpha|Psi|^2 + (beta/2)|Psi|^4 + kappa|grad Psi|^2 - lam*V*Re(Psi)] dV
    C(t) = A exp(-t/tau) + C0

Predictions: NCC > 0.9 during coupling, tau_decay ~ 40-60 steps

References: Al-Kahwati (2026), BPR Deterministic Framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TDGLConfig:
    """Parameters for a 2-D TDGL simulation on a regular grid.

    Parameters
    ----------
    alpha : float
        Linear coefficient (controls T - T_c distance).
    beta : float
        Nonlinear saturation coefficient.
    kappa : float
        Gradient (stiffness) coefficient.
    lam : float
        Boundary coupling strength lambda.
    noise_sigma : float
        Standard deviation of additive white noise xi(t).
    dx : float
        Spatial grid spacing.
    dt : float
        Time step for Euler integration.
    nx, ny : int
        Grid dimensions.
    """
    alpha: float = -1.0
    beta: float = 1.0
    kappa: float = 1.0
    lam: float = 0.5
    noise_sigma: float = 0.01
    dx: float = 1.0
    dt: float = 0.05
    nx: int = 64
    ny: int = 64


# ---------------------------------------------------------------------------
# Boundary potential patterns
# ---------------------------------------------------------------------------

def boundary_coupling_potential(
    nx: int,
    ny: int,
    pattern: str = "stripe",
    wavelength: float = 8.0,
) -> np.ndarray:
    """Create a boundary coupling potential V(x) on an (nx, ny) grid.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    pattern : str
        One of 'stripe', 'checkerboard', 'radial'.
    wavelength : float
        Characteristic length scale of the pattern (in grid units).

    Returns
    -------
    V : ndarray of shape (nx, ny)
    """
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    k = 2.0 * np.pi / wavelength

    if pattern == "stripe":
        V = np.cos(k * X)
    elif pattern == "checkerboard":
        V = np.cos(k * X) * np.cos(k * Y)
    elif pattern == "radial":
        cx, cy = nx / 2.0, ny / 2.0
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        V = np.cos(k * R)
    else:
        raise ValueError(f"Unknown pattern '{pattern}'")

    return V


# ---------------------------------------------------------------------------
# Laplacian (periodic boundary conditions)
# ---------------------------------------------------------------------------

def _laplacian_2d(psi: np.ndarray, dx: float) -> np.ndarray:
    """5-point discrete Laplacian with periodic BCs."""
    lap = (
        np.roll(psi, 1, axis=0)
        + np.roll(psi, -1, axis=0)
        + np.roll(psi, 1, axis=1)
        + np.roll(psi, -1, axis=1)
        - 4.0 * psi
    ) / dx**2
    return lap


# ---------------------------------------------------------------------------
# Single TDGL step
# ---------------------------------------------------------------------------

def tdgl_step(
    psi: np.ndarray,
    config: TDGLConfig,
    V: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """One forward-Euler step of the TDGL equation on a 2-D grid.

    dPsi/dt = -(alpha*Psi + beta*|Psi|^2*Psi - kappa*Lap Psi - lam*V) + xi

    Parameters
    ----------
    psi : ndarray, shape (nx, ny), complex or real
    config : TDGLConfig
    V : ndarray, shape (nx, ny), boundary coupling potential
    rng : numpy Generator (optional)

    Returns
    -------
    psi_new : ndarray same shape/dtype as psi
    """
    lap = _laplacian_2d(psi, config.dx)
    # Deterministic part
    dpsi = -(
        config.alpha * psi
        + config.beta * np.abs(psi) ** 2 * psi
        - config.kappa * lap
        - config.lam * V
    )
    # Stochastic noise
    if config.noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        noise = config.noise_sigma * rng.standard_normal(psi.shape)
        if np.iscomplexobj(psi):
            noise = noise + 1j * config.noise_sigma * rng.standard_normal(psi.shape)
        dpsi = dpsi + noise

    return psi + config.dt * dpsi


# ---------------------------------------------------------------------------
# Free energy functional
# ---------------------------------------------------------------------------

def free_energy(psi: np.ndarray, config: TDGLConfig, V: np.ndarray) -> float:
    """Compute the Ginzburg-Landau free energy F[psi].

    F = integral [ alpha|psi|^2 + (beta/2)|psi|^4 + kappa|grad psi|^2
                   - lam * V * Re(psi) ] dV

    Returns
    -------
    F : float
        Total free energy (in units of dx^2 per grid cell).
    """
    psi_abs2 = np.abs(psi) ** 2

    # Gradient energy via finite differences (periodic)
    grad_x = (np.roll(psi, -1, axis=0) - psi) / config.dx
    grad_y = (np.roll(psi, -1, axis=1) - psi) / config.dx
    grad_sq = np.abs(grad_x) ** 2 + np.abs(grad_y) ** 2

    integrand = (
        config.alpha * psi_abs2
        + 0.5 * config.beta * psi_abs2 ** 2
        + config.kappa * grad_sq
        - config.lam * V * np.real(psi)
    )
    return float(np.sum(integrand) * config.dx**2)


# ---------------------------------------------------------------------------
# Green's function linear response
# ---------------------------------------------------------------------------

def greens_function_response(
    psi: np.ndarray,
    V: np.ndarray,
    config: TDGLConfig,
) -> np.ndarray:
    """Linear response in Fourier space: psi_lin(k) = lam * V(k) / (alpha + kappa*k^2).

    Valid in the linearised regime |psi| << 1.

    Returns
    -------
    psi_response : ndarray, same shape as V
    """
    nx, ny = V.shape
    kx = np.fft.fftfreq(nx, d=config.dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=config.dx) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2

    V_hat = np.fft.fft2(V)
    # Green's function G(k) = 1 / (alpha + kappa*k^2)
    denom = config.alpha + config.kappa * k2
    # Avoid division by zero at the critical point
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    psi_hat = config.lam * V_hat / denom
    return np.real(np.fft.ifft2(psi_hat))


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_tdgl_simulation(
    config: TDGLConfig,
    V: np.ndarray,
    n_steps: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Run a full TDGL simulation.

    Parameters
    ----------
    config : TDGLConfig
    V : ndarray (nx, ny)
    n_steps : int
    seed : int

    Returns
    -------
    history : ndarray shape (n_steps+1, nx, ny)
        Full psi field at each timestep.
    """
    rng = np.random.default_rng(seed)
    psi = 0.01 * rng.standard_normal((config.nx, config.ny))
    history = np.empty((n_steps + 1, config.nx, config.ny))
    history[0] = psi

    for t in range(n_steps):
        psi = tdgl_step(psi, config, V, rng)
        history[t + 1] = np.real(psi) if np.iscomplexobj(psi) else psi

    return history


# ---------------------------------------------------------------------------
# Normalized cross-correlation
# ---------------------------------------------------------------------------

def normalized_cross_correlation(field1: np.ndarray, field2: np.ndarray) -> float:
    """Compute NCC between two real-valued fields.

    NCC = sum(f1_centered * f2_centered) / (||f1_centered|| * ||f2_centered||)

    Returns
    -------
    ncc : float in [-1, 1]
    """
    f1 = field1.ravel() - np.mean(field1)
    f2 = field2.ravel() - np.mean(field2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    if norm1 < 1e-30 or norm2 < 1e-30:
        return 0.0
    return float(np.dot(f1, f2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Coherence decay fitting
# ---------------------------------------------------------------------------

def coherence_decay_fit(
    coherence_history: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit C(t) = A * exp(-t/tau) + C0 to a coherence time-series.

    Parameters
    ----------
    coherence_history : 1-D array of NCC values vs time step index

    Returns
    -------
    (A, tau, C0) : tuple of floats
    """
    t = np.arange(len(coherence_history), dtype=float)
    c = np.asarray(coherence_history, dtype=float)

    def _model(t, A, tau, C0):
        return A * np.exp(-t / tau) + C0

    # Initial guesses from the data endpoints
    C0_guess = c[-1] if len(c) > 1 else 0.0
    A_guess = c[0] - C0_guess if len(c) > 0 else 1.0
    tau_guess = max(len(c) / 4.0, 1.0)

    # Ensure initial guess is feasible within bounds
    A_guess_abs = max(abs(A_guess), 1e-6)

    try:
        popt, _ = curve_fit(
            _model, t, c,
            p0=[A_guess_abs, tau_guess, C0_guess],
            bounds=([0, 0.1, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        # Fallback: return initial guesses
        return float(A_guess), float(tau_guess), float(C0_guess)
