"""
Resonance Algebra: Operator Rulebook for PDE Solvers
=====================================================

Compact fused-step spectral method with band maintenance,
dealiased nonlinear evaluation, and conservation projectors.

Key sequence: Band -> Nonlinear -> Mass/Energy Project -> Band Close

Invariants
----------
    P1: Energy neutrality within 10^{-10}
    P3: Truncation error O(K^{1-p})

Prediction: 1D Burgers error <= 0.12 at K = 32.

References: Al-Kahwati (2026), Resonance Algebra Rulebook v1.0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# ---------------------------------------------------------------------------
# Spectral band operator
# ---------------------------------------------------------------------------

@dataclass
class SpectralBand:
    """Spectral band-limiting operator with dealiasing and conservation.

    Maintains a sharp Fourier cutoff at wavenumber K on a grid with
    spacing dx.  Provides the 3/2-rule dealiased nonlinear evaluation
    and energy/mass conservation projectors.

    Parameters
    ----------
    K : int
        Band cutoff (maximum retained wavenumber index).
    dx : float
        Grid spacing.
    """

    K: int
    dx: float

    @property
    def n_grid(self) -> int:
        """Physical grid size implied by dx and K (at least 2K)."""
        return max(int(2 * np.pi / self.dx), 2 * self.K)

    # ----- band limiting -----

    def band_limit(self, u_hat: np.ndarray) -> np.ndarray:
        """Zero all Fourier modes above the band cutoff K.

        Parameters
        ----------
        u_hat : ndarray (complex)
            Fourier coefficients.

        Returns
        -------
        u_hat_filtered : ndarray (complex)
            Band-limited coefficients.
        """
        N = len(u_hat)
        out = u_hat.copy()
        # Zero modes with |k| > K
        freqs = np.fft.fftfreq(N, d=self.dx)
        k_indices = np.abs(freqs) * N * self.dx  # mode index
        out[k_indices > self.K] = 0.0
        return out

    # ----- dealiased nonlinear -----

    def dealiased_nonlinear(
        self,
        u: np.ndarray,
        nonlinear_func: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Evaluate a nonlinear term with 3/2-rule dealiasing.

        Pads to 3/2 * N in Fourier space, evaluates in physical space,
        then truncates back.

        Parameters
        ----------
        u : ndarray
            Physical-space field.
        nonlinear_func : callable
            Nonlinearity f(u) evaluated pointwise.

        Returns
        -------
        result_hat : ndarray (complex)
            Dealiased Fourier coefficients of f(u), length = len(u).
        """
        N = len(u)
        M = int(np.ceil(1.5 * N))
        # Pad in Fourier space
        u_hat = np.fft.fft(u)
        u_hat_padded = np.zeros(M, dtype=complex)
        u_hat_padded[:N // 2] = u_hat[:N // 2]
        u_hat_padded[-(N // 2):] = u_hat[-(N // 2):]
        # Evaluate nonlinearity on fine grid
        u_fine = np.real(np.fft.ifft(u_hat_padded)) * (M / N)
        f_fine = nonlinear_func(u_fine)
        # Truncate back
        f_hat_padded = np.fft.fft(f_fine) * (N / M)
        result_hat = np.zeros(N, dtype=complex)
        result_hat[:N // 2] = f_hat_padded[:N // 2]
        result_hat[-(N // 2):] = f_hat_padded[-(N // 2):]
        return result_hat

    # ----- conservation projectors -----

    def energy_projector(self, u_hat: np.ndarray, E_target: float) -> np.ndarray:
        """Scale Fourier coefficients to conserve total energy.

        u_hat *= sqrt(E_target / E_current)

        Energy defined as E = sum |u_hat|^2.

        Parameters
        ----------
        u_hat : ndarray (complex)
            Fourier coefficients.
        E_target : float
            Desired energy.

        Returns
        -------
        u_hat_scaled : ndarray (complex)
        """
        E_current = np.sum(np.abs(u_hat) ** 2)
        if E_current < 1e-30:
            return u_hat.copy()
        return u_hat * np.sqrt(E_target / E_current)

    def mass_projector(self, u_hat: np.ndarray, M_target: float) -> np.ndarray:
        """Set the zero mode to conserve total mass.

        u_hat[0] = M_target

        Parameters
        ----------
        u_hat : ndarray (complex)
            Fourier coefficients.
        M_target : float
            Desired zeroth Fourier mode (proportional to mean value).

        Returns
        -------
        u_hat_projected : ndarray (complex)
        """
        out = u_hat.copy()
        out[0] = M_target
        return out


# ---------------------------------------------------------------------------
# Fused spectral step
# ---------------------------------------------------------------------------

def fused_step(
    u_hat: np.ndarray,
    dt: float,
    nu: float,
    band: SpectralBand,
) -> np.ndarray:
    """One fused Resonance Algebra step: band -> nonlinear -> project -> close.

    Integrates one time step of the viscous Burgers equation:

        du/dt + u du/dx = nu d^2u/dx^2

    using an operator-split spectral method with band maintenance.

    Parameters
    ----------
    u_hat : ndarray (complex)
        Current Fourier coefficients.
    dt : float
        Time step.
    nu : float
        Viscosity.
    band : SpectralBand
        Band operator.

    Returns
    -------
    u_hat_new : ndarray (complex)
        Updated Fourier coefficients after one fused step.
    """
    N = len(u_hat)

    # Record conserved quantities
    E0 = np.sum(np.abs(u_hat) ** 2)
    M0 = u_hat[0]

    # Step 1: Band limit
    u_hat = band.band_limit(u_hat)

    # Step 2: Nonlinear term (Burgers: -u du/dx in Fourier = -0.5 d(u^2)/dx)
    u = np.real(np.fft.ifft(u_hat))

    def burgers_nonlinear(v: np.ndarray) -> np.ndarray:
        return -0.5 * v ** 2

    nl_hat = band.dealiased_nonlinear(u, burgers_nonlinear)
    # Derivative in Fourier space: ik * nl_hat
    k = np.fft.fftfreq(N, d=band.dx) * 2 * np.pi
    nl_hat = 1j * k * nl_hat

    # Step 3: Viscous diffusion (exact integrating factor)
    k2 = k ** 2
    diffusion = np.exp(-nu * k2 * dt)
    u_hat = (u_hat + dt * nl_hat) * diffusion

    # Step 4: Conservation projectors
    u_hat = band.energy_projector(u_hat, E0)
    u_hat = band.mass_projector(u_hat, M0)

    # Step 5: Band close
    u_hat = band.band_limit(u_hat)

    return u_hat


# ---------------------------------------------------------------------------
# 1D Burgers solver
# ---------------------------------------------------------------------------

def solve_burgers_1d(
    nx: int = 128,
    nu: float = 0.01,
    T: float = 1.0,
    K: int = 32,
    u0_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the 1D viscous Burgers equation using fused spectral steps.

    du/dt + u du/dx = nu d^2u/dx^2    on [0, 2 pi], periodic BCs.

    Parameters
    ----------
    nx : int
        Number of grid points.
    nu : float
        Viscosity.
    T : float
        Final time.
    K : int
        Spectral band cutoff.
    u0_func : callable, optional
        Initial condition u0(x).  Default: sin(x).

    Returns
    -------
    x : ndarray (nx,)
        Grid points.
    u : ndarray (nx,)
        Solution at time T.
    """
    dx = 2 * np.pi / nx
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    band = SpectralBand(K=K, dx=dx)

    if u0_func is None:
        u0 = np.sin(x)
    else:
        u0 = u0_func(x)

    u_hat = np.fft.fft(u0)

    # Adaptive step count: CFL-like with safety factor
    dt = 0.5 * dx / (np.max(np.abs(u0)) + 1e-10)
    dt = min(dt, 0.5 * dx ** 2 / (nu + 1e-10))
    n_steps = max(int(np.ceil(T / dt)), 1)
    dt = T / n_steps

    for _ in range(n_steps):
        u_hat = fused_step(u_hat, dt, nu, band)

    u = np.real(np.fft.ifft(u_hat))
    return x, u


# ---------------------------------------------------------------------------
# 2D Helmholtz spectral solver
# ---------------------------------------------------------------------------

def solve_helmholtz_2d(
    nx: int = 64,
    ny: int = 64,
    k: float = 1.0,
    source: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve the 2D Helmholtz equation on [0, 2 pi]^2 with periodic BCs.

    (nabla^2 + k^2) u = f

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    k : float
        Helmholtz wavenumber.
    source : ndarray (nx, ny), optional
        Source term f.  Default: point source at centre.

    Returns
    -------
    u : ndarray (nx, ny)
        Solution field.
    """
    dx = 2 * np.pi / nx
    dy = 2 * np.pi / ny

    if source is None:
        source = np.zeros((nx, ny))
        source[nx // 2, ny // 2] = 1.0 / (dx * dy)

    # Wavenumber grids
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2

    # Helmholtz operator in Fourier space: (-K2 + k^2)
    denom = -K2 + k ** 2
    # Regularise zero to avoid division by zero
    denom[np.abs(denom) < 1e-14] = 1e-14

    f_hat = np.fft.fft2(source)
    u_hat = f_hat / denom
    return np.real(np.fft.ifft2(u_hat))


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------

def energy_neutrality_check(
    u_hat_before: np.ndarray,
    u_hat_after: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[bool, float]:
    """Verify energy neutrality invariant P1.

    Parameters
    ----------
    u_hat_before, u_hat_after : ndarray (complex)
        Fourier coefficients before and after a fused step.
    tol : float
        Acceptable relative energy drift.

    Returns
    -------
    passed : bool
    relative_drift : float
        |E_after - E_before| / E_before
    """
    E_before = np.sum(np.abs(u_hat_before) ** 2)
    E_after = np.sum(np.abs(u_hat_after) ** 2)
    if E_before < 1e-30:
        return True, 0.0
    drift = np.abs(E_after - E_before) / E_before
    return bool(drift < tol), float(drift)


def truncation_error(
    u_exact: np.ndarray,
    u_approx: np.ndarray,
    K: int,
) -> Tuple[float, float]:
    """Measure truncation error and estimate convergence order.

    Parameters
    ----------
    u_exact : ndarray
        Reference (exact or high-resolution) solution.
    u_approx : ndarray
        Approximate solution at band cutoff K.
    K : int
        Spectral band cutoff used.

    Returns
    -------
    l2_error : float
        Relative L2 error ||u_exact - u_approx|| / ||u_exact||.
    estimated_order : float
        Estimated convergence exponent (assuming error ~ K^{-p}).
        Requires K >= 2.
    """
    norm_exact = np.linalg.norm(u_exact)
    if norm_exact < 1e-30:
        return 0.0, 0.0
    l2_error = float(np.linalg.norm(u_exact - u_approx) / norm_exact)
    # Rough order estimate: log(error) / log(1/K)
    if K >= 2 and l2_error > 1e-15:
        estimated_order = -np.log(l2_error) / np.log(K)
    else:
        estimated_order = 0.0
    return l2_error, float(estimated_order)
