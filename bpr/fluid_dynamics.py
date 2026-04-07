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


# ---------------------------------------------------------------------------
# 2-D turbulence  — Kraichnan-Batchelor-Leith cascade predictions
# ---------------------------------------------------------------------------

@dataclass
class TwoDTurbulence:
    """BPR predictions for 2-D turbulence energy spectra.

    In d = 2, substrate topology produces two distinct spectral cascades
    (Kraichnan-Batchelor-Leith theory, CONSISTENT with BPR Class B, d=2):

    1. **Enstrophy cascade** (k > k_f, forward):
           E(k) ∝ k^{−3}
       Enstrophy (Z = ∫ω² dx) cascades to small scales; energy is blocked.

    2. **Inverse energy cascade** (k < k_f, backward):
           E(k) ∝ k^{−5/3}
       Energy cascades to large scales (condensation in finite domains).

    BPR interpretation: the d=2 substrate lattice has β=(d-2)/(d+2)=0
    for Class B transitions, meaning the order-parameter boundary at
    k_f is degenerate — energy cannot escape to smaller scales, forcing
    the inverse cascade.

    These are the same exponents as Kolmogorov 3-D (−5/3) and as
    Kraichnan's analytical result (−3); BPR Class B in d=2 is CONSISTENT
    with both.
    """
    d: int = 2

    @property
    def enstrophy_cascade_exponent(self) -> float:
        """Spectral index in enstrophy cascade range: E(k) ∝ k^{−3}."""
        return -3.0

    @property
    def inverse_energy_cascade_exponent(self) -> float:
        """Spectral index in inverse energy cascade range: E(k) ∝ k^{−5/3}."""
        return -5.0 / 3.0

    def kolmogorov_wavenumber(self, epsilon: float, nu: float) -> float:
        """Kolmogorov dissipation scale k_d = (ε/ν³)^{1/4}."""
        if nu <= 0 or epsilon <= 0:
            return float("nan")
        return (epsilon / nu ** 3) ** 0.25

    def enstrophy_dissipation_wavenumber(self, eta: float, nu: float) -> float:
        """Enstrophy dissipation scale k_η = (η/ν³)^{1/6}.

        η = enstrophy dissipation rate (s^{-3}).
        """
        if nu <= 0 or eta <= 0:
            return float("nan")
        return (eta / nu ** 3) ** (1.0 / 6.0)

    @staticmethod
    def radial_spectrum_2d(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Radially-averaged 2-D kinetic energy spectrum E(k).

        Parameters
        ----------
        field_2d : (ny, nx) velocity component (or vorticity)

        Returns
        -------
        (k_bins, E_k) — wavenumber bins and mean spectral power per bin
        """
        ny, nx = field_2d.shape
        fft2 = np.fft.fft2(field_2d)
        power = np.abs(fft2) ** 2 / (nx * ny)
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)
        k_max = int(min(nx, ny) / 2)
        k_bins = np.arange(1, k_max, dtype=float)
        E_k = np.zeros(len(k_bins))
        for i, kb in enumerate(k_bins):
            mask = (K >= kb - 0.5) & (K < kb + 0.5)
            if mask.any():
                E_k[i] = power[mask].mean()
        return k_bins, E_k

    def fit_cascade_exponent(self, k_bins: np.ndarray, E_k: np.ndarray,
                              k_lo_frac: float = 0.15,
                              k_hi_frac: float = 0.45) -> float:
        """Fit E(k) ∝ k^α in the inertial range.

        Uses the middle fraction of the spectrum to avoid forcing and
        dissipation ranges.
        """
        k_lo = k_bins.max() * k_lo_frac
        k_hi = k_bins.max() * k_hi_frac
        mask = (k_bins >= k_lo) & (k_bins <= k_hi) & (E_k > 0)
        if mask.sum() < 3:
            return float("nan")
        p = np.polyfit(np.log(k_bins[mask]), np.log(E_k[mask]), 1)
        return float(p[0])


# ---------------------------------------------------------------------------
# Stratified fluids — Brunt-Väisälä and internal gravity waves
# ---------------------------------------------------------------------------

@dataclass
class StratifiedFluid:
    """BPR framework for density-stratified fluid dynamics.

    Stable density stratification introduces a characteristic frequency
    N (Brunt-Väisälä / buoyancy frequency) that controls:
    - Maximum frequency of internal gravity waves
    - Suppression of vertical turbulent mixing
    - Transition from isotropic to anisotropic (layered) turbulence

    BPR maps stratification to a Class C impedance transition:
    - For Fr > 1 (inertia dominant): isotropic 3-D turbulence, E(k)∝k^{-5/3}
    - For Fr < 1 (stratification dominant): layered 2-D turbulence,
      E(k)∝k^{-3} in horizontal; energy in internal gravity wave band
    - Transition at Fr_c = 1  (Froude number as BPR impedance ratio)

    The Ozmidov scale L_O = √(ε/N³) is the scale at which stratification
    and inertia are equal — BPR's boundary resonance condition.

    References: Al-Kahwati (2026), §6 Class C transitions; Lilly (1983)
    """
    N_buoyancy: float = 0.01    # Brunt-Väisälä frequency (rad/s)
    g: float = 9.81             # gravitational acceleration (m/s²)

    def gravity_wave_frequency(self, k_h: float, k_total: float) -> float:
        """Internal gravity wave frequency: ω = N k_h / k.

        Valid for k_total > 0 and k_h ≤ k_total.
        """
        if k_total <= 0:
            return 0.0
        return self.N_buoyancy * k_h / k_total

    def ozmidov_scale(self, epsilon: float) -> float:
        """Ozmidov scale L_O = (ε/N³)^{1/2}.

        At scales L ≫ L_O: stratification dominates (anisotropic, layered).
        At scales L ≪ L_O: inertia dominates (isotropic Kolmogorov cascade).
        """
        if self.N_buoyancy <= 0 or epsilon <= 0:
            return float("nan")
        return (epsilon / self.N_buoyancy ** 3) ** 0.5

    def froude_number(self, U: float, L: float) -> float:
        """Froude number Fr = U / (N·L).

        Fr > 1: weakly stratified (turbulence dominates).
        Fr < 1: strongly stratified (wave regime).
        Fr ~ 1: BPR Class C critical point (Ozmidov scale).
        """
        if self.N_buoyancy <= 0 or L <= 0:
            return float("inf")
        return U / (self.N_buoyancy * L)

    def buoyancy_reynolds_number(self, epsilon: float, nu: float) -> float:
        """Buoyancy Reynolds Re_b = ε / (N² · ν).

        Re_b ≫ 1: turbulence unaffected by stratification.
        Re_b ~ 1: BPR impedance transition; turbulence collapses to waves.
        """
        if self.N_buoyancy <= 0 or nu <= 0:
            return float("inf")
        return epsilon / (self.N_buoyancy ** 2 * nu)

    def brunt_vaisala_from_density(self, rho: np.ndarray,
                                    dz: float = 1.0) -> float:
        """Estimate N² from a vertical density profile.

        N² = −(g/ρ₀)(∂ρ/∂z)  — positive in stable stratification.

        Parameters
        ----------
        rho : 1-D array of densities along vertical axis (increasing z).
        dz  : grid spacing in z.
        """
        rho = np.asarray(rho, dtype=float)
        rho_0 = float(np.mean(rho))
        drho_dz = np.gradient(rho, dz)
        N2 = float(np.mean(-self.g / rho_0 * drho_dz))
        return N2

    @staticmethod
    def wave_dispersion_residual(k_h: np.ndarray, k_z: np.ndarray,
                                  omega: np.ndarray,
                                  N: float) -> np.ndarray:
        """Residuals of the IGW dispersion relation ω² = N² k_h² / k².

        Returns |ω² − N² k_h² / k²| / ω² for each mode.
        """
        k_total = np.sqrt(k_h ** 2 + k_z ** 2)
        k_total = np.where(k_total > 0, k_total, 1.0)
        omega_pred_sq = N ** 2 * k_h ** 2 / k_total ** 2
        omega_sq = omega ** 2
        return np.abs(omega_sq - omega_pred_sq) / np.where(omega_sq > 0, omega_sq, 1.0)


# ---------------------------------------------------------------------------
# Rayleigh-Taylor instability
# ---------------------------------------------------------------------------

@dataclass
class RayleighTaylorInstability:
    """BPR framework for Rayleigh-Taylor instability.

    When a heavy fluid sits on top of a light fluid under gravity,
    the interface is unstable. BPR maps this to a Class D (symmetry-breaking)
    boundary frustration transition.

    Key predictions:
    - Linear growth rate: sigma(k) = sqrt(A*g*k)  (Atwood number A, gravity g, wavenumber k)
    - Nonlinear mixing width: h(t) = alpha*A*g*t**2  (self-similar, alpha ~ 0.04-0.07)
    - Energy spectrum in mixing zone: E(k) proportional to k^{-5/3}  (Kolmogorov cascade)
    """
    A: float = 0.5     # Atwood number (rho2-rho1)/(rho2+rho1)
    g: float = 9.81     # gravitational acceleration

    def linear_growth_rate(self, k: float) -> float:
        """sigma = sqrt(A*g*k) for wavenumber k."""
        return np.sqrt(self.A * self.g * k)

    def mixing_width(self, t: float, alpha: float = 0.05) -> float:
        """h(t) = alpha*A*g*t**2 (self-similar mixing zone growth)."""
        return alpha * self.A * self.g * t**2

    @staticmethod
    def atwood_number(rho_heavy: float, rho_light: float) -> float:
        return (rho_heavy - rho_light) / (rho_heavy + rho_light)


# ---------------------------------------------------------------------------
# Sedov-Taylor blast wave
# ---------------------------------------------------------------------------

@dataclass
class SedovTaylorBlast:
    """BPR framework for Sedov-Taylor self-similar blast waves.

    A strong explosion in a uniform medium produces a self-similar
    blast wave with radius R(t) proportional to (E/rho0)^{1/5} * t^{2/5}.

    BPR interpretation: the blast front is a Class A (winding) transition
    where the shock boundary phase jumps by pi. The self-similar exponent
    2/5 is exact from dimensional analysis in d=3.
    """
    E: float = 1.0       # explosion energy
    rho_0: float = 1.0   # ambient density
    d: int = 3            # spatial dimension

    def blast_radius(self, t: float) -> float:
        """R(t) = xi0 * (E/rho0)^{1/(d+2)} * t^{2/(d+2)}."""
        xi_0 = 1.15   # Sedov constant for d=3, gamma=5/3
        exponent = 2.0 / (self.d + 2)
        return xi_0 * (self.E / self.rho_0) ** (1.0/(self.d+2)) * t ** exponent

    @property
    def time_exponent(self) -> float:
        """Self-similar exponent: R proportional to t^n where n = 2/(d+2)."""
        return 2.0 / (self.d + 2)

    @property
    def energy_spectrum_exponent(self) -> float:
        """Post-shock turbulence spectrum E(k) proportional to k^alpha. BPR: alpha = -5/3."""
        return -5.0 / 3.0


# ---------------------------------------------------------------------------
# Shear instability (Kelvin-Helmholtz)
# ---------------------------------------------------------------------------

@dataclass
class ShearInstability:
    """BPR framework for shear-driven instability (Kelvin-Helmholtz).

    BPR maps shear instability to a Class D (symmetry-breaking) transition:
    the shear layer breaks translational symmetry along the flow direction.

    In 2D turbulent shear at high Re, the energy spectrum follows the
    enstrophy cascade E(k) proportional to k^{-3} (2D substrate topology, d=2).
    """
    Re: float = 10000.0    # Reynolds number

    @property
    def turbulent_spectrum_exponent(self) -> float:
        """2D shear: enstrophy cascade E(k) proportional to k^{-3}."""
        return -3.0

    def richardson_critical(self) -> float:
        """Critical Richardson number Ri_c = 1/4 for stratified shear."""
        return 0.25


# ---------------------------------------------------------------------------
# Geostrophic turbulence (planetary shallow water)
# ---------------------------------------------------------------------------

@dataclass
class GeostrophicTurbulence:
    """BPR framework for quasi-geostrophic turbulence on a rotating sphere.

    Charney (1971): on a rotating planet, large-scale atmospheric/oceanic
    turbulence follows quasi-2D dynamics with E(k) proportional to k^{-3}
    (enstrophy cascade) for k > k_Rossby.

    BPR maps this to 2D substrate topology (d=2) with the Rossby deformation
    radius as the characteristic boundary scale.
    """

    @property
    def spectrum_exponent(self) -> float:
        return -3.0
