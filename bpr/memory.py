"""
Theory I: Boundary Memory Dynamics
===================================

Formalises non-Markovian temporal correlations in BPR via a prime-periodic
memory kernel.  The standard BPR equation is extended from instantaneous to
history-dependent coupling:

    κ ∇²_B φ(x,t) = ∫ M(t,t') [Σ_i χ_i(t') δ(x-x_i)] dt' + η(x,t)

Key objects
----------
* ``MemoryKernel``  – exponential-oscillatory kernel M(t,t')
* ``BoundaryMemoryField`` – solver for the generalised BPR equation with memory
* ``TemporalCorrelation`` – two-time correlation function C_φ(t,t')

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Physical constants (natural units unless stated otherwise)
# ---------------------------------------------------------------------------
_HBAR = 1.054571817e-34  # J·s
_C = 299792458.0          # m/s
_K_B = 1.380649e-23       # J/K
_PLANCK_TIME = 5.391247e-44  # s


# ---------------------------------------------------------------------------
# §3.2  Memory Kernel  M(t,t')
# ---------------------------------------------------------------------------

@dataclass
class MemoryKernelParams:
    """Parameters defining the boundary memory kernel.

    Attributes
    ----------
    tau_m : float
        Memory timescale (s).
    omega_r : float
        Resonance frequency (rad/s).  Quantised as 2πn/p.
    p : int
        Substrate prime modulus.
    n : int
        Mode number (dominant mode n=1).
    W : float
        Topological winding number.
    alpha : float
        Topological protection exponent (≥ 1).
    tau_0 : float
        Bare memory time (s), before topological enhancement.
    """
    tau_m: float = 1.0
    omega_r: float = 1.0
    p: int = 7
    n: int = 1
    W: float = 0.0
    alpha: float = 1.0
    tau_0: float = 1.0

    def __post_init__(self):
        # Property 2: quantised resonance frequency
        self.omega_r = 2.0 * np.pi * self.n / self.p
        # Property 3: topologically-protected memory timescale
        if abs(self.W) > 0:
            self.tau_m = self.tau_0 * abs(self.W) ** self.alpha
        else:
            self.tau_m = self.tau_0


def memory_kernel(t: np.ndarray, t_prime: np.ndarray,
                  params: Optional[MemoryKernelParams] = None) -> np.ndarray:
    """Evaluate the boundary memory kernel M(t,t').

    Equation (Property 1, §3.2):
        M(t,t') = exp(-|t-t'|/τ_m) cos(ω_r (t-t'))

    Parameters
    ----------
    t, t_prime : array_like
        Time arguments (broadcastable).
    params : MemoryKernelParams, optional
        Kernel parameters; defaults constructed if *None*.

    Returns
    -------
    M : ndarray
        Kernel values, same shape as broadcast of *t* and *t_prime*.
    """
    if params is None:
        params = MemoryKernelParams()
    dt = np.asarray(t) - np.asarray(t_prime)
    return np.exp(-np.abs(dt) / params.tau_m) * np.cos(params.omega_r * dt)


def memory_kernel_multimode(t: np.ndarray, t_prime: np.ndarray,
                            p: int = 7,
                            n_modes: int = 4,
                            amplitudes: Optional[Sequence[float]] = None,
                            damping_rates: Optional[Sequence[float]] = None,
                            ) -> np.ndarray:
    """Multi-mode memory kernel (§3.3, full derivation).

    M(t,t') = Σ_n a_n exp(-γ_n |t-t'|) cos(2πn(t-t')/p)

    Parameters
    ----------
    p : int
        Substrate prime modulus.
    n_modes : int
        Number of harmonic modes to sum.
    amplitudes : sequence of float, optional
        Mode amplitudes a_n.  Default: 1/n² falloff.
    damping_rates : sequence of float, optional
        Coarse-graining damping rates γ_n.  Default: n-linear.
    """
    if amplitudes is None:
        amplitudes = [1.0 / (n ** 2) for n in range(1, n_modes + 1)]
    if damping_rates is None:
        damping_rates = [0.1 * n for n in range(1, n_modes + 1)]

    dt = np.asarray(t) - np.asarray(t_prime)
    M = np.zeros_like(dt, dtype=float)
    for k in range(n_modes):
        n = k + 1
        omega_n = 2.0 * np.pi * n / p
        M += amplitudes[k] * np.exp(-damping_rates[k] * np.abs(dt)) * np.cos(omega_n * dt)
    return M


# ---------------------------------------------------------------------------
# §3.3  Temporal auto-correlation  C_φ(t,t')
# ---------------------------------------------------------------------------

@dataclass
class TemporalCorrelation:
    """Two-time correlation of coarse-grained boundary observables.

    C_φ(t,t') = <φ(x,t)φ(x,t')> - <φ(x,t)><φ(x,t')>

    For the prime-periodic substrate the correlation inherits oscillatory
    recurrences at τ_p = p · τ_Planck.
    """
    p: int = 7
    sigma_phi: float = 1.0

    def __call__(self, t: np.ndarray, t_prime: np.ndarray) -> np.ndarray:
        """Evaluate correlation function."""
        dt = np.asarray(t) - np.asarray(t_prime)
        tau_p = self.p * _PLANCK_TIME
        # Connected correlator with prime-periodic recurrence
        return self.sigma_phi ** 2 * np.exp(-np.abs(dt) / tau_p) * np.cos(
            2.0 * np.pi * dt / tau_p
        )


# ---------------------------------------------------------------------------
# §3.4  Non-Markovian quantum error correlations
# ---------------------------------------------------------------------------

def non_markovian_error_correlation(tau: np.ndarray,
                                    params: Optional[MemoryKernelParams] = None,
                                    gamma_markov: float = 1.0) -> np.ndarray:
    """Predicted temporal autocorrelation of quantum error rates (P1.1 / P1.4).

    Standard Markov model:  C_err(τ) = exp(-γ τ)
    BPR prediction:         C_err(τ) = M(τ, 0)  (sign-changing, oscillatory)

    Returns
    -------
    C_bpr, C_markov : tuple of ndarray
        BPR oscillatory and standard Markov correlations.
    """
    tau = np.asarray(tau)
    C_markov = np.exp(-gamma_markov * tau)
    C_bpr = memory_kernel(tau, np.zeros_like(tau), params)
    return C_bpr, C_markov


# ---------------------------------------------------------------------------
# §3.4.3  Consciousness temporal integration
# ---------------------------------------------------------------------------

def consciousness_memory_timescale(W: float, tau_0: float = 1.0,
                                    alpha: float = 1.0) -> float:
    """Topologically-protected memory timescale (Property 3).

    τ_m = τ_0 |W|^α    for W ≠ 0
    τ_m = τ_0           for W = 0

    Systems with W ≠ 0 (conscious) possess divergent memory timescales,
    enabling coherent temporal narratives (the "specious present").
    """
    if abs(W) > 0:
        return tau_0 * abs(W) ** alpha
    return tau_0


# ---------------------------------------------------------------------------
# Generalised BPR equation with memory  (discretised solver)
# ---------------------------------------------------------------------------

@dataclass
class BoundaryMemoryField:
    """Discretised solver for the BPR equation with memory kernel.

    κ ∇²_B φ(x,t) = ∫ M(t,t') S(x,t') dt' + η(x,t)

    Uses a simple Euler-in-time, spectral-in-space scheme for demonstration.
    """
    kappa: float = 1.0
    kernel_params: MemoryKernelParams = field(default_factory=MemoryKernelParams)
    dt: float = 0.01
    n_spatial: int = 64

    def solve(self, source: np.ndarray, n_steps: int = 200,
              noise_amplitude: float = 0.0) -> np.ndarray:
        """Integrate the memory-extended BPR equation on a 1-D ring.

        Parameters
        ----------
        source : ndarray, shape (n_spatial, n_steps)
            Source function S(x, t) sampled at spatial nodes and time steps.
        n_steps : int
            Number of time steps (must match source.shape[1]).
        noise_amplitude : float
            Amplitude of stochastic η(x,t) term.

        Returns
        -------
        phi : ndarray, shape (n_spatial, n_steps)
            Boundary phase field history.
        """
        N = self.n_spatial
        phi = np.zeros((N, n_steps))
        times = np.arange(n_steps) * self.dt

        # Spectral Laplacian eigenvalues on a ring of length 2π
        k = np.fft.fftfreq(N, d=1.0 / N)
        lap_eigenvalues = -(2.0 * np.pi * k) ** 2

        for step in range(1, n_steps):
            # Memory convolution: ∫ M(t, t') S(x, t') dt'
            conv = np.zeros(N)
            for s in range(step):
                M_val = memory_kernel(
                    np.array([times[step]]),
                    np.array([times[s]]),
                    self.kernel_params,
                )
                conv += M_val.item() * source[:, s] * self.dt

            # Noise term
            eta = noise_amplitude * np.random.randn(N) if noise_amplitude > 0 else 0.0

            # Spectral step: κ λ_k φ_k = (conv + η)_k  →  φ_k = RHS_k / (κ λ_k)
            rhs = conv + eta
            rhs_hat = np.fft.fft(rhs)
            phi_hat = np.zeros(N, dtype=complex)
            for i in range(N):
                if abs(lap_eigenvalues[i]) > 1e-12:
                    phi_hat[i] = rhs_hat[i] / (self.kappa * lap_eigenvalues[i])
            phi[:, step] = np.real(np.fft.ifft(phi_hat))

        return phi


# ---------------------------------------------------------------------------
# Prime-harmonic frequency spectrum  (§3.3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §3.5  Quantum error correlation periodicity  (Prediction 16)
# ---------------------------------------------------------------------------

def quantum_error_period(p: int) -> float:
    """Period of oscillatory quantum error correlations.

    T_error = p × τ_Planck

    On a superconducting qubit chip, error rates should show temporal
    correlations with this period.  For p ~ 10⁵:  T ~ 5 × 10⁻³⁹ s.
    While individual oscillations are unresolvable, the *envelope*
    modulates decoherence at T_envelope = T_error × p ≈ p² τ_Pl.

    Returns
    -------
    float – oscillation period (s)
    """
    return p * _PLANCK_TIME


def quantum_error_envelope_period(p: int) -> float:
    """Envelope period of quantum error modulation.

    T_envelope = p² × τ_Planck

    This longer timescale may be detectable in qubit calibration drift.
    For p = 104729:  T_envelope ≈ 5.9 × 10⁻³⁴ s (still extremely short,
    but integrated effects accumulate in repeated measurements).
    """
    return float(p) ** 2 * _PLANCK_TIME


# ---------------------------------------------------------------------------
# §3.6  Casimir fine structure from memory kernel  (Prediction 9)
# ---------------------------------------------------------------------------

def casimir_fine_structure(distances: np.ndarray, p: int = 104729,
                            n_modes: int = 5,
                            base_amplitude: float = 1e-8) -> np.ndarray:
    """Oscillatory fine structure on the Casimir deviation curve.

    The Casimir deviation δF(d) predicted by Eq 7 is modulated by
    prime harmonics from the memory kernel:

        δF_fine(d) = δF₀ × Σ_n (1/n²) cos(2πn d / (p × l_Pl))

    This predicts that if you measure the Casimir force deviation
    with sufficient precision, the deviation itself wiggles.

    Parameters
    ----------
    distances : ndarray – plate separations (m)
    p : int – substrate prime
    n_modes : int – number of harmonic modes
    base_amplitude : float – relative amplitude of fine structure

    Returns
    -------
    ndarray – fractional modulation δF_fine / F_casimir
    """
    l_planck = 1.616255e-35  # m
    d = np.asarray(distances, dtype=float)
    modulation = np.zeros_like(d)
    for n in range(1, n_modes + 1):
        k_n = 2.0 * np.pi * n / (p * l_planck)
        modulation += (1.0 / n ** 2) * np.cos(k_n * d)
    return base_amplitude * modulation


def prime_harmonic_frequencies(p: int, n_max: int = 10) -> np.ndarray:
    """Return the first *n_max* prime-harmonic resonance frequencies.

    ω_n = 2πn / p   for n = 1, …, n_max
    """
    return 2.0 * np.pi * np.arange(1, n_max + 1) / p
