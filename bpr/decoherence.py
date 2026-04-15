"""
Boundary-Induced Decoherence
=================================================

Derives explicit decoherence rates from impedance mismatch between
system and environment, and a geometric pointer-basis selection rule.

Key objects
-----------
* ``DecoherenceRate``     – Γ_dec from impedance mismatch (Theorem, §5.2)
* ``PointerBasis``        – selection from boundary coupling operator (§5.3)
* ``QuantumClassicalBoundary`` – critical winding W_crit (§5.4)

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §5
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_Z0 = 376.730313668        # vacuum impedance (Ω)


# ---------------------------------------------------------------------------
# §5.2  Decoherence rate from impedance mismatch
# ---------------------------------------------------------------------------

@dataclass
class DecoherenceRate:
    """Boundary decoherence rate (Theorem, §5.2).

    Γ_dec = (k_B T / ℏ) (ΔZ / Z₀)² (A_eff / λ_dB²)

    Parameters
    ----------
    T : float
        Temperature (K).
    Z_system : float
        Boundary impedance of the quantum system (Ω).
    Z_environment : float
        Boundary impedance of the environment (Ω).
    A_eff : float
        Effective boundary surface area (m²).
    lambda_dB : float
        Thermal de Broglie wavelength (m).
    """
    T: float = 300.0
    Z_system: float = _Z0
    Z_environment: float = _Z0 * 1.01
    A_eff: float = 1e-14          # ~ nm² scale
    lambda_dB: float = 1e-10      # ~ Å

    @property
    def delta_Z(self) -> float:
        return abs(self.Z_system - self.Z_environment)

    @property
    def gamma_dec(self) -> float:
        """Decoherence rate (s⁻¹)."""
        return (_K_B * self.T / _HBAR) * (self.delta_Z / _Z0) ** 2 * (
            self.A_eff / self.lambda_dB ** 2
        )

    @property
    def decoherence_time(self) -> float:
        """Decoherence time τ_dec = 1 / Γ_dec (s)."""
        g = self.gamma_dec
        return 1.0 / g if g > 0 else np.inf


def decoherence_rate(T: float, delta_Z: float, A_eff: float,
                     lambda_dB: float) -> float:
    """Functional form of the boundary decoherence rate.

    Γ_dec = (k_B T / ℏ) (ΔZ / Z₀)² (A_eff / λ_dB²)
    """
    return (_K_B * T / _HBAR) * (delta_Z / _Z0) ** 2 * (A_eff / lambda_dB ** 2)


# ---------------------------------------------------------------------------
# §5.3  Pointer basis selection
# ---------------------------------------------------------------------------

@dataclass
class PointerBasis:
    """Pointer basis from boundary coupling operator B(φ).

    The pointer states |π_k⟩ are eigenstates of B(φ) = κ ∇²_B φ
    restricted to the system–environment interface.

    For demonstration we model B as a real symmetric matrix and
    diagonalise it.
    """
    B_matrix: Optional[np.ndarray] = None  # coupling operator (N×N)

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalise the boundary coupling operator.

        Returns
        -------
        eigenvalues : ndarray
            Pointer basis eigenvalues b_k.
        eigenvectors : ndarray
            Pointer states (columns).
        """
        if self.B_matrix is None:
            raise ValueError("Boundary coupling matrix B not set.")
        eigenvalues, eigenvectors = np.linalg.eigh(self.B_matrix)
        return eigenvalues, eigenvectors

    @staticmethod
    def from_boundary_geometry(kappa: float, phi_gradient: np.ndarray,
                               n_points: int = 32) -> "PointerBasis":
        """Construct B(φ) from boundary stiffness and phase gradient field.

        B_ij = κ (∂_i φ · ∂_j φ)  restricted to the S–E interface.
        """
        # Outer product of gradient at interface points → coupling matrix
        grad = np.asarray(phi_gradient)
        if grad.ndim == 1:
            grad = grad[:, None]
        B = kappa * (grad @ grad.T)
        return PointerBasis(B_matrix=B)


# ---------------------------------------------------------------------------
# §5.4  Quantum–classical boundary
# ---------------------------------------------------------------------------

def critical_winding(gamma_dec: float, omega_system: float) -> float:
    """Critical winding number for the quantum–classical transition.

    W_crit = √(Γ_dec / ω_system)

    Systems with |W| > W_crit maintain quantum coherence.
    Systems with |W| < W_crit behave classically.
    """
    if omega_system <= 0:
        return np.inf
    return np.sqrt(gamma_dec / omega_system)


def is_quantum(W: float, gamma_dec: float, omega_system: float) -> bool:
    """Return True if the system is in the quantum regime (|W| > W_crit)."""
    return bool(abs(W) > critical_winding(gamma_dec, omega_system))


# ---------------------------------------------------------------------------
# §5.5  Decoherence-free subspaces (P3.3)
# ---------------------------------------------------------------------------

def decoherence_free_modes(B_matrix: np.ndarray,
                           tol: float = 1e-10) -> np.ndarray:
    """Identify decoherence-free subspaces (zero-reflection modes).

    These are eigenstates of B(φ) with eigenvalue ≈ 0 (zero impedance
    reflection), which are immune to environmental decoherence.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(B_matrix)
    mask = np.abs(eigenvalues) < tol
    return eigenvectors[:, mask]


# ---------------------------------------------------------------------------
# Cryogenic deviation from linear-in-T  (P3.4)
# ---------------------------------------------------------------------------

def decoherence_rate_with_quantum_correction(T: np.ndarray, delta_Z: float,
                                              A_eff: float, lambda_dB: float,
                                              T_quantum: float = 1.0) -> np.ndarray:
    """Decoherence rate with quantum impedance fluctuation correction.

    At low T the rate deviates from linear-in-T:
        Γ(T) = Γ_classical(T) × [1 + (T_quantum/T)²]^{-1/2}
    """
    T = np.asarray(T, dtype=float)
    gamma_classical = (_K_B * T / _HBAR) * (delta_Z / _Z0) ** 2 * (A_eff / lambda_dB ** 2)
    quantum_suppression = 1.0 / np.sqrt(1.0 + (T_quantum / (T + 1e-30)) ** 2)
    return gamma_classical * quantum_suppression


# ---------------------------------------------------------------------------
# §5.6  Impedance-driven Landau crossover
# ---------------------------------------------------------------------------

def impedance_to_landau_coefficient(delta_Z, Z_0, T, T_c, alpha_0=1.0, alpha_eta=0.1):
    """a(T,L,G,η) = α₀·[T/T_c - 1] + α_η·η where η = (ΔZ/Z₀)²
    Maps impedance mismatch to Landau coefficient for quantum-classical crossover.
    When a < 0: quantum regime. When a > 0: classical regime."""
    eta = (delta_Z / Z_0)**2
    return alpha_0 * (T / T_c - 1) + alpha_eta * eta

def decoherence_rate_near_crossover(gamma_dec_base, T, T_c, Delta_T):
    """Γ_dec(T) = γ_base / cosh²((T-T_c)/ΔT)
    Decoherence rate PEAKS at T_c (susceptibility maximum) rather than
    growing monotonically. This is a key falsifiable prediction."""
    return gamma_dec_base / np.cosh((T - T_c) / Delta_T)**2

def crossover_susceptibility(T, T_c, Delta_T, chi_max=1.0):
    """χ(T) = χ_max / cosh²((T-T_c)/ΔT) — peaked susceptibility at crossover.
    Maximum response to perturbation occurs exactly at T_c."""
    return chi_max / np.cosh((T - T_c) / Delta_T)**2

def impedance_driven_crossover(Z_system, Z_environment, T, L, N_eff, m,
                                 hbar=1.055e-34, k_B=1.38e-23,
                                 alpha_0=1.0, alpha_eta=0.1):
    """Full impedance → Landau crossover pipeline.

    Takes impedance values and returns Landau coefficients and crossover observables.
    Bridges impedance-based decoherence to Landau order parameter framework.

    Returns dict with: T_c, a_landau, eta, Delta_T, regime ('quantum'/'classical')"""
    Z_0 = max(abs(Z_system), 1e-15)
    delta_Z = abs(Z_system - Z_environment)
    T_c = hbar**2 / (2 * np.pi * m * k_B) * (N_eff / L)**2
    eta = (delta_Z / Z_0)**2
    a = alpha_0 * (T / T_c - 1) + alpha_eta * eta
    Delta_T = T_c * 0.1 * (1 + eta)  # broadened by impedance mismatch
    regime = "quantum" if a < 0 else "classical"
    return {
        "T_c": T_c, "a_landau": a, "eta": eta,
        "Delta_T": Delta_T, "regime": regime,
        "delta_Z": delta_Z, "Z_0": Z_0,
    }

def decoherence_with_consciousness(gamma_dec, chi_coupling, Phi_measure, Phi_crit,
                                     T=None, T_c=None):
    """Γ_eff = γ_dec · [1 - χ·Φ/Φ_crit]
    Consciousness (integrated information Φ) weakly modulates decoherence rate.
    Near T_c, the effect is amplified by susceptibility.

    Prediction: visibility shift δV = χ·Φ·dV/dT under cognitive tasks."""
    modulation = 1 - chi_coupling * Phi_measure / Phi_crit
    modulation = max(modulation, 0.0)  # can't go negative
    gamma_eff = gamma_dec * modulation
    # Susceptibility amplification near T_c
    if T is not None and T_c is not None:
        Delta_T = T_c * 0.1
        susceptibility = 1.0 / np.cosh((T - T_c) / Delta_T)**2
        delta_visibility = chi_coupling * Phi_measure * susceptibility
    else:
        delta_visibility = chi_coupling * Phi_measure
    return {"gamma_eff": gamma_eff, "modulation_factor": modulation,
            "delta_visibility": delta_visibility}

def born_rule_from_substrate(p, N_modes=10, n_samples=10000, seed=None):
    """Derive Born rule from RPST microstate counting.

    On Z_p substrate, coarse-grain over microstates to obtain
    emergent probability P(α,x) = |ψ_α(x)|²/Σ|ψ(x')|².

    Deviation from standard Born rule scales as 1/p.
    For p=104761: deviation ~ 10⁻⁵.

    Returns dict with probabilities, born_correction, deviation."""
    rng = np.random.default_rng(seed)
    # Sample microstates on Z_p
    phases = rng.uniform(0, 2*np.pi, (n_samples, N_modes))
    # Coarse-grained amplitude: sum of phase factors
    psi = np.sum(np.exp(1j * phases), axis=1) / np.sqrt(N_modes)
    # Standard Born probabilities
    P_born = np.abs(psi)**2 / np.sum(np.abs(psi)**2)
    # BPR correction: substrate granularity introduces 1/p correction
    correction = 1.0 / p
    P_bpr = P_born * (1 + correction * np.cos(np.angle(psi)))
    P_bpr = np.abs(P_bpr) / np.sum(np.abs(P_bpr))
    deviation = np.mean(np.abs(P_bpr - P_born))
    return {"P_born": P_born, "P_bpr": P_bpr,
            "born_correction": correction,
            "mean_deviation": deviation,
            "p": p, "expected_deviation": 1.0/p}
