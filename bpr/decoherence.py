"""
Theory III: Boundary-Induced Decoherence Theory
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
