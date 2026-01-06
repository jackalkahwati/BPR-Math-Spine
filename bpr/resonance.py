"""
BPR Resonance Utilities (Eq 4b scaffolding)

This module provides:
- Riemann zero table loading (optionally via mpmath if installed)
- Pair correlation / GUE comparisons
- Helpers to derive a "fractal exponent" proxy from zero spacings

Design goal: keep the math spine runnable with default deps (numpy/scipy/sympy).
For large tables of zeros, mpmath is optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# A small, verified seed list of nontrivial zeta zeros (imaginary parts).
# For more zeros, install mpmath and use load_riemann_zeros(n>len(RIEMANN_ZEROS)).
RIEMANN_ZEROS = np.array(
    [
        14.134725141734693,
        21.022039638771554,
        25.010857580145688,
        30.424876125859513,
        32.93506158773919,
        37.58617815882567,
        40.918719012147495,
        43.327073280914996,
        48.00515088116716,
        49.7738324776723,
        52.97032147771446,
        56.446247697063394,
        59.34704400260235,
        60.83177852460981,
        65.11254404808161,
        67.07981052949417,
        69.54640171117398,
        72.0671576744819,
        75.70469069908393,
        77.1448400688748,
    ],
    dtype=float,
)


def load_riemann_zeros(n: int = 20) -> np.ndarray:
    """
    Load first n Riemann zeta zeros (imaginary parts).

    - If n <= len(RIEMANN_ZEROS), returns the built-in seed list.
    - If n is larger, attempts to extend via mpmath.zetazero (optional dependency).
    """

    n = int(n)
    if n <= 0:
        return np.array([], dtype=float)
    if n <= RIEMANN_ZEROS.size:
        return RIEMANN_ZEROS[:n].copy()

    zeros = list(RIEMANN_ZEROS.tolist())
    try:
        from mpmath import mp, zetazero

        mp.dps = 50
        for k in range(len(zeros) + 1, n + 1):
            zeros.append(float(zetazero(k).imag))
        return np.array(zeros, dtype=float)
    except Exception as e:
        raise RuntimeError(
            f"Requested n={n} zeros but only {len(RIEMANN_ZEROS)} are bundled. "
            "Install mpmath to extend (pip install mpmath)."
        ) from e


def montgomery_odlyzko_correlation(r: np.ndarray) -> np.ndarray:
    """
    GUE pair correlation (Montgomery–Odlyzko):

      R2(r) = 1 - (sin(πr)/(πr))^2
    """

    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    nz = r != 0
    out[nz] = 1.0 - (np.sin(np.pi * r[nz]) / (np.pi * r[nz])) ** 2
    out[~nz] = 0.0
    return out


@dataclass
class RiemannZeroVerification:
    """Verification helper for comparisons against Riemann zero statistics."""

    n_zeros: int = 20

    def __post_init__(self):
        self.zeros = load_riemann_zeros(self.n_zeros)
        self.n = int(self.zeros.size)

    def verify_first_n_modes(
        self,
        measured_wavenumbers: np.ndarray,
        boundary_radius: float,
        tolerance: float = 0.05,
    ) -> Tuple[bool, Dict]:
        """
        Check k_n ≈ γ_n / R (within relative tolerance).
        """

        measured = np.asarray(measured_wavenumbers, dtype=float)
        R = float(boundary_radius)
        if R <= 0:
            raise ValueError("boundary_radius must be > 0")
        n_compare = int(min(measured.size, self.n))
        if n_compare == 0:
            return False, {"reason": "no data"}

        predicted = self.zeros[:n_compare] / R
        meas = measured[:n_compare]
        rel_err = np.abs(meas - predicted) / np.maximum(np.abs(predicted), 1e-15)
        max_err = float(np.max(rel_err))
        mean_err = float(np.mean(rel_err))
        return max_err < float(tolerance), {
            "n_compared": n_compare,
            "max_relative_error": max_err,
            "mean_relative_error": mean_err,
            "relative_errors": rel_err,
            "predicted": predicted,
            "measured": meas,
        }

    def verify_pair_correlation(
        self,
        eigenvalues: Optional[np.ndarray] = None,
        max_r: float = 3.0,
        bins: int = 50,
        sigma_threshold: float = 3.0,
    ) -> Tuple[bool, Dict]:
        """
        Compare an empirical pair-correlation proxy to the GUE prediction.
        Returns a simple chi^2 per dof diagnostic.
        """

        eigs = np.asarray(self.zeros if eigenvalues is None else eigenvalues, dtype=float)
        eigs = np.sort(eigs)
        if eigs.size < 10:
            return False, {"reason": "insufficient eigenvalues"}
        spacings = np.diff(eigs)
        mean_spacing = float(np.mean(spacings))
        if mean_spacing == 0:
            return False, {"reason": "zero mean spacing"}
        normalized = eigs / mean_spacing

        r_vals, R2_emp = self._compute_pair_correlation(normalized, max_r=max_r, bins=bins)
        R2_th = montgomery_odlyzko_correlation(r_vals)

        residuals = R2_emp - R2_th
        chi_sq = float(np.sum((residuals**2) / (np.abs(R2_th) + 0.01)))
        chi_sq_per_dof = chi_sq / float(len(r_vals))

        return chi_sq_per_dof < float(sigma_threshold), {
            "chi_sq_per_dof": chi_sq_per_dof,
            "r_values": r_vals,
            "R2_empirical": R2_emp,
            "R2_theory": R2_th,
            "residuals": residuals,
        }

    def _compute_pair_correlation(
        self, normalized_eigs: np.ndarray, max_r: float = 3.0, bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = int(normalized_eigs.size)
        distances: List[float] = []
        # local-window sampling for speed
        for i in range(N):
            for j in range(i + 1, min(i + 25, N)):
                dist = float(abs(normalized_eigs[j] - normalized_eigs[i]))
                if 0.01 <= dist < float(max_r):
                    distances.append(dist)
        if len(distances) == 0:
            r = np.linspace(0.01, max_r, bins)
            return r, np.zeros_like(r)

        hist, edges = np.histogram(distances, bins=bins, range=(0.01, max_r), density=True)
        centers = (edges[:-1] + edges[1:]) / 2

        # normalize tail towards 1
        if hist.size >= 10:
            tail = float(np.mean(hist[-10:]))
            if tail > 0:
                hist = hist / tail
        return centers, hist

    def compute_fractal_exponent_from_zeros(self, n_zeros: Optional[int] = None) -> float:
        """
        Proxy for a "fractal exponent" derived from mean spacing:

          δ = 2π / (⟨Δγ⟩ log(γ_N / 2π))
        """

        n = self.n if n_zeros is None else int(min(n_zeros, self.n))
        if n < 3:
            return float("nan")
        zeros = self.zeros[:n]
        spacings = np.diff(zeros)
        mean_spacing = float(np.mean(spacings))
        if mean_spacing == 0:
            return float("nan")
        return float(2 * np.pi / (mean_spacing * np.log(zeros[-1] / (2 * np.pi))))


