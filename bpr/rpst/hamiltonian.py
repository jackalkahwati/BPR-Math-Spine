"""
RPST Hamiltonian - spectral analysis scaffolding (Eq 4b / GUE link)

Implements a simple Legendre-symbol-coupled Hamiltonian on Z_p and utilities
for nearest-neighbor spacing statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .substrate import legendre_symbol


def _is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


class RPSTHamiltonian:
    """
    RPST Hamiltonian on Z_p with Legendre symbol coupling:

        H[a,b] = (a/p) (b/p)

    This is a rank-1 structure (outer product) but still gives a clean,
    testable pipeline for spacing statistics utilities.
    """

    def __init__(self, p: int):
        p = int(p)
        if not _is_prime(p):
            raise ValueError(f"{p} is not prime")
        self.p = p
        self._H = None
        self._eigenvalues = None

    def build_hamiltonian(self) -> np.ndarray:
        if self._H is not None:
            return self._H
        p = self.p
        leg = np.array([legendre_symbol(a, p) for a in range(p)], dtype=float)
        self._H = np.outer(leg, leg)
        return self._H

    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is not None:
            return self._eigenvalues
        H = self.build_hamiltonian()
        self._eigenvalues = np.sort(np.linalg.eigvalsh(H))
        return self._eigenvalues

    def normalized_eigenvalues(self) -> np.ndarray:
        eigs = self.eigenvalues()
        nonzero = eigs[np.abs(eigs) > 1e-10]
        if nonzero.size < 2:
            return nonzero
        spacings = np.diff(np.sort(nonzero))
        mean_spacing = float(np.mean(spacings))
        if mean_spacing == 0:
            return nonzero
        return np.sort(nonzero) / mean_spacing

    def level_spacing_distribution(self, bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        eigs = self.normalized_eigenvalues()
        if eigs.size < 2:
            return np.array([]), np.array([])
        spacings = np.diff(eigs)
        spacings = spacings / float(np.mean(spacings))
        hist, edges = np.histogram(spacings, bins=bins, range=(0, 4), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, hist


def compute_spectral_zeta(H: RPSTHamiltonian, s: complex) -> complex:
    eigs = H.eigenvalues()
    nonzero = eigs[np.abs(eigs) > 1e-10]
    if nonzero.size == 0:
        return 0.0 + 0.0j
    return complex(np.sum(np.abs(nonzero) ** (-s)))


def wigner_surmise_gue(s: np.ndarray) -> np.ndarray:
    """
    GUE Wigner surmise:
      P(s) = (32/π²) s² exp(-4s²/π)
    """
    s = np.asarray(s, dtype=float)
    return (32.0 / np.pi**2) * (s**2) * np.exp(-(4.0 / np.pi) * (s**2))


def _ks_statistic(samples: np.ndarray, cdf_fn) -> float:
    """Simple Kolmogorov–Smirnov statistic D_n vs a provided CDF."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n == 0:
        return 1.0
    cdf_vals = np.array([cdf_fn(v) for v in x], dtype=float)
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - cdf_vals)))


def verify_gue_level_spacing(H: RPSTHamiltonian, tolerance: float = 0.3) -> Tuple[bool, float]:
    """
    Verify that normalized nearest-neighbor spacings are not wildly inconsistent
    with the GUE Wigner surmise CDF.

    Returns:
        (passes, KS_statistic)
    """

    eigs = H.normalized_eigenvalues()
    if eigs.size < 10:
        return False, 1.0
    spacings = np.diff(eigs)
    spacings = spacings / float(np.mean(spacings))

    # Wigner surmise CDF for GUE: 1 - exp(-4 s^2 / π)
    def wigner_cdf(s):
        s = float(max(0.0, s))
        return 1.0 - float(np.exp(-(4.0 / np.pi) * (s**2)))

    try:
        from scipy.stats import kstest

        stat, _p = kstest(spacings, wigner_cdf)
        D = float(stat)
    except Exception:
        D = _ks_statistic(spacings, wigner_cdf)

    return D < float(tolerance), D


