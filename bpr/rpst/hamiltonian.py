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


class RiemannZeroStatistics:
    """GUE statistics of the Riemann zeros — numerical validation.

    Implements the analysis in Al-Kahwati (2026), *Numerical Evidence for
    Riemann Zero Statistics in the RPST Paley Hamiltonian*, Section 4.

    The key results (Table 2 of that paper):
      - 0 of 99 spacings below s=0.3 (vs Poisson prediction of ~26%)
      - Pair correlation R2(0) ≈ 0 (level repulsion)
      - GUE Wigner surmise KS stat D = 0.141, p = 0.047 for N=100 zeros
    """

    @staticmethod
    def smooth_count(T: float) -> float:
        r"""Smooth counting function N_smooth(T) (Eq. 12 of numerical paper).

        N_smooth(T) = (T/2π) log(T/2πe) + 7/8

        Used to unfold the Riemann zeros to unit mean spacing.

        Parameters
        ----------
        T : float
            Height on the critical line.

        Returns
        -------
        float
            Smooth approximation to N(T) = #{zeros with Im(ρ) ≤ T}.
        """
        if T <= 0:
            return 0.0
        return (T / (2.0 * np.pi)) * np.log(T / (2.0 * np.pi * np.e)) + 7.0 / 8.0

    @classmethod
    def unfold(cls, zeros: np.ndarray) -> np.ndarray:
        r"""Unfold Riemann zeros to unit mean spacing.

        Define γ̂_n = N_smooth(γ_n) so that the unfolded sequence has
        mean spacing 1.

        Parameters
        ----------
        zeros : ndarray
            Imaginary parts γ_n of Riemann zeros (positive, increasing).

        Returns
        -------
        ndarray
            Unfolded zero sequence with mean spacing ≈ 1.
        """
        zeros = np.sort(np.asarray(zeros, dtype=float))
        return np.array([cls.smooth_count(g) for g in zeros])

    @classmethod
    def nearest_neighbor_spacings(cls, zeros: np.ndarray) -> np.ndarray:
        r"""Nearest-neighbor spacings of unfolded Riemann zeros.

        s_n = γ̂_{n+1} - γ̂_n, normalized to unit mean.

        Parameters
        ----------
        zeros : ndarray
            Imaginary parts of Riemann zeros.

        Returns
        -------
        ndarray
            Spacings normalized to mean 1.
        """
        unfolded = cls.unfold(zeros)
        spacings = np.diff(unfolded)
        mean = float(np.mean(spacings))
        if mean > 0:
            spacings = spacings / mean
        return spacings

    @classmethod
    def pair_correlation(
        cls,
        zeros: np.ndarray,
        r_max: float = 4.0,
        n_bins: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Pair correlation function R2(r) of unfolded zeros.

        Counts ordered pairs (m, n) with m ≠ n and
        γ̂_m - γ̂_n ∈ [r, r+dr] (positive differences only),
        normalized so that R2(r) → 1 for Poisson statistics.

        GUE prediction (Eq. 14):
            R2_GUE(r) = 1 - (sin(πr) / (πr))²

        Parameters
        ----------
        zeros : ndarray
            Imaginary parts of Riemann zeros.
        r_max : float
            Maximum separation to compute.
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        r_centers : ndarray
            Bin center values.
        R2 : ndarray
            Pair correlation function values.
        """
        unfolded = cls.unfold(zeros)
        N = len(unfolded)
        dr = r_max / n_bins
        counts = np.zeros(n_bins)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                sep = unfolded[j] - unfolded[i]
                if 0.0 < sep < r_max:
                    idx = int(sep / dr)
                    if idx < n_bins:
                        counts[idx] += 1

        r_centers = (np.arange(n_bins) + 0.5) * dr
        # Normalize: Poisson expectation is N * dr per bin
        R2 = counts / (N * dr)
        return r_centers, R2

    @staticmethod
    def gue_pair_correlation(r: np.ndarray) -> np.ndarray:
        r"""GUE sine-kernel pair correlation (Eq. 14 of numerical paper).

        R2_GUE(r) = 1 - (sin(πr) / (πr))²

        Parameters
        ----------
        r : ndarray
            Separation values (in units of mean spacing).

        Returns
        -------
        ndarray
            GUE pair correlation.
        """
        r = np.asarray(r, dtype=float)
        result = np.zeros_like(r)  # r=0 → R2=0 (level repulsion, sin(πr)/πr→1)
        nonzero = r > 1e-10
        pir = np.pi * r[nonzero]
        result[nonzero] = 1.0 - (np.sin(pir) / pir) ** 2
        return result

    @classmethod
    def ks_test_gue(cls, zeros: np.ndarray) -> Tuple[float, float]:
        r"""KS test of spacing distribution vs GUE Wigner surmise.

        GUE Wigner surmise CDF approximation: 1 - exp(-4s²/π).

        Parameters
        ----------
        zeros : ndarray
            Imaginary parts of Riemann zeros.

        Returns
        -------
        D : float
            KS statistic.
        p_value : float
            Two-sided p-value (nan if scipy unavailable).
        """
        spacings = cls.nearest_neighbor_spacings(zeros)

        def gue_cdf(s: float) -> float:
            s = max(0.0, float(s))
            return 1.0 - float(np.exp(-(4.0 / np.pi) * s ** 2))

        try:
            from scipy.stats import kstest
            stat, pval = kstest(spacings, gue_cdf)
            return float(stat), float(pval)
        except Exception:
            x = np.sort(spacings)
            n = len(x)
            if n == 0:
                return 1.0, float("nan")
            ecdf = np.arange(1, n + 1) / n
            cdf_vals = np.array([gue_cdf(v) for v in x])
            return float(np.max(np.abs(ecdf - cdf_vals))), float("nan")

    @classmethod
    def fraction_small_spacings(
        cls, zeros: np.ndarray, threshold: float = 0.3
    ) -> float:
        r"""Fraction of nearest-neighbor spacings below threshold.

        Level repulsion signature:
          - GUE predicts ≈ 5% below s = 0.3
          - Poisson predicts ≈ 26% below s = 0.3
          - Observed for 99 Riemann zeros: 0% (Table 2 of numerical paper)

        Parameters
        ----------
        zeros : ndarray
            Imaginary parts of Riemann zeros.
        threshold : float
            Spacing threshold (in units of mean spacing).

        Returns
        -------
        float
            Fraction of spacings below threshold.
        """
        spacings = cls.nearest_neighbor_spacings(zeros)
        if len(spacings) == 0:
            return 0.0
        return float(np.mean(spacings < threshold))


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


