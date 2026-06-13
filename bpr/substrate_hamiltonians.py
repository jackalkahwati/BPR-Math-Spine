"""Candidate full-rank substrate Hamiltonians on Z_p.

The original ``RPSTHamiltonian`` in ``bpr/rpst/hamiltonian.py`` builds
``H = outer(leg, leg)`` (Legendre-symbol outer product), which is rank-1
by construction — one nonzero eigenvalue, no level-spacing statistics
possible. The earlier audit locked this in by test, downgrading the
GUE/Riemann conjecture from 'Tier 2 with numerical support' to 'not
currently supported by the implementation'.

This module provides **candidate full-rank Hermitian operators** on Z_p
that use the prime modular structure intrinsically, and tests whether
their spectra exhibit Wigner-Dyson (random-matrix) level spacing rather
than Poisson (integrable) or arithmetic degeneracy (cat-map-like).

Constructions
-------------
1. **Legendre Hankel**: H_{ij} = Legendre((i+j) mod p, p)
   Hermitian by construction (anti-)symmetric Hankel; ±1, 0 entries;
   uses the multiplicative Legendre structure intrinsically.

2. **Legendre Multiplicative**: H_{ij} = Legendre((i*j) mod p, p)
   Hermitian (Legendre is multiplicative); uses multiplicative
   subgroup structure of (Z/pZ)*.

3. **Legendre Circulant**: H_{ij} = Legendre((i-j) mod p, p)
   Eigenvalues are Gauss sums — known to be degenerate (~2 distinct
   eigenvalues). Included as a NEGATIVE-control baseline; expected to
   fail random-matrix statistics.

4. **Discrete Berry-Keating**: H = (X̂P̂ + P̂X̂)/2 on Z_p, with
   P̂ = F†X̂F (DFT-conjugate of X̂).
   The discrete analog of Berry-Keating's H = xp conjectured to
   reproduce Riemann zeros in the continuum limit.

5. **Sum Legendre + small random perturbation**: structural matrix +
   Gaussian noise. Tests whether arithmetic structure is robust to
   perturbation.

Honest expectation
------------------
- (1) Legendre Hankel and (2) multiplicative SHOULD show level
  repulsion (Wigner-Dyson) for typical p — generic Hermitian random
  matrices with structured signs do.
- (3) Circulant SHOULD fail (Gauss-sum degeneracies — already
  documented in the earlier audit).
- (4) Discrete Berry-Keating is uncertain — relevant to Hilbert-Pólya
  / Riemann; spectrum depends sensitively on the discretization choice.

We report which constructions actually exhibit Wigner-Dyson statistics
under the Kolmogorov-Smirnov test, and lock the findings into tests.
This either provides a usable replacement for the rank-1 RPSTHamiltonian
or honestly closes the question by showing no Z_p-Hermitian operator in
this class reproduces GUE.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest

from .prime_state_check import legendre_symbol


# ---------------------------------------------------------------------------
# Wigner-Dyson surmise CDFs for random-matrix-theory comparison
# ---------------------------------------------------------------------------

def _goe_pdf(s: float) -> float:
    """GOE (β=1) Wigner surmise: p(s) = (π/2) s exp(-π s² / 4)."""
    return 0.5 * np.pi * s * np.exp(-np.pi * s ** 2 / 4.0)


def _gue_pdf(s: float) -> float:
    """GUE (β=2) Wigner surmise: p(s) = (32/π²) s² exp(-4 s² / π)."""
    return (32.0 / np.pi ** 2) * s ** 2 * np.exp(-4.0 * s ** 2 / np.pi)


def _poisson_pdf(s: float) -> float:
    return np.exp(-s)


def _make_cdf(pdf: Callable[[float], float]) -> Callable[[np.ndarray], np.ndarray]:
    def cdf(s: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(s).astype(float)
        return np.array([quad(pdf, 0.0, max(0.0, x))[0] for x in s])
    return cdf


GOE_CDF = _make_cdf(_goe_pdf)
GUE_CDF = _make_cdf(_gue_pdf)
POISSON_CDF = _make_cdf(_poisson_pdf)


def unfolded_spacings(eigs: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Nearest-neighbor spacings normalized to mean 1.

    Removes near-degenerate eigenvalue pairs (within tol) which would
    otherwise dominate the spacing statistics with zero spacings.
    """
    e = np.sort(np.real(eigs))
    s = np.diff(e)
    s = s[s > tol]
    if s.size == 0:
        return s
    return s / s.mean()


def classify_spacing_statistics(spacings: np.ndarray) -> dict:
    """Run K-S tests against GOE, GUE, Poisson; pick the closest."""
    if spacings.size < 20:
        return {"best_fit": "insufficient_data", "n_spacings": int(spacings.size)}
    d_goe = float(kstest(spacings, GOE_CDF).statistic)
    d_gue = float(kstest(spacings, GUE_CDF).statistic)
    d_poisson = float(kstest(spacings, POISSON_CDF).statistic)
    results = {"GOE": d_goe, "GUE": d_gue, "Poisson": d_poisson}
    best = min(results, key=results.get)
    return {
        "n_spacings": int(spacings.size),
        "ks_statistic": results,
        "best_fit": best,
        "best_fit_D": results[best],
    }


# ---------------------------------------------------------------------------
# Candidate Hamiltonian constructions
# ---------------------------------------------------------------------------

def legendre_hankel_hamiltonian(p: int) -> np.ndarray:
    """H_{ij} = Legendre((i+j) mod p, p). Hermitian by Hankel symmetry."""
    H = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            H[i, j] = legendre_symbol((i + j) % p, p)
    return H


def legendre_multiplicative_hamiltonian(p: int) -> np.ndarray:
    """H_{ij} = Legendre((i*j) mod p, p). Hermitian (multiplicative)."""
    H = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            H[i, j] = legendre_symbol((i * j) % p, p)
    return H


def legendre_circulant_hamiltonian(p: int) -> np.ndarray:
    """H_{ij} = Legendre((i-j) mod p, p). NEGATIVE control (Gauss-sum degen)."""
    H = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            H[i, j] = legendre_symbol((i - j) % p, p)
    return (H + H.T) / 2.0


def discrete_berry_keating_hamiltonian(p: int) -> np.ndarray:
    """Discrete Berry-Keating H = (X̂P̂ + P̂X̂)/2 on Z_p with DFT-conjugate P̂."""
    x = np.arange(p, dtype=np.complex128)
    X = np.diag(x)
    # DFT matrix
    k, j = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    F = np.exp(2j * np.pi * k * j / p) / np.sqrt(p)
    P = F.conj().T @ X @ F
    H = 0.5 * (X @ P + P @ X)
    return (H + H.conj().T) / 2.0  # ensure Hermitian to machine precision


# ---------------------------------------------------------------------------
# Spectral statistics test harness
# ---------------------------------------------------------------------------

def spectral_statistics(H: np.ndarray) -> dict:
    """Eigenvalues, spacings, and Wigner-Dyson classification of H."""
    eigs = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    rank = int(np.linalg.matrix_rank(H, tol=1e-9))
    spacings = unfolded_spacings(eigs)
    classification = classify_spacing_statistics(spacings)
    return {
        "n_eigs": int(eigs.size),
        "rank": rank,
        "n_distinct_eigs": int(
            np.unique(np.round(eigs, 8)).size
        ),
        "spacings_mean": float(spacings.mean()) if spacings.size else float("nan"),
        "spacings_var": float(spacings.var()) if spacings.size else float("nan"),
        **classification,
    }


def survey_candidates(primes: tuple[int, ...] = (211, 503, 1009)) -> dict:
    """Run spectral-statistics survey on all candidate constructions."""
    constructions = {
        "legendre_hankel": legendre_hankel_hamiltonian,
        "legendre_multiplicative": legendre_multiplicative_hamiltonian,
        "legendre_circulant_neg_control": legendre_circulant_hamiltonian,
        "discrete_berry_keating": discrete_berry_keating_hamiltonian,
    }
    results = {}
    for name, ctor in constructions.items():
        results[name] = {}
        for p in primes:
            H = ctor(p)
            results[name][p] = spectral_statistics(H)
    return results


# ---------------------------------------------------------------------------
# Additional candidates for the negative-finding context
# ---------------------------------------------------------------------------

def random_goe_sanity_check(p: int, seed: int = 0) -> np.ndarray:
    """Generic random real symmetric matrix (GOE class).

    SANITY CHECK that the spacing-statistics methodology works: this
    construction is the textbook GOE ensemble and MUST give D_GOE < D_GUE,
    D_Poisson for any reasonable p. If it doesn't, the methodology is wrong.
    Carries no BPR content.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((p, p))
    H = (M + M.T) / np.sqrt(2)
    return H


def discrete_berry_keating_perturbed(
    p: int, eps: float = 0.05, seed: int = 0
) -> np.ndarray:
    """Discrete Berry-Keating + small Gaussian Hermitian perturbation.

    Tests whether the integrable Berry-Keating spectrum (Poisson) transitions
    to chaotic (GOE) under perturbation. The standard Berry-Tabor /
    Bohigas-Giannoni-Schmit transition.
    """
    H0 = discrete_berry_keating_hamiltonian(p)
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((p, p))
    V = (M + M.T) / np.sqrt(2)
    return H0 + eps * V


def legendre_2d_berry_keating(p: int) -> np.ndarray:
    """2D Berry-Keating tensor product on Z_p × Z_p (dim p²).

    H = X̂ ⊗ P̂ + P̂ ⊗ X̂  with X̂, P̂ on Z_p as in the 1D BK.

    Larger Hilbert space (p²) gives access to richer spectral statistics
    that might support GUE/CUE class behavior unavailable to the 1D
    rank-restricted constructions.
    """
    x = np.arange(p, dtype=np.complex128)
    X = np.diag(x)
    k, j = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    F = np.exp(2j * np.pi * k * j / p) / np.sqrt(p)
    P = F.conj().T @ X @ F
    Ip = np.eye(p, dtype=np.complex128)
    XP = np.kron(X, P)
    PX = np.kron(P, X)
    H = 0.5 * (XP + PX)
    return (H + H.conj().T) / 2.0


def extended_candidate_survey(primes_small: tuple[int, ...] = (211, 503)) -> dict:
    """Run survey including the extended candidates."""
    results = {}
    for name, ctor in [
        ("random_GOE_sanity_check", lambda p: random_goe_sanity_check(p)),
        ("discrete_BK_perturbed_eps_0.05", lambda p: discrete_berry_keating_perturbed(p, eps=0.05)),
        ("discrete_BK_perturbed_eps_0.5", lambda p: discrete_berry_keating_perturbed(p, eps=0.5)),
    ]:
        results[name] = {p: spectral_statistics(ctor(p)) for p in primes_small}
    # 2D BK only for the smallest prime (p² states is expensive)
    p = primes_small[0]
    H2 = legendre_2d_berry_keating(min(53, p))  # cap to avoid memory blowup
    results["legendre_2d_berry_keating"] = {min(53, p): spectral_statistics(H2)}
    return results
