"""Tests for candidate Z_p substrate Hamiltonians and spectral statistics.

Honest findings locked in: none of the prime-modular Z_p Hermitian operator
constructions naturally exhibits Wigner-Dyson level statistics.
"""
import numpy as np
import pytest
from bpr.substrate_hamiltonians import (
    legendre_hankel_hamiltonian,
    legendre_multiplicative_hamiltonian,
    legendre_circulant_hamiltonian,
    discrete_berry_keating_hamiltonian,
    random_goe_sanity_check,
    spectral_statistics,
)


def test_legendre_multiplicative_is_rank_one():
    """H_{ij} = leg(i*j mod p) = leg(i)leg(j) — same rank-1 outer product as
    the original RPSTHamiltonian. Documented limit, not improvement."""
    s = spectral_statistics(legendre_multiplicative_hamiltonian(31))
    assert s["rank"] == 1


def test_legendre_hankel_has_arithmetic_degeneracy():
    """Legendre Hankel is nearly full rank but has only ~3 distinct
    eigenvalues (Gauss-sum-like degeneracy) — can't do level spacing."""
    s = spectral_statistics(legendre_hankel_hamiltonian(211))
    assert s["rank"] >= 210
    assert s["n_distinct_eigs"] <= 4


def test_legendre_circulant_gauss_sum_degenerate():
    """Legendre circulant: ≤3 distinct eigenvalues from Gauss-sum structure."""
    s = spectral_statistics(legendre_circulant_hamiltonian(211))
    assert s["n_distinct_eigs"] <= 4


def test_discrete_berry_keating_full_rank_but_poisson():
    """Discrete BK is full rank with distinct eigenvalues, but INTEGRABLE:
    spectral statistics are Poisson, not Wigner-Dyson. The prime-modular
    structure is incompatible with GUE in this class."""
    s = spectral_statistics(discrete_berry_keating_hamiltonian(211))
    assert s["rank"] == 211
    assert s["n_distinct_eigs"] == 211
    assert s["best_fit"] == "Poisson"


def test_random_goe_sanity_check_passes_methodology():
    """Methodology sanity check: a generic GOE matrix MUST classify as GOE.
    Locks in that the K-S testing infrastructure is correct."""
    s = spectral_statistics(random_goe_sanity_check(211, seed=0))
    assert s["best_fit"] == "GOE"
    assert s["ks_statistic"]["GOE"] < s["ks_statistic"]["GUE"]
    assert s["ks_statistic"]["GOE"] < s["ks_statistic"]["Poisson"]


def test_berry_keating_transitions_to_goe_under_large_perturbation():
    """Sanity: BK + perturbation ≳ 10× mean spacing transitions to GOE,
    confirming the integrable structure breaks under sufficient noise."""
    p = 211
    H_bk = discrete_berry_keating_hamiltonian(p)
    eigs = np.linalg.eigvalsh(H_bk)
    mean_spacing = float((eigs.max() - eigs.min()) / (p - 1))
    rng = np.random.default_rng(0)
    M = rng.standard_normal((p, p))
    V = (M + M.T) / np.sqrt(2)
    H = H_bk + 10.0 * mean_spacing * V
    s = spectral_statistics(H)
    assert s["best_fit"] == "GOE"
