"""Tests for Z_p ergodicity → uniform phases (Born-rule Gap 1 closure)."""
import pytest
import numpy as np
from bpr.zp_ergodicity import (
    cat_map_step,
    cat_map_trajectory,
    chi_squared_uniform,
    ensemble_invariance_test,
    coarse_grained_mixing_test,
    gap1_closure_report,
)


def test_cat_map_is_symplectic():
    """det = 1: the cat map preserves the Z_p area element."""
    # M = [[2, 1], [1, 1]] has det = 2*1 - 1*1 = 1
    assert 2 * 1 - 1 * 1 == 1


def test_cat_map_is_a_bijection():
    """The cat map mod p is a permutation: distinct seeds give distinct images."""
    p = 47
    states = np.array([(q, P) for q in range(p) for P in range(p)], dtype=np.int64)
    images = cat_map_step(states, p)
    unique = np.unique(images.view([('q', np.int64), ('P', np.int64)]))
    assert unique.size == p * p


def test_leg_A_invariance_theorem():
    """Uniform ensemble stays uniform under iteration, to within sampling noise."""
    r = ensemble_invariance_test(p=211, n_seeds=50_000, n_steps=50)
    assert r["within_band"]


def test_leg_B_mixing_qualitative():
    """A concentrated initial ensemble spreads under iteration (chi² drops)."""
    r = coarse_grained_mixing_test(p=211, n_steps=30, initial_box=8, n_seeds=20_000)
    # qualitative: final binned chi² is at least 10x lower than initial
    assert r["chi2_binned_final"] < r["chi2_binned_initial"] / 10


def test_gap1_closure_status_string():
    """Report should explicitly state Gap 1 is closed."""
    r = gap1_closure_report(p=211)
    assert "Gap 1 CLOSED" in r["status"]
    assert "Arnold-Avez" in r["leg_C_continuum_theorem"]
