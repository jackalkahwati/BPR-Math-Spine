"""Tests for constructive Born-rule microstate counting (Gap 2 of born_rule conjecture)."""
import numpy as np
import pytest
from bpr.born_rule import (
    coarse_grained_amplitude,
    born_frequencies_from_microstates,
    two_slit_wavefunction,
    gaussian_wavefunction,
)


def test_coarse_grained_amplitude_shape():
    cfg = np.zeros(32, dtype=int)
    A = coarse_grained_amplitude(cfg, n_cells=4, p=101)
    assert A.shape == (4,)
    # all-zero config: every phase = 1, cell sum = sites_per_cell = 8
    assert np.allclose(A, 8.0)


def test_born_recovered_uniform():
    psi = np.ones(16, dtype=complex)
    r = born_frequencies_from_microstates(psi, n_samples=40000, sites_per_cell=40, seed=1)
    assert r["l1_error"] < 0.06


def test_born_recovered_two_slit_with_real_fringes():
    """Two-slit must have genuine non-uniform fringes AND recover Born."""
    psi = two_slit_wavefunction(16)
    born = np.abs(psi) ** 2
    born = born / born.sum()
    assert np.var(born) > 1e-3, "two-slit must have real fringes, not flat modulus"
    r = born_frequencies_from_microstates(psi, n_samples=40000, sites_per_cell=40, seed=1)
    assert r["l1_error"] < 0.09


def test_born_recovered_gaussian():
    psi = gaussian_wavefunction(16)
    r = born_frequencies_from_microstates(psi, n_samples=40000, sites_per_cell=40, seed=1)
    assert r["l1_error"] < 0.10


def test_born_weights_normalized():
    psi = gaussian_wavefunction(16)
    r = born_frequencies_from_microstates(psi, n_samples=5000, seed=2)
    assert r["born_weights"].sum() == pytest.approx(1.0)
    assert r["measured_freq"].sum() == pytest.approx(1.0)
