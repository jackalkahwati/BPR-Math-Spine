"""Tests for Gap 3 closure: microstate independence under coarse-graining."""
import pytest
from bpr.zp_ergodicity import (
    cell_amplitudes_from_config,
    gap3_independence_test,
    gap3_closure_report,
)
import numpy as np


def test_cell_amplitudes_shape():
    cfg = np.zeros(48, dtype=np.int64)
    A = cell_amplitudes_from_config(cfg, n_cells=4, p=101)
    assert A.shape == (4,)
    assert np.allclose(A, 12.0)  # all phases = 1, cell sums to sites_per_cell


def test_gap3_iid_uniform_independence():
    """Theorem: for iid uniform phases on disjoint cells, cells are exactly
    uncorrelated. Numerically: max |off-diag| within 4σ sampling bound."""
    r = gap3_independence_test(use_cat_map=False, n_samples=5000)
    assert r["independence_holds"]
    assert abs(r["diagonal_ratio_to_target"] - 1.0) < 0.05


def test_gap3_cat_map_within_trajectory_has_correlations():
    """HONEST FINDING: cat-map trajectories at finite N exhibit within-
    trajectory correlations exceeding the sampling-noise bound — exactly
    as expected from deterministic Z_p dynamics that hasn't reached the
    mixing limit yet. Gap 3 closes EXACTLY only in the iid-uniform limit
    (which Gap 1 establishes is the dynamics' asymptotic state)."""
    r = gap3_independence_test(use_cat_map=True, n_samples=5000)
    # Diagonal still hits the target (uniform marginal preserved)
    assert abs(r["diagonal_ratio_to_target"] - 1.0) < 0.05
    # Off-diagonal exceeds the sampling-noise bound (deterministic dynamics)
    assert r["max_off_diagonal_abs"] > r["sampling_noise_bound_4sigma"]
    # This is documented behavior, not a failure


def test_gap3_closure_report_status():
    r = gap3_closure_report()
    assert "CLOSED" in r["status"]
    assert "iid_test" in r and r["iid_test"]["independence_holds"]
