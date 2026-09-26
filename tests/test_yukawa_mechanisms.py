"""Checks for doc/derivations/yukawa_mechanisms_2026-09-26.md.

Oracles: explicit spin-1 rotation matrices (invariants computed as a nullspace),
the explicit spin-weighted zero modes near the pole, and the round-3 Yang-Mills
level -|n|/2 for g = 2.
"""

import json

import numpy as np
import pytest

import bpr.yukawa_mechanisms as y


def test_jz_invariant_symmetric_yukawas_are_the_m_plus_mprime_zero_entries():
    basis = y.invariant_symmetric_matrices()
    assert len(basis) == 2
    mask = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool)  # (m, m') with m + m' = 0
    for B in basis:
        assert np.allclose(B[~mask], 0)
    # Independent check: every such matrix commutes with rotations about z, D^T Y D = Y, D = diag(e^{-i m a}).
    for B in basis:
        for a in (0.3, 1.7):
            D = np.diag(np.exp(-1j * a * np.array([1, 0, -1])))
            assert np.allclose(D.T @ B @ D, B)


def test_a_single_brane_always_gives_a_degenerate_pair():
    rng = np.random.default_rng(2)
    for _ in range(50):
        sv, _ = y.brane_spectrum(rng)
        diffs = [abs(sv[0] - sv[1]), abs(sv[1] - sv[2])]
        assert min(diffs) < 1e-10 * max(sv)


@pytest.mark.parametrize("k,orders", [(3, {1: 0, 0: 1, -1: 2}), (5, {2: 0, 1: 1, 0: 2, -1: 3, -2: 4})])
def test_zero_mode_vanishing_orders_at_the_pole(k, orders):
    assert y.vanishing_orders(k) == orders


def test_all_brane_allowed_entries_have_the_same_suppression():
    assert set(y.brane_suppression_orders(3).values()) == {2}


def test_two_branes_lift_the_degeneracy_without_a_hierarchy():
    rng = np.random.default_rng(5)
    ratios = []
    for _ in range(200):
        sv = y.two_brane_spectrum(rng)
        assert sv[1] - sv[2] > 1e-8 * sv[0] or sv[0] - sv[1] > 1e-8 * sv[0]
        ratios.append(sv[2] / sv[0])
    # O(1) random coefficients give O(1) ratios: an up-quark-like 1e-5 ratio is not generic.
    assert np.median(ratios) > 0.05
    assert np.mean(np.array(ratios) < 1e-4) < 0.02


def test_proca_level_and_tuning():
    assert y.proca_lowest_level(2, 0) == -3  # round 3: Yang-Mills value -|n|/2 for |n| = 6
    assert y.proca_lowest_level(2, 3) == 0 and y.proca_lowest_level(1, 0) == 0
    assert y.higgs_tuning() < 1e-29


def test_report():
    report = y.demonstration_report()
    json.loads(json.dumps(report))
    assert report["jz_invariant_dimension"] == 2 and report["empirical_validation"] is False
    assert report["limitations"] == y.LIMITATIONS


def test_single_brane_never_gives_three_distinct_masses_for_any_jz_charge():
    expected = {-4: "zero", -3: "zero", -2: "rank1", -1: "pair+zero", 0: "pair+one",
                1: "pair+zero", 2: "rank1", 3: "zero", 4: "zero"}
    for c, pattern in expected.items():
        assert y.single_brane_spectrum_pattern(c, np.random.default_rng(c + 20)) == pattern
    report = y.demonstration_report()
    assert all("distinct" not in p for p in report["single_brane_patterns_by_c"].values())
