"""Checks for doc/derivations/yukawa_mechanisms_2026-09-26.md.

Oracles: explicit spin-1 rotation matrices (invariants computed as a nullspace),
the explicit spin-weighted harmonics and their eth-derivatives near the pole, the
spin-1 coherent-state Vandermonde, and the round-3 Yang-Mills level -|n|/2 for g = 2.
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


@pytest.mark.parametrize("c", [0, 1, 2])
def test_all_brane_allowed_entries_have_the_same_suppression(c):
    orders = y.brane_suppression_orders(3, c=c)
    assert orders and set(orders.values()) == {2 - c}


@pytest.mark.parametrize("s,l", [(-1, 2), (2, 3), (3, 4), (-3, 4)])
def test_only_m_equal_minus_s_survives_at_the_pole(s, l):
    assert y.pole_nonzero_m(s, l) == [-s]


def test_eth_derivatives_of_zero_modes_at_the_pole():
    # eth-bar annihilates the zero modes; eth^a f_m is nonzero at the pole exactly when m = 1 - a.
    table = y.derivative_pole_table()
    for key, nonzero in table.items():
        m, a = map(int, key.split(","))
        assert nonzero == (m == 1 - a)


def test_brane_jz_charge_scalar_never_couples_vector_gives_rank_one():
    # F-charge -6: spin weight +3 from the monopole, so c = s_h + 3.
    assert y.brane_jz_charge(0) == 3 and y.brane_jz_charge(-1) == 2 and y.brane_jz_charge(-3) == 0
    assert y.twisted_invariant_symmetric_matrices(y.brane_jz_charge(0)) == []
    assert y.single_brane_spectrum_pattern(y.brane_jz_charge(-1), np.random.default_rng(1)) == "rank1"
    # Consistent with the pole oracle: a spin-weight-3 field is nonzero at the pole only for m = -3.
    assert y.pole_nonzero_m(3, 4) == [-3]


def test_120_on_a_brane_gives_a_degenerate_pair_or_nothing():
    for c in range(-3, 4):
        basis = y.twisted_invariant_symmetric_matrices(c, antisymmetric=True)
        assert (len(basis) == 1) == (abs(c) <= 1)
        for B in basis:
            assert np.allclose(B, -B.T)
            sv = sorted(np.linalg.svd(B, compute_uv=False), reverse=True)
            assert abs(sv[0] - sv[1]) < 1e-10 and sv[2] < 1e-10


def test_c0_branes_generic_but_antipodal_branes_keep_the_pair():
    rng = np.random.default_rng(5)
    for _ in range(50):
        sv = y.two_brane_spectrum(rng, gamma=1.0, c=0)
        assert sv[1] - sv[2] > 1e-8 * sv[0] and sv[0] - sv[1] > 1e-8 * sv[0]
        sv = y.two_brane_spectrum(rng, gamma=np.pi, c=0)  # antipodal: same axis, J_z still exact
        assert min(abs(sv[0] - sv[1]), abs(sv[1] - sv[2])) < 1e-10 * sv[0]


def test_two_vector_branes_give_rank_two_with_separation_hierarchy():
    rng = np.random.default_rng(6)
    for gamma in (0.3, 0.1, 0.01):
        sv = np.array([y.two_brane_spectrum(rng, gamma=gamma, c=2) for _ in range(400)])
        assert np.all(sv[:, 2] < 1e-12 * sv[:, 0])  # one family stays massless
        assert 0.1 < np.median(sv[:, 1] / sv[:, 0]) / gamma ** 2 < 0.3  # m2/m1 ~ 0.2 gamma^2


def test_three_clustered_vector_branes_give_a_froggatt_nielsen_form():
    # Masses ~ (1, gamma^2, gamma^4): the ratios over gamma^2 and gamma^4 stay O(1) as gamma shrinks.
    scaling = y.clustered_brane_scaling(np.random.default_rng(7), samples=300)
    values = np.array(list(scaling.values()))
    assert np.all(values > 0.005) and np.all(values < 2)
    assert np.ptp(np.log10(values[:, 0])) < 0.3 and np.ptp(np.log10(values[:, 1])) < 0.3


def test_jz_breaking_texture_gives_hierarchical_masses():
    rng = np.random.default_rng(8)
    logs = np.median([np.log10(np.array(sv) / sv[0]) for sv in (y.fn_texture_spectrum(rng, 0.1) for _ in range(2000))],
                     axis=0)
    assert np.allclose(logs, [0, -2, -4], atol=0.3)


def test_proca_level_and_tuning():
    assert y.proca_lowest_level(2, 0) == -3  # round 3: Yang-Mills value -|n|/2 for |n| = 6
    assert y.proca_lowest_level(2, 3) == 0 and y.proca_lowest_level(1, 0) == 0
    assert y.higgs_tuning() < 1e-29


def test_report():
    report = y.demonstration_report()
    json.loads(json.dumps(report))
    assert report["jz_invariant_dimension"] == 2 and report["empirical_validation"] is False
    assert report["brane_jz_charge"] == {"scalar": 3, "normal_vector": 2}
    assert report["proca_M2r2_tuning_at_g2"] == 3
    assert report["limitations"] == y.LIMITATIONS


def test_single_brane_never_gives_three_distinct_masses_for_any_jz_charge():
    expected = {-4: "zero", -3: "zero", -2: "rank1", -1: "pair+zero", 0: "pair+one",
                1: "pair+zero", 2: "rank1", 3: "zero", 4: "zero"}
    for c, pattern in expected.items():
        assert y.single_brane_spectrum_pattern(c, np.random.default_rng(c + 20)) == pattern
    report = y.demonstration_report()
    # "distinct" also matches "two_distinct"; both would be failures here.
    assert all("distinct" not in p for p in report["single_brane_patterns_by_c"].values())
