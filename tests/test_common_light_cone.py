"""Checks for doc/derivations/common_light_cone_2026-09-25.md (Bogoliubov level).

The oracle builds the full two-species BdG matrix on the n^3 lattice in the
site basis and diagonalizes it, independently of the module's 4x4 blocks.
"""

import itertools
import json
import math

import numpy as np
import pytest

import bpr.common_light_cone as c


def _site_bdg(n, C_pair, G):
    """2*(2M) x 2*(2M) site-basis BdG generator about uniform condensates."""
    sites = list(itertools.product(range(n), repeat=3))
    index = {s: i for i, s in enumerate(sites)}
    M = len(sites)
    A = np.zeros((M, M))
    for s in sites:
        for axis in range(3):
            t = list(s)
            t[axis] = (t[axis] + 1) % n
            A[index[s], index[tuple(t)]] += 1
            A[index[tuple(t)], index[s]] += 1
    blocks = []
    for i in range(2):
        # h - mu_i + 2 G_ii with h = -C A and the Hartree chemical potential -6C + G_ii + G_ij.
        blocks.append(-C_pair[i] * A + 6 * C_pair[i] * np.eye(M))
    G = np.asarray(G, dtype=float)
    D = np.block([[blocks[0] + G[0, 0] * np.eye(M), G[0, 1] * np.eye(M)],
                  [G[1, 0] * np.eye(M), blocks[1] + G[1, 1] * np.eye(M)]])
    O = np.block([[G[0, 0] * np.eye(M), G[0, 1] * np.eye(M)],
                  [G[1, 0] * np.eye(M), G[1, 1] * np.eye(M)]])
    return np.block([[D, O], [-O, -D]])


CASES = [((1.0, 1.0), [[1.0, 0.0], [0.0, 1.0]]),
         ((1.0, 2.0), [[2.0, 0.0], [0.0, 1.0]]),
         ((1.0, 1.0), [[1.0, 0.3], [0.3, 1.0]]),
         ((1.0, 1.3), [[1.0, 0.2], [0.2, 0.7]])]


@pytest.mark.parametrize("C_pair,G", CASES)
def test_block_formula_matches_site_basis_bdg(C_pair, G):
    n = 3
    numeric = np.sort(np.linalg.eigvals(_site_bdg(n, C_pair, G)).real)
    formula = []
    for m in itertools.product(range(n), repeat=3):
        k = [2 * math.pi * x / n for x in m]
        if any(m):
            freqs = c.formula_frequencies(k, C_pair, G)
            formula.extend(freqs + [-f for f in freqs])
        else:
            formula.extend([0.0] * 4)
    assert np.max(np.abs(np.sort(formula) - numeric)) < 1e-6


@pytest.mark.parametrize("C_pair,G", CASES)
def test_speed_matrix_matches_long_wave_slopes(C_pair, G):
    speeds = c.phonon_speeds(list(C_pair), G)
    numeric = c.long_wave_speeds(C_pair, G, step=1e-5)
    assert np.allclose(sorted(speeds), sorted(numeric), rtol=1e-6)


def test_common_cone_iff_decoupled_and_tuned():
    rng = np.random.default_rng(3)
    for _ in range(200):
        kappa = rng.uniform(0.5, 2.0, size=2)
        mu = rng.uniform(0.1, 2.0, size=2)
        mu_ab = rng.choice([0.0, rng.uniform(-0.5, 0.5)])
        G = [[mu[0], mu_ab], [mu_ab, mu[1]]]
        expected = mu_ab == 0.0 and abs(kappa[0] * mu[0] - kappa[1] * mu[1]) < 1e-12
        assert c.common_cone(list(kappa), G) == expected
    tuned = [[2.0, 0.0], [0.0, 1.0]]
    assert c.common_cone([1.0, 2.0], tuned)


def test_z2_point_splits_at_first_order():
    for mu_ab in (1e-3, 1e-2, 0.1):
        low, high = c.z2_point(1.0, 1.0, mu_ab)
        assert c.phonon_speeds([1.0, 1.0], [[1.0, mu_ab], [mu_ab, 1.0]]) == pytest.approx([low, high])
        assert (high - low) / math.sqrt(2) == pytest.approx(mu_ab, rel=0.02)  # (c+ - c-)/c ~ mu_ab/mu


def test_su2_point_has_a_quadratic_branch():
    G = [[1.0, 1.0], [1.0, 1.0]]
    for step in (1e-2, 1e-3):
        k = [step, 0.0, 0.0]
        freqs = c.bdg_frequencies(k, (1.0, 1.0), G)
        assert freqs[0] == pytest.approx(c.dispersion(k, 1.0), rel=1e-9)
    assert c.phonon_speeds([1.0, 1.0], G)[0] == 0.0


def test_immiscible_mixture_is_dynamically_unstable():
    G = [[1.0, 1.4], [1.4, 1.0]]
    assert not c.miscible(G)
    assert c.lattice_scan(8, (1.0, 1.0), G)["unstable_modes"] > 0
    assert c.lattice_scan(8, (1.0, 1.0), [[1.0, 0.9], [0.9, 1.0]])["unstable_modes"] == 0


def test_report_is_strict_json():
    report = c.demonstration_report()
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["empirical_validation"] is False
    names = {case["name"]: case for case in report["cases"]}
    assert names["decoupled_identical"]["common_cone"] is True
    assert names["z2_symmetric_coupled"]["common_cone"] is False
    assert names["su2_symmetric"]["common_cone"] is False
    for case in report["cases"]:
        assert case["lattice_scan_n8"]["max_formula_error"] < 1e-12


def test_input_validation():
    with pytest.raises(ValueError):
        c.speed_matrix([1.0, -1.0], [[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        c.speed_matrix([1.0, 1.0], [[1, 0.1], [0.2, 1]])
