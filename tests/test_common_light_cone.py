"""Checks for doc/derivations/common_light_cone_2026-09-25.md (Bogoliubov level).

Two oracles: a site-basis BdG matrix (checks the Fourier block reduction), and
a finite-difference Jacobian of the two-species lattice Gross-Pitaevskii
equations at unequal densities (checks the linearization and G itself).
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
        # h - mu_i + (Hartree terms): with mu_i = -6C_i + sum_j G_ij-type shifts, the
        # inter-species Hartree shift cancels, leaving -C_i A + 6C_i + G_ii on the diagonal.
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


def _gp_jacobian_frequencies(n, C_pair, g, nu, h=1e-6):
    """Frequencies from a finite-difference Jacobian of the two-species lattice GP flow."""
    sites = list(itertools.product(range(n), repeat=3))
    index = {s_: i for i, s_ in enumerate(sites)}
    M = len(sites)
    A = np.zeros((M, M))
    for s_ in sites:
        for axis in range(3):
            t = list(s_)
            t[axis] = (t[axis] + 1) % n
            A[index[s_], index[tuple(t)]] += 1
            A[index[tuple(t)], index[s_]] += 1
    g = np.asarray(g, dtype=float)
    mu = [-6 * C_pair[i] + sum(g[i, j] * nu[j] for j in range(2)) for i in range(2)]

    def flow(x):
        psi = [x[0:M] + 1j * x[M:2 * M], x[2 * M:3 * M] + 1j * x[3 * M:4 * M]]
        dens = [np.abs(p) ** 2 for p in psi]
        out = []
        for i in range(2):
            F = -C_pair[i] * A @ psi[i] - mu[i] * psi[i] + sum(g[i, j] * dens[j] for j in range(2)) * psi[i]
            d = -1j * F
            out.extend([d.real, d.imag])
        return np.concatenate(out)

    x0 = np.concatenate([np.full(M, np.sqrt(nu[0])), np.zeros(M), np.full(M, np.sqrt(nu[1])), np.zeros(M)])
    J = np.zeros((4 * M, 4 * M))
    for k in range(4 * M):
        e = np.zeros(4 * M)
        e[k] = h
        J[:, k] = (flow(x0 + e) - flow(x0 - e)) / (2 * h)
    return np.sort(np.abs(np.linalg.eigvals(J).imag))


@pytest.mark.parametrize("C_pair,g,nu", [((1.0, 1.7), [[1.0, 0.4], [0.4, 0.6]], (0.4, 2.3)),
                                         ((0.8, 0.8), [[0.5, 0.3], [0.3, 2.0]], (1.5, 0.2))])
def test_gp_jacobian_confirms_symmetric_hartree_matrix(C_pair, g, nu):
    n = 3
    G = c.hartree_matrix(g, nu)
    expected = [0.0] * 4
    for m in itertools.product(range(n), repeat=3):
        if any(m):
            k = [2 * math.pi * x / n for x in m]
            for f in c.formula_frequencies(k, C_pair, G):
                expected.extend([f, f])
    numeric = _gp_jacobian_frequencies(n, C_pair, g, nu)
    assert np.max(np.abs(np.sort(expected) - numeric)) < 1e-4
    # The asymmetric "physical potential" form g_ij nu_j in both slots is wrong at unequal densities.
    wrong = np.array([[g[0][0] * nu[0], g[0][1] * nu[1]], [g[0][1] * nu[1], g[1][1] * nu[1]]])
    k = [2 * math.pi / n, 0.0, 0.0]
    assert np.max(np.abs(np.array(c.formula_frequencies(k, C_pair, wrong))
                         - np.array(c.formula_frequencies(k, C_pair, G)))) > 1e-3


def test_tuned_manifold_gives_common_cone():
    rng = np.random.default_rng(11)
    for _ in range(50):
        kappa = rng.uniform(0.5, 2.0, size=2)
        mu_a = rng.uniform(0.1, 2.0)
        G = [[mu_a, 0.0], [0.0, kappa[0] * mu_a / kappa[1]]]
        assert c.common_cone(list(kappa), G)
        coupled = [[G[0][0], 1e-3], [1e-3, G[1][1]]]
        assert not c.common_cone(list(kappa), coupled)


def test_unstable_or_degenerate_mixtures_never_report_a_common_cone():
    assert not c.common_cone([1.0, 1.0], [[-1.0, 0.0], [0.0, -1.0]])
    assert not c.common_cone([1.0, 1.0], [[0.0, 0.0], [0.0, 0.0]])
    assert c.phonon_speeds([1.0, 1.0], [[1.0, 1.4], [1.4, 1.0]]) is None
    assert c.z2_point(1.0, 1.0, 1.2) is None


def test_finite_lattice_can_hide_long_wave_immiscibility():
    G = [[1.0, 1.05], [1.05, 1.0]]
    assert not c.miscible(G)
    # Equal hopping: instability needs eps_* = 4 sin^2(pi/n) < 2|m_-| = 0.1.
    assert 4 * math.sin(math.pi / 16) ** 2 > 0.1 > 4 * math.sin(math.pi / 20) ** 2
    assert c.lattice_scan(16, (1.0, 1.0), G)["unstable_modes"] == 0
    assert c.lattice_scan(20, (1.0, 1.0), G)["unstable_modes"] > 0


def test_su2_rank_one_at_unequal_densities():
    G = c.hartree_matrix([[1.0, 1.0], [1.0, 1.0]], (0.3, 1.7))
    assert abs(np.linalg.det(G)) < 1e-12
    k = [1e-3, 0.0, 0.0]
    freqs = c.bdg_frequencies(k, (1.0, 1.0), G)
    assert freqs[0] == pytest.approx(c.dispersion(k, 1.0), rel=1e-9)
