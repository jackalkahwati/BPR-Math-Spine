"""Tests for the Max-Cut phase-oscillator heuristic (bpr/optimization.py) and Bridge 9.

Until 2026-10 the coupling was ferromagnetic, so the solver minimized the cut and returned 0 on almost every
graph. These tests pin the corrected sign and the module's honest scope: a heuristic with no approximation
guarantee, checked against exact brute force on small graphs. They assert bounds, never exactness.
"""

import numpy as np
import pytest

import bpr.optimization as opt
from bpr.optimization import (
    MaxCutBPR, alpha_gw, brute_force_max_cut, local_search_polish, random_graph,
)


def complete_bipartite(a, b):
    W = np.zeros((a + b, a + b))
    W[:a, a:] = 1.0
    W[a:, :a] = 1.0
    return W


def cycle(n):
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1.0
    return W


SMALL_GRAPHS = {
    "K3,3": complete_bipartite(3, 3),
    "C6": cycle(6),
    "G(10,1/2) seed 3": random_graph(10, 0.5, seed=3),
}


@pytest.mark.parametrize("name", list(SMALL_GRAPHS))
def test_sign_regression_cut_at_least_half_the_edges(name):
    """The ferromagnetic sign returned cut 0 on these graphs; the corrected solver must beat a random cut."""
    W = SMALL_GRAPHS[name]
    mc = MaxCutBPR(W)
    cut = mc.cut_value(mc.extract_cut(mc.solve(seed=0)))
    assert cut >= W.sum() / 4  # |E|/2 for unit weights


def test_dynamics_are_gradient_descent_on_the_energy():
    """dphi/dt = -dE/dphi at lambda = 0, checked by central differences. Pins the antiferromagnetic sign."""
    W = random_graph(9, 0.5, seed=1)
    mc = MaxCutBPR(W, beta=1.3)
    phi = np.random.default_rng(2).uniform(-np.pi, np.pi, 9)
    h = 1e-6
    grad = np.array([(mc.energy(phi + h * e) - mc.energy(phi - h * e)) / (2 * h) for e in np.eye(9)])
    np.testing.assert_allclose(mc.phase_dynamics(phi, 0.0, lam=0.0), -grad, atol=1e-6)


def test_binary_energy_is_decreasing_in_the_cut():
    """On phases in {0, pi}: E = (beta/2) (sum W - 4 C), so the binary ground state is a maximum cut."""
    W = random_graph(8, 0.5, seed=4)
    mc = MaxCutBPR(W, beta=0.7)
    rng = np.random.default_rng(5)
    for _ in range(20):
        x = rng.integers(0, 2, 8)
        phi = np.where(x == 1, 0.0, np.pi)
        assert np.isclose(mc.energy(phi), 0.35 * (W.sum() - 4 * mc.cut_value(mc.extract_cut(phi))))


@pytest.mark.parametrize("W, expected", [
    (complete_bipartite(3, 3), 9.0),
    (cycle(6), 6.0),
    (cycle(5), 4.0),
    (np.ones((4, 4)) - np.eye(4), 4.0),
])
def test_brute_force_known_values(W, expected):
    value, x = brute_force_max_cut(W)
    assert value == expected
    assert MaxCutBPR(W).cut_value(x) == expected


def test_brute_force_refuses_large_graphs():
    with pytest.raises(ValueError):
        brute_force_max_cut(np.zeros((21, 21)))


@pytest.mark.parametrize("n", [8, 10, 12])
def test_bounds_half_weight_le_polished_cut_le_optimum(n):
    """For non-negative weights a 1-flip local optimum cuts at least half the weight; nothing beats brute force.
    The ratio to the optimum is recorded, not asserted: the heuristic is not exact for small n in general."""
    for seed in range(5):
        W = random_graph(n, 0.5, seed=100 * n + seed)
        mc = MaxCutBPR(W)
        polished = local_search_polish(W, mc.extract_cut(mc.solve(seed=seed)))
        cut = mc.cut_value(polished)
        best, _ = brute_force_max_cut(W)
        assert W.sum() / 4 <= cut <= best


def test_local_search_reaches_a_one_flip_optimum():
    W = random_graph(15, 0.4, seed=7)
    x = local_search_polish(W, np.zeros(15, dtype=int))
    mc = MaxCutBPR(W)
    base = mc.cut_value(x)
    for i in range(15):
        y = x.copy()
        y[i] ^= 1
        assert mc.cut_value(y) <= base + 1e-12


def test_solver_is_reproducible_with_a_seed():
    W = random_graph(12, 0.5, seed=8)
    a = MaxCutBPR(W).solve(seed=11)
    b = MaxCutBPR(W).solve(seed=11)
    np.testing.assert_array_equal(a, b)


def test_alpha_gw_constant():
    """Goemans-Williamson 1995: min over -1 <= rho < 1 of 2 arccos(rho) / (pi (1 - rho)), equivalently
    min over 0 <= t <= 1 of (1 + (2/pi) arcsin t) / (1 + t)."""
    a = alpha_gw()
    assert round(a, 6) == 0.878567
    t = np.linspace(0.0, 1.0, 200001)
    assert np.isclose(a, np.min((1 + (2 / np.pi) * np.arcsin(t)) / (1 + t)), atol=1e-9)


def test_module_docstring_states_its_scope():
    doc = opt.__doc__
    assert "Exact optima" not in doc and "~5 pct" not in doc
    assert "no approximation guarantee" in doc
    assert "Burer" in doc


def test_bridge_9_runs():
    """Bridge 9 raised TypeError before 2026-10. Its magic-number flags are a lookup, not derived."""
    from bpr.bridges.particles_matter import nuclear_shell_optimization

    r = nuclear_shell_optimization(A=16, Z=8, seed=0)
    assert r["doubly_magic"] and r["cut_graph_size"] == 16
    assert len(r["cut_partition"]) == r["cut_graph_size"]
    assert "uncertified" in r["cut_method"] and "not derived" in r["status"]
    r = nuclear_shell_optimization()
    assert r["cut_graph_size"] == 30 and r["cut_value"] > 0
