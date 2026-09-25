"""Independent checks for doc/derivations/emergent_gauge_link_bosons_2026-09-25.md.

The oracle builds the full hard-core Fock space for 12-link clusters from
Pauli matrices, independently of the module's bitmask builders, and computes
the Schrieffer-Wolff terms by dense linear algebra. These are finite checks,
not evidence for any thermodynamic phase.
"""

import itertools
import json
from functools import reduce

import numpy as np
import pytest
import scipy.sparse as sparse

import bpr.link_boson_gauge as g


# ---------------------------------------------------------------------------
# Independent dense oracle (full 2^L Fock space)
# ---------------------------------------------------------------------------

def _dense_operators(L):
    """Hard-core annihilators on the full 2^L Fock space as sparse Kronecker products."""
    sm = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))  # basis (|0>, |1>)
    eye = sparse.identity(2, format="csr")
    ops = []
    for l in range(L):
        factors = [eye] * L
        factors[l] = sm
        ops.append(reduce(lambda a, b: sparse.kron(a, b, format="csr"), factors))
    return ops


def _dense_model(graph, t, U):
    L = len(graph["links"])
    b = _dense_operators(L)
    n = [(op.T @ op).tocsr() for op in b]
    dim = 2 ** L
    H0 = sparse.csr_matrix((dim, dim))
    for q, inc in zip(graph["targets"], graph["incident"]):
        Q = sum(n[l] for l in inc)
        D = Q - q * sparse.identity(dim, format="csr")
        H0 = H0 + U * (D @ D)
    V = sparse.csr_matrix((dim, dim))
    for inc in graph["incident"]:
        for a in inc:
            for c in inc:
                if a != c:
                    V = V - t * (b[a].T @ b[c])
    number = sum(n)
    return H0.tocsr(), V.tocsr(), number.tocsr(), n


def _dense_schrieffer_wolff(graph, t, U):
    H0, V, number, _ = _dense_model(graph, t, U)
    N = graph["particles"]
    diag0 = H0.diagonal()
    diagN = number.diagonal()
    sector = np.nonzero(np.isclose(diagN, N))[0]
    ice = sector[np.isclose(diag0[sector], 0.0)]
    Vs = V[sector][:, sector].toarray()
    pos = {s: i for i, s in enumerate(sector)}
    ip = [pos[s] for s in ice]
    inv = np.array([-1.0 / diag0[s] if diag0[s] > 0.5 * U else 0.0 for s in sector])
    R = np.diag(inv)
    P = np.zeros((len(sector), len(ip)))
    for k, i in enumerate(ip):
        P[i, k] = 1.0
    H2 = P.T @ Vs @ R @ Vs @ P
    H3 = P.T @ Vs @ R @ Vs @ R @ Vs @ P
    # Fock index -> link bitmask (link 0 is the most significant tensor factor).
    L = len(graph["links"])
    masks = []
    for s in ice:
        masks.append(sum(((int(s) >> (L - 1 - l)) & 1) << l for l in range(L)))
    full = (H0 + V)[sector][:, sector].toarray()
    return masks, H2, H3, full, np.max(np.abs(Vs[np.ix_(ip, ip)]))


def _align(module_masks, module_matrix, oracle_masks):
    perm = [module_masks.index(m) for m in oracle_masks]
    return module_matrix[np.ix_(perm, perm)]


CLUSTERS_12 = [("open_cube", g.open_cubes(1)), ("adamantane", g.adamantane())]


@pytest.mark.parametrize("name,graph", CLUSTERS_12)
def test_first_order_vanishes_on_the_ice_manifold(name, graph):
    _, _, _, _, pvp = _dense_schrieffer_wolff(graph, 0.1, 1.0)
    assert pvp == 0.0


@pytest.mark.parametrize("name,graph", CLUSTERS_12)
@pytest.mark.parametrize("t", [0.01, 0.037])
def test_second_order_theorem_matches_dense_oracle(name, graph, t):
    masks, H2, _, _, _ = _dense_schrieffer_wolff(graph, t, 1.0)
    basis, Heff = g.effective_hamiltonian(graph, t, 1.0)
    assert sorted(masks) == sorted(basis)
    assert np.max(np.abs(_align(basis, Heff.toarray(), masks) - H2)) < 1e-14


@pytest.mark.parametrize("name,graph", CLUSTERS_12)
@pytest.mark.parametrize("t", [0.01, 0.037])
def test_third_order_theorem_matches_dense_oracle(name, graph, t):
    masks, _, H3, _, _ = _dense_schrieffer_wolff(graph, t, 1.0)
    basis, A = g.analytic_third_order(graph, t, 1.0)
    assert np.max(np.abs(_align(basis, A, masks) - H3)) < 1e-15


@pytest.mark.parametrize("name,graph", CLUSTERS_12)
def test_full_spectrum_matches_dense_oracle(name, graph):
    _, _, _, Hs, _ = _dense_schrieffer_wolff(graph, 0.2, 1.0)
    _, H = g.full_hamiltonian(graph, 0.2, 1.0)
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(Hs)) - np.sort(np.linalg.eigvalsh(H.toarray())))) < 1e-10


def test_gauss_law_is_not_exact_but_ice_manifold_is_gapped():
    graph = g.open_cubes(1)
    H0, V, number, n = _dense_model(graph, 0.1, 1.0)
    for inc in graph["incident"]:
        Q = sum(n[l] for l in inc)
        assert abs((H0 + V) @ Q - Q @ (H0 + V)).max() > 0.05
        assert abs(H0 @ Q - Q @ H0).max() == 0.0
    assert all(x > 0 for x in g.gauss_commutator_norms(graph, 0.1, 1.0))
    assert all(x == 0 for x in g.gauss_commutator_norms(graph, 0.0, 1.0))
    diag = H0.diagonal()
    sector = np.isclose(number.diagonal(), graph["particles"])
    energies = sorted(set(np.round(diag[sector], 12)))
    assert energies[0] == 0.0 and energies[1] == 2.0


# ---------------------------------------------------------------------------
# Larger clusters: module Schrieffer-Wolff against the analytic theorems
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_cubes():
    return g.open_cubes(2)


def test_two_cubes_second_and_third_order(two_cubes):
    t, U = 0.013, 1.0
    ice, H2, H3 = g.schrieffer_wolff(two_cubes, t, U)
    basis, E2 = g.effective_hamiltonian(two_cubes, t, U)
    _, A3 = g.analytic_third_order(two_cubes, t, U)
    perm = [basis.index(m) for m in ice]
    assert np.max(np.abs(H2 - E2.toarray()[np.ix_(perm, perm)])) < 1e-15
    assert np.max(np.abs(H3 - A3[np.ix_(perm, perm)])) < 1e-16
    shifts = sorted({round(g.third_order_plaquette_shift(two_cubes, c, t, U) / (t ** 3 / U ** 2), 9)
                     for c in g.four_cycles(two_cubes)})
    assert shifts == [3.0, 4.5, 6.0]


def test_square_torus_second_order_and_triangle_guard():
    graph = g.square_torus(3)
    assert not g.is_triangle_free(graph)
    ice, H2, _ = g.schrieffer_wolff(graph, 0.02, 1.0)
    basis, E2 = g.effective_hamiltonian(graph, 0.02, 1.0)
    perm = [basis.index(m) for m in ice]
    assert np.max(np.abs(H2 - E2.toarray()[np.ix_(perm, perm)])) < 1e-15
    with pytest.raises(ValueError):
        g.analytic_third_order(graph, 0.02, 1.0)


@pytest.mark.parametrize("name,graph", CLUSTERS_12)
def test_perturbative_error_scaling(name, graph):
    summary = g.compare_low_energy(graph, 1.0, (0.01, 0.02))
    assert 7.0 < summary["second_order_error_ratio"] < 9.5
    assert 14.0 < summary["third_order_error_ratio"] < 18.5
    for record in summary["records"]:
        assert record["analytic_vs_numeric_second_order"] < 1e-15
        assert record["third_order_error"] < record["second_order_error"]


# ---------------------------------------------------------------------------
# Structure: cycles, line-graph equivalence, RK point, lattice coefficients
# ---------------------------------------------------------------------------

def _brute_cycles(graph, length):
    nbrs = [set() for _ in graph["vertices"]]
    for a, b in graph["links"]:
        nbrs[a].add(b)
        nbrs[b].add(a)
    found = set()
    for perm in itertools.permutations(range(len(graph["vertices"])), length):
        if perm[0] != min(perm):
            continue
        if all(perm[(i + 1) % length] in nbrs[perm[i]] for i in range(length)):
            found.add(frozenset(frozenset((perm[i], perm[(i + 1) % length])) for i in range(length)))
    return len(found)


def test_cycle_enumeration_against_brute_force():
    cube = g.open_cubes(1)
    assert len(g.four_cycles(cube)) == _brute_cycles(cube, 4) == 6
    assert len(g.cycles_of_length(cube, 6)) == _brute_cycles(cube, 6)
    ad = g.adamantane()
    assert len(g.four_cycles(ad)) == 0
    assert len(g.cycles_of_length(ad, 6)) == 4
    assert len(g.four_cycles(g.cubic_torus(6))) == 3 * 216
    with pytest.raises(ValueError):
        g.cubic_torus(5)  # q=3 on 125 vertices gives an odd total charge


def test_line_graph_equivalence_uniform_targets():
    graph = g.square_torus(3)
    basis = g.sector_basis(graph)
    pairs = set()
    for inc in graph["incident"]:
        for a, b in itertools.combinations(inc, 2):
            pairs.add((a, b))
    values = set()
    for mask in basis[:2000]:
        occ = [(mask >> l) & 1 for l in range(len(graph["links"]))]
        charging = sum((sum(occ[l] for l in inc) - q) ** 2
                       for q, inc in zip(graph["targets"], graph["incident"]))
        repulsion = 2 * sum(occ[a] * occ[b] for a, b in pairs)
        values.add(charging - repulsion)
    assert len(values) == 1


@pytest.mark.parametrize("graph", [g.open_cubes(1), g.adamantane(), g.square_torus(3)])
def test_rk_point_ground_states(graph):
    report = g.rk_point_report(graph)
    assert report["zero_energy_states"] == report["components"]
    assert report["min_eigenvalue"] > -1e-10
    assert report["equal_superposition_residual"] < 1e-12


def test_lattice_coefficients():
    cubic = g.cubic_torus(6)
    assert g.is_bipartite(g.cubic_torus(4)) and not g.is_bipartite(g.cubic_torus(5, q=2))
    assert g.is_triangle_free(cubic)
    t, U = 0.1, 1.0
    square = g.four_cycles(cubic)[0]
    assert g.third_order_plaquette_shift(cubic, square, t, U) == pytest.approx(12 * t ** 3 / U ** 2)
    assert g.ring_exchange_coefficient(t, U) == pytest.approx(2 * t ** 2 / U)
    assert g.hexagon_coefficient(t, U) == pytest.approx(3 * t ** 3 / U ** 2)
    per_vertex_2 = g.effective_constant(cubic, t, U) / len(cubic["vertices"])
    per_vertex_3 = g.third_order_constant(cubic, t, U) / len(cubic["vertices"])
    assert per_vertex_2 == pytest.approx(-4.5 * t ** 2 / U)
    assert per_vertex_3 == pytest.approx(-9 * t ** 3 / U ** 2)


# ---------------------------------------------------------------------------
# Validation and report
# ---------------------------------------------------------------------------

def test_graph_validation():
    with pytest.raises(ValueError):
        g.make_graph([0, 1], [(0, 1), (1, 0)], 1)  # multigraph
    with pytest.raises(ValueError):
        g.make_graph([0, 1, 2], [(0, 1), (1, 2)], {0: 1, 1: 1, 2: 1})  # odd total
    with pytest.raises(ValueError):
        g.make_graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0)], {0: 2, 1: 1, 2: 2, 3: 1})
    with pytest.raises(ValueError):
        g.make_graph([0, 1], [(0, 0)], 0)


def test_demonstration_report_is_strict_json():
    report = g.demonstration_report()
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report
    assert report["empirical_validation"] is False
    assert report["coulomb_phase_established"] is False
    assert report["limitations"] == g.LIMITATIONS
    for comparison in report["low_energy_comparison"]:
        assert comparison["third_order_error_ratio"] > comparison["second_order_error_ratio"]
