"""Independent graph-coordinate oracles; never allocate the seven-link space."""
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from bpr import gauge_heat_kernel as inherited
from bpr import gauge_two_plaquette as gp


GROUPS = (5, 8, 9)
IDENTITY = (0, 1)
VERTICES = ("A", "B", "Ltop", "Lbot", "Rbot", "Rtop")
EDGES = {
    "a": ("A", "Ltop"), "b": ("Ltop", "Lbot"),
    "c": ("Lbot", "B"), "d": ("A", "B"),
    "e": ("B", "Rbot"), "f": ("Rbot", "Rtop"),
    "h": ("Rtop", "A"),
}
TREE = ("a", "b", "d", "e", "f")
LOOPS = ((('a', 1), ('b', 1), ('c', 1), ('d', -1)),
         (('d', 1), ('e', 1), ('f', 1), ('h', 1)))
ATOL = 2e-10


# No inherited or production group, conjugacy, holonomy, or action helpers.
def _elements(n):
    return tuple((k, sign) for sign in (1, -1) for k in range(n))


def _mul(a, b, n):
    return ((a[0] + a[1] * b[0]) % n, a[1] * b[1])


def _inv(a, n):
    return ((-a[1] * a[0]) % n, a[1])


def _conjugate(t, a, n):
    return _mul(_mul(t, a, n), _inv(t, n), n)


def _walk(links, path, n):
    value = IDENTITY
    for edge, direction in path:
        factor = links[edge] if direction == 1 else _inv(links[edge], n)
        value = _mul(value, factor, n)
    return value


def _holonomies(links, n):
    return tuple(_walk(links, path, n) for path in LOOPS)


def _slice(u, v):
    links = dict.fromkeys(EDGES, IDENTITY)
    links.update(c=u, h=v)
    return links


def _gauge(links, values, n):
    return {edge: _mul(_mul(values[source], links[edge], n),
                       _inv(values[target], n), n)
            for edge, (source, target) in EDGES.items()}


def _tree_fix(links, n):
    # Traverse the actual tree, rather than repeating production path formulas.
    paths = {"A": IDENTITY}
    while len(paths) < len(VERTICES):
        previous = len(paths)
        for edge in TREE:
            source, target = EDGES[edge]
            if source in paths and target not in paths:
                paths[target] = _mul(paths[source], links[edge], n)
            if target in paths and source not in paths:
                paths[source] = _mul(paths[target], _inv(links[edge], n), n)
        assert len(paths) > previous
    return _gauge(links, paths, n), paths


def _shift(links, edge, g, n):
    result = dict(links)
    result[edge] = _mul(g, links[edge], n)
    return result


@lru_cache(None)
def _weights(n):
    # Integer weights make every central gauge-covariance comparison exact.
    reflections = sorted({_conjugate(t, (0, -1), n) for t in _elements(n)})
    denominator = len(reflections)
    terms = [(denominator, (1, 1)), (denominator, (n - 1, 1))]
    terms.extend((1, g) for g in reflections)
    return tuple(terms), denominator


@lru_cache(None)
def _orbit_key(n, pair):
    return min(tuple(_conjugate(t, u, n) for u in pair) for t in _elements(n))


@lru_cache(None)
def _orbits(n):
    pairs = tuple(product(_elements(n), repeat=2))
    buckets = {}
    for row, pair in enumerate(pairs):
        buckets.setdefault(_orbit_key(n, pair), []).append(row)
    return pairs, tuple(tuple(rows) for rows in buckets.values())


def _central_orbit_counter(links, edge, n):
    terms, _ = _weights(n)
    result = Counter()
    for weight, g in terms:
        pair = _holonomies(_shift(links, edge, g, n), n)
        result[_orbit_key(n, pair)] += weight
    return result


@lru_cache(None)
def _edge_matrix(n, edge):
    pairs, _ = _orbits(n)
    index = {pair: row for row, pair in enumerate(pairs)}
    terms, denominator = _weights(n)
    result = 3 * np.eye(len(pairs))
    for col, (u, v) in enumerate(pairs):
        for numerator, g in terms:
            shifted = _shift(_slice(u, v), edge, g, n)
            row = index[_holonomies(shifted, n)]
            result[row, col] -= numerator / denominator
    return result


def _configurations(n):
    raw = (
        ((1, 1), (2, -1), (3, 1), (0, -1), (2, 1), (1, -1), (3, -1)),
        ((3, -1), (1, 1), (2, -1), (2, 1), (1, -1), (3, 1), (2, 1)),
        ((2, 1), (3, 1), (1, 1), (1, -1), (3, -1), (0, -1), (1, 1)),
        ((0, -1), (1, -1), (2, -1), (3, -1), (1, 1), (2, 1), (3, 1)),
    )
    return [dict(zip(EDGES, ((k % n, sign) for k, sign in row))) for row in raw]


@pytest.fixture(params=GROUPS)
def n(request):
    return request.param


def test_independent_group_oracle(n):
    for a in _elements(n):
        assert _mul(a, _inv(a, n), n) == IDENTITY
        assert _mul(_inv(a, n), a, n) == IDENTITY
        for b in _elements(n):
            for c in _elements(n):
                assert _mul(_mul(a, b, n), c, n) == _mul(a, _mul(b, c, n), n)


def test_canonical_graph_all_seven_fixed_element_actions(n):
    for u, v in product(_elements(n), repeat=2):
        links = _slice(u, v)
        assert gp.canonical_links(n, u, v) == links
        assert gp.holonomies(n, links) == (u, v)
        for edge in EDGES:
            for g in _elements(n):
                moved = _shift(links, edge, g, n)
                expected = _holonomies(moved, n)
                fixed, _ = _tree_fix(moved, n)
                assert (fixed['c'], fixed['h']) == expected
                assert gp.link_action(n, edge, g, u, v) == expected
                if edge == 'd':
                    assert _mul(*expected, n) == _mul(u, v, n)


def test_every_vertex_gauge_action_and_path_transporter_tree_fix(n):
    for links in _configurations(n):
        base = _holonomies(links, n)
        assert gp.holonomies(n, links) == base
        expected_links, expected_values = _tree_fix(links, n)
        actual = gp.tree_fix(n, links)
        assert actual['links'] == expected_links
        assert actual['vertex_values'] == expected_values
        assert all(actual['links'][edge] == IDENTITY for edge in TREE)
        assert (actual['links']['c'], actual['links']['h']) == base
        for vertex in VERTICES:
            for t in _elements(n):
                values = dict.fromkeys(VERTICES, IDENTITY)
                values[vertex] = t
                transformed = _gauge(links, values, n)
                assert gp.gauge_transform(n, links, values) == transformed
                root = values['A']
                expected = tuple(_conjugate(root, u, n) for u in base)
                assert _holonomies(transformed, n) == expected
                assert gp.holonomies(n, transformed) == expected
                fixed = gp.tree_fix(n, transformed)['links']
                assert all(fixed[edge] == IDENTITY for edge in TREE)
                assert (fixed['c'], fixed['h']) == expected
        # Simultaneous nontrivial transformations exercise both link endpoints.
        values = dict(zip(VERTICES, _elements(n)[1:7]))
        assert gp.gauge_transform(n, links, values) == _gauge(links, values, n)


def test_central_weighted_actions_off_slice_at_every_vertex(n):
    matrices = {edge: gp.edge_laplacian(n, edge) for edge in EDGES}
    pairs, orbits = _orbits(n)
    index = {pair: row for row, pair in enumerate(pairs)}
    _, denominator = _weights(n)
    for links in _configurations(n):
        base_pair = _holonomies(links, n)
        for edge in EDGES:
            expected = _central_orbit_counter(links, edge, n)
            # All class-function matrix elements, not one favored probe.
            col = index[base_pair]
            translation = -matrices[edge][:, col].copy()
            translation[col] += 3
            for rows in orbits:
                key = _orbit_key(n, pairs[rows[0]])
                assert_allclose(sum(translation[list(rows)]) * denominator,
                                expected[key], rtol=0, atol=5e-12)
            for vertex in VERTICES:
                for t in _elements(n):
                    values = dict.fromkeys(VERTICES, IDENTITY)
                    values[vertex] = t
                    transformed = _gauge(links, values, n)
                    assert _central_orbit_counter(transformed, edge, n) == expected


def test_single_fixed_element_is_not_a_gauge_invariant_action(n):
    # Root unchanged. Conjugating the shift by a nonroot path changes its orbit.
    links = _slice(IDENTITY, (0, -1))
    values = dict.fromkeys(VERTICES, IDENTITY)
    values['Ltop'] = (1, 1)
    transformed = _gauge(links, values, n)
    before = _holonomies(_shift(links, 'b', (0, -1), n), n)
    after = _holonomies(_shift(transformed, 'b', (0, -1), n), n)
    assert before == ((0, -1), (0, -1))
    assert after == (((n - 2) % n, -1), (0, -1))
    assert _orbit_key(n, before) != _orbit_key(n, after)
    assert _central_orbit_counter(links, 'b', n) == _central_orbit_counter(transformed, 'b', n)


def test_independent_burnside_orbits_and_normalized_projector(n):
    data = gp.orbit_basis(n)
    pairs, expected_orbits = _orbits(n)
    assert tuple(data['elements']) == _elements(n)
    assert tuple(data['pairs']) == pairs
    assert tuple(tuple(rows) for rows in data['orbits']) == expected_orbits
    size = len(pairs)
    projector = np.zeros((size, size))
    index = {pair: row for row, pair in enumerate(pairs)}
    fixed_pair_counts = []
    for t in _elements(n):
        destinations = [index[tuple(_conjugate(t, u, n) for u in pair)] for pair in pairs]
        fixed_pair_counts.append(sum(row == col for col, row in enumerate(destinations)))
        projector[destinations, np.arange(size)] += 1 / (2 * n)
    burnside = sum(fixed_pair_counts) // (2 * n)
    assert sum(fixed_pair_counts) % (2 * n) == 0
    assert data['count'] == data['burnside_count'] == burnside == {5: 22, 8: 64, 9: 56}[n]
    expected_histogram = {
        5: {1: 1, 2: 12, 5: 3, 10: 6},
        8: {1: 4, 2: 30, 4: 12, 8: 18},
        9: {1: 1, 2: 40, 9: 3, 18: 12},
    }[n]
    assert Counter(map(len, expected_orbits)) == expected_histogram
    # Same separate face classes need not mean the same joint gauge orbit.
    equal_reflections = ((0, -1), (0, -1))
    distinct_reflections = ((0, -1), (2, -1))
    assert min(_conjugate(t, (0, -1), n) for t in _elements(n)) == min(
        _conjugate(t, (2, -1), n) for t in _elements(n))
    assert _orbit_key(n, equal_reflections) != _orbit_key(n, distinct_reflections)
    basis = data['basis']
    independent_basis = np.zeros_like(basis)
    for col, rows in enumerate(expected_orbits):
        independent_basis[list(rows), col] = 1 / np.sqrt(len(rows))
    assert_allclose(basis, independent_basis, rtol=0, atol=5e-12)
    assert_allclose(basis.T @ basis, np.eye(burnside), rtol=0, atol=5e-12)
    assert_allclose(basis @ basis.T, projector, rtol=0, atol=5e-12)
    assert_allclose(projector @ projector, projector, rtol=0, atol=5e-12)


def test_each_raw_edge_matrix_and_shared_coupling_negative_control(n):
    pairs, _ = _orbits(n)
    expected = {edge: _edge_matrix(n, edge) for edge in EDGES}
    actual = {edge: gp.edge_laplacian(n, edge) for edge in EDGES}
    for edge in EDGES:
        assert_allclose(actual[edge], expected[edge], rtol=0, atol=5e-12)
        assert_allclose(actual[edge], actual[edge].T, rtol=0, atol=5e-12)
        assert_allclose(actual[edge] @ np.ones(len(pairs)), 0, rtol=0, atol=5e-12)
    assert_allclose(actual['a'], actual['b'], rtol=0, atol=5e-12)
    assert_allclose(actual['a'], actual['c'], rtol=0, atol=5e-12)
    assert_allclose(actual['e'], actual['f'], rtol=0, atol=5e-12)
    assert_allclose(actual['e'], actual['h'], rtol=0, atol=5e-12)
    outer_identity = np.array([_mul(u, v, n) == IDENTITY for u, v in pairs], dtype=float)
    assert_allclose(actual['d'] @ outer_identity, 0, rtol=0, atol=5e-12)
    true_electric = sum(actual.values())
    independent_squares = 4 * (actual['a'] + actual['h'])
    origin = pairs.index((IDENTITY, IDENTITY))
    assert_allclose((true_electric @ outer_identity)[origin], 18, rtol=0, atol=5e-12)
    assert_allclose((independent_squares @ outer_identity)[origin], 24, rtol=0, atol=5e-12)
    # A real simultaneous change of both holonomies is absent from a tensor sum.
    changed = pairs.index(((n - 1, 1), (1, 1)))
    assert actual['d'][changed, origin] == -1
    assert independent_squares[changed, origin] == 0


def test_single_square_four_link_and_tree_controls(n):
    elements = _elements(n)
    square_classes = {}
    for g in elements:
        square_classes.setdefault(min(_conjugate(t, g, n) for t in elements), []).append(g)
    assert len(square_classes) == {5: 4, 8: 7, 9: 6}[n]
    index = {g: row for row, g in enumerate(elements)}
    delta = 3 * np.eye(len(elements))
    terms, denominator = _weights(n)
    for col, h in enumerate(elements):
        for weight, g in terms:
            delta[index[_mul(g, h, n)], col] -= weight / denominator
    assert_allclose(delta, inherited.central_laplacian(n), rtol=0, atol=5e-12)
    pairs, _ = _orbits(n)
    total = sum(gp.edge_laplacian(n, edge) for edge in EDGES)
    for members in square_classes.values():
        single = np.array([g in members for g in elements], dtype=float)
        for face in (0, 1):
            lifted = np.array([single[index[pair[face]]] for pair in pairs])
            expected = np.array([(4 * delta @ single)[index[pair[face]]] for pair in pairs])
            assert_allclose(total @ lifted, expected, rtol=0, atol=5e-12)
    # On the tree subgraph every assignment, including each shifted edge, fixes
    # to the unique all-identity representative. No fictitious tree cycle remains.
    for links in _configurations(n):
        for edge in TREE:
            for g in elements:
                moved = _shift(links, edge, g, n)
                expected, values = _tree_fix(moved, n)
                actual = gp.tree_fix(n, moved)
                assert actual['vertex_values'] == values
                assert all(expected[label] == actual['links'][label] == IDENTITY for label in TREE)


@pytest.fixture(scope='module', params=GROUPS)
def built_model(request):
    return gp.model(request.param, 1.3)


def test_model_operators_invariant_subspace_and_magnetic_oracle(built_model):
    data = built_model
    n, lam = data['n'], data['lam']
    pairs, orbits = _orbits(n)
    basis = data['basis']
    electric = sum(_edge_matrix(n, edge) for edge in EDGES) / lam
    # Independent E1 formula, not the inherited character table.
    character = lambda g: 2 * np.cos(2 * np.pi * g[0] / n) if g[1] == 1 else 0.0
    potential = np.array([2 - character(u) / 2 - character(v) / 2 for u, v in pairs])
    magnetic = np.diag(potential)
    assert data['coordinate_dimension'] == (2 * n) ** 2 <= 324
    assert data['physical_dimension'] == {5: 22, 8: 64, 9: 56}[n]
    assert data['physical_dimension'] > {5: 4, 8: 7, 9: 6}[n] ** 2
    assert set(data['reflection_class']) == {g for _, g in _weights(n)[0] if g[1] == -1}
    if n == 8:
        assert {g[0] for g in data['reflection_class']} == {0, 2, 4, 6}
    for key, expected in [('coordinate_electric', electric),
                          ('coordinate_magnetic', magnetic),
                          ('coordinate_hamiltonian', electric + lam * magnetic)]:
        assert_allclose(data[key], expected, rtol=0, atol=ATOL)
    for key, expected in [('electric', electric), ('magnetic', magnetic),
                          ('hamiltonian', electric + lam * magnetic)]:
        assert_allclose(data[key], basis.T @ expected @ basis, rtol=0, atol=ATOL)
        assert_allclose(expected @ basis, basis @ data[key], rtol=0, atol=ATOL)
        assert_allclose(data[key], data[key].T, rtol=0, atol=ATOL)
    assert_allclose(data['magnetic'], np.diag(data['magnetic_diagonal']), rtol=0, atol=5e-12)
    for rows in orbits:
        assert_allclose(potential[list(rows)], potential[rows[0]], rtol=0, atol=5e-12)
    constant = basis.T @ np.ones(len(pairs))
    assert_allclose(data['electric'] @ constant, 0, rtol=0, atol=ATOL)
    values = np.linalg.eigvalsh(data['electric'])
    assert min(values) >= -ATOL
    assert np.count_nonzero(np.abs(values) < ATOL) == 1
    assert min(data['magnetic_diagonal']) >= 0
    assert np.count_nonzero(data['magnetic_diagonal'] == 0) == 1
    du, dv, shared = (data[key] for key in ('laplacian_u', 'laplacian_v', 'laplacian_shared'))
    for first, second in ((du, dv), (du, shared), (dv, shared)):
        assert_allclose(first @ second, second @ first, rtol=0, atol=ATOL)
    assert np.linalg.norm(data['electric'] @ data['magnetic'] -
                          data['magnetic'] @ data['electric']) > 1


def test_transfer_full_fixed_refinement_and_independent_bounds(built_model):
    data = built_model
    a = data['electric']
    b = data['lam'] * data['magnetic']
    h = a + b
    ab = a @ b - b @ a
    first = np.linalg.norm(a @ ab - ab @ a, 2)
    second = np.linalg.norm(b @ ab - ab @ b, 2)
    constant = first / 12 + second / 24
    scale = np.linalg.norm(a, 2) + np.linalg.norm(b, 2)
    steps = (0.08, 0.04, 0.02)
    report = gp.diagnostics(data['n'], data['lam'], steps)
    assert tuple(report['steps']) == steps
    assert_allclose(report['analytic_constant_C'], constant, rtol=0, atol=ATOL)
    assert_allclose(report['commutator_norms']['a_ab'], first, rtol=0, atol=ATOL)
    assert_allclose(report['commutator_norms']['b_ab'], second, rtol=0, atol=ATOL)
    assert len(report['transfer_diagnostics']) == 3
    effective_values, errors = [], []
    for dt, record in zip(steps, report['transfer_diagnostics']):
        half = expm(-dt * b / 2)
        expected = half @ expm(-dt * a) @ half
        transfer = gp.transfer(data, dt)
        assert_allclose(transfer, expected, rtol=0, atol=ATOL)
        assert_allclose(transfer, transfer.T, rtol=0, atol=ATOL)
        eigenvalues = np.linalg.eigvalsh(transfer)
        assert min(eigenvalues) > 0
        assert max(eigenvalues) <= 1 + ATOL
        bound = np.exp(-dt * scale)
        assert min(eigenvalues) >= bound - ATOL
        effective = gp.effective_hamiltonian(transfer, dt)
        effective_values.append(effective)
        assert_allclose(expm(-dt * effective), transfer, rtol=0, atol=ATOL)
        error = np.linalg.norm(effective - h, 2)
        errors.append(error)
        exact_error = np.linalg.norm(transfer - expm(-dt * h), 2)
        assert exact_error <= dt ** 3 * constant + ATOL
        assert error <= dt ** 2 * np.exp(dt * scale) * constant + ATOL
        assert_allclose(record['transfer_min_eigenvalue'], min(eigenvalues), rtol=0, atol=ATOL)
        assert_allclose(record['transfer_max_eigenvalue'], max(eigenvalues), rtol=0, atol=ATOL)
        assert_allclose(record['condition_number'], max(eigenvalues) / min(eigenvalues), rtol=0, atol=ATOL)
        assert_allclose(record['analytic_transfer_min_bound'], bound, rtol=0, atol=ATOL)
        assert_allclose(record['effective_hamiltonian_error'], error, rtol=0, atol=ATOL)
        assert_allclose(record['transfer_exact_error'], exact_error, rtol=0, atol=ATOL)
        assert_allclose(record['analytic_transfer_error_bound'], dt ** 3 * constant, rtol=0, atol=ATOL)
        assert_allclose(record['analytic_generator_error_bound'],
                        dt ** 2 * np.exp(dt * scale) * constant, rtol=0, atol=ATOL)
        assert record['positivity_resolution'] >= 64 * np.finfo(float).eps * len(a) * max(eigenvalues) * (1 - 1e-10)
        assert record['transfer_min_eigenvalue'] > record['positivity_resolution']
    assert errors[2] < errors[1] < errors[0]
    assert report['refinement_decreasing']
    assert_allclose(report['error_ratios'], [errors[0] / errors[1], errors[1] / errors[2]], rtol=0, atol=ATOL)
    differences = [np.linalg.norm(left - right, 2)
                   for left, right in zip(effective_values, effective_values[1:])]
    assert_allclose(report['refinement_differences'], differences, rtol=0, atol=ATOL)
    assert np.all(np.array(differences) <= np.array(report['analytic_refinement_bounds']) + ATOL)
    assert_allclose(gp.transfer(data, 0), np.eye(len(a)), rtol=0, atol=5e-12)


def test_group_cap_precedes_enumeration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported group reached enumeration')
    monkeypatch.setattr(gp, 'elements', forbidden)
    for bad in (12, 1000000, 0, -5, 5.0, True, '5', None):
        for operation in (gp.orbit_basis, gp.model):
            with pytest.raises(ValueError):
                operation(bad)
        with pytest.raises(ValueError):
            gp.edge_laplacian(bad, 'a')


@pytest.mark.parametrize('bad', [(5, 1), (-1, 1), (0, 0), (0, 2), (True, 1), (0, True),
                                  (1.0, 1), [0, 1], (0,), None])
def test_invalid_elements_rejected(bad):
    with pytest.raises(ValueError):
        gp.canonical_links(5, bad, IDENTITY)
    with pytest.raises(ValueError):
        gp.link_action(5, 'd', bad, IDENTITY, IDENTITY)


def test_graph_key_and_edge_validation():
    links = _slice(IDENTITY, IDENTITY)
    values = dict.fromkeys(VERTICES, IDENTITY)
    for bad_links in ({}, [], dict(links, extra=IDENTITY), dict(links, a=(5, 1))):
        for operation in (gp.holonomies, gp.tree_fix):
            with pytest.raises(ValueError):
                operation(5, bad_links)
    for bad_values in ({}, [], dict(values, extra=IDENTITY), dict(values, A=(0, 0))):
        with pytest.raises(ValueError):
            gp.gauge_transform(5, links, bad_values)
    for bad_edge in ('g', '', 1, None, ['a']):
        with pytest.raises(ValueError):
            gp.edge_laplacian(5, bad_edge)
        with pytest.raises(ValueError):
            gp.link_action(5, bad_edge, IDENTITY, IDENTITY, IDENTITY)


@pytest.mark.parametrize('bad', [0, -1, True, None, '1.3', float('nan'), float('inf')])
def test_invalid_couplings(bad):
    with pytest.raises(ValueError):
        gp.model(5, bad)


@pytest.mark.parametrize('bad', [np.nextafter(0., 1.), 1e-200, 1e200, np.finfo(float).max])
def test_unresolved_couplings_raise_without_flooring(bad):
    with pytest.raises(FloatingPointError):
        gp.model(5, bad)


@pytest.mark.parametrize('bad', [-1, True, None, '0.08', float('nan'), float('inf')])
def test_transfer_time_validation(built_model, bad):
    with pytest.raises(ValueError):
        gp.transfer(built_model, bad)


def test_log_domain_resolution_and_inherited_cap():
    assert_allclose(gp.effective_hamiltonian(np.eye(64), 0.1), 0, rtol=0, atol=5e-12)
    with pytest.raises(ValueError):
        inherited.effective_hamiltonian(np.eye(25), 0.1)
    for matrix in (np.eye(325), np.zeros((0, 0)), np.zeros((2, 3)), [1, 2]):
        with pytest.raises(ValueError):
            gp.effective_hamiltonian(matrix, 0.1)
    for matrix in (np.diag([1., 0.]), np.diag([1., -0.1]), np.diag([1., 1e-16]),
                   np.diag([1., np.nextafter(0., 1.)])):
        with pytest.raises((ValueError, FloatingPointError)):
            gp.effective_hamiltonian(matrix, 0.1)
    for matrix in (np.array([[1., 1.], [0., 1.]]), np.diag([1., np.nan]), np.diag([1., np.inf])):
        with pytest.raises((ValueError, FloatingPointError)):
            gp.effective_hamiltonian(matrix, 0.1)
    for bad in (0, -1, True, None, '0.1', np.nan, np.inf):
        with pytest.raises(ValueError):
            gp.effective_hamiltonian(np.eye(2), bad)
    with pytest.raises(FloatingPointError):
        gp.effective_hamiltonian(np.eye(2), np.nextafter(0., 1.))
    # A resolved small positive eigenvalue is logged raw, not clipped.
    matrix = np.diag([1., 1e-10])
    assert_allclose(gp.effective_hamiltonian(matrix, 0.2),
                    np.diag([0., -np.log(1e-10) / 0.2]), rtol=0, atol=ATOL)


@pytest.mark.parametrize('steps', [(), [], 0.08, True, '0.08', (0,), (-0.1,), (np.nan,), (np.inf,)])
def test_diagnostics_steps_validation(steps):
    with pytest.raises(ValueError):
        gp.diagnostics(5, 1.3, steps)


def test_demo_stdout_only_text_and_strict_json(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/demo_gauge_two_plaquette.py'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONWARNINGS='error')
    def reject_constant(value):
        raise AssertionError('nonfinite JSON constant: ' + value)
    for flags in ([], ['--json']):
        completed = subprocess.run([sys.executable, '-W', 'error', str(script)] + flags,
                                   cwd=str(tmp_path), env=env, capture_output=True,
                                   text=True, timeout=120, check=True)
        assert completed.stderr == ''
        assert completed.stdout.strip()
        assert list(tmp_path.iterdir()) == []
        if flags:
            report = json.loads(completed.stdout, parse_constant=reject_constant)
            assert report['model_id'] == gp.MODEL_ID
            assert len(report['controls']) == 3
            assert [row['n'] for row in report['controls']] == list(GROUPS)
            assert report['scope'] and report['limitations']
        else:
            assert 'D5' in completed.stdout and 'D8' in completed.stdout and 'D9' in completed.stdout


def test_electric_heat_semigroup_after_removing_magnetic_halves(built_model):
    data = built_model
    b = data['lam'] * data['magnetic']
    recovered = []
    for dt in (0.02, 0.04, 0.06):
        inverse_half = expm(dt * b / 2)
        electric_step = inverse_half @ gp.transfer(data, dt) @ inverse_half
        assert_allclose(electric_step, expm(-dt * data['electric']), rtol=0, atol=ATOL)
        recovered.append(electric_step)
    assert_allclose(recovered[0] @ recovered[1], recovered[2], rtol=0, atol=ATOL)


def test_unresolved_positive_transfer_times_raise_but_identity_log_is_exact(built_model):
    for dt in (np.nextafter(0., 1.), np.finfo(float).tiny, 1e-200, 1e-20, 1e200):
        with pytest.raises(FloatingPointError):
            gp.transfer(built_model, dt)
    for dt in (np.finfo(float).tiny, 1e-200, 1e-20, 1e200):
        assert np.array_equal(gp.effective_hamiltonian(np.eye(2), dt), np.zeros((2, 2)))


def test_positive_fraction_time_cannot_round_to_zero(built_model):
    tiny_positive = Fraction(1, 10 ** 1000)
    assert tiny_positive > 0 and float(tiny_positive) == 0
    with pytest.raises(FloatingPointError):
        gp.transfer(built_model, tiny_positive)


def test_negative_fraction_time_rejected_before_float_conversion(built_model):
    tiny_negative = Fraction(-1, 10 ** 400)
    assert tiny_negative < 0 and float(tiny_negative) == 0
    with pytest.raises(ValueError):
        gp.transfer(built_model, tiny_negative)


def test_log_finite_input_with_overflowing_norm_rejects_explicitly():
    transfer = np.array([[1e308, 8e307], [8e307, 1e308]])
    assert np.all(np.isfinite(transfer))
    # A RuntimeWarning under warnings-as-errors is not the domain contract.
    with pytest.raises(FloatingPointError):
        gp.effective_hamiltonian(transfer, 1.0)


@pytest.mark.parametrize('offdiagonal', [1e-18, 1e-18j], ids=['real', 'complex'])
def test_log_near_identity_offdiagonal_is_resolved_or_rejected(offdiagonal):
    # eigh can round both eigenvalues to 1 while a representable offdiagonal
    # still produces an order-one generator after division by dt.
    dt = 1e-18
    perturbation = np.array([[0, offdiagonal], [np.conjugate(offdiagonal), 0]])
    transfer = np.eye(2) + perturbation
    try:
        effective = gp.effective_hamiltonian(transfer, dt)
    except FloatingPointError:
        return
    # Higher log-series terms / dt are at most 1e-18 here.
    assert_allclose(effective, -perturbation / dt, rtol=0, atol=5e-12)


@pytest.mark.parametrize('scale, spectator', [(1.0, True), (2.0, False)],
                         ids=['identity-block-with-spectator', 'scaled-identity'])
def test_log_degenerate_block_offdiagonal_is_resolved_or_rejected(scale, spectator):
    dt = 1e-18
    transfer = np.diag([scale, scale, 2.0] if spectator else [scale, scale])
    transfer[0, 1] = transfer[1, 0] = 1e-18
    try:
        effective = gp.effective_hamiltonian(transfer, dt)
    except FloatingPointError:
        return
    # The unrelated or common scalar log gives huge diagonals. Inspect the
    # active entries directly so those diagonals cannot mask a lost coupling.
    assert_allclose(effective[0, 1], -1.0 / scale, rtol=0, atol=5e-12)
    assert_allclose(effective[1, 0], -1.0 / scale, rtol=0, atol=5e-12)


def test_generic_complex_hermitian_log_above_one_not_clipped():
    unitary = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    values = np.array([0.4, 1.7])
    transfer = (unitary * values) @ unitary.conj().T
    expected = (unitary * (-np.log(values) / 0.13)) @ unitary.conj().T
    result = gp.effective_hamiltonian(transfer, 0.13)
    assert_allclose(result, expected, rtol=0, atol=5e-12)
    assert_allclose(result, result.conj().T, rtol=0, atol=5e-12)
    assert np.min(np.linalg.eigvalsh(result)) < 0
    assert_allclose(expm(-0.13 * result), transfer, rtol=0, atol=5e-12)


def test_genuine_square_graph_after_deleting_right_path(n):
    square_edges = {edge: EDGES[edge] for edge in ('a', 'b', 'c', 'd')}
    square_vertices = ('A', 'Ltop', 'Lbot', 'B')
    terms, denominator = _weights(n)
    for full_links in _configurations(n):
        links = {edge: full_links[edge] for edge in square_edges}
        loop = _walk(links, LOOPS[0], n)
        class_key = lambda g: min(_conjugate(t, g, n) for t in _elements(n))
        for vertex in square_vertices:
            for t in _elements(n):
                values = dict.fromkeys(square_vertices, IDENTITY)
                values[vertex] = t
                transformed = {edge: _mul(_mul(values[x], links[edge], n),
                                         _inv(values[y], n), n)
                               for edge, (x, y) in square_edges.items()}
                assert class_key(_walk(transformed, LOOPS[0], n)) == class_key(loop)
        coordinate_counter = Counter()
        for numerator, g in terms:
            coordinate_counter[class_key(_mul(g, loop, n))] += numerator
        graph_total = Counter()
        for edge in square_edges:
            for numerator, g in terms:
                shifted = _shift(links, edge, g, n)
                graph_total[class_key(_walk(shifted, LOOPS[0], n))] += numerator
        assert graph_total == Counter({key: 4 * value for key, value in coordinate_counter.items()})
        assert sum(graph_total.values()) == 12 * denominator


def test_generic_logarithm_and_noncommuting_symmetric_split_sign():
    a = np.array([[1.0, 0.2], [0.2, 0.6]])
    b = np.array([[0.5, -0.1], [-0.1, 1.4]])
    ab = a @ b - b @ a
    correction = (a @ ab - ab @ a) / 12 + (b @ ab - ab @ b) / 24
    errors = []
    for dt in (0.02, 0.01, 0.005):
        half = expm(-dt * b / 2)
        transfer = half @ expm(-dt * a) @ half
        effective = gp.effective_hamiltonian(transfer, dt)
        assert_allclose(expm(-dt * effective), transfer, rtol=0, atol=5e-12)
        errors.append(np.linalg.norm((effective - a - b) / dt ** 2 - correction, 2))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-6
    assert np.linalg.norm(correction, 2) > 1e-3


def test_python38_grammar_new_files():
    root = Path(__file__).resolve().parents[1]
    for name in ('bpr/gauge_two_plaquette.py', 'scripts/demo_gauge_two_plaquette.py',
                 'tests/test_gauge_two_plaquette.py'):
        ast.parse((root / name).read_text(), filename=name, feature_version=(3, 8))
