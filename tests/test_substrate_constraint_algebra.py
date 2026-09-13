"""Independent finite-sector and exterior-power oracles for the frozen audit.

This file deliberately does not use existing model/projection helpers.  Small
Fourier minors use Leibniz sums, not numpy.linalg.det.  Numerical comparisons
are diagnostics; structural classifications have separate assertions.
"""

import ast
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(np.float64).eps
ATOL = 3.0e-12


def occupations(length, number):
    """Enumerate weak compositions without importing a production constructor."""
    if length == 1:
        return ((number,),)
    return tuple(
        (first,) + rest
        for first in range(number + 1)
        for rest in occupations(length - 1, number - first)
    )


def bose_oracle(length, number, coupling=1.0, interaction=0.7):
    basis = occupations(length, number)
    indices = {state: index for index, state in enumerate(basis)}
    dimension = len(basis)
    hopping = np.zeros((dimension, dimension), dtype=np.float64)
    edges = set()
    for column, state in enumerate(basis):
        for source in range(length):
            for destination in ((source - 1) % length, (source + 1) % length):
                if state[source] == 0:
                    continue
                changed = list(state)
                changed[source] -= 1
                changed[destination] += 1
                row = indices[tuple(changed)]
                hopping[row, column] -= coupling * math.sqrt(
                    state[source] * (state[destination] + 1)
                )
                edges.add(tuple(sorted((row, column))))
    onsite = np.array([
        sum(value * (value - 1) // 2 for value in state) for state in basis
    ], dtype=np.float64)
    return basis, tuple(sorted(edges)), hopping + interaction * np.diag(onsite)


def components_oracle(vertex_count, edges):
    neighbors = [set() for _ in range(vertex_count)]
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    pending = set(range(vertex_count))
    components = []
    while pending:
        root = min(pending)
        reached = {root}
        frontier = [root]
        while frontier:
            vertex = frontier.pop()
            new = neighbors[vertex] - reached
            reached.update(new)
            frontier.extend(new)
        pending.difference_update(reached)
        components.append(tuple(sorted(reached)))
    labels = tuple(
        next(label for label, component in enumerate(components) if vertex in component)
        for vertex in range(vertex_count)
    )
    return tuple(components), labels


def permutation_sign(permutation):
    return (-1) ** sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )


def leibniz_minor(matrix, rows, columns):
    """The only determinants used by the oracle have order at most three."""
    assert len(rows) == len(columns) <= 3
    return sum(
        permutation_sign(permutation) * math.prod(
            matrix[rows[index], columns[permutation[index]]]
            for index in range(len(rows))
        )
        for permutation in itertools.permutations(range(len(rows)))
    )


def exterior_oracle(length, number):
    states = tuple(
        value for value in range(1 << length)
        if bin(value).count("1") == number
    )
    columns = tuple(itertools.combinations(range(3), number))
    modes = (-1, 0, 1)
    twist = 0.0 if number % 2 else np.pi
    fourier = np.array([
        [np.exp(1j * (2 * np.pi * mode + twist) * site / length) / math.sqrt(length)
         for mode in modes]
        for site in range(length)
    ], dtype=np.complex128)
    window = np.zeros((len(states), len(columns)), dtype=np.complex128)
    for row, state in enumerate(states):
        occupied = tuple(site for site in range(length) if state & (1 << site))
        for column, chosen_modes in enumerate(columns):
            window[row, column] = leibniz_minor(fourier, occupied, chosen_modes)
    densities = tuple(
        np.diag([float(bool(state & (1 << site))) for state in states]).astype(
            np.complex128
        )
        for site in range(length)
    )
    return states, window, densities, fourier


def exterior_lift(one_particle, number):
    """Independently apply sum of one-body operators to ordered wedge states."""
    basis = tuple(itertools.combinations(range(3), number))
    indices = {state: index for index, state in enumerate(basis)}
    lifted = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for column, state in enumerate(basis):
        for position, source in enumerate(state):
            remaining = list(state)
            remaining.pop(position)
            for destination in range(3):
                if destination in remaining:
                    continue
                insertion = sum(value < destination for value in remaining)
                changed = tuple(sorted(remaining + [destination]))
                lifted[indices[changed], column] += (
                    (-1) ** (position + insertion) * one_particle[destination, source]
                )
    return lifted


def norm2(matrix):
    if not matrix.size:
        return 0.0
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def defect_oracle(left, right, window):
    compressed_left = window.conj().T @ left @ window
    compressed_right = window.conj().T @ right @ window
    direct = window.conj().T @ left @ right @ window - compressed_left @ compressed_right
    complement = np.eye(window.shape[0]) - window @ window.conj().T
    factorized = window.conj().T @ left @ complement @ right @ window
    leak_left_adjoint = complement @ left.conj().T @ window
    leak_right = complement @ right @ window
    gram = leak_left_adjoint.conj().T @ leak_right
    error = window.conj().T @ window - np.eye(window.shape[1])
    correction = compressed_left @ error @ compressed_right
    return {
        "compressed_left": compressed_left,
        "compressed_right": compressed_right,
        "direct": direct,
        "factorized": factorized,
        "gram": gram,
        "correction": correction,
        "isometry_residual": float(np.linalg.norm(error, ord="fro")),
        "leakage_left": norm2(leak_left_adjoint),
        "leakage_right": norm2(leak_right),
        "operator_norm": norm2(direct),
        "frobenius_norm": float(np.linalg.norm(direct, ord="fro")),
    }


def assert_close(actual, expected, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=atol)


def assert_read_only_detached(array, *inputs):
    assert isinstance(array, np.ndarray)
    assert not array.flags.writeable
    for original in inputs:
        assert not np.shares_memory(array, original)
    if array.size:
        with pytest.raises(ValueError):
            array.flat[0] = 0


def reject_constant(value):
    raise AssertionError("Nonstandard JSON constant: " + value)


def decode_complex(payload):
    assert set(payload) == {"shape", "real", "imag"}
    shape = tuple(payload["shape"])
    real = np.asarray(payload["real"], dtype=np.float64).reshape(shape)
    imaginary = np.asarray(payload["imag"], dtype=np.float64).reshape(shape)
    return real + 1j * imaginary


@pytest.fixture(scope="module")
def api():
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("bpr.substrate_constraint_algebra")


@pytest.mark.parametrize("vertex_count,edges", [
    (1, []), (5, []), (6, [(4, 2), (1, 0), (2, 0)]),
    (4, [(3, 2), (1, 3), (0, 1), (2, 0)]),
    (512, [(511, 0), (8, 7)]),
])
def test_graph_components_independent(api, vertex_count, edges):
    before = list(edges)
    report = api.graph_components(vertex_count, edges)
    components, labels = components_oracle(vertex_count, edges)
    assert report["vertex_count"] == vertex_count
    assert report["components"] == components
    assert report["labels"] == labels
    assert report["component_count"] == len(components)
    assert report["edges"] == tuple(sorted(tuple(sorted(edge)) for edge in edges))
    assert edges == before


@pytest.mark.parametrize("vertex_count,edges", [
    (0, []), (-1, []), (513, []), (True, []), (3.0, []), ("3", []),
    (3, [(0, 0)]), (3, [(0, 3)]), (3, [(-1, 1)]),
    (3, [(False, 1)]), (3, [(0.0, 1)]), (3, [("0", 1)]),
    (3, [(0, 1), (1, 0)]), (3, [(0, 1), (0, 1)]),
    (3, [(0, 1, 2)]), (3, [0]), (3, "01"),
    (3, {(0, 1)}), (1, [(0, 0)]),
])
def test_graph_rejects_invalid_inputs(api, vertex_count, edges):
    with pytest.raises(ValueError):
        api.graph_components(vertex_count, edges)


def test_graph_never_consumes_arbitrary_iterator(api):
    def forbidden():
        raise AssertionError("arbitrary iterator must not be consumed")
        yield (0, 1)
    with pytest.raises(ValueError):
        api.graph_components(3, forbidden())


BOSE_CASES = tuple((length, number) for length in (3, 4, 5)
                   for number in sorted({1, 2, length}))
PROJECTION_CASES = tuple((length, number) for length in (3, 4, 5)
                         for number in (0, 1, 2, 3)) + ((4, 4), (5, 5))


@pytest.mark.parametrize("length,number", BOSE_CASES + ((3, 0), (5, 0)))
def test_exact_sector_graph(api, length, number):
    basis, edges, _ = bose_oracle(length, number)
    report = api.sector_graph(length, number)
    assert report["ambient"] == "complete_bose_fixed_number"
    assert (report["L"], report["N"]) == (length, number)
    assert report["dimension"] == math.comb(length + number - 1, number)
    assert report["basis"] == basis
    assert report["edges"] == edges
    assert report["components"] == (tuple(range(len(basis))),)
    assert report["labels"] == (0,) * len(basis)
    assert report["diagonal_commutant_dimension"] == 1
    if number:
        assert (number,) + (0,) * (length - 1) in basis
        assert len(edges) > 0
    else:
        assert edges == ()


@pytest.mark.parametrize("length,number", BOSE_CASES + ((3, 0),))
@pytest.mark.parametrize("interaction", (0.7, 40.0))
def test_constraint_residue_classification_and_commutators(api, length, number, interaction):
    basis, edges, hamiltonian = bose_oracle(length, number, interaction=interaction)
    report = api.constraint_report(length, number, g=interaction)
    assert report["ambient"] == "complete_bose_fixed_number"
    assert report["dimension"] == len(basis)
    assert tuple(tuple(state) for state in report["basis"]) == basis
    assert report["diagonal_commutant_dimension"] == 1
    assert report["graph"]["component_count"] == 1
    controls = report["onsite_projectors"]
    assert len(controls) == 5 * length
    assert {(entry["site"], entry["q"], entry["residue"]) for entry in controls} == {
        (site, modulus, residue) for site in range(length)
        for modulus in (2, 3) for residue in range(modulus)
    }
    for entry in controls:
        values = np.array([
            float(state[entry["site"]] % entry["q"] == entry["residue"])
            for state in basis
        ])
        diagonal = np.diag(values)
        commutator = hamiltonian @ diagonal - diagonal @ hamiltonian
        expected_status = ("zero" if not np.any(values) else
                           "identity" if np.all(values) else "nonconstant")
        assert entry["status"] == expected_status
        assert entry["conserved"] is (expected_status != "nonconstant")
        norm = float(np.linalg.norm(commutator, ord="fro"))
        assert_close(entry["commutator_frobenius_norm"], norm)
        edge_sum = 2 * sum(abs(hamiltonian[left, right]) ** 2 *
                           abs(values[right] - values[left]) ** 2
                           for left, right in edges)
        assert_close(norm ** 2, edge_sum)
        if expected_status == "nonconstant":
            assert norm > 0.0
            assert any(values[left] != values[right] for left, right in edges)
        else:
            assert norm == 0.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("coupling", (0.25, 2.0, 2.0 ** -128))
def test_constraint_graph_independent_of_hopping_scale(api, coupling):
    report = api.constraint_report(4, 2, C=coupling, g=40.0)
    reference = api.constraint_report(4, 2, C=1.0, g=0.0)
    assert report["graph"] == reference["graph"]
    for control, unit in zip(report["onsite_projectors"], reference["onsite_projectors"]):
        assert control["status"] == unit["status"]
        assert_close(control["commutator_frobenius_norm"] / coupling,
                     unit["commutator_frobenius_norm"])


def assert_audit(report, left, right, window, serialized=False):
    oracle = defect_oracle(left, right, window)
    expected_arrays = {
        "compressed_A": oracle["compressed_left"],
        "compressed_B": oracle["compressed_right"],
        "direct_defect": oracle["direct"],
        "complement_defect": oracle["factorized"],
        "gram_factor": oracle["gram"],
        "gram_correction": oracle["correction"],
        "corrected_gram_defect": oracle["gram"] - oracle["correction"],
    }
    complement = np.eye(window.shape[0]) - window @ window.conj().T
    expected_arrays["leakage_adjoint_A"] = complement @ left.conj().T @ window
    expected_arrays["leakage_B"] = complement @ right @ window
    for key, expected in expected_arrays.items():
        actual = decode_complex(report[key]) if serialized else report[key]
        assert actual.shape == expected.shape
        assert_close(actual, expected)
        if not serialized:
            assert actual.dtype == np.dtype(np.complex128)
            assert_read_only_detached(actual, left, right, window)
    dimension, rank = window.shape
    assert report["ambient_dimension"] == dimension
    assert report["rank"] == rank
    assert report["status"] == ("empty_candidate" if rank == 0 else
                                "full_window" if rank == dimension else "proper_window")
    assert_close(report["isometry_residual"], oracle["isometry_residual"])
    assert report["isometry_tolerance"] == 64 * EPS * max(dimension, rank, 1)
    assert_close(report["defect_operator_norm"], oracle["operator_norm"])
    product = oracle["leakage_left"] * oracle["leakage_right"]
    assert_close(report["leakage_product_bound"], product)
    assert_close(report["gram_correction_operator_norm"], norm2(oracle["correction"]))
    correction_bound = (norm2(oracle["compressed_left"]) *
                        norm2(window.conj().T @ window - np.eye(rank)) *
                        norm2(oracle["compressed_right"]))
    assert_close(report["finite_isometry_bound"], product + correction_bound)
    assert report["defect_operator_norm"] <= report["finite_isometry_bound"] + ATOL
    assert report["direct_complement_residual"] < ATOL
    assert report["corrected_gram_residual"] < ATOL
    return oracle


@pytest.mark.parametrize("length,number", PROJECTION_CASES)
def test_projection_all_ordered_pairs_with_independent_exterior_oracle(api, length, number):
    states, window, densities, fourier = exterior_oracle(length, number)
    report = api.projection_report(length, number)
    assert report["ambient"] == "hard_core_fixed_number"
    assert report["dimension"] == math.comb(length, number)
    assert report["rank"] == (math.comb(3, number) if number <= 3 else 0)
    assert tuple(report["basis_bits"]) == states
    assert tuple(report["modes"]) == (-1, 0, 1)
    assert_close(decode_complex(report["W"]), window)
    assert report["status"] == ("empty_candidate" if window.shape[1] == 0 else
                                "full_window" if window.shape[0] == window.shape[1]
                                else "proper_window")
    assert len(report["density_pairs"]) == length ** 2
    assert {tuple(pair["sites"]) for pair in report["density_pairs"]} == set(
        itertools.product(range(length), repeat=2)
    )
    defects = {}
    for pair in report["density_pairs"]:
        left_site, right_site = pair["sites"]
        left, right = densities[left_site], densities[right_site]
        oracle = assert_audit(pair["audit"], left, right, window, serialized=True)
        compressed_commutator = (oracle["compressed_left"] @ oracle["compressed_right"] -
                                 oracle["compressed_right"] @ oracle["compressed_left"])
        assert_close(decode_complex(pair["compressed_commutator"]), compressed_commutator)
        assert_close(pair["compressed_commutator_operator_norm"], norm2(compressed_commutator))
        assert pair["commutator_identity_residual"] < ATOL
        # All ambient densities commute, but their compressions need not.
        reverse = defect_oracle(right, left, window)["direct"]
        assert_close(compressed_commutator, reverse - oracle["direct"])
        single_left = np.outer(fourier[left_site].conj(), fourier[left_site])
        single_right = np.outer(fourier[right_site].conj(), fourier[right_site])
        one_body_defect = ((single_left if left_site == right_site else
                            np.zeros((3, 3))) - single_left @ single_right)
        assert_close(oracle["direct"], exterior_lift(one_body_defect, number))
        defects[left_site, right_site] = decode_complex(pair["audit"]["direct_defect"])
        if left_site == right_site:
            assert_close(oracle["direct"], (1.0 - 3.0 / length) * oracle["compressed_left"])
            assert np.linalg.eigvalsh(oracle["direct"]).min(initial=0.0) >= -ATOL
            if 1 <= number <= 3:
                expected_norm = (3.0 / length) * (1.0 - 3.0 / length)
                assert_close(oracle["operator_norm"], expected_norm)
                assert_close(oracle["frobenius_norm"],
                             math.sqrt(math.comb(2, number - 1)) * expected_norm)
                assert_close(oracle["leakage_right"], math.sqrt(expected_norm))
        if number == 3:
            assert_close(compressed_commutator, np.zeros((1, 1)))
            if length > 3 and left_site == right_site:
                assert oracle["operator_norm"] > 0.18
        if length == 3 or number == 0 or number > 3:
            assert_close(oracle["direct"], np.zeros_like(oracle["direct"]))
    for site in range(length):
        assert_close(sum(defects[site, other] for other in range(length)),
                     np.zeros((window.shape[1], window.shape[1])))
        assert_close(sum(defects[other, site] for other in range(length)),
                     np.zeros((window.shape[1], window.shape[1])))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("number", (1, 2))
def test_exact_l4_noncommuting_witness(api, number):
    report = api.projection_report(4, number)
    pair = next(pair for pair in report["density_pairs"] if pair["sites"] == [0, 1])
    commutator = decode_complex(pair["compressed_commutator"])
    assert_close(commutator[0, 0], -1j / 8)
    assert_close(norm2(commutator), math.sqrt(2) / 8)
    assert_close(np.linalg.norm(commutator, ord="fro"), 1 / 4)
    assert report["predetermined_witness"] is not None
    if number == 2:
        signed_complement = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        transformed = signed_complement @ commutator @ signed_complement.T
        one_report = api.projection_report(4, 1)
        one_pair = next(pair for pair in one_report["density_pairs"] if pair["sites"] == [0, 1])
        assert_close(transformed, -decode_complex(one_pair["compressed_commutator"]).T)
        assert_close(transformed[0, 0], 1j / 8)


@pytest.mark.parametrize("number", (1, 2))
def test_exact_l5_commutator_norms(api, number):
    report = api.projection_report(5, number)
    for separation in (1, 2):
        pair = next(pair for pair in report["density_pairs"]
                    if pair["sites"] == [0, separation])
        sign = 1 if separation == 1 else -1
        expected = math.sqrt(10 + sign * 3 * math.sqrt(5)) / 25
        assert_close(pair["compressed_commutator_operator_norm"], expected)


def test_nonhermitian_adjoint_factor_is_not_naive_gram(api):
    left = np.array([[1 + 1j, 2, 3j], [0, -1j, 4], [2, 1j, -2]], dtype=complex)
    right = np.array([[2, 0, 1], [1j, 3, 0], [0, -2j, 1]], dtype=complex)
    window = np.array([[1, 0], [0, 1], [0, 0]], dtype=complex)
    copies = tuple(array.copy() for array in (left, right, window))
    report = api.multiplication_defect(left, right, window)
    oracle = assert_audit(report, left, right, window)
    complement = np.eye(3) - window @ window.conj().T
    naive = (complement @ left @ window).conj().T @ (complement @ right @ window)
    assert norm2(naive - oracle["direct"]) > 1.0
    for original, snapshot in zip((left, right, window), copies):
        np.testing.assert_array_equal(original, snapshot)
        assert original.flags.writeable
    left[:] = 0
    assert_close(report["compressed_A"], window.conj().T @ copies[0] @ window)


def test_generic_positive_gram_for_exact_isometry(api):
    operator = np.array([[1, 2j, -1], [0, 3, 1j], [2, 0, -2]], dtype=complex)
    window = np.array([[1, 0], [0, 1], [0, 0]], dtype=complex)
    report = api.multiplication_defect(operator.conj().T, operator, window)
    assert_audit(report, operator.conj().T, operator, window)
    assert_close(report["direct_defect"], report["direct_defect"].conj().T)
    assert np.linalg.eigvalsh(report["direct_defect"]).min() >= -ATOL


@pytest.mark.parametrize("rank", (0, 1, 3))
def test_generic_empty_proper_full_and_scalar_controls(api, rank):
    window = np.eye(3, dtype=complex)[:, :rank]
    left = 2 * np.eye(3, dtype=complex)
    right = np.array([[0, 1j, 2], [3, 1, 0], [0, 2j, 1]], dtype=complex)
    report = api.multiplication_defect(left, right, window)
    assert_audit(report, left, right, window)
    assert_close(report["direct_defect"], np.zeros((rank, rank)))
    assert report["direct_defect"].shape == (rank, rank)
    assert report["leakage_B"].shape == (3, rank)


def test_approximate_isometry_correction_and_negative_scalar_defect(api):
    left = np.ones((1, 1), dtype=complex)
    right = left.copy()
    window = np.array([[math.sqrt(1.0 + 16 * EPS)]], dtype=complex)
    report = api.multiplication_defect(left, right, window)
    assert_audit(report, left, right, window)
    assert 0 < report["isometry_residual"] <= report["isometry_tolerance"]
    assert report["direct_defect"][0, 0].real < 0.0
    assert report["gram_factor"][0, 0].real >= 0.0
    assert report["gram_correction_operator_norm"] > 0.0
    # Absolute 1e-12 would conceal this correction entirely.
    assert_close(report["gram_factor"],
                 report["direct_defect"] + report["gram_correction"], atol=1e-28)
    assert_close(report["corrected_gram_defect"], report["direct_defect"], atol=1e-28)


def test_approximate_isometry_complex_matrix_correction(api):
    left = np.array([[2, 3j, 1], [1, -1, 2j], [0, 1, 4]], dtype=complex)
    right = np.array([[0, 1, 2], [3j, 2, 0], [1, -1j, 3]], dtype=complex)
    window = np.array([[1 + 8 * EPS, 0], [0, 1 - 8 * EPS], [0, 0]], dtype=complex)
    report = api.multiplication_defect(left, right, window)
    oracle = assert_audit(report, left, right, window)
    assert norm2(oracle["correction"]) > 1e-15
    assert_close(report["gram_factor"] - report["gram_correction"],
                 report["direct_defect"], atol=2e-14)


def test_isometry_uses_frobenius_not_entrywise_threshold(api):
    dimension = 4
    tolerance = 64 * EPS * dimension
    window = math.sqrt(1 + 0.75 * tolerance) * np.eye(dimension, dtype=complex)
    error = window.conj().T @ window - np.eye(dimension)
    assert np.max(abs(error)) < tolerance
    assert np.linalg.norm(error, ord="fro") > tolerance
    with pytest.raises(ValueError):
        api.multiplication_defect(np.eye(dimension), np.eye(dimension), window)


BAD_SCALARS = (True, False, "1", None, 1 + 0j, float("nan"), float("inf"),
               -float("inf"), np.nextafter(0.0, 1.0), np.finfo(float).tiny,
               2.0 ** -129, 2.0 ** 21)


@pytest.mark.parametrize("value", BAD_SCALARS)
@pytest.mark.parametrize("parameter", ("C", "g"))
def test_constraint_scalar_domain_before_allocation(api, value, parameter):
    with mock.patch.object(api, "fixed_number_model", side_effect=AssertionError("allocated model")):
        with pytest.raises(ValueError):
            api.constraint_report(3, 0, **{parameter: value})


@pytest.mark.parametrize("value", BAD_SCALARS + (0.0, -1.0))
def test_projection_scalar_domain_before_empty_dispatch(api, value):
    with mock.patch.object(api.matching, "projected_model", side_effect=AssertionError("allocated model")):
        with pytest.raises(ValueError):
            api.projection_report(4, 4, C=value)


@pytest.mark.parametrize("length,number", [
    (True, 1), (3, True), (3.0, 1), (3, 1.0), ("3", 1), (3, "1"),
    (2, 1), (13, 1), (3, -1), (3, 4), (12, 12), (1000000, 1),
])
def test_bose_sector_validation_and_dimension_cap(api, length, number):
    with mock.patch.object(api, "fixed_number_model", side_effect=AssertionError("allocated model")):
        with pytest.raises(ValueError):
            api.sector_graph(length, number)
        with pytest.raises(ValueError):
            api.constraint_report(length, number)


@pytest.mark.parametrize("length,number", [
    (True, 1), (3, True), (3.0, 1), (3, 1.0), ("3", 1), (3, "1"),
    (2, 1), (9, 1), (3, -1), (3, 4), (1000000, 1),
])
def test_hard_core_domain_validation_before_allocation(api, length, number):
    with mock.patch.object(api.matching, "projected_model", side_effect=AssertionError("allocated model")):
        with pytest.raises(ValueError):
            api.projection_report(length, number)


@pytest.mark.parametrize("function,kwargs", [
    ("constraint_report", {"C": 0.0}),
    ("constraint_report", {"C": -1.0}),
    ("constraint_report", {"g": -1.0}),
])
def test_scalar_signs(api, function, kwargs):
    with pytest.raises(ValueError):
        getattr(api, function)(3, 0, **kwargs)


@pytest.mark.parametrize("coupling", (2.0 ** -128, 2.0 ** 20))
def test_scalar_supported_endpoints_and_zero_interaction(api, coupling):
    report = api.constraint_report(3, 0, C=coupling, g=0.0)
    assert report["C"] == coupling
    assert report["g"] == 0.0
    report = api.projection_report(3, 0, C=coupling)
    assert report["C"] == coupling


@pytest.mark.parametrize("bad", [
    np.array([[True]]), np.array([["1"]]), np.array([[object()]], dtype=object),
    np.array([[float("nan")]]), np.array([[float("inf")]]),
    np.array([[1 + float("inf") * 1j]]),
    np.array([[np.nextafter(0.0, 1.0)]]), np.array([[np.finfo(float).tiny]]),
    np.array([[2.0 ** -129]]), np.array([[2.0 ** 21]]),
    np.array([[1 + 1j * 2.0 ** -129]]), np.array([[1 + 1j * 2.0 ** 21]]),
    [[True]], [["1"]], [[None]], [[1, 2], [3]],
])
@pytest.mark.parametrize("operand", ("A", "B", "W"))
def test_generic_matrix_domain_rejection_including_empty_candidate(api, bad, operand):
    operands = {"A": np.ones((1, 1)), "B": np.ones((1, 1)), "W": np.zeros((1, 0))}
    operands[operand] = bad
    with pytest.raises(ValueError):
        api.multiplication_defect(**operands)


@pytest.mark.parametrize("left_shape,right_shape,window_shape", [
    ((0, 0), (0, 0), (0, 0)),
    ((2, 3), (2, 2), (2, 1)),
    ((2, 2), (3, 3), (2, 1)),
    ((2, 2), (2, 2), (3, 1)),
    ((2, 2), (2, 2), (2, 3)),
    ((513, 513), (513, 513), (513, 0)),
    ((1,), (1, 1), (1, 1)),
    ((1, 1), (1, 1), (1,)),
    ((1, 1, 1), (1, 1), (1, 1)),
])
def test_generic_shapes_precede_numeric_conversion(api, left_shape, right_shape, window_shape):
    operands = [np.broadcast_to(np.array(1.0), shape)
                for shape in (left_shape, right_shape, window_shape)]
    with mock.patch.object(np, "asarray", side_effect=AssertionError("premature conversion")):
        with mock.patch.object(np, "array", side_effect=AssertionError("premature allocation")):
            with pytest.raises(ValueError):
                api.multiplication_defect(*operands)


def test_opaque_array_protocol_is_never_invoked(api):
    class DangerousArray:
        shape = (1, 1)

        def __array__(self, *args, **kwargs):
            raise AssertionError("untrusted array protocol invoked")

    with pytest.raises(ValueError):
        api.multiplication_defect(DangerousArray(), np.ones((1, 1)), np.ones((1, 1)))


def test_generic_rejects_array_subclasses_before_protocol(api):
    class ArraySubclass(np.ndarray):
        pass

    array = np.ones((1, 1)).view(ArraySubclass)
    with pytest.raises(ValueError):
        api.multiplication_defect(array, np.ones((1, 1)), np.ones((1, 1)))


def test_material_nonisometry_not_silently_repaired(api):
    window = np.array([[1, 1], [0, 0]], dtype=complex)
    before = window.copy()
    with pytest.raises(ValueError):
        api.multiplication_defect(np.eye(2), np.eye(2), window)
    np.testing.assert_array_equal(window, before)


@pytest.mark.parametrize("scale", (0.0, 2.0 ** -128, -(2.0 ** -128), 2.0 ** 20, -(2.0 ** 20)))
def test_generic_component_domain_endpoints(api, scale):
    left = np.array([[0, scale], [scale * 1j, 0]], dtype=complex)
    right = np.array([[0, 1], [1, 0]], dtype=complex)
    window = np.array([[1], [0]], dtype=complex)
    report = api.multiplication_defect(left, right, window)
    assert_audit(report, left, right, window)
    assert report["direct_defect"][0, 0] == scale


def test_bounded_list_operands_and_zero_rank_list_shape(api):
    report = api.multiplication_defect([[1, 0], [0, 1]], ((0, 1), (1, 0)), [[], []])
    assert report["status"] == "empty_candidate"
    assert report["rank"] == 0
    assert report["direct_defect"].shape == (0, 0)
    assert report["leakage_B"].shape == (2, 0)


def test_cap512_empty_candidate_supported(api):
    identity = np.eye(512)
    window = np.zeros((512, 0))
    report = api.multiplication_defect(identity, identity, window)
    assert report["ambient_dimension"] == 512
    assert report["rank"] == 0
    assert report["status"] == "empty_candidate"
    assert report["direct_defect"].shape == (0, 0)


def test_caller_readonly_noncontiguous_arrays_and_detached_outputs(api):
    storage = np.arange(16, dtype=float).reshape(4, 4)
    left = storage[::2, ::2]
    right = np.eye(2)
    window = np.array([[1.0], [0.0]])
    left.setflags(write=False)
    right.setflags(write=False)
    window.setflags(write=False)
    snapshots = [value.copy() for value in (left, right, window)]
    report = api.multiplication_defect(left, right, window)
    assert_audit(report, left, right, window)
    for original, snapshot in zip((left, right, window), snapshots):
        np.testing.assert_array_equal(original, snapshot)
        assert not original.flags.writeable
    for value in report.values():
        if isinstance(value, np.ndarray):
            assert_read_only_detached(value, storage, left, right, window)
            with pytest.raises(ValueError):
                value.setflags(write=True)


def test_numerical_unavailable_is_clear_value_error_subclass(api):
    assert issubclass(api.NumericalUnavailable, ValueError)
    # Synthetic solver-failure injection tests exception plumbing, not a claim
    # that the bounded domain has a demonstrated arithmetic counterexample.
    operator = np.array([[0.0, 1.0], [1.0, 0.0]])
    window = np.array([[1.0], [0.0]])
    with mock.patch.object(np.linalg, "norm", side_effect=np.linalg.LinAlgError("synthetic failure")):
        with pytest.raises(api.NumericalUnavailable):
            api.multiplication_defect(operator, operator, window)


def test_report_norm_labels_and_diagnostic_not_exact_certificate(api):
    report = api.multiplication_defect(np.eye(2), np.eye(2), np.array([[1.0], [0.0]]))
    assert report["residual_norm"] == "Frobenius"
    assert report["defect_and_bound_norm"] == "operator_2"
    assert report["numerical_status"] == "diagnostic_only"
    assert report["exactness_certificate"] is False
    assert "exact" in report["pure_leakage_bound_requires"]


def test_scalar_report_positive_controls_are_explicit(api):
    number = 2
    constraint = api.constraint_report(4, number)
    controls = constraint["scalar_controls"]
    assert {entry["name"] for entry in controls} == {
        "identity", "global_number", "global_number_residue"
    }
    residues = set()
    for entry in controls:
        expected = (1 if entry["name"] == "identity" else number if
                    entry["name"] == "global_number" else
                    int(number % entry["q"] == entry["residue"]))
        assert entry["value"] == expected
        assert entry["status"] == "scalar"
        assert entry["conserved"] is True
        assert entry["commutator_frobenius_norm"] == 0.0
        if entry["name"] == "global_number_residue":
            residues.add((entry["q"], entry["residue"]))
    assert residues == {(modulus, residue) for modulus in (2, 3) for residue in range(modulus)}
    projection = api.projection_report(4, number)
    assert {entry["name"] for entry in projection["scalar_controls"]} == {
        "zero", "identity", "global_number"
    }
    for entry in projection["scalar_controls"]:
        assert entry["scalar"] == {"zero": 0, "identity": 1, "global_number": number}[entry["name"]]
        assert entry["structural_status"] == "scalar_preserves_exact_window"
        assert_close(decode_complex(entry["audit"]["direct_defect"]), np.zeros((3, 3)))
        assert entry["audit"]["exactness_certificate"] is False


def test_projection_report_serialization_preserves_empty_matrix_shapes(api):
    report = json.loads(json.dumps(api.projection_report(4, 4), allow_nan=False),
                        parse_constant=reject_constant)
    assert report["dimension"] == 1
    assert report["rank"] == 0
    assert report["status"] == "empty_candidate"
    assert decode_complex(report["W"]).shape == (1, 0)
    for pair in report["density_pairs"]:
        assert decode_complex(pair["compressed_commutator"]).shape == (0, 0)
        for key in ("direct_defect", "complement_defect", "gram_factor", "gram_correction"):
            assert decode_complex(pair["audit"][key]).shape == (0, 0)
        assert decode_complex(pair["audit"]["leakage_B"]).shape == (1, 0)


def assert_demo_grid(report):
    assert {(entry["L"], entry["N"], entry["g"]) for entry in report["constraint_cases"]} == {
        (length, number, interaction) for length, number in BOSE_CASES
        for interaction in (0.7, 40.0)
    }
    assert len(report["constraint_cases"]) == 18
    assert {(entry["L"], entry["N"]) for entry in report["projection_cases"]} == {
        (length, number) for length, number in PROJECTION_CASES if number != 0
    }
    assert len(report["projection_cases"]) == 11
    assert report["limitations"]
    assert report["numerical_domain"]
    assert "model_id" in report


def test_demonstration_report_fixed_grid_strict_json(api):
    report = api.demonstration_report()
    decoded = json.loads(json.dumps(report, allow_nan=False), parse_constant=reject_constant)
    assert_demo_grid(decoded)
    assert decoded["vacuum_controls"]["complete_bose"]["N"] == 0
    assert decoded["vacuum_controls"]["hard_core"]["N"] == 0


@pytest.mark.parametrize("json_mode", (False, True))
def test_standalone_demo_empty_cwd_stdout_only(tmp_path, json_mode):
    command = [sys.executable, "-B", str(ROOT / "scripts" / "demo_substrate_constraint_algebra.py")]
    if json_mode:
        command.append("--json")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(command, cwd=str(tmp_path), env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=180)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []
    assert completed.stdout.strip()
    if json_mode:
        report = json.loads(completed.stdout, parse_constant=reject_constant)
        assert_demo_grid(report)
    else:
        text = completed.stdout.lower()
        assert "bose" in text
        assert "hard" in text
        assert "diagonal" in text
        assert "empty" in text


@pytest.mark.parametrize("path", (
    ROOT / "bpr" / "substrate_constraint_algebra.py",
    ROOT / "scripts" / "demo_substrate_constraint_algebra.py",
    Path(__file__),
))
def test_new_files_parse_as_python38(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
