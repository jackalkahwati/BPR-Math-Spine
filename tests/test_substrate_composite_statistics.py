"""Independent kinematic controls for the frozen composite-statistics contract.

Authored from doc/derivations/substrate_composite_statistics_2026-09-12.md
before reading the new implementation or running these tests. Binary expected
entries use changing-bit elementary actions, never a tensor/JW constructor.
Bose expected entries use recursive complete occupations and elementary square
root coefficients, never inherited model or composite constructors. Numerical
Bose tolerances are diagnostics, not exact arithmetic or physical certificates.
"""

import ast
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "bpr.substrate_composite_statistics"
BINARY_LENGTHS = (3, 4, 6, 8)
BOSE_LENGTHS = (3, 4, 5)
DEMO_BOSE_SUPPORTS = ((0,), (0, 1), (0, 1, 2))
PAIR_CASES = (
    (6, (0, 1, 2), (3, 4, 5)),
    (6, (0, 2, 4), (1, 3, 5)),
    (6, (0, 1, 2), (2, 3, 4)),
    (4, (0,), (1,)),
    (4, (0, 1), (2, 3)),
    (4, (0, 1), (1, 2)),
)
ATOL = 3.0e-12


def frozen_supports(length):
    candidates = (
        (0,), (length - 1,), (0, 1), (0, length - 1), (0, 1, 2),
        (0, 2, length - 1), tuple(range(length)),
    )
    result = []
    for sites in candidates:
        if len(set(sites)) == len(sites) and sites not in result:
            result.append(sites)
    return tuple(result)


BINARY_CASES = tuple(
    (length, sites)
    for length in BINARY_LENGTHS
    for sites in frozen_supports(length)
)
BOSE_CASES = tuple(
    (length, number, sites)
    for length in BOSE_LENGTHS
    for number in range(4)
    for sites in DEMO_BOSE_SUPPORTS
)


@pytest.fixture(scope="module")
def api():
    return importlib.import_module(MODULE_NAME)


def popcount(bits):
    # int.bit_count is deliberately not required by the Python 3.8 test oracle.
    return bin(bits).count("1")


def bit_action(bits, site, creation, fermionic):
    occupied = bool(bits & (1 << site))
    if occupied == creation:
        return None, 0
    coefficient = 1
    if fermionic and popcount(bits & ((1 << site) - 1)) % 2:
        coefficient = -1
    return bits ^ (1 << site), coefficient


def bit_word_matrix(length, actions, fermionic=True):
    """Actions are supplied in chronological, not written-product, order."""
    dimension = 1 << length
    answer = np.zeros((dimension, dimension), dtype=np.int64)
    for source in range(dimension):
        current = source
        amplitude = 1
        for site, creation in actions:
            current, factor = bit_action(current, site, creation, fermionic)
            amplitude *= factor
            if not amplitude:
                break
        if amplitude:
            answer[current, source] = amplitude
    return answer


def lowering_word(sites):
    return tuple((site, False) for site in reversed(sites))


def raising_word(sites):
    return tuple((site, True) for site in sites)


def binary_expected(length, sites, fermionic=True):
    return bit_word_matrix(length, lowering_word(sites), fermionic)


def occupations(length, number):
    """Independent lexicographic weak-composition recursion, no local cutoff."""
    if length == 1:
        return ((number,),)
    return tuple(
        (first,) + remaining
        for first in range(number + 1)
        for remaining in occupations(length - 1, number - first)
    )


def bose_word_matrix(length, number, actions):
    source_basis = occupations(length, number)
    target_number = number + sum(1 if creation else -1 for _, creation in actions)
    target_basis = () if target_number < 0 else occupations(length, target_number)
    matrix = np.zeros((len(target_basis), len(source_basis)), dtype=np.float64)
    rows = {state: row for row, state in enumerate(target_basis)}
    for column, source in enumerate(source_basis):
        current = list(source)
        amplitude = 1.0
        for site, creation in actions:
            if creation:
                current[site] += 1
                amplitude *= math.sqrt(current[site])
            elif current[site]:
                amplitude *= math.sqrt(current[site])
                current[site] -= 1
            else:
                amplitude = 0.0
                break
        if amplitude:
            matrix[rows[tuple(current)], column] = amplitude
    return source_basis, target_basis, matrix


def parity_sites(length, sites):
    return tuple(
        site for site in range(length)
        if site not in sites and sum(selected > site for selected in sites) % 2
    )


def binary_product_diagonals(length, sites):
    occupied = np.array([
        int(all(bits & (1 << site) for site in sites))
        for bits in range(1 << length)
    ], dtype=np.int64)
    empty = np.array([
        int(all(not bits & (1 << site) for site in sites))
        for bits in range(1 << length)
    ], dtype=np.int64)
    return occupied, empty


def mixed_expected(length, left, right):
    forward = bit_word_matrix(length, raising_word(right) + lowering_word(left))
    reverse = bit_word_matrix(length, lowering_word(left) + raising_word(right))
    return forward, reverse


def overlap_formula_expected(length, left, right):
    """Additional tensor-factor identity, separate from the signed-bit oracle."""
    selected_left, selected_right = set(left), set(right)
    only_left = selected_left - selected_right
    only_right = selected_right - selected_left
    overlap = selected_left & selected_right
    strings_left = set(parity_sites(length, left))
    strings_right = set(parity_sites(length, right))
    strings = (strings_left ^ strings_right) - (selected_left | selected_right)
    alpha = len(only_left & strings_right) + len(only_right & strings_left)
    epsilon = (-1) ** (
        len(left) * (len(left) - 1) // 2 + len(right) * (len(right) - 1) // 2
    )
    forward = np.zeros((1 << length, 1 << length), dtype=np.int64)
    reverse = np.zeros_like(forward)
    for source in range(1 << length):
        if any(not source & (1 << site) for site in only_left):
            continue
        if any(source & (1 << site) for site in only_right):
            continue
        target = source
        for site in only_left | only_right:
            target ^= 1 << site
        sign = epsilon * (-1) ** sum(bool(source & (1 << site)) for site in strings)
        if all(not source & (1 << site) for site in overlap):
            forward[target, source] = sign * (-1) ** alpha
        if all(source & (1 << site) for site in overlap):
            reverse[target, source] = sign
    return forward, reverse, alpha


def assert_immutable_detached(array, expected_dtype):
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.dtype(expected_dtype)
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)
    if array.size:
        with pytest.raises(ValueError):
            array.flat[0] = 91
    ancestor = array
    while isinstance(ancestor, np.ndarray):
        assert not ancestor.flags.writeable
        with pytest.raises(ValueError):
            ancestor.setflags(write=True)
        ancestor = ancestor.base
    assert isinstance(ancestor, bytes), "primitive storage must be immutable bytes"


def assert_native_json(value):
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        assert math.isfinite(value)
        return
    if type(value) is list:
        for child in value:
            assert_native_json(child)
        return
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for child in value.values():
            assert_native_json(child)
        return
    pytest.fail("non-JSON-native value of type {}".format(type(value).__name__))


def decode_matrix(value):
    assert set(value) == {"shape", "real", "imag"}
    shape = tuple(value["shape"])
    assert len(shape) == 2
    real = np.asarray(value["real"], dtype=np.float64).reshape(shape)
    imaginary = np.asarray(value["imag"], dtype=np.float64).reshape(shape)
    assert np.isfinite(real).all()
    assert np.isfinite(imaginary).all()
    return real + 1j * imaginary


BINARY_REPORT_KEYS = {
    "L", "sites", "degree", "dimension", "kind", "modulus", "neutrality_rule",
    "number_charge", "modular_charge", "grading", "modularly_neutral",
    "parity_sites", "tensor_support", "tensor_phase", "neutral_basis",
    "neutral_composite", "neutral_compression_status", "self_anticommutator_diagonal",
    "canonical_car", "nilpotent", "charge_identity_exact", "self_products_exact",
    "tensor_support_verified", "hard_core_self_products_exact", "exact_binary_arithmetic",
}
PAIR_REPORT_KEYS = {
    "L", "S", "T", "degree_S", "degree_T", "disjoint", "exchange_sign",
    "annihilator_exchange_exact", "mixed_graded_bracket_zero", "overlap_formula_exact",
    "hard_core_commuting_annihilators", "exact_binary_arithmetic", "mixed_forward",
    "mixed_reverse", "mixed_anticommutator", "mixed_graded_bracket",
}
BOSE_MAP_KEYS = {
    "L", "source_N", "target_N", "sites", "source_basis", "target_basis",
    "source_dimension", "target_dimension", "matrix", "status",
}
BOSE_REPORT_KEYS = {
    "L", "N", "sites", "degree", "dimension", "basis", "lowering_status",
    "lowering_shape", "creation_shape", "number_charge", "annihilation_product_diagonal",
    "creation_product_diagonal", "expected_annihilation_diagonal", "expected_creation_diagonal",
    "annihilation_product_frobenius_residual", "creation_product_frobenius_residual",
    "double_creation_vacuum", "exactness_certificate",
}


@pytest.mark.parametrize("length,sites", BINARY_CASES)
@pytest.mark.parametrize("kind,fermionic", (("jw", True), ("hard_core", False)))
def test_binary_complete_signed_bit_entries_and_detachment(api, length, sites, kind, fermionic):
    expected = binary_expected(length, sites, fermionic)
    support_input = list(sites)
    result = api.binary_composite(length, support_input, kind=kind)
    assert result.shape == (1 << length, 1 << length)
    np.testing.assert_array_equal(result, expected)
    assert set(np.unique(result)).issubset({-1, 0, 1})
    assert_immutable_detached(result, np.int64)
    support_input[:] = [length + 8]
    again = api.binary_composite(length, sites, kind=kind)
    assert not np.shares_memory(result, again)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(again, expected)


@pytest.mark.parametrize("length,sites", BINARY_CASES)
def test_binary_report_complete_grid_charge_self_products_and_locality(api, length, sites):
    report = api.binary_report(length, sites)
    assert BINARY_REPORT_KEYS <= set(report)
    degree = len(sites)
    dimension = 1 << length
    expected = binary_expected(length, sites)
    occupied, empty = binary_product_diagonals(length, sites)
    strings = parity_sites(length, sites)
    support = tuple(sorted(set(sites) | set(strings)))
    assert report["L"] == length
    assert tuple(report["sites"]) == sites
    assert report["degree"] == degree
    assert report["dimension"] == dimension
    assert report["kind"] == "jw"
    assert report["modulus"] == length
    assert report["neutrality_rule"] == "inherited"
    assert report["number_charge"] == -degree
    assert report["modular_charge"] == (-degree) % length
    assert report["grading"] == degree % 2
    assert report["modularly_neutral"] is (degree % length == 0)
    assert tuple(report["parity_sites"]) == strings
    assert tuple(report["tensor_support"]) == support
    assert report["tensor_phase"] == (-1) ** (degree * (degree - 1) // 2)
    neutral = (0, dimension - 1)
    assert tuple(report["neutral_basis"]) == neutral
    compressed = expected[np.ix_(neutral, neutral)]
    np.testing.assert_array_equal(decode_matrix(report["neutral_composite"]), compressed)
    assert report["neutral_compression_status"] == ("nonzero" if np.any(compressed) else "zero")
    np.testing.assert_array_equal(report["self_anticommutator_diagonal"], occupied + empty)
    assert report["canonical_car"] is (degree == 1)
    for field in (
        "nilpotent", "charge_identity_exact", "self_products_exact", "tensor_support_verified",
        "hard_core_self_products_exact", "exact_binary_arithmetic",
    ):
        assert report[field] is True, field
    np.testing.assert_array_equal(expected.T @ expected, np.diag(occupied))
    np.testing.assert_array_equal(expected @ expected.T, np.diag(empty))
    np.testing.assert_array_equal(expected @ expected, np.zeros_like(expected))
    numbers = np.array([popcount(bits) for bits in range(dimension)])
    np.testing.assert_array_equal(
        (numbers[:, None] - numbers[None, :]) * expected, -degree * expected,
    )
    # Z/density alone misses the strings. Onsite X is an essential probe.
    result = api.binary_composite(length, sites)
    for site in range(length):
        flipped = np.arange(dimension) ^ (1 << site)
        z = 1 - 2 * ((np.arange(dimension) >> site) & 1)
        x_commutes = np.array_equal(result[flipped, :], result[:, flipped])
        z_commutes = np.array_equal(z[:, None] * result, result * z[None, :])
        assert (not x_commutes or not z_commutes) is (site in support)
        if site in strings:
            assert z_commutes
            assert not x_commutes


@pytest.mark.parametrize("length,sites,q", (
    (4, (0, 1), 2), (4, (0, 1, 2), 3), (3, (0,), 3),
    (3, (0, 1), 2), (6, (0, 2, 4), 3), (8, (0, 7), 2),
))
def test_modulus_labels_and_complete_neutral_compression(api, length, sites, q):
    report = api.binary_report(length, sites, q=q)
    expected = binary_expected(length, sites)
    neutral = tuple(bits for bits in range(1 << length) if popcount(bits) % q == 0)
    assert report["modulus"] == q
    assert report["neutrality_rule"] == ("inherited" if q == length else "counterfactual")
    assert report["degree"] == len(sites)
    assert report["number_charge"] == -len(sites)
    assert report["modular_charge"] == (-len(sites)) % q
    assert report["modularly_neutral"] is (len(sites) % q == 0)
    assert report["grading"] == len(sites) % 2
    assert tuple(report["neutral_basis"]) == neutral
    np.testing.assert_array_equal(
        decode_matrix(report["neutral_composite"]), expected[np.ix_(neutral, neutral)],
    )
    assert api.binary_report(length, sites, q=length)["neutrality_rule"] == "inherited"


@pytest.mark.parametrize("length", BINARY_LENGTHS)
def test_full_neutral_operator_is_not_full_space_car(api, length):
    report = api.binary_report(length, tuple(range(length)))
    phase = (-1) ** (length * (length - 1) // 2)
    expected = np.zeros((1 << length, 1 << length), dtype=np.int64)
    expected[0, -1] = phase
    np.testing.assert_array_equal(api.binary_composite(length, tuple(range(length))), expected)
    np.testing.assert_array_equal(decode_matrix(report["neutral_composite"]), [[0, phase], [0, 0]])
    assert report["modularly_neutral"] is True
    assert report["canonical_car"] is False
    assert sum(report["self_anticommutator_diagonal"]) == 2
    assert report["grading"] == length % 2


def test_odd_degree_is_not_canonical_and_density_does_not_certify_locality(api):
    odd = api.binary_report(6, (0, 1, 2))
    assert odd["grading"] == 1
    assert odd["canonical_car"] is False
    assert odd["self_anticommutator_diagonal"][1] == 0
    assert api.binary_report(6, (5,))["canonical_car"] is True
    assert tuple(api.binary_report(6, (2, 3))["tensor_support"]) == (2, 3)
    assert tuple(api.binary_report(6, (1, 4))["tensor_support"]) == (1, 2, 3, 4)
    assert tuple(api.binary_report(6, (2,))["tensor_support"]) == (0, 1, 2)


@pytest.mark.parametrize("length,left,right", PAIR_CASES + (
    (3, (1,), (1, 2)), (3, (1, 2), (1,)), (4, (0, 2), (0, 2)),
    (4, (3,), (1, 3)), (6, (1, 3, 5), (0, 2, 4)),
))
def test_pair_matrices_dynamic_bits_overlap_and_exact_schema(api, length, left, right):
    report = api.binary_pair_report(length, left, right)
    assert PAIR_REPORT_KEYS <= set(report)
    forward, reverse = mixed_expected(length, left, right)
    formula_forward, formula_reverse, _ = overlap_formula_expected(length, left, right)
    np.testing.assert_array_equal(forward, formula_forward)
    np.testing.assert_array_equal(reverse, formula_reverse)
    sign = (-1) ** (len(left) * len(right))
    expected_fields = {
        "mixed_forward": forward, "mixed_reverse": reverse,
        "mixed_anticommutator": forward + reverse,
        "mixed_graded_bracket": forward - sign * reverse,
    }
    assert report["L"] == length
    assert tuple(report["S"]) == left
    assert tuple(report["T"]) == right
    assert report["degree_S"] == len(left)
    assert report["degree_T"] == len(right)
    assert report["disjoint"] is set(left).isdisjoint(right)
    assert report["exchange_sign"] == sign
    for key, expected in expected_fields.items():
        np.testing.assert_array_equal(report[key], expected)
        assert_immutable_detached(report[key], np.int64)
    assert report["mixed_graded_bracket_zero"] is (not np.any(forward - sign * reverse))
    for field in (
        "annihilator_exchange_exact", "overlap_formula_exact",
        "hard_core_commuting_annihilators", "exact_binary_arithmetic",
    ):
        assert report[field] is True, field
    left_then_right = bit_word_matrix(length, lowering_word(right) + lowering_word(left))
    right_then_left = bit_word_matrix(length, lowering_word(left) + lowering_word(right))
    np.testing.assert_array_equal(left_then_right, sign * right_then_left)
    if set(left) & set(right):
        assert not np.any(left_then_right)
        assert np.any(forward) and np.any(reverse)
    else:
        assert np.any(left_then_right)
    hc_forward = bit_word_matrix(length, lowering_word(right) + lowering_word(left), False)
    hc_reverse = bit_word_matrix(length, lowering_word(left) + lowering_word(right), False)
    np.testing.assert_array_equal(hc_forward, hc_reverse)


def test_alpha_counterexample_has_fixed_nonzero_entries(api):
    # S=(1,), T=(1,2): alpha=0, whereas rs-|I|=1 gives the wrong sign.
    forward, reverse, alpha = overlap_formula_expected(3, (1,), (1, 2))
    assert alpha == 0
    assert (len((1,)) * len((1, 2)) - len({1} & {1, 2})) % 2 == 1
    assert forward[4, 0] == -1
    assert forward[5, 1] == 1
    assert reverse[6, 2] == -1
    assert reverse[7, 3] == 1
    report = api.binary_pair_report(3, (1,), (1, 2))
    np.testing.assert_array_equal(report["mixed_forward"], forward)
    np.testing.assert_array_equal(report["mixed_reverse"], reverse)
    assert report["mixed_graded_bracket_zero"] is False


def bose_primitive_cases():
    for length in BOSE_LENGTHS:
        supports = DEMO_BOSE_SUPPORTS + ((length - 1,), (0, length - 1))
        for number in range(7):
            for sites in supports:
                yield length, number, sites


@pytest.mark.parametrize("length,number,sites", tuple(bose_primitive_cases()))
def test_bose_primitives_complete_untruncated_sectors(api, length, number, sites):
    source, target, expected = bose_word_matrix(length, number, lowering_word(sites))
    report = api.bose_composite_map(length, number, sites)
    assert BOSE_MAP_KEYS <= set(report)
    assert report["L"] == length
    assert report["source_N"] == number
    assert tuple(report["sites"]) == sites
    assert tuple(map(tuple, report["source_basis"])) == source
    assert tuple(map(tuple, report["target_basis"])) == target
    assert report["source_dimension"] == math.comb(number + length - 1, length - 1)
    assert report["target_dimension"] == len(target)
    assert report["source_dimension"] <= 512
    assert report["target_dimension"] <= 512
    assert report["target_N"] == (number - len(sites) if number >= len(sites) else None)
    assert report["status"] == ("complete_sector_map" if number >= len(sites) else "absent_target")
    matrix = report["matrix"]
    assert matrix.shape == (len(target), len(source))
    np.testing.assert_allclose(matrix, expected, rtol=0, atol=ATOL)
    assert_immutable_detached(matrix, np.float64)
    other = api.bose_composite_map(length, number, sites)["matrix"]
    assert not np.shares_memory(matrix, other)
    if number < len(sites):
        assert report["target_basis"] == ()
        assert matrix.shape[0] == 0
    if number > length:
        assert max(max(state) for state in source) == number


@pytest.mark.parametrize("length,number,sites", BOSE_CASES)
def test_bose_report_products_and_double_creation_against_elementary_oracle(api, length, number, sites):
    report = api.bose_report(length, number, sites)
    assert BOSE_REPORT_KEYS <= set(report)
    basis, target, lowering = bose_word_matrix(length, number, lowering_word(sites))
    _, created_basis, creation = bose_word_matrix(length, number, raising_word(sites))
    degree = len(sites)
    expected_annihilation = [math.prod(state[site] for site in sites) for state in basis]
    expected_creation = [math.prod(state[site] + 1 for site in sites) for state in basis]
    assert report["L"] == length
    assert report["N"] == number
    assert tuple(report["sites"]) == sites
    assert report["degree"] == degree
    assert report["dimension"] == len(basis)
    assert tuple(map(tuple, report["basis"])) == basis
    assert report["number_charge"] == -degree
    assert report["lowering_status"] == ("absent_target" if number < degree else "complete_sector_map")
    assert tuple(report["lowering_shape"]) == (len(target), len(basis))
    assert tuple(report["creation_shape"]) == (len(created_basis), len(basis))
    assert report["exactness_certificate"] is False
    for name in (
        "annihilation_product_diagonal", "creation_product_diagonal",
        "expected_annihilation_diagonal", "expected_creation_diagonal",
    ):
        assert type(report[name]) is list
    np.testing.assert_array_equal(report["expected_annihilation_diagonal"], expected_annihilation)
    np.testing.assert_array_equal(report["expected_creation_diagonal"], expected_creation)
    np.testing.assert_allclose(report["annihilation_product_diagonal"], expected_annihilation, rtol=0, atol=ATOL)
    np.testing.assert_allclose(report["creation_product_diagonal"], expected_creation, rtol=0, atol=ATOL)
    np.testing.assert_allclose(lowering.T @ lowering, np.diag(expected_annihilation), rtol=0, atol=ATOL)
    np.testing.assert_allclose(creation.T @ creation, np.diag(expected_creation), rtol=0, atol=ATOL)
    for prefix in ("annihilation", "creation"):
        residual = report[prefix + "_product_frobenius_residual"]
        assert math.isfinite(residual) and 0 <= residual < ATOL
        # Independently reconstruct the raw Frobenius residual from reported diagonals.
        observed = np.asarray(report[prefix + "_product_diagonal"])
        predicted = np.asarray(report["expected_" + prefix + "_diagonal"])
        diagonal_residual = math.sqrt(sum(float(value) ** 2 for value in observed - predicted))
        assert residual == pytest.approx(diagonal_residual, rel=1e-12, abs=1e-30)
    witness = report["double_creation_vacuum"]
    assert {
        "target_N", "target_basis", "amplitudes", "expected_amplitudes", "frobenius_residual",
        "norm_squared", "expected_norm_squared",
    } <= set(witness)
    _, final_basis, twice = bose_word_matrix(length, 0, raising_word(sites) * 2)
    expected_amplitudes = np.zeros(len(final_basis), dtype=np.float64)
    final_state = tuple(2 if site in sites else 0 for site in range(length))
    expected_amplitudes[final_basis.index(final_state)] = math.sqrt(2 ** degree)
    assert witness["target_N"] == 2 * degree <= 6
    assert tuple(map(tuple, witness["target_basis"])) == final_basis
    assert type(witness["amplitudes"]) is list
    assert type(witness["expected_amplitudes"]) is list
    np.testing.assert_allclose(witness["amplitudes"], twice[:, 0], rtol=0, atol=ATOL)
    np.testing.assert_allclose(witness["expected_amplitudes"], expected_amplitudes, rtol=0, atol=ATOL)
    assert witness["norm_squared"] == pytest.approx(2 ** degree, rel=0, abs=ATOL)
    assert witness["expected_norm_squared"] == 2 ** degree
    residual = witness["frobenius_residual"]
    raw = math.sqrt(sum(
        (float(observed) - float(expected)) ** 2
        for observed, expected in zip(witness["amplitudes"], witness["expected_amplitudes"])
    ))
    assert math.isfinite(residual) and 0 <= residual < ATOL
    assert residual == pytest.approx(raw, rel=1e-12, abs=1e-30)
    if number < degree:
        assert not any(report["annihilation_product_diagonal"])
        assert all(value > 0 for value in report["creation_product_diagonal"])


def test_bose_auxiliary_maximum_sector_and_raw_roundoff_are_not_erased(api):
    maximum = api.bose_composite_map(5, 6, (0, 1, 2))
    assert maximum["source_dimension"] == 210
    assert (6, 0, 0, 0, 0) in maximum["source_basis"]
    report = api.bose_report(3, 2, (0,))
    diagonal = report["annihilation_product_diagonal"]
    expected = report["expected_annihilation_diagonal"]
    assert any(observed != target for observed, target in zip(diagonal, expected))
    assert report["annihilation_product_frobenius_residual"] > 0.0
    assert report["exactness_certificate"] is False


INVALID_SUPPORTS = (
    (), [], (0, 0), (1, 0), (True,), (False,), (np.bool_(True),), (1.0,),
    (-1,), (3,), (0, 1, 2, 3), "0", {0}, frozenset((0,)), np.array([0]), None,
)


@pytest.mark.parametrize("sites", INVALID_SUPPORTS)
@pytest.mark.parametrize("function", ("binary_composite", "binary_report", "binary_pair_report", "bose_composite_map", "bose_report"))
def test_invalid_supports_raise_value_error(api, sites, function):
    call = getattr(api, function)
    if function == "binary_pair_report":
        with pytest.raises(ValueError):
            call(3, sites, (1,))
        with pytest.raises(ValueError):
            call(3, (0,), sites)
    elif function.startswith("bose"):
        with pytest.raises(ValueError):
            call(3, 1, sites)
    else:
        with pytest.raises(ValueError):
            call(3, sites)


class UntouchableIterable:
    def __iter__(self):
        raise AssertionError("unbounded input must be rejected without iteration")


class UntouchableInteger:
    def __int__(self):
        raise AssertionError("overlong support must be rejected before element conversion")

    def __index__(self):
        raise AssertionError("overlong support must be rejected before element conversion")


def test_iterators_and_overlong_support_rejected_before_access(api):
    for sites in (UntouchableIterable(), iter((0,)), (value for value in range(3))):
        with pytest.raises(ValueError):
            api.binary_composite(3, sites)
        with pytest.raises(ValueError):
            api.bose_composite_map(3, 2, sites)
    for sites in ([UntouchableInteger()] * 9, tuple(UntouchableInteger() for _ in range(9))):
        with pytest.raises(ValueError):
            api.binary_composite(8, sites)
        with pytest.raises(ValueError):
            api.bose_composite_map(5, 6, sites)


@pytest.mark.parametrize("bad", (True, False, np.bool_(True), 3.0, "3", None, -1, 0, 2, 9, 10 ** 100))
def test_binary_length_validation(api, bad):
    for function, args in (
        (api.binary_composite, (bad, (0,))),
        (api.binary_report, (bad, (0,))),
        (api.binary_pair_report, (bad, (0,), (1,))),
    ):
        with pytest.raises(ValueError):
            function(*args)


@pytest.mark.parametrize("bad", (True, False, np.bool_(True), 3.0, "3", None, -1, 0, 2, 6, 10 ** 100))
def test_bose_length_validation(api, bad):
    for function in (api.bose_composite_map, api.bose_report):
        with pytest.raises(ValueError):
            function(bad, 1, (0,))


@pytest.mark.parametrize("bad", (True, False, np.bool_(True), 1.0, "1", None, -1, 7, 10 ** 100))
def test_bose_number_validation(api, bad):
    for function in (api.bose_composite_map, api.bose_report):
        with pytest.raises(ValueError):
            function(3, bad, (0,))


@pytest.mark.parametrize("number", (4, 5, 6))
def test_bose_reference_reports_reject_auxiliary_only_numbers(api, number):
    with pytest.raises(ValueError):
        api.bose_report(3, number, (0,))
    assert api.bose_composite_map(3, number, (0,))["source_N"] == number


@pytest.mark.parametrize("bad", (True, np.bool_(True), 2.0, "2", 0, -1, 1, 5, 10 ** 100))
def test_modulus_validation(api, bad):
    with pytest.raises(ValueError):
        api.binary_report(4, (0, 1), q=bad)


@pytest.mark.parametrize("bad", (None, True, 0, "JW", "fermion", "hard-core", "", ["jw"]))
def test_kind_is_exact_and_validated(api, bad):
    with pytest.raises(ValueError):
        api.binary_composite(3, (0,), kind=bad)


def test_numpy_integer_scalars_and_list_supports_are_accepted(api):
    length = np.int64(4)
    sites = [np.int32(0), np.int64(1)]
    np.testing.assert_array_equal(api.binary_composite(length, sites), binary_expected(4, (0, 1)))
    assert api.binary_report(length, sites, q=np.int64(2))["modulus"] == 2
    assert api.binary_pair_report(length, sites, [np.int64(2)])["degree_T"] == 1
    assert api.bose_composite_map(length, np.int64(2), sites)["target_N"] == 0
    assert api.bose_report(length, np.int64(2), sites)["N"] == 2


def test_invalid_public_inputs_reject_before_dense_allocation(api, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation attempted before invalid input was rejected")

    for name in ("zeros", "ones", "empty", "full", "eye", "kron"):
        monkeypatch.setattr(np, name, forbidden)
    calls = (
        (api.binary_composite, (9, (0,))),
        (api.binary_composite, (8, (0, 0))),
        (api.binary_pair_report, (9, (0,), (1,))),
        (api.binary_report, (4, (0,)), {"q": 17}),
        (api.bose_composite_map, (5, 7, (0,))),
        (api.bose_composite_map, (5, 6, (0, 1, 2, 3))),
        (api.bose_report, (5, 4, (0,))),
    )
    for item in calls:
        function, args = item[:2]
        kwargs = item[2] if len(item) == 3 else {}
        with pytest.raises(ValueError):
            function(*args, **kwargs)


@pytest.mark.parametrize("family,cap,function,args", (
    ("binary", 255, "binary_composite", (8, (0,))),
    ("binary", 255, "binary_report", (8, (0, 1))),
    ("binary", 63, "binary_pair_report", (6, (0, 1, 2), (3, 4, 5))),
    ("bose", 209, "bose_composite_map", (5, 6, (0, 1, 2))),
    ("bose", 0, "bose_composite_map", (5, 0, (0, 1, 2))),
    # The reference sector fits both reductions. Auxiliary sectors do not.
    ("bose", 100, "bose_report", (5, 3, (0, 1, 2))),
    # Here even N+r fits: only the double-creation vacuum sector exceeds cap.
    ("bose", 100, "bose_report", (5, 0, (0, 1, 2))),
))
def test_dimension_caps_and_joint_auxiliary_preflight_before_allocation(api, monkeypatch, family, cap, function, args):
    assert api.MAX_BINARY_DIMENSION == 256
    assert api.MAX_BOSE_DIMENSION == 512
    constant = "MAX_BINARY_DIMENSION" if family == "binary" else "MAX_BOSE_DIMENSION"
    monkeypatch.setattr(api, constant, cap)

    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError("dense allocation occurred before all sector caps were checked")

    for name in ("zeros", "zeros_like", "ones", "ones_like", "empty", "full", "eye", "kron"):
        monkeypatch.setattr(np, name, forbidden)
    with pytest.raises(ValueError):
        getattr(api, function)(*args)


def test_caps_are_inclusive_and_not_local_occupation_truncations(api, monkeypatch):
    monkeypatch.setattr(api, "MAX_BINARY_DIMENSION", 8)
    assert api.binary_composite(3, (0,)).shape == (8, 8)
    monkeypatch.setattr(api, "MAX_BOSE_DIMENSION", 210)
    assert api.bose_composite_map(5, 6, (0,))["source_dimension"] == 210
    witness = api.bose_report(5, 0, (0, 1, 2))["double_creation_vacuum"]
    assert len(witness["target_basis"]) == 210


def test_algebra_does_not_require_eigenanalysis(api, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("kinematic map unexpectedly requested eigenanalysis")

    for name in ("eig", "eigh", "eigvals", "eigvalsh", "svd"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    assert api.binary_report(3, (0, 1))["self_products_exact"] is True
    assert api.binary_pair_report(3, (1,), (1, 2))["overlap_formula_exact"] is True
    assert api.bose_report(3, 0, (0, 1, 2))["exactness_certificate"] is False
    assert issubclass(api.NumericalUnavailable, ValueError)


@pytest.fixture(scope="module")
def demonstration(api):
    return api.demonstration_report()


def test_demonstration_has_exact_preregistered_grids_and_strict_json(demonstration):
    assert {
        "binary_cases", "pair_cases", "counterfactual_cases", "bose_cases",
        "limitations", "numerical_domain",
    } <= set(demonstration)
    assert_native_json(demonstration)
    serialized = json.dumps(demonstration, allow_nan=False)
    assert json.loads(serialized) == demonstration
    actual_binary = [(row["L"], tuple(row["sites"])) for row in demonstration["binary_cases"]]
    assert len(actual_binary) == len(BINARY_CASES)
    assert set(actual_binary) == set(BINARY_CASES)
    assert all(row["modulus"] == row["L"] for row in demonstration["binary_cases"])
    actual_pairs = [(row["L"], tuple(row["S"]), tuple(row["T"])) for row in demonstration["pair_cases"]]
    assert len(actual_pairs) == 6
    assert set(actual_pairs) == set(PAIR_CASES)
    actual_bose = [(row["L"], row["N"], tuple(row["sites"])) for row in demonstration["bose_cases"]]
    assert len(actual_bose) == 36
    assert set(actual_bose) == set(BOSE_CASES)
    actual_counterfactual = [
        (row["L"], tuple(row["sites"]), row["modulus"])
        for row in demonstration["counterfactual_cases"]
    ]
    assert len(actual_counterfactual) == 2
    assert set(actual_counterfactual) == {(4, (0, 1), 2), (4, (0, 1, 2), 3)}
    assert all(row["neutrality_rule"] == "counterfactual" for row in demonstration["counterfactual_cases"])
    for row in demonstration["pair_cases"]:
        forward, reverse = mixed_expected(row["L"], tuple(row["S"]), tuple(row["T"]))
        sign = (-1) ** (row["degree_S"] * row["degree_T"])
        for field, expected in (
            ("mixed_forward", forward), ("mixed_reverse", reverse),
            ("mixed_anticommutator", forward + reverse),
            ("mixed_graded_bracket", forward - sign * reverse),
        ):
            np.testing.assert_array_equal(decode_matrix(row[field]), expected)
    # Domain metadata and limitations must remain substantive, not empty flags.
    assert demonstration["numerical_domain"]
    limitations = json.dumps(demonstration["limitations"]).lower()
    for word in ("grading", "canonical", "tensor", "neutral", "binding", "fermion", "empirical"):
        assert word in limitations
    assert any(phrase in limitations for phrase in ("no empirical", "not empirical", "without empirical"))


def test_empty_map_shapes_survive_explicit_strict_json_serialization(api, demonstration):
    primitive = api.bose_composite_map(5, 0, (0, 1, 2))
    matrix = primitive["matrix"]
    payload = {
        "shape": list(matrix.shape), "real": matrix.real.tolist(),
        "imag": matrix.imag.tolist(),
    }
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded == {"shape": [0, 1], "real": [], "imag": []}
    assert decode_matrix(decoded).shape == (0, 1)
    for row in demonstration["bose_cases"]:
        if row["N"] < row["degree"]:
            assert row["lowering_shape"] == [0, row["dimension"]]
            assert row["creation_shape"][0] > 0
            assert row["lowering_status"] == "absent_target"


@pytest.mark.parametrize("json_mode", (False, True))
def test_demo_stdout_only_from_empty_directory(json_mode, tmp_path, demonstration):
    command = [sys.executable, "-B", "-W", "error", str(ROOT / "scripts" / "demo_substrate_composite_statistics.py")]
    if json_mode:
        command.append("--json")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command, cwd=str(tmp_path), env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.strip()
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        def reject_nonfinite(token):
            raise AssertionError("non-standard JSON numeric token: " + token)

        result = json.loads(completed.stdout, parse_constant=reject_nonfinite)
        assert_native_json(result)
        assert result == demonstration
    else:
        lower = completed.stdout.lower()
        assert "bose" in lower
        assert "binary" in lower
        assert "empirical" in lower


@pytest.mark.parametrize("relative", (
    "bpr/substrate_composite_statistics.py", "scripts/demo_substrate_composite_statistics.py",
    "tests/test_substrate_composite_statistics.py",
))
def test_new_python_files_parse_with_python38_grammar(relative):
    source = (ROOT / relative).read_text(encoding="utf-8")
    ast.parse(source, filename=relative, feature_version=(3, 8))
