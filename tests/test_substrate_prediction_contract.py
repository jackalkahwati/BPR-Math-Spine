"""Independent frozen module6 contract tests, authored before control execution.

The occupation construction, projector contractions and full-space resolvent
oracle do not call production source, Lehmann or comparison implementations.
Only the four contracted fixture helpers and inherited construction boundaries
are used for numerical/adversarial injections. Extra report metadata is allowed.
"""
import ast
import copy
from fractions import Fraction
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_prediction_contract as api
from bpr import substrate_joint_source_kernel as joint
from bpr import substrate_current_response as current
from bpr import substrate_energy_response as energy
from bpr import substrate_fermionization as occupation_module


ROOT = Path(__file__).resolve().parents[1]
ETAS = (0.5, 1.0, 2.0)
SCALES = (0.5, 1.0, 2.0)
GRID = tuple((L, q) for L in (3, 4, 5) for q in (0.0, 0.7, 40.0))
TOL = 2e-10
AVAILABLE = "available_heuristic"


def same(actual, expected):
    """Separate real/imaginary per-entry reference tolerances, not a norm."""
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape
    np.testing.assert_allclose(a.real, b.real, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(a.imag, b.imag, atol=TOL, rtol=TOL)


def decode(value):
    assert {"shape", "real", "imag"} <= set(value)
    result = np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
    assert list(result.shape) == value["shape"]
    return result


def scalar(value):
    assert {"real", "imag"} <= set(value)
    return complex(value["real"], value["imag"])


def native(value):
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            native(item)
    elif isinstance(value, list):
        for item in value:
            native(item)
    else:
        assert value is None or type(value) in (int, float, bool, str)
        if type(value) is float:
            assert math.isfinite(value)


def scope_flags(scope):
    assert scope["empirical_status"] == "empirical_test_unavailable"
    assert scope["numerical_error_certified"] is False
    assert scope["empirical_validation"] is False


def record(record, status=AVAILABLE, complex_value=False):
    assert {"status", "value", "reason"} <= set(record)
    assert record["status"] == status
    if status == AVAILABLE:
        assert record["reason"] is None
        if complex_value:
            return scalar(record["value"])
        assert type(record["value"]) in (int, float)
        return record["value"]
    assert record["value"] is None
    assert type(record["reason"]) is str and record["reason"]
    if complex_value:
        assert record["proxy"] is None
        assert record["witness_resolved"] is False
    return None


def comparator(item, expected="consistent"):
    required = {"status", "max_real_difference", "max_imag_difference", "failed_components"}
    assert required <= set(item)
    assert item["status"] == expected
    if expected == "inconclusive":
        assert item["max_real_difference"] is None
        assert item["max_imag_difference"] is None
        assert item["failed_components"] is None
    else:
        assert item["max_real_difference"] >= 0
        assert item["max_imag_difference"] >= 0
        if expected == "consistent":
            assert item["failed_components"] == 0
        else:
            assert item["failed_components"] > 0


def comparator_leaves(value):
    if isinstance(value, list):
        for item in value:
            yield from comparator_leaves(item)
    else:
        yield value


def occupations(L, N):
    if L == 1:
        yield (N,)
    else:
        for n in range(N + 1):
            for rest in occupations(L - 1, N - n):
                yield (n,) + rest


def occupation_oracle(L, g, C):
    """Independent complete occupation H, symmetric sources and number current."""
    basis = tuple(occupations(L, L))
    index = {state: i for i, state in enumerate(basis)}
    dimension = math.comb(2 * L - 1, L)
    assert dimension == len(basis)
    bonds, onsite, currents = [], [], []
    for x in range(L):
        y = (x + 1) % L
        transfer = np.zeros((dimension, dimension))
        for col, state in enumerate(basis):
            if state[x]:
                target = list(state)
                target[x] -= 1
                target[y] += 1
                transfer[index[tuple(target)], col] = math.sqrt(state[x] * (state[y] + 1))
        bonds.append(-C * (transfer + transfer.T))
        currents.append(1j * C * (transfer - transfer.T))
        onsite.append(np.diag([g * state[x] * (state[x] - 1) / 2 for state in basis]))
    H = sum(bonds) + sum(onsite)
    weights = []
    for x in range(L):
        if x == 0:
            cosine = 1.0
        elif 2 * x == L:
            cosine = -1.0
        elif 4 * x in (L, 3 * L):
            cosine = 0.0
        else:
            cosine = math.cos(2 * math.pi * x / L)
        weights.append(math.sqrt(2.0 / L) * cosine)
    rho = np.diag([sum(weights[x] * (state[x] - 1) for x in range(L)) for state in basis])
    h = sum(weights[x] * (onsite[x] + (bonds[x - 1] + bonds[x]) / 2) for x in range(L))
    energies, vectors = np.linalg.eigh(H)
    ground = vectors[:, 0]
    sources = np.array([rho, h], dtype=complex)
    columns = np.column_stack([operator @ ground for operator in sources])
    connected = columns - np.outer(ground, ground.conj() @ columns / np.vdot(ground, ground))
    excitation = H - energies[0] * np.eye(dimension)
    return {"basis": basis, "H": H, "sources": sources, "currents": currents,
            "energies": energies, "gaps": energies - energies[0], "ground": ground,
            "connected": connected, "W": connected.conj().T @ connected,
            "M": connected.conj().T @ excitation @ connected, "A": excitation}


def resolvent_kernel(oracle, z):
    """Full-space solves, not reused eigenstate denominators/Lehmann sums."""
    connected = oracle["connected"]
    eye = np.eye(len(oracle["H"]))
    forward = connected.conj().T @ np.linalg.solve(z * eye - oracle["A"], connected)
    backward = connected.conj().T @ np.linalg.solve(z * eye + oracle["A"], connected)
    return forward - backward.T


def proxy_oracle(amplitudes, errors, resolution_over_C, delta1, eta):
    y = eta * delta1
    gamma = 2 * resolution_over_C
    return np.array([[2 * (errors[a] * amplitudes[b] + errors[b] * amplitudes[a] +
                           errors[a] * errors[b]) / y +
                      2 * (1 + eta) * gamma * (amplitudes[a] + errors[a]) *
                      (amplitudes[b] + errors[b]) / (y * y)
                      for b in range(2)] for a in range(2)])


def rational_complex(value):
    z = complex(value)
    return Fraction(z.real), Fraction(z.imag)


def multiply(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def exact_ratio(kernel):
    numerator = multiply(rational_complex(kernel[0, 1]), rational_complex(kernel[1, 0]))
    denominator = multiply(rational_complex(kernel[0, 0]), rational_complex(kernel[1, 1]))
    norm = denominator[0] ** 2 + denominator[1] ** 2
    return complex(float((numerator[0] * denominator[0] + numerator[1] * denominator[1]) / norm),
                   float((numerator[1] * denominator[0] - numerator[0] * denominator[1]) / norm))


def detached_arrays(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
    if isinstance(value, dict):
        return {k: detached_arrays(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detached_arrays(v) for v in value)
    return value


def owned_fixture(gaps=None, rotation=None, unresolved=False, complex_sources=False):
    """Consistent 10-dimensional private owned fixture at inherited boundary.

    Deliberately not a public arbitrary-model API. All excited gaps are actual
    diagonal entries, and complete degenerate subspaces can be unitarily rotated.
    """
    if gaps is None:
        gaps = np.array([0., 2., 2., 3., 4., 5., 6., 7., 8., 9.])
    gaps = np.asarray(gaps, dtype=float)
    dimension = len(gaps)
    vectors = np.eye(dimension, dtype=complex)
    if rotation is not None:
        vectors[:, 1:3] = vectors[:, 1:3] @ rotation
    H = np.diag(gaps).astype(complex)
    sources = np.zeros((2, dimension, dimension), dtype=complex)
    amplitudes = np.array([[1., 2.], [2., -1.], [0.5, 3.]], dtype=complex)
    if complex_sources:
        amplitudes[0, 1] += 0.5j  # W01 imaginary part is -1/4, not canceled.
        amplitudes[2, 0] += 0.25j
    for a in range(2):
        sources[a, 1:4, 0] = amplitudes[:, a]
        sources[a, 0, 1:4] = amplitudes[:, a].conj()
    if unresolved:
        sources[0] *= 2.0 ** -40
        sources[0, 0, 0] = 100.0
    ground = vectors[:, 0]
    columns = np.column_stack([source @ ground for source in sources])
    connected = columns - np.outer(ground, ground.conj() @ columns)
    transitions = vectors[:, 1:].conj().T @ connected
    grams = np.array([np.outer(row.conj(), row) for row in transitions])
    item = {"sources": sources, "connected": connected, "transitions": transitions, "grams": grams}
    return detached_arrays({"L": 3, "N": 3, "C": 1.0, "g": 0.7, "m": 1,
                            "k": 2 * math.pi / 3, "dimension": dimension,
                            "basis": tuple(occupations(3, 3)), "H": H, "energies": gaps,
                            "vectors": vectors, "ground": ground, "gaps": gaps,
                            "resolution": 1e-12,
                            "partitions": {"symmetric": item, "improved": copy.deepcopy(item)}})


def patch_joint(monkeypatch, replacement):
    """Patch the documented inherited boundary, including direct imported aliases."""
    original = joint.joint_system
    monkeypatch.setattr(joint, "joint_system", replacement)
    for name, value in list(vars(api).items()):
        if value is original:
            monkeypatch.setattr(api, name, replacement)


def fixed_fixture(monkeypatch, system=None):
    system = owned_fixture() if system is None else system
    calls = []
    def construct(L, g, C=1.0, m=1):
        calls.append((L, g, C, m))
        return system
    patch_joint(monkeypatch, construct)
    return system, calls


@pytest.fixture(scope="module")
def demonstration():
    """The one scientific demonstration also audits 27 solves, never a 28th."""
    solves = []
    original = np.linalg.eigh
    def counted(*args, **kwargs):
        solves.append(np.shape(args[0]))
        return original(*args, **kwargs)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np.linalg, "eigh", counted)
        report = api.demonstration_report()
    assert len(solves) == 27
    assert solves == [(math.comb(2 * L - 1, L),) * 2 for L, q in GRID for scale in SCALES]
    return report


@pytest.fixture(scope="module", params=tuple(range(9)), ids=lambda i: "L%d-q%g" % GRID[i])
def scaling_case(request, demonstration):
    return GRID[request.param], demonstration["scaling_cases"][request.param]


def test_fixed_constants_and_no_injected_public_data_or_fitting_parameters():
    assert api.NumericalUnavailable is current.NumericalUnavailable
    assert api.MAX_DIMENSION == 512
    assert tuple(api.ETAS) == ETAS
    assert tuple(api.SCALES) == SCALES
    assert api.ATOL == api.RTOL == TOL
    assert list(inspect.signature(api.case_report).parameters) == ["L", "g", "C"]
    assert inspect.signature(api.case_report).parameters["C"].default == 1.0
    assert list(inspect.signature(api.scaling_report).parameters) == ["L", "g_over_C"]
    assert not inspect.signature(api.heldout_report).parameters
    assert not inspect.signature(api.demonstration_report).parameters


def test_full_case_oracles_and_componentwise_scaling(scaling_case):
    (L, q), scaling = scaling_case
    assert {"L", "g_over_C", "scales", "cases", "comparisons", "scope"} <= set(scaling)
    assert scaling["L"] == L and scaling["g_over_C"] == q
    assert scaling["scales"] == list(SCALES)
    assert [slot["scale"] for slot in scaling["cases"]] == list(SCALES)
    scope_flags(scaling["scope"])
    for slot in scaling["cases"]:
        C = slot["scale"]
        report = slot["report"]
        assert slot["status"] == AVAILABLE and slot["reason"] is None
        required = {"L", "N", "C", "g", "m", "dimension", "status", "scope", "gaps", "sources",
                    "total_weight", "first_moment", "normalized_weight", "normalized_first_moment",
                    "frequencies", "observables", "free_control"}
        assert required <= set(report)
        assert (report["L"], report["N"], report["C"], report["g"], report["m"]) == (L, L, C, q * C, 1)
        assert report["dimension"] == math.comb(2 * L - 1, L)
        assert report["status"] == AVAILABLE
        scope_flags(report["scope"])
        assert report["scope"]["gap_convention"] == "counted_multiplicity"
        assert report["scope"]["partition"] == "symmetric"
        oracle = occupation_oracle(L, q * C, C)
        gaps = report["gaps"]
        for j in (1, 2):
            same(gaps["delta%d" % j], oracle["gaps"][j])
            same(gaps["normalized_delta%d" % j], oracle["gaps"][j] / C)
        same(record(gaps["ratio"]), oracle["gaps"][2] / oracle["gaps"][1])
        same(gaps["excited_splitting"], oracle["gaps"][2] - oracle["gaps"][1])
        assert gaps["excited_splitting_status"] in ("resolved_excited_splitting", "unresolved_excited_splitting")
        degrees = np.array([0, 1])
        factors = C ** (degrees[:, None] + degrees[None, :])
        same(decode(report["total_weight"]), oracle["W"])
        same(decode(report["first_moment"]), oracle["M"])
        same(decode(report["normalized_weight"]), oracle["W"] / factors)
        same(decode(report["normalized_first_moment"]), oracle["M"] / (C * factors))
        amplitudes, errors = [], []
        for a, name in enumerate(("rho", "h")):
            operator = oracle["sources"][a]
            source = report["sources"][name]
            assert source["status"] == AVAILABLE
            amplitude = np.linalg.norm(oracle["connected"][:, a]) / C ** degrees[a]
            norm = np.linalg.norm(operator, "fro") / C ** degrees[a]
            error = 256 * np.finfo(float).eps * len(operator) * norm
            same(source["amplitude"], amplitude)
            same(source["operator_frobenius_norm"], norm)
            np.testing.assert_allclose(source["amplitude_proxy"], error, atol=0., rtol=TOL)
            amplitudes.append(amplitude)
            errors.append(error)
            commutator = oracle["H"] @ operator - operator @ oracle["H"]
            double = operator @ commutator - commutator @ operator
            first = np.vdot(oracle["ground"], double @ oracle["ground"]) / 2
            same(first, oracle["M"][a, a])
            expected = (oracle["M"][a, a] / (oracle["gaps"][1] * oracle["W"][a, a])).real
            same(record(source["normalized_moment"]), expected)
            assert expected >= 1 - TOL
        assert [item["eta"] for item in report["frequencies"]] == list(ETAS)
        assert [item["role"] for item in report["frequencies"]] == ["development", "development", "heldout"]
        for eta, frequency in zip(ETAS, report["frequencies"]):
            assert frequency["evaluation_status"] == AVAILABLE and frequency["reason"] is None
            z = 1j * eta * oracle["gaps"][1]
            same(scalar(frequency["z"]), z)
            kernel = resolvent_kernel(oracle, z)
            normalized = kernel * C / factors
            same(decode(frequency["kernel"]), kernel)
            same(decode(frequency["normalized_kernel"]), normalized)
            U = proxy_oracle(amplitudes, errors, gaps["resolution"] / C, gaps["delta1"] / C, eta)
            np.testing.assert_allclose(frequency["entry_proxy"], U, atol=0., rtol=TOL)
            value = record(frequency["ratio"], complex_value=True)
            same(value, exact_ratio(normalized))
            assert -TOL <= value.real <= 1 + TOL
            assert np.linalg.eigvalsh(-normalized.real).min() >= -TOL
            assert frequency["ratio"]["proxy"] >= 0
        observables = report["observables"]
        assert observables["gap_ratio"] == gaps["ratio"]
        assert observables["density_moment"] == report["sources"]["rho"]["normalized_moment"]
        assert observables["energy_moment"] == report["sources"]["h"]["normalized_moment"]
        assert observables["joint_ratios"] == [frequency["ratio"] for frequency in report["frequencies"]]
        if q == 0:
            free = report["free_control"]
            assert free["expected"] == [1] * 6
            assert len(free["comparisons"]) == 6
            for comparison in free["comparisons"]:
                comparator(comparison)
            alpha = -C * (1 + math.cos(2 * math.pi / L))
            same(decode(report["total_weight"]), np.array([[1., alpha], [alpha, alpha * alpha]]))
            same(gaps["delta1"], 4 * C * math.sin(math.pi / L) ** 2)
            same(record(gaps["ratio"]), 1.0)
        else:
            assert report["free_control"] is None
    assert [item["scale"] for item in scaling["comparisons"]] == list(SCALES)
    for comparison in scaling["comparisons"]:
        assert comparison["status"] == "consistent"
        for name in ("gaps", "total_weight", "first_moment", "kernels", "sources", "number_currents", "observables"):
            for item in comparator_leaves(comparison[name]):
                comparator(item)
    native(scaling)
    json.dumps(scaling, allow_nan=False)


@pytest.mark.parametrize("scale_index", (0, 2))
def test_raw_scaling_exponents_include_mixed_channels(scaling_case, scale_index):
    _, report = scaling_case
    base = report["cases"][1]["report"]
    scaled = report["cases"][scale_index]["report"]
    s = SCALES[scale_index]
    exponents = np.array([[0, 1], [1, 2]])
    same(decode(scaled["total_weight"]), decode(base["total_weight"]) * s ** exponents)
    same(decode(scaled["first_moment"]), decode(base["first_moment"]) * s ** (exponents + 1))
    for actual, reference in zip(scaled["frequencies"], base["frequencies"]):
        same(decode(actual["kernel"]), decode(reference["kernel"]) * s ** (exponents - 1))
    # This is all four entries, including the nontrivial mixed source channels.
    same(scaled["gaps"]["delta1"], s * base["gaps"]["delta1"])
    same(scaled["gaps"]["delta2"], s * base["gaps"]["delta2"])


def test_multiplicity_is_not_distinct_level_and_no_excited_splitting_gate():
    available = api._gap_ratio([0., 2., 2., 4.], 1e-6)
    assert record(available) == 1.0
    unsnapped = np.nextafter(2.0, math.inf)
    value = record(api._gap_ratio([0., 2., unsnapped, 4.], 1e-6))
    assert value == unsnapped / 2.0 and value != 1.0
    record(api._gap_ratio([0., 1e-6, 2., 3.], 1e-6), "unresolved_gap")
    assert record(api._gap_ratio([0., np.nextafter(1e-6, math.inf), 2., 3.], 1e-6)) > 1


def test_actual_gaps_not_cluster_centers_and_degenerate_unitary_invariants(monkeypatch):
    system = owned_fixture(complex_sources=True)
    fixed_fixture(monkeypatch, system)
    reference = api.case_report(3, 0.7)
    phase = complex(0.6, 0.8)
    rotation = np.array([[1, phase], [-phase.conjugate(), 1]], dtype=complex) / math.sqrt(2)
    rotated = owned_fixture(rotation=rotation, complex_sources=True)
    fixed_fixture(monkeypatch, rotated)
    actual = api.case_report(3, 0.7)
    for name in ("total_weight", "first_moment", "normalized_weight", "normalized_first_moment"):
        same(decode(actual[name]), decode(reference[name]))
    assert abs(decode(reference["total_weight"])[0, 1].imag) > 0.1
    for actual_f, reference_f in zip(actual["frequencies"], reference["frequencies"]):
        same(decode(actual_f["kernel"]), decode(reference_f["kernel"]))
    # An actual near-degenerate gap remains distinct in contractions, never snapped.
    gaps = np.array(system["gaps"], copy=True)
    gaps[2] += 2.0 ** -20
    changed = owned_fixture(gaps=gaps, complex_sources=True)
    changed = dict(changed, resolution=1e-3)
    fixed_fixture(monkeypatch, changed)
    report = api.case_report(3, 0.7)
    assert record(report["gaps"]["ratio"]) == gaps[2] / gaps[1]
    assert report["gaps"]["excited_splitting_status"] == "unresolved_excited_splitting"
    connected = changed["partitions"]["symmetric"]["connected"]
    expected_M = connected.conj().T @ changed["H"] @ connected
    same(decode(report["first_moment"]), expected_M)
    oracle = {"H": changed["H"], "A": changed["H"], "connected": connected}
    for eta, frequency in zip(ETAS, report["frequencies"]):
        same(decode(frequency["kernel"]), resolvent_kernel(oracle, 1j * eta * gaps[1]))


def test_number_current_operators_are_canonical_validated_and_degree_one(monkeypatch):
    original_current = current.bond_currents
    original_validate = energy._validate_model
    current_calls, validated = [], []
    def validate(model):
        value = original_validate(model)
        validated.append(id(model))
        return value
    def currents(model, phases=None):
        assert type(model) is occupation_module.FixedNumberModel
        assert id(model) in validated
        assert phases is None
        expected = occupation_oracle(model.L, model.g, model.C)
        assert model.basis == expected["basis"]
        same(model.H, expected["H"])
        result = original_current(model, phases)
        same(np.asarray(result), np.asarray(expected["currents"]))
        current_calls.append((model.L, model.N, model.C, model.g))
        return result
    monkeypatch.setattr(energy, "_validate_model", validate)
    monkeypatch.setattr(current, "bond_currents", currents)
    report = api.scaling_report(3, 0.7)
    assert current_calls == [(3, 3, s, 0.7 * s) for s in SCALES]
    for comparison in report["comparisons"]:
        for item in comparator_leaves(comparison["number_currents"]):
            comparator(item)
    # No current response observable was added to the frozen six slots.
    for slot in report["cases"]:
        assert {"gap_ratio", "density_moment", "energy_moment", "joint_ratios"} <= set(slot["report"]["observables"])
        assert len(slot["report"]["observables"]["joint_ratios"]) == 3


def test_single_imaginary_current_entry_mismatch_not_hidden_by_large_channels(monkeypatch):
    original = current.bond_currents
    def changed(model, phases=None):
        result = [np.array(a, copy=True) for a in original(model, phases)]
        if model.C == 2.0:
            result[-1][0, 0] += 1j * 1e-6
        return result
    monkeypatch.setattr(current, "bond_currents", changed)
    report = api.scaling_report(3, 0.7)
    result = report["comparisons"][2]
    assert result["status"] == "mismatch"
    assert any(item["status"] == "mismatch" for item in comparator_leaves(result["number_currents"]))
    for name in ("gaps", "total_weight", "first_moment", "kernels", "sources", "observables"):
        assert all(item["status"] == "consistent" for item in comparator_leaves(result[name]))


@pytest.mark.parametrize("kernel", (
    [[2., 3.], [5., 7.]],
    [[1 + 2j, 3 - 4j], [-5 + 6j, 7 + 8j]],
    [[1., 0.], [3., 2.]],
    [[1., -2.], [3., 2.]],
))
def test_ratio_exact_complex_products_no_psd_clipping(kernel):
    kernel = np.asarray(kernel)
    item = api._joint_ratio(kernel, np.zeros((2, 2)))
    assert record(item, complex_value=True) == exact_ratio(kernel)
    assert item["proxy"] == 0.0
    assert item["witness_resolved"] == bool(kernel[0, 1] != 0 and kernel[1, 0] != 0)


def test_product_gate_not_entry_relative_floor_and_proxy_diagnostic_only():
    # Product margin one passes even with a large relative diagonal proxy.
    K = np.array([[1., 0.5], [0.5, 0.125]])
    U = np.array([[0., 0.], [0., 0.0625]])
    value = api._joint_ratio(K, U)
    assert record(value, complex_value=True) == 2
    assert value["proxy"] == 2
    # Huge mixed uncertainty affects UQ, not denominator availability.
    U[0, 1] = U[1, 0] = 1e6
    assert record(api._joint_ratio(K, U), complex_value=True) == 2
    # The margin is exactly one, no eightfold/precision gate.
    U = np.array([[0., 0.], [0., np.nextafter(0.125, 0.0)]])
    assert record(api._joint_ratio(K, U), complex_value=True) == 2
    U[1, 1] = 0.125
    record(api._joint_ratio(K, U), "unresolved_denominator", complex_value=True)
    U[1, 1] = np.nextafter(0.125, math.inf)
    record(api._joint_ratio(K, U), "unresolved_denominator", complex_value=True)
    record(api._joint_ratio([[0., 1.], [1., 1.]], np.zeros((2, 2))), "zero_denominator", complex_value=True)


def test_ratio_uncertainty_formula_and_both_individual_witnesses():
    K = np.array([[4., 2.], [3., 5.]])
    U = np.array([[0.1, 0.2], [0.3, 0.4]])
    item = api._joint_ratio(K, U)
    Q = record(item, complex_value=True)
    ED = abs(K[0, 0]) * U[1, 1] + abs(K[1, 1]) * U[0, 0] + U[0, 0] * U[1, 1]
    EN = abs(K[0, 1]) * U[1, 0] + abs(K[1, 0]) * U[0, 1] + U[0, 1] * U[1, 0]
    same(item["proxy"], (EN + abs(Q) * ED) / (abs(K[0, 0] * K[1, 1]) - ED))
    assert item["witness_resolved"] is True
    for a, b in ((0, 1), (1, 0)):
        proxy = U.copy()
        proxy[a, b] = abs(K[a, b])
        tested = api._joint_ratio(K, proxy)
        assert tested["status"] == AVAILABLE and tested["witness_resolved"] is False
        proxy[a, b] = np.nextafter(abs(K[a, b]), 0.0)
        assert api._joint_ratio(K, proxy)["witness_resolved"] is True


@pytest.mark.parametrize("exponent", (-600, 600))
def test_ratio_never_materializes_underflowing_or_overflowing_products(exponent):
    t = math.ldexp(1.0, exponent)
    kernel = np.array([[2 * t, t], [3 * t, 4 * t]])
    item = api._joint_ratio(kernel, np.zeros((2, 2)))
    assert record(item, complex_value=True) == 0.375
    assert item["proxy"] == 0.0
    # Product AND product error share this scale: denominator gate is scale safe.
    U = np.array([[0., 0.], [0., t]])
    assert record(api._joint_ratio(kernel, U), complex_value=True) == 0.375
    U[1, 1] = 4 * t
    record(api._joint_ratio(kernel, U), "unresolved_denominator", complex_value=True)


def test_preserve_representable_subnormal_and_tiny_imaginary_component():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    kernel = np.array([[1., complex(1., tiny)], [1., 1.]])
    value = record(api._joint_ratio(kernel, np.zeros((2, 2))), complex_value=True)
    assert value.real == 1.0 and value.imag == tiny
    all_tiny = np.full((2, 2), tiny)
    assert record(api._joint_ratio(all_tiny, np.zeros((2, 2))), complex_value=True) == 1.0
    moment = api._normalized_moment(1., tiny, 1., 1., 0.)
    assert record(moment) == tiny


@pytest.mark.parametrize("kernel", (
    [[1., 2.0 ** -600], [2.0 ** -600, 1.]],
    [[1., 2.0 ** 600], [2.0 ** 600, 1.]],
    [[1., complex(1., 2.0 ** -600)], [2.0 ** -600, 1.]],
))
def test_nonrepresentable_nonzero_final_components_are_not_erased(kernel):
    record(api._joint_ratio(kernel, np.zeros((2, 2))), "numerical_unavailable", complex_value=True)


def test_proxy_overflow_invalidates_ratio_instead_of_clipping():
    record(api._joint_ratio(np.ones((2, 2)), [[0., 1e308], [1e308, 0.]]),
           "numerical_unavailable", complex_value=True)


@pytest.mark.parametrize("weight,moment,gap,expected", (
    (2.0 ** -600, 2.0 ** -600, 2., 0.5),
    (2.0 ** 600, 2.0 ** 600, 2.0 ** 600, 2.0 ** -600),
    (2.0 ** -600, 2.0 ** -600, 2.0 ** -600, 2.0 ** 600),
    (2., 0., 3., 0.),
))
def test_normalized_moment_exact_products_and_no_numerator_gate(weight, moment, gap, expected):
    assert record(api._normalized_moment(weight, moment, gap, 1., 0.)) == expected


@pytest.mark.parametrize("weight,moment,gap", (
    (2.0 ** 600, 2.0 ** -600, 2.0 ** 600),
    (2.0 ** -600, 2.0 ** 600, 2.0 ** -600),
))
def test_normalized_moment_nonrepresentable_final_is_unavailable(weight, moment, gap):
    record(api._normalized_moment(weight, moment, gap, 1., 0.), "numerical_unavailable")


def test_structural_zero_not_inferred_from_numerical_smallness():
    record(api._normalized_moment(0., 0., 1., 0., 0., structural_zero=True), "zero_source")
    record(api._normalized_moment(0., 0., 1., 0., 0.), "unresolved_source")
    record(api._normalized_moment(0., 1., 1., 1., 0.), "unresolved_source")
    record(api._normalized_moment(1., 1., 1., 1., 1.), "unresolved_source")
    assert record(api._normalized_moment(1., 0., 1., np.nextafter(1., math.inf), 1.)) == 0.0


def test_reference_only_component_tolerance_boundaries_and_failure_counts():
    comparator(api._compare_dimensionless(TOL, 0.0))
    comparator(api._compare_dimensionless(np.nextafter(TOL, math.inf), 0.0), "mismatch")
    comparator(api._compare_dimensionless(1j * TOL, 0j))
    comparator(api._compare_dimensionless(1j * np.nextafter(TOL, math.inf), 0j), "mismatch")
    # Reference zero defeats max(observed,reference) and combined-complex norms.
    just_outside = TOL + TOL * TOL / 2
    comparator(api._compare_dimensionless(just_outside, 0.), "mismatch")
    comparator(api._compare_dimensionless(complex(TOL, TOL), 0j))
    observed = np.array([[1e150, 3 * TOL], [3j * TOL, complex(3 * TOL, 3 * TOL)]])
    reference = np.array([[1e150, 0.], [0., 0.]])
    result = api._compare_dimensionless(observed, reference)
    comparator(result, "mismatch")
    assert result["failed_components"] == 4
    assert result["max_real_difference"] == 3 * TOL
    assert result["max_imag_difference"] == 3 * TOL
    # Nonzero reference boundaries are independently computed on supplied floats.
    for reference_value in (1., -3., 2.0 ** 30):
        threshold = Fraction(TOL) + Fraction(TOL) * abs(Fraction(reference_value))
        observation = float(Fraction(reference_value) + threshold)
        for candidate in (np.nextafter(observation, -math.inf), observation,
                          np.nextafter(observation, math.inf)):
            expected = "consistent" if abs(Fraction(float(candidate)) - Fraction(reference_value)) <= threshold else "mismatch"
            comparator(api._compare_dimensionless(candidate, reference_value), expected)


def test_comparison_exact_differences_and_nonrepresentable_summary():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    item = api._compare_dimensionless(complex(1., tiny), 1.)
    comparator(item)
    assert item["max_imag_difference"] == tiny
    # The mismatch is known but its required finite raw difference cannot fit.
    # No infinite/NaN JSON diagnostic or silently clipped difference is allowed.
    try:
        item = api._compare_dimensionless(1e308, -1e308)
    except api.NumericalUnavailable:
        return
    comparator(item, "inconclusive")
    native(item)


class Convertible:
    def __float__(self):
        raise AssertionError("custom numeric conversion must not be invoked")

    def __complex__(self):
        raise AssertionError("custom numeric conversion must not be invoked")


@pytest.mark.parametrize("bad", (True, np.bool_(False), "1", Convertible(), None, float("inf"), float("nan")))
def test_all_fixture_helpers_reject_nonnumeric_nonfinite_inputs(bad):
    for call in (
        lambda: api._joint_ratio([[bad, 1], [1, 1]], np.zeros((2, 2))),
        lambda: api._joint_ratio(np.ones((2, 2)), [[bad, 0], [0, 0]]),
        lambda: api._normalized_moment(bad, 1., 1., 1., 0.),
        lambda: api._gap_ratio([0., 1., bad], 0.),
        lambda: api._compare_dimensionless(bad, 1.),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("field", range(5))
@pytest.mark.parametrize("bad", (-1., 1 + 0j, True, "1", Convertible(), float("inf"), float("nan")))
def test_normalized_moment_strict_nonnegative_real_fields(field, bad):
    values = [1., 1., 1., 1., 0.]
    values[field] = bad
    with pytest.raises(ValueError):
        api._normalized_moment(*values)


@pytest.mark.parametrize("flag", (0, 1, np.bool_(True), "yes", None))
def test_structural_zero_requires_builtin_bool(flag):
    with pytest.raises(ValueError):
        api._normalized_moment(1., 1., 1., 1., 0., structural_zero=flag)


@pytest.mark.parametrize("kernel,proxy", (
    (np.ones(4), np.zeros((2, 2))),
    (np.ones((2, 3)), np.zeros((2, 2))),
    (np.ones((2, 2)), np.zeros(4)),
    (np.ones((2, 2)), [[-1., 0.], [0., 0.]]),
    (np.ones((2, 2)), np.zeros((2, 2), dtype=complex)),
    ([[True, 1], [1, 1]], np.zeros((2, 2))),
    (np.ones((2, 2)), [[False, 0.], [0., 0.]]),
))
def test_ratio_shape_real_proxy_and_nested_bool_validation(kernel, proxy):
    with pytest.raises(ValueError):
        api._joint_ratio(kernel, proxy)


@pytest.mark.parametrize("gaps,resolution", (
    ([0., 1.], 0.), ([0., 2., 1.], 0.), ([1., 2., 3.], 0.),
    ([0., -1., 2.], 0.), ([0., 1., 2.], -1.),
    ([0., 1., 2.], 0j), ([0., 1., 2.], True),
    ([0j, 1., 2.], 0.), ([False, 1., 2.], 0.),
    ([[0., 1., 2.]], 0.), ([0.] + [1.] * 512, 0.),
))
def test_gap_validation(gaps, resolution):
    with pytest.raises(ValueError):
        api._gap_ratio(gaps, resolution)


@pytest.mark.parametrize("observed,reference", (
    ([1., 2.], [1.]), ([1.], 1.),
    ([True, 1.], [0., 1.]), ([1., 0.], [False, 1.]),
    (np.ones((513, 512)), np.ones((513, 512))),
    (np.array([1.], dtype=object), np.array([1.])),
    ([[1.], [1., 2.]], [[1.], [1., 2.]]),
))
def test_comparator_bounded_equal_shapes_and_validation(observed, reference):
    with pytest.raises(ValueError):
        api._compare_dimensionless(observed, reference)


def test_gap_unavailable_propagates_without_discarding_raw_weights(monkeypatch):
    system = dict(owned_fixture(), resolution=2.0)
    fixed_fixture(monkeypatch, system)
    report = api.case_report(3, 0.7)
    assert report["status"] == AVAILABLE
    record(report["gaps"]["ratio"], "unresolved_gap")
    for name in ("rho", "h"):
        record(report["sources"][name]["normalized_moment"], "unresolved_gap")
    assert np.any(decode(report["total_weight"]))
    assert np.any(decode(report["first_moment"]))
    for frequency in report["frequencies"]:
        assert frequency["ratio"]["status"] != AVAILABLE


def test_unresolved_source_is_not_structural_zero_and_raw_data_survives(monkeypatch):
    fixed_fixture(monkeypatch, owned_fixture(unresolved=True))
    report = api.case_report(3, 0.7)
    assert report["status"] == AVAILABLE
    assert report["sources"]["rho"]["status"] == "unresolved_source"
    record(report["sources"]["rho"]["normalized_moment"], "unresolved_source")
    assert decode(report["total_weight"])[0, 0].real > 0
    assert decode(report["first_moment"])[0, 0].real > 0
    for frequency in report["frequencies"]:
        assert frequency["kernel"] is not None
        assert frequency["entry_proxy"] is not None
        record(frequency["ratio"], "unresolved_source", complex_value=True)


def check_heldout(report):
    assert {"reference", "readout_gains", "source_gains", "template", "positive", "negative",
            "development", "heldout", "detection_status", "scope"} <= set(report)
    assert report["readout_gains"] == [2, -3]
    assert report["source_gains"] == [5, 7]
    reference = report["reference"]
    assert (reference["L"], reference["N"], reference["C"], reference["g"], reference["m"]) == (3, 3, 1., 0.7, 1)
    scope_flags(report["scope"])
    assert report["template"] == reference["frequencies"]
    for branch in ("positive", "negative"):
        assert [item["eta"] for item in report[branch]] == list(ETAS)
        assert len(report["development"][branch]) == 2
    gains = np.outer([2., -3.], [5., 7.])
    for i, template in enumerate(report["template"]):
        expected = gains * decode(template["normalized_kernel"])
        positive = report["positive"][i]
        negative = report["negative"][i]
        np.testing.assert_array_equal(decode(positive["kernel"]), expected)
        expected_negative = expected.copy()
        if i == 2:
            expected_negative[0, 1] *= -1
        np.testing.assert_array_equal(decode(negative["kernel"]), expected_negative)
        np.testing.assert_array_equal(positive["entry_proxy"], abs(gains) * np.asarray(template["entry_proxy"]))
        np.testing.assert_array_equal(negative["entry_proxy"], positive["entry_proxy"])
        same(record(positive["ratio"], complex_value=True), record(template["ratio"], complex_value=True))
        same(record(negative["ratio"], complex_value=True),
             (-1 if i == 2 else 1) * record(template["ratio"], complex_value=True))
    for branch in ("positive", "negative"):
        for item in report["development"][branch]:
            comparator(item)
    comparator(report["heldout"]["positive"])
    comparator(report["heldout"]["negative"], "mismatch")
    assert report["detection_status"] == "mismatch_detected"
    native(report)


def test_fixed_one_entry_mutation_not_refitting_or_two_entry_reversal(demonstration):
    check_heldout(demonstration["heldout"])


def test_owned_system_and_reports_are_not_mutated_or_shared(monkeypatch):
    system, calls = fixed_fixture(monkeypatch)
    before = copy.deepcopy(system)
    for partition in system["partitions"].values():
        with pytest.raises(ValueError):
            partition["sources"].setflags(write=True)
    first = api.case_report(3, 0.7)
    first["observables"]["joint_ratios"][0]["value"]["real"] = 123456.
    assert first["frequencies"][0]["ratio"]["value"]["real"] != 123456.
    first["sources"]["rho"]["normalized_moment"]["value"] = 98765.
    assert first["observables"]["density_moment"]["value"] != 98765.
    second = api.case_report(3, 0.7)
    assert second["observables"]["joint_ratios"][0]["value"]["real"] != 123456.
    heldout = api.heldout_report()
    heldout["negative"][0]["kernel"]["real"][0][0] = 777.
    assert heldout["positive"][0]["kernel"]["real"][0][0] != 777.
    assert heldout["template"][0]["normalized_kernel"]["real"][0][0] != 777.
    heldout["template"][0]["ratio"]["value"]["real"] = 555.
    assert heldout["reference"]["frequencies"][0]["ratio"]["value"]["real"] != 555.
    assert len(calls) == 3
    for name in ("H", "energies", "vectors", "ground", "gaps"):
        np.testing.assert_array_equal(system[name], before[name])
    for name in ("sources", "connected", "transitions", "grams"):
        np.testing.assert_array_equal(system["partitions"]["symmetric"][name], before["partitions"]["symmetric"][name])
    native(second)
    json.dumps(heldout, allow_nan=False)


def fixture_kernels(system):
    connected = system["partitions"]["symmetric"]["connected"]
    oracle = {"H": system["H"], "A": system["H"], "connected": connected}
    return [resolvent_kernel(oracle, 1j * eta * system["gaps"][1]) for eta in ETAS]


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("mode", ("mismatch", "unavailable"))
def test_positive_and_development_prerequisites_block_false_rejection(monkeypatch, index, mode):
    system, _ = fixed_fixture(monkeypatch)
    expected = np.outer([2., -3.], [5., 7.]) * fixture_kernels(system)[index]
    original = api._joint_ratio
    injections = []
    def altered(kernel, entry_proxy):
        item = original(kernel, entry_proxy)
        if np.allclose(kernel, expected, atol=1e-13, rtol=1e-13):
            injections.append(True)
            item = copy.deepcopy(item)
            if mode == "mismatch":
                item["value"]["real"] += 1.0
            else:
                item.update(status="numerical_unavailable", value=None, proxy=None,
                            witness_resolved=False, reason="injected unavailable positive control")
        return item
    monkeypatch.setattr(api, "_joint_ratio", altered)
    report = api.heldout_report()
    assert injections
    assert report["detection_status"] == "inconclusive"
    assert type(report["detection_reason"]) is str and report["detection_reason"]
    target = report["heldout"]["positive"] if index == 2 else report["development"]["positive"][index]
    comparator(target, "mismatch" if mode == "mismatch" else "inconclusive")
    if index < 2:
        comparator(report["development"]["negative"][index], "mismatch" if mode == "mismatch" else "inconclusive")


@pytest.mark.parametrize("mode,expected", (
    ("unavailable", "inconclusive"),
    ("witness", "unresolved_mismatch_witness"),
    ("same", "mismatch_not_detected"),
))
def test_heldout_negative_status_hierarchy(monkeypatch, mode, expected):
    system, _ = fixed_fixture(monkeypatch)
    template = fixture_kernels(system)[2]
    negative = np.outer([2., -3.], [5., 7.]) * template
    negative[0, 1] *= -1
    original = api._joint_ratio
    def altered(kernel, entry_proxy):
        item = original(kernel, entry_proxy)
        if np.allclose(kernel, negative, atol=1e-13, rtol=1e-13):
            item = copy.deepcopy(item)
            if mode == "unavailable":
                item.update(status="numerical_unavailable", value=None, proxy=None,
                            witness_resolved=False, reason="injected heldout failure")
            elif mode == "witness":
                item["witness_resolved"] = False
            else:
                q = exact_ratio(template)
                item["value"] = {"real": q.real, "imag": q.imag}
        return item
    monkeypatch.setattr(api, "_joint_ratio", altered)
    report = api.heldout_report()
    assert report["detection_status"] == expected
    for branch in ("positive", "negative"):
        for item in report["development"][branch]:
            comparator(item)
    comparator(report["heldout"]["positive"])


def test_pointwise_signed_frequency_dependent_gain_cancellation_not_raw_reciprocity():
    kernels = [np.array([[3 + eta * 1j, 2 - 0.1j], [4 + 0.2j, 5 - eta * 1j]]) for eta in ETAS]
    for i, K in enumerate(kernels):
        row = np.array([2. + i, -3. - i])
        column = np.array([5. - i / 2, 7. + i])
        observed = row[:, None] * K * column[None, :]
        same(record(api._joint_ratio(observed, np.zeros((2, 2))), complex_value=True), exact_ratio(K))
        assert observed[0, 1] != observed[1, 0]
        both = observed.copy()
        both[0, 1] *= -1
        both[1, 0] *= -1
        same(record(api._joint_ratio(both, np.zeros((2, 2))), complex_value=True), exact_ratio(K))
        background = observed + np.eye(2)
        assert abs(exact_ratio(background) - exact_ratio(K)) > TOL


def test_unavailable_ratio_keeps_kernel_and_independent_comparisons(monkeypatch):
    original = api._joint_ratio
    def unavailable(kernel, entry_proxy):
        item = original(kernel, entry_proxy)
        item.update(status="numerical_unavailable", value=None, proxy=None,
                    witness_resolved=False, reason="injected ratio-only arithmetic failure")
        return item
    monkeypatch.setattr(api, "_joint_ratio", unavailable)
    report = api.scaling_report(3, 0.0)
    for slot in report["cases"]:
        case = slot["report"]
        for frequency in case["frequencies"]:
            assert frequency["kernel"] is not None and frequency["entry_proxy"] is not None
            record(frequency["ratio"], "numerical_unavailable", complex_value=True)
        for item in case["free_control"]["comparisons"][:3]:
            comparator(item)
        for item in case["free_control"]["comparisons"][3:]:
            comparator(item, "inconclusive")
    for comparison in report["comparisons"]:
        assert comparison["status"] == "inconclusive"
        for name in ("gaps", "total_weight", "first_moment", "kernels", "sources", "number_currents"):
            for item in comparator_leaves(comparison[name]):
                comparator(item)
    heldout = api.heldout_report()
    assert heldout["detection_status"] == "inconclusive"
    for branch in ("positive", "negative"):
        comparator(heldout["heldout"][branch], "inconclusive")


def test_scalar_arithmetic_failure_only_invalidates_dependent_moment(monkeypatch):
    original = api._normalized_moment
    calls = []
    def failed(weight, moment, gap, amplitude, amplitude_proxy, structural_zero=False):
        calls.append(True)
        if len(calls) == 1:
            return {"status": "numerical_unavailable", "value": None, "reason": "injected scalar output failure"}
        return original(weight, moment, gap, amplitude, amplitude_proxy, structural_zero)
    monkeypatch.setattr(api, "_normalized_moment", failed)
    fixed_fixture(monkeypatch)
    report = api.case_report(3, 0.7)
    assert report["status"] == AVAILABLE
    record(report["sources"]["rho"]["normalized_moment"], "numerical_unavailable")
    record(report["sources"]["h"]["normalized_moment"])
    assert report["total_weight"] is not None and report["first_moment"] is not None
    for frequency in report["frequencies"]:
        record(frequency["ratio"], complex_value=True)


@pytest.mark.parametrize("L,g,C", (
    (True, 0.7, 1.), (3., 0.7, 1.), (2, 0.7, 1.), (6, 0.7, 1.),
    (3, True, 1.), (3, "0.7", 1.), (3, 0.7 + 0j, 1.), (3, -1., 1.),
    (3, float("nan"), 1.), (3, float("inf"), 1.), (3, 40.0001, 1.),
    (3, 0.7, False), (3, 0.7, 0.49), (3, 0.7, 2.01),
    (3, 0.7, "1"), (3, 0.7, 1 + 0j), (3, Convertible(), 1.),
    (3, 10 ** 1000, 1.),
))
def test_public_validation_precedes_any_joint_allocation(monkeypatch, L, g, C):
    def forbidden(*args, **kwargs):
        raise AssertionError("joint allocation before public input validation")
    patch_joint(monkeypatch, forbidden)
    with pytest.raises(ValueError):
        api.case_report(L, g, C)


@pytest.mark.parametrize("L,q", ((True, 0.7), (3., 0.7), (6, 0.7), (3, -1.),
                                  (3, 41.), (3, True), (3, "1"), (3, 1 + 0j)))
def test_scaling_validation_precedes_any_joint_allocation(monkeypatch, L, q):
    def forbidden(*args, **kwargs):
        raise AssertionError("construction before scaling validation")
    patch_joint(monkeypatch, forbidden)
    with pytest.raises(ValueError):
        api.scaling_report(L, q)


def test_module_own_cap_and_unsnapped_tiny_interaction_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense allocation before cap/tiny interaction guard")
    patch_joint(monkeypatch, forbidden)
    with pytest.raises(api.NumericalUnavailable):
        api.case_report(3, 2.0 ** -41)
    monkeypatch.setattr(api, "MAX_DIMENSION", 9)
    with pytest.raises(ValueError):
        api.case_report(3, 0.7)
    with pytest.raises(ValueError):
        api.scaling_report(3, 0.7)


def test_one_joint_construction_per_case_and_exact_boundary_not_snapped(monkeypatch):
    _, calls = fixed_fixture(monkeypatch)
    api.case_report(3, 2.0 ** -40)
    assert calls == [(3, 2.0 ** -40, 1., 1)]


def test_construction_failure_public_raise_fixed_wrappers_preserve_slots(monkeypatch):
    calls = []
    def failed(L, g, C=1.0, m=1):
        calls.append((L, g, C, m))
        raise api.NumericalUnavailable("injected failed eigensystem")
    patch_joint(monkeypatch, failed)
    with pytest.raises(api.NumericalUnavailable):
        api.case_report(3, 0.7)
    scaling = api.scaling_report(3, 0.7)
    assert [slot["scale"] for slot in scaling["cases"]] == list(SCALES)
    for slot in scaling["cases"]:
        assert slot["status"] == "numerical_unavailable"
        assert slot["report"] is None and slot["reason"]
    for comparison in scaling["comparisons"]:
        assert comparison["status"] == "inconclusive"
        for name in ("gaps", "total_weight", "first_moment", "kernels", "sources", "number_currents", "observables"):
            for item in comparator_leaves(comparison[name]):
                comparator(item, "inconclusive")
    heldout = api.heldout_report()
    assert heldout["reference"]["status"] == "numerical_unavailable"
    assert heldout["reference"]["reason"]
    assert heldout["detection_status"] == "inconclusive"
    calls.clear()
    demonstration = api.demonstration_report()
    assert calls == [(L, q * s, s, 1) for L, q in GRID for s in SCALES]
    check_demo(demonstration, available=False)


def test_failed_scale_does_not_prevent_independent_surviving_comparisons(monkeypatch):
    original = joint.joint_system
    def failed(L, g, C=1.0, m=1):
        if C == 0.5:
            raise api.NumericalUnavailable("injected one scale failure")
        return original(L, g, C, m)
    patch_joint(monkeypatch, failed)
    report = api.scaling_report(3, 0.7)
    assert report["cases"][0]["report"] is None
    assert [item["status"] for item in report["comparisons"]] == ["inconclusive", "consistent", "consistent"]


def test_known_mismatch_takes_precedence_over_independent_inconclusive(monkeypatch):
    original_ratio = api._joint_ratio
    original_current = current.bond_currents
    def unavailable(kernel, entry_proxy):
        item = original_ratio(kernel, entry_proxy)
        item.update(status="numerical_unavailable", value=None, proxy=None,
                    witness_resolved=False, reason="injected ratio failure")
        return item
    def changed(model, phases=None):
        result = [np.array(a, copy=True) for a in original_current(model, phases)]
        if model.C == 2.:
            result[0][0, 0] += 1j
        return result
    monkeypatch.setattr(api, "_joint_ratio", unavailable)
    monkeypatch.setattr(current, "bond_currents", changed)
    report = api.scaling_report(3, 0.7)
    assert report["comparisons"][2]["status"] == "mismatch"
    assert any(item["status"] == "inconclusive" for item in comparator_leaves(report["comparisons"][2]["observables"]))


def test_programmer_errors_are_not_swallowed_by_fixed_wrappers(monkeypatch):
    def bug(*args, **kwargs):
        raise ValueError("injected programmer validation error")
    patch_joint(monkeypatch, bug)
    for operation in (lambda: api.scaling_report(3, 0.7), api.heldout_report, api.demonstration_report):
        with pytest.raises(ValueError, match="programmer"):
            operation()


def check_demo(report, available=True):
    assert {"module", "scaling_cases", "heldout", "measurement_requirements", "limitations", "scope"} <= set(report)
    assert [(r["L"], r["g_over_C"]) for r in report["scaling_cases"]] == list(GRID)
    assert sum(len(r["cases"]) for r in report["scaling_cases"]) == 27
    for scaling in report["scaling_cases"]:
        assert [slot["scale"] for slot in scaling["cases"]] == list(SCALES)
        assert [slot["status"] for slot in scaling["cases"]] == [AVAILABLE if available else "numerical_unavailable"] * 3
    scope_flags(report["scope"])
    assert report["measurement_requirements"] and report["limitations"]
    text = json.dumps(report["measurement_requirements"]).lower()
    for alternatives in (("prepar",), ("spectral",), ("multiplic",), ("probe",),
                         ("gain",), ("background",), ("crosstalk", "cross-talk", "channel mix"),
                         ("clock",), ("length",), ("symmetric",), ("imaginary",), ("uncertaint",)):
        assert any(word in text for word in alternatives), alternatives
    native(report)
    json.dumps(report, allow_nan=False)


def test_demonstration_slots_and_explicit_missing_empirical_requirements(demonstration):
    check_demo(demonstration)


def test_python38_grammar_for_only_new_module_demo_and_test():
    for path in (ROOT / "bpr/substrate_prediction_contract.py",
                 ROOT / "scripts/demo_substrate_prediction_contract.py", Path(__file__)):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))


def test_isolated_stdout_only_text_and_strict_json_demos(tmp_path):
    script = ROOT / "scripts/demo_substrate_prediction_contract.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    for flags in ((), ("--json",)):
        before = set(tmp_path.iterdir())
        run = subprocess.run([sys.executable, "-B", "-W", "error", str(script)] + list(flags),
                             cwd=str(tmp_path), env=environment, capture_output=True,
                             text=True, timeout=900, check=False)
        assert run.returncode == 0, run.stderr
        assert run.stderr == ""
        assert set(tmp_path.iterdir()) == before
        assert run.stdout.strip()
        if flags:
            def reject_constant(value):
                raise AssertionError("non-strict JSON token " + value)
            report = json.loads(run.stdout, parse_constant=reject_constant)
            check_demo(report)
            check_heldout(report["heldout"])
        else:
            assert "empirical_test_unavailable" in run.stdout
