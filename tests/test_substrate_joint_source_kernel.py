"""Frozen independent module5 tests; authored without reading its implementation.

Occupation selection rules, full-space resolvents, reduced static solves and all
25 finite-difference points are independent oracles. No scientific controls were
executed while authoring this file. Tiny scalar fixtures use exact input rational
arithmetic rather than ordinary fixture tolerances. Reports remain diagnostic.
"""
import ast
from fractions import Fraction
import itertools
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_joint_source_kernel as api
from bpr import substrate_fermionization as occupation_module


ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("symmetric", "improved")
FREQUENCIES = (0.5j, 1 + 0.5j, 4 + 1j)
STEPS = (2.0 ** -6, 2.0 ** -7, 2.0 ** -8)
GRID = tuple(itertools.product((3, 4, 5), (0.0, 0.7, 40.0)))
FIXTURES = tuple((L, g, 1.0, 1) for L, g in GRID) + (
    (4, 0.0, 1.0, 2), (3, 1.05, 1.5, 1))
TOL = 2e-10


def same(actual, expected):
    """Fixed PER ENTRY tolerance, never rescaled by the largest channel."""
    np.testing.assert_allclose(actual, expected, atol=TOL, rtol=TOL)


def decode(value):
    assert set(value) == {"shape", "real", "imag"}
    result = np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
    assert list(result.shape) == value["shape"]
    return result


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
    for key in ("numerical_error_certified", "matter_kernel_is_graviton",
                "empirical_validation"):
        assert scope[key] is False


def occupations(L, N):
    if L == 1:
        yield (N,)
    else:
        for n in range(N + 1):
            for rest in occupations(L - 1, N - n):
                yield (n,) + rest


def cosine_weights(L, m):
    """Exact quadrantal values without separately rounded Fourier sources."""
    values = []
    for x in range(L):
        turn = (m * x) % L
        if turn == 0:
            value = 1.0
        elif 2 * turn == L:
            value = -1.0
        elif 4 * turn in (L, 3 * L):
            value = 0.0
        else:
            value = math.cos(2 * math.pi * turn / L)
        values.append(math.sqrt(2.0 / L) * value)
    return np.array(values)


def occupation_oracle(L, g, C=1.0, m=1):
    basis = tuple(occupations(L, L))
    index = {state: row for row, state in enumerate(basis)}
    d = math.comb(2 * L - 1, L)
    assert len(basis) == d
    onsite, bonds = [], []
    for x in range(L):
        y = (x + 1) % L
        transfer = np.zeros((d, d))
        for col, state in enumerate(basis):
            if state[x]:
                destination = list(state)
                destination[x] -= 1
                destination[y] += 1
                transfer[index[tuple(destination)], col] = math.sqrt(
                    state[x] * (state[y] + 1))
        bonds.append(-C * (transfer + transfer.T))
        onsite.append(np.diag([g * n[x] * (n[x] - 1) / 2 for n in basis]))
    H = sum(onsite) + sum(bonds)
    w = cosine_weights(L, m)
    rho = np.diag([sum(w[x] * (n[x] - 1) for x in range(L)) for n in basis])
    sources = {}
    for label in PARTITIONS:
        left, right = (0.5, 0.5) if label == "symmetric" else (0.25, 0.75)
        local = [onsite[x] + left * bonds[x - 1] + right * bonds[x]
                 for x in range(L)]
        same(sum(local), H)
        sources[label] = np.array([rho, sum(w[x] * local[x] for x in range(L))])
    energies, vectors = np.linalg.eigh(H)
    return {"H": H, "basis": basis, "sources": sources,
            "energies": energies, "vectors": vectors, "ground": vectors[:, 0],
            "gap": energies[1] - energies[0], "L": L, "C": C, "g": g, "m": m}


def connected_columns(sources, ground):
    columns = np.column_stack([source @ ground for source in sources])
    return columns - np.outer(ground, ground.conj() @ columns / np.vdot(ground, ground))


def direct_kernel(H, E0, ground, sources, z):
    """Full-space resolvents; neither implementation nor Lehmann helpers."""
    eye = np.eye(len(H))
    A = H - E0 * eye
    connected = connected_columns(sources, ground)
    forward = np.linalg.solve(z * eye - A, connected)
    backward = np.linalg.solve(z * eye + A, connected)
    first = connected.conj().T @ forward
    second = connected.conj().T @ backward
    return first - second.T


def direct_static(H, E0, ground, sources):
    """Independent reduced inverse via H-E0+P0, never chi(0)."""
    connected = connected_columns(sources, ground)
    reduced = H - E0 * np.eye(len(H)) + np.outer(ground, ground.conj())
    solution = np.linalg.solve(reduced, connected)
    return -2 * (connected.conj().T @ solution).real


@pytest.fixture(scope="module", params=FIXTURES, ids=lambda p: "L%d-g%g-C%g-m%d" % p)
def frozen(request):
    L, g, C, m = request.param
    oracle = occupation_oracle(L, g, C, m)
    return oracle, api.joint_system(L, g, C, m), api.joint_report(L, g, C, m)


def test_complete_occupation_hamiltonian_and_common_eigensystem(frozen):
    oracle, system, report = frozen
    required = {"L", "N", "C", "g", "m", "k", "dimension", "basis", "H",
                "energies", "vectors", "ground", "gaps", "resolution", "partitions"}
    assert required <= set(system)
    L, H = oracle["L"], oracle["H"]
    d = len(H)
    assert system["L"] == system["N"] == L
    assert system["dimension"] == math.comb(2 * L - 1, L)
    assert type(system["basis"]) is tuple
    assert all(type(state) is tuple for state in system["basis"])
    assert system["basis"] == oracle["basis"]
    assert any(max(state) == L for state in system["basis"])
    assert set(system["partitions"]) == set(PARTITIONS)
    same(system["H"], H)
    same(system["energies"], oracle["energies"])
    same(H @ system["vectors"], system["vectors"] * system["energies"])
    same(system["vectors"].conj().T @ system["vectors"], np.eye(d))
    np.testing.assert_array_equal(system["ground"], system["vectors"][:, 0])
    same(system["gaps"], system["energies"] - system["energies"][0])
    assert system["gaps"][0] == 0
    assert system["gaps"][1] > system["resolution"] > 0
    same(report["gap"], oracle["gap"])
    native(report)
    json.dumps(report, allow_nan=False)
    scope_flags(report["scope"])
    for key in ("L", "N", "C", "g", "m", "dimension", "k"):
        assert report[key] == system[key]


@pytest.mark.parametrize("partition", PARTITIONS)
def test_both_real_standing_sources_and_connected_grams(frozen, partition):
    oracle, system, report = frozen
    item = system["partitions"][partition]
    d = len(oracle["H"])
    assert {"sources", "connected", "transitions", "grams"} <= set(item)
    sources = oracle["sources"][partition]
    assert item["sources"].shape == (2, d, d)
    same(item["sources"], sources)
    np.testing.assert_array_equal(item["sources"].imag, 0)
    for source in item["sources"]:
        same(source, source.conj().T)
    connected = connected_columns(sources, system["ground"])
    transitions = system["vectors"][:, 1:].conj().T @ connected
    grams = np.array([np.outer(row.conj(), row) for row in transitions])
    same(item["connected"], connected)
    same(item["transitions"], transitions)
    same(item["grams"], grams)
    assert item["connected"].shape == (d, 2)
    assert item["transitions"].shape == (d - 1, 2)
    assert item["grams"].shape == (d - 1, 2, 2)
    same(system["ground"].conj() @ item["connected"], np.zeros(2))
    summary = report["partitions"][partition]
    total = connected.conj().T @ connected
    same(decode(summary["total_gram"]), total)
    assert summary["closure_residual"] >= 0
    expected_residual = np.linalg.norm(item["grams"].sum(axis=0) -
                                       item["connected"].conj().T @ item["connected"])
    same(summary["closure_residual"], expected_residual)


@pytest.mark.parametrize("partition", PARTITIONS)
def test_full_resolvents_reduced_static_physical_scaling_and_real_shear(frozen, partition):
    oracle, _, report = frozen
    H, ground, E0 = oracle["H"], oracle["ground"], oracle["energies"][0]
    sources = oracle["sources"][partition]
    C = oracle["C"]
    D = np.diag([C, 1.0])
    R = np.array([[1.0, 1.0], [0.0, 1.0]])
    equal_sources = np.array([C * sources[0], sources[1]])
    changed = np.einsum("ab,bij->aij", R, equal_sources)
    summary = report["partitions"][partition]
    K = direct_static(H, E0, ground, sources)
    Kf = D @ K @ D
    same(summary["static_hessian_physical"], K)
    same(summary["static_hessian_dimensionless"], Kf)
    W = connected_columns(sources, ground).conj().T @ connected_columns(sources, ground)
    Wf = D @ W @ D
    same(decode(summary["dimensionless_gram"]), Wf)
    same(summary["gram_eigenvalues_dimensionless"], np.linalg.eigvalsh(Wf))
    same(summary["static_eigenvalues_dimensionless"], np.linalg.eigvalsh(Kf))
    assert np.linalg.eigvalsh(Wf).min() >= -TOL
    assert np.linalg.eigvalsh(Kf).max() <= TOL
    if oracle["g"] > 0:
        assert summary["free_control"] is None
        assert "analytic_rank" not in summary
        assert "numerical_rank" not in summary
    basis_change = summary["basis_change"]
    same(basis_change["R"], R)
    U = connected_columns(changed, ground)
    same(decode(basis_change["transformed_gram"]), U.conj().T @ U)
    same(decode(basis_change["transformed_gram"]), R @ Wf @ R.T)
    same(basis_change["transformed_hessian"], direct_static(H, E0, ground, changed))
    same(basis_change["transformed_hessian"], R @ Kf @ R.T)
    assert basis_change["gram_residual"] >= 0
    assert basis_change["hessian_residual"] >= 0
    assert len(summary["responses"]) == len(basis_change["responses"]) == 3
    for z, response, transformed in zip(FREQUENCIES, summary["responses"],
                                         basis_change["responses"]):
        assert response["z"] == transformed["z"] == {"real": z.real, "imag": z.imag}
        kernel = direct_kernel(H, E0, ground, sources, z)
        same(decode(response["kernel_physical"]), kernel)
        same(decode(response["kernel_dimensionless"]), D @ kernel @ D)
        actual = decode(response["kernel_physical"])
        same(response["reciprocity_residual"], np.linalg.norm(actual - actual.T))
        same(decode(transformed["transformed_kernel"]),
             direct_kernel(H, E0, ground, changed, z))
        same(decode(transformed["transformed_kernel"]), R @ D @ kernel @ D @ R.T)
        assert transformed["residual"] >= 0
    # This fails if the C!=1 fixture accidentally reports an E/C Hessian.
    fprime = np.array([0.2, -0.3])
    f = R.T @ fprime
    same(np.einsum("a,aij->ij", fprime, changed),
         np.einsum("a,aij->ij", f, equal_sources))


@pytest.mark.parametrize("partition", PARTITIONS)
def test_projector_group_psd_coverage_and_first_member_width(frozen, partition):
    _, system, report = frozen
    groups = report["partitions"][partition]["groups"]
    grams = system["partitions"][partition]["grams"]
    gaps = system["gaps"]
    resolution = report["resolution"]
    flattened = []
    expected = []
    for n in range(1, len(gaps)):
        if not expected or gaps[n] - gaps[expected[-1][0]] > resolution:
            expected.append([n])
        else:
            expected[-1].append(n)
    assert [group["indices"] for group in groups] == expected
    for group in groups:
        indices = group["indices"]
        flattened.extend(indices)
        same(group["gap_min"], min(gaps[indices]))
        same(group["gap_max"], max(gaps[indices]))
        assert group["gap_max"] - group["gap_min"] <= resolution
        W = decode(group["gram"])
        same(W, grams[np.asarray(indices) - 1].sum(axis=0))
        same(W, W.conj().T)
        assert np.linalg.eigvalsh(W).min() >= -TOL
    assert flattened == list(range(1, len(gaps)))


def test_uniform_controls_are_structural_with_raw_numerical_norms(frozen):
    oracle, system, report = frozen
    ground = system["ground"]
    uniform = report["uniform_controls"]
    assert uniform["number_structural_zero"] is True
    assert uniform["energy_structural_zero"] is True
    scale = math.sqrt(2.0 / oracle["L"])
    for key, source in (("number_connected_norm", scale * oracle["L"] * np.eye(len(ground))),
                        ("energy_connected_norm", scale * oracle["H"])):
        raw = connected_columns([source], ground)
        same(uniform[key], np.linalg.norm(raw))
        assert uniform[key] >= 0


@pytest.mark.parametrize("partition", PARTITIONS)
def test_free_rank_distinction_source_gap_and_pi_dark_operator(frozen, partition):
    oracle, system, report = frozen
    if oracle["g"] != 0:
        return
    L, m, C = oracle["L"], oracle["m"], oracle["C"]
    k = 2 * math.pi * m / L
    at_pi = 2 * m == L
    alpha = 0.0 if at_pi else -C * (1 + math.cos(k))
    beta = 0.0 if at_pi or partition == "symmetric" else C * math.sin(k) / 2
    weight = 2.0 if at_pi else 1.0
    W = weight * np.array([[1.0, alpha], [alpha, alpha * alpha + beta * beta]])
    source_gap = 4 * C * math.sin(k / 2) ** 2
    control = report["partitions"][partition]["free_control"]
    assert control["analytic_rank"] == (1 if partition == "symmetric" or at_pi else 2)
    same(control["expected_gap"], source_gap)
    same(decode(control["expected_gram"]), W)
    same(control["expected_hessian"], -2 * W / source_gap)
    same(decode(report["partitions"][partition]["total_gram"]), W)
    same(report["partitions"][partition]["static_hessian_physical"], -2 * W / source_gap)
    assert control["energy_dark"] is at_pi
    assert control["gram_residual"] >= 0
    assert control["hessian_residual"] >= 0
    sources = oracle["sources"][partition]
    same(control["energy_source_operator_norm"], np.linalg.norm(sources[1], 2))
    same(control["energy_connected_norm"],
         np.linalg.norm(connected_columns(sources, oracle["ground"])[:, 1]))
    for z, response in zip(FREQUENCIES, control["responses"]):
        assert response["z"] == {"real": z.real, "imag": z.imag}
        same(decode(response["expected_kernel"]), 2 * source_gap * W / (z * z - source_gap ** 2))
        assert response["residual"] >= 0
    if partition == "symmetric":
        columns = connected_columns(sources, oracle["ground"])
        same(columns[:, 1], alpha * columns[:, 0])
    elif not at_pi:
        assert np.linalg.det(W) > 0
    if at_pi:
        same(report["gap"], 2 * C)
        same(control["expected_gap"], 4 * C)
        assert control["expected_gap"] > report["gap"]
        if partition == "symmetric":
            np.testing.assert_array_equal(system["partitions"][partition]["sources"][1], 0)
        else:
            assert control["energy_source_operator_norm"] > 0.1


def test_complex_two_level_transpose_and_static_not_zero_frequency(monkeypatch):
    delta = 2.0
    H = np.diag([0.0, delta])
    ground = np.array([1.0, 0.0])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1j], [1j, 0.0]])
    sources = np.array([sx, sy])
    transitions = np.array([[1.0, 1j]])
    grams = api._grams(transitions)
    same(grams, np.array([[[1.0, 1j], [-1j, 1.0]]]))
    assert grams[0, 0, 1] == 1j
    for z in FREQUENCIES:
        result = api._kernel(np.array([delta]), grams, z)
        same(result, direct_kernel(H, 0.0, ground, sources, z))
        same(result[0, 1], 1j / (z - delta) + 1j / (z + delta))
        same(result[1, 0], -result[0, 1])
        assert abs(result[0, 1] - result[1, 0]) > 0.1
    def forbidden_kernel(*args, **kwargs):
        raise AssertionError("static Hessian must not call retarded kernel")
    monkeypatch.setattr(api, "_kernel", forbidden_kernel)
    actual = api._static_hessian(np.array([delta]), grams)
    same(actual, direct_static(H, 0.0, ground, sources))
    assert actual[0, 1] == actual[1, 0] == 0


def test_exact_degenerate_unitary_invariance_and_actual_unequal_gaps():
    transitions = np.array([[1 + 2j, 3 - 1j], [2 - 1j, -1 + 4j],
                            [0.5 + 0.25j, -2j]])
    gaps = np.array([2.0, 2.0, 2.0 + 2.0 ** -20])
    U = np.array([[1.0, 1j], [1j, 1.0]]) / math.sqrt(2.0)
    changed = transitions.copy()
    changed[:2] = U.conj().T @ transitions[:2]
    original = api._grams(transitions)
    rotated = api._grams(changed)
    same(original[:2].sum(axis=0), rotated[:2].sum(axis=0))
    assert not np.allclose(original[0], rotated[0])
    same(original[2], rotated[2])
    for W in (original[:2].sum(axis=0), rotated[:2].sum(axis=0)):
        assert np.linalg.eigvalsh(W).min() >= -TOL
    same(api._static_hessian(gaps, original), api._static_hessian(gaps, rotated))
    for z in FREQUENCIES:
        expected = np.array([[sum(original[n, a, b] / (z - gap) -
                                   original[n, b, a] / (z + gap)
                                   for n, gap in enumerate(gaps))
                              for b in range(2)] for a in range(2)])
        actual = api._kernel(gaps, original, z)
        same(actual, expected)
        same(actual, api._kernel(gaps, rotated, z))
        collapsed = api._kernel(np.array([2.0]), original.sum(axis=0)[None, :, :], z)
        assert np.max(np.abs(actual - collapsed)) > 1e-8


# Exact complex pairs make cancellation oracles independent of floating complex
# multiplication, summation, division and the implementation's rational helpers.
def rational(z):
    z = complex(z)
    return Fraction(z.real), Fraction(z.imag)


def radd(a, b):
    return a[0] + b[0], a[1] + b[1]


def rmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def rdiv(a, b):
    denominator = b[0] ** 2 + b[1] ** 2
    return ((a[0] * b[0] + a[1] * b[1]) / denominator,
            (a[1] * b[0] - a[0] * b[1]) / denominator)


def rational_kernel(gaps, grams, z):
    result = []
    for a in range(2):
        row = []
        for b in range(2):
            value = (Fraction(0), Fraction(0))
            for gap, W in zip(gaps, grams):
                x, y = rational(z)
                gap = Fraction(float(gap))
                forward = rdiv(rational(W[a, b]), (x - gap, y))
                backward = rdiv(rational(W[b, a]), (x + gap, y))
                value = radd(value, (forward[0] - backward[0], forward[1] - backward[1]))
            row.append(value)
        result.append(row)
    return result


def exact_components_or_unavailable(operation, expected):
    try:
        actual = np.asarray(operation())
    except api.NumericalUnavailable:
        return
    assert actual.shape == (len(expected), len(expected[0]))
    for a, row in enumerate(expected):
        for b, components in enumerate(row):
            for value, reference in zip((actual[a, b].real, actual[a, b].imag), components):
                if reference == 0:
                    assert value == 0
                else:
                    converted = float(reference)
                    assert abs(converted) >= np.finfo(float).tiny
                    assert value != 0
                    assert value == pytest.approx(converted, rel=2e-10, abs=0)


def test_grams_do_not_erase_small_component_from_product_cancellation():
    epsilon = 2.0 ** -27
    transitions = np.array([[complex(1 + epsilon, 1), complex(1, 1 - epsilon)]])
    expected = []
    for left in transitions[0]:
        row = []
        a = rational(left)
        for right in transitions[0]:
            row.append(rmul((a[0], -a[1]), rational(right)))
        expected.append(row)
    assert expected[0][1][1] == -Fraction(1, 2 ** 54)
    exact_components_or_unavailable(lambda: api._grams(transitions)[0], expected)


def test_tiny_frequency_component_is_preserved_or_explicitly_unavailable():
    gaps = np.array([2.0])
    grams = np.array([[[1.0, 0.5], [0.5, 0.25]]], dtype=complex)
    z = complex(2.0 ** -500, 0.5)
    expected = rational_kernel(gaps, grams, z)
    assert expected[0][0][0] and expected[0][0][1]
    exact_components_or_unavailable(lambda: api._kernel(gaps, grams, z), expected)


def test_tiny_channel_sum_cancellation_no_threshold_zeroing():
    tiny = 2.0 ** -500
    transitions = np.array([[1.0, 1.0], [1.0, tiny], [1.0, -1.0]])
    grams = np.array([np.outer(row, row) for row in transitions], dtype=complex)
    gaps = np.ones(3)
    expected_static = [[(-2 * sum((Fraction(float(W[a, b].real)) for W in grams), Fraction(0)),
                        Fraction(0)) for b in range(2)] for a in range(2)]
    assert expected_static[0][1][0] == -2 * Fraction(tiny)
    exact_components_or_unavailable(lambda: api._static_hessian(gaps, grams), expected_static)
    z = 1 + 0.5j
    exact_components_or_unavailable(lambda: api._kernel(gaps, grams, z),
                                    rational_kernel(gaps, grams, z))


@pytest.mark.parametrize("power", (-520, 520))
def test_nonrepresentable_gram_component_is_numerical_unavailable(power):
    transitions = np.array([[2.0 ** power, 1.0]])
    with pytest.raises(api.NumericalUnavailable):
        api._grams(transitions)


def test_nonrepresentable_kernel_and_static_components_are_unavailable():
    tiny = np.finfo(float).tiny
    grams = np.array([[[tiny, 0.0], [0.0, 0.0]]])
    with pytest.raises(api.NumericalUnavailable):
        api._static_hessian(np.array([16.0]), grams)
    with pytest.raises(api.NumericalUnavailable):
        api._kernel(np.array([16.0]), grams, 0.5j)


def points_at(h):
    return ((0.0, 0.0), (h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h),
            (h, h), (h, -h), (-h, h), (-h, -h))


def model_bounds(L, g, C):
    weak = 4 * C * math.sin(math.pi / L) ** 2 - g * (L - 1) / 2
    strong = g - 2 * C * L
    delta = max(0.0, weak, strong)
    wmax = math.sqrt(2.0 / L)
    rho = C * L * wmax
    energy = wmax * (g * math.comb(L, 2) + 2 * C * L)
    return delta, {"rho": rho, "h": energy, "plus": rho + energy, "minus": rho + energy}


def envelope(b, h, delta):
    if b == 0:
        return 0.0, "available_conditional"
    if delta <= 0:
        return None, "analytic_gap_unavailable"
    if 2 * b * h >= delta:
        return None, "sufficient_radius_unavailable"
    return 16 * b ** 4 * h ** 2 / (delta ** 3 * (1 - (2 * b * h / delta) ** 2)), "available_conditional"


def expected_bounds(norms, h, gap):
    rho, srho = envelope(norms["rho"], h, gap)
    energy, sh = envelope(norms["h"], h, gap)
    plus, sp = envelope(norms["plus"], h, gap)
    minus, sm = envelope(norms["minus"], h, gap)
    mixed = (plus + minus) / 4 if plus is not None and minus is not None else None
    mixed_status = "available_conditional" if mixed is not None else (sp if plus is None else sm)
    return [[rho, mixed], [mixed, energy]], [[srho, mixed_status], [mixed_status, sh]]


@pytest.fixture(scope="module", params=GRID + ((3, 1.05),), ids=lambda p: "FD-L%d-g%g" % p)
def fd_frozen(request):
    L, g = request.param
    C = 1.5 if g == 1.05 else 1.0
    oracle = occupation_oracle(L, g, C)
    report = api.finite_difference_report(L, g, C)
    # One origin plus 8 independently evaluated points at each of the fixed
    # three steps, per partition. Deliberately no reuse of report energies.
    independent = {}
    for partition in PARTITIONS:
        sources = oracle["sources"][partition]
        X, Y = C * sources[0], sources[1]
        data = {(0.0, 0.0): float(oracle["energies"][0])}
        for h in STEPS:
            for f in points_at(h)[1:]:
                Hf = oracle["H"] + f[0] * X + f[1] * Y
                data[f] = float(np.linalg.eigh(Hf)[0][0])
        assert len(data) == 25
        independent[partition] = data
    return oracle, report, independent


def test_fd_model_bounds_and_known_unavailable_case(fd_frozen):
    oracle, report, _ = fd_frozen
    L, C, g = oracle["L"], oracle["C"], oracle["g"]
    assert report["L"] == report["N"] == L
    assert report["C"] == C and report["g"] == g and report["m"] == 1
    assert report["dimension"] == len(oracle["H"])
    same(report["gap"], oracle["gap"])
    native(report)
    json.dumps(report, allow_nan=False)
    scope_flags(report["scope"])
    delta, norms = model_bounds(L, g, C)
    same(report["model_gap_lower_bound"], delta)
    assert set(report["model_source_norm_bounds"]) == set(norms)
    for key in norms:
        same(report["model_source_norm_bounds"][key], norms[key])
    if L == 5 and g == 0.7:
        assert (5 - math.sqrt(5)) / 2 - 7 / 5 < 0
        assert report["model_gap_lower_bound"] == 0
        for partition in PARTITIONS:
            for step in report["partitions"][partition]["steps"]:
                assert step["model_bounds"]["values"] == [[None, None], [None, None]]
                assert step["model_bounds"]["status"] == [
                    ["analytic_gap_unavailable"] * 2, ["analytic_gap_unavailable"] * 2]
                # Unavailable theorem does not remove measured stencils.
                assert all(value is not None for row in step["stencil"] for value in row)


@pytest.mark.parametrize("partition", PARTITIONS)
def test_fd_all_points_fixed_stencils_bounds_and_energy_scaled_proxies(fd_frozen, partition):
    oracle, report, independent = fd_frozen
    H, C, L, g = oracle["H"], oracle["C"], oracle["L"], oracle["g"]
    sources = oracle["sources"][partition]
    X, Y = C * sources[0], sources[1]
    norms = {"rho": np.linalg.norm(X, 2), "h": np.linalg.norm(Y, 2),
             "plus": np.linalg.norm(X + Y, 2), "minus": np.linalg.norm(X - Y, 2)}
    delta, analytic_norms = model_bounds(L, g, C)
    target = direct_static(H, oracle["energies"][0], oracle["ground"], [X, Y])
    item = report["partitions"][partition]
    contact = C * np.diag([1.0, 0.0])
    same(item["target_hessian"], target)
    same(item["contact_matrix"], contact)
    same(item["contact_shifted_hessian"], target + contact)
    assert [step["step"] for step in item["steps"]] == list(STEPS)
    data = independent[partition]
    for h, step in zip(STEPS, item["steps"]):
        assert len(step["points"]) == 9
        assert [point["f"] for point in step["points"]] == [list(f) for f in points_at(h)]
        proxies, reported_energies = [], []
        for f, point in zip(points_at(h), step["points"]):
            V = f[0] * X + f[1] * Y
            Hf = H + V
            norm = np.linalg.norm(V, 2)
            margin = report["gap"] - 2 * norm
            proxy = 256 * np.finfo(float).eps * len(H) * max(C, np.linalg.norm(Hf, "fro"))
            assert point["evaluation_status"] == "available"
            same(point["energy"], data[f])
            same(point["perturbation_norm"], norm)
            same(point["isolation_margin"], margin)
            assert point["isolation_status"] == ("sufficient_heuristic" if margin > 0
                                                   else "isolation_not_established")
            assert point["energy_proxy"] == pytest.approx(proxy, rel=TOL, abs=0)
            proxies.append(proxy)
            reported_energies.append(point["energy"])
        def stencil(values):
            # Exact sums of supplied floating energies avoid a test-side
            # cancellation error masquerading as an implementation mismatch.
            values = [Fraction(float(value)) for value in values]
            square = Fraction(h) ** 2
            rho = float((values[1] + values[2] - 2 * values[0]) / square)
            energy = float((values[3] + values[4] - 2 * values[0]) / square)
            mixed = float((values[5] - values[6] - values[7] + values[8]) / (4 * square))
            return np.array([[rho, mixed], [mixed, energy]])
        expected = stencil([data[f] for f in points_at(h)])
        actual = np.asarray(step["stencil"])
        # Independent eigensolve subtraction has a separately labeled error
        # budget. This is not a claim that the Hessian discrepancy is small.
        proxy_rho = (proxies[1] + proxies[2] + 2 * proxies[0]) / h ** 2
        proxy_h = (proxies[3] + proxies[4] + 2 * proxies[0]) / h ** 2
        proxy_mix = sum(proxies[5:]) / (4 * h ** 2)
        proxy_matrix = np.array([[proxy_rho, proxy_mix], [proxy_mix, proxy_h]])
        assert np.all(np.abs(actual - expected) <= 2 * proxy_matrix + TOL)
        same(actual, stencil(reported_energies))
        assert step["stencil"][0][1] == step["stencil"][1][0]
        same(step["absolute_error"], np.abs(actual - np.asarray(item["target_hessian"])))
        np.testing.assert_allclose(step["roundoff_proxy"], proxy_matrix, rtol=TOL, atol=0)
        same(step["contact_shifted_stencil"], actual + contact)
        for a in range(2):
            for b in range(2):
                expected_status = ("above_roundoff_proxy" if abs(actual[a, b]) > proxy_matrix[a, b]
                                   else "below_roundoff_proxy")
                assert step["roundoff_status"][a][b] == expected_status
        for key, direction_norms, gap in (("heuristic_bounds", norms, report["gap"]),
                                          ("model_bounds", analytic_norms, delta)):
            bound = step[key]
            same(bound["gap_used"], gap)
            assert set(bound["directional_norms"]) == set(direction_norms)
            for direction in direction_norms:
                same(bound["directional_norms"][direction], direction_norms[direction])
            values, statuses = expected_bounds(direction_norms, h, gap)
            assert bound["status"] == statuses
            for a in range(2):
                for b in range(2):
                    if values[a][b] is None:
                        assert bound["values"][a][b] is None
                    else:
                        assert bound["values"][a][b] == pytest.approx(values[a][b], rel=TOL, abs=0)
                        # Conditional diagnostic plus subtraction proxies, not
                        # a machine certificate and never asserted unavailable.
                        assert step["absolute_error"][a][b] <= values[a][b] + 2 * proxy_matrix[a, b] + TOL


def arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from arrays(item)


def test_frozen_public_signatures_no_arbitrary_system_source_or_cache_inputs():
    assert issubclass(api.NumericalUnavailable, ValueError)
    for name in ("joint_system", "joint_report", "finite_difference_report"):
        parameters = inspect.signature(getattr(api, name)).parameters
        assert list(parameters) == ["L", "g", "C", "m"]
        assert parameters["C"].default == 1.0
        assert parameters["m"].default == 1
        assert parameters["L"].default is inspect.Parameter.empty
        assert parameters["g"].default is inspect.Parameter.empty
        assert all(p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD) for p in parameters.values())
    assert not inspect.signature(api.demonstration_report).parameters


def test_fd_shared_origin_24_nonzero_solves_each_and_no_contact_diagonalizations(monkeypatch):
    oracle = occupation_oracle(3, 0.7)
    original = np.linalg.eigh
    observed = []
    def recording(matrix, *args, **kwargs):
        if np.asarray(matrix).shape == oracle["H"].shape:
            observed.append(np.array(matrix, copy=True))
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(np.linalg, "eigh", recording)
    report = api.finite_difference_report(3, 0.7)
    # The inherited base solver diagonalizes H/||H||F, whereas the finite
    # points diagonalize raw H(f). Identify the base independently, not by
    # call order; redundant origin/contact solves must still fail the count.
    normalized_base = oracle["H"] / np.linalg.norm(oracle["H"], "fro")
    is_base = [np.allclose(matrix, normalized_base, rtol=0, atol=1e-13)
               for matrix in observed]
    assert len(observed) == 49
    assert sum(is_base) == 1
    nonzero = [matrix for matrix, base in zip(observed, is_base) if not base]
    assert len(nonzero) == 48
    expected = []
    for partition in PARTITIONS:
        X, Y = oracle["sources"][partition]
        for h in STEPS:
            for f in points_at(h)[1:]:
                expected.append(oracle["H"] + f[0] * X + f[1] * Y)
    # Multiset matching includes coincident density-axis points in both
    # partitions, without imposing an implementation-specific call order.
    for target in expected:
        index = next((i for i, matrix in enumerate(nonzero)
                      if np.allclose(matrix, target, rtol=0, atol=1e-13)), None)
        assert index is not None
        nonzero.pop(index)
    assert not nonzero
    for partition in PARTITIONS:
        origins = [step["points"][0] for step in report["partitions"][partition]["steps"]]
        assert origins[0] == origins[1] == origins[2]


def test_immutable_bytes_backed_detached_primitives_and_no_public_cache():
    system = api.joint_system(3, 0.7)
    second = api.joint_system(3, 0.7)
    exported = list(arrays(system))
    other = list(arrays(second))
    assert exported and len(exported) == len(other)
    for array, separate in zip(exported, other):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 0
        assert not np.shares_memory(array, separate)
        base = array
        while isinstance(base, np.ndarray) and base.base is not None:
            base = base.base
        if isinstance(base, memoryview):
            assert base.readonly
        else:
            assert isinstance(base, bytes)
    def no_cache(value):
        if isinstance(value, dict):
            assert all("cache" not in str(key).lower() for key in value)
            for child in value.values():
                no_cache(child)
    no_cache(system)
    baseline = api.joint_report(3, 0.7)
    baseline_serialized = json.dumps(baseline, sort_keys=True, allow_nan=False)
    system["H"] = np.zeros_like(system["H"])
    system["partitions"]["symmetric"]["sources"] = np.zeros((2, 10, 10))
    system["resolution"] = 1e100
    system["_transition_cache"] = {"poisoned": True}
    second["partitions"].clear()
    baseline["partitions"]["symmetric"]["responses"][0]["kernel_physical"]["real"][0][0] = 1e100
    assert json.dumps(api.joint_report(3, 0.7), sort_keys=True, allow_nan=False) == baseline_serialized
    fresh = api.joint_system(3, 0.7)
    same(fresh["H"], occupation_oracle(3, 0.7)["H"])
    assert set(fresh["partitions"]) == set(PARTITIONS)


class ConversionTrap:
    def __float__(self):
        raise AssertionError("custom conversion is prohibited")

    def __int__(self):
        raise AssertionError("custom conversion is prohibited")

    def __index__(self):
        raise AssertionError("custom conversion is prohibited")


INVALID = (
    {"L": True}, {"L": np.bool_(False)}, {"L": 3.0}, {"L": "3"},
    {"L": 2}, {"L": 6}, {"L": 10 ** 100}, {"L": ConversionTrap()},
    {"m": True}, {"m": np.bool_(True)}, {"m": 1.0}, {"m": 0},
    {"m": 2}, {"m": "1"}, {"m": ConversionTrap()},
    {"C": True}, {"C": np.bool_(True)}, {"C": 0.49}, {"C": 2.01},
    {"C": float("nan")}, {"C": float("inf")}, {"C": 1 + 0j}, {"C": "1"},
    {"C": ConversionTrap()}, {"C": np.array(1.0)},
    {"g": True}, {"g": np.bool_(True)}, {"g": -0.1}, {"g": 40.01},
    {"g": float("nan")}, {"g": float("inf")}, {"g": "0.7"},
    {"g": 0.7 + 0j}, {"g": ConversionTrap()}, {"g": np.array(0.7)},
)


def block_model_builds(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validation/cap must precede complete-model allocation")
    original = occupation_module.fixed_number_model
    monkeypatch.setattr(occupation_module, "fixed_number_model", forbidden)
    for name, value in list(vars(api).items()):
        if value is original:
            monkeypatch.setattr(api, name, forbidden)


@pytest.mark.parametrize("function", ("joint_system", "joint_report", "finite_difference_report"))
@pytest.mark.parametrize("override", INVALID)
def test_public_invalid_types_and_domain_before_build(monkeypatch, function, override):
    block_model_builds(monkeypatch)
    arguments = {"L": 3, "g": 0.7, "C": 1.0, "m": 1}
    arguments.update(override)
    with pytest.raises(ValueError) as caught:
        getattr(api, function)(**arguments)
    assert not isinstance(caught.value, api.NumericalUnavailable)


@pytest.mark.parametrize("function", ("joint_system", "joint_report", "finite_difference_report"))
def test_cap_and_tiny_interaction_gate_precede_allocation(monkeypatch, function):
    assert api.MAX_DIMENSION == 512
    block_model_builds(monkeypatch)
    with pytest.raises(api.NumericalUnavailable):
        getattr(api, function)(3, 2.0 ** -41)
    monkeypatch.setattr(api, "MAX_DIMENSION", 9)
    with pytest.raises(ValueError):
        getattr(api, function)(3, 0.0)


@pytest.mark.parametrize("C,g", ((0.5, 0.0), (2.0, 80.0), (1.0, 2.0 ** -40)))
def test_accepted_boundaries_numpy_scalar_types_and_nonzero_g_not_snapped(C, g):
    result = api.joint_system(np.int64(3), np.float64(g), np.float64(C), np.int32(1))
    assert result["L"] == result["N"] == 3
    assert result["C"] == C and result["g"] == g
    if g:
        assert result["g"] != 0


@pytest.mark.parametrize("function", ("joint_system", "joint_report", "finite_difference_report"))
@pytest.mark.parametrize("mode", ("exception", "nonfinite", "nonorthogonal", "wrong_eigenpair"))
def test_base_eigensolver_failure_never_fabricates_report(monkeypatch, function, mode):
    original = np.linalg.eigh
    def broken(matrix, *args, **kwargs):
        if mode == "exception":
            raise np.linalg.LinAlgError("independent injected eigensolver failure")
        values, vectors = original(matrix, *args, **kwargs)
        if mode == "nonfinite":
            values[0] = np.nan
        elif mode == "nonorthogonal":
            vectors[:, 1] = vectors[:, 0]
        else:
            values[0] += 0.01
        return values, vectors
    monkeypatch.setattr(np.linalg, "eigh", broken)
    with pytest.raises(api.NumericalUnavailable):
        getattr(api, function)(3, 0.7)


@pytest.mark.parametrize("failed_point", ((STEPS[0], 0.0), (0.0, STEPS[0]),
                                           (STEPS[0], STEPS[0])))
@pytest.mark.parametrize("mode", ("exception", "nonfinite", "nonorthogonal", "wrong_eigenpair"))
def test_point_failure_propagates_only_actual_stencil_dependencies(monkeypatch, failed_point, mode):
    oracle = occupation_oracle(3, 0.7)
    sources = oracle["sources"]["symmetric"]
    target_matrix = oracle["H"] + failed_point[0] * sources[0] + failed_point[1] * sources[1]
    original = np.linalg.eigh
    hits = []
    def broken(matrix, *args, **kwargs):
        matches = np.asarray(matrix).shape == target_matrix.shape and np.allclose(
            matrix, target_matrix, rtol=0, atol=1e-13)
        if not matches:
            return original(matrix, *args, **kwargs)
        hits.append(True)
        if mode == "exception":
            raise np.linalg.LinAlgError("point-specific independent injection")
        values, vectors = original(matrix, *args, **kwargs)
        if mode == "nonfinite":
            values[0] = np.inf
        elif mode == "nonorthogonal":
            vectors[:, 1] = vectors[:, 0]
        else:
            values[0] += 0.01
        return values, vectors
    monkeypatch.setattr(np.linalg, "eigh", broken)
    report = api.finite_difference_report(3, 0.7)
    assert hits, "targeted full perturbed eigh was not called"
    native(report)
    scope_flags(report["scope"])
    assert len(report["partitions"]) == 2
    for partition in PARTITIONS:
        steps = report["partitions"][partition]["steps"]
        assert [step["step"] for step in steps] == list(STEPS)
        for step in steps:
            points = step["points"]
            assert len(points) == 9
            failed_indices = {i for i, point in enumerate(points)
                              if point["evaluation_status"] == "numerical_unavailable"}
            for i in failed_indices:
                point = points[i]
                assert isinstance(point["reason"], str) and point["reason"]
                for key in ("energy", "energy_proxy", "perturbation_norm", "isolation_margin"):
                    assert point[key] is None
            dependencies = (((0, 1, 2), (5, 6, 7, 8)), ((5, 6, 7, 8), (0, 3, 4)))
            for a in range(2):
                for b in range(2):
                    missing = bool(failed_indices.intersection(dependencies[a][b]))
                    for key in ("stencil", "absolute_error", "roundoff_proxy", "contact_shifted_stencil"):
                        assert (step[key][a][b] is None) is missing
                    if missing:
                        assert step["roundoff_status"][a][b] == "unavailable"
                    else:
                        assert step["roundoff_status"][a][b] in (
                            "above_roundoff_proxy", "below_roundoff_proxy")
                    # Neither bound family depends on a perturbed solve.
                    for key in ("model_bounds", "heuristic_bounds"):
                        assert step[key]["status"][a][b] != "input_numerical_unavailable"
            assert step["stencil"][0][1] == step["stencil"][1][0]
    chosen = report["partitions"]["symmetric"]["steps"][0]
    location = points_at(STEPS[0]).index(failed_point)
    assert chosen["points"][location]["evaluation_status"] == "numerical_unavailable"
    if location in (1, 3):
        assert chosen["stencil"][0][1] is not None
    else:
        assert chosen["stencil"][0][0] is not None
        assert chosen["stencil"][1][1] is not None


def check_demo(payload):
    native(payload)
    assert {"joint_cases", "finite_difference_cases", "dark_control", "limitations"} <= set(payload)
    assert payload["limitations"]
    for key in ("joint_cases", "finite_difference_cases"):
        cases = payload[key]
        assert len(cases) == 9
        assert [(case["L"], case["g"]) for case in cases] == list(GRID)
        for case in cases:
            assert case["C"] == 1 and case["m"] == 1
            if case.get("status") == "numerical_unavailable":
                assert type(case["reason"]) is str and case["reason"]
            else:
                assert case["N"] == case["L"]
                assert set(case["partitions"]) == set(PARTITIONS)
                scope_flags(case["scope"])
    dark = payload["dark_control"]
    assert (dark["L"], dark["g"], dark["C"], dark["m"]) == (4, 0.0, 1.0, 2)
    if dark.get("status") != "numerical_unavailable":
        for partition in PARTITIONS:
            assert dark["partitions"][partition]["free_control"]["energy_dark"] is True


def test_demo_preserves_every_unavailable_slot_without_running_controls(monkeypatch):
    calls = []
    def unavailable(kind):
        def report(L, g, C=1.0, m=1):
            calls.append((kind, L, g, C, m))
            raise api.NumericalUnavailable("injected unavailable " + kind)
        return report
    monkeypatch.setattr(api, "joint_report", unavailable("joint"))
    monkeypatch.setattr(api, "finite_difference_report", unavailable("fd"))
    payload = api.demonstration_report()
    check_demo(payload)
    assert len(calls) == 19
    assert [(L, g) for kind, L, g, C, m in calls if kind == "fd"] == list(GRID)
    assert [(L, g, m) for kind, L, g, C, m in calls if kind == "joint"] == [
        (L, g, 1) for L, g in GRID] + [(4, 0.0, 2)]
    for key in ("joint_cases", "finite_difference_cases"):
        assert all(case["status"] == "numerical_unavailable" for case in payload[key])


def test_python38_grammar_for_new_module_demo_and_test():
    for path in (ROOT / "bpr/substrate_joint_source_kernel.py",
                 ROOT / "scripts/demo_substrate_joint_source_kernel.py", Path(__file__)):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))


def test_isolated_stdout_only_text_and_strict_json_demos(tmp_path):
    script = ROOT / "scripts/demo_substrate_joint_source_kernel.py"
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
            check_demo(json.loads(run.stdout, parse_constant=reject_constant))
