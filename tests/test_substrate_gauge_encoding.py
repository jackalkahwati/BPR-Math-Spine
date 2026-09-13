"""Independent tests frozen from the module1 mathematics, not its implementation.

Authored without reading the new module/demo and without executing controls.
Occupation hopping, group permutations and scipy exponential actions are the
independent oracles. Analytic statements are distinct from heuristic numerical
regression checks; no test claims a roundoff or empirical certificate.
"""
from collections import Counter
import copy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from bpr import gauge_heat_kernel as heat
from bpr import substrate_current_response as current
from bpr import substrate_fermionization as occupation_module
from bpr import substrate_gauge_encoding as api


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo_substrate_gauge_encoding.py"
SIZES = (3, 4, 5)
COUPLINGS = (0.0, 0.7, 40.0)
TIMES = (0.0, 0.01, 0.1)
GRID = tuple(itertools.product(SIZES, COUPLINGS))
TOL = 2e-10
AVAILABLE = "available_heuristic"
UNAVAILABLE = "numerical_unavailable"
BMAX = (7 + math.sqrt(5)) / 2
GRAM_SPECTRA = {
    3: [0, 0, 0, 6, 6, 18],
    4: [0, 4, 4, 12, 16, 24, 24, 28],
    5: [9] * 5 + [27] * 5,
}
LEAK_NORMS = {3: 3 * math.sqrt(2), 4: 2 * math.sqrt(7), 5: 3 * math.sqrt(3)}
SEED_NORMS = {3: math.sqrt(5), 4: math.sqrt(14), 5: math.sqrt(18)}


def same(actual, reference):
    """Per-real/imaginary-component tolerance, never max-channel scaling."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual.real, reference.real, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(actual.imag, reference.imag, atol=TOL, rtol=TOL)


def opnorm(matrix):
    """Test-side direct SVD, not the production norm seam."""
    return float(np.linalg.svd(np.asarray(matrix), compute_uv=False)[0])


def decode(matrix):
    if isinstance(matrix, list):
        return np.asarray(matrix)
    assert {"shape", "real", "imag"} <= set(matrix)
    result = np.asarray(matrix["real"]) + 1j * np.asarray(matrix["imag"])
    assert list(result.shape) == matrix["shape"]
    return result


def native(tree):
    if type(tree) is dict:
        assert all(type(key) is str for key in tree)
        for value in tree.values():
            native(value)
    elif type(tree) is list:
        for value in tree:
            native(value)
    else:
        assert tree is None or type(tree) in (str, int, float, bool)
        if type(tree) is float:
            assert math.isfinite(tree)


def comparison(record, status="consistent"):
    assert {"status", "max_absolute_error", "reason"} <= set(record)
    assert record["status"] == status
    if record["max_absolute_error"] is not None:
        assert math.isfinite(record["max_absolute_error"])
        assert record["max_absolute_error"] >= 0
    if status != "consistent":
        assert isinstance(record["reason"], str) and record["reason"]


def scope(record):
    assert record["diagnostic_encoding"] is True
    assert record["local_gauge_emergence"] is False
    assert record["numerical_error_certified"] is False
    assert record["empirical_validation"] is False
    assert record["empirical_status"] == "empirical_test_unavailable"


def product(a, b, L):
    return ((a[0] + a[1] * b[0]) % L, a[1] * b[1])


def inverse(a, L):
    return ((-a[1] * a[0]) % L, a[1])


def spatial_state(state, label):
    result = [0] * len(state)
    for x, number in enumerate(state):
        result[(label[0] + label[1] * x) % len(state)] = number
    return tuple(result)


def independent_oracle(L, g):
    """Stars-and-bars occupancy enumeration and individual directed Bose hops."""
    basis = tuple(sorted({tuple(sites.count(x) for x in range(L))
                          for sites in itertools.combinations_with_replacement(range(L), L)}))
    dimension = math.comb(2 * L - 1, L)
    assert len(basis) == dimension
    index = {state: i for i, state in enumerate(basis)}
    labels = tuple((k, sign) for sign in (1, -1) for k in range(L))
    label_index = {label: i for i, label in enumerate(labels)}
    seed = (L - 1, 1) + (0,) * (L - 2)
    W = np.column_stack([np.eye(dimension)[:, index[spatial_state(seed, a)]] for a in labels])
    left, right = [], []
    for a in labels:
        la, ra = np.zeros((2 * L, 2 * L)), np.zeros((2 * L, 2 * L))
        for j, b in enumerate(labels):
            la[label_index[product(a, b, L)], j] = 1
            ra[label_index[product(b, inverse(a, L), L)], j] = 1
        left.append(la)
        right.append(ra)
    H = np.diag([g * sum(n * (n - 1) // 2 for n in state) for state in basis])
    for column, state in enumerate(basis):
        for source in range(L):
            if state[source] == 0:
                continue
            for target in ((source - 1) % L, (source + 1) % L):
                moved = list(state)
                moved[source] -= 1
                moved[target] += 1
                H[index[tuple(moved)], column] -= math.sqrt(state[source] * (state[target] + 1))
    P = W @ W.T
    Hc = W.T @ H @ W
    B = H @ W - W @ Hc
    E = g * (L - 1) * (L - 2) / 2
    identity = np.eye(2 * L)
    Rs, Ra = right[L], right[L + 1]
    analytic_Hc = E * identity - (Rs + 2 * Ra if L == 3 else 0)
    analytic_gram = ({3: lambda: 3 * (identity + Rs) + 2 * np.ones_like(identity),
                      4: lambda: 14 * identity + 8 * Rs + 6 * Ra,
                      5: lambda: 18 * identity + 9 * Rs}[L])()
    delta = None
    if L == 5:
        delta = 3 * identity - left[1] - left[L - 1] - sum(left[L:]) / L
    return dict(L=L, g=g, basis=basis, labels=labels, seed=seed, index=index,
                W=W, P=P, H=H, Hc=Hc, B=B, E=E, left=left, right=right,
                analytic_Hc=analytic_Hc, analytic_gram=analytic_gram, delta=delta)


def ambient_spatial(oracle, label):
    dimension = len(oracle["basis"])
    U = np.zeros((dimension, dimension))
    for column, state in enumerate(oracle["basis"]):
        U[oracle["index"][spatial_state(state, label)], column] = 1
    return U


def centered_actions(oracle, t):
    """Independent scaling/Taylor exponential action, not Hermitian eigh."""
    W, H, Hc, E = (oracle[key] for key in ("W", "H", "Hc", "E"))
    ambient = expm_multiply(-1j * t * (H - E * np.eye(len(H))), W)
    compressed = expm_multiply(-1j * t * (Hc - E * np.eye(len(Hc))), np.eye(len(Hc)))
    target = None if oracle["delta"] is None else expm_multiply(
        -1j * t * oracle["delta"], np.eye(len(Hc)))
    return ambient, compressed, target


def patch_callable_aliases(monkeypatch, original, replacement):
    """Instrument inherited boundary whether imported directly or by module."""
    for module in (api, occupation_module, heat):
        for name, value in tuple(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, name, replacement)


@pytest.fixture(scope="module")
def oracles():
    return {key: independent_oracle(*key) for key in GRID}


@pytest.fixture(scope="module")
def successful_demo():
    """One nine-case run shared by report/math tests and exact work budget."""
    calls = {"model": [], "screen": [], "eigh": [], "eigvalsh": []}
    original_model = occupation_module.fixed_number_model
    original_screen = api._screened_eigensystem
    original_eigh, original_eigvalsh = np.linalg.eigh, np.linalg.eigvalsh

    def model(*args, **kwargs):
        bound = inspect.signature(original_model).bind(*args, **kwargs)
        bound.apply_defaults()
        calls["model"].append(dict(bound.arguments))
        return original_model(*args, **kwargs)

    def screen(H):
        calls["screen"].append(np.array(H, copy=True))
        return original_screen(H)

    def eigh(H, *args, **kwargs):
        calls["eigh"].append(np.array(H, copy=True))
        return original_eigh(H, *args, **kwargs)

    def eigvalsh(H, *args, **kwargs):
        calls["eigvalsh"].append(np.array(H, copy=True))
        return original_eigvalsh(H, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_callable_aliases(monkeypatch, original_model, model)
        patch_callable_aliases(monkeypatch, original_eigh, eigh)
        patch_callable_aliases(monkeypatch, original_eigvalsh, eigvalsh)
        monkeypatch.setattr(api, "_screened_eigensystem", screen)
        monkeypatch.setattr(np.linalg, "eigh", eigh)
        monkeypatch.setattr(np.linalg, "eigvalsh", eigvalsh)
        report = api.demonstration_report()
    return report, calls


@pytest.fixture(scope="module")
def reports(successful_demo):
    return {(case["L"], case["g"]): case for case in successful_demo[0]["cases"]}


def test_constants_and_inherited_exception_identity():
    assert api.MAX_DIMENSION == 512
    assert api.SIZES == SIZES and api.COUPLINGS == COUPLINGS and api.TIMES == TIMES
    assert api.ATOL == TOL and api.RTOL == TOL
    assert api.NumericalUnavailable is current.NumericalUnavailable


def test_successful_demo_controls_scope_order_and_counts(successful_demo):
    report, _ = successful_demo
    native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["status"] == AVAILABLE
    assert report["model_id"]
    scope(report["scope"])
    assert report["controls"] == {"sizes": list(SIZES), "couplings": list(COUPLINGS),
                                  "times": list(TIMES), "C": 1.0}
    assert [(case["L"], case["g"]) for case in report["cases"]] == list(GRID)
    assert report["summary"]["case_status_counts"].get(AVAILABLE) == 9
    assert sum(report["summary"]["case_status_counts"].values()) == 9
    assert report["summary"]["analytic_d5_obstruction_count"] == 3
    for case in report["cases"]:
        assert case["status"] == AVAILABLE and case["reason"] is None
        assert case["model_id"] == report["model_id"]
        assert case["N"] == case["L"] and case["C"] == 1
        assert case["dimension"] == math.comb(2 * case["L"] - 1, case["L"])
        assert case["code_dimension"] == 2 * case["L"]
        scope(case["scope"])
        assert case["conventions"]
        assert {"seed", "labels", "orbit_dimension", "isometry_residual", "left_action_residual",
                "right_action_residual", "endpoint_commutator_residual", "coordinate_covariance_residual",
                "transport_identity_residual", "comparison"} <= set(case["orbit"])
        assert "H" not in case and "W" not in case
        assert [row["t"] for row in case["times"]] == list(TIMES)
        assert [row["tau"] for row in case["times"]] == list(TIMES)


def test_successful_demo_exact_work_budget_and_centered_scaled_eigh(successful_demo, oracles):
    _, calls = successful_demo
    assert len(calls["model"]) == 9
    assert [(row["L"], row["g"]) for row in calls["model"]] == list(GRID)
    assert all(row["N"] == row["L"] and row["C"] == 1 for row in calls["model"])
    assert len(calls["screen"]) == len(calls["eigh"]) == 21
    assert len(calls["eigvalsh"]) == 9
    assert Counter(H.shape for H in calls["screen"]) == Counter({(10, 10): 9, (6, 6): 3,
                                                               (35, 35): 3, (8, 8): 3,
                                                               (126, 126): 3})
    # The helper diagonalizes H/max(1,Frobenius(H)), NOT the raw centered H.
    for screened, diagonalized in zip(calls["screen"], calls["eigh"]):
        scale = max(1.0, float(np.linalg.norm(screened, ord="fro")))
        same(diagonalized, screened / scale)
    expected = []
    for oracle in oracles.values():
        expected.extend([oracle["H"] - oracle["E"] * np.eye(len(oracle["H"])),
                         oracle["Hc"] - oracle["E"] * np.eye(2 * oracle["L"])])
        if oracle["delta"] is not None:
            expected.append(oracle["delta"])
    unmatched = list(calls["screen"])
    for H in expected:
        matches = [i for i, got in enumerate(unmatched)
                   if got.shape == H.shape and np.allclose(got, H, atol=TOL, rtol=TOL)]
        assert matches, "Missing independent centered propagation Hamiltonian"
        unmatched.pop(matches[0])
    assert not unmatched
    assert Counter(H.shape for H in calls["eigvalsh"]) == Counter({(6, 6): 3, (8, 8): 3, (10, 10): 3})
    for gram_input in calls["eigvalsh"]:
        assert np.all(np.isfinite(gram_input))
        same(gram_input, gram_input.conj().T)
    # The full code dimension is essential: rectangular B singular values can
    # omit dark directions. Report-value tests separately check signed minima.


@pytest.mark.parametrize("L", SIZES)
def test_orbit_exact_basis_and_regular_coordinate_algebra(L, oracles, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Orbit fixture must not construct H or diagonalize")
    patch_callable_aliases(monkeypatch, occupation_module.fixed_number_model, forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    data = api._orbit_data(L)
    oracle = oracles[(L, 0.0)]
    assert {"L", "N", "basis", "labels", "seed", "W", "left", "right"} <= set(data)
    assert data["L"] == data["N"] == L
    for key in ("basis", "labels", "seed"):
        assert type(data[key]) is tuple and data[key] == oracle[key]
    assert type(data["left"]) is type(data["right"]) is tuple
    assert len(data["left"]) == len(data["right"]) == 2 * L
    assert all(np.isrealobj(action) for action in data["left"] + data["right"])
    W, labels = data["W"], data["labels"]
    np.testing.assert_array_equal(W, oracle["W"])
    assert len({spatial_state(data["seed"], label) for label in labels}) == 2 * L
    assert [label for label in labels if spatial_state(data["seed"], label) == data["seed"]] == [(0, 1)]
    np.testing.assert_array_equal(W.T @ W, np.eye(2 * L))
    assert all(max(state) > 1 for i, state in enumerate(data["basis"]) if np.any(W[i]))
    coordinates = [np.diag(np.eye(2 * L)[i]) for i in range(2 * L)]
    for i, a in enumerate(labels):
        la, ra = data["left"][i], data["right"][i]
        np.testing.assert_array_equal(la, oracle["left"][i])
        np.testing.assert_array_equal(ra, oracle["right"][i])
        for j, b in enumerate(labels):
            ab = labels.index(product(a, b, L))
            np.testing.assert_array_equal(la @ data["left"][j], data["left"][ab])
            np.testing.assert_array_equal(ra @ data["right"][j], data["right"][ab])
            np.testing.assert_array_equal(la @ data["right"][j], data["right"][j] @ la)
            np.testing.assert_array_equal(la @ coordinates[j] @ la.T, coordinates[ab])
            right_image = labels.index(product(b, inverse(a, L), L))
            np.testing.assert_array_equal(ra @ coordinates[j] @ ra.T, coordinates[right_image])
            unit = coordinates[i] @ data["left"][labels.index(product(a, inverse(b, L), L))] @ coordinates[j]
            expected = np.zeros_like(unit)
            expected[i, j] = 1
            np.testing.assert_array_equal(unit, expected)
    P = W @ W.T
    np.testing.assert_array_equal(W @ data["left"][0] @ W.T, P)
    assert not np.array_equal(P, np.eye(len(P)))


@pytest.mark.parametrize("L", SIZES)
def test_orbit_owned_arrays_and_no_cache_poisoning(L):
    first, second = api._orbit_data(L), api._orbit_data(L)
    for key in ("W", "left", "right"):
        aa = (first[key],) if key == "W" else first[key]
        bb = (second[key],) if key == "W" else second[key]
        for a, b in zip(aa, bb):
            assert not np.shares_memory(a, b)
            a.flat[0] = 999
    third = api._orbit_data(L)
    for key in ("W", "left", "right"):
        aa = (second[key],) if key == "W" else second[key]
        bb = (third[key],) if key == "W" else third[key]
        for a, b in zip(aa, bb):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("key", GRID)
def test_compression_full_leakage_gram_spectra_and_multiplication_defect(key, reports, oracles):
    report, oracle = reports[key], oracles[key]
    L, g = key
    H, W, B, Hc = (oracle[name] for name in ("H", "W", "B", "Hc"))
    compression, leakage = report["compression"], report["leakage"]
    same(Hc, oracle["analytic_Hc"])
    same(decode(compression["matrix"]), Hc)
    same(decode(compression["analytic_matrix"]), oracle["analytic_Hc"])
    same(decode(compression["centered_matrix"]), Hc - oracle["E"] * np.eye(2 * L))
    same(compression["onsite_energy"], oracle["E"])
    spectrum = [-3, -math.sqrt(3), -math.sqrt(3), math.sqrt(3), math.sqrt(3), 3] if L == 3 else [0] * (2 * L)
    same(compression["analytic_centered_eigenvalues"], spectrum)
    same(np.linalg.eigvalsh(Hc - oracle["E"] * np.eye(2 * L)), spectrum)
    reflection = oracle["right"][L]
    expected_commutator = 2 * math.sqrt(3) if L == 3 else 0
    same(opnorm(Hc @ reflection - reflection @ Hc), expected_commutator)
    same(compression["right_reflection_commutator_norm"], expected_commutator)
    same(compression["analytic_right_reflection_commutator_norm"], expected_commutator)
    gram = B.T @ B
    same(gram, oracle["analytic_gram"])
    same(decode(leakage["gram"]), gram)
    same(decode(leakage["analytic_gram"]), oracle["analytic_gram"])
    same(np.linalg.eigvalsh(gram), GRAM_SPECTRA[L])
    same(W.T @ (H @ H) @ W - Hc @ Hc, gram)
    same((H @ W).T @ (H @ W) - Hc @ Hc, gram)
    for field in ("norm", "analytic_norm"):
        same(leakage[field], LEAK_NORMS[L])
    for field in ("seed_column_norm", "analytic_seed_column_norm"):
        same(leakage[field], SEED_NORMS[L])
    for field in ("minimum_gram_eigenvalue", "analytic_minimum_gram_eigenvalue"):
        same(leakage[field], min(GRAM_SPECTRA[L]))
    same(opnorm(B), LEAK_NORMS[L])
    same(np.linalg.norm(B[:, 0]), SEED_NORMS[L])
    assert leakage["all_states_leak"] is (L == 5)
    assert isinstance(leakage["invariance_status"], str) and leakage["invariance_status"]
    comparison(compression["comparison"])
    comparison(leakage["comparison"])
    for field, value in report["orbit"].items():
        if field.endswith("_residual"):
            same(value, 0)
    assert report["orbit"]["seed"] == list(oracle["seed"])
    assert report["orbit"]["labels"] == [list(label) for label in oracle["labels"]]
    assert report["orbit"]["orbit_dimension"] == 2 * L
    comparison(report["orbit"]["comparison"])


@pytest.mark.parametrize("L", SIZES)
def test_exact_hop_witness_and_sign_character_dark_states(L, oracles):
    oracle = oracles[(L, 0.7)]
    seed = oracle["seed"]
    merged = (L,) + (0,) * (L - 1)
    row = oracle["index"][merged]
    same(oracle["B"][row, 0], -math.sqrt(L))
    assert merged not in {spatial_state(seed, label) for label in oracle["labels"]}
    sign = np.array([label[1] for label in oracle["labels"]]) / math.sqrt(2 * L)
    if L in (3, 4):
        same(oracle["B"] @ sign, np.zeros(len(oracle["H"])))
        energy = oracle["E"] + (3 if L == 3 else 0)
        same(oracle["H"] @ oracle["W"] @ sign, energy * (oracle["W"] @ sign))
    else:
        same(np.linalg.norm(oracle["B"] @ sign), 3)
    if L == 3:
        exchanged = (1, 2, 0)
        same(oracle["H"][oracle["index"][exchanged], oracle["index"][seed]], -2)


@pytest.mark.parametrize("L", SIZES)
def test_global_spatial_vs_partial_symmetry_density_support_and_dense_block_oracle(L, reports, oracles):
    oracle = oracles[(L, 0.7)]
    observed = reports[(L, 0.7)]["symmetry"]
    H, W, P = (oracle[name] for name in ("H", "W", "P"))
    for label in ((1, 1), (0, -1)):
        U = ambient_spatial(oracle, label)
        left = oracle["left"][oracle["labels"].index(label)]
        same(H @ U, U @ H)
        same(U @ W, W @ left)
        assert not np.array_equal(U, W @ left @ W.T)
    same(observed["spatial_intertwining_residual"], 0)
    same(observed["spatial_hamiltonian_commutator_norm"], 0)
    R = oracle["right"][L]
    partial = W @ R @ W.T
    dense_commutator = H @ partial - partial @ H
    Qrows = np.flatnonzero(np.diag(P) == 0)
    Bq = oracle["B"][Qrows]
    A = oracle["Hc"] @ R - R @ oracle["Hc"]
    block = np.block([[A, -R @ Bq.T], [Bq @ R, np.zeros((len(Qrows), len(Qrows)))]])
    expected = 3 * math.sqrt(2) if L == 3 else LEAK_NORMS[L]
    same(opnorm(dense_commutator), expected)
    same(opnorm(block), expected)
    same(observed["partial_right_reflection_commutator_norm"], expected)
    if L == 3:
        assert observed["analytic_partial_right_reflection_commutator_norm"] is None
    else:
        same(observed["analytic_partial_right_reflection_commutator_norm"], expected)
        same(W.T @ dense_commutator @ W, np.zeros((2 * L, 2 * L)))
    same(opnorm(H @ P - P @ H), LEAK_NORMS[L])
    densities = [np.diag([state[x] for state in oracle["basis"]]) for x in range(L)]
    for nx in densities:
        same(opnorm(nx @ partial - partial @ nx), 1)
        same(nx @ P, P @ nx)
        for ny in densities:
            same(W.T @ nx @ ny @ W, (W.T @ nx @ W) @ (W.T @ ny @ W))
    same(observed["density_right_reflection_commutator_norms"], np.ones(L))
    same(observed["analytic_density_right_reflection_commutator_norms"], np.ones(L))
    same(observed["density_multiplication_defect_residual"], 0)
    comparison(observed["comparison"])


@pytest.mark.parametrize("g", COUPLINGS)
def test_d5_fixed_target_spectrum_unfitted_mismatch_and_maximal_witness(g, reports, oracles):
    target, oracle = reports[(5, g)]["target"], oracles[(5, g)]
    delta, B, W = oracle["delta"], oracle["B"], oracle["W"]
    spectrum = [0, 2] + [(7 - math.sqrt(5)) / 2] * 4 + [BMAX] * 4
    same(heat.central_laplacian(5), delta)
    same(decode(target["generator"]), delta)
    same(target["analytic_eigenvalues"], spectrum)
    same(np.linalg.eigvalsh(delta), spectrum)
    assert target["status"] == "analytic_obstruction"
    for field in ("generator_mismatch_norm", "analytic_generator_mismatch_norm"):
        same(target[field], BMAX)
    F = oracle["H"] @ W - W @ (oracle["E"] * np.eye(10) + delta)
    same(F, B - W @ delta)
    same(F.T @ F, B.T @ B + delta @ delta)
    full_norm = math.sqrt(27 + BMAX ** 2)
    same(opnorm(F), full_norm)
    for field in ("full_residual_norm", "analytic_full_residual_norm"):
        same(target[field], full_norm)
    witness = np.array([math.cos(4 * math.pi * k / 5) / math.sqrt(5) for k, _ in oracle["labels"]])
    same(target["maximal_witness"], witness)
    same(np.linalg.norm(witness), 1)
    same(delta @ witness, BMAX * witness)
    same((F.T @ F) @ witness, full_norm ** 2 * witness)
    same(np.linalg.norm(F @ witness), full_norm)
    same(target["witness_residual"], 0)
    comparison(target["comparison"])


@pytest.mark.parametrize("L", (3, 4))
def test_unsupported_target_is_not_zero_mismatch(L, reports):
    target = reports[(L, 0.7)]["target"]
    assert target["status"] == "not_applicable"
    assert isinstance(target["reason"], str) and target["reason"]
    for key in ("generator", "analytic_eigenvalues", "generator_mismatch_norm",
                "analytic_generator_mismatch_norm", "full_residual_norm",
                "analytic_full_residual_norm", "maximal_witness", "witness_residual", "comparison"):
        assert target[key] is None
    assert reports[(L, 0.7)]["eigensystems"]["target"]["status"] == "not_applicable"
    for row in reports[(L, 0.7)]["times"]:
        record = row["target"]
        assert record["status"] == "not_applicable" and record["reason"]
        for field in ("full_space_error", "projected_error", "leakage_amplitude",
                      "analytic_upper_bound", "analytic_lower_bound", "upper_bound_comparison", "lower_bound_comparison"):
            assert record[field] is None


@pytest.mark.parametrize("key", GRID)
def test_dynamics_independent_exponential_action_and_duhamel_bounds(key, reports, oracles):
    oracle, report = oracles[key], reports[key]
    L, _ = key
    W, P = oracle["W"], oracle["P"]
    for row in report["times"]:
        t = row["t"]
        ambient, compressed, target = centered_actions(oracle, t)
        leakage = opnorm(ambient - P @ ambient)
        for name, expected in (("compression", compressed), ("target", target)):
            if expected is None:
                continue
            record = row[name]
            assert record["status"] == AVAILABLE and record["reason"] is None
            full = opnorm(ambient - W @ expected)
            projected = opnorm(W.T @ ambient - expected)
            same(record["full_space_error"], full)
            same(record["projected_error"], projected)
            same(record["leakage_amplitude"], leakage)
            norm = LEAK_NORMS[L] if name == "compression" else math.sqrt(27 + BMAX ** 2)
            upper = min(2, abs(t) * norm)
            lower = 0 if name == "compression" else max(0, 2 * math.sin(abs(t) * BMAX / 2) - 27 * t ** 2 / 2)
            same(record["analytic_upper_bound"], upper)
            same(record["analytic_lower_bound"], lower)
            assert record["full_space_error"] <= upper + TOL + TOL * abs(upper)
            assert record["full_space_error"] >= lower - TOL - TOL * abs(lower)
            comparison(record["upper_bound_comparison"])
            comparison(record["lower_bound_comparison"])
            assert "probability" not in record
        if L in (4, 5):
            return_error = opnorm(W.T @ ambient - np.eye(2 * L))
            assert return_error <= t * t * LEAK_NORMS[L] ** 2 / 2 + TOL
        if L == 5 and t:
            assert row["target"]["analytic_lower_bound"] > 0
        if t:
            # Compression error has first-order leakage; projected error need not.
            assert row["compression"]["full_space_error"] > row["compression"]["projected_error"]


@pytest.mark.parametrize("key", GRID)
def test_eigensystem_report_metadata_is_heuristic_and_array_free(key, reports, oracles):
    oracle = oracles[key]
    matrices = {"ambient": oracle["H"] - oracle["E"] * np.eye(len(oracle["H"])),
                "compression": oracle["Hc"] - oracle["E"] * np.eye(2 * key[0]),
                "target": oracle["delta"]}
    for name, H in matrices.items():
        if H is None:
            continue
        metadata = reports[key]["eigensystems"][name]
        assert metadata["status"] == AVAILABLE and metadata["reason"] is None
        assert "vectors" not in metadata and "values" not in metadata
        tolerance = 256 * np.finfo(float).eps * len(H)
        scale = max(1.0, float(np.linalg.norm(H, "fro")))
        same(metadata["tolerance"], tolerance)
        same(metadata["scale"], scale)
        assert 0 <= metadata["orthogonality_residual"] <= tolerance
        assert 0 <= metadata["eigenpair_residual"] <= tolerance * scale


class ConversionTrap:
    def __float__(self):
        raise AssertionError("Must reject custom conversion protocols")

    def __int__(self):
        raise AssertionError("Must reject custom conversion protocols")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Must reject custom conversion protocols")


@pytest.mark.parametrize("L,g", [
    (True, 0.7), (np.bool_(True), 0.7), (3.0, 0.7), ("3", 0.7),
    (2, 0.7), (6, 0.7), (-3, 0.7), (3 + 0j, 0.7), (ConversionTrap(), 0.7),
    (3, True), (3, np.bool_(False)), (3, "0.7"), (3, 0.7 + 0j),
    (3, np.complex128(40)), (3, float("nan")), (3, float("inf")),
    (3, -float("inf")), (3, -0.7), (3, 1.0), (3, np.float32(0.7)),
    (3, np.nextafter(0.7, 1.0)), (3, np.nextafter(40.0, 0.0)),
    (3, np.array(0.7)), (3, [0.7]), (3, Fraction(7, 10)),
    (3, Decimal("0.7")), (3, ConversionTrap()),
])
def test_public_rejects_invalid_enum_before_model_work(L, g, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid public inputs must fail before construction")
    patch_callable_aliases(monkeypatch, occupation_module.fixed_number_model, forbidden)
    monkeypatch.setattr(api, "_orbit_data", forbidden)
    with pytest.raises((TypeError, ValueError)) as error:
        api.case_report(L, g)
    assert not isinstance(error.value, api.NumericalUnavailable)


@pytest.mark.parametrize("extra", [{"C": 2}, {"N": 2}, {"model": object()}, {"cache": {}}])
def test_public_api_has_no_parameter_search_or_caller_cache(extra):
    with pytest.raises(TypeError):
        api.case_report(3, 0.7, **extra)


def test_extended_precision_cannot_round_into_allowed_coupling(monkeypatch):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("This platform has no floating dtype wider than binary64")
    wider = np.nextafter(np.longdouble(0.7), np.longdouble(1))
    assert float(wider) == 0.7 and wider.as_integer_ratio() != (0.7).as_integer_ratio()
    with pytest.raises((TypeError, ValueError)):
        api.case_report(3, wider)


@pytest.mark.parametrize("L,g", [(np.int32(3), np.float64(0.7)),
                                 (np.int64(3), np.float32(40)),
                                 (np.uint8(3), np.float32(0)),
                                 (3, np.longdouble(0.7)), (3, 40), (3, -0.0)])
def test_exact_numpy_and_builtin_members_are_admitted(L, g):
    # This is input validation over the same frozen cases, not extra controls.
    report = api.case_report(L, g)
    assert report["L"] == 3 and report["g"] == float(g)
    assert report["status"] == AVAILABLE
    native(report)


@pytest.mark.parametrize("bad", [True, np.bool_(True), "3", 3.0, 2, 6, -1, ConversionTrap()])
def test_orbit_invalid_size_before_enumeration(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid orbit size reached occupation enumeration")
    patch_callable_aliases(monkeypatch, occupation_module._occupations, forbidden)
    with pytest.raises((TypeError, ValueError)):
        api._orbit_data(bad)


@pytest.mark.parametrize("bad", [
    [], [1, 2], [[1, 0]], [[1, 0], [0]], np.zeros((2, 2, 1)),
    np.zeros((0, 0)), np.broadcast_to(0.0, (513, 513)),
    np.array([[True, False], [False, True]]), [[1, False], [False, 1]],
    [["1", "0"], ["0", "1"]], [[ConversionTrap()]],
    [[float("nan")]], [[float("inf")]], [[complex(0, float("inf"))]],
    [[0, 1], [0, 0]], [[1j]], ConversionTrap(),
])
def test_screen_rejects_invalid_input_before_eigh(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid matrix reached diagonalization")
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    with pytest.raises((TypeError, ValueError)) as error:
        api._screened_eigensystem(bad)
    assert not isinstance(error.value, api.NumericalUnavailable)


def test_screen_scaled_copied_hermitian_input_and_owned_outputs(monkeypatch):
    H = np.array([[2, 1j], [-1j, 4]], dtype=complex)
    before = H.copy()
    original = np.linalg.eigh
    captured = []

    def eigh(matrix, *args, **kwargs):
        captured.append(np.array(matrix, copy=True))
        assert not np.shares_memory(H, matrix)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    result = api._screened_eigensystem(H)
    assert {"values", "vectors", "orthogonality_residual", "eigenpair_residual", "tolerance", "scale"} <= set(result)
    scale = max(1, np.linalg.norm(H, "fro"))
    same(captured[0], H / scale)
    same(result["values"], [3 - math.sqrt(2), 3 + math.sqrt(2)])
    tau = 256 * np.finfo(float).eps * 2
    same(result["tolerance"], tau)
    same(result["scale"], scale)
    residual = np.linalg.norm(H @ result["vectors"] - result["vectors"] * result["values"], "fro")
    orthogonality = np.linalg.norm(result["vectors"].conj().T @ result["vectors"] - np.eye(2), "fro")
    same(result["eigenpair_residual"], residual)
    same(result["orthogonality_residual"], orthogonality)
    assert residual <= tau * scale and orthogonality <= tau
    assert result["values"].flags.owndata and result["vectors"].flags.owndata
    np.testing.assert_array_equal(H, before)
    H[:] = 0
    same(result["values"], [3 - math.sqrt(2), 3 + math.sqrt(2)])


@pytest.mark.parametrize("variant", ["wrong_values_shape", "wrong_vectors_shape", "nan_value",
                                     "complex_values", "nan_vector", "nonorthogonal", "wrong_pairs",
                                     "boolean_values", "string_vectors"])
def test_screen_invalid_eigenoutputs_raise_numerical_unavailable(variant, monkeypatch):
    def bad_eigh(H, *args, **kwargs):
        values, vectors = np.array([1.0, 2.0]) / math.sqrt(5), np.eye(2)
        if variant == "wrong_values_shape":
            values = values[:, None]
        elif variant == "wrong_vectors_shape":
            vectors = vectors[:, :1]
        elif variant == "nan_value":
            values[0] = np.nan
        elif variant == "complex_values":
            values = values.astype(complex) + 1j
        elif variant == "nan_vector":
            vectors[0, 0] = np.inf
        elif variant == "nonorthogonal":
            vectors[0, 0] = 2
        elif variant == "wrong_pairs":
            values = values[::-1]
        elif variant == "boolean_values":
            values = values.astype(bool)
        elif variant == "string_vectors":
            vectors = vectors.astype(str)
        return values, vectors
    monkeypatch.setattr(np.linalg, "eigh", bad_eigh)
    with pytest.raises(api.NumericalUnavailable):
        api._screened_eigensystem(np.diag([1.0, 2.0]))


def test_screen_eigh_failure_and_valid_rotated_degenerate_eigenvectors(monkeypatch):
    def unavailable(*args, **kwargs):
        raise np.linalg.LinAlgError("injected eigensolver failure")
    monkeypatch.setattr(np.linalg, "eigh", unavailable)
    with pytest.raises(api.NumericalUnavailable):
        api._screened_eigensystem(np.zeros((2, 2)))
    rotated = np.array([[1, 1j], [1j, 1]], dtype=complex) / math.sqrt(2)
    values = np.zeros(2)
    monkeypatch.setattr(np.linalg, "eigh", lambda H: (values, rotated))
    result = api._screened_eigensystem(np.zeros((2, 2)))
    same(result["values"], [0, 0])
    same(result["vectors"], rotated)
    assert not np.shares_memory(result["values"], values)
    assert not np.shares_memory(result["vectors"], rotated)
    columns = np.array([[1, 1j], [0, 2]], dtype=complex)
    evolved = api._unitary_columns(result, 0.1, columns)
    same(evolved, columns)
    assert evolved.dtype.kind == "c" and evolved.flags.owndata
    assert not np.shares_memory(evolved, columns)
    assert not np.shares_memory(evolved, result["vectors"])


def test_screen_signed_tiny_spectrum_is_not_clipped():
    tiny = np.nextafter(0.0, 1.0)
    system = api._screened_eigensystem(np.array([[-tiny]]))
    assert system["values"][0] == -tiny
    assert system["values"][0] < 0
    zero = api._screened_eigensystem(np.zeros((2, 2)))
    same(zero["values"], [0, 0])


def test_screen_retains_allowed_small_asymmetry_in_original_residual(monkeypatch):
    tau = 256 * np.finfo(float).eps * 2
    H = np.array([[1.0, tau / 8], [0, 1.0]])
    seen = []

    def eigh(matrix):
        seen.append(matrix.copy())
        return np.full(2, 1 / np.linalg.norm(H, "fro")), np.eye(2)

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    system = api._screened_eigensystem(H)
    same(seen[0], H / np.linalg.norm(H, "fro"))
    assert seen[0][0, 1] != seen[0][1, 0], "Do not silently symmetrize accepted inputs"
    assert system["eigenpair_residual"] > 0
    assert system["eigenpair_residual"] <= tau * system["scale"]


def test_unitary_columns_matches_independent_exponential_with_complex_basis():
    H = np.array([[2, 1j], [-1j, -1]], dtype=complex)
    columns = np.array([[1, 2j, -1], [3j, 0, 1]], dtype=complex)
    system = api._screened_eigensystem(H)
    before = columns.copy()
    expected = expm_multiply(-0.1j * H, columns)
    got = api._unitary_columns(system, 0.1, columns)
    same(got, expected)
    assert got.dtype.kind == "c" and got.flags.owndata
    np.testing.assert_array_equal(columns, before)
    assert not np.shares_memory(got, columns)


def test_operator_norm_is_spectral_not_frobenius_and_no_spectrum_clipping():
    matrix = np.diag([3.0, 4.0]).astype(complex)
    same(api._operator_norm(matrix), 4)
    same(api._operator_norm(np.array([[3, 4j]])), 5)
    same(api._operator_norm(np.zeros((3, 2))), 0)


@pytest.mark.parametrize("complex_input", [False, True])
def test_operator_norm_preserves_smallest_subnormal_representable_value(complex_input):
    tiny = np.nextafter(0.0, 1.0)
    # A single nonzero entry has exactly its modulus as the spectral norm.
    # Pure imaginary input exercises complex scaling without sqrt(2) rounding.
    matrix = np.array([[complex(0.0, tiny)]]) if complex_input else np.array([[tiny]])
    observed = api._operator_norm(matrix)
    assert math.isfinite(observed)
    assert observed == tiny
    assert observed > 0


def assert_exact_references_survive(report, L):
    same(report["leakage"]["analytic_norm"], LEAK_NORMS[L])
    same(report["leakage"]["analytic_seed_column_norm"], SEED_NORMS[L])
    same(report["leakage"]["analytic_minimum_gram_eigenvalue"], min(GRAM_SPECTRA[L]))
    assert report["leakage"]["all_states_leak"] is (L == 5)
    if L == 5:
        assert report["target"]["status"] == "analytic_obstruction"
        same(report["target"]["analytic_generator_mismatch_norm"], BMAX)
        same(report["target"]["analytic_full_residual_norm"], math.sqrt(27 + BMAX ** 2))
    assert len(report["times"]) == 3
    for row, t in zip(report["times"], TIMES):
        assert row["t"] == row["tau"] == t
        same(row["compression"]["analytic_upper_bound"], min(2, abs(t) * LEAK_NORMS[L]))
        same(row["compression"]["analytic_lower_bound"], 0)
        if L == 5:
            same(row["target"]["analytic_upper_bound"], min(2, abs(t) * math.sqrt(27 + BMAX ** 2)))
            same(row["target"]["analytic_lower_bound"], max(0, 2 * math.sin(abs(t) * BMAX / 2) - 27 * t ** 2 / 2))
    native(report)
    json.dumps(report, allow_nan=False)


def assert_unavailable_dynamics(record):
    assert record["status"] == UNAVAILABLE and record["reason"]
    for name in ("full_space_error", "projected_error"):
        assert record[name] is None
    if record["leakage_amplitude"] is not None:
        assert math.isfinite(record["leakage_amplitude"]) and record["leakage_amplitude"] >= 0
    comparison(record["upper_bound_comparison"], "inconclusive")
    comparison(record["lower_bound_comparison"], "inconclusive")


@pytest.mark.parametrize("failed", ["ambient", "compression", "target"])
def test_failed_eigensystem_isolated_to_dependent_dynamics(failed, monkeypatch):
    original = api._screened_eigensystem

    def screen(H):
        H = np.asarray(H)
        kind = "ambient" if H.shape == (126, 126) else "compression" if np.max(np.abs(H)) < TOL else "target"
        if kind == failed:
            raise api.NumericalUnavailable("injected " + failed + " eigensystem failure")
        return original(H)

    monkeypatch.setattr(api, "_screened_eigensystem", screen)
    report = api.case_report(5, 0.7)
    assert report["status"] == UNAVAILABLE and report["reason"]
    assert report["eigensystems"][failed]["status"] == UNAVAILABLE
    for section in ("orbit", "compression", "leakage", "symmetry", "target"):
        # A failed comparison eigensystem leaves its spectrum check unavailable.
        expected = "inconclusive" if section == failed else "consistent"
        comparison(report[section]["comparison"], expected)
    assert report["compression"]["matrix"] is not None
    assert report["compression"]["centered_matrix"] is not None
    assert report["target"]["generator"] is not None
    assert_exact_references_survive(report, 5)
    for row in report["times"]:
        for name in ("compression", "target"):
            if failed in ("ambient", name):
                assert_unavailable_dynamics(row[name])
            else:
                assert row[name]["status"] == AVAILABLE
                assert row[name]["full_space_error"] is not None


@pytest.mark.parametrize("failed", ["ambient", "compression", "target"])
def test_single_time_propagation_failure_does_not_erase_other_times(failed, monkeypatch):
    original = api._unitary_columns

    def propagate(system, t, columns):
        kind = "ambient" if len(system["values"]) == 126 else "compression" if np.max(np.abs(system["values"])) < TOL else "target"
        if t == 0.01 and kind == failed:
            raise api.NumericalUnavailable("injected single-time " + failed)
        return original(system, t, columns)

    monkeypatch.setattr(api, "_unitary_columns", propagate)
    report = api.case_report(5, 0.7)
    assert report["status"] == UNAVAILABLE
    assert_exact_references_survive(report, 5)
    for row in report["times"]:
        for name in ("compression", "target"):
            if row["t"] == 0.01 and failed in ("ambient", name):
                assert_unavailable_dynamics(row[name])
            else:
                assert row[name]["status"] == AVAILABLE


def test_gram_minimum_failure_preserves_independent_observations(monkeypatch):
    def eigvalsh(*args, **kwargs):
        raise np.linalg.LinAlgError("injected full Gram spectrum failure")
    monkeypatch.setattr(np.linalg, "eigvalsh", eigvalsh)
    report = api.case_report(3, 0.7)
    assert report["status"] == UNAVAILABLE and report["reason"]
    assert report["leakage"]["minimum_gram_eigenvalue"] is None
    comparison(report["leakage"]["comparison"], "inconclusive")
    same(report["leakage"]["norm"], LEAK_NORMS[3])
    assert report["leakage"]["gram"] is not None
    for name in ("orbit", "compression", "symmetry"):
        comparison(report[name]["comparison"])
    for row in report["times"]:
        assert row["compression"]["status"] == AVAILABLE
    assert_exact_references_survive(report, 3)


def test_norm_diagnostic_failures_keep_exact_analytic_envelopes(monkeypatch):
    def norm(matrix):
        raise api.NumericalUnavailable("injected norm diagnostic failure")

    monkeypatch.setattr(api, "_operator_norm", norm)
    report = api.case_report(5, 0.7)
    assert report["status"] == UNAVAILABLE
    assert report["leakage"]["norm"] is None
    comparison(report["leakage"]["comparison"], "inconclusive")
    assert_exact_references_survive(report, 5)


def test_norm_failure_in_one_dynamics_record_is_time_local(monkeypatch):
    original_norm, original_propagate = api._operator_norm, api._unitary_columns
    pending = {"time": None, "fired": False}

    def propagate(system, t, columns):
        result = original_propagate(system, t, columns)
        if len(system["values"]) == 10:
            pending["time"] = t
        return result

    def norm(matrix):
        if pending["time"] == 0.01 and not pending["fired"] and np.asarray(matrix).shape == (10, 6):
            pending["fired"] = True
            raise api.NumericalUnavailable("injected one-record norm failure")
        return original_norm(matrix)

    monkeypatch.setattr(api, "_operator_norm", norm)
    monkeypatch.setattr(api, "_unitary_columns", propagate)
    report = api.case_report(3, 0.7)
    assert pending["fired"]
    assert report["status"] == UNAVAILABLE
    for row in report["times"]:
        if row["t"] == 0.01:
            assert row["compression"]["status"] == UNAVAILABLE
        else:
            assert row["compression"]["status"] == AVAILABLE
    assert_exact_references_survive(report, 3)


def test_raw_positive_zero_time_roundoff_residual_not_forced_to_zero(monkeypatch):
    original = api._unitary_columns

    def propagate(system, t, columns):
        result = original(system, t, columns)
        if len(system["values"]) == 10 and t == 0:
            result = result + 1e-11 * columns
        return result

    monkeypatch.setattr(api, "_unitary_columns", propagate)
    report = api.case_report(3, 0.7)
    zero = report["times"][0]["compression"]
    assert 0 < zero["full_space_error"] < TOL
    assert zero["analytic_upper_bound"] == 0
    comparison(zero["upper_bound_comparison"])
    assert zero["status"] == AVAILABLE


def test_target_lower_bound_compares_full_space_not_projected(monkeypatch, oracles):
    original = api._unitary_columns
    oracle = oracles[(5, 0.7)]
    W, P = oracle["W"], oracle["P"]
    q = np.eye(len(W))[:, np.flatnonzero(np.diag(P) == 0)[0]]
    target = expm_multiply(-0.1j * oracle["delta"], np.eye(10))
    synthetic = W @ target + 0.4 * np.outer(q, np.eye(10)[0])

    def propagate(system, t, columns):
        if len(system["values"]) == 126 and t == 0.1:
            return synthetic.copy()
        return original(system, t, columns)

    monkeypatch.setattr(api, "_unitary_columns", propagate)
    report = api.case_report(5, 0.7)
    record = report["times"][2]["target"]
    same(record["full_space_error"], 0.4)
    same(record["projected_error"], 0)
    assert record["analytic_lower_bound"] > record["projected_error"] + TOL
    comparison(record["lower_bound_comparison"])
    comparison(record["upper_bound_comparison"])
    assert record["status"] == AVAILABLE


def test_bad_bound_observation_is_mismatch_not_numerical_failure(monkeypatch):
    original = api._unitary_columns

    def propagate(system, t, columns):
        result = original(system, t, columns)
        if len(system["values"]) == 10 and t == 0:
            result = result + 1e-6 * columns
        return result

    monkeypatch.setattr(api, "_unitary_columns", propagate)
    report = api.case_report(3, 0.7)
    record = report["times"][0]["compression"]
    assert record["status"] == "diagnostic_mismatch"
    assert record["full_space_error"] > TOL
    comparison(record["upper_bound_comparison"], "mismatch")
    assert report["status"] == "diagnostic_mismatch" and report["reason"]
    assert all(row["compression"]["status"] == AVAILABLE for row in report["times"][1:])


def test_mismatch_precedes_unavailability_in_case_aggregation(monkeypatch):
    original_norm = api._operator_norm

    def norm(matrix):
        value = original_norm(matrix)
        if abs(value - BMAX) < 1e-9:
            return value + 0.01
        return value

    def screen(*args, **kwargs):
        raise api.NumericalUnavailable("injected later unavailable eigensystem")

    monkeypatch.setattr(api, "_operator_norm", norm)
    monkeypatch.setattr(api, "_screened_eigensystem", screen)
    report = api.case_report(5, 0.7)
    comparison(report["target"]["comparison"], "mismatch")
    assert report["status"] == "diagnostic_mismatch"
    assert "target" in report["reason"].lower()
    assert_exact_references_survive(report, 5)


def test_programming_errors_not_converted_to_numerical_unavailability(monkeypatch):
    def bug(*args, **kwargs):
        raise RuntimeError("injected programming error")
    monkeypatch.setattr(api, "_unitary_columns", bug)
    with pytest.raises(RuntimeError, match="injected programming error"):
        api.case_report(3, 0.7)


def test_failed_construction_retains_case_and_analytic_placeholders(monkeypatch):
    def fail(*args, **kwargs):
        raise api.NumericalUnavailable("injected orbit construction failure")
    monkeypatch.setattr(api, "_orbit_data", fail)
    report = api.case_report(5, 0.7)
    assert report["status"] == UNAVAILABLE
    assert report["L"] == report["N"] == 5 and report["g"] == 0.7
    assert report["dimension"] == 126 and report["code_dimension"] == 10
    assert_exact_references_survive(report, 5)
    for row in report["times"]:
        assert_unavailable_dynamics(row["compression"])
        assert_unavailable_dynamics(row["target"])


def test_demo_preserves_all_cases_when_one_model_is_unavailable(monkeypatch):
    original = occupation_module.fixed_number_model

    def model(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["L"] == 4 and bound.arguments["g"] == 0.7:
            raise api.NumericalUnavailable("injected one model construction failure")
        return original(*args, **kwargs)

    patch_callable_aliases(monkeypatch, original, model)
    demo = api.demonstration_report()
    assert [(case["L"], case["g"]) for case in demo["cases"]] == list(GRID)
    assert demo["status"] == UNAVAILABLE
    assert demo["summary"]["case_status_counts"][UNAVAILABLE] == 1
    assert demo["summary"]["case_status_counts"][AVAILABLE] == 8
    assert demo["summary"]["analytic_d5_obstruction_count"] == 3
    for case in demo["cases"]:
        if (case["L"], case["g"]) == (4, 0.7):
            assert_exact_references_survive(case, 4)
            assert case["status"] == UNAVAILABLE
        else:
            assert case["status"] == AVAILABLE


def test_demo_mismatch_status_precedes_unavailable_without_case_substitution(monkeypatch, reports):
    calls = []

    def case_report(L, g):
        calls.append((L, g))
        result = copy.deepcopy(reports[(L, g)])
        if (L, g) == (3, 0.0):
            result["status"] = UNAVAILABLE
            result["reason"] = "injected unavailable"
        if (L, g) == (5, 40.0):
            result["status"] = "diagnostic_mismatch"
            result["reason"] = "injected mismatch"
        return result

    monkeypatch.setattr(api, "case_report", case_report)
    demo = api.demonstration_report()
    assert calls == list(GRID)
    assert demo["status"] == "diagnostic_mismatch"
    counts = demo["summary"]["case_status_counts"]
    assert counts[AVAILABLE] == 7 and counts[UNAVAILABLE] == counts["diagnostic_mismatch"] == 1
    assert len(demo["cases"]) == 9
    native(demo)


def test_componentwise_oracle_acceptance_does_not_hide_small_offdiagonal_in_large_scalar(monkeypatch):
    original = occupation_module.fixed_number_model
    oracle = independent_oracle(4, 40.0)
    rows = [int(np.argmax(oracle["W"][:, column])) for column in (0, 1)]

    def model(*args, **kwargs):
        result = original(*args, **kwargs)
        changed = result.H.copy()
        changed[rows[0], rows[1]] += 1e-8
        changed[rows[1], rows[0]] += 1e-8
        return replace(result, H=changed)

    patch_callable_aliases(monkeypatch, original, model)
    report = api.case_report(4, 40.0)
    # E*=120 permits >2e-8 under an incorrect maximum-channel tolerance.
    # The exact zero offdiagonal instead permits only ATOL=2e-10.
    comparison(report["compression"]["comparison"], "mismatch")
    assert report["status"] == "diagnostic_mismatch"


def test_signed_observed_minimum_is_not_singular_value_or_clipped_zero(monkeypatch):
    original = np.linalg.eigvalsh

    def eigenvalues(H, *args, **kwargs):
        values = original(H, *args, **kwargs)
        values[0] = -1e-8
        return values

    monkeypatch.setattr(np.linalg, "eigvalsh", eigenvalues)
    report = api.case_report(3, 0.7)
    assert report["leakage"]["minimum_gram_eigenvalue"] < -TOL
    assert report["leakage"]["analytic_minimum_gram_eigenvalue"] == 0
    assert report["leakage"]["all_states_leak"] is False
    comparison(report["leakage"]["comparison"], "mismatch")
    assert report["status"] == "diagnostic_mismatch"


def test_report_mutation_does_not_poison_subsequent_case_or_demo(reports):
    baseline = copy.deepcopy(reports[(3, 0.7)])
    first = api.case_report(3, 0.7)
    first["orbit"]["labels"][0][0] = 999
    first["compression"]["matrix"] = ["poisoned"]
    first["leakage"]["analytic_norm"] = -999
    first["times"][0]["compression"]["full_space_error"] = 999
    first["scope"]["local_gauge_emergence"] = True
    second = api.case_report(3, 0.7)
    assert second["orbit"]["labels"] == baseline["orbit"]["labels"]
    same(decode(second["compression"]["matrix"]), decode(baseline["compression"]["matrix"]))
    same(second["leakage"]["analytic_norm"], baseline["leakage"]["analytic_norm"])
    assert second["times"][0]["compression"]["full_space_error"] < TOL
    scope(second["scope"])
    native(second)


@pytest.mark.parametrize("json_mode", [False, True])
def test_demo_stdout_only_modes_using_frozen_report(monkeypatch, capsys, tmp_path, successful_demo, json_mode):
    import runpy
    monkeypatch.setattr(api, "demonstration_report", lambda: copy.deepcopy(successful_demo[0]))
    monkeypatch.setattr(sys, "argv", [str(DEMO)] + (["--json"] if json_mode else []))
    monkeypatch.chdir(tmp_path)
    try:
        runpy.run_path(str(DEMO), run_name="__main__")
    except SystemExit as error:
        assert error.code in (None, 0)
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out.strip()
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        result = json.loads(captured.out)
        assert result == successful_demo[0]
        native(result)
    else:
        text = captured.out.lower()
        assert "heuristic" in text
        assert "empirical" in text


def test_module_and_demo_imports_do_not_execute_models_or_eigensolves(tmp_path):
    # This subprocess is a future test; authoring/static review does not run it.
    code = '''
import importlib.util
import runpy
import numpy as np
from bpr import substrate_fermionization as occupation

def forbidden(*args, **kwargs):
    raise AssertionError("Import executed dense/model work")
occupation.fixed_number_model = forbidden
np.linalg.eigh = forbidden
np.linalg.eigvalsh = forbidden
np.linalg.svd = forbidden
import bpr.substrate_gauge_encoding
runpy.run_path(DEMO_PATH, run_name="import_only")
'''.replace("DEMO_PATH", repr(str(DEMO)))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == ""
    assert list(tmp_path.iterdir()) == []
