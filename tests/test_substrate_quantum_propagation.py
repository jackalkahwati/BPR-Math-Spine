"""Source-free tests of the frozen 2026-09-13 propagation contract.

Authored from the derivation and existing helper contracts, without reading the
new implementation or demo and without executing scientific code. Independent
stars-and-bars hopping, a degree-six operator polynomial, Fourier second
quantization, a small permanent lift, and Decimal positive series are oracles.
Numerical tolerances are regression screens, not certified floating error.
"""
from collections import Counter
import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import inspect
import itertools
import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

from bpr import substrate_fermionization as occupation
from bpr import substrate_gauge_encoding as inherited
from bpr import substrate_quantum_propagation as api


SECTORS = ((3, 3), (4, 4), (5, 5), (8, 2))
COUPLINGS = (0.0, 0.7, 40.0)
TIMES = (0.0, 0.001, 0.01, 0.1)
GRID = tuple((L, N, g) for L, N in SECTORS for g in COUPLINGS)
DIMENSIONS = {(3, 3): 10, (4, 4): 35, (5, 5): 126, (8, 2): 36}
TOL = 2e-10
AVAILABLE = "available_heuristic"
UNAVAILABLE = "numerical_unavailable"
MISMATCH = "diagnostic_mismatch"
AGGREGATE_STATUSES = (AVAILABLE, UNAVAILABLE, MISMATCH)
TAIL_STATUSES = ("structural_zero", "below_stipulated_cap", "trivial_cap", UNAVAILABLE)
CASE_KEYS = {"model_id", "L", "N", "C", "g", "dimension", "hamiltonian_norm_bound",
             "status", "reason", "conventions", "scope", "eigensystem", "times"}
TIME_KEYS = {"t", "tau", "z", "status", "reason", "distances"}
DISTANCE_KEYS = {"x", "y", "distance", "status", "reason", "commutator_norm",
                 "analytic_upper_bound", "bound_evaluation", "signed_excess",
                 "literal_inequality", "comparison"}
TAIL_KEYS = {"status", "value", "partial_sum", "remainder_bound", "terms", "reason"}
EIGEN_KEYS = {"status", "reason", "orthogonality_residual", "eigenpair_residual",
              "tolerance", "scale"}


def same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.real, expected.real, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(actual.imag, expected.imag, atol=TOL, rtol=TOL)


def opnorm(matrix):
    """Independent direct SVD. Never call the aliased production norm oracle."""
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def native(value):
    if type(value) is dict:
        assert all(type(k) is str for k in value)
        for child in value.values():
            native(child)
    elif type(value) is list:
        for child in value:
            native(child)
    else:
        assert value is None or type(value) in (bool, int, float, str)
        if type(value) is float:
            assert math.isfinite(value)


def strict_json(value):
    native(value)
    text = json.dumps(value, allow_nan=False)
    assert json.loads(text) == value


def norm_bound(N, g):
    return 2 * N + g * N * (N - 1) / 2


def observations(report):
    return [obs for row in report["times"] for obs in row["distances"]]


def check_scope(scope):
    assert scope["finite_sector_only"] is True
    for key in ("site_tensor_product_claim", "thermodynamic_velocity",
                "relativistic_spacetime", "numerical_error_certified", "empirical_validation"):
        assert scope[key] is False
    assert scope["empirical_status"] == "empirical_test_unavailable"


def check_available_reason(record):
    assert record["status"] == AVAILABLE
    reason = record["reason"]
    # Heuristic provenance and literal-inequality explanations are permitted
    # on available records; availability does not require silence about them.
    assert reason is None or (type(reason) is str and reason.strip())


def check_tail_schema(record):
    assert set(record) == TAIL_KEYS
    assert record["status"] in TAIL_STATUSES
    assert type(record["terms"]) is int and 0 <= record["terms"] <= 256
    for name in ("value", "partial_sum", "remainder_bound"):
        if record[name] is not None:
            assert type(record[name]) in (int, float)
            assert math.isfinite(record[name]) and record[name] >= 0
    if record["status"] == UNAVAILABLE:
        assert record["value"] is None
        assert isinstance(record["reason"], str) and record["reason"]
    else:
        assert 0 <= record["value"] <= 2
    strict_json(record)


def check_comparison(obs):
    comparison = obs["comparison"]
    assert set(comparison) == {"status", "max_absolute_error", "reason"}
    observed, bound = obs["commutator_norm"], obs["analytic_upper_bound"]
    if observed is None or bound is None:
        assert comparison["status"] == "inconclusive"
        assert comparison["max_absolute_error"] is None
        assert obs["signed_excess"] is None and obs["literal_inequality"] is None
        assert comparison["reason"]
    else:
        excess = observed - bound
        assert obs["signed_excess"] == excess
        assert obs["literal_inequality"] is (observed <= bound)
        assert comparison["max_absolute_error"] == abs(excess)
        consistent = excess <= TOL + TOL * abs(bound)
        assert comparison["status"] == ("consistent" if consistent else "mismatch")
        if excess > 0 and consistent:
            text = comparison["reason"].lower()
            assert "literal" in text and "inequality" in text
            assert "regression" in text and "consisten" in text


def independent_bose(L, N, g):
    """Actual stars-and-bars positions, not inherited occupation enumeration."""
    length = L + N - 1
    states = []
    for bars in itertools.combinations(range(length), L - 1):
        endpoints = (-1,) + bars + (length,)
        states.append(tuple(endpoints[j + 1] - endpoints[j] - 1 for j in range(L)))
    basis = tuple(sorted(states))
    dimension = math.comb(length, N)
    assert len(basis) == len(set(basis)) == dimension == DIMENSIONS[L, N]
    index = {state: i for i, state in enumerate(basis)}
    H = np.zeros((dimension, dimension), dtype=float)
    for column, state in enumerate(basis):
        H[column, column] = g * sum(n * (n - 1) for n in state) / 2
        for source in range(L):
            if not state[source]:
                continue
            for target in ((source + 1) % L, (source - 1) % L):
                moved = list(state)
                moved[source] -= 1
                moved[target] += 1
                H[index[tuple(moved)], column] -= math.sqrt(state[source] * (state[target] + 1))
    density = np.asarray(basis, dtype=float) / N
    return {"basis": basis, "index": index, "H": H, "density": density, "N": N, "L": L}


def commutator_diagonal(matrix, diagonal):
    return matrix * (diagonal[None, :] - diagonal[:, None])


def lift_one_body(matrix, oracle):
    """Independent dGamma, including correctly normalized diagonal a†a terms."""
    basis, index = oracle["basis"], oracle["index"]
    result = np.zeros((len(basis), len(basis)), dtype=complex)
    for column, state in enumerate(basis):
        for source, count in enumerate(state):
            if not count:
                continue
            result[column, column] += matrix[source, source] * count
            for target in range(oracle["L"]):
                if target == source:
                    continue
                moved = list(state)
                moved[source] -= 1
                moved[target] += 1
                result[index[tuple(moved)], column] += (
                    matrix[target, source] * math.sqrt(count * (state[target] + 1)))
    return result


def fourier_unitary(L, t):
    """Direct finite scalar Fourier sums, not a many-body exponential/eigh."""
    U = np.empty((L, L), dtype=complex)
    for a in range(L):
        for b in range(L):
            U[a, b] = sum(
                np.exp(2j * math.pi * k * (a - b) / L)
                * np.exp(2j * t * math.cos(2 * math.pi * k / L))
                for k in range(L)) / L
    return U


def permanent(matrix):
    return sum(math.prod(matrix[row, column] for row, column in enumerate(perm))
               for perm in itertools.permutations(range(len(matrix))))


def permanent_lift(U, basis):
    """Fock-state amplitudes per(U[n,m])/sqrt(prod(n!)prod(m!))."""
    expanded = [tuple(x for x, n in enumerate(state) for _ in range(n)) for state in basis]
    normalizers = [math.prod(math.factorial(n) for n in state) for state in basis]
    result = np.empty((len(basis), len(basis)), dtype=complex)
    for row, rows in enumerate(expanded):
        for column, columns in enumerate(expanded):
            result[row, column] = permanent(U[np.ix_(rows, columns)]) / math.sqrt(
                normalizers[row] * normalizers[column])
    return result


def decimal_tail(z, distance):
    """Independent high-precision positive infinite series, no cancellation."""
    with localcontext() as context:
        context.prec = 110
        x = Decimal.from_float(float(z))
        if not x:
            return Decimal(0)
        if x >= distance:
            return Decimal(2)
        term = x ** distance / Decimal(math.factorial(distance))
        total = term
        order = distance
        for _ in range(4096):
            order += 1
            term = term * x / order
            total += term
            if total >= 1:
                return Decimal(2)
            if term <= total * Decimal("1e-100"):
                return 2 * total
        raise AssertionError("Independent Decimal oracle did not converge")


def patch_builder(monkeypatch, replacement):
    """Locate only aliases of the unchanged documented inherited builder."""
    original = occupation.fixed_number_model
    for module in (api, occupation):
        for name, value in tuple(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, name, replacement)


@pytest.fixture(scope="module")
def oracles():
    return {key: independent_bose(*key) for key in GRID}


@pytest.fixture(scope="module")
def successful_demo():
    """One instrumented successful demo; retain only required bounded matrices."""
    calls = {"model": [], "screen": [], "eigh": [], "evolution": [], "unitary": [], "norm": []}
    matrices = {}
    original_builder = occupation.fixed_number_model
    original_screen = api._screened_eigensystem
    original_evolve = api._evolved_density
    original_unitary = api._unitary_columns
    original_norm = api._operator_norm
    original_eigh = np.linalg.eigh
    system_keys = {}
    current_key = [None]

    def model(*args, **kwargs):
        bound = inspect.signature(original_builder).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        key = (int(arguments["L"]), int(arguments["N"]), float(arguments["g"]))
        current_key[0] = key
        result = original_builder(*args, **kwargs)
        calls["model"].append((key, float(arguments["C"]), result.basis, result.H.copy()))
        return result

    def screen(H):
        calls["screen"].append((current_key[0], np.array(H, copy=True)))
        result = original_screen(H)
        system_keys[id(result)] = current_key[0]
        return result

    def eigh(H, *args, **kwargs):
        calls["eigh"].append(tuple(H.shape))
        return original_eigh(H, *args, **kwargs)

    def unitary(system, t, columns):
        assert columns.ndim == 2 and columns.shape[0] == columns.shape[1] <= 512
        np.testing.assert_array_equal(columns, np.eye(len(columns)))
        calls["unitary"].append((system_keys[id(system)], float(t)))
        return original_unitary(system, t, columns)

    def evolve(system, t, diagonal):
        key = system_keys[id(system)]
        calls["evolution"].append((key, float(t)))
        result = original_evolve(system, t, diagonal)
        assert result.ndim == 2 and result.shape == (DIMENSIONS[key[:2]],) * 2
        assert result.dtype.kind == "c" and result.flags.owndata
        assert np.all(np.isfinite(result))
        if t in (0.001, 0.01) or key[2] == 0:
            matrices[key, float(t)] = result.copy()
        return result

    def norm(matrix):
        calls["norm"].append(tuple(matrix.shape))
        assert matrix.ndim == 2 and max(matrix.shape) <= 512
        return original_norm(matrix)

    def forbidden(*args, **kwargs):
        raise AssertionError("Propagation must not construct a binary model or use a ground/gap gate")

    with pytest.MonkeyPatch.context() as patch:
        patch_builder(patch, model)
        patch.setattr(api, "_screened_eigensystem", screen)
        patch.setattr(api, "_evolved_density", evolve)
        patch.setattr(api, "_unitary_columns", unitary)
        patch.setattr(api, "_operator_norm", norm)
        patch.setattr(np.linalg, "eigh", eigh)
        patch.setattr(np.linalg, "eigvalsh", forbidden)
        patch.setattr(occupation, "binary_operators", forbidden)
        patch.setattr(np, "kron", forbidden)
        report = api.demonstration_report()
    return {"report": report, "calls": calls, "matrices": matrices}


@pytest.fixture(scope="module")
def reports(successful_demo):
    return {(case["L"], case["N"], case["g"]): case for case in successful_demo["report"]["cases"]}


def test_frozen_constants_aliases_and_exact_successful_work_budget(successful_demo, oracles):
    assert tuple(api.SECTORS) == SECTORS
    assert tuple(api.COUPLINGS) == COUPLINGS
    assert tuple(api.TIMES) == TIMES
    assert api.C == 1.0 and api.MAX_DIMENSION == 512
    assert api.ATOL == api.RTOL == TOL
    for name in ("_screened_eigensystem", "_unitary_columns", "_operator_norm", "NumericalUnavailable"):
        assert getattr(api, name) is getattr(inherited, name)
    calls = successful_demo["calls"]
    assert len(calls["model"]) == len(calls["screen"]) == len(calls["eigh"]) == 12
    assert [entry[0] for entry in calls["model"]] == list(GRID)
    expected_work = [(key, t) for key in GRID for t in TIMES]
    assert calls["evolution"] == calls["unitary"] == expected_work
    assert len(calls["norm"]) == 108
    assert Counter(calls["norm"]) == Counter({(10, 10): 12, (35, 35): 24,
                                             (126, 126): 24, (36, 36): 48})
    for key, C, basis, H in calls["model"]:
        oracle = oracles[key]
        assert C == 1 and basis == oracle["basis"]
        same(H, oracle["H"])
        assert max(max(state) for state in basis) == key[1]
    for key, H in calls["screen"]:
        same(H, oracles[key]["H"])
    assert Counter(calls["eigh"]) == Counter({(10, 10): 3, (35, 35): 3,
                                             (126, 126): 3, (36, 36): 3})


def test_exact_12_cases_48_times_108_observations_and_strict_schema(successful_demo):
    demo = successful_demo["report"]
    assert set(demo) == {"model_id", "status", "reason", "scope", "controls", "cases", "summary"}
    assert demo["model_id"] == "substrate_quantum_propagation"
    check_available_reason(demo)
    assert demo["controls"] == {"sectors": [list(sector) for sector in SECTORS],
                                "couplings": list(COUPLINGS), "times": list(TIMES),
                                "C": 1.0, "reference_site": 0}
    assert len(demo["cases"]) == 12
    assert sum(len(case["times"]) for case in demo["cases"]) == 48
    assert sum(len(observations(case)) for case in demo["cases"]) == 108
    assert [(c["L"], c["N"], c["g"]) for c in demo["cases"]] == list(GRID)
    check_scope(demo["scope"])
    counts = {"case_status_counts": Counter(), "observation_status_counts": Counter(),
              "bound_status_counts": Counter()}
    for case in demo["cases"]:
        assert set(case) == CASE_KEYS
        assert case["model_id"] == demo["model_id"]
        L, N, g = case["L"], case["N"], case["g"]
        assert case["C"] == 1 and case["dimension"] == DIMENSIONS[L, N]
        assert case["hamiltonian_norm_bound"] == norm_bound(N, g)
        check_available_reason(case)
        check_scope(case["scope"])
        assert case["conventions"]
        metadata = case["eigensystem"]
        assert set(metadata) == EIGEN_KEYS
        check_available_reason(metadata)
        assert 0 <= metadata["orthogonality_residual"] <= metadata["tolerance"]
        assert 0 <= metadata["eigenpair_residual"] <= metadata["tolerance"] * metadata["scale"]
        assert [row["t"] for row in case["times"]] == list(TIMES)
        counts["case_status_counts"][case["status"]] += 1
        for row in case["times"]:
            assert set(row) == TIME_KEYS and row["t"] == row["tau"]
            assert row["z"] == 2 * norm_bound(N, g) * abs(row["t"])
            check_available_reason(row)
            assert [obs["distance"] for obs in row["distances"]] == list(range(1, L // 2 + 1))
            for obs in row["distances"]:
                assert set(obs) == DISTANCE_KEYS
                assert obs["x"] == 0 and obs["y"] == obs["distance"]
                check_available_reason(obs)
                assert 0 <= obs["commutator_norm"] <= 0.5 + TOL
                check_tail_schema(obs["bound_evaluation"])
                assert obs["analytic_upper_bound"] == obs["bound_evaluation"]["value"]
                assert obs["bound_evaluation"] == api._capped_tail(row["z"], obs["distance"])
                check_comparison(obs)
                counts["observation_status_counts"][obs["status"]] += 1
                counts["bound_status_counts"][obs["bound_evaluation"]["status"]] += 1
    assert set(demo["summary"]) == set(counts)
    for field, observed in counts.items():
        statuses = TAIL_STATUSES if field == "bound_status_counts" else AGGREGATE_STATUSES
        assert demo["summary"][field] == {status: observed[status] for status in statuses}
    strict_json(demo)


@pytest.mark.parametrize("key", GRID)
def test_degree_six_locality_and_operator_norm_integral_remainder(key, oracles, successful_demo, reports):
    oracle = oracles[key]
    H, densities = oracle["H"], oracle["density"]
    L, N, g = key
    nested = [np.diag(densities[:, 0]).astype(complex)]
    for _ in range(6):
        nested.append(H @ nested[-1] - nested[-1] @ H)
    for distance in range(1, L // 2 + 1):
        for order in range(distance):
            cancellation = commutator_diagonal(nested[order], densities[:, distance])
            # The exact support theorem predicts zero; independently assembled
            # floating nested products are only a regression check of that zero.
            assert opnorm(cancellation) <= TOL
    # Independent analytic adjacent-bond coefficient fixes the +it convention.
    bond = np.zeros((L, L))
    bond[0, 1] = bond[1, 0] = 1
    same(commutator_diagonal(nested[1], densities[:, 1]), lift_one_body(bond, oracle) / N ** 2)
    for t in (0.001, 0.01):
        evolved = successful_demo["matrices"][key, t]
        polynomial = sum((1j * t) ** order / math.factorial(order) * matrix
                         for order, matrix in enumerate(nested))
        remainder = 2 * (2 * norm_bound(N, g) * abs(t)) ** 7 / math.factorial(7)
        row = next(row for row in reports[key]["times"] if row["t"] == t)
        for obs in row["distances"]:
            diagonal = densities[:, obs["distance"]]
            actual = commutator_diagonal(evolved, diagonal)
            truncated = commutator_diagonal(polynomial, diagonal)
            # The error of operators, not the difference of their norms and not
            # componentwise equality at 2e-10. Large frozen bounds stay large.
            error_norm = opnorm(actual - truncated)
            assert error_norm <= remainder + TOL + TOL * abs(remainder)
            same(obs["commutator_norm"], opnorm(actual))
        if key == (5, 5, 40.0) and t == 0.01:
            assert remainder > 1, "Do not tighten or clip the loose integral remainder"


@pytest.mark.parametrize("L,N", SECTORS)
def test_free_fourier_lifted_matrices_and_scalar_norm_all_36_rows(L, N, oracles, successful_demo, reports):
    key = (L, N, 0.0)
    oracle = oracles[key]
    records = 0
    for row in reports[key]["times"]:
        t = row["t"]
        U = fourier_unitary(L, t)
        evolved_projector = np.outer(U[0].conj(), U[0])
        expected_density = lift_one_body(evolved_projector, oracle) / N
        actual_density = successful_demo["matrices"][key, t]
        same(actual_density, expected_density)
        for obs in row["distances"]:
            y = obs["distance"]
            P_y = np.zeros(L)
            P_y[y] = 1
            one_commutator = commutator_diagonal(evolved_projector, P_y)
            expected = lift_one_body(one_commutator, oracle) / N ** 2
            actual = commutator_diagonal(actual_density, oracle["density"][:, y])
            same(actual, expected)
            s = abs(U[0, y])
            assert 0 <= s <= 1
            # Do not clip 1-s*s. A failed range check must remain visible.
            scalar = s * math.sqrt(1 - s * s) / N
            same(opnorm(expected), scalar)
            same(obs["commutator_norm"], scalar)
            records += 1
    assert records == 4 * (L // 2)
    assert sum(4 * (size // 2) for size, _ in SECTORS) == 36


def test_small_three_boson_permanent_lift_normalization_and_evolution(oracles, successful_demo):
    key, t = (3, 3, 0.0), 0.01
    oracle = oracles[key]
    U = fourier_unitary(3, t)
    full = permanent_lift(U, oracle["basis"])
    same(full.conj().T @ full, np.eye(10))
    same(full, expm(-1j * t * oracle["H"]))
    normalized = oracle["density"][:, 0]
    expected = full.conj().T @ (normalized[:, None] * full)
    same(successful_demo["matrices"][key, t], expected)
    all_first = oracle["index"][(3, 0, 0)]
    one_each = oracle["index"][(1, 1, 1)]
    same(full[one_each, all_first], math.sqrt(6) * U[0, 0] * U[1, 0] * U[2, 0])


@pytest.mark.parametrize("H", [np.array([[2, 1j], [-1j, -1]], dtype=complex),
                                np.zeros((2, 2)), np.eye(2) * 3])
def test_evolved_density_complex_sign_degeneracy_and_owned_arrays(H, monkeypatch):
    before = H.copy()
    system = api._screened_eigensystem(H)
    system_arrays = {key: value for key, value in system.items() if isinstance(value, np.ndarray)}
    saved = {key: value.copy() for key, value in system_arrays.items()}
    diagonal = np.array([0.2, 0.9])
    untouched = diagonal.copy()
    original = api._unitary_columns
    calls = []

    def unitary(screened, t, columns):
        calls.append(t)
        np.testing.assert_array_equal(columns, np.eye(2))
        return original(screened, t, columns)

    monkeypatch.setattr(api, "_unitary_columns", unitary)
    outputs = []
    for t in (0.0, 0.1):
        result = api._evolved_density(system, t, diagonal)
        U = expm(-1j * t * H)
        same(result, U.conj().T @ np.diag(diagonal) @ U)
        assert result.flags.owndata and result.dtype.kind == "c"
        assert not np.shares_memory(result, diagonal)
        assert all(not np.shares_memory(result, value) for value in system_arrays.values())
        outputs.append(result)
        if t and H[0, 1].imag:
            wrong = U @ np.diag(diagonal) @ U.conj().T
            assert opnorm(result - wrong) > 1e-3
    assert calls == [0.0, 0.1]
    assert not np.shares_memory(*outputs)
    np.testing.assert_array_equal(H, before)
    np.testing.assert_array_equal(diagonal, untouched)
    for key, value in saved.items():
        np.testing.assert_array_equal(system[key], value)
    diagonal[:] = 0
    for value in system.values():
        if isinstance(value, np.ndarray):
            value[:] = 0
    same(outputs[0], np.diag(untouched))


def test_private_diagonal_is_dimension_bounded_not_unit_interval_bounded():
    # Clarified frozen seam: positivity belongs to the public nx/N observable,
    # not to private finite-real diagonal validation.
    H = np.array([[0, 1j], [-1j, 2]], dtype=complex)
    system = api._screened_eigensystem(H)
    diagonal = np.array([-3.0, 7.0])
    U = expm(-0.01j * H)
    same(api._evolved_density(system, 0.01, diagonal), U.conj().T @ np.diag(diagonal) @ U)


def test_zero_spectrum_with_rotated_complex_degenerate_basis(monkeypatch):
    rotated = np.array([[1, 1j], [1j, 1]], dtype=complex) / math.sqrt(2)
    monkeypatch.setattr(np.linalg, "eigh", lambda matrix: (np.zeros(2), rotated.copy()))
    system = api._screened_eigensystem(np.zeros((2, 2)))
    for t in (0.0, 0.1):
        same(api._evolved_density(system, t, [0.25, 0.75]), np.diag([0.25, 0.75]))


class ConversionTrap:
    def __float__(self):
        raise AssertionError("Custom conversion must not be invoked")

    def __int__(self):
        raise AssertionError("Custom conversion must not be invoked")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Custom conversion must not be invoked")


@pytest.mark.parametrize("L,N,g", [
    (True, 3, 0.7), (np.bool_(True), 3, 0.7), (3.0, 3, 0.7), ("3", 3, 0.7),
    (3 + 0j, 3, 0.7), (ConversionTrap(), 3, 0.7), (3, True, 0.7),
    (3, np.bool_(False), 0.7), (3, 3.0, 0.7), (3, "3", 0.7), (3, 3 + 0j, 0.7),
    (3, ConversionTrap(), 0.7), (3, 0, 0.7), (3, -1, 0.7), (3, 2, 0.7),
    (8, 8, 0.7), (9, 9, 0.7), (10 ** 50, 3, 0.7), (3, 10 ** 50, 0.7),
    (3, 3, True), (3, 3, np.bool_(False)), (3, 3, "0.7"), (3, 3, 0.7 + 0j),
    (3, 3, np.complex128(40)), (3, 3, float("nan")), (3, 3, float("inf")),
    (3, 3, -float("inf")), (3, 3, -0.7), (3, 3, 1.0), (3, 3, np.float32(0.7)),
    (3, 3, np.nextafter(0.7, 1)), (3, 3, np.nextafter(40.0, 0)),
    (3, 3, np.array(0.7)), (3, 3, [0.7]), (3, 3, Fraction(7, 10)),
    (3, 3, Decimal("0.7")), (3, 3, ConversionTrap()),
])
def test_public_strict_validation_before_any_allocation(L, N, g, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid public input reached model allocation")
    patch_builder(monkeypatch, forbidden)
    monkeypatch.setattr(api, "_screened_eigensystem", forbidden)
    with pytest.raises((TypeError, ValueError)) as error:
        api.case_report(L, N, g)
    assert not isinstance(error.value, api.NumericalUnavailable)


@pytest.mark.parametrize("extra", [{"C": 2}, {"cache": {}}, {"system": object()},
                                    {"times": (0,)}, {"max_dimension": 1024}])
def test_no_public_cache_or_control_search(extra):
    with pytest.raises(TypeError):
        api.case_report(3, 3, 0.7, **extra)


def test_extended_precision_stored_value_membership_precedes_float_rounding(monkeypatch):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("No floating dtype wider than binary64 on this platform")
    wider = np.nextafter(np.longdouble(0.7), np.longdouble(1))
    assert float(wider) == 0.7 and wider.as_integer_ratio() != (0.7).as_integer_ratio()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unequal stored coupling reached builder")
    patch_builder(monkeypatch, forbidden)
    with pytest.raises((TypeError, ValueError)):
        api.case_report(3, 3, wider)


@pytest.mark.parametrize("L,N,g", [(np.int32(3), np.int64(3), np.float64(0.7)),
                                    (np.uint8(3), np.int16(3), np.float32(40)),
                                    (3, 3, np.float32(0)), (3, 3, np.longdouble(0.7)),
                                    (3, 3, 40), (3, 3, -0.0)])
def test_exact_stored_numpy_and_builtin_members_are_accepted(L, N, g, monkeypatch):
    # Validation only: inject a named model failure rather than solve each alias.
    called = []

    def stop(*args, **kwargs):
        called.append((args, kwargs))
        raise api.NumericalUnavailable("accepted-input-validation-probe")
    patch_builder(monkeypatch, stop)
    report = api.case_report(L, N, g)
    assert len(called) == 1 and report["status"] == UNAVAILABLE
    assert (report["L"], report["N"], report["g"]) == (int(L), int(N), float(g))
    strict_json(report)


def test_complete_dimension_guard_precedes_inherited_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dimension cap must be checked before inherited construction")
    monkeypatch.setattr(api, "MAX_DIMENSION", 9)
    patch_builder(monkeypatch, forbidden)
    with pytest.raises(ValueError):
        api.case_report(3, 3, 0.7)


@pytest.mark.parametrize("bad", [[], [0.2], [0.2, 0.4, 0.6], [[0.2, 0.4]],
                                [True, False], [0.2, False], ["0.2", "0.4"],
                                [0.2 + 0j, 0.4], [np.nan, 0], [np.inf, 0],
                                np.array([0.2, 0.4], dtype=object), ConversionTrap()])
def test_evolved_density_diagonal_validation_before_unitary(bad, monkeypatch):
    system = api._screened_eigensystem(np.diag([0.0, 1.0]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid diagonal reached unitary propagation")
    monkeypatch.setattr(api, "_unitary_columns", forbidden)
    with pytest.raises((TypeError, ValueError)) as error:
        api._evolved_density(system, 0.01, bad)
    assert not isinstance(error.value, api.NumericalUnavailable)


@pytest.mark.parametrize("bad", [True, np.bool_(True), "0.1", 0.1 + 0j,
                                np.nan, np.inf, ConversionTrap(), np.array(0.1)])
def test_evolved_density_time_validation(bad):
    system = api._screened_eigensystem(np.zeros((2, 2)))
    with pytest.raises((TypeError, ValueError)):
        api._evolved_density(system, bad, [0.2, 0.8])


@pytest.mark.parametrize("z,distance", [(True, 1), (np.bool_(False), 1), ("1", 1),
                                        (1 + 0j, 1), (np.array(0.1), 1),
                                        (Fraction(1, 10), 1), (Decimal("0.1"), 1),
                                        (ConversionTrap(), 1), (-1, 1), (np.nan, 1),
                                        (np.inf, 1), (-np.inf, 1), (0.1, True),
                                        (0.1, np.bool_(True)), (0.1, 1.0), (0.1, "1"),
                                        (0.1, 0), (0.1, 5), (0.1, -1),
                                        (0.1, 1 + 0j), (0.1, ConversionTrap())])
def test_tail_strict_types_and_domain(z, distance):
    with pytest.raises((TypeError, ValueError)) as error:
        api._capped_tail(z, distance)
    assert not isinstance(error.value, api.NumericalUnavailable)


@pytest.mark.parametrize("distance", (1, 2, 3, 4))
@pytest.mark.parametrize("z", (0.0, -0.0, np.float32(0), np.float64(0)))
def test_tail_structural_zero_exact_schema(z, distance):
    result = api._capped_tail(z, distance)
    check_tail_schema(result)
    assert result["status"] == "structural_zero"
    assert result["value"] == result["partial_sum"] == result["remainder_bound"] == 0
    assert result["terms"] == 0


@pytest.mark.parametrize("distance", (1, 2, 3, 4))
@pytest.mark.parametrize("z", (1e-50, 1e-8, 0.001, 0.01, 0.1, 0.5, 0.9, 1.5, 2.5, 3.9))
def test_tail_independent_high_precision_positive_scalar_oracle(z, distance):
    result = api._capped_tail(z, distance)
    check_tail_schema(result)
    assert result["status"] != UNAVAILABLE
    expected = decimal_tail(z, distance)
    actual = Decimal.from_float(float(result["value"]))
    # No absolute 2e-10 allowance that would hide errors in tiny tails.
    assert abs(actual - expected) <= abs(expected) * Decimal("2e-14")
    if z >= distance:
        assert result["status"] == "trivial_cap" and result["terms"] == 0
        assert result["partial_sum"] is result["remainder_bound"] is None
        return
    terms = result["terms"]
    assert 1 <= terms <= 256
    with localcontext() as context:
        context.prec = 110
        x = Decimal.from_float(z)
        partial = sum(x ** r / Decimal(math.factorial(r))
                      for r in range(distance, distance + terms))
        recorded_partial = Decimal.from_float(float(result["partial_sum"]))
        assert abs(recorded_partial - partial) <= partial * Decimal("2e-14")
        if result["remainder_bound"] is None:
            assert result["partial_sum"] >= 1 and result["value"] == 2
            assert result["reason"] == "partial sum reaches stipulated cap"
        else:
            m = distance + terms - 1
            remainder = (x ** (m + 1) / Decimal(math.factorial(m + 1))) / (1 - x / (m + 2))
            recorded_remainder = Decimal.from_float(float(result["remainder_bound"]))
            assert abs(recorded_remainder - remainder) <= remainder * Decimal("2e-14")
            assert result["remainder_bound"] <= 8 * np.finfo(float).eps * result["partial_sum"]
            assert result["value"] == min(2, 2 * (result["partial_sum"] + result["remainder_bound"]))
            assert result["status"] == ("trivial_cap" if result["value"] == 2 else "below_stipulated_cap")
            if terms > 1:
                prior_m = m - 1
                prior_s = partial - x ** m / Decimal(math.factorial(m))
                prior_R = (x ** (prior_m + 1) / Decimal(math.factorial(prior_m + 1))) / (1 - x / (prior_m + 2))
                assert prior_R > Decimal(8) * Decimal.from_float(np.finfo(float).eps) * prior_s


@pytest.mark.parametrize("distance", (1, 2, 3, 4))
def test_tail_saturation_is_safe_before_powers_or_exponentials(distance, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Capped tail must not evaluate an exponential")
    for module in (math, np):
        monkeypatch.setattr(module, "exp", forbidden)
        monkeypatch.setattr(module, "expm1", forbidden)
    for z in (float(distance), np.float64(distance), np.finfo(float).max):
        result = api._capped_tail(z, np.int64(distance))
        check_tail_schema(result)
        assert result["status"] == "trivial_cap" and result["value"] == 2
        assert result["terms"] == 0
        assert result["partial_sum"] is result["remainder_bound"] is None
        assert result["reason"]
    result = api._capped_tail(0.01, distance)
    assert result["status"] == "below_stipulated_cap"


@pytest.mark.parametrize("z,distance,terms", [(np.nextafter(0.0, 1.0), 1, 1),
                                              (np.nextafter(0.0, 1.0), 2, 0),
                                              (1e-200, 2, 0), (1e-155, 2, 1),
                                              (1e-75, 4, 1)])
def test_tail_lost_positive_intermediates_are_unavailable_not_zero(z, distance, terms):
    result = api._capped_tail(z, distance)
    check_tail_schema(result)
    assert result["status"] == UNAVAILABLE and result["value"] is None
    assert result["terms"] == terms
    assert result["remainder_bound"] is None
    if terms:
        assert result["partial_sum"] > 0


@pytest.mark.parametrize("z", (1e-77, 1e-80))
def test_tail_retains_representable_positive_subnormal_intermediates(z):
    result = api._capped_tail(z, 3)
    check_tail_schema(result)
    assert result["status"] == "below_stipulated_cap" and result["terms"] == 1
    assert 0 < result["remainder_bound"] < np.finfo(float).tiny
    expected = decimal_tail(z, 3)
    actual = Decimal.from_float(float(result["value"]))
    assert abs(actual - expected) <= expected * Decimal("2e-14")


def test_tail_valid_extended_precision_conversion_underflow_and_overflow():
    if np.finfo(np.longdouble).maxexp <= np.finfo(float).maxexp:
        pytest.skip("No wider floating exponent range on this platform")
    tiny = np.longdouble(np.finfo(float).smallest_subnormal) / np.longdouble(2)
    huge = np.longdouble(np.finfo(float).max) * np.longdouble(2)
    assert tiny > 0 and float(tiny) == 0 and np.isfinite(huge)
    for z in (tiny, huge):
        result = api._capped_tail(z, 1)
        check_tail_schema(result)
        assert result["status"] == UNAVAILABLE and result["terms"] == 0
        assert result["partial_sum"] is result["remainder_bound"] is None


def assert_preserved_placeholders(report, observed_available=False, tail_available=True):
    L, N, g = report["L"], report["N"], report["g"]
    assert set(report) == CASE_KEYS and set(report["eigensystem"]) == EIGEN_KEYS
    assert report["dimension"] == DIMENSIONS[L, N]
    assert report["hamiltonian_norm_bound"] == norm_bound(N, g)
    assert [row["t"] for row in report["times"]] == list(TIMES)
    for row in report["times"]:
        assert set(row) == TIME_KEYS
        assert row["tau"] == row["t"] and row["z"] == 2 * norm_bound(N, g) * row["t"]
        assert [obs["distance"] for obs in row["distances"]] == list(range(1, L // 2 + 1))
        for obs in row["distances"]:
            assert set(obs) == DISTANCE_KEYS
            assert (obs["commutator_norm"] is not None) is observed_available
            check_tail_schema(obs["bound_evaluation"])
            if tail_available:
                assert obs["bound_evaluation"] == api._capped_tail(row["z"], obs["distance"])
            assert obs["analytic_upper_bound"] == obs["bound_evaluation"]["value"]
            check_comparison(obs)
    strict_json(report)


@pytest.mark.parametrize("boundary", ("model", "eigen"))
def test_model_or_eigen_failure_keeps_all_bounds_and_placeholders(boundary, monkeypatch):
    message = "injected-" + boundary + "-failure"

    def failure(*args, **kwargs):
        raise api.NumericalUnavailable(message)

    def forbidden(*args, **kwargs):
        raise AssertionError("Failed dependency must not propagate an invented eigensystem")
    if boundary == "model":
        patch_builder(monkeypatch, failure)
        monkeypatch.setattr(api, "_screened_eigensystem", forbidden)
    else:
        monkeypatch.setattr(api, "_screened_eigensystem", failure)
    monkeypatch.setattr(api, "_evolved_density", forbidden)
    report = api.case_report(4, 4, 0.7)
    assert report["status"] == report["eigensystem"]["status"] == UNAVAILABLE
    assert message in report["reason"]
    assert_preserved_placeholders(report)
    assert all(row["status"] == UNAVAILABLE for row in report["times"])
    assert all(obs["status"] == UNAVAILABLE for obs in observations(report))


def test_inherited_builder_only_numerically_unresolved_valueerror_is_translated(monkeypatch):
    def failure(*args, **kwargs):
        raise ValueError("numerically unresolved inherited arithmetic fixture")
    patch_builder(monkeypatch, failure)
    report = api.case_report(3, 3, 0.7)
    assert report["status"] == UNAVAILABLE
    assert_preserved_placeholders(report)


def test_solver_arithmetic_failure_has_complete_unavailable_report(monkeypatch):
    def failure(*args, **kwargs):
        raise np.linalg.LinAlgError("injected solver failure")
    monkeypatch.setattr(np.linalg, "eigh", failure)
    report = api.case_report(3, 3, 0.7)
    assert report["status"] == UNAVAILABLE
    assert_preserved_placeholders(report)


def test_time_failure_does_not_suppress_later_times_or_independent_bounds(monkeypatch):
    original = api._evolved_density
    called = []

    def evolve(system, t, diagonal):
        called.append(t)
        if t == 0.001:
            raise api.NumericalUnavailable("injected-time-failure")
        return original(system, t, diagonal)
    monkeypatch.setattr(api, "_evolved_density", evolve)
    report = api.case_report(4, 4, 0.7)
    assert called == list(TIMES)
    assert report["status"] == UNAVAILABLE
    assert "injected-time-failure" in report["reason"]
    for row in report["times"]:
        failed = row["t"] == 0.001
        assert row["status"] == (UNAVAILABLE if failed else AVAILABLE)
        for obs in row["distances"]:
            assert (obs["commutator_norm"] is None) is failed
            assert obs["analytic_upper_bound"] is not None
            assert obs["bound_evaluation"]["status"] != UNAVAILABLE
            assert obs["status"] == row["status"]
            check_comparison(obs)
    strict_json(report)


def test_distance_norm_failure_is_local_and_next_distance_still_runs(monkeypatch):
    original = api._operator_norm
    called = []

    def norm(matrix):
        called.append(matrix.shape)
        if len(called) == 1:
            raise api.NumericalUnavailable("injected-first-distance-norm")
        return original(matrix)
    monkeypatch.setattr(api, "_operator_norm", norm)
    report = api.case_report(4, 4, 0.7)
    assert len(called) == 8
    rows = observations(report)
    assert rows[0]["status"] == UNAVAILABLE and rows[0]["commutator_norm"] is None
    assert rows[0]["analytic_upper_bound"] == 0
    assert all(obs["status"] == AVAILABLE and obs["commutator_norm"] is not None for obs in rows[1:])
    assert report["status"] == report["times"][0]["status"] == UNAVAILABLE
    for obs in rows:
        check_comparison(obs)
    strict_json(report)


def test_commutator_arithmetic_failure_is_local_to_one_distance(monkeypatch):
    original = api._evolved_density
    attempts = []

    # Injection-only probe of the frozen A_ij*(b_j-b_i) route, not a public
    # requirement to preserve caller ndarray subclasses.
    class ArithmeticProbe(np.ndarray):
        def __mul__(self, other):
            attempts.append(tuple(self.shape))
            if len(attempts) == 1:
                raise FloatingPointError("injected-first-commutator-arithmetic")
            return np.asarray(self) * other

    def evolve(system, t, diagonal):
        result = original(system, t, diagonal)
        return result.view(ArithmeticProbe) if t == 0 else result

    monkeypatch.setattr(api, "_evolved_density", evolve)
    report = api.case_report(4, 4, 0.7)
    first, second = report["times"][0]["distances"]
    assert len(attempts) == 2
    assert first["status"] == UNAVAILABLE and first["commutator_norm"] is None
    assert first["analytic_upper_bound"] == 0
    assert second["status"] == AVAILABLE and second["commutator_norm"] is not None
    assert all(row["status"] == AVAILABLE for row in report["times"][1:])
    assert report["status"] == UNAVAILABLE
    for obs in observations(report):
        check_comparison(obs)
    strict_json(report)


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_nonfinite_unitary_is_unavailable_for_only_the_affected_time(bad, monkeypatch):
    original = api._unitary_columns
    called = []

    def unitary(system, t, columns):
        called.append(t)
        result = original(system, t, columns)
        if t == 0.001:
            result[0, 0] = bad
        return result

    monkeypatch.setattr(api, "_unitary_columns", unitary)
    report = api.case_report(3, 3, 0.7)
    assert called == list(TIMES)
    assert report["status"] == UNAVAILABLE
    for row in report["times"]:
        failed = row["t"] == 0.001
        assert row["status"] == (UNAVAILABLE if failed else AVAILABLE)
        obs = row["distances"][0]
        assert (obs["commutator_norm"] is None) is failed
        assert obs["analytic_upper_bound"] is not None
        check_comparison(obs)
    strict_json(report)


def unavailable_tail(reason):
    return {"status": UNAVAILABLE, "value": None, "partial_sum": None,
            "remainder_bound": None, "terms": 0, "reason": reason}


def test_tail_failure_preserves_norm_and_independent_later_work(monkeypatch):
    original = api._capped_tail
    called = []

    def tail(z, distance):
        called.append((z, distance))
        if z == 0 and distance == 1:
            return unavailable_tail("injected-first-tail")
        return original(z, distance)
    monkeypatch.setattr(api, "_capped_tail", tail)
    report = api.case_report(4, 4, 0.7)
    assert len(called) == 8
    rows = observations(report)
    assert rows[0]["status"] == UNAVAILABLE and rows[0]["commutator_norm"] is not None
    assert rows[0]["analytic_upper_bound"] is None
    assert all(obs["status"] == AVAILABLE for obs in rows[1:])
    assert report["status"] == UNAVAILABLE and "injected-first-tail" in report["reason"]
    for obs in rows:
        check_tail_schema(obs["bound_evaluation"])
        check_comparison(obs)
    strict_json(report)


def test_real_t0_unitary_route_retains_small_positive_norm_and_literal_failure(monkeypatch):
    original = api._unitary_columns
    perturbation = 1e-12
    called = []

    def unitary(system, t, columns):
        called.append(t)
        result = original(system, t, columns)
        if t == 0:
            # L=N=3 lex basis: last state=(3,0,0), index3=(0,3,0).
            result[-1, 3] += perturbation
        return result
    monkeypatch.setattr(api, "_unitary_columns", unitary)
    report = api.case_report(3, 3, 0.7)
    first = report["times"][0]["distances"][0]
    assert called == list(TIMES)
    assert 0 < first["commutator_norm"] < TOL
    assert first["analytic_upper_bound"] == 0
    assert first["signed_excess"] == first["commutator_norm"]
    assert first["literal_inequality"] is False
    assert first["comparison"]["status"] == "consistent"
    assert first["status"] == report["times"][0]["status"] == report["status"] == AVAILABLE
    check_comparison(first)
    strict_json(report)


@pytest.mark.parametrize("value,expected", [(TOL / 2, AVAILABLE), (2 * TOL, MISMATCH)])
def test_raw_norm_excess_and_comparison_threshold_are_not_clipped(value, expected, monkeypatch):
    original = api._operator_norm
    count = [0]

    def norm(matrix):
        count[0] += 1
        return value if count[0] == 1 else original(matrix)
    monkeypatch.setattr(api, "_operator_norm", norm)
    report = api.case_report(3, 3, 0.7)
    first = observations(report)[0]
    assert first["commutator_norm"] == first["signed_excess"] == value
    assert first["status"] == report["status"] == expected
    assert first["literal_inequality"] is (value <= 0)
    assert first["comparison"]["max_absolute_error"] == abs(value)
    check_comparison(first)


def test_negative_injected_norm_is_unavailable_without_clipping_or_dependency_loss(monkeypatch):
    original = api._operator_norm
    count = [0]

    def norm(matrix):
        count[0] += 1
        return -1e-12 if count[0] == 1 else original(matrix)

    monkeypatch.setattr(api, "_operator_norm", norm)
    report = api.case_report(4, 4, 0.7)
    assert count[0] == 8
    first, second = report["times"][0]["distances"]
    assert first["status"] == UNAVAILABLE and first["reason"]
    assert first["commutator_norm"] is None
    assert first["analytic_upper_bound"] == 0
    assert first["bound_evaluation"]["status"] == "structural_zero"
    assert second["status"] == AVAILABLE and second["commutator_norm"] is not None
    assert report["status"] == report["times"][0]["status"] == UNAVAILABLE
    assert report["reason"] == report["times"][0]["reason"] == first["reason"]
    assert all(row["status"] == AVAILABLE for row in report["times"][1:])
    for obs in observations(report):
        check_comparison(obs)
    strict_json(report)


def test_aggregate_priority_and_first_declared_reason_across_distance_time_case(monkeypatch):
    original_tail, original_norm = api._capped_tail, api._operator_norm
    count = [0]

    def tail(z, distance):
        if z == 0 and distance == 1:
            return unavailable_tail("earlier-unavailable-tail")
        return original_tail(z, distance)

    def norm(matrix):
        count[0] += 1
        if count[0] in (2, 3):
            return 3.0
        return original_norm(matrix)
    monkeypatch.setattr(api, "_capped_tail", tail)
    monkeypatch.setattr(api, "_operator_norm", norm)
    report = api.case_report(4, 4, 0.7)
    first_time = report["times"][0]
    assert first_time["distances"][0]["status"] == UNAVAILABLE
    first_mismatch = first_time["distances"][1]
    assert first_mismatch["status"] == MISMATCH
    assert first_time["status"] == report["status"] == MISMATCH
    assert first_time["reason"] == report["reason"] == first_mismatch["reason"]
    assert report["times"][1]["status"] == MISMATCH
    assert all(row["status"] == AVAILABLE for row in report["times"][2:])
    strict_json(report)


def test_demo_failure_order_counts_and_detached_reports(reports, monkeypatch):
    calls = []

    def case_report(L, N, g):
        key = (L, N, g)
        calls.append(key)
        result = copy.deepcopy(reports[key])
        number = GRID.index(key)
        if number in (0, 1, 2):
            status = UNAVAILABLE if number == 0 else MISMATCH
            reason = "earlier-unavailable" if number == 0 else "first-mismatch" if number == 1 else "later-mismatch"
            result["status"], result["reason"] = status, reason
            result["times"][0]["status"], result["times"][0]["reason"] = status, reason
            result["times"][0]["distances"][0]["status"] = status
            result["times"][0]["distances"][0]["reason"] = reason
        return result
    monkeypatch.setattr(api, "case_report", case_report)
    demo = api.demonstration_report()
    assert calls == list(GRID)
    assert demo["status"] == MISMATCH and demo["reason"] == "first-mismatch"
    assert demo["summary"]["case_status_counts"] == {AVAILABLE: 9, UNAVAILABLE: 1, MISMATCH: 2}
    assert demo["summary"]["observation_status_counts"] == {AVAILABLE: 105, UNAVAILABLE: 1, MISMATCH: 2}
    assert set(demo["summary"]["bound_status_counts"]) == set(TAIL_STATUSES)
    strict_json(demo)
    demo["cases"][0]["times"][0]["distances"][0]["comparison"]["status"] = "mutated"
    assert observations(reports[GRID[0]])[0]["comparison"]["status"] == "consistent"


def test_repeated_public_reports_are_detached_owned_json():
    first = api.case_report(3, 3, 0.0)
    before = copy.deepcopy(first)
    second = api.case_report(3, 3, 0.0)
    second["times"][0]["distances"][0]["bound_evaluation"]["value"] = -99
    second["scope"]["finite_sector_only"] = False
    second["eigensystem"]["status"] = "mutated"
    assert first == before
    strict_json(first)


@pytest.mark.parametrize("boundary", ("model", "eigen", "evolution", "unitary", "norm", "tail"))
@pytest.mark.parametrize("error_type", (TypeError, ValueError, RuntimeError, AssertionError, KeyError))
def test_programming_errors_propagate_instead_of_numerical_status(boundary, error_type, monkeypatch):
    sentinel = error_type("injected-programming-error")

    def broken(*args, **kwargs):
        raise sentinel
    if boundary == "model":
        patch_builder(monkeypatch, broken)
    else:
        name = {"eigen": "_screened_eigensystem", "evolution": "_evolved_density",
                "unitary": "_unitary_columns", "norm": "_operator_norm", "tail": "_capped_tail"}[boundary]
        monkeypatch.setattr(api, name, broken)
    with pytest.raises(error_type) as caught:
        api.case_report(3, 3, 0.7)
    assert caught.value is sentinel


@pytest.mark.parametrize("boundary", ("eigen", "evolution", "norm", "tail"))
def test_numerically_unresolved_prefix_does_not_hide_valueerror_outside_builder(boundary, monkeypatch):
    sentinel = ValueError("numerically unresolved is special only at inherited builder")

    def broken(*args, **kwargs):
        raise sentinel
    name = {"eigen": "_screened_eigensystem", "evolution": "_evolved_density",
            "norm": "_operator_norm", "tail": "_capped_tail"}[boundary]
    monkeypatch.setattr(api, name, broken)
    with pytest.raises(ValueError) as caught:
        api.case_report(3, 3, 0.7)
    assert caught.value is sentinel
