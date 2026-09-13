"""Independent frozen-contract tests; authored without source/demo inspection.

Rational density moments and a degree-12 phase series are independent oracles.
All tolerances are heuristic floating comparisons, not numerical certificates.
Authorship validation is AST/hash only; importing this file executes no controls.
"""
import copy
from decimal import Decimal
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from bpr import chiral_source_compatibility as api


ATOL = RTOL = 2e-10
SOURCES = ("one", "z", "p2")
QUADRATURES = ((24, 48), (48, 96))
EPSILONS = (0.0, 0.01, 0.1)
AVAILABLE = "available_heuristic"
UNAVAILABLE = "numerical_unavailable"
MISMATCH = "diagnostic_mismatch"
STATUS_KEYS = {AVAILABLE, UNAVAILABLE, MISMATCH}
CASE_KEYS = set("model_id source_id q R quadrature status reason scope analytic gram B C G moment_comparisons leakage_spectrum phases".split())
PHASE_KEYS = set("epsilon status reason analytic S L leakage_spectrum leading_order_error leading_order_comparison upper_bound_comparison".split())
SPECTRAL_KEYS = set("status reason eigenvalues hermiticity_residual eigensystem psd_status literal_psd lambda_min lambda_max sqrt_lambda_max_candidate".split())
SOLVER_KEYS = set("status reason orthogonality_residual eigenpair_residual tolerance scale".split())
COMPARISON_KEYS = {"status", "max_absolute_error", "reason"}
UPPER_KEYS = COMPARISON_KEYS | {"signed_excess", "literal_inequality"}
CASE_ANALYTIC_KEYS = set("B_diagonal C_diagonal G_diagonal leakage_norm h_range_width source_norm phase_leakage_structural_zero_for_all_epsilon structural_zero_reason first_order_shift_coefficients formal_second_order_status".split())
PHASE_ANALYTIC_KEYS = set("leading_coefficient_diagonal upper_bound remainder_bound structural_zero reason".split())


def moment_z(k, power):
    """Exact integration against the independently specified polynomial density."""
    if k == 2:
        return (-1) ** power * moment_z(0, power)
    if power % 2:
        return Fraction(3, 2 * (power + 2)) if k == 0 else Fraction(0)
    numerator = 3 * (power + 2) if k == 0 else 3
    denominator = (2 if k == 0 else 1) * (power + 1) * (power + 3)
    return Fraction(numerator, denominator)


def moment_h(source, k, power):
    if source == "one":
        return Fraction(1)
    if source == "z":
        return moment_z(k, power)
    return sum((Fraction(math.comb(power, j) * 3 ** j * (-1) ** (power - j), 2 ** power)
                * moment_z(k, 2 * j) for j in range(power + 1)), Fraction(0))


def reference(source, epsilon=0.0):
    B = [moment_h(source, k, 1) for k in range(3)]
    C = [moment_h(source, k, 2) for k in range(3)]
    G = [c - b * b for b, c in zip(B, C)]
    phase = [sum(((-1j * epsilon) ** r * float(moment_h(source, k, r))
                  / math.factorial(r) for r in range(13)), 0j) for k in range(3)]
    return (np.diag([float(x) for x in B]), np.diag([float(x) for x in C]),
            np.diag([float(x) for x in G]), np.diag(phase))


def norm(matrix):
    return float(np.linalg.svd(matrix, compute_uv=False)[0])


def close(actual, expected, extra=0.0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    error = norm(actual - expected) if actual.ndim == 2 else float(np.max(np.abs(actual - expected)))
    scale = norm(expected) if expected.ndim == 2 else float(np.max(np.abs(expected)))
    assert error <= ATOL + RTOL * scale + extra


def decode(record):
    assert type(record) is dict and set(record) == {"real", "imag"}
    result = np.array(record["real"]) + 1j * np.array(record["imag"])
    assert result.shape == (3, 3)
    assert np.isfinite(result).all()
    return result


def native(value):
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            native(item)
    elif type(value) is list:
        for item in value:
            native(item)
    else:
        assert value is None or type(value) in (str, bool, int, float)
        if type(value) is float:
            assert math.isfinite(value)


def reason(record):
    assert record["reason"] is None or type(record["reason"]) is str
    if record["status"] not in (AVAILABLE, "consistent"):
        assert record["reason"]


def comparison(record, status="consistent", upper=False):
    assert set(record) == (UPPER_KEYS if upper else COMPARISON_KEYS)
    assert record["status"] == status
    reason(record)
    if status == "inconclusive":
        assert record["max_absolute_error"] is None
        if upper:
            assert record["signed_excess"] is None and record["literal_inequality"] is None
    else:
        assert record["max_absolute_error"] >= 0
        if upper:
            close(record["max_absolute_error"], abs(record["signed_excess"]))
            assert record["literal_inequality"] is (record["signed_excess"] <= 0)
            if record["signed_excess"] > 0 and status == "consistent":
                assert record["reason"]


def scope(record):
    assert record == {
        "supplied_internal_sphere": True, "physical_spacetime_chirality": False,
        "standard_model_derivation": False, "full_dirac_solver": False,
        "numerical_error_certified": False, "empirical_validation": False,
        "empirical_status": "empirical_test_unavailable",
    }


def solved_spectrum(record, matrix):
    assert set(record) == SPECTRAL_KEYS
    assert record["status"] == AVAILABLE
    reason(record)
    assert set(record["eigensystem"]) == SOLVER_KEYS
    assert record["eigensystem"]["status"] == AVAILABLE
    for field in SOLVER_KEYS - {"status", "reason"}:
        assert math.isfinite(record["eigensystem"][field])
        assert record["eigensystem"][field] >= 0
    values = np.linalg.eigvalsh(matrix)
    close(record["eigenvalues"], values)
    close(record["hermiticity_residual"], np.linalg.norm(matrix - matrix.conj().T, "fro"))
    assert record["lambda_min"] == min(record["eigenvalues"])
    assert record["lambda_max"] == max(record["eigenvalues"])
    assert record["literal_psd"] is (record["lambda_min"] >= 0)
    expected_psd = "psd_consistent_heuristic" if record["literal_psd"] else "negative_within_tolerance"
    assert record["psd_status"] == expected_psd
    if record["lambda_max"] < 0:
        assert record["sqrt_lambda_max_candidate"] is None
    else:
        close(record["sqrt_lambda_max_candidate"], math.sqrt(record["lambda_max"]))


def failed_spectrum(record):
    assert set(record) == SPECTRAL_KEYS
    assert record["status"] == record["psd_status"] == UNAVAILABLE
    assert record["reason"]
    for field in ("eigenvalues", "literal_psd", "lambda_min", "lambda_max", "sqrt_lambda_max_candidate"):
        assert record[field] is None
    assert set(record["eigensystem"]) == SOLVER_KEYS
    assert record["eigensystem"]["status"] == UNAVAILABLE
    assert record["eigensystem"]["reason"]
    for field in SOLVER_KEYS - {"status", "reason"}:
        assert record["eigensystem"][field] is None


@pytest.fixture(scope="module")
def demo():
    return api.demonstration_report()


@pytest.mark.parametrize("source", SOURCES)
def test_rational_moment_reference_tables(source):
    exact = {
        "one": ([1, 1, 1], [1, 1, 1], [0, 0, 0]),
        "z": ([Fraction(1, 2), 0, Fraction(-1, 2)],
              [Fraction(2, 5), Fraction(1, 5), Fraction(2, 5)],
              [Fraction(3, 20), Fraction(1, 5), Fraction(3, 20)]),
        "p2": ([Fraction(1, 10), Fraction(-1, 5), Fraction(1, 10)],
               [Fraction(8, 35), Fraction(1, 7), Fraction(8, 35)],
               [Fraction(153, 700), Fraction(18, 175), Fraction(153, 700)]),
    }
    b = [moment_h(source, k, 1) for k in range(3)]
    c = [moment_h(source, k, 2) for k in range(3)]
    assert (b, c, [y - x * x for x, y in zip(b, c)]) == exact[source]


@pytest.mark.parametrize("case_index", range(6))
def test_six_cases_and_eighteen_phases_against_independent_oracles(demo, case_index):
    case = demo["cases"][case_index]
    source = SOURCES[case_index % 3]
    polar, azimuth = QUADRATURES[case_index // 3]
    B, C, G, _ = reference(source)
    width = {"one": 0.0, "z": 2.0, "p2": 1.5}[source]
    assert set(case) == CASE_KEYS
    assert case["model_id"] == "chiral_source_compatibility"
    assert case["source_id"] == source and case["q"] == 3 and case["R"] == 1
    assert case["quadrature"] == {"n_polar": polar, "n_azimuth": azimuth, "samples": polar * azimuth}
    assert case["status"] == AVAILABLE
    reason(case)
    scope(case["scope"])
    analytic = case["analytic"]
    assert set(analytic) == CASE_ANALYTIC_KEYS
    for label, expected in (("B", B), ("C", C), ("G", G)):
        close(decode(case[label]), expected)
        close(analytic[label + "_diagonal"], expected.diagonal())
        comparison(case["moment_comparisons"][label])
        close(case["moment_comparisons"][label]["max_absolute_error"], norm(decode(case[label]) - expected))
    assert set(case["moment_comparisons"]) == {"B", "C", "G"}
    close(analytic["leakage_norm"], math.sqrt(norm(G)))
    assert analytic["h_range_width"] == width and analytic["source_norm"] == 1
    close(analytic["first_order_shift_coefficients"], B.diagonal())
    assert analytic["formal_second_order_status"] == "structural_zero_by_chirality"
    assert analytic["phase_leakage_structural_zero_for_all_epsilon"] is (source == "one")
    assert bool(analytic["structural_zero_reason"]) is (source == "one")
    assert set(case["gram"]) == {"status", "reason", "matrix", "error_from_identity", "comparison"}
    T = decode(case["gram"]["matrix"])
    close(T, np.eye(3))
    close(case["gram"]["error_from_identity"], norm(T - np.eye(3)))
    comparison(case["gram"]["comparison"])
    solved_spectrum(case["leakage_spectrum"], decode(case["G"]))
    assert [p["epsilon"] for p in case["phases"]] == list(EPSILONS)
    strong = {"one": (0, 0, 0), "z": (Fraction(2, 175), Fraction(3, 175), Fraction(2, 175)),
              "p2": (Fraction(513, 26950), Fraction(81, 13475), Fraction(513, 26950))}[source]
    for phase in case["phases"]:
        assert set(phase) == PHASE_KEYS
        epsilon = phase["epsilon"]
        assert phase["status"] == AVAILABLE
        reason(phase)
        pa = phase["analytic"]
        assert set(pa) == PHASE_ANALYTIC_KEYS
        close(pa["leading_coefficient_diagonal"], G.diagonal())
        upper = min(1.0, epsilon ** 2 * norm(G))
        width_bound = epsilon ** 4 * width ** 2 * norm(G) / 12
        close(pa["upper_bound"], upper)
        close(pa["remainder_bound"], width_bound)
        structural = source == "one" or epsilon == 0
        assert pa["structural_zero"] is structural
        assert bool(pa["reason"]) is structural
        S, L = decode(phase["S"]), decode(phase["L"])
        expected_S = reference(source, epsilon)[3]
        omitted = abs(epsilon) ** 13 / math.factorial(13)
        close(S, expected_S, omitted)
        expected_L = np.eye(3) - expected_S.conj().T @ expected_S
        close(L, expected_L, 2 * omitted + omitted ** 2)
        close(L, np.eye(3) - S.conj().T @ S)
        solved_spectrum(phase["leakage_spectrum"], L)
        leading_error = norm(L - epsilon ** 2 * decode(case["G"]))
        close(phase["leading_order_error"], leading_error)
        comparison(phase["leading_order_comparison"], upper=True)
        comparison(phase["upper_bound_comparison"], upper=True)
        close(phase["leading_order_comparison"]["signed_excess"], leading_error - width_bound)
        close(phase["upper_bound_comparison"]["signed_excess"], norm(L) - upper)
        assert norm(L - epsilon ** 2 * G) <= width_bound + ATOL + RTOL * width_bound
        deficit = epsilon ** 2 * G.diagonal().real - L.diagonal().real
        assert np.all(deficit >= -ATOL)
        assert np.all(deficit <= np.array([float(k) for k in strong]) * epsilon ** 4 + ATOL)
        if source != "one" and epsilon == 0.1:
            assert norm(S - np.diag(np.exp(-1j * epsilon * B.diagonal()))) > 100 * ATOL


def test_demo_refinements_counts_scope_and_strict_json(demo):
    assert set(demo) == set("model_id status reason scope controls cases refinement_comparisons summary".split())
    assert demo["model_id"] == "chiral_source_compatibility" and demo["status"] == AVAILABLE
    reason(demo)
    scope(demo["scope"])
    controls = demo["controls"]
    assert set(controls) == {"q", "R", "sources", "epsilons", "quadratures"}
    assert controls["q"] == 3 and controls["R"] == 1
    assert controls["sources"] == list(SOURCES) and controls["epsilons"] == list(EPSILONS)
    pairs = [(item["n_polar"], item["n_azimuth"]) if type(item) is dict else tuple(item)
             for item in controls["quadratures"]]
    assert pairs == list(QUADRATURES)
    assert len(demo["cases"]) == 6
    for key, total in (("case_status_counts", 6), ("phase_status_counts", 18)):
        assert demo["summary"][key] == {AVAILABLE: total, UNAVAILABLE: 0, MISMATCH: 0}
    assert [r["source_id"] for r in demo["refinement_comparisons"]] == list(SOURCES)
    for i, record in enumerate(demo["refinement_comparisons"]):
        assert set(record) == {"source_id", "status", "reason", "B", "C", "G", "phases"}
        assert record["status"] == AVAILABLE
        for label in ("B", "C", "G"):
            comparison(record[label])
            close(record[label]["max_absolute_error"], norm(decode(demo["cases"][i][label]) - decode(demo["cases"][i + 3][label])))
        assert len(record["phases"]) == 3
        for j, phase in enumerate(record["phases"]):
            assert set(phase) == {"epsilon", "status", "reason", "S", "L"}
            assert phase["epsilon"] == EPSILONS[j] and phase["status"] == AVAILABLE
            for label in ("S", "L"):
                comparison(phase[label])
                close(phase[label]["max_absolute_error"], norm(decode(demo["cases"][i]["phases"][j][label]) - decode(demo["cases"][i + 3]["phases"][j][label])))
    native(demo)
    assert json.loads(json.dumps(demo, allow_nan=False)) == demo


def test_successful_work_budget_pointwise_square_and_no_sample_square(monkeypatch):
    counts = {"systems": 0, "overlaps": 0, "spectra": 0, "phases": 0, "sources": 0}
    current_source = [None]
    phase_arguments = []
    saved = {name: getattr(api, name) for name in ("_sampled_system", "_source_values", "_weighted_overlap", "_phase_overlap", "_screened_eigensystem")}
    ordinary = []

    def system(*args):
        counts["systems"] += 1
        profiles, weights, z = saved["_sampled_system"](*args)
        assert profiles.shape == (args[0] * args[1], 3)
        assert weights.shape == z.shape == (len(profiles),)
        for array in (profiles, weights, z):
            assert type(array) is np.ndarray and array.flags.owndata
        return profiles, weights, z

    def source(source_id, z):
        counts["sources"] += 1
        result = saved["_source_values"](source_id, z)
        current_source[0] = (source_id, z.copy(), result.copy())
        return result

    def overlap(profiles, weights, values):
        counts["overlaps"] += 1
        assert profiles.shape[1:] == (3,) and 1 <= len(profiles) <= 4608
        assert values.shape == weights.shape == (len(profiles),)
        if current_source[0] is not None and not phase_arguments:
            ordinary.append((current_source[0], values.copy()))
        result = saved["_weighted_overlap"](profiles, weights, values)
        assert result.shape == (3, 3) and result.flags.owndata
        return result

    def phase(profiles, weights, values, epsilon):
        counts["phases"] += 1
        phase_arguments.append(epsilon)
        try:
            return saved["_phase_overlap"](profiles, weights, values, epsilon)
        finally:
            phase_arguments.pop()

    def spectrum(matrix):
        counts["spectra"] += 1
        assert matrix.shape == (3, 3)
        return saved["_screened_eigensystem"](matrix)

    # Catch common dense allocation routes without constructing a sampled projector.
    def guarded(original):
        def allocate(shape, *args, **kwargs):
            dimensions = (int(shape),) if np.isscalar(shape) else tuple(shape)
            assert not (len(dimensions) >= 2 and dimensions[-1] >= 1152 and dimensions[-2] >= 1152)
            return original(shape, *args, **kwargs)
        return allocate

    for name in ("zeros", "ones", "empty", "full"):
        monkeypatch.setattr(np, name, guarded(getattr(np, name)))
    original_eye = np.eye

    def eye(n, m=None, *args, **kwargs):
        assert n < 1152 or (m is not None and m < 1152)
        return original_eye(n, m, *args, **kwargs)

    monkeypatch.setattr(np, "eye", eye)
    for name, function in (("_sampled_system", system), ("_source_values", source), ("_weighted_overlap", overlap),
                           ("_phase_overlap", phase), ("_screened_eigensystem", spectrum)):
        monkeypatch.setattr(api, name, function)
    report = api.demonstration_report()
    assert report["status"] == AVAILABLE
    assert counts == {"systems": 2, "overlaps": 32, "spectra": 24, "phases": 18, "sources": 6}
    squares = [(z, h, values) for (source_id, z, h), values in ordinary
               if source_id == "p2" and values.shape == h.shape and np.array_equal(values, h * h)]
    assert len(squares) == 2
    for z, h, values in squares:
        np.testing.assert_allclose(h, (3 * z * z - 1) / 2, atol=1e-15, rtol=1e-15)
        truncated = 1 / 5 + (2 / 7) * h
        assert np.max(np.abs(values - truncated)) > 0.01


def synthetic_samples():
    profiles = np.array([[1, 1j, 2], [1j, 2, -1j], [2, -1, 1], [1, 2j, -2]], dtype=complex) / 4
    weights = np.array([0.2, 0.3, 0.4, 0.1])
    values = np.array([0.2 + 0.1j, -0.7 + 0.4j, 0.6 - 0.3j, 0.8 + 0.2j])
    return profiles, weights, values


def assert_owned(array, *inputs):
    assert type(array) is np.ndarray and array.flags.owndata
    for source in inputs:
        assert not np.shares_memory(array, source)


def test_complex_nonsymmetric_weighted_overlap_is_not_symmetrized():
    profiles, weights, values = synthetic_samples()
    before = [item.copy() for item in (profiles, weights, values)]
    expected = sum((weights[i] * values[i] * np.outer(profiles[i].conj(), profiles[i])
                    for i in range(4)), np.zeros((3, 3), complex))
    assert norm(expected - expected.conj().T) > 0.01
    result = api._weighted_overlap(profiles, weights, values)
    np.testing.assert_allclose(result, expected, atol=1e-15, rtol=1e-14)
    assert_owned(result, profiles, weights, values)
    for actual, old in zip((profiles, weights, values), before):
        np.testing.assert_array_equal(actual, old)


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1, np.float64(-0.1), np.int64(0)])
def test_phase_overlap_uses_actual_complex_exponential_even_at_zero(monkeypatch, epsilon):
    profiles, weights, _ = synthetic_samples()
    h = np.array([-1.0, -0.2, 0.4, 1.0])
    calls = []
    original = api._weighted_overlap

    def spy(p, w, values):
        calls.append(values.copy())
        return original(p, w, values)

    monkeypatch.setattr(api, "_weighted_overlap", spy)
    result = api._phase_overlap(profiles, weights, h, epsilon)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.exp(-1j * float(epsilon) * h), atol=0, rtol=1e-15)
    expected = profiles.conj().T @ ((weights * np.exp(-1j * float(epsilon) * h))[:, None] * profiles)
    close(result, expected)
    assert_owned(result, profiles, weights, h)
    if epsilon == 0:
        assert norm(result - np.eye(3)) > 0.1


def test_common_sample_phase_cancels_without_changing_controls(demo, monkeypatch):
    original = api._sampled_system

    def phased(*args):
        profiles, weights, z = original(*args)
        shared = np.exp(1j * (0.2 * z + 0.1 * z * z))
        return profiles * shared[:, None], weights, z

    monkeypatch.setattr(api, "_sampled_system", phased)
    changed = api.demonstration_report()
    for before, after in zip(demo["cases"], changed["cases"]):
        for label in ("B", "C", "G"):
            close(decode(after[label]), decode(before[label]))
        for p, q in zip(before["phases"], after["phases"]):
            close(decode(q["S"]), decode(p["S"]))
            close(decode(q["L"]), decode(p["L"]))


def test_raw_gram_matrix_uses_matrix_product_without_clipping_or_symmetry():
    B = np.array([[0.1, 0.2j, 0.3], [0.4, -0.2, 0.1j], [-0.1j, 0.2, 0.3]])
    C = np.array([[0.2, 0.1, 0.4j], [0.3j, 0.1, -0.2], [0.1, 0.2j, -0.3]])
    before = B.copy(), C.copy()
    G = api._gram_matrix(B, C)
    close(G, C - B @ B)
    assert norm(G - (C - B * B)) > 0.01
    assert norm(G - G.conj().T) > 0.1
    assert_owned(G, B, C)
    for actual, old in zip((B, C), before):
        np.testing.assert_array_equal(actual, old)
    negative = api._gram_matrix(np.eye(3), (1 - 1e-11) * np.eye(3))
    assert np.all(negative.diagonal().real < 0)
    np.testing.assert_array_equal(negative, (1 - 1e-11) * np.eye(3) - np.eye(3))


def test_raw_phase_leakage_uses_adjoint_product_and_retains_negative_values():
    S = np.array([[1, 0.1j, 0.3], [0.2, 0.7, -0.1j], [0.2j, 0.4, 0.8]])
    before = S.copy()
    L = api._phase_leakage(S)
    close(L, np.eye(3) - S.conj().T @ S)
    assert norm(L - (np.eye(3) - S.conj() * S)) > 0.1
    assert norm(L - (np.eye(3) - S.T @ S)) > 0.1
    assert_owned(L, S)
    np.testing.assert_array_equal(S, before)
    raw = api._phase_leakage((1 + 1e-11) * np.eye(3))
    assert np.all(raw.diagonal().real < 0)


def test_noncommuting_compression_has_cubic_not_generic_quartic_remainder():
    # Four sampled rows and a three-dimensional subspace, never an ambient projector.
    columns = np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]], dtype=complex) / 2
    h = np.array([-1.0, -0.4, 0.2, 0.9])
    weights = np.ones(4)
    B = api._weighted_overlap(columns, weights, h)
    C = api._weighted_overlap(columns, weights, h * h)
    G = api._gram_matrix(B, C)
    commutator = B @ C - C @ B
    assert norm(commutator) > 0.01
    epsilon = 0.001
    S = api._phase_overlap(columns, weights, h, epsilon)
    L = api._phase_leakage(S)
    cubic = 0.5j * (B @ C - C @ B)
    residual = L - epsilon ** 2 * G
    assert norm(residual / epsilon ** 3 - cubic) < 0.002
    false_quartic = epsilon ** 4 * float(np.ptp(h)) ** 2 * norm(G) / 12
    assert norm(residual) > 10 * false_quartic
    general = abs(epsilon) ** 3 * math.sqrt(norm(G)) * max(abs(h)) ** 2 + epsilon ** 4 * max(abs(h)) ** 4 / 4
    assert norm(residual) <= general + ATOL


@pytest.mark.parametrize("eigenvalues,psd,status,literal", [
    ([0.0, 0.2, 0.4], "psd_consistent_heuristic", AVAILABLE, True),
    ([-1e-11, 0.0, 0.4], "negative_within_tolerance", AVAILABLE, False),
    ([-3e-11, -2e-11, -1e-11], "negative_within_tolerance", AVAILABLE, False),
    ([-1e-6, 0.0, 0.4], MISMATCH, MISMATCH, False),
    ([-3.0, -2.0, -1.0], MISMATCH, MISMATCH, False),
])
def test_spectral_raw_signs_and_detached_metadata(eigenvalues, psd, status, literal):
    matrix = np.diag(eigenvalues)
    record = api._spectral_record(matrix)
    assert set(record) == SPECTRAL_KEYS
    assert record["status"] == status and record["psd_status"] == psd
    assert record["literal_psd"] is literal
    np.testing.assert_allclose(record["eigenvalues"], eigenvalues, atol=0, rtol=2e-15)
    assert record["lambda_min"] == min(record["eigenvalues"])
    assert record["lambda_max"] == max(record["eigenvalues"])
    if eigenvalues[-1] < 0:
        assert record["sqrt_lambda_max_candidate"] is None
    else:
        close(record["sqrt_lambda_max_candidate"], math.sqrt(eigenvalues[-1]))
    if not literal:
        assert record["reason"]
    native(record)
    saved_values = list(record["eigenvalues"])
    matrix[:] = 99
    assert record["eigenvalues"] == saved_values


@pytest.mark.parametrize("eigenvalues,expected", [
    ([-1.99e-10, 0, 0], AVAILABLE), ([-2.01e-10, 0, 0], MISMATCH),
    ([-2.0e-8, 0, 100.0], AVAILABLE), ([-2.1e-8, 0, 100.0], MISMATCH),
])
def test_psd_threshold_uses_raw_eigenvalue_scale(monkeypatch, eigenvalues, expected):
    # Freeze solver metadata so this tests the sign screen, not eigensolver tolerances.
    monkeypatch.setattr(api, "_screened_eigensystem", lambda matrix: {
        "values": np.array(eigenvalues), "vectors": np.eye(3),
        "orthogonality_residual": 0.0, "eigenpair_residual": 0.0, "tolerance": 1e-3, "scale": 1e9,
    })
    record = api._spectral_record(np.diag(eigenvalues))
    assert record["status"] == expected
    assert record["literal_psd"] is False


def test_spectral_nonhermitian_failure_preserves_hermiticity_residual():
    matrix = np.eye(3, dtype=complex)
    matrix[0, 1] = 0.25j
    record = api._spectral_record(matrix)
    failed_spectrum(record)
    close(record["hermiticity_residual"], math.sqrt(2) / 4)


def test_spectral_solver_metadata_is_detached(monkeypatch):
    solver = {"values": np.array([0.0, 0.1, 0.2]), "vectors": np.eye(3),
              "orthogonality_residual": 1e-16, "eigenpair_residual": 2e-16,
              "tolerance": 3e-13, "scale": 1.0}
    monkeypatch.setattr(api, "_screened_eigensystem", lambda matrix: solver)
    report = api._spectral_record(np.diag([0.0, 0.1, 0.2]))
    frozen = copy.deepcopy(report)
    solver["values"][:] = -8
    solver["orthogonality_residual"] = 77
    assert report == frozen
    native(report)


class ArraySubclass(np.ndarray):
    pass


class ConversionTrap:
    def __array__(self, *args, **kwargs):
        raise AssertionError("array conversion protocol must not be called")

    def __float__(self):
        raise AssertionError("float conversion protocol must not be called")

    def __int__(self):
        raise AssertionError("integer conversion protocol must not be called")


class StringSubclass(str):
    pass


def test_public_validation_precedes_sampling(monkeypatch):
    def forbidden(*args):
        raise AssertionError("sampling preceded validation")

    monkeypatch.setattr(api, "_sampled_system", forbidden)
    for source in (None, True, 1, b"one", StringSubclass("one"), ConversionTrap()):
        with pytest.raises((TypeError, ValueError)):
            api.case_report(source)
    for source in ("ONE", " z", "p2 ", "P2", "constant", ""):
        with pytest.raises((TypeError, ValueError)):
            api.case_report(source)
    for pair in ((True, 48), (24, False), (24.0, 48), (24, np.float64(48)),
                 (np.bool_(True), 48), (24, ConversionTrap()), (ConversionTrap(), 48),
                 (12, 24), (24, 96), (48, 48), (-24, 48)):
        with pytest.raises((TypeError, ValueError)):
            api.case_report("one", *pair)


@pytest.mark.parametrize("pair", [(np.int64(24), np.int32(48)), (np.uint32(48), np.uint64(96))])
def test_public_numpy_integer_quadratures_are_accepted(pair):
    report = api.case_report("z", *pair)
    assert report["status"] == AVAILABLE
    assert report["quadrature"]["n_polar"] == int(pair[0])
    native(report)


@pytest.mark.parametrize("slot", range(3))
@pytest.mark.parametrize("kind", ["list", "subclass", "bool", "object", "string", "trap"])
def test_weighted_overlap_rejects_nonbase_numeric_arrays(slot, kind):
    args = list(synthetic_samples())
    original = args[slot]
    if kind == "list":
        invalid = original.tolist()
    elif kind == "subclass":
        invalid = original.view(ArraySubclass)
    elif kind == "trap":
        invalid = ConversionTrap()
    else:
        dtype = {"bool": bool, "object": object, "string": "U8"}[kind]
        invalid = np.ones(original.shape, dtype=dtype)
    args[slot] = invalid
    with pytest.raises(TypeError):
        api._weighted_overlap(*args)


@pytest.mark.parametrize("slot", range(3))
@pytest.mark.parametrize("kind", ["nan", "inf", "wrong_shape", "empty", "complex_real_only"])
def test_weighted_overlap_rejects_invalid_values_and_shapes(slot, kind):
    args = list(synthetic_samples())
    if kind == "nan" or kind == "inf":
        args[slot].flat[0] = float(kind)
    elif kind == "wrong_shape":
        args[slot] = np.ones((4, 2)) if slot == 0 else np.ones((4, 1))
    elif kind == "empty":
        args[slot] = np.empty((0, 3)) if slot == 0 else np.empty(0)
    elif slot == 1:
        args[slot] = args[slot].astype(complex)
    else:
        return  # Complex profiles and values are explicitly valid.
    with pytest.raises((TypeError, ValueError)):
        api._weighted_overlap(*args)


@pytest.mark.parametrize("weights", [np.array([0.0, 1, 1, 1]), np.array([-1.0, 1, 1, 1])])
def test_weighted_overlap_requires_strictly_positive_weights(weights):
    profiles, _, values = synthetic_samples()
    with pytest.raises(ValueError):
        api._weighted_overlap(profiles, weights, values)


def test_weighted_overlap_rejects_oversized_sample_count():
    with pytest.raises(ValueError):
        api._weighted_overlap(np.ones((4609, 3)), np.ones(4609), np.ones(4609))


@pytest.mark.parametrize("dtype", [np.int64, np.uint32, np.float32, np.float64, np.complex128])
def test_numeric_dtype_kinds_and_noncontiguous_arrays_are_supported(dtype):
    profiles = np.ones((8, 3), dtype=dtype)[::2]
    weights = np.ones(8, dtype=np.int64)[::2]
    values = np.ones(8, dtype=dtype)[::2]
    result = api._weighted_overlap(profiles, weights, values)
    close(result, 4 * np.ones((3, 3)))
    assert_owned(result, profiles, weights, values)


@pytest.mark.parametrize("seam", ["_gram_matrix", "_phase_leakage", "_spectral_record", "_matrix_comparison"])
@pytest.mark.parametrize("invalid_kind", ["list", "subclass", "trap", "bool", "object", "string", "shape", "nan"])
def test_compressed_matrix_seams_enforce_array_contract(seam, invalid_kind):
    invalid = {"list": [[1, 0, 0]] * 3, "subclass": np.eye(3).view(ArraySubclass),
               "trap": ConversionTrap(), "bool": np.eye(3, dtype=bool),
               "object": np.eye(3, dtype=object), "string": np.ones((3, 3), dtype="U1"),
               "shape": np.eye(2), "nan": np.full((3, 3), np.nan)}[invalid_kind]
    args = [invalid, np.eye(3)] if seam in ("_gram_matrix", "_matrix_comparison") else [invalid]
    expected = ValueError if invalid_kind in ("shape", "nan") else TypeError
    with pytest.raises(expected):
        getattr(api, seam)(*args)
    if len(args) == 2:
        with pytest.raises(expected):
            getattr(api, seam)(args[1], args[0])


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1j, 0j, "0.1", Decimal("0.1"), Fraction(1, 10), ConversionTrap()])
def test_private_phase_epsilon_rejects_custom_and_nonreal_scalars(bad):
    profiles, weights, _ = synthetic_samples()
    with pytest.raises(TypeError):
        api._phase_overlap(profiles, weights, np.ones(4), bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 10 ** 1000])
def test_private_phase_epsilon_rejects_nonfinite_conversion(bad):
    profiles, weights, _ = synthetic_samples()
    with pytest.raises(ValueError):
        api._phase_overlap(profiles, weights, np.ones(4), bad)


@pytest.mark.parametrize("seam", ["source", "phase"])
@pytest.mark.parametrize("kind", ["list", "subclass", "complex", "bool", "object", "trap", "shape", "nan"])
def test_real_source_and_phase_vectors_are_strict(seam, kind):
    invalid = {"list": [1.0] * 4, "subclass": np.ones(4).view(ArraySubclass),
               "complex": np.ones(4, dtype=complex), "bool": np.ones(4, dtype=bool),
               "object": np.ones(4, dtype=object), "trap": ConversionTrap(),
               "shape": np.ones((4, 1)), "nan": np.full(4, np.nan)}[kind]
    with pytest.raises((TypeError, ValueError)):
        if seam == "source":
            api._source_values("p2", invalid)
        else:
            profiles, weights, _ = synthetic_samples()
            api._phase_overlap(profiles, weights, invalid, 0.1)


def test_source_outputs_are_owned_exact_pointwise_polynomials():
    z = np.array([-1.0, -0.3, 0.2, 1.0])
    old = z.copy()
    for source, expected in (("one", np.ones(4)), ("z", z), ("p2", (3 * z * z - 1) / 2)):
        result = api._source_values(source, z)
        np.testing.assert_allclose(result, expected, atol=1e-15, rtol=1e-15)
        assert_owned(result, z)
    np.testing.assert_array_equal(z, old)


def test_extended_precision_conversion_loss_is_rejected_when_representable():
    # Some hosts alias longdouble to binary64; absence is not a control failure.
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    huge = np.finfo(np.longdouble).max
    candidates = []
    if tiny != 0 and tiny < np.longdouble(np.nextafter(0.0, 1.0)):
        candidates.append(tiny)
    if huge > np.longdouble(np.finfo(float).max):
        candidates.append(huge)
    for value in candidates:
        profiles, weights, values = synthetic_samples()
        for slot in range(3):
            args = [profiles.copy(), weights.copy(), values.copy()]
            array = np.ones(args[slot].shape, dtype=np.longdouble)
            array.flat[0] = value
            args[slot] = array
            with pytest.raises(ValueError):
                api._weighted_overlap(*args)
        complex_array = np.ones((4, 3), dtype=np.clongdouble)
        complex_array.imag.flat[0] = value
        with pytest.raises(ValueError):
            api._weighted_overlap(complex_array, weights, values)
        with pytest.raises(ValueError):
            api._phase_overlap(profiles, weights, np.ones(4), value)


@pytest.mark.parametrize("observed,bound,status,literal", [
    (0.0, 1.0, "consistent", True), (1.0, 1.0, "consistent", True),
    (1.0 + 1e-10, 1.0, "consistent", False), (1.0 + 1e-8, 1.0, MISMATCH, False),
    (0.0, 0.0, "consistent", True), (1e-10, 0.0, "consistent", False),
])
def test_upper_comparison_is_one_sided_with_literal_sign(observed, bound, status, literal):
    record = api._upper_comparison(observed, bound)
    comparison(record, status, upper=True)
    assert record["signed_excess"] == observed - bound
    assert record["max_absolute_error"] == abs(observed - bound)
    assert record["literal_inequality"] is literal


@pytest.mark.parametrize("observed,bound", [(None, 1.0), (1.0, None), (None, None)])
def test_upper_missing_operands_are_inconclusive(observed, bound):
    comparison(api._upper_comparison(observed, bound), "inconclusive", upper=True)


@pytest.mark.parametrize("invalid", [True, 1j, "1", Decimal("1"), ConversionTrap(), -1.0, float("nan"), float("inf")])
def test_upper_comparison_validates_each_scalar(invalid):
    for observed, bound in ((invalid, 1.0), (1.0, invalid)):
        with pytest.raises((TypeError, ValueError)):
            api._upper_comparison(observed, bound)


def test_matrix_comparison_uses_operator_norm_and_reference_scale():
    reference_matrix = 100 * np.eye(3)
    difference = np.diag([1.9e-8, 1.9e-8, 1.9e-8])
    record = api._matrix_comparison(reference_matrix + difference, reference_matrix)
    comparison(record)
    expected = norm(reference_matrix + difference - reference_matrix)
    close(record["max_absolute_error"], expected)
    assert expected < ATOL + RTOL * norm(reference_matrix)
    assert np.linalg.norm(reference_matrix + difference - reference_matrix, "fro") > ATOL + RTOL * norm(reference_matrix)
    comparison(api._matrix_comparison(3e-10 * np.eye(3), np.zeros((3, 3))), MISMATCH)
    comparison(api._matrix_comparison(np.zeros((3, 3)), np.eye(3)), MISMATCH)
    for left, right in ((None, np.eye(3)), (np.eye(3), None), (None, None)):
        comparison(api._matrix_comparison(left, right), "inconclusive")


@pytest.mark.parametrize("bad_norm", [-1.0, float("nan"), float("inf"), True, 1j, None])
def test_invalid_norm_outputs_are_unavailable_not_negative_observables(monkeypatch, bad_norm):
    monkeypatch.setattr(api, "_operator_norm", lambda matrix: bad_norm)
    comparison(api._matrix_comparison(np.eye(3), np.eye(3)), "inconclusive")
    report = api.case_report("z")
    assert report["status"] == UNAVAILABLE
    assert report["gram"]["error_from_identity"] is None
    assert report["B"] is not None and report["C"] is not None and report["G"] is not None
    for phase in report["phases"]:
        assert phase["S"] is not None and phase["L"] is not None
        assert phase["leading_order_error"] is None
        comparison(phase["leading_order_comparison"], "inconclusive", upper=True)
        comparison(phase["upper_bound_comparison"], "inconclusive", upper=True)
    native(report)


def inject_nth(monkeypatch, seam, index, exception=None, transform=None):
    original = getattr(api, seam)
    calls = []

    def wrapper(*args, **kwargs):
        calls.append(args)
        if len(calls) == index:
            if exception is not None:
                raise exception
            return transform(original(*args, **kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(api, seam, wrapper)
    return calls


def assert_phase_retained(phase):
    assert phase["S"] is not None and phase["L"] is not None
    assert phase["epsilon"] in EPSILONS


def test_sample_system_failure_is_local_to_one_quadrature(monkeypatch):
    calls = inject_nth(monkeypatch, "_sampled_system", 1, api.NumericalUnavailable("synthetic system failure"))
    report = api.demonstration_report()
    assert len(calls) == 2 and len(report["cases"]) == 6
    assert report["status"] == UNAVAILABLE
    for case in report["cases"][:3]:
        assert case["status"] == UNAVAILABLE
        assert set(case["analytic"]) == CASE_ANALYTIC_KEYS
        assert case["B"] is case["C"] is case["G"] is None
        assert case["gram"]["matrix"] is None
        assert [p["epsilon"] for p in case["phases"]] == list(EPSILONS)
        for phase in case["phases"]:
            assert phase["S"] is phase["L"] is None
            assert set(phase["analytic"]) == PHASE_ANALYTIC_KEYS
    assert all(case["status"] == AVAILABLE for case in report["cases"][3:])
    for refinement in report["refinement_comparisons"]:
        assert refinement["status"] == UNAVAILABLE
        for label in ("B", "C", "G"):
            comparison(refinement[label], "inconclusive")
    native(report)


def test_source_vector_failure_preserves_other_sources_and_quadrature(monkeypatch):
    inject_nth(monkeypatch, "_source_values", 2, api.NumericalUnavailable("synthetic source failure"))
    report = api.demonstration_report()
    assert report["cases"][1]["status"] == UNAVAILABLE
    assert report["cases"][1]["analytic"]["formal_second_order_status"] == "structural_zero_by_chirality"
    assert all(case["status"] == AVAILABLE for i, case in enumerate(report["cases"]) if i != 1)
    assert report["refinement_comparisons"][1]["status"] == UNAVAILABLE
    assert report["refinement_comparisons"][0]["status"] == AVAILABLE
    assert report["refinement_comparisons"][2]["status"] == AVAILABLE


@pytest.mark.parametrize("failed_index,missing", [(2, "B"), (3, "C")])
def test_B_C_overlaps_are_independent_and_all_phases_survive(monkeypatch, failed_index, missing):
    calls = inject_nth(monkeypatch, "_weighted_overlap", failed_index,
                       api.NumericalUnavailable("synthetic " + missing + " overlap failure"))
    case = api.case_report("p2")
    assert len(calls) == 6  # T, B, C and all three S still attempted.
    assert case[missing] is None and case["G"] is None
    other = "C" if missing == "B" else "B"
    assert case[other] is not None
    comparison(case["moment_comparisons"][other])
    comparison(case["moment_comparisons"][missing], "inconclusive")
    comparison(case["moment_comparisons"]["G"], "inconclusive")
    assert case["status"] == UNAVAILABLE
    for phase in case["phases"]:
        assert_phase_retained(phase)
        assert phase["leading_order_error"] is None
        comparison(phase["leading_order_comparison"], "inconclusive", upper=True)
        comparison(phase["upper_bound_comparison"], upper=True)
    native(case)


def test_raw_T_overlap_failure_does_not_suppress_B_C_or_S(monkeypatch):
    calls = inject_nth(monkeypatch, "_weighted_overlap", 1, api.NumericalUnavailable("synthetic T failure"))
    case = api.case_report("z")
    assert len(calls) == 6
    assert case["gram"]["matrix"] is None and case["gram"]["status"] == UNAVAILABLE
    for label in ("B", "C", "G"):
        assert case[label] is not None
        comparison(case["moment_comparisons"][label])
    assert all(p["status"] == AVAILABLE for p in case["phases"])


def test_raw_T_norm_failure_does_not_suppress_other_diagnostics(monkeypatch):
    inject_nth(monkeypatch, "_operator_norm", 1, api.NumericalUnavailable("synthetic T norm failure"))
    case = api.case_report("z")
    assert case["gram"]["matrix"] is not None
    assert case["gram"]["error_from_identity"] is None
    assert case["gram"]["status"] == UNAVAILABLE
    assert case["status"] == UNAVAILABLE
    for label in ("B", "C", "G"):
        assert case[label] is not None
        comparison(case["moment_comparisons"][label])
    assert all(p["status"] == AVAILABLE for p in case["phases"])


def test_gram_mismatch_is_not_repaired_and_affects_only_dependent_quadrature(monkeypatch):
    def perturb(system):
        profiles, weights, z = system
        return 1.001 * profiles, weights, z

    inject_nth(monkeypatch, "_sampled_system", 1, transform=perturb)
    report = api.demonstration_report()
    assert report["status"] == MISMATCH
    for case in report["cases"][:3]:
        assert case["status"] == MISMATCH and case["gram"]["status"] == MISMATCH
        close(decode(case["gram"]["matrix"]), 1.001 ** 2 * np.eye(3))
        close(decode(case["B"]), 1.001 ** 2 * reference(case["source_id"])[0])
        zero = case["phases"][0]
        close(decode(zero["S"]), decode(case["gram"]["matrix"]))
        assert norm(decode(zero["L"])) > 0.001
        assert zero["analytic"]["structural_zero"] is True
    assert all(case["status"] == AVAILABLE for case in report["cases"][3:])


def test_floating_constant_and_zero_time_residuals_are_retained(monkeypatch):
    def perturb(system):
        profiles, weights, z = system
        return (1 + 1e-12) * profiles, weights, z

    inject_nth(monkeypatch, "_sampled_system", 1, transform=perturb)
    case = api.case_report("one")
    assert case["status"] == AVAILABLE
    G = decode(case["G"])
    assert np.any(G.diagonal().real < 0)
    assert case["leakage_spectrum"]["literal_psd"] is False
    assert case["analytic"]["phase_leakage_structural_zero_for_all_epsilon"] is True
    for phase in case["phases"]:
        assert phase["analytic"]["structural_zero"] is True
        assert np.any(decode(phase["L"]).diagonal().real < 0)
        assert phase["leakage_spectrum"]["literal_psd"] is False
        assert phase["upper_bound_comparison"]["literal_inequality"] is False
        assert phase["upper_bound_comparison"]["reason"]


def test_G_arithmetic_failure_preserves_B_C_and_phase_overlaps(monkeypatch):
    inject_nth(monkeypatch, "_gram_matrix", 1, api.NumericalUnavailable("synthetic G arithmetic failure"))
    case = api.case_report("z")
    assert case["B"] is not None and case["C"] is not None and case["G"] is None
    assert case["status"] == UNAVAILABLE
    for phase in case["phases"]:
        assert_phase_retained(phase)
        comparison(phase["leading_order_comparison"], "inconclusive", upper=True)
        comparison(phase["upper_bound_comparison"], upper=True)


@pytest.mark.parametrize("failed_phase", [1, 2, 3])
def test_phase_overlap_failure_is_local_to_one_epsilon(monkeypatch, failed_phase):
    calls = inject_nth(monkeypatch, "_phase_overlap", failed_phase,
                       api.NumericalUnavailable("synthetic phase overlap failure"))
    case = api.case_report("z")
    assert len(calls) == 3 and case["G"] is not None
    for i, phase in enumerate(case["phases"], 1):
        if i == failed_phase:
            assert phase["status"] == UNAVAILABLE
            assert phase["S"] is phase["L"] is None
            assert phase["reason"]
        else:
            assert phase["status"] == AVAILABLE
            assert_phase_retained(phase)


def test_L_arithmetic_failure_retains_S_and_later_phases(monkeypatch):
    inject_nth(monkeypatch, "_phase_leakage", 2, api.NumericalUnavailable("synthetic L arithmetic failure"))
    case = api.case_report("z")
    failed = case["phases"][1]
    assert failed["S"] is not None and failed["L"] is None
    assert failed["status"] == UNAVAILABLE and failed["reason"]
    comparison(failed["leading_order_comparison"], "inconclusive", upper=True)
    comparison(failed["upper_bound_comparison"], "inconclusive", upper=True)
    assert case["phases"][0]["status"] == case["phases"][2]["status"] == AVAILABLE


@pytest.mark.parametrize("failed_index", [1, 2, 3, 4])
def test_eigensystem_failure_preserves_raw_matrices_and_direct_comparisons(monkeypatch, failed_index):
    calls = inject_nth(monkeypatch, "_screened_eigensystem", failed_index,
                       np.linalg.LinAlgError("synthetic solver failure"))
    case = api.case_report("p2")
    assert len(calls) == 4
    assert case["status"] == UNAVAILABLE
    assert case["G"] is not None
    for label in ("B", "C", "G"):
        comparison(case["moment_comparisons"][label])
    if failed_index == 1:
        failed_spectrum(case["leakage_spectrum"])
    for i, phase in enumerate(case["phases"], 2):
        assert_phase_retained(phase)
        comparison(phase["leading_order_comparison"], upper=True)
        comparison(phase["upper_bound_comparison"], upper=True)
        if i == failed_index:
            failed_spectrum(phase["leakage_spectrum"])
            assert phase["status"] == UNAVAILABLE
        else:
            assert phase["status"] == AVAILABLE
    native(case)


@pytest.mark.parametrize("failure", [FloatingPointError("synthetic arithmetic"), OverflowError("synthetic overflow"),
                                     np.linalg.LinAlgError("synthetic solver"),
                                     ValueError("H fails the heuristic Hermiticity screen")])
def test_expected_spectral_failures_are_narrowly_translated(monkeypatch, failure):
    inject_nth(monkeypatch, "_screened_eigensystem", 1, failure)
    record = api._spectral_record(np.eye(3))
    failed_spectrum(record)
    assert record["hermiticity_residual"] == 0


@pytest.mark.parametrize("seam", ["_sampled_system", "_source_values", "_weighted_overlap", "_gram_matrix",
                                  "_phase_overlap", "_phase_leakage", "_screened_eigensystem", "_operator_norm", "_frobenius"])
@pytest.mark.parametrize("exception_type", [ZeroDivisionError, RuntimeError, ValueError, TypeError])
def test_programming_exceptions_propagate_without_unavailable_masking(monkeypatch, seam, exception_type):
    marker = "programming sentinel " + seam
    inject_nth(monkeypatch, seam, 1, exception_type(marker))
    with pytest.raises(exception_type, match="programming sentinel"):
        api.case_report("z")


def test_matrix_comparison_programming_zero_division_propagates(monkeypatch):
    inject_nth(monkeypatch, "_operator_norm", 1, ZeroDivisionError("programming comparison sentinel"))
    with pytest.raises(ZeroDivisionError, match="programming comparison sentinel"):
        api._matrix_comparison(np.eye(3), np.eye(3))


def test_failure_priority_chooses_first_reason_at_highest_priority(monkeypatch):
    # Raw Gram mismatch comes before moment mismatch and later numerical loss.
    inject_nth(monkeypatch, "_weighted_overlap", 1, transform=lambda matrix: 1.01 * matrix)
    inject_nth(monkeypatch, "_screened_eigensystem", 1, api.NumericalUnavailable("later solver unavailable"))
    case = api.case_report("one")
    assert case["status"] == MISMATCH
    assert case["gram"]["status"] == MISMATCH
    assert case["reason"] == case["gram"]["reason"]


def test_unavailability_chooses_first_failure_reason_in_declared_order(monkeypatch):
    inject_nth(monkeypatch, "_weighted_overlap", 2, api.NumericalUnavailable("first B overlap failure"))
    inject_nth(monkeypatch, "_phase_overlap", 1, api.NumericalUnavailable("later S overlap failure"))
    case = api.case_report("z")
    assert case["status"] == UNAVAILABLE
    assert case["reason"] == case["moment_comparisons"]["B"]["reason"]


def test_demonstration_is_detached_from_samples_and_other_reports(monkeypatch):
    retained = []
    original = api._sampled_system

    def retain(*args):
        result = original(*args)
        retained.extend(result)
        return result

    monkeypatch.setattr(api, "_sampled_system", retain)
    report = api.demonstration_report()
    snapshot = copy.deepcopy(report)
    for array in retained:
        array[:] = 0
    assert report == snapshot
    second = api.demonstration_report()
    second["cases"][0]["B"]["real"][0][0] = 77.0
    second["cases"][1]["analytic"]["B_diagonal"][0] = 88.0
    second["cases"][0]["phases"][0]["analytic"]["leading_coefficient_diagonal"][0] = 99.0
    assert report == snapshot
    assert second["cases"][3]["B"]["real"][0][0] != 77.0
    assert second["cases"][4]["analytic"]["B_diagonal"][0] != 88.0
    assert second["cases"][0]["phases"][1]["analytic"]["leading_coefficient_diagonal"][0] != 99.0
    native(report)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("which", ["leading", "upper"])
def test_phase_norm_failure_is_local_to_one_comparison(monkeypatch, which):
    original_gram = api._gram_matrix
    original_leakage = api._phase_leakage
    original_norm = api._operator_norm
    state = {"G": None, "target": None, "phase": 0, "failed": False}

    def gram(B, C):
        result = original_gram(B, C)
        state["G"] = result.copy()
        return result

    def leakage(S):
        result = original_leakage(S)
        state["phase"] += 1
        if state["phase"] == 2:
            state["target"] = result - 0.01 ** 2 * state["G"] if which == "leading" else result.copy()
        return result

    def failing_norm(matrix):
        target = state["target"]
        if target is not None and not state["failed"] and np.array_equal(matrix, target):
            state["failed"] = True
            raise api.NumericalUnavailable("isolated " + which + " norm failure")
        return original_norm(matrix)

    monkeypatch.setattr(api, "_gram_matrix", gram)
    monkeypatch.setattr(api, "_phase_leakage", leakage)
    monkeypatch.setattr(api, "_operator_norm", failing_norm)
    case = api.case_report("z")
    assert state["failed"]
    assert case["phases"][0]["status"] == case["phases"][2]["status"] == AVAILABLE
    phase = case["phases"][1]
    assert phase["status"] == UNAVAILABLE
    assert_phase_retained(phase)
    assert phase["leakage_spectrum"]["status"] == AVAILABLE
    if which == "leading":
        assert phase["leading_order_error"] is None
        comparison(phase["leading_order_comparison"], "inconclusive", upper=True)
        comparison(phase["upper_bound_comparison"], upper=True)
    else:
        assert phase["leading_order_error"] is not None
        comparison(phase["leading_order_comparison"], upper=True)
        comparison(phase["upper_bound_comparison"], "inconclusive", upper=True)


def test_refinement_mismatch_does_not_retroactively_change_cases(monkeypatch):
    original_phase = api._phase_overlap
    original_comparison = api._matrix_comparison
    state = {"phases": 0, "injected": False}

    def phase(*args):
        state["phases"] += 1
        return original_phase(*args)

    def compare(observed, reference_matrix):
        result = original_comparison(observed, reference_matrix)
        if state["phases"] == 18 and not state["injected"]:
            assert np.array_equal(reference_matrix, np.zeros((3, 3)))
            state["injected"] = True
            return {"status": MISMATCH, "reason": "synthetic refinement mismatch", "max_absolute_error": 1e-5}
        return result

    monkeypatch.setattr(api, "_phase_overlap", phase)
    monkeypatch.setattr(api, "_matrix_comparison", compare)
    report = api.demonstration_report()
    assert state["injected"] and report["status"] == MISMATCH
    assert all(case["status"] == AVAILABLE for case in report["cases"])
    assert report["refinement_comparisons"][0]["status"] == MISMATCH
    assert all(r["status"] == AVAILABLE for r in report["refinement_comparisons"][1:])
    assert report["reason"] == report["refinement_comparisons"][0]["reason"]


@pytest.mark.parametrize("seam", ["overlap", "gram", "leakage", "phase"])
def test_valid_finite_arithmetic_overflow_is_numerical_unavailable(seam):
    huge = np.finfo(float).max
    with pytest.raises(api.NumericalUnavailable):
        if seam == "overlap":
            api._weighted_overlap(np.full((4, 3), huge), np.ones(4), np.ones(4))
        elif seam == "gram":
            api._gram_matrix(np.full((3, 3), huge), np.eye(3))
        elif seam == "leakage":
            api._phase_leakage(np.full((3, 3), huge))
        else:
            api._phase_overlap(np.ones((4, 3)), np.ones(4), np.full(4, huge), huge)


@pytest.mark.parametrize("sample_count", [1, 4608])
def test_weighted_overlap_inclusive_sample_count_boundaries(sample_count):
    profiles = np.ones((sample_count, 3))
    result = api._weighted_overlap(profiles, np.ones(sample_count), np.ones(sample_count))
    close(result, sample_count * np.ones((3, 3)))
    assert_owned(result, profiles)
