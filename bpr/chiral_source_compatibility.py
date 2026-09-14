"""Supplied internal sphere source compatibility, not spacetime chirality.

Frozen contract: doc/derivations/chiral_source_compatibility_2026-09-13.md.
Raw compressions and leakage diagnostics are retained without repair. Exact
operator statements do not certify floating arithmetic or empirical physics.
Conjugating the odd Dirac operator transports its kernel and preserves internal
chirality. Adding a scalar instead has first-order shifts epsilon*B; the formal
second-order term vanishes by chirality, not because the leakage Gram vanishes.
No nonzero-mode Dirac operator is constructed here.
"""
from math import isfinite, sqrt

import numpy as np

from bpr import chiral_flavor_prototype
from bpr.substrate_gauge_encoding import (
    NumericalUnavailable,
    _frobenius as _frobenius,
    _operator_norm as _operator_norm,
    _screened_eigensystem as _screened_eigensystem,
)

MODEL_ID = "chiral_source_compatibility"
SOURCES = ("one", "z", "p2")
QUADRATURES = ((24, 48), (48, 96))
EPSILONS = (0.0, 0.01, 0.1)
Q = 3
R = 1.0
ATOL = RTOL = 2e-10
MAX_SAMPLES = 4608
_STATUSES = ("diagnostic_mismatch", "numerical_unavailable", "available_heuristic")
_NUMERICAL = (NumericalUnavailable, FloatingPointError, OverflowError,
              np.linalg.LinAlgError)
_HEURISTIC = "Floating consistency is heuristic, not certified numerical error."


def _source_id(source_id):
    if type(source_id) is not str:
        raise TypeError("source_id must be a canonical built-in string")
    if source_id not in SOURCES:
        raise ValueError("source_id must be one, z or p2")
    return source_id


def _quadrature(n_polar, n_azimuth):
    for value in (n_polar, n_azimuth):
        if isinstance(value, (bool, np.bool_)) or not (
                type(value) is int or isinstance(value, np.integer)):
            raise TypeError("quadrature orders must be built-in or NumPy integers")
    pair = (int(n_polar), int(n_azimuth))
    if pair not in QUADRATURES:
        raise ValueError("quadrature must be (24,48) or (48,96)")
    return pair


def _real_scalar(value, name, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) in (int, float) or isinstance(value, (np.integer, np.floating))):
        raise TypeError(name + " must be a built-in or NumPy real scalar")
    if not (type(value) is int or isinstance(value, np.integer)) and not np.isfinite(value):
        raise ValueError(name + " must be finite")
    if nonnegative and value < 0:
        raise ValueError(name + " must be nonnegative")
    try:
        result = float(value)
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError(name + " is not representable in binary64") from exc
    if not isfinite(result) or (value != 0 and result == 0):
        raise ValueError(name + " is not representable in binary64")
    return result


def _array(value, name, kind, real=False, samples=None):
    """Bound base-array shape before conversion; own every returned array."""
    if type(value) is not np.ndarray:
        raise TypeError(name + " must be a base NumPy array")
    if kind == "matrix":
        valid = value.shape == (3, 3)
    elif kind == "profiles":
        valid = value.ndim == 2 and value.shape[1] == 3 and 1 <= value.shape[0] <= MAX_SAMPLES
    elif kind == "vector":
        valid = value.ndim == 1 and 1 <= value.shape[0] <= MAX_SAMPLES
    else:
        raise ValueError("unknown internal array kind")
    if not valid or (samples is not None and value.shape[0] != samples):
        raise ValueError(name + " has an invalid or mismatched shape")
    if value.dtype.kind not in ("iuf" if real else "iufc"):
        raise TypeError(name + " must have a non-bool " + ("real" if real else "numeric") + " dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(name + " must be finite")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.array(value, dtype=complex if value.dtype.kind == "c" else float,
                              copy=True)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(name + " is not representable in binary64") from exc
    if not np.all(np.isfinite(result)) or any(
            np.any((before != 0) & (after == 0)) for before, after in
            ((value.real, result.real), (value.imag, result.imag))):
        raise ValueError(name + " loses a nonzero component or overflows in binary64")
    return result


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable(name + " is nonfinite")
    return value


def _arithmetic(name, operation, *args):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _finite(operation(*args), name)
    except _NUMERICAL as exc:
        raise NumericalUnavailable(name + " unavailable: " + str(exc)) from exc


def _attempt(name, operation, *args):
    """Only expected numerical failures are isolated, never programming errors."""
    try:
        return operation(*args), None
    except _NUMERICAL as exc:
        return None, name + " unavailable: " + str(exc)


def _sample_inputs(profiles, weights):
    profiles = _array(profiles, "profiles", "profiles")
    weights = _array(weights, "weights", "vector", real=True, samples=len(profiles))
    if np.any(weights <= 0):
        raise ValueError("weights must be strictly positive")
    return profiles, weights


def _sampled_system(n_polar, n_azimuth):
    n_polar, n_azimuth = _quadrature(n_polar, n_azimuth)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            theta, phi, weights = chiral_flavor_prototype.sphere_quadrature(
                R=R, n_polar=n_polar, n_azimuth=n_azimuth)
            profiles = chiral_flavor_prototype.monopole_profiles(theta, phi, q=Q, R=R)
            z = np.cos(theta)
            profiles, weights = _sample_inputs(_finite(profiles, "profiles"),
                                               _finite(weights, "weights"))
            z = _array(_finite(z, "z"), "z", "vector", real=True, samples=len(profiles))
            return profiles, weights, z
    except _NUMERICAL as exc:
        raise NumericalUnavailable("sample construction unavailable: " + str(exc)) from exc


def _source_values(source_id, z):
    source_id = _source_id(source_id)
    z = _array(z, "z", "vector", real=True)
    if source_id == "one":
        return np.ones_like(z)
    if source_id == "z":
        return z
    return _arithmetic("quadratic source", lambda: (3.0 * z * z - 1.0) / 2.0)


def _weighted_overlap(profiles, weights, values):
    profiles, weights = _sample_inputs(profiles, weights)
    values = _array(values, "values", "vector", samples=len(profiles))
    result = _arithmetic("weighted overlap", lambda:
                         profiles.conj().T @ ((weights * values)[:, None] * profiles))
    return np.array(result, dtype=complex, copy=True)


def _phase_overlap(profiles, weights, h_values, epsilon):
    epsilon = _real_scalar(epsilon, "epsilon")
    profiles, weights = _sample_inputs(profiles, weights)
    h_values = _array(h_values, "h_values", "vector", real=True, samples=len(profiles))
    phases = _arithmetic("phase exponential", lambda: np.exp(-1j * epsilon * h_values))
    return _weighted_overlap(profiles, weights, phases)


def _gram_matrix(B, C):
    B, C = _array(B, "B", "matrix"), _array(C, "C", "matrix")
    result = _arithmetic("leakage Gram", lambda: C - B @ B)
    return np.array(result, dtype=complex, copy=True)


def _phase_leakage(S):
    S = _array(S, "S", "matrix")
    result = _arithmetic("phase leakage", lambda: np.eye(3) - S.conj().T @ S)
    return np.array(result, dtype=complex, copy=True)


def _norm(matrix, operation):
    # Supply the narrow inherited aliases at call time to retain fixture seams.
    matrix = _array(matrix, "norm input", "matrix")
    value = operation(matrix)
    try:
        return _real_scalar(value, "norm output", nonnegative=True)
    except (TypeError, ValueError) as exc:
        raise NumericalUnavailable("norm returned invalid output: " + str(exc)) from exc


def _inconclusive(reason):
    return {"status": "inconclusive", "max_absolute_error": None, "reason": reason}


def _matrix_comparison(observed, reference):
    observed = None if observed is None else _array(observed, "observed", "matrix")
    reference = None if reference is None else _array(reference, "reference", "matrix")
    if observed is None or reference is None:
        return _inconclusive("A matrix comparison operand is unavailable.")
    try:
        difference = _arithmetic("matrix difference", lambda: observed - reference)
        error = _norm(difference, _operator_norm)
        reference_scale = _norm(reference, _operator_norm)
        threshold = _arithmetic("comparison threshold", lambda: ATOL + RTOL * reference_scale)
    except _NUMERICAL as exc:
        return _inconclusive("Matrix comparison unavailable: " + str(exc))
    return {"status": "consistent" if error <= threshold else "diagnostic_mismatch",
            "max_absolute_error": error,
            "reason": _HEURISTIC if error <= threshold else "Matrix disagreement exceeds heuristic tolerance."}


def _upper_comparison(observed, bound):
    observed = None if observed is None else _real_scalar(observed, "observed", nonnegative=True)
    bound = None if bound is None else _real_scalar(bound, "bound", nonnegative=True)
    if observed is None or bound is None:
        record = _inconclusive("A scalar inequality operand is unavailable.")
        record.update(signed_excess=None, literal_inequality=None)
        return record
    try:
        excess = _arithmetic("signed excess", lambda: observed - bound)
        threshold = _arithmetic("inequality threshold", lambda: ATOL + RTOL * abs(bound))
    except _NUMERICAL as exc:
        record = _inconclusive("Scalar comparison unavailable: " + str(exc))
        record.update(signed_excess=None, literal_inequality=None)
        return record
    consistent = excess <= threshold
    reason = _HEURISTIC
    if consistent and excess > 0:
        reason = ("Positive floating excess is within heuristic tolerance but the literal "
                  "inequality is false; the excess is retained, not clipped.")
    elif not consistent:
        reason = "Upper-bound excess exceeds heuristic tolerance."
    return {"status": "consistent" if consistent else "diagnostic_mismatch",
            "max_absolute_error": abs(excess), "reason": reason,
            "signed_excess": excess, "literal_inequality": observed <= bound}


def _empty_spectrum(reason, hermiticity=None):
    return {"status": "numerical_unavailable", "reason": reason, "eigenvalues": None,
            "hermiticity_residual": hermiticity,
            "eigensystem": {"status": "numerical_unavailable", "reason": reason,
                            "orthogonality_residual": None, "eigenpair_residual": None,
                            "tolerance": None, "scale": None},
            "psd_status": "numerical_unavailable", "literal_psd": None,
            "lambda_min": None, "lambda_max": None, "sqrt_lambda_max_candidate": None}


def _spectral_record(matrix):
    matrix = _array(matrix, "spectral input", "matrix")
    hermiticity = None
    try:
        asymmetry = _arithmetic("Hermiticity residual matrix", lambda: matrix - matrix.conj().T)
        hermiticity = _norm(asymmetry, _frobenius)
        try:
            system = _screened_eigensystem(matrix)
        except ValueError as exc:
            if str(exc) == "H fails the heuristic Hermiticity screen":
                raise NumericalUnavailable(str(exc)) from exc
            raise
        # The inherited solver screens its outputs. Validate metadata too so an
        # invalid diagnostic cannot escape into strict JSON through this seam.
        values = system["values"]
        if type(values) is not np.ndarray or values.shape != (3,) or values.dtype.kind not in "iuf":
            raise NumericalUnavailable("eigensystem returned invalid eigenvalues")
        try:
            values = _array(values, "eigenvalues", "vector", real=True, samples=3)
            metadata = {key: _real_scalar(system[key], key, nonnegative=True) for key in
                        ("orthogonality_residual", "eigenpair_residual", "tolerance", "scale")}
        except (TypeError, ValueError) as exc:
            raise NumericalUnavailable("eigensystem returned invalid diagnostics: " + str(exc)) from exc
        low, high = float(np.min(values)), float(np.max(values))
        threshold = _arithmetic("PSD threshold", lambda: ATOL + RTOL * float(np.max(np.abs(values))))
    except _NUMERICAL as exc:
        return _empty_spectrum("Spectrum unavailable: " + str(exc), hermiticity)
    if low < -threshold:
        status, psd = "diagnostic_mismatch", "diagnostic_mismatch"
        reason = "Raw negative eigenvalue exceeds the heuristic PSD tolerance."
    elif low < 0:
        status, psd = "available_heuristic", "negative_within_tolerance"
        reason = "Raw negative eigenvalue is within tolerance; literal PSD is false, without clipping."
    else:
        status, psd = "available_heuristic", "psd_consistent_heuristic"
        reason = _HEURISTIC
    metadata.update(status="available_heuristic", reason=_HEURISTIC)
    return {"status": status, "reason": reason, "eigenvalues": values.tolist(),
            "hermiticity_residual": hermiticity, "eigensystem": metadata,
            "psd_status": psd, "literal_psd": low >= 0, "lambda_min": low,
            "lambda_max": high, "sqrt_lambda_max_candidate": sqrt(high) if high >= 0 else None}


def _aggregate(records):
    aliases = {"consistent": "available_heuristic", "inconclusive": "numerical_unavailable"}
    for status in _STATUSES:
        for record in records:
            if aliases.get(record["status"], record["status"]) == status:
                return status, record["reason"]
    raise ValueError("aggregation requires declared diagnostic records")


def _scope():
    return {"supplied_internal_sphere": True, "physical_spacetime_chirality": False,
            "standard_model_derivation": False, "full_dirac_solver": False,
            "numerical_error_certified": False, "empirical_validation": False,
            "empirical_status": "empirical_test_unavailable"}


def _analytic(source_id):
    B, C, G, norm, width = {
        "one": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.0, 0.0),
        "z": ([0.5, 0.0, -0.5], [2/5, 1/5, 2/5], [3/20, 1/5, 3/20], sqrt(1/5), 2.0),
        "p2": ([1/10, -1/5, 1/10], [8/35, 1/7, 8/35],
               [153/700, 18/175, 153/700], sqrt(153/700), 1.5),
    }[source_id]
    return {"B_diagonal": B, "C_diagonal": C, "G_diagonal": G,
            "leakage_norm": norm, "h_range_width": width, "source_norm": 1.0,
            "phase_leakage_structural_zero_for_all_epsilon": source_id == "one",
            "structural_zero_reason": "A constant source is a common phase on the entire kernel."
            if source_id == "one" else None,
            "first_order_shift_coefficients": list(B),
            "formal_second_order_status": "structural_zero_by_chirality"}


def _phase_analytic(source_id, epsilon, analytic):
    gram_norm = max(analytic["G_diagonal"])
    zero = source_id == "one" or epsilon == 0
    reason = ("At epsilon=0 the exact phase operator is the identity." if epsilon == 0
              else analytic["structural_zero_reason"])
    return {"leading_coefficient_diagonal": list(analytic["G_diagonal"]),
            "upper_bound": min(1.0, epsilon**2 * gram_norm),
            "remainder_bound": epsilon**4 * analytic["h_range_width"]**2 * gram_norm / 12,
            "structural_zero": zero, "reason": reason if zero else None}


def _matrix_record(matrix):
    if matrix is None:
        return None
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _prepare(n_polar, n_azimuth):
    sampled, sample_reason = _attempt("sampled system", _sampled_system, n_polar, n_azimuth)
    T, reason = None, sample_reason
    if sampled is not None:
        profiles, weights, z = sampled
        T, reason = _attempt("sampled Gram overlap", _weighted_overlap,
                             profiles, weights, np.ones_like(z))
    comparison = _matrix_comparison(T, np.eye(3))
    if reason is not None:
        comparison = _inconclusive(reason)
    status, reason = _aggregate([comparison])
    gram = {"status": status, "reason": reason, "matrix": _matrix_record(T),
            "error_from_identity": comparison["max_absolute_error"], "comparison": comparison}
    return sampled, sample_reason, gram


def _spectrum_or_unavailable(matrix, reason):
    if matrix is None:
        return _empty_spectrum(reason or "Input matrix is unavailable.")
    record, failure = _attempt("spectrum", _spectral_record, matrix)
    return _empty_spectrum(failure) if record is None else record


def _phase_report(source_id, epsilon, analytic, sampled, h_values, source_reason, G):
    phase_analytic = _phase_analytic(source_id, epsilon, analytic)
    S, L = None, None
    assembly_reason = source_reason
    if sampled is not None and h_values is not None:
        S, assembly_reason = _attempt("phase overlap", _phase_overlap,
                                      sampled[0], sampled[1], h_values, epsilon)
        if S is not None:
            L, assembly_reason = _attempt("phase leakage", _phase_leakage, S)
    spectrum = _spectrum_or_unavailable(L, assembly_reason)
    leading, leading_reason = None, None
    if L is not None and G is not None:
        difference, leading_reason = _attempt("leading-order difference", _arithmetic,
                                              "leading-order difference", lambda: L - epsilon**2 * G)
        if difference is not None:
            leading, leading_reason = _attempt("leading-order norm", _norm, difference, _operator_norm)
    leading_comparison = _upper_comparison(leading, phase_analytic["remainder_bound"])
    if leading_reason is not None:
        leading_comparison["reason"] = leading_reason
    upper, upper_reason = (None, None) if L is None else _attempt(
        "phase leakage norm", _norm, L, _operator_norm)
    upper_comparison = _upper_comparison(upper, phase_analytic["upper_bound"])
    if upper_reason is not None:
        upper_comparison["reason"] = upper_reason
    records = [spectrum, leading_comparison, upper_comparison]
    if assembly_reason is not None:
        records.insert(0, {"status": "numerical_unavailable", "reason": assembly_reason})
    status, reason = _aggregate(records)
    return {"epsilon": epsilon, "status": status, "reason": reason, "analytic": phase_analytic,
            "S": _matrix_record(S), "L": _matrix_record(L), "leakage_spectrum": spectrum,
            "leading_order_error": leading, "leading_order_comparison": leading_comparison,
            "upper_bound_comparison": upper_comparison}, {"S": S, "L": L}


def _case(source_id, n_polar, n_azimuth, prepared):
    # This internal shared preparation is never accepted from a public caller.
    from copy import deepcopy

    sampled, sample_reason, gram = prepared
    analytic = _analytic(source_id)
    h_values, source_reason = None, sample_reason
    if sampled is not None:
        h_values, source_reason = _attempt("source vector", _source_values, source_id, sampled[2])
    raw = {"B": None, "C": None, "G": None, "phases": []}
    failures = {key: source_reason for key in ("B", "C", "G")}
    if h_values is not None:
        raw["B"], failures["B"] = _attempt("B overlap", _weighted_overlap,
                                           sampled[0], sampled[1], h_values)
        squared, failures["C"] = _attempt("pointwise source square", _arithmetic,
                                          "pointwise source square", lambda: h_values * h_values)
        if squared is not None:
            raw["C"], failures["C"] = _attempt("C overlap", _weighted_overlap,
                                               sampled[0], sampled[1], squared)
        if raw["B"] is not None and raw["C"] is not None:
            raw["G"], failures["G"] = _attempt("G assembly", _gram_matrix, raw["B"], raw["C"])
        else:
            failures["G"] = failures["B"] or failures["C"]
    moments = {}
    for key in ("B", "C", "G"):
        moments[key] = _matrix_comparison(raw[key], np.diag(analytic[key + "_diagonal"]))
        if failures[key] is not None:
            moments[key] = _inconclusive(failures[key])
    spectrum = _spectrum_or_unavailable(raw["G"], failures["G"])
    phases = []
    for epsilon in EPSILONS:
        phase, phase_raw = _phase_report(source_id, epsilon, analytic, sampled, h_values,
                                         source_reason, raw["G"])
        phases.append(phase)
        raw["phases"].append(phase_raw)
    status, reason = _aggregate([gram] + [moments[key] for key in ("B", "C", "G")]
                                + [spectrum] + phases)
    return {"model_id": MODEL_ID, "source_id": source_id, "q": Q, "R": R,
            "quadrature": {"n_polar": n_polar, "n_azimuth": n_azimuth,
                           "samples": n_polar * n_azimuth},
            "status": status, "reason": reason, "scope": _scope(), "analytic": analytic,
            "gram": deepcopy(gram), "B": _matrix_record(raw["B"]), "C": _matrix_record(raw["C"]),
            "G": _matrix_record(raw["G"]), "moment_comparisons": moments,
            "leakage_spectrum": spectrum, "phases": phases}, raw


def case_report(source_id, n_polar=24, n_azimuth=48):
    """Detached raw observations and heuristic diagnostics for one fixed source."""
    source_id = _source_id(source_id)
    n_polar, n_azimuth = _quadrature(n_polar, n_azimuth)
    report, _ = _case(source_id, n_polar, n_azimuth, _prepare(n_polar, n_azimuth))
    return report


def _refinement_comparison(coarse, fine):
    if coarse is None or fine is None:
        return _inconclusive("A coarse/fine matrix is unavailable.")
    difference, reason = _attempt("coarse/fine difference", _arithmetic,
                                  "coarse/fine difference", lambda: coarse - fine)
    if reason is not None:
        return _inconclusive(reason)
    return _matrix_comparison(difference, np.zeros((3, 3)))


def _refinement(source_id, coarse, fine):
    comparisons = {key: _refinement_comparison(coarse[key], fine[key]) for key in ("B", "C", "G")}
    phases = []
    for index, epsilon in enumerate(EPSILONS):
        phase = {key: _refinement_comparison(coarse["phases"][index][key],
                                             fine["phases"][index][key]) for key in ("S", "L")}
        status, reason = _aggregate([phase["S"], phase["L"]])
        phases.append({"epsilon": epsilon, "status": status, "reason": reason, **phase})
    status, reason = _aggregate([comparisons[key] for key in ("B", "C", "G")] + phases)
    return {"source_id": source_id, "status": status, "reason": reason, **comparisons, "phases": phases}


def demonstration_report():
    """Six cases; two sampled systems, 32 overlaps and 24 screened spectra."""
    cases, raw_cases = [], []
    for n_polar, n_azimuth in QUADRATURES:
        prepared = _prepare(n_polar, n_azimuth)
        for source_id in SOURCES:
            report, raw = _case(source_id, n_polar, n_azimuth, prepared)
            cases.append(report)
            raw_cases.append(raw)
    refinements = [_refinement(source_id, raw_cases[index], raw_cases[index + len(SOURCES)])
                   for index, source_id in enumerate(SOURCES)]
    status, reason = _aggregate(cases + refinements)
    return {"model_id": MODEL_ID, "status": status, "reason": reason, "scope": _scope(),
            "controls": {"q": Q, "R": R, "sources": list(SOURCES), "epsilons": list(EPSILONS),
                         "quadratures": [{"n_polar": p, "n_azimuth": a} for p, a in QUADRATURES]},
            "cases": cases, "refinement_comparisons": refinements,
            "summary": {"case_status_counts": {key: sum(case["status"] == key for case in cases)
                                                for key in _STATUSES},
                        "phase_status_counts": {key: sum(phase["status"] == key for case in cases
                                                         for phase in case["phases"])
                                                 for key in _STATUSES}}}
