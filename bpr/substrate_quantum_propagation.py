"""Finite-sector density propagation in the unchanged complete Bose ring.

Frozen contract: doc/derivations/substrate_quantum_propagation_2026-09-13.md.
The envelope concerns exact stipulated operators. Its floating evaluation,
eigensystem screens and regression comparisons are heuristic, not certified
numerical error bounds or a thermodynamic/relativistic propagation theorem.
"""
from math import comb, isfinite

import numpy as np

from bpr import substrate_fermionization
from bpr.substrate_gauge_encoding import (
    NumericalUnavailable,
    _operator_norm as _operator_norm,
    _screened_eigensystem as _screened_eigensystem,
    _unitary_columns as _unitary_columns,
)

SECTORS = ((3, 3), (4, 4), (5, 5), (8, 2))
COUPLINGS = (0.0, 0.7, 40.0)
TIMES = (0.0, 0.001, 0.01, 0.1)
C = 1.0
MAX_DIMENSION = 512
ATOL = RTOL = 2e-10
MODEL_ID = "substrate_quantum_propagation"
_EPS = float(np.finfo(np.float64).eps)
_NUMERICAL = (NumericalUnavailable, FloatingPointError, OverflowError,
              ZeroDivisionError, np.linalg.LinAlgError)
_STATUSES = ("diagnostic_mismatch", "numerical_unavailable", "available_heuristic")
_BOUND_STATUSES = ("structural_zero", "below_stipulated_cap", "trivial_cap",
                   "numerical_unavailable")
_HEURISTIC = "Exact-operator envelope; floating evaluation and regression screens are heuristic."


def _real_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) in (int, float) or isinstance(value, (np.integer, np.floating))):
        raise TypeError(name + " must be a built-in or NumPy real scalar")
    if not (type(value) is int or isinstance(value, np.integer)) and not np.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) is int or isinstance(value, np.integer)):
        raise TypeError(name + " must be a built-in or NumPy integer")
    return int(value)


def _inputs(L, N, g):
    L, N = _integer(L, "L"), _integer(N, "N")
    if N <= 0:
        raise ValueError("N must be positive for normalized density")
    if (L, N) not in SECTORS:
        raise ValueError("(L,N) must be one of the frozen complete sectors")
    source = _real_scalar(g, "g")
    ratio = ((int(source), 1) if type(source) is int or isinstance(source, np.integer)
             else source.as_integer_ratio())
    for reference in COUPLINGS:
        if ratio == reference.as_integer_ratio():
            g = reference
            break
    else:
        raise ValueError("g must have exactly the stored value 0, 0.7 or 40")
    dimension = comb(L + N - 1, N)
    if dimension > MAX_DIMENSION:
        raise ValueError("complete occupation sector exceeds dimension cap")
    return L, N, g, dimension


def _build_model(L, N, g):
    """Translate only the inherited builder's documented arithmetic wrapper."""
    try:
        return substrate_fermionization.fixed_number_model(L, N, C=C, g=g)
    except ValueError as exc:
        if str(exc).startswith("numerically unresolved"):
            raise NumericalUnavailable(str(exc)) from exc
        raise


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable(name + " is nonfinite")
    return value


def _diagonal_data(diagonal, dimension):
    """Validate bounded real data before converting, then own its storage."""
    if isinstance(diagonal, np.ndarray):
        if diagonal.ndim != 1 or diagonal.shape != (dimension,):
            raise ValueError("diagonal dimension does not match screened system")
        if diagonal.dtype.kind not in "iuf":
            raise TypeError("diagonal must have a non-bool real dtype")
        if not np.all(np.isfinite(diagonal)):
            raise ValueError("diagonal must be finite")
    elif isinstance(diagonal, (list, tuple)):
        if len(diagonal) != dimension:
            raise ValueError("diagonal dimension does not match screened system")
        for item in diagonal:
            _real_scalar(item, "diagonal entry")
    else:
        raise TypeError("diagonal must be a bounded real array or sequence")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.array(diagonal, dtype=float, copy=True)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("diagonal is not representable in binary64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("diagonal is not finitely representable in binary64")
    if any(before != 0 and after == 0 for before, after in zip(diagonal, result)):
        raise ValueError("diagonal loses a nonzero entry in binary64")
    return result


def _evolved_density(system, t, diagonal):
    """Owned U† diag(diagonal) U, including the actual numerical t=0 route.

    The system is an internal screened eigensystem, not a public cache API.
    Scalar time and all eigensystem arrays are validated by _unitary_columns.
    """
    if not isinstance(system, dict) or not {"values", "vectors"}.issubset(system):
        raise TypeError("system must be a screened eigensystem dictionary")
    values = system["values"]
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError("screened eigenvalues must be a vector")
    dimension = len(values)
    if not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError("screened system dimension exceeds bounds")
    data = _diagonal_data(diagonal, dimension)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            U = _unitary_columns(system, t, np.eye(dimension, dtype=complex))
            result = U.conj().T @ (data[:, None] * U)
            return np.array(_finite(result, "evolved density"), dtype=complex, copy=True)
    except (FloatingPointError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
        raise NumericalUnavailable("evolved-density arithmetic failed") from exc


def _tail_record(status, value, partial_sum, remainder_bound, terms, reason):
    return {"status": status, "value": value, "partial_sum": partial_sum,
            "remainder_bound": remainder_bound, "terms": terms, "reason": reason}


def _positive(value, name):
    if not isfinite(value):
        raise NumericalUnavailable(name + " is nonfinite")
    if value <= 0:
        raise NumericalUnavailable(name + " lost a positive intermediate")
    return value


def _capped_tail(z, distance):
    """Conservative heuristic evaluation of min(2, 2 sum_{r>=d} z**r/r!).

    Positive-series geometric remainder, no exp-minus-polynomial subtraction.
    Exact arithmetic bounds the tail by s+R; this is not outward rounded.
    A valid but unrepresentable input/intermediate is unavailable, never zero.
    """
    source = _real_scalar(z, "z")
    distance = _integer(distance, "distance")
    if source < 0:
        raise ValueError("z must be nonnegative")
    if not 1 <= distance <= 4:
        raise ValueError("distance must be an integer from 1 through 4")
    s, remainder, terms = None, None, 0
    try:
        z = float(source)
        if not isfinite(z) or (source > 0 and z == 0):
            raise NumericalUnavailable("z is not representable as finite positive binary64")
        if z == 0:
            return _tail_record("structural_zero", 0.0, 0.0, 0.0, 0,
                                "z=0 makes the exact positive-order tail zero")
        if z >= distance:
            return _tail_record("trivial_cap", 2.0, None, None, 0,
                                "z>=distance implies z**distance/distance!>=1; stipulated cap2")
        a = 1.0
        for order in range(1, distance + 1):
            product = _positive(a * z, "starting-term multiplication")
            a = _positive(product / order, "starting-term division")
        s, m, terms = a, distance, 1
        while True:
            if s >= 1:
                return _tail_record("trivial_cap", 2.0, s, None, terms,
                                    "partial sum reaches stipulated cap")
            remainder = None
            product = _positive(a * z, "next-term multiplication")
            next_term = _positive(product / (m + 1), "next-term division")
            ratio = _positive(z / (m + 2), "remainder ratio")
            denominator = _positive(1.0 - ratio, "remainder denominator")
            remainder = _positive(next_term / denominator, "remainder bound")
            threshold = _positive((8.0 * _EPS) * s, "stopping threshold")
            if remainder <= threshold:
                upper_sum = _positive(s + remainder, "tail upper sum")
                value = min(2.0, _positive(2.0 * upper_sum, "doubled tail upper sum"))
                status = "trivial_cap" if value >= 2.0 else "below_stipulated_cap"
                return _tail_record(status, value, s, remainder, terms,
                                    "positive-series geometric remainder; heuristic floating evaluation")
            if terms == 256:
                raise NumericalUnavailable("positive-series tail exhausted 256 accumulated terms")
            s = _positive(s + next_term, "partial sum")
            a, m, terms = next_term, m + 1, terms + 1
    except _NUMERICAL as exc:
        return _tail_record("numerical_unavailable", None, s, remainder, terms,
                            "tail evaluation unavailable: " + str(exc))


def _scope():
    return {"finite_sector_only": True, "site_tensor_product_claim": False,
            "thermodynamic_velocity": False, "relativistic_spacetime": False,
            "numerical_error_certified": False, "empirical_validation": False,
            "empirical_status": "empirical_test_unavailable"}


def _conventions():
    return {"basis": "full lexicographic occupation basis at fixed total N; no local cutoff",
            "evolution": "U=exp(-itH); evolved density U†dxU=exp(itH)dx exp(-itH)",
            "density": "dx=nx/N with N>0",
            "norm": "spectral/operator 2-norm, state-independent",
            "screens": "heuristic eigensystem and regression screens, not certified floating error",
            "envelope": "exact-operator distance tail with stipulated cap2; heuristic floating evaluation",
            "units": "C=1, tau=Ct; no physical length or clock supplied"}


def _aggregate(records):
    for status in _STATUSES:
        for record in records:
            if record["status"] == status:
                return status, record["reason"]
    raise ValueError("aggregate requires records with declared numeric statuses")


def _comparison(observed, bound):
    if observed is None or bound is None:
        return {"status": "inconclusive", "max_absolute_error": None,
                "reason": "Observed norm or heuristic envelope evaluation is unavailable."}
    excess = observed - bound
    consistent = excess <= ATOL + RTOL * abs(bound)
    if consistent and excess > 0:
        reason = ("Regression consistency does not satisfy the literal inequality; "
                  "positive floating excess is retained, not clipped.")
    elif consistent:
        reason = "Floating comparison is regression-consistent, not an exact-operator certificate."
    else:
        reason = "Observed norm exceeds the evaluated envelope beyond the frozen regression tolerance."
    return {"status": "consistent" if consistent else "mismatch",
            "max_absolute_error": abs(excess), "reason": reason}


def _distance_record(evolved, diagonal, z, distance, evolution_reason):
    observed, norm_reason = None, evolution_reason
    if evolved is not None:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                commutator = _finite(evolved * (diagonal[None, :] - diagonal[:, None]),
                                     "density commutator")
                observed = float(_operator_norm(commutator))
                if not isfinite(observed) or observed < 0:
                    raise NumericalUnavailable("commutator norm is nonfinite or negative")
        except _NUMERICAL as exc:
            observed = None
            norm_reason = "commutator/norm unavailable: " + str(exc)
    # Tail evaluation is independent even after evolution or norm failure.
    try:
        tail = _capped_tail(z, distance)
    except _NUMERICAL as exc:
        tail = _tail_record("numerical_unavailable", None, None, None, 0,
                            "tail evaluation unavailable: " + str(exc))
    bound = tail["value"]
    comparison = _comparison(observed, bound)
    excess = None if observed is None or bound is None else observed - bound
    if comparison["status"] == "mismatch":
        status, reason = "diagnostic_mismatch", comparison["reason"]
    elif observed is None or bound is None:
        status = "numerical_unavailable"
        reason = norm_reason if observed is None else tail["reason"]
    else:
        status, reason = "available_heuristic", comparison["reason"]
    return {"x": 0, "y": distance, "distance": distance, "status": status,
            "reason": reason, "commutator_norm": observed,
            "analytic_upper_bound": bound, "bound_evaluation": tail,
            "signed_excess": excess,
            "literal_inequality": None if excess is None else observed <= bound,
            "comparison": comparison}


def _eigensystem_metadata(system, reason):
    fields = ("orthogonality_residual", "eigenpair_residual", "tolerance", "scale")
    metadata = {"status": "available_heuristic" if system is not None else "numerical_unavailable",
                "reason": _HEURISTIC if system is not None else reason}
    for key in fields:
        metadata[key] = None if system is None else float(system[key])
    return metadata


def case_report(L, N, g):
    """One frozen complete-sector case, four times and every nonzero distance.

    All inputs are validated before model allocation. Expected numerical failures
    preserve placeholders and independent bounds; programming errors propagate.
    """
    L, N, g, dimension = _inputs(L, N, g)
    bound_norm = 2.0 * C * N + g * N * (N - 1) / 2.0
    model, system, failure = None, None, None
    try:
        model = _build_model(L, N, g)
    except _NUMERICAL as exc:
        failure = "model unavailable: " + str(exc)
    if model is not None:
        try:
            system = _screened_eigensystem(model.H)
        except _NUMERICAL as exc:
            failure = "eigensystem unavailable: " + str(exc)
    times = []
    for t in TIMES:
        evolved, evolution_reason = None, failure
        z = 2.0 * bound_norm * abs(t)
        if system is not None:
            try:
                reference = np.array([state[0] / N for state in model.basis], dtype=float)
                evolved = _evolved_density(system, t, reference)
            except _NUMERICAL as exc:
                evolution_reason = "density evolution unavailable: " + str(exc)
        distances = []
        for distance in range(1, L // 2 + 1):
            diagonal = (None if evolved is None else
                        np.array([state[distance] / N for state in model.basis], dtype=float))
            distances.append(_distance_record(evolved, diagonal, z, distance, evolution_reason))
        status, reason = _aggregate(distances)
        times.append({"t": t, "tau": C * t, "z": z, "status": status,
                      "reason": reason, "distances": distances})
    status, reason = _aggregate(times)
    return {"model_id": MODEL_ID, "L": L, "N": N, "C": C, "g": g,
            "dimension": dimension, "hamiltonian_norm_bound": bound_norm,
            "status": status, "reason": reason, "conventions": _conventions(),
            "scope": _scope(), "eigensystem": _eigensystem_metadata(system, failure),
            "times": times}


def demonstration_report():
    """Owned JSON-native report: 12 models/eigensystems, 48 evolves, 108 norms.

    These are successful-path work budgets, not a reason to omit unavailable
    records. No ambient matrices, propagators, or mutable caller caches escape.
    """
    cases = [case_report(L, N, g) for L, N in SECTORS for g in COUPLINGS]
    case_counts = dict.fromkeys(_STATUSES, 0)
    observation_counts = dict.fromkeys(_STATUSES, 0)
    bound_counts = dict.fromkeys(_BOUND_STATUSES, 0)
    for case in cases:
        case_counts[case["status"]] += 1
        for time_record in case["times"]:
            for observation in time_record["distances"]:
                observation_counts[observation["status"]] += 1
                bound_counts[observation["bound_evaluation"]["status"]] += 1
    status, reason = _aggregate(cases)
    return {"model_id": MODEL_ID, "status": status, "reason": reason, "scope": _scope(),
            "controls": {"sectors": [list(sector) for sector in SECTORS],
                         "couplings": list(COUPLINGS), "times": list(TIMES),
                         "C": C, "reference_site": 0},
            "cases": cases,
            "summary": {"case_status_counts": case_counts,
                        "observation_status_counts": observation_counts,
                        "bound_status_counts": bound_counts}}
