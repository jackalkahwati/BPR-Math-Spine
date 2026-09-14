"""Frozen finite-record reconstruction of the connected two-source response.

Exact-input arithmetic protects supplied components, not source assembly,
libm or eigenpairs. Conditional model envelopes are not numerical enclosures.
See substrate_sampled_response_2026-09-13.md for the frozen protocol.
"""
from copy import deepcopy
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
from math import comb, cos, exp, expm1, isfinite, sin

import numpy as np

from bpr import substrate_current_response as current
from bpr import substrate_joint_source_kernel as joint
from bpr import substrate_prediction_contract as prediction

NumericalUnavailable = current.NumericalUnavailable
MAX_DIMENSION = 512
PARTITIONS = ("symmetric", "improved")
ETAS = (0.5, 1.0, 2.0)
SCHEDULES = ((4.0, 128), (8.0, 256))
_ZERO = (Fraction(0), Fraction(0))
_NUMERICAL = (NumericalUnavailable, FloatingPointError, OverflowError,
              np.linalg.LinAlgError, DecimalException)
_ONES = ((1.0, 1.0), (1.0, 1.0))
_GAINS = ((10.0, 14.0), (-15.0, -21.0))


def _scope():
    return {"empirical_status": "empirical_test_unavailable", "empirical_validation": False,
            "numerical_error_certified": False,
            "analytic_bounds": "exact-model conditional formulas; floating evaluation not enclosed",
            "assembly_proxy": "heuristic, excludes eigensystem certification",
            "contacts": "not reconstructed", "clock": "model C units only"}


def _leaf(value, name, real_only=False):
    allowed = type(value) in (int, float, complex) or (
        isinstance(value, np.generic) and value.dtype.kind in "iufc"
        and type(value) is value.dtype.type)
    if not allowed or isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must contain ordinary numeric scalars")
    if isinstance(value, np.generic):
        limit = 16 if value.dtype.kind == "c" else 8
        if value.dtype.itemsize > limit:
            raise ValueError(name + " must not use extended precision")
    if type(value) is complex or isinstance(value, np.complexfloating):
        if real_only:
            raise ValueError(name + " must have real, not complex, storage")
        return (_leaf(value.real, name, True)[0], _leaf(value.imag, name, True)[0])
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(name + " must fit finite binary64") from exc
    if not isfinite(result):
        raise ValueError(name + " must be finite")
    if isinstance(value, (int, np.integer)) and Fraction(result) != int(value):
        raise ValueError(name + " integer conversion must be exact")
    return Fraction(result), Fraction(0)


def _array(value, name, real_only=False, shape=None):
    """Validate containers, leaves and bounded rectangular shape before allocation."""
    count = [0]

    def visit(item, depth):
        if depth > 4:
            raise ValueError(name + " has too many axes")
        if type(item) is np.ndarray:
            if item.dtype.kind not in "iufc" or (real_only and item.dtype.kind == "c"):
                raise ValueError(name + " has invalid storage")
            if item.dtype.itemsize > (16 if item.dtype.kind == "c" else 8):
                raise ValueError(name + " must not use extended precision")
            if not 1 <= item.size <= MAX_DIMENSION**2:
                raise ValueError(name + " exceeds component cap")
            count[0] += int(item.size)
            if count[0] > MAX_DIMENSION**2:
                raise ValueError(name + " exceeds component cap")
            return item.shape, [_leaf(x, name, real_only) for x in item.flat]
        if type(item) in (list, tuple):
            if not 1 <= len(item) <= MAX_DIMENSION**2:
                raise ValueError(name + " has an empty or oversized axis")
            children = [visit(x, depth + 1) for x in item]
            subshape = children[0][0]
            if any(s != subshape for s, _ in children):
                raise ValueError(name + " must be rectangular")
            return (len(item),) + subshape, [x for _, leaves in children for x in leaves]
        count[0] += 1
        if count[0] > MAX_DIMENSION**2:
            raise ValueError(name + " exceeds component cap")
        return (), [_leaf(item, name, real_only)]

    if type(value) not in (np.ndarray, list, tuple):
        raise ValueError(name + " must be a base ndarray or builtin list/tuple")
    actual, leaves = visit(value, 0)
    if shape is not None and not (shape(actual) if callable(shape) else actual == shape):
        raise ValueError(name + " has wrong shape")
    dtype = float if real_only else complex
    values = [float(x[0]) if real_only else complex(float(x[0]), float(x[1])) for x in leaves]
    return np.array(values, dtype=dtype).reshape(actual)


def _scalar(value, name, positive=False, nonnegative=False):
    result = _leaf(value, name, True)[0]
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(name + " outside supported range")
    return float(result)


def _intervals(value):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) is int or isinstance(value, np.integer) and type(value) is value.dtype.type):
        raise ValueError("intervals must be an integer")
    if not 1 <= value <= 256:
        raise ValueError("intervals must be in 1..256")
    return int(value)


def _spectral_inputs(gaps, transitions):
    gaps = _array(gaps, "gaps", True, lambda s: len(s) == 1 and 1 <= s[0] <= 511)
    if gaps.ndim != 1 or not 1 <= len(gaps) <= 511 or np.any(gaps <= 0):
        raise ValueError("gaps must be a positive vector of length 1..511")
    transitions = _array(transitions, "transitions", shape=(len(gaps), 2))
    return gaps, transitions


def _times(times):
    times = _array(times, "times", True, lambda s: len(s) == 1 and 1 <= s[0] <= 257)
    if times.ndim != 1 or not 1 <= len(times) <= 257 or np.any(times < 0):
        raise ValueError("times must be a nonnegative vector of length 1..257")
    if np.any(times[1:] <= times[:-1]):
        raise ValueError("times must be strictly increasing")
    return times


def _pair(value):
    return Fraction(float(value.real)), Fraction(float(value.imag))


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _subtract(a, b):
    return a[0] - b[0], a[1] - b[1]


def _factor(pair, factor):
    return pair[0] * factor, pair[1] * factor


def _grams(transitions):
    # Rational pairs, not a binary64 Gram that could lose intermediate products.
    rows = [[_pair(value) for value in row] for row in transitions]
    return [[current._rmul(current._rconj(row[a]), row[b])
             for a in range(2) for b in range(2)] for row in rows]


def _sqrt(value):
    if value < 0:
        raise NumericalUnavailable("negative computed bound")
    with localcontext() as context:
        context.prec = 80
        result = prediction._decimal(value).sqrt()
        if value and not result:
            raise NumericalUnavailable("nonzero square-root underflow")
        return Fraction(result)


def _modulus(pair):
    return _sqrt(prediction._abs2(pair))


def _phase(value):
    result = prediction._computed(value, "phase")
    if abs(value) > 2**20:
        raise NumericalUnavailable("required phase exceeds fixed range 2**20")
    return result


def _decay(value):
    if value == 0:
        return Fraction(1)
    argument = prediction._computed(value, "exponential argument")
    result = exp(-argument)
    if result == 0:
        raise NumericalUnavailable("required exponential has nonzero underflow")
    return Fraction(result)


def _finite_factor(nu, delta, theta):
    """Stable exact-lifted numerator divided by nu+i*delta."""
    a, b = nu * theta, delta * theta
    phase = _phase(b)
    decay = _decay(a)
    loss = -expm1(-prediction._computed(a, "expm1 argument"))
    if a and loss == 0:
        raise NumericalUnavailable("positive exponential difference underflow")
    sine, cosine = Fraction(sin(phase)), Fraction(cos(phase))
    if abs(b) <= 1:
        one_minus_cos = sine**2 / (1 + cosine)
    else:
        half_sine = Fraction(sin(_phase(b / 2)))
        one_minus_cos = 2 * half_sine**2
    numerator = Fraction(loss) + decay * one_minus_cos, decay * sine
    return current._rdiv(numerator, (nu, delta))


def _time_trace(gaps, transitions, times):
    """Owned modal commutator samples, including the continuous right limit."""
    gaps, transitions = _spectral_inputs(gaps, transitions)
    times = _times(times)
    grams = _grams(transitions)
    result = np.empty((len(times), 2, 2), dtype=complex)
    try:
        for j, time in enumerate(times):
            totals = [_ZERO for _ in range(4)]
            for delta, gram in zip(gaps, grams):
                if all(w == _ZERO for w in gram):
                    continue
                phase = _phase(Fraction(float(delta)) * Fraction(float(time)))
                c, s = Fraction(cos(phase)), Fraction(sin(phase))
                for a in range(2):
                    for b in range(2):
                        positive = current._rmul(gram[2 * a + b], (c, -s))
                        negative = current._rmul(gram[2 * b + a], (c, s))
                        difference = _subtract(positive, negative)
                        totals[2 * a + b] = _add(totals[2 * a + b],
                                                (difference[1], -difference[0]))
            for index, total in enumerate(totals):
                result[j, index // 2, index % 2] = prediction._complex_result(total, "time response")
    except (FloatingPointError, OverflowError, DecimalException) as exc:
        raise NumericalUnavailable(str(exc)) from exc
    return result


def _reference(gaps, grams, nu, gains, theta=None):
    totals = [_ZERO for _ in range(4)]
    for delta, gram in zip(gaps, grams):
        delta = Fraction(float(delta))
        if all(w == _ZERO for w in gram):
            continue
        factor = _finite_factor(nu, delta, theta) if theta is not None else None
        for a in range(2):
            for b in range(2):
                gain = Fraction(float(gains[a, b]))
                if gain == 0:
                    continue
                if theta is None:
                    positive = current._rdiv(gram[2 * a + b], (-delta, nu))
                    negative = current._rdiv(gram[2 * b + a], (delta, nu))
                    term = _subtract(positive, negative)
                else:
                    positive = current._rmul(gram[2 * a + b], factor)
                    negative = current._rmul(gram[2 * b + a], current._rconj(factor))
                    difference = _subtract(positive, negative)
                    term = difference[1], -difference[0]
                totals[2 * a + b] = _add(totals[2 * a + b], _factor(term, gain))
    return np.array([prediction._complex_result(x, "reference kernel") for x in totals]).reshape(2, 2)


def _infinite_reference(gaps, grams, nu, gains):
    return _reference(gaps, grams, nu, gains)


def _finite_reference(gaps, grams, nu, gains, theta):
    return _reference(gaps, grams, nu, gains, theta)


def _grid(theta, intervals):
    prediction._computed(Fraction(theta) / intervals, "positive sample step")
    times = np.array([prediction._computed(Fraction(theta) * j / intervals, "sample time")
                      for j in range(intervals + 1)], dtype=float)
    if np.any(times[1:] <= times[:-1]):
        raise NumericalUnavailable("rounded sample grid is not strictly increasing")
    return times


def _weights(times, nu, step):
    return [step * (Fraction(1, 2) if j in (0, len(times) - 1) else 1)
            * _decay(nu * Fraction(float(time))) for j, time in enumerate(times)]


def _gain_trace(trace, gains):
    result = np.empty(trace.shape, dtype=complex)
    for j, a, b in np.ndindex(trace.shape):
        result[j, a, b] = prediction._complex_result(
            _factor(_pair(trace[j, a, b]), Fraction(float(gains[a, b]))), "gained sample")
    return result


def _perturb(trace, epsilon):
    result = np.empty(trace.shape, dtype=complex)
    for j, a, b in np.ndindex(trace.shape):
        pair = _pair(trace[j, a, b])
        result[j, a, b] = prediction._complex_result(
            (pair[0] + epsilon * (-1)**(j + a + b), pair[1]), "synthetic sample")
    return result


def _estimate(samples, weights):
    result = np.empty((2, 2), dtype=complex)
    for a, b in np.ndindex(2, 2):
        total = _ZERO
        for sample, weight in zip(samples, weights):
            total = _add(total, _factor(_pair(sample[a, b]), weight))
        result[a, b] = prediction._complex_result(total, "trapezoid estimate")
    return result


def _real_result(values, name):
    if any(value < 0 for value in values):
        raise NumericalUnavailable(name + ": negative computed bound")
    return np.array([prediction._computed(value, name) for value in values], dtype=float).reshape(2, 2)


def _tail_envelope(grams, nu, theta, gains):
    diagonal = [sum((row[3 * a][0] for row in grams), Fraction(0)) for a in range(2)]
    values = []
    decay = None
    for a, b in np.ndindex(2, 2):
        weight = _sqrt(diagonal[a] * diagonal[b]) * abs(Fraction(float(gains[a, b])))
        if weight == 0:
            values.append(Fraction(0))
        else:
            if decay is None:
                decay = _decay(nu * theta)
            values.append(2 * decay * weight / nu)
    return _real_result(values, "tail envelope")


def _quadrature_envelope(gaps, grams, nu, theta, step, gains):
    values = []
    for a, b in np.ndindex(2, 2):
        terms = sum((_modulus(row[2 * a + b]) * (nu + Fraction(float(delta)))**2
                     for delta, row in zip(gaps, grams)), Fraction(0))
        values.append(step**2 * theta * terms * abs(Fraction(float(gains[a, b]))) / 6)
    return _real_result(values, "quadrature envelope")


def _observation_envelope(weights, epsilon):
    return _real_result([epsilon * sum(weights, Fraction(0))] * 4, "observation envelope")


def _sum_budgets(*arrays):
    return _real_result([sum((Fraction(float(array[a, b])) for array in arrays), Fraction(0))
                         for a, b in np.ndindex(2, 2)], "sum of envelopes")


def _arithmetic_proxy(gaps, grams, nu, theta, weights, gains, epsilon):
    A = sum(weights, Fraction(0))
    factor = (256 * Fraction(1, 2**52) * (len(gaps) + len(weights) + 1)
              * (1 + theta * (nu + Fraction(float(max(gaps))))) * A)
    values = []
    for a, b in np.ndindex(2, 2):
        B = sum((_modulus(row[2 * a + b]) + _modulus(row[2 * b + a])
                 for row in grams), Fraction(0))
        values.append(factor * (abs(Fraction(float(gains[a, b]))) * B + epsilon))
    return _real_result(values, "arithmetic proxy")


def _ratio_empty(detail=None):
    return {"status": "dependency_unavailable", "value": None, "radius": None,
            "reason": "dependency_unavailable", "detail": detail,
            "denominator_margin_status": "not_evaluated", "conditional": True,
            "numerical_error_certified": False, "exact_denominator_gate": False,
            "center_evaluation": "rounded_non_enclosed",
            "center_convention": "exact_quotient_radius_rounded_display",
            "error_radius_evaluation": "non_enclosed"}


def _ratio_disk(kernel, entry_error):
    kernel = _array(kernel, "kernel", shape=(2, 2))
    errors = _array(entry_error, "entry_error", True, (2, 2))
    if np.any(errors < 0):
        raise ValueError("entry_error must be nonnegative")
    k = [_pair(x) for x in kernel.flat]
    u = [Fraction(float(x)) for x in errors.flat]
    n, d = current._rmul(k[1], k[2]), current._rmul(k[0], k[3])
    report = _ratio_empty()
    report["exact_denominator_gate"] = True
    if d == _ZERO:
        report.update(status="zero_denominator", denominator_margin_status="center_zero",
                      reason="denominator_center_zero", detail="Exact supplied center product is zero.")
        return report
    if not prediction._denominator_resolved(prediction._abs2(k[0]), prediction._abs2(k[3]), u[0], u[3]):
        report.update(status="unresolved_denominator", denominator_margin_status="contains_zero",
                      reason="denominator_disk_contains_zero",
                      detail="The conservative denominator product disk contains zero.")
        return report
    report["denominator_margin_status"] = "excludes_zero"
    try:
        quotient = current._rdiv(n, d)
        report["value"] = prediction._z(prediction._complex_result(quotient, "ratio center"))
        radius = prediction._ratio_proxy(k, u, n, d, quotient)
        if radius < 0:
            raise NumericalUnavailable("negative computed ratio radius")
        report["radius"] = float(radius)
    except _NUMERICAL as exc:
        report.update(status="numerical_unavailable", reason="ratio_arithmetic_unavailable", detail=str(exc))
        return report
    report.update(status="conditional_available", reason=None, detail=None)
    return report


def _attempt(function, *args):
    try:
        return function(*args), None
    except _NUMERICAL as exc:
        return None, str(exc)


def _complex_json(array):
    return None if array is None else prediction._matrix(array)


def _real_json(array):
    return None if array is None else array.tolist()


def _residual(left, right):
    return _real_result([_modulus(_subtract(_pair(left[a, b]), _pair(right[a, b])))
                         for a, b in np.ndindex(2, 2)], "diagnostic residual")


def _diagnostic(left, right, limit, proxy, detail):
    residual, failure = (None, None) if left is None or right is None else _attempt(_residual, left, right)
    result = {"status": "inconclusive", "entry_status": [["inconclusive"] * 2 for _ in range(2)],
              "residual": _real_json(residual), "analytic_limit": _real_json(limit),
              "proxy": _real_json(proxy), "reason": "dependency_unavailable",
              "detail": failure or detail or "A required diagnostic dependency is unavailable."}
    if residual is not None and limit is not None and proxy is not None:
        entries = [["within_diagnostic_envelope" if Fraction(float(residual[a, b])) <=
                    Fraction(float(limit[a, b])) + Fraction(float(proxy[a, b]))
                    else "outside_diagnostic_envelope" for b in range(2)] for a in range(2)]
        status = ("outside_diagnostic_envelope" if any("outside_diagnostic_envelope" in row
                  for row in entries) else "within_diagnostic_envelope")
        result.update(status=status, entry_status=entries, reason=None, detail=None)
    return result


def _schedule(theta, intervals):
    step, _ = _attempt(prediction._computed, Fraction(theta) / intervals, "positive sample step")
    return {"theta": float(theta), "intervals": intervals, "sample_count": intervals + 1,
            "step": step}


def _blank_record(theta, intervals, gains, epsilon, reason, detail, nu=None):
    errors = {name: None for name in ("tail", "quadrature", "observation", "total")}
    return {"partition": None, "schedule": _schedule(theta, intervals), "eta": None,
            "nu": nu, "error_level": float(epsilon), "gains": [[float(x) for x in row] for row in gains],
            "status": "numerical_unavailable", "reason": reason, "detail": detail,
            "finite_reference": None, "infinite_reference": None, "estimate": None,
            "errors": errors, "error_status": "numerical_unavailable",
            "error_reason": "envelope_unavailable", "error_detail": detail,
            "arithmetic_proxy": None, "proxy_status": "dependency_unavailable",
            "proxy_reason": "dependency_unavailable", "proxy_detail": detail,
            "diagnostics": {name: _diagnostic(None, None, None, None, detail)
                            for name in ("window", "quadrature", "total")},
            "ratio": _ratio_empty(detail), "numerical_error_certified": False}


def _acquire_owned(gaps, transitions, nu, theta, intervals, gains, error_level,
                   times, trace, trace_detail=None, prepared=None):
    """Consume only internally owned trace data; keep independent stage results."""
    record = _blank_record(theta, intervals, gains, error_level, None, None, nu)
    failures = []
    qnu, qtheta, epsilon = Fraction(nu), Fraction(theta), Fraction(error_level)
    step = qtheta / intervals
    grams = _grams(transitions)
    noiseless = samples = None
    sample_reason = sample_detail = None
    if trace is None:
        sample_reason, sample_detail = "trace_unavailable", trace_detail
    else:
        noiseless, failure = prepared if prepared is not None else _attempt(_gain_trace, trace, gains)
        if failure is None:
            samples, failure = _attempt(_perturb, noiseless, epsilon)
        if failure is not None:
            sample_reason, sample_detail = "sample_assembly_unavailable", failure
    if sample_reason:
        failures.append((sample_reason, sample_detail))
    # References and analytic envelopes do not depend on trace/sample assembly.
    finite, finite_failure = _attempt(_finite_reference, gaps, grams, qnu, gains, qtheta)
    infinite, infinite_failure = _attempt(_infinite_reference, gaps, grams, qnu, gains)
    if finite_failure or infinite_failure:
        failures.append(("reference_unavailable", finite_failure or infinite_failure))
    weights = weight_failure = None
    if times is not None:
        weights, weight_failure = _attempt(_weights, times, qnu, step)
    else:
        weight_failure = trace_detail or "Sample grid unavailable."
    estimate = None
    estimate_failure = sample_detail or weight_failure
    if samples is not None and weights is not None:
        estimate, estimate_failure = _attempt(_estimate, samples, weights)
    if estimate is None:
        failures.append(("estimate_unavailable", estimate_failure or "Samples unavailable."))
    tail, tail_failure = _attempt(_tail_envelope, grams, qnu, qtheta, gains)
    quadrature, quad_failure = _attempt(_quadrature_envelope, gaps, grams, qnu, qtheta, step, gains)
    observation = None
    obs_failure = weight_failure
    if epsilon == 0:
        observation, obs_failure = _attempt(_real_result, [Fraction(0)] * 4, "zero observation envelope")
    elif weights is not None:
        observation, obs_failure = _attempt(_observation_envelope, weights, epsilon)
    total = None
    total_failure = None
    if all(x is not None for x in (tail, quadrature, observation)):
        total, total_failure = _attempt(_sum_budgets, tail, quadrature, observation)
    budgets = {"tail": tail, "quadrature": quadrature, "observation": observation, "total": total}
    error_detail = tail_failure or quad_failure or obs_failure or total_failure
    if total is None:
        error_detail = error_detail or "A required envelope is unavailable."
        failures.append(("envelope_unavailable", error_detail))
    proxy = None
    proxy_failure = weight_failure
    if weights is not None:
        proxy, proxy_failure = _attempt(_arithmetic_proxy, gaps, grams, qnu, qtheta, weights, gains, epsilon)
    if proxy is None:
        failures.append(("arithmetic_proxy_unavailable", proxy_failure))
    ratio = _ratio_empty(error_detail or estimate_failure)
    if estimate is not None and total is not None:
        computed_ratio, ratio_failure = _attempt(_ratio_disk, estimate, total)
        if computed_ratio is None:
            ratio = _ratio_empty(ratio_failure)
            ratio.update(status="numerical_unavailable", reason="ratio_arithmetic_unavailable")
        else:
            ratio = computed_ratio
        if ratio["status"] == "numerical_unavailable":
            failures.append(("ratio_arithmetic_unavailable", ratio["detail"]))
    quad_limit = None
    if quadrature is not None and observation is not None:
        quad_limit, limit_failure = _attempt(_sum_budgets, quadrature, observation)
        if limit_failure:
            failures.append(("envelope_unavailable", limit_failure))
    detail = failures[0][1] if failures else None
    diagnostics = {
        "window": _diagnostic(finite, infinite, tail, proxy, detail),
        "quadrature": _diagnostic(estimate, finite, quad_limit, proxy, detail),
        "total": _diagnostic(estimate, infinite, total, proxy, detail),
    }
    if any(d["status"] == "inconclusive" for d in diagnostics.values()) and not failures:
        failures.append(("arithmetic_proxy_unavailable", "Diagnostic arithmetic unavailable."))
    completed = sum(x is not None for x in budgets.values())
    status = "available_conditional" if not failures else (
        "partially_unavailable" if any(x is not None for x in (estimate, finite, infinite))
        else "numerical_unavailable")
    record.update(
        status=status, reason=failures[0][0] if failures else None,
        detail=failures[0][1] if failures else None,
        finite_reference=_complex_json(finite), infinite_reference=_complex_json(infinite),
        estimate=_complex_json(estimate), errors={key: _real_json(value) for key, value in budgets.items()},
        error_status="available_conditional" if completed == 4 else (
            "partially_unavailable" if completed else "numerical_unavailable"),
        error_reason="envelope_unavailable" if completed != 4 else None, error_detail=error_detail,
        arithmetic_proxy=_real_json(proxy), proxy_status="available_heuristic" if proxy is not None else (
            "dependency_unavailable" if weights is None else "numerical_unavailable"),
        proxy_reason=None if proxy is not None else (
            "dependency_unavailable" if weights is None else "arithmetic_proxy_unavailable"),
        proxy_detail=proxy_failure, diagnostics=diagnostics, ratio=ratio)
    return {"times": None if times is None else times.copy(), "noiseless_samples": noiseless,
            "samples": samples, "sample_status": "available" if samples is not None else "numerical_unavailable",
            "sample_reason": sample_reason, "sample_detail": sample_detail, "record": record}


def _acquire(gaps, transitions, nu, theta, intervals, gains, error_level):
    gaps, transitions = _spectral_inputs(gaps, transitions)
    nu, theta = _scalar(nu, "nu", positive=True), _scalar(theta, "theta", positive=True)
    intervals = _intervals(intervals)
    gains = _array(gains, "gains", True, (2, 2))
    error_level = _scalar(error_level, "error_level", nonnegative=True)
    times, failure = _attempt(_grid, theta, intervals)
    trace = None
    if times is not None:
        trace, failure = _attempt(_time_trace, gaps, transitions, times)
    return _acquire_owned(gaps, transitions, nu, theta, intervals, gains, error_level,
                          times, trace, failure)


def _parameters(L, g, C):
    if type(L) is not int or not 3 <= L <= 5:
        raise ValueError("L must be a builtin int in 3..5")
    for name, value in (("g", g), ("C", C)):
        if type(value) not in (int, float):
            raise ValueError(name + " must be a builtin int or float")
    g, C = _scalar(g, "g", nonnegative=True), _scalar(C, "C", positive=True)
    if not 0.5 <= C <= 2 or Fraction(g) / Fraction(C) > 40:
        raise ValueError("require C in [0.5,2] and g/C in [0,40]")
    if comb(2 * L - 1, L) > MAX_DIMENSION:
        raise ValueError("complete sector exceeds dense dimension 512")
    return L, g, C


def _source_metadata():
    return {name: {"normalized_operator_frobenius_norm": None,
                   "normalized_transition_norm": None} for name in ("rho", "h")}


def _retain_frequency(record, ground_gap):
    if ground_gap is not None:
        nu, failure = _attempt(prediction._computed,
                               Fraction(record["eta"]) * Fraction(ground_gap), "frequency")
        record["nu"] = nu
        if failure is not None:
            record["detail"] = (record["detail"] + "; " if record["detail"] else "") + failure
    return record


def _blank_partition(partition, reason, detail, ground_gap=None):
    records = []
    for theta, intervals in SCHEDULES:
        for eta in ETAS:
            record = _blank_record(theta, intervals, _ONES, 0.0, reason, detail)
            record.update(partition=partition, eta=eta)
            records.append(_retain_frequency(record, ground_gap))
    return {"partition": partition, "status": "numerical_unavailable", "reason": reason,
            "detail": detail, "normalized_total_weight": None,
            "source_metadata": _source_metadata(), "records": records}


def _blank_case(L, g, C, reason, detail):
    return {"L": L, "N": L, "g": g, "C": C, "m": 1, "dimension": None,
            "status": "numerical_unavailable", "reason": reason, "detail": detail,
            "ground_gap": None, "normalized_ground_gap": None, "normalized_resolution": None,
            "partitions": [_blank_partition(partition, reason, detail) for partition in PARTITIONS],
            "scope": _scope()}


def _aggregate(children, known=False, failure=None):
    failures = [(item["reason"], item["detail"]) for item in children
                if item["status"] != "available_conditional"]
    if failure:
        failures.insert(0, failure)
    if not failures:
        return {"status": "available_conditional", "reason": None, "detail": None}
    completed = known or any(item["status"] != "numerical_unavailable" for item in children)
    return {"status": "partially_unavailable" if completed else "numerical_unavailable",
            "reason": failures[0][0], "detail": failures[0][1]}


def _normalize_gaps(system, C):
    factor = Fraction(1) / Fraction(C)
    gaps = np.array([prediction._computed(Fraction(float(x)) * factor, "normalized gap")
                     for x in system["gaps"][1:]], dtype=float)
    resolution = prediction._computed(Fraction(float(system["resolution"])) * factor,
                                     "normalized resolution")
    if not len(gaps) or np.any(gaps <= 0) or gaps[0] <= resolution:
        raise NumericalUnavailable("ground gap does not exceed inherited resolution")
    return gaps, float(gaps[0]), resolution


def _normalize_partition(data, C):
    sources = np.array([prediction._scale(source, Fraction(1) / Fraction(C)**a,
                                          "normalized source")
                        for a, source in enumerate(data["sources"])], dtype=complex)
    transitions = np.empty(data["transitions"].shape, dtype=complex)
    for a in range(2):
        transitions[:, a] = prediction._scale(data["transitions"][:, a],
                                               Fraction(1) / Fraction(C)**a,
                                               "normalized transition")
    grams = _grams(transitions)
    totals = [_ZERO for _ in range(4)]
    for row in grams:
        totals = [_add(x, y) for x, y in zip(totals, row)]
    total_weight = np.array([prediction._complex_result(x, "normalized total weight")
                             for x in totals], dtype=complex).reshape(2, 2)
    if any(total_weight[a, a].real < 0 for a in range(2)):
        raise NumericalUnavailable("negative computed diagonal weight")
    return {"sources": sources, "transitions": transitions, "total_weight": total_weight,
            "maximum_times": None, "maximum_trace": None,
            "trace_status": "numerical_unavailable", "trace_reason": "trace_unavailable",
            "trace_detail": None}


def _records_from_owned(normalized, data, partition, gains, epsilon, prepared=None):
    records = []
    if prepared is None and data["maximum_trace"] is not None:
        prepared = _attempt(_gain_trace, data["maximum_trace"], gains)
    for theta, intervals in SCHEDULES:
        times = None if data["maximum_times"] is None else data["maximum_times"][:intervals + 1]
        trace = None if data["maximum_trace"] is None else data["maximum_trace"][:intervals + 1]
        for eta in ETAS:
            nu, failure = _attempt(prediction._computed,
                                   Fraction(eta) * Fraction(normalized["ground_gap"]), "frequency")
            if failure is not None:
                record = _blank_record(theta, intervals, gains, epsilon,
                                       "normalized_data_unavailable", failure)
            else:
                acquisition = _acquire_owned(normalized["gaps"], data["transitions"], nu,
                                             theta, intervals, gains, epsilon, times, trace,
                                             data["trace_detail"],
                                             None if prepared is None else (
                                                 None if prepared[0] is None else prepared[0][:intervals + 1],
                                                 prepared[1]))
                record = acquisition["record"]
            record.update(partition=partition, eta=eta)
            records.append(record)
    return records


def _owned_case(L, g, C):
    """One owned joint system and at most one maximum trace per partition."""
    L, g, C = _parameters(L, g, C)
    if 0 < Fraction(g) / Fraction(C) < Fraction(1, 2**40):
        return {"system": None, "normalized": None,
                "report": _blank_case(L, g, C, "tiny_g_unavailable",
                                      "Positive g/C lies below inherited arithmetic resolution.")}
    report = _blank_case(L, g, C, "base_system_unavailable", "Owned system unavailable.")
    try:
        system = joint.joint_system(L, g, C, m=1)
    except NumericalUnavailable as exc:
        report = _blank_case(L, g, C, "base_system_unavailable", str(exc))
        return {"system": None, "normalized": None, "report": report}
    report["dimension"] = int(system["dimension"])
    report["ground_gap"] = float(system["gaps"][1])
    gap_data, failure = _attempt(_normalize_gaps, system, C)
    if failure is not None:
        reason = ("ground_gap_unresolved" if "ground gap" in failure
                  else "normalized_data_unavailable")
        report.update(status="partially_unavailable", reason=reason, detail=failure)
        report["partitions"] = [_blank_partition(partition, reason, failure) for partition in PARTITIONS]
        return {"system": system, "normalized": None, "report": report}
    gaps, ground_gap, resolution = gap_data
    normalized = {"gaps": gaps, "ground_gap": ground_gap, "resolution": resolution,
                  "partitions": {partition: None for partition in PARTITIONS}}
    report.update(normalized_ground_gap=ground_gap, normalized_resolution=resolution)
    for index, partition in enumerate(PARTITIONS):
        data, failure = _attempt(_normalize_partition, system["partitions"][partition], C)
        if failure is not None:
            report["partitions"][index] = _blank_partition(
                partition, "normalized_data_unavailable", failure, ground_gap)
            continue
        normalized["partitions"][partition] = data
        metadata = _source_metadata()
        metadata_failure = None
        for a, name in enumerate(("rho", "h")):
            for key, array in (("normalized_operator_frobenius_norm", data["sources"][a]),
                               ("normalized_transition_norm", data["transitions"][:, a])):
                value, failure = _attempt(prediction._norm, array)
                metadata[name][key] = value
                metadata_failure = metadata_failure or failure
        times, failure = _attempt(_grid, 8.0, 256)
        trace = None
        if times is not None:
            trace, failure = _attempt(_time_trace, gaps, data["transitions"], times)
        data.update(maximum_times=times, maximum_trace=trace,
                    trace_status="available" if trace is not None else "numerical_unavailable",
                    trace_reason=None if trace is not None else "trace_unavailable", trace_detail=failure)
        records = _records_from_owned(normalized, data, partition, np.array(_ONES), 0.0)
        public = {"partition": partition, "normalized_total_weight": _complex_json(data["total_weight"]),
                  "source_metadata": metadata, "records": records}
        public.update(_aggregate(records, known=True,
                                 failure=("normalized_data_unavailable", metadata_failure)
                                 if metadata_failure else None))
        report["partitions"][index] = public
    report.update(_aggregate(report["partitions"], known=True))
    return {"system": system, "normalized": normalized, "report": report}


def _owned_slot(L, g, C):
    try:
        return _owned_case(L, g, C)
    except NumericalUnavailable as exc:
        return {"system": None, "normalized": None,
                "report": _blank_case(L, g, C, "base_system_unavailable", str(exc))}


def case_report(L, g, C=1.0):
    """Detached JSON-native case; arbitrary models or cached reports are not inputs."""
    return deepcopy(_owned_slot(L, g, C)["report"])


def _synthetic_owned(owned):
    normalized = owned["normalized"]
    data = None if normalized is None else normalized["partitions"]["symmetric"]
    records = []
    gains = np.array(_GAINS)
    prepared = None
    if data is not None and data["maximum_trace"] is not None:
        prepared = _attempt(_gain_trace, data["maximum_trace"], gains)
    for epsilon in (0.0, 1e-8):
        if data is not None:
            records.extend(_records_from_owned(normalized, data, "symmetric", gains, epsilon, prepared))
        else:
            reference = owned["report"]["partitions"][0]
            for theta, intervals in SCHEDULES:
                for eta in ETAS:
                    record = _blank_record(theta, intervals, _GAINS, epsilon,
                                           reference["reason"], reference["detail"])
                    record.update(partition="symmetric", eta=eta)
                    records.append(_retain_frequency(
                        record, None if normalized is None else normalized["ground_gap"]))
    result = {
        "reference": {"L": 3, "N": 3, "C": 1.0, "g": 0.7, "m": 1, "partition": "symmetric"},
        "readout_gains": [2, -3], "source_gains": [5, 7], "error_levels": [0.0, 1e-8],
        "perturbation_definition": "ideal postgain delta[j,a,b]=epsilon*(-1)^(j+a+b); common j across nested schedules",
        "records": records, "scope": _scope(),
    }
    result.update(_aggregate(records))
    return result


def synthetic_report():
    """One standalone reference attempt; no cache shared with previous calls."""
    return deepcopy(_synthetic_owned(_owned_slot(3, 0.7, 1.0)))


def demonstration_report():
    """Nine prescribed owned slots, both partitions, and reused synthetic reference."""
    cases = []
    reference = None
    for L in (3, 4, 5):
        for g in (0.0, 0.7, 40.0):
            owned = _owned_slot(L, g, 1.0)
            cases.append(owned["report"])
            if L == 3 and g == 0.7:
                reference = owned
    return deepcopy({
        "module": "substrate_sampled_response", "physical_cases": cases,
        "synthetic": _synthetic_owned(reference),
        "counts": {"hamiltonian_slots": 9, "partition_slots": 18, "physical_records": 108,
                   "synthetic_records": 12, "max_samples": 257},
        "limitations": [
            "Complete finite Bose Hamiltonians and linear sources are stipulated.",
            "The second schedule lengthens the window without refining its step.",
            "Model-known spectral envelopes are not experimental bandwidth knowledge.",
            "Observation errors describe ideal postgain samples, not certified stored-array errors.",
            "Conditional ratio radii refer to exact quotients; displayed disks are not enclosed.",
            "The regulator is not dissipation; contacts and physical calibration are not reconstructed.",
            "No empirical validation, thermodynamic limit or seventh module is supplied.",
        ],
        "scope": _scope(),
    })
