"""Frozen same-model scaling and synthetic held-out observation contract.

See doc/derivations/substrate_prediction_contract_2026-09-13.md.  Exact-input
arithmetic protects supplied components, not eigenpairs or source assembly.
All comparisons are conditional software checks, never empirical validation.
"""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb

import numpy as np

from bpr import substrate_current_response as current
from bpr import substrate_energy_response as energy
from bpr import substrate_joint_source_kernel as joint
from bpr.substrate_fermionization import fixed_number_model

NumericalUnavailable = current.NumericalUnavailable
MAX_DIMENSION = 512
ETAS = (0.5, 1.0, 2.0)
SCALES = (0.5, 1.0, 2.0)
ATOL = RTOL = 2e-10
_ZERO = (Fraction(0), Fraction(0))
_DEGREES = (0, 1)
_NAMES = ("rho", "h")


def _real_input(value, name, nonnegative=False, positive=False):
    if (type(value) not in (int, float)
            and not isinstance(value, (np.integer, np.floating))):
        raise ValueError(name + " must be a finite real numeric scalar")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must not be boolean")
    if isinstance(value, (int, np.integer)):
        result = Fraction(int(value))
    else:
        if not bool(np.isfinite(value)):
            raise ValueError(name + " must be finite")
        numerator, denominator = value.as_integer_ratio()
        result = Fraction(int(numerator), int(denominator))
    if nonnegative and result < 0 or positive and result <= 0:
        raise ValueError(name + " must be " + ("positive" if positive else "nonnegative"))
    return result


def _numeric_input(value, name, real_only=False):
    if type(value) is complex or isinstance(value, np.complexfloating):
        if real_only:
            raise ValueError(name + " must have real, not complex, storage")
        return (_real_input(value.real, name), _real_input(value.imag, name))
    return _real_input(value, name), Fraction(0)


def _array_input(value, name, real_only=False):
    """Validate bounds and every leaf before coercion can erase its type/value."""
    count = [0]

    def visit(item, depth):
        if depth > 32:
            raise ValueError(name + " has too many dimensions")
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in "iufc" or real_only and item.dtype.kind == "c":
                raise ValueError(name + " must have numeric " + ("real " if real_only else "") + "storage")
            if item.size == 0 or item.size > MAX_DIMENSION ** 2:
                raise ValueError(name + " exceeds the bounded array domain")
            count[0] += int(item.size)
            if count[0] > MAX_DIMENSION ** 2:
                raise ValueError(name + " exceeds the bounded array domain")
            return item.shape, [_numeric_input(x, name, real_only) for x in item.flat]
        if isinstance(item, (list, tuple)):
            if not 1 <= len(item) <= MAX_DIMENSION ** 2:
                raise ValueError(name + " has an empty or oversized axis")
            children = [visit(x, depth + 1) for x in item]
            shape = children[0][0]
            if any(child[0] != shape for child in children):
                raise ValueError(name + " must be rectangular")
            return (len(item),) + shape, [x for child in children for x in child[1]]
        count[0] += 1
        if count[0] > MAX_DIMENSION ** 2:
            raise ValueError(name + " exceeds the bounded array domain")
        return (), [_numeric_input(item, name, real_only)]

    return visit(value, 0)


def _computed(value, name):
    """Checked final binary64 conversion, permitting representable subnormals."""
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise NumericalUnavailable(name + " is outside finite binary64 range") from exc
    if not np.isfinite(result) or result == 0 and value != 0:
        raise NumericalUnavailable(name + " has a nonrepresentable nonzero component")
    return result


def _complex_result(pair, name):
    return complex(_computed(pair[0], name), _computed(pair[1], name))


def _z(value):
    return {"real": float(value.real), "imag": float(value.imag)}


def _matrix(matrix):
    return {"shape": list(matrix.shape), "real": matrix.real.tolist(),
            "imag": matrix.imag.tolist()}


def _r(value):
    return _numeric_input(value, "computed input")


def _abs2(pair):
    return pair[0] ** 2 + pair[1] ** 2


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _norm(array):
    _, items = _array_input(array, "norm input")
    square = sum((_abs2(item) for item in items), Fraction(0))
    with localcontext() as context:
        context.prec = 80
        return _computed(_decimal(square).sqrt(), "Frobenius/amplitude norm")


def _scale(array, factor, name):
    """A single checked rounding per component, never an intermediate product."""
    result = np.empty(np.shape(array), dtype=complex)
    for index in np.ndindex(result.shape):
        pair = _r(array[index])
        result[index] = _complex_result((pair[0] * factor, pair[1] * factor), name)
    return result


def _scalar_record(status, value=None, reason=None):
    return {"status": status, "value": value, "reason": reason}


def _ratio_unavailable(status, reason):
    return {"status": status, "value": None, "reason": reason,
            "proxy": None, "witness_resolved": False}


def _source_status(amplitude, proxy, structural_zero=False):
    if structural_zero:
        return "zero_source", "source is dark by a supplied exact identity"
    if amplitude <= proxy:
        return "unresolved_source", "source amplitude does not exceed its heuristic proxy"
    return "available_heuristic", None


def _normalized_moment(weight, moment, gap, amplitude, amplitude_proxy,
                       structural_zero=False):
    """Already C-normalized inputs; no independent numerator-resolution gate."""
    if type(structural_zero) is not bool:
        raise ValueError("structural_zero must be a builtin bool")
    w, m, a, e = tuple(_real_input(value, name, nonnegative=True) for value, name in (
        (weight, "weight"), (moment, "moment"), (amplitude, "amplitude"),
        (amplitude_proxy, "amplitude_proxy")))
    delta = _real_input(gap, "gap", positive=True)
    status, reason = _source_status(a, e, structural_zero)
    if status == "available_heuristic" and w == 0:
        status, reason = "unresolved_source", "zero weight without structural proof"
    if status != "available_heuristic":
        return _scalar_record(status, reason=reason)
    try:
        return _scalar_record(status, _computed(m / (delta * w), "normalized moment"))
    except NumericalUnavailable as exc:
        return _scalar_record("numerical_unavailable", reason=str(exc))


def _gap_ratio(gaps, resolution):
    shape, values = _array_input(gaps, "gaps", real_only=True)
    if len(shape) != 1 or not 3 <= shape[0] <= MAX_DIMENSION:
        raise ValueError("gaps must be a full vector of length 3..512")
    values = [value[0] for value in values]
    screen = _real_input(resolution, "resolution", nonnegative=True)
    if values[0] != 0 or any(x < 0 for x in values) or any(
            right < left for left, right in zip(values, values[1:])):
        raise ValueError("gaps must be sorted, nonnegative and start at ground zero")
    if values[1] <= screen or values[2] <= screen:
        return _scalar_record("unresolved_gap", reason="one of the first two ground gaps is unresolved")
    try:
        return _scalar_record("available_heuristic", _computed(values[2] / values[1], "gap ratio"))
    except NumericalUnavailable as exc:
        return _scalar_record("numerical_unavailable", reason=str(exc))


def _denominator_resolved(a2, b2, ua, ub):
    """Exact radical-free test |ab| > |a|ub + |b|ua + ua*ub.

    First isolate |b|(|a|-ua) > ub(|a|+ua), then square
    with sign checks. This remains exact at equality and at extreme scales.
    """
    if a2 <= ua * ua:
        return False
    left = (b2 - ub * ub) * (a2 + ua * ua)
    right_coefficient = 2 * ua * (b2 + ub * ub)
    return left > 0 and left * left > right_coefficient ** 2 * a2


def _ratio_proxy(k, u, numerator, denominator, quotient):
    """Guarded irrational moduli, separate from the exact product-margin gate."""
    # Decimal has a wide exponent range; adaptive precision protects subtraction
    # in |D|-ED. No binary64 modulus or product is materialized along the way.
    precision = 80
    while True:
        with localcontext() as context:
            context.prec = precision
            magnitudes = [_decimal(_abs2(x)).sqrt() for x in k]
            p = [_decimal(x) for x in u]
            d = _decimal(_abs2(denominator)).sqrt()
            ed = magnitudes[0] * p[3] + magnitudes[3] * p[0] + p[0] * p[3]
            margin = d - ed
            if margin > 0 and (ed == 0 or margin > max(d, ed) * Decimal(10) ** (20 - precision)):
                en = magnitudes[1] * p[2] + magnitudes[2] * p[1] + p[1] * p[2]
                q = _decimal(_abs2(quotient)).sqrt()
                return _computed((en + q * ed) / margin, "joint ratio proxy")
        precision *= 2
        if precision > 10240:
            raise NumericalUnavailable("joint ratio proxy precision exhausted")


def _joint_ratio(kernel, entry_proxy):
    shape, k = _array_input(kernel, "kernel")
    proxy_shape, raw_proxy = _array_input(entry_proxy, "entry_proxy", real_only=True)
    if shape != (2, 2) or proxy_shape != (2, 2):
        raise ValueError("kernel and entry_proxy must both have shape (2,2)")
    u = [item[0] for item in raw_proxy]
    if any(item < 0 for item in u):
        raise ValueError("entry_proxy must be nonnegative")
    n = current._rmul(k[1], k[2])
    d = current._rmul(k[0], k[3])
    if d == _ZERO:
        return _ratio_unavailable("zero_denominator", "exact supplied-input denominator product is zero")
    if not _denominator_resolved(_abs2(k[0]), _abs2(k[3]), u[0], u[3]):
        return _ratio_unavailable("unresolved_denominator", "denominator product does not exceed its heuristic proxy")
    try:
        q = current._rdiv(n, d)
        value = _z(_complex_result(q, "joint ratio"))
        proxy = _ratio_proxy(k, u, n, d, q)
    except NumericalUnavailable as exc:
        return _ratio_unavailable("numerical_unavailable", str(exc))
    witness = _abs2(k[1]) > u[1] ** 2 and _abs2(k[2]) > u[2] ** 2
    return {"status": "available_heuristic", "value": value, "reason": None,
            "proxy": proxy, "witness_resolved": bool(witness)}


def _inconclusive(reason):
    return {"status": "inconclusive", "max_real_difference": None,
            "max_imag_difference": None, "failed_components": None, "reason": reason}


def _compare_dimensionless(observed, reference):
    shape, left = _array_input(observed, "observed")
    other_shape, right = _array_input(reference, "reference")
    if shape != other_shape:
        raise ValueError("comparison inputs must have equal shape")
    differences = [Fraction(0), Fraction(0)]
    failures = 0
    for a, b in zip(left, right):
        for component in (0, 1):
            difference = abs(a[component] - b[component])
            differences[component] = max(differences[component], difference)
            if difference > Fraction(ATOL) + Fraction(RTOL) * abs(b[component]):
                failures += 1
    try:
        real, imag = [_computed(x, "comparison component difference") for x in differences]
    except NumericalUnavailable as exc:
        return _inconclusive(str(exc))
    return {"status": "mismatch" if failures else "consistent",
            "max_real_difference": real, "max_imag_difference": imag,
            "failed_components": failures}


def _compare_optional(observed, reference):
    if observed is None or reference is None:
        return _inconclusive("a required comparison input is unavailable")
    return _compare_dimensionless(observed, reference)


def _value(record):
    value = record.get("value")
    if value is None:
        return None
    if isinstance(value, dict):
        return complex(value["real"], value["imag"])
    return value


def _decode(matrix):
    if matrix is None:
        return None
    # Do not use real + 1j*imag: that complex product can erase a component.
    real, imag = matrix["real"], matrix["imag"]
    return np.array([[complex(r, i) for r, i in zip(rr, ii)]
                     for rr, ii in zip(real, imag)], dtype=complex)


def _scope():
    return {"empirical_status": "empirical_test_unavailable",
            "numerical_error_certified": False, "empirical_validation": False,
            "gap_convention": "counted_multiplicity", "partition": "symmetric",
            "conditioning": "frozen C-normalized heuristic proxies, not error bounds",
            "comparison": "reference-only per-real/per-imaginary software tolerance",
            "identification": "no parameter injectivity or unique model identification claim"}


def _parameters(L, g, C):
    if isinstance(L, (bool, np.bool_)) or not isinstance(L, (int, np.integer)):
        raise ValueError("L must be a builtin or NumPy integer")
    if not 3 <= L <= 5:
        raise ValueError("L must lie in 3..5")
    # Our own cap is checked before even invoking the inherited constructor.
    if comb(2 * int(L) - 1, int(L)) > MAX_DIMENSION:
        raise ValueError("complete sector exceeds this module's dense dimension cap")
    L, g, C, _ = joint._parameters(L, g, C, 1)
    return L, g, C


def _spectral_matrices(transitions, gaps):
    w = np.empty((2, 2), dtype=complex)
    m = np.empty((2, 2), dtype=complex)
    pairs = [[_r(x) for x in row] for row in transitions]
    for a in range(2):
        for b in range(2):
            total, moment = _ZERO, _ZERO
            for row, gap in zip(pairs, gaps):
                gram = current._rmul(current._rconj(row[a]), row[b])
                total = current._radd(total, gram)
                delta = _real_input(gap, "excited gap")
                moment = current._radd(moment, (delta * gram[0], delta * gram[1]))
            w[a, b] = _complex_result(total, "total weight")
            m[a, b] = _complex_result(moment, "first moment")
    return w, m


def _normalize_matrix(matrix, C, extra_degree=0):
    result = np.empty((2, 2), dtype=complex)
    c = Fraction(C)
    for a in range(2):
        for b in range(2):
            factor = c ** (-_DEGREES[a] - _DEGREES[b] - extra_degree)
            pair = _r(matrix[a, b])
            result[a, b] = _complex_result((pair[0] * factor, pair[1] * factor), "C-normalized matrix")
    return result


def _entry_proxy(eta, delta1, resolution, amplitudes, proxies):
    y = Fraction(eta) * Fraction(delta1)
    gamma = 2 * Fraction(resolution)
    r, e = [Fraction(x) for x in amplitudes], [Fraction(x) for x in proxies]
    result = []
    for a in range(2):
        row = []
        for b in range(2):
            value = (2 * (e[a] * r[b] + e[b] * r[a] + e[a] * e[b]) / y
                     + 2 * (1 + Fraction(eta)) * gamma * (r[a] + e[a]) * (r[b] + e[b]) / y ** 2)
            row.append(_computed(value, "kernel entry proxy"))
        result.append(row)
    return result


def _frequency_report(system, eta, delta1, resolution, amplitudes, proxies, source_statuses):
    record = {"eta": eta, "role": "heldout" if eta == 2.0 else "development",
              "z": None, "normalized_kernel": None, "kernel": None,
              "entry_proxy": None, "ratio": None,
              "evaluation_status": "available_heuristic", "reason": None}
    if delta1 <= resolution:
        reason = "the ground gap is unresolved"
        record.update(evaluation_status="numerical_unavailable", reason=reason,
                      ratio=_ratio_unavailable("unresolved_gap", reason))
        return record
    try:
        z = complex(0, _computed(Fraction(eta) * Fraction(float(system["gaps"][1])), "frequency"))
        record["z"] = _z(z)
        data = system["partitions"]["symmetric"]
        kernel = joint._kernel(system["gaps"][1:], data["grams"], z)
        normalized = _normalize_matrix(kernel, system["C"], extra_degree=-1)
        record["kernel"], record["normalized_kernel"] = _matrix(kernel), _matrix(normalized)
    except NumericalUnavailable as exc:
        record.update(evaluation_status="numerical_unavailable", reason=str(exc),
                      ratio=_ratio_unavailable("numerical_unavailable", str(exc)))
        return record
    try:
        record["entry_proxy"] = _entry_proxy(eta, delta1, resolution, amplitudes, proxies)
        record["ratio"] = _joint_ratio(normalized, record["entry_proxy"])
    except NumericalUnavailable as exc:
        record["ratio"] = _ratio_unavailable("numerical_unavailable", str(exc))
    if record["ratio"]["status"] == "numerical_unavailable":
        record["evaluation_status"] = "numerical_unavailable"
        record["reason"] = record["ratio"]["reason"]
    for status in ("zero_source", "unresolved_source"):
        if status in source_statuses:
            record["ratio"] = _ratio_unavailable(status, "at least one required source has status " + status)
            break
    return record


def _observable_records(report):
    observations = report["observables"]
    return [observations["gap_ratio"], observations["density_moment"],
            observations["energy_moment"]] + observations["joint_ratios"]


def _owned_case(L, g, C):
    L, g, C = _parameters(L, g, C)
    system = joint.joint_system(L, g, C=C, m=1)
    data = system["partitions"]["symmetric"]
    gaps = system["gaps"]
    c = Fraction(C)
    normalized_gaps = [_computed(Fraction(float(x)) / c, "normalized gap") for x in gaps]
    resolution = _computed(Fraction(system["resolution"]) / c, "normalized resolution")
    gap_record = _gap_ratio(gaps, system["resolution"])
    splitting = _computed(Fraction(float(gaps[2])) - Fraction(float(gaps[1])), "excited splitting")
    w, m = _spectral_matrices(data["transitions"], gaps[1:])
    wn, mn = _normalize_matrix(w, C), _normalize_matrix(m, C, extra_degree=1)
    source_matrices, sources, amplitudes, proxies = [], {}, [], []
    tau = Fraction(256 * int(system["dimension"])) * Fraction(float(np.finfo(float).eps))
    for a, name in enumerate(_NAMES):
        operator = _scale(data["sources"][a], c ** -_DEGREES[a], "normalized source")
        transitions = _scale(data["transitions"][:, a], c ** -_DEGREES[a], "normalized transitions")
        norm, amplitude = _norm(operator), _norm(transitions)
        proxy = _computed(tau * Fraction(norm), "source amplitude proxy")
        status, _ = _source_status(amplitude, proxy)
        if normalized_gaps[1] <= resolution:
            moment = _scalar_record("unresolved_gap", reason="ground gap is unresolved")
        else:
            moment = _normalized_moment(float(wn[a, a].real), float(mn[a, a].real),
                                        normalized_gaps[1], amplitude, proxy)
        sources[name] = {"status": status, "amplitude": amplitude,
                         "amplitude_proxy": proxy, "operator_frobenius_norm": norm,
                         "normalized_moment": moment}
        source_matrices.append(operator)
        amplitudes.append(amplitude)
        proxies.append(proxy)
    frequencies = [_frequency_report(system, eta, normalized_gaps[1], resolution,
                                     amplitudes, proxies, [sources[x]["status"] for x in _NAMES])
                   for eta in ETAS]
    report = {"L": L, "N": L, "C": C, "g": g, "m": 1,
              "dimension": int(system["dimension"]), "status": "available_heuristic",
              "scope": _scope(), "gaps": {
                  "delta1": float(gaps[1]), "delta2": float(gaps[2]),
                  "normalized_delta1": normalized_gaps[1], "normalized_delta2": normalized_gaps[2],
                  "resolution": float(system["resolution"]), "excited_splitting": splitting,
                  "excited_splitting_status": ("unresolved_excited_splitting" if
                      splitting <= system["resolution"] else "resolved_excited_splitting"),
                  "ratio": gap_record},
              "sources": sources, "total_weight": _matrix(w), "first_moment": _matrix(m),
              "normalized_weight": _matrix(wn), "normalized_first_moment": _matrix(mn),
              "frequencies": frequencies, "observables": {
                  "gap_ratio": deepcopy(gap_record),
                  "density_moment": deepcopy(sources["rho"]["normalized_moment"]),
                  "energy_moment": deepcopy(sources["h"]["normalized_moment"]),
                  "joint_ratios": [deepcopy(x["ratio"]) for x in frequencies]},
              "free_control": None}
    if g == 0:
        report["free_control"] = {"expected": [1, 1, 1, 1, 1, 1],
                                  "comparisons": [_compare_optional(_value(x), 1.0)
                                                  for x in _observable_records(report)]}
    return {"system": system, "report": report, "sources": source_matrices}


def case_report(L, g, C=1.0):
    """One independently constructed unit-filled m1 symmetric conditional case."""
    return deepcopy(_owned_case(L, g, C)["report"])


def _number_currents(owned):
    system = owned["system"]
    model = fixed_number_model(system["L"], system["N"], system["C"], system["g"])
    energy._validate_model(model)
    if model.basis != system["basis"] or not np.array_equal(model.H, system["H"]):
        raise ValueError("canonical current model does not match the owned joint system")
    return [_scale(matrix, 1 / Fraction(system["C"]), "normalized number current")
            for matrix in current.bond_currents(model, phases=None)]


def _aggregate(comparisons):
    statuses = []
    for record in comparisons:
        if isinstance(record, list):
            statuses.extend(x["status"] for x in record)
        else:
            statuses.append(record["status"])
    if "mismatch" in statuses:
        return "mismatch"
    if "inconclusive" in statuses:
        return "inconclusive"
    return "consistent"


def _scale_comparison(scale, owned, baseline, currents, baseline_currents, L):
    left = owned["report"] if owned else None
    right = baseline["report"] if baseline else None

    def matrices(name):
        return _compare_optional(_decode(left[name]) if left else None,
                                 _decode(right[name]) if right else None)

    def gap_vector(report):
        if report is None or report["gaps"]["ratio"]["status"] != "available_heuristic":
            return None
        return [report["gaps"]["normalized_delta1"], report["gaps"]["normalized_delta2"]]

    result = {"scale": scale,
              "gaps": _compare_optional(gap_vector(left), gap_vector(right)),
              "total_weight": matrices("normalized_weight"),
              "first_moment": matrices("normalized_first_moment"),
              "kernels": [_compare_optional(_decode(left["frequencies"][i]["normalized_kernel"]) if left else None,
                                             _decode(right["frequencies"][i]["normalized_kernel"]) if right else None)
                          for i in range(3)],
              "sources": [_compare_optional(owned["sources"][i] if owned else None,
                                             baseline["sources"][i] if baseline else None)
                          for i in range(2)],
              "number_currents": [_compare_optional(currents[i] if currents is not None else None,
                                                     baseline_currents[i] if baseline_currents is not None else None)
                                  for i in range(L)],
              "observables": [_compare_optional(_value(_observable_records(left)[i]) if left else None,
                                                 _value(_observable_records(right)[i]) if right else None)
                              for i in range(6)]}
    result["status"] = _aggregate([value for key, value in result.items() if key != "scale"])
    return result


def _scaling_owned(L, g_over_C):
    # Validate programmer inputs once; tiny positive ratios remain numerical
    # failures in all three fixed slots instead of disappearing from the grid.
    try:
        L, ratio, _ = _parameters(L, g_over_C, 1.0)
    except NumericalUnavailable:
        ratio = _computed(_real_input(g_over_C, "g_over_C", nonnegative=True), "g_over_C")
        L = int(L)
    cases, owners, currents = [], [], []
    for scale in SCALES:
        try:
            g = _computed(Fraction(ratio) * Fraction(scale), "scaled coupling")
            owned = _owned_case(L, g, scale)
        except NumericalUnavailable as exc:
            owned = None
            cases.append({"scale": scale, "status": "numerical_unavailable",
                          "report": None, "reason": str(exc)})
        else:
            cases.append({"scale": scale, "status": "available_heuristic",
                          "report": deepcopy(owned["report"]), "reason": None})
        owners.append(owned)
        try:
            currents.append(_number_currents(owned) if owned else None)
        except NumericalUnavailable:
            currents.append(None)
    report = {"L": int(L), "g_over_C": ratio, "scales": list(SCALES), "cases": cases,
              "comparisons": [_scale_comparison(scale, owned, owners[1], currents[i], currents[1], L)
                              for i, (scale, owned) in enumerate(zip(SCALES, owners))],
              "scope": _scope()}
    return report, owners[1], cases[1]["reason"]


def scaling_report(L, g_over_C):
    """Independent fixed-scale systems, with baseline-referenced comparisons."""
    return deepcopy(_scaling_owned(L, g_over_C)[0])


def _observation(frequency, readout, source, reverse=False):
    result = {"eta": frequency["eta"], "kernel": None, "entry_proxy": None,
              "ratio": _ratio_unavailable("numerical_unavailable", "template dependency unavailable")}
    kernel = _decode(frequency["normalized_kernel"])
    if kernel is None:
        result["ratio"] = deepcopy(frequency["ratio"])
        return result
    try:
        observed = np.empty((2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                gain = Fraction(readout[a]) * Fraction(source[b])
                if reverse and (a, b) == (0, 1):
                    gain = -gain
                pair = _r(kernel[a, b])
                observed[a, b] = _complex_result((pair[0] * gain, pair[1] * gain), "observed kernel")
        result["kernel"] = _matrix(observed)
        if frequency["entry_proxy"] is not None:
            result["entry_proxy"] = [[_computed(abs(Fraction(readout[a]) * Fraction(source[b]))
                                                * Fraction(frequency["entry_proxy"][a][b]), "observed entry proxy")
                                      for b in range(2)] for a in range(2)]
            result["ratio"] = _joint_ratio(observed, result["entry_proxy"])
        if frequency["ratio"]["status"] != "available_heuristic":
            result["ratio"] = deepcopy(frequency["ratio"])
    except NumericalUnavailable as exc:
        result["ratio"] = _ratio_unavailable("numerical_unavailable", str(exc))
    return result


def _heldout_owned(owned, failure_reason=None):
    readout, source = [2, -3], [5, 7]
    result = {"reference": None, "readout_gains": readout, "source_gains": source,
              "template": None, "positive": None, "negative": None,
              "development": None, "heldout": None, "detection_status": "inconclusive",
              "detection_reason": None, "scope": _scope()}
    if owned is None:
        reason = failure_reason or "fixed reference construction unavailable"
        result["reference"] = {"L": 3, "N": 3, "C": 1.0, "g": 0.7, "m": 1,
                               "status": "numerical_unavailable", "reason": reason}
        result["detection_reason"] = reason
        return result
    reference = owned["report"]
    result["reference"] = deepcopy(reference)
    result["template"] = deepcopy(reference["frequencies"])
    positive = [_observation(x, readout, source) for x in reference["frequencies"]]
    negative = deepcopy(positive)
    # Only the prescribed held-out entry changes; development records are copies.
    negative[2] = _observation(reference["frequencies"][2], readout, source, reverse=True)
    result["positive"], result["negative"] = positive, negative

    def compare(observation, index):
        return _compare_optional(_value(observation[index]["ratio"]),
                                 _value(reference["frequencies"][index]["ratio"]))

    development = {"positive": [compare(positive, i) for i in range(2)],
                   "negative": [compare(negative, i) for i in range(2)]}
    heldout = {"positive": compare(positive, 2), "negative": compare(negative, 2)}
    result["development"], result["heldout"] = development, heldout
    controls = development["positive"] + development["negative"] + [heldout["positive"]]
    if any(x["status"] != "consistent" for x in controls):
        result["detection_reason"] = "a development or heldout-positive control is not consistent"
    elif heldout["negative"]["status"] == "inconclusive":
        result["detection_reason"] = "a required heldout ratio or comparison is unavailable"
    elif not all(x["ratio"]["witness_resolved"] for x in (
            reference["frequencies"][2], positive[2], negative[2])):
        result["detection_status"] = "unresolved_mismatch_witness"
        result["detection_reason"] = "both selected mixed entries must individually exceed their entry proxies"
    elif heldout["negative"]["status"] == "mismatch":
        result["detection_status"] = "mismatch_detected"
        result["detection_reason"] = "only the prescribed synthetic heldout mutation fails the frozen comparator"
    else:
        result["detection_status"] = "mismatch_not_detected"
        result["detection_reason"] = "resolved witness but the negative ratio remains inside frozen tolerance"
    return result


def heldout_report():
    """Sole preselected full-model template; no gain fitting or case selection."""
    try:
        owned = _owned_case(3, 0.7, 1.0)
    except NumericalUnavailable as exc:
        return _heldout_owned(None, str(exc))
    return _heldout_owned(owned)


def demonstration_report():
    """Nine ordered grids, 27 independent eigensolves, one reused owned template."""
    scaling_cases = []
    reference, failure_reason = None, None
    for L in (3, 4, 5):
        for ratio in (0.0, 0.7, 40.0):
            report, baseline, reason = _scaling_owned(L, ratio)
            scaling_cases.append(report)
            if L == 3 and ratio == 0.7:
                reference, failure_reason = baseline, reason
    return {"module": "substrate_prediction_contract", "scaling_cases": scaling_cases,
            "heldout": _heldout_owned(reference, failure_reason),
            "measurement_requirements": [
                "A physical preparation mapped to the complete unit-filled Bose ring and its ground state.",
                "Complete spectral coverage and multiplicity, including the actual ground gap rather than a first visible source pole.",
                "Calibrated spatial probe profiles for the stipulated m1 density and symmetric energy sources.",
                "Nonzero factorized readout/source gains; independent control of backgrounds and channel crosstalk.",
                "Constant spectral gain across all contributing lines for normalized moments.",
                "A common clock measuring response frequencies and the actual ground gap, plus independent length calibration.",
                "The specified symmetric local-energy partition, not an improved partition.",
                "An imaginary-frequency reconstruction from observations with quantified uncertainties."],
            "limitations": [
                "Common scaling fixes dimensionless relations, not a physical clock, energy, length or probe normalization.",
                "Each finite ring is a separate conditional model, not a common particle spectrum.",
                "The heldout template is the complete frozen model, not a two-frequency interpolation or interacting single pole.",
                "Entry and ratio proxies are heuristic diagnostics, not certified eigenpair or source errors.",
                "Pointwise gain cancellation does not cover additive backgrounds, channel mixing or nonfactorizable pair gains.",
                "Raw observed reciprocity and PSD need not survive arbitrary signed row and column gains.",
                "Across-frequency entry ratios require constant gain products or calibration.",
                "Synthetic inconsistency is not rejection of nature; empirical_test_unavailable.",
                "No unique parameter identification, physical measurement, fitting or empirical validation is supplied."],
            "scope": _scope()}
