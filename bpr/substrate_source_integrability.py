"""Frozen mixed external-source integrability diagnostic (Module4).

All derivatives are of Hcal=H/C in q=(f,A,u), u=v/C. Screens, residuals,
and arithmetic proxies are heuristic, not eigenvector or forward-error bounds.
This externally sourced finite Bose family introduces no metric dynamics.
The frozen contract is doc/derivations/substrate_source_integrability_2026-09-13.md.
"""
from fractions import Fraction
import math

import numpy as np

from bpr.substrate_fermionization import fixed_number_model
from bpr.substrate_neutral_response import bilinear
from bpr.substrate_gauge_encoding import NumericalUnavailable, _screened_eigensystem

MODEL_ID = "substrate_source_integrability"
SIZES = (3, 4, 5)
COUPLINGS = (0.0, 0.7, 40.0)
PARTITIONS = ("symmetric", "improved")
BACKGROUNDS = ("uniform", "modulated")
FD_STEPS = (2.0 ** -6, 2.0 ** -7, 2.0 ** -8)
ATOL = RTOL = 2e-10
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny
_NUMERICAL = (NumericalUnavailable, FloatingPointError, OverflowError,
              np.linalg.LinAlgError)
_STATUSES = ("available_heuristic", "numerical_unavailable", "numerical_unresolved",
             "diagnostic_mismatch", "diagnostic_only", "not_applicable")
_SCOPE = ("Diagnostic external-source family in dimensionless q=(f,A,u), u=v/C, "
          "Hcal=H/C and time tau=C*t; no metric dynamics or gravitational degrees "
          "of freedom. Numerical availability and all thresholds are heuristic.")
_ARITHMETIC_SCOPE = (
    "Entry-local heuristic arithmetic proxies cover supplied final spectral "
    "products and contact expectation sums only. They exclude source assembly, "
    "derivative actions, excited projections, eigensystem and eigenvector errors. "
    "arithmetic_resolved is not physical resolution or a certified error bound. "
    "The FD gate is conditional on the screened supplied Hessian, not a total "
    "forward-error certificate. S is a real Gram construction (PSD analytically); "
    "no additional eigenvalue solve, centering, projection, or symmetry repair.")


def _numpy_scalar(value, kinds):
    return (isinstance(value, np.generic) and type(value) is value.dtype.type
            and value.dtype.kind in kinds)


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) is int or _numpy_scalar(value, "iu")):
        raise ValueError(name + " must be a builtin/NumPy integer, not bool")
    if not lower <= value <= upper:
        raise ValueError(name + " outside allowed range")
    return int(value)


def _real_type(value, name):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) in (int, float) or _numpy_scalar(value, "iuf")):
        raise ValueError(name + " must be a builtin/NumPy real scalar")
    if not (type(value) is int or isinstance(value, np.integer)) and not np.isfinite(value):
        raise NumericalUnavailable(name + " is nonfinite")
    return value


def _ratio(value):
    return ((int(value), 1) if type(value) is int or isinstance(value, np.integer)
            else value.as_integer_ratio())


def _scalar(value, name):
    _real_type(value, name)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = float(value)
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalUnavailable(name + " is not representable in binary64") from exc
    if not math.isfinite(result) or _ratio(value) != result.as_integer_ratio():
        raise NumericalUnavailable(name + " changes value on binary64 conversion")
    if result != 0 and abs(result) < _TINY:
        raise NumericalUnavailable(name + " has a nonzero subnormal component")
    return result


def _parameters(L, N, g, partition):
    L = _integer(L, "L", 3, 5)
    N = _integer(N, "N", 3, 5)
    if L != N or math.comb(L + N - 1, N) > 512:
        raise ValueError("require L=N in (3,4,5), complete dimension <=512")
    _real_type(g, "g")
    matches = [value for value in COUPLINGS if _ratio(g) == value.as_integer_ratio()]
    if not matches:
        raise ValueError("g must equal the original stored value 0, 0.7 or 40")
    if type(partition) is not str or partition not in PARTITIONS:
        raise ValueError("partition must be base string symmetric or improved")
    return L, N, matches[0], partition


def _array_header(value, name, shape=None, real=False):
    if type(value) is not np.ndarray or value.dtype.kind not in ("iuf" if real else "iufc"):
        raise ValueError(name + " must be a base numeric ndarray of allowed dtype")
    if value.ndim not in (1, 2) or any(not 1 <= n <= 512 for n in value.shape):
        raise ValueError(name + " has invalid or unbounded shape")
    if shape is not None and value.shape != shape:
        raise ValueError(name + " has incorrect shape")


def _array_copy(value, name):
    """Called only after every relevant header/shape has been checked."""
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable(name + " contains nonfinite values")
    for part in (value.real, value.imag) if value.dtype.kind == "c" else (value,):
        if np.any((part != 0) & (np.abs(part.astype(np.longdouble)) < _TINY)):
            raise NumericalUnavailable(name + " contains a nonzero subnormal component")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.array(value, dtype=complex if value.dtype.kind == "c" else float, copy=True)
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalUnavailable(name + " is not representable in binary64") from exc
    if not np.all(np.isfinite(result)):
        raise NumericalUnavailable(name + " is not representable in binary64")
    # Comparing in NumPy's common dtype would hide integer narrowing above 2**53.
    if value.dtype.kind in "iu" or value.dtype.itemsize > (16 if value.dtype.kind == "c" else 8):
        pairs = ((value.real, result.real), (value.imag, result.imag)) if value.dtype.kind == "c" else ((value, result),)
        for before, after in pairs:
            for original, converted in zip(before.flat, after.flat):
                if _ratio(original) != float(converted).as_integer_ratio():
                    raise NumericalUnavailable(name + " changes value on binary64 conversion")
    return result


def _finite(value, name, normal=False):
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable(name + " is nonfinite")
    if normal:
        a = np.asarray(value)
        for part in (a.real, a.imag) if np.iscomplexobj(a) else (a,):
            if np.any((part != 0) & (np.abs(part) < _TINY)):
                raise NumericalUnavailable(name + " has a nonzero subnormal component")
    return value


def _mul(a, b, name="product"):
    """Checked real binary64 products; exact structural zeros remain zeros."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.multiply(a, b)
    _finite(result, name, normal=True)
    if np.any((np.asarray(a) != 0) & (np.asarray(b) != 0) & (result == 0)):
        raise NumericalUnavailable(name + " erases a nonzero product")
    return result


def _div(a, b, name="division"):
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result = np.divide(a, b)
    _finite(result, name, normal=True)
    if np.any((np.asarray(a) != 0) & (result == 0)):
        raise NumericalUnavailable(name + " erases a nonzero quotient")
    return result


def _sum(values, name="sum"):
    terms = [float(x) for x in values]
    result = math.fsum(terms)
    _finite(result, name, normal=True)
    if result == 0.0 and any(terms) and sum((Fraction(x) for x in terms), Fraction()) != 0:
        raise NumericalUnavailable(name + " erases a nonzero sum")
    return result


def _norm(value):
    """Scaled Frobenius norm, never a spectral solve or a unit-floor norm."""
    a = np.asarray(value, dtype=np.clongdouble)
    scale = np.max(np.maximum(np.abs(a.real), np.abs(a.imag)))
    if scale == 0:
        return 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        r, i = a.real / scale, a.imag / scale
        extended = scale * np.sqrt(np.sum(r * r + i * i))
        result = float(extended)
    if not math.isfinite(result) or (extended != 0 and result == 0):
        raise NumericalUnavailable("Frobenius norm is not representable")
    return result


def _partition_weights(system):
    return (0.5, 0.5) if system["partition"] == "symmetric" else (0.25, 0.75)


def _source_system(L, N, g, partition, f, A, u, C=1.0):
    L, N, g, partition = _parameters(L, N, g, partition)
    _real_type(C, "C")
    if not 0.5 <= C <= 2.0:
        raise ValueError("C must lie in [0.5,2]")
    for value, name in ((f, "f"), (A, "A"), (u, "u")):
        _array_header(value, name, (L,), real=True)
    for value, name, lo, hi in ((f, "f", 0.5, 1.5), (A, "A", -2, 2), (u, "u", -1, 1)):
        if not np.all(np.isfinite(value)):
            raise NumericalUnavailable(name + " contains nonfinite values")
        if np.any(value < lo) or np.any(value > hi):
            raise ValueError(name + " outside the frozen source domain")
    C = _scalar(C, "C")
    f, A, u = (_array_copy(value, name) for value, name in ((f, "f"), (A, "A"), (u, "u")))
    # The inherited helper receives physical g and C. Its ValueErrors are NOT
    # reclassified; that helper has no NumericalUnavailable exception alias.
    model = fixed_number_model(L, N, C=C, g=g)
    system = dict(L=L, N=N, g=g, C=C, partition=partition, model=model,
                  basis=tuple(tuple(n for n in state) for state in model.basis),
                  occupations=np.array(model.basis, dtype=int, copy=True), f=f, A=A, u=u)
    a, b = _partition_weights(system)
    system["w"] = b * f + a * np.roll(f, -1)
    system["H"] = _assemble(system, A)
    return system


def _bond_at(system, x, phase):
    T = bilinear(system["model"], (x + 1) % system["L"], x)
    z = complex(math.cos(float(phase)), math.sin(float(phase)))
    forward = z * T
    return -(forward + forward.conj().T), 1j * (forward - forward.conj().T)


def _bond(system, x):
    x = _integer(x, "bond index", 0, system["L"] - 1)
    return _bond_at(system, x, system["A"][x])


def _onsite(system, x):
    n = system["occupations"][:, x]
    return np.diag((system["g"] / system["C"]) * (n * (n - 1) / 2)).astype(complex)


def _assemble(system, phases):
    n = system["occupations"]
    diagonal = ((system["g"] / system["C"]) * ((n * (n - 1) / 2) @ system["f"])
                + n @ system["u"])
    H = np.diag(diagonal).astype(complex)
    for x in range(system["L"]):
        V, _ = _bond_at(system, x, phases[x])
        H += system["w"][x] * V
    return _finite(H, "assembled Hcal")


def _first_derivative(system, index):
    L = system["L"]
    index = _integer(index, "source index", 0, 3 * L - 1)
    block, x = divmod(index, L)
    if block == 2:
        return np.diag(system["occupations"][:, x]).astype(complex)
    V, j = _bond(system, x)
    if block == 1:
        return -system["w"][x] * j
    a, b = _partition_weights(system)
    incoming, _ = _bond(system, (x - 1) % L)
    return _onsite(system, x) + a * incoming + b * V


def _second_derivative(system, i, j):
    L = system["L"]
    i = _integer(i, "first source index", 0, 3 * L - 1)
    j = _integer(j, "second source index", 0, 3 * L - 1)
    bi, x = divmod(i, L)
    bj, y = divmod(j, L)
    if bi == bj == 1 and x == y:
        V, _ = _bond(system, x)
        return -system["w"][x] * V
    if (bi, bj) in ((0, 1), (1, 0)):
        site, link = (x, y) if bi == 0 else (y, x)
        a, b = _partition_weights(system)
        coefficient = b * (site == link) + a * (site == (link + 1) % L)
        if coefficient:
            _, current = _bond(system, link)
            return -coefficient * current
    return np.zeros_like(system["H"], dtype=complex)


def _screened_ground(H):
    _array_header(H, "H")
    if H.ndim != 2 or H.shape[0] != H.shape[1] or not 2 <= len(H) <= 512:
        raise ValueError("H must be square of dimension 2..512")
    H = _array_copy(H, "H")
    if not np.array_equal(H, H.conj().T):
        raise ValueError("H must be exactly Hermitian; no repair is permitted")
    scale = _norm(H)
    if scale == 0:
        raise NumericalUnavailable("zero Hamiltonian scale cannot isolate a ground state")
    try:
        screened = _screened_eigensystem(H)
    except ValueError as exc:
        # This one named screen is the sole permitted ValueError translation.
        if str(exc) != "H fails the heuristic Hermiticity screen":
            raise
        raise NumericalUnavailable(str(exc)) from exc
    values = screened["values"].copy()
    vectors = screened["vectors"].copy()
    d = len(H)
    if np.any(values[1:] < values[:-1]):
        raise NumericalUnavailable("ground energies are not nondecreasing")
    gaps = values - values[0]
    _finite(gaps, "ground gaps")
    if np.any(gaps[1:] <= 0):
        raise NumericalUnavailable("nonpositive or erased excited ground gap")
    # Residuals use the supplied raw columns and actual sourced Hcal norm.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        orthogonality = _norm(vectors.conj().T @ vectors - np.eye(d))
        residual = _norm(H @ vectors - vectors * values)
    # Solver-output residuals may be representable subnormals. A subordinate
    # candidate must not fail a normality screen before taking the maximum.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        threshold = np.longdouble(64 * _EPS * d) * np.longdouble(scale)
        residual_candidate = np.longdouble(8) * np.longdouble(residual)
        extended_resolution = max(threshold, residual_candidate)
        resolution = float(extended_resolution)
    _finite(resolution, "ground resolution", normal=True)
    if extended_resolution != 0 and resolution == 0:
        raise NumericalUnavailable("ground resolution loses a nonzero component")
    if orthogonality > 64 * _EPS * d or residual > threshold:
        raise NumericalUnavailable("raw eigensystem fails tighter orthogonality/eigenpair screens")
    if gaps[1] <= resolution:
        raise NumericalUnavailable("ground gap is unresolved against heuristic rho")
    return dict(values=values, vectors=vectors, ground=vectors[:, 0].copy(), gaps=gaps.copy(),
                ground_gap=float(gaps[1]), resolution=resolution,
                orthogonality_residual=orthogonality, eigenpair_residual=residual, scale=scale)


def _contract(transitions, gaps, contact, contact_scale):
    """Internal raw-contact path; unlike the fixture it never demands symmetry."""
    m, p = transitions.shape
    spectral = np.empty((p, p), dtype=float)
    spectral_scale = np.empty_like(spectral)
    hessian = np.empty_like(spectral)
    proxy = np.empty_like(spectral)
    ratios = np.empty((p, p), dtype=object)
    factor = 64 * _EPS * (2 * m + 2)
    for a in range(p):
        for b in range(p):
            rr = _mul(transitions[:, a].real, transitions[:, b].real, "spectral real products")
            ii = _mul(transitions[:, a].imag, transitions[:, b].imag, "spectral imaginary products")
            # Retain phases, including cancellation between real and imaginary
            # products. Dividing components separately also exposes overflow.
            real_terms = _div(rr, gaps, "spectral real gap division")
            imag_terms = _div(ii, gaps, "spectral imaginary gap division")
            spectral[a, b] = float(_mul(2.0, _sum(list(real_terms) + list(imag_terms)), "spectral factor"))
            spectral_scale[a, b] = float(_mul(2.0, _sum(list(np.abs(real_terms)) + list(np.abs(imag_terms))), "absolute spectral scale"))
            hessian[a, b] = _sum((contact[a, b], -spectral[a, b]), "contact completion")
            scale = _sum((contact_scale[a, b], spectral_scale[a, b]), "contraction scale")
            proxy[a, b] = float(_mul(factor, scale, "entry arithmetic proxy"))
            ratios[a, b] = None if scale == 0 else float(_div(abs(hessian[a, b]), scale, "cancellation ratio"))
    return dict(contact=contact.copy(), spectral=spectral, connected=-spectral,
                hessian=hessian, spectral_absolute_scale=spectral_scale,
                contact_absolute_scale=contact_scale.copy(), arithmetic_proxy=proxy,
                arithmetic_resolved=np.abs(hessian) > proxy, cancellation_ratio=ratios)


def _hessian_from_transitions(transitions, gaps, contact):
    _array_header(transitions, "transitions")
    if transitions.ndim != 2 or not 1 <= transitions.shape[0] <= 511 or not 1 <= transitions.shape[1] <= 15:
        raise ValueError("transitions must have shape (m,p), m=1..511, p=1..15")
    m, p = transitions.shape
    _array_header(gaps, "gaps", (m,), real=True)
    _array_header(contact, "contact", (p, p), real=True)
    for value, name in ((transitions, "transitions"), (gaps, "gaps"), (contact, "contact")):
        if not np.all(np.isfinite(value)):
            raise NumericalUnavailable(name + " contains nonfinite values")
    if np.any(gaps <= 0):
        raise ValueError("excited gaps must be strictly positive")
    if not np.array_equal(contact, contact.T):
        raise ValueError("supplied fixture contact must be exactly symmetric")
    transitions = _array_copy(transitions, "transitions")
    gaps = _array_copy(gaps, "gaps")
    contact = _array_copy(contact, "contact")
    try:
        return _contract(transitions, gaps, contact, np.abs(contact))
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalUnavailable("supplied Hessian contraction arithmetic is unavailable") from exc


def _status(status="available_heuristic", reason=None, **fields):
    return dict(status=status, reason=reason, **fields)


def _aggregate(records):
    records = list(records)
    for status in ("diagnostic_mismatch", "numerical_unavailable", "numerical_unresolved"):
        reasons = [record["reason"] or status for record in records if record["status"] == status]
        if reasons:
            return _status(status, "; ".join(dict.fromkeys(reasons)))
    return _status()


def _identity(observed, expected):
    error = _norm(np.asarray(observed) - np.asarray(expected))
    scale = _sum((_norm(observed), _norm(expected)), "identity scale")
    return _residual_record(error, scale)


def _residual_record(error, scale):
    tolerance = _sum((ATOL, float(_mul(RTOL, scale, "identity relative tolerance"))))
    consistent = bool(error <= tolerance)
    return _status("available_heuristic" if consistent else "diagnostic_mismatch",
                   None if consistent else "raw identity residual exceeds heuristic tolerance",
                   error=error, scale=scale, tolerance=tolerance, consistent=consistent)


def _unavailable_identity(reason):
    return _status("numerical_unavailable", reason, error=None, scale=None,
                   tolerance=None, consistent=None)


def _fd_empty(step, weights, denominator, target=None, target_proxy=None, reason=None):
    record = {name: None for name in (
        "energies", "resolutions", "energy_proxy", "subtraction_proxy", "numerator",
        "value", "numerator_proxy", "derivative_proxy", "combined_proxy", "error",
        "tolerance", "agreement")}
    record.update(_status("numerical_unavailable", reason, weights=list(weights),
                          denominator=denominator, target=target, target_proxy=target_proxy,
                          step=step, diagnostic_only=(step == FD_STEPS[0])))
    return record


def _finite_difference_record(energies, resolutions, weights, denominator, target,
                              target_proxy, step, diagnostic_only=False):
    _array_header(energies, "energies", real=True)
    if energies.ndim != 1 or len(energies) not in (3, 4):
        raise ValueError("FD energies must have length 3 or 4")
    n = len(energies)
    _array_header(resolutions, "resolutions", (n,), real=True)
    _array_header(weights, "weights", (n,), real=True)
    for value, name in ((denominator, "denominator"), (target, "target"),
                        (target_proxy, "target_proxy"), (step, "step")):
        _real_type(value, name)
    if type(diagnostic_only) is not bool:
        raise ValueError("diagnostic_only must be builtin bool")
    if _ratio(step) not in [h.as_integer_ratio() for h in FD_STEPS]:
        raise ValueError("step must be in the frozen grid")
    step = _scalar(step, "step")
    wanted = (1, -2, 1) if n == 3 else (1, -1, -1, 1)
    for value, name in ((energies, "energies"), (resolutions, "resolutions"), (weights, "weights")):
        if not np.all(np.isfinite(value)):
            raise NumericalUnavailable(name + " contains nonfinite values")
    if not np.array_equal(weights, np.array(wanted)):
        raise ValueError("FD weights differ from the frozen stencil")
    wanted_denominator = (1 if n == 3 else 4) * step * step
    if _ratio(denominator) != wanted_denominator.as_integer_ratio():
        raise ValueError("denominator must equal step squared (four times for mixed)")
    if diagnostic_only != (step == FD_STEPS[0]):
        raise ValueError("diagnostic_only must identify exactly the largest frozen step")
    if np.any(resolutions < 0) or target_proxy < 0:
        raise ValueError("resolutions and target_proxy must be nonnegative")
    energies = _array_copy(energies, "energies")
    resolutions = _array_copy(resolutions, "resolutions")
    weights = _array_copy(weights, "weights")
    denominator = _scalar(denominator, "denominator")
    target = _scalar(target, "target")
    target_proxy = _scalar(target_proxy, "target_proxy")
    record = _fd_empty(step, weights.tolist(), denominator, target, target_proxy)
    record.update(energies=energies.tolist(), resolutions=resolutions.tolist())
    try:
        record["tolerance"] = float(_mul(1e-4, max(1.0, abs(target)), "FD tolerance"))
        weighted = _mul(weights, energies, "weighted FD energies")
        record["numerator"] = _sum(weighted, "FD numerator")
        record["energy_proxy"] = _sum(_mul(np.abs(weights), resolutions, "weighted endpoint resolution"))
        record["subtraction_proxy"] = float(_mul(16 * _EPS, _sum(np.abs(weighted)), "FD subtraction proxy"))
        record["numerator_proxy"] = _sum((record["energy_proxy"], record["subtraction_proxy"]))
        record["value"] = float(_div(record["numerator"], denominator, "FD value"))
        record["derivative_proxy"] = float(_div(record["numerator_proxy"], denominator, "FD derivative proxy"))
        record["combined_proxy"] = _sum((record["derivative_proxy"], target_proxy))
        record["error"] = abs(_sum((record["value"], -target), "FD comparison"))
        if diagnostic_only:
            record.update(_status("diagnostic_only", "largest fixed step is diagnostic only"))
        elif abs(record["numerator"]) <= record["numerator_proxy"] or record["combined_proxy"] > record["tolerance"] / 10:
            record.update(_status("numerical_unresolved", "nonstructural numerator or combined heuristic proxy fails the fixed resolution gate"))
        else:
            agreement = bool(record["error"] <= record["tolerance"])
            record.update(_status("available_heuristic" if agreement else "diagnostic_mismatch",
                                  None if agreement else "fixed-step diagnostic mismatch; truncation may contribute"))
            record["agreement"] = agreement
    except _NUMERICAL as exc:
        record.update(_status("numerical_unavailable", str(exc)))
    return record


def _json(value):
    """Detach every report; reject rather than serialize nonfinite information."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=_json(value.real), imag=_json(value.imag))
        return _json(value.tolist())
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (complex, np.complexfloating)):
        return dict(real=_json(float(value.real)), imag=_json(float(value.imag)))
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json(value.item())
    if type(value) is float and not math.isfinite(value):
        raise NumericalUnavailable("nonfinite report field")
    return value


def _directions(L):
    angle = 2 * np.pi * np.arange(L) / L
    p, r, d = (np.zeros(3 * L) for _ in range(3))
    p[0] = 1
    r[L] = 1
    d[:L], d[L:2 * L], d[2 * L:] = np.cos(angle), r[L:2 * L], np.sin(angle)
    return p, r, d


def _gauge_check(system):
    L = system["L"]
    theta = 0.2 * np.cos(2 * np.pi * np.arange(L) / L)
    phases = system["A"] + np.roll(theta, -1) - theta
    gauged = _assemble(system, phases)
    occupation_phases = np.exp(1j * (system["occupations"] @ theta))
    expected = occupation_phases[:, None] * system["H"] * occupation_phases.conj()[None, :]
    return _identity(gauged, expected)


def _continuity_check(system):
    L, H = system["L"], system["H"]
    checks = []
    for y in range(L):
        density = system["occupations"][:, y]
        observed = 1j * (H * density[None, :] - density[:, None] * H)
        _, outgoing = _bond(system, y)
        _, incoming = _bond(system, (y - 1) % L)
        incoming = system["w"][(y - 1) % L] * incoming
        outgoing = system["w"][y] * outgoing
        error = _norm(observed - incoming + outgoing)
        scale = _sum((_norm(observed), _norm(incoming), _norm(outgoing)))
        checks.append(_residual_record(error, scale))
    result = _residual_record(_norm(np.array([c["error"] for c in checks])),
                              _sum(c["scale"] for c in checks))
    # Retain the individual site screens even if the aggregate norm is small.
    result.update(_aggregate([result] + checks))
    result["consistent"] = all(c["consistent"] for c in checks) and result["consistent"]
    result["sites"] = checks
    result["analytic"] = "i[Hcal,n_y]=w_(y-1)jcal_(y-1)-w_y*jcal_y in tau=C*t"
    return result


def _density_check(system):
    d = len(system["basis"])
    observed = np.diag(np.sum(system["occupations"], axis=1))
    result = _identity(observed, system["N"] * np.eye(d))
    result["analytic"] = "sum_x n_x=N*I; uniform u first derivative is N, not zero"
    return result


def _work_check(system):
    L = system["L"]
    _, _, rates = _directions(L)
    observed = np.zeros_like(system["H"], dtype=complex)
    for index, rate in enumerate(rates):
        derivative = _first_derivative(system, index)
        observed += rate * derivative
    expected = np.zeros_like(observed)
    a, b = _partition_weights(system)
    for x in range(L):
        V, current = _bond(system, x)
        expected += rates[x] * _onsite(system, x)
        expected += (b * rates[x] + a * rates[(x + 1) % L]) * V
        expected -= rates[L + x] * system["w"][x] * current
        expected += rates[2 * L + x] * np.diag(system["occupations"][:, x])
    matrix = _identity(observed, expected)
    k = np.arange(len(system["basis"])) % 3
    psi = (1 + 1j * k) / math.sqrt(float(np.sum(1 + k * k)))
    expected_value = complex(np.vdot(psi, expected @ psi))
    observed_value = complex(np.vdot(psi, observed @ psi))
    expectation = _identity(observed_value, expected_value)
    record = dict(expectation)
    record.update(_aggregate((matrix, expectation)))
    record.update(expected=_json(expected_value), observed=_json(observed_value),
                  matrix_error=matrix["error"], matrix_scale=matrix["scale"],
                  matrix_tolerance=matrix["tolerance"], matrix_consistent=matrix["consistent"],
                  consistent=bool(matrix["consistent"] and expectation["consistent"]),
                  analytic="Schrodinger work theorem uses i[Hcal,Hcal]=0; fixed-state source differentiation, not time evolution")
    return record


def _operator_checks(system, failure=None):
    checks = {}
    for name, function in (("gauge_covariance", _gauge_check), ("continuity", _continuity_check),
                           ("density_sum", _density_check), ("work_identity", _work_check)):
        try:
            if system is None:
                raise NumericalUnavailable(failure or "source system unavailable")
            checks[name] = function(system)
        except _NUMERICAL as exc:
            checks[name] = _unavailable_identity(str(exc))
            if name == "work_identity":
                checks[name].update(expected=None, observed=None, matrix_error=None,
                                    matrix_scale=None, matrix_tolerance=None, matrix_consistent=None)
    return checks


def _ground_record(ground=None, reason=None):
    keys = ("energy", "ground_gap", "resolution", "orthogonality_residual", "eigenpair_residual", "scale")
    if ground is None:
        return _status("numerical_unavailable", reason, **{key: None for key in keys})
    return _status(energy=float(ground["values"][0]),
                   **{key: float(ground[key]) for key in keys if key != "energy"})


def _hessian_empty(reason):
    fields = ("contact", "spectral", "connected", "hessian", "contact_absolute_scale",
              "spectral_absolute_scale", "arithmetic_proxy", "arithmetic_resolved", "cancellation_ratio")
    result = _status("numerical_unavailable", reason, **{key: None for key in fields})
    result.update(symmetry=_unavailable_identity(reason), arithmetic_scope=_ARITHMETIC_SCOPE,
                  contact_realness=dict(_unavailable_identity(reason), imaginary=None))
    return result


def _owning_hessian(system, ground):
    """Stream actions and contacts; preserve both raw triangles and raw columns."""
    d, p = len(system["basis"]), 3 * system["L"]
    vector = ground["ground"]
    actions = np.empty((d, p), dtype=complex)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        for a in range(p):
            derivative = _first_derivative(system, a)
            actions[:, a] = derivative @ vector
        transitions = ground["vectors"][:, 1:].conj().T @ actions
    _finite(actions, "raw derivative actions")
    _finite(transitions, "raw excited projections")
    del actions, derivative
    contact = np.zeros((p, p))
    imaginary = np.zeros((p, p))
    contact_scale = np.zeros((p, p))
    absolute_vector = np.abs(vector)
    for a in range(p):
        for b in range(p):
            operator = _second_derivative(system, a, b)
            if not np.any(operator):
                continue  # Analytic zero, not a selected transition projection.
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                expectation = np.vdot(vector, operator @ vector)
            _finite(expectation, "raw contact expectation")
            contact[a, b], imaginary[a, b] = expectation.real, expectation.imag
            # Only occupied matrix entries enter Qc, avoiding d*d scalar loops.
            rows, columns = np.nonzero(operator)
            absolute_terms = _mul(absolute_vector[rows], np.abs(operator[rows, columns]), "contact absolute product")
            absolute_terms = _mul(absolute_terms, absolute_vector[columns], "contact absolute triple product")
            contact_scale[a, b] = _sum(absolute_terms, "contact absolute scale")
    record = _hessian_empty("spectral contraction not yet computed")
    record.update(contact=contact, contact_absolute_scale=contact_scale)
    realness = _residual_record(_norm(imaginary), _norm(contact_scale))
    realness["imaginary"] = imaginary
    record["contact_realness"] = realness
    try:
        raw = _contract(transitions, ground["gaps"][1:], contact, contact_scale)
        record.update(raw)
        symmetry_checks = {name: _identity(raw[name], raw[name].T)
                           for name in ("contact", "spectral", "hessian")}
        symmetry = dict(symmetry_checks["hessian"])
        symmetry.update(_aggregate(symmetry_checks.values()))
        symmetry["consistent"] = all(check["consistent"] for check in symmetry_checks.values())
        symmetry["matrices"] = symmetry_checks
        record["symmetry"] = symmetry
        record.update(_aggregate((symmetry, realness)))
    except _NUMERICAL as exc:
        record.update(_aggregate((realness, _status("numerical_unavailable", str(exc)))))
        record["symmetry"] = _unavailable_identity(str(exc))
    return record


def _ward_checks(L, hessian):
    B = np.zeros((L, L))
    for x in range(L):
        B[x, x] = -1
        B[x, (x + 1) % L] = 1
    specifications = [("fA", slice(0, L), slice(L, 2 * L), B, "K_fA B=0"),
                      ("AA", slice(L, 2 * L), slice(L, 2 * L), B, "K_AA B=0"),
                      ("uA", slice(2 * L, 3 * L), slice(L, 2 * L), B, "K_uA B=0"),
                      ("constant_density", slice(0, 3 * L), slice(2 * L, 3 * L),
                       np.ones(L), "K_(alpha,u) 1=0 from sum n=N*I")]
    records = {}
    for name, rows, columns, direction, analytic in specifications:
        record = dict(_unavailable_identity("Hessian prerequisite unavailable"), analytic=analytic,
                      contact_contraction=None, spectral_contraction=None, completed_contraction=None)
        try:
            if hessian["hessian"] is None:
                raise NumericalUnavailable("Hessian prerequisite unavailable")
            for field, key in (("contact", "contact_contraction"), ("spectral", "spectral_contraction"),
                               ("hessian", "completed_contraction")):
                record[key] = _finite(hessian[field][rows, columns] @ direction, "raw Ward contraction")
            comparison = _residual_record(_norm(record["completed_contraction"]),
                                          _sum((_norm(record["contact_contraction"]),
                                                _norm(record["spectral_contraction"]))))
            record.update(comparison)
        except _NUMERICAL as exc:
            record.update(_status("numerical_unavailable", str(exc)))
        records[name] = record
    return records


def _free_oracle(L, N, g, background, ground, hessian):
    if g != 0 or background != "uniform":
        return _status("not_applicable", "requires g=0 and uniform background",
                       expected_energy=None, energy_comparison=None, expected_AA=None, AA_comparison=None)
    expected_energy = -2 * N * math.cos(0.3 / L)
    expected_AA = np.full((L, L), 2 * N * math.cos(0.3 / L) / (L * L))
    record = _status(expected_energy=expected_energy, expected_AA=expected_AA,
                     energy_comparison=_unavailable_identity("ground prerequisite unavailable"),
                     AA_comparison=_unavailable_identity("Hessian prerequisite unavailable"))
    if ground["energy"] is not None:
        try:
            record["energy_comparison"] = _identity(ground["energy"], expected_energy)
        except _NUMERICAL as exc:
            record["energy_comparison"] = _unavailable_identity(str(exc))
    if hessian["hessian"] is not None:
        try:
            record["AA_comparison"] = _identity(hessian["hessian"][L:2 * L, L:2 * L], expected_AA)
        except _NUMERICAL as exc:
            record["AA_comparison"] = _unavailable_identity(str(exc))
    record.update(_aggregate((record["energy_comparison"], record["AA_comparison"])))
    return record


def _missing_contact(L, g, partition, background, hessian):
    fields = dict(d=None, expected_AA=None, observed_AA=None, AA_comparison=None,
                  expected_fA=None, observed_fA=None, fA_comparison=None)
    if L != 3 or g != 0 or background != "uniform":
        return _status("not_applicable", "requires L=N=3,g=0,uniform", **fields)
    d = np.array([-1.0, 0.0, 1.0])
    a, b = (0.5, 0.5) if partition == "symmetric" else (0.25, 0.75)
    expected_AA = -2 * math.cos(0.1) * d
    expected_fA = -2 * math.sin(0.1) * np.array([a - b, -a, b])
    record = _status(d=d, expected_AA=expected_AA, observed_AA=None,
                     expected_fA=expected_fA, observed_fA=None,
                     AA_comparison=_unavailable_identity("spectral prerequisite unavailable"),
                     fA_comparison=_unavailable_identity("spectral prerequisite unavailable"))
    for name, rows, expected in (("AA", slice(L, 2 * L), expected_AA),
                                 ("fA", slice(0, L), expected_fA)):
        try:
            if hessian["spectral"] is None:
                raise NumericalUnavailable("spectral prerequisite unavailable")
            observed = -hessian["spectral"][rows, L:2 * L] @ d
            _finite(observed, "contact-omitted contraction")
            record["observed_" + name] = observed
            record[name + "_comparison"] = _identity(observed, expected)
        except _NUMERICAL as exc:
            record[name + "_comparison"] = _unavailable_identity(str(exc))
    record.update(_aggregate((record["AA_comparison"], record["fA_comparison"])))
    record["analytic"] = "Omitting contacts yields nonzero -C_AA*d and -C_fA*d; matching these is successful negative-control evidence"
    return record


def _displaced_ground(system, offset):
    """One endpoint H/eigensystem; reuse only the owning case's complete basis."""
    L = system["L"]
    shifted = dict(system)
    shifted["f"] = system["f"] + offset[:L]
    shifted["A"] = system["A"] + offset[L:2 * L]
    shifted["u"] = system["u"] + offset[2 * L:]
    a, b = _partition_weights(shifted)
    shifted["w"] = b * shifted["f"] + a * np.roll(shifted["f"], -1)
    H = _assemble(shifted, shifted["A"])
    ground = _screened_ground(H)
    return float(ground["values"][0]), float(ground["resolution"])


def _direction_target(hessian, left, right):
    nonzero_left, nonzero_right = np.flatnonzero(left), np.flatnonzero(right)
    values, proxies = [], []
    for i in nonzero_left:
        for j in nonzero_right:
            weight = float(_mul(left[i], right[j], "target directional coefficient"))
            values.append(float(_mul(weight, hessian["hessian"][i, j], "directional target")))
            proxies.append(float(_mul(abs(weight), hessian["arithmetic_proxy"][i, j], "directional target proxy")))
    return _sum(values, "directional target sum"), _sum(proxies, "directional target proxy sum")


def _fd_targets(L, hessian):
    p, r, d = _directions(L)
    result = []
    for name, directions, left, right in (("link", [r], r, r), ("mixed", [p, r], p, r),
                                          ("joint", [d], d, d)):
        record = dict(name=name, directions=directions, target=None, target_proxy=None, reason=None)
        try:
            if hessian["hessian"] is None:
                raise NumericalUnavailable("ground/Hessian prerequisite unavailable")
            record["target"], record["target_proxy"] = _direction_target(hessian, left, right)
        except _NUMERICAL as exc:
            record["reason"] = str(exc)
        result.append(record)
    return result


def _fd_report(system, L, g, ground, targets):
    if L != 3 or g != 0.7:
        return _status("not_applicable", "not one of the four frozen FD anchors", stencils=[])
    stencils = []
    for specification in targets:
        name = specification["name"]
        directions = specification["directions"]
        target, target_proxy = specification["target"], specification["target_proxy"]
        stencil = _status(name=name, directions=directions, target=target,
                          target_proxy=target_proxy, steps=[])
        previous = None
        for step in FD_STEPS:
            if name == "mixed":
                p, r = directions
                offsets = [step * p + step * r, step * p - step * r,
                           -step * p + step * r, -step * p - step * r]
                weights, denominator = (1, -1, -1, 1), 4 * step * step
            else:
                direction = directions[0]
                offsets = [-step * direction, np.zeros(3 * L), step * direction]
                weights, denominator = (1, -2, 1), step * step
            failure = specification["reason"]
            endpoints = []
            for offset in offsets:
                endpoint = _status("numerical_unavailable", failure or "endpoint not yet computed",
                                   offsets=offset.tolist(), energy=None, resolution=None)
                try:
                    if not np.any(offset) and ground["energy"] is not None:
                        # This known center survives a later Hessian/target failure;
                        # no dependent solve is needed to retain it.
                        energy, resolution = ground["energy"], ground["resolution"]
                    else:
                        if failure is not None or system is None or ground["energy"] is None:
                            raise NumericalUnavailable(failure or "ground/Hessian prerequisite unavailable")
                        energy, resolution = _displaced_ground(system, offset)
                    endpoint.update(_status(energy=energy, resolution=resolution))
                except _NUMERICAL as exc:
                    endpoint.update(_status("numerical_unavailable", str(exc)))
                endpoints.append(endpoint)
            record = _fd_empty(step, weights, denominator, target, target_proxy)
            record["energies"] = [endpoint["energy"] for endpoint in endpoints]
            record["resolutions"] = [endpoint["resolution"] for endpoint in endpoints]
            if all(endpoint["status"] == "available_heuristic" for endpoint in endpoints):
                try:
                    record = _finite_difference_record(
                        np.array(record["energies"]), np.array(record["resolutions"]),
                        np.array(weights, dtype=float), denominator, target, target_proxy,
                        step, diagnostic_only=(step == FD_STEPS[0]))
                except _NUMERICAL as exc:
                    record.update(_status("numerical_unavailable", str(exc)))
            else:
                record.update(_aggregate(endpoints))
            record.update(endpoints=endpoints, drift=None)
            if record["value"] is not None and record["status"] != "numerical_unavailable":
                if previous is not None:
                    try:
                        record["drift"] = abs(_sum((record["value"], -previous), "FD refinement drift"))
                    except _NUMERICAL as exc:
                        record.update(_status("numerical_unavailable", str(exc)))
                        record["agreement"] = None
                if record["status"] != "numerical_unavailable":
                    previous = record["value"]
            stencil["steps"].append(record)
        # Only a successfully computed diagnostic-only first step is ignored.
        stencil.update(_aggregate(stencil["steps"]))
        stencils.append(stencil)
    return dict(_aggregate(stencils), stencils=stencils)


def case_report(L, N, g, partition="symmetric", background="uniform"):
    """One independently owned labeled case, with no cross-case cache."""
    L, N, g, partition = _parameters(L, N, g, partition)
    if type(background) is not str or background not in BACKGROUNDS:
        raise ValueError("background must be base string uniform or modulated")
    angle = 2 * np.pi * np.arange(L) / L
    f = np.ones(L) if background == "uniform" else 1 + 0.05 * np.cos(angle)
    A = np.full(L, 0.3 / L)
    u = np.zeros(L) if background == "uniform" else 0.05 * np.sin(angle)
    report = _status(model_id=MODEL_ID, L=L, N=N, g=g, C=1.0, partition=partition,
                     background=background, dimension=math.comb(L + N - 1, N),
                     source_order=[kind + str(x) for kind in ("f", "A", "u") for x in range(L)],
                     sources=dict(f=f.tolist(), A=A.tolist(), u=u.tolist()), scope=_SCOPE)
    system, ground = None, None
    source_failure = None
    try:
        system = _source_system(L, N, g, partition, f, A, u)
    except _NUMERICAL as exc:
        source_failure = str(exc)
    report["operator_checks"] = _operator_checks(system, source_failure)
    try:
        if system is None:
            raise NumericalUnavailable(source_failure or "source system unavailable")
        ground = _screened_ground(system["H"])
        report["ground"] = _ground_record(ground)
    except _NUMERICAL as exc:
        report["ground"] = _ground_record(reason=str(exc))
    try:
        if ground is None:
            raise NumericalUnavailable("ground prerequisite unavailable")
        hessian = _owning_hessian(system, ground)
    except _NUMERICAL as exc:
        hessian = _hessian_empty(str(exc))
    report["ward_checks"] = _ward_checks(L, hessian)
    report["free_oracle"] = _free_oracle(L, N, g, background, report["ground"], hessian)
    report["missing_contact_control"] = _missing_contact(L, g, partition, background, hessian)
    targets = _fd_targets(L, hessian) if L == 3 and g == 0.7 else []
    report["hessian"] = _json(hessian)
    # Do not hold a background eigensystem or actual H alongside endpoint solves.
    del ground, hessian
    if system is not None:
        del system["H"]
    report["finite_differences"] = _fd_report(system, L, g, report["ground"], targets)
    report.update(_aggregate(list(report["operator_checks"].values()) +
                             [report["ground"], report["hessian"], report["free_oracle"],
                              report["missing_contact_control"], report["finite_differences"]] +
                             list(report["ward_checks"].values())))
    return _json(report)


def demonstration_report():
    """Frozen 36 sequential cases, at most 132 ground eigendecompositions."""
    controls = dict(sectors=[[L, L] for L in SIZES], couplings=list(COUPLINGS), C=1.0,
                    partitions=list(PARTITIONS), backgrounds=list(BACKGROUNDS), total_flux=0.3,
                    source_order="f,A,u", energy_normalization="H/C", gauge_amplitude=0.2,
                    fd_steps=list(FD_STEPS), fd_anchor=dict(L=3, N=3, g=0.7),
                    fd_directions=["link", "mixed", "joint"], fd_tolerance=0.0001,
                    max_ground_solves=132)
    cases = []
    for L in SIZES:
        for g in COUPLINGS:
            for partition in PARTITIONS:
                for background in BACKGROUNDS:
                    cases.append(case_report(L, L, g, partition, background))
    stencils = [stencil for case in cases for stencil in case["finite_differences"]["stencils"]]
    steps = [step for stencil in stencils for step in stencil["steps"]]
    summary = dict(case_count=len(cases),
                   case_status_counts={status: sum(case["status"] == status for case in cases) for status in _STATUSES},
                   ground_available_count=sum(case["ground"]["status"] == "available_heuristic" for case in cases),
                   fd_anchor_count=sum(case["finite_differences"]["status"] != "not_applicable" for case in cases),
                   fd_stencil_count=len(stencils), fd_step_count=len(steps),
                   fd_step_status_counts={status: sum(step["status"] == status for step in steps) for status in _STATUSES},
                   free_oracle_count=sum(case["free_oracle"]["status"] != "not_applicable" for case in cases),
                   missing_contact_control_count=sum(case["missing_contact_control"]["status"] != "not_applicable" for case in cases))
    return _json(dict(_aggregate(cases), model_id=MODEL_ID, scope=_SCOPE,
                      controls=controls, cases=cases, summary=summary))
