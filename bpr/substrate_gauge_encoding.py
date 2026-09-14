"""Frozen occupation-orbit encoding, not an emergent local gauge theory.

The complete Bose sector, seed, comparison phase and D5 normalization are
specified in substrate_gauge_encoding_2026-09-13.md. Numerical screens and
regression tolerances are heuristic. Analytic obstructions are reference
claims, independent of whether a numerical diagnostic is available.
"""
from math import comb, sqrt, sin, pi

import numpy as np

from bpr import substrate_fermionization, gauge_heat_kernel
from bpr.substrate_current_response import NumericalUnavailable

MAX_DIMENSION = 512
SIZES = (3, 4, 5)
COUPLINGS = (0.0, 0.7, 40.0)
TIMES = (0.0, 0.01, 0.1)
ATOL = RTOL = 2e-10
MODEL_ID = "conditional-substrate-gauge-encoding-v1"
_EPS = np.finfo(np.float64).eps
_NUMERICAL = (NumericalUnavailable, FloatingPointError, OverflowError,
              np.linalg.LinAlgError)
_UNSUPPORTED = "Only L=5 has the frozen central D5 target; L=3,4 are algebra controls."


def _size(L):
    if isinstance(L, (bool, np.bool_)) or not (type(L) is int or isinstance(L, np.integer)):
        raise TypeError("L must be a built-in or NumPy integer")
    if L not in SIZES:
        raise ValueError("L must be 3, 4 or 5")
    L = int(L)
    if comb(2 * L - 1, L) > MAX_DIMENSION:
        raise ValueError("complete occupation sector exceeds dimension cap")
    return L


def _real_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not (
            type(value) in (int, float) or isinstance(value, (np.integer, np.floating))):
        raise TypeError(name + " must be a built-in or NumPy real scalar")
    # Arbitrarily large Python integers need no floating conversion to test finiteness.
    if not (type(value) is int or isinstance(value, np.integer)) and not np.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _coupling(g):
    g = _real_scalar(g, "g")
    ratio = (int(g), 1) if type(g) is int or isinstance(g, np.integer) else g.as_integer_ratio()
    for reference in COUPLINGS:
        if ratio == reference.as_integer_ratio():
            return reference
    raise ValueError("g must have exactly the stored value 0, 0.7 or 40")


def _numeric_array(value, name, *, square=False, vector=False):
    """Bound shape/type before conversion; never invoke arbitrary array protocols."""
    if isinstance(value, np.ndarray):
        shape = value.shape
        if value.dtype.kind not in "iufc":
            raise TypeError(name + " must have a non-bool numeric dtype")
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= MAX_DIMENSION:
            raise ValueError(name + " has an empty or oversized dimension")
        if vector:
            shape = (len(value),)
            entries = value
        else:
            n = len(value)
            if any(not isinstance(row, (list, tuple, np.ndarray)) for row in value):
                raise ValueError(name + " must be a matrix")
            if any(isinstance(row, np.ndarray) and row.ndim != 1 for row in value):
                raise ValueError(name + " must have one-dimensional rows")
            lengths = [len(row) for row in value]
            if not lengths or not 1 <= lengths[0] <= MAX_DIMENSION or any(
                    length != lengths[0] for length in lengths):
                raise ValueError(name + " has empty, ragged or oversized rows")
            shape = (n, lengths[0])
            entries = (item for row in value for item in row)
        for item in entries:
            if isinstance(item, (bool, np.bool_)) or not (
                    type(item) in (int, float, complex)
                    or isinstance(item, (np.integer, np.floating, np.complexfloating))):
                raise TypeError(name + " must contain only non-bool numeric scalars")
    else:
        raise TypeError(name + " must be a bounded array or sequence")
    rank = 1 if vector else 2
    if len(shape) != rank or any(not 1 <= n <= MAX_DIMENSION for n in shape):
        raise ValueError(name + " has invalid shape or exceeds dimension cap")
    if square and shape[0] != shape[1]:
        raise ValueError(name + " must be square")
    # At most 512 by 512 elements reach either conversion.
    try:
        source = np.asarray(value)
        if source.dtype.kind not in "iufc":
            raise TypeError(name + " must have a non-bool numeric dtype")
        if not np.all(np.isfinite(source)):
            raise ValueError(name + " must be finite")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.array(source, dtype=complex if np.iscomplexobj(source) else float,
                              copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError(name + " is not finitely representable in binary64")
        if any(np.any((before != 0) & (after == 0)) for before, after in
               ((source.real, result.real), (source.imag, result.imag))):
            raise ValueError(name + " loses a nonzero component in binary64")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(name + " is not finitely representable in binary64") from exc
    return result


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable(name + " is nonfinite")
    return value


def _frobenius(matrix):
    """Scaled extended intermediate; no ban on representable subnormals."""
    a = np.asarray(matrix, dtype=np.clongdouble)
    scale = np.max(np.maximum(np.abs(a.real), np.abs(a.imag)))
    if scale == 0:
        return 0.0
    real, imag = a.real / scale, a.imag / scale
    norm = scale * np.sqrt(np.sum(real * real + imag * imag))
    result = float(norm)
    if not np.isfinite(result) or (norm != 0 and result == 0):
        raise NumericalUnavailable("Frobenius diagnostic is not representable")
    return result


def _screened_eigensystem(H):
    """Owned full eigensystem; degeneracy is permitted, not ground-state gated.

    The accepted Hermiticity, residual and orthogonality screens are heuristic.
    H is not symmetrized and eigenvalues are never clipped or fitted.
    """
    H = _numeric_array(H, "H", square=True)
    n = len(H)
    tolerance = 256 * _EPS * n
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scale = max(1.0, _frobenius(H))
            extended = H.astype(np.clongdouble)
            asymmetry = _frobenius(extended - extended.conj().T)
            if asymmetry / scale > tolerance:
                raise ValueError("H fails the heuristic Hermiticity screen")
            values, vectors = np.linalg.eigh(H / scale)
            # Reject invalid solver outputs before any broadcasting or rescaling.
            if not isinstance(values, np.ndarray) or values.shape != (n,) or values.dtype.kind not in "iuf":
                raise NumericalUnavailable("eigensolver returned invalid real eigenvalues")
            if not isinstance(vectors, np.ndarray) or vectors.shape != (n, n) or vectors.dtype.kind not in "iufc":
                raise NumericalUnavailable("eigensolver returned invalid eigenvectors")
            _finite(values, "eigenvalues")
            _finite(vectors, "eigenvectors")
            raw_values, raw_vectors = values, vectors
            values = np.array(raw_values, dtype=float, copy=True)
            vectors = np.array(raw_vectors, dtype=complex, copy=True)
            if np.any((raw_values != 0) & (values == 0)) or any(
                    np.any((before != 0) & (after == 0)) for before, after in
                    ((raw_vectors.real, vectors.real), (raw_vectors.imag, vectors.imag))):
                raise NumericalUnavailable("eigenoutput conversion loses a nonzero component")
            values = values * scale
            _finite(values, "rescaled eigenvalues")
            _finite(vectors, "converted eigenvectors")
            orthogonality = _frobenius(vectors.conj().T @ vectors - np.eye(n))
            residual = _frobenius(H @ vectors - vectors * values)
            if orthogonality > tolerance or residual / scale > tolerance:
                raise NumericalUnavailable("eigensolver fails orthogonality or eigenpair screen")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise NumericalUnavailable("eigensystem arithmetic or solver failed") from exc
    return {"values": values.copy(), "vectors": vectors.copy(),
            "orthogonality_residual": orthogonality, "eigenpair_residual": residual,
            "tolerance": tolerance, "scale": scale}


def _unitary_columns(system, t, columns):
    """V exp(-itE) V† columns, including the actual numerical t=0 residual."""
    if not isinstance(system, dict) or not {"values", "vectors"}.issubset(system):
        raise TypeError("system must be a screened eigensystem dictionary")
    values = _numeric_array(system["values"], "system values", vector=True)
    if np.iscomplexobj(values):
        raise ValueError("system eigenvalues must be real")
    vectors = _numeric_array(system["vectors"], "system vectors", square=True)
    columns = _numeric_array(columns, "columns")
    if vectors.shape != (len(values), len(values)) or columns.shape[0] != len(values):
        raise ValueError("eigensystem and column dimensions disagree")
    source_t = _real_scalar(t, "t")
    try:
        t = float(source_t)
    except OverflowError as exc:
        raise ValueError("t is not finitely representable") from exc
    if not np.isfinite(t) or (source_t != 0 and t == 0):
        raise ValueError("t is not finitely representable")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            angles = _finite(-t * values, "propagation angles")
            phases = np.exp(1j * angles)
            result = vectors @ (phases[:, None] * (vectors.conj().T @ columns))
            return np.array(_finite(result, "unitary columns"), dtype=complex, copy=True)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise NumericalUnavailable("unitary-column arithmetic failed") from exc


def _operator_norm(matrix):
    """Finite spectral 2-norm, never a clipped spectrum or a Frobenius proxy."""
    vector = isinstance(matrix, np.ndarray) and matrix.ndim == 1
    matrix = _numeric_array(matrix, "operator norm input", vector=vector)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scale = float(np.max(np.maximum(np.abs(matrix.real), np.abs(matrix.imag))))
            if scale == 0:
                return 0.0
            scaled = matrix.real / scale + 1j * (matrix.imag / scale)
            norm = float(np.linalg.norm(scaled, ord=2)) * scale
            if not np.isfinite(norm) or norm < 0:
                raise NumericalUnavailable("operator norm returned invalid output")
            return norm
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise NumericalUnavailable("operator norm arithmetic or SVD failed") from exc


def _mul(a, b, L):
    return ((a[0] + a[1] * b[0]) % L, a[1] * b[1])


def _inverse(a, L):
    return ((-a[1] * a[0]) % L, a[1])


def _label_actions(L):
    labels = tuple((k, s) for s in (1, -1) for k in range(L))
    index = {label: j for j, label in enumerate(labels)}
    left, right = [], []
    for a in labels:
        la, ra = np.zeros((2 * L, 2 * L)), np.zeros((2 * L, 2 * L))
        for j, b in enumerate(labels):
            la[index[_mul(a, b, L)], j] = 1.0
            ra[index[_mul(b, _inverse(a, L), L)], j] = 1.0
        left.append(la)
        right.append(ra)
    return labels, tuple(left), tuple(right)


def _spatial_state(state, label):
    k, s = label
    moved = [0] * len(state)
    for x, n in enumerate(state):
        moved[(k + s * x) % len(state)] = n
    return tuple(moved)


def _orbit_data(L):
    L = _size(L)  # Explicit complete-sector dimension guard precedes enumeration.
    basis = tuple(substrate_fermionization._occupations(L, L))
    labels, left, right = _label_actions(L)
    seed = (L - 1, 1) + (0,) * (L - 2)
    index = {state: j for j, state in enumerate(basis)}
    W = np.zeros((len(basis), 2 * L))
    for j, label in enumerate(labels):
        W[index[_spatial_state(seed, label)], j] = 1.0
    return {"L": L, "N": L, "basis": basis, "labels": labels, "seed": seed,
            "W": W, "left": left, "right": right}


def _spatial_permutation(orbit, label):
    """Stream one full spatial action as its permutation, not a matrix family."""
    index = {state: j for j, state in enumerate(orbit["basis"])}
    return np.array([index[_spatial_state(state, label)] for state in orbit["basis"]],
                    dtype=int)


def _max_component(matrix):
    _finite(matrix, "algebra residual")
    return float(max(np.max(np.abs(np.real(matrix))), np.max(np.abs(np.imag(matrix)))))


def _orbit_algebra(orbit):
    L, W = orbit["L"], orbit["W"]
    labels, left, right = orbit["labels"], orbit["left"], orbit["right"]
    m = len(labels)
    index = {label: j for j, label in enumerate(labels)}
    eye = np.eye(m)
    projectors = tuple(np.diag(eye[j]) for j in range(m))
    lr = rr = endpoint = coordinate = 0.0
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            product = index[_mul(a, b, L)]
            lr = max(lr, _max_component(left[i] @ left[j] - left[product]))
            rr = max(rr, _max_component(right[i] @ right[j] - right[product]))
            endpoint = max(endpoint, _max_component(left[i] @ right[j] - right[j] @ left[i]))
            left_image = index[_mul(a, b, L)]
            right_image = index[_mul(b, _inverse(a, L), L)]
            coordinate = max(coordinate,
                _max_component(left[i] @ projectors[j] @ left[i].T - projectors[left_image]),
                _max_component(right[i] @ projectors[j] @ right[i].T - projectors[right_image]))
            unit = np.zeros((m, m))
            unit[i, j] = 1
            shift = index[_mul(a, _inverse(b, L), L)]
            coordinate = max(coordinate, _max_component(
                projectors[i] @ left[shift] @ projectors[j] - unit))
    # Representative ambient identity check uses actual transported W, not I_ambient.
    support = np.diag(np.sum(W * W, axis=1))
    return {"isometry_residual": _max_component(W.T @ W - eye),
            "left_action_residual": lr, "right_action_residual": rr,
            "endpoint_commutator_residual": endpoint,
            "coordinate_covariance_residual": coordinate,
            "transport_identity_residual": _max_component(W @ left[0] @ W.T - support)}


def _partial_commutator(Hc, B, R, outside):
    """Exact code/complement block norm for [H, W R W†], R=R†.

    QR only changes the complement basis. Selecting complement rows first
    keeps even rank-deficient QR completion orthogonal to the code. No rank
    threshold, Gram square-root clipping, or ambient cubic product is used.
    """
    _, triangular = np.linalg.qr(B[outside, :], mode="reduced")
    q = triangular.shape[0]
    block = np.block([[Hc @ R - R @ Hc, -R @ triangular.conj().T],
                      [triangular @ R, np.zeros((q, q))]])
    return _operator_norm(block)


def _gram_minimum(gram):
    values = np.linalg.eigvalsh(gram)
    if not isinstance(values, np.ndarray) or values.shape != (len(gram),) or values.dtype.kind not in "iuf":
        raise NumericalUnavailable("Gram eigvalsh returned invalid real eigenvalues")
    _finite(values, "Gram eigenvalues")
    return float(np.min(values))  # Signed; null directions must not be dropped.


def _canonical_model(L, g):
    try:
        return substrate_fermionization.fixed_number_model(L, L, C=1.0, g=g)
    except ValueError as exc:
        # The inherited model wraps numerical errors as ValueError. Do not hide
        # arbitrary ValueError bugs or invalid-input errors under availability.
        if str(exc).startswith("numerically unresolved"):
            raise NumericalUnavailable(str(exc)) from exc
        raise


def _scope():
    return {"diagnostic_encoding": True, "local_gauge_emergence": False,
            "numerical_error_certified": False, "empirical_validation": False,
            "empirical_status": "empirical_test_unavailable"}


def _conventions():
    return {
        "spatial_action": "x -> k+s*x modulo L; labels s=+1 then -1, k ascending",
        "right_endpoint": "R_g|h> = |h*g^-1>",
        "basis": "complete lexicographic sum(n)=N occupation basis; no cutoff",
        "scalar_phase_centering": "H-E_star*I and Hc-E_star*I; E_star=g*(N-1)*(N-2)/2; no fit",
        "target": "L=5 only, central_laplacian(5), fixed single-link lambda=1 and C=1",
        "norms": "scientific norms are operator 2-norms; leakage is amplitude, not probability",
        "algebra_residuals": "maximum absolute real/imaginary component, not operator norms",
        "numerical_screens": "Frobenius residual screens are heuristic, not propagation error certificates",
        "supported_identity": "transported endpoint identity is P=WW†, not ambient I",
        "L3_partial_reflection_reference": "analytic ambient norm is null and its oracle comparison is omitted; observed norm is still evaluated",
        "bound_comparisons": "one-sided regression tolerance on full-space error only; raw residuals retained at t=0",
    }


def _comparison(status, error=None, reason=None):
    return {"status": status, "max_absolute_error": error, "reason": reason}


def _equal(observed, reference, name, reason=None):
    if observed is None:
        return _comparison("inconclusive", reason=reason or name + " unavailable")
    a, b = np.asarray(observed), np.asarray(reference)
    if a.shape != b.shape:
        return _comparison("mismatch", reason=name + " shape mismatch")
    errors = [np.abs(x - y) for x, y in ((a.real, b.real), (a.imag, b.imag))]
    consistent = all(np.all(error <= ATOL + RTOL * np.abs(ref))
                     for error, ref in zip(errors, (b.real, b.imag)))
    return _comparison("consistent" if consistent else "mismatch",
                       float(max(np.max(error) for error in errors)),
                       None if consistent else name + " differs from analytic reference")


def _combine(checks):
    for status in ("mismatch", "inconclusive"):
        selected = next((check for check in checks if check["status"] == status), None)
        if selected is not None:
            break
    else:
        selected = _comparison("consistent")
    errors = [check["max_absolute_error"] for check in checks
              if check["max_absolute_error"] is not None]
    return _comparison(selected["status"], max(errors) if errors else None, selected["reason"])


def _bound(observed, reference, name, *, upper):
    if observed is None:
        return _comparison("inconclusive", reason=name + ": full-space error unavailable")
    violation = observed - reference if upper else reference - observed
    consistent = violation <= ATOL + RTOL * abs(reference)
    literal = violation <= 0
    reason = None if literal else (
        name + (" is regression-consistent but the raw value violates the literal inequality"
                if consistent else " violates the one-sided regression inequality"))
    return _comparison("consistent" if consistent else "mismatch",
                       abs(observed - reference), reason)


def _system_metadata(system, reason=None, applicable=True):
    record = {"status": "available_heuristic" if system is not None else (
        "numerical_unavailable" if applicable else "not_applicable"), "reason": reason}
    for key in ("orthogonality_residual", "eigenpair_residual", "tolerance", "scale"):
        record[key] = system[key] if system is not None else None
    return record


def _dynamics_reference(t, norm, *, target=False, applicable=True):
    if not applicable:
        upper = lower = None
    else:
        upper = min(2.0, abs(t) * norm)
        b = (7 + sqrt(5)) / 2
        lower = max(0.0, 2 * sin(abs(t) * b / 2) - 27 * t * t / 2) if target else 0.0
        if target and abs(t) * b > pi:
            lower = None  # Frozen times are entirely inside the proved interval.
    return {"status": "numerical_unavailable" if applicable else "not_applicable",
            "reason": "numerical dynamics not available" if applicable else _UNSUPPORTED,
            "full_space_error": None, "projected_error": None, "leakage_amplitude": None,
            "analytic_upper_bound": upper, "analytic_lower_bound": lower,
            "upper_bound_comparison": _comparison("inconclusive", reason="full-space error unavailable") if applicable else None,
            "lower_bound_comparison": _comparison("inconclusive", reason="full-space error unavailable") if applicable else None}


def _json_tree(value):
    """Detached JSON primitives; complex matrices retain both channels."""
    if isinstance(value, dict):
        return {key: _json_tree(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"shape": list(value.shape), "real": _json_tree(value.real.tolist()),
                    "imag": _json_tree(value.imag.tolist())}
        return _json_tree(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_tree(item) for item in value]
    if isinstance(value, np.generic):
        return _json_tree(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise NumericalUnavailable("nonfinite report value")
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("unexpected non-JSON report value")


def case_report(L, g):
    """One fixed model and three times; failures preserve independent observations."""
    L, g = _size(L), _coupling(g)
    dimension, m = comb(2 * L - 1, L), 2 * L
    labels, reference_left, reference_right = _label_actions(L)
    eye = np.eye(m)
    R0, Ra = reference_right[L], reference_right[L + 1]
    onsite = g * (L - 1) * (L - 2) / 2
    analytic_center = -(R0 + 2 * Ra) if L == 3 else np.zeros((m, m))
    analytic_matrix = onsite * eye + analytic_center
    analytic_gram = (3 * (eye + R0) + 2 * np.ones((m, m)) if L == 3 else
                     14 * eye + 8 * R0 + 6 * Ra if L == 4 else 18 * eye + 9 * R0)
    analytic_norm = {3: 3 * sqrt(2), 4: 2 * sqrt(7), 5: 3 * sqrt(3)}[L]
    seed_norm = sqrt({3: 5, 4: 14, 5: 18}[L])
    compression_spectrum = [-3.0, -sqrt(3), -sqrt(3), sqrt(3), sqrt(3), 3.0] if L == 3 else [0.0] * m
    b = (7 + sqrt(5)) / 2
    target_norm = sqrt(27 + b * b)
    target_spectrum = [0.0, 2.0] + [(7 - sqrt(5)) / 2] * 4 + [b] * 4
    witness = np.array([np.cos(4 * np.pi * k / 5) / sqrt(5) for k, s in labels]) if L == 5 else None
    report = {
        "model_id": MODEL_ID, "L": L, "N": L, "C": 1.0, "g": g,
        "dimension": dimension, "code_dimension": m, "status": None, "reason": None,
        "conventions": _conventions(), "scope": _scope(),
        "orbit": {"seed": (L - 1, 1) + (0,) * (L - 2), "labels": labels,
                  "orbit_dimension": m, **{key: None for key in (
                      "isometry_residual", "left_action_residual", "right_action_residual",
                      "endpoint_commutator_residual", "coordinate_covariance_residual",
                      "transport_identity_residual")}, "comparison": None},
        "compression": {"onsite_energy": onsite, "matrix": None, "centered_matrix": None,
                        "analytic_matrix": analytic_matrix,
                        "analytic_centered_eigenvalues": compression_spectrum,
                        "right_reflection_commutator_norm": None,
                        "analytic_right_reflection_commutator_norm": 2 * sqrt(3) if L == 3 else 0.0,
                        "comparison": None},
        "leakage": {"gram": None, "analytic_gram": analytic_gram, "norm": None,
                    "analytic_norm": analytic_norm, "seed_column_norm": None,
                    "analytic_seed_column_norm": seed_norm, "minimum_gram_eigenvalue": None,
                    "analytic_minimum_gram_eigenvalue": 9.0 if L == 5 else 0.0,
                    "invariance_status": "analytic_noninvariant", "all_states_leak": L == 5,
                    "comparison": None},
        "symmetry": {"spatial_intertwining_residual": None,
                     "spatial_hamiltonian_commutator_norm": None,
                     "partial_right_reflection_commutator_norm": None,
                     "analytic_partial_right_reflection_commutator_norm": analytic_norm if L != 3 else None,
                     "density_right_reflection_commutator_norms": None,
                     "analytic_density_right_reflection_commutator_norms": [1.0] * L,
                     "density_multiplication_defect_residual": None, "comparison": None},
        "target": {"status": "analytic_obstruction" if L == 5 else "not_applicable",
                   "reason": "Reference claim: the zero centered compression differs from the unfitted D5 target." if L == 5 else _UNSUPPORTED,
                   "generator": None, "analytic_eigenvalues": target_spectrum if L == 5 else None,
                   "generator_mismatch_norm": None,
                   "analytic_generator_mismatch_norm": b if L == 5 else None,
                   "full_residual_norm": None,
                   "analytic_full_residual_norm": target_norm if L == 5 else None,
                   "maximal_witness": witness, "witness_residual": None, "comparison": None},
        "eigensystems": {},
        "times": [{"t": t, "tau": t,
                   "compression": _dynamics_reference(t, analytic_norm),
                   "target": _dynamics_reference(t, target_norm, target=True, applicable=L == 5)}
                  for t in TIMES],
    }
    failures = {}

    def observe(name, calculation):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                result = calculation()
                if isinstance(result, (np.ndarray, float, np.number)):
                    _finite(result, name)
                return result
        except _NUMERICAL as exc:
            failures[name] = name + ": " + str(exc)
            return None

    def unavailable(name):
        if name in failures:
            return failures[name]
        aliases = {"compression.centered_eigenvalues": "eigensystems.compression",
                   "target.eigenvalues": "eigensystems.target"}
        if aliases.get(name) in failures:
            return name + ": " + failures[aliases[name]]
        # Prefer a dependency failure, never an unrelated earlier diagnostic.
        if name.startswith("times["):
            prefix = name.split("]", 1)[0] + "]"
            for dependency in (prefix + ".ambient_columns", "eigensystems.ambient",
                               prefix + ".projected_columns", prefix + ".leakage_amplitude"):
                if dependency in failures:
                    return name + ": " + failures[dependency]
            for comparison in ("compression", "target"):
                if "." + comparison + "." in name:
                    for dependency in (prefix + "." + comparison + ".columns",
                                       "eigensystems." + comparison):
                        if dependency in failures:
                            return name + ": " + failures[dependency]
        for dependency in ("orbit construction", "model construction", "Hamiltonian", "HW"):
            if dependency in failures:
                return name + ": " + failures[dependency]
        return name + " unavailable"

    def equal(value, reference, name):
        return _equal(value, reference, name, unavailable(name) if value is None else None)

    orbit = observe("orbit construction", lambda: _orbit_data(L))
    model = observe("model construction", lambda: _canonical_model(L, g))
    W = orbit["W"] if orbit is not None else None
    H = observe("Hamiltonian", lambda: np.array(_finite(model.H, "H"), copy=True)) if model is not None else None
    Hcenter = observe("centered Hamiltonian", lambda: H - onsite * np.eye(dimension)) if H is not None else None
    HW = observe("HW", lambda: H @ W) if H is not None and W is not None else None
    Hc = observe("compression.matrix", lambda: W.T @ HW) if HW is not None else None
    center = observe("compression.centered_matrix", lambda: Hc - onsite * eye) if Hc is not None else None
    B = observe("leakage columns", lambda: HW - W @ Hc) if Hc is not None else None
    compression, leakage, symmetry, target = (report[key] for key in ("compression", "leakage", "symmetry", "target"))
    compression["matrix"], compression["centered_matrix"] = Hc, center

    if orbit is not None:
        algebra = observe("orbit algebra", lambda: _orbit_algebra(orbit))
        if algebra is not None:
            report["orbit"].update(algebra)
    R = orbit["right"][L] if orbit is not None else None
    if Hc is not None:
        compression["right_reflection_commutator_norm"] = observe(
            "compression.right_reflection_commutator_norm", lambda: _operator_norm(Hc @ R - R @ Hc))
    gram = observe("leakage.gram", lambda: B.conj().T @ B) if B is not None else None
    leakage["gram"] = gram
    if B is not None:
        leakage["norm"] = observe("leakage.norm", lambda: _operator_norm(B))
        leakage["seed_column_norm"] = observe("leakage.seed_column_norm", lambda: _operator_norm(B[:, 0]))
    if gram is not None:
        leakage["minimum_gram_eigenvalue"] = observe("leakage.minimum_gram_eigenvalue", lambda: _gram_minimum(gram))
    h_defect = observe("leakage.Hamiltonian_defect", lambda: HW.conj().T @ HW - Hc @ Hc) if Hc is not None else None

    if orbit is not None:
        spatial_intertwining, spatial_commutators = [], []
        for label in ((1, 1), (0, -1)):
            permutation = observe("spatial permutation " + str(label), lambda: _spatial_permutation(orbit, label))
            if permutation is None:
                spatial_intertwining.append(None)
                spatial_commutators.append(None)
                continue
            inverse = np.argsort(permutation)
            action = orbit["left"][labels.index(label)]
            spatial_intertwining.append(observe("symmetry.spatial_intertwining " + str(label),
                lambda: _max_component(W[inverse, :] - W @ action)))
            spatial_commutators.append(observe("symmetry.spatial_H_commutator " + str(label),
                lambda: _operator_norm(H[:, permutation] - H[inverse, :])) if H is not None else None)
        if all(value is not None for value in spatial_intertwining):
            symmetry["spatial_intertwining_residual"] = max(spatial_intertwining)
        if all(value is not None for value in spatial_commutators):
            symmetry["spatial_hamiltonian_commutator_norm"] = max(spatial_commutators)
        if B is not None:
            outside = np.flatnonzero(np.sum(W * W, axis=1) == 0)
            symmetry["partial_right_reflection_commutator_norm"] = observe(
                "symmetry.partial_right_reflection_commutator_norm",
                lambda: _partial_commutator(Hc, B, R, outside))
        densities = np.array(orbit["basis"], dtype=float).T
        compressed_densities = [W.T @ (density[:, None] * W) for density in densities]
        density_norms = [observe("symmetry.density_right_reflection_commutator_norms " + str(x),
            lambda nx=nx: _operator_norm(nx @ R - R @ nx)) for x, nx in enumerate(compressed_densities)]
        symmetry["density_right_reflection_commutator_norms"] = density_norms
        symmetry["density_multiplication_defect_residual"] = observe(
            "symmetry.density_multiplication_defect_residual", lambda: max(
                _max_component(W.T @ ((densities[x] * densities[y])[:, None] * W)
                               - compressed_densities[x] @ compressed_densities[y])
                for x in range(L) for y in range(L)))

    delta = F = None
    if L == 5:
        delta = observe("target.generator", lambda: gauge_heat_kernel.central_laplacian(5))
        target["generator"] = delta
        if delta is not None and center is not None:
            target["generator_mismatch_norm"] = observe("target.generator_mismatch_norm", lambda: _operator_norm(center - delta))
            F = observe("target residual columns", lambda: HW - W @ (onsite * eye + delta))
        if F is not None:
            target["full_residual_norm"] = observe("target.full_residual_norm", lambda: _operator_norm(F))
            target["witness_residual"] = observe("target.witness_residual",
                lambda: _operator_norm(F.conj().T @ (F @ witness) - target_norm ** 2 * witness))

    # The only propagation eigensolves. Every time record reuses these systems.
    ambient_system = observe("eigensystems.ambient", lambda: _screened_eigensystem(Hcenter)) if Hcenter is not None else None
    compression_system = observe("eigensystems.compression", lambda: _screened_eigensystem(center)) if center is not None else None
    target_system = observe("eigensystems.target", lambda: _screened_eigensystem(delta)) if delta is not None else None
    for name, system in (("ambient", ambient_system), ("compression", compression_system), ("target", target_system)):
        applicable = name != "target" or L == 5
        report["eigensystems"][name] = _system_metadata(system,
            None if system is not None else unavailable("eigensystems." + name) if applicable else _UNSUPPORTED,
            applicable=applicable)

    orbit_checks = [equal(value, 0.0, "orbit." + key) for key, value in report["orbit"].items()
                    if key.endswith("residual")]
    report["orbit"]["comparison"] = _combine(orbit_checks)
    compression["comparison"] = _combine([
        equal(Hc, analytic_matrix, "compression.matrix"),
        equal(center, analytic_center, "compression.centered_matrix"),
        equal(compression["right_reflection_commutator_norm"], compression["analytic_right_reflection_commutator_norm"],
              "compression.right_reflection_commutator_norm"),
        equal(np.sort(compression_system["values"]) if compression_system is not None else None,
              compression_spectrum, "compression.centered_eigenvalues")])
    leakage["comparison"] = _combine([
        equal(gram, analytic_gram, "leakage.gram"),
        equal(leakage["norm"], analytic_norm, "leakage.norm"),
        equal(leakage["seed_column_norm"], seed_norm, "leakage.seed_column_norm"),
        equal(leakage["minimum_gram_eigenvalue"], leakage["analytic_minimum_gram_eigenvalue"], "leakage.minimum_gram_eigenvalue"),
        equal(h_defect, analytic_gram, "leakage.Hamiltonian_defect")])
    symmetry_checks = [equal(symmetry[key], 0.0, "symmetry." + key) for key in (
        "spatial_intertwining_residual", "spatial_hamiltonian_commutator_norm", "density_multiplication_defect_residual")]
    density_norms = symmetry["density_right_reflection_commutator_norms"]
    symmetry_checks.extend(equal(density_norms[x] if density_norms is not None else None, 1.0,
        "symmetry.density_right_reflection_commutator_norms " + str(x)) for x in range(L))
    if L != 3:
        symmetry_checks.append(equal(symmetry["partial_right_reflection_commutator_norm"], analytic_norm,
                                     "symmetry.partial_right_reflection_commutator_norm"))
    elif symmetry["partial_right_reflection_commutator_norm"] is None:
        symmetry_checks.append(_comparison("inconclusive", reason=unavailable("symmetry.partial_right_reflection_commutator_norm")))
    symmetry["comparison"] = _combine(symmetry_checks)
    if L == 3 and symmetry["comparison"]["status"] == "consistent":
        symmetry["comparison"]["reason"] = "L3 analytic partial-right-reflection norm comparison explicitly omitted; observed norm retained."
    if L == 5:
        analytic_delta = 3 * eye - reference_left[1] - reference_left[4] - sum(reference_left[5:]) / 5
        target["comparison"] = _combine([
            equal(delta, analytic_delta, "target.generator"),
            equal(np.sort(target_system["values"]) if target_system is not None else None,
                  target_spectrum, "target.eigenvalues"),
            equal(target["generator_mismatch_norm"], b, "target.generator_mismatch_norm"),
            equal(target["full_residual_norm"], target_norm, "target.full_residual_norm"),
            equal(target["witness_residual"], 0.0, "target.witness_residual")])

    for time_record in report["times"]:
        t = time_record["t"]
        prefix = "times[" + str(t) + "]"
        evolved = observe(prefix + ".ambient_columns", lambda: _unitary_columns(ambient_system, t, W)) if ambient_system is not None and W is not None else None
        projected = observe(prefix + ".projected_columns", lambda: W.T @ evolved) if evolved is not None else None
        amplitude = observe(prefix + ".leakage_amplitude", lambda: _operator_norm(evolved - W @ projected)) if projected is not None else None
        for name, system in (("compression", compression_system), ("target", target_system)):
            if name == "target" and L != 5:
                continue
            record = time_record[name]
            record["leakage_amplitude"] = amplitude
            comparison_columns = observe(prefix + "." + name + ".columns", lambda: _unitary_columns(system, t, eye)) if system is not None and evolved is not None else None
            if comparison_columns is not None:
                record["full_space_error"] = observe(prefix + "." + name + ".full_space_error", lambda: _operator_norm(evolved - W @ comparison_columns))
                if projected is not None:
                    record["projected_error"] = observe(prefix + "." + name + ".projected_error", lambda: _operator_norm(projected - comparison_columns))
            record["upper_bound_comparison"] = _bound(record["full_space_error"], record["analytic_upper_bound"],
                prefix + "." + name + " upper bound", upper=True)
            record["lower_bound_comparison"] = _bound(record["full_space_error"], record["analytic_lower_bound"],
                prefix + "." + name + " lower bound", upper=False)
            comparison = _combine([record["upper_bound_comparison"], record["lower_bound_comparison"]])
            missing = next((key for key in ("full_space_error", "projected_error", "leakage_amplitude") if record[key] is None), None)
            if comparison["status"] == "mismatch":
                record["status"], record["reason"] = "diagnostic_mismatch", comparison["reason"]
            elif missing is not None:
                record["status"], record["reason"] = "numerical_unavailable", unavailable(prefix + "." + name + "." + missing)
            else:
                record["status"], record["reason"] = "available_heuristic", None

    checks = [report[key]["comparison"] for key in ("orbit", "compression", "leakage", "symmetry")]
    if L == 5:
        checks.append(target["comparison"])
    for metadata in report["eigensystems"].values():
        if metadata["status"] == "numerical_unavailable":
            checks.append(_comparison("inconclusive", reason=metadata["reason"]))
    for time_record in report["times"]:
        for name in ("compression", "target"):
            record = time_record[name]
            if record["status"] == "not_applicable":
                continue
            checks.extend((record["upper_bound_comparison"], record["lower_bound_comparison"]))
            if record["status"] == "numerical_unavailable":
                checks.append(_comparison("inconclusive", reason=record["reason"]))
    aggregate = _combine(checks)
    report["status"] = {"consistent": "available_heuristic", "mismatch": "diagnostic_mismatch",
                        "inconclusive": "numerical_unavailable"}[aggregate["status"]]
    report["reason"] = aggregate["reason"]
    return _json_tree(report)


def demonstration_report():
    """All nine frozen controls, in L-major/g-minor order, without replacement."""
    cases = [case_report(L, g) for L in SIZES for g in COUPLINGS]
    counts = {status: sum(case["status"] == status for case in cases) for status in
              ("available_heuristic", "diagnostic_mismatch", "numerical_unavailable")}
    status = ("diagnostic_mismatch" if counts["diagnostic_mismatch"] else
              "numerical_unavailable" if counts["numerical_unavailable"] else "available_heuristic")
    return {"model_id": MODEL_ID, "status": status, "scope": _scope(),
            "controls": {"sizes": list(SIZES), "couplings": list(COUPLINGS),
                         "times": list(TIMES), "C": 1.0}, "cases": cases,
            "summary": {"case_status_counts": counts,
                        "analytic_d5_obstruction_count": sum(case["L"] == 5 for case in cases)}}
