"""Joint standing-wave matter sources for the stipulated complete Bose ring.

The frozen module5 contract is in substrate_joint_source_kernel_2026-09-12.md.
Exact-input scalar arithmetic does not certify source assembly or eigenpairs.
These externally specified matter kernels are not a graviton propagator.
"""
from fractions import Fraction
from math import comb

import numpy as np

from bpr import substrate_current_response as current
from bpr import substrate_energy_response as energy
from bpr.substrate_fermionization import fixed_number_model

NumericalUnavailable = current.NumericalUnavailable
MAX_DIMENSION = 512
PARTITIONS = ("symmetric", "improved")
FREQUENCIES = (0.5j, 1.0 + 0.5j, 4.0 + 1.0j)
STEPS = (2.0 ** -6, 2.0 ** -7, 2.0 ** -8)
_ZERO = (Fraction(0), Fraction(0))
_R = np.array([[1.0, 1.0], [0.0, 1.0]])
LIMITATIONS = (
    "The complete finite quantum Bose Hamiltonian and external sources are stipulated.",
    "Exact projector Gram positivity is not a positivity claim for the retarded kernel.",
    "Numerical grouping does not certify exact degeneracy; interacting ranks are not inferred.",
    "Model envelopes are exact-model theorems evaluated without floating enclosures.",
    "Computed-gap/norm envelopes and isolation screens are heuristic, not error certificates.",
    "Subtraction roundoff proxies are separate from truncation envelopes and do not bound eigensolver error.",
    "The scalar contact is an algebraic source-family ambiguity, not new matter dynamics.",
    "No graviton, dynamical metric, physical calibration or empirical validation is supplied.",
)


def _parameters(L, g, C, m):
    for value, name in ((L, "L"), (m, "m")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be a builtin or NumPy integer")
    L, m = int(L), int(m)
    if not 3 <= L <= 5 or not 1 <= m <= L // 2:
        raise ValueError("require L in 3..5 and m in 1..floor(L/2)")
    values = []
    for value, name in ((C, "C"), (g, "g")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a builtin or NumPy real scalar")
        try:
            finite = bool(np.isfinite(value))
        except TypeError:
            # Very large builtin integers need no float conversion to reject.
            finite = isinstance(value, int)
        if not finite:
            raise ValueError(name + " must be finite")
        if (name == "C" and not 0.5 <= value <= 2) or (name == "g" and value < 0):
            raise ValueError("require C in [0.5,2] and g/C in [0,40]")
        if name == "g" and value > 40 * values[0]:
            raise ValueError("require g/C in [0,40]")
        values.append(current._component(value, name))
    C, g = values
    ratio = Fraction(g) / Fraction(C)
    if 0 < ratio < Fraction(1, 2 ** 40):
        raise NumericalUnavailable("nonzero g/C below inherited arithmetic-resolution domain")
    if comb(2 * L - 1, L) > MAX_DIMENSION:
        raise ValueError("complete sector exceeds dense dimension 512")
    return L, g, C, m


def _array(value, name, shape=None):
    """Bounded helper inputs, preserving supplied real and imaginary components."""
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise ValueError(name + " must be a bounded numeric array")
    a = np.asarray(value)
    if a.dtype.kind not in "iufc" or a.size > MAX_DIMENSION * MAX_DIMENSION * 2:
        raise ValueError(name + " must be a bounded numeric array")
    if shape is not None and a.shape != shape:
        raise ValueError(name + " has wrong shape")
    return np.asarray(current._normal(a, name))


def _grams(transitions):
    """Excited-state W[n,a,b]=conj(u[n,a])*u[n,b], rounding once per component."""
    u = _array(transitions, "transitions")
    if u.ndim != 2 or u.shape[1] != 2 or not 1 <= len(u) < MAX_DIMENSION:
        raise ValueError("transitions must have shape (excited_count,2)")
    result = np.empty((len(u), 2, 2), dtype=complex)
    for n in range(len(u)):
        pair = tuple(current._rational(z) for z in u[n])
        for a in range(2):
            for b in range(2):
                result[n, a, b] = energy._computed_complex(
                    current._rmul(current._rconj(pair[a]), pair[b]), "Gram component")
    return result


def _spectral_inputs(gaps, grams):
    gaps = _array(gaps, "excited gaps")
    if gaps.ndim != 1 or not 1 <= len(gaps) < MAX_DIMENSION or np.iscomplexobj(gaps):
        raise ValueError("gaps must be an excited-only real vector")
    if np.any(gaps <= 0):
        raise ValueError("excited gaps must be positive")
    grams = _array(grams, "spectral grams", (len(gaps), 2, 2))
    return gaps, grams


@current._guard
def _kernel(gaps, grams, z):
    """Retarded kernel with actual gaps and transposed negative-frequency numerator."""
    gaps, grams = _spectral_inputs(gaps, grams)
    z = energy._frequency(z)
    zr = current._rational(z)
    result = np.empty((2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            total = _ZERO
            for gap, gram in zip(gaps, grams):
                delta = Fraction(float(gap))
                # Do not round z +/- delta before dividing.
                positive = current._rdiv(current._rational(gram[a, b]),
                                         (zr[0] - delta, zr[1]))
                negative = current._rdiv(current._rational(gram[b, a]),
                                         (zr[0] + delta, zr[1]))
                total = current._radd(total, (positive[0] - negative[0],
                                             positive[1] - negative[1]))
            result[a, b] = energy._computed_complex(total, "retarded kernel component")
    return result


@current._guard
def _static_hessian(gaps, grams):
    """Direct inverse-gap real Hessian; never a zero-frequency retarded call."""
    gaps, grams = _spectral_inputs(gaps, grams)
    result = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            total = sum((Fraction(float(w[a, b].real)) / Fraction(float(gap))
                         for gap, w in zip(gaps, grams)), Fraction(0))
            result[a, b] = energy._computed_component(-2 * total, "static Hessian component")
    return result


def _sum_grams(grams):
    return np.array([[energy._scalar_sum(tuple((w[a, b],) for w in grams), "total Gram")
                      for b in range(2)] for a in range(2)], dtype=complex)


def _column_gram(vectors):
    return np.array([[energy._computed_complex(current._rdot(
        [current._rconj(current._rational(z)) for z in vectors[:, a]],
        [current._rational(z) for z in vectors[:, b]]), "connected Gram")
        for b in range(2)] for a in range(2)], dtype=complex)


def _congruence(matrix, R):
    return np.array([[energy._scalar_sum(
        tuple((R[a, i], matrix[i, j], R[b, j]) for i in range(2) for j in range(2)),
        "source congruence") for b in range(2)] for a in range(2)], dtype=complex)


def _transform_transitions(u, R):
    return np.array([[energy._scalar_sum(tuple((R[a, b], row[b]) for b in range(2)),
                                       "transformed transition")
                      for a in range(2)] for row in u], dtype=complex)


def _residual(left, right):
    difference = np.empty(np.shape(left), dtype=complex)
    for index in np.ndindex(difference.shape):
        difference[index] = energy._scalar_sum(((left[index],), (-1, right[index])),
                                               "diagnostic difference")
    return current._norm(difference, "raw absolute residual")


def _cosine(m, x, L):
    residue = (m * x) % L
    if residue == 0:
        return 1.0
    if 2 * residue == L:
        return -1.0
    if 4 * residue == L or 4 * residue == 3 * L:
        return 0.0
    return float(np.cos(2 * np.pi * residue / L))


def _connected_vector(system, source):
    """Raw Q O|0>, with exact-input elastic subtraction, no structural clipping."""
    v, vd = current._dyadic(system["ground"])
    operator, od = current._dyadic(source)
    vc = (v[0], -v[1])
    direct = current._imatmul(operator, v)
    expectation = current._imatmul(vc, direct)
    norm = int(current._imatmul(vc, v)[0])
    elastic = (v[0] * expectation[0] - v[1] * expectation[1],
               v[0] * expectation[1] + v[1] * expectation[0])
    denominator = int(od * vd * norm)
    return np.array([energy._computed_complex(
        (Fraction(int(r * norm - er), denominator),
         Fraction(int(i * norm - ei), denominator)), "raw connected vector")
        for r, i, er, ei in zip(direct[0], direct[1], elastic[0], elastic[1])])


def _assemble(L, g, C, m):
    L, g, C, m = _parameters(L, g, C, m)
    model = fixed_number_model(L, L, C, g)
    try:
        energy._validate_model(model)
        system = energy._validate_system(current.spectral_system(model.H), model.H)
    except ValueError as exc:
        raise NumericalUnavailable("base model/eigensystem unavailable: " + str(exc)) from exc
    norm = float(np.sqrt(2.0 / L))
    weights = tuple(energy._scalar_sum(((norm, _cosine(m, x, L)),),
                                       "standing coefficient").real for x in range(L))
    density = np.diag([energy._scalar_sum(
        tuple((weights[x], state[x] - 1) for x in range(L)), "standing density").real
        for state in model.basis])
    partitions = {}
    for partition in PARTITIONS:
        local = energy._local_parts(model, partition)["h"]
        h = energy._linear(tuple(zip(weights, local)), "standing energy")
        sources = np.array((density, h), dtype=complex)
        connected = np.column_stack([_connected_vector(system, source) for source in sources])
        transitions = np.column_stack([current._exact_transitions(system, source)[1]
                                       for source in sources])
        partitions[partition] = {"sources": sources, "connected": connected,
                                 "transitions": transitions, "grams": _grams(transitions)}
    return {"L": L, "N": L, "C": C, "g": g, "m": m,
            "k": float(2 * np.pi * m / L), "dimension": len(model.basis),
            "basis": tuple(tuple(state) for state in model.basis),
            "H": system["H"], "energies": system["energies"], "vectors": system["vectors"],
            "ground": system["ground"], "gaps": system["gaps"],
            "resolution": float(system["resolution"]), "partitions": partitions}


def _detached(value):
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detached(item) for item in value)
    return value


@current._guard
def joint_system(L, g, C=1.0, m=1):
    """Detached, bytes-backed immutable primitives for both source partitions."""
    return _detached(_assemble(L, g, C, m))


def _complex_matrix(matrix):
    matrix = current._normal(matrix, "complex report matrix")
    return {"shape": list(matrix.shape), "real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _real_matrix(matrix):
    matrix = current._normal(matrix, "real report matrix")
    if np.any(matrix.imag != 0):
        raise NumericalUnavailable("a real report matrix has a nonzero imaginary component")
    return matrix.real.tolist()


def _z(z):
    return {"real": float(z.real), "imag": float(z.imag)}


def _scope():
    return {"numerical_error_certified": False, "matter_kernel_is_graviton": False,
            "empirical_validation": False,
            "physical_source_conversion": "q=(C f_rho,f_h), D=diag(C,1), K_f=D K_q D",
            "hessian_energy": "E, not E/C; the Hessian of E/C would be K_f/C",
            "source_basis_conversion": "X=(C O_rho,O_h); X'=R X, f=R^T f', f'=R^{-T}f",
            "exact_ground_uniqueness": "positive-C complete hopping: Perron-Frobenius",
            "numerical_resolution": "separate heuristic eigensystem screen",
            "gram_positivity": "exact spectral-projector theorem, not a kernel positivity claim",
            "degeneracy_grouping": "numerical width from first member, not exact degeneracy",
            "residuals": "raw absolute diagnostics, not acceptance certificates",
            "model_bounds": "exact stipulated-operator theorem with floating non-enclosed evaluation",
            "heuristic_bounds": "computed gap and spectral norms; conditional diagnostics only",
            "roundoff_proxy": "energy subtraction proxy, separate from truncation and eigensolver error",
            "contact": "algebraic C diag(1,0) shift; no separate identity-shifted eigensolver validation"}


def _metadata(system, names):
    return {name: system[name] for name in names}


def _groups(system, grams):
    gaps = system["gaps"]
    groups = []
    start = 1
    while start < len(gaps):
        stop = start + 1
        while stop < len(gaps) and Fraction(float(gaps[stop])) - Fraction(float(gaps[start])) <= Fraction(system["resolution"]):
            stop += 1
        groups.append({"indices": list(range(start, stop)), "gap_min": float(gaps[start]),
                       "gap_max": float(gaps[stop - 1]),
                       "gram": _complex_matrix(_sum_grams(grams[start - 1:stop - 1]))})
        start = stop
    return groups


def _basis_change(gaps, transitions, gram, hessian, responses):
    transformed = _transform_transitions(transitions, _R)
    grams = _grams(transformed)
    total = _sum_grams(grams)
    static = _static_hessian(gaps, grams)
    records = []
    for z, response in zip(FREQUENCIES, responses):
        kernel = _kernel(gaps, grams, z)
        records.append({"z": _z(z), "transformed_kernel": _complex_matrix(kernel),
                        "residual": _residual(kernel, _congruence(response, _R))})
    return {"R": _R.tolist(), "transformed_gram": _complex_matrix(total),
            "transformed_hessian": _real_matrix(static),
            "gram_residual": _residual(total, _congruence(gram, _R)),
            "hessian_residual": _residual(static, _congruence(hessian, _R)),
            "responses": records}


def _spectral_norm(matrix):
    return current._component(np.linalg.norm(matrix, ord=2), "spectral source norm")


def _free_kernel(gap, gram, z):
    zr = current._rational(z)
    delta = Fraction(gap)
    denominator = current._radd(current._rmul(zr, zr), (-delta * delta, Fraction(0)))
    return np.array([[energy._computed_complex(current._rdiv(
        current._rmul((2 * delta, Fraction(0)), current._rational(gram[a, b])), denominator),
        "free expected kernel") for b in range(2)] for a in range(2)], dtype=complex)


def _free_control(system, partition, data, total, static, responses):
    if system["g"] != 0:
        return None
    L, m, C, k = (system[name] for name in ("L", "m", "C", "k"))
    pi_mode = 2 * m == L
    alpha = 0.0 if pi_mode else energy._scalar_sum(
        ((-C,), (-C, _cosine(m, 1, L))), "free alpha").real
    beta = 0.0 if partition == "symmetric" or pi_mode else energy._scalar_sum(
        ((C, 0.5, float(np.sin(k))),), "free beta").real
    weight = float(2 if pi_mode else 1)  # N/L=1 in every supported complete sector.
    expected = np.array([[weight, energy._scalar_sum(((weight, alpha),), "free cross Gram").real],
                         [energy._scalar_sum(((weight, alpha),), "free cross Gram").real,
                          energy._scalar_sum(((weight, alpha, alpha), (weight, beta, beta)),
                                             "free energy Gram").real]], dtype=complex)
    sine = 1.0 if pi_mode else float(np.sin(k / 2))
    gap = energy._scalar_sum(((4, C, sine, sine),), "active free source gap").real
    expected_hessian = _static_hessian(np.array([gap]), expected[np.newaxis, :, :])
    records = []
    for z, response in zip(FREQUENCIES, responses):
        kernel = _free_kernel(gap, expected, z)
        records.append({"z": _z(z), "expected_kernel": _complex_matrix(kernel),
                        "residual": _residual(response, kernel)})
    return {"analytic_rank": 1 if partition == "symmetric" or pi_mode else 2,
            "expected_gap": gap, "expected_gram": _complex_matrix(expected),
            "expected_hessian": _real_matrix(expected_hessian),
            "gram_residual": _residual(total, expected),
            "hessian_residual": _residual(static, expected_hessian), "responses": records,
            "energy_source_operator_norm": _spectral_norm(data["sources"][1]),
            "energy_connected_norm": current._norm(data["connected"][:, 1]),
            "energy_dark": bool(pi_mode)}


def _joint_report(system):
    report = _metadata(system, ("L", "N", "C", "g", "m", "k", "dimension", "resolution"))
    report["gap"] = float(system["gaps"][1])
    gaps = system["gaps"][1:]
    D = np.diag([system["C"], 1.0])
    partitions = {}
    for partition, data in system["partitions"].items():
        grams = data["grams"]
        total = _sum_grams(grams)
        dimensionless = _congruence(total, D)
        static = _static_hessian(gaps, grams)
        static_f = _congruence(static, D)
        responses, physical_responses, dimensionless_responses = [], [], []
        for z in FREQUENCIES:
            physical = _kernel(gaps, grams, z)
            scaled = _congruence(physical, D)
            physical_responses.append(physical)
            dimensionless_responses.append(scaled)
            responses.append({"z": _z(z), "kernel_physical": _complex_matrix(physical),
                              "kernel_dimensionless": _complex_matrix(scaled),
                              "reciprocity_residual": _residual(physical, physical.T)})
        partitions[partition] = {
            "total_gram": _complex_matrix(total), "dimensionless_gram": _complex_matrix(dimensionless),
            "groups": _groups(system, grams), "responses": responses,
            "static_hessian_physical": _real_matrix(static),
            "static_hessian_dimensionless": _real_matrix(static_f),
            "gram_eigenvalues_dimensionless": current._normal(np.linalg.eigvalsh(dimensionless), "Gram eigenvalues").tolist(),
            "static_eigenvalues_dimensionless": current._normal(np.linalg.eigvalsh(static_f.real), "static eigenvalues").tolist(),
            "closure_residual": _residual(total, _column_gram(data["connected"])),
            "basis_change": _basis_change(gaps, _transform_transitions(data["transitions"], D),
                                           dimensionless, static_f, dimensionless_responses),
            "free_control": _free_control(system, partition, data, total, static, physical_responses)}
    normalization = float(np.sqrt(2.0 / system["L"]))
    number = energy._linear(((normalization * system["N"], np.eye(system["dimension"])),),
                            "uniform number source")
    h_uniform = energy._linear(((normalization, system["H"]),), "uniform energy source")
    report.update({"partitions": partitions,
                   "uniform_controls": {"number_structural_zero": True, "energy_structural_zero": True,
                                        "number_connected_norm": current._norm(_connected_vector(system, number)),
                                        "energy_connected_norm": current._norm(_connected_vector(system, h_uniform))},
                   "scope": _scope()})
    return report


@current._guard
def joint_report(L, g, C=1.0, m=1):
    """JSON-native joint spectral, static, source-basis and fixed free controls."""
    return _joint_report(_assemble(L, g, C, m))


def _real_fraction(value, name):
    return energy._computed_component(value, name)


def _model_bounds(system):
    L, N, C, g = (system[name] for name in ("L", "N", "C", "g"))
    sine = Fraction(float(np.sin(np.pi / L)))
    cf, gf = Fraction(C), Fraction(g)
    weak = 4 * cf * sine * sine - gf * N * (N - 1) / (2 * L)
    strong = gf - 2 * cf * N
    gap = _real_fraction(max(Fraction(0), weak, strong), "model gap bound")
    wmax = Fraction(float(np.sqrt(2.0 / L)))
    rho = _real_fraction(cf * N * wmax, "model density norm bound")
    h = _real_fraction(wmax * (gf * comb(N, 2) + 2 * cf * N), "model energy norm bound")
    plus = _real_fraction(Fraction(rho) + Fraction(h), "model directional norm bound")
    return gap, {"rho": rho, "h": h, "plus": plus, "minus": plus}


def _axial_bound(b, h, delta, model):
    if b is None or delta is None:
        return None, "input_numerical_unavailable"
    if b == 0:
        return 0.0, "available_conditional"
    if delta <= 0:
        return None, "analytic_gap_unavailable" if model else "input_numerical_unavailable"
    bf, hf, df = Fraction(b), Fraction(h), Fraction(delta)
    if 2 * bf * hf >= df:
        return None, "sufficient_radius_unavailable"
    try:
        value = 16 * bf ** 4 * hf ** 2 / (df ** 3 * (1 - (2 * bf * hf / df) ** 2))
        return _real_fraction(value, "axial truncation envelope"), "available_conditional"
    except NumericalUnavailable:
        return None, "input_numerical_unavailable"


def _bounds(norms, h, gap, model=False):
    axial = {key: _axial_bound(norms[key], h, gap, model) for key in ("rho", "h", "plus", "minus")}
    if all(axial[key][0] is not None for key in ("plus", "minus")):
        try:
            mixed = _real_fraction((Fraction(axial["plus"][0]) + Fraction(axial["minus"][0])) / 4,
                                   "mixed truncation envelope")
            mixed_status = "available_conditional"
        except NumericalUnavailable:
            mixed, mixed_status = None, "input_numerical_unavailable"
    else:
        mixed = None
        unavailable = {axial[key][1] for key in ("plus", "minus")}
        mixed_status = next(status for status in ("input_numerical_unavailable", "analytic_gap_unavailable",
                                                  "sufficient_radius_unavailable") if status in unavailable)
    return {"values": [[axial["rho"][0], mixed], [mixed, axial["h"][0]]],
            "status": [[axial["rho"][1], mixed_status], [mixed_status, axial["h"][1]]],
            "directional_norms": dict(norms), "gap_used": gap}


def _directions(sources, C):
    X = energy._linear(((C, sources[0]),), "dimensionless density source")
    Y = sources[1]
    return {"rho": X, "h": Y,
            "plus": energy._linear(((1, X), (1, Y)), "plus direction"),
            "minus": energy._linear(((1, X), (-1, Y)), "minus direction")}


def _direction_norms(directions):
    norms = {}
    for key, matrix in directions.items():
        try:
            norms[key] = _spectral_norm(matrix)
        except (NumericalUnavailable, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            norms[key] = None
    return norms


def _origin_point(system):
    """Reuse the privately screened base eigensystem, without another solve."""
    scale = max(system["C"], current._norm(system["H"], "base Hamiltonian norm"))
    proxy = _real_fraction(Fraction(256 * system["dimension"]) *
                           Fraction(float(np.finfo(float).eps)) * Fraction(scale),
                           "origin per-energy proxy")
    return {"f": [0.0, 0.0], "energy": float(system["energies"][0]),
            "energy_proxy": proxy, "perturbation_norm": 0.0,
            "isolation_margin": float(system["gaps"][1]),
            "isolation_status": "sufficient_heuristic", "evaluation_status": "available"}


def _point(system, X, Y, f):
    point = {"f": [float(f[0]), float(f[1])]}
    try:
        perturbation = energy._linear(((f[0], X), (f[1], Y)), "source perturbation")
        norm = _spectral_norm(perturbation)
        matrix = energy._linear(((1, system["H"]), (1, perturbation)), "perturbed Hamiltonian")
        values, vectors = np.linalg.eigh(matrix)
        values = current._normal(values, "perturbed eigenvalues")
        vectors = current._normal(vectors, "perturbed eigenvectors")
        dimension, C = system["dimension"], system["C"]
        tolerance = _real_fraction(Fraction(256 * dimension) * Fraction(float(np.finfo(float).eps)),
                                   "perturbed relative tolerance")
        scale = max(C, current._norm(matrix, "perturbed Hamiltonian norm"))
        proxy = _real_fraction(Fraction(tolerance) * Fraction(scale), "per-energy proxy")
        orthogonality = current._norm(vectors.conj().T @ vectors - np.eye(dimension), "perturbed orthogonality")
        residual = current._norm(matrix @ vectors - vectors * values, "perturbed eigenpair residual")
        if orthogonality > tolerance or residual > proxy:
            raise NumericalUnavailable("perturbed eigensystem failed consistency screen")
        margin = _real_fraction(Fraction(float(system["gaps"][1])) - 2 * Fraction(norm), "isolation margin")
        point.update({"energy": float(values[0]), "energy_proxy": proxy, "perturbation_norm": norm,
                      "isolation_margin": margin,
                      "isolation_status": "sufficient_heuristic" if margin > 0 else "isolation_not_established",
                      "evaluation_status": "available"})
    except (NumericalUnavailable, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        point.update({"energy": None, "energy_proxy": None, "perturbation_norm": None,
                      "isolation_margin": None, "isolation_status": "isolation_not_established",
                      "evaluation_status": "numerical_unavailable", "reason": str(exc)})
    return point


def _stencil_entry(points, field, indices, coefficients, denominator):
    if any(points[index][field] is None for index in indices):
        return None
    try:
        numerator = sum((Fraction(points[index][field]) * coefficient
                         for index, coefficient in zip(indices, coefficients)), Fraction(0))
        return _real_fraction(numerator / denominator, "finite-difference " + field)
    except NumericalUnavailable:
        return None


def _entry_map(matrix, function):
    result = [[None, None], [None, None]]
    for a, b in ((0, 0), (0, 1), (1, 1)):
        if matrix[a][b] is not None:
            try:
                result[a][b] = function(matrix[a][b], a, b)
            except NumericalUnavailable:
                pass
        result[b][a] = result[a][b]
    return result


def _step(system, directions, origin, h, target, contact, heuristic_norms, model_gap, model_norms):
    coordinates = ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h),
                   (h, h), (h, -h), (-h, h), (-h, -h))
    points = [dict(origin, f=list(origin["f"]))]
    points.extend(_point(system, directions["rho"], directions["h"], f) for f in coordinates)
    h2 = Fraction(h) ** 2
    stencil, proxy = [[None, None], [None, None]], [[None, None], [None, None]]
    for a, b, indices, coefficients, denominator in (
            (0, 0, (1, 0, 2), (1, -2, 1), h2),
            (1, 1, (3, 0, 4), (1, -2, 1), h2),
            (0, 1, (5, 6, 7, 8), (1, -1, -1, 1), 4 * h2)):
        stencil[a][b] = _stencil_entry(points, "energy", indices, coefficients, denominator)
        proxy[a][b] = _stencil_entry(points, "energy_proxy", indices,
                                      tuple(abs(c) for c in coefficients), denominator)
        stencil[b][a], proxy[b][a] = stencil[a][b], proxy[a][b]
    errors = _entry_map(stencil, lambda value, a, b: _real_fraction(
        abs(Fraction(value) - Fraction(float(target[a, b]))), "stencil discrepancy"))
    shifted = _entry_map(stencil, lambda value, a, b: _real_fraction(
        Fraction(value) + Fraction(float(contact[a, b])), "contact-shifted stencil"))
    statuses = [[("unavailable" if stencil[a][b] is None or proxy[a][b] is None else
                  "above_roundoff_proxy" if abs(stencil[a][b]) > proxy[a][b] else "below_roundoff_proxy")
                 for b in range(2)] for a in range(2)]
    return {"step": h, "points": points, "stencil": stencil, "absolute_error": errors,
            "roundoff_proxy": proxy, "roundoff_status": statuses,
            "heuristic_bounds": _bounds(heuristic_norms, h, float(system["gaps"][1])),
            "model_bounds": _bounds(model_norms, h, model_gap, model=True),
            "contact_shifted_stencil": shifted}


@current._guard
def finite_difference_report(L, g, C=1.0, m=1):
    """Fixed-step Hessian stencils with independent bounds and per-entry failures."""
    system = _assemble(L, g, C, m)
    model_gap, model_norms = _model_bounds(system)
    report = _metadata(system, ("L", "N", "C", "g", "m", "dimension"))
    report.update({"gap": float(system["gaps"][1]), "model_gap_lower_bound": model_gap,
                   "model_source_norm_bounds": model_norms, "partitions": {}, "scope": _scope()})
    D = np.diag([system["C"], 1.0])
    contact = np.diag([system["C"], 0.0])
    origin = _origin_point(system)
    for partition, data in system["partitions"].items():
        directions = _directions(data["sources"], system["C"])
        norms = _direction_norms(directions)
        target = _congruence(_static_hessian(system["gaps"][1:], data["grams"]), D).real
        shifted = np.array([[energy._scalar_sum(((target[a, b],), (contact[a, b],)),
                                                "contact-shifted Hessian").real
                             for b in range(2)] for a in range(2)])
        report["partitions"][partition] = {
            "target_hessian": _real_matrix(target), "contact_matrix": _real_matrix(contact),
            "contact_shifted_hessian": _real_matrix(shifted),
            "steps": [_step(system, directions, origin, h, target, contact, norms, model_gap, model_norms)
                      for h in STEPS]}
    return report


def _case(function, L, g, m=1):
    try:
        return function(L, g, C=1.0, m=m)
    except NumericalUnavailable as exc:
        return {"L": L, "g": g, "C": 1.0, "m": m,
                "status": "numerical_unavailable", "reason": str(exc)}


def demonstration_report():
    """The frozen nine-case grid and the distinct free pi dark-source control."""
    cases = tuple((L, g) for L in (3, 4, 5) for g in (0.0, 0.7, 40.0))
    return {"joint_cases": [_case(joint_report, L, g) for L, g in cases],
            "finite_difference_cases": [_case(finite_difference_report, L, g) for L, g in cases],
            "dark_control": _case(joint_report, 4, 0.0, m=2), "limitations": list(LIMITATIONS)}
