"""Classical canonical modes and finite quantum rank-one projected memory.

Frozen conventions: doc/derivations/substrate_collective_dynamics_2026-09-12.md.
A generic ``source`` is a vector, not an operator, and need not be connected.
Its resolvent uses all of H-E0, including the ground direction. Only the
quantum density wrapper constructs a connected positive-excitation source.
Numerical screens and residuals are diagnostics, never error certificates.
"""
from fractions import Fraction
from numbers import Integral

import numpy as np

from bpr.substrate_current_response import (
    MAX_DENSE_DIMENSION, NumericalUnavailable, _component, _dyadic, _guard,
    _imatmul, _json_complex, _matrix, _normal, _norm, _radd, _rational,
    _rcomplex, _rconj, _rdiv, _rdot, _rmul, _scalar, retarded_response,
    spectral_system,
)
from bpr.substrate_fermionization import fixed_number_model
from bpr.substrate_neutral_response import density_diagonal

MODEL_ID = "conditional-substrate-collective-dynamics-v1"
CLASSICAL_CONTROLS = tuple((L, 1.0, 1.0, g) for g in (0.0, 0.7, 40.0)
                           for L in (3, 4, 5))
QUANTUM_CONTROLS = tuple((L, 1.0, g, 1) for g in (0.7, 40.0) for L in (3, 4, 5))
FREQUENCIES = (0.5j, 1.0 + 0.5j, 4.0 + 1.0j)
REGULATORS = (0.25, 0.5, 1.0)
TIMES = (0.0, 0.5, 1.0, 2.0)
LIMITATIONS = (
    "Classical DNLS and the stipulated quantum Bose ring are separate models.",
    "Canonical density and phase are one conjugate pair, not two displacement fields.",
    "The generic source vector need not be connected; its full shifted Hamiltonian includes the ground.",
    "Finite nonzero memory is almost periodic; samples do not establish irreversible decay.",
    "The upper-half-plane regulator is not a physical damping rate.",
    "No internal phason, physical modulus, calibration or Markov approximation is derived.",
    "Floating residuals, inverse conditioning and eigenpair screens are heuristic, not certificates.",
)
_EPS = np.finfo(float).eps


def _ring(L, C, g):
    if isinstance(L, (bool, np.bool_)) or not isinstance(L, Integral) or L not in (3, 4, 5):
        raise ValueError("public ring cases require L=3,4,5")
    C, g = _scalar(C, "C", real=True), _scalar(g, "g", real=True)
    if C <= 0 or g < 0:
        raise ValueError("C must be positive and g nonnegative")
    return int(L), C, g


def _momentum(L, m):
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral) or not 0 <= m < L:
        raise ValueError("m must be a canonical integer momentum 0,...,L-1")
    return int(m)


def _frequency(z):
    z = _scalar(z, "z")
    if z.imag <= 0:
        raise ValueError("z must lie in the upper half plane")
    return z


@_guard
def classical_modes(L, nbar=1.0, C=1.0, g=0.7):
    """JSON-safe Hessian coefficients and dispersion, with k=0 excluded."""
    L, C, g = _ring(L, C, g)
    nbar = _scalar(nbar, "nbar", real=True)
    if nbar <= 0:
        raise ValueError("nbar must be positive")
    # Keep products extended until _normal checks the binary64 conversion.
    # Python float multiplication can silently underflow before np.errstate.
    nc, cc, gc = np.longdouble(nbar), np.longdouble(C), np.longdouble(g)
    modes = []
    for m in range(L):
        k = 2 * np.pi * m / L
        epsilon = float(_normal(4 * cc * np.longdouble(np.sin(k / 2)) ** 2, "epsilon")) if m else 0.0
        ec = np.longdouble(epsilon)
        alpha = float(_normal(gc + ec / (2 * nc), "alpha"))
        beta = float(_normal(2 * nc * ec, "beta"))
        omega2 = float(_normal(ec * (ec + 2 * gc * nc), "omega squared"))
        modes.append({"m": m, "k": float(k), "epsilon": epsilon, "alpha": alpha,
                      "beta": beta, "omega_squared": omega2, "omega": float(np.sqrt(omega2)),
                      "static_susceptibility": float(_normal(1 / alpha, "static susceptibility")) if m else None,
                      "status": "oscillator" if m else "fixed_norm_density_removed_global_phase_symmetry"})
    return {"parameters": {"L": L, "nbar": nbar, "C": C, "g": g}, "modes": modes,
            "real_oscillator_count": L - 1,
            "gradient_stiffness": float(_normal(2 * cc * nc, "gradient stiffness")),
            "total_flux_curvature": float(_normal(2 * cc * nc / L, "flux curvature")),
            "response_convention": "chi_nn for +f*n; positive static susceptibility is for -f*n"}


@_guard
def classical_response(L, m, z, nbar=1.0, C=1.0, g=0.7):
    """beta/(z*z-omega_squared); the fixed-norm uniform response is zero."""
    modes = classical_modes(L, nbar, C, g)
    m, z = _momentum(L, m), _frequency(z)
    if m == 0:
        return 0j
    mode = modes["modes"][m]
    zz = _rmul(_rational(z), _rational(z))
    denominator = (zz[0] - Fraction(mode["omega_squared"]), zz[1])
    return _rcomplex(_rdiv((Fraction(mode["beta"]), Fraction(0)), denominator), "classical response")


def _vector(value, dimension):
    # Inspect shape before conversion, including hostile __array__ inputs.
    if isinstance(value, np.ndarray):
        if value.shape != (dimension,):
            raise ValueError("source must be a vector of Hamiltonian dimension")
    elif isinstance(value, (list, tuple)):
        if len(value) != dimension:
            raise ValueError("source must be a vector of Hamiltonian dimension")
    else:
        raise ValueError("source must be a bounded vector")
    return np.array([_scalar(x, "source") for x in value], dtype=complex)


def _exact_matvec(A, v):
    """Exact dyadic contraction of bounded binary64 inputs; rational output."""
    a, ad = _dyadic(A)
    b, bd = _dyadic(v)
    real, imag = _imatmul(a, b)
    denominator = ad * bd
    return [(Fraction(int(x), denominator), Fraction(int(y), denominator))
            for x, y in zip(real, imag)]


def _safe_matmul(A, B):
    """Extended contraction with bounded exact fallback for cancellation.

    This protects contractions of the supplied arrays, not their eigensolver
    provenance. Broadly cancellation-dominated requests are explicitly unsupported.
    """
    A, B = np.asarray(A, dtype=complex), np.asarray(B, dtype=complex)
    a, b = A.astype(np.clongdouble), B.astype(np.clongdouble)
    value = a @ b
    real_scale = np.abs(a.real) @ np.abs(b.real) + np.abs(a.imag) @ np.abs(b.imag)
    imag_scale = np.abs(a.real) @ np.abs(b.imag) + np.abs(a.imag) @ np.abs(b.real)
    suspicious = ((real_scale != 0) & (np.abs(value.real) <= 1e-10 * real_scale))
    suspicious |= ((imag_scale != 0) & (np.abs(value.imag) <= 1e-10 * imag_scale))
    indices = np.argwhere(suspicious)
    if len(indices) > 4096:
        raise NumericalUnavailable("projection has too many cancellation-dominated contractions")
    # Exact output is inserted after conversion, so underflow cannot silently erase it.
    result = np.empty(value.shape, dtype=complex)
    if value.ndim != 2:
        raise ValueError("internal contraction requires matrices")
    for i, j in indices:
        exact = _rdot([_rational(x) for x in A[i]], [_rational(x) for x in B[:, j]])
        value[i, j] = 0  # Avoid converting the less accurate cancelled component.
        result[i, j] = _rcomplex(exact, "projection contraction")
    normal = _normal(value, "projection contraction")
    result[~suspicious] = normal[~suspicious]
    return result


def _shifted_matrix(H, E0):
    A = H.copy()
    for i in range(len(H)):
        A[i, i] = _component(Fraction(float(H[i, i].real)) - Fraction(float(E0)), "shifted diagonal")
    return A


@_guard
def projected_system(H, source):
    """Build rank-one Schur data for a VECTOR using the full A=H-E0.

    Generic sources need not be ground-connected. Q includes the ground unless
    it belongs to the source direction. Exact zero sources have F=0, but their
    normalized projection, self energy and memory are undefined (ValueError).
    Arrays in the result are reusable implementation data, not JSON output.
    """
    # H shape/cap and source shape are screened before any eigensolver allocation.
    H = _matrix(H, "H")
    source = _vector(source, len(H))
    return _projected(spectral_system(H), source)


def _projected(system, source):
    H, E0 = system["H"], float(system["energies"][0])
    n = len(H)
    source = _vector(source, n)
    A = _shifted_matrix(H, E0)
    base = {"system": system, "A": A, "source": source,
            "source_contract": "generic vector; full shifted Hamiltonian including ground"}
    if not np.any(source != 0):
        base.update({"status": "zero_source", "s": 0.0, "e": None, "W": None,
                     "a": None, "b": None, "B": None, "beta_squared": None,
                     "B_energies": None, "B_vectors": None, "diagnostics": {}})
        return base
    vr = [_rational(x) for x in source]
    sr = _rdot([_rconj(x) for x in vr], vr)[0]
    s = _component(sr, "source squared norm")
    norm = _component(np.sqrt(np.longdouble(s)), "source norm")
    e = _normal(source.astype(np.clongdouble) / norm, "normalized source")
    hv = _exact_matvec(H, source)
    rayleigh = _rdot([_rconj(x) for x in vr], hv)
    if rayleigh[1] != 0:
        raise NumericalUnavailable("exact Hermitian Rayleigh contraction has imaginary part")
    lam = rayleigh[0] / sr
    a = _component(lam - Fraction(E0), "projected a")
    # Subtract before conversion: do not declare b=0 from rounded Hv-lambda*v.
    residual_rational = [_radd(h, tuple(-x for x in _rmul((lam, Fraction(0)), v)))
                         for h, v in zip(hv, vr)]
    structural_b_zero = all(x == (0, 0) for x in residual_rational)
    residual = np.array([_rcomplex(_rdiv(x, (Fraction(norm), Fraction(0))),
                                   "source residual") for x in residual_rational])
    # Complete QR supplies an orthonormal complement without selecting a ground.
    U, _ = np.linalg.qr(e[:, None], mode="complete")
    W = U[:, 1:]
    AW = _safe_matmul(A, W)
    Braw = _safe_matmul(W.conj().T, AW)
    hermitian_residual = _norm(Braw - Braw.conj().T)
    B = (Braw + Braw.conj().T) / 2
    B = _normal(B, "Hermitian compression")
    b = np.zeros(n - 1, dtype=complex) if structural_b_zero else _safe_matmul(W.conj().T, residual[:, None])[:, 0]
    if not structural_b_zero and not np.any(b != 0):
        raise NumericalUnavailable("nonzero source residual lost in complementary projection")
    br = [_rational(x) for x in b]
    beta2 = _component(_rdot([_rconj(x) for x in br], br)[0], "beta squared")
    # B may have a degenerate minimum; module2's unique-ground helper is NOT used.
    Bscale = _norm(B)
    if Bscale:
        Bvalues, Bvectors = np.linalg.eigh(_normal(B.astype(np.clongdouble) / Bscale, "scaled B"))
        Bvalues = _normal(Bvalues.astype(np.longdouble) * Bscale, "B eigenvalues")
    else:
        Bvalues, Bvectors = np.zeros(n - 1), np.eye(n - 1, dtype=complex)
    diagnostics = {
        "source_normalization_residual": float(abs(np.vdot(e, e) - 1)),
        "basis_orthogonality_residual": _norm(W.conj().T @ W - np.eye(n - 1)),
        "source_complement_residual": _norm(W.conj().T @ e),
        "compression_hermiticity_residual": hermitian_residual,
        "compression_eigenpair_residual": _norm(B @ Bvectors - Bvectors * Bvalues),
        "block_coupling_residual": _norm(W.conj().T @ (A @ e) - b),
        "source_rayleigh_residual": float(abs(np.vdot(e, A @ e) - a)),
        "variance_subtraction_raw": float(np.vdot(A @ e, A @ e).real - a * a),
        "variance_identity_residual": float(abs(np.vdot(A @ e, A @ e).real - a * a - beta2)),
        "structural_b_zero": structural_b_zero,
        "status": "floating_diagnostics_not_certificates",
    }
    base.update({"status": "active", "s": s, "e": e, "W": W, "a": a,
                 "b": b, "B": B, "beta_squared": beta2,
                 "B_energies": Bvalues, "B_vectors": Bvectors, "diagnostics": diagnostics})
    return base


def _active(projected):
    if projected["status"] == "zero_source":
        raise ValueError("zero source: normalized projection, self energy and memory are not applicable")


def _positive_spectral_sum(values, weights, z):
    """Exact rational sum for binary64 poles and nonnegative binary64 weights."""
    zr, zi = _rational(z)
    total = (Fraction(0), Fraction(0))
    for energy, weight in zip(values, weights):
        if weight:
            total = _radd(total, _rdiv((Fraction(float(weight)), Fraction(0)),
                                      (zr - Fraction(float(energy)), zi)))
    return _rcomplex(total, "spectral resolvent")


def _weights(vectors, source):
    amplitudes = _safe_matmul(vectors.conj().T, source[:, None])[:, 0]
    result = []
    for a in amplitudes:
        ar, ai = _rational(a)
        result.append(_component(ar * ar + ai * ai, "spectral weight"))
    return np.array(result)


@_guard
def self_energy(projected, z):
    """b†(z-B)^-1 b on ran Q, with finite normal Im(z)>0."""
    z = _frequency(z)
    _active(projected)
    if projected["diagnostics"]["structural_b_zero"]:
        return 0j
    if "B_weights" not in projected:
        projected["B_weights"] = _weights(projected["B_vectors"], projected["b"])
    return _positive_spectral_sum(projected["B_energies"], projected["B_weights"], z)


@_guard
def source_resolvent(projected, z):
    """s/(z-a-Sigma), zero for an exactly zero vector; UHP only."""
    z = _frequency(z)
    if projected["status"] == "zero_source":
        return 0j
    sigma = self_energy(projected, z)
    zr, zi = _rational(z)
    sr, si = _rational(sigma)
    denominator = (zr - Fraction(projected["a"]) - sr, zi - si)
    return _rcomplex(_rdiv((Fraction(projected["s"]), Fraction(0)), denominator), "source resolvent")


@_guard
def memory_kernel(projected, t):
    """Finite memory K(t); signed real times permit the conjugation check.

    This is an almost-periodic spectral sum, not a damping model. Tiny physical
    sources or couplings are not clipped. Cancellation-dominated transcendental
    sums are explicitly unavailable rather than falsely reported as exact zero.
    """
    t = _scalar(t, "t", real=True)
    _active(projected)
    if projected["diagnostics"]["structural_b_zero"]:
        return 0j
    if t == 0:
        return complex(projected["beta_squared"])
    if "B_weights" not in projected:
        projected["B_weights"] = _weights(projected["B_vectors"], projected["b"])
    phase = _normal(projected["B_energies"].astype(np.longdouble) * t, "memory phase")
    # Large phase reduction is outside this diagnostic's arithmetic contract.
    if np.max(np.abs(phase)) > 1e12:
        raise NumericalUnavailable("memory phase too large for supported reduction")
    terms = projected["B_weights"].astype(np.clongdouble) * np.exp(-1j * phase.astype(np.clongdouble))
    value = np.sum(terms)
    for component, scale in ((value.real, np.sum(np.abs(terms.real))),
                             (value.imag, np.sum(np.abs(terms.imag)))):
        if scale and abs(component) <= 1e-12 * scale:
            raise NumericalUnavailable("memory sum has unresolved transcendental cancellation")
    return complex(_normal(value, "memory kernel"))


def _connected_density(system, diagonal, uniform=False):
    if uniform:
        return np.zeros(len(diagonal), dtype=complex)
    ground = system["ground"]
    gr = [_rational(x) for x in ground]
    dr = [_rational(x) for x in diagonal]
    norm = _rdot([_rconj(x) for x in gr], gr)
    raw = [_rmul(d, v) for d, v in zip(dr, gr)]
    mean = _rdiv(_rdot([_rconj(x) for x in gr], raw), norm)
    return np.array([_rcomplex(_radd(w, tuple(-x for x in _rmul(mean, v))), "connected density source")
                     for w, v in zip(raw, gr)])


def _solve_diagnostic(A, v, z):
    """Independent direct full-space inverse, not a claimed certified solve."""
    M = z * np.eye(len(A)) - A
    x = np.linalg.solve(M, v)
    residual = _norm(v - M @ x)
    singular = np.linalg.svd(M, compute_uv=False)
    return complex(np.vdot(v, x)), residual, float(singular[0] / singular[-1])


def _projection_report(p):
    responses, memory = [], []
    if p["status"] == "zero_source":
        return {"source": {"status": "zero_source", "s": 0.0, "a": None, "beta_squared": None},
                "projection_diagnostics": {}, "responses": [], "memory": [], "regulator_comparisons": []}
    s, a, beta2 = p["s"], p["a"], p["beta_squared"]
    full_weights = _weights(p["system"]["vectors"], p["source"])
    for z in FREQUENCIES:
        F, sigma = source_resolvent(p, z), self_energy(p, z)
        spectral = _positive_spectral_sum(p["system"]["gaps"], full_weights, z)
        direct, residual, condition = _solve_diagnostic(p["A"], p["source"], z)
        sigma_direct, bresidual, bcondition = _solve_diagnostic(p["B"], p["b"], z)
        bx = np.linalg.solve(z * np.eye(len(p["B"])) - p["B"], p["b"])
        one_line = s / (z - a)
        responses.append({"z": _json_complex(z), "F": _json_complex(F), "Sigma": _json_complex(sigma),
                          "full_spectral_F": _json_complex(spectral), "full_resolvent_F": _json_complex(direct),
                          "spectral_residual": float(abs(F - spectral)), "full_resolvent_residual": float(abs(F - direct)),
                          "self_energy_direct_residual": float(abs(sigma - sigma_direct)),
                          "full_solve_residual": residual, "B_solve_residual": bresidual,
                          "full_condition_estimate": condition, "B_condition_estimate": bcondition,
                          "self_energy_residual_bound_diagnostic": float(np.sqrt(beta2) * bresidual / z.imag),
                          "passivity_identity_residual": float(abs(sigma.imag + z.imag * np.vdot(bx, bx).real)),
                          "minus_Im_Sigma": float(-sigma.imag), "denominator_imag": float(z.imag - sigma.imag),
                          "F_abs": float(abs(F)), "F_bound": float(s / z.imag),
                          "Sigma_abs": float(abs(sigma)), "Sigma_bound": float(beta2 / z.imag),
                          "one_line_error": float(abs(F - one_line)), "one_line_bound": float(s * beta2 / z.imag ** 3),
                          "F_transform_tail_bound_T2": float(s * np.exp(-2 * z.imag) / z.imag),
                          "Sigma_transform_tail_bound_T2": float(beta2 * np.exp(-2 * z.imag) / z.imag)})
    normalized_weights = full_weights / s
    for t in TIMES:
        K = memory_kernel(p, t)
        c = np.sum(normalized_weights * np.exp(-1j * p["system"]["gaps"] * t))
        state = p["system"]["vectors"] @ (np.exp(-1j * p["system"]["gaps"] * t) *
                                                  (p["system"]["vectors"].conj().T @ p["e"]))
        leakage = _norm(p["W"].conj().T @ state) ** 2
        memory.append({"t": t, "K": _json_complex(K), "K_abs": float(abs(K)), "K_bound": beta2,
                       "reality_residual": float(abs(memory_kernel(p, -t) - K.conjugate())),
                       "c": _json_complex(c), "one_line_amplitude_error": float(abs(c - np.exp(-1j * a * t))),
                       "one_line_amplitude_bound": float(min(2, beta2 * t * t / 2)),
                       "complementary_leakage": float(leakage), "leakage_bound": float(min(1, beta2 * t * t))})
    regulators = []
    for i, eta in enumerate(REGULATORS):
        for eta2 in REGULATORS[i + 1:]:
            regulators.append({"eta": eta, "eta_prime": eta2,
                               "F_difference": float(abs(source_resolvent(p, 1 + 1j * eta) - source_resolvent(p, 1 + 1j * eta2))),
                               "F_bound": float(s * abs(eta - eta2) / (eta * eta2)),
                               "Sigma_difference": float(abs(self_energy(p, 1 + 1j * eta) - self_energy(p, 1 + 1j * eta2))),
                               "Sigma_bound": float(beta2 * abs(eta - eta2) / (eta * eta2))})
    return {"source": {"status": p["status"], "s": s, "a": a, "beta_squared": beta2},
            "projection_diagnostics": p["diagnostics"], "responses": responses, "memory": memory,
            "regulator_comparisons": regulators}


@_guard
def quantum_case(L, C=1.0, g=0.7, m=1):
    """JSON-safe complete fixed-N=L density diagnostics, using two adjoint sources."""
    L, C, g = _ring(L, C, g)
    m = _momentum(L, m)
    model = fixed_number_model(L, L, C, g)
    system = spectral_system(model.H)
    diagonal = density_diagonal(model, m)
    source = _connected_density(system, diagonal, uniform=m == 0)
    adjoint_source = _connected_density(system, diagonal.conjugate(), uniform=m == 0)
    p, adjoint = _projected(system, source), _projected(system, adjoint_source)
    report = _projection_report(p)
    for response, z in zip(report["responses"], FREQUENCIES):
        reconstructed = source_resolvent(adjoint, z) + source_resolvent(p, -z.conjugate()).conjugate()
        reference = retarded_response(system, np.diag(diagonal), np.diag(diagonal.conjugate()), z)
        response.update({"retarded_two_source": _json_complex(reconstructed),
                         "retarded_module2": _json_complex(reference),
                         "retarded_identity_residual": float(abs(reconstructed - reference))})
    uniform = _projected(system, np.zeros(len(model.basis), dtype=complex))
    report.update({"parameters": {"L": L, "N": L, "C": C, "g": g, "m": m, "dimension": len(model.basis)},
                   "ground": {"energy": float(system["energies"][0]), "gap": system["ground_gap"],
                              "resolution": system["resolution"], "status": "resolved_nondegenerate"},
                   "adjoint_source": {"s": adjoint["s"], "a": adjoint["a"], "beta_squared": adjoint["beta_squared"]},
                   "source_connectedness_residual": float(abs(np.vdot(system["ground"], source))),
                   "distinct_adjoint_vector_difference": _norm(source - adjoint_source),
                   "uniform_density_control": {"status": uniform["status"], "F": _json_complex(source_resolvent(uniform, 0.5j)),
                                               "self_energy": None, "memory": None,
                                               "reason": "fixed total number makes connected uniform density exactly zero"},
                   "limitations": list(LIMITATIONS)})
    return report


def _classical_oracle(report):
    p = report["parameters"]
    L, C, g, nbar = p["L"], p["C"], p["g"], p["nbar"]
    laplacian = np.zeros((L, L))
    for x in range(L):
        edge = np.zeros(L)
        edge[x], edge[(x + 1) % L] = 1, -1
        laplacian += np.outer(edge, edge)
    Kn, Ktheta = g * np.eye(L) + C * laplacian / (2 * nbar), 2 * C * nbar * laplacian
    cartesian = np.block([[np.zeros((L, L)), C * laplacian],
                          [-(C * laplacian + 2 * g * nbar * np.eye(L)), np.zeros((L, L))]])
    transform = np.diag([2 * np.sqrt(nbar)] * L + [1 / np.sqrt(nbar)] * L)
    canonical = np.block([[np.zeros((L, L)), Ktheta], [-Kn, np.zeros((L, L))]])
    residuals = []
    for mode in report["modes"]:
        v = np.exp(1j * mode["k"] * np.arange(L)) / np.sqrt(L)
        residuals.append(max(_norm(Kn @ v - mode["alpha"] * v), _norm(Ktheta @ v - mode["beta"] * v)))
    report["independent_oracle"] = {"hessian_mode_max_residual": max(residuals),
                                    "cartesian_jacobian_residual": _norm(transform @ cartesian @ np.linalg.inv(transform) - canonical)}
    return report


def demonstration_report():
    """Exactly the frozen controls, strict-JSON-safe and without artifacts."""
    toy = projected_system(np.diag([0.0, 1.0, 3.0]), np.array([0, 1, 1]) / np.sqrt(2))
    toy_report = _projection_report(toy)
    for item, z in zip(toy_report["responses"], FREQUENCIES):
        item["analytic_F_residual"] = float(abs(source_resolvent(toy, z) - (z - 2) / ((z - 1) * (z - 3))))
        item["analytic_Sigma_residual"] = float(abs(self_energy(toy, z) - 1 / (z - 2)))
    for item, t in zip(toy_report["memory"], TIMES):
        item["analytic_K_residual"] = float(abs(memory_kernel(toy, t) - np.exp(-2j * t)))
    single = projected_system(np.diag([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]))
    zero = projected_system(np.diag([0.0, 1.0, 3.0]), np.zeros(3))
    return {"model_id": MODEL_ID,
            "frozen_controls": {"classical": [list(x) for x in CLASSICAL_CONTROLS],
                                "quantum": [list(x) for x in QUANTUM_CONTROLS],
                                "frequencies": [_json_complex(z) for z in FREQUENCIES],
                                "regulators": list(REGULATORS), "times": list(TIMES)},
            "classical_cases": [_classical_oracle(classical_modes(*x)) for x in CLASSICAL_CONTROLS],
            "quantum_cases": [quantum_case(*x) for x in QUANTUM_CONTROLS],
            "toy_controls": {"two_transition": toy_report, "one_transition": _projection_report(single),
                             "zero_source": _projection_report(zero),
                             "self_energy_pole_is_source_zero": {"frequency": 2.0, "analytic_F_zero": 0.0},
                             "finite_memory_nondecay": "nonzero finite spectral memory is almost periodic; samples prove no decay law"},
            "limitations": list(LIMITATIONS)}
