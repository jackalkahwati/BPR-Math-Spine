"""Finite Bose-ring number currents and external-source response.

Conventions and controls are frozen in substrate_current_response_2026-09-12.md.
Numerical resolution screens are heuristic, not error certificates. Peierls
phases are external probes, not a physical gauge field or an elastic modulus.
"""
from functools import wraps
from fractions import Fraction
from hashlib import sha256
from numbers import Complex, Integral

import numpy as np

from bpr.substrate_fermionization import fixed_number_model
from bpr.substrate_neutral_response import bilinear, density_diagonal

MAX_DENSE_DIMENSION = 512
MODEL_ID = "conditional-substrate-current-response-v1"
FROZEN_CONTROLS = tuple((L, 1.0, g) for g in (40.0, 0.7) for L in (3, 4, 5))
FREQUENCIES = (0.5j, 1.0 + 0.5j, 4.0 + 1.0j)
FLUX_STEPS = (0.04, 0.02, 0.01, 0.005)
LIMITATIONS = (
    "The quantum Bose ring and unit filling are stipulated, not derived classical quantization.",
    "Peierls phases are external sources; covariance supplies no physical gauge field or Gauss constraint.",
    "Finite-ring flux curvature is not a thermodynamic stiffness, physical modulus or superfluid classification.",
    "Arithmetic and finite-difference resolution screens are heuristic, not certified error bounds or convergence proofs.",
    "No physical energy or length calibration, damping, chemical potential fit or thermodynamic extrapolation is supplied.",
)
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny


class NumericalUnavailable(ValueError):
    """The supplied arithmetic does not resolve the requested diagnostic."""


def _guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise NumericalUnavailable("numerically unavailable arithmetic") from exc
    return checked


def _component(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must have finite binary64 components") from exc
    if not np.isfinite(result) or (result != 0 and abs(result) < _TINY):
        raise NumericalUnavailable(name + " has nonfinite or subnormal components")
    if result == 0 and value != 0:
        raise NumericalUnavailable(name + " loses a nonzero component")
    return result


def _scalar(value, name, real=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise ValueError(name + " must be a numeric scalar")
    r, i = _component(value.real, name), _component(value.imag, name)
    if real:
        if i != 0:
            raise ValueError(name + " must be real")
        return r
    return complex(r, i)


def _normal(value, name):
    """Componentwise conversion, including checks for erased extended values."""
    source = np.asarray(value)
    dtype = complex if np.iscomplexobj(source) else float
    result = source.astype(dtype)
    for before, after in ((source.real, result.real), (source.imag, result.imag)):
        if not np.all(np.isfinite(after)) or np.any((after != 0) & (np.abs(after) < _TINY)):
            raise NumericalUnavailable(name + " has nonfinite or subnormal components")
        if np.any((before != 0) & (after == 0)):
            raise NumericalUnavailable(name + " loses a nonzero component")
    return result.item() if result.ndim == 0 else result


def _matrix(value, name, dimension=None):
    """Inspect bounded shape before any array conversion or dense allocation."""
    if isinstance(value, np.ndarray):
        shape = value.shape
    elif isinstance(value, (list, tuple)):
        n = len(value)
        if not 1 <= n <= MAX_DENSE_DIMENSION:
            raise ValueError(name + " exceeds dense cap or is empty")
        if any(not isinstance(row, (list, tuple, np.ndarray)) for row in value):
            raise ValueError(name + " must be a square matrix")
        if any(len(row) != n for row in value):
            raise ValueError(name + " must be a square matrix")
        shape = (n, n)
    else:
        # In particular, do not invoke arbitrary __array__ on unbounded inputs.
        raise ValueError(name + " must be a bounded array or nested sequence")
    if len(shape) != 2 or shape[0] != shape[1] or not 1 <= shape[0] <= MAX_DENSE_DIMENSION:
        raise ValueError(name + " must be square with dimension at most 512")
    if dimension is not None and shape != (dimension, dimension):
        raise ValueError(name + " has wrong shape")
    if isinstance(value, np.ndarray) and value.dtype.kind in "fc" and value.dtype.itemsize <= 16:
        result = _normal(value, name)
    else:
        result = np.empty(shape, dtype=complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                result[i, j] = _scalar(value[i][j], name)
    _norm(result, name + " norm")
    return result


def _norm(value, name="norm"):
    # Extended arithmetic avoids overflow in abs(complex) and sum of squares.
    a = np.asarray(value, dtype=np.clongdouble)
    scale = np.max(np.maximum(np.abs(a.real), np.abs(a.imag)))
    if scale == 0:
        return 0.0
    scaled = a / scale
    return float(_normal(scale * np.sqrt(np.sum(scaled.real ** 2 + scaled.imag ** 2)), name))


def _phases(model, phases):
    if phases is None:
        return np.zeros(model.L)
    if not isinstance(phases, (list, tuple, np.ndarray)) or len(phases) != model.L:
        raise ValueError("phases must have one real scalar per link")
    return np.array([_scalar(a, "phase", real=True) for a in phases])


def _model_shape(model):
    if not isinstance(model.L, Integral) or model.L < 3:
        raise ValueError("model must be an existing Bose ring")
    _matrix(model.H, "model H")
    if len(model.basis) != model.H.shape[0]:
        raise ValueError("model basis and Hamiltonian disagree")


@_guard
def bond_currents(model, phases=None):
    """Outgoing j_x(A)=i C (exp(i A_x) T_x - exp(-i A_x) T_x†)."""
    _model_shape(model)
    phases = _phases(model, phases)
    result = []
    for x, a in enumerate(phases):
        T = bilinear(model, (x + 1) % model.L, x)
        hop = np.exp(1j * a) * T
        result.append(_normal(1j * model.C * (hop - hop.conj().T), "bond current"))
    return result


@_guard
def peierls_hamiltonian(model, phases):
    """Complete fixed-number H(A); phases specify outgoing directed links."""
    _model_shape(model)
    phases = _phases(model, phases)
    result = np.diag(model.g * model.D).astype(complex)
    for x, a in enumerate(phases):
        T = bilinear(model, (x + 1) % model.L, x)
        hop = np.exp(1j * a) * T
        result -= model.C * (hop + hop.conj().T)
    return _normal(result, "Peierls Hamiltonian")


@_guard
def spectral_system(H):
    """One reusable eigendecomposition with a resolved nondegenerate ground.

    The returned arrays describe the supplied binary64 Hamiltonian. Ground
    resolution incorporates matrix scale and eigensolver residual heuristically.
    No physical source is assigned to a degenerate/unresolved eigenvector.
    """
    H = _matrix(H, "H").astype(complex, copy=True)
    if not np.array_equal(H, H.conj().T):
        raise ValueError("H must be Hermitian")
    n = len(H)
    if n < 2:
        raise ValueError("a response system requires at least two states")
    scale = _norm(H, "Hamiltonian norm")
    if scale == 0:
        raise NumericalUnavailable("degenerate ground state")
    scaled = _normal(H.astype(np.clongdouble) / scale, "scaled Hamiltonian")
    e, vectors = np.linalg.eigh(scaled)
    energies = _normal(e.astype(np.longdouble) * scale, "eigenvalues")
    gaps = _normal(energies.astype(np.longdouble) - energies[0], "energy gaps")
    residual = _norm(H @ vectors - vectors * energies, "eigensolver residual")
    resolution = _normal(max(64 * _EPS * n * scale, 8 * residual), "spectral resolution")
    if gaps[1] <= resolution:
        raise NumericalUnavailable("ground state is degenerate or numerically unresolved")
    for array in (H, energies, vectors, gaps):
        array.setflags(write=False)
    return {"H": H, "energies": energies, "vectors": vectors,
            "ground": vectors[:, 0], "gaps": gaps, "resolution": float(resolution),
            "ground_gap": float(gaps[1])}


def _stationary_zero(system, A):
    # Exact input identities, not a tolerance on a noisy transition weight.
    if np.array_equal(A, system["H"]):
        return True
    diagonal = np.diag(A)
    return bool(np.all(diagonal == diagonal[0]) and
                np.array_equal(A, np.diag(diagonal)))


def _rational(z):
    return Fraction(float(z.real)), Fraction(float(z.imag))


def _radd(a, b):
    return a[0] + b[0], a[1] + b[1]


def _rmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _rdiv(a, b):
    denominator = b[0] ** 2 + b[1] ** 2
    return ((a[0] * b[0] + a[1] * b[1]) / denominator,
            (a[1] * b[0] - a[0] * b[1]) / denominator)


def _rcomplex(a, name):
    return complex(_component(a[0], name), _component(a[1], name))


def _rconj(a):
    return a[0], -a[1]


def _rdot(left, right):
    total = (Fraction(0), Fraction(0))
    for a, b in zip(left, right):
        if a != (0, 0) and b != (0, 0):
            total = _radd(total, _rmul(a, b))
    return total


def _dyadic(values):
    values = np.asarray(values, dtype=complex)
    ratios = [float(component).as_integer_ratio()
              for part in (values.real, values.imag) for component in part.flat]
    denominator = max(d for _, d in ratios)
    parts = np.array([n * (denominator // d) for n, d in ratios], dtype=object)
    return (parts[:values.size].reshape(values.shape),
            parts[values.size:].reshape(values.shape)), denominator


def _imatmul(a, b):
    return a[0] @ b[0] - a[1] @ b[1], a[0] @ b[1] + a[1] @ b[0]


def _exact_transitions(system, A, indices=None):
    """End-to-end exact connected contractions of the supplied binary64 data.

    Preserve fractions through matvec and elastic subtraction; rounding an
    intermediate A|0> first would erase small pieces beside a large H source.
    """
    if indices is None:
        indices = np.arange(len(A) - 1)
    # Binary64 inputs are dyadic. Common power-of-two denominators let object
    # integer BLAS loops retain every bit without Fraction gcd at each multiply.
    v, vd = _dyadic(system["ground"])
    matrix, ad = _dyadic(A)
    w, wd = _dyadic(system["vectors"][:, np.asarray(indices) + 1])
    vc = (v[0], -v[1])
    wc = (w[0].T, -w[1].T)
    right = _imatmul(matrix, v)
    left = _imatmul(vc, matrix)
    norm = _imatmul(vc, v)[0]
    expectation = _imatmul(vc, right)
    left_direct, right_direct = _imatmul(left, w), _imatmul(wc, right)
    left_overlap, right_overlap = _imatmul(vc, w), _imatmul(wc, v)
    denominator = int(ad * vd * wd * norm)
    result = []
    for direct, overlap in ((left_direct, left_overlap), (right_direct, right_overlap)):
        er = expectation[0] * overlap[0] - expectation[1] * overlap[1]
        ei = expectation[0] * overlap[1] + expectation[1] * overlap[0]
        real, imag = direct[0] * norm - er, direct[1] * norm - ei
        result.append(np.array([_rcomplex((Fraction(int(r), denominator), Fraction(int(i), denominator)),
                                          "connected source projection")
                                for r, i in zip(real, imag)]))
    return tuple(result)


def _transitions(system, A):
    """Connected A_0n and A_n0, with bounded cached cancellation fallback."""
    if _stationary_zero(system, A):
        zero = np.zeros(len(A) - 1, dtype=complex)
        return zero, zero.copy()
    key = (A.dtype.str, sha256(A.tobytes()).digest())
    cache = system.setdefault("_transition_cache", {})
    if key in cache:
        return cache[key]
    v, excited = system["ground"], system["vectors"][:, 1:]
    with np.errstate(all="ignore"):
        right, left = A @ v, v.conj() @ A
        expectation = np.vdot(v, right) / np.vdot(v, v)
        a0n = left @ excited - expectation * (v.conj() @ excited)
        an0 = excited.conj().T @ right - expectation * (excited.conj().T @ v)
        # Bound the absolute terms of each COMPLETE contraction, not just its
        # last dot. This catches an erased small part of an enormous H source.
        absolute = np.abs(A)
        left_scale = np.abs(v) @ absolute @ np.abs(excited)
        right_scale = np.abs(excited).T @ absolute @ np.abs(v)
        left_scale += abs(expectation) * (np.abs(v) @ np.abs(excited))
        right_scale += abs(expectation) * (np.abs(excited).T @ np.abs(v))
        # Each channel and each component must stand alone: another large
        # transition can be removed by B and cannot resolve this channel.
        suspicious = np.full(len(A) - 1, len(A) <= 8, dtype=bool)
        for values, scales in ((a0n, left_scale), (an0, right_scale)):
            material = scales * (64 * _EPS * len(A))
            suspicious |= ~np.isfinite(values) | ~np.isfinite(scales)
            for component in (values.real, values.imag):
                suspicious |= (scales != 0) & (material > 1e-10 * np.abs(component))
                suspicious |= (component != 0) & (np.abs(component) < _TINY)
    indices = np.flatnonzero(suspicious)
    if len(indices):
        exact_left, exact_right = _exact_transitions(system, A, indices)
        a0n[indices], an0[indices] = exact_left, exact_right
    result = (_normal(a0n, "source projection"), _normal(an0, "source projection"))
    if len(cache) >= 64:
        cache.clear()
    cache[key] = result
    return result


@_guard
def retarded_response(system, A, B, z):
    """chi_AB(z), Im(z)>0; mixed numerators and actual eigenvalue gaps.

    The elastic n=0 component is excluded after connected subtraction. A
    supplied scalar identity or the system Hamiltonian gives exact zero by
    stationarity. Small arbitrary operators are never clipped. Eigensolver
    roundoff remains distinct from these structural statements.
    """
    n = len(system["energies"])
    if n > MAX_DENSE_DIMENSION:
        raise ValueError("system exceeds dense cap")
    A, B = _matrix(A, "A", n), _matrix(B, "B", n)
    z = _scalar(z, "z")
    if z.imag <= 0:
        raise ValueError("retarded z must lie in the upper half plane")
    if _stationary_zero(system, A) or _stationary_zero(system, B):
        return 0j
    a0n, an0 = _transitions(system, A)
    b0n, bn0 = _transitions(system, B)
    gaps = system["gaps"][1:]
    with np.errstate(all="ignore"):
        numerator1, numerator2 = a0n * bn0, b0n * an0
        first, second = numerator1 / (z - gaps), numerator2 / (z + gaps)
        value = np.sum(first - second)
        scale_real = float(np.sum(np.abs(first.real) + np.abs(second.real)))
        scale_imag = float(np.sum(np.abs(first.imag) + np.abs(second.imag)))
        scale_all = float(np.sum(np.abs(first) + np.abs(second)))
        fallback = not np.isfinite(scale_all)
        for component, scale in ((value.real, scale_real), (value.imag, scale_imag)):
            fallback = fallback or not np.isfinite(component)
            fallback = fallback or (scale != 0 and abs(component) <= 1e-8 * scale)
            fallback = fallback or (component != 0 and abs(component) < _TINY)
        fallback = fallback or bool(np.any((a0n != 0) & (bn0 != 0) & (numerator1 == 0)))
        fallback = fallback or bool(np.any((b0n != 0) & (an0 != 0) & (numerator2 == 0)))
        fallback = fallback or bool(np.any((numerator1 != 0) & (first == 0)))
        fallback = fallback or bool(np.any((numerator2 != 0) & (second == 0)))
        # Extremely disparate real/imaginary frequency components can erase a
        # component in complex division even without cancellation in the sum.
        fallback = fallback or (z.real != 0 and
                                min(abs(z.imag / z.real), abs(z.real / z.imag)) < 1e-12)
        # Check source products before either complex multiplication can erase
        # a component, and denominators before subtraction hides z.real.
        for a, b in ((a0n, bn0), (b0n, an0)):
            for x, y, u, v in ((a.real, b.real, a.imag, b.imag),
                               (a.real, b.imag, -a.imag, b.real)):
                p, q = x * y, u * v
                scale = np.abs(p) + np.abs(q)
                fallback = fallback or bool(np.any((scale != 0) & (np.abs(p - q) <= 1e-8 * scale)))
                fallback = fallback or bool(np.any((x != 0) & (y != 0) & (p == 0)))
                fallback = fallback or bool(np.any((u != 0) & (v != 0) & (q == 0)))
        if z.real != 0:
            fallback = fallback or bool(np.any((z.real - gaps == -gaps) | (z.real + gaps == gaps)))
    if not fallback:
        return complex(_normal(value, "retarded response"))
    # Exact supplied-binary64 fallback; no certificate for the eigensolver.
    zr, zi = _rational(z)
    total = (Fraction(0), Fraction(0))
    for a, b, c, d, gap in zip(a0n, bn0, b0n, an0, gaps):
        delta = Fraction(float(gap))
        first = _rdiv(_rmul(_rational(a), _rational(b)), (zr - delta, zi))
        second = _rdiv(_rmul(_rational(c), _rational(d)), (zr + delta, zi))
        total = _radd(total, (first[0] - second[0], first[1] - second[1]))
    return _rcomplex(total, "retarded response")


def _expect(system, A):
    v = system["ground"].astype(np.clongdouble)
    return complex(_normal(np.vdot(v, A.astype(np.clongdouble) @ v), "expectation"))


def _comm(A, B):
    return A @ B - B @ A


def _json_complex(z):
    return {"real": float(z.real), "imag": float(z.imag)}


def _curvature(model, system, currents):
    L = model.L
    J = sum(currents)
    kinetic = _expect(system, model.V).real
    dia = -kinetic / L ** 2
    _, jn0 = _transitions(system, J)
    param_raw = float(_normal(2 * np.sum(np.abs(jn0) ** 2 / system["gaps"][1:]) / L ** 2,
                              "paramagnetic curvature"))
    free = model.g == 0
    # At g=0 the unique k=0 condensate is annihilated by total current.
    param = 0.0 if free else param_raw
    difference = dia - param
    resolution = float(64 * _EPS * len(model.basis) * (abs(dia) + abs(param)) +
                       system["resolution"] * (1 + param / system["ground_gap"]) / L ** 2)
    status = "resolved" if abs(difference) > resolution else "unresolved_numerically_unavailable"
    value = difference if status == "resolved" else None
    if free:
        value = 2 * model.C * model.N / L ** 2
        status = "structural_free_control"
    differences = []
    for h in FLUX_STEPS:
        plus = spectral_system(peierls_hamiltonian(model, np.full(L, h / L)))
        minus = spectral_system(peierls_hamiltonian(model, np.full(L, -h / L)))
        numerator = ((np.longdouble(plus["energies"][0]) - system["energies"][0]) +
                     (np.longdouble(minus["energies"][0]) - system["energies"][0]))
        raw = float(_normal(numerator / h ** 2, "finite difference"))
        fd_resolution = (plus["resolution"] + minus["resolution"] +
                         2 * system["resolution"]) / h ** 2
        resolved = abs(raw) > fd_resolution
        differences.append({"step": h, "raw_value": raw,
                            "value": raw if resolved else None,
                            "status": "resolved_heuristic" if resolved else "unresolved_numerically_unavailable",
                            "resolution": float(fd_resolution),
                            "raw_error_against_curvature": float(raw - value) if value is not None else None,
                            "energy_rise_plus": float(plus["energies"][0] - system["energies"][0]),
                            "energy_rise_upper_bound": float(-kinetic * (2 * np.sin(h / (2 * L)) ** 2))})
    return {"diamagnetic": dia, "paramagnetic": param,
            "paramagnetic_raw_roundoff_diagnostic": param_raw,
            "paramagnetic_structural_zero": free,
            "raw_difference": difference, "value": value, "status": status,
            "resolution": resolution,
            "condition_ratio": abs(difference) / (abs(dia) + abs(param)) if dia or param else None,
            "normalization": "d²E0/dPhi² for total flux Phi; link phase Phi/L",
            "finite_differences": differences,
            "finite_difference_convergence_claimed": False}


def _gauge_report(model):
    L = model.L
    theta = 0.2 * np.cos(2 * np.pi * np.arange(L) / L)
    phases = np.full(L, 0.3 / L)
    shifted = phases + np.roll(theta, -1) - theta
    U = np.exp(1j * (np.asarray(model.basis) @ theta))
    H = peierls_hamiltonian(model, phases)
    transformed = U[:, None] * H * U.conj()[None, :]
    original_currents = bond_currents(model, phases)
    new_currents = bond_currents(model, shifted)
    current_residual = max(_norm(new - U[:, None] * old * U.conj()[None, :])
                           for new, old in zip(new_currents, original_currents))
    # Independently redistribute the same total flux onto the last link.
    single = np.zeros(L)
    single[-1] = 0.3
    redistribution = -0.3 * np.arange(L) / L
    W = np.exp(1j * (np.asarray(model.basis) @ redistribution))
    return {"hamiltonian_residual": _norm(peierls_hamiltonian(model, shifted) - transformed),
            "current_max_residual": current_residual,
            "density_max_residual": max(_norm(np.diag(density_diagonal(model, m)) -
                                               U[:, None] * np.diag(density_diagonal(model, m)) * U.conj()[None, :])
                                        for m in range(L)),
            "flux_sum_residual": float(abs(np.sum(phases) - np.sum(shifted))),
            "single_link_hamiltonian_residual": _norm(peierls_hamiltonian(model, single) -
                                                       W[:, None] * H * W.conj()[None, :]),
            "density_commutation_structural": True}


@_guard
def case_report(L, C=1.0, g=40.0):
    """JSON-safe frozen-domain diagnostic; public rings have L=N=3,4,5."""
    if isinstance(L, (bool, np.bool_)) or not isinstance(L, Integral) or L not in (3, 4, 5):
        raise ValueError("public ring cases require L=3,4,5")
    C, g = _scalar(C, "C", real=True), _scalar(g, "g", real=True)
    if C <= 0 or g < 0:
        raise ValueError("C must be positive and g nonnegative")
    L = int(L)
    model = fixed_number_model(L, L, C, g)
    system = spectral_system(model.H)
    currents = bond_currents(model)
    H, E0 = model.H, system["energies"][0]
    local = []
    for x in range(L):
        n = bilinear(model, x, x)
        local.append(_norm(1j * _comm(H, n) - (currents[(x - 1) % L] - currents[x])))
    modes, fourier = [], []
    kinetic = _expect(system, model.V).real
    for m in range(L):
        k = 2 * np.pi * m / L
        q = np.exp(-1j * k) - 1 if m else 0j
        rho = np.diag(density_diagonal(model, m))
        j = sum(np.exp(-1j * k * x) * currents[x] for x in range(L)) / np.sqrt(L)
        dotrho, dotj = 1j * _comm(H, rho), 1j * _comm(H, j.conj().T)
        fourier.append(_norm(dotrho - q * j))
        _, transitions = _transitions(system, rho)
        moment = float(_normal(np.sum(system["gaps"][1:] * np.abs(transitions) ** 2), "density moment"))
        double = 0.5 * _expect(system, _comm(rho.conj().T, _comm(H, rho))).real
        target = -(1 - np.cos(k)) * kinetic / L if m else 0.0
        responses = []
        contact = _expect(system, _comm(rho, j.conj().T))
        analytic_contact = -1j * q * kinetic / L
        for z in FREQUENCIES:
            mixed = retarded_response(system, rho, j.conj().T, z)
            first = z * mixed - contact - 1j * retarded_response(system, dotrho, j.conj().T, z)
            second = z * mixed - contact + 1j * retarded_response(system, rho, dotj, z)
            density = retarded_response(system, rho, rho.conj().T, z)
            current_raw = retarded_response(system, j, j.conj().T, z)
            free_uniform = g == 0 and m == 0
            # The free k=0 condensate has J|0>=0, unlike a generic source.
            current = 0j if free_uniform else current_raw
            combined = z ** 2 * density - abs(q) ** 2 * (current - kinetic / L)
            responses.append({"z": _json_complex(z), "density": _json_complex(density),
                              "mixed": _json_complex(mixed), "current": _json_complex(current),
                              "current_structural_zero": free_uniform,
                              "current_raw_roundoff_diagnostic": _json_complex(current_raw),
                              "contact": _json_complex(contact),
                              "ward_first_residual": float(abs(first)),
                              "ward_second_residual": float(abs(second)),
                              "ward_combined_residual": float(abs(combined))})
        modes.append({"m": m, "k": float(k), "structural_density_zero": m == 0,
                      "fsum": {"spectral": moment, "double_commutator": double, "kinetic": target,
                               "residual": float(max(abs(moment - double), abs(moment - target)))},
                      "contact_analytic": _json_complex(analytic_contact),
                      "contact_residual": float(abs(contact - analytic_contact)), "responses": responses})
    return {"parameters": {"L": L, "N": L, "C": C, "g": g, "dimension": len(model.basis)},
            "ground": {"energy": float(E0), "gap": system["ground_gap"],
                       "resolution": system["resolution"], "status": "resolved_nondegenerate"},
            "continuity": {"local_max_residual": max(local), "fourier_max_residual": max(fourier),
                           "zero_phase_hamiltonian_residual": _norm(peierls_hamiltonian(model, np.zeros(L)) - H)},
            "momenta": modes, "curvature": _curvature(model, system, currents),
            "gauge": _gauge_report(model), "limitations": list(LIMITATIONS)}


def demonstration_report():
    """Frozen interacting cases and analytic free controls; no artifacts."""
    return {"model_id": MODEL_ID,
            "cases": [case_report(L, C, g) for L, C, g in FROZEN_CONTROLS],
            "free_controls": [case_report(L, 1.0, 0.0) for L in (3, 4, 5)],
            "limitations": list(LIMITATIONS)}
