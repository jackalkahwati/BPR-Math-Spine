"""Frozen classical long-wave controls and a bounded uniform-condensate bridge.

Contract: doc/derivations/substrate_longwave_limit_2026-09-12.md.
Classical operator errors are spectral 2-norms in the stated fixed coordinates
and norm weights. Quantum state/interaction errors are Euclidean vector norms;
energy and variance residuals are absolute scalar differences. Eigensystem
residuals explicitly use Frobenius norms. Analytic envelopes exclude roundoff:
no finite residual is clipped, including free and zero-time quantum cases.

The quantum model is the complete unshifted fixed-number Bose Hamiltonian.
No arbitrary-state dynamics, eigenspace selection or public cache is supplied.
"""
from functools import wraps
import math

import numpy as np

from bpr.substrate_vacuum_selection import capped_binomial as _capped_binomial

MAX_QUANTUM_DIMENSION = 512
_CLASSICAL_SIZES = (16, 32, 64, 128, 256)
_MODES = (1, 2)
_ACOUSTIC_COUPLINGS = (0.7, 40.0)
_CLASSICAL_TIMES = (0.0, 0.5, 1.0, 2.0)
_RESPONSE_FREQUENCIES = (0.5j, 1.0 + 0.5j, 4.0 + 1.0j)
_QUANTUM_SIZES = (3, 4, 5)
_QUANTUM_COUPLINGS = (0.0, 0.7, 40.0)
_QUANTUM_TIMES = (0.0, 0.1, 0.5, 1.0)
_LIMITATIONS = (
    "The supplied classical DNLS and stipulated complete quantum Bose Hamiltonian are distinct models.",
    "The classical limit is a linear acoustic or reversible quadratic dispersive limit, not a uniform nonlinear-DNLS theorem.",
    "Bounded rescaled data still require an independently small perturbation amplitude for linearization.",
    "Circumference one is mathematical, not a physical length or light-speed calibration.",
    "A uniform fixed-number condensate is a trial family, not an assumed interacting ground state.",
    "The normalized-orbital variational coupling is g(N-1); for psi=sqrt(N)u it is g(1-1/N), not the supplied classical g.",
    "The quantum Hilbert distance uses the mean-energy phase, not the uncorrected Hartree phase or a phase-optimized distance.",
    "Analytic envelopes and heuristic eigensystem screens are not certified floating-point error bounds.",
    "Raw free and zero-time residuals may exceed their zero analytic bound by roundoff; they are not clipped.",
    "A loose bound alone does not show trial dynamics fail, and finite residuals do not establish a thermodynamic theorem.",
    "No interacting quantum continuum, emergent 3+1 spacetime, relativistic interactions, gravity or empirical validation is derived.",
)


class NumericalUnavailable(ValueError):
    """Computed nonfinite arithmetic or an unresolved numerical eigensystem."""


def _integer(value, name, choices):
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(name + " must be a builtin or NumPy integer")
    if isinstance(value, (bool, np.bool_)) or value not in choices:
        raise ValueError(name + " outside the frozen integer grid")
    return int(value)


def _real(value, name, choices):
    if (type(value) not in (int, float)
            and not isinstance(value, (np.integer, np.floating))):
        raise ValueError(name + " must be a builtin or NumPy real scalar")
    if isinstance(value, (bool, np.bool_)) or value not in choices:
        raise ValueError(name + " must be finite and on the frozen grid")
    return float(value)


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable("nonfinite computed " + name)
    return value


def _json_native(value):
    """Detached report containers, with no arrays or nonfinite JSON numbers."""
    if isinstance(value, np.ndarray):
        _finite(value, "array")
        if np.iscomplexobj(value):
            return {"shape": list(value.shape), "real": value.real.tolist(),
                    "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, complex):
        return {"real": _json_native(value.real), "imag": _json_native(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        raise NumericalUnavailable("nonfinite computed report value")
    return value


def _guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return _json_native(function(*args, **kwargs))
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise NumericalUnavailable("unresolved floating arithmetic or linear algebra") from exc
    return checked


def _operator_norm(matrix, weights):
    """Spectral 2-norm after the fixed, h-independent diagonal similarity."""
    roots = np.sqrt(weights)
    transformed = roots[:, None] * matrix / roots[None, :]
    _finite(transformed, "fixed-norm matrix")
    return float(_finite(np.linalg.norm(transformed, ord=2), "operator norm"))


def _propagator(generator, frequency_squared, T):
    omega = np.sqrt(frequency_squared)
    return _finite(np.cos(omega * T) * np.eye(2)
                   + T * np.sinc(omega * T / np.pi) * generator, "propagator")


def _classical_report(L, m, g, acoustic):
    # Public wrappers validate every input before reaching any array allocation.
    C, nbar = 1.0, 1.0
    h, p = 1.0 / L, 2.0 * np.pi * m
    a_h = 4.0 * C * np.sin(h * p / 2.0)**2 / h**2
    a_0 = C * p**2
    a_second = a_0 - C * h**2 * p**4 / 12.0
    difference_bound = C * h**2 * p**4 / 12.0
    if acoustic:
        b_0 = 2.0 * nbar * a_0
        generator = np.array([[0.0, 2.0 * nbar * a_h],
                              [-g - h**2 * a_h / (2.0 * nbar), 0.0]])
        limit = np.array([[0.0, b_0], [-g, 0.0]])
        weights = np.array([g, b_0])
        frequency_squared = a_h * (2.0 * g * nbar + h**2 * a_h)
        limit_frequency_squared = 2.0 * g * nbar * a_0
        h_max, gamma = 1.0 / 16.0, 1.0 - np.pi**2 / 192.0
        omega_0 = np.sqrt(g * b_0)
        coefficient = omega_0 * max(p**2 / 12.0, C * p**2 / (2.0 * nbar * g))
        uniform_bound = np.sqrt(1.0 / gamma + h_max**2 * C * p**2 / (2.0 * nbar * g))
        generator_bound = h**2 * coefficient
        propagator_cap = uniform_bound + 1.0
        output_factor = 1.0
        response_norm_factor = np.sqrt(b_0 / g)
        microscopic_frequency_factor = h
        coordinates = ["r=rho/h", "theta"]
        time_scaling, source_scaling = "T=h*t", "f=h*F"
    else:
        rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])
        generator, limit = a_h * rotation, a_0 * rotation
        weights = np.ones(2)
        frequency_squared, limit_frequency_squared = a_h**2, a_0**2
        generator_bound = difference_bound
        uniform_bound, propagator_cap = 1.0, 2.0
        output_factor = 2.0 * nbar
        response_norm_factor = 2.0 * nbar
        microscopic_frequency_factor = h**2
        coordinates = ["rho/(2*nbar)", "theta"]
        time_scaling, source_scaling = "T=h^2*t", "f=h^2*F"

    propagator_cases = []
    for T in _CLASSICAL_TIMES:
        finite = _propagator(generator, frequency_squared, T)
        limiting = _propagator(limit, limit_frequency_squared, T)
        uncapped = uniform_bound * abs(T) * generator_bound
        propagator_cases.append({
            "T": T, "propagator": finite, "limit_propagator": limiting,
            "error_operator_norm": _operator_norm(finite - limiting, weights),
            "error_bound": min(propagator_cap, uncapped),
            "uncapped_error_bound": uncapped,
            "bound_status": "informative" if uncapped < propagator_cap else "trivial_cap",
        })

    response_cases = []
    source = np.array([0.0, -1.0])
    numerator = 2.0 * nbar * a_h
    limit_numerator = 2.0 * nbar * a_0
    for zeta in _RESPONSE_FREQUENCIES:
        eta = zeta.imag
        resolvent = _finite(np.linalg.inv(-1j * zeta * np.eye(2) - generator), "resolvent")
        limit_resolvent = _finite(np.linalg.inv(-1j * zeta * np.eye(2) - limit), "limit resolvent")
        response = numerator / (zeta**2 - frequency_squared)
        limit_response = limit_numerator / (zeta**2 - limit_frequency_squared)
        resolvent_error_bound = uniform_bound * generator_bound / eta**2
        response_cases.append({
            "zeta": zeta, "microscopic_z": microscopic_frequency_factor * zeta,
            "response": response, "limit_response": limit_response,
            "absolute_error": abs(response - limit_response),
            "error_bound": response_norm_factor * resolvent_error_bound,
            "resolvent_error_operator_norm": _operator_norm(resolvent - limit_resolvent, weights),
            "resolvent_error_bound": resolvent_error_bound,
            "resolvent_norm_bound": uniform_bound / eta,
            "response_contraction_residual": abs(response - output_factor * (resolvent @ source)[0]),
            "bound_kind": "analytic_absolute_envelope_not_accuracy_certificate",
        })

    report = {
        "L": L, "m": m, "g": g, "C": C, "nbar": nbar, "h": h, "p": p,
        "scaling": "acoustic" if acoustic else "dispersive",
        "coordinates": coordinates, "time_scaling": time_scaling,
        "source_scaling": source_scaling, "density_output_factor": output_factor,
        "norm_weights": weights, "a_h": a_h, "a_0": a_0,
        "a_second_order": a_second, "a_remainder": a_h - a_second,
        "a_remainder_bound": C * h**4 * abs(p)**6 / 360.0,
        "a_difference_bound": difference_bound,
        "generator": generator, "limit_generator": limit,
        "scaled_frequency_squared": frequency_squared,
        "limit_frequency_squared": limit_frequency_squared,
        "generator_error_operator_norm": (_operator_norm(generator - limit, weights)
                                          if acoustic else abs(a_h - a_0)),
        "generator_error_bound": generator_bound,
        "propagator_cases": propagator_cases, "response_cases": response_cases,
        "exactness_certificate": False,
    }
    if acoustic:
        second = (limit_frequency_squared
                  + h**2 * p**4 * (C**2 - C * g * nbar / 6.0))
        report.update({
            "frequency_second_order": second,
            "frequency_remainder": frequency_squared - second,
            "frequency_remainder_bound": h**4 * abs(p)**6 * (C * g * nbar / 180.0 + C**2 / 6.0),
            "h_uniform_max": h_max, "gamma": gamma,
            "propagator_uniform_bound": uniform_bound,
            "generator_bound_coefficient": coefficient,
        })
    return report


@_guard
def acoustic_report(L, m, g):
    """Frozen g>0 acoustic generator, propagators and normalized density response.

    X=(rho/h,theta), T=h*t, f=h*F. The fixed norm is
    g*|X[0]|**2 + 2*nbar*C*p**2*|X[1]|**2; r/F equals rho/f.
    """
    L = _integer(L, "L", _CLASSICAL_SIZES)
    m = _integer(m, "m", _MODES)
    g = _real(g, "g", _ACOUSTIC_COUPLINGS)
    return _classical_report(L, m, g, acoustic=True)


@_guard
def dispersive_report(L, m):
    """Separate reversible g=0 scaling, never a dissipative diffusion limit.

    Y=(rho/(2*nbar),theta), T=h**2*t, f=h**2*F, Euclidean norm.
    The response is h**2*chi_micro(h**2*zeta), with output factor 2*nbar.
    """
    L = _integer(L, "L", _CLASSICAL_SIZES)
    m = _integer(m, "m", _MODES)
    return _classical_report(L, m, 0.0, acoustic=False)


def _occupations(L, N):
    """Complete lexicographic tuples, with no local occupation cutoff."""
    if L == 1:
        yield (N,)
    else:
        for first in range(N + 1):
            for rest in _occupations(L - 1, N - first):
                yield (first,) + rest


def _quantum_model(L, g):
    """Own the full model; reject the capped dimension before enumeration."""
    N, C = L, 1.0
    dimension = _capped_binomial(L + N - 1, N, cap=MAX_QUANTUM_DIMENSION)
    if dimension > MAX_QUANTUM_DIMENSION:
        raise ValueError("complete quantum dimension exceeds cap " + str(MAX_QUANTUM_DIMENSION))
    basis = tuple(_occupations(L, N))
    index = {state: i for i, state in enumerate(basis)}
    pairs = np.array([sum(n * (n - 1) // 2 for n in state) for state in basis], dtype=float)
    H = np.diag(g * pairs)
    for column, state in enumerate(basis):
        for x in range(L):
            y = (x + 1) % L
            for source, destination in ((x, y), (y, x)):
                if state[source]:
                    moved = list(state)
                    moved[source] -= 1
                    moved[destination] += 1
                    H[index[tuple(moved)], column] -= C * np.sqrt(state[source] * (state[destination] + 1))
    # psi_n = sqrt(N! / prod n_x!) prod u_x**n_x, all u_x=1/sqrt(L)>0.
    orbital = 1.0 / np.sqrt(L)
    state = np.array([
        np.sqrt(math.factorial(N) / math.prod(math.factorial(n) for n in occupation)) * orbital**N
        for occupation in basis
    ])
    _finite(H, "complete Hamiltonian")
    _finite(state, "uniform condensate")
    return dimension, basis, pairs, H, state


def _screened_eigensystem(centered):
    """Heuristic checks only; neither screen certifies numerical error."""
    dimension = centered.shape[0]
    eigenvalues, vectors = np.linalg.eigh(centered)
    _finite(eigenvalues, "centered eigenvalues")
    _finite(vectors, "centered eigenvectors")
    tolerance = 256.0 * np.finfo(float).eps * max(dimension, 1)
    orthogonality = float(_finite(np.linalg.norm(vectors.conj().T @ vectors - np.eye(dimension), ord="fro"),
                                 "eigenvector orthogonality residual"))
    eigenpair = float(_finite(np.linalg.norm(centered @ vectors - vectors * eigenvalues[None, :], ord="fro"),
                             "eigenpair residual"))
    scale = max(1.0, float(_finite(np.linalg.norm(centered, ord="fro"), "centered Frobenius norm")))
    if orthogonality > tolerance or eigenpair > tolerance * scale:
        raise NumericalUnavailable("centered eigensystem failed heuristic residual screens")
    return eigenvalues, vectors, eigenpair, orthogonality, tolerance


@_guard
def uniform_condensate_bridge(L, g, t):
    """Complete N=L trial-state bridge in the analytically centered frame.

    The expected energy, not a fitted eigenvalue or numerical expectation,
    centers H. The bound uses the analytic expected variance. Free and t=0
    cases take the same numerical evolution path and retain their raw errors.
    """
    L = _integer(L, "L", _QUANTUM_SIZES)
    g = _real(g, "g", _QUANTUM_COUPLINGS)
    t = _real(t, "t", _QUANTUM_TIMES)
    dimension, basis, pairs, H, state = _quantum_model(L, g)
    N, C = L, 1.0
    mean_pairs = N * (N - 1) / (2.0 * L)
    interaction_energy = g * mean_pairs
    expected_energy = -2.0 * C * N + interaction_energy
    expected_variance = g**2 * (N * (N - 1) // 2) * (L - 1) / L**2
    orbital_frequency = -2.0 * C + g * (N - 1) / L
    energy = float(np.vdot(state, H @ state).real)
    centered = _finite(H - expected_energy * np.eye(dimension), "centered Hamiltonian")
    residual = _finite(centered @ state, "centered state residual")
    interaction_residual = g * (pairs - mean_pairs) * state
    variance = float(np.vdot(residual, residual).real)
    eigenvalues, vectors, eigenpair, orthogonality, tolerance = _screened_eigensystem(centered)
    evolved = _finite(vectors @ (np.exp(-1j * t * eigenvalues) * (vectors.conj().T @ state)),
                      "centered evolution")
    error = float(np.linalg.norm(evolved - state, ord=2))
    uncapped = abs(t) * np.sqrt(expected_variance)
    return {
        "L": L, "N": N, "C": C, "g": g, "t": t,
        "dimension": dimension, "basis": basis,
        "state_normalization_residual": abs(float(np.vdot(state, state).real) - 1.0),
        "energy": energy, "expected_energy": expected_energy,
        "energy_residual": abs(energy - expected_energy),
        "orbital_frequency": orbital_frequency,
        "hartree_lift_frequency": N * orbital_frequency,
        "phase_correction_rate": interaction_energy,
        "variance": variance, "expected_variance": expected_variance,
        "variance_residual": abs(variance - expected_variance),
        "interaction_residual_agreement": float(np.linalg.norm(residual - interaction_residual, ord=2)),
        "phase_corrected_error": error,
        "error_bound": min(2.0, uncapped), "uncapped_error_bound": uncapped,
        "bound_status": "informative" if uncapped < 2.0 else "trivial_cap",
        "free_exact_structure": g == 0.0,
        "eigenpair_frobenius_residual": eigenpair,
        "orthogonality_frobenius_residual": orthogonality,
        "eigensystem_tolerance": tolerance,
        "exactness_certificate": False,
    }


@_guard
def demonstration_report():
    """The entire preregistered bounded grid, with fresh detached reports."""
    return {
        "acoustic_cases": [acoustic_report(L, m, g) for L in _CLASSICAL_SIZES
                           for m in _MODES for g in _ACOUSTIC_COUPLINGS],
        "dispersive_cases": [dispersive_report(L, m) for L in _CLASSICAL_SIZES for m in _MODES],
        "quantum_cases": [uniform_condensate_bridge(L, g, t) for L in _QUANTUM_SIZES
                          for g in _QUANTUM_COUPLINGS for t in _QUANTUM_TIMES],
        "limitations": _LIMITATIONS,
        "numerical_domain": {
            "arithmetic": "float64/complex128; evaluated analytic envelopes are not certified numerical errors",
            "classical_L": _CLASSICAL_SIZES, "m": _MODES, "C": 1.0,
            "classical_nbar": 1.0, "acoustic_g": _ACOUSTIC_COUPLINGS,
            "classical_T": _CLASSICAL_TIMES, "zeta": _RESPONSE_FREQUENCIES,
            "quantum_L_equals_N": _QUANTUM_SIZES, "quantum_g": _QUANTUM_COUPLINGS,
            "quantum_t": _QUANTUM_TIMES, "max_quantum_dimension": MAX_QUANTUM_DIMENSION,
            "classical_operator_norm": "spectral2 after fixed norm-weight similarity",
            "quantum_vector_norm": "Euclidean2; state normalization residual is abs(<psi,psi>-1)",
            "scalar_residuals": "absolute energy/variance differences; signed Taylor remainders",
            "eigensystem_screens": "heuristic Frobenius residuals; dimensionless tolerance 256*eps*dimension",
            "raw_residuals_clipped": False, "exactness_certificate": False,
        },
    }
