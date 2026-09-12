"""First interaction derivative of the unchanged classical defocusing DNLS ring.

In the density-equivalent rotating frame, H0=C(2-T-T†), and
psi(t)=v(t)+g*psi1(t)+R(t). All 27 cubic channels are integrated before
residue aliases are summed. This is a state-dependent quartic density response,
not a static source matrix, selected vacuum, or fermion/mass derivation.

Arithmetic warnings are floating-point diagnostics, NOT rigorous roundoff
certificates. The analytic approximation bounds do not include roundoff error.
Only normal binary64 intermediates are supported: unresolved subnormal,
nonfinite, or excessively large phase calculations raise ValueError. No wider
longdouble format is assumed. This deliberately conservative domain can reject
mathematically finite inputs whose evaluated intermediates are unresolved.
"""
from functools import wraps
from itertools import product
from numbers import Integral, Real

import numpy as np

from bpr.substrate_triplet_projection import MODES, fourier_isometry

MODEL_ID = "defocusing-ring-nonlinear-response-v1"
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny


def _guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, MemoryError) as exc:
            raise ValueError("numerically unresolved intermediate (overflow/underflow/phase/allocation)") from exc
    return checked


def _checked(value, name):
    value = np.asarray(value)
    parts = (value.real, value.imag) if np.iscomplexobj(value) else (value,)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"numerically unresolved {name}: nonfinite")
    if any(np.any((abs(part) > 0) & (abs(part) < _TINY)) for part in parts):
        raise ValueError(f"numerically unresolved {name}: subnormal/underflow")
    return value.item() if value.ndim == 0 else value


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    original = value
    value = _checked(float(value), name)
    if original != 0 and value == 0:
        raise ValueError(f"numerically unresolved {name}: underflow")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _ring(p, C):
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
        raise TypeError("p must be an integer")
    if p < 5:
        raise ValueError("p must be at least 5 (primality is not required)")
    if p > 2**53:
        raise ValueError("numerically unresolved integer ring size")
    return int(p), _real(C, "C", positive=True)


def _amplitudes(amplitudes):
    original = np.asarray(amplitudes)
    z = _checked(np.asarray(amplitudes, dtype=complex), "amplitudes")
    if z.shape != (3,):
        raise ValueError("amplitudes must have shape (3,) in k=(-1,0,1) order")
    if np.any((original != 0) & (z == 0)):
        raise ValueError("numerically unresolved amplitudes: underflow")
    n = _checked(np.sum(z.real*z.real + z.imag*z.imag), "norm")
    if np.any(z != 0) and n == 0:
        raise ValueError("numerically unresolved positive norm")
    return z, n


def _mul(a, b, name):
    value = _checked(np.multiply(a, b), name)
    if a != 0 and b != 0 and value == 0:
        raise ValueError(f"numerically unresolved {name}: underflow")
    return value


def _phase(value):
    value = _checked(value, "phase")
    if abs(value) > 1/(64*_EPS):
        raise ValueError("numerically unresolved oscillatory phase")
    return np.exp(-1j*value)


@_guard
def duhamel_kernel(omega, t):
    """Integral from 0 to t of exp(-i*omega*s), including negative t.

    Resonance is exact, never selected by a numerical near-zero tolerance.
    sinc is continuous near zero; no inverse-detuning pole is evaluated.
    """
    omega, t = _real(omega, "omega"), _real(t, "t")
    if t == 0:
        return 0j
    if omega == 0:
        return complex(t)
    half = _mul(_mul(omega, t, "detuning*time"), 0.5, "half phase")
    phase = _phase(half)
    sinc = _checked(np.sinc(half/np.pi), "sinc")
    return complex(_mul(_mul(t, sinc, "kernel magnitude"), phase, "kernel"))


@_guard
def cubic_channels(p, C=1.0):
    """27 state-independent channel records; stable identity-based detunings.

    W=epsilon1. Inband detunings are 0 or +/-2W. Offband detunings
    are 2W-epsilon2=-2W*cos(2theta), -epsilon2, and
    W-epsilon3=-4C*sin(4theta)*sin(2theta), theta=pi/p.
    These are strictly nonzero for p>=5. No rounded subtraction labels a
    resonance. p=5 aliased raw outputs retain their DIFFERENT detunings.
    """
    p, C = _ring(p, C)
    theta = np.pi/p
    energies = {0: 0.0}
    for k in range(1, 4):
        representative = min(k % p, (-k) % p)
        energies[k] = _mul(C, 4*np.sin(representative*theta)**2, "dispersion")
        if energies[k] <= 0:
            raise ValueError("numerically unresolved positive dispersion")
    w = energies[1]
    result = []
    for i, j, ell in product(MODES, repeat=3):
        k = i-j+ell
        count = i*i-j*j+ell*ell
        retained = k in MODES
        if retained:
            multiplier = count-k*k
            detuning = _mul(multiplier, w, "inband detuning")
            identity = f"{multiplier}*W"
            resonant = multiplier == 0
        elif abs(k) == 2:
            detuning = -energies[2] if count == 0 else _mul(-2*w, np.cos(2*theta), "2W-epsilon2")
            identity = "-epsilon2" if count == 0 else "2W-epsilon2"
            resonant = False
        else:
            detuning = _mul(C, -4*np.sin(4*theta)*np.sin(2*theta), "W-epsilon3")
            identity, resonant = "W-epsilon3", False
        if not resonant and detuning == 0:
            raise ValueError("numerically unresolved nonzero detuning")
        result.append({"input_modes": [i, j, ell], "raw_output": k,
                       "residue": k % p, "retained": retained,
                       "output_energy": energies[abs(k)], "detuning": detuning,
                       "detuning_identity": identity, "exact_resonance": resonant})
    return result


@_guard
def first_order_response(amplitudes, p, C=1.0, t=0.1):
    """Return v, dpsi/dg at g=0, and its retained/discarded components.

    Fourier arrays use residue order 0,...,p-1 and normalized Fourier modes.
    psi1 has NO factor g. The retained derivative is also exactly the first
    derivative of the lifted three-mode Galerkin trajectory at g=0.
    """
    p, C = _ring(p, C)
    t = _real(t, "t")
    z, n = _amplitudes(amplitudes)
    channels = cubic_channels(p, C)
    free = np.zeros(p, complex)
    retained = np.zeros(p, complex)
    discarded = np.zeros(p, complex)
    term_scales = np.zeros(p)
    maximum_phase = 0.0
    w = channels[0]["output_energy"]  # (-1,-1,-1) outputs -1
    for a, k in zip(z, MODES):
        phase = _phase(_mul(w if k else 0.0, t, "free phase"))
        free[k % p] = _mul(a, phase, "free coefficient")
    for channel in channels:
        i, j, ell = channel["input_modes"]
        monomial = _mul(_mul(z[i+1], z[j+1].conjugate(), "quadratic monomial"), z[ell+1], "cubic monomial")
        integral = duhamel_kernel(channel["detuning"], t)
        contribution = _mul(monomial, integral, "integrated monomial")
        contribution = _mul(contribution, -1j/p, "normalized monomial")
        output_phase = _mul(channel["output_energy"], t, "output phase")
        detuning_phase = _mul(channel["detuning"], t, "detuning phase")
        maximum_phase = max(maximum_phase, abs(output_phase), abs(detuning_phase))
        phase = _phase(output_phase)
        contribution = _mul(contribution, phase, "propagated monomial")
        target = retained if channel["retained"] else discarded
        target[channel["residue"]] += contribution
        # Do not suppress the error scale at a sinc zero: differentiating the
        # integral with respect to frequency brings in a factor |t|. Use the
        # unoscillated magnitude even when the displayed integral cancels.
        term_scales[channel["residue"]] += _mul(_mul(abs(monomial), abs(t), "unoscillated magnitude"), 1/p, "unoscillated normalized magnitude")
    # Unit-normalized inverse FFT avoids a dense full-ring Fourier matrix.
    def lift(coefficients):
        return _checked(np.fft.ifft(coefficients)*np.sqrt(p), "site field")
    retained, discarded = _checked(retained, "retained coefficients"), _checked(discarded, "discarded coefficients")
    total = _checked(retained+discarded, "first derivative coefficients")
    # Frequency/phase rounding grows like eps*|epsilon*t|. A magnitude-only
    # summation warning incorrectly resolves tiny differences of large secular
    # terms at long times. This phase-sensitive scale remains a diagnostic,
    # NOT interval arithmetic: returned raw arrays may be inaccurate at long t.
    warning = _mul(_mul(128*_EPS, 1+maximum_phase, "phase resolution factor"),
                   np.hypot.reduce(term_scales), "arithmetic warning")
    off_norm = _checked(np.hypot.reduce(abs(discarded)), "discarded norm")
    structural = t == 0 or np.count_nonzero(z) <= 1
    status = "structural_zero" if structural else ("resolved_nonzero" if off_norm > warning else "unresolved_cancellation")
    return {"p": p, "C": C, "time": t, "norm": n, "channels": channels,
            "free_fourier": free, "psi1_fourier": total,
            "retained_fourier": retained, "discarded_fourier": discarded,
            "free_field": lift(free), "psi1": lift(total),
            "retained_psi1": lift(retained), "discarded_psi1": lift(discarded),
            "arithmetic": {"warning_scale": warning, "maximum_phase": maximum_phase,
                           "phase_sensitive": True, "discarded_status": status,
                           "discarded_raw_norm": off_norm,
                           "discarded_resolved_norm": None if status == "unresolved_cancellation" else off_norm,
                           "rigorous_roundoff_certificate": False}}


@_guard
def finite_time_bounds(N, g, t, source_max=1.0):
    """Norm-conserving absolute finite-time certificates, full AND Galerkin.

    ||psi-v||<=g*N^(3/2)*T; ||R||<=(3/2)*g^2*N^(5/2)*T^2.
    For F=diag(f), observable remainder <=4*||f||inf*g^2*N^3*T^2,
    from 2 Re(v†FR)+delta†Fdelta. The full-minus-Galerkin difference
    uses TWO trajectory remainders. No O(g^3) term is silently discarded.
    These analytic bounds exclude numerical integration and roundoff errors.
    """
    N = _real(N, "N", nonnegative=True)
    g, t = _real(g, "g", nonnegative=True), _real(t, "t")
    source_max = _real(source_max, "source_max", nonnegative=True)
    exact = N == 0 or g == 0 or t == 0
    if exact:
        delta = remainder = observable = relative = 0.0
    else:
        scale = _mul(_mul(g, N, "gN"), abs(t), "gNT")
        relative = _mul(_mul(1.5, scale, "relative remainder scale"), scale, "relative state remainder")
        delta = _mul(scale, np.sqrt(N), "free-state bound")
        remainder = _mul(_mul(1.5, scale, "remainder scale"), delta, "state remainder bound")
        observable = _mul(_mul(4*source_max, delta, "observable scale"), delta, "observable bound") if source_max else 0.0
    return {"free_state_error_bound": delta, "state_remainder_bound": remainder,
            "relative_state_remainder_bound": None if N == 0 else relative,
            "observable_remainder_bound": observable,
            "full_vs_galerkin_observable_remainder_bound": _mul(2.0, observable, "two-trajectory bound"),
            "status": "vacuum_exact" if N == 0 else ("exact_zero" if exact else "conservative_finite_time"),
            "includes_arithmetic_error": False, "includes_solver_error": False}


def _resolution(value, warning, structural=False):
    return {"raw_value": value, "arithmetic_warning_scale": warning,
            "status": "structural_zero" if structural else ("resolved_nonzero" if abs(value) > warning else "unresolved_cancellation"),
            "resolved_value": 0.0 if structural else (value if abs(value) > warning else None)}


@_guard
def observable_response(amplitudes, source, p, C=1.0, g=0.5, t=0.1):
    """JSON-safe density response split, WITHOUT renormalization or clipping.

    'first_order_observable' is v†Fv+g*L, not the full square of v+g*psi1.
    The latter contains an extra g² psi1†Fpsi1 and is reported separately.
    L_P is the Galerkin derivative; L_Q is its omitted local observable term.
    A signal exceeding an approximation bound is NOT a roundoff certificate.
    """
    p, C = _ring(p, C)
    g, t = _real(g, "g", nonnegative=True), _real(t, "t")
    original_source = np.asarray(source)
    if np.iscomplexobj(original_source):
        raise TypeError("source must be real")
    source = _checked(np.asarray(source, dtype=float), "source")
    if source.shape != (p,):
        raise ValueError("source must have shape (p,)")
    if np.any((original_source != 0) & (source == 0)):
        raise ValueError("numerically unresolved source: underflow")
    response = first_order_response(amplitudes, p, C, t)
    n, v = response["norm"], response["free_field"]
    maximum = float(np.max(abs(source)))
    free = _checked(np.vdot(v, source*v).real, "free observable")
    lp = _checked(2*np.vdot(v, source*response["retained_psi1"]).real, "retained derivative")
    lq = _checked(2*np.vdot(v, source*response["discarded_psi1"]).real, "discarded derivative")
    structural = n == 0 or t == 0 or np.count_nonzero(amplitudes) <= 1 or bool(np.all(source == source[0]))
    if structural:
        # Constant total norm and single plane-wave densities are structural
        # controls, not cancellation tests on the evaluated Fourier sums.
        lp = lq = 0.0
    full = _checked(lp+lq, "full derivative")
    retained_term, omitted_term = _mul(g, lp, "retained correction"), _mul(g, lq, "omitted correction")
    correction = _checked(retained_term+omitted_term, "full correction")
    bounds = finite_time_bounds(n, g, t, maximum)
    # A conservative magnitude-based arithmetic WARNING, not interval arithmetic.
    warning = _mul(256*_EPS, _mul(2*maximum*np.sqrt(n),
                   np.hypot.reduce(abs(response["psi1_fourier"])), "observable magnitude"), "observable resolution")
    warning = _checked(warning + 2*maximum*np.sqrt(n)*response["arithmetic"]["warning_scale"], "observable warning")
    def square(field):
        return _checked(np.vdot(field, source*field).real, "squared approximate field observable")
    approx = _checked(v+g*response["psi1"], "approximate field")
    galerkin_approx = _checked(v+g*response["retained_psi1"], "Galerkin approximate field")
    return {"free_observable": free,
            "retained_derivative": lp, "discarded_derivative": lq, "full_derivative": full,
            "retained_correction": retained_term, "discarded_correction": omitted_term,
            "full_correction": correction,
            "first_order_observable": _checked(free+correction, "first-order observable"),
            "galerkin_first_order_observable": _checked(free+retained_term, "Galerkin observable"),
            "squared_approximate_field_observable": square(approx),
            "squared_galerkin_approximate_field_observable": square(galerkin_approx),
            "bounds": bounds,
            "arithmetic": {"retained": _resolution(lp, warning, structural),
                           "discarded": _resolution(lq, warning, structural),
                           "full": _resolution(full, warning, structural),
                           "rigorous_roundoff_certificate": False},
            "full_correction_exceeds_analytic_bound": bool(abs(correction) > bounds["observable_remainder_bound"]),
            "omitted_correction_exceeds_two_trajectory_bound": bool(abs(omitted_term) > bounds["full_vs_galerkin_observable_remainder_bound"]),
            "renormalized": False, "densities_clipped": False}


@_guard
def demonstration():
    """Frozen finite-time analytic-response witness, not a fitted prediction."""
    p, C, g, N, t = 7, 1.0, 0.5, 1.0, 0.1
    s = np.sqrt(N/2)
    z = np.array([0.0, s, s], complex)
    source = np.sin(4*np.pi*np.arange(p)/p)
    response = observable_response(z, source, p, C, g, t)
    w = 4*C*np.sin(np.pi/p)**2
    e2 = 4*C*np.sin(2*np.pi/p)**2
    d = -2*w*np.cos(2*np.pi/p)
    analytic_p = -N*N/(4*p)*np.sin(2*w*t)/(2*w)
    analytic_q = N*N/(4*p)*t*np.sinc(d*t/(2*np.pi))*np.cos((e2+2*w)*t/2)
    generic = np.array([1, 1j, (1+1j)/np.sqrt(2)])/np.sqrt(3)
    return {"model_id": MODEL_ID, "status": "conditional_classical_first_interaction_response",
            "parameters": {"p": p, "C": C, "g": g, "N": N, "t": t},
            "witness": {"amplitudes_re_im": [[float(a.real), float(a.imag)] for a in z],
                        "source": source.tolist(), "source_formula": "sin(4*pi*x/p)",
                        "response": response,
                        "analytic_coefficients": {"retained_derivative": analytic_p,
                                                  "discarded_derivative": analytic_q,
                                                  "full_derivative": analytic_p+analytic_q,
                                                  "small_time_cubic_coefficient": -N*N*e2*(e2+2*w)/(24*p)}},
            "controls": {"total_norm": observable_response(z, np.ones(p), p, C, g, t),
                         "pure_mode": observable_response([0, 1, 0], source, p, C, g, t),
                         "vacuum": observable_response([0, 0, 0], source, p, C, g, t)},
            "generic_complex": {"amplitudes_re_im": [[float(a.real), float(a.imag)] for a in generic],
                                "response": observable_response(generic, source, p, C, g, t)},
            "physical_masses": None, "physical_mixing": None,
            "limitations": ["Initial state, norm and local observable are specified inputs, not dynamically selected.",
                            "Classical finite-time quartic response is not a universal state-independent 3-by-3 source matrix.",
                            "No new interaction, quantum statistics, physical masses, mixing or fermions are derived.",
                            "Nonzero derivative at g=0 does not certify the finite-g sign when the bound exceeds the signal.",
                            "Full-minus-Galerkin comparison uses the sum of two trajectory remainder bounds.",
                            "Arithmetic warning scales are separate from analytic approximation bounds and are not rigorous roundoff certificates.",
                            "No renormalization, density clipping, source selection or complete microscopic matching is performed."]}
