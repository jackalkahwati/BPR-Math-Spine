"""Large-ring density measure of the stipulated one-pair compression.

Exact formulas describe this compression and its leading virtual source only.
Their binary64 evaluations and residuals are not roundoff certificates or a
full-Hamiltonian spectral theorem. The energy origin is the unperturbed
product energy, not the exact interacting ground-state energy.
"""
from fractions import Fraction
import math
from numbers import Integral

import numpy as np

from bpr.substrate_fermionization import (
    _EPS, _guard, _integer, _mul, _normal, _real, _scale,
)
from bpr.substrate_vacuum_selection import rational_record

MODEL_ID = "conditional-substrate-neutral-continuum-v1"
MAX_SITES = 4096
MAX_MOMENT_DEGREE = 32
FROZEN_REFERENCE = (5, 1, 1.0, 40.0)
FROZEN_LENGTHS = (16, 64, 256)
LIMITATIONS = (
    "The quantum Bose Hamiltonian and N=L=q neutrality are stipulated.",
    "The absolutely continuous weak limit is exact only for the leading one-pair compression measure.",
    "Compression energies are measured from the unperturbed product energy zero, not certified frequencies above the interacting ground state.",
    "Finite numerical evaluations, phase screens and residuals have no certified roundoff bound.",
    "Population selection g>=4C does not imply the sufficient excitation separation g>4CL.",
    "No Schur norm is computed and no uniform full-Hamiltonian thermodynamic spectral theorem is available.",
    "A missing full-model certificate does not rule out a bound particle.",
    "Fixed lattice spacing and increasing ring size do not define a spatial continuum limit.",
    "Growing g translates energy to infinity and is not a tight probability limit on the original energy axis.",
    "A relative-motion continuum or quadratic threshold expansion neither establishes nor disproves massive relativistic behavior.",
    "No physical lattice spacing, limiting speed, masses, mixing or empirical calibration is supplied.",
)


def _parameters(L, m, C, g, capped=False):
    # Validate the allocation cap before any arrays, including zero controls.
    L = _integer(L, "L", 3, MAX_SITES if capped else None)
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    m = int(m) % L
    return L, m, _real(C, "C", positive=True), _real(g, "g", lower=0)


def _fraction_float(value, name):
    try:
        result = _normal(float(value), name)
    except OverflowError as exc:
        raise ValueError("numerically unresolved " + name + ": overflow") from exc
    if value and result == 0:
        raise ValueError("numerically unresolved " + name + ": underflow")
    return result


def _lattice_angle(L, m):
    """Reduce integers first; never replace an unresolved nonzero mode by zero."""
    signed = m if 2 * m <= L else m - L
    k = _fraction_float(Fraction(signed, L) * Fraction.from_float(2 * math.pi),
                        "lattice angle")
    if abs(k) == math.pi and 2 * m != L:
        raise ValueError("numerically unresolved non-pi lattice angle rounded to pi")
    return k


def _angle(k):
    k = _real(k, "k")
    if not -math.pi <= k <= math.pi:
        raise ValueError("k must lie in the bounded interval [-pi, pi]")
    return k


def _geometry(k):
    if abs(k) == math.pi:
        return 1.0, 0.0
    sine = _normal(math.sin(k), "momentum sine")
    real = 2 + math.cos(k)
    theta = _normal(math.atan2(-sine, real), "hopping phase")
    if k != 0 and (sine == 0 or theta == 0):
        raise ValueError("numerically unresolved nonzero hopping phase")
    return math.hypot(real, sine), theta


def _product(numerators, denominators=(), name="product"):
    """Scaled scalar product, allowing a normal final answer without overflow.

    Factors here are finite normal floats or structural zeros. No subnormal
    final answer or nonzero underflow is returned as a physical zero.
    """
    if any(x == 0 for x in numerators):
        return 0.0
    mantissa, exponent = 1.0, 0
    for values, divide in ((numerators, False), (denominators, True)):
        for value in values:
            part, power = math.frexp(value)
            mantissa = mantissa / part if divide else mantissa * part
            exponent += -power if divide else power
            mantissa, power = math.frexp(mantissa)
            exponent += power
    try:
        result = _normal(math.ldexp(mantissa, exponent), name)
    except OverflowError as exc:
        raise ValueError("numerically unresolved " + name + ": overflow") from exc
    if result == 0:
        raise ValueError("numerically unresolved " + name + ": underflow")
    return result


def _optional(function):
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            return function(), None
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return None, "Numerically unavailable: " + str(exc)


@_guard
def endpoint_source(L, m, C=1.0, g=40.0):
    """Two complex endpoint coefficients in the existing |k,r> convention.

    This scalar-size helper is not capped in L. At g=0 the virtual source is
    undefined even at m=0. For g>0 the zero mode has two structural zeros.
    Half-angle components avoid subtracting exp(ik) from one.
    """
    L, m, C, g = _parameters(L, m, C, g)
    if g == 0:
        raise ValueError("virtual source undefined at g=0, including the zero mode")
    if m == 0:
        return np.zeros(2, dtype=complex)
    k = _lattice_angle(L, m)
    half_sin = _normal(math.sin(k / 2), "source half angle")
    half_cos = 0.0 if 2 * m == L else _normal(math.cos(k / 2), "source half angle")
    real = _product((-2 * math.sqrt(2), C, half_sin, half_sin), (g,), "source real component")
    imag = _product((-2 * math.sqrt(2), C, half_sin, half_cos), (g,), "source imaginary component")
    return np.array((complex(real, imag), complex(real, -imag)))


@_guard
def limiting_shape(k):
    """Normalized semicircle in s=(E-g)/C, independent of C and g.

    This standalone shape exists at k=0; the exactly zero density-source mode
    cannot itself be normalized. Continuous k must be in [-pi,pi].
    """
    k = _angle(k)
    a, _ = _geometry(k)
    return {"k": k, "a": a, "coordinate": "s=(E-g)/C",
            "support": [-2 * a, 2 * a], "mass": 1.0, "mean": 0.0,
            "second_moment": a * a, "absolute_first_moment": 8 * a / (3 * math.pi),
            "density_formula": "sqrt(4*a^2-s^2)/(2*pi*a^2)",
            "dimensional_density": "density_s((E-g)/C)/C",
            "scope": "normalized leading compression weak limit only",
            "full_hamiltonian_certificate_available": False}


@_guard
def semicircle_density(s, k):
    """Scalar density in dimensionless s; support checked before squaring."""
    s, k = _real(s, "s"), _angle(k)
    a, _ = _geometry(k)
    width = 2 * a
    if abs(s) >= width:
        return 0.0
    # Endpoint differences retain precision instead of subtracting squares.
    return math.sqrt((width - abs(s)) / width) * math.sqrt((width + abs(s)) / width) / (math.pi * a)


@_guard
def semicircle_moment(degree, k):
    """Centered dimensionless moment: odd zero, even Cat_n*a^(2n)."""
    degree = _integer(degree, "degree", 0, MAX_MOMENT_DEGREE)
    a, _ = _geometry(_angle(k))
    if degree % 2:
        return 0.0
    n = degree // 2
    return float(math.comb(2 * n, n) // (n + 1)) * a ** degree


@_guard
def weak_limit_bound(L, m, k_target, lipschitz=1.0):
    """K[32*a(k_L)/(3L)+2*d_circle(k_L,k)] in s, uncapped in L.

    The exact compressed-model inequality excludes binary64 roundoff.
    For F(q)=sin^2(q)[f(-2a cos(q))-f(0)], Var(F)<=16aK/3.
    Trapezoidal error and endpoint interference each cost Var(F)/L;
    coupling the two semicircles gives the circular-momentum term.
    """
    L, m, _, _ = _parameters(L, m, 1.0, 1.0)
    k_target = _angle(k_target)
    lipschitz = _real(lipschitz, "lipschitz", lower=0)
    if m == 0:
        raise ValueError("normalized-source weak limit requires a nonzero aliased mode")
    k = _lattice_angle(L, m)
    a, _ = _geometry(k)
    if lipschitz == 0:
        return 0.0
    term = _fraction_float(Fraction.from_float(32 * a / 3) / L,
                           "finite-size weak-limit term")
    difference = abs(Fraction.from_float(k) - Fraction.from_float(k_target))
    period = Fraction.from_float(2 * math.pi)
    distance = _fraction_float(min(difference, period - difference),
                               "circular momentum distance")
    return _mul(lipschitz, term + 2 * distance, "weak-limit bound")


@_guard
def validity_audit(L, C=1.0, g=40.0):
    """Uncapped, exact threshold comparisons of accepted binary64 inputs.

    No large Hilbert space or B1 is constructed. Satisfying separation is a
    sufficient finite-size condition, not a computed Schur correction norm.
    """
    L, _, C, g = _parameters(L, 0, C, g)
    cq, gq = Fraction.from_float(C), Fraction.from_float(g)
    population, separation = gq - 4 * cq, gq - 4 * L * cq
    return {"population_selection_condition": "g >= 4C (q=L)",
            "population_selection_sufficient": population >= 0,
            "excitation_separation_condition": "g > 4CL",
            "excitation_separation_sufficient": separation > 0,
            "population_margin": rational_record(population),
            "excitation_margin": rational_record(separation),
            "population_threshold": rational_record(4 * cq),
            "excitation_threshold": rational_record(4 * L * cq),
            "threshold_evaluation": "exact Fractions of accepted binary64 inputs",
            "schur_norm_computed": False,
            "full_hamiltonian_thermodynamic_certificate_available": False,
            "roundoff_included": False,
            "interpretation": "Failure of a sufficient condition is not a disproof of separation or bound particles."}


def _dimensional_energies(offsets, g):
    result = _normal(offsets + g, "dimensional energies")
    # An explicitly heuristic screen: keep s regardless of loss when adding g.
    error = float(np.max(np.abs((result - g) - offsets)))
    tolerance = 64 * _EPS * float(np.max(np.abs(offsets)))
    if error > tolerance or np.any((offsets != 0) & (result == g)):
        raise ValueError("translation by g loses bandwidth precision (heuristic screen)")
    return result


@_guard
def finite_compression_measure(L, m, C=1.0, g=40.0):
    """O(L) sine-chain nodes and finite endpoint-interference probabilities.

    Cap L<=4096 is checked before allocations. No Fock solver or eigensolver.
    Array index j-1 denotes j=1..L-1. structural_dark_indices are ZERO-BASED
    indices, assigned only by the exact k=pi parity rule (even j are dark).
    Probabilities are squared amplitudes, never clipped or renormalized.
    At g=0 no virtual-source probabilities or weights are exposed. At g>0,
    m=0 has exactly zero absolute measure and no normalized probabilities.
    Dimensional outputs may be unavailable while dimensionless nodes survive.
    """
    L, m, C, g = _parameters(L, m, C, g, capped=True)
    k = _lattice_angle(L, m)
    a, theta = _geometry(k)
    phi = k - (L - 2) * theta
    beta = -math.cos(phi)
    j = np.arange(1, L, dtype=int)
    q = math.pi * np.minimum(j, L - j) / L
    cosine = np.cos(q) * np.where(2 * j < L, 1.0, -1.0)
    cosine[2 * j == L] = 0.0
    nodes = -2 * a * cosine
    dark = j[j % 2 == 0] - 1 if 2 * m == L else np.empty(0, dtype=int)
    result = {"L": L, "m": m, "k": k, "C": C, "g": g,
              "a": a, "theta": theta, "phi": phi, "beta": beta,
              "coordinate": "s=(E-g)/C", "dimensionless_offsets": nodes,
              "probabilities": None, "normalization_available": False,
              "normalization_reason": None, "total_weight": None,
              "total_weight_reason": None, "absolute_weights": None,
              "absolute_weight_reason": None, "energy_offsets": None,
              "energy_offset_reason": None, "energies": None, "energy_reason": None,
              "structural_dark_indices": dark, "numerical_diagnostics": None}
    result["energy_offsets"], result["energy_offset_reason"] = _optional(
        lambda: _scale(nodes, C, "dimensional energy offsets"))
    if result["energy_offsets"] is None:
        result["energy_reason"] = result["energy_offset_reason"]
    else:
        result["energies"], result["energy_reason"] = _optional(
            lambda: _dimensional_energies(result["energy_offsets"], g))
    if g == 0:
        reason = "Virtual-source measure undefined at g=0, including m=0."
        result.update(normalization_reason=reason, total_weight_reason=reason,
                      absolute_weight_reason=reason)
        return result
    if m == 0:
        result.update(normalization_reason="Zero density mode: zero measure cannot be normalized.",
                      total_weight=0.0, absolute_weights=np.zeros(L - 1))
        return result
    odd_factor = math.sin(phi / 2) ** 2
    even_factor = 0.0 if 2 * m == L else math.cos(phi / 2) ** 2
    if 2 * m == L:
        odd_factor = 1.0
    p = _normal((4.0 / L) * np.sin(q) ** 2 * np.where(j % 2, odd_factor, even_factor),
                "normalized line probabilities")
    result.update(probabilities=p, normalization_available=True)
    half_sin = math.sin(k / 2)
    weight, reason = _optional(lambda: _product(
        (16.0, C, C, half_sin, half_sin), (g, g), "total source weight"))
    result.update(total_weight=weight, total_weight_reason=reason)
    if weight is None:
        result["absolute_weight_reason"] = reason
    else:
        result["absolute_weights"], result["absolute_weight_reason"] = _optional(
            lambda: _scale(p, weight, "absolute line weights"))
    mass = float(np.sum(p))
    moments = {str(d): float(np.dot(p, nodes ** d)) for d in range(9)}
    residuals = {str(d): moments[str(d)] - semicircle_moment(d, k)
                 for d in range(min(9, L - 2))}
    diagnostics = {"mass": mass, "mass_residual": mass - 1,
                   "max_line_fraction": float(np.max(p)), "line_fraction_bound": 4.0 / L,
                   "line_fraction_bound_excess": float(np.max(p)) - 4.0 / L,
                   "centered_moments": moments, "moment_residuals": residuals,
                   "phase_roundoff_scale_heuristic": _EPS * (abs(k) + (L - 2) * abs(theta)),
                   "roundoff_certified": False}
    if L - 2 <= 8:
        d = L - 2
        discrepancy = moments[str(d)] - semicircle_moment(d, k)
        expected = beta * (-a) ** d
        diagnostics["first_mismatch"] = {"degree": d, "measured_discrepancy": discrepancy,
                                         "predicted_discrepancy": expected,
                                         "residual": discrepancy - expected}
    result["numerical_diagnostics"] = diagnostics
    return result


def _range_or_none(values):
    return None if values is None else [float(np.min(values)), float(np.max(values))]


def case_report(L, m, C=1.0, g=40.0, k_target=None):
    """JSON-safe summaries; resource/input errors raise, not empty spectra."""
    measure = finite_compression_measure(L, m, C, g)
    L, m, C, g, k = (measure[key] for key in ("L", "m", "C", "g", "k"))
    if k_target is not None:
        k_target = _angle(k_target)
    summary = {key: measure[key] for key in (
        "coordinate", "normalization_available", "normalization_reason", "total_weight",
        "total_weight_reason", "absolute_weight_reason", "energy_offset_reason", "energy_reason")}
    summary.update(line_count=L - 1,
                   dimensionless_support=_range_or_none(measure["dimensionless_offsets"]),
                   energy_offset_range=_range_or_none(measure["energy_offsets"]),
                   energy_range=_range_or_none(measure["energies"]),
                   absolute_weights_available=measure["absolute_weights"] is not None,
                   structural_dark_indices=measure["structural_dark_indices"].tolist())
    return {"parameters": {"L": L, "m": m, "C": C, "g": g, "k": k},
            "compression_measure": summary,
            "numerical_checks": measure["numerical_diagnostics"],
            "validity_audit": validity_audit(L, C, g),
            "limiting_shape": limiting_shape(k if k_target is None else k_target),
            "weak_limit_bound": (weak_limit_bound(L, m, k if k_target is None else k_target)
                                 if measure["normalization_available"] else None),
            "theorem_scope": "exact compressed-model formulas; numerical evaluation without certified roundoff"}


def demonstration_report():
    """Frozen mathematical illustrations, not physical parameter fitting."""
    cases = []
    sequences = [(FROZEN_REFERENCE, "small reference", None)]
    sequences += [((L, L // 4, 1.0, 40.0), "fixed k=pi/2, fixed C,g", math.pi / 2)
                  for L in FROZEN_LENGTHS]
    sequences += [((L, 1, 1.0, 40.0), "fixed integer m=1, fixed C,g, k tends to zero", 0.0)
                  for L in FROZEN_LENGTHS]
    sequences += [((5, 0, 1.0, 40.0), "zero density mode control", None),
                  ((5, 1, 1.0, 0.0), "undefined g=0 virtual source control", None),
                  ((5, 0, 1.0, 0.0), "undefined g=0 zero-mode virtual source control", None)]
    for (L, m, C, g), sequence, target in sequences:
        case = case_report(L, m, C, g, target)
        case["limit_sequence"] = sequence
        cases.append(case)
    return {"model_id": MODEL_ID, "cases": cases, "limitations": list(LIMITATIONS),
            "frozen_threshold_audits": [
                {"parameters": {"L": L, "C": 1.0, "g": 40.0},
                 "audit": validity_audit(L, 1.0, 40.0)} for L in (9, 10)],
            "limit_statements": {
                "fixed_nonzero_k": "Normalized measures converge weakly to an absolutely continuous semicircle; individual line fractions are <=4/L.",
                "fixed_integer_m": "W ~ 16*pi^2*m^2*(C/g)^2/L^2; absolute measure vanishes while normalized shape tends to the k=0 semicircle.",
                "growing_g": "Use s=(E-g)/C; translations to infinity are not tight on the original energy axis.",
                "threshold": "g-6C+(2C/3)k^2+O(C*k^4), a threshold not a particle dispersion."},
            "physical_predictions": {"masses": None, "mixing": None,
                                     "limiting_light_speed": None, "spatial_continuum_scaling": None,
                                     "full_hamiltonian_thermodynamic_spectrum": None}}
