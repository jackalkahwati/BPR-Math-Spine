"""Volume-uniform bounds for the unshifted complete Bose ring.

Exact rational inequalities are separate from bounded, heuristic eigenvalue
checks. These bounds neither select a population nor establish a thermodynamic
limit. The frozen contract is substrate_extensive_stability_2026-09-13.md.
"""
from fractions import Fraction
from math import isfinite
import sys

import numpy as np

from bpr.substrate_gauge_encoding import (
    NumericalUnavailable, _frobenius, _screened_eigensystem,
)
from bpr.substrate_vacuum_selection import (
    MAX_SITES, _bounds, _metadata_integer, all_number_model, capped_binomial,
    rational_record,
)

MODEL_ID = "substrate_extensive_stability"
MAX_DIMENSION = 512
_NUMERICAL = (NumericalUnavailable, FloatingPointError, OverflowError,
              np.linalg.LinAlgError)
_SCOPE = (
    "Unshifted complete Bose ring; exact extensive bounds do not select a "
    "population, prove a thermodynamic limit, or supply empirical validation. "
    "The completed-square bound applies only for g>0, not g=0."
)
_COMPARISON_SCOPE = (
    "Exact binary64-lifted margins with a frozen heuristic allowance; neither "
    "the floating eigenvalue, matrix assembly nor the proxy is certified."
)


def _integer(value, name, lower, upper=None):
    if type(value) is not int:
        raise TypeError(name + " must be a base builtin int")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(name + " outside supported range")
    return value


def _rational(value, name):
    if type(value) not in (int, Fraction):
        raise TypeError(name + " must be a base builtin int or Fraction")
    return Fraction(value)


def _serialize(value):
    """Lossless exact records, with no array construction even for displays."""
    if type(value) is Fraction:
        return rational_record(value)
    if type(value) is int:
        return _metadata_integer(value)
    if type(value) is dict:
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _exact_bounds(L, N, C, g):
    """Owned exact fixture; no resource cap, float decisions or basis path."""
    L = _integer(L, "L", 3)
    N = _integer(N, "N", 0)
    C, g = _rational(C, "C"), _rational(g, "g")
    if C <= 0 or g < 0:
        raise ValueError("C must be positive and g nonnegative")
    inherited = _bounds(L, N, C, g)
    remainder = N % L
    alpha = g / 2 + 2 * C
    all_population_lower = -L * alpha**2 / (2 * g) if g else None
    envelope = None
    if N == L:
        envelope = {"lower": -2 * C,
                    "upper": min(Fraction(0), -2 * C + g * Fraction(L - 1, L) / 2)}
    structural_energy = None
    if N == 0:
        structural_energy = Fraction(0)
    elif g == 0 or N == 1:
        structural_energy = -2 * C * N
    return {
        "L": L, "N": N, "C": C, "g": g, "density": Fraction(N, L),
        "interaction_minimum": inherited["interaction_minimum"],
        "remainder": remainder,
        "quadratic_lower": g * N**2 / (2 * L) - alpha * N,
        "balanced_correction": g * remainder * (L - remainder) / (2 * L),
        "lower_bound": inherited["lower_bound"],
        "trial_upper_bounds": inherited["trial_upper_bounds"],
        "best_upper_bound": inherited["best_upper_bound"],
        "best_trial": inherited["best_trial"],
        "all_population_lower": all_population_lower,
        "all_population_lower_per_site": all_population_lower / L if g else None,
        "quadratic_minimizer": L * alpha / g if g else None,
        "lower_per_site": inherited["lower_bound"] / L,
        "trial_upper_per_site": {name: value / L for name, value in
                                 inherited["trial_upper_bounds"].items()},
        "best_upper_per_site": inherited["best_upper_bound"] / L,
        "unit_filling_envelope": envelope,
        "classification": {
            "all_population": "bounded_below_all_populations" if g else "unbounded_below",
            "fixed_density": "fixed_density_extensive_bounds" if g else "exact_free_extensivity",
            "completed_square": "applicable" if g else "not_applicable_g_zero",
            "population_selection": "not_selected_by_these_bounds",
        },
        "structural_energy": structural_energy,
    }


def stability_bounds(L, N, C=1, g=Fraction(7, 10)):
    """Exact scalar bounds; coefficients accept only base int/Fraction values."""
    return {"model_id": MODEL_ID, "status": "analytic",
            "parameter_semantics": "exact_rational",
            "bounds": _serialize(_exact_bounds(L, N, C, g)), "scope": _SCOPE}


def _dense_coefficient(value, name, lower, upper):
    if type(value) not in (int, float):
        raise TypeError(name + " must be a base builtin int or float")
    if not lower <= value <= upper:
        raise ValueError(name + " outside supported range")
    result = float(value)
    if not isfinite(result) or (0 < abs(result) < sys.float_info.min):
        raise ValueError(name + " must be finite and zero or normal binary64")
    if type(value) is int and result != value:
        raise ValueError(name + " integer conversion must be exact")
    return result


def _screened_spectrum(H):
    """One screened eigensystem, with owned eigenvalues but no returned vectors."""
    if type(H) is not np.ndarray:
        raise TypeError("H must be a base ndarray")
    if not H.dtype.isnative or H.dtype not in (np.dtype("float64"), np.dtype("complex128")):
        raise TypeError("H must have native float64 or complex128 dtype")
    if H.ndim != 2 or H.shape[0] != H.shape[1] or not 1 <= H.shape[0] <= MAX_DIMENSION:
        raise ValueError("H must be square with dimension 1 through 512")
    if not np.all(np.isfinite(H.real)) or not np.all(np.isfinite(H.imag)):
        raise ValueError("H must have finite real and imaginary components")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            owned = H.copy()
            scale = max(1.0, _frobenius(owned))
            scaled = owned / scale
            if any(np.any((before != 0) & (after == 0)) for before, after in
                   ((owned.real, scaled.real), (owned.imag, scaled.imag))):
                raise NumericalUnavailable("Hamiltonian scaling loses a nonzero component")
            system = _screened_eigensystem(owned)
            values = system["values"]
            if np.any(values[1:] < values[:-1]):
                raise NumericalUnavailable("eigensolver eigenvalues are not nondecreasing")
            return {
                "values": values.copy(), "ground_energy": float(values[0]),
                "dimension": len(owned), "scale": float(system["scale"]),
                "tolerance": float(system["tolerance"]),
                "orthogonality_residual": float(system["orthogonality_residual"]),
                "eigenpair_residual": float(system["eigenpair_residual"]), "certified": False,
            }
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise NumericalUnavailable(str(exc)) from exc


def _energy_comparison(energy, scale, dimension, lower, upper, C):
    """Signed exact margins; display approximations never enter the decision."""
    if type(energy) is not float or type(scale) is not float:
        raise TypeError("energy and scale must be base builtin floats")
    if not isfinite(energy) or not isfinite(scale) or scale < 1:
        raise ValueError("energy must be finite and scale finite and at least one")
    dimension = _integer(dimension, "dimension", 1, MAX_DIMENSION)
    lower, upper, C = (_rational(lower, "lower"), _rational(upper, "upper"),
                       _rational(C, "C"))
    if C <= 0 or lower > upper:
        raise ValueError("C must be positive and lower must not exceed upper")
    e = Fraction.from_float(energy) / C
    l, u = lower / C, upper / C
    S = max(Fraction(1), Fraction.from_float(scale) / C, abs(e), abs(l), abs(u))
    proxy = 1024 * Fraction(1, 2**52) * dimension * S
    lower_margin, upper_margin = e - l, u - e
    consistent = lower_margin >= -proxy and upper_margin >= -proxy
    return {
        "status": "available_heuristic" if consistent else "diagnostic_mismatch",
        "consistent": consistent, "lower_margin": rational_record(lower_margin),
        "upper_margin": rational_record(upper_margin), "scale": rational_record(S),
        "proxy": rational_record(proxy), "normalization": "energy_over_C",
        "certified": False, "scope": _COMPARISON_SCOPE,
    }


def _unavailable(report, reason_code, reason):
    report.update(status="numerical_unavailable", reason_code=reason_code, reason=reason)
    return report


def sector_diagnostic(L, N, C=1.0, g=0.7):
    """Exact lifted bounds and at most one owned full-sector spectrum attempt."""
    L = _integer(L, "L", 3)
    N = _integer(N, "N", 0)
    C = _dense_coefficient(C, "C", 0.5, 2.0)
    g = _dense_coefficient(g, "g", 0.0, 40.0)
    exact = _exact_bounds(L, N, Fraction.from_float(C), Fraction.from_float(g))
    report = {
        "model_id": MODEL_ID, "status": "available_heuristic",
        "reason": None, "reason_code": None,
        "parameter_semantics": "binary64_lifted_exactly",
        "L": _metadata_integer(L), "N": _metadata_integer(N), "C": C, "g": g,
        "dimension": None, "dimension_cap": MAX_DIMENSION,
        "dimension_exceeds_cap": None, "analytic_bounds": _serialize(exact),
        "spectrum": None, "comparison": None, "scope": _SCOPE,
    }
    if L > MAX_SITES:
        return _unavailable(report, "site_cap", "Site count exceeds inherited cap %d." % MAX_SITES)
    dimension = capped_binomial(L + N - 1, N, MAX_DIMENSION)
    if dimension > MAX_DIMENSION:
        report["dimension_exceeds_cap"] = True
        return _unavailable(report, "dimension_cap", "Complete sector dimension exceeds cap 512.")
    report.update(dimension=dimension, dimension_exceeds_cap=False)
    if N == 0:
        report.update(status="structural_vacuum", reason_code="structural_vacuum",
                      reason="Vacuum sector has exactly zero energy.",
                      spectrum={"ground_energy": 0.0, "dimension": 1, "scale": None,
                                "tolerance": None, "orthogonality_residual": None,
                                "eigenpair_residual": None, "certified": True})
        return report
    try:
        try:
            model = all_number_model(L, N, C, g)
        except ValueError as exc:
            if str(exc).startswith("numerically unresolved "):
                raise NumericalUnavailable(str(exc)) from exc
            raise
    except _NUMERICAL as exc:
        return _unavailable(report, "assembly_failure", str(exc))
    try:
        spectrum = _screened_spectrum(model.H)
    except _NUMERICAL as exc:
        return _unavailable(report, "spectrum_failure", str(exc))
    except ValueError as exc:
        if str(exc) == "H fails the heuristic Hermiticity screen":
            return _unavailable(report, "hermiticity_screen_failed", str(exc))
        raise
    report["spectrum"] = {key: spectrum[key] for key in (
        "ground_energy", "dimension", "scale", "tolerance",
        "orthogonality_residual", "eigenpair_residual", "certified")}
    try:
        report["comparison"] = _energy_comparison(
            spectrum["ground_energy"], spectrum["scale"], dimension,
            exact["lower_bound"], exact["best_upper_bound"], exact["C"])
    except _NUMERICAL as exc:
        return _unavailable(report, "comparison_failure", str(exc))
    report["status"] = report["comparison"]["status"]
    if report["status"] == "diagnostic_mismatch":
        report.update(reason="Ground energy lies outside the frozen heuristic bound allowance.",
                      reason_code="energy_bound_mismatch")
    return report


def _unit_filling_exclusion():
    witnesses = []
    for semantics, g in (("exact_rational", Fraction(7, 10)),
                         ("binary64_lifted_exactly", Fraction.from_float(0.7))):
        C = Fraction(1)
        reference, witness = -10 * C, -30 * C + 21 * g
        witnesses.append({
            "parameter_semantics": semantics, "C": rational_record(C),
            "g": rational_record(g), "reference_lower": rational_record(reference),
            "witness_upper": rational_record(witness),
            "strict_margin": rational_record(reference - witness),
            "excluded": witness < reference,
        })
    return {
        "status": "analytic", "L": 5, "q": 5, "reference_N": 5, "witness_N": 15,
        "threshold": "21g<20C", "witnesses": witnesses,
        "scope": "Inherited scalar witness excludes unit filling, not a global population search or winner.",
    }


def demonstration_report():
    """The frozen 18 scalar cases and nine dense attempts, with no retries."""
    volumes = (10, 1000, 1000000)
    densities = (Fraction(1, 2), Fraction(1), Fraction(2))
    couplings = (Fraction(7, 10), Fraction(40))
    scalar_cases = []
    for L in volumes:
        for density in densities:
            population = density * L
            if population.denominator != 1:
                raise ValueError("Exact density requires an integer population")
            for g in couplings:
                scalar_cases.append(stability_bounds(L, population.numerator, 1, g))
    dense_cases = [sector_diagnostic(L, L, 1.0, g)
                   for L in (3, 4, 5) for g in (0.0, 0.7, 40.0)]
    counts = {name: 0 for name in ("available_heuristic", "diagnostic_mismatch",
                                  "numerical_unavailable", "structural_vacuum")}
    for case in dense_cases:
        counts[case["status"]] += 1
    status, reason = "available_heuristic", None
    if counts["diagnostic_mismatch"]:
        status = "diagnostic_mismatch"
        reason = "One or more dense diagnostics violate the frozen heuristic bound allowance."
    elif counts["numerical_unavailable"]:
        status = "numerical_unavailable"
        reason = "One or more dense diagnostics are numerically unavailable."
    return {
        "model_id": MODEL_ID, "status": status, "reason": reason, "scope": _SCOPE,
        "controls": {
            "dense_sectors": [[3, 3], [4, 4], [5, 5]],
            "dense_couplings": [0.0, 0.7, 40.0], "dense_C": 1.0,
            "scalar_volumes": list(volumes),
            "scalar_densities": [rational_record(value) for value in densities],
            "scalar_couplings": [rational_record(value) for value in couplings],
            "scalar_C": rational_record(Fraction(1)), "max_physical_eigensolves": 9,
        },
        "scalar_cases": scalar_cases, "dense_cases": dense_cases,
        "free_classification": {
            "status": "analytic", "formula": "E0=-2CN",
            "fixed_density": "exact_free_extensivity", "all_population": "unbounded_below",
            "scope": "For g=0, fixed-density energy is exact; unrestricted populations are unbounded below.",
        },
        "unit_filling_exclusion": _unit_filling_exclusion(),
        "summary": {"scalar_case_count": len(scalar_cases), "dense_case_count": len(dense_cases),
                    "dense_status_counts": counts},
    }
