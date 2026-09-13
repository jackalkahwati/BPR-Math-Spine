"""Independent tests of the frozen 2026-09-13 extensive-stability contract.

Written without reading the new implementation or demo. No scientific code was
imported or executed while authoring. The final/reconciled contract overrides
its early shorthand. These checks do not certify floating arithmetic or physics.

PHYSICAL INVOCATION LIST (frozen before execution)
* One shared demonstration: (L,N,C,g)=(L,L,1.,g), L=3,4,5 in that order,
  g=0.,0.7,40. in that order. Nine owned model attempts / at most nine eigh
  attempts, keyed by case. Fully successful runs require nine eigh attempts;
  numerical unavailability retains reasons/partials, never promotes mismatch.
  Matrix/lowering/trial consumers reuse successfully captured models, with
  explicit per-case skips only for genuinely unavailable dependent data.
* Two separate isolated CLI demonstrations, text and JSON: the same nine each.
* No other test constructs a physical model or performs a physical eigensolve.
  Boundary/aggregate tests use explicit test-only synthetic model/screen stubs.

SYNTHETIC _screened_spectrum INVOCATION LIST (frozen before execution)
* Valid, one call each: zero1, zero3, degenerate2, diagonal2, complex2,
  subnormal_real1, subnormal_imag2. At most one actual eigh attempt per call.
* Ownership: diagonal2 twice, exactly two inherited screen/eigh attempts.
* Invalid input, one call each: list, tuple, protocol, subclass, int64, float32,
  complex64, bool, object, nonnative64, scalar, vector, rectangular, empty,
  oversized513, nan_real, inf_real, nan_imag, inf_imag. No screen/eigh entry.
* Hermiticity: asymmetric2 once, no eigh entry.
* Scaling loss: real_loss2 and imag_loss2 once each, no inherited screen entry.
* Malformed eigh output on diagonal2, one call each: values_list,
  values_short, values_matrix, values_complex, values_bool, values_nan,
  values_inf, vectors_list, vectors_short, vectors_bool, vectors_nan,
  vectors_inf, not_orthogonal, wrong_eigenpairs, descending, component_loss.
  Exactly one mocked eigh attempt, no actual eigensolve in each invocation.
* Solver exceptions on diagonal2: NumericalUnavailable, FloatingPointError,
  OverflowError, LinAlgError, TypeError, ValueError, RuntimeError, once each.
  Exactly one mocked attempt; only the four numerical classes are adapted.
* Inherited-screen exceptions on diagonal2: the same seven classes, once each.
  Exactly one mocked screen attempt and no eigh attempt.
* Failed-demo solver boundary: nine zero matrices, dimensions10,35,126 three
  each in the physical metadata order, each delegated once to a mocked failing
  eigh; no actual eigensolve and no physical model construction. Assembly and
  pre-screen failed-demo variants never invoke the genuine spectrum fixture.
* Extreme descending adjacent comparison: diag(-1e308,1e308) once with a
  mocked inherited screen (no eigh); reject direct descending comparisons
  without overflow-prone subtraction or sorting.

The lists above are synthetic API fixtures, NOT a hidden physical control grid.
Frobenius identity tolerance is fixed at 2e-10 + 2e-10 * the displayed-term
scale, C-normalized for energies. Raw trial norm tolerance is 2e-10. No SVD,
positivity eigensolve, tolerance fitting, retries or post hoc normalization.
"""
from collections import Counter
import copy
from decimal import Decimal
from fractions import Fraction
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bpr import substrate_current_response as current
from bpr import substrate_fermionization as occupation
from bpr import substrate_gauge_encoding as encoding
from bpr import substrate_vacuum_selection as inherited
from bpr import substrate_extensive_stability as api


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo_substrate_extensive_stability.py"
MODEL_ID = "substrate_extensive_stability"
AVAILABLE = "available_heuristic"
MISMATCH = "diagnostic_mismatch"
UNAVAILABLE = "numerical_unavailable"
VACUUM = "structural_vacuum"
STATUSES = (AVAILABLE, MISMATCH, UNAVAILABLE, VACUUM)
DENSE_GRID = tuple((L, L, 1.0, g) for L in (3, 4, 5) for g in (0.0, 0.7, 40.0))
SCALAR_GRID = tuple((L, int(rho * L), Fraction(1), g)
                    for L in (10, 1000, 1000000)
                    for rho in (Fraction(1, 2), Fraction(1), Fraction(2))
                    for g in (Fraction(7, 10), Fraction(40)))
EPSILON = Fraction(1, 2**52)
TOL = 2e-10
COUNT_KEYS = {"L", "N", "interaction_minimum", "remainder"}
EXACT_KEYS = set("L N C g density interaction_minimum remainder quadratic_lower "
                 "balanced_correction lower_bound trial_upper_bounds best_upper_bound "
                 "best_trial all_population_lower all_population_lower_per_site "
                 "quadratic_minimizer lower_per_site trial_upper_per_site "
                 "best_upper_per_site unit_filling_envelope classification "
                 "structural_energy".split())
SCALAR_KEYS = {"model_id", "status", "parameter_semantics", "bounds", "scope"}
DENSE_KEYS = set("model_id status reason reason_code parameter_semantics L N C g "
                 "dimension dimension_cap dimension_exceeds_cap analytic_bounds "
                 "spectrum comparison scope".split())
SPECTRUM_KEYS = set("ground_energy dimension scale tolerance orthogonality_residual "
                    "eigenpair_residual certified".split())
COMPARISON_KEYS = set("status consistent lower_margin upper_margin scale proxy "
                      "normalization certified scope".split())
DEMO_KEYS = set("model_id status reason scope controls scalar_cases dense_cases "
                "free_classification unit_filling_exclusion summary".split())
REASONS = {AVAILABLE: None,
           MISMATCH: "Ground energy lies outside the frozen heuristic bound allowance.",
           VACUUM: "Vacuum sector has exactly zero energy."}
AGGREGATE_REASONS = {
    AVAILABLE: None,
    UNAVAILABLE: "One or more dense diagnostics are numerically unavailable.",
    MISMATCH: "One or more dense diagnostics violate the frozen heuristic bound allowance.",
}
REASON_CODES = {None, "structural_vacuum", "site_cap", "dimension_cap",
                "assembly_failure", "spectrum_failure", "hermiticity_screen_failed",
                "comparison_failure", "energy_bound_mismatch"}
MODULES = (api, inherited, occupation, encoding, current, math, np, np.linalg)


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class FractionSubclass(Fraction):
    pass


class ArraySubclass(np.ndarray):
    pass


class ProtocolBomb:
    def __float__(self):
        raise AssertionError("untrusted float protocol was invoked")

    def __int__(self):
        raise AssertionError("untrusted integer protocol was invoked")

    def __index__(self):
        raise AssertionError("untrusted index protocol was invoked")

    def __array__(self, *args, **kwargs):
        raise AssertionError("untrusted array protocol was invoked")


def deny(*args, **kwargs):
    raise AssertionError("forbidden allocation, physics construction or extra eigensolve")


def patch_aliases(monkeypatch, original, replacement):
    """Patch provider and direct imports by object identity, never source reading."""
    patched = []
    for module in MODULES:
        for name, value in tuple(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, name, replacement)
                patched.append((module.__name__, name))
    assert patched, "test did not instrument its intended provider seam"
    return patched


def block_physics(monkeypatch, arrays=False, capped=True):
    targets = [inherited.all_number_model, occupation.fixed_number_model,
               occupation._occupations, occupation.binary_operators,
               inherited.sector_ground_diagnostic, encoding._screened_eigensystem,
               np.linalg.eigh, np.linalg.eigvalsh, np.linalg.eig,
               np.linalg.eigvals, np.linalg.svd, math.comb]
    if capped:
        targets.append(inherited.capped_binomial)
    if arrays:
        targets.extend(getattr(np, name) for name in (
            "ndarray", "matrix", "array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray",
            "empty", "zeros", "ones", "full", "empty_like", "zeros_like",
            "ones_like", "full_like", "eye", "identity", "arange", "linspace",
            "logspace", "geomspace", "fromiter", "frombuffer", "fromfunction",
            "diag", "diagonal", "stack", "vstack", "hstack", "column_stack",
            "concatenate", "block"))
    seen = set()
    for target in targets:
        if id(target) not in seen:
            seen.add(id(target))
            patch_aliases(monkeypatch, target, deny)


def native(value):
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            native(item)
    elif type(value) is list:
        for item in value:
            native(item)
    else:
        assert value is None or type(value) in (int, float, str, bool)
        if type(value) is float:
            assert math.isfinite(value)


def rational(record, expected):
    """Independent lossless and optional-display oracle, not rational_record."""
    expected = Fraction(expected)
    assert type(record) is dict
    assert set(record) == {"numerator", "denominator", "approximate", "approximation_status"}
    assert type(record["numerator"]) is str and type(record["denominator"]) is str
    assert record["numerator"] == str(expected.numerator)
    assert record["denominator"] == str(expected.denominator)
    try:
        display = float(expected)
    except OverflowError:
        display, status = None, "overflow"
    else:
        if not math.isfinite(display):
            display, status = None, "overflow"
        elif expected and display == 0:
            display, status = None, "underflow"
        elif 0 < abs(display) < sys.float_info.min:
            display, status = None, "subnormal"
        else:
            status = "normal" if expected else "exact_zero"
    assert record["approximation_status"] == status
    assert record["approximate"] == display
    if display is not None:
        assert type(record["approximate"]) is float


def metadata(actual, expected):
    if abs(expected) <= 2**53 - 1:
        assert type(actual) is int and actual == expected
    else:
        assert type(actual) is str and actual == str(expected)


def exact_oracle(L, N, C, g):
    """All arithmetic is independent builtin integer/Fraction arithmetic."""
    C, g = Fraction(C), Fraction(g)
    a, r = divmod(N, L)
    minimum = L * a * (a - 1) // 2 + a * r
    quadratic = g * Fraction(N * N, 2 * L) - (g / 2 + 2 * C) * N
    correction = g * Fraction(r * (L - r), 2 * L)
    lower = g * minimum - 2 * C * N
    trials = {"balanced_occupation": g * minimum,
              "uniform_condensate": -2 * C * N + g * Fraction(N * (N - 1), 2 * L)}
    if N == L:
        trials["unit_filling_two_state"] = (
            -2 * C * C * (g + 4 * C) / ((g + 2 * C)**2 + 2 * C * C))
    best = min(trials, key=trials.get)
    alpha = g / 2 + 2 * C
    global_lower = -L * alpha * alpha / (2 * g) if g else None
    result = dict(L=L, N=N, C=C, g=g, density=Fraction(N, L),
                  interaction_minimum=minimum, remainder=r,
                  quadratic_lower=quadratic, balanced_correction=correction,
                  lower_bound=lower, trial_upper_bounds=trials,
                  best_upper_bound=trials[best], best_trial=best,
                  all_population_lower=global_lower,
                  all_population_lower_per_site=global_lower / L if g else None,
                  quadratic_minimizer=L * alpha / g if g else None,
                  lower_per_site=lower / L,
                  trial_upper_per_site={key: value / L for key, value in trials.items()},
                  best_upper_per_site=trials[best] / L,
                  unit_filling_envelope=(dict(lower=-2 * C, upper=min(
                      Fraction(0), -2 * C + g * Fraction(L - 1, 2 * L)))
                      if N == L else None),
                  classification={
                      "all_population": "bounded_below_all_populations" if g else "unbounded_below",
                      "fixed_density": "fixed_density_extensive_bounds" if g else "exact_free_extensivity",
                      "completed_square": "applicable" if g else "not_applicable_g_zero",
                      "population_selection": "not_selected_by_these_bounds"},
                  structural_energy=(Fraction(0) if N == 0 else -2 * C * N
                                     if not g or N == 1 else None))
    assert set(result) == EXACT_KEYS
    assert lower == quadratic + correction
    return result


def assert_exact(actual, expected):
    assert type(actual) is dict and set(actual) == EXACT_KEYS
    assert actual == expected
    for key, item in actual.items():
        if key in COUNT_KEYS:
            assert type(item) is int
        elif type(expected[key]) is Fraction:
            assert type(item) is Fraction
        elif key in ("trial_upper_bounds", "trial_upper_per_site", "unit_filling_envelope"):
            if item is not None:
                assert list(item) == list(expected[key])
                assert all(type(value) is Fraction for value in item.values())


def assert_payload(actual, expected):
    assert type(actual) is dict and set(actual) == EXACT_KEYS
    for key, value in expected.items():
        if key in COUNT_KEYS:
            metadata(actual[key], value)
        elif type(value) is Fraction:
            rational(actual[key], value)
        elif key in ("trial_upper_bounds", "trial_upper_per_site", "unit_filling_envelope") and value is not None:
            assert list(actual[key]) == list(value)
            for name, entry in value.items():
                rational(actual[key][name], entry)
        else:
            assert actual[key] == value
    native(actual)


def assert_scalar(report, args):
    assert set(report) == SCALAR_KEYS
    assert report["model_id"] == MODEL_ID
    assert report["status"] == "analytic"
    assert report["parameter_semantics"] == "exact_rational"
    assert type(report["scope"]) is str and report["scope"].strip()
    assert_payload(report["bounds"], exact_oracle(*args))
    native(report)


def comparison_oracle(energy, scale, dimension, lower, upper, C):
    E, s = Fraction.from_float(energy), Fraction.from_float(scale)
    lower, upper, C = Fraction(lower), Fraction(upper), Fraction(C)
    e, l, u = E / C, lower / C, upper / C
    S = max(Fraction(1), s / C, abs(e), abs(l), abs(u))
    proxy = 1024 * EPSILON * dimension * S
    lm, um = e - l, u - e
    return lm, um, S, proxy, lm >= -proxy and um >= -proxy


def assert_comparison(report, args):
    lm, um, S, proxy, consistent = comparison_oracle(*args)
    assert set(report) == COMPARISON_KEYS
    assert report["status"] == (AVAILABLE if consistent else MISMATCH)
    assert report["consistent"] is consistent
    for key, value in (("lower_margin", lm), ("upper_margin", um), ("scale", S), ("proxy", proxy)):
        rational(report[key], value)
    assert report["normalization"] == "energy_over_C"
    assert report["certified"] is False
    assert type(report["scope"]) is str and report["scope"].strip()
    native(report)


def assert_dense(report, args):
    L, N, C, g = args
    C, g = float(C), float(g)
    assert set(report) == DENSE_KEYS
    assert report["model_id"] == MODEL_ID
    assert report["parameter_semantics"] == "binary64_lifted_exactly"
    metadata(report["L"], L)
    metadata(report["N"], N)
    assert type(report["C"]) is float and report["C"] == C
    assert type(report["g"]) is float and report["g"] == g
    assert report["dimension_cap"] == 512
    assert report["reason_code"] in REASON_CODES
    assert type(report["scope"]) is str and report["scope"].strip()
    exact = exact_oracle(L, N, Fraction.from_float(C), Fraction.from_float(g))
    assert_payload(report["analytic_bounds"], exact)
    if report["status"] in REASONS:
        assert report["reason"] == REASONS[report["status"]]
        assert report["reason_code"] == {
            AVAILABLE: None, MISMATCH: "energy_bound_mismatch", VACUUM: "structural_vacuum"
        }[report["status"]]
    else:
        assert report["status"] == UNAVAILABLE
        assert type(report["reason"]) is str and report["reason"].strip()
    if report["dimension"] is not None:
        assert type(report["dimension"]) is int
        assert report["dimension"] == math.comb(L + N - 1, N)
        assert report["dimension_exceeds_cap"] is False
    if report["status"] == UNAVAILABLE:
        code = report["reason_code"]
        assert report["comparison"] is None
        if code == "comparison_failure":
            assert report["spectrum"] is not None and report["dimension"] is not None
        else:
            assert code in {"site_cap", "dimension_cap", "assembly_failure",
                            "spectrum_failure", "hermiticity_screen_failed"}
            assert report["spectrum"] is None
        if code == "site_cap":
            assert report["dimension"] is None and report["dimension_exceeds_cap"] is None
        elif code == "dimension_cap":
            assert report["dimension"] is None and report["dimension_exceeds_cap"] is True
        else:
            assert report["dimension"] is not None and report["dimension_exceeds_cap"] is False
    if report["spectrum"] is not None:
        spectrum = report["spectrum"]
        assert set(spectrum) == SPECTRUM_KEYS
        assert spectrum["dimension"] == report["dimension"]
        if report["status"] == VACUUM:
            assert spectrum["ground_energy"] == 0.0
            assert spectrum["certified"] is True
            for key in SPECTRUM_KEYS - {"ground_energy", "dimension", "certified"}:
                assert spectrum[key] is None
            assert report["comparison"] is None
        else:
            assert type(spectrum["ground_energy"]) is float
            assert spectrum["certified"] is False
            assert spectrum["scale"] >= 1.0
            assert spectrum["tolerance"] == 256 * float(EPSILON) * spectrum["dimension"]
            assert 0 <= spectrum["orthogonality_residual"] <= spectrum["tolerance"]
            assert 0 <= spectrum["eigenpair_residual"] / spectrum["scale"] <= spectrum["tolerance"]
    if report["comparison"] is not None:
        spectrum = report["spectrum"]
        assert_comparison(report["comparison"], (
            spectrum["ground_energy"], spectrum["scale"], spectrum["dimension"],
            exact["lower_bound"], exact["best_upper_bound"], Fraction.from_float(C)))
        assert report["status"] == report["comparison"]["status"]
    native(report)


@pytest.fixture(scope="module")
def shared_demo():
    """The ONLY in-process physical run: attempt counters precede delegation."""
    calls = {"model": [], "fixed_model": [], "screen": [], "eigh": [], "scalar": []}
    models = {}
    active = {"model": None, "screen": None}
    original_fixed = occupation.fixed_number_model
    original_model = inherited.all_number_model
    original_screen = api._screened_spectrum
    original_eigh = np.linalg.eigh
    original_scalar = api.stability_bounds

    def fixed_model(*args, **kwargs):
        bound = inspect.signature(original_fixed).bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments[name] for name in ("L", "N", "C", "g"))
        calls["fixed_model"].append(key)
        assert key in DENSE_GRID and calls["fixed_model"].count(key) == 1
        assert key == active["model"]
        return original_fixed(*args, **kwargs)

    def model(*args, **kwargs):
        bound = inspect.signature(original_model).bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments[name] for name in ("L", "N", "C", "g"))
        calls["model"].append(key)
        assert len(calls["model"]) <= 9
        assert key == DENSE_GRID[len(calls["model"]) - 1], "hidden physical grid"
        active["model"] = key
        try:
            result = original_model(*args, **kwargs)
        finally:
            active["model"] = None
        models[key] = result
        return result

    def screen(H):
        key = calls["model"][-1]
        assert key in models and key not in calls["screen"]
        calls["screen"].append(key)
        assert H is models[key].H or np.array_equal(H, models[key].H)
        active["screen"] = key
        try:
            return original_screen(H)
        finally:
            active["screen"] = None

    def eigh(H, *args, **kwargs):
        key = active["screen"]
        assert key in DENSE_GRID and key in calls["screen"]
        assert all(prior != key for prior, _ in calls["eigh"])
        calls["eigh"].append((key, H.copy()))
        return original_eigh(H, *args, **kwargs)

    def scalar(*args, **kwargs):
        bound = inspect.signature(original_scalar).bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments[name] for name in ("L", "N", "C", "g"))
        calls["scalar"].append(key)
        assert len(calls["scalar"]) <= 18
        assert key == SCALAR_GRID[len(calls["scalar"]) - 1], "hidden scalar grid"
        assert type(key[0]) is int and type(key[1]) is int
        assert type(key[2]) in (int, Fraction) and type(key[3]) in (int, Fraction)
        with pytest.MonkeyPatch.context() as guard:
            block_physics(guard, arrays=True)
            return original_scalar(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, original_fixed, fixed_model)
        patch_aliases(patch, original_model, model)
        patch_aliases(patch, original_screen, screen)
        patch_aliases(patch, original_eigh, eigh)
        patch_aliases(patch, original_scalar, scalar)
        for other in (np.linalg.eigvalsh, np.linalg.eig, np.linalg.eigvals,
                      np.linalg.svd, inherited.sector_ground_diagnostic):
            patch_aliases(patch, other, deny)
        report = api.demonstration_report()
    return report, calls, models


def assert_demo_metadata(report):
    assert set(report) == DEMO_KEYS
    assert report["model_id"] == MODEL_ID
    assert type(report["scope"]) is str and report["scope"].strip()
    controls = report["controls"]
    assert set(controls) == {"dense_sectors", "dense_couplings", "dense_C",
                             "scalar_volumes", "scalar_densities", "scalar_couplings",
                             "scalar_C", "max_physical_eigensolves"}
    assert controls["dense_sectors"] == [[3, 3], [4, 4], [5, 5]]
    assert controls["dense_couplings"] == [0.0, 0.7, 40.0]
    assert all(type(value) is float for value in controls["dense_couplings"])
    assert controls["dense_C"] == 1.0 and type(controls["dense_C"]) is float
    assert controls["scalar_volumes"] == [10, 1000, 1000000]
    assert len(controls["scalar_densities"]) == 3
    assert len(controls["scalar_couplings"]) == 2
    for actual, value in zip(controls["scalar_densities"], (Fraction(1, 2), Fraction(1), Fraction(2))):
        rational(actual, value)
    for actual, value in zip(controls["scalar_couplings"], (Fraction(7, 10), Fraction(40))):
        rational(actual, value)
    rational(controls["scalar_C"], 1)
    assert type(controls["max_physical_eigensolves"]) is int
    assert controls["max_physical_eigensolves"] == 9
    assert len(report["scalar_cases"]) == 18 and len(report["dense_cases"]) == 9
    for row, args in zip(report["scalar_cases"], SCALAR_GRID):
        assert_scalar(row, args)
    for row, args in zip(report["dense_cases"], DENSE_GRID):
        assert_dense(row, args)
    summary = report["summary"]
    assert set(summary) == {"scalar_case_count", "dense_case_count", "dense_status_counts"}
    assert summary["scalar_case_count"] == 18 and summary["dense_case_count"] == 9
    counts = {status: 0 for status in STATUSES}
    counts.update(Counter(case["status"] for case in report["dense_cases"]))
    assert summary["dense_status_counts"] == counts
    assert all(type(value) is int for value in counts.values())
    status = MISMATCH if counts[MISMATCH] else UNAVAILABLE if counts[UNAVAILABLE] else AVAILABLE
    assert report["status"] == status and report["reason"] == AGGREGATE_REASONS[status]
    free = report["free_classification"]
    assert set(free) == {"status", "formula", "fixed_density", "all_population", "scope"}
    assert free["status"] == "analytic" and free["formula"] == "E0=-2CN"
    assert free["fixed_density"] == "exact_free_extensivity"
    assert free["all_population"] == "unbounded_below"
    assert type(free["scope"]) is str and free["scope"].strip()
    exclusion = report["unit_filling_exclusion"]
    assert set(exclusion) == {"status", "L", "q", "reference_N", "witness_N", "threshold", "scope", "witnesses"}
    assert exclusion["status"] == "analytic"
    assert (exclusion["L"], exclusion["q"], exclusion["reference_N"], exclusion["witness_N"]) == (5, 5, 5, 15)
    assert exclusion["threshold"] == "21g<20C"
    assert type(exclusion["scope"]) is str and exclusion["scope"].strip()
    assert len(exclusion["witnesses"]) == 2
    for row, semantics, g in zip(exclusion["witnesses"],
                                 ("exact_rational", "binary64_lifted_exactly"),
                                 (Fraction(7, 10), Fraction.from_float(0.7))):
        assert set(row) == {"parameter_semantics", "C", "g", "reference_lower", "witness_upper", "strict_margin", "excluded"}
        assert row["parameter_semantics"] == semantics
        for key, value in (("C", 1), ("g", g), ("reference_lower", -10),
                           ("witness_upper", -30 + 21 * g), ("strict_margin", 20 - 21 * g)):
            rational(row[key], value)
        assert row["excluded"] is True
    assert exclusion["witnesses"][0]["g"] != exclusion["witnesses"][1]["g"]
    native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_demo_complete_grid_metadata_and_exact_attempt_budget(shared_demo):
    report, calls, models = shared_demo
    assert_demo_metadata(report)
    assert report["status"] != MISMATCH, report
    assert calls["model"] == list(DENSE_GRID)
    assert calls["scalar"] == list(SCALAR_GRID)
    for stage in ("model", "fixed_model", "screen"):
        assert set(calls[stage]) <= set(DENSE_GRID)
        assert all(count == 1 for count in Counter(calls[stage]).values())
    eigh_keys = [key for key, _ in calls["eigh"]]
    assert len(eigh_keys) == len(set(eigh_keys)) <= 9
    assert set(eigh_keys) <= set(calls["screen"]) <= set(models)
    assert len({id(model) for model in models.values()}) == len(models)
    for key in DENSE_GRID:
        row = next(row for row in report["dense_cases"]
                   if (row["L"], row["N"], row["C"], row["g"]) == key)
        if row["status"] == AVAILABLE:
            assert key in models and key in calls["screen"] and key in eigh_keys
        else:
            assert row["status"] == UNAVAILABLE
            if row["reason_code"] == "assembly_failure":
                assert key not in models and key not in calls["screen"] and key not in eigh_keys
            elif row["reason_code"] in ("spectrum_failure", "hermiticity_screen_failed"):
                assert key in models and key in calls["screen"]
            elif row["reason_code"] == "comparison_failure":
                assert key in models and key in calls["screen"] and key in eigh_keys
            else:
                pytest.fail("frozen small dense grid unexpectedly hit a resource cap")
        if key in models:
            assert len(models[key].basis) == {3: 10, 4: 35, 5: 126}[key[0]]
    if report["status"] == AVAILABLE:
        assert len(calls["screen"]) == len(calls["eigh"]) == len(models) == 9
    assert occupation.MAX_SITES == 12
    assert occupation.MAX_DENSE_DIM == 512 and occupation.MAX_BINARY_DIM == 256
    assert api.NumericalUnavailable is current.NumericalUnavailable


def independent_occupations(L, N):
    """Combinations of occupied site labels, not the inherited recursion."""
    return tuple(sorted(tuple(particles.count(site) for site in range(L))
                        for particles in itertools.combinations_with_replacement(range(L), N)))


def frobenius(matrix):
    return math.sqrt(float(np.sum(np.abs(matrix)**2)))


def matrix_identity(left_terms, right_terms):
    """Terms are exactly those frozen in the doc, not residual-based scales."""
    residual = frobenius(sum(left_terms) - sum(right_terms))
    scale = sum(frobenius(term) for term in left_terms + right_terms)
    assert residual <= TOL + TOL * scale


def scalar_identity(actual, expected):
    discrepancy = actual - expected  # Keep the signed discrepancy before testing.
    assert abs(discrepancy) <= TOL + TOL * (abs(actual) + abs(expected))


@pytest.mark.parametrize("args", DENSE_GRID)
def test_captured_matrices_independent_moves_lowering_sos_and_raw_trials(shared_demo, args):
    """No extra model/solver, including on the auxiliary N-1 row space."""
    report, _, models = shared_demo
    rows = {(row["L"], row["N"], row["C"], row["g"]): row for row in report["dense_cases"]}
    assert rows[args]["status"] != MISMATCH, rows[args]
    if args not in models:
        row = rows[args]
        assert row["status"] == UNAVAILABLE and row["reason_code"] == "assembly_failure"
        pytest.skip("matrix/trial fixture unavailable for {!r}: {}".format(args, row["reason"]))
    # One parameterized consumer per key makes every unavailable matrix visible.
    for model in (models[args],):
        L, N, C, g = args
        basis = independent_occupations(L, N)
        rows = independent_occupations(L, N - 1)
        assert tuple(model.basis) == basis
        assert len(basis) == math.comb(L + N - 1, N)
        assert (N,) + (0,) * (L - 1) in basis  # No local cutoff/hard-core replacement.
        index, row_index = ({state: i for i, state in enumerate(states)} for states in (basis, rows))
        d = len(basis)
        I = np.eye(d)
        onsite = np.diag([g * sum(n * (n - 1) // 2 for n in state) for state in basis])
        hopping = np.zeros((d, d))
        lowering = [np.zeros((len(rows), d)) for _ in range(L)]
        for column, state in enumerate(basis):
            for source, count in enumerate(state):
                if not count:
                    continue
                lowered = list(state)
                lowered[source] -= 1
                lowering[source][row_index[tuple(lowered)], column] = math.sqrt(count)
                for target in ((source - 1) % L, (source + 1) % L):
                    moved = list(lowered)
                    moved[target] += 1
                    hopping[index[tuple(moved)], column] -= C * math.sqrt(count * (state[target] + 1))
        matrix_identity([model.H / C], [onsite / C, hopping / C])
        matrix_identity([model.V / C], [hopping / C])
        bonds = [(lowering[x] - lowering[(x + 1) % L]).T @
                 (lowering[x] - lowering[(x + 1) % L]) for x in range(L)]
        matrix_identity([hopping / C, 2 * N * I], bonds)
        occupation_square = np.diag([sum(n * n for n in state) for state in basis])
        deviations = [np.diag([(state[x] - N / L)**2 for state in basis]) for x in range(L)]
        matrix_identity([occupation_square], [N * N / L * I] + deviations)
        if g > 0:
            alpha = g / 2 + 2 * C
            Nstar = L * alpha / g
            B = L * alpha * alpha / (2 * g)
            matrix_identity([model.H / C, B * I / C], [
                g * (N - Nstar)**2 / (2 * L * C) * I,
                g / (2 * C) * sum(deviations), sum(bonds)])
        # For g=0, never form B or Nstar: only the preceding dimensionless and
        # hopping identities apply, not a semibounded completed-square theorem.
        a, r = divmod(N, L)
        balanced_state = (a + 1,) * r + (a,) * (L - r)
        balanced = np.zeros(d)
        balanced[index[balanced_state]] = 1.0
        uniform = np.array([math.sqrt(math.factorial(N) /
                           (L**N * math.prod(math.factorial(n) for n in state))) for state in basis])
        omega, pair = index[(1,) * L], index[(2, 0) + (1,) * (L - 2)]
        s = C / (g + 2 * C)
        z = np.zeros(d)
        z[omega] = 1 / math.sqrt(1 + 2 * s * s)
        z[pair] = math.sqrt(2) * s / math.sqrt(1 + 2 * s * s)
        scalar_identity(model.H[omega, omega] / C, 0.0)
        scalar_identity(model.H[pair, pair] / C, g / C)
        scalar_identity(model.H[omega, pair] / C, -math.sqrt(2))
        scalar_identity(model.H[pair, omega] / C, -math.sqrt(2))
        exact = exact_oracle(L, N, Fraction.from_float(C), Fraction.from_float(g))
        for vector, name in ((balanced, "balanced_occupation"), (uniform, "uniform_condensate"),
                             (z, "unit_filling_two_state")):
            assert abs(np.vdot(vector, vector) - 1) <= TOL  # Raw, never renormalized.
            energy = np.vdot(vector, model.H @ vector)
            expected = float(exact["trial_upper_bounds"][name])
            scalar_identity(energy / C, expected / C)


@pytest.mark.parametrize("args", DENSE_GRID)
def test_captured_free_ground_energy_and_frozen_spectral_scales(shared_demo, args):
    report, calls, models = shared_demo
    rows = {(row["L"], row["N"], row["C"], row["g"]): row for row in report["dense_cases"]}
    delegated = dict(calls["eigh"])
    row = rows[args]
    assert row["status"] != MISMATCH, row
    if row["spectrum"] is None:
        assert row["status"] == UNAVAILABLE
        pytest.skip("spectral fixture unavailable for {!r}: {} ({})".format(
            args, row["reason"], row["reason_code"]))
    assert args in models and args in delegated
    model, spectrum = models[args], row["spectrum"]
    scale = max(1.0, frobenius(model.H))
    assert spectrum["scale"] == pytest.approx(scale, rel=2e-15, abs=0)
    np.testing.assert_allclose(delegated[args], model.H / scale, rtol=2e-15, atol=0)
    assert spectrum["tolerance"] == 256 * float(EPSILON) * len(model.H)
    if model.g == 0:
        scalar_identity(spectrum["ground_energy"] / model.C, -2.0 * model.N)
    assert row["analytic_bounds"]["structural_energy"] is None or model.g == 0


def test_scalar_eigh_basis_array_and_serialization_tripwires():
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        for args in SCALAR_GRID:
            result = api.stability_bounds(*args)
            assert_scalar(result, args)
            assert_exact(api._exact_bounds(*args), exact_oracle(*args))
            json.dumps(result, allow_nan=False)


def test_scalar_balanced_integer_transfers_correction_and_intensive_formulas():
    # Pure integer occupation samples: no matrices, basis provider or hidden grid.
    samples = ((3, (7, 0, 2)), (4, (3, 1, 0, 5)), (5, (0, 4, 2, 1, 0)))
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        for L, original in samples:
            state = list(original)
            N = sum(state)
            count = lambda ns: sum(n * (n - 1) // 2 for n in ns)
            while max(state) - min(state) >= 2:
                hi, lo = state.index(max(state)), state.index(min(state))
                u, v, before = state[hi], state[lo], count(state)
                state[hi] -= 1
                state[lo] += 1
                assert before - count(state) == u - v - 1 > 0
            expected = exact_oracle(L, N, Fraction(3, 2), Fraction(7, 10))
            actual = api._exact_bounds(L, N, Fraction(3, 2), Fraction(7, 10))
            assert_exact(actual, expected)
            assert count(state) == actual["interaction_minimum"]
            rho, delta = Fraction(N, L), Fraction(N % L, L)
            assert Fraction(count(state), L) == (rho**2 - rho + delta * (1 - delta)) / 2
            assert actual["lower_bound"] - actual["quadratic_lower"] == Fraction(7, 10) * (N % L) * (L - N % L) / (2 * L)
        for L, N, C, g in SCALAR_GRID:
            actual = api._exact_bounds(L, N, C, g)
            assert actual["interaction_minimum"] == (L if N == 2 * L else 0)
            assert actual["lower_per_site"] == actual["lower_bound"] / L
            assert actual["best_upper_per_site"] == actual["best_upper_bound"] / L
            if N == L:
                assert actual["unit_filling_envelope"]["lower"] == -2 * C
                assert actual["best_upper_per_site"] <= actual["unit_filling_envelope"]["upper"]


@pytest.mark.parametrize("args", [
    (3, 0, 1, 0), (3, 0, 1, 40), (5, 1, Fraction(2, 3), 40),
    (5, 15, 1, 0), (5, 5, 1, 0), (5, 15, 1, Fraction(7, 10)),
    (5, 15, 1, Fraction.from_float(0.7)),
    (2**53 - 1, 2**53 - 1, 1, Fraction(7, 10)),
    (2**53, 2**53 + 1, 1, Fraction(7, 10)),
    (10**100, 10**200 + 7, Fraction(10**400), Fraction(1, 10**400)),
    (10**80 + 3, 10**100 + 5, Fraction(1, 10**400), Fraction(10**400)),
    (3, 2, Fraction(1, 10**310), Fraction(1, 10**310)),
])
def test_scalar_exact_extremes_structural_identities_and_metadata(args):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        assert_exact(api._exact_bounds(*args), exact_oracle(*args))
        assert_scalar(api.stability_bounds(*args), args)


def test_scalar_default_tie_order_and_binary64_distinction():
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        assert_scalar(api.stability_bounds(5, 5), (5, 5, 1, Fraction(7, 10)))
        assert_exact(api._exact_bounds(5, 5, 1, Fraction(7, 10)), exact_oracle(5, 5, 1, Fraction(7, 10)))
        vacuum = api._exact_bounds(3, 0, 1, 40)
        assert vacuum["best_trial"] == "balanced_occupation"
        assert list(vacuum["trial_upper_bounds"]) == ["balanced_occupation", "uniform_condensate"]
        exact = api._exact_bounds(5, 15, 1, Fraction(7, 10))
        lifted = api._exact_bounds(5, 15, 1, Fraction.from_float(0.7))
        assert exact["trial_upper_bounds"]["uniform_condensate"] == Fraction(-153, 10)
        assert lifted["trial_upper_bounds"]["uniform_condensate"] == -30 + 21 * Fraction.from_float(0.7)
        assert exact != lifted
        assert exact["classification"]["population_selection"] == "not_selected_by_these_bounds"


STRICT_BAD = (True, False, IntSubclass(3), FloatSubclass(1), FractionSubclass(1),
              np.int64(3), np.float64(1), np.bool_(True), Decimal("1"),
              "3", 1 + 0j, ProtocolBomb(), None)


@pytest.mark.parametrize("name", ("L", "N", "C", "g"))
@pytest.mark.parametrize("bad", STRICT_BAD)
def test_scalar_rejects_nonbase_types_without_allocation(name, bad):
    args = dict(L=3, N=3, C=1, g=Fraction(7, 10))
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        for function in (api.stability_bounds, api._exact_bounds):
            with pytest.raises(TypeError):
                function(**args)


@pytest.mark.parametrize("name,bad", [("L", 3.0), ("L", Fraction(3)), ("N", 1.0),
                                       ("N", Fraction(1)), ("C", 1.0), ("g", 0.7)])
def test_scalar_rejects_even_exact_float_and_fraction_counts(name, bad):
    args = dict(L=3, N=3, C=1, g=Fraction(7, 10))
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        for function in (api.stability_bounds, api._exact_bounds):
            with pytest.raises(TypeError):
                function(**args)


@pytest.mark.parametrize("name,bad", [("L", 2), ("L", -1), ("N", -1),
                                       ("C", 0), ("C", -1), ("g", Fraction(-1, 10**400))])
def test_scalar_domain_errors_before_allocation(name, bad):
    args = dict(L=3, N=3, C=1, g=Fraction(7, 10))
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        for function in (api.stability_bounds, api._exact_bounds):
            with pytest.raises(ValueError):
                function(**args)


def test_scalar_reports_and_exact_fixtures_are_detached():
    args = (5, 5, Fraction(1), Fraction(7, 10))
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        a, b = api.stability_bounds(*args), api.stability_bounds(*args)
        assert a is not b and a["bounds"] is not b["bounds"]
        a["bounds"]["trial_upper_bounds"]["uniform_condensate"]["numerator"] = "corrupt"
        a["bounds"]["classification"]["all_population"] = "corrupt"
        assert_scalar(b, args)
        assert_scalar(api.stability_bounds(*args), args)
        x, y = api._exact_bounds(*args), api._exact_bounds(*args)
        x["trial_upper_bounds"]["balanced_occupation"] = Fraction(99)
        x["unit_filling_envelope"]["lower"] = Fraction(99)
        assert_exact(y, exact_oracle(*args))
        assert_exact(api._exact_bounds(*args), exact_oracle(*args))


@pytest.mark.parametrize("args", [
    (0.0, 1.0, 1, 0, 0, 1),
    (-0.0, 1.0, 512, Fraction(-1, 3), Fraction(1, 7), Fraction(3, 2)),
    (-6.1, 17.0, 10, -7, -6, 2),
    (-1.0, 1.0, 1, Fraction(-1) + Fraction(1, 2**42), 1, 1),
    (-1.0, 1.0, 1, Fraction(-1) + Fraction(1, 2**42) + Fraction(1, 2**100), 1, 1),
    (1.0, 1.0, 1, -1, Fraction(1) - Fraction(1, 2**42), 1),
    (1.0, 1.0, 1, -1, Fraction(1) - Fraction(1, 2**42) - Fraction(1, 2**100), 1),
    (0.7, 3.0, 35, Fraction(7, 10), Fraction(7, 10), Fraction(1, 3)),
    (0.0, 1.0, 512, -10**400, 10**400, Fraction(1, 10**400)),
    (0.0, 1.0, 1, Fraction(-1, 10**400), Fraction(1, 10**400), 10**400),
    (sys.float_info.max, sys.float_info.max, 2, 0, 10**400, Fraction(1, 10**400)),
    (float.fromhex('0x0.0000000000001p-1022'), 1.0, 1, 0, 1, 1),
])
def test_energy_comparison_exact_signed_margins_scale_proxy_and_displays(args):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        result = api._energy_comparison(*args)
    assert_comparison(result, args)


def test_energy_comparison_boundary_is_inclusive_not_float_rounded():
    quantum = Fraction(1, 2**42)
    cases = [(-1.0, 1.0, 1, -1 + quantum, 1, 1),
             (-1.0, 1.0, 1, -1 + quantum + Fraction(1, 2**100), 1, 1),
             (1.0, 1.0, 1, -1, 1 - quantum, 1),
             (1.0, 1.0, 1, -1, 1 - quantum - Fraction(1, 2**100), 1)]
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        reports = [api._energy_comparison(*args) for args in cases]
    assert [row["consistent"] for row in reports] == [True, False, True, False]
    for report, args in zip(reports, cases):
        assert_comparison(report, args)
    assert reports[0]["lower_margin"]["numerator"].startswith("-")
    assert reports[2]["upper_margin"]["numerator"].startswith("-")


@pytest.mark.parametrize("name,bad,error", [
    ("energy", 0, TypeError), ("energy", Fraction(0), TypeError),
    ("energy", FloatSubclass(0), TypeError), ("energy", np.float64(0), TypeError),
    ("energy", True, TypeError), ("energy", ProtocolBomb(), TypeError),
    ("energy", float("nan"), ValueError), ("energy", float("inf"), ValueError),
    ("scale", 1, TypeError), ("scale", True, TypeError),
    ("scale", np.float64(1), TypeError), ("scale", FloatSubclass(1), TypeError),
    ("scale", Fraction(1), TypeError), ("scale", 0.9999999999999999, ValueError),
    ("scale", float("inf"), ValueError), ("scale", float("nan"), ValueError),
    ("dimension", True, TypeError), ("dimension", IntSubclass(1), TypeError),
    ("dimension", np.int64(1), TypeError), ("dimension", 1.0, TypeError),
    ("dimension", 0, ValueError), ("dimension", 513, ValueError),
    ("lower", 0.0, TypeError), ("upper", 1.0, TypeError), ("C", 1.0, TypeError),
    ("lower", True, TypeError), ("upper", FractionSubclass(1), TypeError),
    ("C", IntSubclass(1), TypeError), ("lower", np.int64(0), TypeError),
    ("upper", ProtocolBomb(), TypeError), ("C", Decimal("1"), TypeError),
    ("lower", 2, ValueError), ("C", 0, ValueError), ("C", -1, ValueError),
])
def test_energy_comparison_strict_validation_before_numerical_work(name, bad, error):
    args = dict(energy=0.0, scale=1.0, dimension=1, lower=0, upper=1, C=1)
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        with pytest.raises(error):
            api._energy_comparison(**args)


def test_energy_comparison_all_exact_inputs_reject_subclasses_and_protocols():
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        for name in ("lower", "upper", "C"):
            for bad in STRICT_BAD:
                args = dict(energy=0.0, scale=1.0, dimension=1, lower=0, upper=1, C=1)
                args[name] = bad
                with pytest.raises(TypeError):
                    api._energy_comparison(**args)


@pytest.mark.parametrize("name", ("L", "N", "C", "g"))
@pytest.mark.parametrize("bad", STRICT_BAD)
def test_dense_rejects_nonbase_types_before_model_or_dimension(name, bad):
    args = dict(L=3, N=3, C=1.0, g=0.7)
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        with pytest.raises(TypeError):
            api.sector_diagnostic(**args)


@pytest.mark.parametrize("name,bad,error", [
    ("L", 3.0, TypeError), ("N", 1.0, TypeError),
    ("L", Fraction(3), TypeError), ("N", Fraction(1), TypeError),
    ("C", Fraction(1), TypeError), ("g", Fraction(7, 10), TypeError),
    ("L", 2, ValueError), ("N", -1, ValueError),
    ("C", 0.0, ValueError), ("C", 0.49999999999999994, ValueError),
    ("C", 2.0000000000000004, ValueError), ("g", -0.1, ValueError),
    ("g", 40.00000000000001, ValueError),
    ("C", float("nan"), ValueError), ("C", float("inf"), ValueError),
    ("g", float("nan"), ValueError), ("g", float("-inf"), ValueError),
    ("C", float.fromhex('0x0.0000000000001p-1022'), ValueError),
    ("g", float.fromhex('0x0.0000000000001p-1022'), ValueError),
    ("g", float.fromhex('0x0.fffffffffffffp-1022'), ValueError),
    ("C", 2**53 + 1, ValueError), ("g", 2**53 + 1, ValueError),
    ("C", 10**400, ValueError), ("g", 10**400, ValueError),
])
def test_dense_domain_and_binary64_conversion_validation(name, bad, error):
    args = dict(L=3, N=3, C=1.0, g=0.7)
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        with pytest.raises(error):
            api.sector_diagnostic(**args)


def test_resource_screen_order_site_before_binomial_and_no_allocation():
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        site = api.sector_diagnostic(13, 0)
        huge_site = api.sector_diagnostic(10**80, 10**120)
    for report, args in ((site, (13, 0, 1.0, 0.7)),
                         (huge_site, (10**80, 10**120, 1.0, 0.7))):
        assert_dense(report, args)
        assert report["status"] == UNAVAILABLE and report["reason_code"] == "site_cap"
        assert report["dimension"] is None and report["dimension_exceeds_cap"] is None
        assert report["spectrum"] is report["comparison"] is None


def test_capped_dimension_sentinel_not_misreported_and_exact_bounds_survive():
    calls = []
    original = inherited.capped_binomial

    def capped(n, k, cap=512):
        calls.append((n, k, cap))
        return original(n, k, cap)

    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True, capped=False)
        patch_aliases(patch, original, capped)
        report = api.sector_diagnostic(12, 12)
        huge = api.sector_diagnostic(3, 10**100)
    assert calls == [(23, 12, 512), (10**100 + 2, 10**100, 512)]
    for row, args in ((report, (12, 12, 1.0, 0.7)), (huge, (3, 10**100, 1.0, 0.7))):
        assert_dense(row, args)
        assert row["status"] == UNAVAILABLE and row["reason_code"] == "dimension_cap"
        assert row["dimension"] is None and row["dimension_exceeds_cap"] is True
        assert row["spectrum"] is row["comparison"] is None
        assert row["dimension_cap"] == 512


@pytest.mark.parametrize("C,g", [(0.5, 0.0), (2.0, 40.0), (1, 40), (1.0, -0.0),
                                  (1.0, sys.float_info.min)])
def test_structural_vacuum_accepts_closed_dense_domain_without_model(C, g):
    calls = []
    original = inherited.capped_binomial

    def capped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True, capped=False)
        patch_aliases(patch, original, capped)
        report = api.sector_diagnostic(12, 0, C, g)
    assert len(calls) == 1
    assert_dense(report, (12, 0, C, g))
    assert report["status"] == VACUUM and report["dimension"] == 1


def synthetic_spectrum(dimension=10, energy=-3.0):
    """A test-only report stub: never a physical or synthetic eigensolve."""
    return dict(values=np.full(dimension, energy, dtype=np.float64),
                ground_energy=float(energy), dimension=dimension, scale=100.0,
                tolerance=256 * float(EPSILON) * dimension,
                orthogonality_residual=0.0, eigenpair_residual=0.0, certified=False)


def install_owned_stubs(patch, assembly=None, screening=None, comparison=None):
    """Replace all physical builders and count attempted owned calls first."""
    calls = {"model": [], "screen": [], "comparison": []}
    objects = []
    original_model, original_screen = inherited.all_number_model, api._screened_spectrum
    original_compare = api._energy_comparison
    model_alias_names = [name for name, value in vars(api).items() if value is original_model]
    real_comb = math.comb
    block_physics(patch, capped=False)

    def model(L, N, C=1.0, g=0.7):
        calls["model"].append((L, N, C, g))
        if assembly is not None:
            return assembly(L, N, C, g)
        result = SimpleNamespace(H=np.zeros((real_comb(L + N - 1, N),) * 2),
                                 L=L, N=N, C=C, g=g)
        objects.append(result)
        return result

    def screen(H):
        calls["screen"].append(H)
        if screening is not None:
            return screening(H)
        return synthetic_spectrum(len(H), -3.0)

    def compare(*args, **kwargs):
        calls["comparison"].append((args, kwargs))
        if comparison is not None:
            return comparison(*args, **kwargs)
        return original_compare(*args, **kwargs)

    # block_physics replaced the model identity already; recover only the owned
    # provider and its original API aliases, not old fixed-number/basis seams.
    patch.setattr(inherited, "all_number_model", model)
    for name in model_alias_names:
        patch.setattr(api, name, model)
    patch_aliases(patch, original_screen, screen)
    patch_aliases(patch, original_compare, compare)
    return calls, objects


def throwing(exception):
    def fail(*args, **kwargs):
        raise exception
    return fail


NUMERICAL_TYPES = (current.NumericalUnavailable, FloatingPointError, OverflowError, np.linalg.LinAlgError)
PROGRAMMING_TYPES = (TypeError, ValueError, RuntimeError)


@pytest.mark.parametrize("stage,code", [("assembly", "assembly_failure"),
                                        ("screening", "spectrum_failure"),
                                        ("comparison", "comparison_failure")])
@pytest.mark.parametrize("exception_type", NUMERICAL_TYPES)
def test_owned_numerical_boundaries_preserve_partial_reports_and_attempts(stage, code, exception_type):
    exception = exception_type("frozen numerical sentinel")
    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, **{stage: throwing(exception)})
        report = api.sector_diagnostic(3, 3)
    assert_dense(report, (3, 3, 1.0, 0.7))
    assert report["status"] == UNAVAILABLE and report["reason_code"] == code
    assert report["reason"] == str(exception)
    assert report["dimension"] == 10 and report["dimension_exceeds_cap"] is False
    assert len(calls["model"]) == 1
    assert len(calls["screen"]) == (0 if stage == "assembly" else 1)
    assert len(calls["comparison"]) == (1 if stage == "comparison" else 0)
    assert report["comparison"] is None
    if stage == "comparison":
        assert report["spectrum"] == {key: value for key, value in synthetic_spectrum().items() if key != "values"}
    else:
        assert report["spectrum"] is None


@pytest.mark.parametrize("stage", ("assembly", "screening", "comparison"))
@pytest.mark.parametrize("exception_type", PROGRAMMING_TYPES)
def test_unexpected_programming_errors_propagate_from_exact_owned_stage(stage, exception_type):
    exception = exception_type("programming sentinel")
    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, **{stage: throwing(exception)})
        with pytest.raises(exception_type) as caught:
            api.sector_diagnostic(3, 3)
    assert caught.value is exception
    assert len(calls["model"]) == 1
    assert len(calls["screen"]) == (0 if stage == "assembly" else 1)
    assert len(calls["comparison"]) == (1 if stage == "comparison" else 0)


@pytest.mark.parametrize("message,adapt", [
    ("numerically unresolved arithmetic or singular solve", True),
    ("numerically unresolved injected inherited failure", True),
    ("numerically unresolved ", True),
    ("numerically unresolved", False),
    (" numerically unresolved arithmetic or singular solve", False),
    ("Numerically unresolved arithmetic or singular solve", False),
    ("prefix numerically unresolved arithmetic or singular solve", False),
    ("H fails the heuristic Hermiticity screen", False),
])
def test_assembly_valueerror_adapter_uses_exact_prefix_only(message, adapt):
    exception = ValueError(message)
    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, assembly=throwing(exception))
        if adapt:
            report = api.sector_diagnostic(3, 3)
        else:
            with pytest.raises(ValueError) as caught:
                api.sector_diagnostic(3, 3)
            assert caught.value is exception
    assert len(calls["model"]) == 1 and not calls["screen"] and not calls["comparison"]
    if adapt:
        assert_dense(report, (3, 3, 1.0, 0.7))
        assert report["reason"] == message and report["reason_code"] == "assembly_failure"
        assert report["spectrum"] is report["comparison"] is None


@pytest.mark.parametrize("message,adapt", [
    ("H fails the heuristic Hermiticity screen", True),
    ("H fails the heuristic Hermiticity screen ", False),
    (" H fails the heuristic Hermiticity screen", False),
    ("H fails the heuristic Hermiticity screen: details", False),
    ("numerically unresolved arithmetic or singular solve", False),
])
def test_owned_spectrum_valueerror_adapter_matches_whole_text_only(message, adapt):
    exception = ValueError(message)
    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, screening=throwing(exception))
        if adapt:
            report = api.sector_diagnostic(3, 3)
        else:
            with pytest.raises(ValueError) as caught:
                api.sector_diagnostic(3, 3)
            assert caught.value is exception
    assert len(calls["model"]) == len(calls["screen"]) == 1 and not calls["comparison"]
    if adapt:
        assert_dense(report, (3, 3, 1.0, 0.7))
        assert report["reason"] == message and report["reason_code"] == "hermiticity_screen_failed"
        assert report["spectrum"] is report["comparison"] is None


@pytest.mark.parametrize("message", ["H fails the heuristic Hermiticity screen",
                                      "numerically unresolved arithmetic or singular solve"])
def test_no_valueerror_adapter_applies_to_comparison(message):
    exception = ValueError(message)
    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, comparison=throwing(exception))
        with pytest.raises(ValueError) as caught:
            api.sector_diagnostic(3, 3)
    assert caught.value is exception
    assert len(calls["model"]) == len(calls["screen"]) == len(calls["comparison"]) == 1


@pytest.mark.parametrize("exception_type", NUMERICAL_TYPES + PROGRAMMING_TYPES)
def test_exact_bounds_errors_stay_outside_numerical_boundary(exception_type):
    exception = exception_type("exact construction sentinel")
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        patch.setattr(api, "_exact_bounds", throwing(exception))
        with pytest.raises(exception_type) as caught:
            api.sector_diagnostic(3, 3)
    assert caught.value is exception


def test_dense_owned_reports_uncached_detached_and_mismatch_reason():
    with pytest.MonkeyPatch.context() as patch:
        calls, objects = install_owned_stubs(patch)
        first = api.sector_diagnostic(3, 3)
        second = api.sector_diagnostic(3, 3)
        first["analytic_bounds"]["trial_upper_bounds"]["uniform_condensate"]["numerator"] = "corrupt"
        first["spectrum"]["ground_energy"] = 123.0
        third = api.sector_diagnostic(3, 3)
    assert len(calls["model"]) == len(calls["screen"]) == len(calls["comparison"]) == 3
    assert len({id(model) for model in objects}) == 3
    assert second == third and second is not third
    assert_dense(second, (3, 3, 1.0, 0.7))
    assert second["status"] == MISMATCH
    assert second["reason"] == REASONS[MISMATCH]


def test_nonvacuum_zero_and_degenerate_ground_are_not_rejected():
    # Synthetic boundary stubs deliberately report a zero/degenerate spectrum;
    # the exact bound comparison, not uniqueness/gap/nonzero gates, decides.
    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, screening=lambda H: synthetic_spectrum(len(H), 0.0))
        zero = api.sector_diagnostic(3, 1, 1, 40)
    assert len(calls["model"]) == len(calls["screen"]) == len(calls["comparison"]) == 1
    assert_dense(zero, (3, 1, 1, 40))
    assert zero["status"] == MISMATCH and zero["reason_code"] == "energy_bound_mismatch"
    rational(zero["analytic_bounds"]["structural_energy"], -2)
    assert zero["spectrum"]["ground_energy"] == 0.0


VALID_MATRICES = ("zero1", "zero3", "degenerate2", "diagonal2", "complex2",
                  "subnormal_real1", "subnormal_imag2")
INVALID_MATRICES = ("list", "tuple", "protocol", "subclass", "int64", "float32",
                    "complex64", "bool", "object", "nonnative64", "scalar", "vector",
                    "rectangular", "empty", "oversized513", "nan_real", "inf_real",
                    "nan_imag", "inf_imag")
MALFORMED_OUTPUTS = ("values_list", "values_short", "values_matrix", "values_complex",
                     "values_bool", "values_nan", "values_inf", "vectors_list",
                     "vectors_short", "vectors_bool", "vectors_nan", "vectors_inf",
                     "not_orthogonal", "wrong_eigenpairs", "descending", "component_loss")


def synthetic_matrix(name):
    tiny = float.fromhex('0x0.0000000000001p-1022')
    if name == "zero1":
        return np.zeros((1, 1), dtype=np.float64), [0.0]
    if name == "zero3":
        return np.zeros((3, 3), dtype=np.float64), [0.0] * 3
    if name == "degenerate2":
        return np.eye(2, dtype=np.float64), [1.0, 1.0]
    if name == "diagonal2":
        return np.diag(np.array([-1.0, 2.0], dtype=np.float64)), [-1.0, 2.0]
    if name == "complex2":
        return np.array([[0, 1j], [-1j, 0]], dtype=np.complex128), [-1.0, 1.0]
    if name == "subnormal_real1":
        return np.array([[tiny]], dtype=np.float64), [tiny]
    if name == "subnormal_imag2":
        return np.array([[0, complex(0, tiny)], [complex(0, -tiny), 0]], dtype=np.complex128), [-tiny, tiny]
    raise AssertionError("unlisted synthetic matrix")


@pytest.mark.parametrize("name", VALID_MATRICES)
def test_screened_spectrum_valid_native_matrices_one_attempt_no_repair(name):
    H, expected = synthetic_matrix(name)
    before = H.copy()
    H.flags.writeable = False
    calls = {"screen": [], "eigh": []}
    original_screen, original_eigh = encoding._screened_eigensystem, np.linalg.eigh

    def screen(matrix):
        calls["screen"].append(matrix.copy())
        assert matrix is not H and not np.shares_memory(matrix, H)
        assert matrix.dtype == H.dtype
        np.testing.assert_array_equal(matrix, before)
        return original_screen(matrix)

    def eigh(matrix, *args, **kwargs):
        calls["eigh"].append(matrix.copy())
        assert len(calls["eigh"]) <= 1
        return original_eigh(matrix, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, original_screen, screen)
        patch_aliases(patch, original_eigh, eigh)
        for other in (np.linalg.eigvalsh, np.linalg.eig, np.linalg.eigvals, np.linalg.svd):
            patch_aliases(patch, other, deny)
        result = api._screened_spectrum(H)
    assert len(calls["screen"]) == len(calls["eigh"]) == 1
    assert set(result) == SPECTRUM_KEYS | {"values"}
    assert type(result["values"]) is np.ndarray
    assert result["values"].dtype == np.dtype(np.float64)
    assert not np.shares_memory(result["values"], H)
    np.testing.assert_allclose(result["values"], expected, rtol=2e-14, atol=2e-14)
    if name in ("zero1", "zero3", "subnormal_real1", "subnormal_imag2"):
        np.testing.assert_array_equal(result["values"], expected)
    assert result["ground_energy"] == float(result["values"][0])
    assert type(result["ground_energy"]) is float
    assert result["dimension"] == len(H)
    assert result["scale"] == pytest.approx(max(1.0, frobenius(H)), rel=2e-15, abs=0)
    assert result["tolerance"] == 256 * float(EPSILON) * len(H)
    assert result["certified"] is False
    assert result["orthogonality_residual"] <= result["tolerance"]
    assert result["eigenpair_residual"] / result["scale"] <= result["tolerance"]
    np.testing.assert_array_equal(H, before)
    assert H.flags.writeable is False


def test_screened_spectrum_owned_values_detached_across_calls():
    H, _ = synthetic_matrix("diagonal2")
    original = encoding._screened_eigensystem
    outputs = []

    def capture(matrix):
        result = original(matrix)
        outputs.append(result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, original, capture)
        first = api._screened_spectrum(H)
        second = api._screened_spectrum(H)
    assert len(outputs) == 2
    for result, provider in zip((first, second), outputs):
        assert not np.shares_memory(result["values"], provider["values"])
    first["values"][0] = 100
    outputs[1]["values"][:] = 200
    H[:] = 300
    np.testing.assert_allclose(second["values"], [-1, 2], atol=2e-14, rtol=2e-14)
    assert second["ground_energy"] == pytest.approx(-1.0)


def invalid_matrix(name):
    if name == "list":
        return [[1.0]], TypeError
    if name == "tuple":
        return ((1.0,),), TypeError
    if name == "protocol":
        return ProtocolBomb(), TypeError
    if name == "subclass":
        return np.eye(1).view(ArraySubclass), TypeError
    if name in ("int64", "float32", "complex64", "bool", "object"):
        return np.ones((1, 1), dtype=name), TypeError
    if name == "nonnative64":
        dtype = np.dtype(">f8" if sys.byteorder == "little" else "<f8")
        return np.ones((1, 1), dtype=dtype), TypeError
    if name == "scalar":
        return np.array(1.0), ValueError
    if name == "vector":
        return np.ones(2), ValueError
    if name == "rectangular":
        return np.ones((2, 3)), ValueError
    if name == "empty":
        return np.empty((0, 0)), ValueError
    if name == "oversized513":
        return np.zeros((513, 513)), ValueError
    if name in ("nan_real", "inf_real"):
        return np.array([[float("nan") if name == "nan_real" else float("inf")]]), ValueError
    if name in ("nan_imag", "inf_imag"):
        return np.array([[complex(0, float("nan") if name == "nan_imag" else float("inf"))]]), ValueError
    raise AssertionError("unlisted invalid matrix")


@pytest.mark.parametrize("name", INVALID_MATRICES)
def test_spectrum_input_validation_precedes_copy_and_solver(name):
    H, error = invalid_matrix(name)
    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, encoding._screened_eigensystem, deny)
        patch_aliases(patch, np.linalg.eigh, deny)
        # Invalid shape/dtype must not get a conversion/allocation repair.
        if name not in ("nan_real", "inf_real", "nan_imag", "inf_imag"):
            patch_aliases(patch, np.array, deny)
            patch_aliases(patch, np.asarray, deny)
        with pytest.raises(error):
            api._screened_spectrum(H)


def test_direct_spectrum_hermiticity_valueerror_not_adapted_or_symmetrized():
    H = np.array([[0.0, 1.0], [0.0, 0.0]])
    before = H.copy()
    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, np.linalg.eigh, deny)
        with pytest.raises(ValueError, match="^H fails the heuristic Hermiticity screen$"):
            api._screened_spectrum(H)
    np.testing.assert_array_equal(H, before)


@pytest.mark.parametrize("part", ("real_loss2", "imag_loss2"))
def test_componentwise_scaling_loss_rejected_before_inherited_screen(part):
    tiny = float.fromhex('0x0.0000000000001p-1022')
    if part == "real_loss2":
        H = np.array([[1e100, tiny], [tiny, 0.0]], dtype=np.float64)
    else:
        H = np.array([[1e100, complex(0, tiny)], [complex(0, -tiny), 0.0]], dtype=np.complex128)
    before = H.copy()
    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, encoding._screened_eigensystem, deny)
        patch_aliases(patch, np.linalg.eigh, deny)
        with pytest.raises(current.NumericalUnavailable):
            api._screened_spectrum(H)
    np.testing.assert_array_equal(H, before)


@pytest.mark.parametrize("name", MALFORMED_OUTPUTS)
def test_malformed_solver_outputs_are_not_repaired_or_silently_sorted(name):
    H, _ = synthetic_matrix("diagonal2")
    calls = []

    def malformed(matrix, *args, **kwargs):
        calls.append(matrix.copy())
        assert len(calls) == 1
        values = np.diag(matrix).copy()
        vectors = np.eye(2)
        if name == "values_list":
            values = values.tolist()
        elif name == "values_short":
            values = values[:1]
        elif name == "values_matrix":
            values = values.reshape((2, 1))
        elif name == "values_complex":
            values = values.astype(np.complex128)
        elif name == "values_bool":
            values = values.astype(bool)
        elif name == "values_nan":
            values[0] = float("nan")
        elif name == "values_inf":
            values[0] = float("inf")
        elif name == "vectors_list":
            vectors = vectors.tolist()
        elif name == "vectors_short":
            vectors = vectors[:, :1]
        elif name == "vectors_bool":
            vectors = vectors.astype(bool)
        elif name == "vectors_nan":
            vectors[0, 0] = float("nan")
        elif name == "vectors_inf":
            vectors[0, 0] = float("inf")
        elif name == "not_orthogonal":
            vectors *= 2
        elif name == "wrong_eigenpairs":
            values += 0.1
        elif name == "descending":
            values = values[::-1].copy()
            vectors = vectors[:, ::-1].copy()
        elif name == "component_loss":
            values = values.astype(np.longdouble)
            values[0] = np.longdouble(float.fromhex('0x0.0000000000001p-1022')) / np.longdouble(2)
            if values[0] == 0:
                pytest.skip("platform longdouble has no wider subnormal component")
        else:
            raise AssertionError("unlisted malformed solver output")
        return values, vectors

    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, np.linalg.eigh, malformed)
        for other in (np.linalg.eigvalsh, np.linalg.eig, np.linalg.eigvals, np.linalg.svd):
            patch_aliases(patch, other, deny)
        with pytest.raises(current.NumericalUnavailable):
            api._screened_spectrum(H)
    assert len(calls) == 1


@pytest.mark.parametrize("exception_type", NUMERICAL_TYPES + PROGRAMMING_TYPES)
@pytest.mark.parametrize("boundary", ("eigh", "inherited_screen"))
def test_spectrum_exception_adapters_only_expected_numerical_classes(exception_type, boundary):
    H, _ = synthetic_matrix("diagonal2")
    exception = exception_type("synthetic solver exception")
    attempts = []

    def failure(*args, **kwargs):
        attempts.append(1)
        raise exception

    with pytest.MonkeyPatch.context() as patch:
        target = np.linalg.eigh if boundary == "eigh" else encoding._screened_eigensystem
        patch_aliases(patch, target, failure)
        if boundary == "inherited_screen":
            patch_aliases(patch, np.linalg.eigh, deny)
        expected = current.NumericalUnavailable if exception_type in NUMERICAL_TYPES else exception_type
        with pytest.raises(expected) as caught:
            api._screened_spectrum(H)
    assert attempts == [1]
    if exception_type in PROGRAMMING_TYPES:
        assert caught.value is exception


def test_eigenvalue_ordering_uses_adjacent_comparisons_not_difference_or_sort():
    H = np.diag(np.array([-1e308, 1e308], dtype=np.float64))
    calls = []

    def descending(matrix):
        calls.append(matrix)
        return dict(values=np.array([1e308, -1e308]), vectors=np.eye(2)[:, ::-1],
                    scale=math.sqrt(2) * 1e308, tolerance=512 * float(EPSILON),
                    orthogonality_residual=0.0, eigenpair_residual=0.0)

    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, encoding._screened_eigensystem, descending)
        patch_aliases(patch, np.linalg.eigh, deny)
        patch_aliases(patch, np.diff, deny)
        patch_aliases(patch, np.sort, deny)
        with np.errstate(over="raise", invalid="raise"):
            with pytest.raises(current.NumericalUnavailable):
                api._screened_spectrum(H)
    assert len(calls) == 1


@pytest.fixture(scope="module")
def synthetic_aggregation_baseline():
    """JSON-only successful diagnostics, independent of physical availability."""
    def successful_case(L, N, C=1.0, g=0.7):
        exact = exact_oracle(L, N, Fraction.from_float(C), Fraction.from_float(g))
        energy = float((exact["lower_bound"] + exact["best_upper_bound"]) / 2)
        d = {3: 10, 4: 35, 5: 126}[L]
        spectrum = dict(ground_energy=energy, dimension=d, scale=1000.0,
                        tolerance=256 * float(EPSILON) * d,
                        orthogonality_residual=0.0, eigenpair_residual=0.0, certified=False)
        comparison = api._energy_comparison(energy, 1000.0, d, exact["lower_bound"],
                                             exact["best_upper_bound"], Fraction.from_float(C))
        assert comparison["status"] == AVAILABLE
        return dict(model_id=MODEL_ID, status=AVAILABLE, reason=None, reason_code=None,
                    parameter_semantics="binary64_lifted_exactly", L=L, N=N, C=C, g=g,
                    dimension=d, dimension_cap=512, dimension_exceeds_cap=False,
                    analytic_bounds=api.stability_bounds(
                        L, N, Fraction.from_float(C), Fraction.from_float(g))["bounds"],
                    spectrum=spectrum, comparison=comparison,
                    scope="Synthetic aggregation fixture, not a physical diagnostic.")

    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        patch_aliases(patch, api.sector_diagnostic, successful_case)
        baseline = api.demonstration_report()
    assert_demo_metadata(baseline)
    return baseline


@pytest.mark.parametrize("scenario", ("all_available", "one_unavailable", "all_unavailable",
                                       "one_mismatch", "mismatch_and_unavailable", "one_vacuum"))
def test_demonstration_all_status_counts_failure_priority_and_no_omission(synthetic_aggregation_baseline, scenario):
    baseline = synthetic_aggregation_baseline
    rows = copy.deepcopy(baseline["dense_cases"])
    if scenario in ("one_unavailable", "all_unavailable", "mismatch_and_unavailable"):
        positions = range(9) if scenario == "all_unavailable" else (8,)
        for i in positions:
            rows[i].update(status=UNAVAILABLE, reason="synthetic assembly unavailable",
                           reason_code="assembly_failure", spectrum=None, comparison=None)
    if scenario in ("one_mismatch", "mismatch_and_unavailable"):
        row = rows[0]
        row["spectrum"]["ground_energy"] = 100.0
        bounds = exact_oracle(*DENSE_GRID[0])
        row["comparison"] = api._energy_comparison(
            100.0, row["spectrum"]["scale"], row["dimension"],
            bounds["lower_bound"], bounds["best_upper_bound"], 1)
        row.update(status=MISMATCH, reason=REASONS[MISMATCH], reason_code="energy_bound_mismatch")
    if scenario == "one_vacuum":
        # Aggregation-only injection. This is NOT a physical L=N vacuum claim;
        # dense case field semantics are checked separately on actual reports.
        rows[0].update(status=VACUUM, reason=REASONS[VACUUM], reason_code="structural_vacuum")
    seen = []

    original = api.sector_diagnostic

    def case(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments[name] for name in ("L", "N", "C", "g"))
        seen.append(key)
        assert len(seen) <= 9 and key == DENSE_GRID[len(seen) - 1]
        return copy.deepcopy(rows[len(seen) - 1])

    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        patch_aliases(patch, original, case)
        report = api.demonstration_report()
    assert seen == list(DENSE_GRID)
    assert report["dense_cases"] == rows
    if scenario != "one_vacuum":
        assert_demo_metadata(report)
    else:
        counts = {AVAILABLE: 8, MISMATCH: 0, UNAVAILABLE: 0, VACUUM: 1}
        assert report["summary"]["dense_status_counts"] == counts
        assert report["summary"]["scalar_case_count"] == 18
        assert report["summary"]["dense_case_count"] == 9
        assert report["status"] == AVAILABLE and report["reason"] is None
        assert len(report["scalar_cases"]) == 18
    expected = (MISMATCH if "mismatch" in scenario else UNAVAILABLE
                if "unavailable" in scenario else AVAILABLE)
    assert report["status"] == expected and report["reason"] == AGGREGATE_REASONS[expected]
    assert report["unit_filling_exclusion"] == baseline["unit_filling_exclusion"]
    assert report["free_classification"] == baseline["free_classification"]
    assert report["controls"] is not baseline["controls"]
    assert report["controls"]["scalar_densities"] is not baseline["controls"]["scalar_densities"]
    assert report["unit_filling_exclusion"] is not baseline["unit_filling_exclusion"]
    assert report["unit_filling_exclusion"]["witnesses"] is not baseline["unit_filling_exclusion"]["witnesses"]
    assert report["free_classification"] is not baseline["free_classification"]


@pytest.mark.parametrize("failure_stage", ("assembly", "screen", "solver"))
def test_failed_demo_counts_attempts_not_successes_and_never_retries(failure_stage):
    """All nine case failures preserved; no real model or eigensolver invoked."""
    calls = {"model": [], "screen": [], "eigh": []}
    original_model, original_screen = inherited.all_number_model, api._screened_spectrum
    original_eigh = np.linalg.eigh
    original_comb = math.comb
    model_aliases = [name for name, value in vars(api).items() if value is original_model]

    def model(L, N, C=1.0, g=0.7):
        key = (L, N, C, g)
        calls["model"].append(key)
        assert len(calls["model"]) <= 9 and key == DENSE_GRID[len(calls["model"]) - 1]
        if failure_stage == "assembly":
            raise OverflowError("attempted assembly failure")
        return SimpleNamespace(H=np.zeros((original_comb(L + N - 1, N),) * 2))

    def screen(H):
        calls["screen"].append(H)
        assert len(calls["screen"]) <= 9
        if failure_stage == "screen":
            raise current.NumericalUnavailable("attempted screen failure")
        return original_screen(H)

    def solver(*args, **kwargs):
        calls["eigh"].append(1)
        assert len(calls["eigh"]) <= 9
        raise np.linalg.LinAlgError("attempted solver failure")

    with pytest.MonkeyPatch.context() as patch:
        # Keep the genuine spectrum seam for the solver-failure case, but forbid
        # every physical model and all auxiliary eigensolver paths.
        for target in (occupation.fixed_number_model, occupation._occupations,
                       occupation.binary_operators, inherited.sector_ground_diagnostic,
                       np.linalg.eigvalsh, np.linalg.eig, np.linalg.eigvals, np.linalg.svd):
            patch_aliases(patch, target, deny)
        patch.setattr(inherited, "all_number_model", model)
        for name in model_aliases:
            patch.setattr(api, name, model)
        patch_aliases(patch, original_screen, screen)
        patch_aliases(patch, original_eigh, solver)
        report = api.demonstration_report()
    assert_demo_metadata(report)
    assert report["status"] == UNAVAILABLE
    assert calls["model"] == list(DENSE_GRID)
    assert len(calls["screen"]) == (0 if failure_stage == "assembly" else 9)
    assert len(calls["eigh"]) == (9 if failure_stage == "solver" else 0)
    expected_code = "assembly_failure" if failure_stage == "assembly" else "spectrum_failure"
    assert all(row["reason_code"] == expected_code for row in report["dense_cases"])
    assert all(row["spectrum"] is row["comparison"] is None for row in report["dense_cases"])
    assert report["summary"]["dense_status_counts"] == {
        AVAILABLE: 0, MISMATCH: 0, UNAVAILABLE: 9, VACUUM: 0}


def test_required_inherited_rational_seams_reused_without_public_float_bridge():
    seen = {"bounds": [], "minimum": [], "record": []}
    originals = (inherited._bounds, inherited.balanced_interaction_minimum, inherited.rational_record)

    def bounds(L, N, C, g):
        seen["bounds"].append((L, N, C, g))
        assert type(C) is Fraction and type(g) is Fraction
        return originals[0](L, N, C, g)

    def minimum(L, N):
        seen["minimum"].append((L, N))
        return originals[1](L, N)

    def record(value):
        seen["record"].append(value)
        return originals[2](value)

    args = (1000, 500, Fraction(1, 3), Fraction(7, 10))
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch, arrays=True)
        patch_aliases(patch, inherited.sector_energy_bounds, deny)
        for original, replacement in zip(originals, (bounds, minimum, record)):
            patch_aliases(patch, original, replacement)
        report = api.stability_bounds(*args)
    assert_scalar(report, args)
    assert seen["bounds"] and all(row == args for row in seen["bounds"])
    assert seen["minimum"] and all(row == args[:2] for row in seen["minimum"])
    assert seen["record"]
    assert Fraction(7, 10) in seen["record"]


def test_dense_cap_near_boundary_without_building_additional_physical_models():
    # C(32,30)=496 passes; C(33,31)=528 exceeds. Failed assembly is deliberate.
    attempts = []

    def assembly(L, N, C, g):
        attempts.append((L, N, C, g))
        raise OverflowError("intentional boundary assembly stop")

    with pytest.MonkeyPatch.context() as patch:
        calls, _ = install_owned_stubs(patch, assembly=assembly)
        inside = api.sector_diagnostic(3, 30)
        outside = api.sector_diagnostic(3, 31)
    assert attempts == [(3, 30, 1.0, 0.7)]
    assert calls["model"] == attempts and not calls["screen"]
    assert_dense(inside, (3, 30, 1.0, 0.7))
    assert inside["dimension"] == 496 and inside["reason_code"] == "assembly_failure"
    assert_dense(outside, (3, 31, 1.0, 0.7))
    assert outside["dimension"] is None and outside["reason_code"] == "dimension_cap"
    assert outside["dimension_exceeds_cap"] is True


# This bootstrap is intentionally instrumentation, not a test of the script's
# own sys.path bootstrap. The coordinator's separate unwrapped absolute CLI
# gate tests that. Each wrapped mode here has its own independent nine budget.
CLI_BOOTSTRAP = r'''
import inspect
import json
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve()
mode = sys.argv[2:]
assert script.is_absolute()
assert not list(Path.cwd().iterdir())
sys.path.insert(0, str(script.parents[1]))  # Instrumentation imports only.
import numpy as np
from bpr import substrate_extensive_stability as api
from bpr import substrate_vacuum_selection as inherited
from bpr import substrate_fermionization as occupation
from bpr import substrate_gauge_encoding as encoding

modules = (api, inherited, occupation, encoding, np, np.linalg)
def patch(original, replacement):
    changed = 0
    for module in modules:
        for name, value in tuple(vars(module).items()):
            if value is original:
                setattr(module, name, replacement)
                changed += 1
    assert changed

expected = [(L, L, 1.0, g) for L in (3, 4, 5) for g in (0.0, 0.7, 40.0)]
calls = {'model': [], 'screen': [], 'eigh': [], 'demo': []}
models = {}
active_screen = None
original_model = inherited.all_number_model
original_screen = api._screened_spectrum
original_eigh = np.linalg.eigh
original_demo = api.demonstration_report

def model(*args, **kwargs):
    bound = inspect.signature(original_model).bind(*args, **kwargs)
    bound.apply_defaults()
    key = tuple(bound.arguments[name] for name in ('L', 'N', 'C', 'g'))
    calls['model'].append(key)
    assert len(calls['model']) <= 9
    assert key == expected[len(calls['model']) - 1]
    result = original_model(*args, **kwargs)
    models[key] = result
    return result

def screen(H):
    global active_screen
    key = calls['model'][-1]
    assert key in models and key not in calls['screen']
    assert H is models[key].H or np.array_equal(H, models[key].H)
    calls['screen'].append(key)
    active_screen = key
    try:
        return original_screen(H)
    finally:
        active_screen = None

def eigh(H, *args, **kwargs):
    key = active_screen
    assert key in expected and key in calls['screen'] and key not in calls['eigh']
    calls['eigh'].append(key)
    return original_eigh(H, *args, **kwargs)

def forbidden(*args, **kwargs):
    raise AssertionError('unlisted eigensolver or old ground-state diagnostic')

def demonstration():
    calls['demo'].append(1)
    assert len(calls['demo']) == 1
    report = original_demo()
    assert len(report['dense_cases']) == 9 and len(report['scalar_cases']) == 18
    assert report['status'] in ('available_heuristic', 'numerical_unavailable'), report
    assert [(r['L'], r['N'], r['C'], r['g']) for r in report['dense_cases']] == expected
    counts = {'available_heuristic': 0, 'diagnostic_mismatch': 0,
              'numerical_unavailable': 0, 'structural_vacuum': 0}
    for row in report['dense_cases']:
        key = (row['L'], row['N'], row['C'], row['g'])
        counts[row['status']] += 1
        assert row['status'] in ('available_heuristic', 'numerical_unavailable'), row
        assert row['dimension'] == {3: 10, 4: 35, 5: 126}[row['L']]
        assert row['dimension_exceeds_cap'] is False
        assert row['analytic_bounds']
        if row['status'] == 'available_heuristic':
            assert row['reason'] is None and row['reason_code'] is None
            assert key in models and key in calls['screen'] and key in calls['eigh']
            assert row['spectrum'] is not None and row['comparison']['consistent'] is True
        else:
            assert isinstance(row['reason'], str) and row['reason']
            assert row['comparison'] is None
            code = row['reason_code']
            if code == 'assembly_failure':
                assert key not in models and key not in calls['screen'] and key not in calls['eigh']
                assert row['spectrum'] is None
            elif code in ('spectrum_failure', 'hermiticity_screen_failed'):
                assert key in models and key in calls['screen']
                assert row['spectrum'] is None
            else:
                assert code == 'comparison_failure'
                assert key in models and key in calls['screen'] and key in calls['eigh']
                assert row['spectrum'] is not None
    assert report['summary']['dense_status_counts'] == counts
    if counts['numerical_unavailable']:
        assert report['status'] == 'numerical_unavailable'
        assert report['reason'] == 'One or more dense diagnostics are numerically unavailable.'
    else:
        assert report['status'] == 'available_heuristic' and report['reason'] is None
        assert len(calls['eigh']) == 9
    json.dumps(report, allow_nan=False)
    return report

patch(original_model, model)
patch(original_screen, screen)
patch(original_eigh, eigh)
patch(original_demo, demonstration)
for other in (np.linalg.eigvalsh, np.linalg.eig, np.linalg.eigvals,
              np.linalg.svd, inherited.sector_ground_diagnostic):
    patch(other, forbidden)
sys.argv = [str(script)] + mode
try:
    runpy.run_path(str(script), run_name='__main__')
except SystemExit as exc:
    assert exc.code in (None, 0)
assert calls['model'] == expected
assert len(calls['screen']) == len(set(calls['screen'])) <= 9
assert len(calls['eigh']) == len(set(calls['eigh'])) <= 9
assert set(calls['eigh']) <= set(calls['screen']) <= set(models)
assert calls['demo'] == [1]
assert not list(Path.cwd().iterdir())
'''


@pytest.mark.parametrize("mode", ("text", "json"))
def test_wrapped_absolute_cli_stdout_only_isolation_and_separate_nine_budget(tmp_path, mode):
    """Separate direct-script bootstrap verification belongs to the outer gate."""
    assert DEMO.is_absolute() and DEMO.is_file()
    cwd = tmp_path / mode
    cwd.mkdir()
    assert list(cwd.iterdir()) == []
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, "-B", "-c", CLI_BOOTSTRAP, str(DEMO)]
    if mode == "json":
        command.append("--json")
    completed = subprocess.run(command, cwd=str(cwd), env=env, capture_output=True,
                               text=True, timeout=180, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.strip()
    assert list(cwd.iterdir()) == []
    if mode == "json":
        # loads rejects any non-JSON prefix/suffix, proving no budget/debug chatter.
        report = json.loads(completed.stdout)
        assert_demo_metadata(report)
        assert report["status"] != MISMATCH, report
    else:
        assert not completed.stdout.lstrip().startswith("{")
        assert "stability" in completed.stdout.lower()
