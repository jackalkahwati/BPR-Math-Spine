"""Exact neutral number-sector bounds for the stipulated finite Bose ring.

H = g D + V is unshifted, C > 0, g >= 0, L >= 3 and neutrality means
N = 0 (mod q), q >= 2. Accepted normal binary64 parameters are converted
immediately to exact Fractions. Only these bounds, never eigensolver output,
enter sector selection. Energetic preference does not prepare a population:
[H, Nhat] = 0 and every number-sector probability is conserved.
"""
from fractions import Fraction
from math import isfinite
import sys

import numpy as np

from bpr.substrate_fermionization import (
    FixedNumberModel, MAX_DENSE_DIM, MAX_SITES, _EPS, _guard, _integer,
    _mul, _normal, _occupations, _real, _scale, fixed_number_model,
)

MODEL_ID = "conditional-substrate-vacuum-selection-v1"
MAX_CANDIDATES = 256
MAX_NUMERICAL_SECTORS = 4
FROZEN_CONTROLS = ((5, 5, 1.0, 40.0), (5, 5, 1.0, 0.7),
                   (5, 5, 1.0, 0.0), (3, 4, 1.0, 40.0))
LIMITATIONS = (
    "The quantum Bose-Hubbard Hamiltonian and neutrality constraint are stipulated.",
    "Exact certificates concern finite-L number sectors, not a product-state vacuum or a thermodynamic limit.",
    "Number conservation prevents relaxation or preparation between number sectors in this model.",
    "Numerical eigenvalues and near-ties are diagnostics, never analytic comparison certificates.",
    "Adding 2 C Nhat changes the Hamiltonian for comparisons between sectors.",
    "No physical spacetime vacuum, empirical calibration, masses, mixing or TOE conclusion is derived.",
)


@_guard
def _parameters(L, C, g, q=None):
    L = _integer(L, "L", 3)
    if q is not None:
        q = _integer(q, "q", 2)
    C = Fraction.from_float(_real(C, "C", positive=True))
    g = Fraction.from_float(_real(g, "g", lower=0))
    return L, q, C, g


def _decimal_integer(value):
    """Decimal text without changing Python's global integer-string limit."""
    value = int(value)
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    chunks = []
    while value:
        value, part = divmod(value, 1_000_000_000)
        chunks.append(part)
    return sign + str(chunks[-1]) + "".join(f"{part:09d}" for part in reversed(chunks[:-1]))


def _metadata_integer(value):
    return int(value) if abs(value) <= 2**53 - 1 else _decimal_integer(value)


def rational_record(value):
    """Lossless rational JSON record; unresolved float summaries are null.

    Nonzero subnormal/underflow outputs never masquerade as zero. Numerator
    and denominator are always decimal strings, including ordinary counts.
    """
    value = Fraction(value)
    approximate = None
    try:
        approximation = float(value)
    except OverflowError:
        status = "overflow"
    else:
        if not isfinite(approximation):
            status = "overflow"
        elif value and approximation == 0:
            status = "underflow"
        elif 0 < abs(approximation) < sys.float_info.min:
            status = "subnormal"
        else:
            approximate = approximation
            status = "normal" if value else "exact_zero"
    return {"numerator": _decimal_integer(value.numerator),
            "denominator": _decimal_integer(value.denominator),
            "approximate": approximate, "approximation_status": status}


def _json_safe(value):
    if isinstance(value, Fraction):
        return rational_record(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return _metadata_integer(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def balanced_interaction_minimum(L, N):
    """Exact min sum n_x(n_x-1)/2, obtained by balanced occupations."""
    L = _integer(L, "L", 3)
    N = _integer(N, "N", 0)
    a, r = divmod(N, L)
    return L * a * (a - 1) // 2 + a * r


def _bounds(L, N, C, g):
    minimum = balanced_interaction_minimum(L, N)
    trials = {"balanced_occupation": g * minimum,
              "uniform_condensate": -2 * C * N + g * N * (N - 1) / (2 * L)}
    if N == L:
        # Omega + sqrt(2)*s*pair, s=C/(g+2C), normalized after addition.
        trials["unit_filling_two_state"] = (
            -2 * C**2 * (g + 4 * C) / ((g + 2 * C)**2 + 2 * C**2))
    best = min(trials, key=trials.get)
    return {"lower_bound": g * minimum - 2 * C * N,
            "interaction_minimum": minimum, "trial_upper_bounds": trials,
            "best_upper_bound": trials[best], "best_trial": best}


def sector_energy_bounds(L, N, C=1.0, g=0.7):
    """Exact Fraction lower bound and named variational upper bounds.

    V >= -2 C N. A balanced number state has zero hopping expectation.
    A uniform-mode condensate has <D>=N(N-1)/(2L). The optional unit-filling
    trial is a rational evaluation, not a rounded two-state eigenvalue.
    No neutrality is presumed here; only neutral trials may compare sectors.
    """
    L, _, C, g = _parameters(L, C, g)
    N = _integer(N, "N", 0)
    return _bounds(L, N, C, g)


def _tail(L, q, C, g):
    if not g:
        return {"status": "unbounded_below", "cutoff": None,
                "max_neutral_number": None, "candidate_count": None,
                "first_excluded_neutral": None,
                "reason": "Uniform condensates have E=-2CN for all neutral multiples N."}
    cutoff = L * (1 + 4 * C / g)
    last_multiple = cutoff // q
    return {"status": "finite", "cutoff": cutoff,
            "max_neutral_number": last_multiple * q,
            "candidate_count": last_multiple + 1,
            "first_excluded_neutral": (last_multiple + 1) * q,
            "strict_exclusion": "N > cutoff",
            "reason": "g/2*(N^2/L-N)-2CN > 0 above cutoff; vacuum trial energy is zero."}


def neutral_tail_bound(L, q, C=1.0, g=0.7):
    """Exact rational cutoff and integer counts; equality is not excluded.

    For g=0 no finite cutoff or minimum exists. This low-level API retains
    Fraction/int values; analyze_neutral_sectors serializes them losslessly.
    """
    L, q, C, g = _parameters(L, C, g, q)
    return _tail(L, q, C, g)


def capped_binomial(n, k, cap=MAX_DENSE_DIM):
    """Return binomial(n,k) when <=cap, otherwise the sentinel cap+1.

    Monotone multiplicative accumulation stops as soon as the cap is crossed.
    Thus enormous occupation counts never trigger full dimension computation.
    """
    n = _integer(n, "n", 0)
    k = _integer(k, "k", 0, n)
    cap = _integer(cap, "cap", 1)
    k = min(k, n - k)
    result = 1
    for j in range(1, k + 1):
        result = result * (n - k + j) // j
        if result > cap:
            return cap + 1
    return result


@_guard
def all_number_model(L, N, C=1.0, g=0.7):
    """Full fixed-N bosonic model, including N>L, with preallocation caps.

    N<=L delegates unchanged to the inherited builder. For N>L, P is empty
    structurally, not an assertion about the ground state or hard-core physics.
    """
    L = _integer(L, "L", 3, MAX_SITES)
    N = _integer(N, "N", 0)
    if capped_binomial(L + N - 1, N) > MAX_DENSE_DIM:
        raise ValueError(f"fixed-number dimension exceeds cap {MAX_DENSE_DIM}")
    if N <= L:
        return fixed_number_model(L, N, C, g)
    C = _real(C, "C", positive=True)
    g = _real(g, "g", lower=0)
    _mul(C, 2 * N, "hopping scale")
    _mul(g, N * (N - 1) // 2, "interaction scale")
    basis = tuple(_occupations(L, N))
    index = {state: i for i, state in enumerate(basis)}
    V_unit = np.zeros((len(basis), len(basis)))
    D = np.array([sum(n * (n - 1) // 2 for n in state) for state in basis], dtype=float)
    for col, state in enumerate(basis):
        for x in range(L):
            y = (x + 1) % L
            for target, source in ((x, y), (y, x)):
                if state[source]:
                    moved = list(state)
                    moved[source] -= 1
                    moved[target] += 1
                    V_unit[index[tuple(moved)], col] -= np.sqrt(state[source] * (state[target] + 1))
    V = _scale(V_unit, C, "hopping matrix")
    H = _normal(V + np.diag(_scale(D, g, "interaction diagonal")), "Hamiltonian")
    p = np.array([], dtype=int)
    q = np.arange(len(basis), dtype=int)
    return FixedNumberModel(L, N, C, g, basis, p, q, D, V_unit, V, H,
                            H[np.ix_(p, p)], H[np.ix_(p, q)], H[np.ix_(q, q)])


@_guard
def sector_ground_diagnostic(model):
    """Bounded numerical eigensystem summary, never a sector certificate.

    A scale-relative near-tie is unresolved diagnostic ordering. Arithmetic
    failures raise ValueError for callers to label separately from exclusions.
    """
    d = len(model.basis)
    if model.L > MAX_SITES or d > MAX_DENSE_DIM:
        raise ValueError("numerical model exceeds site/dimension cap")
    result = {"status": "resolved", "certified": False, "N": model.N,
              "dimension": d, "ground_energy": None, "gap": None,
              "multiplicity": None, "hard_core_dimension": len(model.p_indices),
              "ordering": "numerical_only"}
    if model.N == 0:
        result.update(ground_energy=0.0, multiplicity=1, status="structural_vacuum")
        return result
    H = _normal(model.H, "Hamiltonian")
    scale = float(np.max(np.abs(H)))
    if scale == 0:
        raise ValueError("numerically unresolved zero Hamiltonian in nonvacuum sector")
    scaled = _normal(H / scale, "scaled Hamiltonian")
    if np.any((H != 0) & (scaled == 0)):
        raise ValueError("numerically unresolved Hamiltonian scaling underflow")
    values = _normal(np.linalg.eigvalsh(scaled), "eigenvalues")
    tolerance = 128 * _EPS * d * max(1.0, float(np.max(np.abs(values))))
    if abs(values[0]) <= tolerance:
        raise ValueError("numerically unresolved ground energy near zero")
    result["ground_energy"] = _mul(float(values[0]), scale, "ground energy")
    multiplicity = int(np.count_nonzero(values - values[0] <= tolerance))
    result["multiplicity"] = multiplicity
    if multiplicity > 1:
        result.update(status="near_degenerate", ordering="unresolved_near_tie")
    elif d > 1:
        result["gap"] = _mul(float(values[1] - values[0]), scale, "spectral gap")
    return result


def _diagnostics(rows, L, C, g, enabled, budget):
    attempts = 0
    # Analytically surviving candidates receive the bounded diagnostic budget first.
    for row in sorted(rows, key=lambda item: (item["status"] == "excluded", item["N"])):
        N = row["N"]
        if not enabled:
            diagnostic = {"status": "disabled"}
        elif L > MAX_SITES:
            diagnostic = {"status": "site_cap", "max_sites": MAX_SITES}
        elif capped_binomial(L + N - 1, N) > MAX_DENSE_DIM:
            diagnostic = {"status": "dimension_cap", "dimension_exceeds": MAX_DENSE_DIM}
        elif N != 0 and attempts >= budget:
            diagnostic = {"status": "budget_exhausted"}
        else:
            # Vacuum is structural, requiring no diagonalization.
            if N != 0:
                attempts += 1
            try:
                diagnostic = sector_ground_diagnostic(all_number_model(L, N, float(C), float(g)))
            except ValueError as exc:
                diagnostic = {"status": "arithmetic_unresolved", "reason": str(exc),
                              "certified": False}
        row["numerical"] = diagnostic
    return attempts


def analyze_neutral_sectors(L, q, C=1.0, g=0.7, *, numerical=True,
                            candidate_cap=MAX_CANDIDATES,
                            numerical_budget=MAX_NUMERICAL_SECTORS):
    """JSON-safe exact sector analysis with separately bounded diagnostics.

    A sector is removed only if its lower bound STRICTLY exceeds the best
    available neutral trial upper bound. Equality always remains a candidate.
    Resource limits never count as analytic exclusions. Candidate and numerical
    budgets may be reduced but never increased above 256 and 4 respectively.
    """
    L, q, C, g = _parameters(L, C, g, q)
    candidate_cap = _integer(candidate_cap, "candidate_cap", 1, MAX_CANDIDATES)
    numerical_budget = _integer(numerical_budget, "numerical_budget", 0, MAX_NUMERICAL_SECTORS)
    if not isinstance(numerical, (bool, np.bool_)):
        raise TypeError("numerical must be boolean")
    tail = _tail(L, q, C, g)
    report = {"model_id": MODEL_ID, "parameters": {"L": L, "q": q, "C": C, "g": g},
              "hamiltonian": "H = g D + V (unshifted)",
              "neutrality": "N = 0 modulo q (assumed)", "tail": tail,
              "status": None, "selected_sector": None, "surviving_sectors": None,
              "best_trial": None, "sectors": [], "certificate": None,
              "enumeration_complete": False, "number_conserving": True,
              "preparation_mechanism": None,
              "unit_filling_status": "not_established" if L % q == 0 else "not_neutral",
              "numerical_diagonalizations": 0, "numerical_budget": numerical_budget,
              "candidate_cap": candidate_cap,
              "physical_predictions": {"masses": None, "mixing": None},
              "limitations": LIMITATIONS}
    if not g:
        report.update(status="unbounded_below", enumeration_complete=True,
                      certificate={"type": "uniform_condensate_unbounded_sequence",
                                   "energy": "-2 C k q tends to minus infinity as k tends to infinity"})
        if L % q == 0:
            report["unit_filling_status"] = "excluded"
        return _json_safe(report)

    strong = q == L and g >= 4 * C
    if tail["candidate_count"] > candidate_cap and not strong:
        report.update(status="resource_unresolved",
                      resource_reason="candidate_count exceeds enumeration cap")
        report["best_trial"] = {"N": 0, "name": "balanced_occupation", "upper_bound": Fraction(0)}
        if L % q == 0:
            unit = _bounds(L, L, C, g)
            if unit["best_upper_bound"] < 0:
                report["best_trial"] = {"N": L, "name": unit["best_trial"],
                                        "upper_bound": unit["best_upper_bound"]}
        return _json_safe(report)

    # The strong theorem requires at most three sectors, even for enormous L.
    # With a caller-reduced cap it instead certifies directly, without enumeration.
    if strong and tail["candidate_count"] > candidate_cap:
        bounds = _bounds(L, L, C, g)
        report.update(status="selected", selected_sector=L, surviving_sectors=[L],
                      best_trial={"N": L, "name": bounds["best_trial"],
                                  "upper_bound": bounds["best_upper_bound"]},
                      certificate={"type": "q_equals_L_g_at_least_4C",
                                   "proof": "All k>=2 have LB>=0, vacuum E=0, and the N=L two-state trial is strictly negative."},
                      unit_filling_status="selected", enumeration_complete=False)
        return _json_safe(report)

    rows = []
    for k in range(tail["candidate_count"]):
        N = k * q
        rows.append({"N": N, **_bounds(L, N, C, g)})
    best_row = min(rows, key=lambda row: row["best_upper_bound"])
    best_upper = best_row["best_upper_bound"]
    survivors = []
    for row in rows:
        excluded = row["lower_bound"] > best_upper
        row.update(status="excluded" if excluded else "retained",
                   exclusion_reason="lower_bound > best_neutral_trial_upper_bound" if excluded else None)
        if not excluded:
            survivors.append(row["N"])
    report.update(sectors=rows, surviving_sectors=survivors, enumeration_complete=True,
                  status="selected" if len(survivors) == 1 else "bounded_candidates",
                  best_trial={"N": best_row["N"], "name": best_row["best_trial"],
                              "upper_bound": best_upper})
    if len(survivors) == 1:
        report["selected_sector"] = survivors[0]
        report["certificate"] = {
            "type": "q_equals_L_g_at_least_4C" if strong else "strict_rational_bounds",
            "proof": "Every other enumerated sector has LB > best neutral trial UB; all remaining neutral sectors are excluded by the exact tail bound."}
    if L % q == 0:
        report["unit_filling_status"] = (
            "selected" if report["selected_sector"] == L else
            "excluded" if L not in survivors else "not_established")
    report["numerical_diagonalizations"] = _diagnostics(
        rows, L, C, g, bool(numerical), numerical_budget)
    return _json_safe(report)


def shifted_convention_result(L, q, C=1.0, g=0.7):
    """Theorem for the DIFFERENT Hamiltonian H+2 C Nhat, not a repair.

    V+2CNhat = C sum_x (b_x-b_{x+1})^dagger(b_x-b_{x+1}). Its fixed-N
    kernel on the connected ring is the uniform condensate. For N>=2 it has
    positive <D>=N(N-1)/(2L), so no vector is in both nonnegative kernels.
    All nonvacuum neutral sectors have N>=q>=2. For g>0 the shifted tail
    grows, and the empty sector uniquely minimizes energy at zero. For g=0
    infinitely many neutral condensates tie at zero.
    """
    L, q, C, g = _parameters(L, C, g, q)
    return _json_safe({
        "status": "selected" if g else "infinitely_many_tied_sectors",
        "selected_sector": 0 if g else None,
        "parameters": {"L": L, "q": q, "C": C, "g": g},
        "different_hamiltonian": True, "hamiltonian": "H_shifted = H + 2 C Nhat",
        "sector_energy_shift": "2 C N", "minimum_energy": Fraction(0),
        "number_conserving": True, "preparation_mechanism": None,
        "proof": ("Positive hopping Laplacian has only the uniform-condensate kernel at fixed N. "
                  "For N>=2 that vector is not in the D=0 kernel; g>0 gives positive nonvacuum neutral energies."
                  if g else "Every neutral uniform condensate is in the hopping-Laplacian kernel and has zero energy."),
        "same_within_sector_dynamics_up_to_phase": True})


def demonstration_report():
    """Frozen, bounded examples; no files, fitting or physical predictions."""
    witness = all_number_model(3, 6, 1.0, 0.7)
    return {"model_id": MODEL_ID,
            "cases": [analyze_neutral_sectors(*case) for case in FROZEN_CONTROLS],
            "direct_above_filling_witness": {
                "parameters": {"L": 3, "N": 6, "C": 1.0, "g": 0.7},
                "bounds": _json_safe(sector_energy_bounds(3, 6, 1.0, 0.7)),
                "numerical": sector_ground_diagnostic(witness)},
            "shifted_convention": [shifted_convention_result(5, 5, 1.0, g) for g in (40.0, 0.0)],
            "limitations": list(LIMITATIONS),
            "physical_predictions": {"masses": None, "mixing": None}}
