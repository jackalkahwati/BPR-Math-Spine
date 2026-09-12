"""Conditional hard-core/Jordan-Wigner diagnostic for a finite quantum ring.

Extends the stipulated three-site Bose-Hubbard prescription, not a derivation
of quantization from classical DNLS. Occupations use lexicographic tuple order;
binary operators use bit index sum(2**x*n_x), NOT tensor index 9*n0+3*n1+n2.
All certificate norms are spectral/operator 2-norms. Analytic bounds exclude
floating-point error; numerical residuals are checks, not rigorous certificates.
Only normal binary64 arithmetic is supported; unresolved intermediates raise
ValueError rather than yielding a false zero or an infinite certificate.
"""
from dataclasses import dataclass
from functools import wraps
from math import comb
from numbers import Integral, Real

import numpy as np

MODEL_ID = "conditional-substrate-fermionization-v1"
MAX_SITES = 12
MAX_DENSE_DIM = 512
MAX_BINARY_DIM = 256
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny

# Frozen before evaluation. These are illustrations, not empirical calibration.
FROZEN_CONTROLS = (
    (5, 2, 1.0, 40.0, 5),
    (5, 2, 1.0, 0.7, 5),
    (3, 3, 1.0, 40.0, 3),
)
LIMITATIONS = (
    "The L-site quantum Bose-Hubbard prescription is an explicit extension of the three-site toy, not unique classical quantization.",
    "Hard-core/Jordan-Wigner CAR uses nonlocal one-dimensional strings, not physical relativistic fermions, spin or chirality.",
    "PHP is an exact compression, not an invariant finite-g dynamics when PHQ is nonzero.",
    "The boundary twist depends on particle-number parity; one untwisted periodic free-fermion Hamiltonian does not cover all sectors.",
    "Schur elimination is energy dependent, not an exact autonomous Hamiltonian or all-time approximation.",
    "Only Schur operator and low-eigenstate leakage bounds are certified analytically; eigenvalue comparisons and residuals are numerical checks.",
    "Analytic bounds exclude roundoff; arithmetic-resolution diagnostics are not rigorous roundoff certificates.",
    "Finite-g virtual processes introduce interactions and correlated hopping; finite-g dynamics is not exactly free fermionic.",
    "Neutrality is an independent state-selection assumption, not dynamical confinement; Jordan-Wigner strings retain charge one.",
    "Projected neutrality counts do not remove finite-g doublons or settle arbitrary composite statistics.",
    "No physical masses, mixing, selected vacuum or three-family conclusion is derived.",
)


def _guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ValueError("numerically unresolved arithmetic or singular solve") from exc
    return checked


def _normal(value, name):
    a = np.asarray(value)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"numerically unresolved {name}: nonfinite")
    if np.any((np.abs(a) > 0) & (np.abs(a) < _TINY)):
        raise ValueError(f"numerically unresolved {name}: subnormal")
    return a.item() if a.ndim == 0 else a


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} outside supported range")
    return int(value)


def _real(value, name, lower=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    result = _normal(float(value), name)
    if result == 0 and value != 0:
        raise ValueError(f"numerically unresolved {name}: underflow")
    if (lower is not None and result < lower) or (positive and result <= 0):
        raise ValueError(f"{name} outside supported range")
    return result


def _mul(a, b, name):
    result = _normal(np.multiply(a, b), name)
    if a != 0 and b != 0 and result == 0:
        raise ValueError(f"numerically unresolved {name}: underflow")
    return result


def _div(a, b, name):
    result = _normal(np.divide(a, b), name)
    if a != 0 and result == 0:
        raise ValueError(f"numerically unresolved {name}: underflow")
    return result


def _scale(matrix, scale, name):
    result = _normal(matrix * scale, name)
    if scale != 0 and np.any((matrix != 0) & (result == 0)):
        raise ValueError(f"numerically unresolved {name}: underflow")
    return result


def _norm(matrix):
    """Scaled spectral norm (Euclidean for vectors), including empty blocks."""
    if matrix.size == 0:
        return 0.0
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return 0.0
    return _mul(scale, float(np.linalg.norm(matrix / scale, ord=2)), "2-norm")


def _sector(L, N):
    L = _integer(L, "L", 3, MAX_SITES)
    N = _integer(N, "N", 0, L)
    dim = comb(L + N - 1, N)
    if dim > MAX_DENSE_DIM:
        raise ValueError(f"fixed-number dimension {dim} exceeds cap {MAX_DENSE_DIM}")
    return L, N


def _occupations(L, N):
    if L == 1:
        yield (N,)
    else:
        for first in range(N + 1):
            for rest in _occupations(L - 1, N - first):
                yield (first,) + rest


@dataclass(frozen=True)
class FixedNumberModel:
    L: int
    N: int
    C: float
    g: float
    basis: tuple
    p_indices: np.ndarray
    q_indices: np.ndarray
    D: np.ndarray
    V_unit: np.ndarray
    V: np.ndarray
    H: np.ndarray
    PHP: np.ndarray
    B: np.ndarray
    QHQ: np.ndarray

    @property
    def hard_core_basis(self):
        return tuple(self.basis[i] for i in self.p_indices)


@_guard
def fixed_number_model(L, N, C=1.0, g=0.7):
    """Full sum(n)=N basis, H=gD+V, P:n_x<=1, B=PHQ.

    Caps are checked before enumeration/allocation, including the vacuum L cap.
    No local cutoff is used. The old local n<=2 toy is an oracle only for N<=2.
    """
    L, N = _sector(L, N)
    C = _real(C, "C", positive=True)
    g = _real(g, "g", lower=0)
    _mul(C, 2 * N, "hopping scale")
    _mul(g, N * (N - 1) // 2, "interaction scale")
    basis = tuple(_occupations(L, N))
    index = {state: i for i, state in enumerate(basis)}
    d = len(basis)
    V_unit = np.zeros((d, d))
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
    H = V + np.diag(_scale(D, g, "interaction diagonal"))
    p = np.array([i for i, state in enumerate(basis) if max(state) <= 1], dtype=int)
    q = np.array([i for i, state in enumerate(basis) if max(state) > 1], dtype=int)
    return FixedNumberModel(L, N, C, g, basis, p, q, D, V_unit, V, H,
                            H[np.ix_(p, p)], H[np.ix_(p, q)], H[np.ix_(q, q)])


def binary_operators(L):
    """Complete bit-ordered binary space: b_x, c_x=Z_0...Z_(x-1)b_x, n_x.

    CAR cannot be checked by compressing lowering operators into fixed N.
    """
    L = _integer(L, "L", 3, MAX_SITES)
    d = 1 << L
    if d > MAX_BINARY_DIM:
        raise ValueError(f"binary dimension {d} exceeds cap {MAX_BINARY_DIM}")
    b, c, n = [], [], []
    for x in range(L):
        bx = np.zeros((d, d))
        cx = np.zeros((d, d))
        nx = np.zeros((d, d))
        for bit in range(d):
            if bit & (1 << x):
                row = bit ^ (1 << x)
                bx[row, bit] = 1
                cx[row, bit] = (-1) ** bin(bit & ((1 << x) - 1)).count("1")
                nx[bit, bit] = 1
        b.append(bx)
        c.append(cx)
        n.append(nx)
    return {"b": tuple(b), "c": tuple(c), "n": tuple(n), "number": sum(n)}


@_guard
def algebra_diagnostics(L):
    """Full-space CAR, offsite b commutation, charge and neutral-bilinear checks."""
    ops = binary_operators(L)
    b, c, n, number = (ops[k] for k in ("b", "c", "n", "number"))
    identity = np.eye(len(number))
    residuals = {"car_annihilation": 0.0, "car_mixed": 0.0,
                 "offsite_hard_core_commutator": 0.0, "density_identity": 0.0,
                 "creation_charge_one": 0.0, "neutral_bilinear_charge": 0.0}
    for x in range(L):
        residuals["density_identity"] = max(residuals["density_identity"], _norm(c[x].T @ c[x] - n[x]))
        residuals["creation_charge_one"] = max(residuals["creation_charge_one"], _norm(number @ c[x].T - c[x].T @ number - c[x].T))
        for y in range(L):
            residuals["car_annihilation"] = max(residuals["car_annihilation"], _norm(c[x] @ c[y] + c[y] @ c[x]))
            residuals["car_mixed"] = max(residuals["car_mixed"], _norm(c[x] @ c[y].T + c[y].T @ c[x] - (identity if x == y else 0)))
            bilinear = c[x].T @ c[y]
            residuals["neutral_bilinear_charge"] = max(residuals["neutral_bilinear_charge"], _norm(number @ bilinear - bilinear @ number))
            if x != y:
                residuals["offsite_hard_core_commutator"] = max(residuals["offsite_hard_core_commutator"], _norm(b[x] @ b[y] - b[y] @ b[x]))
    return residuals


def boundary_twist(N):
    """(-1)^(N-1), including the physically immaterial vacuum twist -1."""
    N = _integer(N, "N", 0, MAX_SITES)
    return 1 if N % 2 else -1


@_guard
def fermionic_hopping(L, N, C=1.0, twist=None):
    """Independent signed-bit hopping, returned in lexicographic P ordering.

    Interior links have no twist. The closing fermionic link carries
    (-1)^(N-1), canceling the sign from the N-1 intervening occupied sites.
    A caller-specified +/-1 twist supports a deliberately wrong-twist control.
    """
    L, N = _sector(L, N)
    C = _real(C, "C", positive=True)
    if twist is None:
        twist = boundary_twist(N)
    if isinstance(twist, (bool, np.bool_)) or not isinstance(twist, Integral):
        raise TypeError("twist must be integer +1 or -1")
    if twist not in (-1, 1):
        raise ValueError("twist must be +1 or -1")
    basis = tuple(state for state in _occupations(L, N) if max(state) <= 1)
    bits = [sum(n << x for x, n in enumerate(state)) for state in basis]
    index = {bit: i for i, bit in enumerate(bits)}
    hopping = np.zeros((len(bits), len(bits)))
    for col, bit in enumerate(bits):
        for x in range(L):
            y = (x + 1) % L
            weight = twist if x == L - 1 else 1
            for target, source in ((x, y), (y, x)):
                if not (bit & (1 << source)) or bit & (1 << target):
                    continue
                intermediate = bit ^ (1 << source)
                sign = (-1) ** (bin(bit & ((1 << source) - 1)).count("1")
                                + bin(intermediate & ((1 << target) - 1)).count("1"))
                row = intermediate | (1 << target)
                hopping[index[row], col] -= weight * sign
    return _scale(hopping, C, "fermionic hopping")


@_guard
def validity_certificate(model):
    """Sufficient low-cluster bounds, not numerical eigensolver certificates.

    Qempty is structural and exact for every g. A positive margin comparable
    to roundoff is rejected; weak/equal repulsion is merely uncertified.
    """
    v = _mul(model.C, 2 * model.N, "v")
    B_norm = _norm(model.B)
    result = {"status": "unavailable", "available": False,
              "q_empty": len(model.q_indices) == 0, "v": v, "B_norm": B_norm,
              "margin": None, "leakage_ratio_bound": None,
              "schur_shift_bound": None, "second_order_remainder_bound": None,
              "low_cluster_interval": None, "high_cluster_lower_bound": None,
              "roundoff_included": False}
    if result["q_empty"]:
        result.update(status="exact_q_empty", available=True, leakage_ratio_bound=0.0,
                      schur_shift_bound=0.0, second_order_remainder_bound=0.0,
                      low_cluster_interval=[-v, v])
        return result
    twice_v = _mul(2.0, v, "2v")
    margin = _normal(model.g - twice_v, "separation margin")
    result["margin"] = margin
    if margin <= 0:
        return result
    resolution = _mul(64 * _EPS, max(model.g, twice_v), "margin resolution")
    if margin <= resolution:
        raise ValueError("numerically unresolved positive separation margin g-2v")
    ratio = _div(B_norm, margin, "leakage ratio")
    shift = _mul(B_norm, ratio, "Schur shift bound")
    remainder = _mul(_div(twice_v, model.g, "relative resolvent scale"), shift, "resolvent remainder bound")
    result.update(status="certified_separated_cluster", available=True,
                  leakage_ratio_bound=ratio, schur_shift_bound=shift,
                  second_order_remainder_bound=remainder,
                  low_cluster_interval=[-v, v], high_cluster_lower_bound=model.g - v)
    return result


@_guard
def second_order_effective(model):
    """H2=PHP+Delta, Delta=-B(gD_Q)^(-1)B^T, scaled separately.

    Available algebraically for every g>0, not only in the certified regime.
    A small correction can be resolved alone but lost upon addition to PHP.
    """
    if len(model.q_indices) == 0:
        delta = np.zeros_like(model.PHP)
        return {"hamiltonian": model.PHP.copy(), "correction": delta,
                "correction_coefficient": delta.copy(), "scale": 0.0,
                "total_addition_resolved": True}
    if model.g == 0:
        raise ValueError("second-order elimination requires g>0 when Q is nonempty")
    B_unit = model.V_unit[np.ix_(model.p_indices, model.q_indices)]
    coefficient = -(B_unit / model.D[model.q_indices]) @ B_unit.T
    scale = _mul(model.C, _div(model.C, model.g, "C/g"), "C squared / g")
    delta = _scale(coefficient, scale, "second-order correction")
    total = _normal(model.PHP + delta, "second-order total")
    mask = delta != 0
    increments = total[mask] - model.PHP[mask]
    relative = increments / delta[mask]
    resolved = bool(np.all(np.abs(relative - 1) <= 64 * _EPS))
    return {"hamiltonian": total, "correction": delta,
            "correction_coefficient": coefficient, "scale": scale,
            "total_addition_resolved": resolved}


@_guard
def schur_correction(model, E):
    """-B(QHQ-E)^(-1)B^T; a resolvent correction, not an autonomous H."""
    E = _real(E, "E")
    if len(model.q_indices) == 0:
        return np.zeros_like(model.PHP)
    # Normalize before solving and multiply dimensionless B blocks: do not
    # square a tiny/huge energy scale or subtract two nearly equal totals.
    scale = max(model.g, model.C, abs(E))
    q = model.q_indices
    Q_scaled = (_scale(model.V_unit[np.ix_(q, q)], _div(model.C, scale, "C/solve scale"), "scaled Q hopping")
                + np.diag(_scale(model.D[q], _div(model.g, scale, "g/solve scale"), "scaled Q interaction")))
    Q_scaled -= np.eye(len(q)) * _div(E, scale, "E/solve scale")
    singular = np.linalg.svd(Q_scaled, compute_uv=False)
    if singular[-1] <= 64 * _EPS * max(1.0, float(singular[0])):
        raise ValueError("singular or numerically unresolved QHQ-E resolvent")
    B_unit = model.V_unit[np.ix_(model.p_indices, q)]
    coefficient = -B_unit @ np.linalg.solve(Q_scaled, B_unit.T)
    correction_scale = _mul(model.C, _div(model.C, scale, "C/solve scale"), "Schur correction scale")
    return _scale(coefficient, correction_scale, "Schur correction")


@_guard
def schur_remainder(model, E):
    """S(E)-H2 via the resolvent identity, not cancellation of full totals.

    B R0 (QVQ-E) (QHQ-E)^(-1) B^T, R0=(gD_Q)^(-1).
    Structural zeros are handled before scaling potentially tiny quantities.
    """
    E = _real(E, "E")
    q = model.q_indices
    if len(q) == 0:
        return np.zeros_like(model.PHP)
    if model.g == 0:
        raise ValueError("second-order remainder requires g>0 when Q is nonempty")
    Vq = model.V_unit[np.ix_(q, q)]
    if E == 0 and not np.any(Vq):
        return np.zeros_like(model.PHP)
    solve_scale = max(model.g, model.C, abs(E))
    w_scale = max(model.C, abs(E))
    W = _scale(Vq, _div(model.C, w_scale, "C/W scale"), "scaled QVQ")
    W -= np.eye(len(q)) * _div(E, w_scale, "E/W scale")
    K = _scale(Vq, _div(model.C, solve_scale, "C/solve scale"), "scaled Q hopping")
    K += np.diag(_scale(model.D[q], _div(model.g, solve_scale, "g/solve scale"), "scaled interaction"))
    K -= np.eye(len(q)) * _div(E, solve_scale, "E/solve scale")
    singular = np.linalg.svd(K, compute_uv=False)
    if singular[-1] <= 64 * _EPS * max(1.0, float(singular[0])):
        raise ValueError("singular or numerically unresolved QHQ-E resolvent")
    B_unit = model.V_unit[np.ix_(model.p_indices, q)]
    coefficient = (B_unit / model.D[q]) @ W @ np.linalg.solve(K, B_unit.T)
    second_scale = _mul(model.C, _div(model.C, model.g, "C/g"), "C squared / g")
    scale = _mul(second_scale, _div(w_scale, solve_scale, "remainder relative scale"), "remainder scale")
    return _scale(coefficient, scale, "Schur remainder")


@_guard
def schur_operator(model, E):
    """S(E)=PHP-B(QHQ-E)^(-1)B^T, exact algebraic Schur complement."""
    return _normal(model.PHP + schur_correction(model, E), "Schur operator")


def neutrality_counts(L, q):
    """Strict hard-core sectors N mod q=0 under an INDEPENDENT assumption."""
    L = _integer(L, "L", 3, MAX_SITES)
    q = _integer(q, "q", 2)
    sectors = [{"N": N, "dimension": comb(L, N)} for N in range(L + 1) if N % q == 0]
    return {"L": L, "q": q, "assumed": True, "rule": "N mod q = 0",
            "sectors": sectors, "allowed_dimension": sum(row["dimension"] for row in sectors),
            "full_binary_dimension": 1 << L, "dynamical_confinement_derived": False}


def _virtual_witnesses(model, second):
    targets = []
    if (model.L, model.N) == (5, 2):
        targets = [((1, 1, 0, 0, 0), (1, 1, 0, 0, 0)),
                   ((1, 1, 0, 0, 0), (0, 1, 1, 0, 0))]
    elif (model.L, model.N) == (3, 3):
        targets = [((1, 1, 1), (1, 1, 1))]
    witnesses = []
    for initial, final in targets:
        col = model.hard_core_basis.index(initial)
        row = model.hard_core_basis.index(final)
        paths = []
        for j, index in enumerate(model.q_indices):
            if model.B[row, j] != 0 and model.B[col, j] != 0:
                paths.append({"intermediate": list(model.basis[index]),
                              "initial_hop": float(model.B[col, j]),
                              "final_hop": float(model.B[row, j]), "D": float(model.D[index])})
        witnesses.append({"initial": list(initial), "final": list(final), "paths": paths,
                          "coefficient_in_C_squared_over_g": float(second["correction_coefficient"][row, col]),
                          "matrix_element": float(second["correction"][row, col])})
    return witnesses


@_guard
def case_report(L, N, C, g, q=None):
    """JSON-safe finite example, with numerical checks separated from bounds.

    The report rejects an eigensolver resolution proxy 64*eps*||H|| >= 1e-6
    times the smaller hopping/correction scale. This conservative arithmetic
    screen is not a theorem restriction or a rigorous roundoff certificate;
    low-level analytic/matrix APIs remain usable outside the report domain.
    """
    model = fixed_number_model(L, N, C, g)
    # Fail the independent binary cap before any eigensystem or CAR work.
    if 1 << model.L > MAX_BINARY_DIM:
        raise ValueError("binary dimension exceeds report cap")
    neutrality = None if q is None else neutrality_counts(model.L, q)
    if neutrality is not None:
        q = neutrality["q"]
        neutrality["comparison_sector_neutral"] = bool(model.N % q == 0)
    certificate = validity_certificate(model)
    second = None if model.g == 0 and len(model.q_indices) else second_order_effective(model)
    resolution = _mul(64 * _EPS, _norm(model.H), "eigensolver resolution")
    relevant_scale = certificate["v"]
    if second is not None and len(model.q_indices):
        relevant_scale = min(relevant_scale, _norm(second["correction"]))
    if relevant_scale > 0 and resolution >= _mul(1e-6, relevant_scale, "report resolution threshold"):
        raise ValueError("numerically unresolved report eigensystem relative to hopping/correction; analytic certificate remains separate")
    p_dim = len(model.p_indices)
    eigenvalues, vectors = np.linalg.eigh(model.H)
    hard_eigenvalues = np.linalg.eigvalsh(model.PHP)
    fermionic = fermionic_hopping(model.L, model.N, model.C)
    correction = None
    checks = {"hard_core_fermionic_matrix_residual": _norm(model.PHP - fermionic),
              "hopping_norm": _norm(model.V),
              "full_eigensolver_absolute_resolution_scale": resolution,
              "report_eigensystem_resolution_screen_passed": True,
              "low_eigenstate_leakage_ratios": None,
              "schur_test_energy": None, "schur_shift_norm": None,
              "second_order_remainder_norm": None,
              "H2_low_eigenvalue_max_difference_not_certified": None}
    if second is not None:
        correction = {"operator_norm": _norm(second["correction"]),
                      "scale_C_squared_over_g": second["scale"],
                      "total_addition_resolved": second["total_addition_resolved"],
                      "evaluated_separately_from_total": True,
                      "virtual_witnesses": _virtual_witnesses(model, second)}
        checks["H2_low_eigenvalue_max_difference_not_certified"] = float(np.max(np.abs(np.linalg.eigvalsh(second["hamiltonian"]) - eigenvalues[:p_dim])))
    if certificate["available"]:
        checks["low_eigenstate_leakage_ratios"] = [
            _div(_norm(vectors[model.q_indices, j]), _norm(vectors[model.p_indices, j]), "observed leakage")
            for j in range(p_dim)]
        delta_exact = schur_correction(model, 0.0)
        checks.update(schur_test_energy=0.0, schur_shift_norm=_norm(delta_exact),
                      second_order_remainder_norm=_norm(schur_remainder(model, 0.0)))
    return {"parameters": {"L": model.L, "N": model.N, "C": model.C, "g": model.g, "q": q},
            "dimensions": {"full": len(model.basis), "hard_core": p_dim,
                           "discarded": len(model.q_indices), "binary": 1 << model.L},
            "boundary_twist": boundary_twist(model.N), "algebra_residuals": algebra_diagnostics(model.L),
            "spectra": {"full": eigenvalues.tolist(), "hard_core": hard_eigenvalues.tolist(),
                        "fermionic": np.linalg.eigvalsh(fermionic).tolist(),
                        "second_order": None if second is None else np.linalg.eigvalsh(second["hamiltonian"]).tolist()},
            "correction": correction, "certificate": certificate,
            "neutrality": neutrality, "numerical_checks": checks}


def demonstration_report():
    """Frozen, illustrative, stdout-only metadata; no fits or physical targets."""
    keys = ("L", "N", "C", "g", "q")
    controls = [dict(zip(keys, values)) for values in FROZEN_CONTROLS]
    return {"model_id": MODEL_ID, "frozen_controls": controls,
            "controls_frozen_before_evaluation": True, "empirical_calibration": False,
            "caps": {"sites": MAX_SITES, "fixed_number_dimension": MAX_DENSE_DIM,
                     "binary_dimension": MAX_BINARY_DIM},
            "cases": [case_report(**parameters) for parameters in controls],
            "physical_predictions": {"masses": None, "mixing": None},
            "limitations": list(LIMITATIONS)}
