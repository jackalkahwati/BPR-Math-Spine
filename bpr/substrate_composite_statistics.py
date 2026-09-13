"""Bounded composite algebra on complete binary and Bose number spaces.

Contract: doc/derivations/substrate_composite_statistics_2026-09-12.md.
Binary operators use exact int64 original-site tensor factors, site 0 the
least-significant bit. Bose bases are complete lexicographic occupation tuples;
their float64 residuals are numerical checks, not exactness certificates.
No Hamiltonian, eigensystem, binding interaction or physical fermion is supplied.
"""
from functools import wraps
from numbers import Integral

import numpy as np

from bpr.substrate_charged_response import local_annihilation_map
from bpr.substrate_fermionization import _occupations
from bpr.substrate_vacuum_selection import capped_binomial

MAX_BINARY_DIMENSION = 256
MAX_BOSE_DIMENSION = 512
LIMITATIONS = (
    "Exchange grading is distinct from canonical CAR; odd composites need not obey full-space CAR.",
    "Original-site tensor support includes surviving Jordan-Wigner strings in a fixed ordering, not relativistic locality.",
    "Number charge, constituent count and modular neutrality are distinct; q=L is inherited and q=2,3 alternatives are counterfactuals.",
    "Full-support grading is formal because there is no disjoint nonempty partner in that ring.",
    "A canonical lowering operator on a chosen two-state neutral space does not establish full-space CAR or a physical fermion.",
    "No binding mechanism, protected code, physical spin, spacetime chirality or physical fermions are derived.",
    "The Bose ladders use complete sectors without a local occupation cutoff; their raw float64 residuals are not exact arithmetic certificates.",
    "These are bounded kinematic algebra controls, with no Hamiltonian or eigensystem calculation and no empirical validation.",
)


class NumericalUnavailable(ValueError):
    """Computed floating arithmetic is nonfinite or otherwise unavailable."""


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + " must be a Python or NumPy integer, not bool or float")
    if value < lower or value > upper:
        raise ValueError(name + " outside supported range")
    return int(value)


def _sites(value, L, maximum):
    # Inspect the finite container length before touching or converting elements.
    if not isinstance(value, (list, tuple)):
        raise ValueError("sites must be a finite list or tuple")
    if not 1 <= len(value) <= maximum:
        raise ValueError("sites length outside supported range")
    result = tuple(_integer(x, "site", 0, L - 1) for x in value)
    if any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError("sites must be strictly ascending and distinct")
    return result


def _binary_parameters(L, sites):
    L = _integer(L, "L", 3, 8)
    sites = _sites(sites, L, L)
    if (1 << L) > MAX_BINARY_DIMENSION:
        raise ValueError("binary dimension exceeds cap 256")
    return L, sites


def _bose_parameters(L, N, sites, max_N):
    L = _integer(L, "L", 3, 5)
    N = _integer(N, "N", 0, max_N)
    return L, N, _sites(sites, L, 3)


def _immutable(array, dtype):
    """Detached immutable storage, including zero-row rectangular matrices."""
    value = np.asarray(array, dtype=dtype)
    return np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)


def _matrix_record(matrix):
    return {"shape": list(matrix.shape), "real": matrix.real.tolist(),
            "imag": matrix.imag.tolist()}


def _json_native(value):
    if isinstance(value, np.ndarray):
        return _matrix_record(value)
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parity_sites(L, sites):
    selected = set(sites)
    return tuple(x for x in range(L)
                 if x not in selected and sum(i > x for i in sites) % 2)


def _phase(sites):
    r = len(sites)
    return -1 if (r * (r - 1) // 2) % 2 else 1


def _tensor(L, factors):
    """Ordered tensor product L-1 down to 0, with exact signed entries."""
    local = {
        "identity": np.eye(2, dtype=np.int64),
        "lower": np.array([[0, 1], [0, 0]], dtype=np.int64),
        "raise": np.array([[0, 0], [1, 0]], dtype=np.int64),
        "parity": np.diag(np.array([1, -1], dtype=np.int64)),
        "occupied": np.diag(np.array([0, 1], dtype=np.int64)),
        "empty": np.diag(np.array([1, 0], dtype=np.int64)),
    }
    result = np.ones((1, 1), dtype=np.int64)
    for x in range(L - 1, -1, -1):
        result = np.kron(result, local[factors.get(x, "identity")])
    return result


def binary_composite(L, sites, kind="jw"):
    """C_S on all 2**L binary states, built from reviewed tensor factors."""
    L, sites = _binary_parameters(L, sites)
    if not isinstance(kind, str) or kind not in ("jw", "hard_core"):
        raise ValueError("kind must be exactly 'jw' or 'hard_core'")
    factors = {x: "lower" for x in sites}
    phase = 1
    if kind == "jw":
        factors.update({x: "parity" for x in _parity_sites(L, sites)})
        phase = _phase(sites)
    return _immutable(phase * _tensor(L, factors), np.int64)


def _is_zero(matrix):
    return not bool(np.any(matrix))


def _tensor_support(L, matrix):
    """Test commutation with BOTH onsite X and Z on the full tensor space.

    Row/column permutations implement X multiplication; signed row/column
    scaling implements Z multiplication, avoiding dense onsite intermediates.
    """
    indices = np.arange(1 << L, dtype=np.int64)
    support = []
    for x in range(L):
        flipped = indices ^ (1 << x)
        z = 1 - 2 * ((indices >> x) & 1)
        commutes_x = np.array_equal(matrix[flipped, :], matrix[:, flipped])
        commutes_z = np.array_equal(z[:, None] * matrix, matrix * z[None, :])
        if not (commutes_x and commutes_z):
            support.append(x)
    return tuple(support)


def binary_report(L, sites, q=None):
    """Exact full-binary-space identities and explicit neutral compression."""
    L, sites = _binary_parameters(L, sites)
    modulus = L if q is None else _integer(q, "q", 2, L)
    if modulus not in (L, 2, 3):
        raise ValueError("q must equal L, 2 or 3")
    r = len(sites)
    dimension = 1 << L
    composite = binary_composite(L, sites)
    hard_core = binary_composite(L, sites, kind="hard_core")
    annihilation_product = composite.T @ composite
    creation_product = composite @ composite.T
    occupied = _tensor(L, {x: "occupied" for x in sites})
    empty = _tensor(L, {x: "empty" for x in sites})
    anticommutator = annihilation_product + creation_product
    numbers = np.array([bin(bits).count("1") for bits in range(dimension)],
                       dtype=np.int64)
    neutral_basis = tuple(i for i in range(dimension) if numbers[i] % modulus == 0)
    neutral_composite = composite[np.ix_(neutral_basis, neutral_basis)]
    parity = _parity_sites(L, sites)
    support = tuple(sorted(set(sites).union(parity)))
    charge_commutator = numbers[:, None] * composite - composite * numbers[None, :]
    return {
        "L": L, "sites": sites, "degree": r, "dimension": dimension,
        "kind": "jw", "modulus": modulus,
        "neutrality_rule": "inherited" if modulus == L else "counterfactual",
        "number_charge": -r, "modular_charge": (-r) % modulus,
        "grading": r % 2, "modularly_neutral": r % modulus == 0,
        "parity_sites": parity, "tensor_support": support,
        "tensor_phase": _phase(sites), "neutral_basis": neutral_basis,
        "neutral_composite": _matrix_record(neutral_composite),
        "neutral_compression_status": "zero" if _is_zero(neutral_composite) else "nonzero",
        "self_anticommutator_diagonal": np.diag(anticommutator).tolist(),
        "canonical_car": bool(np.array_equal(anticommutator, np.eye(dimension, dtype=np.int64))),
        "nilpotent": _is_zero(composite @ composite),
        "charge_identity_exact": bool(np.array_equal(charge_commutator, -r * composite)),
        "self_products_exact": bool(np.array_equal(annihilation_product, occupied)
                                    and np.array_equal(creation_product, empty)),
        "tensor_support_verified": _tensor_support(L, composite) == support,
        "hard_core_self_products_exact": bool(np.array_equal(hard_core.T @ hard_core, occupied)
                                              and np.array_equal(hard_core @ hard_core.T, empty)),
        "exact_binary_arithmetic": True,
    }


def _overlap_products(L, S, T):
    """Independent original-site overlap formula, not products of C matrices."""
    s, t = set(S), set(T)
    js, jt = set(_parity_sites(L, S)), set(_parity_sites(L, T))
    u, v, overlap = s - t, t - s, s & t
    k = (js ^ jt) - (s | t)
    alpha = len(u & jt) + len(v & js)
    factors = {x: "parity" for x in k}
    factors.update({x: "lower" for x in u})
    factors.update({x: "raise" for x in v})
    forward_factors = dict(factors)
    reverse_factors = dict(factors)
    forward_factors.update({x: "empty" for x in overlap})
    reverse_factors.update({x: "occupied" for x in overlap})
    phase = _phase(S) * _phase(T)
    return (phase * (-1 if alpha % 2 else 1) * _tensor(L, forward_factors),
            phase * _tensor(L, reverse_factors))


def binary_pair_report(L, S, T):
    """Direct exact mixed products, checked against the tensor overlap formula."""
    L, S = _binary_parameters(L, S)
    T = _sites(T, L, L)
    a, b = binary_composite(L, S), binary_composite(L, T)
    forward, reverse = a @ b.T, b.T @ a
    sign = -1 if (len(S) * len(T)) % 2 else 1
    bracket = forward - sign * reverse
    expected_forward, expected_reverse = _overlap_products(L, S, T)
    hard_a = binary_composite(L, S, kind="hard_core")
    hard_b = binary_composite(L, T, kind="hard_core")
    return {
        "L": L, "S": S, "T": T, "degree_S": len(S), "degree_T": len(T),
        "disjoint": set(S).isdisjoint(T), "exchange_sign": sign,
        "annihilator_exchange_exact": bool(np.array_equal(a @ b, sign * (b @ a))),
        "mixed_graded_bracket_zero": _is_zero(bracket),
        "overlap_formula_exact": bool(np.array_equal(forward, expected_forward)
                                      and np.array_equal(reverse, expected_reverse)),
        "hard_core_commuting_annihilators": bool(np.array_equal(hard_a @ hard_b, hard_b @ hard_a)),
        "exact_binary_arithmetic": True,
        "mixed_forward": _immutable(forward, np.int64),
        "mixed_reverse": _immutable(reverse, np.int64),
        "mixed_anticommutator": _immutable(forward + reverse, np.int64),
        "mixed_graded_bracket": _immutable(bracket, np.int64),
    }


def _float_guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise NumericalUnavailable("computed Bose arithmetic unavailable") from exc
    return checked


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise NumericalUnavailable(name + " contains nonfinite values")
    return value


def _lowering_sectors(N, degree):
    return tuple(range(N - degree, N + 1)) if N >= degree else (N,)


def _check_bose_dimensions(L, sectors):
    """Check every required full sector together, before bases or matrices."""
    dimensions = {}
    for N in sorted(set(sectors)):
        N = _integer(N, "auxiliary N", 0, 6)
        dimension = capped_binomial(L + N - 1, N, cap=MAX_BOSE_DIMENSION)
        if dimension > MAX_BOSE_DIMENSION:
            # capped_binomial's cap+1 sentinel is not the exact dimension.
            raise ValueError("complete Bose sector dimension exceeds cap 512")
        dimensions[N] = dimension
    return dimensions


def _local_ladder(L, N, site, rows, columns):
    try:
        inherited = np.asarray(local_annihilation_map(L, N, site))
    except ValueError as exc:
        # Inputs and every full-sector dimension were already validated.
        raise NumericalUnavailable("inherited rectangular ladder unavailable") from exc
    _finite(inherited, "inherited rectangular ladder")
    if inherited.shape != (rows, columns) or np.any(inherited.imag != 0):
        raise NumericalUnavailable("inherited rectangular ladder has unexpected shape or phase")
    return _finite(np.asarray(inherited.real, dtype=np.float64), "real rectangular ladder")


def _bose_lowering(L, N, sites, dimensions):
    """Inputs and all intermediate dimensions have already been checked."""
    source_basis = tuple(_occupations(L, N))
    r = len(sites)
    if N < r:
        target_N, target_basis, target_dimension = None, (), 0
        matrix = np.zeros((0, dimensions[N]), dtype=np.float64)
        status = "absent_target"
    else:
        target_N = N - r
        target_basis = tuple(_occupations(L, target_N))
        target_dimension = dimensions[target_N]
        matrix = None
        for step, site in enumerate(reversed(sites)):
            current_N = N - step
            ladder = _local_ladder(L, current_N, site,
                                   dimensions[current_N - 1], dimensions[current_N])
            matrix = ladder if matrix is None else _finite(ladder @ matrix, "composite lowering")
        status = "complete_sector_map"
    return {
        "L": L, "source_N": N, "target_N": target_N, "sites": sites,
        "source_basis": source_basis, "target_basis": target_basis,
        "source_dimension": dimensions[N], "target_dimension": target_dimension,
        "matrix": _immutable(matrix, np.float64), "status": status,
    }


@_float_guard
def bose_composite_map(L, N, sites):
    """Complete rectangular lowering; N<r has no target, not a vacuum target."""
    L, N, sites = _bose_parameters(L, N, sites, 6)
    dimensions = _check_bose_dimensions(L, _lowering_sectors(N, len(sites)))
    return _bose_lowering(L, N, sites, dimensions)


def _frobenius(value):
    _finite(value, "residual matrix")
    return float(_finite(np.linalg.norm(value), "raw Frobenius residual"))


def _occupation_product(state, sites, offset):
    result = 1
    for site in sites:
        result *= state[site] + offset
    return result


@_float_guard
def bose_report(L, N, sites):
    """Complete-sector product identities and double creation FROM VACUUM."""
    L, N, sites = _bose_parameters(L, N, sites, 3)
    r = len(sites)
    sectors = (_lowering_sectors(N, r) + tuple(range(N, N + r + 1))
               + tuple(range(2 * r + 1)))
    dimensions = _check_bose_dimensions(L, sectors)
    lowering = _bose_lowering(L, N, sites, dimensions)
    upper = _bose_lowering(L, N + r, sites, dimensions)
    a, creation = lowering["matrix"], upper["matrix"].T
    annihilation_product = _finite(a.T @ a, "annihilation product")
    creation_product = _finite(creation.T @ creation, "creation product")
    basis = lowering["source_basis"]
    expected_annihilation = [_occupation_product(state, sites, 0) for state in basis]
    expected_creation = [_occupation_product(state, sites, 1) for state in basis]

    vacuum_ladder = _bose_lowering(L, r, sites, dimensions)
    double_ladder = _bose_lowering(L, 2 * r, sites, dimensions)
    once = vacuum_ladder["matrix"].T[:, 0]
    twice = _finite(double_ladder["matrix"].T @ once, "double creation amplitudes")
    target_basis = double_ladder["source_basis"]
    target_state = tuple(2 if x in sites else 0 for x in range(L))
    expected_twice = np.zeros(dimensions[2 * r], dtype=np.float64)
    expected_twice[target_basis.index(target_state)] = np.sqrt(float(2 ** r))
    norm_squared = float(_finite(np.dot(twice, twice), "double creation norm squared"))
    return {
        "L": L, "N": N, "sites": sites, "degree": r,
        "dimension": dimensions[N], "basis": basis,
        "lowering_status": lowering["status"], "lowering_shape": a.shape,
        "creation_shape": creation.shape, "number_charge": -r,
        "annihilation_product_diagonal": np.diag(annihilation_product).tolist(),
        "creation_product_diagonal": np.diag(creation_product).tolist(),
        "expected_annihilation_diagonal": expected_annihilation,
        "expected_creation_diagonal": expected_creation,
        "annihilation_product_frobenius_residual": _frobenius(
            annihilation_product - np.diag(expected_annihilation)),
        "creation_product_frobenius_residual": _frobenius(
            creation_product - np.diag(expected_creation)),
        "double_creation_vacuum": {
            "target_N": 2 * r, "target_basis": target_basis,
            "amplitudes": twice.tolist(), "expected_amplitudes": expected_twice.tolist(),
            "frobenius_residual": _frobenius(twice - expected_twice),
            "norm_squared": norm_squared, "expected_norm_squared": 2 ** r,
        },
        "exactness_certificate": False,
    }


def demonstration_report():
    """The frozen, deduplicated bounded grid, as a JSON-native dictionary."""
    binary_cases = []
    for L in (3, 4, 6, 8):
        seen = set()
        candidates = ((0,), (L - 1,), (0, 1), (0, L - 1),
                      (0, 1, 2), (0, 2, L - 1), tuple(range(L)))
        for sites in candidates:
            if len(set(sites)) != len(sites) or sites in seen:
                continue
            seen.add(sites)
            binary_cases.append(binary_report(L, sites))
    pair_controls = (
        (6, (0, 1, 2), (3, 4, 5)),
        (6, (0, 2, 4), (1, 3, 5)),
        (6, (0, 1, 2), (2, 3, 4)),
        (4, (0,), (1,)),
        (4, (0, 1), (2, 3)),
        (4, (0, 1), (1, 2)),
    )
    return _json_native({
        "binary_cases": binary_cases,
        "pair_cases": [binary_pair_report(L, S, T) for L, S, T in pair_controls],
        "counterfactual_cases": [binary_report(4, (0, 1), q=2),
                                 binary_report(4, (0, 1, 2), q=3)],
        "bose_cases": [bose_report(L, N, sites) for L in (3, 4, 5)
                       for N in (0, 1, 2, 3)
                       for sites in ((0,), (0, 1), (0, 1, 2))],
        "limitations": LIMITATIONS,
        "numerical_domain": {
            "binary_L": (3, 4, 6, 8), "binary_dimension_cap": MAX_BINARY_DIMENSION,
            "binary_dtype": "int64", "binary_basis": "ascending bit integers, site 0 least significant",
            "bose_L": (3, 4, 5), "bose_reference_N": (0, 1, 2, 3),
            "bose_auxiliary_N_max": 6, "bose_sector_dimension_cap": MAX_BOSE_DIMENSION,
            "bose_largest_supported_sector_dimension": 210,
            "bose_dtype": "float64", "bose_basis": "complete lexicographic occupation tuples",
            "bose_residuals": "raw Frobenius residuals, not exactness certificates",
            "no_combined_cross_sector_enlargement": True,
        },
    })
