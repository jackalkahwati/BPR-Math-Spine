"""Bounded interval-symmetry hypotheses and complete-sector finite diagnostics.

The conditional infinite-Fock theorem is algebraic, not inferred from these
cutoff matrices. All numerical comparisons are binary64 heuristics, with raw
real and imaginary residuals retained. No eigensystem or commutant is solved.
"""
from contextlib import contextmanager
from math import comb, hypot, isfinite, sqrt
from numbers import Real

import numpy as np

from bpr.substrate_charged_response import local_annihilation_map
from bpr.substrate_vacuum_selection import all_number_model, capped_binomial

MODEL_ID = "conditional-substrate-local-symmetry-v1"
MAX_DIMENSION = 512
_EPS = 2.0 ** -52
LIMITATIONS = (
    "The interval theorem assumes boundedness, weak conservation, C>0, and at least two exterior sites.",
    "Support reports check hypotheses, not conservation of an arbitrary operator.",
    "Finite complete-number-sector compressions are diagnostics, not unrestricted commutator norms.",
    "Vanishing finite residuals do not establish conservation on unrestricted Fock space.",
    "Comparison allowances are heuristic, not certified numerical error bounds or empirical validation.",
    "The theorem does not exclude constrained-subspace, approximate, dressed, nonlocal or low-energy gauge emergence.",
    "No physical fermions, Standard Model gauge sector or broader physical no-go theorem is supplied.",
)
_ARITHMETIC_LABELS = (
    "C", "g", "hopping scale", "interaction scale", "hopping matrix",
    "interaction diagonal", "annihilation map real component",
    "annihilation map imaginary component",
)


class NumericalUnavailable(Exception):
    """Recognized arithmetic failure, distinct from invalid input/integration."""


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(name + " must be a built-in int in the supported range")
    return value


def interval_support_report(L, start, length):
    """Check cyclic interval hypotheses and peel in the ORIGINAL L-site ring."""
    L = _integer(L, "L", 3, 64)
    start = _integer(start, "start", 0, L - 1)
    length = _integer(length, "length", 1, L)
    support = [(start + j) % L for j in range(length)]
    steps = []
    if length <= L - 2:
        status = "theorem_applies"
        for j, site in enumerate(support):
            steps.append({"removed_site": site,
                          "exterior_neighbor": (site - 1) % L,
                          "remaining_support": support[j + 1:]})
    else:
        status = ("outside_theorem_full_support" if length == L
                  else "outside_theorem_one_exterior")
    return {"L": L, "start": start, "length": length, "support": support,
            "complement_size": L - length, "status": status,
            "peel_steps": steps,
            "scope": {"checks_support_hypotheses_only": True,
                      "conditional_on": ["boundedness", "weak conservation", "C>0"],
                      "low_energy_gauge_emergence_excluded": False,
                      "description": "Not a conservation test for an arbitrary operator; the original ring is never shortened."}}


def _preflight(L):
    """Check every dimension against the live cap, before calls/allocations."""
    dimension = capped_binomial(L + 2, 2, cap=MAX_DIMENSION)
    sectors = [capped_binomial(L + N - 1, N, cap=MAX_DIMENSION)
               for N in range(3)]
    if dimension > MAX_DIMENSION or any(d > MAX_DIMENSION for d in sectors):
        raise ValueError("complete-sector dimension exceeds cap " + str(MAX_DIMENSION))
    return dimension, sectors


@contextmanager
def _arithmetic():
    """Translate owned arithmetic only, never ValueError or solver errors."""
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            yield
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalUnavailable("owned numerical arithmetic: " + str(exc)) from exc


def _zeros(L, shape, dtype=np.complex128):
    _preflight(L)
    with _arithmetic():
        return np.zeros(shape, dtype=dtype)


def _recognized_inherited_failure(exc):
    if str(exc) == "numerically unresolved arithmetic or singular solve":
        return isinstance(exc.__cause__,
                          (FloatingPointError, OverflowError, np.linalg.LinAlgError))
    return any(str(exc) == "numerically unresolved " + label + ": " + reason
               for label in _ARITHMETIC_LABELS
               for reason in ("nonfinite", "subnormal", "underflow"))


def _model(L, N, g):
    _preflight(L)
    try:
        return all_number_model(L, N, C=1.0, g=float(g))
    except ValueError as exc:
        if _recognized_inherited_failure(exc):
            raise NumericalUnavailable(str(exc)) from exc
        raise


def _ladder(L, N):
    _preflight(L)
    try:
        return local_annihilation_map(L, N, 0)
    except ValueError as exc:
        if _recognized_inherited_failure(exc):
            raise NumericalUnavailable(str(exc)) from exc
        raise


def _finite_scalar(value, label):
    if not isfinite(value.real) or not isfinite(value.imag):
        raise NumericalUnavailable(label + ": nonfinite")
    return value


def _finite_array(array, label):
    # Scalar checks avoid allocating unchecked dense masks or modulus arrays.
    for value in array.flat:
        _finite_scalar(value, label)
    return array


def _array(array, shape, label):
    if (not isinstance(array, np.ndarray) or array.shape != shape
            or array.dtype.kind not in "biufc"):
        raise ValueError(label + " has invalid numeric array structure or shape")
    return _finite_array(array, label)


def _sector_basis(L, N):
    if L == 1:
        yield (N,)
    else:
        for first in range(N + 1):
            for rest in _sector_basis(L - 1, N - first):
                yield (first,) + rest


def _validate_model(model, L, N, g, dimension):
    try:
        metadata = (model.L, model.N, model.C, model.g)
        basis, H = model.basis, model.H
    except AttributeError as exc:
        raise ValueError("inherited model is missing required data") from exc
    if (type(metadata[0]) is not int or type(metadata[1]) is not int
            or metadata[:2] != (L, N)):
        raise ValueError("inherited model L/N metadata mismatch")
    for value, expected in zip(metadata[2:], (1.0, g)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or value != expected):
            raise ValueError("inherited model C/g metadata mismatch")
    if not isinstance(basis, (list, tuple)) or len(basis) != dimension:
        raise ValueError("inherited model basis has invalid structure")
    for state in basis:
        if (not isinstance(state, (list, tuple)) or len(state) != L
                or any(type(n) is not int or n < 0 for n in state)):
            raise ValueError("inherited model occupation has invalid structure")
    basis = tuple(tuple(state) for state in basis)
    if basis != tuple(_sector_basis(L, N)):
        raise ValueError("inherited model basis is not complete lexicographic occupations")
    _array(H, (dimension, dimension), "inherited Hamiltonian")
    return basis, H


def _permutation(L, basis, index, sites):
    result = _zeros(L, (len(basis), len(basis)))
    for col, state in enumerate(basis):
        moved = [0] * L
        for site, target in enumerate(sites):
            moved[target] = state[site]
        result[index[tuple(moved)], col] = 1.0
    return result


def _unit_vector(L, dimension, index, occupations):
    result = _zeros(L, (dimension,))
    result[index[tuple(occupations)]] = 1.0
    return result


def _owned_case(L, g):
    """Private array ownership seam; no mutable scientific arrays escape publicly."""
    dimension, dimensions = _preflight(L)
    sectors = []
    for N, size in enumerate(dimensions):
        model = _model(L, N, g)
        sectors.append(_validate_model(model, L, N, g, size))
    ladders = []
    for N in (1, 2):
        ladder = _ladder(L, N)
        ladders.append(_array(ladder, (dimensions[N - 1], dimensions[N]),
                              "inherited annihilation map"))
    basis = tuple(state for sector, _ in sectors for state in sector)
    index = {state: i for i, state in enumerate(basis)}
    offsets = (0, dimensions[0], dimensions[0] + dimensions[1])
    H = _zeros(L, (dimension, dimension))
    with _arithmetic():
        for offset, (sector, block) in zip(offsets, sectors):
            H[offset:offset + len(sector), offset:offset + len(sector)] = block
    _finite_array(H, "direct-sum Hamiltonian")
    operators = []

    def add(name, support_class, conserved, matrix, witness=None):
        _finite_array(matrix, name)
        operators.append({"name": name, "support_class": support_class,
                          "exact_conservation": conserved, "matrix": matrix,
                          "witness": witness})

    identity = _zeros(L, (dimension, dimension))
    for i in range(dimension):
        identity[i, i] = 1.0
    add("identity", "empty_support", True, identity)

    flip = _zeros(L, (dimension, dimension))
    with _arithmetic():
        for N, ladder in zip((1, 2), ladders):
            for col, state in enumerate(sectors[N][0]):
                if state[0] == 1:
                    for row in range(dimensions[N - 1]):
                        value = ladder[row, col]
                        i, j = offsets[N - 1] + row, offsets[N] + col
                        flip[i, j] = value
                        flip[j, i] = value.conjugate()
    vacuum = [0] * L
    at_zero, at_one = list(vacuum), list(vacuum)
    at_zero[0], at_one[1] = 1, 1
    v = _unit_vector(L, dimension, index, at_one)
    w = _unit_vector(L, dimension, index, vacuum)
    add("local_flip", "singleton", False, flip,
        {"row": v, "column": w, "expected": -1.0})

    projector = _zeros(L, (dimension, dimension))
    for i, state in enumerate(basis):
        projector[i, i] = float(state[0] == 0)
    w = _unit_vector(L, dimension, index, at_zero)
    add("local_vacuum_projector", "singleton", False, projector,
        {"row": v, "column": w, "expected": 1.0})

    parity = _zeros(L, (dimension, dimension))
    for i, state in enumerate(basis):
        parity[i, i] = (-1.0) ** sum(state)
    add("total_number_parity", "full_support", True, parity)
    reflection = _permutation(L, basis, index, [(2 - x) % L for x in range(L)])
    add("reflection", "one_exterior_counterexample", True, reflection)

    if L == 4:
        swap = _permutation(L, basis, index, (2, 1, 0, 3))
        add("opposite_site_swap", "disconnected_opposite_sites", True, swap)
        dark = _zeros(L, (dimension, dimension))
        with _arithmetic():
            for i, state in enumerate(basis):
                n = state[0] + state[2]
                for j, other in enumerate(basis):
                    if (state[1] == other[1] and state[3] == other[3]
                            and n == other[0] + other[2]):
                        dark[i, j] = sqrt(comb(n, state[0]) * comb(n, other[0])) / 2 ** n
        bright_vector = _zeros(L, (dimension,))
        dark_vector = _zeros(L, (dimension,))
        with _arithmetic():
            for state, bright, dark_coefficient in (
                    ((2, 0, 0, 0), 0.5, 0.5),
                    ((1, 0, 1, 0), 1 / sqrt(2), -1 / sqrt(2)),
                    ((0, 0, 2, 0), 0.5, 0.5)):
                bright_vector[index[state]] = bright
                dark_vector[index[state]] = dark_coefficient
        add("dark_vacuum_projector", "disconnected_opposite_sites", g == 0, dark,
            {"row": dark_vector, "column": bright_vector, "expected": g / 2.0})
    return {"L": L, "g": g, "dimension": dimension, "basis": basis,
            "H": H, "operators": operators}


def _complex_record(value):
    _finite_scalar(value, "reported scalar")
    return {"real": float(value.real), "imag": float(value.imag)}


def _witness_report(L, commutator, scale, row, column, expected):
    dimension = commutator.shape[0]
    _array(commutator, (dimension, dimension), "witness commutator")
    _array(scale, (dimension, dimension), "witness scale")
    _array(row, (dimension,), "witness row")
    _array(column, (dimension,), "witness column")
    _finite_scalar(expected, "expected witness")
    product = _zeros(L, (dimension,))
    with _arithmetic():
        np.matmul(commutator, column, out=product)
        _finite_array(product, "witness product")
        value = _finite_scalar(np.vdot(row, product), "witness value")
        delta = _finite_scalar(value - expected, "witness difference")
        weighted_scale = 0.0
        for i in range(dimension):
            for j in range(dimension):
                term = _finite_scalar(abs(row[i]) * scale[i, j].real * abs(column[j]),
                                      "witness weighted scale term")
                weighted_scale = _finite_scalar(weighted_scale + term,
                                                "witness weighted scale")
        allowance = _finite_scalar(256 * _EPS * dimension * weighted_scale,
                                   "witness allowance")
        sign_required = bool(expected.real != 0)
        sign_satisfied = (not sign_required or
                          (value.real > 0 if expected.real > 0 else value.real < 0))
        passed = (abs(delta.real) <= allowance and abs(delta.imag) <= allowance
                  and sign_satisfied)
    return {"value": _complex_record(value), "expected": _complex_record(expected),
            "allowance": float(allowance),
            "status": ("within_heuristic_allowance" if passed
                       else "outside_heuristic_allowance"),
            "nonzero_sign_required": sign_required,
            "nonzero_sign_satisfied": bool(sign_satisfied)}


def _operator_report(L, H, operator):
    dimension = H.shape[0]
    A = _array(operator["matrix"], H.shape, "operator")
    commutator = _zeros(L, H.shape)
    reverse = _zeros(L, H.shape)
    absolute_H = _zeros(L, H.shape, dtype=np.float64)
    absolute_A = _zeros(L, H.shape, dtype=np.float64)
    scale = _zeros(L, H.shape, dtype=np.float64)
    reverse_scale = _zeros(L, H.shape, dtype=np.float64)
    allowance = _zeros(L, H.shape, dtype=np.float64)
    with _arithmetic():
        np.matmul(H, A, out=commutator)
        _finite_array(commutator, "H A product")
        np.matmul(A, H, out=reverse)
        _finite_array(reverse, "A H product")
        np.subtract(commutator, reverse, out=commutator)
        _finite_array(commutator, "finite commutator")
        np.absolute(H, out=absolute_H)
        np.absolute(A, out=absolute_A)
        _finite_array(absolute_H, "absolute Hamiltonian")
        _finite_array(absolute_A, "absolute operator")
        np.matmul(absolute_H, absolute_A, out=scale)
        _finite_array(scale, "forward scale")
        np.matmul(absolute_A, absolute_H, out=reverse_scale)
        _finite_array(reverse_scale, "reverse scale")
        np.add(scale, reverse_scale, out=scale)
        _finite_array(scale, "pre-cancellation scale")
        np.multiply(scale, 256 * _EPS * dimension, out=allowance)
        _finite_array(allowance, "comparison allowance")
        norm = 0.0
        for value in commutator.flat:
            norm = _finite_scalar(hypot(norm, value.real, value.imag),
                                  "finite Frobenius norm")
        passed = all(abs(value.real) <= tolerance and abs(value.imag) <= tolerance
                     for value, tolerance in zip(commutator.flat, allowance.flat))
    conserved = operator["exact_conservation"]
    diagnostic = (("within_heuristic_allowance" if passed else "outside_heuristic_allowance")
                  if conserved else "not_applicable_nonconserved")
    witness = operator["witness"]
    if witness is not None:
        witness = _witness_report(L, commutator, scale, witness["row"],
                                  witness["column"], witness["expected"])
    return {"name": operator["name"], "support_class": operator["support_class"],
            "exact_conservation": bool(conserved),
            "commutator": {"shape": list(commutator.shape),
                           "real": commutator.real.tolist(),
                           "imag": commutator.imag.tolist()},
            "finite_frobenius_norm": float(norm),
            "conservation_diagnostic": diagnostic, "witness": witness}


def _case_report(owned):
    L, g, dimension = owned["L"], owned["g"], owned["dimension"]
    H = _array(owned["H"], (dimension, dimension), "owned Hamiltonian")
    return {"L": L, "g": g, "C": 1, "cutoff": 2, "dimension": dimension,
            "basis_order": [list(state) for state in owned["basis"]],
            "status": "available_diagnostic",
            "operators": [_operator_report(L, H, operator) for operator in owned["operators"]],
            "scope": {"finite_compression_only": True,
                      "numerical_error_certified": False,
                      "empirical_validation": False,
                      "infinite_theorem_from_numerics": False,
                      "description": "Complete sectors N=0,1,2; exact labels are algebraic, not norm-sign inferences.",
                      "global_controls": "Total-number parity is globally conserved and scalar within each fixed sector; reflection fixes vertex 1 and may fix additional vertices.",
                      "comparison": "Componentwise absolute residuals use 256*u*D times the entrywise pre-cancellation scale; source rounding is not certified."}}


def case_report(L, g):
    """Construct one fixed C=1, K=2 control; availability is not agreement."""
    L = _integer(L, "L", 3, 5)
    g = _integer(g, "g", 0, 1)
    _preflight(L)
    return _case_report(_owned_case(L, g))


def demonstration_report():
    """Retain all six predetermined slots; never retry or replace a failure."""
    cases = []
    for L in (3, 4, 5):
        for g in (0, 1):
            try:
                report = case_report(L, g)
            except NumericalUnavailable as exc:
                cases.append({"L": L, "g": g, "status": "numerical_unavailable",
                              "reason": str(exc) or "numerical arithmetic unavailable", "report": None})
            else:
                cases.append({"L": L, "g": g, "status": "available_diagnostic",
                              "reason": None, "report": report})
    intervals = [interval_support_report(L, 0, length)
                 for L in (3, 4, 5) for length in range(1, L + 1)]
    intervals.extend(interval_support_report(*args)
                     for args in ((3, 2, 1), (4, 3, 2), (5, 4, 3)))
    return {"module": MODEL_ID, "cases": cases, "interval_controls": intervals,
            "limitations": list(LIMITATIONS)}
