"""Diagonal Bose constraints and multiplication in the inherited hard-core window.

See doc/derivations/substrate_constraint_algebra_2026-09-12.md. Graph statements
are combinatorial. Binary64 matrix residuals are diagnostics, not exact-input
certificates or certified roundoff bounds. The two ambient spaces are distinct.
"""
from functools import wraps
from math import comb
from numbers import Integral

import numpy as np

from .substrate_fermionization import fixed_number_model
from . import substrate_quantum_matching as matching

MODEL_ID = "conditional-substrate-constraint-algebra-v1"
MAX_DENSE_DIMENSION = 512
MIN_COMPONENT = 2.0 ** -128
MAX_COMPONENT = 2.0 ** 20
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny
FROZEN_CONSTRAINT_CASES = tuple(
    (L, N, 1.0, g) for L in (3, 4, 5) for N in sorted({1, 2, L})
    for g in (0.7, 40.0)
)
FROZEN_PROJECTION_CASES = tuple(
    (L, N, 1.0) for L in (3, 4, 5) for N in (1, 2, 3)
) + ((4, 4, 1.0), (5, 5, 1.0))
LIMITATIONS = (
    "The complete Bose ring is stipulated; the diagonal commutant is not the full commutant of H.",
    "The hard-core window has a distinct ambient space and no interaction parameter g.",
    "A scalar in a fixed-number sector is not a nontrivial local conserved constraint.",
    "An empty window is not a physical encoding; commutativity alone does not establish multiplication closure.",
    "No gauge field, Gauss-operator embedding or equivalence to the separate finite-group model is derived.",
    "These obstructions are not a universal no-go theorem for emergent gauge constructions.",
    "Binary64 residuals, ranks and norm bounds are diagnostics, not exact-input or roundoff certificates.",
)
NUMERICAL_DOMAIN = {
    "maximum_dimension": MAX_DENSE_DIMENSION,
    "minimum_nonzero_component": MIN_COMPONENT,
    "maximum_component": MAX_COMPONENT,
    "arithmetic": "guarded complex128; finite normal components or zero",
    "isometry_norm": "Frobenius",
    "isometry_tolerance": "64*binary64_eps*max(d,r,1)",
    "exactness_certificate": False,
}


class NumericalUnavailable(ValueError):
    """Valid supported inputs whose computed binary64 result is unavailable."""


def _guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise NumericalUnavailable("unresolved floating-point arithmetic") from exc
    return checked


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if not low <= value <= high:
        raise ValueError(f"{name} outside [{low},{high}]")
    return int(value)


def _components(value, name):
    """Check native numeric components before narrowing their storage."""
    parts = (value.real, value.imag) if np.iscomplexobj(value) else (value,)
    for part in parts:
        if (not np.all(np.isfinite(part)) or np.any(part > MAX_COMPONENT)
                or np.any(part < -MAX_COMPONENT)
                or np.any((part > 0) & (part < MIN_COMPONENT))
                or np.any((part < 0) & (part > -MIN_COMPONENT))):
            raise ValueError(f"{name} components must be zero or have magnitude in [2^-128,2^20]")


def _numeric_scalar(value, name):
    # Do not execute a caller-defined __array__, __float__ or __complex__.
    builtin = type(value) in (int, float, complex)
    numpy_number = isinstance(value, np.generic) and value.dtype.kind in "iufc"
    if not (builtin or numpy_number):
        raise ValueError(f"{name} entries must be numeric, not boolean/string/object")
    if type(value) is int and abs(value) > MAX_COMPONENT:
        raise ValueError(f"{name} component exceeds supported magnitude")
    _components(value, name)


def _real(value, name, *, positive=False):
    _numeric_scalar(value, name)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = float(value)
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _shape(value, name):
    """Bound all shapes without conversion or caller array protocols."""
    if type(value) is np.ndarray:
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        shape = value.shape
    elif type(value) in (list, tuple):
        if not 1 <= len(value) <= MAX_DENSE_DIMENSION:
            raise ValueError(f"{name} must have 1..512 rows")
        width = None
        for row in value:
            if type(row) is np.ndarray:
                if row.ndim != 1:
                    raise ValueError(f"{name} rows must be one-dimensional")
            elif type(row) not in (list, tuple):
                raise ValueError(f"{name} must contain bounded numeric rows")
            if len(row) > MAX_DENSE_DIMENSION:
                raise ValueError(f"{name} exceeds dimension cap512")
            if width is not None and len(row) != width:
                raise ValueError(f"{name} rows must have equal lengths")
            width = len(row)
        shape = (len(value), width)
    else:
        raise ValueError(f"{name} must be an ndarray or bounded list/tuple")
    if not 1 <= shape[0] <= MAX_DENSE_DIMENSION or not 0 <= shape[1] <= MAX_DENSE_DIMENSION:
        raise ValueError(f"{name} exceeds dimension cap512")
    return shape


def _matrix(value, name):
    # _shape for ALL operands has already run before reaching this conversion.
    rows = (value,) if type(value) is np.ndarray else value
    for row in rows:
        if type(row) is np.ndarray:
            if row.dtype.kind not in "iufc":
                raise ValueError(f"{name} dtype must be numeric, not boolean/string/object")
            _components(row, name)
        else:
            for scalar in row:
                _numeric_scalar(scalar, name)
    return np.array(value, dtype=np.complex128, copy=True)


def _normal(value, name):
    a = np.asarray(value)
    for part in (a.real, a.imag) if np.iscomplexobj(a) else (a,):
        if not np.all(np.isfinite(part)):
            raise NumericalUnavailable(f"nonfinite computed {name}")
        if np.any((part != 0) & (np.abs(part) < _TINY)):
            raise NumericalUnavailable(f"subnormal computed {name}")
    return value


def _product(left, right):
    return _normal(left @ right, "matrix product")


def _norm(matrix, order=2):
    """Scaled operator 2-norm or Frobenius norm, with explicit empty handling."""
    _normal(matrix, "norm input")
    if matrix.size == 0:
        return 0.0
    scale = float(max(np.max(np.abs(matrix.real)), np.max(np.abs(matrix.imag))))
    if not scale:
        return 0.0
    normalized = _normal(matrix / scale, "normalized norm input")
    result = float(_normal(scale * np.linalg.norm(normalized, ord=order), "norm"))
    if result == 0:
        raise NumericalUnavailable("nonzero matrix norm underflowed")
    return result


def _readonly(array):
    # A bytes-backed copy is detached and cannot be made writeable again.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _json(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "real": value.real.tolist(), "imag": value.imag.tolist()}
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def graph_components(vertex_count, edges):
    """Exact connected components of a bounded simple undirected graph.

    Components and vertices are sorted by their least vertex and vertex number;
    edges are canonical sorted pairs. Isolated vertices are retained.
    """
    n = _integer(vertex_count, "vertex_count", 1, MAX_DENSE_DIMENSION)
    if type(edges) not in (list, tuple) or len(edges) > n * (n - 1) // 2:
        raise ValueError("edges must be a bounded list/tuple of simple undirected edges")
    pairs = set()
    adjacency = [set() for _ in range(n)]
    for edge in edges:
        if type(edge) not in (list, tuple) or len(edge) != 2:
            raise ValueError("each edge must be a list/tuple of two integer endpoints")
        a, b = (_integer(endpoint, "endpoint", 0, n - 1) for endpoint in edge)
        pair = (min(a, b), max(a, b))
        if a == b or pair in pairs:
            raise ValueError("self loops and duplicate unoriented edges are forbidden")
        pairs.add(pair)
        adjacency[a].add(b)
        adjacency[b].add(a)
    labels = [-1] * n
    components = []
    for root in range(n):
        if labels[root] != -1:
            continue
        label = len(components)
        labels[root] = label
        stack, component = [root], []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in sorted(adjacency[vertex]):
                if labels[neighbor] == -1:
                    labels[neighbor] = label
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return {"vertex_count": n, "edges": tuple(sorted(pairs)),
            "components": tuple(components), "labels": tuple(labels),
            "component_count": len(components)}


def _bose_parameters(L, N):
    L = _integer(L, "L", 3, 12)
    N = _integer(N, "N", 0, L)
    if comb(L + N - 1, N) > MAX_DENSE_DIMENSION:
        raise ValueError("complete Bose sector exceeds dense dimension512")
    return L, N


def _sector_graph(L, N, basis):
    index = {state: i for i, state in enumerate(basis)}
    edges = set()
    for i, state in enumerate(basis):
        for x in range(L):
            for y in ((x - 1) % L, (x + 1) % L):
                if state[x]:
                    moved = list(state)
                    moved[x] -= 1
                    moved[y] += 1
                    j = index[tuple(moved)]
                    edges.add((min(i, j), max(i, j)))
    graph = graph_components(len(basis), tuple(edges))
    graph.update({"L": L, "N": N, "dimension": len(basis), "basis": basis,
                  "ambient": "complete_bose_fixed_number",
                  "diagonal_commutant_dimension": graph["component_count"]})
    return graph


@_guard
def sector_graph(L, N):
    """Exact occupation-move graph, independent of hopping numerical thresholds."""
    L, N = _bose_parameters(L, N)
    model = fixed_number_model(L, N, C=1.0, g=0.0)
    return _sector_graph(L, N, model.basis)


@_guard
def constraint_report(L, N, C=1.0, g=0.7):
    """JSON-safe classification only of occupation-diagonal conserved operators."""
    L, N = _bose_parameters(L, N)
    C, g = _real(C, "C", positive=True), _real(g, "g")
    model = fixed_number_model(L, N, C, g)
    _normal(model.H, "canonical Bose Hamiltonian")
    graph = _sector_graph(L, N, model.basis)
    projectors = []
    for site in range(L):
        for q in (2, 3):
            for residue in range(q):
                values = np.array([state[site] % q == residue for state in model.basis], dtype=float)
                status = "identity" if np.all(values) else ("zero" if not np.any(values) else "nonconstant")
                commutator = _normal(model.H * (values[None, :] - values[:, None]), "commutator")
                witness = next((edge for edge in graph["edges"] if values[edge[0]] != values[edge[1]]), None)
                projectors.append({"site": site, "q": q, "residue": residue,
                                   "status": status, "conserved": status != "nonconstant",
                                   "commutator_frobenius_norm": _norm(commutator, "fro"),
                                   "crossing_edge": witness})
    controls = [{"name": "identity", "value": 1, "status": "scalar", "conserved": True,
                 "commutator_frobenius_norm": 0.0},
                {"name": "global_number", "value": N, "status": "scalar", "conserved": True,
                 "commutator_frobenius_norm": 0.0}]
    for q in (2, 3):
        for residue in range(q):
            controls.append({"name": "global_number_residue", "q": q, "residue": residue,
                             "value": int(N % q == residue), "status": "scalar", "conserved": True,
                             "commutator_frobenius_norm": 0.0})
    return _json({"model_id": MODEL_ID, "source_model": "fixed_number_model",
                  "ambient": "complete_bose_fixed_number", "L": L, "N": N, "C": C, "g": g,
                  "dimension": len(model.basis), "basis": model.basis, "basis_order": "lexicographic occupations",
                  "graph": graph, "diagonal_commutant_dimension": graph["component_count"],
                  "classification": "diagonal F commutes iff constant on each graph component",
                  "structural_status": "singleton_vacuum" if N == 0 else "connected_ring",
                  "onsite_projectors": projectors, "scalar_controls": controls,
                  "commutator_norm": "Frobenius", "limitations": LIMITATIONS})


@_guard
def multiplication_defect(A, B, W):
    """Audit compression without replacing a nearly isometric W by an isometry.

    For ANY W, D=W†ABW-AcBc=W†AQBW. With E=W†W-I and K_X=QXW,
    K_(A†)†K_B = D + Ac E Bc. Consequently the finite-W triangle bound
    adds ||Ac||2 ||E||2 ||Bc||2 to the pure leakage product. The sharper
    ||Ac E Bc||2 correction norm is also reported. Agreements are numerical.
    """
    ashape, bshape, wshape = _shape(A, "A"), _shape(B, "B"), _shape(W, "W")
    d, r = wshape
    if ashape != (d, d) or bshape != (d, d) or r > d:
        raise ValueError("A,B must be common square ambient matrices and W must have shape(d,r), r<=d")
    A, B, W = _matrix(A, "A"), _matrix(B, "B"), _matrix(W, "W")
    adjoint = W.conj().T
    error = _normal(_product(adjoint, W) - np.eye(r), "isometry error")
    residual = _norm(error, "fro")
    tolerance = float(64 * _EPS * max(d, r, 1))
    if residual > tolerance:
        raise ValueError("W fails the dimension-scaled Frobenius isometry tolerance")
    Q = _normal(np.eye(d) - _product(W, adjoint), "complement")
    AW, BW = _product(A, W), _product(B, W)
    Ac, Bc = _product(adjoint, AW), _product(adjoint, BW)
    direct = _normal(_product(adjoint, _product(A, BW)) - _product(Ac, Bc), "direct defect")
    complement = _product(_product(adjoint, A), _product(Q, BW))
    ka = _product(Q, _product(A.conj().T, W))
    kb = _product(Q, BW)
    gram = _product(ka.conj().T, kb)
    correction = _product(_product(Ac, error), Bc)
    corrected = _normal(gram - correction, "corrected Gram defect")
    leakage_bound = float(_normal(_norm(ka) * _norm(kb), "leakage product bound"))
    correction_norm = _norm(correction)
    correction_bound = float(_normal(_norm(Ac) * _norm(error) * _norm(Bc), "Gram correction bound"))
    result = {
        "compressed_A": Ac, "compressed_B": Bc,
        "direct_defect": direct, "complement_defect": complement,
        "gram_factor": gram, "gram_correction": correction,
        "corrected_gram_defect": corrected,
        "leakage_adjoint_A": ka, "leakage_B": kb,
        "isometry_residual": residual, "isometry_tolerance": tolerance,
        "direct_complement_residual": _norm(_normal(direct - complement, "direct agreement"), "fro"),
        "corrected_gram_residual": _norm(_normal(direct - corrected, "corrected Gram agreement"), "fro"),
        "defect_operator_norm": _norm(direct), "leakage_product_bound": leakage_bound,
        "gram_correction_operator_norm": correction_norm,
        "gram_correction_bound": correction_bound,
        "finite_isometry_bound": float(_normal(leakage_bound + correction_bound, "finite isometry bound")),
        "leakage_adjoint_A_operator_norm": _norm(ka), "leakage_B_operator_norm": _norm(kb),
        "status": "empty_candidate" if r == 0 else ("full_window" if r == d else "proper_window"),
        "ambient_dimension": d, "rank": r,
        "residual_norm": "Frobenius", "defect_and_bound_norm": "operator_2",
        "pure_leakage_bound_requires": "exact W†W=I (not certified by a floating-point residual)",
        "numerical_status": "diagnostic_only", "exactness_certificate": False,
    }
    return {key: _readonly(value) if isinstance(value, np.ndarray) else value for key, value in result.items()}


@_guard
def projection_report(L, N, C=1.0):
    """All ordered density pairs in a fresh canonical hard-core three-mode window."""
    L = _integer(L, "L", 3, matching.MAX_SITES)
    N = _integer(N, "N", 0, L)
    C = _real(C, "C", positive=True)
    # Never trust a shallow-frozen caller model or its mutable arrays/caches.
    model = matching.projected_model(L, N, C)
    W, d = model.W, len(model.bits)
    pairs = []
    for x in range(L):
        for y in range(L):
            audit = multiplication_defect(model.densities[x], model.densities[y], W)
            ac, bc = audit["compressed_A"], audit["compressed_B"]
            compressed_commutator = _normal(_product(ac, bc) - _product(bc, ac), "compressed commutator")
            reverse = _normal(_product(W.conj().T, _product(model.densities[y], _product(model.densities[x], W)))
                              - _product(bc, ac), "reverse defect")
            identity_residual = _normal(compressed_commutator + audit["direct_defect"] - reverse,
                                        "compressed commutator identity")
            # Independent canonical omitted Slater states, not I-WW†.
            left = _product(model.complement.conj().T, _product(model.densities[x], W))
            right = _product(model.complement.conj().T, _product(model.densities[y], W))
            contraction = _product(left.conj().T, right)
            row = {"sites": [x, y], "audit": audit,
                   "compressed_commutator": compressed_commutator,
                   "compressed_commutator_operator_norm": _norm(compressed_commutator),
                   "commutator_identity_residual": _norm(identity_residual, "fro"),
                   "independent_complement_defect": contraction,
                   "independent_complement_residual": _norm(_normal(audit["direct_defect"] - contraction,
                                                                       "independent complement agreement"), "fro")}
            if x == y and 1 <= N <= 3:
                p = 3.0 / L
                target = _normal((1 - p) * ac, "density target")
                row["reviewed_density_target"] = {
                    "defect_operator_norm": p * (1 - p),
                    "leakage_operator_norm": float(np.sqrt(p * (1 - p))),
                    "identity_residual": _norm(_normal(audit["direct_defect"] - target, "density identity"), "fro"),
                }
            pairs.append(row)
    identity = np.eye(d)
    controls = [{"name": name, "scalar": scalar,
                 "structural_status": "scalar_preserves_exact_window",
                 "audit": multiplication_defect(scalar * identity, identity, W)}
                for name, scalar in (("zero", 0.0), ("identity", 1.0), ("global_number", float(N)))]
    witness = None
    if L == 4 and N in (1, 2):
        pair = next(row for row in pairs if row["sites"] == [0, 1])
        witness = {"sites": [0, 1], "expected_commutator_operator_norm": float(np.sqrt(2) / 8),
                   "actual_commutator_operator_norm": pair["compressed_commutator_operator_norm"],
                   "basis_note": "original mode order (-1,0,1)" if N == 1 else
                                 "N2 norm follows signed-complement conjugation; no original-basis entry asserted"}
        if N == 1:
            witness["expected_first_diagonal_entry"] = {"real": 0.0, "imag": -0.125}
            entry = pair["compressed_commutator"][0, 0]
            witness["actual_first_diagonal_entry"] = {"real": float(entry.real), "imag": float(entry.imag)}
    rank = W.shape[1]
    status = "empty_candidate" if rank == 0 else ("full_window" if rank == d else "proper_window")
    summed_density = _normal(sum(model.projected_densities, np.zeros((rank, rank), complex)), "total density")
    finite_number = _normal(N * _product(W.conj().T, W), "finite-W number")
    number_diagnostics = {
        "ideal_number_frobenius_residual": _norm(_normal(summed_density - N * np.eye(rank), "ideal number residual"), "fro"),
        "finite_W_number_frobenius_residual": _norm(_normal(summed_density - finite_number, "finite number residual"), "fro"),
        "exact_identity": "sum_x Bx=N W†W; equals N I only for exact isometry",
    }
    return _json({"model_id": MODEL_ID, "source_model": matching.MODEL_ID,
                  "ambient": "hard_core_fixed_number", "L": L, "N": N, "C": C,
                  "dimension": d, "rank": rank, "status": status,
                  "structural_status": "empty_not_encoding" if rank == 0 else
                                       ("full_space_closure" if rank == d else "proper_three_mode_window"),
                  "basis_bits": model.bits, "basis_order": "ascending binary integers",
                  "modes": matching.MODES, "mode_subsets": model.mode_subsets, "W": W,
                  "density_pairs": pairs, "scalar_controls": controls,
                  "total_number_diagnostics": number_diagnostics,
                  "predetermined_witness": witness, "limitations": LIMITATIONS})


def demonstration_report():
    """Frozen complete-Bose and hard-core grids, plus explicit vacuum controls."""
    return {
        "model_id": MODEL_ID,
        "constraint_cases": [constraint_report(*case) for case in FROZEN_CONSTRAINT_CASES],
        "projection_cases": [projection_report(*case) for case in FROZEN_PROJECTION_CASES],
        "vacuum_controls": {
            "complete_bose": constraint_report(3, 0),
            "hard_core": projection_report(3, 0),
        },
        "numerical_domain": dict(NUMERICAL_DOMAIN),
        "limitations": list(LIMITATIONS),
    }
