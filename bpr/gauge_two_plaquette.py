"""Bounded coupled two-face extension of the central finite-group gauge model.

Counting-measure coordinates are pairs of based holonomies. Simultaneous
conjugation orbits give the physical basis; no seven-link tensor is formed.
The even-n reflection class is inherited unchanged from gauge_heat_kernel.
This is a supplied dimensionless model, not a substrate-derived gauge theory.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from numbers import Integral, Real

import numpy as np

from .gauge_heat_kernel import character_table, elements, heat_kernel, inv, mul, reflection_class

MODEL_ID = "central-class-two-plaquette-v1"
ALLOWED_N = (5, 8, 9)
COORDINATE_CAP = 324
VERTICES = ("A", "B", "Ltop", "Lbot", "Rbot", "Rtop")
EDGES = ("a", "b", "c", "d", "e", "f", "h")
ENDPOINTS = {
    "a": ("A", "Ltop"), "b": ("Ltop", "Lbot"),
    "c": ("Lbot", "B"), "d": ("A", "B"),
    "e": ("B", "Rbot"), "f": ("Rbot", "Rtop"),
    "h": ("Rtop", "A"),
}
TREE_EDGES = ("a", "b", "d", "e", "f")
IDENTITY = (0, 1)


def _group(n: int) -> int:
    # Check the bound before calling an inherited enumerator or allocating.
    if isinstance(n, bool) or not isinstance(n, Integral) or n not in ALLOWED_N:
        raise ValueError("n must be one of (5, 8, 9)")
    n = int(n)
    if (2 * n) ** 2 > COORDINATE_CAP:
        raise ValueError("two-holonomy coordinate dimension exceeds cap324")
    return n


def _real(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real number")
    if value < 0:
        raise ValueError(name + " must be nonnegative")
    original_nonzero = value != 0
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " must be a finite real number") from exc
    if original_nonzero and value == 0:
        raise FloatingPointError(name + " loses a nonzero scale in float64 conversion")
    if not np.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(name + " must be finite and " + ("nonnegative" if zero else "positive"))
    if value != 0 and value < np.finfo(float).tiny:
        raise FloatingPointError(name + " is below the normal float64 range")
    return value


def _element(n: int, value: tuple, name: str) -> tuple:
    if (not isinstance(value, tuple) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in value)
            or not 0 <= value[0] < n or value[1] not in (-1, 1)):
        raise ValueError(name + " must be a canonical (k, +/-1) element of D_n")
    return (int(value[0]), int(value[1]))


def _mapping(n: int, values: Mapping, keys: tuple, name: str) -> dict:
    if not isinstance(values, Mapping) or set(values) != set(keys):
        raise ValueError(name + " must map exactly the declared names to group elements")
    return {key: _element(n, values[key], name + "[" + key + "]") for key in keys}


def canonical_links(n: int, u: tuple, v: tuple) -> dict:
    """Tree links are identity, c=u, h=v, in the frozen edge order."""
    n = _group(n)
    u, v = _element(n, u, "u"), _element(n, v, "v")
    return {edge: u if edge == "c" else v if edge == "h" else IDENTITY for edge in EDGES}


def holonomies(n: int, links: Mapping) -> tuple:
    """Based u=abc d^-1 and v=defh; their product is the outer loop."""
    n = _group(n)
    links = _mapping(n, links, EDGES, "links")
    u = mul(mul(mul(links["a"], links["b"], n), links["c"], n), inv(links["d"], n), n)
    v = mul(mul(mul(links["d"], links["e"], n), links["f"], n), links["h"], n)
    return u, v


def gauge_transform(n: int, links: Mapping, vertex_values: Mapping) -> dict:
    """Apply independently specified endpoint actions U_xy -> t_x U_xy t_y^-1."""
    n = _group(n)
    links = _mapping(n, links, EDGES, "links")
    values = _mapping(n, vertex_values, VERTICES, "vertex_values")
    return {
        edge: mul(mul(values[x], links[edge], n), inv(values[y], n), n)
        for edge, (x, y) in ENDPOINTS.items()
    }


def tree_fix(n: int, links: Mapping) -> dict:
    """Return fixed links and the root-identity vertex transporters used."""
    n = _group(n)
    links = _mapping(n, links, EDGES, "links")
    paths = {"A": IDENTITY}
    for edge in TREE_EDGES:
        x, y = ENDPOINTS[edge]
        paths[y] = mul(paths[x], links[edge], n)
    values = {vertex: paths[vertex] for vertex in VERTICES}
    return {"links": gauge_transform(n, links, values), "vertex_values": values}


def _slice_action(n: int, edge: str, g: tuple, u: tuple, v: tuple) -> tuple:
    if edge in ("a", "b", "c"):
        return mul(g, u, n), v
    if edge == "d":
        return mul(u, inv(g, n), n), mul(g, v, n)
    return u, mul(g, v, n)


def _edge(edge: str) -> str:
    if not isinstance(edge, str) or edge not in EDGES:
        raise ValueError("edge must be one of " + repr(EDGES))
    return edge


def link_action(n: int, edge: str, g: tuple, u: tuple, v: tuple) -> tuple:
    """Fixed-g action on the canonical slice, NOT an off-slice gauge operator.

    The shared edge action (u g^-1, g v) preserves the outer product u v.
    Only the central weighted sums descend to physical class functions.
    """
    n, edge = _group(n), _edge(edge)
    g, u, v = (_element(n, x, name) for x, name in ((g, "g"), (u, "u"), (v, "v")))
    return _slice_action(n, edge, g, u, v)


def orbit_basis(n: int) -> dict:
    """Orbit indicators normalized in counting measure; deterministic row order."""
    n = _group(n)
    els = tuple(elements(n))
    pairs = tuple(product(els, repeat=2))
    index = {pair: row for row, pair in enumerate(pairs)}
    unseen, orbits = set(range(len(pairs))), []
    while unseen:
        u, v = pairs[min(unseen)]
        rows = tuple(sorted({
            index[(mul(mul(g, u, n), inv(g, n), n), mul(mul(g, v, n), inv(g, n), n))]
            for g in els
        }))
        orbits.append(rows)
        unseen.difference_update(rows)
    basis = np.zeros((len(pairs), len(orbits)))
    for column, rows in enumerate(orbits):
        basis[list(rows), column] = 1 / np.sqrt(len(rows))
    fixed_sum = sum(sum(mul(g, h, n) == mul(h, g, n) for h in els) ** 2 for g in els)
    if fixed_sum % len(els):
        raise ArithmeticError("Burnside fixed-point sum is not divisible by group order")
    return {
        "n": n, "elements": els, "pairs": pairs, "orbits": tuple(orbits),
        "basis": basis, "count": len(orbits), "burnside_count": fixed_sum // len(els),
    }


def edge_laplacian(n: int, edge: str) -> np.ndarray:
    """Unscaled central generator for one graph link, acting on columns."""
    n, edge = _group(n), _edge(edge)
    els = tuple(elements(n))
    pairs = tuple(product(els, repeat=2))
    index = {pair: row for row, pair in enumerate(pairs)}
    cl = reflection_class(n)
    terms = [((1, 1), 1.0), (inv((1, 1), n), 1.0)]
    terms.extend((g, 1.0 / len(cl)) for g in cl)
    laplacian = 3 * np.eye(len(pairs))
    for column, (u, v) in enumerate(pairs):
        for g, weight in terms:
            laplacian[index[_slice_action(n, edge, g, u, v)], column] -= weight
    return laplacian


def _normal_array(value: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(name + " exceeds finite float64 range")
    absolute = np.abs(value)
    if np.any((absolute != 0) & (absolute < np.finfo(float).tiny)):
        raise FloatingPointError(name + " contains unresolved subnormal values")
    return value


def model(n: int, lam: float = 1.3) -> dict:
    """Build graph-derived E and unscaled magnetic M, with H=E+lam*M.

    Shared-edge central convolution is included once, each outside face action
    three times. This is not a sum of two isolated-square electric operators.
    """
    n, lam = _group(n), _real(lam, "lam")
    data = orbit_basis(n)
    basis, pairs = data["basis"], data["pairs"]
    du, dv, shared = (edge_laplacian(n, edge) for edge in ("a", "e", "d"))
    chi = character_table(n)["E1"]
    potential = np.array([2 - chi[u] / 2 - chi[v] / 2 for u, v in pairs])
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        coordinate_electric = _normal_array((3 * du + 3 * dv + shared) / lam, "electric")
        coordinate_magnetic = np.diag(potential)
        coordinate_hamiltonian = _normal_array(
            coordinate_electric + lam * coordinate_magnetic, "Hamiltonian")
        electric = _normal_array(basis.T @ coordinate_electric @ basis, "physical electric")
        # Each orbit is a level set of the class-function magnetic potential.
        magnetic_diagonal = potential[[rows[0] for rows in data["orbits"]]]
        magnetic = np.diag(magnetic_diagonal)
        hamiltonian = _normal_array(electric + lam * magnetic, "physical Hamiltonian")
    scale = float(np.linalg.norm(hamiltonian, ord=np.inf))
    resolution = 64 * np.finfo(float).eps * data["count"] * scale
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        trial_bound = min(2 * lam, 21 / lam)
    if not np.isfinite(scale) or resolution >= trial_bound:
        raise FloatingPointError("Hamiltonian energy scales are numerically unresolved at this lam")
    data.update({
        "model_id": MODEL_ID, "lam": lam,
        "coordinate_dimension": len(pairs), "physical_dimension": data["count"],
        "reflection_class": reflection_class(n),
        "laplacian_u": du, "laplacian_v": dv, "laplacian_shared": shared,
        "coordinate_electric": coordinate_electric, "coordinate_magnetic": coordinate_magnetic,
        "coordinate_hamiltonian": coordinate_hamiltonian,
        "electric": electric, "magnetic": magnetic, "hamiltonian": hamiltonian,
        "magnetic_diagonal": magnetic_diagonal,
    })
    return data


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Bound shape before coercing an array-like, then validate its components."""
    shape = getattr(value, "shape", None)
    if shape is None:
        if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= COORDINATE_CAP
                or any(not isinstance(row, (list, tuple, np.ndarray))
                       or len(row) != len(value) for row in value)):
            raise ValueError(name + " must be a square matrix of size 1..324")
    elif (len(shape) != 2 or shape[0] != shape[1]
          or not 1 <= shape[0] <= COORDINATE_CAP):
        raise ValueError(name + " must be a square matrix of size 1..324")
    array = np.asarray(value)
    if array.dtype.kind not in "iufc":
        raise ValueError(name + " must contain real or complex numbers, not booleans or objects")
    if not np.all(np.isfinite(array)):
        raise ValueError(name + " must contain finite numbers")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.asarray(array, dtype=complex if np.iscomplexobj(array) else float)
    for before, after in ((array.real, converted.real), (array.imag, converted.imag)):
        _normal_array(after, name)
        if np.any((before != 0) & (after == 0)):
            raise FloatingPointError(name + " loses components in float64 conversion")
    return converted


def _hermitian(value: np.ndarray, name: str) -> tuple:
    matrix = _matrix(value, name)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = float(np.linalg.norm(matrix, ord=np.inf))
    _normal_array(np.asarray(scale), name + " scale")
    with np.errstate(over="ignore", invalid="ignore"):
        defect = float(np.linalg.norm(matrix - matrix.conj().T, ord=np.inf))
    if not np.isfinite(defect) or defect > 64 * np.finfo(float).eps * len(matrix) * scale:
        raise ValueError(name + " must be Hermitian")
    return matrix, scale


def _positive_spectrum(value: np.ndarray) -> tuple:
    matrix, _ = _hermitian(value, "transfer")
    values, vectors = np.linalg.eigh(matrix)
    scale = float(np.max(np.abs(values)))
    resolution = 64 * np.finfo(float).eps * len(values) * scale
    _normal_array(np.array([scale, resolution]), "transfer spectral resolution")
    if float(values[0]) <= resolution:
        raise FloatingPointError("transfer is nonpositive or numerically unresolved; reduce dt")
    return matrix, values, vectors, resolution


def effective_hamiltonian(T: np.ndarray, dt: float) -> np.ndarray:
    """Hermitian -log(T)/dt on resolved positive matrices of size at most324.

    This general logarithm does not require T<=I: positive eigenvalues above1
    correctly give negative energies. No eigenvalue is clipped or floored.
    """
    dt = _real(dt, "dt")
    matrix, values, vectors, _ = _positive_spectrum(T)
    if np.array_equal(matrix, np.eye(len(matrix))):
        return np.zeros_like(matrix)
    # A normwise-resolved positive spectrum can still erase tiny active blocks
    # near a scalar matrix. In particular eigh(I+1e-18*sigma_x) may return only
    # unit eigenvalues. Test supplied nonzero components of T-I separately:
    # no global tolerance may hide them behind an unrelated spectator block.
    # The relative check is a conservative support policy, not a roundoff proof.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        deviation = matrix - np.eye(len(matrix))
        reconstructed = (vectors * (values - 1)) @ vectors.conj().T
    for supplied, recovered in ((deviation.real, reconstructed.real),
                                (deviation.imag, reconstructed.imag)):
        active = supplied != 0
        if np.any(np.abs(recovered[active] - supplied[active]) > 1e-6 * np.abs(supplied[active])):
            raise FloatingPointError("transfer has numerically unresolved nonzero logarithm components")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        logs = np.log(values)
        energies = _normal_array(-logs / dt, "effective energies")
        if np.any((logs != 0) & (energies == 0)):
            raise FloatingPointError("effective energies lose a nonzero scale at this dt")
        result = (vectors * energies) @ vectors.conj().T
    _normal_array(result.real, "effective Hamiltonian real components")
    _normal_array(result.imag, "effective Hamiltonian imaginary components")
    if np.any(energies != 0) and not np.any(result != 0):
        raise FloatingPointError("effective Hamiltonian loses its nonzero scale")
    return result


def _model_operators(data: Mapping) -> tuple:
    if not isinstance(data, Mapping):
        raise ValueError("model must be the mapping returned by model(n,lam)")
    required = ("model_id", "n", "lam", "electric", "magnetic", "hamiltonian")
    if any(key not in data for key in required) or data["model_id"] != MODEL_ID:
        raise ValueError("model mapping is missing the two-plaquette operator specification")
    n, lam = _group(data["n"]), _real(data["lam"], "lam")
    electric, _ = _hermitian(data["electric"], "electric")
    magnetic, _ = _hermitian(data["magnetic"], "magnetic")
    hamiltonian, _ = _hermitian(data["hamiltonian"], "Hamiltonian")
    dimension = {5: 22, 8: 64, 9: 56}[n]
    if any(matrix.shape != (dimension, dimension) for matrix in (electric, magnetic, hamiltonian)):
        raise ValueError("model operators have the wrong physical dimension")
    if (np.iscomplexobj(magnetic) and np.any(magnetic.imag != 0)
            or not np.array_equal(magnetic, np.diag(np.diag(magnetic)))
            or np.any(np.diag(magnetic).real < 0)):
        raise ValueError("model magnetic operator must be real nonnegative diagonal")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_magnetic = _normal_array(lam * magnetic.real, "scaled magnetic")
        expected = _normal_array(electric + scaled_magnetic, "Hamiltonian sum")
    scale = float(np.linalg.norm(expected, ord=np.inf))
    if np.linalg.norm(hamiltonian - expected, ord=np.inf) > 64 * np.finfo(float).eps * dimension * scale:
        raise ValueError("model Hamiltonian must equal electric+lam*magnetic")
    return electric, scaled_magnetic, hamiltonian


def _electric_heat_factor(n: int, lam: float, dt: float, edge: str, data: dict) -> np.ndarray:
    """Compress a central heat convolution, retaining its exact trivial weight."""
    weights = _normal_array(heat_kernel(n, lam, dt), "central heat weights")
    pairs, basis = data["pairs"], data["basis"]
    index = {pair: row for row, pair in enumerate(pairs)}
    coordinate = np.zeros((len(pairs), len(pairs)))
    for column, (u, v) in enumerate(pairs):
        for g, weight in zip(data["elements"], weights):
            coordinate[index[_slice_action(n, edge, g, u, v)], column] += weight
    return basis.T @ coordinate @ basis


def transfer(model_dict: Mapping, dt: float) -> np.ndarray:
    """Symmetric magnetic-half/electric/magnetic-half step, not exp(-dt H).

    Commuting central generators give electric heat factors at times3dt,3dt,dt.
    Their exact trivial-irrep attenuation is1; no rounded electric eigenvalue is
    exponentiated. Each coordinate convolution is compressed before multiplying.
    """
    dt = _real(dt, "dt", zero=True)
    electric, magnetic, hamiltonian = _model_operators(model_dict)
    n, lam = _group(model_dict["n"]), _real(model_dict["lam"], "lam")
    # Do not silently apply canonical heat factors to a modified operator dict.
    canonical = model(n, lam)
    if (not np.array_equal(electric, canonical["electric"])
            or not np.array_equal(magnetic, lam * canonical["magnetic"])):
        raise ValueError("model operators must match the declared canonical graph model")
    if dt == 0:
        return np.eye(len(electric))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        step_scale = dt * float(np.linalg.norm(hamiltonian, ord=np.inf))
    if not np.isfinite(step_scale) or step_scale <= 64 * np.finfo(float).eps * len(electric):
        raise FloatingPointError("transfer time scale is numerically unresolved")
    # Reject unresolvable attenuation before calling the inherited heat helper,
    # which has a wider underflow-tolerant contract. This is a conservative cap.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        worst_exponent = dt * (42 / lam + 4 * lam)
    if not np.isfinite(worst_exponent) or worst_exponent > -np.log(np.finfo(float).tiny):
        raise FloatingPointError("transfer attenuation is below the supported normal range")
    factors = [
        _electric_heat_factor(n, lam, multiplier * dt, edge, canonical)
        for multiplier, edge in ((3, "a"), (3, "e"), (1, "d"))
    ]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        magnetic_exponents = _normal_array(-dt * np.diag(magnetic) / 2, "magnetic exponents")
        magnetic_half = _normal_array(np.exp(magnetic_exponents), "magnetic attenuation")
        if np.any(magnetic_half == 0):
            raise FloatingPointError("transfer attenuation loses a nonzero scale")
        electric_step = factors[0] @ factors[1] @ factors[2]
        result = _normal_array(magnetic_half[:, None] * electric_step * magnetic_half[None, :], "transfer")
    _, eigenvalues, _, threshold = _positive_spectrum(result)
    if eigenvalues[-1] > 1 + threshold:
        raise FloatingPointError("model transfer contraction is numerically unresolved")
    return result


def _norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord=2))


def _positive_bound(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0 or value < np.finfo(float).tiny:
        raise FloatingPointError(name + " is not representable as a positive normal float64")
    return value


def diagnostics(n: int, lam: float = 1.3, steps: tuple = (0.08, 0.04, 0.02)) -> dict:
    """All requested steps, raw numerical residuals, and analytic sufficient bounds.

    Numerical tolerances are acceptance policies, not certified roundoff bounds.
    The analytic Strang/log bounds assume the exact positive operators.
    """
    n, lam = _group(n), _real(lam, "lam")
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError("steps must be a nonempty list or tuple of positive times")
    steps = tuple(_real(step, "step") for step in steps)
    data = model(n, lam)
    electric, magnetic, hamiltonian = _model_operators(data)
    a, b = _norm(electric), _norm(magnetic)
    commutator = electric @ magnetic - magnetic @ electric
    a_ab = _norm(electric @ commutator - commutator @ electric)
    b_ab = _norm(magnetic @ commutator - commutator @ magnetic)
    constant = a_ab / 12 + b_ab / 24
    _positive_bound(constant, "Strang bound constant")
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    energy_resolution = 64 * np.finfo(float).eps * len(eigenvalues) * _norm(hamiltonian)
    if eigenvalues[0] <= energy_resolution:
        raise FloatingPointError("Hamiltonian ground energy is numerically unresolved")
    records, effective_matrices = [], []
    for dt in steps:
        step = transfer(data, dt)
        _, values, _, resolution = _positive_spectrum(step)
        effective = effective_hamiltonian(step, dt)
        effective_matrices.append(effective)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            lower = _positive_bound(np.exp(-dt * (a + b)), "analytic transfer minimum")
            transfer_bound = _positive_bound(dt ** 3 * constant, "analytic transfer error bound")
            generator_bound = _positive_bound(dt ** 2 * np.exp(dt * (a + b)) * constant,
                                              "analytic generator error bound")
            exact_decay = _normal_array(np.exp(-dt * eigenvalues), "exact transfer attenuation")
            if np.any(exact_decay == 0):
                raise FloatingPointError("exact transfer attenuation loses a nonzero scale")
            exact = (eigenvectors * exact_decay) @ eigenvectors.conj().T
        records.append({
            "dt": dt, "transfer_min_eigenvalue": float(values[0]),
            "transfer_max_eigenvalue": float(values[-1]), "positivity_resolution": resolution,
            "condition_number": float(values[-1] / values[0]),
            "analytic_transfer_min_bound": lower,
            "transfer_hermiticity_error": _norm(step - step.conj().T),
            "transfer_exact_error": _norm(step - exact),
            "analytic_transfer_error_bound": transfer_bound,
            "effective_hamiltonian_error": _norm(effective - hamiltonian),
            "analytic_generator_error_bound": generator_bound,
        })
    errors = [record["effective_hamiltonian_error"] for record in records]
    ratios = [left / right if right > 0 else None for left, right in zip(errors, errors[1:])]
    differences = [_norm(left - right) for left, right in zip(effective_matrices, effective_matrices[1:])]
    bounds = [left["analytic_generator_error_bound"] + right["analytic_generator_error_bound"]
              for left, right in zip(records, records[1:])]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        a0, b0 = _positive_bound(42 / lam, "apriori electric norm"), _positive_bound(4 * lam, "apriori magnetic norm")
        c0 = _positive_bound(a0 * a0 * b0 / 3 + a0 * b0 * b0 / 6, "apriori Strang constant")
    return {
        "model_id": MODEL_ID, "n": n, "lam": lam, "steps": list(steps),
        "coordinate_dimension": data["coordinate_dimension"],
        "physical_dimension": data["physical_dimension"], "burnside_count": data["burnside_count"],
        "reflection_class": [list(g) for g in data["reflection_class"]],
        "electric_min_eigenvalue": float(np.linalg.eigvalsh(electric)[0]),
        "hamiltonian_energies": eigenvalues.tolist(),
        "hamiltonian_hermiticity_error": _norm(hamiltonian - hamiltonian.conj().T),
        "electric_hermiticity_error": _norm(electric - electric.conj().T),
        "orbit_orthonormality_error": _norm(data["basis"].T @ data["basis"] - np.eye(data["count"])),
        "commutator_norms": {"a_ab": a_ab, "b_ab": b_ab},
        "analytic_constant_C": constant, "electric_norm": a, "scaled_magnetic_norm": b,
        "apriori_bounds": {"electric_norm": a0, "scaled_magnetic_norm": b0, "analytic_constant_C": c0},
        "transfer_diagnostics": records, "error_ratios": ratios,
        "refinement_decreasing": all(left > right for left, right in zip(errors, errors[1:])),
        "refinement_differences": differences, "analytic_refinement_bounds": bounds,
    }


def demonstration_report() -> dict:
    """Frozen D5,D8,D9 controls, without changing parameters after evaluation."""
    return {
        "model_id": MODEL_ID,
        "scope": "dimensionless coupled two-plaquette central finite-group model",
        "controls": [diagnostics(n) for n in ALLOWED_N],
        "limitations": [
            "The Hamiltonian and Gauss structure are supplied, not derived from the Bose ring.",
            "No Standard Model group, continuum limit, physical matching, or empirical validation follows.",
            "Analytic sufficient bounds are distinct from floating-point residual diagnostics.",
        ],
    }
