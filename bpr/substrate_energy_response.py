"""Complete finite Bose-ring energy continuity and external-source response.

The two local partitions are conventions for supplied microscopic dynamics,
not a stress tensor, lapse, metric or gravitational theory.  See the frozen
2026-09-12 preregistration. Exact-input arithmetic below concerns binary64
inputs only; neither it nor the resolution screens certify eigenpairs.
"""
from fractions import Fraction
from math import comb
from numbers import Integral

import numpy as np

from bpr import substrate_current_response as current
from bpr.substrate_fermionization import FixedNumberModel, fixed_number_model

NumericalUnavailable = current.NumericalUnavailable
MAX_DENSE_DIMENSION = 512
MODEL_ID = "conditional-substrate-energy-response-v1"
PARTITIONS = ("symmetric", "improved")
FROZEN_CONTROLS = tuple((L, 1.0, g) for g in (0.7, 40.0) for L in (3, 4, 5))
FREQUENCIES = current.FREQUENCIES
IDENTITY_SHIFTS = (-3.0, 0.0, 2.5)
LIMITATIONS = (
    "The complete quantum Bose ring is stipulated; no classical quantization is derived.",
    "The externally supplied source H[f]=sum f_x h_x is not identified with a lapse or metric.",
    "Local energy partitions agree only in total energy, not in finite-momentum response.",
    "Exact-input contractions do not certify eigenpairs; residuals and resolution screens are diagnostics.",
    "No stress tensor, spacetime covariance, graviton, Newton constant or cosmological constant is derived.",
    "No physical calibration, thermodynamic extrapolation or production evolution is supplied.",
)
NUMERICAL_POLICY = (
    "Complete sectors only, dense dimension at most 512; public cases L=N=3,4,5.",
    "C>0 and g>=0; nonzero g/C must lie in [2^-40,2^40], a heuristic computational domain.",
    "Componentwise finite normal binary64 results only; exact zeros are retained without clipping.",
    "Sparse dyadic operator arithmetic and rational scalar contractions preserve cancellation components or reject.",
    "Unchanged module2 eigenvalue denominators, connected transitions and retarded evaluator are used.",
    "Public systems pass heuristic eigenpair/orthogonality checks and use detached readonly arrays without caller caches.",
    "Energy-projector grouping uses eigensolver resolution diagnostically, not as an exact degeneracy certificate.",
)
_ZERO = (Fraction(0), Fraction(0))


def _partition(partition):
    if not isinstance(partition, str) or partition not in PARTITIONS:
        raise ValueError("partition must be symmetric or improved")
    return (0.5, 0.5) if partition == "symmetric" else (0.25, 0.75)


def _momentum(m, L):
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise ValueError("momentum must be an integer, not a boolean")
    return int(m % L)


def _parameters(C, g):
    C = current._scalar(C, "C", real=True)
    g = current._scalar(g, "g", real=True)
    if C <= 0 or g < 0:
        raise ValueError("C must be positive and g nonnegative")
    if g:
        ratio = Fraction(g) / Fraction(C)
        if not Fraction(1, 2 ** 40) <= ratio <= 2 ** 40:
            raise NumericalUnavailable("g/C outside heuristic arithmetic-resolution domain")
    return C, g


def _validate_model(model):
    """Reject altered canonical models, before any new dense allocation."""
    if type(model) is not FixedNumberModel:
        raise ValueError("model must be supplied by fixed_number_model")
    L, N = model.L, model.N
    for value, name in ((L, "L"), (N, "N")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(name + " must be an integer")
    if not 3 <= L <= 12 or not 0 <= N <= L:
        raise ValueError("unsupported complete Bose-ring sector")
    dimension = comb(int(L + N - 1), int(N))
    if dimension > MAX_DENSE_DIMENSION:
        raise ValueError("complete sector exceeds dense dimension 512")
    C, g = _parameters(model.C, model.g)
    if not isinstance(model.basis, tuple) or len(model.basis) != dimension:
        raise ValueError("model must contain the complete canonical basis")
    # The constructor has no eigensolver, SVD, or hard-core operator transforms.
    # A fresh bounded reconstruction also checks all mutable dataclass arrays.
    expected = fixed_number_model(int(L), int(N), C, g)
    if model.basis != expected.basis:
        raise ValueError("model basis is not canonical and complete")
    for name in ("D", "V_unit", "V", "H", "p_indices", "q_indices", "PHP", "B", "QHQ"):
        actual, reference = getattr(model, name), getattr(expected, name)
        if not isinstance(actual, np.ndarray) or actual.shape != reference.shape:
            raise ValueError("model " + name + " has wrong shape")
        if actual.dtype.kind not in "iufc":
            raise ValueError("model arrays must be numeric")
        current._normal(actual, "model " + name)
        if not np.array_equal(actual, reference):
            raise ValueError("modified model " + name + " is unsupported")
    current._matrix(model.H, "model H", dimension)
    return int(L), dimension


def _validate_system(system, H):
    if not isinstance(system, dict):
        raise ValueError("system must be an unchanged module2 spectral system")
    n = len(H)
    supplied = current._matrix(system.get("H"), "system H", n)
    if not np.array_equal(supplied, H):
        raise ValueError("system Hamiltonian must exactly equal model Hamiltonian")
    for name, shape in (("vectors", (n, n)), ("ground", (n,)),
                        ("energies", (n,)), ("gaps", (n,))):
        a = system.get(name)
        if not isinstance(a, np.ndarray) or a.shape != shape:
            raise ValueError("system " + name + " has wrong shape")
        current._normal(a, "system " + name)
    if n < 2 or not np.array_equal(system["ground"], system["vectors"][:, 0]):
        raise ValueError("system ground does not match its eigenvectors")
    energies, gaps = system["energies"], system["gaps"]
    if np.iscomplexobj(energies) or np.iscomplexobj(gaps):
        raise ValueError("system energies and gaps must be real")
    expected = current._normal(energies.astype(np.longdouble) - energies[0], "system gaps")
    if not np.array_equal(gaps, expected) or np.any(gaps[1:] <= 0):
        raise ValueError("inconsistent or unresolved system gaps")
    if np.any(energies[1:] < energies[:-1]):
        raise ValueError("system eigenvalues must be sorted")
    if current._scalar(system.get("ground_gap"), "ground gap", real=True) != gaps[1]:
        raise ValueError("inconsistent system ground gap")
    for name in ("vectors", "ground", "energies", "gaps"):
        if system[name].dtype.kind not in "fc" or system[name].dtype.itemsize > 16:
            raise ValueError("system arrays must use bounded floating-point storage")
    vectors = np.asarray(system["vectors"], dtype=complex)
    scale = current._norm(supplied, "system Hamiltonian norm")
    tolerance = _computed_component(Fraction(64 * n) * Fraction(float(np.finfo(float).eps)),
                                    "eigenpair relative tolerance")
    absolute_tolerance = _computed_component(Fraction(tolerance) * Fraction(scale),
                                             "eigenpair absolute tolerance")
    orthogonality = current._norm(vectors.conj().T @ vectors - np.eye(n), "orthogonality diagnostic")
    residual = current._norm(supplied @ vectors - vectors * energies, "eigenpair diagnostic")
    if orthogonality > tolerance or residual > absolute_tolerance:
        raise ValueError("system eigenvectors fail bounded consistency diagnostics")
    resolution = current._scalar(system.get("resolution"), "resolution", real=True)
    residual_floor = _computed_component(Fraction(8) * Fraction(residual), "eigenpair resolution floor")
    resolution_floor = max(absolute_tolerance, residual_floor)
    if resolution <= 0 or gaps[1] <= max(resolution, resolution_floor):
        raise NumericalUnavailable("ground state is numerically unresolved")
    # Boundary isolation: neither trust nor mutate the caller's transition
    # cache. Copies prevent a later array mutation from invalidating this call.
    result = {}
    for name in ("H", "vectors", "energies", "gaps"):
        result[name] = np.array(system[name], copy=True)
        result[name].setflags(write=False)
    result["ground"] = result["vectors"][:, 0]
    result.update({"resolution": max(resolution, resolution_floor), "ground_gap": float(gaps[1])})
    return result


def _computed_component(value, name):
    try:
        return current._component(value, name)
    except ValueError as exc:
        if isinstance(exc.__cause__, OverflowError):
            raise NumericalUnavailable(name + " exceeds binary64 computed-output range") from exc
        raise


def _computed_complex(value, name):
    return complex(_computed_component(value[0], name), _computed_component(value[1], name))


def _retarded(system, A, B, z):
    try:
        return current.retarded_response(system, A, B, z)
    except ValueError as exc:
        if isinstance(exc.__cause__, OverflowError):
            raise NumericalUnavailable("retarded computed output exceeds binary64 range") from exc
        raise


def _connected(system, A):
    try:
        return current._transitions(system, A)
    except ValueError as exc:
        if isinstance(exc.__cause__, OverflowError):
            raise NumericalUnavailable("connected computed output exceeds binary64 range") from exc
        raise


def _frequency(z):
    z = current._scalar(z, "z")
    if z.imag <= 0:
        raise ValueError("retarded z must lie in the upper half plane")
    return z


def _dyadic(matrix):
    """Bounded sparse integer representation of exact supplied binary64 data."""
    matrix = current._matrix(matrix, "operator")
    parts, denominator = current._dyadic(matrix)
    rows = []
    for i in range(len(matrix)):
        rows.append({int(j): (int(parts[0][i, j]), int(parts[1][i, j]))
                     for j in np.flatnonzero(matrix[i] != 0)})
    return rows, denominator


def _iadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def _imul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _idot(left, right):
    total = (0, 0)
    for a, b in zip(left, right):
        total = _iadd(total, _imul(a, b))
    return total


def _matrix_from_rows(rows, denominator, name):
    result = np.zeros((len(rows), len(rows)), dtype=complex)
    for i, row in enumerate(rows):
        for j, value in row.items():
            if value != (0, 0):
                result[i, j] = _computed_complex(
                    (Fraction(value[0], denominator), Fraction(value[1], denominator)), name)
    return current._matrix(result, name)


def _linear(terms, name="operator sum"):
    """Round once per output component, never between cancelling summands."""
    records = []
    for coefficient, matrix in terms:
        coefficient = current._scalar(coefficient, "coefficient")
        rows, denominator = _dyadic(matrix)
        cparts, cd = current._dyadic(np.asarray([coefficient]))
        c = (int(cparts[0][0]), int(cparts[1][0]))
        records.append((c, rows, denominator * cd))
    n = len(records[0][1])
    if any(len(rows) != n for _, rows, _ in records):
        raise ValueError("operator sum shape mismatch")
    denominator = max(d for _, _, d in records)
    result = [{} for _ in range(n)]
    for c, rows, d in records:
        factor = denominator // d
        for i, row in enumerate(rows):
            for j, value in row.items():
                value = _imul(c, value)
                value = (factor * value[0], factor * value[1])
                result[i][j] = _iadd(result[i].get(j, (0, 0)), value)
    return _matrix_from_rows(result, denominator, name)


def _commutator(A, B, factor=1.0):
    """Exact-input sparse AB-BA, with one componentwise rounding at the end."""
    a, ad = _dyadic(A)
    b, bd = _dyadic(B)
    if len(a) != len(b):
        raise ValueError("commutator shape mismatch")
    cparts, cd = current._dyadic(np.asarray([current._scalar(factor, "commutator factor")]))
    c = (int(cparts[0][0]), int(cparts[1][0]))
    result = [{} for _ in a]
    for left, right, sign in ((a, b, 1), (b, a, -1)):
        for i, row in enumerate(left):
            for k, x in row.items():
                for j, y in right[k].items():
                    product = _imul(x, y)
                    product = (sign * product[0], sign * product[1])
                    result[i][j] = _iadd(result[i].get(j, (0, 0)), product)
    for row in result:
        for j in row:
            row[j] = _imul(c, row[j])
    return _matrix_from_rows(result, ad * bd * cd, "commutator")


def _local_parts(model, partition):
    a, b = _partition(partition)
    L, n = model.L, len(model.basis)
    index = {state: i for i, state in enumerate(model.basis)}
    U, V = [], []
    for x in range(L):
        onsite = np.diag(np.asarray([s[x] * (s[x] - 1) // 2 for s in model.basis], dtype=float))
        U.append(_linear(((model.g, onsite),), "onsite energy"))
        T = np.zeros((n, n))
        for col, state in enumerate(model.basis):
            if state[x]:
                moved = list(state)
                moved[x] -= 1
                moved[(x + 1) % L] += 1
                T[index[tuple(moved)], col] = np.sqrt(state[x] * (state[(x + 1) % L] + 1))
        V.append(_linear(((-model.C, T), (-model.C, T.T)), "bond energy"))
    h = tuple(_linear(((1, U[x]), (a, V[(x - 1) % L]), (b, V[x])), "local energy")
              for x in range(L))
    return {"U": tuple(U), "V": tuple(V), "h": h, "a": a, "b": b}


def _components(model, parts):
    L, U, V = model.L, parts["U"], parts["V"]
    a, b = parts["a"], parts["b"]
    Q = tuple(_commutator(V[x], V[(x + 1) % L], 1j) for x in range(L))
    A = tuple(_linear(((a, _commutator(U[x], V[x], 1j)),
                       (b, _commutator(V[x], U[(x + 1) % L], 1j)),
                       (a * a, Q[(x - 1) % L]), (b * b, Q[x])), "adjacent transfer component")
              for x in range(L))
    return Q, A


def _transfers(parts):
    h = parts["h"]
    # Direct exact-input commutators independently retain every modular overlap.
    return tuple(tuple(_commutator(hx, hy, 1j) for hy in h) for hx in h)


@current._guard
def local_energies(model, partition="symmetric"):
    """Tuple of complete local matrices h_x; no local occupation cutoff."""
    _partition(partition)
    _validate_model(model)
    return _local_parts(model, partition)["h"]


@current._guard
def energy_transfers(model, partition="symmetric"):
    """Tuple of tuples J[x][y]=i[h_x,h_y], including wrapped coincidences."""
    _partition(partition)
    _validate_model(model)
    return _transfers(_local_parts(model, partition))


def _phase(m, x, L):
    residue = (m * x) % L
    if residue == 0:
        return 1.0 + 0j
    if 2 * residue == L:
        return -1.0 + 0j
    if 4 * residue == L:
        return -1j
    if 4 * residue == 3 * L:
        return 1j
    return complex(np.exp(-2j * np.pi * residue / L))


def _fourier(model, m, partition, parts, components):
    L = model.L
    m = _momentum(m, L)
    k = float(2 * np.pi * m / L)
    normalization = float(1 / np.sqrt(L))
    phases = tuple(_phase(m, x, L) * normalization for x in range(L))
    Q, A = components
    def transform(local):
        return _linear(tuple(zip(phases, local)), "Fourier observable")
    h = transform(parts["h"]) if m else _linear(((normalization, model.H),), "uniform energy")
    V, Ak, Qk = transform(parts["V"]), transform(A), transform(Q)
    R = _commutator(model.H, h, 1j) if m else np.zeros_like(h)
    dotV = _commutator(model.H, V, 1j)
    q = _phase(m, 1, L) - 1 if m else 0j
    # Canonical exact zero also when 2m wraps to zero (e.g. the L4 pi mode).
    q2 = _phase(2 * m, 1, L) - 1 if (2 * m) % L else 0j
    return {"h": h, "R": R, "A": Ak, "Q": Qk, "V": V, "dotV": dotV,
            "q": q, "q2": q2, "d": -q, "k": k, "m": m, "partition": partition}


@current._guard
def fourier_observables(model, m, partition="symmetric"):
    """Matrices h,R=dot(h),A,Q,V,dotV and scalar q,q2,d,k,m,partition."""
    _partition(partition)
    _momentum(m, 1)
    _validate_model(model)
    parts = _local_parts(model, partition)
    return _fourier(model, m, partition, parts, _components(model, parts))


def _expectation_fraction(system, matrices):
    """Normalized ground expectation of a product, without rounded matvecs."""
    v, vd = current._dyadic(system["ground"])
    vector = [(int(r), int(i)) for r, i in zip(v[0], v[1])]
    ground = tuple(vector)
    denominator = vd
    for matrix in reversed(matrices):
        rows, md = _dyadic(matrix)
        vector = [_idot(tuple(row.values()), tuple(vector[j] for j in row))
                  for row in rows]
        denominator *= md
    total = _idot(tuple((r, -i) for r, i in ground), vector)
    norm = sum(r * r + i * i for r, i in ground)
    if not norm:
        raise ValueError("zero ground vector")
    denominator *= norm
    return Fraction(total[0] * vd, denominator), Fraction(total[1] * vd, denominator)


def _expectation_sum(system, terms, name):
    result = _ZERO
    for coefficient, matrices in terms:
        result = current._radd(result, current._rmul(current._rational(complex(coefficient)),
                                                    _expectation_fraction(system, matrices)))
    return _computed_complex(result, name)


def _scalar_sum(terms, name):
    result = _ZERO
    for factors in terms:
        product = (Fraction(1), Fraction(0))
        for factor in factors:
            product = current._rmul(product, current._rational(complex(factor)))
        result = current._radd(result, product)
    return _computed_complex(result, name)


def _weights_moment(amplitudes, gaps):
    weights, moment = [], Fraction(0)
    for amplitude, gap in zip(amplitudes, gaps):
        r, i = current._rational(amplitude)
        weight = r * r + i * i
        weights.append(_computed_component(weight, "transition weight"))
        moment += Fraction(float(gap)) * weight
    result = np.asarray(weights)
    result.setflags(write=False)
    return result, _computed_component(moment, "energy spectral moment")


def _source_data(system, obs):
    E, R = obs["h"], obs["R"]
    uniform = obs["m"] == 0
    if uniform:
        # Use the EXACT supplied H to get module2's structural zero. The
        # rounded matrix H/sqrt(L) must not be treated as generically stationary.
        left, right = _connected(system, system["H"])
        C = D = double = 0j
    else:
        left, right = _connected(system, E)
        C = _expectation_sum(system, ((1, (E, E.conj().T)), (-1, (E.conj().T, E))), "contact C")
        D = _expectation_sum(system, ((1, (R, E.conj().T)), (-1, (E.conj().T, R))), "contact D")
        H, Ed = system["H"], E.conj().T
        double = _expectation_sum(system, ((1, (E, H, Ed)), (-1, (E, Ed, H)),
                                          (-1, (H, Ed, E)), (1, (Ed, H, E))), "double commutator")
    gaps = system["gaps"][1:].copy()
    gaps.setflags(write=False)
    plus, Mplus = _weights_moment(right, gaps)
    minus, Mminus = _weights_moment(left, gaps)
    return {"contact_C": C, "contact_D": D, "double_commutator": double,
            "Mplus": Mplus, "Mminus": Mminus,
            "transition_weights_plus": plus, "transition_weights_minus": minus, "gaps": gaps,
            "energy_expectation": _expectation_sum(system, ((1, (E,)),), "energy expectation"),
            "rate_expectation": 0j if uniform else _expectation_sum(system, ((1, (R,)),), "rate expectation")}


def _response(system, obs, z, data=None):
    z = _frequency(z)
    E, R = obs["h"], obs["R"]
    # Validate sources even on the uniform structural branch.
    current._matrix(E, "energy source", len(system["H"]))
    current._matrix(R, "energy rate", len(system["H"]))
    if data is None:
        data = _source_data(system, obs)
    if obs["m"] == 0:
        zero = _retarded(system, system["H"], system["H"], z)
        EE = RE = ER = RR = zero
    else:
        EE = _retarded(system, E, E.conj().T, z)
        RE = _retarded(system, R, E.conj().T, z)
        ER = _retarded(system, E, R.conj().T, z)
        RR = _retarded(system, R, R.conj().T, z)
    C, D = data["contact_C"], data["contact_D"]
    first_left = _scalar_sum(((z, EE),), "first Ward left")
    first_right = _scalar_sum(((C,), (1j, RE)), "first Ward right")
    second_right = _scalar_sum(((C,), (-1j, ER)), "second Ward right")
    combined_left = _scalar_sum(((z, z, EE),), "combined Ward left")
    combined_right = _scalar_sum(((z, C), (1j, D), (RR,)), "combined Ward right")
    result = dict(data)
    # Never expose aliases into module2's transition cache or shared report data.
    for name in ("gaps", "transition_weights_plus", "transition_weights_minus"):
        result[name] = data[name].copy()
        result[name].setflags(write=False)
    result.update({"partition": obs["partition"], "m": obs["m"], "k": obs["k"], "z": z,
                   "uniform_structural_zero": obs["m"] == 0,
                   "chi_EE": EE, "chi_RE": RE, "chi_ER": ER, "chi_RR": RR,
                   "ward_first_left": first_left, "ward_first_right": first_right,
                   "ward_second_left": first_left, "ward_second_right": second_right,
                   "ward_combined_left": combined_left, "ward_combined_right": combined_right,
                   "ward_first_residual": float(abs(first_left - first_right)),
                   "ward_second_residual": float(abs(first_left - second_right)),
                   "ward_combined_residual": float(abs(combined_left - combined_right)),
                   "fsum_residual": float(abs(data["double_commutator"] - data["Mplus"] - data["Mminus"]))})
    return result


@current._guard
def energy_response(system, model, m, z, partition="symmetric"):
    """Mixed energy/rate response and contacts, both first Wards and f-sum.

    Mplus uses |h_n0|^2 and Mminus uses |h_0n|^2; these names do not denote
    denominator signs. Returned arrays are independent, read-only copies.
    All residual fields are floating-point diagnostics, not certificates.
    """
    _partition(partition)
    _momentum(m, 1)
    z = _frequency(z)
    _validate_model(model)
    system = _validate_system(system, model.H)
    parts = _local_parts(model, partition)
    obs = _fourier(model, m, partition, parts, _components(model, parts))
    return _response(system, obs, z)


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (complex, np.complexfloating)):
        z = current._scalar(value, "JSON complex")
        return [z.real, z.imag]
    if isinstance(value, (float, np.floating)):
        return current._scalar(value, "JSON number", real=True)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _projector_groups(system):
    gaps, resolution = system["gaps"][1:], system["resolution"]
    groups = []
    start = 0
    for i in range(1, len(gaps)):
        if gaps[i] - gaps[start] > resolution:
            groups.append(tuple(range(start, i)))
            start = i
    groups.append(tuple(range(start, len(gaps))))
    return groups


def _partition_report(model, system, partition, parts, transfers, observables, sources):
    H, h, L = model.H, parts["h"], model.L
    local_rates = [1j * (H @ hx - hx @ H) for hx in h]
    diagnostics = {
        "local_sum_residual": current._norm(sum(h) - H),
        "local_hermiticity_max_residual": max(current._norm(hx - hx.conj().T) for hx in h),
        "transfer_antisymmetry_max_residual": max(current._norm(transfers[x][y] + transfers[y][x])
                                                   for x in range(L) for y in range(L)),
        "transfer_hermiticity_max_residual": max(current._norm(J - J.conj().T) for row in transfers for J in row),
        "continuity_max_residual": max(current._norm(local_rates[x] + sum(transfers[x])) for x in range(L)),
        "range2_witness_norm": current._norm(transfers[0][2]),
    }
    momenta = []
    for obs, data in zip(observables, sources):
        a, b = parts["a"], parts["b"]
        residual = current._norm(obs["R"] - obs["q"] * obs["A"] - a * b * obs["q2"] * obs["Q"])
        momenta.append({"m": obs["m"], "k": obs["k"],
                        "diagnostics": {"fourier_continuity_residual": residual},
                        "responses": [_response(system, obs, z, data) for z in FREQUENCIES]})
    return {"partition": partition, "diagnostics": diagnostics, "momenta": momenta}


def _partition_change(model, system, parts, transfers, observables, sources):
    symmetric, improved = parts
    L, groups = model.L, _projector_groups(system)
    delta = [_linear(((0.25, symmetric["V"][x]), (-0.25, symmetric["V"][(x - 1) % L])))
             for x in range(L)]
    local = max(current._norm(improved["h"][x] - symmetric["h"][x] - delta[x]) for x in range(L))
    residuals = []
    for x in range(L):
        for y in range(L):
            transformed = _linear(((1, transfers[0][x][y]),
                                   (1, _commutator(symmetric["h"][x], delta[y], 1j)),
                                   (1, _commutator(delta[x], symmetric["h"][y], 1j)),
                                   (1, _commutator(delta[x], delta[y], 1j))))
            residuals.append(current._norm(transfers[1][x][y] - transformed))
    modes, strict, strict_imaginary = [], None, None
    for m in range(L):
        old, new = observables[0][m], observables[1][m]
        factor = float(abs(old["d"]) ** 2 / 16)
        vleft, vright = _connected(system, old["V"])
        wp, _ = _weights_moment(vright, system["gaps"][1:])
        wm, _ = _weights_moment(vleft, system["gaps"][1:])
        weight_residuals = {}
        for suffix, vw in (("plus", wp), ("minus", wm)):
            key = "transition_weights_" + suffix
            residual = 0.0
            for group in groups:
                terms = []
                for j in group:
                    terms.extend(((sources[1][m][key][j],), (-1, sources[0][m][key][j]), (-factor, vw[j])))
                residual = max(residual, abs(_scalar_sum(terms, "projector-weight difference")))
            weight_residuals["projector_weight_" + suffix + "_max_residual"] = float(residual)
        diagnostics = {"h_transform_residual": current._norm(new["h"] - old["h"] - old["d"] * old["V"] / 4),
                       "R_transform_residual": current._norm(new["R"] - old["R"] - old["d"] * old["dotV"] / 4)}
        diagnostics.update(weight_residuals)
        responses = []
        for z in FREQUENCIES:
            chi_old = _retarded(system, system["H"] if m == 0 else old["h"],
                                                system["H"] if m == 0 else old["h"].conj().T, z)
            chi_new = _retarded(system, system["H"] if m == 0 else new["h"],
                                                system["H"] if m == 0 else new["h"].conj().T, z)
            vv = _retarded(system, old["V"], old["V"].conj().T, z)
            predicted = _scalar_sum(((chi_old,), (factor, vv)), "partition response prediction")
            difference = _scalar_sum(((chi_new,), (-1, chi_old)), "partition response difference")
            if L == 3 and m == 1 and z == FREQUENCIES[0]:
                strict = float(difference.real)
                strict_imaginary = float(difference.imag)
            responses.append({"z": z, "chi_symmetric": chi_old, "chi_improved": chi_new,
                              "chi_VV": vv, "predicted_improved": predicted, "difference": difference,
                              "identity_residual": float(abs(chi_new - predicted))})
        modes.append({"m": m, "diagnostics": diagnostics, "responses": responses})
    return {"diagnostics": {"local_transform_max_residual": local,
                             "transfer_transform_max_residual": max(residuals),
                             "projector_groups_excited_indices": groups,
                             "projector_grouping_is_diagnostic": True},
            "momenta": modes, "strict_L3_m1_imaginary_z_difference": strict,
            "strict_L3_m1_imaginary_part_diagnostic": strict_imaginary}


def _identity_controls(model, system, parts, observables):
    result, L = [], model.L
    identity = np.eye(len(model.H))
    for kappa in IDENTITY_SHIFTS:
        shifted_H = _linear(((1, model.H), (kappa, identity)), "shifted Hamiltonian")
        # An actual independent eigensystem, never a modified or shallow-copied
        # system with a stale transition cache, including the zero-shift control.
        shifted = current.spectral_system(shifted_H)
        residual = local_residual = 0.0
        for p, obs_set in zip(parts, observables):
            shifted_local = tuple(_linear(((1, h), (kappa / L, identity)), "shifted local energy") for h in p["h"])
            local_residual = max(local_residual, current._norm(sum(shifted_local) - shifted_H))
            for obs in obs_set:
                m = obs["m"]
                if m:
                    phases = tuple(_phase(m, x, L) / np.sqrt(L) for x in range(L))
                    Eshift = _linear(tuple(zip(phases, shifted_local)), "shifted Fourier energy")
                    Ebase = obs["h"]
                else:
                    # Normalize outside a structural exact-H contraction. Zero
                    # connected weights are not inferred from rounded Fourier h.
                    Ebase, Eshift = system["H"], shifted["H"]
                for z in FREQUENCIES:
                    old = _retarded(system, Ebase, Ebase.conj().T, z)
                    new = _retarded(shifted, Eshift, Eshift.conj().T, z)
                    residual = max(residual, abs(new - old))
        result.append({"kappa": kappa,
                       "ground_energy_shift": float(shifted["energies"][0] - system["energies"][0]),
                       "gap_max_residual": float(np.max(np.abs(shifted["gaps"] - system["gaps"]))),
                       "local_sum_max_residual": local_residual, "response_max_residual": float(residual),
                       "uniform_structural_zero": True, "uniform_raw_fourier_not_used": True})
    return result


@current._guard
def case_report(L, C=1.0, g=0.7):
    """Strict-JSON-safe case in the fixed L=N=3,4,5 computational domain."""
    if isinstance(L, (bool, np.bool_)) or not isinstance(L, Integral) or L not in (3, 4, 5):
        raise ValueError("case reports require L=N=3,4,5")
    C, g = _parameters(C, g)
    model = fixed_number_model(int(L), int(L), C, g)
    _validate_model(model)
    system = current.spectral_system(model.H)
    parts = [_local_parts(model, partition) for partition in PARTITIONS]
    transfers = [_transfers(p) for p in parts]
    observables, sources = [], []
    for partition, p in zip(PARTITIONS, parts):
        components = _components(model, p)
        obs = [_fourier(model, m, partition, p, components) for m in range(model.L)]
        observables.append(obs)
        sources.append([_source_data(system, o) for o in obs])
    partitions = [_partition_report(model, system, partition, p, J, obs, data)
                  for partition, p, J, obs, data in zip(PARTITIONS, parts, transfers, observables, sources)]
    report = {"parameters": {"L": int(L), "N": int(L), "C": C, "g": g, "dimension": len(model.basis)},
              "ground": {"energy": float(system["energies"][0]), "gap": system["ground_gap"],
                         "resolution": system["resolution"], "status": "resolved_nondegenerate_diagnostic"},
              "partitions": partitions,
              "partition_change": _partition_change(model, system, parts, transfers, observables, sources),
              "identity_shifts": _identity_controls(model, system, parts, observables)}
    return _json(report)


@current._guard
def demonstration_report():
    """Six frozen complete-sector cases; no fit, scan or file output."""
    return {"model_id": MODEL_ID,
            "conventions": {"local_transfer": "Jxy=i[hx,hy]; dot(hx)=-sum_y Jxy",
                            "fourier": "O_k=sum_x exp(-ikx) O_x/sqrt(L)",
                            "retarded": "Im(z)>0; unchanged signed Lehmann denominators",
                            "moments": "Mplus=sum gap*|h_n0|^2; Mminus=sum gap*|h_0n|^2",
                            "source": "H[f]=sum_x f_x h_x externally supplied; no metric identification"},
            "numerical_policy": list(NUMERICAL_POLICY), "limitations": list(LIMITATIONS),
            "cases": [case_report(L, C, g) for L, C, g in FROZEN_CONTROLS]}
