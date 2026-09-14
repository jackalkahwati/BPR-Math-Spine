"""Parity-aware exterior-power candidate in the unchanged strict hard-core ring.

Equations/controls: doc/derivations/substrate_quantum_matching_2026-09-12.md.
Determinant rows use ascending binary occupations, not lexicographic subsets.
Ranks and spectral ties are bounded binary64 diagnostics, not exact proofs.
No link Hilbert space or map of Gauss operators into the ring is supplied.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import substrate_chirality as sc
from . import substrate_fermionization as sf
from .quantum_flavor_sources import overlap_operators, spin_one_generators

MODEL_ID = "conditional-substrate-quantum-matching-v1"
MODES = (-1, 0, 1)
MAX_SITES = 8
MAX_BINARY_DIM = 256
RANK_TOLERANCE = 1e-11
FROZEN_CONTROLS = tuple((L, N) for L in (5, 6, 7) for N in (1, 2, 3)) + (
    (3, 3), (4, 4), (5, 5), (5, 0),
)
COMMON_SCALES = (0.25, 4.0)
LIMITATIONS = (
    "This is strict hard-core compression, not invariant finite-g Bose dynamics.",
    "The parity-specific three-mode window has many-body dimension binom(3,N), not three generations.",
    "Neutral N=L is empty in this candidate for L>3 and rank one for L=N=3; neutral doublon-hole excitations lie outside hard core.",
    "The even window chooses one member of a paired level and generally leaks under reflection; many-body isolation is measured independently.",
    "The real-linear density image and its generated complex associative algebra are different objects; neither proves conserved SU(3) or a gauge theory.",
    "The odd literal A1+E1 versus internal A2+E1 vector obstruction depends on the chosen reflection lift; bilinear conjugation cannot distinguish the two signs.",
    "Independent canonical internal fermions and monopole sources are comparison targets, not identified ring modes, spin, or physical statistics.",
    "No link Hilbert space, endpoint link action, or intertwining map of Gauss operators into the ring is supplied; absence is not a universal emergent-gauge no-go.",
    "All reported ranks, degeneracies, residuals and separation classifications are numerical, not certified roundoff bounds or exact proofs.",
)


def _sector(L, N):
    L = sf._integer(L, "L", 3, MAX_SITES)
    N = sf._integer(N, "N", 0, L)
    if (1 << L) > MAX_BINARY_DIM:
        raise ValueError("complete binary dimension exceeds cap256")
    return L, N


def occupation_basis(L, N):
    """Tuple of integer bit strings in ascending order; site x is bit 2**x."""
    L, N = _sector(L, N)
    return tuple(bit for bit in range(1 << L) if bin(bit).count("1") == N)


def _subsets(bits, L):
    return tuple(tuple(x for x in range(L) if bit & (1 << x)) for bit in bits)


def _matrix(matrix, name):
    # Validate structure BEFORE dtype conversion, which can allocate a full
    # complex copy. Unknown array protocols/iterators cannot promise a cap.
    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a matrix")
        shape = matrix.shape
    elif isinstance(matrix, Sequence) and not isinstance(matrix, (str, bytes)):
        if len(matrix) > MAX_SITES:
            raise ValueError(f"{name} exceeds one-body cap8")
        width = None
        for row in matrix:
            if isinstance(row, np.ndarray):
                valid_row = row.ndim == 1
            else:
                valid_row = isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            if not valid_row:
                raise ValueError(f"{name} must contain one-dimensional rows")
            if len(row) > MAX_SITES:
                raise ValueError(f"{name} exceeds one-body cap8")
            if width is not None and len(row) != width:
                raise ValueError(f"{name} must have equal row lengths")
            width = len(row)
            if any(not np.isscalar(value) for value in row):
                raise ValueError(f"{name} entries must be scalar")
        if width is None:
            raise ValueError(f"{name} must have explicit two-dimensional shape")
        shape = (len(matrix), width)
    else:
        raise TypeError(f"{name} must be a NumPy array or bounded nested sequence")
    if max(shape, default=0) > MAX_SITES:
        raise ValueError(f"{name} exceeds one-body cap8")
    a = np.asarray(matrix, dtype=complex)
    return sf._normal(a, name)


@sf._guard
def fourier_window(L, N):
    """L-by-3 columns m=(-1,0,1), with theta=0 odd and pi even."""
    L, N = _sector(L, N)
    angles = (2 * np.pi * np.array(MODES) + sc.boundary_phase(N)) / L
    return sf._normal(np.exp(1j * np.arange(L)[:, None] * angles) / np.sqrt(L), "Fourier window")


def _determinants(matrix, rows, columns):
    result = np.empty((len(rows), len(columns)), complex)
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            # Singular minors are valid structural zeros. NumPy's determinant
            # implementation can signal log(0) internally for these minors.
            with np.errstate(divide="ignore", invalid="ignore"):
                value = np.linalg.det(matrix[np.ix_(row, col)]) if row else 1.0
            result[i, j] = sf._normal(value, "minor determinant")
    return sf._normal(result, "determinant map")


@sf._guard
def exterior_power(matrix, N):
    """Multiplicative wedge^N(matrix), both subset orders lexicographic.

    This is NOT the additive lift dGamma. Rectangular one-body maps are allowed;
    dimensions are capped at8 before combinations/allocation. N=0 gives [[1]].
    """
    matrix = _matrix(matrix, "one-body map")
    N = sf._integer(N, "N", 0, MAX_SITES)
    rows = tuple(combinations(range(matrix.shape[0]), N))
    columns = tuple(combinations(range(matrix.shape[1]), N))
    return _determinants(matrix, rows, columns)


@sf._guard
def exterior_lift(matrix, N):
    """Additive dGamma(matrix) in lexicographic column-combination order."""
    matrix = _matrix(matrix, "one-body generator")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("one-body generator must be square")
    N = sf._integer(N, "N", 0, MAX_SITES)
    basis = tuple(combinations(range(len(matrix)), N))
    index = {state: i for i, state in enumerate(basis)}
    result = np.zeros((len(basis), len(basis)), complex)
    for col, state in enumerate(basis):
        for position, source in enumerate(state):
            rest = state[:position] + state[position + 1:]
            for target in range(len(matrix)):
                if target in rest:
                    continue
                insertion = sum(x < target for x in rest)
                moved = tuple(sorted(rest + (target,)))
                result[index[moved], col] += (-1)**(position + insertion) * matrix[target, source]
    return sf._normal(result, "additive exterior lift")


@sf._guard
def wedge_embedding(L, N):
    """Actual det(U[X,I]) map: ascending bit rows, lexicographic I columns.

    Shape (comb(L,N), comb(3,N)); the second dimension is zero for N>3.
    """
    L, N = _sector(L, N)
    return _determinants(fourier_window(L, N), _subsets(occupation_basis(L, N), L),
                         tuple(combinations(range(3), N)))


@dataclass(frozen=True)
class QuantumMatchingModel:
    L: int
    N: int
    C: float
    bits: tuple
    site_subsets: tuple
    mode_subsets: tuple
    U: np.ndarray
    W: np.ndarray
    complement: np.ndarray
    h: np.ndarray
    H: np.ndarray
    Hproj: np.ndarray
    densities: tuple
    projected_densities: tuple
    hamiltonian_closure: np.ndarray
    density_leakages: tuple


@sf._guard
def projected_model(L, N, C=1.0):
    """Bounded actual hard-core matrices and Fourier Slater complement.

    Remaining one-body columns follow the retained labels, then unused residues
    in ascending order. Complement columns are all other N-fold combinations.
    No full Bose or complete binary operator allocation is performed.
    """
    L, N = _sector(L, N)
    C = sf._real(C, "C", positive=True)
    sf._mul(2 * L, C, "spectral scale")
    bits = occupation_basis(L, N)
    sites = _subsets(bits, L)
    modes = tuple(combinations(range(3), N))
    U = fourier_window(L, N)
    W = _determinants(U, sites, modes)
    used = tuple(m % L for m in MODES)
    remaining = tuple(m for m in range(L) if m not in used)
    angles = (2 * np.pi * np.array(remaining) + sc.boundary_phase(N)) / L
    full_U = np.column_stack((U, np.exp(1j * np.arange(L)[:, None] * angles) / np.sqrt(L)))
    omitted = tuple(state for state in combinations(range(L), N) if state not in modes)
    complement = _determinants(full_U, sites, omitted)
    h = sc.one_body_hopping(L, N, C)
    H = sc.hard_core_hopping(L, N, C)
    Hproj = sf._normal(W.conj().T @ H @ W, "projected H")
    densities = tuple(np.diag([float(bool(bit & (1 << x))) for bit in bits]) for x in range(L))
    projected = tuple(W.conj().T @ density @ W for density in densities)
    leakage = tuple(density @ W - W @ proj for density, proj in zip(densities, projected))
    closure = sf._normal(H @ W - W @ Hproj, "Hamiltonian closure")
    return QuantumMatchingModel(L, N, C, bits, sites, modes, U, W, complement, h, H,
                                Hproj, densities, projected, closure, leakage)


def _rank(matrix):
    if not matrix.size:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return 0
    return int(np.count_nonzero(singular > RANK_TOLERANCE * singular[0]))


def _span(matrices, *, real=False):
    if not matrices:
        return 0
    vectors = np.array([a.reshape(-1) for a in matrices]).T
    if real:
        vectors = np.concatenate((vectors.real, vectors.imag), axis=0)
    return _rank(vectors)


def _algebra_dimension(matrices, d):
    """Numerical complex dimension of the UNITAL associative algebra."""
    if not d:
        return 0
    basis = []

    def insert(a, product_scale=None):
        vector = a.reshape(-1).copy()
        norm = sf._norm(vector)
        if not norm:
            return False
        # A nearly-zero product of independent normalized operands can be
        # entirely multiplication roundoff. Do not magnify it into a new
        # algebra direction. This cutoff does NOT apply to input generators.
        if product_scale is not None and norm <= RANK_TOLERANCE * product_scale:
            return False
        vector /= norm
        for _ in range(2):
            for old in basis:
                vector -= np.vdot(old, vector) * old
        norm = sf._norm(vector)
        if norm <= RANK_TOLERANCE:
            return False
        basis.append(vector / norm)
        return True

    insert(np.eye(d, dtype=complex))
    generators = []
    for a in matrices:
        # Normalize each supplied generator independently. Small but nonzero
        # independent generators remain legitimate, and rescaling an input
        # does not alter the generated algebra or the product-zero threshold.
        scale = sf._norm(a.reshape(-1))
        if scale:
            normalized = a / scale
            generators.append(normalized)
            insert(normalized)
    cursor = 0
    while cursor < len(basis) and len(basis) < d*d:
        a = basis[cursor].reshape(d, d)
        for b in generators:
            insert(a @ b, sf._norm(a.reshape(-1)) * sf._norm(b.reshape(-1)))
            if len(basis) == d*d:
                break
        cursor += 1
    return len(basis)


def _commutant_dimension(matrices, d):
    if not d:
        return 0
    identity = np.eye(d)
    constraints = []
    for a in matrices:
        # Subtract the scalar part before normalization so tiny roundoff on a
        # scalar compression cannot manufacture a symmetry breaking operator.
        scale = sf._norm(a)
        if scale == 0:
            continue
        centered = (a - np.trace(a) / d * identity) / scale
        if sf._norm(centered) <= RANK_TOLERANCE:
            continue
        constraints.append(np.kron(identity, centered) - np.kron(centered.T, identity))
    return d*d - (_rank(np.vstack(constraints)) if constraints else 0)


def _complex_json(a):
    a = np.asarray(a)
    sf._normal(a, "serialized matrix")
    return {"real": a.real.tolist(), "imag": a.imag.tolist(), "shape": list(a.shape)}


def _source_report(model):
    d = model.W.shape[1]
    matrices = model.projected_densities
    # The internal source family is independent. This arithmetic comparison
    # does not supply an identification of physical modes or source profiles.
    internal = tuple(exterior_lift(a, model.N) for a in overlap_operators())
    return {
        "status": "available" if d else "empty_candidate",
        "direct_real_span_dimension": _span(matrices, real=True),
        "unital_complex_associative_algebra_dimension": _algebra_dimension(matrices, d),
        "full_matrix_algebra_dimension": d*d,
        "internal_source_real_span_dimension": _span(internal, real=True),
        "internal_comparison": "independent monopole operators with additive exterior lift; no physical source matching map",
        "density_leakage_norms": [sf._norm(a) for a in model.density_leakages],
        "number_sum_residual": sf._norm(sum(matrices) - model.N * np.eye(d)),
        "projected_densities": [_complex_json(a) for a in matrices],
        "rank_relative_tolerance": RANK_TOLERANCE,
        "algebra_is_conserved_su3": False,
    }


@sf._guard
def source_diagnostics(L, N):
    return _source_report(projected_model(L, N))


def _one_body_symmetries(L, N):
    translation = np.zeros((L, L), complex)
    for x in range(L - 1):
        translation[x, x + 1] = 1
    translation[L - 1, 0] = 1 if N % 2 else -1
    reflection = np.zeros((L, L), complex)
    reflection[0, 0] = 1
    for x in range(1, L):
        reflection[x, L - x] = 1 if N % 2 else -1
    return translation, reflection


def _occupation_permutation(bits, L, transform):
    index = {bit: i for i, bit in enumerate(bits)}
    result = np.zeros((len(bits), len(bits)), complex)
    for col, bit in enumerate(bits):
        target = sum(1 << transform(x) for x in range(L) if bit & (1 << x))
        result[index[target], col] = 1
    return result


def _symmetry_report(model):
    L, N, W = model.L, model.N, model.W
    d = W.shape[1]
    t, r = _one_body_symmetries(L, N)
    t_full = _determinants(t, model.site_subsets, model.site_subsets)
    r_full = _determinants(r, model.site_subsets, model.site_subsets)
    T = _occupation_permutation(model.bits, L, lambda x: (x - 1) % L)
    R = _occupation_permutation(model.bits, L, lambda x: (-x) % L)
    phase = (-1)**(N * (N - 1) // 2)
    projected_T, projected_R = (W.conj().T @ a @ W for a in (T, R))
    t3, r3 = (model.U.conj().T @ a @ model.U for a in (t, r))
    # Compute at unit C to avoid scale-dependent numerical rank classification.
    Hunit = sc.hard_core_hopping(L, N)
    Hprojected_unit = W.conj().T @ Hunit @ W
    generator = exterior_lift(spin_one_generators()[0], N)
    commutator_unit = Hprojected_unit @ generator - generator @ Hprojected_unit
    witness_norm_unit = sf._norm(commutator_unit)
    return {
        "status": "available" if d else "empty_candidate",
        "boundary_phase": sc.boundary_phase(N),
        "one_body_translation": _complex_json(t),
        "one_body_reflection": _complex_json(r),
        "one_body_h_commutator_norms": [sf._norm(a @ model.h - model.h @ a) for a in (t, r)],
        "one_body_window_translation_leakage": sf._norm(t @ model.U - model.U @ t3),
        "one_body_window_reflection_leakage": sf._norm(r @ model.U - model.U @ r3),
        "occupation_translation": _complex_json(T),
        "occupation_reflection": _complex_json(R),
        "exterior_translation": _complex_json(t_full),
        "exterior_reflection": _complex_json(r_full),
        "translation_lift_vs_occupation_residual": sf._norm(t_full - T),
        "reflection_lift_vs_occupation_raw_residual": sf._norm(r_full - R),
        "reflection_lift_global_phase": phase,
        "reflection_lift_vs_occupation_phase_corrected_residual": sf._norm(r_full - phase * R),
        "occupation_h_commutator_norms": [sf._norm(a @ model.H - model.H @ a) for a in (T, R)],
        "projected_translation": _complex_json(projected_T),
        "projected_reflection": _complex_json(projected_R),
        "many_body_translation_leakage": sf._norm(T @ W - W @ projected_T),
        "many_body_reflection_leakage": sf._norm(R @ W - W @ projected_R),
        "projected_translation_unitarity_residual": sf._norm(projected_T.conj().T @ projected_T - np.eye(d)),
        "projected_reflection_unitarity_residual": sf._norm(projected_R.conj().T @ projected_R - np.eye(d)),
        "hamiltonian_commutant_complex_dimension": _commutant_dimension([Hprojected_unit], d),
        "discrete_compression_commutant_complex_dimension": _commutant_dimension([projected_T, projected_R], d),
        "joint_h_discrete_commutant_complex_dimension": _commutant_dimension([Hprojected_unit, projected_T, projected_R], d),
        "discrete_commutant_scope": "commutant of actual compressed occupation operators; not a symmetry representation when leakage is nonzero",
        "nonconserved_generator": {
            "status": "witness" if d == 3 and witness_norm_unit > RANK_TOLERANCE else "not_applicable_scalar_or_empty",
            "operator": _complex_json(generator),
            "construction": "dGamma(Jx) from independent internal spin-one matrices in column order",
            "commutator_norm": sf._mul(witness_norm_unit, model.C, "generator commutator norm"),
            "commutator_norm_over_C": witness_norm_unit,
        },
    }


@sf._guard
def symmetry_diagnostics(L, N, C=1.0):
    return _symmetry_report(projected_model(L, N, C))


@sf._guard
def intertwiner_diagnostics(L):
    """Solve T A=A T, R A=A(s R), for both s=-1 and s=+1.

    Returns a numerical nullspace, analytic saturating witness, and maximum
    rank for this bounded odd one-body window, not a universal vector no-go.
    """
    L, _ = _sector(L, 1)
    T = np.diag(np.exp(2j * np.pi * np.array(MODES) / L))
    R = np.eye(3, dtype=complex)[::-1]
    identity = np.eye(3)
    results = {}
    for sign, label in ((-1, "proper_half_turn"), (1, "alternate_parity_twisted")):
        internal_R = sign * R
        constraints = np.vstack((np.kron(identity, T) - np.kron(T.T, identity),
                                 np.kron(identity, R) - np.kron(internal_R.T, identity)))
        _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
        rank = int(np.count_nonzero(singular > RANK_TOLERANCE * singular[0]))
        basis = [v.conj().reshape((3, 3), order="F") for v in vh[rank:]]
        witness = np.diag([1., 0., -1.]) if sign == -1 else np.eye(3)
        # The equations force diagonal A, then a_- = sign*a_+ and
        # a_0=sign*a_0. The displayed witness attains the resulting bound.
        results[label] = {
            "reflection_sign": sign,
            "internal_decomposition": "A2+E1" if sign == -1 else "A1+E1",
            "solution_complex_dimension": len(basis),
            "basis": [_complex_json(a) for a in basis],
            "maximum_rank": 2 if sign == -1 else 3,
            "maximum_rank_scope": "specific diagonal translation and specified reflection lift; algebraic constraint bound with saturating witness",
            "witness": _complex_json(witness),
            "witness_rank_numerical": _rank(witness),
            "translation_residual": sf._norm(T @ witness - witness @ T),
            "reflection_residual": sf._norm(R @ witness - witness @ internal_R),
            "nullspace_residual": max((sf._norm(T @ a - a @ T) + sf._norm(R @ a - a @ internal_R) for a in basis), default=0.0),
        }
    matrix_units = [np.eye(9, dtype=complex)[i].reshape(3, 3) for i in range(9)]
    bilinear_residual = max(sf._norm(R @ a @ R.conj().T - (-R) @ a @ (-R).conj().T) for a in matrix_units)
    return {"L": L, "ring_decomposition": "A1+E1", "translation": _complex_json(T),
            "literal_reflection": _complex_json(R), "lifts": results,
            "bilinear_conjugation_identity_residual": bilinear_residual,
            "bilinear_test_basis": "all nine complex matrix units",
            "universal_source_or_gauge_obstruction": False,
            "rank_relative_tolerance": RANK_TOLERANCE}


def _spectral_report(model):
    W, Q = model.W, model.complement
    # Diagonalize actual compressed many-body matrices at unit scale, not sums
    # inferred from one-body cutoff levels. Scale only after stable differences.
    Hunit = sc.hard_core_hopping(model.L, model.N)
    retained = np.linalg.eigvalsh(W.conj().T @ Hunit @ W)
    omitted = np.linalg.eigvalsh(Q.conj().T @ Hunit @ Q)
    full = np.linalg.eigvalsh(Hunit)
    tolerance_unit = 256 * np.finfo(float).eps * max(1, len(full)) * max(1., sf._norm(Hunit))
    gap = None
    if not len(retained):
        status = "empty_candidate"
    elif not len(omitted):
        status = "empty_complement"
    else:
        gap_unit = float(np.min(np.abs(retained[:, None] - omitted[None, :])))
        gap = sf._mul(gap_unit, model.C, "many-body separation")
        status = "separated_numerically" if gap_unit > tolerance_unit else "overlap_within_numerical_tolerance"
    return {
        "status": status,
        "minimum_separation": gap,
        "retained_spectrum": sf._scale(retained, model.C, "retained spectrum").tolist(),
        "complement_spectrum": sf._scale(omitted, model.C, "complement spectrum").tolist(),
        "full_spectrum": sf._scale(full, model.C, "full spectrum").tolist(),
        "tie_absolute_tolerance": sf._mul(tolerance_unit, model.C, "spectral tie tolerance"),
        "complement_closure_norm": sf._mul(sf._norm(Hunit @ Q - Q @ (Q.conj().T @ Hunit @ Q)), model.C, "complement closure"),
        "scope": "minimum pairwise distance of actual retained/complement many-body spectra; numerical tolerance is not a certified bound",
    }


@sf._guard
def matching_diagnostics(L, N, C=1.0):
    model = projected_model(L, N, C)
    L, N = model.L, model.N
    d = model.W.shape[1]
    h3 = model.U.conj().T @ model.h @ model.U
    additive = exterior_lift(h3, N)
    combined = np.column_stack((model.W, model.complement))
    return {
        "model_id": MODEL_ID,
        "parameters": {"L": model.L, "N": model.N, "C": model.C},
        "candidate": {
            "status": "available" if d else "empty_candidate", "dimension": d,
            "hard_core_dimension": len(model.bits), "complement_dimension": model.complement.shape[1],
            "boundary_phase": sc.boundary_phase(N), "mode_labels": list(MODES),
            "momenta": ((2 * np.pi * np.array(MODES) + sc.boundary_phase(N)) / L).tolist(),
            "window_choice": "periodic lowest singlet and doublet" if N % 2 else "antiperiodic representative; omitted -3pi/L partner for L>=4 (L=3 is complete)",
            "bits": list(model.bits), "site_subsets": [list(x) for x in model.site_subsets],
            "mode_subsets": [list(x) for x in model.mode_subsets],
            "W": _complex_json(model.W), "Hproj": _complex_json(model.Hproj),
            "orthonormality_residual": sf._norm(model.W.conj().T @ model.W - np.eye(d)),
            "complete_slater_basis_residual": sf._norm(combined.conj().T @ combined - np.eye(len(model.bits))),
            "hamiltonian_closure_norm": sf._norm(model.hamiltonian_closure),
            "additive_lift_residual": sf._norm(model.Hproj - additive),
        },
        "neutrality": {"neutrality_modulus": L, "neutrality_rule": "N mod L = 0",
                       "is_neutral": N % L == 0, "number_charge": N,
                       "modular_charge": N % L, "is_unit_filling": N == L,
                       "charge_deviation_from_unit_filling": N - L,
                       "candidate_dimension": d, "full_hard_core_dimension": len(model.bits),
                       "neutral_candidate_dimension": d if N % L == 0 else 0,
                       "neutral_hard_core_dimension": len(model.bits) if N % L == 0 else 0,
                       "unit_filling_candidate_dimension": 1 if L == 3 else 0,
                       "unit_filling_hard_core_dimension": 1,
                       "doublon_hole_excitations": "outside hard core"},
        "spectral_separation": _spectral_report(model),
        "sources": _source_report(model),
        "symmetries": _symmetry_report(model),
        "gauge_comparison": {"link_hilbert_space_supplied": False,
                             "endpoint_link_actions_supplied": False,
                             "gauss_intertwining_map": "absent",
                             "universal_emergent_gauge_no_go": False},
        "numerical_scope": {"rank_relative_tolerance": RANK_TOLERANCE,
                            "arithmetic": "finite normal binary64 and structural zero; residuals are not rigorous error bounds",
                            "matrix_norm": "spectral/operator 2-norm"},
    }


@sf._guard
def demonstration_report():
    """Frozen controls and common C scaling; stdout serialization is in script."""
    cases = [matching_diagnostics(L, N) for L, N in FROZEN_CONTROLS]
    scale_controls = []
    for C in COMMON_SCALES:
        for L, N in ((5, 1), (6, 2), (7, 3)):
            model = projected_model(L, N, C)
            unit = projected_model(L, N)
            separation = _spectral_report(model)
            scale_controls.append({
                "L": L, "N": N, "C": C,
                "Hproj_scaling_residual": sf._norm(model.Hproj - C * unit.Hproj),
                "density_scaling_residual": max(sf._norm(a - b) for a, b in zip(model.projected_densities, unit.projected_densities)),
                "spectral_separation": separation,
            })
    return {"model_id": MODEL_ID, "cases": cases,
            "intertwiners": [intertwiner_diagnostics(L) for L in (5, 6, 7)],
            "common_scale_controls": scale_controls, "limitations": list(LIMITATIONS)}
