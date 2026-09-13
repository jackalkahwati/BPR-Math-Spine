"""Second-order all-D Schrieffer-Wolff diagnostic for the stipulated Bose ring.

Convention: U=exp(S), Htilde=U H U†, S=lambda*S1+lambda²*S2,
lambda=C/g. All coefficients use the complete fixed-number occupation basis,
not a local occupation cutoff. Dense work inherits the dimension-512 cap
(unit filling L=3..6). Coefficients are distinct from scaled energies.

Analytic BCH/Weyl bounds exclude floating-point error. The observable bound
controls unitary conjugation, NOT the error from replacing the transformed
exact ground state by Omega. Resummed spectral weights are numerical model
diagnostics, not certified residues or complete fourth-order predictions.
"""
from dataclasses import dataclass

import numpy as np

from bpr.substrate_fermionization import (
    FixedNumberModel, MAX_DENSE_DIM, _EPS, _div, _guard, _mul, _norm,
    _normal, _real, _scale,
)
from bpr.substrate_neutral_response import (
    NeutralModel, _momentum, density_diagonal, grouped_spectral_measure,
    neutral_model,
)

MODEL_ID = "conditional-substrate-neutral-effective-v1"
FROZEN_CONTROLS = ((5, 1.0, 40.0, 1), (5, 1.0, 0.7, 1),
                   (5, 1.0, 40.0, 0), (5, 1.0, 0.7, 0))
LIMITATIONS = (
    "The complete quantum Bose ring and N=L=q neutrality are stipulated, not derived.",
    "BCH and Weyl bounds are exact-arithmetic inequalities; roundoff is not certified.",
    "Global sorted eigenvalue bounds identify D blocks only after explicit interval separation.",
    "Rrho bounds observable conjugation, not the true-ground density-source error.",
    "Squared truncated sources include only a partial fourth-order contribution, not the missing z3 interference.",
    "Effective line weights are partially resummed diagnostics, not certified individual residues.",
    "Finite-ring lines do not establish a thermodynamic bound particle or physical fermions.",
    "No observed masses, mixing, spacetime, gravity, or empirical support for BPR is derived.",
)


@dataclass(frozen=True)
class SWCoefficients:
    """Unscaled full-space coefficients; p1_channels[D] is P1 x P1.

    Channel D contributes P1 T P_D T P1/(1-D). The D=0 channel is
    positive; reachable higher-D channels are negative semidefinite.
    """
    adapter: NeutralModel
    Td: np.ndarray
    To: np.ndarray
    F: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    K: np.ndarray
    p1_channels: dict


@dataclass(frozen=True)
class EffectiveModel:
    """Full block truncation and its centered P1 compression.

    H2_dimensionless=D+lambda Td+lambda²K, H2=g*H2_dimensionless.
    H1 is the P1 energy matrix, E0=-4LC²/g; gaps1=H1-E0 I is a
    matrix, not its eigenvalues. No exact block identification is implied.
    """
    adapter: NeutralModel
    coefficients: SWCoefficients
    lam: float
    S: np.ndarray
    H2_dimensionless: np.ndarray
    H2: np.ndarray
    H1: np.ndarray
    E0: float
    gaps1: np.ndarray

    @property
    def model(self):
        return self.adapter.model


@dataclass(frozen=True)
class DressedSource:
    """Observable coefficients and vacuum transition source in the full basis.

    rho (also rho0), rho1, rho2 are unscaled full matrices; z1/rho1 Omega
    and z2/rho2 Omega are unscaled vectors. z=lambda z1+lambda²z2.
    p1_source and by_D contain local-block components of this scaled z.
    The three weight_order* fields are separate lambda-scaled contributions.
    weight_order2+weight_order3 is the cubic-order polynomial; the extra
    weight_partial_order4 is ||lambda²z2||², NOT the full fourth-order term.
    """
    m: int
    rho: np.ndarray
    rho1: np.ndarray
    rho2: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    z: np.ndarray
    p1_source: np.ndarray
    by_D: dict
    weight_order2: float
    weight_order3: float
    weight_partial_order4: float
    truncated_weight: float

    @property
    def rho0(self):
        return self.rho


def _comm(a, b):
    return a @ b - b @ a


def _adapter(value):
    if isinstance(value, NeutralModel):
        adapter = value
    elif isinstance(value, FixedNumberModel):
        p = np.flatnonzero(value.D == 1)
        q = np.flatnonzero(value.D != 1)
        if value.N != value.L:
            raise ValueError("effective neutral coefficients require N=L")
        omega = value.basis.index((1,) * value.L)
        adapter = NeutralModel(value, omega, p, q, value.H[np.ix_(p, p)],
                               value.H[np.ix_(p, q)], value.H[np.ix_(q, q)])
    else:
        raise TypeError("adapter must be a NeutralModel or FixedNumberModel")
    model = adapter.model
    if model.N != model.L or not 3 <= model.L <= 6:
        raise ValueError("effective neutral model requires N=L and L=3..6")
    if len(model.basis) > MAX_DENSE_DIM:
        raise ValueError("fixed-number dimension exceeds inherited cap")
    _normal(model.D, "interaction coefficient")
    _normal(model.V_unit, "hopping coefficient")
    return adapter


@_guard
def sw_coefficients(adapter):
    """Canonical zero-equal-D generators, independent of C and g (even g=0)."""
    adapter = _adapter(adapter)
    model = adapter.model
    difference = model.D[:, None] - model.D[None, :]
    equal = difference == 0
    Td = np.where(equal, model.V_unit, 0.0)
    To = model.V_unit - Td
    S1 = np.divide(model.V_unit, difference, out=np.zeros_like(Td), where=~equal)
    F = _comm(S1, Td) + 0.5 * _comm(S1, To)
    K = np.where(equal, F, 0.0)
    S2 = np.divide(F, difference, out=np.zeros_like(F), where=~equal)
    channels = {}
    p = adapter.p1_indices
    for level in np.unique(model.D):
        if level == 1:
            continue
        q = np.flatnonzero(model.D == level)
        coupling = model.V_unit[np.ix_(p, q)]
        if np.any(coupling):
            channels[int(level)] = _normal(coupling @ coupling.T / (1 - level),
                                           "signed virtual channel")
    for name, matrix in (("S1", S1), ("S2", S2), ("F", F), ("K", K)):
        _normal(matrix, name)
    return SWCoefficients(adapter, Td, To, F, S1, S2, K, channels)


@_guard
def effective_model(L, C=1.0, g=40.0):
    """Build the full second-order block model; g=0 is undefined and raises.

    Only normal binary64 scales are supported. Powers and retained scaled
    coefficients are checked rather than silently underflowing to zero.
    """
    adapter = neutral_model(L, C, g)
    model = adapter.model
    if model.g == 0:
        raise ValueError("Schrieffer-Wolff perturbation requires g>0")
    coefficients = sw_coefficients(adapter)
    lam = _div(model.C, model.g, "lambda=C/g")
    lam2 = _mul(lam, lam, "lambda squared")
    # Both are used by the remainder and source-weight expansion.
    _mul(lam2, lam, "lambda cubed")
    _mul(lam2, lam2, "lambda fourth power")
    second_scale = _mul(model.C, lam, "C squared/g")
    S = _normal(_scale(coefficients.S1, lam, "lambda S1")
                + _scale(coefficients.S2, lam2, "lambda squared S2"), "S")
    H2dim = _normal(np.diag(model.D) + _scale(coefficients.Td, lam, "lambda Td")
                     + _scale(coefficients.K, lam2, "lambda squared K"), "H2/g")
    H2 = _normal(np.diag(_scale(model.D, model.g, "gD"))
                 + _scale(coefficients.Td, model.C, "C Td")
                 + _scale(coefficients.K, second_scale, "C squared K/g"), "H2")
    p = adapter.p1_indices
    H1 = H2[np.ix_(p, p)].copy()
    E0 = -_mul(4 * model.L, second_scale, "ground shift")
    gaps1 = _normal(H1 - E0 * np.eye(len(p)), "centered P1 matrix")
    return EffectiveModel(adapter, coefficients, lam, S, H2dim, H2, H1, E0, gaps1)


@_guard
def dressed_source(effective, m):
    """Return all-D transformed rho and vacuum source through second order.

    Omega is the transformed ground through second order only. Rrho below
    does not bound the omitted true-ground-state correction. Exactly m=0
    (including aliases) gives structural zeros in every coefficient.
    """
    model = effective.model
    m = _momentum(m, model.L)
    diagonal = density_diagonal(model, m)
    rho = np.diag(diagonal)
    coeff = effective.coefficients
    rho1 = _normal(coeff.S1 * (diagonal[None, :] - diagonal[:, None]), "rho1")
    rho2 = _normal(coeff.S2 * (diagonal[None, :] - diagonal[:, None])
                   + 0.5 * _comm(coeff.S1, rho1), "rho2")
    omega = effective.adapter.omega_index
    z1, z2 = rho1[:, omega].copy(), rho2[:, omega].copy()
    first = _scale(z1, effective.lam, "first-order source")
    second = _scale(z2, _mul(effective.lam, effective.lam, "lambda squared"),
                    "second-order source")
    z = _normal(first + second, "truncated source")
    w2 = _mul(_norm(first), _norm(first), "second-order source weight")
    w3 = _mul(2.0, float(np.vdot(first, second).real), "cubic source weight")
    w4 = _mul(_norm(second), _norm(second), "partial quartic source weight")
    total = _normal(float(np.vdot(z, z).real), "squared truncated source")
    if np.any(z != 0) and total == 0:
        raise ValueError("numerically unresolved squared truncated source")
    by_D = {int(d): z[model.D == d].copy() for d in np.unique(model.D)}
    return DressedSource(m, rho, rho1, rho2, z1, z2, z,
                         z[effective.adapter.p1_indices].copy(), by_D,
                         w2, w3, w4, total)


def _hermitian_norm(matrix):
    """Spectral norm without an SVD for a Hermitian coefficient matrix."""
    return float(np.max(np.abs(np.linalg.eigvalsh(matrix))))


def _block_spectra(effective):
    model = effective.model
    return {int(d): np.linalg.eigvalsh(effective.H2_dimensionless[np.ix_(model.D == d, model.D == d)])
            for d in np.unique(model.D)}


@_guard
def remainder_certificate(effective, m=None):
    """Conservative BCH and sorted Weyl bounds, with explicit block gate.

    RH is dimensionless, energy_bound=g RH. Rrho (when m is supplied)
    bounds ||U rho U†-(rho+lambda rho1+lambda²rho2)|| only. When the D0
    gap is isolated, source_error_bound additionally includes the transformed
    ground-vector error. No individual residue bound follows. Convex hull
    intervals give a sufficient (not necessary) effective-block labeling gate.
    """
    c, model, lam = effective.coefficients, effective.model, effective.lam
    s1 = _hermitian_norm(1j * c.S1)
    s2 = _hermitian_norm(1j * c.S2)
    d = float(np.max(model.D))
    t = _hermitian_norm(model.V_unit)
    l2 = _mul(lam, lam, "lambda squared")
    l3 = _mul(l2, lam, "lambda cubed")
    l4 = _mul(l2, l2, "lambda fourth power")
    s = _normal(_mul(lam, s1, "lambda s1") + _mul(l2, s2, "lambda squared s2"), "s")
    s_squared = _mul(s, s, "s squared")
    s_cubed = _mul(s_squared, s, "s cubed")
    cross = _mul(_mul(4 * l3, s1, "cross s1"), s2, "cross s1 s2")
    quartic = _mul(_mul(2 * l4, s2, "quartic s2"), s2, "quartic s2 squared")
    terms = {
        "D_taylor_remainder": _mul(4 / 3 * s_cubed, d, "D Taylor remainder"),
        "T_taylor_remainder": _mul(_mul(2 * lam, s_squared, "T Taylor factor"), t, "T Taylor remainder"),
        "D_cubic_cross": _mul(cross, d, "D cubic cross"),
        "D_quartic": _mul(quartic, d, "D quartic"),
        "T_S2": _mul(_mul(2 * l3, s2, "T S2 factor"), t, "T S2 remainder"),
    }
    RH = _normal(sum(terms.values()), "Hamiltonian remainder")
    energy_bound = _mul(model.g, RH, "energy remainder")
    rho_norm, Rrho = None, None
    if m is not None:
        m = _momentum(m, model.L)
        rho_norm = float(np.max(np.abs(density_diagonal(model, m))))
        Rrho = _mul(rho_norm, _normal(4 / 3 * s_cubed + cross + quartic,
                                    "observable remainder factor"), "observable remainder")
    spectra = _block_spectra(effective)
    intervals = {d: [float(v[0]), float(v[-1])] for d, v in spectra.items()}
    e0 = intervals[0][0]
    ground_separation = min(v[0] - e0 for d, v in intervals.items() if d != 0)
    p1low, p1high = intervals[1]
    # Negative signed separation means that convex hulls overlap.
    p1_separation = min(max(v[0] - p1high, p1low - v[1])
                        for d, v in intervals.items() if d != 1)
    twice = _mul(2.0, RH, "twice Hamiltonian remainder")
    resolution = _mul(128 * _EPS * len(model.basis),
                      max(1.0, max(abs(x) for v in intervals.values() for x in v), twice),
                      "block separation resolution")
    ground_available = ground_separation - twice > resolution
    p1_available = p1_separation - twice > resolution
    available = bool(ground_available and p1_available)
    ground_vector_error, source_error = None, None
    if ground_available:
        leakage = _div(RH, _normal(ground_separation - RH, "ground resolvent margin"),
                       "ground leakage bound")
        ground_vector_error = _mul(np.sqrt(2.0), leakage, "ground vector error bound")
        if m is not None:
            source_error = _normal(Rrho + _mul(rho_norm, ground_vector_error,
                                               "ground contribution to source bound"),
                                   "true-ground source error bound")
    return {
        "available": True, "RH": RH, "Rrho": Rrho, "energy_bound": energy_bound,
        "terms": terms, "norms": {"s1": s1, "s2": s2, "d": d, "t": t,
                                   "s": s, "rho": rho_norm},
        "block_identification_available": available,
        "ground_identification_available": bool(ground_available),
        "P1_isolation_available": bool(p1_available),
        "block_intervals_dimensionless": intervals,
        "ground_separation_dimensionless": ground_separation,
        "P1_separation_dimensionless": p1_separation,
        "required_separation_dimensionless": twice,
        "separation_resolution_proxy": resolution,
        "block_identification_condition": "D0 below all other block hulls and P1 hull disjoint from every other hull by more than 2 RH, plus a roundoff screen",
        "gap_error_bound": _mul(2.0, energy_bound, "gap remainder") if available else None,
        "reason": None if available else "Sufficient effective-block interval separation unavailable; no exact excited-block assignment.",
        "global_sorted_eigenvalue_bound_available": True,
        "individual_weight_bound_available": False,
        "ground_vector_error_bound": ground_vector_error,
        "source_error_bound": source_error,
        "true_ground_source_bound_available": source_error is not None,
        "ground_vector_bound_scope": "phase-aligned transformed exact ground versus Omega, sqrt(2)*RH/(Delta0-RH)",
        "observable_bound_scope": "Rrho is conjugation only; source_error_bound additionally includes the isolated-ground vector error",
        "dynamics_bound_formula": "min(2, abs(time)*energy_bound), hbar=1, after the same U",
        "roundoff_included": False,
    }


@_guard
def dynamics_error_bound(effective, time):
    """Bound transformed full versus truncated propagators, in energy units."""
    time = _real(time, "time")
    bound = remainder_certificate(effective)["energy_bound"]
    # Compare before multiplying, so a mathematically capped bound cannot overflow.
    if time == 0 or bound == 0:
        return 0.0
    # Binary64 numbers have exact integer ratios. Compare their product to
    # two without an overflowing product or a subnormal saturation reciprocal.
    time_num, time_den = abs(time).as_integer_ratio()
    bound_num, bound_den = float(bound).as_integer_ratio()
    if time_num * bound_num >= 2 * time_den * bound_den:
        return 2.0
    return _mul(abs(time), bound, "dynamics bound")


def _unitary_from_generator(S):
    values, vectors = np.linalg.eigh(1j * S)
    return _normal((vectors * np.exp(-1j * values)) @ vectors.conj().T, "SW unitary")


def _diagonalize(matrix):
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return np.zeros(len(matrix)), np.eye(len(matrix)), 1.0
    scaled = _normal(np.divide(matrix, scale), "scaled Hamiltonian")
    if np.any((matrix != 0) & (scaled == 0)):
        raise ValueError("numerically unresolved scaled Hamiltonian: underflow")
    values, vectors = np.linalg.eigh(scaled)
    return _scale(values, scale, "eigenvalues"), vectors, scale


def _sum_rules(matrix, reference, source, measure):
    total = _mul(_norm(source), _norm(source), "sum-rule total")
    centered = matrix - reference * np.eye(len(matrix))
    first = _normal(float(np.vdot(source, centered @ source).real), "sum-rule first moment")
    return {"total_weight_direct": total, "first_moment_direct": first,
            "total_weight_residual": abs(total - measure["total_weight"]),
            "first_moment_residual": abs(first - measure["first_moment"])}


@_guard
def case_report(L, m=1, C=1.0, g=40.0):
    """Exact versus partially resummed full/P1 spectra and grouped density weights.

    Reports at g=0 retain the exact response and explicitly mark perturbation
    unavailable. Weak-coupling truncated spectra remain diagnostics, without
    labeling their lines as exact excitations or claiming certified weights.
    """
    adapter = neutral_model(L, C, g)
    model = adapter.model
    m = _momentum(m, model.L)
    scale = float(np.max(np.abs(model.H)))
    proxy = 64 * _EPS * len(model.basis)
    hop_relative = _div(model.C, scale, "relative hopping")
    if proxy >= _mul(1e-6, hop_relative, "hopping resolution threshold"):
        raise ValueError("numerically unresolved report eigensystem relative to hopping")
    source_relative = min(1.0, _div(model.C, model.g, "relative response")) if model.g else 1.0
    if m and proxy >= _mul(1e-6, source_relative, "source resolution threshold"):
        raise ValueError("numerically unresolved report eigensystem relative to density response")
    exact_values, exact_vectors, scale = _diagonalize(model.H)
    exact_ground = exact_vectors[:, 0].copy()
    if exact_ground[adapter.omega_index] < 0:
        exact_ground *= -1
    rho_diagonal = density_diagonal(model, m)
    exact_source = _normal(rho_diagonal * exact_ground, "exact source")
    exact_gaps = _normal(exact_values - exact_values[0], "exact gaps")
    tolerance = _mul(128 * _EPS * len(model.basis), scale, "exact grouping tolerance")
    exact_measure = grouped_spectral_measure(exact_gaps, exact_vectors, exact_source, tolerance)
    report = {
        "parameters": {"L": model.L, "N": model.N, "q": model.L, "C": model.C, "g": model.g, "m": m},
        "dimensions": {"full": len(model.basis), "P0": 1, "P1": len(adapter.p1_indices)},
        "perturbation_available": model.g > 0,
        "perturbation_unavailable_reason": None if model.g else "g=0 makes C/g perturbation undefined; the exact response need not vanish.",
        "ground_energy": float(exact_values[0]),
        "exact_eigenvalues": exact_values.tolist(), "exact_gaps": exact_gaps.tolist(),
        "exact_density_measure": exact_measure,
        "exact_sum_rules": _sum_rules(model.H, exact_values[0], exact_source, exact_measure),
        "exact_elastic_weight": float(abs(np.vdot(exact_ground, exact_source)) ** 2),
        "remainder_certificate": None, "effective": None, "numerical_checks": None,
        "neutrality_assumed": True,
        "physical_predictions": {"masses": None, "mixing": None, "bound_particle": None},
    }
    if model.g == 0:
        return report
    effective = effective_model(L, C, g)
    source = dressed_source(effective, m)
    certificate = remainder_certificate(effective, m)
    approximate_values, approximate_vectors, approximate_scale = _diagonalize(effective.H2)
    approximate_gaps = _normal(approximate_values - effective.E0, "effective gaps")
    approximate_measure = grouped_spectral_measure(
        approximate_gaps, approximate_vectors, source.z,
        _mul(128 * _EPS * len(model.basis), approximate_scale, "effective grouping tolerance"))
    approximate_measure["interpretation"] = "partially resummed full block model centered on E0[2], not certified exact line weights"
    p1_values, p1_vectors, p1_scale = _diagonalize(effective.gaps1)
    p1_measure = grouped_spectral_measure(
        p1_values, p1_vectors, source.p1_source,
        _mul(128 * _EPS * len(adapter.p1_indices), p1_scale, "P1 grouping tolerance"))
    p1_measure["interpretation"] = "partially resummed P1 gaps H1[2]-E0[2] and squared truncated source; only partial order-four weight"
    U = _unitary_from_generator(effective.S)
    Hdim = np.diag(model.D) + _scale(model.V_unit, effective.lam, "dimensionless exact hopping")
    h_residual = _norm(U @ Hdim @ U.conj().T - effective.H2_dimensionless)
    rho_truncated = source.rho + _scale(source.rho1, effective.lam, "lambda rho1") + _scale(
        source.rho2, _mul(effective.lam, effective.lam, "lambda squared"), "lambda squared rho2")
    rho_residual = _norm(U @ source.rho @ U.conj().T - rho_truncated)
    omega = np.zeros(len(model.basis))
    omega[adapter.omega_index] = 1.0
    sorted_error = float(np.max(np.abs(exact_values - approximate_values)))
    cluster = None
    if certificate["block_identification_available"]:
        # Weyl's sorted pairing identifies block labels only after the gate above.
        labelled = sorted((float(e), level) for level, values in _block_spectra(effective).items() for e in values)
        indices = [i for i, (_, level) in enumerate(labelled) if level == 1]
        exact_p1_gaps = exact_gaps[indices]
        cluster = {"count": len(indices), "exact_sorted_indices": indices,
                   "exact_gaps": exact_p1_gaps.tolist(), "effective_gaps": p1_values.tolist(),
                   "max_gap_error": float(np.max(np.abs(exact_p1_gaps - p1_values))),
                   "gap_error_bound": certificate["gap_error_bound"],
                   "individual_weights_certified": False}
    # One fixed dimensionless time tau=g*time=1; no fit to a favorable time.
    dimensionless_values, dimensionless_vectors = np.linalg.eigh(Hdim)
    block_values, block_vectors = np.linalg.eigh(effective.H2_dimensionless)
    exact_propagator = (dimensionless_vectors * np.exp(-1j * dimensionless_values)) @ dimensionless_vectors.conj().T
    block_propagator = (block_vectors * np.exp(-1j * block_values)) @ block_vectors.conj().T
    dynamics_residual = _norm(U @ exact_propagator @ U.conj().T - block_propagator)
    channel_summary = {}
    for level, matrix in effective.coefficients.p1_channels.items():
        values = np.linalg.eigvalsh(matrix)
        channel_summary[level] = {"min_eigenvalue": float(values[0]), "max_eigenvalue": float(values[-1]),
                                  "norm": float(np.max(np.abs(values))), "denominator": 1 - level}
    checks = {
        "global_sorted_eigenvalue_max_error": sorted_error,
        "global_sorted_eigenvalue_error_bound": certificate["energy_bound"],
        "Hamiltonian_conjugation_residual_dimensionless": h_residual,
        "observable_conjugation_residual": rho_residual,
        "unitarity_residual": _norm(U.conj().T @ U - np.eye(len(U))),
        "transformed_ground_minus_omega_norm": _norm(U @ exact_ground - omega),
        "transformed_exact_source_minus_truncated_norm": _norm(U @ exact_source - source.z),
        "true_ground_source_error_is_not_bounded_by_Rrho": True,
        "dynamics": {"time": _div(1.0, model.g, "diagnostic time"), "g_times_time": 1.0,
                     "operator_error": dynamics_residual, "bound": min(2.0, certificate["RH"])},
        "source_weight_polynomial_residual": abs(source.truncated_weight - source.weight_order2
                                                  - source.weight_order3 - source.weight_partial_order4),
        "total_weight_effective_absolute_error": abs(exact_measure["total_weight"] - approximate_measure["total_weight"]),
        "arithmetic_resolution_proxy_relative": proxy, "roundoff_certified": False,
    }
    report.update(remainder_certificate=certificate, numerical_checks=checks, effective={
        "lambda": effective.lam, "ground_shift": effective.E0,
        "ground_shift_coefficient": float(effective.coefficients.K[adapter.omega_index, adapter.omega_index]),
        "ground_energy_absolute_error": abs(float(exact_values[0]) - effective.E0),
        "eigenvalues": approximate_values.tolist(), "P1_gaps": p1_values.tolist(),
        "full_density_measure": approximate_measure, "P1_density_measure": p1_measure,
        "full_sum_rules": _sum_rules(effective.H2, effective.E0, source.z, approximate_measure),
        "P1_sum_rules": _sum_rules(effective.H1, effective.E0, source.p1_source, p1_measure),
        "source_weights": {"order2": source.weight_order2, "order3": source.weight_order3,
                           "through_order3": source.weight_order2 + source.weight_order3,
                           "partial_order4": source.weight_partial_order4, "squared_truncated": source.truncated_weight,
                           "complete_fourth_order": False,
                           "by_D": {d: _mul(_norm(z), _norm(z), "D-block source weight") for d, z in source.by_D.items()}},
        "signed_P1_channels": channel_summary, "identified_P1_cluster": cluster,
        "partially_resummed": True, "individual_weights_certified": False,
    })
    return report


def demonstration_report():
    """Frozen L=5 strong/weak and zero-mode cases; strict JSON-compatible data."""
    controls = [dict(zip(("L", "C", "g", "m"), case)) for case in FROZEN_CONTROLS]
    return {"model_id": MODEL_ID, "frozen_controls": controls,
            "controls_frozen_before_evaluation": True, "empirical_calibration": False,
            "caps": {"fixed_number_dimension": MAX_DENSE_DIM, "unit_filling_sites": [3, 6]},
            "cases": [case_report(**case) for case in controls],
            "limitations": list(LIMITATIONS),
            "physical_predictions": {"masses": None, "mixing": None, "bound_particle": None}}
