"""Neutral finite-repulsion diagnostics in the stipulated quantum Bose ring.

N=L=q is a state-selection assumption, not confinement or vacuum selection.
P1 denotes D=1, NOT the inherited hard-core projector. Compression spectra are
not full energies. Analytic bounds exclude roundoff; arithmetic screens and
spectral residuals are numerical diagnostics, not rigorous error certificates.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from bpr.substrate_fermionization import (
    FixedNumberModel, MAX_DENSE_DIM, _EPS, _div, _guard, _integer, _mul,
    _norm, _normal, _real, _scale, fixed_number_model,
)

MODEL_ID = "conditional-substrate-neutral-response-v1"
# Frozen before evaluation; the zero modes are structural controls.
FROZEN_CONTROLS = ((5, 1.0, 40.0, 1), (5, 1.0, 0.7, 1),
                   (5, 1.0, 40.0, 0), (5, 1.0, 0.7, 0))
LIMITATIONS = (
    "The quantum Bose-Hubbard prescription is stipulated, not derived from classical dynamics.",
    "N=L=q neutrality is assumed; neither confinement nor a vacuum population is selected.",
    "Defect compression eigenvalues are not individually certified physical excitation energies.",
    "Excited Schur corrections are indefinite and energy dependent, not autonomous dynamics.",
    "The density bound certifies total weight only, not individual lines or first-band confinement.",
    "Analytic approximation bounds exclude roundoff; arithmetic screens are heuristic.",
    "Number-preserving bosonic bilinears do not supply fermionic CAR or physical particle statistics.",
    "No empirical calibration, physical masses, mixing, or theory-of-everything conclusion is supplied.",
)


@dataclass(frozen=True)
class NeutralModel:
    model: FixedNumberModel
    omega_index: int
    p1_indices: np.ndarray
    q1_indices: np.ndarray
    A1: np.ndarray
    B1: np.ndarray
    K1: np.ndarray


def _momentum(m, L):
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    return int(m % L)


@_guard
def neutral_model(L, C=1.0, g=40.0):
    """Full unit-filling adapter; inherited preallocation cap accepts L=3..6."""
    model = fixed_number_model(L, L, C, g)
    p = np.flatnonzero(model.D == 1)
    q = np.flatnonzero(model.D != 1)
    omega = model.basis.index((1,) * model.L)
    return NeutralModel(model, omega, p, q, model.H[np.ix_(p, p)],
                        model.H[np.ix_(p, q)], model.H[np.ix_(q, q)])


def translation(model):
    """Full occupation translation T|n> with particles shifted x -> x+1."""
    index = {state: i for i, state in enumerate(model.basis)}
    result = np.zeros_like(model.H)
    for j, state in enumerate(model.basis):
        result[index[(state[-1],) + state[:-1]], j] = 1.0
    return result


def _defect_index(adapter):
    return {(state.index(2), state.index(0)): i
            for i in adapter.p1_indices for state in (adapter.model.basis[i],)}


@_guard
def momentum_isometry(adapter, m):
    """Full-space |k,r> columns, r=1..L-1, T eigenvalue exp(+ik).

    Coefficients are exp(-ik*h)/sqrt(L), with doublon at h+r and hole h.
    """
    model = adapter.model
    m = _momentum(m, model.L)
    k = 2 * np.pi * m / model.L
    index = _defect_index(adapter)
    result = np.zeros((len(model.basis), model.L - 1), dtype=complex)
    for h in range(model.L):
        for r in range(1, model.L):
            result[index[((h + r) % model.L, h)], r - 1] = np.exp(-1j * k * h) / np.sqrt(model.L)
    return result


@_guard
def defect_compression(adapter):
    """Independent defect hopping oracle, returned in adapter.p1_indices order.

    Doublon hops cost -2C, hole hops -C. Collision hops leave P1 and are
    omitted, never wrapped across the relative-coordinate chain endpoints.
    """
    model = adapter.model
    full_index = _defect_index(adapter)
    local = {full: i for i, full in enumerate(adapter.p1_indices)}
    result = np.eye(len(local)) * model.g
    for (d, h), col in full_index.items():
        for step in (-1, 1):
            dn, hn = (d + step) % model.L, (h + step) % model.L
            if dn != h:
                result[local[full_index[(dn, h)]], local[col]] -= 2 * model.C
            if hn != d:
                result[local[full_index[(d, hn)]], local[col]] -= model.C
    return _normal(result, "defect compression")


@_guard
def momentum_block(adapter, m):
    """Independent open-chain compression with row(r+1),col(r)=-C(2+e^-ik)."""
    model = adapter.model
    m = _momentum(m, model.L)
    k = 2 * np.pi * m / model.L
    result = np.eye(model.L - 1, dtype=complex) * model.g
    hop = -model.C * (2 + np.exp(-1j * k))
    for r in range(model.L - 2):
        result[r + 1, r] = hop
        result[r, r + 1] = hop.conjugate()
    return _normal(result, "momentum block")


@_guard
def compression_eigenvalues(adapter, m=None):
    """Sorted analytic compression oracle, all momenta if m is omitted."""
    model = adapter.model
    momenta = range(model.L) if m is None else (_momentum(m, model.L),)
    values = [model.g - 2 * model.C * np.sqrt(5 + 4 * np.cos(2 * np.pi * k / model.L))
              * np.cos(np.pi * j / model.L)
              for k in momenta for j in range(1, model.L)]
    return _normal(np.sort(values), "compression eigenvalues")


@_guard
def density_diagonal(model, m):
    """rho_m=L^-1/2 sum exp(-ikx)(n_x-1); m=0 is structurally zero at N=L.

    This API requires unit filling; subtraction of one is an observable
    convention and does not insert compensating background charge into H.
    """
    if model.N != model.L:
        raise ValueError("neutral density requires N=L")
    m = _momentum(m, model.L)
    if m == 0:
        return np.zeros(len(model.basis), dtype=complex)
    phase = np.exp(-2j * np.pi * m * np.arange(model.L) / model.L)
    return _normal((np.asarray(model.basis) - 1) @ phase / np.sqrt(model.L), "density diagonal")


@_guard
def bilinear(model, d, h):
    """Complete fixed-number bosonic a†_d a_h, not a P1-compressed algebra."""
    d = _integer(d, "d", 0, model.L - 1)
    h = _integer(h, "h", 0, model.L - 1)
    index = {state: i for i, state in enumerate(model.basis)}
    result = np.zeros_like(model.H)
    for col, state in enumerate(model.basis):
        if d == h:
            result[col, col] = state[d]
        elif state[h]:
            moved = list(state)
            moved[h] -= 1
            moved[d] += 1
            result[index[tuple(moved)], col] = np.sqrt(state[h] * (state[d] + 1))
    return result


@_guard
def excited_certificate(adapter):
    """Separated D=1 cluster, resolvent correction and eigenstate leakage bounds.

    For g>2v, precisely one eigenvalue is in [-v,v], L(L-1) in
    [g-v,g+v], and the rest are >=2g-v. dist(E,spec K1)>=g-2v on
    the middle interval. This is not a sorted full-eigenvalue error bound.
    """
    model = adapter.model
    v = _mul(2 * model.L, model.C, "hopping bound")
    twice_v = _mul(2, v, "twice hopping bound")
    bnorm = _norm(adapter.B1)
    available = model.g > twice_v
    result = {"available": available, "condition": "g > 2v, v=2CL",
              "v": v, "delta": None, "B1_norm": bnorm,
              "schur_correction_bound": None, "leakage_ratio_bound": None,
              "ground_interval": None, "excited_interval": None,
              "higher_energy_lower_bound": None,
              "ground_count": None, "excited_count": None,
              "reason": None if available else "Sufficient separation condition unavailable, not disproved.",
              "roundoff_included": False}
    if not available:
        return result
    delta = _normal(model.g - twice_v, "separation margin")
    if delta <= _mul(64 * _EPS, max(model.g, twice_v), "margin resolution"):
        raise ValueError("numerically unresolved positive separation margin")
    ratio = _div(bnorm, delta, "excited leakage bound")
    result.update(delta=delta, leakage_ratio_bound=ratio,
                  schur_correction_bound=_mul(bnorm, ratio, "Schur bound"),
                  ground_interval=[-v, v], excited_interval=[model.g - v, model.g + v],
                  higher_energy_lower_bound=_normal(2 * model.g - v, "higher band bound"),
                  ground_count=1, excited_count=model.L * (model.L - 1))
    return result


@_guard
def excited_schur_correction(adapter, E):
    """-B1(K1-E)^-1 B1† evaluated separately from A1, with no sign claim.

    Algebraically available outside the certified interval if the resolvent
    is numerically nonsingular. The Q1 block includes Omega and is indefinite
    across the certified excited interval.
    """
    model = adapter.model
    E = _real(E, "E")
    scale = max(model.C, model.g, abs(E))
    q = adapter.q1_indices
    K = _scale(model.V_unit[np.ix_(q, q)], _div(model.C, scale, "C/solve scale"), "scaled K1 hopping")
    K += np.diag(_scale(model.D[q], _div(model.g, scale, "g/solve scale"), "scaled K1 interaction"))
    K -= np.eye(len(q)) * _div(E, scale, "E/solve scale")
    singular = np.linalg.svd(K, compute_uv=False)
    if singular[-1] <= 64 * _EPS * max(1.0, float(singular[0])):
        raise ValueError("singular or numerically unresolved K1-E resolvent")
    B = model.V_unit[np.ix_(adapter.p1_indices, q)]
    coefficient = -B @ np.linalg.solve(K, B.T)
    factor = _mul(model.C, _div(model.C, scale, "C/solve scale"), "excited Schur scale")
    return _scale(coefficient, factor, "excited Schur correction")


@_guard
def virtual_ground_component(adapter):
    """w=-V Omega/g. Undefined at g=0, even for the zero density mode."""
    model = adapter.model
    if model.g == 0:
        raise ValueError("virtual ground component requires g>0")
    return _scale(-model.V_unit[:, adapter.omega_index], _div(model.C, model.g, "C/g"), "virtual ground component")


@_guard
def density_certificate(adapter, m):
    """Leading density weight and conservative total-weight error bound.

    chi is the exact ground component perpendicular to Omega, with positive
    Omega overlap. ||chi-w||<=eta uses D V Omega=V Omega and alpha>=
    1/sqrt(1+r²). No assertion about individual exact spectral lines follows.
    """
    model = adapter.model
    m = _momentum(m, model.L)
    cert = excited_certificate(adapter)
    d = 0.0 if m == 0 else float(np.sqrt(model.L))
    leading = None
    if model.g > 0:
        amplitude = 0.0 if m == 0 else _mul(4 * np.sin(np.pi * m / model.L), _div(model.C, model.g, "C/g"), "leading amplitude")
        leading = _mul(amplitude, amplitude, "leading density weight")
    result = {"available": cert["available"], "m": m, "leading_weight": leading,
              "b": _mul(2 * np.sqrt(model.L), model.C, "ground coupling bound"),
              "r": None, "eta": None, "density_norm": d,
              "total_weight_error_bound": None, "leading_signal_resolved_by_bound": None,
              "roundoff_included": False,
              "reason": cert["reason"]}
    if not cert["available"]:
        return result
    b = result["b"]
    r = _div(b, cert["delta"], "ground leakage bound")
    bg = _div(b, model.g, "ground first-order norm")
    eta = _normal(_mul(_mul(r, r, "r squared") / 2, bg, "normalization remainder")
                  + _mul(_div(2 * cert["v"], cert["delta"], "resolvent remainder ratio"), bg, "resolvent remainder"), "ground remainder bound")
    de = _mul(d, eta, "density remainder")
    bound = _mul(de, 2 * np.sqrt(leading) + de, "density total-weight bound")
    result.update(r=r, eta=eta, total_weight_error_bound=bound,
                  leading_signal_resolved_by_bound=bool(leading > bound))
    return result


@_guard
def grouped_spectral_measure(eigenvalues, eigenvectors, source, tolerance=None):
    """Group an orthonormal Hermitian eigensystem without transitive chaining.

    Tolerance is ABSOLUTE energy, default 128*eps*n*max(abs(energies)).
    Each group's entire energy span must fit the tolerance measured from its
    first member. Report actual energy spreads and sum w_i E_i for the first
    moment, never replace energies by a representative before taking moments.
    Exact-degeneracy sums are invariant to rotations of the corresponding
    eigenvectors. Eigenvectors are columns; energies need not be sorted.
    """
    raw_values = np.asarray(eigenvalues)
    if np.iscomplexobj(raw_values):
        raise TypeError("eigenvalues must be real")
    values = _normal(np.asarray(raw_values, dtype=float), "spectral energies")
    vectors = _normal(np.asarray(eigenvectors, dtype=complex), "spectral vectors")
    source = _normal(np.asarray(source, dtype=complex), "spectral source")
    if values.ndim != 1 or not len(values) or vectors.ndim != 2 or source.ndim != 1:
        raise ValueError("invalid spectral measure shapes")
    if vectors.shape != (len(source), len(values)):
        raise ValueError("eigenvector columns and source dimensions must match")
    energy_scale = float(np.max(np.abs(values)))
    if tolerance is None:
        tolerance = _mul(128 * _EPS * len(values), energy_scale, "grouping tolerance")
    else:
        tolerance = _real(tolerance, "tolerance", lower=0)
    order = np.argsort(values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    amplitudes = _normal(vectors.conj().T @ source, "spectral amplitudes")
    weights = _normal(np.abs(amplitudes) ** 2, "spectral weights")
    if np.any((amplitudes != 0) & (weights == 0)):
        raise ValueError("numerically unresolved spectral weights")
    scaled = values / energy_scale if energy_scale else values
    moments = _normal(weights * scaled, "scaled spectral moments")
    groups = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] - values[start] <= tolerance:
            end += 1
        weight = float(np.sum(weights[start:end]))
        first = _mul(energy_scale, float(np.sum(moments[start:end])), "group first moment")
        # Midrange representative is descriptive only, not used in moments.
        representative = float(values[start] / 2 + values[end - 1] / 2)
        groups.append({"energy": representative, "energy_min": float(values[start]),
                       "energy_max": float(values[end - 1]),
                       "energy_spread": float(values[end - 1] - values[start]),
                       "weight": weight, "first_moment": first,
                       "multiplicity": end - start})
        start = end
    return {"groups": groups, "total_weight": float(np.sum(weights)),
            "first_moment": _mul(energy_scale, float(np.sum(moments)), "spectral first moment"),
            "tolerance": tolerance, "grouping": "bounded span from first member, no transitive chaining"}


@_guard
def bilinear_diagnostics(model):
    """Representative full-space gl(L) identities and explicit non-CAR witness.

    [E_ab,E_cd]=delta_bc E_ad-delta_ad E_cb. For N>=2, E_01 squared is
    nonzero (where a fermionic hopping bilinear squared would vanish).
    This particular witness is unavailable for N<2.
    """
    e01, e12, e02 = bilinear(model, 0, 1), bilinear(model, 1, 2), bilinear(model, 0, 2)
    e10 = e01.T
    n0, n1 = bilinear(model, 0, 0), bilinear(model, 1, 1)
    witness = None
    if model.N >= 2:
        initial = [0] * model.L
        initial[1] = model.N
        final = initial.copy()
        final[1] -= 2
        final[0] += 2
        square = e01 @ e01
        witness = {"operator": "(a†_0 a_1)^2", "initial": initial,
                   "final": final, "matrix_element": float(square[model.basis.index(tuple(final)), model.basis.index(tuple(initial))])}
    return {"chain_commutator_residual": _norm(e01 @ e12 - e12 @ e01 - e02),
            "reverse_commutator_residual": _norm(e01 @ e10 - e10 @ e01 - n0 + n1),
            "neutral_number_commutator_residual": _norm(model.N * e01 - e01 * model.N),
            "non_car_witness": witness,
            "non_car_witness_unavailable_reason": "Squared-hopping witness requires N>=2." if witness is None else None,
            "scope": "complete fixed-number bosonic space, not P1-compressed algebra"}


@_guard
def case_report(L, C=1.0, g=40.0, m=1):
    """One scaled full eigendecomposition shared by all exact case diagnostics.

    Heuristic screens require hopping and requested first-order source
    amplitude to exceed numerical resolution proxies by 1e6. This rejects
    unresolved small response rather than reporting apparent precision.
    """
    adapter = neutral_model(L, C, g)
    model = adapter.model
    m = _momentum(m, model.L)
    cert = excited_certificate(adapter)
    density_cert = density_certificate(adapter, m)
    scale = float(np.max(np.abs(model.H)))
    proxy = 64 * _EPS * len(model.basis)
    relative_hop = _div(model.C, scale, "relative hopping scale")
    if proxy >= _mul(1e-6, relative_hop, "hopping resolution threshold"):
        raise ValueError("numerically unresolved report eigensystem relative to hopping")
    relative_source = min(1.0, _div(model.C, model.g, "response resolution scale")) if model.g else 1.0
    if m and proxy >= _mul(1e-6, relative_source, "response resolution threshold"):
        raise ValueError("numerically unresolved report eigensystem relative to requested density response")
    Hscaled = _scale(model.H, _div(1.0, scale, "inverse spectral scale"), "scaled Hamiltonian")
    scaled_values, vectors = np.linalg.eigh(Hscaled)
    eigenvalues = _scale(scaled_values, scale, "full eigenvalues")
    ground = vectors[:, 0].copy()
    if ground[adapter.omega_index] < 0:
        ground *= -1
    rho = density_diagonal(model, m)
    source = rho * ground
    gaps_scaled = scaled_values - scaled_values[0]
    gaps = _scale(gaps_scaled, scale, "excitation gaps")
    tolerance = _mul(128 * _EPS * len(model.basis), scale, "full grouping tolerance")
    measure = grouped_spectral_measure(gaps, vectors, source, tolerance)
    total = _mul(_norm(source), _norm(source), "direct total density weight")
    moment = _mul(scale, float(np.vdot(source, Hscaled @ source - scaled_values[0] * source).real), "direct first moment")
    elastic = float(abs(np.vdot(ground, source)) ** 2)
    checks = {"total_weight_direct": total,
              "total_weight_sum_rule_residual": abs(total - measure["total_weight"]),
              "first_moment_direct": moment,
              "first_moment_sum_rule_residual": abs(moment - measure["first_moment"]),
              "elastic_weight": elastic,
              "density_ground_expectation_abs": float(abs(np.vdot(ground, source))),
              "ground_translation_residual": _norm(translation(model) @ ground - ground),
              "full_eigensystem_scaled_residual": _norm(Hscaled @ vectors - vectors * scaled_values),
              "arithmetic_resolution_proxy_relative": proxy,
              "arithmetic_screen_is_not_a_certificate": True,
              "ground_remainder_norm": None, "leading_source_weight": None,
              "total_weight_leading_absolute_error": None,
              "cluster_counts": None, "excited_eigenstate_leakage_ratios": None,
              "schur_midpoint_correction_norm": None}
    leading_measure = None
    if model.g > 0:
        w = virtual_ground_component(adapter)
        leading_source = rho * w
        chi = ground.copy()
        chi[adapter.omega_index] = 0
        checks.update(ground_remainder_norm=_norm(chi - w),
                      leading_source_weight=_mul(_norm(leading_source), _norm(leading_source), "leading source weight"),
                      total_weight_leading_absolute_error=abs(total - density_cert["leading_weight"]))
        if cert["available"]:
            U = momentum_isometry(adapter, m)
            block = momentum_block(adapter, m)
            block_scale = max(model.C, model.g)
            be, bu = np.linalg.eigh(block / block_scale)
            leading_measure = grouped_spectral_measure(_scale(be, block_scale, "leading compression energies"), bu, U.conj().T @ leading_source)
            leading_measure["interpretation"] = "P1 leading measure, zero reference ground energy; not matched exact lines"
    if cert["available"]:
        low, high = cert["excited_interval"]
        selected = np.flatnonzero((eigenvalues >= low) & (eigenvalues <= high))
        checks["cluster_counts"] = {"ground": int(np.count_nonzero((eigenvalues >= -cert["v"]) & (eigenvalues <= cert["v"]))),
                                    "excited": len(selected),
                                    "higher": int(np.count_nonzero(eigenvalues >= cert["higher_energy_lower_bound"]))}
        checks["excited_eigenstate_leakage_ratios"] = [
            _div(_norm(vectors[adapter.q1_indices, j]), _norm(vectors[adapter.p1_indices, j]), "observed excited leakage") for j in selected]
        checks["schur_midpoint_correction_norm"] = _norm(excited_schur_correction(adapter, model.g))
    return {"parameters": {"L": model.L, "N": model.N, "q": model.L,
                           "C": model.C, "g": model.g, "m": m},
            "dimensions": {"full": len(model.basis), "P0": 1,
                           "P1": len(adapter.p1_indices), "Q1": len(adapter.q1_indices)},
            "excited_certificate": cert, "density_certificate": density_cert,
            "ground_energy": float(eigenvalues[0]), "exact_eigenvalues": eigenvalues.tolist(),
            "exact_density_measure": measure, "leading_compression_measure": leading_measure,
            "leading_measure_suppressed_reason": None if leading_measure is not None else "Sufficient cluster separation unavailable; no approximate physical-looking spectrum.",
            "numerical_checks": checks, "bilinear_diagnostics": bilinear_diagnostics(model),
            "neutrality_assumed": True, "physical_predictions": {"masses": None, "mixing": None}}


def demonstration_report():
    """Frozen strong/weak illustrations and structural m=0 controls, JSON safe."""
    controls = [dict(zip(("L", "C", "g", "m"), values)) for values in FROZEN_CONTROLS]
    return {"model_id": MODEL_ID, "frozen_controls": controls,
            "controls_frozen_before_evaluation": True, "empirical_calibration": False,
            "caps": {"fixed_number_dimension": MAX_DENSE_DIM, "unit_filling_sites": [3, 6]},
            "cases": [case_report(**parameters) for parameters in controls],
            "physical_predictions": {"masses": None, "mixing": None},
            "limitations": list(LIMITATIONS)}
