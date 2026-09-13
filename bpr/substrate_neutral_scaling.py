"""Joint strong-coupling/large-ring neutral characteristic-function bounds.

The full fixed-number Bose model H=gD+CT and N=L=q are stipulated. The
frozen module-2 inequalities are independently reviewed conditional
real-arithmetic theorems, not floating-point enclosures. Scalar paths allocate no
Fock space and impose no site cap; a separate dense oracle accepts L=3..6.
"""
from fractions import Fraction
import math

import numpy as np

from bpr.substrate_fermionization import _guard, _integer, _normal, _real
from bpr.substrate_neutral_continuum import (
    _angle, _fraction_float, _geometry, _lattice_angle, _parameters, _product,
    validity_audit, weak_limit_bound,
)

MODEL_ID = "conditional-substrate-neutral-scaling-v1"
THEOREM_STATUS = "independently reviewed conditional theorem"
MAX_EXACT_SITES = 6
FROZEN_LENGTHS = (16, 64, 256)
FROZEN_TIMES = (0.0, 0.5, 1.0)
LIMITATIONS = (
    "The complete quantum Bose Hamiltonian and N=L=q neutrality are stipulated.",
    "The reviewed full-model inequalities are conditional analytical results, not a fixed-coupling thermodynamic theorem.",
    "Analytical bounds exclude roundoff; numerical oracle residuals are diagnostics only.",
    "The joint limit requires increasing g/C; fixed coupling or g proportional to L is not covered by these sufficient estimates.",
    "The normalized limit has no atom, but this does not exclude a finite-g bound particle at fixed coupling.",
    "No individual full-model spectral weights or uniform unbounded moments are controlled.",
    "Absolute source weight vanishes in the sufficient sequences even though the normalized probability has a weak limit.",
    "Spatial generator convergence does not establish a conditioned near-edge spectral limit.",
    "A quadratic threshold shift is not a pole dispersion or a derivation of Lorentz covariance.",
    "No physical limiting speed, Planck spacing, masses or empirical calibration is supplied.",
)


def _positive(value, name):
    value = _normal(value, name)
    if value <= 0:
        raise ValueError("numerically unresolved nonzero " + name)
    return value


@_guard
def comparison_bound(L, m, C=1.0, g=40.0, time=1.0, k_target=None):
    """Bound normalized full/compression and full/semicircle characteristic functions.

    Full coordinates are s=(E_n-E_0-g)/C, compression coordinates (E-g)/C,
    with characteristic sign exp(+it s). Return None for unavailable bounds,
    never a normalized zero measure. Failure of either sufficient gate says
    nothing about whether a particle exists. At valid t=0 the bounds are zero.

    Exact Fractions compare g>4CL and form dimensionless source quantities;
    no gD, C*L or lambda-squared float intermediate is needed. The source gate
    uses the evaluated transcendental source norm, not a roundoff enclosure.
    All exposed nonzero numeric quantities must be normal, finite binary64.
    """
    L, m, C, g = _parameters(L, m, C, g)
    time = _real(time, "time")
    k = _lattice_angle(L, m)
    target = k if k_target is None else _angle(k_target)
    audit = validity_audit(L, C, g)
    result = dict(L=L, m=m, C=C, g=g, k=k, k_target=target, time=time,
                  validity_audit=audit,
                  separation_sufficient=audit["excitation_separation_sufficient"],
                  source_normalization_sufficient=False, bound_available=False,
                  unavailable_reason=None, roundoff_included=False,
                  theorem_status=THEOREM_STATUS,
                  full_coordinate="s=(E_n-E_0-g)/C",
                  compression_coordinate="s=(E-g)/C",
                  characteristic_sign="exp(+it s)",
                  ground_virtual_remainder_scope="||(I-|Omega><Omega|)G-w|| <= eta",
                  numerical_evaluation="binary64 evaluation, not a certified roundoff enclosure")
    fields = ("lambda", "source_norm", "source_error_bound", "relative_source_error",
              "ground_virtual_remainder_bound", "ground_complement_ratio", "separation_ratio",
              "s1_norm_bound", "generator_bound", "ground_shift_bound", "source_comparison_term",
              "rotation_term", "time_term", "compression_weak_bound",
              "full_compression_bound_uncapped", "full_compression_bound",
              "full_limit_bound_uncapped", "full_limit_bound")
    result.update(dict.fromkeys(fields))
    if g == 0:
        result["unavailable_reason"] = "g=0: virtual-source comparison is undefined, including the zero mode."
        return result
    lamq = Fraction.from_float(C) / Fraction.from_float(g)
    lam = _fraction_float(lamq, "lambda=C/g")
    result["lambda"] = lam
    if m == 0:
        result.update(source_norm=0.0,
                      unavailable_reason="Zero density mode: the zero measure cannot be normalized.")
        return result
    half_sine = _positive(abs(math.sin(k / 2)), "source half-angle sine")
    a = _product((4.0, lam, half_sine), name="source norm")
    length = _fraction_float(Fraction(L), "site count")
    s1 = _product((math.pi, length), name="S1 norm bound")
    bh = _fraction_float(8 * Fraction.from_float(math.pi) * lamq * L * L,
                         "dimensionless generator bound")
    rotation = _fraction_float(2 * Fraction.from_float(math.pi) * lamq * L,
                               "source rotation term")
    result.update(source_norm=a, s1_norm_bound=s1, generator_bound=bh, rotation_term=rotation)
    if not result["separation_sufficient"]:
        result["unavailable_reason"] = "Sufficient excitation separation g>4CL fails; normalized-source comparison is unavailable."
        return result
    dq = 1 - 4 * lamq * L
    d = _fraction_float(dq, "dimensionless separation margin")
    root = math.sqrt(length)
    r = _product((2.0, lam, root), (d,), "ground complement ratio")
    # sqrt(L)*eta cancels both square roots analytically. Fraction evaluation
    # avoids gratuitous underflow of lambda**2 or overflow of L**2.
    deltaq = 4 * lamq**3 * L**2 / dq**2 + 8 * lamq**2 * L**2 / dq
    delta = _fraction_float(deltaq, "density source error bound")
    eta = _product((delta,), (root,), "ground virtual remainder bound")
    ratio = _fraction_float(deltaq / Fraction.from_float(a), "relative source error")
    b0 = _fraction_float(4 * lamq * L / dq, "ground shift bound")
    source_term = _product((4.0, ratio), name="normalized source comparison term")
    result.update(separation_ratio=d, ground_complement_ratio=r,
                  source_error_bound=delta, ground_virtual_remainder_bound=eta,
                  relative_source_error=ratio, ground_shift_bound=b0,
                  source_comparison_term=source_term)
    if not deltaq < Fraction.from_float(a):
        result["unavailable_reason"] = "Sufficient relative-source normalization gate delta_source<a fails."
        return result
    result.update(source_normalization_sufficient=True, bound_available=True)
    # Zero is structural only after the source has been proved nonzero.
    if time == 0:
        result.update(time_term=0.0, compression_weak_bound=0.0,
                      full_compression_bound_uncapped=0.0, full_compression_bound=0.0,
                      full_limit_bound_uncapped=0.0, full_limit_bound=0.0)
        return result
    time_term = _fraction_float(Fraction.from_float(abs(time)) *
                               (Fraction.from_float(bh) + Fraction.from_float(b0)),
                               "time comparison term")
    weak = _product((2.0, weak_limit_bound(L, m, target, lipschitz=abs(time))),
                    name="complex compression weak-limit bound")
    full_comp = _normal(math.fsum((source_term, rotation, time_term)), "full/compression bound")
    total = _normal(math.fsum((full_comp, weak)), "full/semicircle bound")
    result.update(time_term=time_term, compression_weak_bound=weak,
                  full_compression_bound_uncapped=full_comp,
                  full_compression_bound=min(2.0, full_comp),
                  full_limit_bound_uncapped=total, full_limit_bound=min(2.0, total))
    return result


@_guard
def spatial_scaling(L, ell=1.0, kappa=1.0):
    """Fixed-circumference m=1 audit: h=ell/L, C=kappa/h², lambda=L^-5.

    kappa is arbitrary, not a speed. The threshold is the continuum support
    edge g-2CA, not the finite compression minimum g-2CA*cos(pi/L).
    Their positive offset can persist when C grows as L²; the first bright
    density line may also differ because finite compression lines can be dark.
    The stable support-edge shift is 16*C*sin²(k/2)/(3+A); subtracting
    g-6C from an enormous g is avoided.
    For u=8sin²(k/2), A=sqrt(9-u)>=1,
      2(3-A)=u/3+u²/[3(3+A)²].
    |u-2k²|<=k⁴/6 and u<=2k² give remainder <=(5/36)C*k⁴
    relative to (2/3)C*k². This is an analytical remainder, not roundoff.
    The reported physical threshold residual is only a binary64 diagnostic.
    """
    L = _integer(L, "L", 3)
    ell, kappa = _real(ell, "ell", positive=True), _real(kappa, "kappa", positive=True)
    eq, kapq = Fraction.from_float(ell), Fraction.from_float(kappa)
    h = _fraction_float(eq / L, "lattice spacing")
    cq = kapq * L**2 / eq**2
    C = _fraction_float(cq, "spatial hopping scale")
    lam = _fraction_float(Fraction(1, L**5), "spatial lambda")
    # The actual accepted g,C are checked by comparison_bound below.
    g = _fraction_float(Fraction.from_float(C) * L**5, "spatial repulsion scale")
    p = _fraction_float(Fraction.from_float(2 * math.pi) / eq, "physical momentum")
    k = _lattice_angle(L, 1)
    shape, _ = _geometry(k)
    sine = _positive(abs(math.sin(k / 2)), "threshold half-angle sine")
    shift = _product((16.0, C, sine, sine), (3.0 + shape,), "threshold shift")
    quadratic = _product((2.0 / 3.0, kappa, p, p), name="quadratic threshold shift")
    remainder = _product((5.0 / 36.0, C, k, k, k, k), name="threshold remainder bound")
    edge_sine = _positive(abs(math.sin(k / 4)), "finite compression edge sine")
    edge_offset = _product((4.0, C, shape, edge_sine, edge_sine),
                           name="finite compression edge offset")
    response = comparison_bound(L, 1, C, g, k_target=0.0)
    return dict(L=L, m=1, ell=ell, kappa=kappa, h=h, physical_momentum=p,
                C=C, g=g, **{"lambda": lam}, k=k,
                threshold_shift=shift, quadratic_threshold_shift=quadratic,
                threshold_remainder_bound=remainder,
                threshold_scope="continuum support edge g-2CA, not the finite compression minimum or first bright line",
                finite_compression_edge_offset=edge_offset,
                finite_compression_edge_offset_formula="4 C A sin^2(pi/(2L)) above g-2CA",
                finite_compression_edge_scope="the offset need not vanish under C proportional to L^2; first bright density line may differ",
                threshold_residual=_normal(shift - quadratic, "threshold residual"),
                physical_generator_bound=_product((C, response["generator_bound"]),
                                                   name="physical generator bound"),
                physical_ground_shift_bound=_product((C, response["ground_shift_bound"]),
                                                      name="physical ground shift bound"),
                response_bound=response, roundoff_included=False,
                theorem_status=THEOREM_STATUS,
                scaling="fixed ell,m=1; C=kappa L^2/ell^2; lambda=L^-5",
                near_edge_conditioned_measure_convergence_proved=False,
                pole_dispersion_derived=False, limiting_speed=None)


def _complex_record(value):
    value = complex(_normal(value, "characteristic function"))
    return {"real": value.real, "imag": value.imag}


@_guard
def exact_case_report(L, m, C=1.0, g=40.0, time=1.0, k_target=None):
    """Bounded complete-space numerical oracle, never used by scalar bounds.

    The first-order S1=T_ab/(D_a-D_b) is computed directly without S2.
    No probability clipping or renormalization is used. Floating-point
    cancellation and phase screens are heuristic; no numerical certificate
    follows from observed agreement with the analytical bounds.
    """
    L = _integer(L, "L", 3, MAX_EXACT_SITES)  # before model allocation
    bound = comparison_bound(L, m, C, g, time, k_target)
    result = {"comparison_bound": bound, "dimension": math.comb(2 * L - 1, L),
              "numerical_diagnostics": None, "unavailable_reason": None,
              "roundoff_included": False}
    if bound["g"] == 0 or bound["m"] == 0:
        result["unavailable_reason"] = bound["unavailable_reason"]
        return result
    from bpr.substrate_fermionization import _EPS, _norm, _scale
    from bpr.substrate_neutral_response import density_diagonal, neutral_model

    adapter = neutral_model(L, bound["C"], bound["g"])
    model, lam = adapter.model, bound["lambda"]
    # Diagonalize in units of g; form centered bandwidth nodes afterwards.
    h_scaled = np.diag(model.D) + _scale(model.V_unit, lam, "oracle hopping")
    # Inherited neutral-report heuristic: the scaled spectral arithmetic floor
    # must be at least 1e6 below the requested hopping/bandwidth scale. This
    # also screens cancellation when subtracting the D=1 energy before /lambda.
    # It is a resolution policy, NOT an eigensolver roundoff enclosure.
    spectral_scale = _positive(float(np.max(np.sum(np.abs(h_scaled), axis=1))),
                               "oracle spectral scale")
    energy_proxy = _product((64.0, _EPS, float(len(model.basis)), spectral_scale),
                            name="oracle scaled energy resolution proxy")
    if energy_proxy >= _product((1e-6, lam), name="oracle bandwidth resolution threshold"):
        raise ValueError("numerically unresolved oracle centered bandwidth (heuristic screen)")
    energies, vectors = np.linalg.eigh(h_scaled)
    ground_gap = _positive(float(energies[1] - energies[0]), "oracle ground gap")
    if ground_gap <= 2 * energy_proxy:
        raise ValueError("numerically unresolved oracle ground isolation (heuristic screen)")
    # The small ground shift is a distinct reported coordinate. Require its
    # absolute magnitude to resolve the proxy by 1e3, without forcing a sign
    # or comparing a numerical residual with the analytical approximation.
    if energy_proxy >= _product((1e-3, abs(float(energies[0]))),
                                name="oracle ground-energy resolution threshold"):
        raise ValueError("numerically unresolved oracle ground energy (heuristic screen)")
    source_proxy = _product((energy_proxy, math.sqrt(L)), (ground_gap,),
                            "oracle ground-source resolution proxy")
    if source_proxy >= _product((1e-6, bound["source_norm"]),
                                name="oracle source resolution threshold"):
        raise ValueError("numerically unresolved oracle tiny density source (heuristic screen)")
    ground = vectors[:, 0].copy()
    if ground[adapter.omega_index] < 0:
        ground *= -1
    rho = density_diagonal(model, bound["m"])
    x = _normal(rho * ground, "oracle ground source")
    y = _normal(rho * _scale(-model.V_unit[:, adapter.omega_index], lam,
                            "oracle virtual ground"), "oracle leading source")
    nx, ny = _positive(_norm(x), "oracle ground source norm"), _positive(_norm(y), "oracle leading source norm")
    if source_proxy >= _product((1e-6, nx), name="oracle measured source resolution threshold"):
        raise ValueError("numerically unresolved oracle measured density source (heuristic screen)")
    xn, yn = x / nx, y / ny
    difference = model.D[:, None] - model.D[None, :]
    equal = difference == 0
    td = np.where(equal, model.V_unit, 0.0)
    s1 = np.divide(model.V_unit, difference, out=np.zeros_like(td), where=~equal)
    q, v = np.linalg.eigh(1j * s1)
    phases = _scale(q, -lam, "oracle unitary phases")
    if np.max(np.abs(phases)) > 1 / (128 * _EPS):
        raise ValueError("numerically unresolved oracle unitary phases (heuristic screen)")
    u = (v * np.exp(1j * phases)) @ v.conj().T
    hbd_scaled = np.diag(model.D) + lam * td
    transformed_difference = _normal((u @ h_scaled @ u.conj().T - hbd_scaled) / lam,
                                     "oracle transformed generator difference")
    generator_error = float(np.max(np.abs(np.linalg.eigvalsh(transformed_difference))))
    e0 = float(energies[0] / lam)
    nodes = _normal((energies - energies[0] - 1.0) / lam, "oracle centered full nodes")
    p1 = adapter.p1_indices
    comp_nodes, comp_vectors = np.linalg.eigh(model.V_unit[np.ix_(p1, p1)])
    full_prob = _normal(np.abs(vectors.conj().T @ xn)**2, "oracle full probabilities")
    comp_prob = _normal(np.abs(comp_vectors.conj().T @ yn[p1])**2, "oracle compression probabilities")
    if bound["time"] == 0:
        chi_full = chi_comp = complex(1.0)
    else:
        full_phase = _scale(nodes, bound["time"], "oracle full characteristic phases")
        comp_phase = _scale(comp_nodes, bound["time"], "oracle compression characteristic phases")
        if max(float(np.max(np.abs(full_phase))), float(np.max(np.abs(comp_phase)))) > 1 / (128 * _EPS):
            raise ValueError("numerically unresolved characteristic phases (heuristic screen)")
        chi_full = complex(np.dot(full_prob, np.exp(1j * full_phase)))
        chi_comp = complex(np.dot(comp_prob, np.exp(1j * comp_phase)))
    result["numerical_diagnostics"] = dict(
        ground_energy_over_C=e0, ground_source_norm=nx, leading_source_norm=ny,
        source_error_norm=_norm(x - y),
        normalized_transformed_source_distance=_norm(u @ xn - yn),
        s1_norm=float(np.max(np.abs(q))), generator_error_over_C=generator_error,
        chi_full=_complex_record(chi_full), chi_compression=_complex_record(chi_comp),
        full_compression_error=abs(chi_full - chi_comp),
        full_mass_residual=float(np.sum(full_prob) - 1),
        compression_mass_residual=float(np.sum(comp_prob) - 1),
        scaled_energy_resolution_proxy=energy_proxy,
        ground_source_resolution_proxy=source_proxy,
        centered_bandwidth_resolution_proxy=energy_proxy / lam,
        arithmetic_resolution_policy="heuristic proxy margins: 1e6 for source and bandwidth, 1e3 for ground-energy magnitude; not certified roundoff",
        roundoff_certified=False)
    return result


def demonstration_report():
    """Frozen controls and sufficient sequences; no files, fitting or Fock work."""
    cases = []
    for label, mode, exponent, target in (
        ("fixed nonzero k=pi/2, lambda=L^-3", lambda L: L // 4, 3, math.pi / 2),
        ("fixed m=1, lambda=L^-4", lambda L: 1, 4, 0.0),
    ):
        for L in FROZEN_LENGTHS:
            for time in FROZEN_TIMES:
                case = comparison_bound(L, mode(L), g=float(L**exponent), time=time, k_target=target)
                case["limit_sequence"] = label
                cases.append(case)
    controls = ((1, 40.0, "frozen original reference"),
                (1, 0.7, "failed separation control"),
                (0, 40.0, "exact zero density mode"),
                (1, 0.0, "undefined g=0 virtual source"),
                (0, 0.0, "undefined g=0 zero mode"))
    for m, g, label in controls:
        for time in FROZEN_TIMES:
            case = comparison_bound(5, m, g=g, time=time)
            case["limit_sequence"] = label
            cases.append(case)
    return {"model_id": MODEL_ID, "theorem_status": THEOREM_STATUS,
            "cases": cases, "spatial_scaling": [spatial_scaling(L) for L in FROZEN_LENGTHS],
            "limitations": list(LIMITATIONS), "roundoff_included": False,
            "full_model_limit": "normalized characteristic functions in centered bandwidth units, uniformly on compact time intervals along sufficient joint sequences",
            "physical_predictions": {"masses": None, "mixing": None, "limiting_speed": None,
                                     "lorentz_covariance": None, "planck_spacing": None}}
