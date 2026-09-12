"""Exact Fourier restriction of the existing defocusing ring, not flavor matching.

Modes are ordered k=(-1,0,1), U[x,k]=exp(2*pi*i*x*k/p)/sqrt(p).
The free window is invariant but split (singlet plus doublet). Its local real
scalar source image has real dimension five, not the monopole target's nine.
Its generated *complex associative* algebra is nevertheless all M_3(C). The quartic
Galerkin restriction is generally not invariant under the full DNLS flow.

All energies retain the old ring's unshifted convention. No statistics change,
physical masses, flux, chiral Yukawa source, or families are derived here.
"""
from numbers import Integral, Real

import numpy as np

from bpr.quantum_flavor_sources import overlap_operators

MODEL_ID = "defocusing-ring-triplet-projection-v1"
MODES = (-1, 0, 1)
_EPS = np.finfo(float).eps
_WIDE = np.longdouble
_CWIDE = np.clongdouble


def _p(p):
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
        raise TypeError("p must be an integer")
    if p < 5:
        raise ValueError("p must be at least 5 (primality is not required)")
    try:
        value = float(p)
    except OverflowError as exc:
        raise ValueError("numerically unresolved ring size") from exc
    if not np.isfinite(value):
        raise ValueError("numerically unresolved ring size")
    return int(p)


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"nonfinite {name}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _checked(value, name):
    """Reject overflow and loss of a nonzero component to floating-point zero."""
    value = np.asarray(value)
    dtype = complex if np.iscomplexobj(value) else float
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = value.astype(dtype)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"numerically unresolved {name}: nonfinite result")
    parts = ((value.real, out.real), (value.imag, out.imag)) if dtype is complex else ((value, out),)
    if any(np.any((before != 0) & (after == 0)) for before, after in parts):
        raise ValueError(f"numerically unresolved {name}: underflow")
    return out.item() if out.ndim == 0 else out


def _amplitudes(amplitudes):
    z = np.asarray(amplitudes, dtype=_CWIDE)
    if z.shape != (3,):
        raise ValueError("amplitudes must have shape (3,) in k=(-1,0,1) order")
    if not np.all(np.isfinite(z)):
        raise ValueError("amplitudes must be finite")
    _checked(z, "amplitudes")
    scale = np.max(np.abs(z))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        n = _checked(np.sum(z.real**2 + z.imag**2), "norm")
    if scale > 0 and n == 0:
        raise ValueError("numerically unresolved positive norm: underflow")
    return z, n


def _positive(value, name):
    value = _checked(value, name)
    if value <= 0:
        raise ValueError(f"numerically unresolved positive {name}: underflow")
    return value


def _cubic_product(a, b, c):
    with np.errstate(under="ignore"):
        product = a*b*c
    if a != 0 and b != 0 and c != 0 and product == 0:
        raise ValueError("numerically unresolved cubic monomial: underflow")
    return product


def _normalized(z, scale):
    with np.errstate(under="ignore"):
        normalized = z/scale if scale else z
    if np.any((z != 0) & (normalized == 0)):
        raise ValueError("numerically unresolved normalized amplitude: underflow")
    return normalized


def _cubic_scale(g, p, scale):
    if g == 0 or scale == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = np.exp(np.log(_WIDE(g))-np.log(_WIDE(p))+3*np.log(scale))
    return _positive(value, "cubic scale")


def fourier_isometry(p):
    """p-by-3 Fourier isometry; no dense p-by-p projector is formed."""
    p = _p(p)
    return np.exp(2j*np.pi*(np.arange(p)/p)[:, None]*np.array(MODES))/np.sqrt(p)


def project(vector):
    """Apply UU† to a finite ring vector in O(p) storage."""
    vector = np.asarray(vector, dtype=complex)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("vector must be a finite one-dimensional ring field")
    u = fourier_isometry(len(vector))
    return _checked(u @ (u.conj().T @ vector), "projected field")


def restricted_symmetries(p):
    """T psi[x]=psi[x+1], R psi[x]=psi[-x], hence RTR=T†."""
    p = _p(p)
    return {"translation": np.diag(np.exp(2j*np.pi*np.array(MODES)/p)),
            "reflection": np.eye(3)[::-1].copy()}


def free_window(p, C=1.0):
    """Stable shifted dispersion and isolation; no subtractive gap evaluation.

    Unshifted energies can coalesce numerically even while the positive shifted
    scales remain resolved. Report this, rather than claim exact degeneracy.
    The limiting isolation/bandwidth ratio is three, not infinity.
    """
    p, C = _p(p), _real(C, "C", positive=True)
    theta = _WIDE(np.pi)/_WIDE(p)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = _checked(_WIDE(C)*(4*np.sin(theta)**2), "positive bandwidth")
        gap = _checked(_WIDE(C)*(4*np.sin(3*theta)*np.sin(theta)), "positive isolation gap")
    if w <= 0 or gap <= 0:
        raise ValueError("numerically unresolved positive free scales")
    with np.errstate(over="ignore"):
        baseline = _checked(-2*_WIDE(C), "unshifted energy")
    return {"p": p, "C": C, "modes": list(MODES),
            "shifted_energies": [w, 0.0, w],
            "unshifted_energies": [baseline+w, baseline, baseline+w],
            "bandwidth": w, "isolation_gap": gap,
            "gap_to_bandwidth": float(1+2*np.cos(2*theta)),
            "unshifted_split_resolved": bool(w > 32*_EPS*abs(baseline)),
            "free_invariant": True, "exactly_degenerate_triplet": False,
            "asymptotic_gap_to_bandwidth": 3.0}


def source_generators(p):
    """Five independent real local fields: 1, cos(x), sin(x), cos(2x), sin(2x)."""
    p = _p(p)
    theta = 2*np.pi*np.arange(p)/p
    return np.array([np.ones(p), np.cos(theta), np.sin(theta),
                     np.cos(2*theta), np.sin(2*theta)])


def compress_local_source(source):
    """U† diag(source) U for a REAL local scalar; constant-diagonal Toeplitz."""
    source = np.asarray(source)
    if np.iscomplexobj(source):
        raise TypeError("local scalar source must be real")
    source = np.asarray(source, dtype=float)
    if source.ndim != 1 or not np.all(np.isfinite(source)):
        raise ValueError("source must be a finite one-dimensional real field")
    u = fourier_isometry(len(source))
    return _checked(u.conj().T @ (source[:, None]*u), "source compression")


def local_source_closure_residual(source):
    """Frobenius norm of (I-UU†) diag(f) U, computed without a dense projector.

    Only constant local f preserves this band (p>=5). Compression by itself
    does not establish invariance. Subtractive near-zero output is arithmetic,
    not a proof of exact closure; the constant control is structural.
    """
    compress_local_source(source)  # validate the real local field
    source = np.asarray(source, dtype=float)
    u = fourier_isometry(len(source))
    # The constant part preserves the band exactly. Remove it BEFORE taking
    # a small difference; a large constant must not obscure angular leakage.
    centered = _checked(source-source[0], "centered source")
    lifted = centered[:, None]*u
    residual = lifted-u @ (u.conj().T @ lifted)
    raw = _checked(np.linalg.norm(residual), "local source closure residual")
    scale = _checked(64*_EPS*np.linalg.norm(lifted), "source closure resolution scale")
    structural = bool(np.all(source == source[0]))
    return {"raw_norm": raw, "arithmetic_warning_scale": scale,
            "resolved_norm": 0.0 if structural else (raw if raw > scale else None),
            "status": "structural_zero" if structural else ("resolved_nonzero" if raw > scale else "unresolved_cancellation")}


def real_gram_rank(matrices):
    """Real Hilbert-Schmidt Gram rank, appropriate for Hermitian source images."""
    matrices = np.asarray(matrices, dtype=complex)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not np.all(np.isfinite(matrices)):
        raise ValueError("matrices must be finite with shape (n,3,3)")
    flat = matrices.reshape(len(matrices), 9)
    scale = np.max(abs(flat)) if flat.size else 0.0
    if scale == 0:
        return 0
    flat = flat/scale
    gram = (flat.conj() @ flat.T).real
    return int(np.linalg.matrix_rank(gram))


def generated_matrix_units(p):
    """Construct all nine E_ij as words in the compressed source generators.

    S=P(cos x)-i P(sin x) is the upper shift, I-SS†=E_22;
    E_ij=S**(2-i) E_22 (S†)**(2-j). This concerns the COMPLEX associative
    algebra, not the real linear source map (products are new operators).
    """
    matrices = [compress_local_source(s) for s in source_generators(p)]
    shift = matrices[1]-1j*matrices[2]
    corner = matrices[0]-shift @ shift.conj().T
    return np.array([np.linalg.matrix_power(shift, 2-i) @ corner
                     @ np.linalg.matrix_power(shift.conj().T, 2-j)
                     for i in range(3) for j in range(3)])


def source_matching_diagnostic(p=7):
    """Target overlap operators are geometry only, not substrate statistics."""
    p = _p(p)
    sources = np.array([compress_local_source(s) for s in source_generators(p)])
    target = overlap_operators()
    units = generated_matrix_units(p)
    witness = np.diag([1., 0., -1.])
    return {"direct_source_real_rank": real_gram_rank(sources),
            "target_source_real_rank": real_gram_rank(target),
            "generated_algebra_complex_rank": int(np.linalg.matrix_rank(units.reshape(9, 9))),
            "direct_matching": False,
            "image": "constant-diagonal Hermitian Toeplitz",
            "witness": witness.tolist(),
            "witness_hilbert_schmidt_distance": float(np.linalg.norm(witness)),
            "scope": "direct real local scalar source compression only; not a universal no-go"}


def projected_energy_terms(amplitudes, p, C=1.0, g=0.5):
    """Separated energy terms; a rounded total need not resolve its corrections."""
    p, g = _p(p), _real(g, "g", nonnegative=True)
    window = free_window(p, C)
    z, n = _amplitudes(amplitudes)
    scale = np.max(abs(z))
    a, b, c = _normalized(z, scale)
    shape = (abs(a)**2+abs(b)**2+abs(c)**2)**2 + 2*abs(b*a.conjugate()+c*b.conjugate())**2 + 2*abs(c*a.conjugate())**2
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        baseline = _checked(-2*_WIDE(window["C"])*n, "energy baseline")
        shifted = _checked(_WIDE(window["bandwidth"])*(abs(z[0])**2+abs(z[2])**2), "shifted energy")
        quartic = 0.0 if g == 0 or scale == 0 else _positive(
            np.exp(np.log(_WIDE(g))-np.log(2*_WIDE(p))+4*np.log(scale)+np.log(shape)), "quartic energy")
        total = _checked(baseline+shifted+quartic, "projected energy")
    if n > 0 and baseline == 0 or (z[0] != 0 or z[2] != 0) and shifted == 0:
        raise ValueError("numerically unresolved positive energy term")
    warning = 32*_EPS*(abs(baseline)+shifted+quartic)
    return {"baseline": baseline, "shifted_kinetic": shifted, "quartic": quartic,
            "total": total,
            "corrections_resolved_in_total": bool((shifted == 0 or shifted > warning)
                                                   and (quartic == 0 or quartic > warning)),
            "total_cancellation_warning": bool(warning > 0 and abs(total) <= warning)}


def projected_energy(amplitudes, p, C=1.0, g=0.5):
    """Exact unshifted H(Uz); see projected_energy_terms for rounding metadata."""
    return projected_energy_terms(amplitudes, p, C, g)["total"]


def projected_gradient(amplitudes, p, C=1.0, g=0.5):
    """dH/dz* by exact cubic momentum conservation; i dz/dt = this gradient."""
    p, g = _p(p), _real(g, "g", nonnegative=True)
    window = free_window(p, C)
    z, _ = _amplitudes(amplitudes)
    scale = np.max(abs(z))
    normalized = _normalized(z, scale) if g > 0 else np.zeros(3, dtype=_CWIDE)
    cubic = np.zeros(3, dtype=_CWIDE)
    for i, ki in enumerate(MODES):
        for j, kj in enumerate(MODES):
            for ell, kl in enumerate(MODES):
                k = ki-kj+kl
                if k in MODES:  # no in-band aliases for p>=5
                    cubic[k+1] += _cubic_product(normalized[i], normalized[j].conjugate(), normalized[ell])
    kinetic = -2*_WIDE(window["C"])+np.array(window["shifted_energies"], dtype=_WIDE)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return _checked(kinetic*z+_cubic_scale(g, p, scale)*cubic, "projected gradient")


def _offband(z, p, g):
    scale = np.max(abs(z))
    factor = _cubic_scale(g, p, scale)
    a, b, c = _normalized(z, scale) if g > 0 else np.zeros(3, dtype=_CWIDE)
    terms = {-3: [_cubic_product(a, a, c.conjugate())],
             -2: [_cubic_product(a, a, b.conjugate()), 2*_cubic_product(b, a, c.conjugate())],
             2: [_cubic_product(c, c, b.conjugate()), 2*_cubic_product(b, c, a.conjugate())],
             3: [_cubic_product(c, c, a.conjugate())]}
    coefficients, scales = {}, {}
    for k, summands in terms.items():
        residue = k % p
        coefficients[residue] = coefficients.get(residue, _CWIDE(0)) + sum(summands)
        scales[residue] = scales.get(residue, _WIDE(0)) + sum(abs(s) for s in summands)
    return ({k: factor*v for k, v in coefficients.items()},
            {k: factor*v for k, v in scales.items()})


def offband_coefficients(amplitudes, p, g=0.5):
    """Discarded normalized Fourier coefficients of g|Uz|²Uz, aliases summed.

    p=5 pairs (-3,2) and (-2,3); p=6 pairs (-3,3). For p>=7 all four
    residues are distinct. Small/cancelled coefficients alone do not establish
    exact closure: see leakage_diagnostic for arithmetic resolution metadata.
    """
    p, g = _p(p), _real(g, "g", nonnegative=True)
    z, _ = _amplitudes(amplitudes)
    coeffs, _ = _offband(z, p, g)
    return {k: _checked(v, "off-band coefficient") for k, v in coeffs.items()}


def leakage_diagnostic(amplitudes, p, g=0.5):
    """Analytic leakage norm with explicit cancellation/underflow honesty.

    A 64-epsilon sum-of-term-magnitudes scale is an arithmetic warning, NOT a
    physical threshold. Only vacuum/free/pure-mode controls get structural_zero.
    Near cancellation yields null resolved_norm, never a false exact-closure
    assertion. raw_norm is merely the evaluated floating-point diagnostic.
    """
    p, g = _p(p), _real(g, "g", nonnegative=True)
    z, n = _amplitudes(amplitudes)
    coeffs, scales = _offband(z, p, g)
    raw = _checked(np.hypot.reduce([abs(v) for v in coeffs.values()]), "leakage norm")
    arithmetic = _checked(64*_EPS*np.hypot.reduce(list(scales.values())), "leakage resolution scale")
    structural = g == 0 or np.count_nonzero(z) <= 1
    if not structural and any(v > 0 for v in scales.values()) and arithmetic == 0:
        raise ValueError("numerically unresolved leakage arithmetic scale")
    status = "structural_zero" if structural else ("resolved_nonzero" if raw > arithmetic else "unresolved_cancellation")
    return {"raw_norm": raw, "resolved_norm": None if status == "unresolved_cancellation" else raw,
            "status": status, "arithmetic_warning_scale": arithmetic,
            "residues": sorted(coeffs), "aliasing": p in (5, 6),
            "uniform_residual_bound": _checked(3*_cubic_scale(g, p, np.sqrt(_WIDE(n))), "residual bound"),
            "exact_invariant_truncation": False}


def interaction_gap_diagnostic(N, p, C=1.0, g=0.5):
    """gN/(p Delta) is a diagnostic only, not an all-time error theorem."""
    N, p, g = _real(N, "N", nonnegative=True), _p(p), _real(g, "g", nonnegative=True)
    gap = free_window(p, C)["isolation_gap"]
    if g == 0 or N == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.exp(np.log(_WIDE(g))+np.log(_WIDE(N))-np.log(_WIDE(p))-np.log(_WIDE(gap)))
    return _positive(result, "interaction/gap diagnostic")


def finite_time_error_bound(N, p, g, t):
    """Conservative classical DNLS versus lifted Galerkin finite-time bound.

    With identical in-band initial data, both conserve N. The cubic Lipschitz
    constant is <=3gN on the norm ball, while ||residual||<=3gN**(3/2)/p.
    Duhamel/Gronwall gives relative error <=min(2,expm1(3gN|t|)/p).
    Cap before exponentiating using log1p(2p), so even enormous finite inputs
    cannot overflow an exponential. Vacuum has zero absolute, undefined relative
    error. This is not quantum, many-body, or infinite-time control.
    """
    N, p = _real(N, "N", nonnegative=True), _p(p)
    g, t = _real(g, "g", nonnegative=True), _real(t, "t")
    if N == 0:
        return {"relative_bound": None, "absolute_bound": 0.0, "capped": False,
                "status": "vacuum_exact", "time": t}
    if g == 0 or t == 0:
        return {"relative_bound": 0.0, "absolute_bound": 0.0, "capped": False,
                "status": "exact_zero", "time": t}
    log_exponent = np.log(_WIDE(3))+np.log(_WIDE(g))+np.log(_WIDE(N))+np.log(abs(_WIDE(t)))
    threshold = np.log(_WIDE(p))+np.log(2+1/_WIDE(p))
    capped = bool(log_exponent >= np.log(threshold))
    with np.errstate(under="ignore"):
        exponent = np.exp(log_exponent) if not capped else 0.0
        # expm1(A)/p may be finite even when expm1(A) overflows. For A>1
        # use exp(A-log p)*(1-exp(-A)); retain expm1 precision near zero.
        relative = (_WIDE(2) if capped else
                    (np.expm1(exponent)/_WIDE(p) if exponent <= 1 else
                     np.exp(exponent-np.log(_WIDE(p)))*(-np.expm1(-exponent))))
    if relative <= 0:
        raise ValueError("numerically unresolved positive finite-time bound")
    return {"relative_bound": _checked(relative, "finite-time relative bound"),
            "absolute_bound": _checked(relative*np.sqrt(_WIDE(N)), "finite-time absolute bound"),
            "capped": capped, "status": "vacuous_norm_cap" if capped else "conservative_bound",
            "time": t}


def demonstration():
    """Frozen p=7,C=1,g=.5,N=1 illustration; JSON safe, no fit or scan."""
    p, C, g = 7, 1.0, 0.5
    cases = [("single_mode", np.array([0., 1., 0.], complex)),
             ("two_mode", np.array([1., 1j, 0.])/np.sqrt(2)),
             ("complex_multimode", np.array([1., 1j, (1+1j)/np.sqrt(2)])/np.sqrt(3))]
    examples = []
    for name, z in cases:
        examples.append({"name": name, "amplitudes_re_im": [[float(v.real), float(v.imag)] for v in z],
                         "norm": _amplitudes(z)[1], "energy": projected_energy(z, p, C, g),
                         "energy_terms": projected_energy_terms(z, p, C, g),
                         "leakage": leakage_diagnostic(z, p, g)})
    return {"model_id": MODEL_ID, "parameters": {"p": p, "C": C, "g": g, "N": 1.0},
            "free_window": free_window(p, C), "source_matching": source_matching_diagnostic(p),
            "interaction_gap_diagnostic": interaction_gap_diagnostic(1.0, p, C, g),
            "examples": examples,
            "finite_time_bounds": [finite_time_error_bound(1.0, p, g, t) for t in (0.0, 0.1, 1.0, 2.0)],
            "physical_masses": None, "physical_mixing": None,
            "limitations": ["Fixed norm is inherited, not dynamically selected; fixed density is a different scaling limit.",
                            "Fourier projection preserves bosonic statistics; global U(1) charge is not site momentum.",
                            "Local density is not automatically a Lorentz-scalar chiral Yukawa source.",
                            "No flux, physical families, masses or mixing are derived.",
                            "Rank obstruction concerns the direct local linear map, not its full generated algebra or dressed/nonlocal sources.",
                            "Finite-time bound is conservative and may be vacuous; no quantum or all-time effective-theory proof."]}
