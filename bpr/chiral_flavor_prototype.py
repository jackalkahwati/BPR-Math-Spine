"""Conditional four-dimensional chiral-flavor EFT, not a flavor prediction.

The family-space S² and its monopole sections are assumed. They do not establish
an interacting six-dimensional compactification or a Spin(10) Yukawa coupling.
Our conventional 4D fields are Q_L=(3,2,1/6), u_R=(3,1,2/3),
d_R=(3,1,-1/3), L_L=(1,2,-1/2), e_R=(1,1,-1), and H=(1,2,1/2).
The allowed contractions are -bar(Q_L)^a_A (Y_u) tilde(H)_a u_R^A,
-bar(Q_L)^a_A (Y_d) H_a d_R^A, and -bar(L_L)^a (Y_e) H_a e_R,
plus Hermitian conjugates; tilde(H)=i sigma_2 H*, a is a weak index,
A is a color index, and family indices contract through Y. Optional neutral
nu_R permits -bar(L_L)^a (Y_nu) tilde(H)_a nu_R. These are 4D gauge singlets.

The scalar solver is bounded to the nine real l<=2 harmonics. A source is an
external input, not spontaneous symmetry breaking. The action is strictly
convex, but its value need not be positive. Those nine inputs still span every
Hermitian 3x3 overlap matrix; this model does not solve source selection.
No physical vev, mass unit, empirical flavor data, or optimizer is supplied.
"""
from fractions import Fraction
from math import lgamma
from numbers import Integral, Real

import numpy as np

from bpr.flavor_foundations import sphere_line_zero_modes, su_bundle_twisted_index

MODEL_ID = "conditional-4d-sm-s2-forced-flavor-v1"
HARMONIC_NAMES = ("1", "x", "y", "z", "xy", "yz", "P2(z)", "xz", "x^2-y^2")
HARMONIC_DEGREES = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
# Y_A = HARMONIC_NORMALIZATIONS[A] * the named unit-sphere polynomial.
HARMONIC_NORMALIZATIONS = np.sqrt(np.array([1, 3, 3, 3, 15, 15, 5, 15, 15 / 4]) / (4 * np.pi))


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _real(value, name, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value


def _real_array(value, name):
    a = np.asarray(value)
    if np.iscomplexobj(a) and np.any(a.imag != 0):
        raise ValueError(f"{name} must be real")
    a = np.asarray(a.real, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _coefficients(value, name="coefficients"):
    a = _real_array(value, name)
    if a.shape != (9,):
        raise ValueError(f"{name} must have shape (9,)")
    return a


def _angles(theta, phi):
    theta, phi = np.broadcast_arrays(_real_array(theta, "theta"), _real_array(phi, "phi"))
    if np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("theta must lie in [0, pi]")
    return theta, phi


def sphere_quadrature(R=1.0, n_polar=24, n_azimuth=48):
    """Return flattened (theta, phi, dA weights), Gauss-Legendre x periodic.

    The polar rule is in z=cos(theta); weights sum to 4*pi*R**2.
    Increase both orders to check arbitrary callable integrands for convergence.
    """
    R = _real(R, "R", positive=True)
    n_polar = _integer(n_polar, "n_polar", 1)
    n_azimuth = _integer(n_azimuth, "n_azimuth", 1)
    z, wz = np.polynomial.legendre.leggauss(n_polar)
    theta, phi = np.meshgrid(np.arccos(z), 2 * np.pi * np.arange(n_azimuth) / n_azimuth, indexing="ij")
    weights = np.broadcast_to(wz[:, None] * (2 * np.pi / n_azimuth) * R**2, theta.shape)
    return theta.ravel(), phi.ravel(), weights.ravel()


def _xyz(theta, phi):
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)


def real_harmonics(theta, phi):
    """Real orthonormal dOmega basis in HARMONIC_NAMES order, final axis 9.

    Norm is integral dOmega Y_A Y_B=delta_AB, not area normalization.
    Y_P2=sqrt(5/(4*pi))*(3*z*z-1)/2. No SciPy harmonic API is used.
    """
    theta, phi = _angles(theta, phi)
    x, y, z = _xyz(theta, phi)
    polynomials = np.stack((np.ones_like(z), x, y, z, x*y, y*z,
                            (3*z*z-1)/2, x*z, x*x-y*y), axis=-1)
    return polynomials * HARMONIC_NORMALIZATIONS


def monopole_profiles(theta, phi, q=3, R=1.0, patch_phase=0.0):
    """Normalized positive-flux zero modes, shape (..., q), north patch.

    n=q-1; psi_k=sqrt(q*binom(n,k)/(4*pi*R²))
    cos(theta/2)**(n-k) sin(theta/2)**k exp(i*k*phi), k=0,...,n.
    Thus k=0 is north-localized. These are local section components, with the
    common spin/gauge frame phase suppressed. A common real patch_phase
    multiplies every component by exp(i*patch_phase), cancelling in overlaps.
    Integral dA psi_i* psi_j=delta_ij. Negative/zero q counts are available
    separately; this profile API deliberately rejects them.
    """
    q = _integer(q, "q", 1)
    R = _real(R, "R", positive=True)
    theta, phi = _angles(theta, phi)
    phase = np.broadcast_to(_real_array(patch_phase, "patch_phase"), theta.shape)
    n = q - 1
    # Log amplitudes avoid overflow in binomial coefficients for general q.
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    result = []
    with np.errstate(divide="ignore"):
        for k in range(q):
            log_amp = 0.5 * (np.log(q / (4 * np.pi)) + lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) - np.log(R)
            if n-k:
                log_amp = log_amp + (n-k) * np.log(c)
            if k:
                log_amp = log_amp + k * np.log(s)
            result.append(np.exp(log_amp) * np.exp(1j * (k * phi + phase)))
    return np.stack(result, axis=-1)


def signed_mode_bookkeeping(q, gauge_rank=3, parent="weyl", parent_chirality=1):
    """Conditional bookkeeping only, not a 6D completion of this 4D EFT.

    Choose chi_6=chi_4*chi_internal and label chi_4=+ as left. A positive
    parent Weyl chirality then maps positive internal modes to 4D left modes.
    A 6D Dirac parent contains both parent Weyl chiralities and is vectorlike
    in 4D. q counts family copies; rank*q counts gauge components, not families.
    The SU(rank) index is topological and alone excludes no paired modes;
    the explicit sphere line-twist counts below assume the stated factorization.
    """
    modes = sphere_line_zero_modes(q)
    component_index = su_bundle_twisted_index(gauge_rank, q)
    chirality = _integer(parent_chirality, "parent_chirality")
    if chirality not in (-1, 1):
        raise ValueError("parent_chirality must be +1 or -1")
    if parent not in ("weyl", "dirac"):
        raise ValueError("parent must be 'weyl' or 'dirac'")
    left, right = modes.positive, modes.negative
    if parent == "dirac":
        left = right = modes.positive + modes.negative
    elif chirality == -1:
        left, right = right, left
    return {"flux": int(q), "internal_positive": modes.positive,
            "internal_negative": modes.negative, "internal_index": modes.index,
            "su_twisted_component_index": component_index, "parent": parent,
            "parent_chirality": chirality if parent == "weyl" else None,
            "left_family_copies": left, "right_family_copies": right,
            "net_left_families": left-right,
            "net_left_gauge_components": int(gauge_rank)*(left-right),
            "status": "conditional counting, not a six-dimensional EFT construction"}


def _field_values(field, theta, phi):
    if callable(field):
        values = field(*_xyz(theta, phi))
    elif np.ndim(field) == 0:
        values = field
    else:
        values = real_harmonics(theta, phi) @ _coefficients(field, "field coefficients")
    return np.broadcast_to(_real_array(values, "scalar field"), theta.shape)


def source_coefficients(source, n_polar=24, n_azimuth=48):
    """Project a real callable source J(x,y,z) onto l<=2 using dOmega.

    Higher harmonics are discarded by this bounded solver, not solved for.
    Scalar constants are also accepted. Coordinates are on the unit sphere.
    """
    theta, phi, weights = sphere_quadrature(n_polar=n_polar, n_azimuth=n_azimuth)
    return real_harmonics(theta, phi).T @ (weights * _field_values(source, theta, phi))


def overlap_matrix(field, q=3, R=1.0, n_polar=None, n_azimuth=None, patch_phase=0.0):
    """Compute P(field)_ij=integral dA psi_i* field psi_j by quadrature.

    field is a real constant, nine harmonic coefficients, or callable (x,y,z).
    Callable higher harmonics are integrated directly, not truncated first.
    patch_phase is a real constant or callable (x,y,z), shared by all sections.
    """
    q = _integer(q, "q", 1)
    theta, phi, weights = sphere_quadrature(
        R, max(24, q+2) if n_polar is None else n_polar,
        max(48, 2*q+8) if n_azimuth is None else n_azimuth)
    phase = patch_phase(*_xyz(theta, phi)) if callable(patch_phase) else patch_phase
    psi = monopole_profiles(theta, phi, q, R, phase)
    values = _field_values(field, theta, phi)
    return psi.conj().T @ ((weights * values)[:, None] * psi)


def _scalar_parameters(R, kappa, mu2, h0):
    return (_real(R, "R", positive=True), _real(kappa, "kappa", nonnegative=True),
            _real(mu2, "mu2", positive=True), _real(h0, "h0"))


def _response_eigenvalues(R, kappa, mu2):
    ell = HARMONIC_DEGREES
    return mu2 + kappa * ell * (ell+1) / R**2


def scalar_minimizer(source, R=1.0, kappa=1.0, mu2=1.0, h0=2.0):
    """Unique l<=2 minimizer: h_A=h0*sqrt(4*pi)*delta_A0+J_A/D_A.

    D_A=mu2+kappa*l*(l+1)/R². Inputs and output are dOmega-normalized
    coefficients. No source fitting, positivity restriction on h, or SSB.
    """
    R, kappa, mu2, h0 = _scalar_parameters(R, kappa, mu2, h0)
    result = _coefficients(source, "source") / _response_eigenvalues(R, kappa, mu2)
    result[0] += h0 * np.sqrt(4 * np.pi)
    return result


def scalar_action(h, source, R=1.0, kappa=1.0, mu2=1.0, h0=2.0):
    """Exact bounded S=integral dA[kappa|grad h|²/2+mu2(h-h0)²/2-Jh].

    Harmonics are dOmega normalized: gradient term has no R² prefactor,
    while potential and source terms do. Strict convexity does not imply S>=0.
    """
    R, kappa, mu2, h0 = _scalar_parameters(R, kappa, mu2, h0)
    h, source = _coefficients(h, "h"), _coefficients(source, "source")
    delta = h.copy()
    delta[0] -= h0 * np.sqrt(4 * np.pi)
    ell = HARMONIC_DEGREES
    return float(kappa/2 * np.dot(ell*(ell+1)*h, h)
                 + R**2 * (mu2/2 * np.dot(delta, delta) - np.dot(source, h)))


def scalar_stationarity(h, source, R=1.0, kappa=1.0, mu2=1.0, h0=2.0):
    """Gradient dS/dh_A (includes area R²), zero at the unique minimizer."""
    R, kappa, mu2, h0 = _scalar_parameters(R, kappa, mu2, h0)
    h, source = _coefficients(h, "h"), _coefficients(source, "source")
    gradient = _response_eigenvalues(R, kappa, mu2) * h - source
    gradient[0] -= mu2 * h0 * np.sqrt(4*np.pi)
    return R**2 * gradient


def scalar_hessian(R=1.0, kappa=1.0, mu2=1.0):
    """Positive-definite coefficient Hessian diag(R²*mu2+kappa*l*(l+1))."""
    R, kappa, mu2, _ = _scalar_parameters(R, kappa, mu2, 0)
    return np.diag(R**2 * _response_eigenvalues(R, kappa, mu2))


def yukawa_matrix(field, y_eff=1.0, **overlap_options):
    """Dimensionless Y=y_eff*P(h); y_eff is an explicit real EFT input."""
    return _real(y_eff, "y_eff") * overlap_matrix(field, **overlap_options)


def _matrix(value, name):
    a = np.asarray(value, dtype=complex)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return a


def mass_matrix(Y, v):
    """M=v*Y/sqrt(2); positive v and its mass units must be supplied by caller."""
    return (_real(v, "v", positive=True) / np.sqrt(2)) * _matrix(Y, "Y")


def svd_sector(Y):
    """Ascending singular values and paired rotations; Y=left diag(s) right_adjoint.

    left diagonalizes YY†, not the signed Hermitian Y. Vector phases and bases
    within degenerate subspaces are intentionally not assigned physical meaning.
    """
    Y = _matrix(Y, "Y")
    left, singular, right_adjoint = np.linalg.svd(Y)
    order = np.argsort(singular)
    left, singular, right_adjoint = left[:, order], singular[order], right_adjoint[order, :]
    return {"matrix": Y, "singular_values": singular, "left": left,
            "right_adjoint": right_adjoint,
            "reconstruction_error": float(np.linalg.norm(Y - (left * singular) @ right_adjoint))}


def mixing_observables(Y_up, Y_down, degeneracy_rtol=1e-10, degeneracy_atol=1e-12):
    """Nondegenerate V=U_uL†U_dL and J=Im(V00 V11 V01* V10*).

    Return None observables and an explicit reason if either singular spectrum
    is degenerate at the declared tolerance, including opposite signed masses.
    V phases are convention dependent; |V| and this signed quartet are invariant
    under quark rephasings. Ordering is ascending singular value in each sector.
    """
    rtol = _real(degeneracy_rtol, "degeneracy_rtol", nonnegative=True)
    atol = _real(degeneracy_atol, "degeneracy_atol", nonnegative=True)
    up, down = svd_sector(Y_up), svd_sector(Y_down)
    degenerate = []
    for name, sector in (("up", up), ("down", down)):
        s = sector["singular_values"]
        if np.any(np.diff(s) <= atol + rtol * np.maximum(s[1:], s[:-1])):
            degenerate.append(name)
    reason = "degenerate singular spectrum in " + ", ".join(degenerate) if degenerate else None
    V = None if reason else up["left"].conj().T @ down["left"]
    return {"up": up, "down": down, "mixing": V,
            "abs_mixing": None if V is None else np.abs(V),
            "J": None if V is None else float(np.imag(V[0, 0]*V[1, 1]*V[0, 1].conj()*V[1, 0].conj())),
            "mixing_undefined_reason": reason,
            "degeneracy_rtol": rtol, "degeneracy_atol": atol,
            "left_squared_commutator_norm": float(np.linalg.norm(
                (up["matrix"] @ up["matrix"].conj().T) @ (down["matrix"] @ down["matrix"].conj().T)
                - (down["matrix"] @ down["matrix"].conj().T) @ (up["matrix"] @ up["matrix"].conj().T)))}


def sm_anomalies(families=1, include_neutral_neutrino=False):
    """Exact 4D SM anomalies in an all-left-handed Weyl convention.

    Right-handed physical fields are replaced by u^c,d^c,e^c,(nu^c), reversing
    hypercharge and color chirality. T(fund)=1/2, SU(3) cubic A(fund)=+1,
    A(antifund)=-1. The SU(2) local cubic anomaly vanishes identically; its
    global Witten condition counts doublets including color. Mixed nonabelian
    cross anomalies vanish by tracelessness. Cancellation does not select N_f.
    """
    families = _integer(families, "families", 0)
    if not isinstance(include_neutral_neutrino, (bool, np.bool_)):
        raise TypeError("include_neutral_neutrino must be boolean")
    # name, color dimension, weak dimension, hypercharge, color cubic sign
    fields = [("Q", 3, 2, Fraction(1, 6), 1), ("u^c", 3, 1, Fraction(-2, 3), -1),
              ("d^c", 3, 1, Fraction(1, 3), -1), ("L", 1, 2, Fraction(-1, 2), 0),
              ("e^c", 1, 1, Fraction(1), 0)]
    if include_neutral_neutrino:
        fields.append(("nu^c", 1, 1, Fraction(0), 0))
    half = Fraction(1, 2)
    contributions = {}
    for name, color, weak, charge, color_sign in fields:
        contributions[name] = {"SU3_cubed": Fraction(weak*color_sign),
            "SU3_squared_U1": weak*half*charge if color == 3 else Fraction(0),
            "SU2_squared_U1": color*half*charge if weak == 2 else Fraction(0),
            "U1_cubed": color*weak*charge**3, "gravity_squared_U1": color*weak*charge}
    per_family = {key: sum((c[key] for c in contributions.values()), Fraction(0))
                  for key in next(iter(contributions.values()))}
    doublets = sum(color for _, color, weak, _, _ in fields if weak == 2)
    return {"convention": "all-left-handed 4D Weyl fields; T(fundamental)=1/2",
            "families": families, "field_contributions_per_family": contributions,
            "per_family": per_family,
            "totals": {key: families*value for key, value in per_family.items()},
            "weak_doublets_per_family": doublets, "witten_mod2": families*doublets % 2,
            "family_count_selected": False,
            "higher_dimensional_status": "six-dimensional and extra-U(1) anomalies unresolved"}


def assumption_ledger():
    """Separate mathematical results from model inputs and unresolved physics."""
    return {"scope": "conventional four-dimensional SM EFT, not literal 6D or Spin(10) Yukawa",
            "assumed": ["SM chiral representations and Higgs (1,2,1/2)",
                        "family-space sphere with positive q=3 line flux and identical left/right family profiles",
                        "fixed radius, scalar parameters, effective couplings and external sources",
                        "real scalar l<=2 truncation and dimensionless h, J in declared toy units"],
            "established_conditionally": ["normalized monopole overlaps and signed index arithmetic",
                "strictly convex sourced action has a unique harmonic minimizer; action need not be positive",
                "SVD masses and identifiable mixing only for nondegenerate singular spectra",
                "exact 4D SM anomaly cancellation per family"],
            "not_solved": ["flux/family-count selection", "radius stabilization", "substrate-derived sources",
                "six-dimensional anomalies and extra-U(1) completion", "UV Yukawa contractions",
                "physical vev, units, coupling matching and measured flavor spectrum"],
            "source_freedom": "nine real l<=2 sources still span any Hermitian 3x3 matrix after invertible scalar response",
            "selection_status": "forced linear response, not spontaneous symmetry breaking or predictive flavor selection",
            "data_policy": "fixed toy sources before evaluation; no empirical targets, fitting, optimization or MC"}


def fixed_toy_sources():
    """Frozen analytic coefficients, not inferred from resulting singular values.

    Ju=z+(3*z²-1)/4=z+P2(z)/2; Jd=x+z/3+x*y/2.
    Divide polynomial coefficients by the declared harmonic normalizations.
    """
    up, down = np.zeros(9), np.zeros(9)
    up[3], up[6] = 1/HARMONIC_NORMALIZATIONS[3], 0.5/HARMONIC_NORMALIZATIONS[6]
    down[1], down[3], down[4] = (1/HARMONIC_NORMALIZATIONS[1],
                                1/(3*HARMONIC_NORMALIZATIONS[3]),
                                0.5/HARMONIC_NORMALIZATIONS[4])
    return up, down


def _case(source_up, source_down):
    hu, hd = scalar_minimizer(source_up), scalar_minimizer(source_down)
    result = mixing_observables(yukawa_matrix(hu), yukawa_matrix(hd))
    result.update({"source_coefficients_up": source_up, "source_coefficients_down": source_down,
                   "profile_coefficients_up": hu, "profile_coefficients_down": hd,
                   "action_up": scalar_action(hu, source_up), "action_down": scalar_action(hd, source_down),
                   "stationarity_norm_up": float(np.linalg.norm(scalar_stationarity(hu, source_up))),
                   "stationarity_norm_down": float(np.linalg.norm(scalar_stationarity(hd, source_down)))})
    return result


def demonstration():
    """Reproducible, stdout-serializer-friendly report; no physical v by default."""
    up, down = fixed_toy_sources()  # Freeze all sources before computing any cases.
    zero = np.zeros(9)
    axial_up, axial_down = zero.copy(), zero.copy()
    axial_up[3], axial_down[3] = up[3], up[3]/2
    dihedral_source = zero.copy()
    dihedral_source[6] = up[6]
    cyclic = _case(axial_up, axial_down)
    cyclic.update({"orders": [5, 8, 9, 12],
                   "symmetry": "shared axial profiles are invariant under each listed cyclic rotation",
                   "conclusion": "nondegenerate sectors: no mixing beyond phases and permutations",
                   "proper_dihedral_control": _case(dihedral_source, dihedral_source),
                   "dihedral_symmetry": "adding the proper pi rotation about x removes the z term and forces outer-entry degeneracy"})
    return {"model_id": MODEL_ID,
            "inputs": {"q": 3, "R": 1.0, "kappa": 1.0, "mu2": 1.0, "h0": 2.0,
                "y_eff_up": 1.0, "y_eff_down": 1.0, "v": None,
                "output_units": "dimensionless effective Yukawa couplings, not physical masses",
                "source_up": "z+(3*z*z-1)/4", "source_down": "x+z/3+x*y/2",
                "harmonic_names": HARMONIC_NAMES, "harmonic_normalizations": HARMONIC_NORMALIZATIONS.copy(),
                "coefficient_normalization": "Y_A=N_A*named polynomial; integral dOmega Y_A Y_B=delta_AB",
                "source_coefficients_up": up, "source_coefficients_down": down},
            "bookkeeping": signed_mode_bookkeeping(3), "anomalies": sm_anomalies(3),
            "unforced": _case(zero, zero), "cyclic": cyclic, "forced": _case(up, down),
            "assumption_ledger": assumption_ledger()}
