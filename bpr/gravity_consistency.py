"""Action-consistent gravity formulas and bounded identifiability diagnostics.

Frozen source: doc/derivations/gravity_consistency_2026-09-12.md.  The natural
unit action is M**2 R/2 + alpha R**2/2, with M the REDUCED Planck scale.
These conditional conversions do not derive gravity from a finite Bose ring.
Nonzero inputs/intermediates/outputs must be normal finite binary64 numbers.
Numerical residuals are diagnostics, not rigorous roundoff certificates.
"""
from collections.abc import Sequence
from functools import wraps
from math import fsum
from numbers import Integral, Number, Real

import numpy as np

from .constants import C, HBAR

MODEL_ID = "action-consistent-gravity-v1"
MAX_DENSE_DIM = 32
MAX_TIMES = 32
_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps
_EIGHT_PI = 8.0 * np.pi
_INDUCED_DENOMINATOR = 48.0 * np.pi**2
_HBAR_C = HBAR * C
_NEWTON_ENERGY_FACTOR = HBAR * C**5 / _EIGHT_PI
LIMITATIONS = (
    "Exact formulas are conditional on the supplied Jordan action and positive reduced scale, not a derivation of dynamical spacetime.",
    "Scalar amplitude, epsilon, n_s and r use the leading large-e-fold slow-roll approximation; inverse amplitude matching is calibration, not prediction.",
    "The induced coefficient requires supplied field content, regulator, subtraction assumptions and cutoff; no absolute scale is predicted.",
    "A constant ZG propagator amplitude cannot identify Z and G separately, and an effective coefficient cannot identify bare, counterterm and cutoff contributions separately.",
    "SI Planck scales here are energies in joules, not masses in kilograms; entropy is dimensionless S/k_B for the Einstein-Hilbert contribution or Einstein-frame area, not the full curved-horizon Jordan R+R squared Wald expression.",
    "The small-alpha slow-roll controls are algebraic checks and can yield As above unity; they are not viable perturbative cosmologies.",
    "Flat-space spectra and conjugation evolution are bounded dense numerical checks, not production simulation or roundoff certificates.",
    "Only a global identity shift preserves the ordering across number sectors; a number-dependent shift need not do so.",
    "A finite-ring offset supplies no covariant vacuum density, volume, subtraction prescription or independent gravitational coupling; no cosmological constant is predicted.",
)


def _guard(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ValueError("numerically unresolved arithmetic or eigensystem") from exc
    return checked


def _normal(value, name):
    a = np.asarray(value)
    if not np.all(np.isfinite(a)):
        raise ValueError("numerically unresolved {}: nonfinite".format(name))
    # Check real/imaginary components separately: abs(complex) can hide a
    # subnormal component or itself overflow for finite components.
    components = (a.real, a.imag) if np.iscomplexobj(a) else (a,)
    for part in components:
        if np.any((np.abs(part) > 0) & (np.abs(part) < _TINY)):
            raise ValueError("numerically unresolved {}: subnormal".format(name))
    return a.item() if a.ndim == 0 else a


def _real(value, name, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("{} must be real".format(name))
    result = _normal(float(value), name)
    if result == 0 and value != 0:
        raise ValueError("numerically unresolved {}: underflow".format(name))
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError("{} outside supported range".format(name))
    return result


def _multiplicity(p):
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral):
        raise TypeError("p must be an integer multiplicity")
    result = _real(p, "p", positive=True)
    if int(result) != int(p):
        raise ValueError("numerically unresolved integer multiplicity")
    return int(p)


def _mul(a, b, name):
    value = _normal(np.multiply(a, b), name)
    if a != 0 and b != 0 and value == 0:
        raise ValueError("numerically unresolved {}: underflow".format(name))
    return value


def _div(a, b, name):
    value = _normal(np.divide(float(a), float(b)), name)
    if a != 0 and value == 0:
        raise ValueError("numerically unresolved {}: underflow".format(name))
    return value


def _add(a, b, name):
    return _normal(np.add(a, b), name)


def _square(a, name):
    return _mul(a, a, name)


def _action_parameters(M, alpha):
    return _real(M, "M", positive=True), _real(alpha, "alpha", positive=True)


@_guard
def plateau_energy_density(M=1.0, alpha=1.0):
    """Exact V0=M**4/(8 alpha), in the fourth power of M's energy unit."""
    M, alpha = _action_parameters(M, alpha)
    return _div(_square(_square(M, "M squared"), "M fourth power"),
                _mul(8.0, alpha, "8 alpha"), "plateau")


@_guard
def einstein_potential(phi, M=1.0, alpha=1.0):
    """Exact Einstein-frame V, canonical phi in the same natural unit as M.

    F=exp(sqrt(2/3)*phi/M)>0; expm1 avoids loss near phi=0.  This
    evaluates V directly, without claiming a dynamical inflation solution.
    """
    M, alpha = _action_parameters(M, alpha)
    phi = _real(phi, "phi")
    exponent = _mul(-np.sqrt(2.0 / 3.0), _div(phi, M, "phi/M"), "exponent")
    shape = _normal(-np.expm1(exponent), "potential shape")
    if phi != 0 and shape == 0:
        raise ValueError("numerically unresolved nonzero potential shape")
    return _mul(plateau_energy_density(M, alpha), _square(shape, "shape squared"), "potential")


@_guard
def scalaron_mass_squared(M=1.0, alpha=1.0):
    """Exact vacuum curvature V''(0)=M**2/(6 alpha)."""
    M, alpha = _action_parameters(M, alpha)
    return _div(_square(M, "M squared"), _mul(6.0, alpha, "6 alpha"), "mass squared")


@_guard
def scalaron_mass(M=1.0, alpha=1.0):
    """Positive mass sqrt(V''(0)), in M's natural energy unit."""
    return _normal(np.sqrt(scalaron_mass_squared(M, alpha)), "mass")


@_guard
def leading_scalar_amplitude(Ne, alpha):
    """Leading large-Ne As=Ne**2/(144 pi**2 alpha), NOT an exact solution."""
    Ne = _real(Ne, "Ne", positive=True)
    alpha = _real(alpha, "alpha", positive=True)
    return _div(_square(Ne, "Ne squared"),
                _mul(144.0 * np.pi**2, alpha, "amplitude denominator"), "leading As")


@_guard
def alpha_from_scalar_amplitude(Ne, As):
    """Synthetic or empirical inverse calibration, not a predicted alpha."""
    Ne = _real(Ne, "Ne", positive=True)
    As = _real(As, "As", positive=True)
    return _div(_square(Ne, "Ne squared"),
                _mul(144.0 * np.pi**2, As, "inverse denominator"), "calibrated alpha")


@_guard
def leading_slow_roll(Ne, alpha):
    """Leading epsilon, As, n_s and r; positive Ne need not imply validity."""
    Ne = _real(Ne, "Ne", positive=True)
    alpha = _real(alpha, "alpha", positive=True)
    n2 = _square(Ne, "Ne squared")
    return {"Ne": Ne, "alpha": alpha, "As": leading_scalar_amplitude(Ne, alpha),
            "epsilon": _div(0.75, n2, "epsilon"),
            "n_s": _add(1.0, -_div(2.0, Ne, "2/Ne"), "n_s"),
            "r": _div(12.0, n2, "r"),
            "status": "leading large-Ne slow roll, not exact"}


@_guard
def reduced_to_unreduced_energy(reduced_energy):
    """M_unreduced=sqrt(8 pi)*M_reduced; input/output use the same energy unit."""
    return _mul(_real(reduced_energy, "reduced energy", positive=True),
                np.sqrt(_EIGHT_PI), "unreduced energy")


@_guard
def unreduced_to_reduced_energy(unreduced_energy):
    """Inverse energy conversion; neither argument denotes a kilogram mass."""
    return _div(_real(unreduced_energy, "unreduced energy", positive=True),
                np.sqrt(_EIGHT_PI), "reduced energy")


@_guard
def natural_newton_constant(M):
    """Natural-unit G=1/(8 pi M**2), in inverse energy squared (hbar=c=1)."""
    M = _real(M, "M", positive=True)
    return _div(1.0, _mul(_EIGHT_PI, _square(M, "M squared"), "8 pi M squared"), "natural G")


@_guard
def newton_constant_from_reduced_energy(reduced_energy_j):
    """SI G [m**3 kg**-1 s**-2] from REDUCED energy [J], hbar*c**5/(8pi M_E**2)."""
    energy = _real(reduced_energy_j, "reduced energy [J]", positive=True)
    return _div(_NEWTON_ENERGY_FACTOR, _square(energy, "energy squared [J2]"), "SI G")


@_guard
def reduced_energy_from_newton_constant(G_si):
    """REDUCED Planck energy [J] from positive SI G, not inverse cutoff prediction."""
    G_si = _real(G_si, "G [m3 kg-1 s-2]", positive=True)
    return _normal(np.sqrt(_div(_NEWTON_ENERGY_FACTOR, G_si, "reduced energy squared")), "reduced energy [J]")


@_guard
def induced_reduced_coefficient(p, cutoff):
    """Conditional p*cutoff**2/(48 pi**2), natural-unit reduced M**2.

    p is a supplied positive integer multiplicity, not a selected field count.
    cutoff and M must use the same energy unit; b=c=0 is not inferred.
    """
    p = _multiplicity(p)
    cutoff = _real(cutoff, "cutoff", positive=True)
    return _mul(_div(p, _INDUCED_DENOMINATOR, "induced k"),
                _square(cutoff, "cutoff squared"), "induced coefficient")


@_guard
def induced_spacing_planck_ratio(p):
    """a/lP=sqrt(p/(6 pi)), conditional on the induced term alone (b=c=0)."""
    p = _multiplicity(p)
    return _normal(np.sqrt(_div(p, 6.0 * np.pi, "spacing ratio squared")), "a/lP")


@_guard
def cutoff_spacing_si(cutoff_energy_j):
    """a=hbar*c/cutoff [m], given a supplied positive cutoff energy in joules."""
    return _div(_HBAR_C, _real(cutoff_energy_j, "cutoff energy [J]", positive=True), "spacing [m]")


@_guard
def wald_entropy_natural(area, M):
    """Einstein-Hilbert / Einstein-frame S/k_B=2*pi*area*M**2.

    Area uses inverse square of M's natural energy unit. For the full Jordan
    R+R**2 action the entropy is instead 2*pi*integral(M**2+2*alpha*R)dA_J;
    dA_E=F*dA_J, so the frame and area convention matter.
    """
    area = _real(area, "natural area", nonnegative=True)
    M = _real(M, "M", positive=True)
    return _mul(2.0 * np.pi, _mul(area, _square(M, "M squared"), "area M squared"), "natural entropy")


@_guard
def wald_entropy_si(area_m2, reduced_energy_j):
    """Einstein-Hilbert / Einstein-frame S/k_B=2*pi*A*M_E**2/(hbar*c)**2.

    A[m2] is Einstein-frame area, reduced M_E[J] is an energy. This is not
    the full Jordan curved-horizon R+R**2 entropy at fixed Jordan area.
    Equals A/(4*lP**2) for lP**2=hbar*G/c**3 and the G derived from M_E.
    This does not alter any pre-existing fixed physical Planck-length anchor.
    """
    area = _real(area_m2, "area [m2]", nonnegative=True)
    energy = _real(reduced_energy_j, "reduced energy [J]", positive=True)
    numerator = _mul(2.0 * np.pi, _mul(area, _square(energy, "energy squared"), "area energy squared"), "entropy numerator")
    return _div(numerator, _square(_HBAR_C, "hbar c squared"), "SI entropy")


@_guard
def propagator_amplitude(Z, G, k):
    """Conditional A(k)=ZG/k**2 for positive Z,G and positive momentum magnitude k."""
    Z = _real(Z, "Z", positive=True)
    G = _real(G, "G", positive=True)
    k = _real(k, "k", positive=True)
    return _div(_mul(Z, G, "ZG"), _square(k, "k squared"), "propagator amplitude")


@_guard
def propagator_rescaling(Z, G, factor):
    """Return (factor*Z,G/factor); logarithmic null direction is (1,-1)."""
    Z = _real(Z, "Z", positive=True)
    G = _real(G, "G", positive=True)
    factor = _real(factor, "factor", positive=True)
    return (_mul(factor, Z, "rescaled Z"), _div(G, factor, "rescaled G"))


@_guard
def induced_effective_coefficient(b, c, p, cutoff):
    """Physical positive M_eff**2=b+c+p*cutoff**2/(48pi**2), natural units.

    b and c are independent real energy-squared coefficients, not assumed zero.
    Compensated summation retains the supplied rounded terms. It cannot repair
    uncertainty in the computed induced term: reject if the heuristic scale
    16*eps*sum(abs(terms)) exceeds 1e-10 times the positive physical result.
    This numerical conditioning policy is not a rigorous error certificate.
    """
    b, c = _real(b, "b"), _real(c, "c")
    induced = induced_reduced_coefficient(p, cutoff)
    result = _real(_normal(fsum((b, c, induced)), "effective coefficient"), "M_eff squared", positive=True)
    scale = _normal(fsum((abs(b), abs(c), induced)), "coefficient arithmetic scale")
    if _div(_mul(16.0 * _EPS, scale, "coefficient resolution estimate"), result, "relative coefficient resolution") > 1e-10:
        raise ValueError("numerically unresolved effective coefficient cancellation")
    return result


def _coefficient_invariance(original, transformed):
    # Compare to the physical result, never to the much larger bare terms.
    relative = _div(abs(_add(transformed, -original, "coefficient difference")), original, "relative coefficient difference")
    if relative > 256.0 * _EPS:
        raise ValueError("numerically unresolved coefficient invariance after transformation")


@_guard
def induced_bare_counterterm_shift(b, c, p, cutoff, shift):
    """Return (b+shift,c-shift), natural units, only if numerically resolved.

    Both coefficients pass the cancellation guard; the returned coefficient
    must agree within 256*eps relative to the original physical coefficient.
    """
    original = induced_effective_coefficient(b, c, p, cutoff)
    b, c, shift = _real(b, "b"), _real(c, "c"), _real(shift, "shift")
    new_b, new_c = _add(b, shift, "shifted b"), _add(c, -shift, "shifted c")
    transformed = induced_effective_coefficient(new_b, new_c, p, cutoff)
    _coefficient_invariance(original, transformed)
    return new_b, new_c


@_guard
def induced_cutoff_shift(b, c, p, cutoff, new_cutoff):
    """Return (new_b,c,new_cutoff), new_b=b+k*(cutoff**2-new_cutoff**2).

    Apply the same cancellation/invariance policy as the bare/counterterm shift.
    """
    original = induced_effective_coefficient(b, c, p, cutoff)
    b, c = _real(b, "b"), _real(c, "c")
    p = _multiplicity(p)
    cutoff = _real(cutoff, "cutoff", positive=True)
    new_cutoff = _real(new_cutoff, "new cutoff", positive=True)
    delta = _add(_square(cutoff, "cutoff squared"), -_square(new_cutoff, "new cutoff squared"), "cutoff square difference")
    new_b = _add(b, _mul(_div(p, _INDUCED_DENOMINATOR, "k"), delta, "bare compensation"), "new b")
    transformed = induced_effective_coefficient(new_b, c, p, new_cutoff)
    _coefficient_invariance(original, transformed)
    return new_b, c, new_cutoff


@_guard
def induced_identifiability_report(b, c, p, cutoff):
    """Jacobian and exact symbolic null directions in coordinates (b,c,cutoff).

    The tangent parameter on the second direction has energy units; its b
    component -2*k*cutoff therefore has energy units. Vectors are coordinate
    components, not a dimensionless orthonormal basis or an SVD certificate.
    """
    coefficient = induced_effective_coefficient(b, c, p, cutoff)
    p, cutoff = _multiplicity(p), _real(cutoff, "cutoff", positive=True)
    slope = _mul(2.0, _mul(_div(p, _INDUCED_DENOMINATOR, "k"), cutoff, "k cutoff"), "Jacobian slope")
    jacobian = np.array([1.0, 1.0, slope])
    nulls = np.array([[1.0, -1.0, 0.0], [-slope, 0.0, 1.0]])
    residuals = _normal(nulls @ jacobian, "Jacobian null residuals")
    return {"M_eff_squared": coefficient, "coordinates": ["b", "c", "cutoff"],
            "units": ["energy squared", "energy squared", "energy"],
            "jacobian": jacobian.tolist(), "null_directions": nulls.tolist(),
            "numerical_null_residuals": residuals.tolist(),
            "status": "nonidentifiable without independent bare/counterterm/cutoff assumptions"}


@_guard
def cosmological_constant_shift(rho_v, M_eff_squared):
    """Conditional natural-unit delta Lambda=rho_v/M_eff**2 [energy squared].

    rho_v is a supplied covariant energy density [energy**4], NOT a finite H
    offset. The independent coupling -sqrt(-g)*rho_v gives T=-rho_v*g.
    """
    return _div(_real(rho_v, "covariant vacuum density"),
                _real(M_eff_squared, "M_eff squared", positive=True), "cosmological constant shift")


def _matrix(value, name):
    # Shape/size checks precede dtype conversion, allocation or eigensolvers.
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError("{} must be a matrix".format(name))
        shape = value.shape
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not 1 <= len(value) <= MAX_DENSE_DIM:
            raise ValueError("{} dimension outside cap32".format(name))
        widths = []
        for row in value:
            if isinstance(row, np.ndarray):
                valid = row.ndim == 1
            else:
                valid = isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            if not valid or len(row) > MAX_DENSE_DIM:
                raise ValueError("{} requires bounded one-dimensional rows".format(name))
            widths.append(len(row))
        if len(set(widths)) != 1:
            raise ValueError("{} has ragged rows".format(name))
        shape = (len(value), widths[0])
    else:
        raise TypeError("{} must be a NumPy array or bounded nested sequence".format(name))
    if not 1 <= shape[0] <= MAX_DENSE_DIM or shape[0] != shape[1]:
        raise ValueError("{} must be nonempty square with dimension at most32".format(name))
    for row in value:
        for entry in row:
            if isinstance(entry, (bool, np.bool_)) or not isinstance(entry, Number):
                raise TypeError("{} entries must be numerical scalars".format(name))
            # Validate each original component before binary64 conversion so a
            # wider scalar cannot silently lose a small imaginary component.
            real = _real(entry.real, name + " real component")
            imag = _real(entry.imag, name + " imaginary component")
            z = complex(real, imag)
            _normal(z, name)
            if z == 0 and entry != 0:
                raise ValueError("numerically unresolved matrix conversion")
    a = _normal(np.asarray(value, dtype=complex), name)
    scale = float(np.max(np.abs(a)))
    if scale and _norm((a / scale) - (a / scale).conj().T) > 64.0 * _EPS:
        raise ValueError("{} must be Hermitian to binary64 tolerance".format(name))
    # Do not silently symmetrize or feed a non-Hermitian matrix to eigh.
    if not np.array_equal(a, a.conj().T):
        raise ValueError("{} must have explicitly Hermitian entries".format(name))
    return a


def _norm(a):
    a = _normal(a, "norm input")
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    if scale == 0:
        return 0.0
    scaled = _normal(a / scale, "scaled norm input")
    return _mul(scale, float(np.linalg.norm(scaled, ord=2)), "spectral norm")


def _times(values):
    if isinstance(values, np.ndarray):
        valid = values.ndim == 1
    else:
        valid = isinstance(values, Sequence) and not isinstance(values, (str, bytes))
    if not valid:
        raise TypeError("times must be a bounded one-dimensional sequence")
    if not 1 <= len(values) <= MAX_TIMES:
        raise ValueError("times must contain between1 and32 values")
    return tuple(_real(t, "time") for t in values)


def _unitary(eigenvalues, eigenvectors, time):
    phase = _normal(np.multiply(-time, eigenvalues), "dynamical phases")
    if time != 0 and np.any((eigenvalues != 0) & (phase == 0)):
        raise ValueError("numerically unresolved dynamical phases")
    # Large arguments cannot resolve the input phase to even one radian.
    if np.any(np.abs(phase) > 1.0 / _EPS):
        raise ValueError("numerically unresolved dynamical phase argument")
    return _normal((eigenvectors * np.exp(1j * phase)) @ eigenvectors.conj().T, "unitary")


def _spectral_resolution(matrix, energies, gaps, name):
    # Independent centering is only a resolution diagnostic: reported spectra
    # and dynamics still come from the actual uncentered matrix eigensystem.
    centered = _normal(matrix - matrix[0, 0] * np.eye(len(matrix)), name + " centered matrix")
    centered_scale = _norm(centered)
    full_scale = _norm(matrix)
    resolution = _mul(64.0 * _EPS, full_scale, name + " spectral resolution")
    if centered_scale and centered_scale <= resolution:
        raise ValueError("numerically unresolved {} nonscalar spectrum".format(name))
    nonzero_gaps = np.abs(gaps[gaps != 0])
    if centered_scale and not nonzero_gaps.size:
        raise ValueError("numerically unresolved {} collapsed spectrum".format(name))
    if nonzero_gaps.size and float(np.min(nonzero_gaps)) <= resolution:
        raise ValueError("numerically unresolved {} spectral gaps".format(name))
    centered_energies = _normal(np.linalg.eigvalsh(centered), name + " centered energies")
    centered_gaps = _normal(centered_energies[:, None] - centered_energies[None, :], name + " centered gaps")
    if _norm(gaps - centered_gaps) > _mul(512.0 * _EPS, centered_scale, name + " centered gap tolerance"):
        raise ValueError("numerically unresolved {} uncentered gaps".format(name))
    return centered_scale


@_guard
def flat_identity_shift_report(H, source, offset, times=(0.0, 0.3, 1.0)):
    """Actual eigh(H) and eigh(H+offset*I) spectra and conjugation dynamics.

    H and the noncommuting diagnostic source must be explicitly Hermitian;
    a commuting source is allowed but gives a weaker check. Natural hbar=1,
    so time has inverse H-energy units. All matrices and time lists cap32.
    The global phase is checked at operator level, hence for every state.
    Both original and shifted spectra must resolve nonscalar parts and nonzero
    gaps above 64*eps times their full spectral norm. Centered eigensystems
    independently check their gaps within 512*eps times the centered norm.
    These conservative binary64 rejection heuristics are not error bounds.
    """
    H, source = _matrix(H, "H"), _matrix(source, "source")
    if source.shape != H.shape:
        raise ValueError("H and source must have the same dimension")
    offset, times = _real(offset, "offset"), _times(times)
    identity = np.eye(len(H), dtype=complex)
    shifted = _normal(H + offset * identity, "shifted H")
    # Reject an offset that erases H, or is itself wholly unresolvable.
    reconstruction = _norm(_normal(shifted - offset * identity - H, "shift reconstruction"))
    if reconstruction > _mul(128.0 * _EPS, _norm(H), "reconstruction tolerance"):
        raise ValueError("numerically unresolved H after identity shift")
    if offset != 0 and np.array_equal(shifted, H):
        raise ValueError("numerically unresolved identity offset")
    energies, vectors = np.linalg.eigh(H)
    shifted_energies, shifted_vectors = np.linalg.eigh(shifted)
    energies = _normal(energies, "energies")
    shifted_energies = _normal(shifted_energies, "shifted energies")
    vectors, shifted_vectors = _normal(vectors, "eigenvectors"), _normal(shifted_vectors, "shifted eigenvectors")
    gaps = _normal(energies[:, None] - energies[None, :], "all pairwise gaps")
    shifted_gaps = _normal(shifted_energies[:, None] - shifted_energies[None, :], "shifted pairwise gaps")
    original_scale = _spectral_resolution(H, energies, gaps, "original")
    shifted_scale = _spectral_resolution(shifted, shifted_energies, shifted_gaps, "shifted")
    gap_residual = _norm(shifted_gaps - gaps)
    if gap_residual > _mul(512.0 * _EPS, max(original_scale, shifted_scale), "gap comparison tolerance"):
        raise ValueError("numerically unresolved gaps after identity shift")
    dynamics = []
    for time in times:
        U = _unitary(energies, vectors, time)
        shifted_U = _unitary(shifted_energies, shifted_vectors, time)
        phase_arg = _mul(-time, offset, "global phase argument")
        if abs(phase_arg) > 1.0 / _EPS:
            raise ValueError("numerically unresolved global phase argument")
        phase = _normal(np.exp(1j * phase_arg), "global phase")
        evolved = _normal(U @ source @ U.conj().T, "source evolution")
        shifted_evolved = _normal(shifted_U @ source @ shifted_U.conj().T, "shifted source evolution")
        dynamics.append({"time": time,
                         "global_phase": [float(phase.real), float(phase.imag)],
                         "state_phase_operator_residual": _norm(shifted_U - phase * U),
                         "conjugation_residual": _norm(shifted_evolved - evolved),
                         "unitarity_residual": _norm(U @ U.conj().T - identity),
                         "shifted_unitarity_residual": _norm(shifted_U @ shifted_U.conj().T - identity)})
    return {"dimension": len(H), "offset": offset, "energies": energies.tolist(),
            "shifted_energies": shifted_energies.tolist(),
            "pairwise_gaps": gaps.tolist(), "shifted_pairwise_gaps": shifted_gaps.tolist(),
            "spectrum_translation_residual": _norm(shifted_energies - energies - offset),
            "pairwise_gap_residual": _norm(shifted_gaps - gaps),
            "source_commutator_norm": _norm(_normal(H @ source - source @ H, "commutator")),
            "dynamics": dynamics, "status": "bounded binary64 diagnostic, not gravitational inference"}


@_guard
def demonstration_report():
    """Frozen synthetic controls only; JSON-native report, no files or fitting."""
    actions = []
    for M in (1.0, 2.0):
        for alpha in (0.5, 2.0, 10.0):
            actions.append({"M": M, "alpha": alpha,
                            "mass_squared": scalaron_mass_squared(M, alpha),
                            "mass": scalaron_mass(M, alpha),
                            "plateau": plateau_energy_density(M, alpha),
                            "potential": [{"phi_over_M": x, "V": einstein_potential(x * M, M, alpha)}
                                          for x in (0.0, 0.1, 1.0, 5.0)]})
    calibrations = []
    for Ne in (40.0, 55.0, 70.0):
        for As in (1e-3, 1e-6):
            alpha = alpha_from_scalar_amplitude(Ne, As)
            calibrations.append({"Ne": Ne, "synthetic_As": As, "calibrated_alpha": alpha,
                                 "round_trip_residual": leading_scalar_amplitude(Ne, alpha) - As,
                                 "status": "synthetic inverse calibration, not prediction"})
    units = []
    for energy in (1.0, 2.0, 10.0):
        G_si = newton_constant_from_reduced_energy(energy)
        lP2 = _div(_mul(HBAR, G_si, "hbar G"), C**3, "Planck length squared")
        entropy = wald_entropy_si(1.0, energy)
        units.append({"reduced_energy_j": energy,
                      "unreduced_energy_j": reduced_to_unreduced_energy(energy), "G_si": G_si,
                      "energy_round_trip_residual": unreduced_to_reduced_energy(reduced_to_unreduced_energy(energy)) - energy,
                      "G_energy_round_trip_residual": reduced_energy_from_newton_constant(G_si) - energy,
                      "area_m2": 1.0, "wald_entropy_S_over_kB": entropy,
                      "area_lP_entropy_relative_residual": _div(entropy, _div(1.0, 4.0 * lP2, "area entropy"), "entropy ratio") - 1.0})
    induced_units = []
    for p in (5, 11):
        for cutoff in (1.0, 2.0, 10.0):
            reduced = _normal(np.sqrt(induced_reduced_coefficient(p, cutoff)), "induced energy")
            G_si = newton_constant_from_reduced_energy(reduced)
            lP = _normal(np.sqrt(_div(_mul(HBAR, G_si, "hbar G"), C**3, "lP squared")), "lP")
            ratio = induced_spacing_planck_ratio(p)
            induced_units.append({"p": p, "supplied_cutoff_energy_j": cutoff, "reduced_energy_j": reduced,
                                  "reduced_to_cutoff_ratio": _div(reduced, cutoff, "M/cutoff"),
                                  "spacing_planck_ratio": ratio,
                                  "spacing_ratio_residual": _div(cutoff_spacing_si(cutoff), lP, "a/lP") - ratio,
                                  "status": "conditional induced term only, assumes b=c=0"})
    propagator = []
    for factor in (0.25, 4.0):
        Z, G = propagator_rescaling(2.0, 3.0, factor)
        propagator.append({"Z": 2.0, "G": 3.0, "factor": factor, "new_Z": Z, "new_G": G,
                           "momentum": 1.0, "amplitude": propagator_amplitude(Z, G, 1.0),
                           "amplitude_residual": propagator_amplitude(Z, G, 1.0) - propagator_amplitude(2.0, 3.0, 1.0),
                           "log_gradient": [1.0, 1.0], "log_null_direction": [1.0, -1.0],
                           "log_null_residual": 0.0})
    b, c, p, cutoff = 2.0, -0.5, 5, 3.0
    identifiability = induced_identifiability_report(b, c, p, cutoff)
    identifiability["parameters"] = {"b": b, "c": c, "p": p, "cutoff": cutoff}
    identifiability["bare_counterterm_shifts"] = []
    identifiability["cutoff_shifts"] = []
    original = identifiability["M_eff_squared"]
    for shift in (-0.25, 0.25):
        new_b, new_c = induced_bare_counterterm_shift(b, c, p, cutoff, shift)
        identifiability["bare_counterterm_shifts"].append({"shift": shift, "b": new_b, "c": new_c,
                                                         "coefficient_residual": induced_effective_coefficient(new_b, new_c, p, cutoff) - original})
    for new_cutoff in (2.0, 4.0):
        new_b, new_c, new_cutoff = induced_cutoff_shift(b, c, p, cutoff, new_cutoff)
        identifiability["cutoff_shifts"].append({"cutoff": new_cutoff, "b": new_b, "c": new_c,
                                               "coefficient_residual": induced_effective_coefficient(new_b, new_c, p, new_cutoff) - original})
    H, source = [[1.0, 0.2], [0.2, 2.0]], [[0.0, 1.0], [1.0, 0.0]]
    return {"model_id": MODEL_ID, "exact_action_formulas": actions,
            "leading_slow_roll": [leading_slow_roll(Ne, alpha) for Ne in (40.0, 55.0, 70.0) for alpha in (0.5, 2.0, 10.0)],
            "inverse_calibrations": calibrations, "si_energy_and_entropy_controls": units,
            "conditional_induced_units": induced_units, "constant_ZG_degeneracy": propagator,
            "induced_identifiability": identifiability,
            "flat_identity_shifts": [flat_identity_shift_report(H, source, offset) for offset in (-3.0, 4.0)],
            "limitations": list(LIMITATIONS)}
