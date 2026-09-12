"""Conditional fixed-occupation source selection, not a BPR flavor derivation.

Two positive trace-one mean-field matrices on the assumed q=3 round sphere
source real scalars through n_rho=Tr(rho T), T_ij=psi_i* psi_j. The fixed-trace
constraint makes the joint scalar/occupation action bounded. These variables
are not asserted Grassmann condensates or fully occupied fermion states.

Eliminating the positive scalar Hessian gives a strictly convex quadratic
reward on the product density spaces. Any maximum must therefore be pure in
both sectors. For a pure spin-one state, with spin length s, the density powers
are (1/4, 3*s²/16, (4-3*s²)/80)/(pi R⁴). For either inverse-response weight
w_l=a_l or b_l, the slope of its weighted norm is proportional to 5*w_1-w_2.
It is strictly positive for a_l, including kappa=0; for b_l it is positive
when eta>0. Weighted Cauchy-Schwarz bounds the cross reward by the product of
the maximal weighted norms. All inequalities saturate for aligned coherent
projectors. Since b_l>0 on every harmonic when eta>0, equality forces identical
density coefficients, hence identical occupations. At eta=0 the coherent
axes are independently free: relative mixing is unselected, not predicted.

This is a classical minimizing orbit, not an exact quantum ground state or a
proof of spontaneous symmetry breaking on a finite sphere. No physical masses,
empirical targets, optimization, or Monte Carlo are used.
"""
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from bpr.chiral_flavor_prototype import (
    HARMONIC_DEGREES, HARMONIC_NAMES, HARMONIC_NORMALIZATIONS,
    mixing_observables, monopole_profiles, overlap_matrix, real_harmonics,
    scalar_action, sphere_quadrature, yukawa_matrix,
)

MODEL_ID = "conditional-fixed-occupation-flavor-selection-v1"
_ROUNDOFF = 128 * np.finfo(float).eps
_COHERENT_POWERS = np.array([1 / 4, 3 / 16, 1 / 80]) / np.pi


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"numerically unresolved {name}: nonfinite result")
    return value


def _radius_squared(R):
    R = _real(R, "R", positive=True)
    with np.errstate(over="ignore", under="ignore"):
        r2 = np.float64(R) * R
    if not np.isfinite(r2) or r2 == 0 or not np.isfinite(1 / r2):
        raise ValueError("numerically unresolved radius squared")
    return float(r2)


@dataclass(frozen=True)
class SelectionParameters:
    """Declared toy inputs; no tuning or physical matching is performed.

    Floating-point unresolved Hessians/scales are rejected, not regularized.
    Rejection can be conservative: an overflowing intermediate may be rejected
    even if a rescaled or higher-precision calculation could resolve the final
    value. No assertion of mathematically infinite energy is made in that case.
    y_eff inputs are separate from these source/action couplings.
    """
    R: float = 1.0
    kappa: float = 1.0
    mu2: float = 1.0
    eta: float = 0.25
    g_u: float = 1.0
    g_d: float = 1.0
    h0: float = 2.0

    def __post_init__(self):
        for name in ("R", "mu2", "g_u", "g_d"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        for name in ("kappa", "eta", "h0"):
            object.__setattr__(self, name, _real(getattr(self, name), name, nonnegative=True))
        if self.mu2 <= self.eta:
            raise ValueError("scalar Hessian requires mu2 > eta >= 0")
        if self.mu2 - self.eta <= _ROUNDOFF * self.mu2:
            raise ValueError("numerically unresolved scalar Hessian gap")
        r2 = _radius_squared(self.R)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            d = self.mu2 + self.kappa * np.array([0., 2., 6.]) / r2
            hessian_eigenvalues = r2 * np.r_[d - self.eta, d + self.eta]
            inverse = 1 / np.r_[d - self.eta, d + self.eta]
        _finite(hessian_eigenvalues, "scalar Hessian")
        _finite(inverse, "inverse scalar Hessian")
        if np.any(hessian_eigenvalues <= 0) or np.any(inverse <= 0):
            raise ValueError("numerically unresolved scalar Hessian scale")


def _parameters(parameters):
    if parameters is None:
        return SelectionParameters()
    if not isinstance(parameters, SelectionParameters):
        raise TypeError("parameters must be SelectionParameters")
    return parameters


def _q(q):
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, Integral):
        raise TypeError("q must be a positive integer")
    if q <= 0:
        raise ValueError("q must be a positive integer")
    return int(q)


def validated_density_matrix(rho, q=3):
    """Copy a finite Hermitian PSD trace-one matrix without clipping/renormalizing.

    Absolute roundoff tolerance is 128 machine eps on this trace-one space.
    Eigenvalues below minus this tolerance are invalid; tiny roundoff negatives
    are accepted unchanged. No approximate physical projection is performed.
    """
    q = _q(q)
    rho = np.array(rho, dtype=complex, copy=True)
    if rho.shape != (q, q):
        raise ValueError(f"rho must have shape ({q}, {q})")
    _finite(rho, "density matrix")
    if np.max(np.abs(rho - rho.conj().T)) > _ROUNDOFF:
        raise ValueError("rho must be Hermitian")
    if abs(np.trace(rho) - 1) > _ROUNDOFF:
        raise ValueError("rho must have trace one; no renormalization is applied")
    if np.linalg.eigvalsh(rho).min() < -_ROUNDOFF:
        raise ValueError("rho must be positive semidefinite")
    return rho


def equal_occupation(q=3):
    """Normalized equal-filled control I/q, not a fully filled Pauli state I."""
    q = _q(q)
    return np.eye(q, dtype=complex) / q


def coherent_occupation(theta=0.0, phi=0.0, q=3):
    """Projector at the stated physical direction; conjugate the profile vector.

    With T_ij=psi_i* psi_j, v=conj(psi(theta,phi))/||psi|| is essential.
    Omitting conjugation would reflect a nonzero source azimuth.
    """
    q = _q(q)
    theta, phi = _real(theta, "theta"), _real(phi, "phi")
    psi = monopole_profiles(theta, phi, q=q)
    v = psi.conj() / np.linalg.norm(psi)
    return validated_density_matrix(np.outer(v, v.conj()), q)


def occupation_density(rho, theta, phi, R=1.0, patch_phase=0.0):
    """Nonnegative n=Tr(rho T), integral dA n=1; supports any positive q.

    Patch phases may be arrays broadcastable to the supplied angles. Real
    roundoff residuals are returned unchanged rather than positivity-clipped.
    """
    array = np.asarray(rho)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("rho must be square")
    rho = validated_density_matrix(rho, array.shape[0])
    _radius_squared(R)
    psi = monopole_profiles(theta, phi, q=len(rho), R=R, patch_phase=patch_phase)
    result = np.einsum("...i,ij,...j->...", psi, rho, psi.conj())
    _finite(result, "occupation density")
    scale = len(rho) / (4 * np.pi * R**2)
    if np.any(np.abs(result.imag) > _ROUNDOFF * scale) or np.any(result.real < -_ROUNDOFF * scale):
        raise ValueError("numerically unresolved real positive occupation density")
    return result.real


def density_coefficients(rho, R=1.0):
    """Exact q=3 l<=2 dOmega coefficients d_A=Tr(rho P(Y_A))/R².

    Products of spin-one sections contain only l=0,1,2. Nothing induced is
    dropped. P(Y_A) is radius independent; reuse the prototype quadrature.
    """
    rho = validated_density_matrix(rho)
    r2 = _radius_squared(R)
    matrices = np.array([overlap_matrix(row) for row in np.eye(9)])
    traces = np.einsum("ij,aji->a", rho, matrices)
    if np.max(np.abs(traces.imag)) > _ROUNDOFF:
        raise ValueError("numerically unresolved real density coefficients")
    return _finite(traces.real / r2, "density coefficients")


def density_multipole_powers(rho, R=1.0):
    """Return sum_{A:degree=l} d_A² for l=0,1,2."""
    coefficients = density_coefficients(rho, R)
    with np.errstate(over="ignore", under="ignore"):
        result = np.array([np.dot(coefficients[HARMONIC_DEGREES == ell],
                                  coefficients[HARMONIC_DEGREES == ell]) for ell in range(3)])
    _finite(result, "multipole powers")
    if result[0] == 0:
        raise ValueError("numerically unresolved density monopole power")
    return result


def spin_vector(rho):
    """Physical-axis spin-one expectation (2*P(x),2*P(y),2*P(z)).

    The y matrix is minus the conventional magnetic-basis J_y because the
    section evaluation vector is conjugated in the density convention.
    """
    rho = validated_density_matrix(rho)
    matrices = (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2),
                np.array([[0, 1j, 0], [-1j, 0, 1j], [0, -1j, 0]]) / np.sqrt(2),
                np.diag([1, 0, -1]))
    return np.array([np.trace(rho @ matrix).real for matrix in matrices])


def response_weights(parameters=None):
    """Return (a_l,b_l), l=0,1,2, inverse blocks [[a,b],[b,a]].

    Use factored eigenvalues rather than d²-eta² to avoid gratuitous overflow
    and cancellation. Positive eta with an underflowed b is rejected.
    """
    p = _parameters(parameters)
    d = p.mu2 + p.kappa * np.array([0., 2., 6.]) / p.R**2
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        a = (0.5 / (d - p.eta)) + (0.5 / (d + p.eta))
        b = (p.eta / (d + p.eta)) / (d - p.eta)
    _finite(a, "diagonal response")
    _finite(b, "off-diagonal response")
    if np.any(a <= 0) or (p.eta > 0 and np.any(b <= 0)):
        raise ValueError("numerically unresolved positive response weights")
    return a, b


def occupation_sources(rho_u, rho_d, parameters=None):
    """Return j_aA=g_a*d_aA, shape (2,9), up then down."""
    p = _parameters(parameters)
    with np.errstate(over="ignore", under="ignore"):
        result = np.array([p.g_u * density_coefficients(rho_u, p.R),
                           p.g_d * density_coefficients(rho_d, p.R)])
    _finite(result, "occupation sources")
    if np.any(result[:, 0] <= 0):
        raise ValueError("numerically unresolved source monopole")
    return result


def _profiles(h):
    h = np.asarray(h)
    if h.shape != (2, 9) or (np.iscomplexobj(h) and np.any(h.imag != 0)):
        raise ValueError("profiles must be real with shape (2,9)")
    return _finite(np.asarray(h.real, dtype=float), "profiles")


def _fluctuations(h, p):
    delta = _profiles(h).copy()
    delta[:, 0] -= p.h0 * np.sqrt(4 * np.pi)
    return _finite(delta, "scalar fluctuations")


def coupled_scalar_response(rho_u, rho_d, parameters=None):
    """Unique full scalar minimizer h_A=h0 sqrt(4pi) delta_A0+D_l^-1 j_A."""
    p = _parameters(parameters)
    j = occupation_sources(rho_u, rho_d, p)
    a, b = response_weights(p)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = a[HARMONIC_DEGREES] * j + b[HARMONIC_DEGREES] * j[::-1]
        h = delta.copy()
        h[:, 0] += p.h0 * np.sqrt(4 * np.pi)
    _finite(h, "scalar response")
    recovered = _fluctuations(h, p)
    if np.any(delta[:, 0] <= 0) or np.any(np.abs(recovered[:, 0] - delta[:, 0]) > 1e-10 * delta[:, 0]):
        raise ValueError("numerically unresolved scalar response relative to h0")
    return h


def coupled_scalar_hessian(parameters=None):
    """Positive 18x18 Hessian in sector-major (up[0:9],down[0:9]) order."""
    p = _parameters(parameters)
    diagonal = np.diag(p.R**2 * p.mu2 + p.kappa * HARMONIC_DEGREES * (HARMONIC_DEGREES + 1))
    cross = -p.R**2 * p.eta * np.eye(9)
    return _finite(np.block([[diagonal, cross], [cross, diagonal]]), "scalar Hessian")


def coupled_scalar_action(h, rho_u, rho_d, parameters=None):
    """Independent action evaluation: two prototype actions minus eta integral f_u f_d."""
    p = _parameters(parameters)
    h = _profiles(h)
    j = occupation_sources(rho_u, rho_d, p)
    delta = _fluctuations(h, p)
    with np.errstate(over="ignore", invalid="ignore"):
        value = sum(scalar_action(h[a], j[a], R=p.R, kappa=p.kappa, mu2=p.mu2, h0=p.h0)
                    for a in range(2)) - p.R**2 * p.eta * np.dot(delta[0], delta[1])
    return float(_finite(value, "joint action"))


def coupled_scalar_stationarity(h, rho_u, rho_d, parameters=None):
    """Area-normalized action gradient, shape (2,9)."""
    p = _parameters(parameters)
    delta = _fluctuations(h, p)
    j = occupation_sources(rho_u, rho_d, p)
    with np.errstate(over="ignore", invalid="ignore"):
        result = (coupled_scalar_hessian(p) @ delta.ravel()).reshape(2, 9) - p.R**2 * j
    return _finite(result, "scalar stationarity")


def _energy_from_reward(p, reward):
    _finite(reward, "positive quadratic reward")
    if reward <= 0:
        raise ValueError("numerically unresolved positive quadratic reward")
    baseline = p.h0 * (p.g_u + p.g_d)
    value = -baseline - reward
    _finite(value, "effective energy")
    if abs((-value-baseline)-reward) > 1e-10 * reward:
        raise ValueError("numerically unresolved energy reward relative to baseline")
    return float(value)


def effective_energy(rho_u, rho_d, parameters=None):
    """Exact eliminated E=-h0(g_u+g_d)-R²/2 sum_A j_A^T D_l^-1 j_A."""
    p = _parameters(parameters)
    j = occupation_sources(rho_u, rho_d, p)
    a, b = response_weights(p)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        reward = np.sum(a[HARMONIC_DEGREES] * (j[0]**2 + j[1]**2)
                        + 2 * b[HARMONIC_DEGREES] * j[0] * j[1])
        reward *= p.R**2 / 2
    return _energy_from_reward(p, reward)


def analytic_energy_bound(parameters=None):
    """Exact global lower bound, achieved by the minimizing family.

    Strict convexity forces pure occupations. The spin-one power identity and
    5*a_1>a_2 select s²=1. Weighted Cauchy-Schwarz selects a common coherent
    axis when eta>0; eta=0 leaves both axes free. See module proof.
    """
    p = _parameters(parameters)
    a, b = response_weights(p)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        g = np.array([p.g_u, p.g_d])
        reward = np.dot(_COHERENT_POWERS, a * np.sum(g*g) + 2*b*g[0]*g[1]) / (2*p.R**2)
    return _energy_from_reward(p, reward)


def energy_bound_gap(rho_u, rho_d, parameters=None):
    """Stable E_eff-E_min and its arithmetic resolution, excluding common l=0.

    Fixed trace fixes the monopole exactly; subtracting total energies would
    erase a small but resolved angular gap under that common baseline. A gap
    within the reported roundoff scale is consistent with saturation, not a
    numerical proof of a minimizing occupation. No clipping is performed.
    """
    p = _parameters(parameters)
    a, b = response_weights(p)
    g = np.array([p.g_u, p.g_d])
    densities = np.array([density_coefficients(rho_u), density_coefficients(rho_d)])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        j = g[:, None] * densities
        terms, scales = [], []
        for ell in (1, 2):
            part = j[:, HARMONIC_DEGREES == ell]
            upper = _COHERENT_POWERS[ell] * (a[ell]*np.sum(g*g) + 2*b[ell]*g[0]*g[1])
            diagonal = a[ell] * np.sum(part*part)
            cross = 2*b[ell] * np.dot(part[0], part[1])
            terms.append((upper-diagonal-cross)/(2*p.R**2))
            scales.append((abs(upper)+abs(diagonal)+abs(cross))/(2*p.R**2))
        gap = float(np.sum(terms))
        resolution = float(_ROUNDOFF*np.sum(scales))
    _finite(gap, "angular energy bound gap")
    _finite(resolution, "angular energy bound gap resolution")
    if not np.any(np.asarray(scales) > 0) or resolution == 0:
        raise ValueError("numerically unresolved angular energy bound gap")
    if gap < -resolution:
        raise ValueError("numerically unresolved violation of analytic energy bound")
    status = ("positive angular energy gap" if gap > resolution else
              "consistent with saturation at arithmetic resolution; not a numerical selection proof")
    return {"gap": gap, "resolution": resolution, "status": status}


def coherent_axis_profiles(parameters=None):
    """Analytic z-axis (A,B,C) for h=A+B*z+C*P2(z), shape (2,3).

    B>=3C>0 and A-h0>=2B/3 ensure the smallest overlap is at least
    h0+B/6+C/10>0. Gaps B/2 +/- 3C/10 are strictly positive. Reject
    numerically unresolved spectral gaps rather than inventing eigenvectors.
    """
    p = _parameters(parameters)
    a, b = response_weights(p)
    g = np.array([p.g_u, p.g_d])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        w = g[:, None] * a + g[::-1, None] * b
        abc = w * np.array([1/(4*np.pi), 3/(8*np.pi), 1/(8*np.pi)]) / p.R**2
        abc[:, 0] += p.h0
    _finite(abc, "coherent axis profile")
    A, B, C = abc.T
    gaps = np.stack((B/2 - 3*C/10, B/2 + 3*C/10))
    if np.any(C <= 0) or np.any(gaps <= _ROUNDOFF * np.maximum(A, B)):
        raise ValueError("numerically unresolved coherent spectral gaps")
    return abc


def analytic_overlap_diagonal(parameters=None):
    """North-axis overlap entries in original profile order, not ascending order."""
    A, B, C = coherent_axis_profiles(parameters).T
    diagonal = np.stack((A+B/2+C/10, A-C/5, A-B/2+C/10), axis=1)
    if np.any(diagonal <= 0):
        raise ValueError("numerically unresolved positive coherent overlap")
    return _finite(diagonal, "analytic overlap diagonal")


def minimizing_family(theta=0.0, phi=0.0, *, down_theta=None, down_phi=None, parameters=None):
    """Return analytic minimizing occupations, shape (2,3,3).

    eta>0 permits a common axis only. eta=0 permits independent down angles;
    omitting them simply chooses an aligned representative, not an alignment
    prediction. Angles denote absolute axes, not relative Euler parameters.
    """
    p = _parameters(parameters)
    up = coherent_occupation(theta, phi)
    down = coherent_occupation(theta if down_theta is None else down_theta,
                               phi if down_phi is None else down_phi)
    if p.eta > 0 and not np.allclose(up, down, rtol=0, atol=_ROUNDOFF):
        raise ValueError("eta>0 global minima require identical coherent occupations")
    return np.stack((up, down))


def solution_diagnostics(rho_u, rho_d, parameters=None, *, y_eff_u=1.0, y_eff_d=1.0):
    """Scalar solution, action/bound residuals, and inherited SVD observables.

    This function reports a chosen pair, not a proof that the pair minimizes
    over occupations. Degenerate SVD observables retain the prototype's None.
    """
    p = _parameters(parameters)
    couplings = [_real(y_eff_u, "y_eff_u", positive=True), _real(y_eff_d, "y_eff_d", positive=True)]
    rho = np.stack((validated_density_matrix(rho_u), validated_density_matrix(rho_d)))
    h = coupled_scalar_response(*rho, p)
    matrices = [yukawa_matrix(h[a], y_eff=couplings[a], R=p.R) for a in range(2)]
    for matrix in matrices:
        _finite(matrix, "Yukawa matrix")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        obs = mixing_observables(*matrices)
    def check_diagnostics(value):
        if isinstance(value, dict):
            for item in value.values():
                check_diagnostics(item)
        elif isinstance(value, (np.ndarray, Real, complex)):
            _finite(value, "Yukawa/SVD diagnostics")
    check_diagnostics(obs)
    energy, bound = effective_energy(*rho, p), analytic_energy_bound(p)
    gap = energy_bound_gap(*rho, p)
    return {"density_matrices": rho, "density_coefficients": np.array([density_coefficients(r, p.R) for r in rho]),
            "source_coefficients": occupation_sources(*rho, p), "profile_coefficients": h,
            "action": coupled_scalar_action(h, *rho, p), "effective_energy": energy,
            "energy_bound": bound, "bound_residual": gap["gap"],
            "bound_residual_resolution": gap["resolution"], "bound_residual_status": gap["status"],
            "stationarity_norm": float(np.linalg.norm(coupled_scalar_stationarity(h, *rho, p))),
            "observables": obs, "status": "chosen occupations with unique minimizing scalar response"}


def assumption_ledger():
    """Separate model inputs, proven conditional selection, and unresolved physics."""
    return {"scope": "new conditional fixed-occupation mean-field model, not sources derived from BPR",
            "assumed": ["q=3, round S2, fixed radius and inherited conventional 4D chiral EFT/overlap prescription",
                "two PSD trace-one occupation matrices as added collective variables, not SM Grassmann expectations",
                "positive source couplings, mu2>eta>=0, kappa>=0, h0>=0 and declared dimensionless toy units",
                "fixed trace is a boundedness constraint, not a derivation of three families"],
            "density_convention": "T_ij=psi_i*psi_j; n=Tr(rho T); integral dA h n=Tr(rho P(h)); d_A=Tr(rho P(Y_A))/R^2",
            "source_freedom": "positive trace-one matrices have eight continuous directions near I/3; their constrained sources do not cover all signed sources",
            "established_conditionally": ["strict positive scalar Hessian and exact elimination with no induced l>2 source",
                "global minima are coherent occupations; eta>0 aligns their axes and gives no nontrivial mixing",
                "eta=0 leaves relative mixing unselected; equal sector couplings give equal spectra"],
            "quantum_boundary": "classical minimizing orbit, not an exact quantum ground state or proof of SSB on a finite sphere",
            "pauli_boundary": "a fully filled three-state fermion system with rho<=I and trace=3 forces rho=I and constant density, unlike normalized trace=1",
            "not_solved": ["substrate derivation of occupation dynamics and coupling signs", "flux/family-count selection",
                "radius stabilization", "UV completion", "physical units/masses and nontrivial predicted flavor mixing"],
            "data_policy": "frozen declared toy parameters; no empirical targets, fitting, optimizer, MC or benchmark imports"}


def demonstration():
    """Frozen structured report; arrays/complex values use the caller's serializer."""
    p = SelectionParameters()  # Freeze inputs before calculating any outputs.
    p0 = SelectionParameters(eta=0.0)
    equal = equal_occupation()
    selected = minimizing_family(parameters=p)
    control = solution_diagnostics(equal, equal, p)
    control["status"] = "equal-filled stationary control, not a global occupation minimum"
    aligned = solution_diagnostics(*selected, p)
    aligned.update({"status": "analytic global minimum", "axis_profile_ABC": coherent_axis_profiles(p),
                    "analytic_overlap_diagonal": analytic_overlap_diagonal(p),
                    "selection_status": "aligned coherent occupations; no mixing beyond phases and permutations"})
    representatives = {}
    for name, theta, phi in (("aligned", 0.0, 0.0), ("tilted", np.pi/2, np.pi/3)):
        rho = minimizing_family(down_theta=theta, down_phi=phi, parameters=p0)
        representatives[name] = solution_diagnostics(*rho, p0)
        representatives[name]["status"] = "chosen representative of independent coherent minimizing axes"
    # Exact rotational averaging of coherent projectors: integral dOmega rho/(4pi)=I/3.
    theta, phi, weights = sphere_quadrature(n_polar=8, n_azimuth=16)
    psi = monopole_profiles(theta, phi)
    vectors = psi.conj() / np.sqrt(3/(4*np.pi))
    averaged = np.einsum("p,pi,pj->ij", weights/(4*np.pi), vectors, vectors.conj())
    return {"model_id": MODEL_ID,
            "inputs": {**asdict(p), "q": 3, "y_eff_u": 1.0, "y_eff_d": 1.0,
                "harmonic_names": HARMONIC_NAMES, "harmonic_normalizations": HARMONIC_NORMALIZATIONS.copy(),
                "output_units": "dimensionless effective Yukawa couplings, not physical masses",
                "coefficient_normalization": "real harmonics orthonormal in dOmega, not dA"},
            "assumption_ledger": assumption_ledger(), "equal_filled": control, "aligned": aligned,
            "uncoupled": {"eta": 0.0, "selection_status": "relative mixing unselected",
                "mixing": None, "abs_mixing": None, "J": None,
                "mixing_undefined_reason": "independent relative coherent orientations are not selected by the energy",
                "representatives": representatives},
            "rotational_average": {"density_matrix": averaged,
                "averaged_orbit_energy": aligned["effective_energy"],
                "energy_of_average_density": effective_energy(averaged, averaged, p),
                "explanation": "averaging the common coherent orbit gives I/3 and constant density; averaging nonlinear minimized energies is not evaluating the energy of averaged occupations; no finite-sphere quantum SSB is established"}}
