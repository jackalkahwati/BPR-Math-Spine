"""The cosmological-constant tuning of BPR-6D and the brane self-tuning question.

See doc/derivations/cosmological_constant_2026-09-26.md. Classical 6D Einstein-
Maxwell-Lambda on (A)dS4 x S^2 with two antipodal codimension-2 branes of
tension T ("football"/"rugby ball"): the branes cut a wedge, so the sphere metric
is d theta^2 + alpha^2 sin^2 theta d phi^2 with alpha = 1 - delta/(2 pi), deficit
delta = T / M^4 (from (M^4/2) R, i.e. M^4 = 1/(8 pi G6)). The local equations are
unchanged; flux quantization on the reduced area makes m -> m / alpha.

The 4D curvature H^2 (dS > 0, AdS < 0) follows from the full 6D equations
(flux_compactification section 6 oracle): H^2 = 1/(3 r^2) - B^2/(3 M^4) with
Lambda + (3/2) B^2 - 2 M^4 / r^2 = 0 and B = m / (2 e alpha r^2).

The flux quantum m is held fixed as the tension varies: dF = 0 conserves the flux, and BPR-6D has no
magnetically charged branes. (Carroll-Guica instead hold the field strength B fixed, tuned against Lambda.)
"""

from math import pi

import sympy as sp

MODEL_ID = "bpr6d-cosmological-constant-v2"

M, e, Lam, u, alpha, T = sp.symbols("M e Lambda u alpha T", positive=True)
m = sp.symbols("m", positive=True, integer=True)

LIMITATIONS = [
    "Classical; branes are idealized delta-function sources of tension only (no brane-localized flux or couplings).",
    "Only the unwarped ansatz with two equal tensions is solved; warped solutions with unequal tensions exist and also need one flatness relation.",
    "Radion (breathing-mode) stability is shown; other modes of the football, which breaks SU(2) to U(1), are not analysed.",
    "Only the non-supersymmetric Einstein-Maxwell-Lambda theory is computed; supersymmetric (Salam-Sezgin/SLED) claims are cited, not derived.",
    "Quantum corrections to the 6D and brane vacuum energies are represented only by a shift of T or Lambda.",
]

OBSERVED_RHO_LAMBDA_GEV4 = (2.3e-12) ** 4  # (2.3 meV)^4
REDUCED_PLANCK_GEV = 2.435e18
PLANCK_GEV = 1.221e19


def stationary_radius_squared():
    """Smaller root u = r^2 of Lambda u^2 - 2 M^4 u + (3/8) (m/alpha)^2 / e^2 = 0 (the stable branch)."""
    disc = 4 * M ** 8 - 4 * Lam * sp.Rational(3, 8) * m ** 2 / (e ** 2 * alpha ** 2)
    return (2 * M ** 4 - sp.sqrt(disc)) / (2 * Lam)


def hubble_squared():
    """H^2 as a function of (Lambda, m, alpha) on the stable branch."""
    uu = stationary_radius_squared()
    B2 = m ** 2 / (4 * e ** 2 * alpha ** 2 * uu ** 2)
    return sp.simplify(1 / (3 * uu) - B2 / (3 * M ** 4))


def flat_alpha():
    """alpha at which H^2 = 0: alpha = m sqrt(Lambda / 2) / (M^4 e)."""
    return m * sp.sqrt(Lam / 2) / (M ** 4 * e)


def flatness_conditions():
    """Solve H^2 = 0 with the constraint on either root: u = M^4/(2 Lambda), B^2 = 2 Lambda, alpha = alpha_flat.

    So at fixed flux every tension other than the flat one curves 4D: a global statement, not only first order.
    """
    uu, B2 = sp.symbols("u_s B2_s", positive=True)
    sol = sp.solve([1 / (3 * uu) - B2 / (3 * M ** 4), Lam + sp.Rational(3, 2) * B2 - 2 * M ** 4 / uu], [uu, B2], dict=True)
    assert len(sol) == 1
    u_flat, B2_flat = sol[0][uu], sol[0][B2]
    alpha_sol = sp.solve(sp.Eq(m ** 2 / (4 * e ** 2 * alpha ** 2 * u_flat ** 2), B2_flat), alpha)
    return {"u": u_flat, "B2": B2_flat, "alpha": alpha_sol}


def flat_point_derivative_closed_form():
    """dH^2/dT at the flat point for general (Lambda, M, e, m); expected sqrt(2) e sqrt(Lambda) / (3 pi m M^4) > 0."""
    H2 = hubble_squared()
    d = sp.diff(H2, alpha).subs(alpha, flat_alpha()) * (-1 / (2 * sp.pi * M ** 4))
    return sp.simplify(d)


def radion_potential():
    """Einstein-frame breathing-mode potential at fixed alpha, up to a positive factor.

    The brane action cancels the conical curvature, so the football potential is the round-sphere one with
    m -> m/alpha: V(u) ~ Lambda/u - M^4/u^2 + (m/alpha)^2/(8 e^2 u^3), u = r^2.
    """
    return Lam / u - M ** 4 / u ** 2 + (m / alpha) ** 2 / (8 * e ** 2 * u ** 3)


def radion_stability():
    """V'(u) = 0 reproduces the constraint; V'' ~ u (M^4 - Lambda u) there, so the smaller root is the stable one."""
    V = radion_potential()
    stationary = sp.simplify(sp.diff(V, u) * (-u ** 4))
    constraint = Lam * u ** 2 - 2 * M ** 4 * u + sp.Rational(3, 8) * (m / alpha) ** 2 / e ** 2
    V2 = sp.diff(V, u, 2) * u ** 5 / 2
    V2_on_shell = sp.simplify(V2.subs(m, sp.solve(constraint, m)[0]))
    # The smaller root lies below M^4/Lambda by sqrt(disc)/(2 Lambda) >= 0, so V'' > 0 there (and < 0 at the larger).
    gap = sp.simplify(M ** 4 / Lam - stationary_radius_squared())
    disc = 4 * M ** 8 - 4 * Lam * sp.Rational(3, 8) * m ** 2 / (e ** 2 * alpha ** 2)
    return {"stationarity_is_constraint": sp.simplify(stationary - constraint) == 0,
            "V2_on_shell": V2_on_shell,
            "smaller_root_stable": bool(sp.simplify(V2_on_shell - u * (M ** 4 - Lam * u)) == 0
                                        and sp.simplify(gap - sp.sqrt(disc) / (2 * Lam)) == 0)}


def tension_of(alpha_value):
    return 2 * sp.pi * M ** 4 * (1 - alpha_value)


def self_tuning_test(values=None):
    """At the flat point, dH^2/dT != 0: a shift of the brane tension (brane vacuum energy) curves 4D space."""
    values = values or {M: 1, e: sp.Rational(1, 2), m: 3, Lam: sp.Rational(1, 50)}
    H2 = hubble_squared()
    a_flat = flat_alpha().subs(values)
    H2_flat = sp.nsimplify(sp.simplify(H2.subs(values).subs(alpha, a_flat)))
    dH2_dalpha = sp.diff(H2, alpha).subs(values).subs(alpha, a_flat)
    dH2_dT = sp.simplify(dH2_dalpha * (-1 / (2 * sp.pi * values[M] ** 4)))  # dalpha/dT = -1/(2 pi M^4)
    return {"alpha_flat": a_flat, "H2_at_flat": H2_flat, "dH2_dT": sp.nsimplify(dH2_dT),
            "tension_flat": tension_of(a_flat).subs(values),
            "self_tuning": bool(sp.simplify(dH2_dT) == 0)}


def discrete_flat_tensions(values=None, fluxes=range(1, 8)):
    """For fixed Lambda, M, e the flat tensions form a discrete set, one per flux quantum (0 < alpha <= 1)."""
    values = values or {M: 1, e: sp.Rational(1, 2), Lam: sp.Rational(1, 50)}
    out = []
    for mm in fluxes:
        a = flat_alpha().subs(values).subs(m, mm)
        if 0 < a <= 1:
            out.append({"flux": mm, "alpha": float(a), "tension": float(tension_of(a).subs(values))})
    return out


def tuning_magnitudes(inverse_radius_gev=1.6e17):
    """Observed vacuum energy against natural scales (M_Pl reduced; the unreduced Planck mass gives ~1e-123)."""
    return {"rho_obs_over_MPl4": OBSERVED_RHO_LAMBDA_GEV4 / REDUCED_PLANCK_GEV ** 4,
            "rho_obs_over_unreduced_MPl4": OBSERVED_RHO_LAMBDA_GEV4 / PLANCK_GEV ** 4,
            "rho_obs_over_inverse_radius4": OBSERVED_RHO_LAMBDA_GEV4 / inverse_radius_gev ** 4,
            "rho_obs_over_TeV4": OBSERVED_RHO_LAMBDA_GEV4 / 1e3 ** 4}


def demonstration_report():
    test = self_tuning_test()
    return {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "status": "no_classical_self_tuning",
        "empirical_validation": False,
        "self_tuning_test": {k: str(v) for k, v in test.items()},
        "flat_point_dH2_dT": str(flat_point_derivative_closed_form()),
        "flatness_conditions": {k: str(v) for k, v in flatness_conditions().items()},
        "radion_stability": {k: str(v) for k, v in radion_stability().items()},
        "discrete_flat_tensions": discrete_flat_tensions(),
        "tuning_magnitudes": tuning_magnitudes(),
        "limitations": list(LIMITATIONS),
    }
