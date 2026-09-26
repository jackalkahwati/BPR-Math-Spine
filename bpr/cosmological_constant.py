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
"""

from math import pi

import sympy as sp

MODEL_ID = "bpr6d-cosmological-constant-v1"

M, e, Lam, u, alpha, T = sp.symbols("M e Lambda u alpha T", positive=True)
m = sp.symbols("m", positive=True, integer=True)

LIMITATIONS = [
    "Classical; branes are idealized delta-function sources of tension only (no brane-localized flux or couplings).",
    "Only the non-supersymmetric Einstein-Maxwell-Lambda theory is computed; supersymmetric (Salam-Sezgin/SLED) claims are cited, not derived.",
    "Quantum corrections to the 6D and brane vacuum energies are represented only by a shift of T or Lambda.",
]

OBSERVED_RHO_LAMBDA_GEV4 = (2.3e-12) ** 4  # (2.3 meV)^4
REDUCED_PLANCK_GEV = 2.435e18


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
    """Observed vacuum energy against natural scales."""
    return {"rho_obs_over_MPl4": OBSERVED_RHO_LAMBDA_GEV4 / REDUCED_PLANCK_GEV ** 4,
            "rho_obs_over_inverse_radius4": OBSERVED_RHO_LAMBDA_GEV4 / inverse_radius_gev ** 4,
            "rho_obs_over_TeV4": OBSERVED_RHO_LAMBDA_GEV4 / 1e3 ** 4}


def demonstration_report():
    test = self_tuning_test()
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "no_classical_self_tuning",
        "empirical_validation": False,
        "self_tuning_test": {k: str(v) for k, v in test.items()},
        "discrete_flat_tensions": discrete_flat_tensions(),
        "tuning_magnitudes": tuning_magnitudes(),
        "limitations": list(LIMITATIONS),
    }
