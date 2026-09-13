"""Exact local anomaly of one supplied complex six-dimensional Weyl 16.

Spin(10) rank five and its sixteen half-integral weights are fixed. Neither
this supplied parent nor its anomaly cancellation is derived from BPR. Gauge
curvature is F/(2*pi*i); Q is charge, m is integral_S2 X, and q=Q*m is the
signed internal index. Representation parity is not spacetime chirality.

Anomaly forms are formal descent polynomials, not ordinary nonzero six-forms
on a four-dimensional base. The sphere calculation assumes a product tangent
bundle and Spin(10) curvature pulled back from the base. No reality halving,
cancellation fields, Green-Schwarz completion, or global-anomaly claim is made.
"""
from functools import lru_cache
from itertools import product
from numbers import Integral
from types import MappingProxyType

import sympy as sp

from bpr.chiral_flavor_prototype import signed_mode_bookkeeping

MODEL_ID = "supplied-6d-spin10-weyl16-local-anomaly-v1"
# Computational coefficient-size cap, not a physical charge or flux bound.
MAX_ABS_INTEGER = 10**6
x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5")
CARTAN_SYMBOLS = (x1, x2, x3, x4, x5)
S2, S4, p1, p2, X, X4, Q = sp.symbols("S2 S4 p1 p2 X X4 Q")
POLYNOMIAL_GENERATORS = (S2, S4, p1, p2, X, X4, Q)
FORM_DEGREES = MappingProxyType({
    **{symbol: 2 for symbol in CARTAN_SYMBOLS},
    S2: 4, S4: 8, p1: 4, p2: 8, X: 2, X4: 2, Q: 0,
})
CONTRIBUTION_KEYS = (
    "pure_spin10", "mixed_spin10_u1", "pure_u1",
    "mixed_gravity_spin10", "mixed_gravity_u1", "pure_gravity",
)


def _integer(value, name, minimum, maximum):
    # Check type and bounds before int conversion or any symbolic expansion.
    if isinstance(value, bool) or not isinstance(value, (Integral, sp.Integer)):
        raise TypeError("{} must be an exact integer, not bool/float/string".format(name))
    if value < minimum or value > maximum:
        raise ValueError("{} must lie in [{}, {}]".format(name, minimum, maximum))
    return int(value)


def _sign(value, name):
    result = _integer(value, name, -1, 1)
    if result == 0:
        raise ValueError("{} must be +1 or -1".format(name))
    return result


def _case_inputs(charge, flux, chirality):
    charge = _integer(charge, "charge", -MAX_ABS_INTEGER, MAX_ABS_INTEGER)
    flux = _integer(flux, "flux", -MAX_ABS_INTEGER, MAX_ABS_INTEGER)
    return charge, flux, _sign(chirality, "chirality")


def spinor_weights(parity=1):
    """Return the fixed sixteen weights; parity +1 means even minus signs.

    Opposite parity is the conjugate gauge representation, not an opposite
    six-dimensional Weyl chirality. Both traces agree through power four.
    """
    parity = _sign(parity, "parity")
    return tuple(
        tuple(sp.Rational(sign, 2) for sign in signs)
        for signs in product((-1, 1), repeat=5)
        if (-1)**signs.count(-1) == parity
    )


@lru_cache(maxsize=10)
def _trace_polynomial(power, parity):
    return sp.expand(sum(
        sum(weight * variable for weight, variable in zip(weights, CARTAN_SYMBOLS))**power
        for weights in spinor_weights(parity)
    ))


def trace_polynomial(power, parity=1):
    """Exact Cartesian weight-sum trace for an integer power from zero to four."""
    power = _integer(power, "power", 0, 4)
    parity = _sign(parity, "parity")
    return _trace_polynomial(power, parity)


@lru_cache(maxsize=1)
def _invariant_traces():
    """Extract invariant coefficients and prove the full Cartesian residual zero.

    The basis is fixed at degree <=4. Coefficients are extracted symbolically,
    never by sampling Cartan backgrounds, so no untested mixed monomial remains.
    """
    traces = tuple(trace_polynomial(power) for power in range(5))
    cartan_s2 = sum(variable**2 for variable in CARTAN_SYMBOLS)
    cartan_s4 = sum(variable**4 for variable in CARTAN_SYMBOLS)
    second = sp.Poly(traces[2], *CARTAN_SYMBOLS)
    fourth = sp.Poly(traces[4], *CARTAN_SYMBOLS)
    a = second.coeff_monomial(x1**2)
    b = fourth.coeff_monomial(x1**2 * x2**2) / sp.Integer(2)
    c = fourth.coeff_monomial(x1**4) - b
    residuals = (
        traces[1], traces[3], traces[2] - a * cartan_s2,
        traces[4] - b * cartan_s2**2 - c * cartan_s4,
    )
    if any(sp.expand(residual) != 0 for residual in residuals):
        raise ArithmeticError("fixed spinor trace invariant reduction failed")
    return traces[0], a * S2, b * S2**2 + c * S4


def _parent_contributions(chirality):
    dimension, trace2, trace4 = _invariant_traces()
    # All six degree-eight pieces of Ahat * ch16 * exp(Q X).
    terms = (
        trace4 / sp.factorial(4),
        trace2 * (Q * X)**2 / (sp.factorial(2) * sp.factorial(2)),
        dimension * (Q * X)**4 / sp.factorial(4),
        -p1 * trace2 / (sp.Integer(24) * sp.factorial(2)),
        -p1 * dimension * (Q * X)**2 / (sp.Integer(24) * sp.factorial(2)),
        dimension * (7 * p1**2 - 4 * p2) / sp.Integer(5760),
    )
    return {key: sp.expand(chirality * term) for key, term in zip(CONTRIBUTION_KEYS, terms)}


def parent_polynomial(chirality=1):
    """Full I8 in fixed invariant symbols, retaining the symbolic charge Q."""
    chirality = _sign(chirality, "chirality")
    return sp.expand(sum(_parent_contributions(chirality).values()))


def lower_polynomial(chirality=1):
    """Formal s*[Ahat(T4) ch16 exp(Q X4)]_6, before multiplication by q."""
    chirality = _sign(chirality, "chirality")
    dimension, trace2, _ = _invariant_traces()
    return sp.expand(chirality * (
        trace2 * Q * X4 / sp.factorial(2)
        + dimension * (Q * X4)**3 / sp.factorial(3)
        - p1 * dimension * Q * X4 / sp.Integer(24)
    ))


def sphere_pushforward(charge, flux, chirality=1):
    """Extract internal y coefficient of I8(Q, X4 + m*y), integral_S2 y=1.

    Expansion precedes internal dimension restriction. The fixed polynomial
    has at most y**4. Charge and flux are exact integers of magnitude <=10**6;
    their product q may reach 10**12. This cap is computational, not physical.
    No arbitrary symbolic expression or variable expansion degree is accepted.
    """
    charge, flux, chirality = _case_inputs(charge, flux, chirality)
    internal_y = sp.Dummy("internal_y")
    substituted = parent_polynomial(chirality).subs({Q: charge, X: X4 + flux * internal_y})
    expanded = sp.expand(substituted)
    return sp.expand(sp.Poly(expanded, internal_y).coeff_monomial(internal_y))


def _rational_record(value):
    numerator, denominator = value.as_numer_denom()
    return {"numerator": str(numerator), "denominator": str(denominator)}


def _polynomial_record(expression, generators=POLYNOMIAL_GENERATORS):
    polynomial = sp.Poly(expression, *generators, domain=sp.QQ)
    return {
        "expression": str(sp.expand(expression)),
        "generators": [str(symbol) for symbol in generators],
        "terms": [] if polynomial.is_zero else [
            {"powers": list(powers), "coefficient": _rational_record(coefficient)}
            for powers, coefficient in polynomial.terms()
        ],
    }


def _zero(expression):
    return sp.expand(expression) == 0


def case_report(charge, flux, chirality=1):
    """JSON-native exact report for one bounded (Q,m,s) background.

    Polynomial coefficients serialize as reduced numerator/denominator strings,
    never floating point. Gauge-component and family indices are distinct.
    """
    charge, flux, chirality = _case_inputs(charge, flux, chirality)
    q = charge * flux
    parent = parent_polynomial(chirality).subs(Q, charge)
    lower = lower_polynomial(chirality).subs(Q, charge)
    pushed = sphere_pushforward(charge, flux, chirality)
    opposite_parent = parent_polynomial(-chirality).subs(Q, charge)
    opposite_lower = lower_polynomial(-chirality).subs(Q, charge)
    opposite_push = sphere_pushforward(charge, flux, -chirality)
    p2_coefficient = sp.Poly(parent, p2).coeff_monomial(p2)
    return {
        "model_id": MODEL_ID,
        "parameters": {"charge": charge, "flux": flux, "q": q, "chirality": chirality},
        "parent": _polynomial_record(parent),
        "lower": _polynomial_record(lower),
        "pushforward": _polynomial_record(pushed),
        "expected_index_times_lower": _polynomial_record(q * lower),
        "contributions": {
            key: _polynomial_record(value.subs(Q, charge))
            for key, value in _parent_contributions(chirality).items()
        },
        "p2_coefficient": _rational_record(p2_coefficient),
        "bookkeeping": signed_mode_bookkeeping(
            q, gauge_rank=16, parent="weyl", parent_chirality=chirality,
        ),
        "controls": {
            "pushforward_matches_index_times_lower": _zero(pushed - q * lower),
            "parent_chirality_reversal": _zero(parent + opposite_parent),
            "lower_chirality_reversal": _zero(lower + opposite_lower),
            "pushforward_chirality_reversal": _zero(pushed + opposite_push),
            "parent_vectorlike_cancellation": _zero(parent + opposite_parent),
            "lower_vectorlike_cancellation": _zero(lower + opposite_lower),
            "pushforward_vectorlike_cancellation": _zero(pushed + opposite_push),
            "irreducible_p2_matches_dimension": p2_coefficient == -sp.Rational(chirality * 16, 1440),
            "parent_nonzero": not _zero(parent),
        },
        "scope": "Conditional 16*q gauge-component index versus q families; not a derivation of SM matter.",
    }


def demonstration_report():
    """Fixed exact controls, no fits, user expression parsing, or spectrum search."""
    parent = parent_polynomial()
    lower = lower_polynomial()
    cases = [case_report(1, flux) for flux in (0, 1, -1, 3)]
    cases.extend((case_report(3, 1), case_report(0, 1)))
    pure_gravity = parent.subs({S2: 0, S4: 0, X: 0})
    neutral_u1 = parent.subs(Q, 0)
    parity_equal = all(
        _zero(trace_polynomial(power, 1) - trace_polynomial(power, -1))
        for power in range(5)
    )
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "scope": "Local anomaly obstruction for one supplied complex six-dimensional Weyl Spin(10) 16.",
        "conventions": {
            "gauge_curvature": "F/(2*pi*i)",
            "charge_flux_index": "q=Q*m; m=integral_S2 X; charge Q remains distinct from q",
            "chirality": "s=+1 or -1; I8=s*[Ahat(T)*ch16(F)*exp(QX)]_8",
            "parity": "+1 is even minus signs in five half-integral weights; not spacetime chirality",
            "form_degrees": {str(symbol): degree for symbol, degree in FORM_DEGREES.items()},
            "pushforward": "X=X4+m*y, integral_S2 y=1; expand then extract coefficient of y",
        },
        "resource_policy": {
            "rank": 5, "weight_count": 16, "max_trace_power": 4,
            "max_form_degree": 8, "max_internal_y_power": 4,
            "max_abs_charge_or_flux": MAX_ABS_INTEGER,
            "max_abs_q": MAX_ABS_INTEGER**2,
            "cap_is_physical": False,
            "description": "Computational cap limits exact coefficient bit growth, not physical charge or flux.",
        },
        "symbolic": {
            "parent": _polynomial_record(parent),
            "lower": _polynomial_record(lower),
            "contributions": {
                key: _polynomial_record(value) for key, value in _parent_contributions(1).items()
            },
            "p2_coefficient": _rational_record(-sp.Rational(1, 90)),
            "pure_gravity": _polynomial_record(pure_gravity),
            "neutral_u1": _polynomial_record(neutral_u1),
            "traces": {
                str(power): _polynomial_record(trace_polynomial(power), CARTAN_SYMBOLS)
                for power in range(5)
            },
        },
        "cases": cases,
        "controls": {
            "parity_traces_equal_through_four": parity_equal,
            "full_gauge_conjugation_preserves_parent": parity_equal and _zero(parent.subs(Q, -Q) - parent),
            "full_gauge_conjugate_pair_does_not_cancel": parity_equal and not _zero(parent.subs(Q, -Q) + parent),
            "zero_base_u1_curvature_lower_vanishes": _zero(lower.subs(X4, 0)),
            "zero_base_u1_curvature_parent_remains_anomalous": not _zero(parent.subs(X, 0)),
            "same_q_different_charge_pushforwards_differ": not _zero(
                sphere_pushforward(3, 1) - sphere_pushforward(1, 3)
            ),
            "q_zero_both_pushforwards_zero": _zero(sphere_pushforward(0, 1)) and _zero(sphere_pushforward(1, 0)),
            "q_zero_both_parents_nonzero": not _zero(parent.subs(Q, 0)) and not _zero(parent.subs(Q, 1)),
            "q_zero_different_charge_parents_differ": not _zero(parent.subs(Q, 0) - parent.subs(Q, 1)),
            "neutral_u1_retains_p2": sp.Poly(neutral_u1, p2).coeff_monomial(p2) == -sp.Rational(1, 90),
            "pure_gravity_retains_p2": sp.Poly(pure_gravity, p2).coeff_monomial(p2) == -sp.Rational(1, 90),
        },
        "limitations": [
            "The complex Weyl 16 and its six-dimensional chirality are supplied, not derived from BPR.",
            "The nonzero p2 coefficient is a conditional irreducible local gravitational anomaly witness.",
            "No reality condition halves this anomaly; no cancellation fields or Green-Schwarz completion are added.",
            "Sphere reduction assumes a product tangent bundle and pulled-back Spin(10) gauge curvature.",
            "Anomaly forms are formal descent polynomials, not ordinary nonzero six-forms on a four-dimensional base without extension.",
            "Four-dimensional Standard Model anomaly cancellation is not parent anomaly cancellation or a derivation of SM matter.",
            "Equal gauge-parity traces through degree four do not imply opposite-spacetime-chirality cancellation.",
            "Neither global anomaly cancellation nor a universal exclusion of future BPR parents is established.",
        ],
    }
