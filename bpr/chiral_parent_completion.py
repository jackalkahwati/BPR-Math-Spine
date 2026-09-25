"""Exact local-anomaly completion search for the supplied 6D Spin(10) parent.

See doc/derivations/chiral_parent_completion_2026-09-25.md. The added fields
and the Green-Schwarz 2-form are new model inputs, not derived from BPR. Only
local anomaly polynomials are treated; global anomalies, coupling
quantization, flux stabilization and dynamics are not established.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product

import sympy as sp

MODEL_ID = "supplied-6d-spin10-parent-minimal-completion-v1"
# Computational enumeration caps, not physical bounds.
MAX_CHARGE = 4
MAX_EXTRA_COMPONENTS = 40
MAX_ABS_FLUX = 16

S2, S4, X, p1, p2, X4 = sp.symbols("S2 S4 X p1 p2 X4")
_CARTAN = sp.symbols("x1:6")
REPRESENTATIONS = ("1", "10", "16", "16bar")
CONJUGATE = {"1": "1", "10": "10", "16": "16bar", "16bar": "16"}
# Z4 center of Spin(10): acts as i^c on the representation.
CENTER_CHARGE = {"1": 0, "10": 2, "16": 1, "16bar": 3}
# SU(2)_L doublets inside each representation (SU(5) x U(1) branching).
SU2_DOUBLETS = {"1": 0, "10": 2, "16": 4, "16bar": 4}

LIMITATIONS = [
    "The parent, the added 10 and singlets, and the 2-form are supplied inputs, not BPR derivations.",
    "Only local anomaly polynomials are checked; global anomalies and coupling quantization are not.",
    "The flux m=3 (three families) is chosen, not derived.",
    "Forty-eight massless sterile singlets remain and need masses or decoupling.",
    "No symmetry breaking, Yukawa couplings or substrate realization is supplied.",
]


def _weights(rep):
    if rep == "1":
        return [(0,) * 5]
    if rep == "10":
        out = []
        for i in range(5):
            for sign in (1, -1):
                w = [0] * 5
                w[i] = sign
                out.append(tuple(w))
        return out
    parity = 1 if rep == "16" else -1
    return [tuple(Fraction(s, 2) for s in signs) for signs in product((-1, 1), repeat=5)
            if (-1) ** signs.count(-1) == parity]


@lru_cache(maxsize=None)
def representation_traces(rep):
    """Exact (dim, t1, t2, t3, a4, b4): tr F^k from weights, reduced to S2, S4."""
    if rep not in REPRESENTATIONS:
        raise ValueError("unsupported representation")
    weights = _weights(rep)
    powers = [sp.expand(sum(sum(sp.Rational(w.numerator, w.denominator) * x if isinstance(w, Fraction)
                                    else w * x for w, x in zip(weight, _CARTAN)) ** k
                                for weight in weights)) for k in range(5)]
    s2 = sum(x ** 2 for x in _CARTAN)
    s4 = sum(x ** 4 for x in _CARTAN)
    second = sp.Poly(powers[2], *_CARTAN)
    fourth = sp.Poly(powers[4], *_CARTAN)
    t2 = second.coeff_monomial(_CARTAN[0] ** 2)
    a4 = fourth.coeff_monomial(_CARTAN[0] ** 2 * _CARTAN[1] ** 2) / 2
    b4 = fourth.coeff_monomial(_CARTAN[0] ** 4) - a4
    residual2 = sp.expand(powers[2] - t2 * s2)
    residual4 = sp.expand(powers[4] - a4 * s2 ** 2 - b4 * s4)
    if residual2 != 0 or residual4 != 0 or sp.expand(powers[1]) != 0 or sp.expand(powers[3]) != 0:
        raise ArithmeticError("trace reduction failed")
    return (len(weights), 0, Fraction(int(t2.p), int(t2.q)), 0,
            Fraction(int(a4.p), int(a4.q)), Fraction(int(b4.p), int(b4.q)))


def _field(field):
    if type(field) is not tuple or len(field) != 3:
        raise TypeError("a field is a (chirality, representation, charge) tuple")
    s, rep, charge = field
    if type(s) is not int or s not in (1, -1):
        raise ValueError("chirality must be +1 or -1")
    if rep not in REPRESENTATIONS:
        raise ValueError("unsupported representation")
    if type(charge) is not int or abs(charge) > 10 ** 6:
        raise ValueError("charge must be a bounded builtin int")
    return s, rep, charge


def field_polynomial(field):
    """I8 = s [Ahat ch_R exp(QX)]_8 in the invariant symbols."""
    s, rep, Q = _field(field)
    dim, _, t2, _, a4, b4 = representation_traces(rep)
    t2, a4, b4 = sp.Rational(t2.numerator, t2.denominator), sp.Rational(a4.numerator, a4.denominator), \
        sp.Rational(b4.numerator, b4.denominator)
    trace2 = t2 * S2
    trace4 = a4 * S2 ** 2 + b4 * S4
    body = (trace4 / 24 + (Q * X) ** 2 * trace2 / 4 + dim * (Q * X) ** 4 / 24
            - p1 * trace2 / 48 - p1 * dim * (Q * X) ** 2 / 48
            + dim * (7 * p1 ** 2 - 4 * p2) / 5760)
    return sp.expand(s * body)


def total_polynomial(fields):
    return sp.expand(sum((field_polynomial(f) for f in fields), sp.Integer(0)))


def anomaly_coefficients(fields):
    """Exact coefficients of I8 in u=S2, v=X^2, w=p1 plus the irreducible S4, p2."""
    c = {key: Fraction(0) for key in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "S4", "p2")}
    for field in fields:
        s, rep, Q = _field(field)
        dim, _, t2, _, a4, b4 = representation_traces(rep)
        c["alpha"] += s * a4 / 24
        c["beta"] += s * t2 * Q * Q / 4
        c["gamma"] += Fraction(s * dim * Q ** 4, 24)
        c["delta"] += -s * t2 / 48
        c["epsilon"] += Fraction(-s * dim * Q * Q, 48)
        c["zeta"] += Fraction(7 * s * dim, 5760)
        c["S4"] += s * b4 / 24
        c["p2"] += Fraction(-4 * s * dim, 5760)
    return c


def _is_rational_square(value):
    if value < 0:
        return False
    return bool(sp.sqrt(sp.Rational(value.numerator, value.denominator)).is_rational)


def factorization_status(coefficients):
    """Lemma 2: exact criterion for single-2-form (rank <= 2 product) cancellation."""
    c = coefficients
    if c["S4"] != 0:
        return {"status": "irreducible_S4", "factors": None}
    if c["p2"] != 0:
        return {"status": "irreducible_p2", "factors": None}
    if c["zeta"] != 0:
        raise ArithmeticError("p1^2 must vanish with p2")
    alpha, beta, gamma, delta, epsilon = (c[k] for k in ("alpha", "beta", "gamma", "delta", "epsilon"))
    if delta == 0 and epsilon == 0:
        disc = beta * beta - 4 * alpha * gamma
        if not _is_rational_square(disc):
            return {"status": "not_factorizable_rank3_or_irrational", "factors": None}
        if alpha != 0:
            root = Fraction(str(sp.sqrt(sp.Rational(disc.numerator, disc.denominator))))
            r1 = (-beta + root) / (2 * alpha)
            r2 = (-beta - root) / (2 * alpha)
            left, right = (1, -r1, 0), (alpha, -alpha * r2, 0)
        elif gamma != 0:
            left, right = (0, 1, 0), (beta, gamma, 0)
        else:
            left, right = (1, 0, 0), (0, beta, 0)
        return {"status": "factorizable",
                "factors": [[_rational(v) for v in left], [_rational(v) for v in right]]}
    kernel_value = alpha * epsilon * epsilon - beta * delta * epsilon + gamma * delta * delta
    if kernel_value != 0:
        return {"status": "not_factorizable_rank3_or_irrational", "factors": None,
                "kernel_value": _rational(kernel_value)}
    # q = l*m with l = delta u + epsilon v and m = a u + b v; then I8 = l (m + w).
    if delta != 0 and epsilon != 0:
        a, b = alpha / delta, gamma / epsilon
    elif delta != 0:
        a, b = alpha / delta, beta / delta
    else:
        a, b = beta / epsilon, gamma / epsilon
    left = (delta, epsilon, Fraction(0))
    right = (a, b, Fraction(1))
    return {"status": "factorizable",
            "factors": [[_rational(v) for v in left], [_rational(v) for v in right]]}


def _rational(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _from_rational(record):
    return sp.Rational(record["numerator"], record["denominator"])


def factor_expression(factors):
    """Rebuild the product of the two linear forms in S2, X^2, p1."""
    exprs = []
    for coeffs in factors:
        a, b, cw = (_from_rational(r) for r in coeffs)
        exprs.append(a * S2 + b * X ** 2 + cw * p1)
    return sp.expand(exprs[0] * exprs[1])


# ---------------------------------------------------------------------------
# Four-dimensional descendants
# ---------------------------------------------------------------------------

def zero_modes(fields, flux):
    """Left-handed 4D zero modes on S^2 with U(1)_X flux m (index theorem)."""
    if type(flux) is not int or flux == 0 or abs(flux) > MAX_ABS_FLUX:
        raise ValueError("flux must be a nonzero builtin int of bounded size")
    counts = {}
    for field in fields:
        s, rep, Q = _field(field)
        index = Q * flux
        if index == 0:
            continue
        if s * (1 if index > 0 else -1) > 0:
            key = (rep, Q)
        else:
            key = (CONJUGATE[rep], -Q)
        counts[key] = counts.get(key, 0) + abs(index)
    return [{"representation": rep, "charge": Q, "multiplicity": mult}
            for (rep, Q), mult in sorted(counts.items())]


def four_d_polynomial(modes):
    """Sum of [Ahat ch_R exp(Q X4)]_6 over left-handed zero modes."""
    total = sp.Integer(0)
    for mode in modes:
        dim, _, t2, _, _, _ = representation_traces(mode["representation"])
        t2 = sp.Rational(t2.numerator, t2.denominator)
        Q = mode["charge"]
        total += mode["multiplicity"] * (t2 * S2 * Q * X4 / 2 + dim * (Q * X4) ** 3 / 6
                                         - p1 * dim * Q * X4 / 24)
    return sp.expand(total)


def pushforward(fields, flux):
    """Coefficient of the internal generator y after X -> X4 + m y."""
    if type(flux) is not int or flux == 0 or abs(flux) > MAX_ABS_FLUX:
        raise ValueError("flux must be a nonzero builtin int of bounded size")
    y = sp.Dummy("y")
    expanded = sp.expand(total_polynomial(fields).subs(X, X4 + flux * y))
    return sp.expand(sp.Poly(expanded, y).coeff_monomial(y))


def four_d_ledger(modes):
    """Local 4D anomalies of left-handed Weyl content plus two global counters."""
    mixed = Fraction(0)
    cubic = 0
    gravity = 0
    spin10_cubic = 0
    doublets = 0
    z16 = 0
    z16_applicable = True
    sm_embedded = 0
    for mode in modes:
        dim, _, t2, t3, _, _ = representation_traces(mode["representation"])
        Q, mult = mode["charge"], mode["multiplicity"]
        mixed += mult * Q * t2
        cubic += mult * dim * Q ** 3
        gravity += mult * dim * Q
        spin10_cubic += mult * t3
        doublets += mult * SU2_DOUBLETS[mode["representation"]]
        center = CENTER_CHARGE[mode["representation"]]
        if center % 2 == 0:
            z16_applicable = False
        else:
            contribution = mult * dim * (1 if center == 1 else -1)
            z16 += contribution
            sm_embedded += contribution
    return {
        "spin10_cubic": spin10_cubic,
        "spin10_squared_u1": _rational(mixed),
        "u1_cubic": cubic,
        "gravity_u1": gravity,
        "su2_doublets": doublets,
        "witten_parity_even": doublets % 2 == 0,
        "z16_spin10_charged_count_mod16": sm_embedded % 16,
        "z16_full_spectrum_applicable": z16_applicable,
    }


def standard_model_z16(generations, right_handed_neutrinos):
    """Control: Weyl count with Z4 charge X=5(B-L)-4Y mod 4 (all SM fields have X=1 mod 4)."""
    if type(generations) is not int or type(right_handed_neutrinos) is not int:
        raise TypeError("counts must be builtin ints")
    if generations < 0 or right_handed_neutrinos < 0 or generations > 64 or right_handed_neutrinos > 64:
        raise ValueError("counts outside supported range")
    # (B-L, Y, multiplicity) for left-handed Weyl fields of one generation.
    fields = [(Fraction(1, 3), Fraction(1, 6), 6), (Fraction(-1, 3), Fraction(-2, 3), 3),
              (Fraction(-1, 3), Fraction(1, 3), 3), (Fraction(-1), Fraction(-1, 2), 2),
              (Fraction(1), Fraction(1), 1)]
    count = 0
    for bl, y, mult in fields:
        x = 5 * bl - 4 * y
        if x.denominator != 1 or x.numerator % 4 != 1:
            raise ArithmeticError("unexpected Z4 charge")
        count += mult
    nu_x = 5 * Fraction(1) - 4 * Fraction(0)
    if nu_x.numerator % 4 != 1:
        raise ArithmeticError("unexpected Z4 charge")
    total = generations * count + right_handed_neutrinos
    return {"per_generation_without_nu": count, "total_mod16": total % 16,
            "anomaly_free": total % 16 == 0}


# ---------------------------------------------------------------------------
# Exhaustive minimal-class search
# ---------------------------------------------------------------------------

PARENT = (1, "16", 1)


def _multisets(count, values):
    """All multisets of the given size over the value list, as count vectors."""
    if len(values) == 1:
        yield (count,)
        return
    for first in range(count, -1, -1):
        for rest in _multisets(count - first, values[1:]):
            yield (first,) + rest


def completion_search(q_max, extra_components=36):
    """All completions in class C(q_max) with the given number of added Weyl components."""
    if type(q_max) is not int or not 0 <= q_max <= MAX_CHARGE:
        raise ValueError("q_max outside the enumeration cap")
    if type(extra_components) is not int or not 0 <= extra_components <= MAX_EXTRA_COMPONENTS:
        raise ValueError("extra_components outside the enumeration cap")
    values = list(range(q_max + 1))
    solutions = []
    examined = 0
    for tens_minus in range(extra_components // 20 + 1):
        tens_plus = tens_minus + 1  # Lemma 1, (C1)
        singlets = extra_components - 10 * (tens_plus + tens_minus)
        if singlets < 0 or (singlets + 26) % 2:
            continue
        minus = (singlets + 26) // 2  # Lemma 1, (C2)
        plus = singlets - minus
        if plus < 0:
            continue
        for neg in _multisets(minus, values):
            for pos in _multisets(plus, values):
                examined += 1
                fields = [PARENT] + [(1, "10", 0)] * tens_plus + [(-1, "10", 0)] * tens_minus
                for q, k in zip(values, neg):
                    fields += [(-1, "1", q)] * k
                for q, k in zip(values, pos):
                    fields += [(1, "1", q)] * k
                status = factorization_status(anomaly_coefficients(fields))
                if status["status"] == "factorizable":
                    solutions.append({
                        "tens_plus": tens_plus, "tens_minus": tens_minus,
                        "negative_singlet_charge_counts": {str(q): k for q, k in zip(values, neg) if k},
                        "positive_singlet_charge_counts": {str(q): k for q, k in zip(values, pos) if k},
                        "factors": status["factors"], "fields": [list(f) for f in fields],
                    })
    return {"q_max": q_max, "extra_components": extra_components, "examined": examined,
            "solutions": solutions}


def completion_report(fields, flux=3):
    coefficients = anomaly_coefficients(fields)
    status = factorization_status(coefficients)
    total = total_polynomial(fields)
    modes = zero_modes(fields, flux)
    pushed = pushforward(fields, flux)
    four_d = four_d_polynomial(modes)
    factored = None
    if status["factors"] is not None:
        factored = sp.expand(total - factor_expression(status["factors"])) == 0
    return {
        "fields": [list(f) for f in fields],
        "flux": flux,
        "I8": str(total),
        "I8_factored_sympy": str(sp.factor(total)),
        "status": status["status"],
        "factors": status["factors"],
        "factor_product_matches_I8": factored,
        "zero_modes": modes,
        "pushforward": str(pushed),
        "pushforward_matches_zero_modes": sp.expand(pushed - four_d) == 0,
        "four_d_ledger": four_d_ledger(modes),
    }


def minimal_completion():
    return [PARENT, (1, "10", 0)] + [(-1, "1", 1)] * 16 + [(-1, "1", 0)] * 10


def demonstration_report():
    parent_only = completion_report([PARENT])
    minimal = completion_report(minimal_completion())
    searches = [completion_search(q) for q in (1, 2, 3)]
    for search in searches:
        for solution in search["solutions"]:
            fields = [tuple(f) for f in solution["fields"]]
            modes = zero_modes(fields, 3)
            ledger = four_d_ledger(modes)
            solution["four_d_abelian_anomaly_free"] = ledger["u1_cubic"] == 0 and ledger["gravity_u1"] == 0
            del solution["fields"]
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "local_anomaly_completion_demonstrator",
        "empirical_validation": False,
        "derived_from_bpr": False,
        "parent_only": {key: parent_only[key] for key in ("I8", "status", "zero_modes", "four_d_ledger")},
        "minimal_completion": minimal,
        "searches": searches,
        "standard_model_z16_controls": {
            "3_generations_0_nu": standard_model_z16(3, 0),
            "3_generations_3_nu": standard_model_z16(3, 3),
            "3_generations_19_nu": standard_model_z16(3, 19),
        },
        "limitations": list(LIMITATIONS),
    }
