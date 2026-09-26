"""Dirac quantization of the Green-Schwarz couplings of the BPR-6D matter sector.

See doc/derivations/green_schwarz_quantization_2026-09-26.md. The local anomaly
polynomial of chiral_parent_completion is rewritten in integral generators

    lambda_V = S2 / 2   (instanton number of the Spin(10) bundle; S2 = p1(V_10)),
    x^2               (x = c1 of the unit-U(1)_F-charge line bundle),
    lambda_T = p1 / 2   (integral on spin manifolds),

and tested against the Green-Schwarz form I8 = (1/2) <Y, Y> with Y = b_V lambda_V
+ b_X x^2 + (a/2) lambda_T, b in a unimodular string-charge lattice and a
characteristic. Exact rational arithmetic throughout. This is the standard
characteristic-class ("naive") quantization test; torsion refinements and
Omega_7 bordism are not computed.
"""

from fractions import Fraction
from itertools import product

import sympy as sp

try:
    from .chiral_parent_completion import anomaly_coefficients, zero_modes
except ImportError:  # loaded as a top-level module by the demo script
    from chiral_parent_completion import anomaly_coefficients, zero_modes

MODEL_ID = "bpr6d-green-schwarz-quantization-v1"
lamV, x2, lamT = sp.symbols("lambda_V x2 lambda_T")

LIMITATIONS = [
    "Characteristic-class quantization only: torsion refinements and Omega_7 spin bordism are not computed.",
    "Backgrounds are spin manifolds with Spin(10) x U(1)_F bundles; the Wu-class shift is taken in lambda_T only.",
    "Charge normalization: the unit of U(1)_F charge must be carried by some state (completeness); here only massive states can carry it.",
    "The Green-Schwarz sector is non-chiral 2-forms (signature (n, n)); chiral tensors are outside the class.",
    "A multiple-of-three family number is derived for the minimal completion only; near-minimal completions behave differently.",
]


def minimal_fields(parent_charge):
    """16_+(q) + 16_-(0): the unique minimal local completion, with the parent at charge q."""
    if type(parent_charge) is not int or parent_charge < 1:
        raise ValueError("parent_charge must be a positive int")
    return [(1, "16", parent_charge), (-1, "16", 0)]


def integral_coefficients(fields):
    """Coefficients of I8 in (lambda_V, x^2, lambda_T): S2 = 2 lambda_V, X^2 = x^2, p1 = 2 lambda_T."""
    c = anomaly_coefficients(fields)
    if c["S4"] != 0 or c["p2"] != 0:
        raise ValueError("irreducible S4 or p2 terms: no Green-Schwarz factorization")
    return {"VV": 4 * c["alpha"], "VX": 2 * c["beta"], "XX": c["gamma"],
            "VT": 4 * c["delta"], "XT": 2 * c["epsilon"], "TT": 4 * c["zeta"]}


def integral_polynomial(fields):
    k = integral_coefficients(fields)
    r = lambda value: sp.Rational(value.numerator, value.denominator)  # noqa: E731
    return sp.expand(r(k["VV"]) * lamV ** 2 + r(k["VX"]) * lamV * x2 + r(k["XX"]) * x2 ** 2
                     + r(k["VT"]) * lamV * lamT + r(k["XT"]) * x2 * lamT + r(k["TT"]) * lamT ** 2)


def gram_requirements(fields):
    """Inner products that (1/2)<Y,Y> = I8 forces on (b_V, b_X, a)."""
    k = integral_coefficients(fields)
    return {"bV.bV": 2 * k["VV"], "bX.bX": 2 * k["XX"], "a.a": 8 * k["TT"],
            "bV.bX": k["VX"], "bV.a": 2 * k["VT"], "bX.a": 2 * k["XT"]}


def necessary_conditions(fields):
    """Necessary for ANY unimodular lattice and any number of non-chiral 2-forms.

    Every Gram entry is an integer; characteristic a gives b.b = b.a (mod 2); non-chiral 2-forms have
    signature zero, so a.a = 0 (mod 8).
    """
    g = gram_requirements(fields)
    integral = {key: value.denominator == 1 for key, value in g.items()}
    all_integral = all(integral.values())
    parity_V = all_integral and (g["bV.bV"] - g["bV.a"]) % 2 == 0
    parity_X = all_integral and (g["bX.bX"] - g["bX.a"]) % 2 == 0
    signature = all_integral and g["a.a"] % 8 == 0
    return {"gram": {key: str(value) for key, value in g.items()}, "integral": integral,
            "parity_V": bool(parity_V), "parity_X": bool(parity_X), "signature_mod_8": bool(signature),
            "pass": bool(all_integral and parity_V and parity_X and signature)}


def _linear_factors(poly):
    """I8 = K * L1 * L2 with L1, L2 primitive integral linear forms, or None."""
    if poly == 0:
        return sp.Integer(0), None, None
    content, factors = sp.factor_list(poly, lamV, x2, lamT)
    linear = []
    for fac, mult in factors:
        if sp.Poly(fac, lamV, x2, lamT).total_degree() != 1:
            return None
        linear += [fac] * mult
    if len(linear) != 2:
        return None
    return sp.nsimplify(content), linear[0], linear[1]


def hyperbolic_solution(fields):
    """One non-chiral 2-form (even lattice U): I8 = Y_e Y_g with integral Y_e, Y_g.

    By Gauss's lemma this exists iff the content K of I8 = K L1 L2 (L primitive) is an integer.
    """
    poly = integral_polynomial(fields)
    fac = _linear_factors(poly)
    if fac is None:
        return {"exists": False, "reason": "not a product of two linear forms"}
    K, L1, L2 = fac
    if K == 0:
        return {"exists": True, "Y_e": "0", "Y_g": "0", "reason": "I8 vanishes"}
    if not K.is_integer:
        return {"exists": False, "reason": "content {} is not an integer".format(K)}
    Y_e, Y_g = sp.expand(K * L1), sp.expand(L2)
    if sp.expand(Y_e * Y_g - poly) != 0:
        raise ArithmeticError("factorization check failed")
    return {"exists": True, "Y_e": str(Y_e), "Y_g": str(Y_g)}


def _coeff_vector(expr):
    p = sp.Poly(expr, lamV, x2, lamT)
    return tuple(sp.nsimplify(p.coeff_monomial(mono)) for mono in (lamV, x2, lamT))


def odd_lattice_solution(fields, bound=24):
    """One self-dual plus one anti-self-dual tensor, lattice I_{1,1} = diag(1, -1), a = (odd, odd).

    Y_i = b_i + (a_i/2) lambda_T and I8 = (Y_1^2 - Y_2^2)/2 = P R / 2 with P = Y_1 - Y_2, R = Y_1 + Y_2:
    P, R integral, equal parity in lambda_V and x^2, opposite parity in lambda_T. Enumerates the
    divisor splittings of 2K over the rational factorization.
    """
    poly = integral_polynomial(fields)
    fac = _linear_factors(poly)
    if fac is None or fac[0] == 0:
        return {"exists": fac is not None, "reason": "no factorization" if fac is None else "I8 vanishes"}
    K, L1, L2 = fac
    target = 2 * K
    if not target.is_integer:
        return {"exists": False, "reason": "2K = {} not an integer".format(target)}
    t = int(target)
    for k1 in range(1, min(abs(t), bound) + 1):
        if t % k1:
            continue
        for sign in (1, -1):
            for A, Bf in ((L1, L2), (L2, L1)):
                P, R = sp.expand(sign * k1 * A), sp.expand(sign * (t // k1) * Bf)
                p, q = _coeff_vector(P), _coeff_vector(R)
                if (p[0] - q[0]) % 2 == 0 and (p[1] - q[1]) % 2 == 0 and (p[2] - q[2]) % 2 == 1:
                    return {"exists": True, "P": str(P), "R": str(R)}
    return {"exists": False, "reason": "parity obstruction for every divisor splitting"}


def lattice_vectors_in_U(solution):
    """Explicit (b_V, b_X, a) in U = <e, g>, e.g = 1, for Y = Y_e e + Y_g g, and their Gram data."""
    ce, cg = _coeff_vector(sp.sympify(solution["Y_e"])), _coeff_vector(sp.sympify(solution["Y_g"]))
    bV, bX, c = (ce[0], cg[0]), (ce[1], cg[1]), (ce[2], cg[2])
    dot = lambda u, w: u[0] * w[1] + u[1] * w[0]  # noqa: E731
    a = (2 * c[0], 2 * c[1])
    return {"b_V": [int(v) for v in bV], "b_X": [int(v) for v in bX], "a": [int(v) for v in a],
            "gram": {"bV.bV": int(dot(bV, bV)), "bX.bX": int(dot(bX, bX)), "a.a": int(dot(a, a)),
                     "bV.bX": int(dot(bV, bX)), "bV.a": int(dot(bV, a)), "bX.a": int(dot(bX, a))}}


def minimal_completion_scan(q_max=12):
    rows = []
    for q in range(1, q_max + 1):
        fields = minimal_fields(q)
        nec = necessary_conditions(fields)
        hyp = hyperbolic_solution(fields)
        odd = odd_lattice_solution(fields)
        families = sum(md["multiplicity"] for md in zero_modes(fields, 1))
        rows.append({"parent_charge": q, "necessary": nec["pass"], "U": hyp["exists"], "I11": odd["exists"],
                     "families_at_unit_flux": families})
    return rows


def near_minimal_scan(parent_charges=(1, 2, 3), singlet_max=4):
    """18-component completions 16_+(q) + 16_-(0) + 1_+(Qa) + 1_-(Qb): which pass, and extra massless content."""
    rows = []
    for q in parent_charges:
        for qa, qb in product(range(singlet_max + 1), repeat=2):
            fields = minimal_fields(q) + [(1, "1", qa), (-1, "1", qb)]
            nec = necessary_conditions(fields)
            hyp = hyperbolic_solution(fields)
            modes = zero_modes(fields, 1)
            singlets = sum(md["multiplicity"] for md in modes if md["representation"] == "1")
            rows.append({"parent_charge": q, "Qa": qa, "Qb": qb, "necessary": nec["pass"], "U": hyp["exists"],
                         "massless_singlets_per_unit_flux": singlets})
    return rows


def family_number_statement():
    """Minimal completion: Dirac quantization needs 3 | q; the index gives n_gen = q |m|; so 3 | n_gen."""
    rows = minimal_completion_scan(12)
    passing = [row["parent_charge"] for row in rows if row["necessary"]]
    return {"passing_parent_charges": passing,
            "all_multiples_of_three": all(q % 3 == 0 for q in passing),
            "every_multiple_of_three_passes_U": all(row["U"] for row in rows if row["parent_charge"] % 3 == 0),
            "family_numbers_unit_flux": sorted({row["families_at_unit_flux"] for row in rows
                                                if row["parent_charge"] in passing}),
            "minimal_family_number": min(row["families_at_unit_flux"] for row in rows
                                         if row["parent_charge"] in passing)}


def unit_flux_landscape():
    """With the parent at charge 3, three families = one flux quantum. Tune Lambda for it:
    fluxes m >= 2 have no vacuum (m^2 < 4/3), and m = 0 has only a maximum (no flux stabilization)."""
    try:
        from .six_dim_flux_vacuum import flux_landscape, reduced_potential, M, e, Lam, r0, r, m
    except ImportError:
        from six_dim_flux_vacuum import flux_landscape, reduced_potential, M, e, Lam, r0, r, m
    land = flux_landscape(1, (1, 2, 3))
    V0 = reduced_potential().subs({M: 1, e: 1, Lam: 2, r0: 1, m: 0})
    crit = [c for c in sp.solve(sp.diff(V0, r), r) if c.is_positive]
    second = [sp.simplify(sp.diff(V0, r, 2).subs(r, c)) for c in crit]
    return {"sectors": land["sectors"],
            "zero_flux_stationary_points": [str(c) for c in crit],
            "zero_flux_second_derivative": [str(v) for v in second],
            "zero_flux_has_minimum": any(v > 0 for v in second)}


def demonstration_report():
    minimal_q1 = minimal_fields(1)
    minimal_q3 = minimal_fields(3)
    hyp3 = hyperbolic_solution(minimal_q3)
    landscape = unit_flux_landscape()
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "characteristic_class_quantization_three_divides_family_number",
        "empirical_validation": False,
        "integral_polynomial_q1": str(integral_polynomial(minimal_q1)),
        "necessary_q1": necessary_conditions(minimal_q1),
        "hyperbolic_q1": hyperbolic_solution(minimal_q1),
        "odd_lattice_q1": odd_lattice_solution(minimal_q1),
        "integral_polynomial_q3": str(integral_polynomial(minimal_q3)),
        "necessary_q3": necessary_conditions(minimal_q3),
        "hyperbolic_q3": hyp3,
        "lattice_vectors_q3": lattice_vectors_in_U(hyp3),
        "odd_lattice_q3": odd_lattice_solution(minimal_q3),
        "minimal_scan": minimal_completion_scan(12),
        "family_number": family_number_statement(),
        "near_minimal_passing": [row for row in near_minimal_scan() if row["U"]],
        "unit_flux_landscape": {"sectors": landscape["sectors"],
                                "zero_flux_has_minimum": landscape["zero_flux_has_minimum"]},
        "limitations": list(LIMITATIONS),
    }
