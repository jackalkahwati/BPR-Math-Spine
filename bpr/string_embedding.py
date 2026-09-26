"""String-embedding checks for BPR-6D: the Morrison-Taylor tension and 6D (1,0) analogues.

See doc/derivations/string_embedding_2026-09-26.md. Two computations:

1. Morrison-Taylor quantities for BPR-6D itself: -a.b~ in the Park-Taylor
   normalization (b~ = 2 b_X, anchored in green_schwarz_quantization) and the gcd
   of the massless U(1)_F charges.
2. Supersymmetric analogues: 6D N=(1,0) supergravity with G = SO(10) x U(1)
   on T = 0 (lattice Z, a = -3), and a pure U(1) on T = 1 (lattice U,
   a = (-2, -2)), the counterpart of BPR-6D's one non-chiral 2-form. The standard
   Kumar-Morrison-Taylor / Park-Taylor anomaly equations are validated on SU(2),
   where they must reproduce the genus g = (b-1)(b-2)/2 of a degree-b plane
   curve and 22 fundamentals for b = 1.

Exact integer / rational arithmetic throughout.
"""

from fractions import Fraction
from math import gcd
from functools import reduce

try:
    from .green_schwarz_quantization import abelian_gram, minimal_fields, necessary_conditions, integral_polynomial
    from .chiral_parent_completion import zero_modes
except ImportError:  # loaded as a top-level module by the demo script
    from green_schwarz_quantization import abelian_gram, minimal_fields, necessary_conditions, integral_polynomial
    from chiral_parent_completion import zero_modes

MODEL_ID = "bpr6d-string-embedding-v2"

LIMITATIONS = [
    "The Morrison-Taylor massless-charge statement concerns supersymmetric F-theory models; BPR-6D is non-supersymmetric, and its sign of -a.b~ depends on which chirality is called the hyperino.",
    "The full Morrison-Taylor text was not accessible here; its exact positivity condition and the height-pairing normalization of b~ are assumed from the abstract.",
    "The supersymmetric analysis is anomaly cancellation only (T = 0, and a pure U(1) on T = 1); F-theory realizability of the example spectra is not established.",
    "A U(1) monopole background in 6D (1,0) supergravity is not a supersymmetric vacuum unless the U(1) is the R-symmetry; vacuum existence is not analysed.",
]

# SO(10) group theory in the Kumar-Morrison-Taylor normalization (tr = tr_10, lambda = 2):
# tr_R F^2 = A_R tr F^2, tr_R F^4 = B_R tr F^4 + C_R (tr F^2)^2.
SO10 = {"lambda": 2,
        "adj": {"dim": 45, "A": 8, "B": 2, "C": 3},
        "10": {"dim": 10, "A": 1, "B": 1, "C": 0},
        "16": {"dim": 16, "A": 2, "B": -1, "C": Fraction(3, 4)}}
# SU(2): tr = tr_2, lambda = 1; tr F^4 = (1/2)(tr F^2)^2 has no independent quartic.
SU2 = {"lambda": 1,
       "adj": {"dim": 3, "A": 4, "C": 8},
       "2": {"dim": 2, "A": 1, "C": Fraction(1, 2)}}


# ---------------------------------------------------------------------------
# 1. Morrison-Taylor quantities for BPR-6D
# ---------------------------------------------------------------------------

def mt_quantities(fields):
    """-a.b~ with b~ = 2 b_X (height pairing), and the gcd of all 6D charges in the spectrum.

    BPR-6D has no gravitino, so which chirality plays the hyperino role is a convention (the lattice U is
    isomorphic to -U). With the charged 16_+ as the hyperino, -a.b~ = 24 > 0; swapping the labels flips the sign.
    In supersymmetric theories -a.b~ = (1/6) sum_hypers q^2 >= 0 identically.
    """
    gram = abelian_gram(fields)
    minus_a_btilde = -2 * gram["bX.a"]
    charges = [abs(f[2]) for f in fields if f[2] != 0]
    g = reduce(gcd, charges) if charges else 0
    return {"minus_a_dot_btilde": minus_a_btilde, "charge_gcd": g,
            "premise_holds_in_hyperino_convention": minus_a_btilde > 0,
            "massless_charges_generate_lattice": g == 1}


def vectorlike_charge_one_extension(q=3, pairs=1):
    """BPR-6D plus massless vector-like 6D singlet pairs 1_+(1) + 1_-(1): gcd 1, I8 unchanged, families unchanged."""
    base = minimal_fields(q)
    extended = base + [(1, "1", 1), (-1, "1", 1)] * pairs
    same_I8 = integral_polynomial(base) == integral_polynomial(extended)
    modes = zero_modes(extended, 1)
    families = sum(md["multiplicity"] for md in modes if md["representation"] == "16")
    singlets = [md for md in modes if md["representation"] == "1"]
    return {"I8_unchanged": bool(same_I8), "quantization_pass": necessary_conditions(extended)["pass"],
            "families_at_unit_flux": families,
            "singlet_zero_modes": singlets,
            "net_singlet_chirality": sum(md["multiplicity"] * (1 if md["charge"] > 0 else -1) for md in singlets),
            "mt": mt_quantities(extended)}


# ---------------------------------------------------------------------------
# 2. 6D (1,0) supergravity anomaly equations on T = 0
# ---------------------------------------------------------------------------

def su2_t0_spectrum(b):
    """SU(2) on T = 0 with anomaly coefficient b (degree-b curve): solve for fundamentals and adjoints.

    a.b = (lambda/6)(A_adj - sum x_R A_R), b.b = (lambda^2/3)(sum x_R C_R - C_adj), with a = -3, b in Z.
    Unknowns: n_f fundamentals and g adjoints.
    """
    lam = SU2["lambda"]
    # -3b = (1/6)(4 - n_f - 4 g)  and  b^2 = (1/3)(n_f/2 + 8 g - 8)
    # => n_f = 4 + 18 b - 4 g and n_f = 6 b^2 + 16 - 16 g.
    g = Fraction(6 * b * b + 16 - 4 - 18 * b, 12)
    n_f = 4 + 18 * b - 4 * g
    ok = (lam * Fraction(SU2["adj"]["A"] - n_f * SU2["2"]["A"] - g * SU2["adj"]["A"], 6) == -3 * b and
          Fraction(lam * lam, 3) * (n_f * SU2["2"]["C"] + g * SU2["adj"]["C"] - SU2["adj"]["C"]) == b * b)
    return {"b": b, "genus": g, "fundamentals": n_f, "equations_hold": ok}


def so10_t0_solutions(max_b=10, max_adjoints=0):
    """Nonabelian SO(10) conditions on T = 0 with n16 16s, n10 10s and g adjoint hypers (default g = 0).

    Quartic: 0 = (1 - g) B_adj - n10 B_10 - n16 B_16  =>  n10 = n16 + 2 (1 - g).
    a.b = (lambda/6)((1 - g) A_adj - n10 - 2 n16) = -3k, b.b = (lambda^2/3)(3/4 n16 + (g - 1) C_adj) = k^2, and the
    gravitational bound H - V <= 273 (H counts only the charged hypers here). Without adjoints only k = 1, 2.
    """
    out = []
    for k in range(1, max_b + 1):
        for g in range(0, max_adjoints + 1):
            n16 = k * k - 4 * (g - 1)
            n10 = n16 + 2 * (1 - g)
            if n16 < 0 or n10 < 0:
                continue
            A = (1 - g) * SO10["adj"]["A"] - n10 * SO10["10"]["A"] - n16 * SO10["16"]["A"]
            ok_ab = Fraction(SO10["lambda"], 6) * A == -3 * k
            ok_bb = Fraction(SO10["lambda"] ** 2, 3) * (n16 * SO10["16"]["C"] + n10 * SO10["10"]["C"]
                                                        + (g - 1) * SO10["adj"]["C"]) == k * k
            ok_quartic = (1 - g) * SO10["adj"]["B"] - n10 * SO10["10"]["B"] - n16 * SO10["16"]["B"] == 0
            ok_grav = 16 * n16 + 10 * n10 + 45 * g - 45 <= 273
            if ok_ab and ok_bb and ok_quartic and ok_grav:
                row = {"b": k, "n16": n16, "n10": n10}
                if max_adjoints:
                    row["adjoints"] = g
                out.append(row)
    return out


def so10_u1_t0_check(k, charges16, charges10, singlet_charges, n_neutral):
    """All anomaly equations for SO(10) x U(1) on T = 0 (a = -3, b = k, b~ = 2 beta).

    U(1) (Park-Taylor, b~ = height pairing): a.b~ = -(1/6) sum dim q^2, b.b~ = lambda sum A_R q^2,
    b~.b~ = (1/3) sum dim q^4; gravitational H - V + 29 T = 273 with V = 46.
    On T = 0 b~ is fixed from b.b~, so that equation holds by construction; the other two are checks.
    """
    n16, n10 = len(charges16), len(charges10)
    nonab = so10_t0_solutions()
    nonab_ok = any(s["b"] == k and s["n16"] == n16 and s["n10"] == n10 for s in nonab)
    sum_dim_q2 = 16 * sum(q * q for q in charges16) + 10 * sum(p * p for p in charges10) + sum(s * s for s in singlet_charges)
    sum_dim_q4 = 16 * sum(q ** 4 for q in charges16) + 10 * sum(p ** 4 for p in charges10) + sum(s ** 4 for s in singlet_charges)
    sum_A_q2 = 2 * sum(q * q for q in charges16) + sum(p * p for p in charges10)
    beta = Fraction(SO10["lambda"] * sum_A_q2, 2 * k)        # b.b~ = k * 2 beta
    eq_a = Fraction(-3 * 2 * beta) == Fraction(-sum_dim_q2, 6)
    eq_bb = Fraction(4 * beta * beta) == Fraction(sum_dim_q4, 3)
    H = 16 * n16 + 10 * n10 + len(singlet_charges) + n_neutral
    grav = H - 46 == 273 and n_neutral >= 0
    charged = [abs(q) for q in list(charges16) + list(charges10) + list(singlet_charges) if q]
    g = reduce(gcd, charged) if charged else 0
    families_per_flux = sum(charges16)
    return {"nonabelian_ok": nonab_ok, "beta": str(beta), "beta_integral": beta.denominator == 1,
            "a_btilde_ok": eq_a, "btilde_sq_ok": eq_bb, "gravitational_ok": grav,
            "all_ok": bool(nonab_ok and beta.denominator == 1 and eq_a and eq_bb and grav and beta > 0),
            "minus_a_dot_btilde": str(6 * beta), "charge_gcd": g,
            "net_16_per_unit_flux": families_per_flux}


def rescale(charges, g):
    return [q // g for q in charges]


def susy_examples():
    """Anomaly-free T = 0 spectra with five 16s (b = 1) and seven neutral 10s.

    (a) three_net: 16 charges (3,3,3,-3,-3) and 70 singlets of charge 6: gcd 3. Per unit flux it has 9 16s, 6 16bars
        and 420 chiral charge-6 singlets, so it is not BPR-like beyond the net count.
    (b) rescaled: the same divided by its gcd, 16 charges (1,1,1,-1,-1), singlets of charge 2: also anomaly-free.
    (c) gcd_one_three_net: 16 charges (3,3,3,-3,-3) with singlets 70 x 1(1) + 49 x 1(5) + 25 x 1(7): gcd 1, still
        3 net 16s per unit flux, so 3 | n_gen is allowed but not forced.
    """
    a = so10_u1_t0_check(1, [3, 3, 3, -3, -3], [0] * 7, [6] * 70, 319 - 80 - 70 - 70)
    b = so10_u1_t0_check(1, [1, 1, 1, -1, -1], [0] * 7, [2] * 70, 319 - 80 - 70 - 70)
    c = so10_u1_t0_check(1, [3, 3, 3, -3, -3], [0] * 7, [1] * 70 + [5] * 49 + [7] * 25, 319 - 80 - 70 - 144)
    return {"three_net": a, "rescaled": b, "gcd_one_three_net": c}


def t1_pure_u1(charge, multiplicity=128, n_neutral=117):
    """Pure U(1) on T = 1, lattice U = [[0,1],[1,0]], a = (-2,-2) (a.a = 8 = 9 - T).

    Solves a.b~ = -(1/6) sum q^2 and b~.b~ = (1/3) sum q^4 for b~ = (x, y): x + y = sum q^2 / 12, 2xy = sum q^4 / 3.
    Consistent iff b~ is a lattice vector (and b~ in 2U, the repository's anchor) and H - V + 29 = 273.
    """
    S2, S4 = multiplicity * charge ** 2, multiplicity * charge ** 4
    s_sum, s_prod = Fraction(S2, 12), Fraction(S4, 6)
    disc = s_sum * s_sum - 4 * s_prod
    root = None
    if disc >= 0 and disc.denominator == 1:
        r = int(round(float(disc) ** 0.5))
        if r * r == disc:
            root = r
    grav = multiplicity + n_neutral - 1 + 29 == 273
    if root is None or s_sum.denominator != 1:
        return {"charge": charge, "btilde": None, "consistent": False, "gravitational_ok": grav}
    x, y = (s_sum + root) / 2, (s_sum - root) / 2
    integral = x.denominator == 1 and y.denominator == 1
    even = integral and x % 2 == 0 and y % 2 == 0
    return {"charge": charge, "btilde": [str(x), str(y)], "consistent": bool(integral and even and grav),
            "minus_a_dot_btilde": str(2 * (x + y)), "gravitational_ok": grav}


def t0_integrality_lemma(max_numerator=20000):
    """On T = 0 (any gauge group): b~ = sum dim q^2 / 18 and 3 b~^2 = sum dim q^4 in Z force b~ in Z, and
    q^4 = q^2 mod 12 then forces b~ even. Checked over sum dim q^2 = N up to max_numerator."""
    integral = all(Fraction(N, 18).denominator == 1 for N in range(1, max_numerator)
                   if (3 * Fraction(N, 18) ** 2).denominator == 1)
    return {"btilde_integral_whenever_3btilde2_integral": integral}


def so10_on_curve(self_intersection):
    """Genus-0 curve of self-intersection n (a.b = -2 - n, b.b = n): n16 = n + 4, n10 = n + 6."""
    n = self_intersection
    n16, n10 = n + 4, n + 6
    ab = Fraction(SO10["lambda"], 6) * (SO10["adj"]["A"] - n10 * SO10["10"]["A"] - n16 * SO10["16"]["A"])
    bb = Fraction(SO10["lambda"] ** 2, 3) * (n16 * SO10["16"]["C"] + n10 * SO10["10"]["C"] - SO10["adj"]["C"])
    quartic = SO10["adj"]["B"] - n10 * SO10["10"]["B"] - n16 * SO10["16"]["B"]
    return {"n": n, "n16": n16, "n10": n10, "a.b": ab, "b.b": bb,
            "consistent": ab == -2 - n and bb == n and quartic == 0 and n16 >= 0}


def gcd_reduction_always_consistent(k, charges16, charges10, singlet_charges, n_neutral):
    """Divide an anomaly-free T = 0 spectrum's charges by their gcd g. Every U(1) equation scales
    homogeneously; the only integrality condition left is beta' = beta / g^2 in Z, i.e.
    k | (sum_10 p'^2 + 2 sum_16 q'^2). Automatic for b = k = 1."""
    full = so10_u1_t0_check(k, charges16, charges10, singlet_charges, n_neutral)
    g = full["charge_gcd"]
    if not full["all_ok"] or g <= 1:
        return {"applicable": False}
    red = so10_u1_t0_check(k, rescale(charges16, g), rescale(charges10, g), rescale(singlet_charges, g), n_neutral)
    return {"applicable": True, "gcd": g, "reduced_all_ok": red["all_ok"], "reduced_beta": red["beta"]}


def reduction_lemma(n_max=10001):
    """T = 0 lemma: after dividing by the gcd, b~' = 2 beta' is an even integer again.

    b~' = n = sum dim q'^2 / 18 is an integer (t0_integrality_lemma). The U(1) equations give sum dim q'^2 = 18 n
    and sum dim q'^4 = 3 n^2, while q^4 - q^2 = q^2 (q - 1)(q + 1) is always divisible by 12, so 3 n (n - 6) must
    be divisible by 12, i.e. 4 | n (n - 6): impossible for odd n. This holds for any gauge group and any b on
    T = 0, so every anomaly-free spectrum there can be normalized to charge gcd 1.
    """
    q_identity = all((q ** 4 - q ** 2) % 12 == 0 for q in range(-50, 51))
    odd_obstructed = all((n * (n - 6)) % 4 != 0 for n in range(1, n_max, 2))
    return {"q4_minus_q2_divisible_by_12": q_identity, "odd_n_obstructed": odd_obstructed,
            "gcd_one_normalization_always_consistent_on_T0": q_identity and odd_obstructed}


def demonstration_report():
    bpr = mt_quantities(minimal_fields(3))
    swapped = mt_quantities([(-1, "16", 3), (1, "16", 0)])
    ext = vectorlike_charge_one_extension()
    return {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "status": "forcing_3Z_conflicts_with_morrison_taylor_not_with_susy",
        "empirical_validation": False,
        "bpr6d_mt": bpr,
        "bpr6d_mt_swapped_chirality": swapped,
        "vectorlike_extension": {k: v for k, v in ext.items() if k != "singlet_zero_modes"},
        "su2_validation": [su2_t0_spectrum(b) for b in (1, 2, 3, 4)],
        "so10_t0": so10_t0_solutions(),
        "so10_t0_with_adjoints": so10_t0_solutions(max_adjoints=4),
        "susy_examples": susy_examples(),
        "t1_pure_u1": [t1_pure_u1(3), t1_pure_u1(1)],
        "t0_integrality_lemma": t0_integrality_lemma(),
        "reduction_lemma": reduction_lemma(),
        "limitations": list(LIMITATIONS),
    }
