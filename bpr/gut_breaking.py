"""Breaking Spin(10) to the Standard Model in BPR-6D: flux and orbifold routes.

See doc/derivations/gut_breaking_2026-09-26.md. Exact weight / root algebra for
Spin(10) (Cartan basis e_1..e_5, colour planes 1-3, weak planes 4-5) and explicit
Wigner rotation matrices for the SU(2) isometry acting on the flux families.

Conventions: the 16 has weights (+-1/2)^5 with an odd number of minus signs (this set carries the
standard left-handed content Q, u^c, d^c, L, e^c, nu^c);
Y = (e4 + e5)/2 - (e1 + e2 + e3)/3, B-L = -(2/3)(e1 + e2 + e3), X = e1 + ... + e5,
T3L = (e4 - e5)/2, T3R = (e4 + e5)/2.
"""

from fractions import Fraction
from itertools import combinations, product

import numpy as np

MODEL_ID = "bpr6d-gut-breaking-v1"
F = Fraction

LIMITATIONS = [
    "Only abelian (Cartan) fluxes on the round S^2 and S^2/(Z2 x Z2) orbifolds with commuting gauge twists are analysed.",
    "The Stueckelberg argument uses the round-4 Green-Schwarz coupling (coefficient 3 of lambda_V in Y_g); its 4D normalization is not computed, only that it is nonzero.",
    "Fixed-point localized matter and localized anomalies on orbifolds are not included.",
    "Conventional Higgs breaking of Spin(10) is possible in principle but requires field content that BPR-6D does not supply.",
]

Y = (F(-1, 3), F(-1, 3), F(-1, 3), F(1, 2), F(1, 2))
BL = (F(-2, 3), F(-2, 3), F(-2, 3), F(0), F(0))
X = (F(1), F(1), F(1), F(1), F(1))
T3L = (F(0), F(0), F(0), F(1, 2), F(-1, 2))
T3R = (F(0), F(0), F(0), F(1, 2), F(1, 2))


def dot(a, b):
    return sum(F(x) * F(y) for x, y in zip(a, b))


def weights16():
    return [w for w in product((F(1, 2), F(-1, 2)), repeat=5) if sum(1 for c in w if c < 0) % 2 == 1]


def roots():
    out = []
    for i, j in combinations(range(5), 2):
        for si, sj in product((1, -1), repeat=2):
            r = [0] * 5
            r[i], r[j] = si, sj
            out.append(tuple(r))
    return out


def sm_label(w):
    """Standard-Model multiplet of a 16 weight from (T3L, Y, colour)."""
    y = dot(Y, w)
    colour = sum(1 for c in w[:3] if c > 0)
    labels = {F(1, 6): "Q", F(-2, 3): "u^c", F(1, 3): "d^c", F(-1, 2): "L", F(1): "e^c", F(0): "nu^c"}
    return labels[y]


def sm_multiplet_dimensions():
    counts = {}
    for w in weights16():
        counts[sm_label(w)] = counts.get(sm_label(w), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Flux breaking
# ---------------------------------------------------------------------------

def zero_mode_indices(h, parent_charge=3, flux=1):
    """Index per 16 weight: 16_+(q) sees q m + w.h, 16_-(0) sees w.h with opposite chirality."""
    rows = []
    for w in weights16():
        plus = parent_charge * flux + dot(w, h)
        minus = -dot(w, h)
        rows.append({"weight": w, "multiplet": sm_label(w), "from_16plus": plus, "from_16minus": minus,
                     "net": plus + minus, "vectorlike_pairs": min(abs(plus), abs(minus)) if plus * minus < 0 else 0})
    return rows


def flux_quantized(h):
    """w.h integral on all weights of the 10 (= +-e_i) and the 16."""
    return all(F(x).denominator == 1 for x in h) and all(dot(w, h).denominator == 1 for w in weights16())


def centralizer(h):
    return [r for r in roots() if dot(r, h) == 0]


def contains_sm(h):
    sm_roots = [r for r in roots() if (r[3] == 0 and r[4] == 0 and sum(r[:3]) == 0 and sorted(r[:3]) == [-1, 0, 1])
                or (r[:3] == (0, 0, 0) and r[3] == -r[4])]
    return all(r in centralizer(h) for r in sm_roots)


def centralizer_is_exactly_sm(h):
    """Nonabelian part of the centralizer = SU(3) x SU(2)_L (8 + 2 = 8 roots: 6 + 2)."""
    return contains_sm(h) and len(centralizer(h)) == 8


def hypercharge_massless(h):
    """The Green-Schwarz term makes the flux direction massive; Y stays massless iff tr(Y h) = 0."""
    return dot(Y, h) == 0


def flux_scan(bound=3):
    """All integral Cartan fluxes h with entries |h_i| <= bound: is any compatible with SM and massless Y?"""
    hits = {"sm_centralizer": 0, "sm_with_massless_Y": 0, "quantized_sm": 0}
    for h in product(range(-bound, bound + 1), repeat=5):
        if not any(h):
            continue
        if centralizer_is_exactly_sm(h):
            hits["sm_centralizer"] += 1
            hits["quantized_sm"] += int(flux_quantized(h))
            hits["sm_with_massless_Y"] += int(hypercharge_massless(h))
    return hits


def flux_plane_theorem():
    """h must lie in span(Y, X) to keep SU(3) x SU(2): h = (a,a,a,b,b). Y massless needs a = b, then SU(5) survives."""
    h_generic = (F(1), F(1), F(1), F(3), F(3))
    h_x = X
    return {"generic_in_plane_is_sm": centralizer_is_exactly_sm(h_generic),
            "generic_in_plane_Y_massless": hypercharge_massless(h_generic),
            "X_direction_Y_massless": hypercharge_massless(h_x),
            "X_direction_centralizer_roots": len(centralizer(h_x)),  # 20 = roots of SU(5)
            "Y_perp_X": dot(Y, X) == 0}


# ---------------------------------------------------------------------------
# Orbifold S^2 / (Z2 x Z2)
# ---------------------------------------------------------------------------

def spin_j_matrices(j):
    """Spin-j generators and pi rotations about z and x (numerical, via matrix exponential)."""
    from scipy.linalg import expm
    dim = int(2 * j + 1)
    ms = [j - k for k in range(dim)]
    Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((dim, dim), complex)
    for k in range(1, dim):
        m = ms[k]
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m * (m + 1))
    Jx = (Jp + Jp.conj().T) / 2
    return expm(-1j * np.pi * Jz), expm(-1j * np.pi * Jx)


def d2_multiplicities(j):
    """Multiplicity of each (R_z, R_x) sign pair in spin j, by simultaneous diagonalization (integer j)."""
    Rz, Rx = spin_j_matrices(j)
    # For integer j these are real commuting involutions; project onto joint eigenspaces.
    out = {}
    for sz, sx in product((1, -1), repeat=2):
        P = (np.eye(Rz.shape[0]) + sz * Rz) @ (np.eye(Rz.shape[0]) + sx * Rx) / 4
        out[(sz, sx)] = int(round(np.real(np.trace(P))))
    return out


def d2_character_formula(j):
    triv = (2 * j + 1 + 3 * (-1) ** j) // 4
    non = (2 * j + 1 - (-1) ** j) // 4
    return {(1, 1): triv, (1, -1): non, (-1, 1): non, (-1, -1): non}


def orbifold_classes():
    """(P_PS, P_SU5) of each SM multiplet: P_PS = +1 on (4,2,1), -1 on (4bar,1,2); P_SU5 = +1 on 10, -1 on 5bar, 1."""
    x_of_10 = {dot(X, w) for w in weights16() if sm_label(w) in ("Q", "u^c", "e^c")}
    if len(x_of_10) != 1:
        raise ArithmeticError("the SU(5) 10 does not carry a single X charge")
    classes = {}
    for w in weights16():
        lab = sm_label(w)
        p_ps = 1 if dot(T3R, w) == 0 else -1  # (4,2,1): T3R = 0; (4bar,1,2): T3R = +-1/2
        p_su5 = 1 if dot(X, w) in x_of_10 else -1
        classes[lab] = (p_ps, p_su5)
    return classes


def orbifold_family_counts(j):
    """Families per SM multiplet for every choice of intrinsic parities (eta, eta')."""
    mult = d2_multiplicities(j)
    classes = orbifold_classes()
    table = {}
    for eta, eta2 in product((1, -1), repeat=2):
        counts = {lab: mult[(eta * c[0], eta2 * c[1])] for lab, c in classes.items()}
        table["{},{}".format(eta, eta2)] = counts
    uniform = [key for key, counts in table.items() if len(set(counts.values())) == 1]
    return {"j": j, "multiplicities": {"{},{}".format(*k): v for k, v in mult.items()},
            "counts": table, "uniform_choices": uniform}


def demonstration_report():
    h_example = (0, 0, 0, 1, 1)  # T3R flux (integral on the 10; check the 16)
    idx = zero_mode_indices((F(0), F(0), F(0), F(1), F(1)))
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "geometric_breaking_obstructed",
        "empirical_validation": False,
        "sm_dimensions": sm_multiplet_dimensions(),
        "flux_example_T3R": {"quantized": flux_quantized(tuple(F(v) for v in h_example)),
                             "net_per_multiplet": sorted({(r["multiplet"], str(r["net"])) for r in idx}),
                             "vectorlike_pairs": sum(r["vectorlike_pairs"] for r in idx)},
        "flux_plane_theorem": flux_plane_theorem(),
        "flux_scan": flux_scan(2),
        "orbifold_classes": {k: list(v) for k, v in orbifold_classes().items()},
        "orbifold_j1": orbifold_family_counts(1),
        "orbifold_uniform_j_up_to_10": [j for j in range(1, 11) if orbifold_family_counts(j)["uniform_choices"]],
        "limitations": list(LIMITATIONS),
    }
