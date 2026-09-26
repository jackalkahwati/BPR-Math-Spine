"""Global anomalies of BPR-6D: Spin bordism of B(Spin(10) x U(1)) in degree 7.

See doc/derivations/global_anomalies_2026-09-26.md. After local anomaly
cancellation by quantized Green-Schwarz terms, a remaining global anomaly is a
homomorphism Omega_7^Spin(BG) -> U(1). Below degree 8, MSpin agrees with ko at
the prime 2 (Anderson-Brown-Peterson), and H_*(BSpin(10) x CP^infty) has no odd
torsion, so Omega_7^Spin(BG) = ko_7(BG). This module computes the E2 page of the
Adams spectral sequence Ext_{A(1)}(H^*(X; F2), F2) => ko_*(X) by an exact
minimal free resolution over GF(2).

Mod-2 cohomology: H^*(BSpin(10)) is the quotient of H^*(BSO(10)) by the
Quillen ideal (w2, w3, w5, w9 below degree 17), i.e. F2[w4, w6, w7, w8, w10]
below degree 32, with Sq^i from the Wu formula; H^*(CP^infty) = F2[x].
"""

from functools import lru_cache
from itertools import product
from math import comb

import numpy as np

MODEL_ID = "bpr6d-global-anomaly-bordism-v1"

LIMITATIONS = [
    "Spin x Spin(10) x U(1) structure only; Spin x_Z2 Spin(10) backgrounds are not included.",
    "E2 of the Adams spectral sequence is computed; conclusions are drawn only where E2 vanishes, so no differentials are needed.",
    "Uses the standard Dai-Freed framework: after local cancellation the anomaly is a bordism invariant.",
    "The Green-Schwarz 7D term is assumed to be the standard well-defined term for integral Y (even lattice U, no refinement).",
    "Modules are truncated in degree; Ext is exact only for stems below the truncation degree.",
]

# ---------------------------------------------------------------------------
# The subalgebra A(1) = <Sq1, Sq2>
# ---------------------------------------------------------------------------

A1_WORDS = ("", "1", "2", "12", "21", "121", "212", "1212")
A1_DEGREES = tuple(sum(int(c) for c in w) for w in A1_WORDS)


def _reduce_word(word):
    """Normal form of a word in Sq1, Sq2: returns a basis word or None (zero)."""
    if sum(int(c) for c in word) > 6:
        return None
    changed = True
    while changed:
        changed = False
        if "11" in word:
            return None
        if "22" in word:
            word = word.replace("22", "121", 1)
            changed = True
            continue
        if "2121" in word:
            word = word.replace("2121", "1212", 1)
            changed = True
    return word if word in A1_WORDS else _fail(word)


def _fail(word):
    raise ArithmeticError("unreduced A(1) word " + word)


@lru_cache(maxsize=None)
def a1_product(i, j):
    """Basis index of A1_WORDS[i] * A1_WORDS[j], or None."""
    red = _reduce_word(A1_WORDS[i] + A1_WORDS[j])
    return None if red is None else A1_WORDS.index(red)


def a1_checks():
    """Associativity of the multiplication table, dimension 8, top class in degree 6."""
    n = len(A1_WORDS)
    assoc = True
    for i, j, k in product(range(n), repeat=3):
        ij = a1_product(i, j)
        left = None if ij is None else a1_product(ij, k)
        jk = a1_product(j, k)
        right = None if jk is None else a1_product(i, jk)
        assoc &= left == right
    return {"dimension": n, "associative": assoc, "top_degree": max(A1_DEGREES)}


# ---------------------------------------------------------------------------
# Polynomial Steenrod modules
# ---------------------------------------------------------------------------

class PolyAlgebra:
    """F2[generators] truncated at degree D, with Sq1, Sq2 on generators given as polynomials.

    A polynomial is a frozenset of exponent tuples (coefficients in F2).
    """

    def __init__(self, names, degrees, sq1, sq2, max_degree):
        self.names, self.degrees, self.D = tuple(names), tuple(degrees), max_degree
        self.n = len(names)
        self._sq = {1: dict(sq1), 2: dict(sq2)}
        self.monomials = sorted((m for m in self._all_monomials() if self.deg(m) <= max_degree),
                                key=lambda m: (self.deg(m), m))

    def _all_monomials(self):
        bounds = [self.D // d for d in self.degrees]
        for exps in product(*[range(b + 1) for b in bounds]):
            yield tuple(exps)

    def deg(self, mono):
        return sum(e * d for e, d in zip(mono, self.degrees))

    def gen(self, i):
        return tuple(1 if k == i else 0 for k in range(self.n))

    @staticmethod
    def mul(p, q):
        out = set()
        for a in p:
            for b in q:
                out ^= {tuple(x + y for x, y in zip(a, b))}
        return frozenset(out)

    @lru_cache(maxsize=None)
    def sq(self, i, mono):
        """Sq^i of a monomial (i in 0, 1, 2) by the Cartan formula."""
        if i == 0:
            return frozenset([mono])
        if sum(mono) == 0:
            return frozenset()
        k = next(idx for idx, e in enumerate(mono) if e)
        g = self.gen(k)
        rest = tuple(e - (1 if idx == k else 0) for idx, e in enumerate(mono))
        sq_g = {0: frozenset([g]), 1: self._sq[1][self.names[k]], 2: self._sq[2][self.names[k]]}
        out = set()
        for a in range(i + 1):
            out ^= set(self.mul(sq_g[a], self.sq(i - a, rest)))
        return frozenset(out)


class Module:
    """A finite graded F2-module with Sq1, Sq2 as sparse maps on a basis."""

    def __init__(self, degrees, sq1, sq2, name=""):
        self.degrees = list(degrees)
        self.N = len(self.degrees)
        self.sq1 = sq1  # list of sets of basis indices
        self.sq2 = sq2
        self.name = name

    def act(self, word, vec):
        """Apply a word (e.g. '12' = Sq1 Sq2) to a vector given as a set of basis indices."""
        for c in reversed(word):
            table = self.sq1 if c == "1" else self.sq2
            out = set()
            for b in vec:
                out ^= table[b]
            vec = out
        return vec

    def check_adem(self):
        ok = True
        for b in range(self.N):
            v = {b}
            ok &= not self.act("11", v)
            ok &= self.act("22", v) == self.act("121", v)
            ok &= self.act("2121", v) == self.act("1212", v)
        return ok

    def truncate(self, D):
        keep = [i for i, d in enumerate(self.degrees) if d <= D]
        idx = {old: new for new, old in enumerate(keep)}
        f = lambda s: {idx[b] for b in s if b in idx}  # noqa: E731
        return Module([self.degrees[i] for i in keep], [f(self.sq1[i]) for i in keep],
                      [f(self.sq2[i]) for i in keep], self.name)


def module_from_poly(alg, reduced=True):
    monos = [m for m in alg.monomials if not (reduced and sum(m) == 0)]
    index = {m: i for i, m in enumerate(monos)}
    to_set = lambda poly: {index[m] for m in poly if m in index}  # noqa: E731
    return Module([alg.deg(m) for m in monos], [to_set(alg.sq(1, m)) for m in monos],
                  [to_set(alg.sq(2, m)) for m in monos])


def tensor(M, N, D):
    """Tensor product with the Cartan (diagonal) action, truncated at degree D."""
    pairs = [(a, b) for a in range(M.N) for b in range(N.N) if M.degrees[a] + N.degrees[b] <= D]
    idx = {p: i for i, p in enumerate(pairs)}

    def img(sq_a, sq_b, a, b):
        out = set()
        for x in sq_a:
            for y in sq_b:
                if (x, y) in idx:
                    out ^= {idx[(x, y)]}
        return out

    sq1, sq2 = [], []
    for a, b in pairs:
        one = img(M.sq1[a], {b}, a, b) ^ img({a}, N.sq1[b], a, b)
        two = img(M.sq2[a], {b}, a, b) ^ img(M.sq1[a], N.sq1[b], a, b) ^ img({a}, N.sq2[b], a, b)
        sq1.append(one)
        sq2.append(two)
    return Module([M.degrees[a] + N.degrees[b] for a, b in pairs], sq1, sq2)


# --- Named cohomology rings ---------------------------------------------------

def wu(i, j, n):
    """Sq^i w_j in H^*(BSO(n)) as a list of (a, b) meaning w_a w_b (w_0 = 1)."""
    terms = []
    for t in range(i + 1):
        c = 1 if (t == 0) else (comb(j - i + t - 1, t) if j - i + t - 1 >= 0 else 0)
        if c % 2 and j + t <= n:
            terms.append((i - t, j + t))
    return terms


def bspin10_algebra(D):
    names = ["w4", "w6", "w7", "w8", "w10"]
    degrees = [4, 6, 7, 8, 10]
    killed = {1, 2, 3, 5, 9}  # Quillen ideal below degree 17 (w1 = 0 on BSO)

    def w_poly(k, alg_names):
        if k == 0:
            return frozenset([tuple(0 for _ in alg_names)])
        if k in killed or k > 10:
            return frozenset()
        return frozenset([tuple(1 if nm == "w{}".format(k) else 0 for nm in alg_names)])

    sq = {1: {}, 2: {}}
    for nm in names:
        j = int(nm[1:])
        for i in (1, 2):
            out = set()
            for a, b in wu(i, j, 10):
                out ^= set(PolyAlgebra.mul(w_poly(a, names), w_poly(b, names)))
            sq[i][nm] = frozenset(out)
    return PolyAlgebra(names, degrees, sq[1], sq[2], D)


def bso_algebra(n, D):
    """H^*(BSO(n)) = F2[w2..wn] for validating the Wu formula through the Adem relations."""
    names = ["w{}".format(k) for k in range(2, n + 1)]
    degrees = list(range(2, n + 1))

    def w_poly(k):
        if k == 0:
            return frozenset([tuple(0 for _ in names)])
        if k == 1 or k > n:
            return frozenset()
        return frozenset([tuple(1 if nm == "w{}".format(k) else 0 for nm in names)])

    sq = {1: {}, 2: {}}
    for nm in names:
        j = int(nm[1:])
        for i in (1, 2):
            out = set()
            for a, b in wu(i, j, n):
                out ^= set(PolyAlgebra.mul(w_poly(a), w_poly(b)))
            sq[i][nm] = frozenset(out)
    return PolyAlgebra(names, degrees, sq[1], sq[2], D)


def cp_infinity_algebra(D):
    return PolyAlgebra(["x"], [2], {"x": frozenset()}, {"x": frozenset([(2,)])}, D)


def bsu_algebra(n, D):
    """H^*(BSU(n)) = F2[c2..cn]; Sq^odd c = 0 and Sq^2 c_j from the Wu formula (doubled degrees)."""
    names = ["c{}".format(k) for k in range(2, n + 1)]
    degrees = [2 * k for k in range(2, n + 1)]

    def c_poly(k):
        if k == 0:
            return frozenset([tuple(0 for _ in names)])
        if k == 1 or k > n:
            return frozenset()
        return frozenset([tuple(1 if nm == "c{}".format(k) else 0 for nm in names)])

    sq1, sq2 = {}, {}
    for nm in names:
        j = int(nm[1:])
        sq1[nm] = frozenset()
        out = set()
        for a, b in wu(1, j, n):
            out ^= set(PolyAlgebra.mul(c_poly(a), c_poly(b)))
        sq2[nm] = frozenset(out)
    return PolyAlgebra(names, degrees, sq1, sq2, D)


def point_module():
    return Module([0], [set()], [set()], "F2")


# ---------------------------------------------------------------------------
# GF(2) linear algebra and the minimal resolution
# ---------------------------------------------------------------------------

def _rref(rows):
    """Row-reduce a list of int bitmasks; returns (pivot-indexed basis dict)."""
    basis = {}
    for r in rows:
        v = r
        while v:
            p = v.bit_length() - 1
            if p in basis:
                v ^= basis[p]
            else:
                basis[p] = v
                break
    return basis


def _in_span(basis, v):
    while v:
        p = v.bit_length() - 1
        if p not in basis:
            return False
        v ^= basis[p]
    return True


def _nullspace(columns):
    """Kernel of the map sending unit vector k to columns[k] (bitmasks): list of bitmasks over k."""
    pivots = {}  # pivot bit -> (reduced image, combination)
    kernel = []
    for k, col in enumerate(columns):
        v, comb_mask = col, 1 << k
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                pv, pc = pivots[p]
                v ^= pv
                comb_mask ^= pc
            else:
                pivots[p] = (v, comb_mask)
                break
        if not v:
            kernel.append(comb_mask)
    return kernel


class FreeModule:
    """Free A(1)-module on generators of given degrees, basis (generator, word) up to degree T."""

    def __init__(self, gen_degrees, T):
        self.gen_degrees = list(gen_degrees)
        self.T = T
        self.basis = [(g, w) for g, d in enumerate(self.gen_degrees) for w in range(8)
                      if d + A1_DEGREES[w] <= T]
        self.index = {b: i for i, b in enumerate(self.basis)}
        self.degrees = [self.gen_degrees[g] + A1_DEGREES[w] for g, w in self.basis]

    def act(self, word, vec):
        """Left action of a word on a set of basis indices."""
        wi = A1_WORDS.index(_reduce_word(word)) if _reduce_word(word) is not None else None
        if wi is None:
            return set()
        out = set()
        for b in vec:
            g, w = self.basis[b]
            prod = a1_product(wi, w)
            if prod is not None and (g, prod) in self.index:
                out ^= {self.index[(g, prod)]}
        return out


def _set_to_mask(s):
    m = 0
    for b in s:
        m |= 1 << b
    return m


def _mask_to_set(m):
    out, k = set(), 0
    while m:
        if m & 1:
            out.add(k)
        m >>= 1
        k += 1
    return out


def minimal_resolution(M, s_max, T):
    """Ext^{s,t}_{A(1)}(M, F2) for s <= s_max, t <= T, as a dict {(s, t): dimension}."""
    ext = {}
    # Target: the module M (as ambient), with "target subspace" = everything.
    ambient_act = M.act
    ambient_deg = M.degrees
    target = {t: [1 << b for b, d in enumerate(ambient_deg) if d == t] for t in range(T + 1)}
    for s in range(s_max + 1):
        gens, images = [], []  # generator degrees and their images (bitmasks in ambient)
        for t in range(T + 1):
            span = _rref([_set_to_mask(ambient_act(A1_WORDS[w], _mask_to_set(img)))
                          for (d, img) in zip(gens, images) for w in range(8) if d + A1_DEGREES[w] == t])
            for v in target.get(t, []):
                if not _in_span(span, v):
                    gens.append(t)
                    images.append(v)
                    # add the new generator's own degree-t image (word "")
                    p = v
                    while p:
                        q = p.bit_length() - 1
                        if q in span:
                            p ^= span[q]
                        else:
                            span[q] = p
                            break
        for t in gens:
            ext[(s, t)] = ext.get((s, t), 0) + 1
        # Kernel of F_s -> ambient, degree by degree.
        F = FreeModule(gens, T)
        new_target = {}
        for t in range(T + 1):
            cols_idx = [b for b, d in enumerate(F.degrees) if d == t]
            cols = []
            for b in cols_idx:
                g, w = F.basis[b]
                cols.append(_set_to_mask(ambient_act(A1_WORDS[w], _mask_to_set(images[g]))))
            kern = _nullspace(cols)
            new_target[t] = [_set_to_mask({cols_idx[k] for k in _mask_to_set(km)}) for km in kern]
        ambient_act, ambient_deg, target = F.act, F.degrees, new_target
    return ext


def ext_chart(M, s_max=12, T=20):
    """{stem: [(s, dim), ...]} from the minimal resolution."""
    ext = minimal_resolution(M, s_max, T)
    chart = {}
    for (s, t), dim in ext.items():
        chart.setdefault(t - s, []).append((s, dim))
    return {stem: sorted(v) for stem, v in sorted(chart.items())}


def sq1_homology(M, degree):
    """dim of ker Sq1 / im Sq1 in a degree (Q0 Margolis homology: detects h0-towers)."""
    idx = [b for b, d in enumerate(M.degrees) if d == degree]
    below = [b for b, d in enumerate(M.degrees) if d == degree - 1]
    ker = len(_nullspace([_set_to_mask(M.sq1[b]) for b in idx]))
    image_rank = len(_rref([_set_to_mask(M.sq1[b]) for b in below]))
    return ker - image_rank


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def bpr6d_modules(D=14):
    bs = module_from_poly(bspin10_algebra(D))
    cp = module_from_poly(cp_infinity_algebra(D))
    return {"BSpin(10)": bs, "CP^infty": cp, "BSpin(10)^CP^infty": tensor(bs, cp, D)}


def stem_summary(chart, stems):
    return {stem: sum(dim for _, dim in chart.get(stem, [])) for stem in stems}


def global_anomaly_verdict(D=14, s_max=12, T=20):
    mods = bpr6d_modules(D)
    out = {}
    for name, M in mods.items():
        chart = ext_chart(M, s_max, T)
        out[name] = {"adem_relations_hold": M.check_adem(),
                     "E2_stem_7": chart.get(7, []),
                     "odd_degree_sq1_homology": {d: sq1_homology(M, d) for d in range(1, D, 2)},
                     "E2_total_by_stem": stem_summary(chart, range(0, 9))}
    vanishes = all(not v["E2_stem_7"] for v in out.values())
    return {"summands": out, "omega7_vanishes": vanishes,
            "global_anomaly": "none" if vanishes else "undetermined"}


def demonstration_report():
    verdict = global_anomaly_verdict()
    checks = validation_checks()
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "omega7_spin_bspin10_x_bu1_vanishes" if verdict["omega7_vanishes"] else "omega7_nonzero",
        "empirical_validation": False,
        "a1": a1_checks(),
        "validation": checks,
        "verdict": {k: v for k, v in verdict.items() if k != "summands"},
        "summands": {name: {"E2_stem_7": [list(p) for p in v["E2_stem_7"]],
                            "E2_total_by_stem": {str(k): n for k, n in v["E2_total_by_stem"].items()},
                            "adem_relations_hold": v["adem_relations_hold"]}
                     for name, v in verdict["summands"].items()},
        "limitations": list(LIMITATIONS),
    }


def validation_checks():
    """Known results the engine must reproduce."""
    pt = ext_chart(point_module(), 12, 20)
    su2 = ext_chart(module_from_poly(bsu_algebra(2, 14)), 12, 20)
    su3 = ext_chart(module_from_poly(bsu_algebra(3, 14)), 12, 20)
    cp = ext_chart(module_from_poly(cp_infinity_algebra(14)), 12, 20)
    lists = lambda entries: [list(p) for p in entries]  # noqa: E731
    return {
        "ko_point_stems_0_to_8": {str(k): lists(pt.get(k, [])) for k in range(9)},
        "bsu2_reduced_stem5": lists(su2.get(5, [])), "bsu2_reduced_stem7": lists(su2.get(7, [])),
        "bsu3_reduced_stem5": lists(su3.get(5, [])), "bsu3_reduced_stem7": lists(su3.get(7, [])),
        "cp_reduced_odd_stems": {str(k): lists(cp.get(k, [])) for k in (1, 3, 5, 7)},
    }
