"""Preregistered scan: do any frozen BPR operators produce the flavor labels?

CONTEXT. Since 2026-09-10 all nine fermion mode labels are CONJECTURAL
(`qcd_flavor.derive_l_modes`). The next step in the open-problems list
(website /status) is to derive at least one label as an eigenvalue of an
explicit operator built from frozen ingredients only. This module is the
first, deliberately narrow attempt: a fixed list of candidate operators and a
fixed scoring rule, written BEFORE the spectra were computed. It cannot
"fit" anything; it can only report hits against a stated chance rate.

TARGET LABELS (z = 6, n_gen = 3; from the retained ansatz):
    up      (1, 24, 283)          mass ~ l^2
    down    (1, 4, 30)            mass ~ l(l + W_c), W_c = sqrt(3)
    lepton  (1, sqrt(210), 59)    mass ~ l^2 (so 210 is the squared label)
Scored quantities: the integers {1, 4, 24, 30, 59, 283} as MODE LABELS and
the eigenvalue-like integers {1, 4, 24, 30, 210, 283, 3481} as EIGENVALUES.

CANDIDATE OPERATORS (frozen ingredients only; nothing tuned):
    C1  round-S^2 scalar Laplacian: eigenvalues l(l+1), labels l.
    C2  S^2 Laplacian restricted to O_h-invariant harmonics (the point group
        of the cubic tiling that fixes z = 6): allowed labels l from the
        Molien series 1/((1-t^4)(1-t^6)).
    C3  same, chiral O only (adds the l = 9 pseudoscalar invariant).
    C4  compact boson on S^2 with the CCR selection rule m = 0 mod 6:
        allowed (l, m) with 6 | m; labels l.
    C5  graph Laplacian and adjacency of the z = 6 coordination shell of a
        cubic site (the octahedron K_{2,2,2}).
    C6  graph Laplacian of the 7-site star (site + 6 neighbours) and of the
        octahedron-with-centre.
    C7  D_n electric Casimirs for the allowed classes n in {5, 8, 9, 12}:
        eps(A1)=0, eps(A2)=2, eps(B1)=4, eps(B2)=6, eps(E_k)=3-2cos(2 pi k/n).
    C8  winding-shifted S^2 spectrum l(l + sqrt(3)) (the down-type rule).

SCORING RULE (fixed). For each candidate and each target set, count hits.
A hit is a target integer that appears exactly among the candidate's labels
(for the label test) or eigenvalues (for the eigenvalue test), with spectra
truncated at 300. The chance rate is the fraction of integers in [1, 300]
that appear in the candidate's spectrum; the expected number of hits among
k targets by chance is k times that fraction. A candidate is REPORTED as
interesting only if the exact binomial tail probability of at least that
many hits is below 0.05 AFTER Bonferroni correction for the total number of
(candidate, test) comparisons made here (16), and it has >= 2 hits. Nothing here is a derivation; a positive result
would justify building the explicit operator for that sector, no more.
"""
from __future__ import annotations

import itertools
import numpy as np
from math import comb

Z = 6
LABELS = {"up": (1, 24, 283), "down": (1, 4, 30), "lepton": (1, 210, 59)}
LABEL_INTS = sorted({1, 4, 24, 30, 59, 283})
EIGEN_INTS = sorted({1, 4, 24, 30, 210, 283, 3481})
CUT = 300


def molien_allowed(gens: tuple[int, ...], cut: int = CUT) -> list[int]:
    """Degrees l with a nonzero coefficient in prod 1/(1-t^g)."""
    coef = np.zeros(cut + 1, dtype=np.int64)
    coef[0] = 1
    for g in gens:
        for i in range(g, cut + 1):
            coef[i] += coef[i - g]
    return [l for l in range(1, cut + 1) if coef[l] > 0]


def c1_round_sphere():
    labels = list(range(1, CUT + 1))
    eig = [l * (l + 1) for l in labels]
    return labels, eig


def c2_octahedral_invariant():
    labels = molien_allowed((4, 6))
    return labels, [l * (l + 1) for l in labels]


def c3_chiral_octahedral():
    labels = molien_allowed((4, 6, 9))
    return labels, [l * (l + 1) for l in labels]


def c4_ccr_selection():
    labels = [l for l in range(1, CUT + 1) if any(m % Z == 0 for m in range(0, l + 1))]
    return labels, [l * (l + 1) for l in labels]


def _graph_spectra(A: np.ndarray):
    D = np.diag(A.sum(axis=1))
    lap = np.linalg.eigvalsh(D - A)
    adj = np.linalg.eigvalsh(A)
    return sorted(set(np.round(lap, 9))), sorted(set(np.round(adj, 9)))


def octahedron_adjacency() -> np.ndarray:
    pts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    A = np.zeros((6, 6))
    for i, j in itertools.combinations(range(6), 2):
        if np.dot(pts[i], pts[j]) == 0:
            A[i, j] = A[j, i] = 1
    return A


def c5_shell_graph():
    lap, adj = _graph_spectra(octahedron_adjacency())
    return [], [x for x in lap + adj if float(x).is_integer()]


def c6_star_graphs():
    A6 = octahedron_adjacency()
    star = np.zeros((7, 7)); star[0, 1:] = star[1:, 0] = 1
    both = star.copy(); both[1:, 1:] = A6
    out = []
    for A in (star, both):
        lap, adj = _graph_spectra(A)
        out += [x for x in lap + adj if float(x).is_integer()]
    return [], sorted(set(out))


def c7_dn_casimirs():
    eig = set()
    for n in (5, 8, 9, 12):
        eig |= {0.0, 2.0, 4.0, 6.0}
        eig |= {round(3 - 2 * np.cos(2 * np.pi * k / n), 9) for k in range(1, n // 2 + 1)}
    return [], sorted(x for x in eig if float(x).is_integer())


def c8_winding_shift():
    labels = list(range(1, CUT + 1))
    eig = [round(l * (l + np.sqrt(3)), 9) for l in labels]
    return labels, [e for e in eig if float(e).is_integer()]


CANDIDATES = {
    "C1 round S2 Laplacian": c1_round_sphere,
    "C2 O_h-invariant harmonics": c2_octahedral_invariant,
    "C3 O-invariant harmonics": c3_chiral_octahedral,
    "C4 CCR m=0 mod 6": c4_ccr_selection,
    "C5 octahedron shell graph": c5_shell_graph,
    "C6 star / star+shell graphs": c6_star_graphs,
    "C7 D_n electric Casimirs": c7_dn_casimirs,
    "C8 winding-shifted l(l+sqrt3)": c8_winding_shift,
}


def score(values: list, targets: list) -> dict:
    vals = {int(v) for v in values if 1 <= v <= CUT and float(v).is_integer()}
    hits = [t for t in targets if t in vals]
    density = len(vals) / CUT
    in_range = [t for t in targets if t <= CUT]
    expected = density * len(in_range)
    sigma = np.sqrt(expected) if expected > 0 else 0.0
    k, n, q = len(hits), len(in_range), density
    p_tail = sum(comb(n, j) * q**j * (1 - q) ** (n - j) for j in range(k, n + 1)) if n else 1.0
    return {"hits": hits, "n_hits": len(hits), "expected": round(float(expected), 3),
            "density": round(float(density), 4),
            "excess_sigma": round(float((len(hits) - expected) / sigma), 2) if sigma > 0 else None,
            "p_tail": float(p_tail)}


def run_scan() -> dict:
    out = {}
    for name, fn in CANDIDATES.items():
        labels, eig = fn()
        out[name] = {
            "labels_test": score(labels, LABEL_INTS) if labels else None,
            "eigen_test": score(eig, EIGEN_INTS),
            "spectrum_head": [float(x) for x in (labels or eig)[:12]],
        }
    return out


N_COMPARISONS = 16   # 8 candidates x 2 tests, fixed before computing


def interesting(scan: dict, alpha: float = 0.05) -> list:
    """Bonferroni-corrected: p_tail * N_COMPARISONS < alpha and >= 2 hits."""
    out = []
    for name, r in scan.items():
        for test in ("labels_test", "eigen_test"):
            s = r[test]
            if s and s["n_hits"] >= 2 and s["p_tail"] * N_COMPARISONS < alpha:
                out.append((name, test, s))
    return out


def report() -> str:
    scan = run_scan()
    lines = ["Flavor-label scan (preregistered candidates, fixed scoring)", ""]
    for name, r in scan.items():
        lines.append(name)
        lines.append(f"  spectrum head: {r['spectrum_head']}")
        for test in ("labels_test", "eigen_test"):
            s = r[test]
            if s:
                lines.append(f"  {test:12s} hits={s['hits']} expected={s['expected']} "
                             f"density={s['density']} p_tail={s['p_tail']:.3g} "
                             f"bonferroni={min(1.0, s['p_tail'] * N_COMPARISONS):.3g}")
    hits = interesting(scan)
    lines.append("")
    lines.append("INTERESTING (Bonferroni p < 0.05, >= 2 hits): "
                 + (", ".join(f"{n} [{t}]" for n, t, _ in hits) if hits else "none"))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
