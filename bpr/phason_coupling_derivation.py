"""Derivation attempt for the phason coupling exponent — how far it actually goes.

The 31-order gap reduces (via casimir_epsilon_toy) to ONE unknown: the net
projection suppression S in ε = g²·O·S. casimir_epsilon_toy spans S over 40
orders and refuses to predict. This module ATTEMPTS to derive S's scaling with
the substrate size p from the cut-and-project geometry, and reports exactly how
far the derivation gets — including where it stops.

THE DERIVATION (cut-and-project projection suppression)
-------------------------------------------------------
Setup. The substrate boundary phase φ takes values in Z_p (p discrete phases).
The full embedding space has dimension D = rank = φ(n); the physical slice E∥ is
2-dimensional; the internal space E⊥ has dimension d_⊥ = D − 2. A topological
phason defect lives in E⊥ and must "register" on E∥ to exert force.

Step 1 — per-direction alignment. For the internal configuration to be read out
by the physical slice, the Z_p phase along each internal direction must align
with the slice. A single phase in Z_p hits a fixed target with probability 1/p
(one value out of p). This is the substrate's fundamental small parameter — the
SAME 1/p that controls every finite-substrate correction in BPR.

Step 2 — number of directions. The defect must bridge the internal space, so the
bare overlap carries one factor of 1/p per internal dimension: (1/p)^{d_⊥}.

Step 3 — topological protection. The defect's Burgers/winding direction is phase-
coherent by construction (its charge is O(1), unsuppressed — this is the whole
point of going topological). Protection removes τ internal factors, τ ∈ {0,1}:
    overlap amplitude  A ~ (1/p)^{d_⊥ − τ}.

Step 4 — amplitude vs energy. ε is a fraction of the reservoir ENERGY. If ε
tracks the overlap probability |A|², it carries c = 2; if it tracks the amplitude
linearly, c = 1. So
    S ~ (1/p)^{k},     k = c · (d_⊥ − τ),   c ∈ {1,2},  τ ∈ {0,1}.

WHAT THIS DERIVES, AND WHAT IT DOESN'T
--------------------------------------
DERIVED: the FORM is a power law in 1/p (not exponential, not arbitrary), and
the exponent is a small integer fixed by (d_⊥, protection, energy-convention) —
NOT a free 40-order span. That is real progress over casimir_epsilon_toy.

NOT DERIVED: which corner (c, τ). BPR's action does not yet fix the energy
convention or whether protection removes a full internal factor. So k is
BRACKETED, not pinned.

THE HONEST PUNCHLINE
--------------------
For the 9-fold class (d_⊥ = 4) the energy-convention corners are k = 6 and k = 8,
which BRACKET the required k_req ≈ 7 — but clean counting yields only EVEN
exponents in the energy convention, and 7 is ODD. So the projection derivation
reproduces the right ORDER (k between 6 and 8) but cannot produce (1/p)^7
exactly: that needs a half-unit of protection or an amplitude×energy cross-term
that the current action does not supply. The (1/p)^7 numerical match is therefore
confirmed as NOT emerging from clean cut-and-project counting — it sits between
two even structural corners. The gap is reduced from "unknown 40-order S" to
"one missing odd half-power in the defect-core Lagrangian."
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log10

from .phason_sector import (
    embedding_rank,
    required_coupling_efficiency,
    ALLOWED_INTERNAL_CLASSES,
)
from .phason_coupling import P_SUBSTRATE


@dataclass(frozen=True)
class ExponentCorner:
    c: int          # 1 = amplitude convention, 2 = energy (|A|²) convention
    tau: int        # topological protection: 0 (none) or 1 (one internal factor)
    k: int          # resulting exponent k = c·(d_⊥ − τ)
    label: str


def d_perp(n: int) -> int:
    """Internal-space dimension d_⊥ = rank − 2 for the n-fold class."""
    return embedding_rank(n) - 2


def exponent_corners(n: int) -> list[ExponentCorner]:
    """The four defensible exponent values k = c·(d_⊥ − τ) for class n."""
    dp = d_perp(n)
    corners = []
    for c in (1, 2):
        for tau in (0, 1):
            k = c * (dp - tau)
            conv = "energy |A|²" if c == 2 else "amplitude A"
            prot = "protected" if tau == 1 else "unprotected"
            corners.append(ExponentCorner(c, tau, k,
                                          f"{conv}, {prot}: k={k}"))
    return corners


def epsilon_from_exponent(k: int, p: int = P_SUBSTRATE) -> float:
    """S ~ (1/p)^k, taking g²·O ~ O(1) so ε ≈ S (the suppression dominates)."""
    return float(p) ** (-k)


def required_exponent(p: int = P_SUBSTRATE) -> float:
    """The exponent k_req that reproduces ε_required: k_req = −log_p(ε_req)."""
    eps_req = required_coupling_efficiency()
    return log10(eps_req) / log10(1.0 / p)


def class_bracket(n: int, p: int = P_SUBSTRATE) -> dict:
    """Exponent bracket, ε bracket, and the energy-convention (even) corners
    for class n, plus whether k_req falls inside the energy-convention bracket.
    """
    corners = exponent_corners(n)
    ks = [cc.k for cc in corners]
    energy_ks = sorted(cc.k for cc in corners if cc.c == 2)   # the even corners
    k_req = required_exponent(p)
    return {
        "n_fold": n,
        "d_perp": d_perp(n),
        "corners": corners,
        "k_min": min(ks),
        "k_max": max(ks),
        "energy_bracket": (energy_ks[0], energy_ks[-1]),
        "k_required": k_req,
        "required_inside_energy_bracket": energy_ks[0] <= k_req <= energy_ks[-1],
        "required_is_odd_between_even_corners": (
            energy_ks[0] <= k_req <= energy_ks[-1]
            and all(kk % 2 == 0 for kk in energy_ks)
            and not (abs(k_req - round(k_req)) < 0.05 and round(k_req) % 2 == 0)
        ),
        "eps_at_k_min": epsilon_from_exponent(max(ks), p),   # smallest ε
        "eps_at_k_max": epsilon_from_exponent(min(ks), p),   # largest ε
    }


def derivation_report(p: int = P_SUBSTRATE) -> str:
    k_req = required_exponent(p)
    lines = [
        "Phason coupling exponent — derivation attempt (cut-and-project)",
        "==============================================================",
        f"Substrate p = {p};  required exponent k_req = {k_req:.2f}  "
        f"(ε_req = {required_coupling_efficiency():.1e})",
        "",
        "Derived FORM:  S ~ (1/p)^k,   k = c·(d_⊥ − τ),  c∈{1,2}, τ∈{0,1}",
        "  (power law in 1/p, exponent a small integer — NOT a 40-order span)",
        "",
    ]
    for n in ALLOWED_INTERNAL_CLASSES:
        b = class_bracket(n, p)
        inside = "YES" if b["required_inside_energy_bracket"] else "no"
        lines.append(
            f"  {n:>2}-fold (d_⊥={b['d_perp']}): k corners {sorted(set(cc.k for cc in b['corners']))}"
            f"  energy-bracket {b['energy_bracket']}  k_req inside? {inside}"
        )
    lines += [
        "",
        "PUNCHLINE (9-fold, the propulsion-relevant class):",
        "  energy-convention corners are k=6 and k=8; k_req≈7 sits BETWEEN them.",
        "  Clean counting gives only EVEN exponents in the energy convention, but",
        "  7 is ODD. So the projection derivation reproduces the right ORDER",
        "  (6 ≤ k_req ≤ 8) yet CANNOT produce (1/p)^7 exactly — that needs a",
        "  half-unit of protection or an amplitude×energy cross-term the current",
        "  defect-core Lagrangian does not supply.",
        "",
        "STATUS: PARTIAL. Form derived; order derived; exact exponent NOT pinned.",
        "The (1/p)^7 match is confirmed to fall between two even structural",
        "corners — a coincidence at the integer level, not a clean prediction.",
        "Remaining task: the defect-core Lagrangian fixing (c, τ).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(derivation_report())
