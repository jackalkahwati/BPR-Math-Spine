"""Phason coupling — the two honest attacks on the 31-order undecidability gap.

CONTEXT
-------
``phason_sector.phason_defect_lift_budget`` reduces controlled phason lift to a
single number ε (the fraction of the J⁴ reservoir a topological defect extracts)
and reports an UNDECIDABLE verdict: ε_required ≈ 6e-36 sits ~31 orders below the
experimental sensitivity ε_bound ≈ 5e-5 of the δ=2 Casimir amplitude. The
reservoir is NOT the problem (J⁴/(ℏc)³ is ~1e35× the lift need); the coupling ε
is simply underdetermined and below experimental reach.

Two — and only two — honest routes close that gap:
  (1) DERIVE ε from first principles, so it need not be measured.
  (2) ENUMERATE the Postulate 0c modes more carefully, in case a channel with
      stronger (less suppressed) coupling was missed.

This module attacks both and reports what actually comes out — including the
negative results. Nothing here is tuned to reach 6e-36; doing so would be a
retrofit of exactly the kind ``doc/CLOSED_AND_DEPRECATED.md`` exists to refuse.

RESULTS (honest)
----------------
  * MODE CENSUS (route 2): NEGATIVE / CLOSED. The phason order-parameter space
    is the internal torus T^{d_⊥}, d_⊥ = rank − 2. A torus has π_1 = ℤ^{d_⊥}
    and *vanishing* higher homotopy (π_k = 0, k ≥ 2), because its universal
    cover ℝ^{d_⊥} is contractible. So the complete topological-defect content
    is the π_1 dislocation lattice already in ``topological_charge_capacity``.
    There is NO point-defect, texture, or domain-wall channel to find. Careful
    enumeration does not rescue the budget — it closes the "missing mode" hope.

  * SCALE LADDER (route 1): UNDERDETERMINED / OPEN, now precisely characterised.
    Reaching ε_required ≈ 6e-36 from BPR's natural small parameters requires a
    HIGH power: ~7 powers of 1/p, or ~3.5 powers of 1/p², or one gravitational
    coupling at extreme suppression. No BPR derivation currently predicts that
    power. The ε ≈ (1/p)^7 coincidence is flagged as a RETROFIT RISK, not a
    result. The gap stays open; it is now a sharp, single missing derivation
    (the defect→slice coupling power) rather than a hand-wave.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log10

from .phason_sector import (
    embedding_rank,
    topological_charge_capacity,
    required_coupling_efficiency,
)

# BPR substrate constants (the natural small parameters available to ε).
P_SUBSTRATE = 104761            # prime modular substrate size
Z_PHASE_MODE = 6               # integer boundary phase mode
ALPHA_EM = 7.2973525693e-3     # fine-structure constant
M_E_OVER_M_TAU = 0.510998950 / 1776.86   # lightest/heaviest charged-lepton ratio


# ---------------------------------------------------------------------------
# Route 2: homotopy census of the Postulate 0c internal-space modes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeChannel:
    """A candidate dynamical / topological channel on the internal space."""
    name: str
    homotopy_group: str        # the classifying group, as text
    rank_of_group: int         # number of independent ℤ generators (0 if trivial)
    couples_to_lift: bool      # can it source a reactionless force?
    note: str


def torus_homotopy_rank(k: int, d_perp: int) -> int:
    """Rank of π_k(T^{d_⊥}) — the number of independent ℤ generators.

    A torus T^{d_⊥} = (S¹)^{d_⊥} has
        π_1(T^{d_⊥}) = ℤ^{d_⊥}   (rank d_⊥),
        π_k(T^{d_⊥}) = 0  for k ≥ 2,
    because its universal cover ℝ^{d_⊥} is contractible (all higher homotopy of
    a contractible space vanishes, and the covering map is a π_{k≥2} iso). Also
    π_0 = 0 (a torus is connected). This is exact, not an approximation.
    """
    if k == 0:
        return 0           # connected: no domain-wall (π_0) defects
    if k == 1:
        return d_perp      # dislocation lattice: ℤ^{d_⊥}
    return 0               # π_{k≥2}(T^n) = 0 — no point defects / textures


def postulate0c_mode_census(n: int = 9) -> dict:
    """Enumerate EVERY mode/defect class the internal space supports.

    The physical slice is 2D, so phason defects in 3D physical space are
    classified by the homotopy of the internal order-parameter space, which is
    the internal torus T^{d_⊥} with d_⊥ = rank − 2 (rank = φ(n) embedding dim).

    Topological-defect ↔ homotopy dictionary (Mermin, Rev. Mod. Phys. 1979):
        domain walls (codim 1)  ↔ π_0
        line defects (codim 2)  ↔ π_1   ← phason dislocations, the lift channel
        point defects (codim 3) ↔ π_2
        textures                ↔ π_3

    Returns the full channel list plus the decisive fact: only π_1 is nonzero,
    so the dislocation sector is the COMPLETE topological content.
    """
    rank = embedding_rank(n)
    d_perp = rank - 2
    channels = [
        ModeChannel(
            "phonon (physical slice)", "—", 0, False,
            "gapless Goldstone of broken translation in E∥; the sector BPR "
            "already has. Perturbative, K-suppressed — does not lift.",
        ),
        ModeChannel(
            "phason (smooth, internal)", "—", 0, False,
            "gapless internal-space mode; softer than phonon (K≈0.05 C). "
            "Perturbative channel makes the force smaller, not larger.",
        ),
        ModeChannel(
            "domain wall (π_0)", "π_0(T^{d_⊥}) = 0", 0, False,
            "torus is connected → no wall defects.",
        ),
        ModeChannel(
            "phason dislocation (π_1)", f"π_1(T^{d_perp}) = Z^{d_perp}", d_perp,
            True,
            "THE lift channel: O(1) quantized Burgers charge, no K/λ "
            "suppression. χ = d_⊥ independent axes.",
        ),
        ModeChannel(
            "point defect (π_2)", f"π_2(T^{d_perp}) = 0", 0, False,
            "vanishes for a torus → no monopole-like channel.",
        ),
        ModeChannel(
            "texture (π_3)", f"π_3(T^{d_perp}) = 0", 0, False,
            "vanishes for a torus → no skyrmion-like channel.",
        ),
    ]
    lift_channels = [c for c in channels if c.couples_to_lift]
    return {
        "n_fold": n,
        "embedding_rank": rank,
        "d_perp": d_perp,
        "channels": channels,
        "n_lift_channels": len(lift_channels),
        "total_topological_charge_rank": topological_charge_capacity(rank),
        "verdict": (
            "COMPLETE: the internal torus has nonzero homotopy ONLY at π_1 "
            f"(= Z^{d_perp}). The phason-dislocation sector already in the code "
            "is the entire topological-defect content; careful enumeration "
            "finds NO new lift channel. Route 2 closes negatively."
        ),
    }


# ---------------------------------------------------------------------------
# Route 1: substrate-scale ladder for the coupling efficiency ε
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleHypothesis:
    """A candidate first-principles value for ε, built from natural BPR scales."""
    label: str
    epsilon: float
    derived: bool              # True only if BPR actually predicts this power
    note: str


def epsilon_scale_ladder() -> dict:
    """Tabulate the ε that natural BPR small-parameters would give, and compare
    to ε_required. The point is to show ε spans dozens of orders depending on an
    UNDERIVED power, so matching 6e-36 is a retrofit unless the power is derived.

    None of these is marked ``derived=True``: BPR does not currently predict the
    coupling power. That is precisely the one missing derivation.
    """
    eps_required = required_coupling_efficiency()   # ≈ 6e-36 (from J⁴ reservoir)

    inv_p = 1.0 / P_SUBSTRATE
    hypotheses = [
        ScaleHypothesis("ε ~ 1 (no suppression)", 1.0, False,
                        "unphysical: would lift trivially and be ubiquitous."),
        ScaleHypothesis("ε ~ α_em", ALPHA_EM, False,
                        "single EM-strength coupling; far too large."),
        ScaleHypothesis("ε ~ 1/p", inv_p, False,
                        "one factor of the substrate modular size."),
        ScaleHypothesis("ε ~ 1/p²", inv_p ** 2, False,
                        "two factors (slice×slice overlap)."),
        ScaleHypothesis("ε ~ (m_e/m_τ)", M_E_OVER_M_TAU, False,
                        "lepton-hierarchy-strength suppression."),
        ScaleHypothesis("ε ~ (1/p)^7", inv_p ** 7, False,
                        "RETROFIT RISK: numerically near ε_required, but no "
                        "BPR derivation predicts the exponent 7."),
    ]

    # The exponent of 1/p that WOULD reproduce ε_required (diagnostic only).
    power_needed = log10(eps_required) / log10(inv_p)

    rows = []
    for h in hypotheses:
        gap = log10(h.epsilon / eps_required)   # +: above need, -: below
        rows.append({
            "label": h.label,
            "epsilon": h.epsilon,
            "orders_vs_required": gap,
            "derived": h.derived,
            "note": h.note,
        })

    return {
        "eps_required": eps_required,
        "power_of_inv_p_needed": power_needed,   # ≈ 7.0
        "ladder": rows,
        "any_derived": any(h.derived for h in hypotheses),
        "verdict": (
            f"ε_required ≈ {eps_required:.1e} corresponds to (1/p)^{power_needed:.2f}. "
            "Natural single-factor suppressions (α, 1/p, 1/p², lepton ratio) are "
            "all FAR too large; reaching 6e-36 needs a high, specific power that "
            "no BPR derivation predicts. The (1/p)^7 match is a retrofit risk, "
            "NOT a result. Route 1 stays OPEN: the missing piece is a derivation "
            "of the defect→slice coupling power from (p, z, J)."
        ),
    }


# ---------------------------------------------------------------------------
# Combined honest verdict
# ---------------------------------------------------------------------------

def gap_attack_report(n: int = 9) -> str:
    """Plain-text summary of both attacks on the 31-order undecidability gap."""
    census = postulate0c_mode_census(n)
    ladder = epsilon_scale_ladder()
    lines = [
        "Attacking the 31-order phason-lift undecidability gap — two routes",
        "==================================================================",
        "",
        "ROUTE 2 — enumerate Postulate 0c modes more carefully:",
        f"  internal space = torus T^{census['d_perp']} (rank {census['embedding_rank']}, "
        f"{n}-fold).",
        f"  nonzero homotopy: π_1 = Z^{census['d_perp']} only; π_0 = π_2 = π_3 = 0.",
        f"  lift channels found: {census['n_lift_channels']} (the known dislocation "
        "sector).",
        "  VERDICT: NEGATIVE — no new channel. The dislocation sector is complete.",
        "",
        "ROUTE 1 — derive the coupling ε instead of measuring it:",
        f"  ε_required ≈ {ladder['eps_required']:.1e}  =  "
        f"(1/p)^{ladder['power_of_inv_p_needed']:.2f}.",
        "  natural single-factor suppressions are all >25 orders too large.",
        "  any hypothesis actually DERIVED by BPR? "
        f"{'yes' if ladder['any_derived'] else 'no'}.",
        "  VERDICT: OPEN — one sharp missing derivation (the coupling power);",
        "           matching (1/p)^7 by hand would be a retrofit.",
        "",
        "NET: the gap does NOT close. Route 2 is a clean negative (mode census",
        "complete). Route 1 reduces the whole question to a single underived",
        "number — the defect→slice coupling power — which is the honest statement",
        "of what BPR is missing. Energy is not the issue; the coupling is.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(gap_attack_report())
