"""Casimir constraint on the δ=2 deviation amplitude ε — what the data allows.

The BPR prediction is a fractional Casimir-force deviation

    ΔF/F = ε · (R/R_f)^(−δ),   δ = 2,  R_f = 1 µm,

so at sub-micron separation the deviation is *amplified* by (R_f/R)². A
precision measurement at separation R with fractional accuracy p that sees NO
anomaly bounds the amplitude:

    ε  <  p · (R/R_f)²            (tighter at smaller R).

This module turns published precision-vs-separation figures into that bound,
compares it to the ε needed for levitation (from ``phason_sector``), and
states exactly how wide the open window is and what sensitivity would close it.

NOTE: it makes NO claim of evidence for δ=2. It quantifies what current data
*rules out* and what it *leaves open*. Precision figures are representative
values from the literature (Lamoreaux 1997; Mohideen & Roy 1998; Decca et al.
2003–07; recent 0.2–8 µm work), not a rigorous meta-analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

R_F_NM = 1000.0   # reference fractal scale R_f = 1 µm


@dataclass(frozen=True)
class CasimirPoint:
    separation_nm: float
    frac_precision: float   # fractional agreement with QED (no anomaly seen)
    reference: str


# Representative published precision points (order-of-magnitude, cited).
DATA = [
    CasimirPoint(62,  0.0175, "Mohideen & Roy 1998 (1.75% @ 62 nm)"),
    CasimirPoint(100, 0.01,   "Mohideen & Roy 1998 (~1%)"),
    CasimirPoint(200, 0.002,  "Decca et al. 2003-07 gradient (~0.2%)"),
    CasimirPoint(300, 0.002,  "Decca et al. (~0.2%)"),
    CasimirPoint(750, 0.005,  "sphere-plate (~0.5%)"),
]


def epsilon_bound(p: CasimirPoint) -> float:
    """Upper bound on ε from a single measurement (no anomaly seen)."""
    return p.frac_precision * (p.separation_nm / R_F_NM) ** 2


def best_bound() -> tuple[float, CasimirPoint]:
    """Tightest current upper bound on ε across the data set."""
    pairs = [(epsilon_bound(p), p) for p in DATA]
    return min(pairs, key=lambda t: t[0])


def required_precision_to_detect(epsilon: float, separation_nm: float) -> float:
    """Fractional precision an experiment needs to see a deviation of
    amplitude ε at the given separation: ΔF/F = ε·(R_f/R)²."""
    return epsilon * (R_F_NM / separation_nm) ** 2


def lift_required_epsilon() -> float:
    """ε needed for levitation, from the phason-sector derivation."""
    from .phason_sector import required_coupling_efficiency
    return required_coupling_efficiency()  # ~5.6e-36 (1500 kg, K=3, 9-fold)


def report() -> str:
    import math
    bound, pt = best_bound()
    lift = lift_required_epsilon()
    window = math.log10(bound / lift)
    # sensitivity needed to reach the lift floor at the best separation
    need = required_precision_to_detect(lift, pt.separation_nm)
    lines = [
        "Casimir constraint on the δ=2 amplitude ε  (ΔF/F = ε·(R/R_f)^-2)",
        "================================================================",
        "ε upper bound from each published precision point:",
    ]
    for p in DATA:
        lines.append(f"  {p.separation_nm:>4.0f} nm @ {p.frac_precision*100:>4.1f}%  "
                     f"-> ε < {epsilon_bound(p):.1e}   [{p.reference}]")
    lines += [
        "",
        f"Tightest current bound:  ε < {bound:.1e}   (from {pt.separation_nm:.0f} nm)",
        f"Lift-required ε (1500 kg): {lift:.1e}",
        "",
        f"OPEN WINDOW: ~{window:.0f} orders of magnitude.  Any ε in",
        f"  [{lift:.0e}, {bound:.0e}] both LEVITATES and is invisible to today's data.",
        "",
        "Two honest consequences:",
        "  • Current non-detection does NOT preclude levitation (huge window).",
        "  • Casimir can CONFIRM δ=2 only if ε is near the current bound (~1e-4).",
        f"    Reaching the lift floor would need precision ~{need:.0e} at "
        f"{pt.separation_nm:.0f} nm —",
        "    ~30+ orders beyond feasible; Casimir can never probe that regime.",
        "",
        "Verdict: no evidence for δ=2; a real upper bound; a vast open window.",
        "A dedicated residual-hunt (50 nm–1 µm, after QED+patch+roughness",
        "subtraction) either detects δ=2 (if ε large) or narrows the window's top.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
