"""TOY estimator of the coupling efficiency ε — MODEL-DEPENDENT, NOT A PREDICTION.

============================  READ THIS FIRST  ============================
This is a TOY. It does NOT predict ε. The rigorous ε requires the full
defect-to-substrate coupling Lagrangian, which BPR has not yet specified, so
any number here is whatever the modeling assumptions put in. Its only honest
job is to show (a) what factors ε decomposes into, (b) how undetermined it is,
and (c) the one real constraint that DOES fall out. Simulating the postulate
cannot validate the postulate (cf. repo commit 9197fa4 on circular sims).
===========================================================================

Decomposition (a modeling choice, not a derivation):

    ε  =  g²  ·  O  ·  S

  g²  phonon–phason coupling squared (dimensionless). In lab quasicrystals
      g = D/√(CK) ~ 0.1–0.7, so g² ~ 1e-2 … 5e-1.   [measurable, bounded]
  O   geometric overlap of the topological defect with the physical slice.
      O(1) if the defect couples fully; small if mostly "internal".  1e-3 … 1.
  S   net projection / decoherence suppression — THE dominant unknown.
      Topological protection argues S ~ O(1); decoherence argues S ≪ 1.
      Spanned over 1e-40 … 1 because it is genuinely undetermined.

The output is therefore a SPAN, classified against two fixed reference values:
  • Casimir bound      ε ≲ 1e-4   (current data; casimir_constraint)
  • lift threshold     ε ≈ 6e-36  (phason_sector.required_coupling_efficiency)
"""
from __future__ import annotations

import numpy as np

# Reference values (imported lazily in functions to avoid import cost at module load)
CASIMIR_BOUND = 1e-4          # current upper bound on ε (see casimir_constraint)

# Plausible factor ranges (log10 bounds). g² and O are loosely grounded;
# S is the dominant unknown, deliberately spanned over 40 orders.
G2_RANGE = (1e-2, 5e-1)       # phonon-phason coupling squared (lab QC)
O_RANGE = (1e-3, 1.0)         # geometric overlap of defect with physical slice
S_RANGE = (1e-40, 1.0)        # net suppression — THE unknown


def epsilon(g2: float, O: float, S: float) -> float:
    """ε = g²·O·S (toy decomposition)."""
    return g2 * O * S


def unsuppressed_epsilon() -> tuple[float, float]:
    """ε with S = 1 (topological, NO suppression): range g²·O.
    If the coupling is unsuppressed, this is where ε sits."""
    lo = G2_RANGE[0] * O_RANGE[0]
    hi = G2_RANGE[1] * O_RANGE[1]
    return lo, hi


def _loguniform(rng, lohi, n):
    lo, hi = np.log10(lohi[0]), np.log10(lohi[1])
    return 10 ** rng.uniform(lo, hi, n)


def monte_carlo(n: int = 200_000, seed: int = 0) -> np.ndarray:
    """Sample ε over the plausible factor ranges (log-uniform)."""
    rng = np.random.default_rng(seed)
    g2 = _loguniform(rng, G2_RANGE, n)
    O = _loguniform(rng, O_RANGE, n)
    S = _loguniform(rng, S_RANGE, n)
    return epsilon(g2, O, S)


def lift_threshold() -> float:
    from .phason_sector import required_coupling_efficiency
    return required_coupling_efficiency()


def classify(eps: np.ndarray) -> dict:
    """Fraction of sampled ε landing in each regime."""
    lift = lift_threshold()
    excluded = float(np.mean(eps > CASIMIR_BOUND))                 # already ruled out
    window = float(np.mean((eps <= CASIMIR_BOUND) & (eps >= lift)))  # lifts & invisible
    nolift = float(np.mean(eps < lift))                            # too weak to lift
    return {"excluded": excluded, "liftable_window": window, "no_lift": nolift,
            "lift_threshold": lift}


def report(n: int = 200_000) -> str:
    eps = monte_carlo(n)
    c = classify(eps)
    us_lo, us_hi = unsuppressed_epsilon()
    pct = np.percentile(eps, [5, 50, 95])
    lines = [
        "TOY ε estimator — MODEL-DEPENDENT, NOT A PREDICTION",
        "===================================================",
        f"ε = g²·O·S sampled log-uniform over plausible ranges (n={n}).",
        f"  ε percentiles (5/50/95): {pct[0]:.1e} / {pct[1]:.1e} / {pct[2]:.1e}",
        f"  (spans ~40 orders — dominated by the unknown suppression S)",
        "",
        "Regime split (vs Casimir bound 1e-4 and lift threshold "
        f"{c['lift_threshold']:.0e}):",
        f"  already EXCLUDED by Casimir (ε>1e-4):     {c['excluded']*100:5.1f}%",
        f"  liftable & invisible window:              {c['liftable_window']*100:5.1f}%",
        f"  too weak to lift (ε<threshold):           {c['no_lift']*100:5.1f}%",
        "",
        "The ONE honest constraint the toy yields (independent of S's value):",
        f"  If the coupling is UNSUPPRESSED (S~1), ε ~ g²·O ~ {us_lo:.0e}–{us_hi:.0e},",
        f"  i.e. AT or ABOVE the Casimir bound (1e-4) -> it would ALREADY be seen.",
        "  Since it has NOT been seen, the model REQUIRES real suppression (S≪1).",
        "  The value of that suppression is exactly what is undetermined.",
        "",
        "Verdict: the toy CANNOT predict ε (it's dominated by an unknown). It only",
        "shows ε is consistent with the liftable window ONLY IF suppression sits in",
        "a specific range — and that ε ~ O(1)·coupling is already excluded. To get a",
        "real number, the defect-coupling Lagrangian must be completed (not toyed).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
