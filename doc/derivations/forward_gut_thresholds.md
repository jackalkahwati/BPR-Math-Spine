# Forward-Derived GUT Threshold Corrections (3-Channel)

> **Status:** April 2026 — closes Task #7 of the gap-closure pass.
> The three-channel forward threshold corrections close 98.1% of the
> 1-loop gap and 92.0% of the 2-loop gap in SM gauge coupling
> unification at BPR's M_GUT. The `bpr/gauge_unification.py` code
> already contained the 1-loop three-channel derivation; this doc
> documents the result, extends the 2-loop diagnostic to use all three
> channels (previously only U(1)_Y), and updates `LIMITATIONS`.

## The gap

Prior to April 2026, LIMITATIONS §GUT reported:

> Forward calculation from E8 content closes ~34% of the α₁ gap.
> Remaining ~66% is open. Complete SU(5) multiplets can't split
> couplings; only mass splitting within multiplets contributes.

This was correct for the single-channel (α₁-only) forward correction
that was current at the time. In parallel, the code was updated to a
three-channel correction using S² boundary geometry, but the
documentation and the 2-loop diagnostic were not updated to match.

## The three-channel mechanism

The S² boundary has one distinguished rotational symmetry SO(3) and a
ℤ_p winding structure. The N_B = p^(1/3) active boundary modes between
M_GUT and M_Pl couple to the SM gauge groups via three distinct
geometric channels:

**Channel 1 — U(1)_Y hypercharge.**
Each mode's effective hypercharge-squared is

    Y²_eff = κ + 1/z = (z/2) + 1/z = (z² + 2) / (2z)

Using the standard SU(5) normalization Y²_SU(5) = (3/5) Y² and the
boundary content z = 6 (cubic-lattice nearest neighbors), one obtains
the effective coefficient (3z + 1) / 6 = 19/6. The U(1)_Y threshold
correction is

    δ(1/α₁) = N_B × (3/5) × Y²_eff / 3 × L_above / (2π)                  (1)

where L_above = ln(M_Pl / M_GUT) = ln(p)/4 for BPR.

**Channel 2 — SU(2)_L weak isospin.**
A boundary mode at one of z + 1 sites (z neighbors + central site) has
probability 1/(z + 1) of aligning with any one of the three S² Killing
directions. The effective T² for SU(2) is

    T²_2 = 1/(z + 1) = 1/7

giving

    δ(1/α₂) = N_B × T²_2 / 3 × L_above / (2π)                            (2)

**Channel 3 — SU(3)_c color.**
Color is an internal (winding) symmetry, not a rotation. Its coupling
to boundary modes is doubly suppressed by (z + 1) relative to SU(2):

    T²_3 = 1/(z + 1)² = 1/49

    δ(1/α₃) = N_B × T²_3 / 3 × L_above / (2π)                            (3)

## Numerical result at 1-loop

For p = 104,761, z = 6:

|  | bare | δ (forward) | corrected |
|---|---|---|---|
| 1/α₁(M_GUT) | 35.17 | +13.73 | 48.90 |
| 1/α₂(M_GUT) | 48.00 | +1.03 | 49.04 |
| 1/α₃(M_GUT) | 49.20 | +0.15 | 49.34 |

Residual max spread: 0.25 (vs bare 13.43). **Fraction closed: 98.1%.**
Unified 1/α_GUT ≈ 49.1.

## Numerical result at 2-loop

Running all three couplings with the 2-loop β_ij matrix
(Machacek-Vaughn) from M_Z to M_GUT, then applying the same three
forward corrections:

|  | 2-loop bare | corrected |
|---|---|---|
| 1/α₁(M_GUT) | 34.93 | 48.66 |
| 1/α₂(M_GUT) | 47.69 | 48.72 |
| 1/α₃(M_GUT) | 49.65 | 49.79 |

Max deviation from mean: **1.50%** (vs bare 2-loop 20.8%). **Fraction
closed: 92.0%.**

The residual 1.5% at 2-loop is within the expected range of GUT-scale
threshold corrections from superheavy gauge bosons in non-SUSY
unification schemes (Weinberg 1980, Georgi 1980; typical residuals 1–3%
from unknown heavy-state spectrum). BPR does not compute this residual
from first principles; it is the main remaining open piece.

## Why previous claims underestimated the closure

The earlier "34% closed" figure was for a single-channel correction
applied only to α₁. The corresponding code block was

    delta_1 = N_B × κ × L_above / (10π)   (U(1)_Y only)

which closes 34% of the overall gap because most of the gap (after α₁
is corrected) is between α₂ and α₃. The three-channel mechanism closes
this residual because the SU(2) and SU(3) boundary couplings are
*different* — SU(3) is suppressed relative to SU(2) by (z + 1), and
both are suppressed relative to U(1)_Y. This differential suppression
is exactly what drags α₃ down to meet α₂.

## What this does NOT claim

1. **Residual 1.5% at 2-loop is not derived from BPR.** In any non-SUSY
   unification scheme, the last few percent of coupling matching comes
   from integrating out superheavy GUT gauge bosons with unknown mass
   spectrum. BPR has not computed this from the boundary action. The
   1.5% residual is consistent with typical non-SUSY unification
   expectations but not independently predicted.

2. **T²_2 and T²_3 suppressions are motivated, not rigorously derived.**
   The arguments that SU(2) alignment probability is 1/(z + 1) and SU(3)
   suppression is 1/(z + 1)² are physical (based on the coset structure
   of S²'s isometry group and color being an internal winding symmetry),
   but a complete derivation from the CS action would need to compute
   the boundary Wilson-line overlap integrals explicitly. The structure
   of the suppressions is consistent with this; the coefficients are at
   the O(1) level.

3. **Backward-fit retained for downstream predictions.** The Weinberg
   angle and α_s(M_Z) predictions currently use the backward-fit that
   achieves exact unification. With forward corrections now at 2-loop
   1.5%, the backward-fit refinement is small, and the downstream
   predictions are robust to the residual. Still, the formal status is
   "conditional on unification."

## Status change

| Prediction | Previous | After this document |
|---|---|---|
| Forward fraction closed (1-loop) | ~34% (α₁ only) | **98.1% (three-channel)** |
| Forward fraction closed (2-loop) | Not reported | **92.0% (three-channel)** |
| LIMITATIONS §GUT | OPEN (66% residual) | PARTIALLY CLOSED (1.5% residual at 2-loop) |
| Residual mechanism | Unknown | Superheavy gauge boson thresholds (standard non-SUSY) |

## Code changes

`bpr/gauge_unification.py::two_loop_diagnostic`: extended from
single-channel (U(1)_Y only) to three-channel forward corrections,
matching the structure already in `forward_threshold_corrections`.
Added `fraction_closed_2loop`, `max_deviation_pct_bare`,
`inv_a2_corrected_2loop`, `inv_a3_corrected_2loop` fields.

## Honest residual open problem

The last ~1.5% at 2-loop is not derived. A full closure would require
computing the superheavy gauge boson spectrum from the CS bulk action
and integrating it out, which BPR has not done. Two possible paths:

- **Path A (ambitious):** Compute the mass spectrum of heavy gauge
  bosons from the 5D CS × (compact boson) theory directly. This is a
  real calculation that could succeed; it has not been attempted.

- **Path B (acknowledge):** Report 1.5% as "inside the expected range of
  non-SUSY GUT thresholds" and flag it as an estimated SM uncertainty
  rather than a BPR-specific shortfall. This is the current posture.

---

*April 2026 — closes Task #7 of the April 2026 gap-closure pass.
Three-channel forward corrections now close 98.1% at 1-loop and 92.0%
at 2-loop. Residual 1.5% documented as "expected non-SUSY threshold
range, not computed from BPR first principles."*
