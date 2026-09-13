# Inflation Potential from the Boundary Action

> **Normalization correction, 2026-09-12:** The operative formulas below follow
> `gravity_consistency_2026-09-12.md`. For the stated action with reduced
> `M_Pl`, the scalar mass is `M_Pl/sqrt(6α)`, the plateau is `M_Pl⁴/(8α)` and
> `A_s = N²/(144π²α)` at leading large-`N` slow roll. This supersedes the
> old `1/(2sqrt(α))`, `3/(16α)` and `96π²` normalization, the `5/3`
> percent-level near-match, and any scalar-amplitude closure implication.
> No parameter, enhancement candidate or comparison input is retuned.

## The gap

The April 2026 note promoted P11.2 (`n_s`) and P11.3 (`r`) from FRAMEWORK
on the basis of the scalar dual of an induced `R²` action. The action-to-scalar
map is exact within that supplied action. It does not by itself derive the
boundary-to-bulk action, the inflationary history, or the scalar amplitude
from the Bose ring. The existing e-fold estimate is retained:

    N = p^(1/3)(1 + 1/d) ≈ 62.855   for p = 104761, d = 3

## The derivation

Write the Starobinsky potential using scalar mass `m_s` and reduced `M_Pl`:

    V(φ) = (3 m_s² M_Pl² / 4) (1 − e^(−√(2/3) φ / M_Pl))²                 (1)

### Step 1 — Induced R² term from boundary-mode integration

The inherited boundary calculation supplies the conditional effective action

    S_grav = ∫ d⁴x √(−g) [ (M_Pl²/2) R + (α/2) R² + O(R³) ]              (2)

with the one-loop coefficient

    α_min = (p / 384π²) × κ²,    κ = z/2                                  (3)

For the unchanged `p = 104761`, `z = 6`, this gives `α_min ≈ 248.778`.
The induced Einstein coefficient obeys `M_Pl² = p Λ_b²/(48π²)`, so
`Λ_b = M_Pl sqrt(48π²/p)`, not the old `M_Pl sqrt(p)` cutoff statement.
`M_Pl` here is reduced: `M_Pl² = 1/(8πG)` in natural units. With
`a = ħc/Λ_b` and physical `l_P² = ħG/c³`, `a/l_P = sqrt(p/(6π))`.
Matching the existing physical Planck anchor gives `Λ_b ≈ 1.64 × 10¹⁷ GeV`
at default `p`; this inverse scale matching is calibration, not prediction.
The induced coefficients remain conditional on field content, regulator and
subtraction assumptions.

### Step 2 — Scalar dual of R² gravity

Introduce an auxiliary field `χ` through `αR²/2 -> αχR − αχ²/2`.
For `F = 1 + 2αχ/M_Pl² > 0`, take `g_E = F g_J` and
`φ = sqrt(3/2) M_Pl log F`. Then

    S_scalar = ∫ d⁴x √(−g_E) [ (M_Pl²/2) R_E − (1/2) (∂φ)² − V(φ) ]      (4)

with

    V(φ) = (M_Pl⁴ / (8α)) (1 − e^(−√(2/3) φ / M_Pl))²                    (5)

Expansion about `φ = 0` gives `m_s² = M_Pl²/(6α)`. Thus (5) is the
Starobinsky shape (1), with its normalization fixed by the stated action.

### Step 3 — Scalar amplitude audit

Using `ε ≈ 3/(4N²)` and `A_s = V/(24π² M_Pl⁴ ε)` at leading large `N`,

    A_s = (N²/24π²) × (m_s/M_Pl)² = N²/(144π²α)                           (6)

At the unchanged parameters above,

    m_s/M_Pl ≈ 0.02588
    V0/M_Pl⁴ ≈ 5.025 × 10⁻⁴
    A_s,min ≈ 1.117 × 10⁻²

Retaining the existing comparison input `A_s,obs = 2.1 × 10⁻⁹` gives the
inverse calibration

    α_required = N² / (144π² A_s,obs) ≈ 1.324 × 10⁹                        (7)
    α_required / α_min ≈ 5.321 × 10⁶                                      (8)

This required enhancement is `2/3` of the old `≈ 7.98 × 10⁶` gap. All
fixed candidate/required ratios grow by `3/2`. The old estimate
`1 + sqrt(z/2)/sqrt(log p) ≈ 1.51` remains far too small. The unchanged
compact-boson square-lattice count gives
`418608 log(p) ≈ 4.84 × 10⁶`, or `0.9094` of the corrected requirement.
It remains a mode-count diagnostic, not a derived loop coefficient.

### Result

The conditional pure-`R²` potential shape still gives the leading slow-roll
relations

    n_s = 1 − 2/N,        r = 12/N²                                       (9)

Their normalization independence does not establish an absolute amplitude or
validate the assumed e-fold estimate.

The following April 2026 comparison table is preserved as historical raw
benchmark material, not a newly evaluated or validated prediction table:

| Quantity | Formula | Value | Observed |
|---|---|---|---|
| N_efolds | p^(1/3)(1 + 1/d) | 63 | 55–65 ✓ |
| n_s | 1 − 2/N | 0.968 | 0.9649 ± 0.004 (+0.78σ) |
| r | 12/N² | 0.003 | < 0.044 ✓ |

## Status correction

The historical April promotion of P11.2/P11.3 to DERIVED and “closes Task #10”
is not an unconditional inflation or amplitude closure. The exact result here
is the scalar dual of the supplied action; `n_s` and `r` use leading slow roll
and the existing assumed e-fold prescription. The boundary coefficient and
physical realization remain conditional.

## Open normalization and other caveats

1. The winding/anyon-loop contribution in (8) is not derived from first
   principles. The unchanged full lattice leaves a corrected residual factor
   `≈ 1.0996`, not `1.649`. The existing current-insertion candidate
   `1 + 2/R² = 5/3` gives a coefficient/required ratio `≈ 1.5157`, not
   `1.010`. The former percent-level near-match and candidate closure are
   explicitly superseded; no replacement factor is selected. The CS/WZW
   operator-compatibility argument does not prove either the doubled/non-chiral
   completion or the absolute finite-`p` bulk coefficient.

2. Step 1 uses the conditional one-loop induced-gravity approximation.
   Higher-loop corrections, the regulator and subtraction prescription require
   independent control; inverse matching of `α` to `A_s` is calibration.

3. The potential (5) is the minimal pure-`R²` result. Changing `α` rescales its
   amplitude but leaves leading `n_s` and `r` unchanged. Additional
   `R_μν R^μν` or Weyl-squared contributions need a separate suppression or
   decoupling argument before claiming the same single-field plateau.
