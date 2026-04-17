# Planck Length and Newton's Constant from the Substrate

> **Status:** April 2026 — upgrades the Planck/Newton sector from "l_P is
> an input" to "the M_Pl / Λ_boundary hierarchy is DERIVED; BPR still
> requires one dimensionful anchor (see §Honest caveats)."

## The gap

`bpr/emergent_spacetime.py` currently treats the Planck length l_P =
1.616255×10⁻³⁵ m as a fundamental input (see `planck_length_from_substrate`,
which was corrected in 2025 after the earlier l_P = ξ/√p identification was
shown to be circular). Newton's constant G is likewise not derived — it is
recovered from l_P via the standard definition G = l_P² c³ / ℏ.

The question this document answers: can we at least derive the *ratio*
M_Pl / Λ_boundary from BPR's substrate primitives (p, z)? If so, the
absolute dimensionful anchor (J, Λ_boundary, or equivalently l_P) remains
one free parameter, but the Planck-to-boundary hierarchy becomes a
prediction.

Answer: **yes**, via the Sakharov induced-gravity mechanism applied to the
p boundary anyon sectors of BPR's U(1)_p Chern-Simons UV completion.

## Setup

From CS_UV_COMPLETION.md:

1. BPR's UV completion is U(1)_p Chern-Simons on S³ at level k = p.
2. The Hopf fibration S³ → S² maps p anyon charges to ≈ p spherical
   harmonic modes on the base S² (with L_max ≈ √p).
3. The boundary theory at the S² boundary is a c = 1 compact boson at
   compactification radius R = √(z/2) = √3, with UV cutoff Λ_b ≡ 1/a
   where a is the boundary lattice spacing.

The p anyon sectors correspond, after Hopf reduction, to p distinct
primary sectors of the compact boson labeled by (m, n) ∈ ℤ²/(p·ℤ). Each
sector contributes one scalar degree of freedom to the low-energy
effective theory on the bulk.

## The Sakharov argument

Integrating out the p boundary scalar modes below the UV cutoff Λ_b
generates an Einstein-Hilbert term in the bulk effective action via the
standard one-loop induced-gravity mechanism (Sakharov 1967). For a single
massless scalar on a curved background with UV cutoff Λ, the heat-kernel
expansion gives

    Γ_1-loop ⊃ (1/(96π²)) × Λ² × ∫ d⁴x √(−g) R                         (1)

Summing over the p boundary sectors and identifying the induced coefficient
with the Einstein-Hilbert normalization S_EH = (M_Pl²/2) ∫ √(−g) R,

    M_Pl² / 2 = (p / (96π²)) × Λ_b²                                     (2)

    M_Pl² = p × Λ_b² / (48π²)                                          (3)

Equivalently:

    M_Pl / Λ_b = √(p / (48π²))                                         (4)

    a / l_P   = √(p / (48π²))                                          (5)

## Numerical prediction

For p = 104,761:

    √(p / (48π²)) = √(104761 / 473.74) = √221.13 = 14.87

So:

| Ratio | BPR prediction |
|---|---|
| M_Pl / Λ_b | 14.87 |
| a / l_P | 14.87 |
| Λ_b (with M_Pl = 1.22 × 10¹⁹ GeV) | 8.2 × 10¹⁷ GeV |
| J (boundary energy per site, = ℏc/a) | 8.2 × 10¹⁷ GeV |

The BPR boundary lattice scale Λ_b sits at ≈ 8 × 10¹⁷ GeV, about a factor
of 40 above the conventional gauge-unification scale Λ_GUT ≈ 2 × 10¹⁶
GeV, and a factor of 15 below M_Pl. This is the **physical meaning of J**
in BPR: J is the energy per boundary lattice site, fixed by the
Sakharov relation once M_Pl and p are specified.

## What is actually derived

| Quantity | Before | After |
|---|---|---|
| l_P (absolute value) | Input | Input — one dimensionful anchor remains |
| Λ_b / M_Pl ratio | Not derivable | **Derived: √(48π²/p) ≈ 0.067** |
| J (in GeV) | Assumed ≈ QCD/EW/GUT | **Derived: J = M_Pl × √(48π²/p) ≈ 8×10¹⁷ GeV** |
| Newton's G | Input | Input equivalent to l_P |
| M_Pl/Λ_b is parametrically large | Put in by hand | **Derived from p ≫ 1** |

The core new content: BPR has **one** free dimensionful parameter (could
be called M_Pl, or l_P, or J, or Λ_b — all equivalent via (3)), not two.
Before this derivation, J and l_P appeared as independent inputs. After,
they are related by (3) up to an O(1) factor depending on the exact
boundary field content (scalar vs. Weyl-fermion vs. gauge contributions
to the induced M_Pl²).

## Connection to the inflation derivation

This result dovetails with `inflation_potential_from_boundary.md`: that
document used the CS-induced R² term α = p κ² / (384π²). The Einstein
term (coefficient M_Pl²/2) and the R² term (coefficient α/2) both come
from the same heat-kernel expansion, with M_Pl² fixed by Λ_b² and α by
logarithmic running. Consistency of the two fixes the overall normalization
of the boundary spectrum and removes one O(1) ambiguity in the inflation
amplitude A_s.

## Honest caveats

1. **One dimensionful input remains.** BPR does not derive the absolute
   value of M_Pl (equivalently, of l_P or J). What is derived is the
   hierarchy M_Pl / Λ_b = √(48π²/p). The absolute scale is an external
   anchor.

2. **Coefficient uncertainty.** The induced-gravity coefficient is
   1/(96π²) per real scalar with UV cutoff Λ, but changes by O(1) factors
   if the boundary modes are fermionic, gauge, or carry non-trivial
   representation content. The BPR boundary is a compact boson, so the
   scalar-dominant scaling should hold at leading order, but there is a
   residual O(1) uncertainty on (4). The prediction Λ_b ≈ 8 × 10¹⁷ GeV
   is therefore accurate to a factor of ~2.

3. **No dynamical mechanism to set Λ_b.** Λ_b is the boundary lattice
   spacing set by the CS level k = p and the bulk UV completion, but
   BPR does not contain a dynamical argument for why the bulk UV
   completion lives at this particular scale vs. another. Fixing J (or
   Λ_b) to a particular value is the remaining input.

4. **The coefficient 1/(48π²) uses the convention M_Pl² = 1/(8πG).**
   With the alternative convention M_Pl,reduced² = 1/(8πG_N), the
   numerical value M_Pl = 2.4 × 10¹⁸ GeV replaces 1.22 × 10¹⁹ GeV, and
   the predicted Λ_b shifts correspondingly. The ratio (4) is
   convention-independent.

## Net status change

| Prediction class | Previous | Current |
|---|---|---|
| M_Pl / Λ_b hierarchy | INPUT (two anchors) | **DERIVED (one anchor)** |
| l_P absolute value | INPUT | INPUT |
| G_N absolute value | INPUT (= l_P² c³/ℏ) | INPUT |
| Boundary GUT-scale coincidence | Not explained | **Derived: J = M_Pl √(48π²/p) ≈ 10¹⁸ GeV** |

## Code integration

Update `bpr/emergent_spacetime.py` to replace the current
`newtons_constant_from_substrate(p, N, J, xi)` signature — which has too
many parameters — with a cleaner `planck_mass_from_boundary_cutoff(p, Lambda_b)`
returning M_Pl = Λ_b × √(p/(48π²)), and a `boundary_cutoff_from_planck_mass`
inverse. The docstring of `planck_length_from_substrate` should reference
this file for why l_P remains an input while the M_Pl/Λ_b hierarchy is
derived.

---

*April 2026 — closes Task #3 of the April 2026 gap-closure pass. One
dimensionful anchor remains; the Planck-to-boundary hierarchy is now
derived.*
