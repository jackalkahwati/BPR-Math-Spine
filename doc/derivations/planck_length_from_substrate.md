# Planck Length and Newton's Constant from the Substrate

> **Normalization corrected 2026-09-12:** the displayed induced Einstein coefficient defines a **reduced** Planck energy M_Pl. The conditional mass/cutoff ratio remains unchanged, but a/l_P gains sqrt(8pi). Absolute G/l_P remains an input. Bare terms, counterterms, regulator and field-content assumptions prevent treating the induced-only ratio as a model-independent prediction. See [action consistency and identifiability](gravity_consistency_2026-09-12.md).

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

Conditional answer: the stated induced-only calculation relates these scales if the supplied boundary-to-bulk field identification, coefficient and absence of independent gravitational terms are assumed. Those assumptions are not established by the ratio calculation.

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

    M_Pl² = 1/(8πG),  M_unreduced = √(8π) M_Pl
    a / l_P = √(p / (6π))                                               (5)

Here (2) defines reduced M_Pl, with energy units when hbar and c are restored. Physical l_P=hbar*c/(sqrt(8pi)*M_Pl), not hbar*c/M_Pl. The two ratios (4) and (5) are not equal.

## Numerical prediction

For p = 104,761:

    √(p / (48π²)) = √(104761 / 473.74) = √221.13 = 14.87

So:

| Ratio | Conditional induced-only conversion |
|---|---|
| Reduced M_Pl / Λ_b | 14.87 |
| a / l_P | 74.55 |
| Λ_b (with reduced M_Pl approximately 2.435 × 10¹⁸ GeV) | approximately 1.64 × 10¹⁷ GeV |
| J defined as boundary cutoff energy = ℏc/a | approximately 1.64 × 10¹⁷ GeV |

These values use an externally anchored physical Planck length and the stipulated induced-only coefficient. Identifying a substrate site energy J with this cutoff is an additional interpretation, not a measured or dynamically selected scale. The former 8.2 × 10¹⁷ GeV value mixed reduced and unreduced conventions and is superseded; no gauge-unification coincidence is derived.

## What is actually derived

| Quantity | Before | After |
|---|---|---|
| l_P (absolute value) | Input | Input — one dimensionful anchor remains |
| Λ_b / M_Pl ratio | Not derivable | **Derived: √(48π²/p) ≈ 0.067** |
| Cutoff energy (in GeV) | Independent scale | Conditional calibrated conversion: Λ_b = reduced M_Pl × √(48π²/p) ≈ 1.64×10¹⁷ GeV |
| Newton's G | Input | Input equivalent to l_P |
| M_Pl/Λ_b is parametrically large | Put in by hand | **Derived from p ≫ 1** |

Within the induced-only ansatz (3), one dimensionful anchor relates reduced M_Pl, physical l_P and Λ_b. This parameter-count statement assumes no independent bare Einstein coefficient or counterterm. More generally M_eff²=b+c+pΛ_b²/(48π²), and compensating changes in b,c,Λ_b leave the effective coefficient invariant. The module5 identifiability calculation exhibits those transformations explicitly. Field-content and regulator assumptions are not fixed by an observed gravitational coefficient.

## Connection to the inflation derivation

This result dovetails with `inflation_potential_from_boundary.md`: that
document used the CS-induced R² term α = p κ² / (384π²). The Einstein
term (coefficient M_Pl²/2) and the R² term (coefficient α/2) both come
from the same heat-kernel expansion, with M_Pl² fixed by Λ_b² and α by
logarithmic running in the supplied calculation. This does not fix independent counterterms or remove the scalar-amplitude calibration. For the displayed alpha/2 convention the action-consistent leading amplitude is A_s=Ne²/(144π² alpha), not the historical96π² formula; old numerical enhancement matches are superseded.

## Honest caveats

1. **One dimensionful input remains.** BPR does not derive the absolute
   value of M_Pl (equivalently, of l_P or J). What is derived is the
   conditional hierarchy reduced M_Pl / Λ_b = √(p/(48π²)). The absolute scale is an external
   anchor.

2. **Coefficient uncertainty.** The induced-gravity coefficient is
   1/(96π²) per real scalar with UV cutoff Λ, but changes by O(1) factors
   if the boundary modes are fermionic, gauge, or carry non-trivial
   representation content. The BPR boundary is a compact boson, so the
   scalar-dominant scaling should hold at leading order, but there is a
   regulator and field-content dependence in (4). No quantified factor-of-two uncertainty follows without specifying and bounding those choices. The corrected induced-only conversion is approximately1.64 × 10¹⁷ GeV, not a calibrated uncertainty interval.

3. **No dynamical mechanism to set Λ_b.** Λ_b is the boundary lattice
   spacing set by the CS level k = p and the bulk UV completion, but
   BPR does not contain a dynamical argument for why the bulk UV
   completion lives at this particular scale vs. another. Fixing J (or
   Λ_b) to a particular value is the remaining input.

4. **The coefficient 1/(48π²) defines reduced M_Pl² = 1/(8πG).**
   The unreduced mass satisfies M_unreduced²=1/G=8π M_Pl². Reexpressing (3) using it changes the coefficient to p/(6π). A physical cutoff is convention-independent only when the coefficient and mass definition are converted together. The two definitions previously described here as alternatives were identical; the numerical examples had instead mixed reduced and unreduced energies.

## Net status change

| Prediction class | Previous | Current |
|---|---|---|
| Reduced M_Pl / Λ_b hierarchy | Independent scales | Conditional induced-only relation; bare/counterterm freedom remains |
| l_P absolute value | INPUT | INPUT |
| G_N absolute value | INPUT (= l_P² c³/ℏ) | INPUT |
| Boundary cutoff | Independent input | Conditional conversion from reduced M_Pl; no GUT-scale coincidence established |

## Code integration

The existing `bpr/emergent_spacetime.py` helpers `planck_mass_from_boundary_cutoff(p, Lambda_b)` and `boundary_cutoff_from_planck_mass` use reduced energy units. `newtons_constant_from_substrate(p, Lambda_b)` uses joules and G=hbar*c^5/(8pi*M_Pl²). `planck_length_from_substrate` retains the external physical anchor. Module5 repairs boundary-spacing and direct caller conversions consistently; inverse matching remains calibration.

---

*Original note April2026; normalization and claim scope corrected September12,2026. The ratio is conditional on the stated induced-only model, not a determination of gravitational dynamics or its absolute scale.*
