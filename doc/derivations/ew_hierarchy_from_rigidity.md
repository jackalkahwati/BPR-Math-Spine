# Electroweak Hierarchy M_Pl / v_EW from Boundary Rigidity × Mode Count

> **Boundary normalization correction, 2026-09-12:** The April 2026 EW
> formulas and raw benchmark table are retained as history, not re-derived or
> revalidated here. Their `M_Pl = 1.22e19 GeV` anchor is unreduced, unlike the
> reduced coefficient in the gravitational action. The copied Sakharov
> cross-check and cutoff conversions below are corrected using
> `gravity_consistency_2026-09-12.md`. The former independent sub-percent
> cross-check and “closes Task #4” implications are superseded. This repair
> does not change EW physics, benchmark results or code status flags.

## Historical EW claim and raw benchmark (not revalidated)

In this historical section only, `M_Pl` denotes the unreduced Planck energy.
The formulas, benchmark values and explanatory claims are preserved as the
April record; their current prediction authority is not established by this
normalization repair.

    M_Pl / v_EW = p^(z/2 + 1/3) × ln(p) / (ln(p) + 1)                    (1)

For the BPR default (p = 104,761, z = 6):

| Quantity | Value |
|---|---|
| p^(10/3) (bare) | 5.41 × 10¹⁶ (9.1% off) |
| p^(10/3) × ln(p)/(ln(p)+1) (corrected) | 4.988 × 10¹⁶ (0.51% off) |
| Observed M_Pl / v_EW | 4.963 × 10¹⁶ |

The exponent 10/3 = z/2 + 1/3 factors into two substrate-derived pieces,
each of which appears independently elsewhere in BPR:

**Factor p^(z/2) — boundary rigidity amplification.** The boundary
rigidity κ = z/2 = 3 is the effective stiffness of the S² surface under
coherent deformation. It is the same κ that sets the compactification
radius R = √(z/2) = √3 in `CS_UV_COMPLETION.md` §3, the same κ that
appears in the inflation R² coefficient α ∝ κ² in
`inflation_potential_from_boundary.md`, and the same κ that appears in
the winding-sector correction to A_s. Each unit of rigidity contributes
a factor of p to the gravitational self-coupling suppression, because
coherent deformation against a rigid boundary costs p units of energy per
rigidity unit. Three units of rigidity → p³ suppression.

**Factor p^(1/3) — active boundary mode count between M_GUT and M_Pl.**
The cube root appears because the active-mode count scales with the
linear size of the boundary lattice: N_boundary = p^(1/3) in three
bulk spatial directions. This is the same N_B used in
`gauge_unification.py::GaugeCouplingRunning.n_boundary_modes`, the same
p^(1/3) in the BPR v_EW formula v = Λ_QCD × p^(1/3) × (ln p + z − 2),
and the same p^(1/3) in n_efolds = p^(1/3)(1 + 1/d).

**Correction ln(p)/(ln(p)+1).** The active fraction of boundary degrees
of freedom is not all ln(p) entropy modes — the ground state (the
winding-zero sector) does not couple to gravitational deformation, so
the active fraction is ln(p)/(ln(p)+1) ≈ 0.920 for p = 104,761. This
~8% correction brings the prediction from 9% off to 0.5% off.

## Corrected boundary conversion; historical cross-check superseded

Let `M` denote the reduced coefficient of `M²R/2`, and `M_P` the
unreduced energy used by the historical EW table. The inherited conditional
Sakharov coefficient in `planck_length_from_substrate.md` is

    M / Λ_b = sqrt(p/(48π²)),    M_P = sqrt(8π) M                         (2)
    M_P / Λ_b = sqrt(p/(6π))
    a/l_P = sqrt(p/(6π)),    a = ħc/Λ_b,    l_P² = ħG/c³

Thus `M_P = 1.22 × 10¹⁹ GeV` corresponds to reduced
`M ≈ 2.43 × 10¹⁸ GeV` and `Λ_b ≈ 1.64 × 10¹⁷ GeV` at default `p`.
The old `8.2 × 10¹⁷ GeV` cutoff inserted the unreduced scale into the
reduced formula. Matching this physical anchor is inverse calibration, not
an independent prediction of an absolute gravitational scale.

If the historical EW ansatz (1) is retained without modification, its
purely algebraic combination with the corrected conversion is

    v_EW / Λ_b = (M_P / Λ_b) / (M_P / v_EW)
              = sqrt(p/(6π)) / (p^(10/3) × ln(p)/(ln(p)+1))
              = p^(−17/6) / (sqrt(6π) × ln(p)/(ln(p)+1))                  (3)

This corrects the copied unit conversion only; it does not validate (1) or
supply an independent EW cross-check.

**Superseded historical comparison, not a current prediction:** the April
text reported `v_EW/Λ_b = 2.98 × 10⁻¹⁶`, an observed comparator
`3.00 × 10⁻¹⁶`, and ratio `0.993` (0.7% agreement), using
`v_EW = 246 GeV` and the old `Λ_b = 8.2 × 10¹⁷ GeV`. Those raw reported
numbers are retained here only as history. The common unreduced/reduced
convention error invalidates their use as an independent consistency check.
No dependent EW benchmark table is recalculated and no replacement agreement
claim is made.

## Historical EW caveats (not revalidated by this repair)

The following caveats record the original EW interpretation. Only the copied
boundary cutoff and scale equivalence are corrected here; no EW dynamics or
hierarchy derivation is assessed.

1. **Does not solve the hierarchy problem's fine-tuning aspect.** The
   formula (1) explains the *numerical value* of M_Pl / v_EW from
   substrate primitives, but the standard naturalness argument about
   quadratic divergences in the Higgs mass is a separate question.
   BPR's response to the fine-tuning problem is that the Higgs is itself
   a boundary mode, so its quadratic divergences are cut off at the
   boundary scale. The copied gravity conversion now gives
   `Λ_b ≈ 1.64 × 10¹⁷ GeV`, rather than the old `8 × 10¹⁷ GeV`;
   unreduced `M_P ≈ 1.2 × 10¹⁹ GeV` and reduced `M ≈ 2.43 × 10¹⁸ GeV`
   are distinct scales. The historical residual-hierarchy estimate and
   naturalness interpretation are not re-evaluated here.

2. **Does not derive v_EW itself.** v_EW is derived from Λ_QCD via
   v_EW = Λ_QCD × p^(1/3) × (ln p + z − 2), which uses Λ_QCD as input.
   The chain is:
   - Λ_QCD: one-flavor QCD scale — external anchor
   - v_EW: derived from Λ_QCD via boundary formula (Task #8)
   - Historical unreduced M_Pl: the EW ansatz uses (1); equation (2) instead
     supplies a conditional reduced coefficient and is not an equivalent
     independent absolute-scale prediction

3. **Correction factor ln(p)/(ln(p)+1) is motivated, not fully derived.**
   The argument that the ground state does not couple to gravitational
   deformation is physical, but the precise functional form of the
   correction is a leading-order approximation. The *structure* (O(1/ln p)
   shift from bare p^(10/3)) is correct; the *coefficient* in that
   structure matches observation to 0.5%.

## Historical status table (closure implications superseded)

The original table is preserved below as an April 2026 status record, not a
current claim that this gravity normalization repair establishes the EW
hierarchy. Existing code flags are outside this repair's scope.

| Prediction | Previous (LIMITATIONS as of April 2026) | Historical April status |
|---|---|---|
| M_Pl / v_EW value | OPEN | **DERIVED (0.5% off)** |
| hierarchy_derived flag in code | already True | Consistent with docs |
| Fine-tuning / naturalness | Framework argument | Same — separate question from value derivation |

## Code integration scope

The historical result came from `HierarchyProblem.hierarchy_comparison` in
`bpr/gauge_unification.py`. No EW implementation, raw benchmark result or
status flag is changed here. The former instruction to promote LIMITATIONS
based on this cross-check is superseded; the corrected reduced/unreduced
conversion does not establish a new EW result.

---

*April 2026 historical note retained; boundary normalization and authority of
the copied cross-check corrected on 2026-09-12.*
