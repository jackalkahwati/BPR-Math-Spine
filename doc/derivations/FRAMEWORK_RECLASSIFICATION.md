# FRAMEWORK Item Reclassification (June 2026 Audit)

The DERIVATION_ROADMAP lists 5 FRAMEWORK predictions (consistent, not
derived). This audit reclassifies each by WHAT KIND of work would close it,
so contributors know which are BPR-specific derivation targets vs. inherited
many-body problems no framework has solved.

| ID | Prediction | Reclassified status | What closing it requires |
|----|------------|---------------------|--------------------------|
| P4.9 | Tc(MgB₂) | **BPR-derivation target (partial)** | Combined N(0)V now DERIVED (`bpr/superconductivity.py`, 0.26% off). Interband factor 1+1/(4 ln p) remains phenomenological; coordination-shell 2/z TESTED & REJECTED. Closing needs a first-principles interband-coupling derivation. SMALL–MEDIUM. |
| P19.7 | B/A(⁵⁶Fe) | **Inherited many-body** | Binding energy per nucleon is a nuclear many-body (liquid-drop + shell) quantity. BPR contributes the shell-closure integers (magic numbers, already DERIVED) but not the absolute B/A, which needs the full EOS. NOT a clean BPR target. |
| P19.9 | n_sat | **Inherited many-body** | Nuclear saturation density is a QCD many-body result. BPR has no special handle beyond the confinement scale. NOT a clean BPR target. |
| P19.10 | M_NS,max | **Inherited many-body** | Neutron-star max mass depends on the dense-matter EOS above n_sat — unsolved by ALL frameworks (the "EOS problem"). BPR inherits this; not a BPR-specific gap. |
| P19.11 | R_NS | **Inherited many-body** | Same as M_NS,max — EOS-limited. NOT a clean BPR target. |

## Verdict

- **1 of 5** (Tc MgB₂) is a genuine BPR-specific derivation target, now
  partially closed: the pairing strength is derived, the interband factor is
  the remaining gap.
- **4 of 5** (the nuclear items) are inherited many-body problems that no
  framework solves from first principles. They should be relabeled in the
  roadmap as "INHERITED (EOS-limited)" rather than "FRAMEWORK (open for BPR
  derivation)" — the latter overstates what BPR could provide.

This prevents future contributors from spending effort trying to "derive"
the neutron-star EOS from substrate parameters, which is not a tractable or
BPR-specific problem.
