# BPR-Math-Spine: Prediction Validation Status

> **Version:** 0.7.0 (Jan 2026)  
> **Policy:** Every prediction is classified honestly.  Failures are documented.

## Classification Key

| Status | Meaning |
|--------|---------|
| **DERIVED** | BPR derives this from (J, p, N) with no hand-tuning |
| **FRAMEWORK** | BPR provides the framework/formula but some inputs are from experiment |
| **CONSISTENT** | Matches data, but also predicted by SM/GR (not uniquely BPR) |
| **CONJECTURAL** | Novel BPR claim, not yet testable |
| **OPEN** | BPR does not yet derive this quantity |
| **SUSPICIOUS** | Value matches because it was likely an input, not a genuine prediction |

## Previously Catastrophic Failures — Root Cause and Fixes

### The Wrong Assumption

**Before v0.7.0**, a single set of lab-scale parameters (R = 1 cm, N = 10,000)
was used to derive everything from the Planck length to galactic dynamics.
This is wrong: the lab parameters describe a Casimir experiment, not the universe.

**The fix:** Different physics probes different scales of the substrate.
- **Casimir/phonon predictions:** use lab-scale (R ~ cm, N ~ 10⁴)
- **Galactic/cosmological predictions:** use cosmological boundary (R_Hubble)
- **Particle physics:** use electroweak/GUT scales
- **The prime p = 104729** is the universal topological invariant across all scales

### Specific Fixes

| Prediction | Before (v0.5) | After (v0.7) | Observed | Fix |
|------------|--------------|-------------|----------|-----|
| MOND a₀ | 1.8×10¹⁸ m/s² | 1.04×10⁻¹⁰ m/s² | 1.2×10⁻¹⁰ | Used cosmological R_Hubble, not lab R |
| Baryon η | 9.4×10⁻²⁷ | 3.0×10⁻¹⁰ | 6.1×10⁻¹⁰ | Removed unjustified 1/√p suppression; used standard sphaleron efficiency |
| Tc(Nb) | 1.3×10⁻²² K | 6.0 K | 9.25 K | Replaced vacuum impedance formula with BCS N(0)V |
| Tc(MgB₂) | 1.1×10⁻⁵⁷ K | 67 K | 39 K | Same; N(0)V from experiment, not derived |
| l_Planck | 3.7×10⁻⁶ m | 1.616×10⁻³⁵ m | 1.616×10⁻³⁵ | Acknowledged as input, not derived from lab ξ |
| Hierarchy | √(pN) = 3.2×10⁴ | OPEN | 5.0×10¹⁶ | Honestly flagged as unsolved |
| θ₂₃ | 45.0° (maximal) | 47.6° | ~49° | Broke μ-τ symmetry via mass hierarchy correction |
| θ₁₂ | 35.26° | 33.65° | 33.41° | Corrected from exact 1/3 tri-bimaximal |

## Full Prediction Audit

### Theory I: Boundary Memory (P1.x)
| # | Prediction | Status | Notes |
|---|-----------|--------|-------|
| P1.1 | Oscillatory memory decay | CONJECTURAL | No direct test yet |
| P1.2 | Prime harmonic ω | DERIVED | ω = 2π/p, from substrate |
| P1.5 | Casimir fine-structure wiggles | CONJECTURAL | Below current precision |

### Theory II: Impedance / Dark Sector (P2.x)
| # | Prediction | Value | Status | Comparison |
|---|-----------|-------|--------|------------|
| P2.2 | MOND a₀ | 1.04×10⁻¹⁰ m/s² | FRAMEWORK | Obs: 1.2×10⁻¹⁰ (13% off) |
| P2.7 | DM σ/m | 0.019 cm²/g | DERIVED | Bound: < 0.6 cm²/g ✓ |
| P2.15 | Proton lifetime | ~10⁵⁰ yr | DERIVED | Bound: > 2.4×10³⁴ yr ✓ |

### Theory III: Decoherence (P3.x)
| # | Prediction | Status | Notes |
|---|-----------|--------|-------|
| P3.1 | Γ ∝ ΔZ² | DERIVED | Testable in molecule interferometry |
| P3.7 | W_crit(C60) = 10⁻³ | CONJECTURAL | Implies C60 is quantum ✓ |

### Theory V: Neutrino (P5.x)
| # | Prediction | Value | Status | Comparison |
|---|-----------|-------|--------|------------|
| P5.1 | Normal hierarchy | YES | DERIVED | T2K+NOvA: slight preference ✓ |
| P5.2 | Σm_ν | 0.06 eV | DERIVED | Bound: < 0.12 eV ✓ |
| P5.5 | θ₁₂ | 33.65° | FRAMEWORK | PDG: 33.41° ± 0.8° ✓ |
| P5.6 | θ₂₃ | 47.6° | FRAMEWORK | PDG: ~49° ± 1.3° ✓ |
| P5.7 | θ₁₃ | 8.63° | DERIVED | PDG: 8.54° ± 0.15° ✓ |
| P5.10 | 3 generations | 3 | DERIVED | Observed: 3 ✓ |

### Theory VII: GW (P7.x)
| # | Prediction | Status | Comparison |
|---|-----------|--------|------------|
| P7.1 | v_GW = c | CONSISTENT | GW170817: |δv/c| < 7×10⁻¹⁶ ✓ (also GR prediction) |

### Theory XI: Cosmology (P11.x)
| # | Prediction | Value | Status | Comparison |
|---|-----------|-------|--------|------------|
| P11.2 | n_s | 0.968 | SUSPICIOUS | Planck: 0.9649 ± 0.004 (Starobinsky input) |
| P11.3 | r | 0.003 | SUSPICIOUS | Bound: r < 0.044 (Starobinsky gives this) |
| P11.7 | η_baryon | 3.0×10⁻¹⁰ | FRAMEWORK | Obs: 6.1×10⁻¹⁰ (factor 2) |
| P11.15 | Ω_DM h² | 0.1196 | SUSPICIOUS | Planck: 0.120 ± 0.001 (too precise, likely input) |

### Theory IV: Phase Transitions (P4.x)
| # | Prediction | Value | Status | Notes |
|---|-----------|-------|--------|-------|
| P4.7 | Tc(Nb) | 6.0 K | FRAMEWORK | Obs: 9.25 K (N(0)V from experiment) |
| P4.9 | Tc(MgB₂) | 67 K | FRAMEWORK | Obs: 39 K (N(0)V from experiment) |

### Theory XVII: Gauge Unification (P17.x)
| # | Prediction | Value | Status | Notes |
|---|-----------|-------|--------|-------|
| P17.1 | GUT scale | 6.8×10¹⁷ GeV | DERIVED | Standard: ~2×10¹⁶ (30× off) |
| P17.4 | Hierarchy value | OPEN | OPEN | Cannot yet derive M_Pl/v from (p,N) |
| P17.6 | Higgs mass protected | YES | DERIVED | UV cutoff from boundary |
| P17.8 | τ_proton(GUT) | ~10⁴³ yr | DERIVED | Bound: > 2.4×10³⁴ yr ✓ |

### Theory XVIII: Charged Leptons (P18.x)
| # | Prediction | Status | Notes |
|---|-----------|--------|-------|
| P18.1-3 | e/μ/τ masses | SUSPICIOUS | c_norms tuned to match known masses |
| P18.4 | Koide Q = 2/3 | SUSPICIOUS | Known empirical relation (Koide 1981) |
| P18.7 | R(K) ≈ 1 | CONSISTENT | LHCb 2023 confirms SM ✓ (not uniquely BPR) |

### Theory XIX: Nuclear Physics (P19.x)
| # | Prediction | Status | Notes |
|---|-----------|--------|-------|
| P19.5 | Magic numbers | CONSISTENT | Known since 1949 (hardcoded list) |
| P19.7 | B/A(Fe56) = 8.85 MeV | FRAMEWORK | Obs: 8.79 MeV (BW formula + BPR term) |

### Theory XX: QG Phenomenology (P20.x)
| # | Prediction | Value | Status | Comparison |
|---|-----------|-------|--------|------------|
| P20.1 | ξ₁ = 0 (no linear LIV) | 0 | DERIVED | LHAASO: E_QG,1 > 10 M_Pl ✓ |
| P20.4 | GUP β = 1/p | 9.5×10⁻⁶ | DERIVED | Bound: β < 4×10⁴ (well within) |
| P20.7 | |δc/c| | 3.4×10⁻²¹ | DERIVED | Fermi: < 6×10⁻²¹ (just below — testable!) |

### Theory XIII: Emergent Spacetime (P13.x)
| # | Prediction | Status | Notes |
|---|-----------|--------|-------|
| P13.3-5 | 3+1 dimensions | DERIVED | Correct ✓ |
| P13.8 | l_Planck | OPEN | Was wrong; now acknowledged as input |

## Summary Scorecard (v0.7.0)

| Category | Count | % |
|----------|-------|---|
| DERIVED (genuine BPR prediction, consistent with data) | ~25 | 12% |
| FRAMEWORK (BPR formula, some inputs from experiment) | ~30 | 15% |
| CONSISTENT (matches, but also SM/GR prediction) | ~20 | 10% |
| SUSPICIOUS (likely reverse-engineered from known data) | ~15 | 7% |
| CONJECTURAL (not yet testable) | ~40 | 20% |
| Standard physics with BPR label | ~70 | 34% |
| OPEN (acknowledged as unsolved) | ~5 | 2% |

## Lessons Learned

1. **Scale separation matters.** Lab-scale substrate parameters cannot derive
   the Planck length or galactic dynamics. Different scales require different
   effective descriptions, connected by the universal prime p.

2. **Claiming "zero free parameters" was overstated.** BPR has 3 substrate
   parameters (J, p, N) but also implicitly uses observation-scale parameters
   (R, geometry) and theory-specific inputs (N(0)V for Tc, c_norms for masses).

3. **Reproducing known physics is not evidence.** The ~70 "predictions" that
   reproduce textbook results (magic numbers, shell structure, hydrogen levels)
   demonstrate consistency but not predictive power.

4. **The genuinely novel content is smaller but real.** The ~25 DERIVED
   predictions—especially the Casimir deviation, phonon coupling channel,
   decoherence scaling, and Lorentz violation bound—are testable and specific.

---
*Generated by BPR-Math-Spine v0.7.0 validation audit*
