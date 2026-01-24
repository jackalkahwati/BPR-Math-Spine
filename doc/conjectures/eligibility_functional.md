# Eligibility Functional for Boundary Phase Rewrite

**Status:** TIER 2 CONJECTURE
**Confidence:** 30-50%
**Source:** "Cache, BPR, RPST Full Architecture" paper, Section 9

---

## Claim

A "rewrite event" (discrete update to boundary conditions) is triggered
when the eligibility functional E(ψ) exceeds a threshold ε.

## Proposed Form

```
E(ψ) = ||∇_∂Ω K||₂ · σ(ψ) / (1 + κ_K / K(ψ))
```

where:
- K(ψ): Global coherence (Kuramoto order parameter)
- σ(ψ): Boundary stress (frustration or Laplacian)
- κ_K: Coherence suppression scale
- ∇_∂Ω K: Boundary gradient of coherence

Rewrite triggers when: E(ψ) > ε

---

## Interpretation

### Numerator: ||∇K|| · σ

- **||∇K||**: Rate of coherence change (something is happening)
- **σ**: Boundary stress (tension in the system)
- **Product**: High stress AND rapid change → likely transition point

### Denominator: 1 + κ_K / K

- **Low coherence (K → 0)**: Denominator large → E suppressed
- **High coherence (K → 1)**: Denominator ≈ 1 → E not suppressed
- **Interpretation**: Don't rewrite when system is already disordered

---

## What's Solid (Tier 1)

1. **Coherence K is well-defined:** Kuramoto order parameter is standard.
   Implemented in `bpr/rpst/coherence.py`.

2. **Stress σ is well-defined:** Frustration and Laplacian stress are
   standard gradient energy measures. Implemented in `bpr/rpst/coherence.py`.

3. **Thresholding is a valid concept:** Many physical systems exhibit
   threshold behavior (phase transitions, nucleation, etc.).

---

## What's Missing (Gaps)

### Gap 1: Why This Functional Form?

**The claim:** E(ψ) should have this specific algebraic structure.

**What's provided:** Heuristic justification (numerator = change × stress,
denominator = coherence suppression).

**What's needed:**
- Derivation from variational principle
- Physical meaning of κ_K
- Why multiplicative rather than additive?

**Alternative forms that seem equally plausible:**
```
E₁(ψ) = ||∇K|| + λσ           # Additive
E₂(ψ) = ||∇K||² + σ²          # Sum of squares
E₃(ψ) = (||∇K|| · σ)^α        # Power law
```

No argument distinguishes the proposed form from alternatives.

### Gap 2: What Sets κ_K and ε?

**The claim:** κ_K and ε are parameters.

**What's provided:** "Tunable hyperparameters."

**What's needed:**
- Physical meaning of κ_K in terms of RPST substrate
- Derivation of ε from system properties
- Scaling behavior with system size N

### Gap 3: What IS a "Rewrite Event"?

**The claim:** Discrete updates to boundary conditions occur.

**What's provided:** Abstract description.

**What's needed:**
- Concrete definition: What changes during rewrite?
- Mechanism: How does E > ε trigger the rewrite?
- Conservation: What's preserved across rewrite?

This is the fundamental gap. Without knowing what a rewrite IS,
the eligibility functional is just a number.

### Gap 4: Physical Observables

**The claim:** Rewrite events have physical consequences.

**What's provided:** Assertion.

**What's needed:**
- Observable signatures of rewrite events
- Experimental or simulation protocol to detect them
- Falsification criteria

---

## Comparison to Known Physics

### Similar Concepts:

1. **Nucleation theory:** Free energy barrier determines transition rate
   ```
   Rate ~ exp(-ΔF/kT)
   ```
   Here, E plays role of activation energy.

2. **Ginzburg-Landau:** Order parameter dynamics near phase transition
   ```
   ∂ψ/∂t = -δF/δψ*
   ```
   Eligibility could be related to |δF/δψ|.

3. **Ising model:** Metropolis acceptance probability
   ```
   P(flip) = min(1, exp(-ΔE/kT))
   ```
   Eligibility could be analogous to ΔE.

### Difference from Known Physics:

The specific form of E(ψ) doesn't match any standard functional.
This suggests either:
1. A genuinely new physical principle (exciting but unlikely)
2. An incomplete formulation (more likely)
3. Curve-fitting dressed as theory (concerning)

---

## Falsification Criteria

### The conjecture is strengthened if:

1. Rewrite events cluster near E > ε peaks (simulation test)
2. System dynamics change qualitatively after rewrite
3. ε shows universal scaling with N

### The conjecture is weakened if:

1. Rewrite events occur independent of E
2. Alternative E forms work equally well
3. κ_K and ε must be tuned per-system

### The conjecture is FALSIFIED if:

1. Systems with E > ε show no qualitative change
2. A simpler model (without eligibility) explains same phenomena

---

## Implementation Status

| Component | Status | Module |
|-----------|--------|--------|
| Coherence K | ✓ Implemented | `bpr/rpst/coherence.py` |
| Stress σ | ✓ Implemented | `bpr/rpst/coherence.py` |
| Eligibility E | ✓ Implemented | `bpr/rpst/coherence.py` |
| Rewrite mechanism | NOT implemented | — |
| Event detection | NOT implemented | — |

---

## Testing Protocol

### Step 1: Compute E Along Trajectories

1. Simulate RPST substrate dynamics
2. Compute E(ψ(t)) at each timestep
3. Identify peaks and threshold crossings

### Step 2: Look for Qualitative Changes

1. Track observables: K(t), σ(t), energy E(t)
2. Test for discontinuities or rapid changes
3. Correlate with E > ε events

### Step 3: Vary Parameters

1. Scan κ_K ∈ [0.01, 10]
2. Scan ε ∈ [0.1, 100]
3. Test sensitivity of event detection

### Step 4: Compare Alternatives

1. Implement E₁, E₂, E₃ (alternative forms)
2. Compare predictive power
3. If all work equally well, original form not special

---

## Honest Assessment

**What we have:**
- A plausible-looking functional
- Standard components (K, σ)
- Implemented and computable

**What we don't have:**
- Derivation from first principles
- Physical meaning of "rewrite"
- Falsification tests

**Conclusion:** This is a HYPOTHESIS, not a result. It may be useful
as a heuristic for detecting phase-transition-like events, but it
lacks theoretical foundation.

---

## Recommendation

### For Core Spine:

**DO NOT include** eligibility as "BPR equation."
**DO include** K and σ as useful diagnostics.

### For Conjectures:

Keep eligibility in Tier 2 with clear gaps documented.
Test numerically before claiming physical significance.

### For Papers:

Present as: "We propose the following eligibility functional..."
NOT as: "BPR predicts that..."

---

**Last Updated:** January 2026
**Assigned:** [Unassigned]
**Priority:** Low-Medium (interesting heuristic, but not foundational)
