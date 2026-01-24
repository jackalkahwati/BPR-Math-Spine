# Born Rule from Microstate Counting

**Status:** TIER 2 CONJECTURE
**Confidence:** 50-70%
**Source:** "Manufactured Randomness" paper

---

## Claim

The Born rule emerges from uniform phase averaging over RPST microstates,
not from fundamental randomness.

## Proposed Derivation

### Setup

For microstates Ω_α(x) corresponding to particle type α at location x, define:

```
ψ_α(x) = Σ_{S ∈ Ω_α(x)} exp(iφ(S))
```

where φ(S) is the phase of microstate S.

### Coarse-Graining Assumption

Under coarse-graining with uniform phase distributions:

```
⟨exp(iφ(S))⟩ = 0  for individual states
⟨|Σ exp(iφ)|²⟩ = N  for N states (by central limit)
```

### Proposed Result

By symmetry under uniform phase distribution:

```
P(α, x) = |ψ_α(x)|² / Σ_{α',x'} |ψ_{α'}(x')|²
```

This is the Born rule.

---

## What's Solid (Tier 1)

1. **Statistical mechanics of counting:** Standard probability from microstate
   counting is well-established (Boltzmann, Gibbs).

2. **Central limit behavior:** Sums of random phases converge to Gaussian with
   variance proportional to N.

3. **Dimensional consistency:** |ψ|² has correct units for probability density.

---

## What's Missing (Gaps)

### Gap 1: Why Uniform Phases?

**The claim:** Phases are uniformly distributed in Ω_α(x).

**What's needed:** Derivation showing that RPST dynamics (the Z_p symplectic
map) produces uniform phase distribution in the continuum limit.

**Possible approach:** Show that RPST Hamiltonian has sufficient mixing
to justify uniform microcanonical distribution.

### Gap 2: Definition of Microstates

**The claim:** Microstates Ω_α(x) correspond to "particle type α at x."

**What's needed:** Explicit construction of Ω_α(x) from RPST substrate.
What does "particle at x" mean in terms of Z_p lattice configurations?

**This is the hardest gap.** Without this, the derivation is circular.

### Gap 3: Independence of Microstates

**The claim:** Different microstates contribute independently.

**What's needed:** Show that correlations between microstate phases
average out under coarse-graining.

### Gap 4: Normalization

**The claim:** Probabilities sum to 1.

**What's needed:** Verify that the counting measure is properly normalized
across all α and x.

---

## Falsification Criteria

### The conjecture is falsified if:

1. **RPST dynamics do NOT produce uniform phase distribution** in the
   continuum limit (can be tested numerically).

2. **Born rule emerges from different mechanism** that doesn't involve
   phase averaging (would need alternative derivation).

3. **Microstate counting gives WRONG probabilities** for test cases
   where Born rule is experimentally verified.

### Test Protocol

1. Simulate RPST substrate with N ~ 10^6 nodes for time T ~ 10^4 steps.
2. Compute phase distribution P(φ) across substrate.
3. Test for uniformity: χ² statistic against uniform distribution.
4. If P(φ) is NOT uniform, conjecture fails at Step 1.

---

## Connection to Existing Literature

### Consistent With:

- **Zurek's existential interpretation:** Probabilities from observer-induced
  branching ratios.
- **Many-worlds counting:** Branch weights proportional to |ψ|².
- **Typicality arguments:** Most observers see Born statistics.

### Inconsistent With:

- **Copenhagen interpretation:** Probability as fundamental, not derived.
- **GRW collapse models:** Physical collapse mechanism, not statistical.

---

## Next Steps for Tier 1 Promotion

1. **Define Ω_α(x) constructively** from RPST lattice.
2. **Prove uniform phase theorem** for RPST dynamics.
3. **Verify numerically** that counting reproduces Born probabilities.
4. **Connect to standard QM** via explicit mapping.

---

## Implementation Status

| Component | Status | Module |
|-----------|--------|--------|
| RPST substrate | ✓ Implemented | `bpr/rpst/substrate.py` |
| Phase distribution | Needs test | `tests/test_phase_distribution.py` |
| Microstate counting | Not implemented | — |
| Born rule verification | Not implemented | — |

---

## Conclusion

**This is a plausible conjecture with significant gaps.**

The idea that probability emerges from phase averaging is attractive and
has precedent in statistical mechanics. However, the specific construction
of Ω_α(x) and the proof of uniform phase distribution are missing.

**Recommendation:** Keep in Tier 2. Test phase uniformity numerically.
If that passes, attempt constructive definition of microstates.

---

**Last Updated:** January 2026
**Assigned:** [Unassigned]
**Priority:** Medium (depends on RPST validation)
