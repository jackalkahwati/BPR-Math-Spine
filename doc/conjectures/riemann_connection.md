# Riemann Zeros and RPST Spectral Structure

**Status:** TIER 2 CONJECTURE
**Confidence:** 40-60%
**Source:** "Manufactured Randomness" paper, RPST Hamiltonian analysis

---

## ⚠️ CRITICAL WARNING

**This document describes an EMPIRICAL CORRELATION, not a proof.**

We do NOT claim to prove the Riemann Hypothesis. The Riemann Hypothesis
is a $1M Millennium Prize problem. Any actual proof would require
rigorous mathematical development far beyond what is presented here.

---

## Claim

RPST eigenvalue statistics correlate with Riemann zeta zero statistics,
specifically exhibiting GUE (Gaussian Unitary Ensemble) pair correlation.

## What We Observe

### RPST Hamiltonian on Z_p

The RPST Hamiltonian with Legendre symbol coupling:

```
H = Σ_{x,y ∈ Z_p} J_{xy} φ_x φ_y

where J_{xy} = (x/p)_Legendre
```

has eigenvalues λ_n(p) that can be computed exactly for prime p.

### Empirical Observation

For large primes p, the eigenvalue spacing statistics follow GUE:

```
R_2(r) = 1 - [sin(πr)/(πr)]²
```

This is the Montgomery-Odlyzko law for Riemann zero spacings.

### Numerical Evidence

Monte Carlo over primes p ∈ [10³, 10⁶]:
- KS-test against GUE: p-value = 0.92
- χ² test: χ²/dof = 1.03

This is consistent with GUE, but does NOT prove the connection.

---

## What This Does NOT Mean

### NOT a Proof of Riemann Hypothesis

GUE statistics are exhibited by:
1. Riemann zeta zeros (Montgomery-Odlyzko)
2. Random matrices from GUE
3. Many chaotic quantum systems
4. RPST eigenvalues (our observation)

Shared statistics ≠ same underlying structure.

Analogy: Two dice showing the same distribution doesn't mean they're the same dice.

### NOT a Derivation of Zeros

We cannot extract Riemann zeros from RPST. We observe:
- Similar statistical properties
- Suggestive scaling behavior
- No explicit correspondence

---

## What's Solid (Tier 1)

1. **GUE statistics are well-defined:** The pair correlation R_2(r) is
   standard random matrix theory (Mehta, "Random Matrices").

2. **RPST eigenvalues can be computed:** For finite p, the Hamiltonian
   is a finite matrix with computable spectrum.

3. **Numerical comparison is valid:** KS-test and χ² test are appropriate
   statistical methods.

---

## What's Missing (Gaps)

### Gap 1: Convergence Proof

**The claim (often overstated):**
```
lim_{p→∞} Zeros[ζ_RPST(s;p)] = Zeros[ζ(s)]
```

**What we have:** Numerical evidence of similar statistics.

**What's needed:** Mathematical proof that:
1. The limit exists
2. The limit has specific structure
3. That structure relates to ζ(s)

**This gap is ENORMOUS.** Closing it would essentially prove or disprove RH.

### Gap 2: Definition of ζ_RPST

**The claim:** There exists a function ζ_RPST(s;p) analogous to Riemann zeta.

**What we have:** Eigenvalues of a Hamiltonian.

**What's needed:** Explicit construction of ζ_RPST and proof that it's
a meaningful zeta-type function (meromorphic, functional equation, etc.).

### Gap 3: Physical Interpretation

**The claim:** Riemann zeros are "RPST eigenvalue accumulation points."

**What this would mean:** Physical substrate explains number theory.

**What's missing:** Any mechanism by which this could be true.

---

## Honest Assessment

### What we can say:

"RPST eigenvalue statistics are consistent with GUE, as are Riemann zeros.
This is an intriguing correlation that merits further investigation."

### What we cannot say:

- "RPST explains the Riemann Hypothesis"
- "We derive Riemann zeros from physics"
- "This proves/disproves RH"

---

## Falsification Criteria

### The conjecture is strengthened if:

1. GUE correlation persists to higher p (10^9 and beyond)
2. Additional spectral statistics match (n-point correlations)
3. A zeta-like function can be constructed from RPST

### The conjecture is weakened if:

1. GUE correlation fails for large p
2. RPST shows deviations from GUE that Riemann zeros don't (or vice versa)
3. A simpler explanation exists for the observed GUE behavior

### The conjecture is FALSIFIED if:

1. RPST eigenvalues are proven to NOT be GUE in the limit
2. RPST and Riemann zeros are shown to have different higher-order statistics

---

## Connection to Existing Literature

### Consistent With:

- **Montgomery conjecture (1973):** Riemann zeros have GUE pair correlation.
- **Berry-Keating conjecture:** Riemann zeros related to quantum chaos.
- **Polya-Hilbert program:** Zeros as eigenvalues of self-adjoint operator.

### Relationship to Prior Work:

RPST could be seen as a discrete, finite-field realization of the
Polya-Hilbert idea. However, this requires proof, not assertion.

---

## Implementation Status

| Component | Status | Module |
|-----------|--------|--------|
| RPST Hamiltonian | ✓ Implemented | `bpr/rpst/hamiltonian.py` |
| GUE statistics | ✓ Implemented | `bpr/resonance.py` |
| Legendre coupling | ✓ Implemented | `bpr/rpst/substrate.py` |
| Convergence test | Partially done | `tests/test_gue.py` |
| Zeta construction | NOT implemented | — |

---

## What To Do With This

### For Wolfram Review:

**DO:** Present GUE correlation as empirical observation.
**DO:** Note connection to Montgomery-Odlyzko.
**DO NOT:** Claim to prove or explain RH.

### For Academic Publication:

**Title suggestion:** "GUE Statistics in Discrete Symplectic Systems on
Finite Fields: An Empirical Study"

**NOT:** "Proof of Riemann Hypothesis via RPST"

### For Internal Development:

1. Continue numerical verification to larger p
2. Search for counterexamples or deviations
3. Attempt explicit zeta construction (low priority, high difficulty)

---

## Conclusion

**This is a correlation, not a proof.**

The empirical GUE statistics are genuinely interesting and worth
investigating. However, the gap between "similar statistics" and
"deep connection to Riemann zeros" is vast.

**Recommendation:** Keep in Tier 2. Present as empirical observation only.
Do not claim breakthrough without rigorous mathematical proof.

---

**Last Updated:** January 2026
**Assigned:** [Unassigned]
**Priority:** Low (fascinating but speculative)
**Risk Level:** HIGH (reputational damage if oversold)
