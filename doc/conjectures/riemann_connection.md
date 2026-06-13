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

---

## UPDATE (June 2026 audit): GUE empirical foundation NOT reproduced

A direct check of the implemented operators found that the conjecture's
empirical premise — "RPST eigenvalue spacing is consistent with GUE
(KS p ≈ 0.92)" — is **not reproduced by the code**:

1. **`RPSTHamiltonian` is rank-1.** It builds `H = outer(leg, leg)` from the
   Legendre-symbol vector, which has **exactly one** nonzero eigenvalue
   (= p−1). A rank-1 Hamiltonian has no spectrum of repelling levels and
   therefore **cannot** exhibit GUE level-spacing statistics. Locked in by
   `tests/test_gue_riemann_honest.py`.

2. **A Legendre circulant doesn't either.** Replacing the outer product with
   a circulant generated by the Legendre symbol gives eigenvalues that are
   Gauss sums — only ~2 distinct magnitudes, highly degenerate — so the
   level spacings collapse and the GUE/Poisson KS tests return D ≈ 0.5 (poor)
   or NaN (degenerate). No GUE.

### Honest status revision

The GUE/Riemann connection is **downgraded from "Tier 2 conjecture with
numerical support" to "not currently supported by the implementation."**
The previously-cited KS p = 0.92 cannot be reproduced with either operator
in the repo. Either:

- a genuinely different RPST quantization (not the rank-1 outer product or
  the Legendre circulant) is needed to even pose the GUE question, or
- the GUE/Riemann claim should be withdrawn until such an operator exists.

This is the "clarify what the correlation does NOT imply" outcome: the
substrate eigenvalue statistics, as implemented, say nothing about the
Riemann zeros. Gaps 1–3 above are therefore moot until a full-spectrum
operator is constructed — that construction is now the prerequisite, not
the convergence proof.

---

## UPDATE — Strengthened negative finding (June 2026, full Hermitian-operator survey)

Following the earlier rank-1 finding for `RPSTHamiltonian`, a comprehensive
survey of candidate full-rank Hermitian operators on Z_p was conducted in
`bpr/substrate_hamiltonians.py`. **Result: no natural prime-modular Z_p
Hermitian operator exhibits Wigner-Dyson level statistics.**

### Constructions tested

1. **Legendre Hankel**: H_{ij} = Legendre((i+j) mod p, p) — nearly full
   rank, but only ~3 distinct eigenvalues (Gauss-sum degeneracy). Cannot
   support level-spacing analysis.
2. **Legendre Multiplicative**: H_{ij} = Legendre((i·j) mod p, p) — by
   multiplicativity equals `outer(leg, leg)`. **Rank 1**, same problem
   as the original `RPSTHamiltonian` from a different angle.
3. **Legendre Circulant**: H_{ij} = Legendre((i−j) mod p, p) — Gauss-sum
   degeneracies, ≤3 distinct eigenvalues.
4. **Discrete Berry-Keating**: H = (X̂P̂ + P̂X̂)/2 on Z_p with
   P̂ = F†X̂F (DFT-conjugate). **Full rank with p distinct eigenvalues**,
   but Poisson level-spacing (integrable), not Wigner-Dyson. K-S
   D_Poisson ≈ 0.14, D_GOE ≈ 0.27, D_GUE ≈ 0.33 across primes 211–1009.
5. **2D Berry-Keating** on Z_p × Z_p — same Poisson result at dim p².
6. **Berry-Keating + scaled perturbation** — transitions to GOE only at
   perturbation strength ≳ 10× mean level spacing, where the random
   component dominates the spectrum (prime-modular structure becomes
   negligible).

### Sanity check (locks in methodology)

Generic random GOE matrix correctly classifies as GOE
(D_GOE ≈ 0.07 ≪ D_GUE ≈ 0.10 < D_Poisson ≈ 0.21). The testing methodology
works; the negative findings are real.

### Status revision

**Strengthened**: the Riemann/GUE conjecture for BPR is downgraded from
"not currently supported by the implementation" (earlier finding) to
"**no Hermitian operator in the natural prime-modular Z_p class
reproduces GUE statistics.**" The prime-modular structure is
incompatible with generic Wigner-Dyson level repulsion at this level.

What this leaves open:

- **Non-Hermitian operators**: relaxing self-adjointness gives more
  spectral freedom; the Hannay-Berry quantum cat map is unitary (not
  Hermitian) and has different statistical properties. Untested here.
- **Larger Hilbert spaces with non-trivial coupling structure**: e.g.,
  Z_p^N with N > 2 and coupling terms — too large to test exhaustively
  but worth considering.
- **Random perturbation as a phenomenological fix**: BK + 10× mean-
  spacing perturbation gives GOE, but the random part dominates — this
  isn't a derivation of GUE from prime structure, it's adding randomness
  to wash out the prime structure.

### Verdict

The Hilbert-Pólya / Berry-Keating program for Riemann zeros via Z_p
Hermitian operators is not realized by any natural construction in this
module's survey. The conjecture's original numerical claim (KS p=0.92
for GUE) is not just unsupported by the framework's specific
implementation — it's structurally incompatible with the prime-modular
operator class. Closure of this question is now substantively negative.

Locked in by 6 tests in `tests/test_substrate_hamiltonians.py`.
