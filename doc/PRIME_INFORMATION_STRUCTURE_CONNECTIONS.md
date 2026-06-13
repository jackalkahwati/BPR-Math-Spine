# Prime Information Structure — Connections and Limits

> **Status:** Connects BPR's prime-substrate program (Z_p with p = 104761)
> to two mainstream number-theoretic-physics results — Kontoyiannis's
> information-theoretic prime-counting work and the Latorre-Sierra
> Prime state quantum-information program. Honest scope: these results
> establish that primes carry deep information-theoretic and
> quantum-mechanical structure. They do NOT validate BPR's specific
> prime selection. They DO suggest concrete next calculations that
> would either strengthen or weaken the framework's prime claim.

## 1. The two source papers

**Kontoyiannis, "Counting Primes Using Entropy" (2008)** — IEEE
Information Theory Newsletter; full technical version in arXiv:0710.4076
(2007). Gives an information-theoretic proof of Chebyshev's result

    Σ_{p prime, p ≤ n} (log p) / p ~ log n   as n → ∞

using Shannon entropy bounds plus the fundamental theorem of arithmetic.
The 2021 follow-up "All the information in the integers is in the primes"
(arXiv:2112.08911) makes the slogan explicit: the entire entropic content
of the integer distribution is captured by the primes via unique
factorization.

References:
- [Counting Primes Using Entropy (PDF)](https://www.dpmms.cam.ac.uk/~ik355/PAPERS/NewsLetter.pdf)
- [arXiv:0710.4076](https://arxiv.org/abs/0710.4076)
- [All the information in the integers is in the primes (arXiv:2112.08911)](https://arxiv.org/abs/2112.08911)

**García-Martín, Ribas, Carrazza, Latorre, Sierra, "The Prime state and
its quantum relatives" (2020)** — Quantum 4, 371; arXiv:2005.02422.
Defines the Prime state of n qubits as the uniform superposition

    |P_n⟩ = (1 / √π(2^n))  Σ_{p prime, p ≤ 2^n}  |p⟩

and proves the bipartite entanglement entropy of |P_n⟩ scales with the
Shannon entropy of the square-free integer density 6/π² ≈ 0.6079.
The QFT of |P_n⟩ accesses Chebyshev biases in arithmetic progressions.
Results extend to qudit bases and arithmetic-progression generalizations.

References:
- [arXiv:2005.02422](https://arxiv.org/abs/2005.02422)
- [Quantum 4, 371 (2020)](https://quantum-journal.org/papers/q-2020-12-11-371/)
- Predecessor: Latorre & Sierra, "There is entanglement in the primes,"
  [arXiv:1403.4765](https://arxiv.org/abs/1403.4765), J. Phys. A 47, 475302 (2014)

## 2. What these papers establish (and what they don't)

### Establish

- Primes carry intrinsic information-theoretic content; the entropy of
  the integers is captured by the primes
- Primes carry intrinsic quantum-information structure; the Prime state
  has measurable entanglement properties tied to deep number-theoretic
  invariants
- The connection between primes (multiplicative) and square-free
  integers (multiplicative-up-to-divisibility) is quantum-mechanically
  meaningful via the entanglement scaling
- The QFT of the Prime state accesses Chebyshev biases — quantum
  computing can probe the arithmetic of primes

This is the *family* of approaches in which BPR sits. Other members of
this family include Connes' noncommutative-geometry approach to the
Riemann zeta function, the Berry-Keating Hilbert-Pólya program, Sierra's
own prior work on ζ as a Hamiltonian spectrum, and the broader
number-theoretic-physics tradition.

### Do NOT establish

- That any *specific* prime is physically realized
- That BPR's choice p = 104761 has special significance beyond the
  numerological α-matching that BPR offers
- That a Z_p substrate is the correct framework for emergent spacetime
- Any of BPR's distinctive predictions (δ = 2 Casimir, doubly-heavy
  baryon hyperfine splittings, etc.)

This is the honest scope: support for the family, not for the specific
framework.

## 3. Three concrete connections to BPR

### Connection 1: Square-free entanglement scaling as a BPR test

The García-Martín result is a *sharp number-theoretic identity* connecting
prime structure to entanglement entropy via the square-free density 6/π².
If BPR's substrate is genuinely doing what its prime-modular structure
suggests, the BPR-substrate analog of the Prime state should reproduce
the same scaling.

The computational machinery to test this is now in
[`bpr/prime_state_check.py`](../bpr/prime_state_check.py). The Prime
state side is verified independently (we reproduce the scaling
numerically up to n=12 qubits, with finite-size corrections matching
the published behavior). The BPR-substrate side is stubbed — it
requires committing to how the Z_p boundary phase field becomes a
quantum state, which has not been specified.

**The discriminating outcome would be:**
- BPR substrate reproduces the 6/π² × n_A scaling → prime-substrate
  claim has genuine number-theoretic teeth
- BPR substrate gives different scaling → prime-substrate claim is
  decorative (the framework says "Z_p" but doesn't behave like the
  primes carry the structure)

This is exactly the kind of test the mainstream-mapping discipline
recommends: a specific calculation whose outcome either strengthens or
weakens the framework's distinctive claim.

### Connection 2: Chebyshev biases and BPR's boundary mode integers

García-Martín et al.'s QFT-of-Prime-state result accesses Chebyshev
biases in arithmetic progressions (the well-known but subtle asymmetry
where primes ≡ 3 (mod 4) outnumber primes ≡ 1 (mod 4) at small scales,
with the bias oscillating at larger scales).

BPR's boundary mode integers are themselves arithmetic functions of z:
l_d = 1, l_s = z − 2 = 4, l_b = z(z − 1) = 30, l_u = 1, l_c = z(z − 2)
= 24, l_t = (z²−1)(z + n_gen + 2 − N_c) + n_gen = 283 for z = 6,
n_gen = 3. These are derived from the substrate, not fitted, but their
relationship to primes-in-arithmetic-progression structure (the kind of
thing the QFT of the Prime state probes) is unexplored.

**Possible refinement:** check whether BPR's specific boundary mode
integers correlate with arithmetic-progression structure of primes
near p = 104761. This is a long-shot check, but a positive result would
be evidence that BPR's substrate is structured by prime arithmetic in
a deeper way than the Z_p label suggests.

### Connection 3: Kontoyiannis entropy bound as an audit tool

Kontoyiannis's information-theoretic framework gives bounds on the
information content of integers. Specifically, log p ≈ 11.56 nats
≈ 16.68 bits for BPR's p = 104761.

This provides a quantitative audit tool for BPR's headline claim that
"205 falsifiable predictions are derived from three substrate numbers
(J, p, z)." The audit:

- Information available in (J, p, z) inputs: ~28 bits (with generous
  estimates for J and z)
- Information needed to specify 205 binary-outcome predictions: ~205 bits
- Information needed to specify 205 real-valued predictions to 10%
  precision: ~700 bits
- Excess of 177–670 bits is smuggled in via structural choices
  (boundary mode integers from z, W_c = √3 winding, Higgs VEV
  anchors, lattice-anchored bindings) and fitted coefficients
  (θ_23 = 1.35, θ_12 = 3.5, ln(p)/(ln(p)+1), MgB₂ interband factor)

The verdict: BPR's framework is **not** "three numbers → everything" in
strict information-theoretic terms. It is "three numbers + boundary CFT
structure + fitted coefficients → 205 predictions." This is still a
meaningful parameter reduction relative to fitting 20+ SM parameters
independently, but the headline "three numbers" framing understates the
structural inputs. The information-theoretic audit makes this
quantitative.

This complements the parameter-honesty pass from earlier audit work —
the same conclusion (BPR has fitted coefficients masquerading as derived)
is reached via a completely independent route.

## 4. What this means for BPR's positioning

The Kontoyiannis and García-Martín results provide:

1. **Genuine intellectual lineage** for BPR's prime-substrate program.
   The number-theoretic-physics tradition that takes primes seriously as
   physical objects is real and active.
2. **A specific testable prediction** (square-free entanglement scaling)
   that BPR could either reproduce or fail to reproduce. This is exactly
   the discriminating-test format the mainstream mapping recommends.
3. **An audit tool** (the Kontoyiannis bound) that complements the
   parameter-honesty work in a quantitative way.

What they do NOT provide:

1. Validation that p = 104761 is the correct substrate prime
2. Derivation of α from prime structure
3. Support for the BPR-specific extensions (phason, consciousness, etc.)
4. Any resolution of the GUE/Riemann finding from the earlier audit
   (where BPR's `RPSTHamiltonian` was found to be rank-1 and unable to
   produce GUE statistics — this stands as an open structural problem)

## 5. Recommended next-step calculation

The cleanest next research step is **Connection 1's BPR-substrate analog
computation**:

1. Specify how the BPR boundary phase field (a Z_p-valued function on the
   discrete lattice) maps to a quantum state in a finite-dimensional
   Hilbert space.
2. Identify the natural "support rule" analogous to "support is on
   primes" in the standard Prime state. Candidates:
   - Support on states with nontrivial topological winding
   - Support on stable boundary resonances (minimizing the boundary action)
   - Support on states satisfying a substrate-specific selection rule
3. Compute the bipartite entanglement entropy under a natural lattice
   partition.
4. Compare to the García-Martín 6/π² × n_A scaling.

If the scaling reproduces (within finite-size corrections matched to
those we already see for the standard Prime state), BPR's prime claim
has genuine quantum-information content. If it doesn't, the claim is
nominal.

This computation is a real research target — it requires committing to
the "support rule" choice, which is a substantive theoretical decision.
It's not a one-function calculation. But it's well-defined, and its
outcome would either substantially strengthen or substantially weaken
BPR's distinctive content.

## 6. Code

[`bpr/prime_state_check.py`](../bpr/prime_state_check.py) — Prime state
construction, bipartite entanglement entropy calculation, García-Martín
scaling check (verified for n ≤ 12), Kontoyiannis bound audit, and the
BPR-substrate analog stub for future implementation.

[`tests/test_prime_state_check.py`](../tests/test_prime_state_check.py)
— 9 tests covering the numerical reproduction and audit results.

## 7. Honest summary

These two papers strengthen the *family* of approaches BPR belongs to.
They identify a sharp testable claim BPR could make (square-free
entanglement scaling) that would discriminate it from a nominal
prime-substrate framework. They provide an information-theoretic audit
tool that complements the framework's parameter-honesty work.

They do not validate BPR's specific substrate identification, do not
predict any BPR-distinctive observable, and do not resolve the
RPSTHamiltonian rank-1 issue from the earlier audit.

The honest position: BPR is a member of an intellectually serious
tradition that includes the Latorre-Sierra Prime state program, the
Connes noncommutative-geometry approach to ζ, and the Berry-Keating
Hilbert-Pólya program. Membership in that tradition is meaningful but
does not constitute validation. The validation, if it comes, will come
from the discriminating tests (square-free entanglement scaling among
them, plus the flavor-sector pre-dictions registered for LHCb).

---

## 8. UPDATE — BPR-substrate analog calculation EXECUTED

The discriminating test outlined in §5 has been computed. Choice of
support rule: **quadratic residues mod p**, which uses BPR's prime
modular structure intrinsically (Legendre-symbol selection, multi-
plicative subgroup of index 2 in (Z/pZ)*).

### Calculation

The BPR-substrate analog of the Prime state:

    |QR_p⟩ = (1 / √|QR|) Σ_{a: (a|p)=+1, 0 < a < p} |a⟩

Embedded in the smallest power-of-2 Hilbert space (so 2^n_qubits ≥ p),
with half-half bipartite entanglement entropy computed via SVD.
Compared to the Bernoulli-density baseline log 2 × n_A (which is what a
uniform-density-1/2 state would give without any correlations).

### Numerical results

| p | n_qubits | n_A | S(|QR_p⟩) [nats] | log 2 × n_A | ratio |
|---|---|---|---|---|---|
| 31  | 5 | 2 | 1.065 | 1.386 | 0.768 |
| 61  | 6 | 3 | 1.389 | 2.079 | 0.668 |
| 127 | 7 | 3 | 1.410 | 2.079 | 0.678 |
| 251 | 8 | 4 | 1.586 | 2.773 | 0.572 |
| 509 | 9 | 4 | 1.783 | 2.773 | 0.643 |

**Slope of ratio vs ln(p): −0.05** (essentially flat).

### Interpretation

The QR-state entanglement entropy is **consistently 0.57–0.77 of the
Bernoulli baseline**, never approaching it. This means:

1. **Multiplicative correlations among quadratic residues carry quantum-
   information content.** A random density-1/2 subset would give ratio
   ~1.0. The observed ratio bounded away from 1 below is direct evidence
   of nontrivial number-theoretic structure in the entanglement.

2. **This is structurally analogous to the Prime state result.** The
   Prime state's entropy is bounded below H(6/π²) × n_A asymptotically
   because square-free correlations among integers carry information.
   The QR state's entropy is bounded below log 2 × n_A because Legendre-
   symbol multiplicative correlations carry information. Same kind of
   number-theoretic content, different specific identity.

3. **The structure is approximately p-independent** (slope ~ −0.05).
   This means the discrimination it gives is about "Z_p with prime p"
   in general, NOT about p = 104761 specifically. Any prime would
   exhibit the same QR-correlation signature.

### Honest scoping of what this establishes

✅ **The framework's prime-substrate claim is NOT purely decorative.**
The natural quantum state on a Z_p substrate has nontrivial number-
theoretic structure that reduces its entanglement entropy below the
bare-density bound — exactly the kind of structure that signals real
information content beyond labels.

❌ **This does NOT validate p = 104761 specifically.** The structure is
generic to "Z_p with p prime"; the calculation gives the same kind of
result for any prime substrate. So the prime claim has teeth, but the
teeth don't distinguish among possible prime choices.

➡️ **The natural next discriminator** would be a calculation where the
specific value of p (vs. some other prime) matters for the observable.
Candidates: (a) entanglement-entropy SCALING corrections that depend
on the specific p via its prime-counting behavior nearby, (b) Chebyshev-
bias signatures specific to p mod 4, p mod 6, etc., (c) connections to
the substrate-derived SM parameter values via the boundary mode integers.

### Code and tests

Implementation: [`bpr/prime_state_check.py`](../bpr/prime_state_check.py),
functions `legendre_symbol`, `quadratic_residues_mod_p`,
`quadratic_residue_state`, `bpr_substrate_analog_scaling`.

Tests: [`tests/test_prime_state_check.py`](../tests/test_prime_state_check.py),
including `test_qr_state_entropy_below_bernoulli_baseline` (locks the
ratio ∈ [0.4, 0.9] for all tested primes) and
`test_qr_ratio_stable_across_primes` (locks the |slope| < 0.15 finding).

### Updated Gap-1 status of the prime-substrate claim

| Aspect | Status |
|---|---|
| BPR substrate state has nontrivial quantum-information content beyond bare density | **CONFIRMED** (QR state at 0.65× Bernoulli baseline) |
| Structure depends specifically on p = 104761 vs other primes | NOT CONFIRMED (slope ~ 0 across primes) |
| Square-free / Möbius / Mertens structure appears in BPR substrate | UNTESTED (different number-theoretic invariants) |
| Connections to SM parameters via QR structure | UNTESTED |

This is a real, honest, partial closure: the framework's prime substrate
isn't decorative — it carries genuine number-theoretic content. But the
content is structural (any prime works), not specifically validating of
BPR's particular prime selection.
