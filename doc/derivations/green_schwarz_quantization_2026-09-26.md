# Dirac quantization of the Green–Schwarz couplings: the family number is a multiple of three

2026-09-26. Status: exact characteristic-class algebra on the supplied BPR-6D
matter sector. The implementation is `bpr/green_schwarz_quantization.py`,
with tests in `tests/test_green_schwarz_quantization.py` and a demo in
`scripts/demo_green_schwarz_quantization.py`. Torsion refinements and global
(Ω₇ bordism) anomalies are not computed. An independent adversarial review
(section 9) found no blocker. It found three major scope issues and seven
minor ones; its repairs are applied.

## 0. Question

The [chiral completion](chiral_parent_completion_2026-09-25.md) cancels the
local anomaly of the 6D parent with one added neutral spinor and one
Green–Schwarz 2-form. Its section 5 flagged an open issue: under a naive
integrality test, that minimal completion fails Dirac quantization of the
Green–Schwarz couplings. The [architecture decision](architecture_decision_2026-09-26.md)
therefore made BPR-6D provisional.

It is also one of the conditions that consistent quantum-gravity
completions of 6D theories are known to impose, after local anomaly
cancellation: the Green–Schwarz data must embed in an integral, unimodular
lattice of string charges. So the question is both "is the matter sector
consistent?" and "does BPR-6D pass this filter on the way to a UV
completion?"

## 1. Integral generators

The backgrounds are spin 6-manifolds (and the 8-manifolds used for anomaly
polynomials) carrying Spin(10) × U(1)_F bundles. In the index-density
normalization of the completion module:
- **Spin(10).** With ch = tr e^F, ch₂(V⊗ℂ) = ½tr₁₀F² = p₁(V). The module's
  S2 is defined by tr₁₀F² = 2S2, so **S2 = p₁(V)**. For Spin(10) bundles
  p₁(V) = 2λ_V, where λ_V (the instanton number) generates H⁴(BSpin(10);ℤ).
  So S2 = 2λ_V.
- **Tangent bundle.** On spin manifolds p₁ = 2λ_T, with λ_T integral and
  λ_T ≡ w₄ (mod 2), which is the Wu class ν₄.
- **U(1)_F.** x = c₁ of the line bundle of **unit** charge. The parent's
  charge is written q in these units, so X_parent = q·x.

These three universal classes can be varied independently, so a universal
4-class built from them is integral iff its coefficients are integers. The one
exception is the standard Wu shift, a half-integral multiple of λ_T allowed
when it comes with a characteristic vector.

The characteristic-vector condition comes from the 8D anomaly theory. On
closed spin 6-manifolds ν₄ = 0, so λ_T is even there. Restricting to
6-manifolds does not rescue q=1: the factorization is unique up to scale, and
the x² coefficient still needs 2/(3k) ∈ ℤ.

## 2. The quantization condition

Green–Schwarz cancellation with 2-form fields whose string charges form a
unimodular lattice Λ requires

    I8 = ½⟨Y, Y⟩,   Y = b_V λ_V + b_X x² + (a/2) λ_T,

with b_V, b_X ∈ Λ and a a characteristic vector of Λ. Non-chiral 2-forms give
Λ of signature (n,n). For a single non-chiral 2-form, Λ = U, the even
hyperbolic plane (the Dirac–Schwinger–Zwanziger pairing of electric and
magnetic strings). The condition then reads I8 = Y_e·Y_g with Y_e and Y_g
integral.

This is the characteristic-class ("naive") form of the conditions developed
for 6D supergravity:
- Seiberg–Taylor, JHEP 06 (2011) 001;
- Kumar–Morrison–Taylor, JHEP 11 (2010) 118;
- Monnier–Moore–Park, JHEP 02 (2018) 020.

Here it is adapted to non-chiral 2-forms without supersymmetry.

**Normalization anchor.** For charged Weyl fermions of chirality +1, the
module gives b_X·b_X = Σq⁴/12 and b_X·a = −Σq²/12. These are Park–Taylor's
U(1) conditions, 3b₁₁·b₁₁ = Σq⁴ and a·b₁₁ = −Σq²/6, with b₁₁ = 2b_X, i.e. the
even height pairing of F-theory. For example, 108 charge-1 fields give
b_X = K̄ = 3H on ℙ², with a = K. The tests check this for random spectra, so
the factors of ½ are tied to the literature and not only to this repository's
trace tables.

Matching coefficients fixes the Gram entries:

    b_V·b_V = 8α,  b_X·b_X = 2γ,  a·a = 32ζ,  b_V·b_X = 2β,  b_V·a = 8δ,  b_X·a = 4ε,

where α…ζ are the completion module's coefficients of S2², S2X², X⁴, S2p1,
X²p1 and p1². All of these must be integers, and a characteristic vector
gives b·b ≡ b·a (mod 2) and a·a ≡ σ(Λ) = 0 (mod 8).

## 3. Necessity: three divides the parent charge

For 16₊(q) ⊕ 16₋(0),

    I8 = q² x² S2 + (2/3) q⁴ x⁴ − (1/3) q² x² p1
       = (2q²/3) · x² · (3λ_V + q² x² − λ_T).

**Theorem 1.** For every integral lattice, unimodular or not, and any
number of 2-forms, chiral tensors included, the minimal completion satisfies
the quantization conditions only if **3 | q**.

*Proof.* b_X·b_X = 2γ = 4q⁴/3 and b_X·a = 4ε = −4q²/3 must both be integers,
and each holds iff 3 | q. The x⁴ coefficient alone already decides it:
- it equals ½b_X·b_X, which lies in ½ℤ;
- chiral tensors contribute nothing to x⁴;
- 16q⁴/24 ∈ ½ℤ iff 3 | q. ∎

The theorem assumes the anomaly is cancelled by 2-form Green–Schwarz terms.
A 6D Stückelberg axion could cancel I8 = x∧X₆, but it would make U(1)_F
massive in six dimensions. The flux vacuum needs a massless 6D U(1)_F.

The minimal completion is unique only within class 𝒞, which admits no extra
massless Spin(10) matter. Outside 𝒞, 16₊(1) ⊕ 16₋(2) also has 16 added
components and passes at q=1 (Y_e = −2x², Y_g = 3λ_V + 5x² − λ_T). But it
leaves one 16 and two 16bar massless per flux quantum, so its net chirality
(1−2)m is not forced into 3ℤ (tested). The multiple-of-three result depends
on 𝒞's restriction.

At q=1, which is the convention used so far, the Gram entry b_X·b_X is 4/3.
That is the failure the completion note recorded, and it is now shown to hold
for every lattice, not just the naive one.

## 4. Sufficiency: an explicit solution

**Theorem 2.** If q = 3t, one non-chiral 2-form suffices:

    Y_e = 6t² x²,   Y_g = 3λ_V + 9t² x² − λ_T,   I8 = Y_e·Y_g.

In U = ⟨e, g⟩ with e·g = 1, take t=1:
- b_V = 3g;
- b_X = 6e + 9g;
- a = −2g, which is characteristic because U is even.

The Gram entries (0, 108, 0, 18, 0, −12) match every requirement in
section 2.

*Checks.* The tests verify:
- the integral polynomial against direct substitution into the symbolic
  anomaly polynomial;
- the factorization;
- the Gram matrix;
- that an independent brute-force integer search (linear algebra over a box
  of Y_e, not factorization) finds (6x²)(3λ_V + 9x² − λ_T) at q=3 and finds
  nothing at q=1.

A scan over q ≤ 12 gives passing charges exactly {3, 6, 9, 12}, and the
necessary conditions and U-lattice existence agree for every q.

**Proposition 3 (odd lattice).** The odd lattice I₁,₁ = diag(1,−1), i.e. one
self-dual plus one anti-self-dual tensor with odd characteristic a, never
works for the minimal completion.

*Proof.* Write P = Y₁−Y₂ and R = Y₁+Y₂. Then PR = 2I8, P and R agree in
parity in λ_V and x², and they have opposite parity in λ_T. Up to order the
factors are c·x² and k·(3λ_V + q²x² − λ_T), with c and k integers.
- The λ_V coefficients are 0 and 3k. Equal parity makes k even.
- The λ_T coefficients are 0 and −k. Opposite parity makes k odd.

These contradict each other. ∎

The test's brute-force search confirms this. It also succeeds on a positive
control, 1₊(0)⊕1₊(1)⊕1₋(3)⊕1₋(4), so the search is not vacuous.

## 5. What this means physically

**The unit of U(1)_F charge is at most one third of the matter's charge.** In
the old convention (parent charge 1), the minimal completion is inconsistent.
It is consistent at this level only if the gauge group is U(1)_F, with states
carrying a smaller charge somewhere in the spectrum. For q=3 that charge is
one third of the parent's. About these states:
- Completeness requires them.
- They are allowed, since vector-like pairs do not change I8 (tested), but
  that does not force them to be massive.
- In BPR-6D as defined they are not in the light spectrum, so they are
  massive. Massless vector-like charge-1 fermions or charged scalars would
  also be consistent, and would keep n_gen = q|m|.

The gauge group is U(1)_F, not U(1)_F/ℤ₃. Below the mass of those states, the
light theory has an approximate ℤ₃ one-form symmetry. That symmetry cannot be
gauged: in a ℤ₃ 2-form background, Y_e = 6x² acquires a non-integral (2/3)B̃²
term. This is the q=1 failure restated.

**A tension with known string realizations.** Morrison and Taylor (JHEP 12
(2021) 040) show that in 6D F-theory models the massless states generate the
full charge lattice whenever the anomaly coefficients satisfy a positivity
condition. Here b_X·b_X = 108 and −a·b_X = 12 are both positive. The precise
condition was not checked here; the review saw the abstract only. If it
applies, then a massless spectrum confined to charges in 3ℤ is exceptional in
the best-understood part of the string landscape. An F-theory realization of
BPR-6D would then be expected to carry massless states of charge 1 as well.

**Theorem 4 (family number).** For the minimal completion in class 𝒞,
Dirac quantization forces n_gen = q|m| ∈ 3ℤ.

*Proof.* By the index theorem, n_gen = q|m| (tested: q=3, m=1 gives exactly
three chiral 16s and nothing else). Theorem 1 requires 3 | q. ∎

Three families needs **both** q=3 and |m|=1. These are two further minimal
choices, not consequences:
- every q ∈ 3ℤ passes with the same component count;
- at q=6, n_gen ∈ 6ℤ, and three families is impossible.

With q=3, what was called "flux 3" in parent units is one flux quantum of the
charge-1 bundle. This assumes the smallest charge is exactly q/3. The family
number changes from an arbitrary input to a multiple of three, with three the
minimal case.

**Corollary 5 (the three-family vacuum is isolated, classically).** Take
q=3 and tune Λ so that the three-family sector (|m|=1) is flat. Then no flux
sector with |m| ≠ 1 has a compactified minimum:
- |m| ≥ 2 violates m² < (4/3)·1², the flux-landscape bound of
  [flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md),
  Theorem 5.
- m = 0 has only a maximum: there is no flux stabilization. The tests check
  this symbolically and by grid.

By contrast, tuning Λ for six families (m=2) leaves the three-family sector
as an anti-de Sitter vacuum below it. So, among the Minkowski flux vacua, only
the three-family one has no lower-flux compact vacuum. Flux-changing
transitions need magnetically charged 2-branes, which completeness also
requires. A transition m=1 → 0 would lead into an unstabilized region, and
its rate and endpoint are not analysed. The corollary is classical only:
quantum corrections to the radion potential can be of the same order (flux
note, section 9).

**What is and is not derived.**
- *Derived.* Given the minimal completion in 𝒞, 2-form Green–Schwarz terms
  and characteristic-class Dirac quantization, n_gen is a multiple of three.
- *Not derived.* Why n_gen is three rather than six or nine. That needs
  q=3 and |m|=1, which rest on minimality, the Λ tuning, or the stability
  reading of Corollary 5.
- The status of the family count moves from "stipulated" to "conditional".

## 6. Near-minimal completions behave differently

Consider the 18-component completions 16₊(q) ⊕ 16₋(0) ⊕ 1₊(Q_a) ⊕ 1₋(Q_b).
- At q=1 with |Q| ≤ 4, exactly two pass: (Q_a,Q_b) = (0,4) and (3,1).
- Each adds four massless singlets per flux quantum, i.e. 12 at the old
  "flux 3".
- Larger charges pass too, for example (0,8) and (3,5), with more massless
  singlets.
- None of them forces 3 | n_gen.

So the family-number result is specific to the minimal completion. The
smallest consistent option in 𝒞 is that completion at q=3: 16 added
components, three families at one flux quantum, and no extra massless matter.
Choosing q=3 over 6, 9, … is itself a minimality choice.

## 7. Relation to a UV completion

Local anomaly cancellation comes first; embedding the Green–Schwarz data in
an integral unimodular lattice is one of the further conditions that string
and F-theory compactifications satisfy. Passing it is necessary, not
sufficient. The remaining conditions include:
- global anomalies. Round 5 settles these for the Spin × Spin(10) × U(1)
  structure: Ω₇^Spin(B(Spin(10)×U(1))) = 0
  ([global_anomalies_2026-09-26.md](global_anomalies_2026-09-26.md));
- the existence of the required charge-1/3 states and magnetic 2-branes in an
  explicit completion;
- an explicit string or F-theory realization, or a supersymmetric embedding
  (decision note, section 7);
- the massless-charge-lattice tension above (Morrison–Taylor 2021);
- other swampland conditions not examined here.

## 8. Consequences for other notes

- **Chiral completion, section 5.** The Dirac-quantization bullet is resolved
  at the characteristic-class level: the completion passes iff the parent
  charge is a multiple of three.
- **Architecture decision.** The adoption is no longer provisional on the
  naive test, but global anomalies remain.
- **Unification map.**
  - family_count moves to conditional.
  - A new node gs_quantization is exact at the characteristic-class level.
- **Family note.** Nothing changes. Only k = q·m = 3 enters.

## 9. Independent review

An independent adversarial review verified the following:
- **I8 from Chern roots.** It recomputed I8 with Â = Π(y/2)/sinh(y/2) and the
  explicit 16 weights, and found an exact match.
- **Generators.** S2 = p₁(V), λ_V generates H⁴(BSpin(10)), λ_T ≡ w₄ = ν₄, and
  X_parent = qx.
- **Literature consistency.** The standard supergravity U(1) congruences
  (Σq⁴, Σq² ≡ 0 mod 12, given integral a and b) reduce to 3 | q for this
  spectrum.
- **Theorem 2 and Proposition 3.** The Gram data and the parity argument are
  correct.
- **Citations.** All correct.
- **Corollary 5.** Correct.

It found no blocker. Its findings, all repaired above:
- **Major:**
  - uniqueness holds only in 𝒞 (counterexample 16₊(1)⊕16₋(2));
  - q=3 and |m|=1 are extra choices;
  - the charge-1 states are allowed, not forced, to be massive, and the
    Morrison–Taylor tension was omitted, along with an overclaimed "first
    condition" framing.
- **Minor:**
  - Theorem 1 strengthened (any integral lattice, chiral tensors) and the
    Stückelberg alternative excluded;
  - 6D-versus-8D integrality;
  - the Spin×ℤ₂Spin(10) limitation partly closed (below);
  - the ℤ₃ symmetry cannot be gauged;
  - Corollary 5 is classical, with |m|≠1;
  - the near-minimal "exactly two" holds for |Q|≤4 only;
  - test hardening: a rational q=1 failure test, the Park–Taylor anchor, the
    out-of-𝒞 counterexample, and the untruncated divisor search.

## 10. Limitations

- Characteristic-class quantization only. Torsion classes, the quadratic
  refinement beyond the Wu shift in λ_T, and Ω₇ bordism are not computed.
- The background class is spin manifolds with Spin(10) × U(1)_F bundles. The
  Spin×_{ℤ₂}Spin(10) structure available to this spectrum adds backgrounds:
  - Necessity survives, because spin backgrounds are a subset.
  - The review argues that the q=3 solution also survives. Its argument:
    3λ_V − λ_T = p₁(V) + (p₁(V) − p₁(T))/2, and p₁ ≡ 𝔓(w₂) + 2w₄ (mod 4)
    makes the second term integral when w₂(V) = w₂(T). This is not tested
    here.
- The charge-1/3 states are required, not constructed.
- Only non-chiral 2-forms are considered; chiral tensors change the
  gravitational anomaly and are outside the class.
- Three rather than six or nine families is not derived. It needs q=3 and
  |m|=1, both minimal choices.
- The result depends on class 𝒞 and on 2-form Green–Schwarz terms.
- A known F-theory result suggests that massless charge-1 states would
  normally be present (section 5).
