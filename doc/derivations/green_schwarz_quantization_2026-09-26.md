# Dirac quantization of the Green–Schwarz couplings: the family number is a multiple of three

2026-09-26. Status: exact characteristic-class algebra on the supplied BPR-6D
matter sector. The implementation is `bpr/green_schwarz_quantization.py`,
with tests in `tests/test_green_schwarz_quantization.py` and a demo in
`scripts/demo_green_schwarz_quantization.py`. Torsion refinements and global
(Ω₇ bordism) anomalies are not computed. The independent review is recorded
in section 9.

## 0. Question

The [chiral completion](chiral_parent_completion_2026-09-25.md) cancels the
local anomaly of the 6D parent with one added neutral spinor and one
Green–Schwarz 2-form. Its section 5 flagged an open issue: under a naive
integrality test, that minimal completion fails Dirac quantization of the
Green–Schwarz couplings. The [architecture decision](architecture_decision_2026-09-26.md)
therefore made BPR-6D provisional.

This is also the first condition that consistent quantum-gravity
completions of 6D theories are known to impose: the Green–Schwarz data must
embed in an integral, unimodular lattice of string charges. So the question
is both "is the matter sector consistent?" and "can BPR-6D pass the first
filter on the way to a UV completion?"

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

These three classes can be varied independently, so a 4-class built from
them is integral on all backgrounds iff its coefficients are integers. The one
exception is the standard Wu shift, a half-integral multiple of λ_T allowed
when it comes with a characteristic vector.

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

Matching coefficients fixes the Gram entries:

    b_V·b_V = 8α,  b_X·b_X = 2γ,  a·a = 32ζ,  b_V·b_X = 2β,  b_V·a = 8δ,  b_X·a = 4ε,

where α…ζ are the completion module's coefficients of S2², S2X², X⁴, S2p1,
X²p1 and p1². All of these must be integers, and a characteristic vector
gives b·b ≡ b·a (mod 2) and a·a ≡ σ(Λ) = 0 (mod 8).

## 3. Necessity: three divides the parent charge

For 16₊(q) ⊕ 16₋(0),

    I8 = q² x² S2 + (2/3) q⁴ x⁴ − (1/3) q² x² p1
       = (2q²/3) · x² · (3λ_V + q² x² − λ_T).

**Theorem 1.** For every unimodular lattice and any number of non-chiral
2-forms, the minimal completion satisfies the quantization conditions only if
**3 | q**.

*Proof.* b_X·b_X = 2γ = 4q⁴/3 and b_X·a = 4ε = −4q²/3 must both be integers,
and each holds iff 3 | q. The x⁴ coefficient alone already decides it: it is
½b_X·b_X ∈ ½ℤ, while 16q⁴/24 ∈ ½ℤ iff 3 | q. ∎

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

**The unit of U(1)_F charge is one third of the matter's charge.** In the old
convention (parent charge 1), the minimal completion is inconsistent. It is
consistent at this level only if the gauge group is U(1)_F with states of one
third of the parent's charge somewhere in the spectrum. Such states are:
- required by completeness;
- necessarily massive, because vector-like pairs do not change I8 (tested).

The gauge group is U(1)_F, not U(1)_F/ℤ₃. Below the mass of those states the
light theory has an approximate ℤ₃ one-form symmetry.

**Theorem 4 (family number).** For the minimal completion, Dirac
quantization forces n_gen = q|m| ∈ 3ℤ. The smallest value, **three**, is one
flux quantum of the unit-charge bundle.

*Proof.* By the index theorem, n_gen = q|m| (tested: q=3, m=1 gives exactly
three chiral 16s and nothing else). Theorem 1 requires 3 | q. ∎

The background is the same geometry as before. What was called "flux 3" in
parent units is **one** flux quantum in the true units. The family number
changes from an arbitrary input to a multiple of three, with three the
minimal case.

**Corollary 5 (the three-family vacuum is isolated).** Tune Λ so that the
three-family sector (m=1) is flat. Then no other flux sector has a
compactified minimum:
- m ≥ 2 violates m² < (4/3)·1², the flux-landscape bound of
  [flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md),
  Theorem 5.
- m = 0 has only a maximum: there is no flux stabilization. The tests check
  this symbolically and by grid.

By contrast, tuning Λ for six families (m=2) leaves the three-family sector
as an anti-de Sitter vacuum below it. So, among the Minkowski flux vacua, only
the three-family one has no lower-flux compact vacuum. Flux-changing
transitions need magnetically charged 2-branes, which completeness also
requires. A transition m=1 → 0 would lead into an unstabilized region, and
its rate and endpoint are not analysed.

**What is and is not derived.**
- *Derived.* Given the minimal completion, one 2-form lattice and
  characteristic-class Dirac quantization, n_gen is a multiple of three.
- *Not derived.* Why n_gen is three rather than six or nine. That still rests
  on the Λ tuning, or on the minimality or stability reading of Corollary 5.
- The status of the family count moves from "stipulated" to "conditional".

## 6. Near-minimal completions behave differently

Consider the 18-component completions 16₊(q) ⊕ 16₋(0) ⊕ 1₊(Q_a) ⊕ 1₋(Q_b).
- At q=1, exactly two pass: (Q_a,Q_b) = (0,4) and (3,1).
- Each adds four massless singlets per flux quantum, i.e. 12 at the old
  "flux 3".
- Neither forces 3 | n_gen.

So the family-number result is specific to the minimal completion. Minimality
now selects that completion **at q=3**: 16 added components, three families
at one flux quantum, and no extra massless matter.

## 7. Relation to a UV completion

Embedding the Green–Schwarz data in an integral unimodular lattice is the
first condition that string and F-theory compactifications satisfy. Passing
it is necessary, not sufficient. The next conditions are:
- global anomalies, the Ω₇ spin bordism of B(Spin(10)×U(1)), together with
  the 2-form's quadratic refinement;
- the existence of the required charge-1/3 states and magnetic 2-branes in an
  explicit completion;
- an explicit string or F-theory realization, or a supersymmetric embedding
  (decision note, section 7).

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

Pending at the time of writing. See the README entry for this date.

## 10. Limitations

- Characteristic-class quantization only. Torsion classes, the quadratic
  refinement beyond the Wu shift in λ_T, and Ω₇ bordism are not computed.
- The background class is spin manifolds with Spin(10) × U(1)_F bundles. The
  Spin×_{ℤ₂}Spin(10) structure available to this spectrum would add
  backgrounds, and so possibly add conditions, but it is not analysed.
- The charge-1/3 states are required, not constructed.
- Only non-chiral 2-forms are considered; chiral tensors change the
  gravitational anomaly and are outside the class.
- Three rather than six or nine families is not derived; it is the minimal
  case.
