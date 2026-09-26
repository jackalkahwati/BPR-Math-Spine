# The Morrison–Taylor tension and supersymmetric analogues of BPR-6D

2026-09-26. Status: exact anomaly arithmetic. The implementation is
`bpr/string_embedding.py`, with tests in `tests/test_string_embedding.py` and
a demo in `scripts/demo_string_embedding.py`. An independent review (section 6)
found a blocker in the first version: the "intrinsically non-supersymmetric
fork" held only on T=0. Its repairs are applied, and the fork is restated.

## 0. Question

Round 4 ([green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md))
found that BPR-6D's minimal matter sector is consistent only if the parent
16 carries U(1)_F charge 3 in units of the smallest charge. That makes the
family number a multiple of three, but requires states of charge 1 that are
absent from the massless spectrum.

Morrison and Taylor studied the massless charge lattice in 6D F-theory
models (arXiv:2108.02309, JHEP 12 (2021) 040):
- **Theorem.** For every 6D supergravity with an F-theory description, charge
  completeness is equivalent to the standard Mordell–Weil assumption.
- **Massless charges.** Massless charged states generate the full charge
  lattice of any U(1) factor whose anomaly coefficient satisfies a simple
  positivity condition, taken here as −a·b̃ > 0. The abstract states this as
  a finding across many F-theory models; this note calls it a conjecture.

The full text was not accessible here. The statements, including the exact
positivity condition and the height-pairing normalization of b̃, are taken
from the abstract and search summaries and are unverified against the text.

Does this conflict with BPR-6D, and what would a string embedding do to the
multiple-of-three result?

## 1. BPR-6D against the Morrison–Taylor statement

Take the Park–Taylor normalization, b̃ = 2b_X, which round 4 anchors to the
literature. Treating the charged 16₊ as the "hyperino":

    −a·b̃ = (1/6) Σ dim q² = (1/6)·16·9 = 24 > 0,   gcd of charges = 3.

**This is a formal analogue, not a test.**
- BPR-6D has no gravitino, so which chirality plays the hyperino role is a
  convention. The lattice U is isomorphic to −U. Swapping the labels gives
  −a·b̃ = −24 (tested).
- In supersymmetric theories −a·b̃ = (1/6)Σ_hypers q² ≥ 0 holds
  identically. If the paper's condition is −a·b̃ > 0, then for a U(1) it only
  says that charged matter exists.
- The statement concerns supersymmetric F-theory models, and BPR-6D is
  non-supersymmetric. No theorem is violated: completeness only requires the
  charge-1 states to exist, possibly massive.

**The tension is removable, at a cost.** Add one massless vector-like 6D pair
of charge-1 singlets, 1₊(1) ⊕ 1₋(1). Then (tested):
- the massless charges generate ℤ;
- I8 is unchanged, and so are the quantization data. Since Ω₇ = 0
  (round 5), I8 fixes the global anomaly too;
- the three families are unchanged;
- in 4D there is one vector-like pair of charge ±1 per flux quantum, with net
  chirality zero.

The costs are these. A 6D Dirac mass for the pair is gauge invariant, so
nothing protects its masslessness. Vector-like 10(1) or 16(1) pairs would
serve equally well, so the extra states need not be Standard-Model neutral.

## 2. Supersymmetric analogues: 6D (1,0) supergravity

The analysis uses the Kumar–Morrison–Taylor / Park–Taylor anomaly equations.
For SO(10) × U(1), the terms linear in F_U(1) (F·trF³ and F³·trF) vanish
identically, because SO(10) has no cubic Casimir, so no condition is missing.

**Validation.** The equations reproduce known spectra:
- SU(2) on a degree-b curve in ℙ² has genus (b−1)(b−2)/2, with 22
  fundamentals for b=1 and 54 fundamentals plus one adjoint for b=3.
- SO(10) on a −4 curve carries exactly two vectors, and on genus-0 curves
  n₁₆ = n + 4, n₁₀ = n + 6.
- SO(10) on ℙ² gives 5×16 + 7×10 on a line and 8×16 + 10×10 on a conic.

**Proposition 1 (10s on T=0).** The SO(10) quartic condition gives
n₁₀ = n₁₆ + 2(1 − g) with g adjoint hypermultiplets. On T=0 (lattice ℤ,
a = −3):
- without adjoints, only b=1 (n₁₆ = 5, n₁₀ = 7) and b=2 (n₁₆ = 8,
  n₁₀ = 10) survive;
- with adjoints, b=3 (9×16, 9×10, 1 adjoint) and b=4 (8×16, 4×10,
  3 adjoints) also pass (tested).

Every T=0 solution has 10s. This does not extend to larger T. The review
found a T=4 example, a genus-3 curve 4H − ΣE_i on dP₄, with 4×16, no 10s and
3 adjoints, that passes every anomaly equation. It has no Weierstrass model,
because its residual discriminant is not effective.

**Proposition 2 (explicit T=0 spectra).** Three spectra with b=1, five 16s
and seven neutral 10s, each checked against every equation (gravitational,
nonabelian and U(1)):
- **Three-net.** 16 charges (3,3,3,−3,−3) and 70 singlets of charge 6; gcd 3.
  Per unit flux it has 9 16s, 6 16bars and 420 chiral charge-6 singlets, so
  it matches BPR-6D only in its net count of 3.
- **Rescaled.** The same with charges divided by 3: (1,1,1,−1,−1) and
  singlets of charge 2.
- **gcd one, three net.** 16 charges (3,3,3,−3,−3) with singlets
  70×1(1) + 49×1(5) + 25×1(7). Its charge gcd is 1 and it still has 3 net
  16s per unit flux. So on T=0, 3 | n_gen is **allowed but not forced**.

**Lemma 3 (the normalization is always reducible on T=0).** Take an
anomaly-free G × U(1) spectrum on T=0, for any gauge group G. Dividing its
charges by their gcd gives an anomaly-free spectrum.

*Proof.* The U(1) equations are homogeneous, so only integrality can fail. On
T=0, b̃′ = Σdim q′²/18 and 3b̃′² = Σdim q′⁴ ∈ ℤ, and together these force
b̃′ ∈ ℤ (tested). Suppose b̃′ = n is odd. Then:
- Σdim q′² = 18n;
- Σdim q′⁴ = 3n²;
- q⁴ − q² = q²(q−1)(q+1) is always divisible by 12, so 4 | n(n−6).

That last condition is impossible for odd n. ∎

A scan at b=2 confirms the lemma: every anomaly-free spectrum with gcd > 1
that the scan found reduces consistently.

**Proposition 4 (on T=1, supersymmetry can force 3ℤ).** BPR-6D has one
non-chiral 2-form. Its supersymmetric counterpart is T=1, not T=0: the
self-dual tensor of the gravity multiplet plus one anti-self-dual tensor.
Take lattice U with a = (−2,−2) (a·a = 8 = 9 − T). A pure U(1) with 128
hypers of charge 3 and 117 neutral hypers is consistent, with:
- b̃ = (72, 24) ∈ 2U;
- a·b̃ = −192;
- b̃² = 3456;
- H − V + 29T = 273.

The same spectrum at charge 1 is inconsistent, because a·b̃ = −128/6 is not
an integer (tested). The review found the same on the odd lattice I₁,₁. So
anomaly cancellation in supersymmetric supergravity can push the massless
charges into 3ℤ, exactly as in BPR-6D. That is the situation the
Morrison–Taylor statement excludes from F-theory.

## 3. Consequence: the fork, restated

The earlier claim, that the multiple-of-three family number is
"intrinsically non-supersymmetric", was a T=0 artifact. The correct picture:
- On T=0 the U(1) normalization is never forced (Lemma 3). 3 | n_gen can
  still be chosen (Proposition 2).
- On T ≥ 1 it can be forced (Proposition 4).
- The real difference from BPR-6D is the repair. Non-supersymmetric BPR-6D
  admits an anomaly-neutral vector-like charge-1 pair. In 6D (1,0) every
  hypermultiplet contributes chirally to the anomaly, so a charge-1 hyper
  cannot be added without changing it.

The fork is therefore between **anomaly-forced 3 | n_gen** and **F-theory
embeddability**, if the Morrison–Taylor statement holds, not between
non-supersymmetric and supersymmetric theories. Under Morrison–Taylor, a
spectrum like Proposition 4's is either completed by extra massless charged
states or lies in the swampland. The gcd-3 three-net spectrum is either a
relabelling of the rescaled one or in the swampland.

There are two branches:
- **Non-supersymmetric BPR-6D.** It keeps n_gen ∈ 3ℤ and needs massive, or
  vector-like massless, charge-1 states. No string realization is known.
  Tachyon-free non-supersymmetric strings exist (SO(16) × SO(16), Sugimoto's
  USp(32), 0′B, 6D heterotic orbifolds), but they are generically
  destabilized by dilaton tadpoles.
- **A supersymmetric BPR-6D′ in F-theory.** If Morrison–Taylor holds,
  3 | n_gen can occur there but is not forced. It is not constructed here:
  vacuum existence with a non-R U(1) flux and F-theory realizability are
  both open.

In 6D (1,0), hypermultiplets have no Yukawa couplings among themselves, so
Yukawa couplings would still have to come from gauge–Higgs or
brane-localized terms (rounds 3 and 8).

## 4. What this settles

- The Morrison–Taylor tension is not a contradiction. The statement is
  supersymmetric, its premise is a convention for BPR-6D, and a vector-like
  charge-1 pair removes the tension, though its masslessness is unprotected.
- Anomaly-forced 3 | n_gen and an F-theory embedding pull in opposite
  directions. This is recorded as a fork, not resolved.

## 5. Limitations

- The Morrison–Taylor statements are taken at abstract level.
- The supersymmetric analysis is anomaly cancellation only: T=0 for
  SO(10) × U(1), and a pure U(1) on T=1. It does not establish F-theory
  realizability or a vacuum with U(1) flux: a non-R U(1) monopole on S² is
  not a supersymmetric vacuum.
- Lattice refinements beyond b̃ ∈ 2Λ (Monnier–Moore–Park) are not checked.
  On T=0 they cannot affect the results, since b̃ ∈ 2Λ is automatic there.

## 6. Independent review

An independent review, with its own scripts, verified:
- the SO(10) constants A, B, C from the weights, and the anomaly equations;
- that no condition is missing for SO(10) × U(1);
- every validation anchor;
- the explicit spectra by hand;
- Lemma 3 on T=0.

Its findings, all repaired:
- **Blocker.** "Intrinsically non-supersymmetric" was a T=0 artifact. T=1 is
  the right counterpart, and there supersymmetry can force 3ℤ
  (Proposition 4). A gcd-1 spectrum with three net families shows that
  3 | n_gen is not forced on T=0. The fork is restated in section 3.
- **Major.** The premise −a·b̃ > 0 is a chirality convention for BPR-6D. It
  is now called a formal analogue, and the flag is renamed.
- **Minor:**
  - Proposition 1 is restricted to T=0, with adjoint solutions added.
  - Lemma 3 is strengthened to any gauge group, with the integrality step
    justified.
  - The costs of the vector-like pair are stated.
  - The first example is relabelled "three-net" and its chiral content given.
  - Code: the neutral-hyper count must be non-negative; the construction of
    b̃ is documented; an unused function is removed.
  - The wording about string models and the Morrison–Taylor status is
    corrected.
