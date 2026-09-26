# The Morrison–Taylor tension and supersymmetric analogues of BPR-6D

2026-09-26. Status: exact anomaly arithmetic. The implementation is
`bpr/string_embedding.py`, with tests in `tests/test_string_embedding.py` and
a demo in `scripts/demo_string_embedding.py`. The independent review is
recorded in section 6.

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
- **Conjecture.** Massless charged states generate the full charge lattice of
  any U(1) factor whose anomaly coefficient satisfies −a·b̃ > 0.

The full text was not accessible here; the statements are taken from the
abstract and search summaries. The standard height-pairing normalization of
b̃ is assumed.

Does this conflict with BPR-6D, and what would a string embedding do to the
multiple-of-three result?

## 1. BPR-6D against the conjecture

Take the Park–Taylor normalization, b̃ = 2b_X, which round 4 anchors to the
literature. Then:

    −a·b̃ = (1/6) Σ dim q² = (1/6)·16·9 = 24 > 0,   gcd of charges = 3.

So the conjecture's premise holds and its conclusion fails.

Two remarks follow.
- **The conjecture does not apply as stated.** It concerns supersymmetric
  F-theory models, and BPR-6D is non-supersymmetric. No theorem is violated,
  because completeness only requires the charge-1 states to exist, possibly
  massive.
- **The tension is removable.** Add one massless vector-like 6D pair of
  charge-1 singlets, 1₊(1) ⊕ 1₋(1). Then:
  - the massless charges generate ℤ;
  - I8 is unchanged, and so are the quantization data and (since bordism does
    not see charges) the global analysis;
  - the three families are unchanged;
  - the 4D effect is one vector-like pair of Spin(10)-singlet states per flux
    quantum, with net chirality zero.

  All of this is tested. A string-like completion of BPR-6D would therefore
  predict light, Standard-Model-neutral, vector-like states carrying one third
  of the matter's U(1)_F charge. This costs nothing else.

## 2. Supersymmetric analogues: 6D (1,0) with SO(10) × U(1)

String and F-theory models are supersymmetric, so the natural question is
what happens to the gauge and matter sector in 6D N=(1,0) supergravity. The
analysis uses the Kumar–Morrison–Taylor / Park–Taylor anomaly equations on
T=0 (lattice ℤ, a = −3).

**Validation.** The equations reproduce known spectra:
- SU(2) on a degree-b curve has genus (b−1)(b−2)/2, with 22 fundamentals for
  b=1 and 54 fundamentals plus one adjoint for b=3.
- SO(10) on a −4 curve carries exactly two vectors.

**Proposition 1 (10s are forced).** The SO(10) quartic condition gives
n₁₀ = n₁₆ + 2 for 16 and 10 hypermultiplets. On T=0 only two anomaly
coefficients survive:
- b=1, with n₁₆ = 5 and n₁₀ = 7;
- b=2, with n₁₆ = 8 and n₁₀ = 10.

Hypermultiplets in the 10, the SO(10) Higgs representation, are therefore
unavoidable in any supersymmetric version.

**Proposition 2 (explicit spectra).** Two anomaly-free T=0 spectra, both
checked against every equation (gravitational, nonabelian and U(1)):
- **BPR-like.** 16 charges (3,3,3,−3,−3), seven neutral 10s, 70 singlets of
  charge 6 and 99 neutral hypers. The gcd is 3, and there are 3 net families
  per unit flux.
- **Rescaled.** The same spectrum with charges divided by 3: (1,1,1,−1,−1) and
  singlets of charge 2. It is also anomaly-free.

**Lemma 3 (the normalization is always reducible on T=0).** Take any
anomaly-free SO(10) × U(1) spectrum on T=0. Dividing its charges by their gcd
gives an anomaly-free spectrum.

*Proof.* The U(1) equations are homogeneous. The only integrality condition
left is β′ = β/g² ∈ ℤ, and on T=0, β′ ∈ ½ℤ. Suppose β′ = n/2 with n odd. Then:
- Σdim q′² = 36β′ = 18n;
- Σdim q′⁴ = 12β′² = 3n²;
- q⁴ − q² = q²(q−1)(q+1) is always divisible by 12, so 4 | n(n−6).

That last condition is impossible for odd n. ∎

A scan at b=2 confirms it: every anomaly-free spectrum with gcd > 1 that the
scan found reduces consistently.

## 3. Consequence: a fork

The multiple-of-three family number is **intrinsically non-supersymmetric**:
- In BPR-6D (non-SUSY, minimal content), the U(1)_F normalization is forced
  by the fermion anomaly coefficient 16q⁴/24, and 3 | q follows.
- In the supersymmetric analogue, anomaly cancellation itself allows charge 1
  for the 16s (Lemma 3). The family number is then q·m with no divisibility
  constraint.

So there are two branches:
- **Non-supersymmetric BPR-6D.** It keeps n_gen ∈ 3ℤ and needs massive (or
  vector-like massless) charge-1 states. No string realization is known, and
  non-supersymmetric string vacua are rare and typically unstable.
- **Supersymmetric BPR-6D′.** It is plausibly within reach of the F-theory
  landscape. It is not constructed here: vacuum existence with a non-R U(1)
  flux and F-theory realizability are both open. It loses the
  multiple-of-three result, but it gains forced 10 hypermultiplets.

In 6D (1,0), hypermultiplets have no Yukawa couplings among themselves.
Yukawa couplings would still have to come from gauge–Higgs or brane-localized
terms (see the round-3 family note).

## 4. What this settles

- The Morrison–Taylor tension is not a contradiction. The conjecture is
  supersymmetric, and a vector-like charge-1 pair removes the tension at no
  cost.
- A string embedding and the multiple-of-three family number pull in
  opposite directions. This is recorded as a genuine fork, not resolved.

## 5. Limitations

- The Morrison–Taylor statements are taken at abstract level.
- The supersymmetric analysis is anomaly cancellation on T=0 only. It does
  not establish F-theory realizability or a vacuum with U(1) flux: a non-R
  U(1) monopole on S² is not a supersymmetric vacuum.
- Only SO(10) × U(1) with 16, 10 and singlet hypermultiplets is considered.

## 6. Independent review

Pending at the time of writing.
