# Families from flux on S²: an SU(2) family symmetry, and no minimal Yukawa

2026-09-26. Status: exact results on a supplied background. The
implementation is `bpr/sphere_family_structure.py`, with tests in
`tests/test_sphere_family_structure.py` and a demo in
`scripts/demo_sphere_family_structure.py`. The methods are explicit
spin-weighted harmonics, eth operators, 6D gamma matrices and
Clebsch–Gordan algebra.

The background is the flux vacuum of
[flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md).
The field content is the minimal completion of
[chiral_parent_completion_2026-09-25.md](chiral_parent_completion_2026-09-25.md).

**The main outcome is negative.** The minimal BPR-6D content has no
renormalizable, or low-derivative, Yukawa coupling among the three flux
families. The Yukawa sector is therefore an open problem, not a result.

## 0. Question

The index theorem gives three chiral 16s at flux 3. What else does the sphere
imply for them? Specifically:
- how the three families transform under the sphere's symmetry;
- which Higgs couplings are allowed;
- what the allowed couplings can and cannot say about masses.

## 1. Zero modes form one SU(2) multiplet

Consider a 6D Weyl spinor of U(1)_F charge Q in flux m, and let k=|Qm|. On S²
its two components are sections of spin weight

    s₊ = (1−k)/2   and   s₋ = s₊ − 1.

Here the charge enters as an effective spin-weight shift of −Qm/2. The
internal Dirac operator is built from ð and ð̄.

**Theorem 1.** Take k ≥ 1.
- The Dirac kernel is spanned by the spin-weighted harmonics
  ₍₋ⱼ₎Y_{j,μ}, for μ = −j, …, j, with j=(k−1)/2.
- These are annihilated by ð̄ and have one fixed internal chirality.
- The other component has no kernel.
- The k zero modes therefore form one irreducible spin-j multiplet of the
  SU(2) isometry of the round sphere.
- The massive levels have

      m² r² = (l + ½)² − k²/4,   l = j, j+1, …,

  with degeneracy 2l+1.

For k=3 the three families are an SU(2) triplet (j=1), and the first massive
level is at m² = 4/r².

*Checks.*
- The Goldberg-formula harmonics match SymPy's Y_lm at s=0.
- They are orthonormal by quadrature.
- They obey the standard ladder relations ð ₛY = +√((l−s)(l+s+1)) ₛ₊₁Y and
  ð̄ ₛY = −√((l+s)(l−s+1)) ₛ₋₁Y, including at half-integer labels.
- For k = 1…4, the tests confirm:
  - explicit annihilation of the zero modes;
  - J_z weights −j…j, each appearing once;
  - that ð acting on the other component has no kernel.
- The massive levels follow from −ðð̄ ₛY = (l+s)(l−s+1) ₛY. Explicit
  derivatives check them at l = 1…4 for k = 3.

This reproduces the classic statement (RSS 1983; Witten 1983) that
flux-induced chiral families carry isometry quantum numbers.

## 2. Six-dimensional chirality lemma

**Lemma 2.** Let ψ and χ have the same 6D chirality. Then:
- the scalar bilinears ψ̄χ and ψᵀCχ vanish, for both charge conjugations
  C±;
- the vector bilinears ψ̄Γ^Mχ and ψᵀCΓ^Mχ do not vanish.

This matches Spin(6)≅SU(4): 4⊗4 = 6⊕10 contains no singlet.

*Checks.* The tests build 8×8 gamma matrices of signature (−,+,+,+,+,+) and
verify:
- the Clifford relations;
- that the chirality matrix is Hermitian, squares to 1 and anticommutes with
  every Γ^M;
- that both charge conjugations are unique up to scale.

They then evaluate random same-chirality spinors: every scalar bilinear
vanishes to 10⁻¹⁵ and every vector bilinear is O(1). As a control,
Cᵀ-pairing between opposite chiralities has full rank 4.

The three families all come from the single 6D field 16₊. So a 16·16·H Yukawa
among them cannot use a 6D scalar H. The same conclusion follows
independently from spin weight: the integrand of a scalar Yukawa has total
spin weight 1 ≠ 0 (section 3).

## 3. Selection rules: only an internal-vector J=2 Higgs couples

U(1)_F neutrality of 16·16·H forces H to have F-charge −2. A component of H
with tangent spin weight s_h then has effective spin weight s_h + k. The
integrand of the triple overlap must have total spin weight zero:

    2 s₊ + s_h + k = 1 + s_h = 0   ⇒   s_h = −1.

So only the internal one-form component contributes, specifically the ∂_z̄
component. Its modes have effective spin weight k−1=2, so SU(2) spin
J ≥ 2. The families have j=1, and 1⊗1 = 0⊕1⊕2. Hence exactly one channel
survives: **J = 2**.

**Theorem 3.** With the families' j=1 fixed, the triple overlap is

    ∫ ₋₁Y_{1,m₁} ₋₁Y_{1,m₂} ₂Y_{2,m₃} dΩ
        = √(45/4π) (1 1 2; m₁ m₂ m₃)(1 1 2; 1 1 −2).

The two factors are Wigner 3j symbols. All nine (m₁,m₂) pairs are checked
against this closed form by quadrature. The Yukawa tensor is therefore the
Clebsch–Gordan tensor ⟨1 m₁ 1 m₂|2 M⟩, and it is symmetric in the two
families.

**Corollary (minimal BPR-6D has no zero-mode Yukawa).**
- A 6D scalar 10 fails by Lemma 2 and by spin weight.
- A derivative coupling ψᵀCΓ^aψ D_aH of a scalar 10 also fails. H has
  effective spin weight k=3, so all its modes have J ≥ 3, and none couples to
  1⊗1.
- The Spin(10)×U(1)_F gauge fields contain no 10.

The only viable channel is therefore an internal-vector 10 of F-charge −2,
which is not in the minimal content.

## 4. What a J=2 vev can and cannot say about masses

Given the channel, the mass matrix of one sector is M = Σ_M v_M C^M, where
the C^M are the J=2 Clebsch–Gordan matrices. Equivalently, M is a complex
symmetric traceless 3×3 matrix in a Cartesian basis.

**Theorem 4 (orientations).**
1. *Highest-weight (null) vev, v ∝ e_{M=2}.* M has rank 1, with spectrum
   (1, 0, 0). One family is massive at leading order.
2. *Real vev, v_{−M} = (−1)^M v̄_M.* M is a real traceless symmetric matrix.
   Its singular values are the absolute eigenvalues, so

       m_heaviest = m_middle + m_lightest   (exact sum rule).

   The uniaxial vev gives 2:1:1.
3. *General complex vev.* Every ordered triple of masses is reachable. The
   explicit construction takes U with UᵀU = [[it, w, 0], [w, it, 0],
   [0, 0, −i]], where t = c/(a+b) and w = √(1−t²). Then M = U diag(a,b,c) Uᵀ
   is traceless.

*Checks.*
- Twenty random real traceless matrices reproduce their absolute eigenvalues.
- Two hundred random real vevs satisfy the sum rule to 10⁻¹⁵.
- Fifty random targets, and an up-type-like hierarchy (1, 7.3e−3, 1.3e−5),
  are reached to 10⁻¹².

*Reading.*
- The real-vev sum rule is far from any charged-fermion sector (for example
  m_t ≫ m_c + m_u), so real orientations are excluded.
- A hierarchy means a near-null complex orientation.
- The symmetry constrains the form of the Yukawa but predicts no mass ratio.
- With one J=2 multiplet of 10s, the five 10s can supply H_u and H_d with
  different orientations. That is the only freedom available for CKM mixing.
  Its dynamics (the Higgs potential) is not specified.

## 5. The obvious completion: SO(12) gauge–Higgs, and why it fails as stated

SO(12) ⊃ SO(10)×U(1) contains the needed field. The adjoint 66 contains
10_{±2}, whose internal components are exactly the s_h=−1, charge −2 one-form
(Manton 1979-type gauge–Higgs). Two problems follow.

**Proposition 5a (even family number).** A 6D Weyl 32 decomposes as
16_{+1} ⊕ 16bar_{−1}, which gives 2k left-handed 16s (tests: 2, 4, 6, 8 for
k = 1…4). Three families are impossible without a further projection, such as
an orbifold.

**Proposition 5b (tachyonic Higgs level).** Take internal gauge components of
monopole number n = 2k.
- Their lowest level has m² r² = −n/2, with degeneracy n−1 and SU(2) spin
  n/2−1.
- This follows from rough Laplacian + Ricci + gyromagnetic term, with the
  aligned component carrying effective spin weight |n|/2−1.
- For k=3 this is the J=2 quintet at m² = −3/r².

This is the Randjbar-Daemi–Salam–Strathdee instability (1983). The Higgs is
tachyonic at the compactification scale, about 10¹⁷ GeV, not near the weak
scale.

*Checks.*
- The formula reproduces the Nielsen–Olesen value −qB r² = −n/2.
- For n = 1…10, the number of complex negative modes equals n−1. That is the
  Atiyah–Bott (1983) Morse index, 2(n−1) real directions, of the SU(2)
  Yang–Mills critical point of W-degree n on S².
- n=1 is stable, at m² = +1/2.
- The opposite helicity assignment would give no negative mode for n ≤ 4,
  contradicting Atiyah–Bott, so the test discriminates between the two.
- A full fluctuation computation was not done.

## 6. The family symmetry is gauged

The SU(2) isometry gives massless 4D gauge bosons at tree level, from g_μa
along the Killing vectors. The three families form their triplet, so these
bosons mediate family-changing neutral currents. They must be massive far
above the weak scale.

A J=2 Higgs vev breaks SU(2) only at the weak scale. A separate high-scale
breaking is needed, for example:
- a squashed sphere: the index still gives k zero modes, but the SU(2) is
  broken;
- another vev with J ≠ 0.

No such mechanism is supplied or derived here.

## 7. Consequences for the program

- **Derived.** The families are an SU(2) triplet, their KK spectrum, the
  chirality lemma, the unique J=2 channel, the orientation theorems, and the
  SO(12) obstructions.
- **Not derived.** Any Yukawa coupling of minimal BPR-6D, any mass ratio, and
  CKM/PMNS.
  - The legacy flavor formulas remain unconnected phenomenology.
  - Nothing here supports them.
  - The one structural statement BPR-6D makes about masses is a constraint
    on possible Higgs sectors, not a prediction.
- **Well-posed next problem.** Find a Higgs sector for BPR-6D that meets all
  of the following:
  - it supplies an F-charge −2 internal one-form 10;
  - it is stable at the compactification scale;
  - it keeps three families;
  - it breaks SU(2)_iso at a high scale.

  Or show that none exists in a stated class.

## 8. Independent review

Pending. See the README entry for this date.

## 9. Limitations

- The round sphere, the flux and the field content are supplied.
- The Higgs sector is unspecified. The J=2 channel assumes a field that is
  not in the minimal content.
- The SO(12) route gives even families and a tachyonic Higgs level.
- The charged-vector level formula is checked against the Atiyah–Bott count,
  not by a full fluctuation computation.
- SU(2)_iso breaking is not supplied.
- No masses are predicted.
