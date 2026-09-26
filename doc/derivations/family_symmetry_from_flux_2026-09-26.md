# Families from flux on S²: an SU(2) family symmetry, and no minimal Yukawa

2026-09-26. Status: exact results on a supplied background. The
implementation is `bpr/sphere_family_structure.py`, with tests in
`tests/test_sphere_family_structure.py` and a demo in
`scripts/demo_sphere_family_structure.py`. The methods are explicit
spin-weighted harmonics, eth operators, 6D gamma matrices,
Clebsch–Gordan algebra and a finite-difference Yang–Mills Hessian. The
independent review in section 8 found four major and six minor issues, all
of scope or interpretation; its repairs are applied.

The background is the flux vacuum of
[flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md).
The field content is the minimal completion of
[chiral_parent_completion_2026-09-25.md](chiral_parent_completion_2026-09-25.md).

**The main outcome is negative.**
- The minimal BPR-6D content has no field in 16⊗16 = 10⊕120⊕126, so it has no
  Yukawa coupling at all. This part is trivial.
- The nontrivial result is an all-orders selection rule for any extension
  (section 3). Only a J=2 channel exists. Perturbatively in U(1)_F it needs an
  F-charge −2 internal one-form in the 10 or 126.
- The Yukawa sector is therefore an open problem, not a result.

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
- **Kernel completeness, independently of the harmonics.** ð̄f=0 at spin
  weight −j means f = sin^jθ·h(w), with w = e^{−iφ}tan(θ/2) and h holomorphic
  on ℂ*. Boundedness at both poles leaves exactly w^d with |d| ≤ j. The tests
  check, for k = 3 and 5:
  - that these 2j+1 functions lie in the span of the harmonic zero modes;
  - that |d| = j+1 is annihilated but unbounded.
- The Goldberg-formula harmonics match SymPy's Y_lm at s=0.
- They are orthonormal by quadrature.
- They obey the standard ladder relations ð ₛY = +√((l−s)(l+s+1)) ₛ₊₁Y and
  ð̄ ₛY = −√((l+s)(l−s+1)) ₛ₋₁Y, including at half-integer labels.
- For k = 1…4, the tests confirm:
  - explicit annihilation of the zero modes;
  - J_z weights −j…j, each appearing once;
  - that ð acting on the other component has no kernel, on sampled l.
- The massive levels follow from −ðð̄ ₛY = (l+s)(l−s+1) ₛY. Explicit
  derivatives check them at l = 1…4 for k = 3.

This reproduces the classic statement that flux-induced chiral families
carry isometry quantum numbers:
- Randjbar-Daemi, Salam and Strathdee, Nucl. Phys. B214 (1983) 491;
- Witten, "Fermion quantum numbers in Kaluza–Klein theory", Shelter Island
  II (1983, published 1985).

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
among them cannot use a 6D scalar H without derivatives. The same conclusion
follows independently from spin weight: the integrand of a scalar Yukawa has
total spin weight 1 ≠ 0 (section 3).

Restricted to the chiral subspace, the vector bilinear ψᵀCΓ^Mψ is
antisymmetric (tested for both C±). Fermions anticommute, so the Yukawa is
symmetric in SO(10) × family:
- the SO(10)-symmetric 10 and 126 pair with the family-symmetric J=0 and J=2;
- the antisymmetric 120 pairs with J=1.

## 3. Selection rules: only an internal-vector J=2 Higgs couples

U(1)_F neutrality of 16·16·H forces H to have F-charge −2. A component of H
with tangent spin weight s_h then has effective spin weight s_h + k. The
integrand of the triple overlap must have total spin weight zero:

    2 s₊ + s_h + k = 1 + s_h = 0   ⇒   s_h = −1.

So, without derivatives, only the internal one-form component contributes,
specifically the ∂_z̄ component. Its modes have effective spin weight k−1=2,
so SU(2) spin J ≥ 2. The families have j=1, and 1⊗1 = 0⊕1⊕2. Hence exactly
one channel survives: **J = 2**.

**Robust form (any field, any number of derivatives or flux insertions).**
- The zero-mode pair has total spin weight 1−k = −2. The Higgs mode therefore
  needs spin weight +2, so J ≥ 2.
- 1⊗1 caps J at 2, so J = 2 exactly.
- ð raises the spin weight at fixed J, and the flux is an SU(2) singlet. So a
  field reaches the channel iff it has an l=2 mode of effective spin weight
  ≤ 2. For example, a component H_z̄z̄ reaches it with one ð.
- A scalar of F-charge −2 has effective spin weight 3, so all its modes have
  J ≥ 3. It never couples, at any derivative order.
- Tests confirm by quadrature that spin-weight-2 modes with J=3 and J=4 have
  vanishing overlap.
- The 120 would need J=1, so it is excluded. The 10 and 126 are allowed.

"F-charge −2" is a perturbative statement. U(1)_F is Stückelberg-massive,
and Spin(10) instantons violate it, leaving at most a discrete remnant.
Non-perturbative, F-violating couplings, for example a neutral scalar 10 with
two ð's, are not excluded by this argument; they would be instanton-suppressed.

**Theorem 3.** With the families' j=1 fixed, the triple overlap is

    ∫ ₋₁Y_{1,m₁} ₋₁Y_{1,m₂} ₂Y_{2,m₃} dΩ
        = √(45/4π) (1 1 2; m₁ m₂ m₃)(1 1 2; 1 1 −2).

The two factors are Wigner 3j symbols. All nine (m₁,m₂) pairs are checked
against this closed form by quadrature. The Yukawa tensor is therefore the
Clebsch–Gordan tensor ⟨1 m₁ 1 m₂|2 M⟩, and it is symmetric in the two
families.

**Corollary.**
- (i) *Minimal content.* No field in 16⊗16 exists, since the Spin(10)×U(1)_F
  gauge fields contain no 10, 120 or 126. So there is no Yukawa at all. This
  is trivial.
- (ii) *Extensions, perturbatively in U(1)_F.* A scalar 10 or 126 of F-charge
  −2 never couples, at any order in derivatives or flux insertions. The 120
  never couples. Only an F-charge −2 internal one-form 10 or 126 in J=2 does.

"Renormalizable" is not a useful qualifier here: every 6D Yukawa has
dimension 7 > 6.

Loopholes checked by the review, none of which opens:
- The 16₋(Q=0) has no zero modes (Lichnerowicz). Its KK modes could mix only
  through a 16⊗16 field of F-charge −1, which is absent.
- The Green–Schwarz 2-form has no fermion couplings.
- Axion-dressed operators still need a 16⊗16 field.

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
- Within one sector, the family symmetry constrains the form of the Yukawa but
  no mass ratio.
- Across sectors, SO(10) adds the familiar relations:
  - With 10s only, M_d = M_eᵀ and M_u = M_ν^Dirac at the matching scale.
    This is excluded for the light generations (for example m_s ≠ m_μ).
  - A 126 in the same J=2 channel is needed. It gives the usual
    Georgi–Jarlskog-type freedom.
- The five components of a J=2 multiplet can supply H_u and H_d with
  different orientations. That is the freedom available for CKM mixing. Its
  dynamics (the Higgs potential) is not specified.

## 5. The obvious completion: SO(12) gauge–Higgs, and why it fails as stated

SO(12) ⊃ SO(10)×U(1) contains a field of the right quantum numbers. The
adjoint 66 contains 10_{±2}, whose internal components are the s_h=−1,
charge −2 one-form. This is Manton-type gauge–Higgs unification (Nucl. Phys.
B158 (1979) 141). It fails in three ways.

**Proposition 5a (even and paired families).**
- A 6D Weyl 32 decomposes as 16_{+1} ⊕ 16bar_{−1}. This gives 2k left-handed
  16s (tests: 2, 4, 6, 8 for k = 1…4).
- The 10_{−2} maps 16_{+1} to 16bar_{−1}, and the only coupling is ψ̄Γ^aA_aψ.
  So the Yukawa pairs the k families from 16_{+1} with the k from 16bar_{−1}.
- The 2k×2k mass matrix is [[0,M],[Mᵀ,0]], so every mass is doubly
  degenerate.
- Any projection that keeps three families (n_A+n_B=3) has rank at most
  2·min(n_A,n_B) ≤ 2. At least one family stays massless at tree level.
- A projection that removes one set entirely removes the Yukawa.

The tests check the doubled singular values and the rank bound.

**Proposition 5b (vacuum instability).** Take internal gauge components of
monopole number n = 2k.
- For |n| ≥ 2, their lowest level has m² r² = −n/2, with degeneracy n−1 and
  SU(2) spin n/2−1.
- This follows from rough Laplacian + Ricci + gyromagnetic term, with the
  aligned component carrying effective spin weight |n|/2−1.
- For k=3 this is the J=2 quintet at m² = −3/r². In SO(12) there are 10(n−1)
  complex tachyonic modes, because the 10 comes with multiplicity.

This is the Randjbar-Daemi–Salam–Strathdee instability (Phys. Lett. B124
(1983) 345). It is not merely a Higgs-hierarchy problem:
- π₁(SO(12)) = ℤ₂, so the U(1) flux is not topologically conserved.
- The condensing 10_{−2} breaks SO(10), and the same coupling gives the
  families Dirac masses of order 1/r.
- The flux relaxes, and the chiral spectrum is lost.

*Checks.*
- **Finite-difference Hessian.** An independent finite-difference second
  variation of the SU(2) Yang–Mills energy on S² was run without gauge fixing,
  so gauge directions are exact zeros. For n = 2, 3, 6 it gives exactly n−1
  modes at −n/2, to 2×10⁻².
- For n=1 its lowest physical level is 7/2. The aligned l=1/2 mode is pure
  gauge; an earlier version of this note wrongly listed +1/2.
- **Atiyah–Bott.** For n = 1…10, the number of complex negative modes equals
  n−1. That is the Atiyah–Bott Morse index, 2(n−1) real directions, of the
  U(2) Yang–Mills critical point of W-degree n on S² (SO(3) for odd n);
  Phil. Trans. R. Soc. A308 (1983) 523.
- **Helicity assignment.** The opposite assignment would give no negative
  mode for n ≤ 4, contradicting both checks.
- **Flat-space limit.** −n/2 = −qBr² is the Nielsen–Olesen value (Nucl.
  Phys. B144 (1978) 376). This is a consistency limit, not an independent
  test.

## 6. The family symmetry is gauged

The SU(2) isometry gives massless 4D gauge bosons at tree level, from g_μa
along the Killing vectors. The three families form their triplet, so these
bosons mediate family-changing neutral currents. They must be massive far
above the weak scale.

A J=2 Higgs vev breaks SU(2) only at the weak scale. A separate high-scale
breaking is needed, for example:
- a squashed sphere: the index still gives k zero modes, but the SU(2) is
  broken. A squashing that keeps a U(1) still leaves a gauged family U(1)
  with non-universal charges.
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
  - it supplies an F-charge −2 internal one-form 10 and 126;
  - it is stable at the compactification scale;
  - it keeps three families;
  - it breaks SU(2)_iso at a high scale.

  Or show that none exists in a stated class.

## 8. Independent review

An independent adversarial review re-derived with its own scripts:
- the harmonics, using Wigner-d functions;
- the eth conventions and the spin-weight bookkeeping for charged fields;
- a separate monopole Dirac discretization;
- the chirality lemma, with a different gamma representation;
- the 3j formula;
- all three orientation theorems;
- the charged-vector levels, from an un-gauge-fixed Yang–Mills Hessian;
- the 2k family count.

It found no wrong computed result, but four major issues of scope or
interpretation and six minor ones. All are repaired above:
- **Major:**
  - the SO(12) families are paired by the Yukawa (Proposition 5a);
  - the tachyon is a vacuum instability, not a hierarchy problem
    (Proposition 5b);
  - the Corollary's trivial and nontrivial parts are separated, with the
    all-orders scalar exclusion;
  - the 126 is added, together with the SO(10) relations.
- **Minor:**
  - the robust selection rule and the perturbative scope of F-charge;
  - the n=1 gauge mode;
  - Atiyah–Bott and Nielsen–Olesen wording;
  - tautological tests replaced by independent oracles (the analytic kernel,
    the Yang–Mills Hessian, J=3/4 overlaps);
  - full references;
  - dead code removed.

## 9. Limitations

- The round sphere, the flux and the field content are supplied.
- The Higgs sector is unspecified. The J=2 channel assumes a field that is
  not in the minimal content.
- The SO(12) route gives paired, even families and a vacuum instability.
- The charged-vector levels are checked by the Atiyah–Bott count and a
  finite-difference Hessian, not by an analytic fluctuation computation.
- The selection rules are perturbative in U(1)_F.
- SU(2)_iso breaking is not supplied.
- No masses are predicted.
