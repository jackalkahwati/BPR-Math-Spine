# A minimal local-anomaly completion of the six-dimensional chiral parent

2026-09-25. Status: new conditional algebra with an exact implementation
(`bpr/chiral_parent_completion.py`), independent tests
(`tests/test_chiral_parent_completion.py`) and a demo
(`scripts/demo_chiral_parent_completion.py`). An independent adversarial review
is recorded in section 8; it corrected the first version's minimality claim.
The added fields are new model inputs, not consequences of BPR. This is a
consistency repair of a supplied target, not a derivation of Standard Model
matter.

## 0. Problem

The [chiral-parent note](chiral_parent_anomaly_2026-09-12.md) considers one
complex six-dimensional Weyl field in the 16 of Spin(10), with commuting U(1)_F
charge Q=1 and chirality s=+1. On M4×S² with U(1)_F flux m it gives |m| chiral
16's in four dimensions; m=3 gives three families, but m=3 is chosen. Here the
parent's commuting U(1) is called U(1)_F, and its curvature symbol in the
formulas is X. This avoids confusion with the Standard Model combination
X_SM=5(B-L)-4Y used in section 4. Its local anomaly is

    I8 = (3S2²-2S4)/24 + X²S2 + (2/3)X⁴ - p1S2/12 - p1X²/3 + (7p1²-4p2)/360.

It contains two irreducible terms, S4 and p2. Neither can be cancelled by a
Green–Schwarz 2-form. The [foundation decision](foundation_decision_2026-09-13.md)
therefore marked the lone parent excluded, and required any future matter sector
to carry an anomaly-consistent completion.

The question here is: what are the smallest sets of additional
six-dimensional Weyl fields that
- allow local anomaly cancellation by a single non-chiral 2-form, and
- leave the four-dimensional Spin(10)-charged massless content unchanged?

Conventions are those of the parent module: F/(2πi) curvature,
S2=Σx_i², S4=Σx_i⁴ in Spin(10) Cartan variables, X=c_1(U(1)_F),
Â=1-p1/24+(7p1²-4p2)/5760, and I8(s,R,Q)=s[Â ch_R(F) exp(QX)]_8.

## 1. Representation data

From the weights, with exact arithmetic:

| R | dim | tr_R F² | tr_R F⁴ |
|---|---|---|---|
| 1 | 1 | 0 | 0 |
| 10 | 10 | 2 S2 | 2 S4 |
| 16 | 16 | 4 S2 | 3 S2² - 2 S4 |

The 16-bar has the same even traces. In six dimensions conjugation preserves
chirality, so (s,R,Q) ≅ (s,R̄,-Q). The Weyl spinor is pseudoreal and the 1 and
10 are real, so no symplectic-Majorana halving is possible and each field
counts fully. A field (s,R,Q) contributes

    s[ tr_R F⁴/24 + Q² X² tr_R F²/4 + dim Q⁴X⁴/24 - p1 tr_R F²/48
       - dim Q² p1 X²/48 + dim (7p1²-4p2)/5760 ].

## 2. Class and conditions

The parent 16₊ with Q=1 is fixed. Additions are drawn from:
- Spin(10) spinors (16) with Q=0 and either chirality;
- 10's with Q=0 and either chirality;
- Spin(10) singlets with |Q|<=Q_max and either chirality.

Q=0 is required for Spin(10)-charged additions. A charged 16 or 10 would add
massless four-dimensional Spin(10)-charged zero modes; neutral ones have none,
because a round S² without flux has no Dirac zero modes. Chiral tensors are
excluded; each would add ±(16p1²-112p2)/5760 and change the p2 condition.
Size is measured in added Weyl components, i.e. dimension-weighted field count.
Call this class 𝒞. The first version of this note also excluded neutral
spinors; that narrower class is 𝒞₀.

Local cancellation by one non-chiral 2-form B, with dH=X_4 and a coupling
∫B∧X̃_4, requires three things:
- (C1) the S4 coefficient vanishes;
- (C2) the p2 coefficient vanishes;
- (C3) the remainder factorizes over the rationals as X_4∧X̃_4.

**Lemma 1.** Let n16_±, n10_± and n1_± count added fields by chirality.
- (C1) ⟺ n10₊-n10₋ = 1+n16₊-n16₋.
- (C2) ⟺ 16(1+n16₊-n16₋)+10(n10₊-n10₋)+n1₊-n1₋ = 0.

When (C2) holds, the p1² coefficient also vanishes.

*Proof.* The S4 coefficient is (2/24)(n10₊-n10₋) from the 10's and
-(2/24)(1+n16₊-n16₋) from the spinors. The p2 and p1² coefficients are both
proportional to Σ s·dim. ∎

**Lemma 2 (factorization criterion).** Suppose (C1) and (C2) hold. Write
u=S2, v=X², w=p1. Then I8=αu²+βuv+γv²+δuw+εvw with rational coefficients.
- If (δ,ε)≠(0,0), I8 factorizes over Q iff q(ε,-δ)=0, where q=αu²+βuv+γv².
  In that case I8=ℓ·(m+w), with ℓ=δu+εv and q=ℓm.
- If δ=ε=0, I8 factorizes iff β²-4αγ is a rational square.

*Proof.* In a product L1L2 of linear forms the w² coefficient c1c2 must vanish.
Take c2=0. Then c1(a2u+b2v)=ℓ, so ℓ divides q. A linear form divides a binary
quadratic iff q vanishes on its kernel, which is spanned by (ε,-δ).
Conversely, divisibility gives the explicit factorization. ∎

One non-chiral 2-form has pairing of signature (1,1). It therefore cancels any
product of rational linear forms, including a perfect square. With two
non-chiral 2-forms every content satisfying (C1)–(C2) can be cancelled.
Because the w² coefficient vanishes, w is isotropic and the remainder is
always a sum of two rational products. So (C3) is exactly the cost of
insisting on ONE 2-form.

## 3. Results

**Theorem 3 (minimal completion in 𝒞).**
(i) Fewer than 16 added components cannot satisfy (C1)–(C2). Without a spinor,
(C1) needs a 10₊, and then (C2) needs at least 26 singlets of negative
chirality, which is 36 components.
(ii) At exactly 16 added components the unique solution is one neutral spinor
of opposite chirality:

    16₊(Q=1) ⊕ 16₋(Q=0),      I8 = (1/3) X² (3S2 + 2X² - p1).             (2)

(iii) At flux m the four-dimensional massless content of (2) is exactly
|m|×(16, Q=+sgn m). For m=3 this is three chiral 16's and nothing else: no
sterile singlets and no extra Spin(10) matter. The neutral 16₋ contributes
only massive Kaluza–Klein modes.

*Proof.* Lemma 1 with n16=n10=0 forces n10₊>=1 and n1₋-n1₊>=26, which proves
(i). At 16 components the only way to satisfy (C1) is n16₋=1, and then (C2)
holds automatically. The parent's pure Spin(10) and Spin(10)-gravity terms are
cancelled exactly, leaving the displayed U(1)_F terms. Here δ=0 and ε=-1/3, so
Lemma 2's criterion holds (the δ=0 branch of the implementation). ∎

This is the familiar structure of an anomalous U(1) cancelled by a
Green–Schwarz term: the Spin(10) sector is vectorlike in six dimensions, and
only the U(1)_F assignment is chiral. Pushing (2) forward over S² gives
I6=m(2S2X4+(8/3)X4³-(2/3)p1X4). For m=3 this is 6S2X4+8X4³-2p1X4, which is
exactly the anomaly of 3×(16,+1). Every term carries a factor X4, so the
four-dimensional axion descending from B cancels the Spin(10)²–U(1)_F, U(1)_F³
and mixed-gravitational anomalies together. U(1)_F becomes Stückelberg-massive;
that is the standard flux-compactification consequence, not derived here.

**Theorem 4 (the narrower class 𝒞₀, no added spinors).** Without spinors the
minimum is 36 components, one 10₊ and twenty-six 1₋. Let the singlets have
charge magnitudes with P2=Σ|Q|² and P4=Σ|Q|⁴. Condition (C3) becomes

    P4 = 16 - (16-P2)(32+P2)/12.                                          (1)

Exhaustive exact enumeration gives 1, 3, 8 and 26 solutions for Q_max=1, 2, 3
and 4, all of the form (1/8)(S2-kX²)(S2+(k+8)X²-p1) with k=(P2-16)/6. For
Q_max=1 the unique solution is sixteen |Q|=1 singlets plus ten neutral ones,
with I8=(1/8)S2(S2+8X²-p1). At flux m its massless content is
3×(16,+1) ⊕ 48×(1,-1).

Among minimal 𝒞₀ solutions it is the unique one whose massless U(1)_F³ and
mixed-gravitational anomalies, m(16-P4) and m(16-P2), vanish. That is a
selection of a massless sector free of abelian anomalies, not a consistency
requirement: the descended axion cancels those anomalies anyway. This 𝒞₀
solution was the first version's headline. It is dominated by (2), which has
fewer fields and no massless singlets.

**Near-minimal alternatives in 𝒞, and why they are easy.** Once the added
neutral 16₋ makes the Spin(10) content vectorlike, α=δ=0. Every remaining term
then carries X², so I8=X²·(βS2+γX²+εp1) factorizes for ANY further singlets
satisfying (C2). The budget scan confirms this:
- no solution at 17 components (parity);
- all 25 chirality-balanced singlet pairs with |Q|<=4 work at 18 components.

For example, 16₊(1) ⊕ 16₋(0) ⊕ 1₊(0) ⊕ 1₋(Q=4) gives I8=X²(S2-10X²), with
massless content 3×(16,+1) ⊕ 12×(1,-4). These alternatives differ from (2) in
extra massless singlets and in the quantization data of the 2-form couplings
(section 5), not in local consistency. Minimality selects (2) uniquely.

## 4. Four-dimensional ledger for completion (2), m=3

The massless content 3×16 has these properties:
- Spin(10)³ vanishes, since tr_R F³=0.
- π4(Spin(10))=0, so there is no Witten anomaly for Spin(10) itself.
- Under SU(2)_L ⊂ Spin(10) each 16 has four doublets, so 3×16 has twelve and
  the Witten parity is even.
- For Z16: the Spin(10) center Z4 acts as i on the 16, and X_SM=5(B-L)-4Y is
  1 mod 4 on every component. Each 16 therefore contributes 16≡0 mod 16. This
  check is automatic for any number of 16's and says nothing about n_gen.
  Every fermion of completion (2), massless or massive, is a Spin(10) spinor
  with odd center charge, so no even-charge fermion obstructs the count.
  The implementation reproduces the known control: the Standard Model with 15
  Weyl fields per generation needs n_ν≡n_gen mod 16 right-handed neutrinos
  (García-Etxebarria–Montero, JHEP 2019).
- The mixed Spin(10)²–U(1)_F, U(1)_F³ and gravitational–U(1)_F anomalies are
  nonzero. They are all proportional to X4 and are cancelled by the descended
  axion.

The ledger key `spin10_squared_u1_trace_sum` is Σ mult·Q·(tr_R F²/S2), which
is 12 here. The corresponding I6 coefficient of S2X4 is half of it, 6.

For the 𝒞₀ solution, the 48 massless singlets have odd U(1)_F charge. U(1)_F
mod 4 is thus an odd Z4 on the massless sector, and its count is 48-48≡0 mod 16.
The obstruction to a full Spin-Z4 structure there comes from the massive modes
of the 10 and the neutral singlets, which have even center charge. That
completion also Higgses U(1)_F.

## 5. What this changes and what it does not

It turns the parent's local obstruction from "excluded" into "completable at
the level of local anomaly polynomials", at modest cost:
- one opposite-chirality neutral 16 and one 2-form, neither derived from BPR;
- a Stückelberg-massive U(1)_F;
- at least one massless four-dimensional axion or 2-form mode (whichever of
  ∫_{S²}B or the dual of B_{μν} is not eaten), coupled to S2 and the
  U(1)_F/gravity terms;
- the S² radius and other moduli.

Not established:
- **Global anomalies.** The Ω7 spin bordism of B(Spin(10)×U(1)), together with
  the 2-form sector's quantization data, is not computed. π6(Spin(10))=0 is
  only the older Witten-type criterion.
- **Dirac quantization** of the Green–Schwarz couplings. It is a live
  question, and it is what distinguishes the near-minimal completions. Under a
  naive integrality test, (2) needs b_F·b_F=4/3 and fails, while the 18-component
  example above passes. The review also sketched an odd-lattice (I₁,₁)
  embedding of the 𝒞₀ solution. None of this is settled here.

  **Update 2026-09-26.**
  [green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md)
  settles this at the characteristic-class level:
  - (2) is quantizable, for any 2-form lattice, iff the parent's charge is a
    multiple of 3 in units of the smallest U(1)_F charge;
  - with parent charge 3, one non-chiral 2-form (lattice U) suffices;
  - the odd lattice I₁,₁ is obstructed.

  The failure recorded above is the parent-charge-1 convention. With the
  parent at charge 3, "flux 3" is one flux quantum, and n_gen is a multiple of
  three. Torsion and Ω₇ remain open.
- **Flux and radius stabilization.** m=3, and hence n_gen=3, is still an input.
  See the update above for what quantization adds: n_gen ∈ 3ℤ.
- Yukawa couplings, breaking of Spin(10) to the Standard Model, and any
  realization of this parent on the bosonic substrate. Nielsen–Ninomiya-type
  doubling constraints on lattice realizations remain open.

## 6. Relation to other sectors

This result addresses only obstruction 4 of the foundation decision, at the
level of local anomalies. It does not interact with the
[condensate regime](cubic_condensate_regime_2026-09-25.md), which is purely
bosonic. The [unification map](unification_map_2026-09-25.md) records how the
two partial results sit in the chain.

## 7. Exact controls

The implementation and tests check, with exact Fraction/SymPy arithmetic:
- representation traces from weights, compared against
  `bpr/chiral_parent_anomaly.py`;
- per-field polynomials against a characteristic-class oracle;
- the budgets 0–18 in 𝒞, including uniqueness at 16;
- the 𝒞₀ search at budget 36 for Q_max=1,2,3, against a SymPy factorization
  oracle for Q_max<=2 and by factorization of all eight solutions for Q_max=3;
- pushforward versus zero-mode I6;
- all branches of the factorization criterion;
- the two-2-form decomposition example;
- the 4D ledger and the Standard Model Z16 control.

## 8. Review record

An independent adversarial review (2026-09-25) confirmed:
- the algebra: traces, per-field I8, Lemma 1, criterion (1), the
  factorizations, 1/3/8/26 solutions in 𝒞₀, and factorization_status on 4000
  random coefficient sets;
- the pushforward on 150 random field lists;
- the zero-mode rule, the Witten and Z16 statements, and the absence of
  halving.

It found one blocker. The first version's claim that 36 components are
minimal rested on excluding added spinors "to keep the four-dimensional
content". That reason is false for neutral spinors, which have no zero modes.
A single neutral 16₋ gives a 16-component completion with no massless
singlets. The note now states Theorem 3 in the corrected class 𝒞 and keeps
the 𝒞₀ classification as Theorem 4.

Other repairs applied:
- stating that one 2-form, not consistency, forces (C3), and that two 2-forms
  suffice in general;
- recasting four-dimensional abelian anomaly freedom as a preference;
- general wording of Lemma 1;
- the U(1)_F / X_SM naming and the reasons given in the Z16 paragraph;
- Ω7 bordism rather than π6 for global anomalies;
- documenting the ledger normalization;
- test coverage of all factorization branches and the Q_max=3 solutions;
- the massless axion and moduli in the price list.

This record is AI review of scoped algebra, not certification or empirical
validation.
