# A minimal local-anomaly completion of the six-dimensional chiral parent

2026-09-25. Status: new conditional algebra with an exact implementation
(`bpr/chiral_parent_completion.py`), independent tests
(`tests/test_chiral_parent_completion.py`) and a demo
(`scripts/demo_chiral_parent_completion.py`). An independent adversarial review
is recorded in section 8. The added fields are new model inputs, not
consequences of BPR. This is a consistency repair of a supplied target, not a
derivation of Standard Model matter.

## 0. Problem

The [chiral-parent note](chiral_parent_anomaly_2026-09-12.md) considers one
complex six-dimensional Weyl field in the 16 of Spin(10), with commuting U(1)_X
charge Q=1 and chirality s=+1. On M4×S² with U(1)_X flux m it gives |m| chiral
16's in four dimensions; q=3 gives three families, but q=3 is chosen. Its local
anomaly

    I8 = (3S2²-2S4)/24 + X²S2 + (2/3)X⁴ - p1S2/12 - p1X²/3 + (7p1²-4p2)/360

contains two irreducible terms, S4 and p2. Neither can be cancelled by a
Green–Schwarz 2-form. The [foundation decision](foundation_decision_2026-09-13.md)
therefore marked the lone parent excluded and required any future matter sector
to carry an anomaly-consistent completion. This note asks a narrow question:
what is the smallest set of additional six-dimensional Weyl fields such that
- the local anomaly can be cancelled by one non-chiral 2-form, and
- the four-dimensional Spin(10)-charged chiral content is unchanged?

Conventions are those of the parent module: F/(2πi) curvature,
S2=Σx_i², S4=Σx_i⁴ in Spin(10) Cartan variables, X=c_1(U(1)_X), Â=1-p1/24+(7p1²-4p2)/5760,
and I8(s,R,Q)=s[Â ch_R(F) exp(QX)]_8.

## 1. Representation data

From the weights, with exact arithmetic:

| R | dim | tr_R F² | tr_R F⁴ |
|---|---|---|---|
| 1 | 1 | 0 | 0 |
| 10 | 10 | 2 S2 | 2 S4 |
| 16 | 16 | 4 S2 | 3 S2² - 2 S4 |

The 16-bar has the same even traces. Hence a field (s,R,Q) contributes

    s[ tr_R F⁴/24 + Q² X² tr_R F²/4 + dim Q⁴X⁴/24 - p1 tr_R F²/48
       - dim Q² p1 X²/48 + dim (7p1²-4p2)/5760 ].

## 2. Class and conditions

Class 𝒞(Q_max): the parent 16 with s=+1 and Q=1 is fixed. Add:
- any number of 10's with Q=0 and either chirality;
- any number of Spin(10) singlets with |Q|<=Q_max and either chirality.

No additional 16 or 16-bar, no U(1)_X-charged 10, and no chiral tensors. The
first two exclusions keep the four-dimensional Spin(10)-charged chiral content
equal to that of the parent.

Local cancellation by one non-chiral 2-form B, with dH=X_4 and a coupling
∫B∧X̃_4, requires three things:
- (C1) the S4 coefficient vanishes;
- (C2) the p2 coefficient vanishes;
- (C3) the remainder factorizes over the rationals as X_4∧X̃_4.

**Lemma 1.** (C1) ⟺ #10₊-#10₋=1. (C2) ⟺ #1₋-#1₊=26. When (C2) holds the p1²
coefficient also vanishes.

*Proof.* The S4 coefficients are -2/24 per 16 and +2/24 per 10, weighted by
chirality. The p2 and p1² coefficients are both proportional to Σ s·dim,
which is 16+10(#10₊-#10₋)+(#1₊-#1₋). ∎

Hence at least one 10₊ and at least 26 negative-chirality singlets are needed.
**The minimal number of added Weyl components is 36**: exactly one 10₊ and
twenty-six 1₋.

**Lemma 2 (factorization criterion).** Suppose (C1) and (C2) hold. Write
u=S2, v=X², w=p1. Then I8=αu²+βuv+γv²+δuw+εvw with rational coefficients.
- If (δ,ε)≠(0,0), I8 factorizes over Q iff q(ε,-δ)=0, where q=αu²+βuv+γv².
  In that case I8=ℓ·(m+w), with ℓ=δu+εv and q=ℓm.
- If δ=ε=0, I8 factorizes iff β²-4αγ is a rational square.

*Proof.* In a product L1L2 of linear forms the w² coefficient c1c2 must vanish.
Take c2=0. Then c1(a2u+b2v)=ℓ, so ℓ divides q. A linear form divides a binary
quadratic iff q vanishes on its kernel, which is spanned by (ε,-δ).
Conversely, divisibility gives the explicit factorization. ∎

For the minimal structure, δ=-(4+2)/48=-1/8 for any charges. Let the
twenty-six singlets have charge magnitudes with P2=Σ|Q|² and P4=Σ|Q|⁴. Then
α=1/8, β=1, γ=2/3-P4/24 and ε=-1/3+P2/48. The criterion becomes

    P4 = 16 - (16-P2)(32+P2)/12.                                          (1)

## 3. Result

**Theorem 3.**
(i) With Q_max=1, exactly one minimal completion satisfies (C1)–(C3):

    16₊(Q=1) ⊕ 10₊(Q=0) ⊕ 16×1₋(|Q|=1) ⊕ 10×1₋(Q=0),
    I8 = (1/8) S2 (S2 + 8X² - p1).                                        (2)

(ii) Uniqueness fails as soon as larger charges are allowed. With Q_max=2 the
exhaustive search finds three minimal solutions:
- (2) itself;
- twelve |Q|=1 and four |Q|=2 singlets, with I8=(1/8)(S2-2X²)(S2+10X²-p1);
- ten |Q|=2 and sixteen neutral singlets, with P2=40 and P4=160.

With Q_max=3 it finds eight. A hand scan of (1) had missed the |Q|=2-only
solution; the exact enumeration is authoritative.
(iii) At flux m, each field (s,R,Q) gives |Qm| four-dimensional left-handed
zero modes in (R,Q) if s·sgn(Qm)=+1, and in (R̄,-Q) otherwise. Neutral fields
have no zero modes, because a round S² without flux has none. For (2) at m=3
the massless content is

    3 × (16, X=+1)   and   48 × (1, X=-1).

(iv) Consider every minimal completion, for any Q_max and any flux m≠0. Its
four-dimensional U(1)_X³ and mixed-gravitational anomalies are m(16-P4) and
m(16-P2). Both vanish iff P2=P4=16, i.e. Σ Q²(Q²-1)=0 with exactly sixteen
|Q|=1 singlets. By (1) this already implies (C3). **So (2) is the unique
minimal completion whose four-dimensional abelian anomalies vanish.** Every
other minimal completion needs a four-dimensional Green–Schwarz term for
U(1)_X³ as well.

*Proof.* Enumerate multisets satisfying (1): an exhaustive exact search in the
implementation, cross-checked by SymPy factorization. (iii) is the index
theorem on S² with flux; the pushforward of (2) is checked against it below.
For (iv), take m>0; m<0 reverses every sign. A negative-chirality singlet of
magnitude |Q| gives m|Q| left-handed modes of charge -|Q|. The parent gives
16m modes of charge +1. ∎

## 4. Four-dimensional ledger for completion (2), m=3

Integrating (2) over S² gives I6=2m·S2·X4=6·S2·X4. This equals the
zero-mode anomaly Σ[Â ch_R exp(QX4)]_6 exactly:
- the parent's 8X4³ and -2p1X4 terms are cancelled by the charged singlets;
- only the mixed Spin(10)²–U(1)_X anomaly remains, and it is already factorized.

It is cancelled by the four-dimensional axion descending from B. The standard
flux-compactification consequence is that U(1)_X acquires a Stückelberg mass;
that dynamics is not derived here.

Further checks on the 3×16 content:
- Spin(10)³: vanishes; tr_R F³=0 for all these representations.
- π4(Spin(10))=0, so there is no Witten anomaly for Spin(10) itself.
- Under SU(2)_L ⊂ Spin(10), each 16 has four doublets, so 3×16 has twelve;
  the Witten parity is even.
- The Spin(10) center Z4 acts as i on the 16, so each 16 contributes 16 to the
  Z16 (Spin-Z4) count, and 3×16 gives 48≡0 mod 16. The implementation also
  reproduces the known control: the Standard Model with X=5(B-L)-4Y and 15
  Weyl fields per generation needs n_ν≡n_gen mod 16 right-handed neutrinos
  (García-Etxebarria–Montero, JHEP 2019). The 48 singlets carry no Spin(10)
  center charge. A Spin-Z4 structure for the full spectrum would need an extra
  odd Z4 assignment, so no Z16 claim is made for the full spectrum.

## 5. What this changes and what it does not

It turns the parent's local obstruction from "excluded" into "completable at
the level of local anomaly polynomials". The price is explicit:
- one 10, 26 singlets and a 2-form, none derived from BPR;
- 48 massless four-dimensional sterile singlets. As they stand, these conflict
  with light-species bounds (for example N_eff) unless masses or decoupling are
  supplied;
- a Stückelberg-massive U(1)_X.

Not established:
- global anomalies in six dimensions beyond π6(Spin(10))=0, Dirac quantization
  of the GS couplings, and gravitational couplings of B in full
  supergravity-free consistency;
- flux and radius stabilization; q=3 is still an input, so n_gen=3 is still an
  input;
- Yukawa couplings, symmetry breaking to the Standard Model, and any
  realization of this parent on the bosonic substrate. Nielsen–Ninomiya-type
  doubling constraints on lattice realizations remain open.

## 6. Relation to other sectors

This result addresses only obstruction 4 of the foundation decision, at the
level of local anomalies. It does not interact with the
[condensate regime](cubic_condensate_regime_2026-09-25.md), which is purely
bosonic. The [unification map](unification_map_2026-09-25.md) records how the
two partial results sit in the chain.

## 7. Exact controls

The implementation checks the following:
- representation traces from weights, compared against `bpr/chiral_parent_anomaly.py`;
- per-field polynomials against the parent module;
- the exhaustive minimal-class search for Q_max=1,2,3;
- SymPy factorization of every solution;
- pushforward versus zero-mode I6 for every solution;
- the 4D ledger;
- the Standard Model Z16 control.

All arithmetic is exact (Fraction / SymPy rationals).

## 8. Review record

See the end of this file once the independent review is complete.
