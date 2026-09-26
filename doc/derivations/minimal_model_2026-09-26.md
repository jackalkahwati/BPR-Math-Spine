# Phase 1: the minimal complete BPR-6D model and its kill checks

2026-09-26. Status: a specified model with exact structural checks and
one-loop scale estimates. The implementation is `bpr/minimal_model.py`
(structure) and `bpr/model_scales.py` (scales), with tests in
`tests/test_minimal_model.py` and `tests/test_model_scales.py`, and demos in
`scripts/demo_minimal_model.py` and `scripts/demo_model_scales.py`. **No
empirical validation is claimed.**

The independent review (section 7) found a blocker in the first version,
which put an independent Higgs copy on each brane. That version fails two
kill checks (section 2.4) and is kept only for the record. The adopted
version uses a single bulk Higgs field that feeds every brane (section 2.5).

## 0. Goal

Rounds 7, 8 and 10 left BPR-6D as a framework with blanks. The Higgs sector
was "supplied" but never written down, and the obvious choice is ruled out.
Phase 1 turns BPR-6D into **one definite model** and runs kill checks on it.
A failed check ends that version of the model.

## 1. The model (BPR-6D-M, version B)

The fixed part of BPR-6D is unchanged: 6D gravity, Spin(10) × U(1)_F, the
matter 16₊(3) ⊕ 16₋(0), the Green–Schwarz 2-form, and the flux sphere.

Earlier rounds constrain the Higgs sector:
- **Round 10.** Every Higgs that gives Yukawas has F-charge −6, so U(1)_F
  acts as a Peccei–Quinn symmetry. A viable axion needs an F-charged
  singlet linked to the Higgs doublets.
- **Round 8.** A Higgs operator localized at a point couples to the families
  only if its normal-bundle spin weight is s_h = −1 (J_z charge
  c = s_h + 3 = 2), and then only with rank 1. A bulk scalar has no 6D
  Yukawa: two 6D Weyl spinors of the same chirality form no Lorentz scalar.
- **Round 7.** Geometric breaking of Spin(10) is obstructed, so a breaking
  Higgs is needed.

The adopted content:

| field | where | SO(10) | F | role |
|---|---|---|---|---|
| 45_H | bulk | 45 | 0 | breaks Spin(10) → SU(3)×SU(2)_L×SU(2)_R×U(1)_B−L at M_GUT |
| 10_H | bulk | 10 (complex) | −6 | Higgs doublets; couples at the branes through ð̄10_H |
| 126bar_H | bulk | 126bar | −6 | breaks B−L at M_I; Yukawas and ν^c masses through ð̄126bar_H |
| S | brane 1 | 1 | +6 | Peccei–Quinn singlet; DFSZ link S²·10_H·10_H at brane 1 |

There are four codimension-2 branes of small tension. Each carries the
brane-localized operators 16·16·(ð̄10_H)(z_a) and 16·16·(ð̄126bar_H)(z_a),
plus brane mass terms for the two bulk Higgs fields. Here (ð̄Φ)(z_a) is the
spin-lowering derivative of the bulk field Φ, evaluated at the brane.

This is the familiar non-supersymmetric "complex 10 + 126bar + 45 with a
Peccei–Quinn symmetry" SO(10) content (Bajc–Melfo–Senjanović–Vissani,
PRD 73 (2006) 055001; Bertolini–Di Luzio–Malinský, PRD 81 (2010) 035015 and
PRD 85 (2012) 095014; cited, not checked against the texts). Here the
Peccei–Quinn symmetry is not imposed: it is U(1)_F. What BPR adds is the
sphere: where fields can live, and what the flux does to their couplings.

## 2. Structural checks

### 2.1 Couplings

Each term is checked for total F-charge zero, for zero total normal weight if
it is a brane term, and for an SO(10) invariant. The invariants come from a
hand-coded table based on Slansky, not recomputed. Yukawas use the round-8
rule c = 2. All results are tested.

- **Allowed:**
  - 16·16·(ð̄10_H) at a brane;
  - 16·16·(ð̄126bar_H) at a brane, which also gives the ν^c Majorana mass;
  - S²·10_H·10_H at brane 1;
  - the bulk and brane mass terms;
  - 126bar_H†·45_H·126bar_H;
  - the 10–126 doublet mixing 10_H†·126bar_H·45_H·45_H, which realistic fits
    need.
- **Forbidden:**
  - 16·16·10_H at a brane, the undifferentiated value, since c = 3;
  - 16·16·(ð̄10_H)*, since F = +12;
  - the μ-term 10_H·10_H, since F = −12;
  - S·10_H·10_H, since F = −6.

Two remarks:
- In F-neutral brane terms the Wu–Yang parts of J_z cancel, so only the
  normal weights need to cancel.
- The forbidden list holds only perturbatively. The axion factor e^{ib}
  carries F = 12, so terms such as 10_H·10_H·e^{ib} are gauge invariant.
  This is the axion-quality question of round 10.

### 2.2 Breaking to exactly the Standard Model

The 45 is represented on vectors and the 126 on 5-forms. The 5-form
representation is tested to be a Lie-algebra homomorphism, and the vev
direction is the SU(5)-singlet holomorphic 5-form (the ν^cν^c direction).
The unbroken dimensions are:

| vevs | unbroken dimension | group |
|---|---|---|
| 45_H along B−L | 15 | SU(3)×SU(2)×SU(2)×U(1) |
| 45_H along T3R | 19 | SU(4)×SU(2)×U(1) |
| 45_H along both | 13 | |
| 126bar_H alone | 24 | SU(5) |
| **45_H along B−L, with 126bar_H** | **12** | SU(3)×SU(2)×U(1) |
| 45_H along both, with 126bar_H | 12 | SU(3)×SU(2)×U(1) |

The unbroken Cartan direction is exactly Y = (−⅓,−⅓,−⅓,½,½). Flipped
hypercharge and X are broken.

This checks the group theory only. For 45 + 126 the potential has no viable
tree-level minimum. It becomes viable at one loop, and the viable vacuum has
light pseudo-Goldstone states from the 45 (BDM 2010, cited). Those light
states violate the extended survival hypothesis used in section 3.

### 2.3 Brane Yukawa mathematics

Take a single light field whose c = 2 operator acts at points z_a. Brane a
contributes Y ∝ u(z_a)u(z_a)ᵀ, where u(z) is the spin-1 coherent state
(tested against explicit rotations, for complex z as well as real).

**Proposition 1 (spin 2 only).** These matrices span only a
five-dimensional subspace of the six-dimensional space of symmetric Yukawa
matrices. They have no J=0 component, since Sym²(spin 1) = J0 ⊕ J2 (tested).
This holds at leading order: brane-localized fermion kinetic terms and the
tension distortion of the zero modes add J=0 pieces.

**The way out.** In the 4D theory, a U(3) rotation of the three 16s is a
field redefinition. The Spin(10) interactions are U(3) invariant, and only
the heavy family gauge bosons are not. Observables therefore depend on
(Y10, Y126) only up to simultaneous congruence UᵀYU.

**Proposition 2 (at least four branes).** Count the real rank of the map from
N branes (positions, the two couplings, plus U(3)) to pairs (Y10, Y126),
whose target space has 24 real dimensions (tested):

| branes | rank | meaning |
|---|---|---|
| 2 | 16 | codimension 8 |
| 3 | 23 | one real relation |
| 4 | 24 | generic pairs |
| 4, without U(3) | 20 | the J=2 pairs only |

Modulo U(3), the space of pairs is 15-dimensional (24 − 9). Four branes
carry about 21 real parameters after overall rotations, so the fibres are
six-dimensional.

**Construction.** For four branes the realization is exact and constructive.
- *Step A.* Choose U ∈ U(3) that removes the J=0 part of both matrices. These
  are four real conditions on U(3), solved numerically. Step A is robust (the
  review tried about 190 structured and adversarial pairs, and none failed)
  but unproven, and the code raises an error if it fails.
- *Step B.* Any hyperplane of the J=2 space that contains both matrices meets
  the curve {u(z)u(z)ᵀ}, a rational normal quartic, in four points: the roots
  of a quartic in z. Those roots are the brane positions, and the couplings
  follow linearly.

Tests reproduce hierarchical targets to machine precision. For example, Takagi
values (10⁻⁵, 3×10⁻³, 1) are recovered to 10⁻⁸ relative. In that example one
brane term is about 1 and three are about 3×10⁻³, and the 10⁻⁵ value comes
from a cancellation among the small terms. The hierarchy sits in the
couplings, not in the geometry.

**Proposition 3 (the three-brane relation).** With three branes, the one
leftover relation is the Bargmann invariant. The phase of ⟨1|2⟩⟨2|3⟩⟨3|1⟩
for spin-1 coherent states equals the solid angle of the geodesic triangle,
which the three pairwise distances fix (l'Huilier). This matches to 10⁻¹⁴
(tested).

### 2.4 Version A (independent Higgs copies on each brane) fails

The first version put an independent 10_H and 126bar_H on each brane. The
review found two fatal problems.

1. **No common light doublet.** No tree-level term links fields on different
   branes: there is no bulk 10, and the 45 is F-neutral. The doublet mass
   matrix is therefore diagonal in the brane index. One tuning puts the light
   doublet on a single brane, and the tree-level Yukawa is rank 1 (tested).
   Cross-brane mixing first appears at one loop.

   The review's toy model suggests that loop admixtures then give hierarchies
   of order (1, 10⁻³, 10⁻⁴). That is a lead for later work, not computed here.
2. **No consistent intermediate scale.** A rank-3 Majorana matrix needs at
   least three condensing Δ_R, one per brane, and their orientations are
   flat at tree level. At one loop (table in section 3), three light Δ_R give
   either M_I ≲ 1 GeV, or, with an extra bidoublet, M_GUT ≈ 3×10¹⁵ GeV, which
   violates the Super-Kamiokande bound. Every configuration with three Δ_R
   fails (tested).

### 2.5 Version B (one bulk Higgs) repairs both

- **The lowest level.** An F-charge −6 bulk scalar has spin weight +3.
  Its levels are m²r² = l(l+1) − 9 with l ≥ 3, so the lowest level is a
  spin-3 multiplet (seven states) at m²r² = 3. The next level is at 11.
  Tested exactly:
  - ð annihilates the lowest level;
  - its Laplacian eigenvalue is 3;
  - ð̄ maps it to spin weight 2 with coefficient −√6.
- **The brane coupling.** At a brane only m = −3 survives in the value (so
  c = 3: no Yukawa), and only m = −2 survives in ð̄ (so c = 2: rank-1 Yukawa).
  The Yukawa operator therefore carries one derivative, suppressed by
  ε = 1/(M r) ≈ 0.2–0.3 inside the window of section 3. An O(1) top Yukawa
  needs brane couplings of a few.
- **One light combination.** Brane mass terms κ_a|Φ(z_a)|² + λ_a|ð̄Φ(z_a)|²
  split the multiplet by an amount of order κ/r², and one tuning makes one
  combination light. With positive value terms alone, three combinations
  stay degenerate, because they vanish at all four branes. So a unique light
  state needs a negative brane term or derivative terms (tested).
- **It feeds every brane.** Over random brane positions and terms, the light
  combination feeds all four branes:
  - the median sorted weights |c_a|/max are (0.12, 0.29, 0.58, 1);
  - 80% of samples have the smallest weight above 0.05;
  - every sampled Yukawa matrix is rank 3 (tested).

  So Y10 = Σ y_a c_a P_a and Y126 = Σ y′_a c′_a P_a with independent
  couplings, and the four-brane construction of section 2.3 applies.
- **Neutrinos.** A single Δ_R condensate gives
  M_R = Σ d_a c_a^Δ P_a, of rank 3.
- **Consistent with the running.** The other six members of each spin-3
  multiplet sit at the brane-term scale, near 1/r. This matches the one-Δ_R,
  one-bidoublet content used in the running.

### 2.6 Family symmetry

One brane leaves the rotation about its axis unbroken. Two branes that are
not antipodal break the SU(2) family isometry completely (tested).

The family gauge bosons get masses m ≈ 0.28√(Nδ)/r. This needs Nδ ~ 0.05–1,
which strains the "small tension" assumption. It is far above the ~100 TeV
that flavour-changing bounds require.

Five physical brane-position moduli remain after the family gauge bosons eat
three. They couple to fermions through position-dependent Yukawas, so
fifth forces, flavour violation and cosmological moduli are all possible.
Unless the moduli are stabilized this is **potentially fatal**, and it is not
addressed here.

## 3. Scales

**One-loop running.** The chain is Spin(10) → 3221 at M_GUT (by 45_H), then
3221 → SM at M_I (by 126bar_H), under the extended survival hypothesis. The
beta coefficients are computed from field content. They reproduce the
textbook anchors (tested):
- SM: (41/10, −19/6, −7);
- two Higgs doublets: (21/5, −3, −7);
- MSSM: (33/5, 1, −3), with MSSM unification at 2.0×10¹⁶ GeV.

The 3221 coefficients, (−7, −3, −7/3, 11/2), are also checked by hand.
Forward re-running confirms that all four couplings meet.

Version B's light content is one bidoublet and one Δ_R:

| below M_I | M_I (GeV) | M_GUT (GeV) | 1/α_G | τ_p naive (yr) | margin over Super-K |
|---|---|---|---|---|---|
| one doublet | 1.0×10⁹ | 4.5×10¹⁶ | 46.2 | 2.6×10³⁸ | 1.1×10⁴ |
| two doublets | 3.1×10⁹ | 2.3×10¹⁶ | 45.4 | 1.6×10³⁷ | 6.9×10² |

The naive lifetime M_GUT⁴/(α_G² m_p⁵) is conservative; the review's
matrix-element estimate gives longer lifetimes.

**Sensitivity** (all tested; `multiplicity_scan`):
- **Three Δ_R** (version A): M_I ≲ 1 GeV, or a Super-K violation.
- **Two doublets:** M_GUT = 2.3×10¹⁶ GeV whatever the number of Δ_R, because
  b₃ = c₃ and b₂ = c₂L.
- **One extra bidoublet:** the one-doublet M_GUT moves from 4.5×10¹⁶ to
  1.4×10¹⁶ GeV.
- **Cited, not computed:** thresholds and the light 45 pseudo-Goldstones can
  move M_I by orders of magnitude. BDM 2012 find M_B−L up to about
  10¹⁴ GeV for this model.

**Compactification window.** The flux module gives 1/r = 2g₄M_Pl/3 and
1/(rM) = 2π^{1/4}√(g₄/3). Here g₄ is the U(1)_F coupling of the charge-1
parent unit in flux 3; the coupling per unit F-charge is g₄/3. Requiring:
- M_GUT ≤ 1/r, a validity condition for the 4D analysis rather than a
  physical constraint;
- rM ≥ 3, a control convention;

gives:
- one doublet: g₄ ∈ [0.028, 0.047] and 1/r ∈ [4.5, 7.6]×10¹⁶ GeV;
- two doublets: g₄ ∈ [0.014, 0.047] and 1/r ∈ [2.3, 7.6]×10¹⁶ GeV.

Requiring M_GUT ≤ 1/r caps rM at 3.9 or 5.5 (tested). The window sits
exactly where the omitted Kaluza–Klein thresholds and the (M_GUT/M)² ~ 0.1
operators are largest. With the extra-bidoublet sensitivity above, the scale
is **not** robustly pinned; "1/r ≈ (2–8)×10¹⁶ GeV" is a one-loop indication
only.

**Seesaw.** A type-I seesaw with M_R ≈ M_I ≈ 10⁹ GeV needs a Dirac neutrino
Yukawa of about 10⁻³ for m_ν ≈ 0.05 eV. SO(10) ties the Dirac matrix to the
up-quark one, with its top-like entry of about 1; that would need
M_R ~ 6×10¹⁴ GeV. A type-II contribution is not an escape in this chain:
Δ_L sits at M_GUT, so v_L ~ λv²v_R/M_GUT² ~ 10⁻¹¹ eV. The tension may be an
artifact of the survival hypothesis (BDM 2012), so it is recorded as a
tension, not a failure.

## 4. Kill-check table

| # | check | version A (brane copies) | version B (bulk Higgs, adopted) |
|---|---|---|---|
| 1 | Couplings: wanted allowed, dangerous forbidden | pass | **pass** (computed; perturbative only) |
| 2 | Vevs leave exactly the SM, with the right hypercharge | pass | **pass** (computed) |
| 3 | Breaking potential has that minimum | conditional | conditional (one loop, light 45 states; cited) |
| 4 | Yukawas reach generic mass matrices | **fail** (rank 1 at tree level) | **pass** (single light combination feeds four branes; exact construction) |
| 5 | Those mass matrices fit the data | — | conditional (non-SUSY 10+126bar fits, e.g. Boucenna–Ohlsson–Pernow 2019, as found by the review; not redone) |
| 6 | Viable invisible QCD axion | — | pass with S at brane 1 (the link exists only there); quality unresolved; domain-wall number > 1 if S breaks U(1)_F after inflation; λS²H_uH_d needs λ ≲ (v/f_S)² |
| 7 | Family gauge bosons heavy | pass | pass (estimate) |
| 8 | Unification and proton decay | **fail** (three Δ_R) | **pass** at one loop, survival hypothesis |
| 9 | Compactification control with M_GUT ≤ 1/r | — | pass narrowly; not robust |
| 10 | Seesaw neutrino masses | — | **tension** (possibly an artifact) |
| 11 | Brane positions stabilized | open | **open, potentially fatal** (five moduli coupled to fermions) |
| 12 | Multi-brane background exists | conditional | conditional (constant-curvature spheres with cone points exist for small deficits: Troyanov 1991, Luo–Tian 1992) |
| 13 | Higgs mass, M_I hierarchy and Λ | tuned | tuned: one Higgs-mass tuning per bulk Higgs, plus the M_I/M_GUT hierarchy, as in 4D minimal SO(10) |

**Verdict:**
- **Version A** fails checks 4 and 8.
- **Version B** passes every check that was computed: 1, 2, 4 and 8, plus
  the pass-level estimates 7 and 9.
- Checks 3, 5, 6 and 12 rest on cited results.
- There is one tension (10) and one potentially fatal open problem, the
  brane moduli (11).

**BPR-6D is now one specific model with these caveats, not a framework with
blanks.**

## 5. What Phase 1 did and did not achieve

**Achieved:**
- A definite field content, with every coupling checked, and an exact
  breaking pattern.
- A clean failure of the brane-copy version.
- Three structural results about Yukawas from points on the flux sphere:
  - they are spin 2 only;
  - generic Yukawas need at least four branes, built from the roots of a
    quartic;
  - three branes obey a Bargmann relation.
- The bulk-Higgs mechanism:
  - the lowest level is a spin-3 multiplet;
  - an ð̄ coupling at each brane;
  - a single light combination that feeds every brane.

**Not achieved:**
- **Nothing about fermion masses is predicted.** Four branes carry about 21
  parameters for a 15-dimensional space of Yukawa pairs.
- The Higgs content is chosen for minimality, not derived.
- Brane positions are free and potentially dangerous moduli.
- The intermediate scale and the seesaw are not settled beyond one loop.

**Next (Phase 2).**
- Stabilize or remove the brane moduli; this is the most urgent.
- Redo a non-SUSY 10+126bar fit inside the four-brane construction.
- Add two-loop running, the 45 thresholds and Kaluza–Klein thresholds.
- Compute the loop-induced hierarchy suggested by version A's toy model.

## 6. Limitations

- The Higgs content is chosen, not derived.
- The breaking potential and the fermion fits are cited, not computed.
- The running is one loop, under the survival hypothesis, with no thresholds
  and no Kaluza–Klein thresholds near 1/r.
- The proton lifetime is a naive, order-of-magnitude estimate.
- The family-gauge-boson masses are estimates.
- Brane positions are moduli; the tension distortion of the zero modes and
  brane kinetic terms are not included.
- The SO(10) invariants are hand-coded.
- The forbidden terms are forbidden perturbatively only.
- The control criterion rM ≥ 3 is a convention.
- Citations were not checked against the texts; arXiv is blocked here.

## 7. Independent review

An independent review, with its own scripts, verified:
- the 5-form group theory, including a Hodge-star check that Ω lies in one
  irreducible 126;
- every stabilizer dimension, including the pure B−L case;
- round 7's conventions;
- Propositions 1 and 3;
- the coherent-state conventions for complex z;
- the Jacobian ranks, with analytic derivatives;
- Step B, and Step A on about 190 structured and adversarial pairs;
- every one-loop number, the proton lifetimes, the window and the seesaw
  arithmetic.

Its findings, all repaired:
- **Blocker.** Independent brane Higgs copies do not mix at tree level, so
  the four-brane construction did not apply. Version A is now recorded as
  failing, and version B, a single bulk Higgs, is adopted and checked.
- **Major:**
  - A rank-3 Majorana matrix needs three Δ_R; the multiplicity scan shows
    version A fails, and version B needs only one.
  - The survival hypothesis conflicts with the cited vacuum; BDM 2012 is
    now cited, and type-II is dropped as an escape.
  - Citations are corrected: Babu–Bajc–Saad (a 10+120+126 model) is
    removed, and Bajc–Melfo–Senjanović–Vissani 2006 and BDM 2012 are added.
  - The verdict is reworded, and the brane moduli are marked potentially
    fatal.
- **Minor:**
  - The reason for where fields live is corrected.
  - S is placed on brane 1.
  - The weight identity and the perturbative nature of the forbidden list
    are stated, and missing invariants are added.
  - The pure B−L case is tested.
  - The parameter count now says 15 effective and "at least four".
  - The cancellation in the example is noted.
  - Loud failure modes are added.
  - The margins (6.9×10² and 1.1×10⁴) are corrected.
  - The g₄ normalization and the validity-condition wording are fixed, with
    the extra-bidoublet sensitivity.
  - The family-gauge-boson estimate is caveated, and Proposition 1 is marked
    leading order.
  - The axion domain-wall and coupling-size caveats are added.
  - Complex-z rotation, J=2 and failure-path tests are added.
