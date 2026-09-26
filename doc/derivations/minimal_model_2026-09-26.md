# Phase 1: the minimal complete BPR-6D model and its kill checks

2026-09-26. Status: a specified model with exact structural checks and
one-loop scale estimates. The implementation is `bpr/minimal_model.py`
(structure) and `bpr/model_scales.py` (scales), with tests in
`tests/test_minimal_model.py` and `tests/test_model_scales.py`, and demos in
`scripts/demo_minimal_model.py` and `scripts/demo_model_scales.py`. **No
empirical validation is claimed.** The independent review is recorded in
section 7.

## 0. Goal

Rounds 7, 8 and 10 left BPR-6D as a framework with blanks. The Higgs sector
was "supplied" but never written down, and the last review showed that the
obvious choice is ruled out. Phase 1 turns BPR-6D into **one definite model**
and runs kill checks on it. A failed check would end the model.

## 1. The model (BPR-6D-M)

The fixed part of BPR-6D is unchanged: 6D gravity, Spin(10) × U(1)_F, the
matter 16₊(3) ⊕ 16₋(0), the Green–Schwarz 2-form, and the flux sphere.

Earlier rounds constrain the Higgs sector:
- **Round 10.** Every Higgs that gives Yukawas has F-charge −6, so U(1)_F
  acts as a Peccei–Quinn symmetry. A viable axion needs an F-charged
  singlet linked to the Higgs doublets.
- **Round 8.** A brane Higgs couples to the families only if its
  normal-bundle spin weight is s_h = −1 (J_z charge c = s_h + 3 = 2), and
  then only with rank 1. A brane scalar never couples.
- **Round 7.** Geometric breaking of Spin(10) is obstructed, so a breaking
  Higgs is needed.
- **Physics.** An F-charged 6D scalar has no zero mode in the flux
  background, so F-charged Higgs fields must live on branes.

The smallest content meeting all of these:

| field | where | SO(10) | F | normal weight | copies | role |
|---|---|---|---|---|---|---|
| 45_H | bulk | 45 | 0 | 0 | 1 | breaks Spin(10) → SU(3)×SU(2)_L×SU(2)_R×U(1)_B−L at M_GUT |
| 10_H | brane | 10 (complex) | −6 | −1 | 4 | Higgs doublets and Yukawas |
| 126bar_H | brane | 126bar | −6 | −1 | 4 | breaks B−L at M_I; Yukawas; ν^c masses |
| S | brane | 1 | +6 | +1 | 1 | Peccei–Quinn singlet (DFSZ link S²·10_H·10_H) |

There are four codimension-2 branes of small tension. Each carries one 10_H
and one 126bar_H. Section 2.3 shows why four.

This is the familiar non-supersymmetric "10 + 126bar + 45 with a
Peccei–Quinn symmetry" SO(10) content (for example Babu–Mohapatra 1993,
Bertolini–Di Luzio–Malinský 2010, Babu–Bajc–Saad 2017; cited from memory).
Here the Peccei–Quinn symmetry is not imposed: it is U(1)_F. The novelty is
where the fields must sit and what the sphere does to their couplings.

## 2. Structural checks (Phase 1a)

### 2.1 Couplings

Each term is checked for three things: total F-charge zero, total
normal-bundle weight zero on the brane, and an SO(10) invariant, taken from
Slansky's tables. Yukawas use the round-8 rule c = 2. All results are tested.

- **Allowed:** 16·16·10_H; 16·16·126bar_H (including the ν^c Majorana mass);
  S²·10_H·10_H; 10_H†10_H; 126bar_H†·45_H·126bar_H.
- **Forbidden:**
  - 16·16·10_H*, since F = +12;
  - the μ-term 10_H·10_H, since F = −12;
  - S·10_H·10_H, since F = −6 (that would need F_S = +12);
  - any Yukawa with a brane scalar (c = 3).

### 2.2 Breaking to exactly the Standard Model

The 45 is represented on vectors and the 126 on 5-forms. The 5-form
representation is tested to be a Lie-algebra homomorphism. The unbroken
dimensions are:
- 45_H along B−L: 15, i.e. SU(3)×SU(2)×SU(2)×U(1);
- 45_H along T3R: 19, i.e. SU(4)×SU(2)×U(1);
- both: 13;
- the SU(5)-singlet 5-form alone: 24, i.e. SU(5);
- **45_H together with 126bar_H: 12, i.e. SU(3)×SU(2)×U(1).**

The unbroken Cartan direction is exactly Y = (−⅓,−⅓,−⅓,½,½). Flipped
hypercharge and X are broken.

This checks the group theory only. Whether the 45 + 126 potential has this
minimum is a separate question. At tree level it does not; one-loop effects
make it viable (Bertolini–Di Luzio–Malinský 2010, cited).

### 2.3 Brane Yukawas: spin 2 only, and why four branes

A c = 2 brane at a point z contributes Y ∝ u(z)u(z)ᵀ, where u(z) is the
spin-1 coherent state at z (tested against explicit rotations).

**Proposition 1 (spin 2 only).** These matrices span only a
five-dimensional subspace of the six-dimensional space of symmetric Yukawa
matrices. They have no J=0 component, since Sym²(spin 1) = J0 ⊕ J2 and the
brane rule selects J=2 (tested). So no number of such branes produces a
generic Yukawa matrix in the family basis fixed by the sphere.

**The way out.** In the 4D theory, a U(3) rotation of the three 16s is a
field redefinition. The Spin(10) interactions are U(3) invariant, and only
the heavy family gauge bosons are not. Observables therefore depend on
(Y10, Y126) only up to simultaneous congruence U^T Y U.

**Proposition 2 (four branes suffice).** Count the real rank of the map from
N branes (positions, the 10_H and 126bar_H couplings, plus U(3)) to pairs
(Y10, Y126), whose target space has 24 real dimensions (tested):

| branes | rank | meaning |
|---|---|---|
| 2 | 16 | codimension 8 |
| 3 | 23 | one real relation |
| 4 | 24 | generic pairs |
| 4, without U(3) | 20 | the J=2 pairs only |

**Construction.** For four branes the realization is exact and constructive.
- *Step A.* Choose U ∈ U(3) that removes the J=0 part of both
  U^T Y10 U and U^T Y126 U. These are four real conditions on U(3), solved
  numerically.
- *Step B.* Any hyperplane of the J=2 space that contains both matrices
  meets the curve {u(z)u(z)ᵀ} (a rational normal quartic) in four points.
  Those points, the roots of a quartic polynomial in z, are the brane
  positions. The couplings then follow linearly.

It reproduces hierarchical targets to machine precision (tested). For
example, a Y10 with Takagi values (10⁻⁵, 3×10⁻³, 1) together with a generic
Y126 is reproduced with the smallest value correct to 10⁻⁸ relative. The
brane positions come out spread over the sphere. The hierarchy then lives in
the couplings, not in the geometry.

**Proposition 3 (the three-brane relation).** With three branes, the one
leftover relation is the Bargmann invariant. The phase of
⟨1|2⟩⟨2|3⟩⟨3|1⟩ for spin-1 coherent states equals the solid angle of the
geodesic triangle, which the three pairwise distances already fix
(l'Huilier). This matches to 10⁻¹⁴ (tested). A three-brane model would
therefore obey one real relation among the fermion-mass data. Its
phenomenological form is not worked out here.

### 2.4 Family symmetry

One brane leaves the rotation about its own axis unbroken. Two branes that
are not antipodal break the SU(2) family isometry completely, and four
generic branes certainly do (tested). The family gauge bosons get masses of
order √(Nδ)·M²/M_Pl ~ 10¹⁵–10¹⁶ GeV (an order-of-magnitude estimate;
δ is the deficit). That is far above flavour-changing-neutral-current bounds.

## 3. Scales (Phase 1b)

**One-loop running.** The chain is Spin(10) → 3221 at M_GUT (by 45_H), then
3221 → SM at M_I (by 126bar_H). The beta coefficients are computed from the
field content under the extended survival hypothesis. They reproduce the
textbook anchors (tested):
- SM: (41/10, −19/6, −7);
- two Higgs doublets: (21/5, −3, −7);
- MSSM: (33/5, 1, −3), with MSSM unification at 2.0×10¹⁶ GeV.

The 3221 coefficients, (−7, −3, −7/3, 11/2), are also checked by hand.
Forward re-running confirms that all four couplings meet.

| below M_I | M_I (GeV) | M_GUT (GeV) | 1/α_G | τ_p naive (yr) |
|---|---|---|---|---|
| one Higgs doublet | 1.0×10⁹ | 4.5×10¹⁶ | 46.2 | 3×10³⁸ |
| two Higgs doublets | 3.1×10⁹ | 2.3×10¹⁶ | 45.4 | 2×10³⁷ |

**Proton decay.** Super-Kamiokande requires τ(p → e⁺π⁰) > 2.4×10³⁴ yr,
i.e. M_GUT ≳ 4.4×10¹⁵ GeV in the naive estimate. Both cases pass by
10³–10⁴. Threshold corrections, which can shift M_GUT by a factor of a few
(τ ∝ M_GUT⁴), could bring the rate toward Hyper-Kamiokande's reach, but
nothing sharp is predicted.

**Compactification window (new constraint).** Using the flux module's
relations, 1/r = 2g_F M_Pl/3 and 1/(rM) = 2π^{1/4}√(g_F/3), and requiring:
- M_GUT ≤ 1/r, so that the breaking has a 4D description;
- rM ≥ 3, for classical control;

gives:
- one doublet: g_F ∈ [0.028, 0.047] and 1/r ∈ [4.5, 7.6]×10¹⁶ GeV;
- two doublets: g_F ∈ [0.014, 0.047] and 1/r ∈ [2.3, 7.6]×10¹⁶ GeV.

Requiring M_GUT ≤ 1/r caps rM at 3.9 (one doublet) or 5.5 (two doublets)
(tested). The window is open but narrow: the compactification scale is
pinned to within about a factor of 3.

**Seesaw (tension).** A type-I seesaw with M_R ≈ M_I ≈ 10⁹ GeV needs a Dirac
neutrino Yukawa of about 10⁻³ for m_ν ≈ 0.05 eV. SO(10) relates the Dirac
neutrino matrix to the up-quark matrix, which contains a top-like entry of
about 1; that would need M_R ~ 6×10¹⁴ GeV. So the model needs cancellations
in the Dirac matrix, a type-II contribution, or thresholds that raise M_I.
This is the known seesaw issue of minimal non-supersymmetric SO(10), not
something specific to BPR, but it is the tightest point found here.

## 4. Kill-check table

| # | check | result |
|---|---|---|
| 1 | Couplings: wanted allowed, dangerous forbidden | **pass** (computed) |
| 2 | Vevs leave exactly the SM, with the right hypercharge | **pass** (computed) |
| 3 | Breaking potential has that minimum | conditional (one-loop, cited) |
| 4 | Brane Yukawas can reach generic mass matrices | **pass** with four branes (computed, exact construction) |
| 5 | Those mass matrices fit the data | conditional (generic 10 + 126bar fits, cited) |
| 6 | Viable invisible QCD axion | **pass** with S (round 10); quality unresolved |
| 7 | Family gauge bosons heavy | **pass** (estimate) |
| 8 | Gauge unification and proton decay | **pass** at one loop |
| 9 | Compactification control with M_GUT ≤ 1/r | **pass, narrowly** (computed) |
| 10 | Seesaw neutrino masses | **tension** (known in minimal SO(10)) |
| 11 | Brane positions stabilized | open |
| 12 | Multi-brane background exists | conditional (constant-curvature spheres with cone points exist for small deficits: Troyanov 1991, Luo–Tian 1992, cited from memory) |
| 13 | Higgs mass and Λ | tuned, as before |

**Verdict.** No kill check fails. BPR-6D is now one specific model, not a
framework with blanks. It is conditional on the cited results, and it has one
genuine tension (the seesaw) and one narrow window (the compactification
scale).

## 5. What Phase 1 did and did not achieve

**Achieved:**
- A definite field content, with every coupling checked.
- An exact breaking pattern.
- Two new structural results about brane Yukawas: they are spin 2 only, and
  generic Yukawas need exactly four branes, built from the roots of a
  quartic.
- A compactification scale pinned to 1/r ≈ (2–8)×10¹⁶ GeV. This also
  narrows the U(1)_F coupling and the upper end of the axion band.

**Not achieved:**
- **Nothing about fermion masses is predicted.** Four branes give about 21
  real parameters (after removing overall rotations), comparable to the
  observables they must fit.
- The Higgs content is chosen for minimality, not derived.
- Brane positions are free moduli.
- The seesaw tension is unresolved.

**Next (Phase 2).** Compare with data at the level the model now allows:
- redo the Yukawa fit, or check that the cited fits land inside the
  four-brane construction;
- add two-loop and threshold effects to tighten M_I, M_GUT and the window;
- turn the window into the axion and Kaluza–Klein statements that
  experiments can test.

## 6. Limitations

- The Higgs content is chosen, not derived.
- The breaking potential and the fermion fits are cited, not computed.
- The running is one-loop, with no thresholds and no Kaluza–Klein
  thresholds near 1/r.
- The proton lifetime is a naive, order-of-magnitude estimate.
- The family-gauge-boson masses are order-of-magnitude estimates.
- Brane positions are moduli, and the tension-induced distortion of the zero
  modes is not included.
- The control criterion rM ≥ 3 is a convention.
- Citations marked "from memory" were not checked against the texts.

## 7. Independent review

Pending at the time of writing.
