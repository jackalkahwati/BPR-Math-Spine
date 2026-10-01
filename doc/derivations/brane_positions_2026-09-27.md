# Phase 1d: pinning the brane positions with a vortex condensate

2026-09-27. Status: an explicit mechanism with exact and numerical checks. The
implementation is `bpr/brane_stabilization.py`, with tests in
`tests/test_brane_stabilization.py` and a demo in
`scripts/demo_brane_stabilization.py`. **No empirical validation is
claimed.**

The independent review (section 7) confirmed every structural number
exactly, and found two blockers in the first version, both repaired:
- Fixed positions **do** constrain the Yukawas.
- The pinning is at best **metastable**, so kill check 11 is conditional,
  not passed.

## 0. The problem

Phase 1 ([minimal_model_2026-09-26.md](minimal_model_2026-09-26.md)) left the
positions of the four branes as free moduli. Five physical position modes
survive after the family gauge bosons eat three, and they couple to fermions
through the position-dependent Yukawas. Light moduli of this kind would mean
fifth forces, flavour violation and a cosmological moduli problem.

## 1. What moves the branes

| effect | size | sign or shape | status |
|---|---|---|---|
| Classical gravity and flux with tension-only branes | 0 | none | positions are moduli: spherical metrics with cone points exist for any positions and small deficits (Troyanov 1991; Luo–Tian 1992) |
| Tree-level exchange of the bulk 45_H (vev at M_GUT, M_GUT·r ≈ 0.6–1) and of other bulk scalars | about κ²/(4πr⁴) | attractive for like-sign couplings of one real scalar | not negligible; favours stacking branes, which makes the Yukawas rank deficient |
| One-loop Casimir forces between conical defects | about N_dof δ²/(16π² r⁴) ≈ 0.006/r⁴ | not computed | |
| One-loop forces from the Higgs brane terms (Phase 1 needs O(1) terms for 10_H and 126bar_H) | the review's crude estimate: up to about 0.3/r⁴, dominated by the derivative terms | not computed | may dominate |
| Pinning by the Δ_R condensate profile | about κ M_I²/r² | small | negligible: 10⁻¹⁶ of 1/r⁴ |

## 2. The mechanism: a charge −4 vortex condensate

Add a bulk scalar χ that is an SO(10) singlet with U(1)_F charge −4.

**Exactly four zeros.** Charge −4 is an integer in the charge-1 unit that the
Green–Schwarz note requires, and scalars do not enter I8, so the anomaly
analysis is untouched. In unit flux χ has spin weight s = 2. Its lowest level
is a spin-2 multiplet at m²r² = 2, with the next level at 8, and ð
annihilates it (tested).

Every lowest-level profile is (1 + |ζ|²)⁻² times a quartic polynomial in
ζ = tan(θ/2)e^{iφ}, up to a gauge phase. So it has exactly four zeros, and a
missing top coefficient puts zeros at the south pole. Both pole cases are
tested, and the polynomial roots are zeros of the profile to 10⁻¹⁶.

**Condensation and the tetrahedron.** For a bulk mass between −8/r² and
−2/r², χ condenses in this level. The quartic then decides the shape, and
the regime matters:
- **Type II:** λ above the critical, BPS value, of order q_χ²/2 with
  q_χ = (4/3)g₄ ≈ 0.04 (tested). The condensate minimizes ∫|χ|⁴ at fixed
  ∫|χ|².
- **Critical coupling:** the four zero positions are exact moduli. The vortex
  moduli space on S² is CP⁴ (Bradlow; Baptista–Manton, JMP 44 (2003) 3495).
- **Type I:** the zeros coalesce.

In the type-II case the minimum is the tetrahedral state
(|2,2⟩ + √2|2,−1⟩)/√3, with its four zeros at a regular tetrahedron. The
exact values below were found by the review via Clebsch–Gordan algebra and
are all tested:

| configuration | ∫\|χ\|⁴ |
|---|---|
| **regular tetrahedron** (global minimum; every start reaches it) | **25/(84π)** |
| equatorial square | 5/(14π) |
| double zeros at both poles | 5/(14π) |
| all four zeros at one point | 25/(36π) |

The Hessian on CP⁴ has three zero modes, which are the rotations. The other
five modes are positive: 10/(21π) twice and 100/(63π) three times.

**Pinning, and why it is only metastable.** A brane term κ|χ(z_a)|² with
κ > 0 vanishes only at the zeros, so it pins branes to vortices. The
stiffness at a zero is 5/(3π), isotropic because the zero is a simple
holomorphic zero. But:
- **Stacking costs nothing.** The pinning energy is zero for any assignment
  of branes to zeros, including two branes on one zero (tested). Attractive
  scalar exchange favours stacking, and stacked branes give rank-deficient
  Yukawas.
- **The barrier is low.** A pinned brane moves to a neighbouring vortex
  across the edge-midpoint saddle, which costs only
  5/(16π)·κ(vr)²/r⁴ ≈ 0.1·κ(vr)²/r⁴ (tested).
- **Two labellings.** The one-brane-per-vortex state comes in two
  degenerate chiral labellings with different Yukawas (the mirror
  configurations).

So the one-brane-per-vortex tetrahedron is **metastable at best**, and
whether it survives depends on the uncomputed one-loop forces of section 1.

**A candidate cure, not yet checked.** Let each brane carry localized U(1)_F
flux, so that χ must vanish there. Each brane would then force one zero, and
stacking would cost quartic energy. But localized fluxes change the fermion
zero-mode counting and profiles, which could spoil the three families. This
is not analysed.

## 3. Scales and the window

Let v be the canonical 4D vev of χ's lowest mode, and λ the quartic of its
amplitude. The modulus masses are then of order

    m r ~ √(κ·5/(3π)) (v r) / (√δ (rM)²) ≈ 0.19 at v r = 1, κ = 1, δ = 0.1, rM = 3.5,

near the compactification scale.

The conditions for v r (tested), taking pinning to win when κ(vr)² exceeds
ten times the one-loop force:

| condition | bound |
|---|---|
| pinning beats the Casimir estimate alone | v r ≳ 0.25 |
| pinning beats the Casimir estimate plus the Higgs brane-term estimate (0.3/r⁴) | v r ≳ 1.75 |
| the lowest-level description holds: λ(vr)² below the gap 6 (λ = 1) | v r ≲ 2.45 |
| the flux stays nearly uniform: U(1)_F mass √2·(4/3)g₄v below 1/r (g₄ = 0.03) | v r ≲ 17.7 |

With the Higgs brane-term estimate included, the window is 1.75 ≲ v r ≲ 2.45.
It closes for κ ≲ 0.5 (tested). The window is **narrow and not robust**.

## 4. Consequences

**Fixed positions constrain the Yukawas.** With the branes fixed, the four
brane matrices span a subspace W of complex codimension 2. A target pair is
reachable only if some U ∈ U(3) puts both UᵀYU in W, which is 8 real
conditions on the 8-dimensional U(3)/U(1). The map still has full rank 24, so
its image contains an open set. But it is a **proper subset**:
- the review reached only 24 of 40 random pairs;
- the hierarchical example of the Phase 1 note is unreachable, with a best
  squared distance of about 5×10⁻⁴ that is stable over 1500 starts, and above
  10⁻⁴ in any search (tested);
- the review's solution counting gives degree zero, so existence is not
  guaranteed anywhere;
- pairs built from the tetrahedral branes are reached exactly (tested).

So pinning the positions **is** a constraint on the flavour sector. Phase 2
has to fit with the branes fixed, not over generic Yukawas.

Phase 2 ([phase2_fermion_fit_2026-09-27.md](phase2_fermion_fit_2026-09-27.md))
identifies W exactly: the pure J = 2 matrices annihilated by χ itself. It
finds that the pinned model fails the charged-fermion data at leading order
(best found χ² ≈ 120; m_s about 3.6–3.9 times too large), while free positions
fit. Phase 2b finds that a one-derivative brane vertex of the same order as the
other corrections, near the top of its natural size, removes the conflict.

**The A4 × Z4 symmetry of the χ sector.** Rotations of the tetrahedron,
combined with the compensating U(1)_F phase, leave χ invariant. On the
families they form A4 × Z4: order 48, and order 12 modulo phases. The
families form an irreducible A4 triplet (review: a split direct product).

This is a symmetry of the χ-plus-geometry sector only. **Identical branes
give zero Yukawas.** With A4-symmetric brane couplings the Higgs lowest level
splits as 1 + 3 + 3. A unique light state must then be the singlet, and its
ð̄ vanishes at every vertex by a 3-fold selection rule (Phase 2,
`fermion_fit.identical_brane_selection_rule`). A4 must therefore be broken
at O(1) by the brane couplings. The A4 neutrino-model literature (for example
Ma–Rajasekaran 2001, Babu–Ma–Valle 2003, Altarelli–Feruglio 2005) is related
by theme only.

**Charge lattice.** Since gcd(3, 4) = 1, χ's charge together with the
matter's generates the full charge lattice. It also breaks the approximate Z₃
one-form symmetry of the Green–Schwarz note.

**χ and the axion: caveats.** The bulk term χ*³·10_H·10_H is allowed
(F-neutral, an SO(10) singlet, higher dimension). It would give
θ̄ = 3b + 9 arg χ (invariant, since 3·12 − 9·4 = 0) and
f ≈ min(f_b/3, v/9) ~ 10¹⁵ GeV. But it comes with three problems:
- **Bμ:** it generates Bμ ~ c(vr)³/((rM)⁴r²), about c·(4×10¹⁵ GeV)².
  Keeping two light doublets needs c ≲ 10⁻²⁷. Otherwise only the one-doublet
  variant of Phase 1 survives.
- **Quality:** χ³e^{ib} is gauge invariant and acts on b + 3 arg χ = θ̄/3,
  which is a quality problem.
- **Domain walls:** the domain-wall number is 3 along that direction, so χ
  must condense before inflation.

It is **not** claimed that χ replaces the singlet S.

## 5. Kill check 11, updated

| check | before | after |
|---|---|---|
| Brane positions stabilized | open, potentially fatal | **conditional**: a metastable one-per-vortex tetrahedron, given the added χ (F = −4, type II). Stacking and the one-loop brane forces are unresolved; the window is 1.75 ≲ v r ≲ 2.45 with the Higgs brane-term estimate. |

## 6. Limitations

- χ is added, not derived.
- Metastability: stacking, the low barrier, and uncomputed one-loop forces.
- The type-II condition, the lowest-level regime and a nearly uniform flux
  are assumed.
- Unequal tensions distort the tetrahedron at O(δ).
- Relaxation before χ condenses, and domain walls between the mirror
  labellings, are not analysed.
- The localized-flux cure is not analysed. Localized-flux branes of this
  kind were studied by Buchmüller–Dierigl–Tatsuta on the orbifold T²/Z₂,
  not on S².

## 7. Independent review

An independent review, with its own code, verified:
- the level and zero count, including the pole cases;
- the charge normalization, and that the Green–Schwarz mechanism, the three
  families and the radion are untouched;
- the global minimum and every exact energy;
- the Hessian, stiffness and residual group (A4 × Z4, a split direct
  product);
- span 4 and rank 24;
- the axion arithmetic, the scale arithmetic and the citations.

Its findings, all addressed:
- **Blocker 1.** "No loss of fitting ability" was false. Fixed positions
  reach only a proper subset. Reachability is now computed and tested, and
  the Phase 1 example is a regression test.
- **Blocker 2.** Kill check 11 was not passed. The stacking degeneracy, the
  low barrier and the Higgs brane-term forces are added, and the verdict is
  downgraded to conditional.
- **Major:**
  - identical branes give zero Yukawas, and A4 belongs to the χ sector only;
  - the type-II condition and the vortex literature are added;
  - the axion-link caveats are stated, and the claim "S may be unnecessary"
    is withdrawn.
- **Minor:**
  - the U(1)_F mass normalization (flux bound 17.7) and the tachyonic mass
    range are fixed;
  - 45_H exchange is added to the table, the Buchmüller–Dierigl–Tatsuta
    reference is rephrased, and the charge-lattice note is added;
  - the caching bugs are fixed;
  - the tests now use exact oracles, with south-pole, stacking and
    reachability tests added.
