# Yukawa mechanisms for BPR-6D: brane and bulk Higgs fields

2026-09-26. Status: exact symmetry analysis with explicit rotation matrices
and zero modes. The implementation is `bpr/yukawa_mechanisms.py`, with tests
in `tests/test_yukawa_mechanisms.py` and a demo in
`scripts/demo_yukawa_mechanisms.py`. An independent review (section 5) found
one blocker, the J_z charge of the brane Higgs; its repairs are applied.

## 0. Question

Two earlier results frame this note:
- Round 3 ([family_symmetry_from_flux_2026-09-26.md](family_symmetry_from_flux_2026-09-26.md))
  showed that the minimal BPR-6D content has no Yukawa couplings, and that a
  bulk Higgs couples only through an internal one-form 10 or 126 in the J=2
  channel.
- Round 7 ([gut_breaking_2026-09-26.md](gut_breaking_2026-09-26.md)) showed
  that the Higgs sector must be supplied anyway.

So which supplied Higgs fields give usable Yukawa couplings? The **families
are an SU(2)-isometry triplet** (m = 1, 0, −1), and that structure is what
constrains every option below.

## 1. A Higgs localized at a point

The first option is a boundary mechanism: a 4D Higgs on a codimension-2
brane at a point of S² (by symmetry, the north pole). It carries F-charge −6,
so that it neutralizes two charge-3 families. The point is fixed by the
rotations about its axis, so J_z is exactly conserved there.

**Lemma 0 (the J_z charge of a brane Higgs).** Three facts, each checked on
the explicit harmonics (tested):
- A harmonic of spin weight s is nonzero at the pole only for m = −s.
- The zero modes f_m have spin weight −1: −3/2 from the monopole (F-charge 3)
  plus +1/2 from the spinor. So only f₁ is nonzero at the pole.
- ð̄ annihilates the zero modes, and ð^a f_m is nonzero at the pole exactly
  when m = 1 − a.

A fermion bilinear with a derivatives ð therefore has spin weight −2 + a and
J_z = 2 − a at the pole. A brane field of F-charge −6 and normal-bundle spin
weight s_h has spin weight s_h + 3, the monopole's Wu–Yang contribution being
+3. A rotation-invariant coupling needs total spin weight zero, so
a = −1 − s_h, and the coupling links families with m + m′ = c, where

    c = s_h + 3.

The cases are:
- **Brane scalar (s_h = 0): c = 3, no Yukawa at any derivative order.** This
  is the brane version of round 3's all-orders exclusion of the bulk scalar.
  The operator ψᵀCΓ^aD_aψ, which an earlier draft used to give a scalar
  c = 1, vanishes: Γ^aD_a is the internal Dirac operator, which annihilates
  the zero modes.
- **Normal-bundle vector (s_h = −1): c = 2.** This is the local restriction
  of round 3's bulk one-form Higgs (its J=2 channel). It needs no derivatives
  and couples only f₁f₁, so it gives rank 1: one massive family.
- **c = 1 and c = 0** need normal-bundle spin weight −2 and −3 respectively.

**Theorem 1 (one brane never gives three distinct masses).** For a Higgs of
J_z charge c at one point, Y_{mm′} = 0 unless m + m′ = c. The J_z-covariant
complex symmetric matrices on the triplet give:
- **|c| ≥ 3:** no coupling at all (this includes the scalar).
- **c = ±2:** a single entry, rank 1, so one massive family (the vector).
- **c = ±1:** entries (1,0) or (0,−1), so the spectrum is (|a|, |a|, 0).
- **c = 0:** entries (1,−1) and (0,0), so the spectrum is (|a|, |a|, |b|).

A 120 on a brane gives antisymmetric matrices: c ∈ {−1, 0, 1} gives
(|a|, |a|, 0), and nothing otherwise.

*Checks.*
- The covariant matrices are computed for c = −4…4 as a nullspace, using
  explicit spin-1 rotation matrices, and random elements are classified by
  their spectra (tested).
- Every c=0 basis element commutes with diag(e^{−ima}). The c=0 case is the
  J=0 singlet plus the M=0 component of J=2, consistent with round 3's
  uniaxial 2:1:1.

**Proposition 2 (derivative counting).** Near the point, the zero modes
vanish as θ^{n_m} with n_m = 1 − m, i.e. orders (0, 1, 2) for m = 1, 0, −1
(and 0…4 for five families). Every entry with m + m′ = c needs
n_m + n_{m′} = 2 − c derivatives, each suppressed by ε = 1/(M_* r). So while
J_z is exact, all entries of one brane sit at the same order and there is no
internal hierarchy (tested for c = 0, 1, 2).

**Several branes.**
- **c = 0 branes** at generic points give generic spectra. Antipodal branes
  share the rotation axis, so J_z stays exact and the degenerate pair
  survives (tested).
- **Two vector (c = 2) branes** at angular separation γ give rank 2: one
  family stays massless, and m₂/m₁ ≈ 0.2γ² (tested for γ = 0.3, 0.1, 0.01).
  Brane separation is a geometric source of hierarchy.
- **Three vector branes within an angle γ** give all three masses, scaling
  as (1, ≈0.45γ², ≈0.03γ⁴) (tested for γ = 0.3, 0.1, 0.03). The reason is
  structural. Each c = 2 brane contributes a rank-1 projector onto a spin-1
  coherent state v(z) ∝ (1, √2z, z²), and the Vandermonde determinant of
  three nearby points scales as γ³.

The O(1) random coefficients set the scatter around these medians.
Statistics such as "how often is a ratio below 10⁻⁵" measure the prior on
those coefficients, not the model.

**Broken J_z at a brane (the Heckman–Vafa point-Yukawa texture).** Round 3
already needs the SU(2) family symmetry broken. Once J_z is broken near the
brane (by other branes, squashing or a bulk vev), the vanishing orders act as
Froggatt–Nielsen charges: Y_{mm′} ~ a_{mm′} ε^{n_m+n_{m′}}. With O(1)
random a and ε = 0.1, the median log₁₀ mass ratios are (0, −1.9, −3.8)
(tested). This is the mechanism of F-theory point Yukawas (Heckman–Vafa,
arXiv:0811.2417; also arXiv:0910.0477, arXiv:0907.4895 and arXiv:1104.2609,
cited from the review and not checked against the texts). The clustered
vector branes above are the same structure: the Taylor expansion of the
coherent states reproduces the orders (0, 1, 2).

**Summary of section 1.** A vector brane Higgs gives one heavy family with
no suppression. J_z breaking, from clustered branes or at a single brane,
gives the form (1, ε², ε⁴). That form is hierarchical, but ε (or γ) is a
free parameter.

## 2. A bulk internal-vector Higgs: allowed, tuned

A non-gauge (Proca) internal one-form 10 or 126 of F-charge −6 has monopole
number |n| = 6 in unit flux. Its aligned lowest level is

    m² r² = |s_e| + 1 − g|n|/2 + M²r²,   |s_e| = |n|/2 − 1 = 2,

with g the gyromagnetic ratio. For the gauge value g = 2 and M = 0 this is
−3, matching round 3, where it was checked against Atiyah–Bott and a
Yang–Mills Hessian. The J=2 Higgs quintet can be made light only by tuning
M²r² to 3g − 3 to a precision of (m_H r)² ≈ 10⁻³⁰ for 1/r ≈ 10¹⁷ GeV. This is
the ordinary hierarchy problem; BPR-6D does not solve it.

When it works, round 3's Theorem 4 applies:
- a real vev obeys m₃ = m₁ + m₂, which is excluded;
- a complex vev reaches any spectrum;
- hierarchies need a near-null orientation.

A null J=2 vev is not a second fine-tuning. It is the ferromagnetic phase of
a spin-2 order parameter, which is the ground state over an open region of
the quartic couplings (Ciobanu–Yip–Ho, PRA 61 (2000) 033607; Koashi–Ueda,
PRL 84 (2000) 1066). It gives rank 1 at leading order, and a hierarchy then
needs a small departure from the null direction. The weak-scale tuning of the
Higgs mass is common to every mechanism in this note, brane Higgs fields
included.

## 3. Status of the Yukawa sector

| mechanism | Yukawa? | hierarchy? | status |
|---|---|---|---|
| bulk scalar 10/126 | no, at any derivative order (round 3) | — | excluded |
| brane scalar 10/126 | no, at any derivative order (c = 3) | — | excluded |
| SO(12) gauge–Higgs | paired families, vacuum instability (round 3) | — | excluded |
| one brane Higgs, J_z exact | vector (c = 2): rank 1; others: a degenerate pair or nothing | none internal | not a sole source |
| two vector branes | rank 2 | m₂/m₁ ≈ 0.2γ² | allowed, γ free |
| three clustered vector branes, or J_z broken at a brane | yes | (1, ε², ε⁴) form | allowed, ε free |
| bulk Proca 10/126 (J=2) | yes, any spectrum | ferromagnetic J=2 phase gives rank 1; hierarchy needs a small breaking | allowed |

Every row also needs the weak-scale Higgs mass tuning of about 10⁻³⁰.

Not analysed here, and open:
- a squashed S² or non-uniform flux, which keeps the zero-mode count
  (Aharonov–Casher) but changes the profiles, and so the normalizations;
- flux localized at a brane (Aharonov–Bohm, as in Buchmüller–Dierigl–Tatsuta,
  arXiv:1804.07497), which shifts the J_z charges and vanishing orders by
  fractions;
- triplet-flavon models of the SU(2) family symmetry (King–Ross type).

Two further routes do not help:
- Wilson-line (Hosotani) Higgs fields need non-contractible loops, and S² is
  simply connected. Internal gauge components are the SO(12) case, already
  excluded.
- Exponential suppression from magnetized tori (Cremades–Ibáñez–Marchesano)
  needs a torus or large flux. With three families on S², the coherent-state
  overlaps are only power-law, cos^{2j}(γ/2).

**Conclusion.** Yukawa couplings can be supplied. The natural brane Higgs, a
normal-bundle vector, gives one heavy family, and J_z breaking gives the
Froggatt–Nielsen form (1, ε², ε⁴). But ε, γ and every coefficient are free,
so **no hierarchy is predicted**. The flavor sector stays open, and the
legacy flavor formulas remain unconnected phenomenology.

## 4. Limitations

- Every Higgs field here is supplied.
- Brane dynamics, tension and backreaction are not modelled. ε, the size of
  J_z breaking and the brane positions are free parameters.
- Brane-localized flux is not included (section 3).
- The UV consistency of a charged Proca field is not addressed.

## 5. Independent review

An independent review, with its own scripts, verified:
- which harmonics survive at the pole, and the ð-derivatives of the zero
  modes;
- the c-table of Theorem 1 and the c = 0 and c = 2 bases;
- the antipodal and c = 2 two-brane cases;
- the Froggatt–Nielsen-type texture;
- the Proca level against the flat-space Nielsen–Olesen limit (1 − g)|eB|,
  with eB r² = |n|/2. The aligned modes are divergence-free and do not mix
  with the 4D vector or longitudinal modes.

Its findings, all repaired:
- **Blocker.** The brane Higgs's J_z charge is c = s_h + 3, so a brane scalar
  gives no Yukawa and a normal-bundle vector gives rank 1. The earlier c = 1
  identification through ψᵀCΓ^aD_aψ was wrong, because that operator vanishes
  on zero modes.
- **Major:**
  - The two-brane analysis now covers antipodal and c = 2 branes. The earlier
    "fewer than 2% below 10⁻⁴" statistic, which only measured the prior, is
    removed.
  - "No natural Yukawa mechanism" becomes "no hierarchy is predicted", with
    the Heckman–Vafa texture and the open routes named.
  - "Doubly tuned" is withdrawn: the null J=2 vev is a ferromagnetic phase,
    and the weak-scale tuning is common to every row.
- **Minor:**
  - Wilson lines, brane-localized flux and magnetized-torus suppression are
    now addressed.
  - The 120 on a brane is treated.
  - Unused imports are removed; a hard-coded report value is now computed;
    and the derivative counting covers every c.
