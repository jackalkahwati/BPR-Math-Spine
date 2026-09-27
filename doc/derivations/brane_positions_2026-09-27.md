# Phase 1d: fixing the brane positions with a vortex condensate

2026-09-27. Status: an explicit mechanism with exact and numerical checks. The
implementation is `bpr/brane_stabilization.py`, with tests in
`tests/test_brane_stabilization.py` and a demo in
`scripts/demo_brane_stabilization.py`. **No empirical validation is
claimed.** The independent review is recorded in section 7.

## 0. The problem

Phase 1 ([minimal_model_2026-09-26.md](minimal_model_2026-09-26.md)) left the
positions of the four branes as free moduli. Five physical position modes
survive after the family gauge bosons eat three, and they couple to fermions
through the position-dependent Yukawas. Light moduli of this kind would mean
fifth forces, flavour violation and a cosmological moduli problem. Phase 1
marked this as potentially fatal.

## 1. What moves the branes

| effect | size | sign or shape | status |
|---|---|---|---|
| Classical gravity and flux with tension-only branes | 0 | none | positions are moduli: constant-curvature spheres with cone points exist for any positions and small deficits (Troyanov 1991, Luo–Tian 1992, cited) |
| Exchange of bulk scalars between brane couplings (e.g. the 45_H) | about κ²/(4πr⁴) | **attractive** for like-sign couplings; four branes cannot all repel pairwise | collapse would make the Yukawas rank 1 |
| One-loop Casimir forces between conical defects | about N_dof δ²/(16π² r⁴) | not computed | unknown |
| Pinning by the Δ_R condensate profile | about κ M_I²/r² | small | negligible: 10⁻¹⁶ of the scale 1/r⁴ |

So nothing in the Phase 1 content both dominates and fixes the positions. A
new ingredient is needed.

## 2. The mechanism: a charge −4 vortex condensate

Add a bulk scalar χ that is an SO(10) singlet with U(1)_F charge −4. The
choice of charge is the point: the flux then forces exactly four vortices.

**The lowest level.** χ has spin weight s = 2 in unit flux (a charge F field
has s = −F/2). Its lowest level is therefore a spin-2 multiplet: five states
at m²r² = l(l+1) − s² = 2. The operator ð annihilates it (tested).

**Exactly four zeros.** Up to a gauge phase and a positive factor, every
lowest-level profile is a quartic polynomial in the stereographic coordinate
ζ = tan(θ/2)e^{iφ}. So it has exactly four zeros, counted with multiplicity.
This follows from the degree of the line bundle, which is the flux quantum
times the charge. The polynomial roots are checked to be true zeros of the
profile, to 10⁻¹⁶ (tested).

**The zeros form a regular tetrahedron.** A slightly tachyonic bulk mass
makes χ condense in the lowest level. Its quartic self-interaction then
selects the profile that minimizes ∫|χ|⁴ at fixed ∫|χ|². This is the sphere
version of the Abrikosov vortex-lattice problem. The minimum:
- is unique up to rotations: every one of 40–60 random starts reaches the same
  value, to 10⁻¹⁴;
- equals the known spin-2 "tetrahedral state"
  (|2,2⟩ + √2|2,−1⟩)/√3, to 10⁻¹⁰;
- has its four zeros at a **regular tetrahedron**: every pairwise dot product
  equals −1/3 to 10⁻⁸ (all tested).

Other four-zero configurations cost more energy:

| configuration | ∫\|χ\|⁴ |
|---|---|
| **regular tetrahedron** | **0.0947** |
| equatorial square | 0.1137 |
| double zeros at both poles | 0.1137 |
| all four zeros at one point | 0.2210 |

**Stability.** The Hessian of the quartic energy on the space of lowest-level
states has exactly three zero modes. These are the rotations, which are gauge
directions eaten by the family gauge bosons. The other five modes are
positive: an A4 doublet at 0.152 and a triplet at 0.505 (tested).

**Pinning.** A brane-localized term κ|χ(z_a)|² with κ > 0 costs nothing only
at a zero, so each brane sits at one vortex. The stiffness at a zero is
positive and isotropic, 0.53 in normalized units (tested). The coupling is
allowed: F-neutral, zero normal weight, an SO(10) singlet.

**Result.** The four branes sit at the vertices of a regular tetrahedron.
Every position mode is either a gauge rotation or massive.

## 3. Scales and the window

Let v be the canonically normalized 4D vev of χ's lowest mode. The
brane-modulus masses are then of order

    m r ~ √κ (v r) / (√δ (rM)²) ≈ 0.26 at v r = 1, κ = 1, δ = 0.1, rM = 3.5,

i.e. near the compactification scale (~10¹⁶ GeV). That is heavy enough to
remove the fifth-force and cosmological-moduli dangers.

Three conditions fix the window for v r (defaults: κ = 1, δ = 0.1,
g₄ = 0.03, λ = 1, N_dof = 100):
- **Pinning beats Casimir forces:** κ(vr)² must exceed ten times
  N_dof δ²/(16π²). This gives v r ≳ 0.25.
- **The flux stays nearly uniform:** the U(1)_F mass that χ induces,
  (4/3)g₄v, must stay below 1/r. With g₄ ≈ 0.03 from the Phase 1 window, this
  allows v r up to about 25.
- **The lowest-level description holds:** λ(vr)² must stay below the level
  gap of 6. This gives v r ≲ 2.4.

The window 0.25 ≲ v r ≲ 2.4 is open. It closes if the pinning is very weak
(κ ~ 10⁻³, tested). Inside it, the flux distortion and the change to the
fermion zero-mode profiles are about (g₄vr)² ~ 10⁻³, and the index theorem
still guarantees three families.

## 4. Consequences

**A residual A4 family symmetry.** The condensate breaks SU(2)_iso × U(1)_F.
The surviving transformations are rotations of the tetrahedron combined with
the U(1)_F phase that compensates χ's phase: 3-fold rotations multiply χ by
e^{±2πi/3}, and 2-fold rotations leave it invariant. On the families
(spin 1, F = 3) they generate a group of order 48. Modulo phases it has
order 12, i.e. **A4 × Z4**, where Z4 ⊂ U(1)_F is the part left unbroken by a
charge-4 condensate. The families form an irreducible **A4 triplet** (Burnside
test).

A4 is the family symmetry used in many neutrino-mixing models (for example
Ma–Rajasekaran 2001, Babu–Ma–Valle 2003, Altarelli–Feruglio 2005; cited from
memory). Here it is not imposed: it is what a flux-forced vortex condensate
leaves of the sphere's isometry.

**Yukawas are not constrained by the positions alone.** With the branes
fixed at the tetrahedron, the four brane matrices span four dimensions. With
free couplings c_a and d_a and U(3) family redefinitions, the map to
(Y10, Y126) still has full rank 24 (tested). Fixing the positions therefore
removes the moduli without costing the model its ability to fit data. It
does not make fermion masses predictive. Predictivity would need A4-related
brane couplings (identical branes), which is left to Phase 2.

**χ can do S's job.** The bulk term χ*³·10_H·10_H is allowed: F-neutral and
an SO(10) singlet, though of higher dimension. It locks the Higgs-doublet
phases to χ's phase, so that θ̄ = 3b + 9 arg χ (invariant, since
3·12 − 9·4 = 0). The QCD axion would then have

    f ≈ (9/f_b² + 81/v²)^{−1/2} ≈ min(f_b/3, v/9),

about 10¹⁵ GeV in the window, i.e. m_a ≈ 6×10⁻⁹ eV. This applies if the χ link
dominates S's, and the separate brane singlet S may then be unnecessary.

## 5. Kill check 11, updated

| check | before | after |
|---|---|---|
| Brane positions stabilized | open, potentially fatal | **pass**, conditional on the added field χ (F = −4) and the window 0.25 ≲ v r ≲ 2.4 |

## 6. Limitations

- χ is added, chosen because charge 4 forces exactly four vortices; it is not
  derived.
- The lowest-level description needs λ(vr)² below the level gap and a nearly
  uniform flux. Outside that regime vortex cores localize the flux, and the
  zero-mode profiles change: these are the Aharonov–Bohm branes of
  Buchmüller–Dierigl–Tatsuta.
- The Casimir forces are estimated by dimensional analysis only.
- Unequal brane tensions distort the tetrahedron at O(δ).
- Before χ condenses the positions are free. Relaxation, and possible domain
  walls between mirror configurations, are not analysed.
- The A4 × Z4 group is computed on the families only. The Higgs condensates
  break it further.

## 7. Independent review

Pending at the time of writing.
