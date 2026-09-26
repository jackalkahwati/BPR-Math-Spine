# Architecture decision: Lorentz invariance is fundamental (BPR-6D)

2026-09-26. Status: this is a decision, with its reasons and consequences. It
is not a derivation. It resolves the decision point at the end of the
[unification map](unification_map_2026-09-25.md) (section 7). The supporting
calculations are
[flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md)
and [family_symmetry_from_flux_2026-09-26.md](family_symmetry_from_flux_2026-09-26.md).

The adoption was **provisional**, because Dirac quantization of the
Green–Schwarz couplings was unsettled for the chosen completion (section 2).
Round 4 settles it at the characteristic-class level
([green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md)).
The completion is quantizable iff the parent's U(1)_F charge is a multiple of
3 in units of the smallest charge, which makes the family number a multiple
of three. Round 5 shows that Ω₇^Spin(B(Spin(10)×U(1))) = 0, so there is no
global anomaly for the Spin × Spin(10) × U(1) structure
([global_anomalies_2026-09-26.md](global_anomalies_2026-09-26.md)). An independent review (section 8) found
one blocker, six major and five minor issues; its repairs are applied.

## 0. The decision point

Round 2 showed a naturalness obstruction
([common_light_cone_2026-09-25.md](common_light_cone_2026-09-25.md)).
Emergent fields on the preferred-frame lattices studied here do not share one
light cone unless something forces them to. Tree-level tuning does not survive
loops (Collins et al. 2004). Collider bounds require isotropic photon–electron
speed differences below about 10⁻¹¹ (Hohensee et al. 2009). The program had to
adopt one of three options:
1. **One relativistic fixed point.** Every low-energy field is an excitation
   of a single theory whose infrared is Lorentz-invariant.
2. **A protecting symmetry.** For example, supersymmetry pushes the lowest
   Lorentz-violating operators to dimension five (Groot Nibbelink–Pospelov
   2005), and sequestering suppresses them further (Pospelov–Shang 2012). Soft
   supersymmetry breaking regenerates dimension-3 and dimension-4 Lorentz
   violation, suppressed by the breaking scale.
3. **Lorentz invariance fundamental.** Keep the "boundary" idea, but place it
   inside a Lorentz-invariant spacetime, as a compact internal space.

## 1. Choice and reasons

**Option 3 is chosen.** The reasons are ranked:

1. **It meets two section-7 links by assumption and gives consistent chiral
   matter.**
   - The common light cone is automatic.
   - A massless spin-2 field with universal coupling is ordinary 6D general
     relativity. Weinberg's soft-graviton theorem forces universality.
     Weinberg–Witten is evaded because the graviton is elementary.
   - Local-anomaly-consistent chiral matter comes from the round-1
     completion.

   It does not meet the rest of section 7:
   - Three families come from a flux choice, which is equivalent to the Λ
     tuning.
   - The Standard Model group is not reached; Spin(10) is unbroken.
   - It abandons section 7's single-microscopic-Hamiltonian standard.
   - There is no tested prediction.

   Options 1 and 2 leave both the light cone and gravity open. Neither has a
   first calculation within this repository's reach:
   - Option 1 needs a lattice model with the Standard Model content at one
     Lorentz-invariant fixed point, and none is known. Weinberg–Witten
     obstructs the emergent graviton it would need.
   - Option 2 needs a lattice with exact supersymmetry, a notoriously hard
     lattice problem, and it still needs a graviton.
2. **It reuses the program's strongest exact result.** The minimal completion
   16₊(Q=1) ⊕ 16₋(Q=0) plus one Green–Schwarz 2-form gives three chiral 16s at
   flux 3 with vanishing Z16 count. It is the matter sector as it stands.
3. **It makes the program a definite effective field theory.** Its claims
   become checkable. Round 3 checked several of them, and some came out
   negative (section 3).

The cost is explicit. **BPR no longer claims to derive spacetime.** The
substrate programme becomes analogue physics. A lattice regulator of BPR-6D
would reintroduce Collins-type Lorentz violation and fermion doubling, so
"regulator" is a possibility to be examined, not an established role.

## 2. Definition of BPR-6D

- **Spacetime and gravity.** A six-dimensional Lorentzian manifold with
  Einstein gravity, (M⁴/2)R₆ − Λ. The vacuum is M4 × S².
- **Gauge group.** Spin(10) × U(1)_F, plus one Green–Schwarz 2-form B.
- **Matter.** Two 6D Weyl spinors:
  - 16 of Spin(10), with U(1)_F charge 1 and chirality +;
  - 16 with charge 0 and chirality −.

  This is the unique minimal (16-component) local-anomaly completion
  (`bpr/chiral_parent_completion.py`). With the parent at charge 1 its
  Green–Schwarz couplings fail Dirac quantization, and they fail for every
  lattice. With the parent at charge 3 in units of the smallest U(1)_F charge
  they pass with one non-chiral 2-form (round 4). BPR-6D therefore takes:
  - parent charge 3, the minimal choice (any multiple of 3 passes);
  - states of charge 1, required by completeness, which are massive in
    BPR-6D as defined;
  - flux |m|=1, which gives three families. That is the same geometry as the
    earlier "flux 3".
- **Background.** U(1)_F monopole flux m=3 on a round S². The S² is the
  "boundary" of the Boundary Phase Resonance name.
- **4D spectrum at tree level.**
  - Three chiral 16s, forming a triplet of the sphere's SU(2) isometry.
  - Massless Spin(10) × SU(2)_iso gauge bosons.
  - A Stückelberg-massive U(1)_F boson (Green–Schwarz with flux). Its
    coupling g4 therefore belongs to a massive vector, and
    M_Pl² = πm²/e² is a relation, not a testable prediction.
  - At least one axion from B that is not eaten.
  - The radion, at mass 1/r.
- **Scope of the U(1)_F selection rules.** The charge and spin-weight rules
  of the family note apply to local 6D operators. Non-local, axion-dressed or
  instanton-type contributions are not analysed.
- **Not specified.** The Higgs sector, the Spin(10) → Standard Model
  breaking, SU(2)_iso breaking, supersymmetry, and any UV completion.

## 3. Round-3 results

| Question | Answer | Status |
|---|---|---|
| Is M4 × S² with m=3 a vacuum? | Yes, with r=m/(2M²e), if Λ=2M⁸e²/m² | exact (classical), one tuning |
| Is the size stable? | Classically, the breathing mode is, with m_ψ=1/r0 exactly; other Einstein–Maxwell modes are cited (RSS 1983) | exact / cited |
| 4D scales | M_Pl²=πm²/e², 1/r=2g4M_Pl/m; classical control needs g4 below an O(1), convention-dependent bound | exact relations |
| Is m=3 selected? | No. At fixed Λ, one flux is flat, lower fluxes are AdS, the window m₀<m<(2/√3)m₀ is dS (empty for m₀=3), and higher fluxes have no vacuum. This is the known Einstein–Maxwell landscape applied to the family count. | exact (classical), negative |
| Family symmetry | Three families = one SU(2) triplet of the sphere's isometry | exact |
| Yukawas | The minimal content has no 16⊗16 field, so no Yukawa. Selection rule for extensions (all orders, perturbative in U(1)_F): only an F-charge −2 internal one-form 10 or 126 in J=2 couples. Scalar Higgs fields and the 120 never couple. | exact, negative |
| SO(12) gauge–Higgs fix | Families come even (2k) and paired, so a projection to three leaves one massless; the J=2 level is tachyonic and destabilizes the flux vacuum | exact / Hessian- and index-checked |
| Masses | Real J=2 vev ⇒ m₃=m₁+m₂ (excluded); complex vevs reach any spectrum | exact, no viable relation |
| Are the Green–Schwarz couplings Dirac-quantizable? (round 4) | Iff the parent charge is a multiple of 3; then n_gen ∈ 3ℤ, and three is one flux quantum | exact (characteristic-class) |

## 4. What survives and what is reclassified

| Earlier piece | New role |
|---|---|
| Ring and cubic substrate theorems, condensate regime, acoustic metric | Exact mathematics about lattice Bose systems. Analogue models, not the origin of spacetime. |
| Link-boson emergent U(1) | An analogue of emergent gauge fields. Not the source of the Standard Model gauge fields, which are fundamental in BPR-6D. |
| Common-light-cone obstruction | Still true for its substrate class. BPR-6D resolves it by assumption, not by solving it. |
| R+R² induced gravity and R+R² inflation | Superseded as the gravity sector by 6D Einstein gravity. |
| Calibrated G | M_Pl²=πm²/e² relates G to the (Stückelberg-massive) U(1)_F coupling but does not predict it. |
| Path B (dihedral 2D gauge) | Separate and unconnected. Its falsified predictions stay withdrawn. |
| Chiral parent + completion | Adopted as the matter sector, provisionally (2-form quantization). |
| Flavor formulas, "205 predictions from (J,p,N)", "41 DERIVED" | Unconnected phenomenology. BPR-6D derives no mass or mixing relation (family note, section 4). The substrate parameters (J,p,N) have no BPR-6D counterpart. |
| Emergent "3+1 dimensions", "Lorentz invariance to exp(−p^{1/3})", ξ₂=1/p dispersion, the GUP | Superseded. In BPR-6D spacetime is six-dimensional and exactly Lorentz-invariant, and p has no counterpart. |
| Legacy Casimir claims, the phonon λ~10⁻⁸ result, the MEMS direction | Not supported by BPR-6D. The only boundary is the internal sphere, of radius r = (m/2g4)ℓ_P, and no laboratory-scale boundary field exists. |
| Fine-structure constant | An inverse calibration in the substrate parameter p, with no BPR-6D counterpart. |

The README's "CURRENT CLAIM (BPR 2.0)" header is superseded, for the
architecture, by section 5 below. The README now says so.

## 5. The changed central claim

Before, the implicit claim was that a boundary substrate generates spacetime,
gauge fields, matter and gravity. That claim is withdrawn.

The claim now is narrower:

> BPR-6D is a six-dimensional, Lorentz-invariant effective field theory with
> gravity, Spin(10)×U(1)_F, an anomaly-free chiral completion and a
> Green–Schwarz 2-form. Compactified on a flux-carrying sphere, it gives:
> - three chiral 16s forming a triplet of a gauged SU(2) family symmetry;
> - a classically stable Minkowski vacuum, after one tuning of the 6D
>   cosmological constant.

**It is not a theory of everything.**
- 6D gravity and gauge theory are non-renormalizable, with a cutoff of order
  M. 1/r is within a factor of about 2 of M unless g4 is small.
- The cosmological constant is tuned, and the flux number is equivalent to
  that tuning.
- The minimal content has no Yukawa couplings.
- The Higgs sector, Spin(10) breaking and SU(2)_iso breaking are unspecified.
- It makes no quantitative prediction yet.

## 6. Precedent

The architecture is not new. The following are cited, not re-derived:
- Randjbar-Daemi, Salam and Strathdee, Nucl. Phys. B214 (1983) 491: 6D
  Einstein–Maxwell on S² with monopole flux, stability, and chiral fermions
  in isometry multiplets.
- Randjbar-Daemi, Salam and Strathdee, Phys. Lett. B124 (1983) 345: the
  instability of non-abelian monopole backgrounds.
- Salam and Sezgin, Phys. Lett. B147 (1984) 47: the supersymmetric,
  tuning-free version.
- Manton, Nucl. Phys. B158 (1979) 141: gauge–Higgs unification on S².
- Witten, Nucl. Phys. B186 (1981) 412: the chirality obstruction for pure
  Kaluza–Klein theory.
- Green, Schwarz and West, Nucl. Phys. B254 (1985) 327, and Sagnotti (1992):
  6D anomaly cancellation.
- Atiyah and Bott (1983): Yang–Mills over Riemann surfaces.
- Blanco-Pillado, Schwartz-Perlov and Vilenkin (2009), and Carroll, Johnson
  and Randall (2009): the Einstein–Maxwell flux landscape.

What BPR-6D adds is narrow:
- the specific minimal completion;
- the Yukawa selection rule for its flux families;
- the observation that the known landscape leaves its family number
  unselected.

## 7. Next well-posed problems

In priority order:

1. **Global anomalies: settled for Spin × Spin(10) × U(1).**
   - Characteristic-class quantization (round 4) requires the parent charge
     to be a multiple of 3.
   - Ω₇^Spin(B(Spin(10)×U(1))) = 0 (round 5).
   - Still open: the Spin ×_{ℤ₂} Spin(10) global form.
   - The even lattice U needs no quadratic refinement.
2. **Remove the Λ tuning.** Find an anomaly-free 6D (1,0) gauged supergravity
   with Spin(10) ⊂ G whose Salam–Sezgin vacuum gives exactly three chiral 16s,
   or show that none exists in a stated class. This is not a relabelling of
   BPR-6D:
   - hyperini have the opposite chirality to gaugini, so 16₊ and 16₋ cannot
     both be matter;
   - the gravitational anomaly condition n_H − n_V + 29n_T = 273 requires 290
     hypermultiplets for Spin(10)×U(1)_R with n_T = 1;
   - Salam–Sezgin fixes the monopole number at ±1, leaves a flat
     dilaton–radius direction and leaves 4D N=1 unbroken.
3. **Higgs sector.** Find a field content that meets all of the following, or
   prove that none exists in a stated class:
   - it supplies an F-charge −2 internal one-form 10;
   - it is stable at 1/r;
   - it keeps three families;
   - it breaks SU(2)_iso at a high scale.
4. **Quantum radion potential.** In D=6 the one-loop potential is
   log-divergent (heat-kernel coefficient a₃). Dimension-6 counterterms scale
   with r like the Casimir term, so the finite part is scheme-dependent.
   Compute the scheme-independent log coefficient, including the
   Green–Schwarz axion and the Stückelberg-massive U(1)_F. Then determine
   whether a regime rM ≫ 1 keeps the classical minimum.

## 8. Independent review

An independent review read this note against the unification map, the
round-2 and round-3 notes and the README. It confirmed the literature
attributions:
- Collins et al.;
- Hohensee et al. (−5.8×10⁻¹² to 1.2×10⁻¹¹);
- Groot Nibbelink–Pospelov (lowest Lorentz violation at dimension five);
- Pospelov–Shang;
- every citation in section 6.

Its findings are repaired above:
- **Blocker.** The earlier text claimed option 3 gives every section-7 link a
  consistent realization. That is false: families, the Standard Model group,
  the single-Hamiltonian standard and predictions are unmet.
- **Major:**
  - the Planck-sized-sphere statements are now conditional on g4;
  - the 2-form quantization caveat now makes the adoption provisional;
  - the 4D spectrum (Stückelberg U(1)_F, axion) is added, together with the
    local-operator scope of the selection rules;
  - the README legacy claims are reclassified;
  - the supergravity next step is re-posed;
  - the one-loop next step is re-posed as scheme-independent.
- **Minor:**
  - Weinberg–Witten wording;
  - soft-breaking Lorentz violation;
  - the dS window;
  - the flux landscape is credited as known;
  - the Yukawa statement is stated as a selection rule;
  - wording fixes.
