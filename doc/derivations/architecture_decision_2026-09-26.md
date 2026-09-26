# Architecture decision: Lorentz invariance is fundamental (BPR-6D)

2026-09-26. Status: this is a decision, with its reasons and consequences. It
is not a derivation. It resolves the decision point at the end of the
[unification map](unification_map_2026-09-25.md) (section 7). The supporting
calculations are
[flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md)
and [family_symmetry_from_flux_2026-09-26.md](family_symmetry_from_flux_2026-09-26.md).

## 0. The decision point

Round 2 showed a naturalness obstruction
([common_light_cone_2026-09-25.md](common_light_cone_2026-09-25.md)).
Emergent fields on a preferred-frame lattice do not share one light cone
unless something forces them to. Tree-level tuning does not survive loops
(Collins et al. 2004). Collider bounds require speed differences below about
10⁻¹¹ (Hohensee et al. 2009). The program had to adopt one of three options:
1. **One relativistic fixed point.** Every low-energy field is an excitation
   of a single theory whose infrared is Lorentz-invariant.
2. **A protecting symmetry.** For example, supersymmetry forbids
   dimension-≤4 Lorentz violation (Groot Nibbelink–Pospelov 2005), and
   sequestering adds further suppression (Pospelov–Shang 2012).
3. **Lorentz invariance fundamental.** Keep the "boundary" idea, but place it
   inside a Lorentz-invariant spacetime, as a compact internal space.

## 1. Choice and reasons

**Option 3 is chosen.** The reasons are ranked:

1. **Every link in map section 7 then has a known consistent realization.**
   - A common light cone is automatic.
   - A massless spin-2 field with universal coupling is ordinary 6D general
     relativity. Weinberg's soft-graviton theorem then forces universality,
     and Weinberg–Witten does not apply.
   - Chiral matter comes from the anomaly-consistent completion already built
     in round 1.
   - Families come from flux.

   Options 1 and 2 leave both the light cone and gravity open. Neither has a
   first calculation within this repository's reach:
   - Option 1 needs a lattice model with the Standard Model content at one
     Lorentz-invariant fixed point; none is known.
   - Option 2 needs a lattice with exact supersymmetry, which is a notoriously
     hard lattice problem, and it still needs a graviton.
2. **It reuses the program's strongest exact result.** The minimal completion
   16₊(Q=1) ⊕ 16₋(Q=0) plus one Green–Schwarz 2-form, with three chiral 16s
   at flux 3 and vanishing Z16 count, is the matter sector as it stands.
3. **It makes the program a definite effective field theory.** Its claims
   become checkable. Round 3 checked several of them, and some came out
   negative (section 4).

The cost is explicit. **BPR no longer claims to derive spacetime.** The
substrate programme becomes analogue physics and a possible regulator.

## 2. Definition of BPR-6D

- **Spacetime and gravity.** A six-dimensional Lorentzian manifold with
  Einstein gravity, (M⁴/2)R₆ − Λ. The vacuum is M4 × S².
- **Gauge group.** Spin(10) × U(1)_F, plus one Green–Schwarz 2-form B.
- **Matter.** Two 6D Weyl spinors:
  - 16 of Spin(10), with U(1)_F charge 1 and chirality +;
  - 16 with charge 0 and chirality −.

  This is the unique minimal (16-component) local-anomaly completion
  (`bpr/chiral_parent_completion.py`).
- **Background.** U(1)_F monopole flux m=3 on a round S². The S² is the
  "boundary" of the Boundary Phase Resonance name.
- **Not specified.** The Higgs sector, the Spin(10) → Standard Model
  breaking, supersymmetry, and any UV completion.

## 3. Round-3 results

| Question | Answer | Status |
|---|---|---|
| Is M4 × S² with m=3 a vacuum? | Yes, with r=m/(2M²e), if Λ=2M⁸e²/m² | exact, one tuning |
| Is the size stable? | Breathing mode yes, m_ψ=1/r0 exactly; other modes cited (RSS 1983) | exact / cited |
| 4D scales | M_Pl²=πm²/e², 1/r=2g4M_Pl/m; control needs g4<m/(4√π) | exact relations |
| Is m=3 selected? | No: fixing Λ makes one flux flat, lower flux AdS, and above (2/√3)m no vacuum | exact, negative |
| Family symmetry | Three families = one SU(2) triplet of the sphere's isometry | exact |
| Yukawas | None in minimal content: 6D chirality + SU(2) leave only an internal-vector J=2 10 | exact, negative |
| SO(12) gauge–Higgs fix | Even families (2k) and a tachyonic J=2 level | exact / index-checked |
| Masses | Real J=2 vev ⇒ m₃=m₁+m₂ (excluded); complex vevs reach any spectrum | exact, no prediction |

## 4. What survives and what is reclassified

| Earlier piece | New role |
|---|---|
| Ring and cubic substrate theorems, condensate regime, acoustic metric | Exact mathematics about lattice Bose systems. Analogue models, not the origin of spacetime. |
| Link-boson emergent U(1) | An analogue of emergent gauge fields. Not the source of Standard Model gauge fields, which are fundamental in BPR-6D. |
| Common-light-cone obstruction | Still true for its substrate class. It is resolved by assumption in BPR-6D, not solved. |
| R+R² induced gravity with calibrated G | Superseded as the gravity sector by 6D Einstein gravity. M_Pl²=πm²/e² relates G to the U(1)_F coupling but does not predict it. |
| Path B (dihedral 2D gauge) | Separate and unconnected. Its falsified predictions stay withdrawn. |
| Chiral parent + completion | Adopted as the matter sector. |
| Flavor formulas | Unconnected phenomenology. BPR-6D derives no mass or mixing relation (family note, section 4). |
| Legacy Casimir and "boundary resonance" laboratory claims | Not supported by BPR-6D. The only boundary is a sphere within one or two orders of magnitude of the Planck length. |
| Fine-structure constant | Still calibrated. |

## 5. The changed central claim

Before, the implicit claim was that a boundary substrate generates spacetime,
gauge fields, matter and gravity. That claim is withdrawn.

The claim now is narrower:

> BPR-6D is a six-dimensional, Lorentz-invariant effective field theory with
> gravity, Spin(10)×U(1)_F, the minimal anomaly-free chiral completion and a
> Green–Schwarz 2-form. Compactified on a flux-carrying sphere, it gives:
> - three chiral 16s forming a triplet of a gauged SU(2) family symmetry;
> - a Minkowski vacuum with a stable breathing mode, after one tuning of the
>   6D cosmological constant.

**It is not a theory of everything.**
- 6D gravity and gauge theory are non-renormalizable, with a cutoff of order
  M.
- The cosmological constant is tuned, and the flux number is equivalent to
  that tuning.
- The Yukawa sector is absent at the minimal level.
- The Higgs sector, Spin(10) breaking and SU(2)_iso breaking are unspecified.
- No prediction has been tested.

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
- the demonstration that its Yukawa sector is empty at the minimal level;
- the demonstration that its flux number is not selected.

## 7. Next well-posed problems

1. **Remove the Λ tuning.** Embed BPR-6D in Salam–Sezgin-type gauged
   supergravity, with U(1)_F → U(1)_R. Then redo anomaly cancellation (the
   gravitino and gaugini become charged) and the family count (the flux is
   fixed by supersymmetry).
2. **Higgs sector.** Find a field content that meets all of the following, or
   prove that none exists in a stated class:
   - it supplies an F-charge −2 internal one-form 10;
   - it is stable at 1/r;
   - it keeps three families;
   - it breaks SU(2)_iso at a high scale.
3. **Quantum radion potential.** Compute the one-loop Casimir energy on the
   Planck-sized sphere, including the Green–Schwarz axion and the
   Stückelberg-massive U(1)_F.
4. **Global anomalies.** Check Ω₇ bordism and the 2-form quantization left
   open by the completion note.
