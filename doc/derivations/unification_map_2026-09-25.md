# Unification map: where the pieces connect and where they do not

2026-09-25. This is a whole-repository review of what a theory of everything
would need and what the repository actually supplies for each link. The
machine-readable version is [`unification_map_2026-09-25.json`](unification_map_2026-09-25.json),
and `tests/test_unification_map.py` checks its integrity. It is a dependency
map, not a completion score. **BPR is not a completed theory of everything, and
this review does not make it one.** It records the new links from three rounds of
work (sections 2–3, 3b and 3c) and names the smallest well-posed problems that stand
between the current state and a connected theory (section 5). Round 3
(2026-09-26) resolved the section 7 decision point by making Lorentz invariance
fundamental; see [architecture_decision_2026-09-26.md](architecture_decision_2026-09-26.md).

## 1. The honest one-paragraph answer

Everything does not yet tie together. The repository contains four largely
separate islands:
1. **Bosonic substrates.** A 1D ring and a supplied 3D cubic lattice, with many
   exact finite-system theorems.
2. **A supplied gauge sector.** A dihedral finite group in 2D; its 1.0 particle
   predictions were falsified by a sealed glueball benchmark.
3. **A supplied six-dimensional chiral matter parent.** Until today it was
   excluded by an irreducible anomaly.
4. **Supplied gravity.** An R+R² action with calibrated G.

Flavor formulas sit on top as phenomenology, with fitted inputs and n_gen=3 as
an input. No derived map connects the substrate to the gauge, matter or gravity
islands. Nor is there an observation that distinguishes the substrate from
conventional Bose physics. This session adds:
- a controlled bridge from the 3D substrate to a condensed vacuum with a
  Lorentz-invariant (acoustic) low-energy sector;
- an anomaly-consistent completion of the matter parent.

Neither bridge crosses between islands.

A second round (section 3b) builds the first bridge from a bosonic substrate
to a gauge field: an emergent compact U(1), via a proposed link-boson
amendment. It then shows that emergent fields from this class of substrate
generically do not share one light cone. That is now the program's sharpest
obstruction.

A third round (section 3c) takes the decision that obstruction forced. It
makes six-dimensional Lorentz invariance and gravity fundamental ("BPR-6D"),
with the lattice substrate reclassified as an analogue. The matter sector
comes from the round-1 completion on a flux-carrying sphere. This connects
the gravity, matter and flavor islands inside one effective theory. It does so
by assumption rather than derivation, and with two new negative results:
- the flux number, and hence three families, is not selected;
- the minimal theory has no Yukawa couplings.

## 2. New link A: substrate → vacuum → acoustic spacetime

[`cubic_condensate_regime_2026-09-25.md`](cubic_condensate_regime_2026-09-25.md)
works on the unchanged cubic lattice (C>0, g>=0). It proves:
- **Exact sector vacuum.** Every fixed-N ground state is unique, positive and
  invariant under all graph automorphisms.
- **Controlled condensation.** In the mean-field regime at fixed lattice size,
  depletion is bounded uniformly in N, and the low spectrum converges to
  Bogoliubov. The proof is self-contained.
- **Lorentz window.** The phonon's deviation from ω=c_s|q| is bounded exactly by
  max(a²/12, ξ²)|q|². This realizes the relativity gate's conditional acoustic
  Lagrangian, with A and B derived rather than assumed.
- **Acoustic metric.** For uniform flows, the long-wave cone is the null cone of
  an acoustic metric. It is Lorentzian iff the flow is dynamically stable, and
  it has no ergoregion iff the flow is long-wave Landau stable.

This is the first place in the repository where a microscopic model yields an
interacting vacuum with a Lorentz-invariant low-energy excitation under
controlled approximations. Its limits are just as concrete:
- one scalar species, so "common speed" is trivial;
- a preferred frame;
- anisotropic and superluminal corrections at order q²;
- no tensor dynamics, no photon, no fermions;
- no thermodynamic limit.

## 3. New link B: an anomaly-consistent matter target

[`chiral_parent_completion_2026-09-25.md`](chiral_parent_completion_2026-09-25.md)
shows that the smallest local completion is one added opposite-chirality
Spin(10) spinor with zero U(1)_F charge. It is the unique 16-component
completion, and nothing smaller works:

    16₊(Q=1) ⊕ 16₋(Q=0),    I8 = (1/3) X² (3S2 + 2X² − p1),

which one Green–Schwarz 2-form cancels. At flux 3 the massless four-dimensional
content is exactly three chiral 16's. These pass the Spin(10)³, Witten SU(2)
and Z16 checks. The remaining U(1)_F anomalies are all proportional to X4 and
are cancelled by the descended axion, which makes U(1)_F massive.

An independent review caught that the first version had excluded neutral
spinors without justification and so reported a larger 36-component
completion. That result is kept as a narrower-class classification.

Open problems: the added spinor and 2-form are inputs, Ω7 bordism and 2-form
quantization are unchecked, flux 3 is chosen, and no substrate realization
exists.

## 3b. Second round: an emergent gauge field, and the light-cone obstruction

**Emergent U(1).**
[`emergent_gauge_link_bosons_2026-09-25.md`](emergent_gauge_link_bosons_2026-09-25.md)
adds a proposed amendment: hard-core bosons on lattice links, with an energy
cost U per unit squared deviation of each vertex charge. Equivalently, these
are ordinary nearest-neighbour-repulsive bosons on the line graph. Results:
- exact second- and third-order effective theories: a compact U(1) quantum link
  model with emergent Gauss law, with K=2t²/U on plaquettes, δK and hexagon
  terms at order t³/U², and constant diagonal terms;
- checks against exact diagonalization on four clusters, and against an
  independent dense oracle;
- on the diamond lattice this is exactly the pyrochlore hard-core boson model.
  Quantum Monte Carlo in the literature reports its Coulomb phase with an
  emergent photon at large U/t;
- on the cubic lattice the photon phase is undetermined.

The route is U(1) only, the charges are bosonic, and the photon relies on
published numerics.

**Light-cone obstruction.**
[`common_light_cone_2026-09-25.md`](common_light_cone_2026-09-25.md) shows:
- at Bogoliubov level, density-coupled condensate species never share a light
  cone, and decoupled species share one only under an unprotected tuning;
  SU(2) symmetry even turns one branch quadratic;
- within one species of link bosons, the photon and the superfluid mode occupy
  different phases; photons and phonons from different bosons have unrelated
  speeds;
- the emergent charges outrun their own photon by a factor of order (U/t)²;
- literature radiative and renormalization-group arguments (Collins et al.
  2004; Anber–Donoghue 2011) say tree-level tuning does not survive, and
  weak-coupling convergence of speeds is only logarithmic.

This is a naturalness obstruction, not a no-go. Emergent Lorentz invariance for
many fields from a preferred-frame lattice therefore needs a new principle. Candidates are one relativistic fixed point
for all fields, or a symmetry forbidding dimension-4 Lorentz violation. The
alternative is to give up deriving spacetime from the substrate.

## 3c. Third round: Lorentz invariance fundamental (BPR-6D)

**Decision.** [architecture_decision_2026-09-26.md](architecture_decision_2026-09-26.md)
chooses option 3 of section 7 over a single relativistic fixed point or a
protecting symmetry. The definition of BPR-6D:
- 6D Einstein gravity;
- Spin(10)×U(1)_F;
- 16₊(Q=1) ⊕ 16₋(Q=0) plus a Green–Schwarz 2-form;
- M4 × S² with flux 3.

The common light cone and a universally coupled graviton then hold by
assumption. The substrate results remain true as lattice mathematics.

**Flux vacuum.** [flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md)
derives the Randjbar-Daemi–Salam–Strathdee vacuum:
- r = m/(2M²e), which requires the tuning Λ = 2M⁸e²/m²;
- a stable breathing mode with mass exactly 1/r0;
- M_Pl² = πm²/e² and 1/r = 2g4M_Pl/m.

The sphere radius is r = (m/2g4)ℓ_P with g4 unknown. It is near-Planckian
unless g4 is tiny, and classical control needs g4 below an O(1),
convention-dependent bound. The adoption is provisional, because the minimal
completion's Green–Schwarz couplings fail a naive Dirac-quantization test.

**Negative result: flux selection.** At fixed Λ:
- only one flux sector is flat;
- lower fluxes are anti-de Sitter;
- fluxes above (2/√3)m₀ have no vacuum.

m=3 is therefore the Λ tuning, not a prediction. This answers critical-path
item 6 at the classical level.

**Family symmetry.** [family_symmetry_from_flux_2026-09-26.md](family_symmetry_from_flux_2026-09-26.md)
shows that the three families form one triplet of the sphere's gauged SU(2)
isometry.

**Negative result: Yukawas.** The minimal content has no field in 16⊗16, so
it has no Yukawa at all. For extensions, an all-orders selection rule
(perturbative in U(1)_F) leaves only an F-charge −2 internal one-form 10 or
126 in J=2; scalars and the 120 never couple. The SO(12) gauge–Higgs source
of that field gives paired, even families and a vacuum instability. Even with
the channel, complex J=2 vevs reach every mass spectrum, so no mass ratio is
predicted.

## 3d. Fourth round: Green–Schwarz quantization and a multiple-of-three family number

[green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md)
tests the minimal completion against Dirac quantization of its Green–Schwarz
couplings. It works at the characteristic-class level, with integral
generators λ_V = S2/2, x² and λ_T = p₁/2.

**Result.** For any integral lattice of 2-form charges, quantization holds
iff the parent's U(1)_F charge q is a multiple of 3 in units of the smallest
charge. This holds for the minimal completion in class 𝒞, and the U(1)
normalization is anchored to Park–Taylor. At q=3 one non-chiral 2-form suffices:
- Y_e = 6x²;
- Y_g = 3λ_V + 9x² − λ_T.

The odd lattice I₁,₁ is obstructed.

**Consequences.**
- States with one third of the matter's charge must exist. In BPR-6D as
  defined they are massive. Morrison–Taylor's F-theory result suggests that
  realizations would normally make them massless.
- By the index theorem, n_gen = q|m| is a **multiple of three**. Three needs
  q=3 and |m|=1, both minimal choices.
- With Λ tuned for it, the three-family vacuum is the only flux sector with a
  compactified minimum.

This is the program's first structural constraint on the family number. It
moves family_count from stipulated to conditional. Why three rather than six
remains a minimality or tuning statement, and torsion and Ω₇ anomalies are
open.

## 3e. Fifth round: no global anomalies

[global_anomalies_2026-09-26.md](global_anomalies_2026-09-26.md) computes
Ω₇^Spin(B(Spin(10)×U(1))) = 0. The route:
- below degree 8, Spin bordism equals ko at the prime 2
  (Anderson–Brown–Peterson), and there is no odd part;
- the Adams E₂ page, Ext over A(1), is computed by an exact minimal
  resolution. It is empty in stem 7 for every summand of the stably split
  (BSpin(10) × ℂP^∞)₊.

The same engine reproduces known results:
- the ko chart;
- Witten's SU(2) anomaly;
- Lee–Tachikawa's Ω₇ = 0 for SU(2) and SU(3);
- the absence of 4D anomalies for Spin(10).

With the round-4 couplings, BPR-6D therefore has no global anomaly for the
Spin × Spin(10) × U(1) structure.

For the twisted Spin ×_{ℤ₂} Spin(10) form, the independent review found a
ℤ/2 in Ω₇. Its anomaly is shown trivial by a reduction argument:
- the η product formula on S² × Wu;
- the branching 16 = (4,4);
- the order bound |Ω₅^{Spin×ℤ₂Spin(5)}| ≤ 4.

The anomalies are therefore cancelled locally, with quantization, and
globally, for both global forms.

## 3f. Rounds 6–7: string embedding and Spin(10) breaking

**String embedding.** See
[string_embedding_2026-09-26.md](string_embedding_2026-09-26.md). The
Morrison–Taylor massless-charge conjecture is supersymmetric. BPR-6D meets its
premise (−a·b̃ = 24) with charge gcd 3, but one massless vector-like charge-1
pair removes the tension without changing anomalies or families.

Supersymmetric SO(10) × U(1) analogues on T=0 behave differently:
- they force n₁₀ = n₁₆ + 2 hypermultiplets in the 10;
- by an arithmetic lemma, they always allow charge gcd 1.

So the multiple-of-three family number is intrinsically non-supersymmetric.
String-embeddability and that result pull in opposite directions.

**Spin(10) breaking.** See [gut_breaking_2026-09-26.md](gut_breaking_2026-09-26.md).
- Spin(10) flux is chirality-neutral: three net families survive.
- But the Green–Schwarz coupling makes the flux U(1) massive, so no flux gives
  SU(3) × SU(2) with a massless hypercharge. This is the sphere version of the
  F-theory hypercharge-flux problem.
- S²/(ℤ₂ × ℤ₂) orbifolds give the right gauge group but non-uniform family
  numbers.

Spin(10) → Standard Model therefore needs a supplied Higgs sector.

## 3g. Rounds 8–9: Yukawa mechanisms and the cosmological constant

**Yukawa mechanisms.** See
[yukawa_mechanisms_2026-09-26.md](yukawa_mechanisms_2026-09-26.md).
- A single point-brane Higgs gives rank 1 or a degenerate pair, never three
  distinct masses, whatever its J_z charge.
- Two branes give generic spectra with no hierarchy.
- A bulk internal-vector 10 or 126 works only with a Higgs-mass tuning of
  about 10⁻³⁰ and a near-null orientation.

BPR-6D therefore has no natural Yukawa mechanism.

**Cosmological constant.** See
[cosmological_constant_2026-09-26.md](cosmological_constant_2026-09-26.md).
Classical brane self-tuning fails. Flux quantization on the reduced area makes
the flat brane tensions discrete, and dH²/dT ≠ 0 at the flat point. The Λ
tuning (about 10⁻¹²⁰) stands.

## 3h. Round 10: predictions

See [predictions_2026-09-26.md](predictions_2026-09-26.md).
- **A QCD axion.** The Green–Schwarz 2-form leaves one physical axion, after
  U(1)_F eats one combination, and it always couples to QCD through λ_V.
  BPR-6D therefore predicts a QCD axion with f_a near the compactification
  scale: m_a ≈ 10⁻¹¹–10⁻⁸ eV, with dark matter requiring a small
  misalignment angle. This is generic to Green–Schwarz models.
- **The family number.** n_gen ∈ 3ℤ is a postdiction that excludes a fourth
  family.
- **Everything else** is at 10¹⁶–10¹⁸ GeV or not predicted.

## 4. The dependency graph

```mermaid
graph TD
  classDef stip fill:#eee,stroke:#888
  classDef exact fill:#cfe8cf,stroke:#2e7d32
  classDef ctrl fill:#e3f0d8,stroke:#558b2f
  classDef open fill:#fff3cd,stroke:#b8860b
  classDef obst fill:#f8d7da,stroke:#a33
  classDef fit fill:#e0e0f8,stroke:#55a

  ring[1D Bose ring]:::stip --> locobs[3D-in-ring encoding obstructed]:::obst
  ring --> gauss[short-interval Gauss obstructed]:::obst
  locobs --> cubic[3D cubic lattice]:::stip
  cubic --> tuned[tuned continuum + hard exclusion]:::ctrl
  tuned --> eqgate[equilibrium transfer gate]:::open
  cubic --> vac[sector vacuum: unique, symmetric]:::exact
  vac --> bec[mean-field BEC + Bogoliubov]:::ctrl
  bec --> thermo[thermodynamic condensation]:::open
  cubic --> lit[literal particle relativity]:::obst
  bec --> window[acoustic Lorentz window]:::ctrl
  window --> metric[acoustic metric, stability = signature]:::ctrl
  metric --> speed[common cone for all species: obstructed in class]:::obst
  metric --> gdyn[metric dynamics / graviton]:::open
  grav[R+R^2 action]:::stip --> gdyn
  grav --> G[G, Planck length]:::stip
  link[link-boson amendment]:::stip --> egauge[emergent U(1): conditional]:::ctrl
  gauss --> egauge
  egauge --> nonab[non-Abelian SU3 x SU2]:::open
  egauge --> speed
  pathb[dihedral gauge proposal]:::stip --> glue[glueball benchmark: withdrawn]:::obst
  parent[6D Spin10 parent]:::obst --> comp[minimal anomaly completion: + neutral 16]:::exact
  comp --> ngen[n_gen in 3Z; 3 = one flux quantum]:::ctrl
  comp --> latchi[lattice chiral fermions]:::open
  cubic --> latchi
  ngen --> flavor[flavor formulas]:::fit
  window --> lv[map phonon LV to real particles]:::open
  speed --> lv
  speed --> arch[BPR-6D: Lorentz invariance fundamental]:::stip
  comp --> arch
  arch --> g6[6D Einstein gravity]:::stip
  g6 --> fvac[M4 x S2 flux vacuum, radion stable]:::exact
  comp --> fvac
  fvac --> fsel[flux selects m=3: obstructed]:::obst
  fsel --> ngen
  comp --> gsq[GS Dirac quantization: parent charge in 3Z]:::exact
  gsq --> ngen
  gsq --> glob[global anomalies: Omega_7 = 0]:::exact
  fvac --> fam[families = SU2 isometry triplet]:::exact
  fam --> yuk[Yukawa sector: none in minimal content]:::obst
```

Colors: green = proved (exact or controlled), yellow = open, red = obstructed
or withdrawn, grey = supplied, blue = fit. The JSON gives sources and
limitations for every node.

## 5. Critical path: the smallest problems that would connect the islands

These are ordered by how many open nodes each would unblock. Each has a
concrete first calculation; none is authorized as physics by this map alone.

1. **Common limiting speed (spacetime), analysed in round 2 and decided in
   round 3.** Round 3 made Lorentz invariance fundamental
   ([architecture_decision_2026-09-26.md](architecture_decision_2026-09-26.md)),
   so this item is bypassed for BPR-6D and remains open only for the substrate
   as an analogue. The analysis in
   [common_light_cone_2026-09-25.md](common_light_cone_2026-09-25.md) confirms
   and extends the reasoning below. Add a second boson species to the
   cubic lattice: a two-component Bose–Hubbard model with g11, g22, g12. At
   Bogoliubov level, with equal hopping, there are two phonon branches with
   c_i²=2κλ_i(G), where G=[[g11ν1, g12√(ν1ν2)],[g12√(ν1ν2), g22ν2]]. This is
   the standard two-component result, recorded here rather than rederived. A
   single cone needs G∝I, i.e. g12=0 and g11ν1=g22ν2, a codimension-two tuning.
   No symmetry of the model protects it:
   - the inter-species density coupling n1n2 is allowed by U(1)×U(1)×Z2;
   - making the couplings SU(2)-symmetric (g11=g22=g12) makes the spin mode
     quadratic, a type-B Goldstone mode, rather than giving it the same cone.

   **In this substrate class, a universal light cone is therefore not natural
   at leading order.** This is a structural reason for caution about "emergent
   Lorentz invariance" claims. It matches the bimetricity known from analogue
   gravity. Next calculation: control the two-component mean-field limit as in
   Theorem 4 of the condensate note. Then find a mechanism that forces all
   low-energy fields to share one cone, for example all of them being
   excitations of a single order parameter, or a Chadha–Nielsen-type
   renormalization-group flow in a gauge theory. If no such mechanism exists
   for a substrate, that substrate cannot underlie relativistic multi-field
   physics.
2. **Emergent gauge field (gauge), built in round 2 for U(1).** See
   [emergent_gauge_link_bosons_2026-09-25.md](emergent_gauge_link_bosons_2026-09-25.md).
   Still open: the cubic-lattice photon phase and any non-Abelian group. The ring theorem excludes exact bounded
   short-interval Gauss operators on unrestricted Fock space. The known escape
   is energetic: bosons on links with a large vertex-charging term, whose
   low-energy sector is a compact U(1) lattice gauge theory with a photon (the
   quantum-ice mechanism). First calculation: the degenerate-perturbation
   ring-exchange coefficient for link bosons on the cubic lattice, and the
   stability of its Coulomb phase. This changes the substrate's degrees of
   freedom and requires explicit approval as a new model.
3. **Chiral matter on the substrate (matter).** The completion supplies
   three chiral 16's, 16 Weyl fermions per generation, with Z16 count zero.
   Any lattice realization must evade
   Nielsen–Ninomiya doubling. Possible routes are domain walls, or gapping a
   mirror sector by symmetric mass generation, for which sixteen Weyl fermions
   per generation is the content usually required. First calculation: the
   free-fermion doubling count for the proposed lattice embedding, then the
   mirror-sector interaction needed to gap it.
4. **Metric dynamics (gravity).** The acoustic metric obeys hydrodynamics, not
   Einstein's equations. Any emergent massless spin-2 state must evade
   Weinberg–Witten, typically through non-fundamental Lorentz symmetry or
   diffeomorphism as a gauge redundancy. First calculation: the induced
   (Sakharov) Einstein–Hilbert coefficient from the phonon loop on a slowly
   varying acoustic background. Compare it with the hydrodynamic back-reaction
   to show which dominates.
5. **Thermodynamic vacuum (vacuum).** Extend condensation beyond the mean-field
   regime at fixed size. Rigorous results exist in the literature only for
   hard-core bosons at half filling. First calculation: state precisely which
   reflection-positivity / infrared-bound results transfer to the g→∞ member of
   the family.
6. **Flux and family number (matter/flavor).** q=3 is chosen, and the flux
   energy alone favors q=0 at fixed radius (constructive extension note,
   chiral section). First calculation:
   the moduli potential including the Green–Schwarz term and Casimir energies.
   It is possible that no minimum selects q=3; that outcome should be recorded
   if found. **Round 3 found it at the classical level.** In the BPR-6D flux
   vacuum at fixed Λ, one flux sector is flat, lower ones are AdS and higher
   ones have no vacuum. Choosing q=3 is the Λ tuning
   ([flux_compactification_2026-09-26.md](flux_compactification_2026-09-26.md),
   Theorem 5). Quantum (Casimir) corrections and a supersymmetric
   (Salam–Sezgin) embedding remain open.
   **Round 4.** Dirac quantization of the Green–Schwarz couplings forces
   n_gen ∈ 3ℤ for the minimal completion
   ([green_schwarz_quantization_2026-09-26.md](green_schwarz_quantization_2026-09-26.md)).
   Three is one flux quantum, and at its tuned Λ it is the only compactified
   vacuum.
7. **Yukawa sector (flavor), added in round 3.** Minimal BPR-6D has no
   zero-mode Yukawa
   ([family_symmetry_from_flux_2026-09-26.md](family_symmetry_from_flux_2026-09-26.md)).
   Find a Higgs sector that meets all of the following, or prove that none
   exists in a stated class:
   - it supplies an F-charge −2 internal one-form 10;
   - it is stable at 1/r;
   - it keeps three families;
   - it breaks the SU(2) family symmetry at a high scale.

## 6. Recorded legacy debt found in this review

Found by read-only surveys and not repaired here. These are recorded so that
they are not mistaken for results.

- Gravity/cosmology:
  - `bpr/black_hole.py` still uses the pre-audit spacing a=l_P·sqrt(p/48π²),
    and `tests/test_extended_predictions.py` asserts it.
  - `bpr/bridges/cosmology_gravity.py` (and `bridges/substrate_quantum.py`)
    use the superseded a=l_P/√p.
  - `dark_energy_from_boundary_action` describes Ω_Λ≈0.69 but returns ~1e-104.
  - `planck_to_hubble` reads a nonexistent attribute, swallows the exception
    and returns the input H0.
  - `cosmology.py` uses slow-roll ε=3/(2N²) and a hard-coded A_s;
    `gravity_consistency` uses 3/(4N²).
  - `emergent_spacetime.py` hard-codes 3+1 dimensions and signature.
  - Three mutually inconsistent gravitational-wave dispersion statements
    coexist.
- Particle side:
  - `DERIVATION_ROADMAP.md` and parts of `VALIDATION_STATUS.md` still carry
    pre-audit "DERIVED" labels.
  - m_τ is described four inconsistent ways.
  - `koide_predicted` returns 2/3 although the model gives 0.6653.
  - `strong_cp_theta` returns 0 for every odd prime.
  - `clifford_bpr.e8_to_sm_decomposition` keeps a withdrawn decomposition.
  - `topological_anomaly_inflow` uses an incorrect b₃ and a family cap.
  - The example hypercharges in the `boundary_action.anomaly_inflow` docstring
    have nonzero Tr Q³.
  - Path B documents still describe the substrate as "deconfined", whereas the
    code now reports an unknown phase.
- Tests: `tests/test_substrate_triplet_projection.py::test_mixed_scale_cubic_underflow_is_not_reported_as_closure`
  fails on x86-64 Linux. There `np.longdouble` is 80-bit extended, so the
  expected underflow does not occur; the test assumes a platform where
  longdouble equals double. The full suite on this platform gives 7,388
  passed, 4 skipped, 1 failed, before the new modules.

## 7. What would count as "everything ties together"

At minimum there must be one microscopic Hamiltonian whose controlled
low-energy limit contains, from the same degrees of freedom:
- a single light cone shared by all propagating fields;
- gauge fields with the Standard Model group;
- anomaly-free chiral matter with three families not put in by hand;
- a dynamical metric with universal coupling.

It must also yield at least one prediction that differs from conventional
physics and survives a preregistered test. Today:
- the first is obstructed within the current substrate class;
- the fourth is open;
- the third has a consistent target but no substrate realization;
- the second has a conditional substrate route for U(1) only;
- the prediction requirement is unmet.

The critical path in section 5 is the most direct honest route. Its first
item is now a decision point, not a calculation. Either find a principle that
gives every emergent field the same light cone, or stop deriving spacetime
from a preferred-frame lattice.

**Round-3 update.** The decision was taken: BPR-6D stops deriving spacetime
from a preferred-frame lattice. Measured against the list above, BPR-6D stands
as follows:
- The single light cone holds by assumption.
- Gauge fields are supplied: Spin(10) is not yet broken to the Standard
  Model group.
- Chiral matter with three families is anomaly-free. The family number is
  equivalent to the Λ tuning, not derived.
- A dynamical metric with universal coupling is supplied by 6D gravity.
- There are no Yukawa couplings at the minimal level.
- No prediction has been tested.

Round 4 refines the chiral-matter item: Dirac quantization makes the family
number a multiple of three, with three the minimal case. Round 5 shows there
is no global anomaly for either global form. For the twisted form this rests
on a reduction argument.

Rounds 6–10 settle the rest of the list:
- **String embedding:** a fork. Supersymmetric analogues lose 3 | n_gen.
- **Spin(10) → Standard Model:** geometric breaking is obstructed, so a Higgs
  sector must be supplied.
- **Yukawas:** no natural mechanism.
- **Cosmological constant:** tuned, with no classical self-tuning.
- **Predictions:** a Green–Schwarz QCD axion and n_gen ∈ 3ℤ.

Every item is supplied or assumed rather than derived from one microscopic
Hamiltonian. The four requirements are now consistent with each other, but
BPR-6D does not meet the "one microscopic Hamiltonian" standard of this
section.
