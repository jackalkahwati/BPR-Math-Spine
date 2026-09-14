# Relativity compatibility gate for the conditional cubic Bose model

2026-09-14. Status: scoped mathematical review cleared with no blockers;
independent scope review found no overclaim; preservation audit cleared. No simulation, fit, physical-state selection or model change.
The previously paused autonomous loop is not restarted by this task.

## 1. Which claim is being tested?

Retain the constructed contact/hard-exclusion continuum model and its supplied
parameters. Use its existing spatial translations and external time as physical
momentum and energy for the LITERAL particle identification, with hbar=1.
Allow a declared constant one-particle energy reference E0 and ordinary changes
of units, but no nonlinear reassignment of momentum or state-dependent clock.
Any different physical dictionary requires its own analysis.

Four claims must not be conflated:
1. The exact existing N=1 sector is a finite-c relativistic particle sector
   under these same time/translation generators.
2. Its low-momentum dispersion approximates a massive relativistic branch.
3. A many-body state supports a low-energy collective acoustic mode.
4. A declared interacting physical sector has relativistic correlations,
   transformations and causal properties with a common limiting speed.

The tests below can exclude the first specified identification while leaving
the second compatible and the third/fourth unestablished. This is not a theorem
against every emergent encoding or a relativistic ultraviolet theory with a
nonrelativistic low-energy limit. No link to observed particles is supplied merely
by naming the microscopic variables bosonic.

## 2. Exact literal one-particle identification fails

Both pair contact and triple exclusion vanish as operators on complete N=1.
The exact continuum generator is h1=-kappa Delta, kappa>0, with momentum spectrum

    E(p)=E0+kappa|p|².                                  (1)

For a particle of mass m>=0 in an exact relativistic identification with finite
c>0, the translation generators would need the mass-shell relation

    E(p)²=m²c⁴+c²|p|².                                  (2)

On an open radial momentum interval put u=|p|². Substitution gives the polynomial

    kappa²u²+(2E0 kappa-c²)u+(E0²-m²c⁴)=0.

If this holds on an interval its coefficients all vanish. Its leading coefficient
kappa² is nonzero, a contradiction. Thus no constant E0, finite c>0 and m>=0
make the specified exact branch satisfy(2) on an open momentum band. A necessary
relativistic condition has failed, so a boost representation with that same
single-particle mass shell cannot repair this literal identification. This does
not classify every possible representation mixing other sectors or observables.

At the empty-vacuum threshold, E-E(0)=kappa|p|² has dynamic exponent2, whereas
a massless relativistic branch has c|p| and exponent1. The literal group velocity
is grad_p E=2kappa p, unbounded over unrestricted continuum momentum. These
facts are consistent with the exact mass-shell mismatch, but they are not proofs
of operational faster-than-light signaling. For that, one must specify detector
observables, interventions, state preparation and a propagation/measurement map.
Nonzero Schrödinger wavefunction tails alone are insufficient. The constrained
sector direct sum has not automatically supplied a canonical local field algebra
or relativistic microcausality.

The E0 in(1) is a fixed one-sector comparison convention. The earlier auxiliary
number shift is not a derived rest mass, chemical potential or vacuum rule;
across number superpositions such shifts are not generally one common phase.

## 3. A massive relativistic approximation is not excluded

For any comparison c>0 choose m=1/(2kappa)>0 and E0=mc² in the stated units.
This matches the constant and quadratic terms of a positive massive branch.
Let x=|p|²/(m²c²). The exact discrepancy is

    Delta E=mc²+|p|²/(2m)-sqrt(m²c⁴+c²|p|²)
       =mc²[1+x/2-sqrt(1+x)]
       =|p|⁴/[2m³c²(1+sqrt(1+x))²].                   (3)

The last equality follows by setting y=sqrt(1+x):
1+x/2-y=(y-1)²/2=x²/[2(1+y)²]. Thus for every real p,

    0<=Delta E<=|p|⁴/(8m³c²).                           (4)

For a specified finite window |p|<=p_max, the same uniform envelope has p_max⁴
in the numerator. In the nonrelativistic regime |p|/(mc)<<1 the leading error
is |p|⁴/(8m³c²), and the ratio to the retained kinetic energy is bounded by
|p|²/(4m²c²) (with zero-momentum value understood by continuity). No numerical
window or tolerance is selected here. At fixed m and bounded momenta, the
rest-energy-subtracted massive branch tends to p²/(2m) as c tends to infinity.
The absolute rest energy itself does not have that finite limit.

Consequently a quadratic dispersion near rest is not evidence against a massive
relativistic approximation. Conversely matching it does not determine c: c can
be any supplied comparison value, with a corresponding comparison E0. It does
not establish antiparticles, physical mass calibration, boost symmetry or a
relativistic completion of the existing exact branch. Conditions(1)–(4) are
analytic comparisons, not experimental discrimination between theories.

## 4. One conditional collective mechanism: density/phase acoustics

The only collective mechanism considered here is a phase-density sound mode.
It is not an adopted state, new Hamiltonian or computed response of the current
attractive contact/hard-exclusion model.

The following prerequisites are NOT established by the completed sector/direct-
sum convergence proofs:
- a thermodynamic homogeneous, isotropic, stationary background at a specified
  density, with the needed phase rigidity;
- a physical density/phase observable and source dictionary, including its
  low-energy symplectic structure;
- stable positive finite compressibility and phase stiffness for that state;
- a stationary-background convention that removes the linear density term without
  silently choosing a physical chemical potential;
- controlled collective poles, residues, coupling to other modes, and derivative
  expansion errors, including any damping.

A normalizable superposition of particle numbers is not such a thermodynamic
state. Bare Bose variables do not by themselves establish canonical creation
fields on the constrained space. Nor does the sign of the renormalized-state
compressibility equal the sign of the diverging bare attractive coupling by
assumption. Setting A=g_a or using e(n)=g_a n²/2 is not justified here.

ONLY CONDITIONALLY, suppose those prerequisites yield real small fluctuations
rho=delta n and theta with effective quadratic Lagrangian density

    L2=-rho partial_t theta-(A/2)rho²-(B/2)|grad theta|²,
    A>0, B>0 finite.                                    (5)

A and B are effective coefficients, not inferred microscopic values. Identifying
A=e''(n0) requires an independently established equation of state, density
normalization and ensemble/source convention. Identifying B=2kappa n0 requires
a proved stiffness/Ward relation for this state and domain; it is not adopted.

Variation in rho gives rho=-partial_t theta/A. Substitution into(5) gives

    L_eff=(partial_t theta)²/(2A)-(B/2)|grad theta|²,
    partial_t² theta-AB Delta theta=0,
    omega²=c_s²|k|², c_s²=AB.                           (6)

Equivalently variation in theta gives partial_t rho+B Delta theta=0, consistent
with(6). The quadratic Hamiltonian A rho²/2+B|grad theta|²/2 is nonnegative under
the stipulated signs. These equations show a stable nondissipative linear sound
mode in this effective model, not existence of the required microscopic phase.

With tau=c_s t, the action is (c_s/(2A)) integral d tau d³x
[(partial_tau theta)²-|grad theta|²]. Rescaling the field by sqrt(c_s/A) puts
this Gaussian action into the unit-speed massless scalar form. This is an acoustic
Lorentz symmetry of one quadratic mode under those assumptions. It neither
removes the background's preferred rest frame from the full system nor constrains
all higher-order interactions and other observable channels to obey the same
symmetry. Choosing units c_s=1 is normalization, not a prediction of physical c.

## 5. Response and controlled-regime requirements

Before assigning the sound mode a physical interpretation, a future derivation
must specify the observable source response, not just a wave-equation eigenvalue.
A schematic inverse kernel for that goal is

    K_eff(omega,k)=Z[(omega+i0)²-c_s²|k|²]+Remainder(omega,k).

This notation asserts no microscopic value or sign of Z, no residue convention,
no analytic expansion order and no damping or remainder bound. The source coupling,
response numerator, positive spectral weight in its stated convention, isolated
pole and error norm must be derived. Near-pole relative errors need a defined
criterion; a uniform denominator estimate alone is not a uniform response bound
on its zeros.

A prospective long-wave regime may require |k|R small and frequencies below
other excitations, but this alone controls neither scattering-length, density,
correlation-length nor damping scales. The state, momentum/frequency/time window,
limit order and comparison norm must be specified before evaluation. No quartic
coefficient, numerical tolerance or favorable window is invented here.

The older classical repulsive-ring result omega²=epsilon(epsilon+2g nbar), with
an acoustic limit for g>0, is not a derivation for the present attractive tuned
quantum hard-domain model. Its variables, background and assumptions differ.
Substituting the current negative bare g_a into that formula is neither a proof
of sound nor a proof of instability of the present thermodynamic state.

Similarly the finite-sector ring commutator tail bound is not an exact light
cone, a state/volume-uniform propagation speed or relativistic microcausality.
Relativity of physical observables requires its own dictionary and controlled
causal/correlation analysis. A linear collective mode alone is insufficient.

## 6. Decision table: success, failure and missing information

| Claim | Success requirement | Legitimate rejection | What does not decide it |
| --- | --- | --- | --- |
| Literal exact particle relativity | Same E,p generators satisfy finite-c shell and compatible boosts on the declared sector | Polynomial contradiction in section2 excludes this specified identification | Matching only the quadratic low-p term |
| Massive low-p comparison | A fixed comparison m,c/window with controlled discrepancy | Failure of a stated error requirement, once its scope is fixed | The mere fact that dispersion is quadratic |
| Collective acoustic candidate | Derived state/variables, positive A,B, source-normalized pole and residue with controlled corrections | Demonstrated instability, absent claimed pole or a controlled persistent violation in that specified regime | Missing state construction or a loose sufficient bound |
| Interacting physical relativistic regime | Declared coupled observable sectors, common limiting speed, compatible boost/Ward/correlation relations and controlled symmetry breaking | Derived incompatible speeds/covariance or causal relations beyond the predeclared error in those sectors | One sound pole, c=1 units or a legacy statement from another model |
| Operational causal claim | Detector/observable and intervention dictionary with a quantitative propagation criterion | A verified violation under those same hypotheses | Nonzero wavefunction tails alone |

The exact literal route is excluded under the fixed dictionary, while a massive
nonrelativistic approximation is compatible at leading order. The sound route
remains UNESTABLISHED, not universally ruled out. Universal/interacting physical
relativity is not supplied by that route's conditional Gaussian calculation.

A future test must preregister the state family and preparation, independent
parameter accounting, physical observables and source normalization, clock and
momentum conventions, limit order, window, norm/error target, competing model
and falsifying outcome. No post-hoc fitting of c or shrinking of a failed window.
For empirical claims, calibration and independent data are additionally required;
none is provided or acquired here. A missing calculation is not a falsification,
and mathematical consistency is not empirical validation.

The next prerequisite for the one chosen collective mechanism is a concrete
proposal to derive a homogeneous-state density/phase response and its stability
for the UNCHANGED model. The background or ensemble would be an explicit supplied
choice until derived. Do not silently select a vacuum, chemical potential or
density, add interactions, or restart the paused loop. No other mechanism is
searched or adopted by this assessment.

## 7. Source register and prohibited transfers

Protected references, relative to doc/derivations/ unless otherwise stated:
- `supplied_cubic_bose_2026-09-13.md:79-147`: exact one-particle reduction and
  nonrelativistic Fourier/continuum generator.
- `cubic_finite_particle_continuum_2026-09-14.md`: exact small-sector identification.
- `cubic_sector_direct_sum_convergence_2026-09-14.md`, sections4–6: number-dependent
  reference phases, auxiliary number shift, no automatic canonical fields,
  thermodynamic state or physical vacuum.
- `substrate_longwave_limit_2026-09-12.md:7-19,31-45`: separate supplied classical
  repulsive-ring acoustic result and its explicit scope restrictions.
- `substrate_quantum_propagation_2026-09-13.md:9-25,37-43`: finite-sector ring
  commutator bounds, explicitly not relativistic microcausality.
- `foundation_decision_2026-09-13.md:33-48`: model/state/operator dictionary,
  common dynamics, physical sectors and falsification obligations.
- `foundation_observation_gate_2026-09-13.md:40-63`: paired ring descriptions
  with identical observation distributions cannot be distinguished by precision.
  This is not automatically a theorem about every new cubic observable.

No legacy light speed inserted by construction counts as derivation of relativity
for the present model. No legacy module, paper, benchmark or claim is repaired,
regraded or recertified here. The assessment's exact-sector algebra and conditional
hydrodynamic implication are new scoped arguments for review, not imported
physical conclusions from those sources.

## 8. Preservation and review status

Baseline692tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-relativity-gate-f1hsi6iv/baseline.json`.
Inventory uses git ls-files --cached --others --exclude-standard; ignored caches,
configuration/build products and .git are excluded. This is the sole new
repository file. No numerical test, scientific import, data acquisition, fitting,
model edit, remote command or publication is part of this assessment.
The autonomous loop remains paused. BPR is not a TOE by completion of this
compatibility assessment.

## 9. Independent review record

Mathematical reviewer a1fa048cc4efd8dd0 cleared the exact-sector obstruction,
massive approximation identity/bounds, conditional acoustic derivation and
stated inference limits with no blocker or required repair. This does not
establish the microscopic collective state or physical relativity.

Separate scope reviewer afb969f37d46ab193 found no scientific-scope overclaim,
without recertifying that mathematics. They confirmed the fixed-dictionary
restriction, unestablished acoustic prerequisites, no signaling shortcut,
scoped falsification criteria and nontransfer of old ring results. Their
next-prerequisite assessment is proposal development for homogeneous-state
response and stability, with state/ensemble/density/source/window choices
explicit, not an automatically authorized construction campaign.

Both reviewers used direct reads only. A stale closing status sentence was
corrected; no mathematical argument, physical hypothesis or constant changed.
Original proof-draft-v1.md and its hash are preserved externally. Independent
auditor a05a6aafc8867a81b confirmed all692baseline files unchanged, exactly one
added note and693current hashes matching final-hashes.json. Assessment sections1–7
are byte-identical to the draft; the note matched pre-audit-note.md. This is
saved-document/scope clearance, not mathematical recertification or proof of
absence of unrecorded execution. The audited final-hashes.json is retained;
closing-hashes.json binds final audit-status edits. No numerical tests or empirical
validation were performed, and no additional model or loop was started.
