# Fixed-density state feasibility: finite boxes, competitors and the phase gate

2026-09-14. Status: scoped mathematical review cleared with no blockers;
separate scope review found no overclaim; preservation audit cleared.
The study establishes finite-volume comparison tools, not a homogeneous phase
or sound response. The autonomous loop remains paused.

## 1. Supplied ensemble, density and questions

Retain the unchanged contact/hard-triple-exclusion continuum family, supplied
R,kappa>0 and fixed finite signed s!=0. Choose ZERO-TEMPERATURE CANONICAL energy
minimization as the study convention. Supply symbolic density rho satisfying

    0<rho R³<1/8.

This is a conservative regime for explicit admissible competitors, not measured
physical density, a sharp packing threshold or an assertion that rho|s|³ is
small. Both signs of s are retained. Neither background density nor an ensemble
choice is a derived physical vacuum or chemical potential.

Use open continuum boxes and the existing threshold-referenced energy. No
joint lattice-spacing/volume/particle-number limit is assumed. Fixed-N continuum
convergence and square-summable sector convergence do not establish canonical
thermodynamics. The questions are separate: finite-box nonemptiness and energy
control, existence of an energy-density limit, existence of a bulk equilibrium
state, its phase character, and only then a collective density response.

## 2. A specified continuum box operator

Let Lambda=(0,L)^3 and O_(N,L)=Omega_N intersect Lambda^N, where Omega_N requires
every labelled triple to have diameter>R. For N=0 retain the vacuum convention.
For N>0 with nonempty O_(N,L), work first on ordered L²(O_(N,L)) and then
restrict to bosons.

Restrict the existing intrinsic compact singular initial domain: each admissible
multiplier has compact support INSIDE O_(N,L), avoids unselected collision
surfaces, and is normal-flat near selected contacts. Retain all contact germs
and their scattering-length condition. Its off-contact action is the same
S_N=-kappa sum_i Delta_(x_i); contact delta sources are not left in the L² action.
This is not the ordinary free Dirichlet Laplacian on O_(N,L).

Zero extension identifies this initial operator as a restriction of the reviewed
whole-space symmetric semibounded operator. Regular compact tests avoiding
contact surfaces are L²-dense in the nonempty open O_(N,L), because those
surfaces have measure zero. Selected singular generators remain included wherever
the geometry permits them. The domain and action are permutation invariant.
The inherited lower bound is

    q_(N,L)[u]>=-B N||u||²,
    B=kappa[1_{s>0}/s²+64pi²/R²].                       (1)

Symmetry, density and(1) yield the specified Friedrichs operator H_(N,L)^D,
then its bosonic restriction. Its closed form is the closure of this restricted
singular class. The map by zero extension is continuous in the shifted energy
norm into the whole-space form domain and is isometric for the inherited
pairing on the initial domain. A form-Cauchy sequence therefore has a whole-
space form limit and an L² limit supported in closure(Lambda)^N; the two limits
agree. This proves the finite-box domain inclusion and retained bound(1).
The converse inclusion of every supported whole-space form vector is not proved
or required here; it would be an additional external-wall density theorem.

For clarity, positive form norms are singular energy norms, not bare H¹ norms.
On the initial box class, and hence its completion, the whole-space localization
identity gives for lambda_N>floor(N/2)E_s, E_s=2kappa*1_{s>0}/s²,

    q_(N,L)[u]+(lambda_N+Wmax_N)||u||²
      =sum_P(a_P[chi_Pu]+lambda_N||chi_Pu||²)
                         +integral(Wmax_N-W_loc,N)|u|²,
    Wmax_N=2kappa floor(N/2)(8pi/R)².

This controls the localized matching form norm, including contact singularities.
Ordinary Rellich compactness cannot simply be applied to the singular wavefunction.
No compact-resolvent or ground-eigenvector attainment theorem is claimed.

Define E(N,L)=inf_{||u||=1}q_(N,L)[u]=inf spectrum H_(N,L)^D. For a nonzero
sector it is finite below by(1), and normalized approximate minimizers from
the intrinsic core exist by form-core density and normalization. Empty allowed
sectors are assigned E=+infinity as variational bookkeeping, not fictitious
ground states. N=0 has energy0. These definitions do not select a finite-volume
or thermodynamic equilibrium wavefunction uniquely.

## 3. A necessary capacity bound

Partition Lambda into half-open cells of side at most R/sqrt(3), shortening
boundary cells as needed. At most ceil(sqrt(3)L/R)^3 cells are needed. Every
cell has Euclidean diameter<=R, so three particle labels inside one cell would
violate exclusion, even if positions coincide. Therefore every allowed
configuration obeys

    N<=2 ceil(sqrt(3)L/R)^3.                            (2)

This is necessary, not sufficient, and not a sharp close-packing density.
It makes explicit why some finite-box sectors can be empty. Boundary conventions
in the cell partition cannot invalidate the diameter upper bound.

## 4. Exact-density separated-orbital competitor

Put d=rho^(-1/3)>2R. For integer m>=1 take L_m=m d, N_m=m³, so N_m/L_m³=rho
exactly. In each d-cell place a centered subcube of side ell=d/4, with center
((j1+1/2)d,(j2+1/2)d,(j3+1/2)d),0<=ji<m. On each subcube choose its normalized
Dirichlet sine ground orbital, extended by zero. Each orbital lies in H¹ and
has kinetic energy3pi²kappa/ell². It need not belong to the ambient smooth
operator core; form-domain membership suffices.

Distinct supports are separated by at least d-ell=3d/4>R. There are N_m
orthonormal orbitals. Sum their products over assignments to particle labels
and normalize by1/sqrt(N_m!). Distinct assignments have disjoint configuration
supports and are orthogonal; the symmetrized vector is normalized and has one
particle in each subcube. Every pair separation exceeds R on its support, so
no contact charge or interaction contributes and all triple exclusions hold.

To establish membership in the box form domain, approximate each sine orbital
in H¹_0 of its subcube by smooth compact interior orbitals. For finite N_m their
tensor products converge in the free sum kinetic form norm; symmetrization
preserves convergence. These approximants are regular intrinsic-core functions
away from all contacts and walls. On that support the contact form equals the
free form, so the limiting trial belongs to the specified box form closure.
Normalization can be restored during approximation without changing the limit.

Energy additivity and orthogonal assignments yield

    q[Psi_m]/N_m=3pi²kappa/(d/4)²
                    =48pi²kappa rho^(2/3).

Thus nonempty finite-energy sectors exist along the exact-density sequence and

    -B rho <= E(m³,m d)/(m d)^3 <=48pi²kappa rho^(5/3).  (3)

These are finite lower and upper envelopes, NOT an energy-density limit or an
equation of state. The trial is spatially inhomogeneous and proves no uniform
phase, stiffness, compressibility or sound mode. m³ objects are an analytic
construction only; no graph, tensor vector or large sector is allocated.

## 5. Separated-box gluing as a canonical comparison lemma

Let finitely many subboxes Lambda_j be contained in a larger box Lambda, with
pairwise spatial separation>R. For admissible particle counts n_j, set N=sum_j n_j.
We claim

    E(N,Lambda)<=sum_j E(n_j,Lambda_j).                 (4)

Choose normalized approximate minimizers in each bosonic intrinsic box core,
with errors tending to0. Initially assign disjoint groups of particle labels
to the boxes and tensor the chosen functions. Expand their finite generator
sums. Selected matchings unite into a matching of all labels; product cutoffs
retain normal flatness, compact support and interior-wall margins. Cross-box
contacts vanish, and any triple using multiple boxes has a cross-box distance>R,
so all cross triples are allowed. Each expanded product is therefore an admitted
global intrinsic-core function, with additive off-contact action and energy.

Symmetrize over assignments of N labels to the boxes, counting each assignment
once since factors are already internally symmetric. The number of assignments
is N!/product_j n_j!. Distinct assignments have disjoint configuration supports,
positive separation from one another and zero differential-action cross pairings.
Normalizing by the square root of this number preserves exactly the sum of
component energies. Letting the finitely many minimizer errors tend to zero
proves(4). No ground-state attainment is used. Empty-sector cases impose no
finite comparison and may be omitted or treated through +infinity.

For a fixed seed n particles in a cube of side ell, a useful consequence at the
same supplied rho is available whenever h=(n/rho)^(1/3)>ell+R. Center m³copies
of the seed box in h-cells of a box of side m h. Their gap exceeds R, and
(4) gives E(m³n,mh)<=m³ E(n,ell), hence

    limsup_(m->infinity) E(m³n,mh)/(mh)^3
                          <=(rho/n)E(n,ell).            (5)

Here n>=1 and the seed is admissible. This is a sequence-specific finite-cluster
competitor bound. An unevaluated seed energy is not an actual improvement over
(3), and(5) does not decide which competitor minimizes energy. A common
thermodynamic limit cannot be inferred merely from these inequalities.

## 6. What uniformity and clustering would mean

Fixed N/volume alone is not a state selection rule. Uniform one-point density
also does not exclude clustering. As a kinematic illustration, on a torus let
f(x) be a nonconstant localized density profile with integral N. Averaging
f(x-a) over uniform translations a gives constant N/volume, without changing
the clustering inside individual profiles. This is a probability-level inference
counterexample, NOT an adopted periodic quantum box, a stationary state of this
model, or a claim that an open finite box is translation invariant.

Three meanings must be kept separate:
- microscopic binding/aggregation: enhanced close-pair or finite-cluster structure;
- macroscopic phase separation: bulk regions or mixtures with distinct densities;
- correlation clustering: decay of connected local-observable correlations at
  large separation in an appropriate extremal thermodynamic state.
A uniform molecular fluid may contain bound pairs and still have clustering of
connected distant correlations. For s>0 the known pair binding energy is
-2kappa/s²; this does not prove a droplet or rule out that fluid. For s<0 absence
of a two-body bound state does not exclude larger clusters or establish uniformity.

The thermodynamic energy limit e(rho), if later established with density-preserving
mixing control, must be distinguished from candidate metastable branch energies.
An affine interval of convex equilibrium energy is compatible with coexistence,
not a constructed morphology. Positive local curvature may support a compressibility
claim with the right ensemble conventions, but does not alone establish a
homogeneous extremal state, correlation decay or phase stiffness. Negative bare
g_a cannot be substituted for thermodynamic curvature; e(rho)=g_a rho²/2 is
not justified for this renormalized hard-domain model.

## 7. Two naive-state tests do not select a correlated phase

For N>=3 an unprojected nonzero constant orbital product in a box has positive
weight on an open set where three labels occupy one sufficiently small ball.
That set is forbidden. Thus it is not an allowed hard-domain vector. More
generally the same argument applies to a smooth product orbital nonzero on an
open ball. No conclusion about all correlated states follows.

Hard-projecting a nonzero smooth product can fail for a different reason.
Construct a local exclusion-boundary patch away from contacts and external
walls with positions x1=c-(R/2)e1, x2=c+(R/2)e1, x3=c. The unique maximal
side is |x1-x2|=R; the other sides are R/2, strictly inside (R/4,R). In a small
neighborhood, the hard boundary for this triple is the smooth hypersurface
|x1-x2|=R. All pair distances there exceed R/4, so every active angle is zero
and the empty-matching localization equals1.

For the sequence in section4 with m>=2, this patch fits in the interior of
one d-cell centered at c because d>2R. Place the N_m-3 remaining labels at
distinct other d-cell centers, leaving two centers unused. There are enough
centers since N_m=m³. Every additional separation is greater than R, including
from the patch's endpoints (at least d-R/2>R). A small configuration neighborhood
therefore has every other triple condition strict. It lies away from every
collision surface and outer wall.

On this patch the projected constant product has a nonzero jump across that
hypersurface. Its zero extension is not locally H¹: the distributional normal
derivative contains nonzero hypersurface measure, not an L² function. The
whole-space localized form characterization requires H¹ here because chi_empty=1.
The box form embeds into that global form, so this projected vector cannot
belong to the box form domain. The same rejection applies to a smooth product
nonzero across this patch. It does NOT assert that every projected product fails:
some project to zero or vanish on the relevant boundary. Nor does it exclude
a homogeneous correlated phase with different short-distance structure.

## 8. Thermodynamic state gate and response not reached

Equations(1)–(5) construct a canonical finite-box problem, an admissible exact-
density sequence with finite energy control, and separated-system competitors.
They do not establish existence or uniqueness of

    e(rho)=lim_(L->infinity,N/L³->rho) E(N,L)/L³.

Missing energy-limit work includes controlling buffer gaps, external boundary
costs, particle-count interpolation and sequence dependence without changing
the imposed density or energy reference. An actual bulk equilibrium state then
needs an observable algebra/topology, limiting construction with density retention,
stationarity and the claimed homogeneity/extremality properties. No such state
is obtained simply by choosing approximate finite-box minimizers, which need
not themselves be stationary.

| Desired claim | Missing proof before promotion |
| --- | --- |
| Canonical energy density | Thermodynamic limit with density, boundary and sequence control |
| Stable homogeneous equilibrium phase | Actual bulk state and comparison with admissible separated/clustered competitors |
| Compressibility | Equation-of-state regularity and justified ensemble/source identification |
| Density response | Stationary state, observable/source convention, existence and limit control |
| Acoustic pole | Appropriate positive stiffness/susceptibility, spectral residue, damping and controlled low-energy errors |

A bounded finite-N smeared density n(f)=sum_i f(x_i), for bounded real f, is a
well-defined symmetric multiplication observable with norm<=N||f||_infinity.
This requires no canonical creation fields, but supplies no limiting response
or phase stiffness. No Kubo kernel or sound speed is derived here because the
thermodynamic phase gate is unresolved. The earlier classical repulsive-ring
Bogoliubov/acoustic formula cannot be imported to bypass it.

Outcome: a homogeneous stable phase has NOT been established or disproved.
The simple product ansätze above fail under their stated hypotheses, but that
is not a universal phase no-go. Even a future sound-mode result would not prove
universal relativity. No physical light speed, chemical potential, condensate,
parameter fit or new interaction is introduced. The task stops here; the loop
remains paused and response analysis is explicitly not reached.

## 9. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_finite_particle_continuum_2026-09-14.md`: intrinsic domain, common
  semiboundedness and symmetry used in the new external-wall restriction.
- `cubic_finite_particle_boundary_density_2026-09-14.md`: localized global form
  norm and contact-free H¹ regularity; not thermodynamic particle density.
- `cubic_finite_particle_convergence_2026-09-14.md`: fixed-N lattice convergence,
  not a joint volume/particle-number limit or a finite-box convergence theorem.
- `cubic_sector_direct_sum_convergence_2026-09-14.md`: number-reference and
  thermodynamic/physical-vacuum limitations.
- `cubic_three_particle_continuum_2026-09-13.md`: pair point-interaction energy.
- `cubic_relativity_gate_2026-09-14.md:97-181`: explicit unestablished phase and
  response prerequisites; conditional acoustic action only.
- `substrate_extensive_stability_2026-09-13.md:51-63`: older ring envelopes do
  not establish this model's equation of state.

Baseline693tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-fixed-density-phase-gate-4b_1q54u/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored caches,
configuration/build products and .git are outside scope. This note is the sole
new repository file. No scientific imports, tests, numerical calculations,
existing-file/model edits, data acquisition or publication are authorized.
## 10. Independent review record

Mathematical reviewer a9a2d3b359b47c933 cleared the finite-box construction,
packing, exact-density trial, separated-box gluing and naive-product exclusions
with no blocker, conditional on the previously reviewed whole-space construction.
No ground-state attainment, thermodynamic limit or actual phase was inferred.

Separate scope reviewer aa2dca60da67bcd02 found no scientific-scope overclaim,
without recertifying those proofs. They confirmed the distinctions among density
input and vacuum selection, energy bounds and a thermodynamic limit, microscopic
binding and phase separation, and ansatz rejection and a general phase no-go.
The exact next prerequisite is canonical thermodynamic energy-density existence
with buffer, boundary, particle-number and sequence control; actual bulk-state
construction and phase assessment would still precede response analysis.

Both reviews used direct reads only. Original proof-draft-v1.md and its hash
are preserved; no mathematical argument, constant or physical assumption changed.
Independent auditor a972770648449528d confirmed all693baseline files unchanged,
exactly one new note and694current files matching final-hashes.json. Assessment
sections1–8 are byte-identical to the draft; the note matched pre-audit-note.md.
This is document/preservation clearance, not mathematical recertification or
proof of absence of unrecorded execution. The audited final-hashes.json is
retained; closing-hashes.json binds final audit-status edits. No numerical tests,
physical-state selection, empirical validation or loop restart occurred.
