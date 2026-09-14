# Continuum hard-exclusion candidate at every fixed finite particle number

2026-09-14. Status: construction at each fixed finite N independently cleared;
independent saved-document/hash audit cleared.
This note constructs sector operators and an extensive bound. General finite-N
boundary density and lattice convergence are separate, unproved gates here.

## 1. Fixed family, geometry and ambient matchings

Fix supplied R,kappa>0 and finite real s!=0. For each integer N>=0 let Omega_N
exclude every three-label subset of diameter<=R in (R³)^N. For N<3 there is
no exclusion. Let M_N be the finite set of matchings of N labels and
m_N=floor(N/2). Each pair e={i,j} uses r_e=xi-xj, c_e=(xi+xj)/2, with
Jacobian1 and kinetic coefficients2kappa in r_e and kappa/2 in c_e.

For a matching P define the ambient operator

    A_P=sum_{e in P}(h_(s,e)-(kappa/2)Delta_(c_e))
                    +sum_{unmatched i}(-kappa Delta_(x_i)),
    A_P>=-|P|E_s, E_s=2kappa*1_{s>0}/s².                (1)

These are finite spectral tensor sums of the established relative point
interactions and free coordinates. No additional physical interaction or
many-contact trace condition is imposed.

After shifting the finitely many constituents to be positive, their joint
spectral variables xi_j obey

    sum_j xi_j² <= (sum_j xi_j)² <= k sum_j xi_j²,

where k is the number of constituents. This identifies the sum operator domain
with the intersection of constituent domains; the sum form domain is the
intersection of their square-root domains. Constants may depend on finite k.
Use these domains rather than assigning independently free-tangential charges
to arbitrarily deep multiple contacts.

The established relative operator core is
C_rel=C_c^infinity(R³ minus{0})+span{b}, with
b=rho(G_mu+c_mu), G_mu=exp(-mu|r|)/(4pi|r|),
c_mu=(mu-1/s)/(4pi), mu>max(0,1/s), and rho smooth radial compact, equal1 near0.
Subtracting the singular coefficient leaves an H² function of zero point trace;
its approximation by smooth functions avoiding0 gives graph-core density as
proved previously. Finite products of these cores with smooth compact center
and unmatched-coordinate factors form an operator core V_P for A_P. To see
this at any fixed finite depth, first restrict the joint spectrum to bounded
rectangles, approximate by finite tensors there, approximate each factor in
its graph core, then remove the spectral restriction in the sum graph norm.

In particular q(z)product_{e in P}b_e(r_e) belongs to D(A_P). Any finite product
of1/|r_e| singularities is locally L² in the independent relative variables:
its squared integral on a product neighborhood factors into finite3D integrals.
When one relative factor is differentiated, the others can remain undifferentiated
L² factors. This finite-depth observation is what later multiplier estimates use.

The earlier two-contact mixed-charge extraction bound must NOT be extrapolated.
For three or more extracted charges the product of spectral weights is not
controlled by the square of their sum. No bounded all-mixed-charge extraction
map on the ordinary operator graph norm, or on the sum form norm, is assumed
or required here.

## 2. Intrinsic compact singular initial domain

A smooth eta is P-admissible if its support is a compact subset of

    U_P=Omega_N minus union_{e not in P}Sigma_e,

and eta is independent of every selected relative coordinate r_e in a uniform
tube about r_e=0, with the other P-coordinates fixed. Define

    D_(0,N)=span{eta v: P in M_N, v in V_P, eta P-admissible}.         (2)

Allow all the admissible choices of relative generators. This agrees with the
N=4 prescription, rather than introducing a partition-dependent initial domain.
Ordinary smooth compact tests away from all collisions are included through
the empty matching with product bumps equal1 on their supports, proving L²
density in Omega_N. Define S_N as the L² free differential action
-kappa sum_i Delta_(x_i) away from contact surfaces, not the full distributional
Laplacian with contact delta sources retained. The next lemma verifies its L²
action and cross-matching compatibility.

## 3. Restricted multiplier lemma for arbitrary finite depth

Expand a tensor-core vector as v=q(z)product_{e in P}w_e(r_e), each w_e either
regular or a permitted singular factor. Suppose C subset P, and F is smooth
compact, normal-flat at every contact in C, and zero in uniform tubes about
contacts in P minus C. Then

    Fv in D(A_C).                                      (3)

For a retained singular e put B_e=q product_{f!=e}w_f. Normal flatness permits

    F b_e B_e=b_e F(0,.)B_e+b_e[F-F(0,.)]B_e.

The first term satisfies the relative contact condition with L²-valued spectator
coefficient. The second vanishes near r_e=0; its relative second derivatives
are L² and have zero trace. Derivatives of b_e occur only away from its own
singularity; all other singular factors are undifferentiated and controlled
by finite tensor-product L² norms. Bounded smooth coefficients do not change
this control. Apply this separately to each retained relative coordinate.

For each dropped pair, Fv vanishes near its collision, so second derivatives
in that relative variable encounter no singularity. The free complement of C
consists of dropped pairs' coordinates, unmatched coordinates and centers.
Differentiation there never differentiates a retained singular relative factor.
The product rule supplies second derivatives in each free coordinate in L².
Fourier ellipticity then gives full H² in the free complement with L² retained-
relative values: sums of squared coordinate second-derivative weights control
all free second derivatives. No separate estimate of each mixed derivative is
needed. The positive tensor-sum domain identity proves(3).

If M contains C and supp(Fv) also avoids every added Sigma_e for e in M minus C,
then Fv belongs to D(A_M). Added pairs involve only labels unmatched in C,
so they never split a retained singular pair. Free-complement H² supplies their
regularity and tube avoidance supplies zero collision trace, hence the contact
operator condition with zero charge. This is not a claim that arbitrary smooth
functions with nonzero trace belong to a zero-charge contact operator domain.

The lemma is restricted to initial tensor generators; it is not yet a completed-
form multiplier theorem at arbitrary N. It proves S_N is an L² action on(2).
Alternative representations of the same vector agree off measure-zero surfaces,
so the L² action is well-defined.

## 4. Cross-matching symmetry without dropping common contacts

Let u,v be generators from P,Q, set C=P intersect Q and
K=supp u intersect supp v. K is compact inside U_C. Construct a real compact
plateau zeta in U_C equal1 near K. Flatten it in the disjoint relative coordinates
of C using smooth maps sending r_e to0 below epsilon and leaving it unchanged
above2epsilon. The maps commute. Each particle moves by at most epsilon,
so every pair distance changes by at most2epsilon, independently of depth.
The FULL configuration displacement may be O(sqrt(N)epsilon); choose epsilon
below the actual compact support/plateau margins with this finite-N factor
included. This preserves support and plateau while making zeta normal-flat.

Lemma(3) puts zeta u,zeta v in D(A_C). Support locality gives

    <u,S_Nv>=<zeta u,A_C(zeta v)>.

Commutator terms are L² by the same finite-tensor derivative argument, and vanish
in the pairing because zeta is constant near the common support. Self-adjointness
of A_C proves symmetry. This works even when the shared matching contains many
contacts; it is not a free-energy overlap argument. Finite bilinear expansion
proves S_N symmetric on the full D_(0,N).

## 5. Singular IMS and a uniform extensive continuum bound

Choose the same smooth theta with r0=R/8,r1=R/4 and |theta'|<=L=8pi/R.
Define chi_J by the sine/cosine product over ALL subsets of the binomial(N,2)
label edges. This is a finite analytic family at fixed N, never enumerated.
The full product identity gives sum_J chi_J²=1.

On Omega_N active edges form a matching: two incident distances<R/4 make all
three distances of their triple<R. Thus nonmatching components vanish, and at
most m_N angles can be active. Near any selected collision, every competing
incident angle vanishes by the all-triples constraint; disjoint angles are
independent of that relative coordinate. The restricted multiplier lemma
therefore proves chi_Mu in D(A_M) for u in D_(0,N). Compact interior source
cutoffs turn this local statement into the global normal-flat multiplier
statement needed by(3).

The operator commutator proof of singular IMS is valid at each finite depth.
Every differentiated singular factor is protected by its vanishing normal cutoff
derivative, or removed by a dropped-contact cutoff; the remaining singular
factors are undifferentiated L² tensors. This yields

    <u,S_Nu>=sum_{M in M_N}<chi_Mu,A_M chi_Mu>
                       -integral W_loc|u|²,
    W_loc=kappa sum_M|grad_X chi_M|².

For the full product partition, sum_J|grad chi_J|²=sum_e|grad theta_e|².
Each varying pair distance contributes squared particle-gradient2. Nonmatching
functions and their derivatives vanish on the open allowed set. Consequently

    0<=W_loc<=2kappa m_N L²<=kappa N L²,
    <u,S_Nu>>=-(m_N E_s+2kappa m_N L²)||u||².            (4)

Density, symmetry and semiboundedness produce the specified Friedrichs operator
H_N^D: its form is the closure of (4)'s intrinsic pairing on D_(0,N), with a
sufficiently positive form-norm shift. This prescription enforces hard exclusion
through closure of compact-interior admissible singular states. It is not yet
identified with an independently defined maximal local-energy domain.

The construction yields the explicit extensive bound

    H_N^D >= -kappa N[1_{s>0}/s²+64pi²/R²].             (5)

The coefficient is independent of N; this is a stability bound, NOT a convergence
estimate uniform in N. The chosen smooth cutoff is the same finite physical-scale
proof tool, not a fitted interaction. Since the initial operator definition
precedes the partition, any partition reproducing the identity estimates the
same form and does not add a physical potential.

All channels and admissible generators are permutation invariant as a family.
The Friedrichs form/operator therefore has a bosonic reducing restriction.
D_(0,N) is a form core by construction, not asserted to be a graph core of
that Friedrichs extension. Existence of finite-energy singular generators is
explicit: place pair centers and unmatched particles sufficiently far apart,
then choose small relative supports and normal-flat compact cutoffs.

## 6. Exact small sectors and compatibility with prior constructions

Define N=0 as the one-dimensional vacuum sector H_0=0. For N=1 there are no
contacts or exclusion; the compact smooth core closes to -kappa Delta.
For N=2 the absence of exclusion alone is not the whole identification argument.
Every compact ambient pair tensor-core vector belongs to D_(0,2): use matching12
and a normal-flat compact plateau equal1 around its support. Conversely the
restricted multiplier lemma places every D_(0,2) generator in the known ambient
two-particle operator domain with the SAME action. Thus the initial restriction
contains an ambient operator core and its closure is that self-adjoint operator.
Its Friedrichs extension is exactly the established full pair point interaction.

At N=3, compatibility with the earlier singular domain can be checked directly.
For a new generator eta q b, let q'(z)=eta(0,z)q(z), and choose an earlier-style
narrow radial generator q'b' with the same mu and support inside the allowed
pair tube. Normal flatness makes the difference vanish in a uniform contact
tube; it is a smooth compact regular function away from all collisions. Conversely,
an earlier generator admits a compact normal-flat plateau equal1 around its
support. The initial domains and off-contact actions therefore agree, and so
do their Friedrichs closures.

At N=4 formula(2) is the previously specified initial-domain prescription, with
all matching channels and their compact admissible multipliers. The construction
therefore retains that candidate, not merely a different operator with a similar
lower bound. No new physical parameters, boundary terms or pair tuning are added.
The N=3/N=4 convergence results retain their previous scopes; this construction
alone does not enlarge them.

## 7. Audit of the remaining arbitrary-finite-N gates

The finite-depth argument above proves a candidate continuum CONSTRUCTION,
subject to independent review. The following steps remain separate:

- Completed-form multiplier bounds and contact dropping/adding must be extended
  by core completion at arbitrary depth; the restricted operator lemma alone
  does not supply every maximal-domain mapping.
- Whole-state boundary density must identify the minimal closure with global
  physical zero-extended matching energies. The collapse geometry is promising:
  each endpoint moves at most epsilon even for many disjoint collapses, so affected
  incident distances stay above R-4epsilon; full-space compact margins still
  need the finite-N displacement accounted for.
- All-matching physical-cell tensor convergence needs a written finite-factor
  proof, not extrapolation from N=4 alone.
- The shifted hard-lattice lower limit requires the domain theorem and correct
  GLOBAL partition correction. For a scalar coordinate, at most N-1 angles
  vary globally, giving the candidate bound M_N=3kappa N(N-1)L² from6Nneighbors.
  This quadratic proof shift differs from the sharper extensive stability bound.
- Individual-generator graph/cutoff recovery and the reviewed abstract positive
  Gram assembly can be applied only after those inputs are established.
- The full form/resolvent/dynamics synthesis must keep the same fixed N, model,
  domain, physical embedding and shift throughout.

There is no inference of uniform convergence as N grows, a thermodynamic or
full-Fock continuum limit, chemical potential, selected vacuum or empirical
validation. A future sector direct-sum strong-resolvent argument might use a
uniform nonreal resolvent norm, but is not part of this construction and does
not itself make unshifted full-Fock energy semibounded. No TOE claim follows.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_four_particle_double_contact_2026-09-13.md`: relative operator-core proof,
  tensor domains and the specific limitations of mixed-charge graph traces.
- `cubic_four_particle_continuum_2026-09-13.md`: intrinsic D0, restricted
  multiplier/common-contact symmetry and Friedrichs prescription.
- `cubic_four_particle_boundary_density_2026-09-13.md`: finite-depth form-domain
  strategy, not already an arbitrary-N theorem.
- `cubic_three_body_exclusion_2026-09-13.md`: previously reviewed all-N lattice
  stability and the physical hard rule.
- `cubic_four_particle_convergence_2026-09-14.md`: completed N=4 scope and the
  need to check arbitrary matching depth explicitly.

Baseline686tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-finite-particle-continuum-nllj84ap/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 9. Independent mathematical review

Reviewer a883f61f294a2049c cleared the construction for each fixed finite N,
including the stated extensive lower bound, with no mathematical blocker.
They checked finite tensor domains/cores, arbitrary-depth restricted multiplier
compatibility, shared-contact symmetry, collapse displacements, singular IMS,
extensive coefficient and exact small-sector/N=4 identifications. Their direct
N=3 initial-domain compatibility argument was added explicitly in section6.

Clearance does not include general boundary density, maximal-domain identification
or lattice convergence. Direct reads only were used; original proof-draft-v1.md
and its hash remain preserved. No physical hypotheses or mathematical constants
changed. Independent auditor ab399e3a39134a987 confirmed all686baseline files
unchanged, exactly one authorized note and687current files matching
final-hashes.json. Draft changes are review/status documentation and the explicit
reviewer-supplied N=3 compatibility argument only; the note matched its pre-audit
snapshot. This is documentary/hash clearance, not mathematical recertification
or proof of absence of unrecorded execution. The audited final-hashes.json is
retained; closing-hashes.json binds final audit-status edits.
