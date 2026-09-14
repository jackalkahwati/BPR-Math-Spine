# Strict three-particle exclusion: operator and uniform lattice stability

2026-09-13. Status: sections1–6 independently reviewed with no mathematical blockers.
Independent continuum-scope review confirms the unresolved gate is accurately
stated; final independent saved-document/preservation audit cleared.
Continuum interacting contact-domain construction and lattice-to-continuum
operator convergence remain explicitly unresolved in section7.

This is a new supplied hard constraint, not a modification of protected model
APIs, a numerical experiment or a claim that BPR is a theory of everything.

## 1. Allowed Hilbert space and boundary convention

Fix supplied R,kappa>0 and finite s!=0. Retain C_a=kappa/a² and the previously
analyzed attractive pair tuning

    1/g_a=-a²W/kappa+a³/(8pi kappa s),

for sufficiently small positive a that it is finite. The threshold-referenced
complete N-particle pair operator on labelled lattice configurations X=(x_i) is

    K_pair,a,N=sum_i t_i+g_a sum_{i<j}1_{x_i=x_j},
    t_i=C_a(6I-A_i).

Restrict to the symmetric subspace for bosons; normalized occupation and labelled
symmetric tensor descriptions are unitarily equivalent. No extra Bose factor is
inserted in the pair potential. The +6C_aN reference is explicit; it does not
silently redefine the unshifted full-Fock population energetics.

Let Q_(a,R,N) count unordered triples of distinct particle labels whose three
physical Euclidean pair distances are all <=R. Repeated positions count. Define

    Omega_a={X:Q_(a,R,N)(X)=0},
    P=1_{Omega_a}, F=I-P, H_allow=ran(P),
    K_hard,a,R,N=(P K_pair,a,N P)|H_allow.               (1)

Every allowed triple has diameter strictly greater than R. P need not commute
with the unconstrained hopping: this is an imposed configuration-space domain,
not an inherited invariant sector. For N<=2, P=I and the complete two-body
operator and all previously scoped scattering results are exactly unchanged.

At fixed a,N, ||sum_i t_i||<=12C_aN and the pair multiplier is bounded by
|g_a|binomial(N,2). Thus K_pair is bounded self-adjoint, as is its compression
on the closed allowed subspace. The direct sum over allowed number sectors is
self-adjoint on

    {psi=(psi_N):sum_N||K_hard,a,R,N psi_N||²<infinity}.

Sector truncations converge in graph norm and give a finite-number-support core.
Finite occupation-support approximants within a sector also converge in graph
norm by boundedness. Empty allowed finite-box sectors are zero Hilbert spaces,
not fictitious ground states. No lower bound on infinite-volume full Fock space
is inferred from self-adjointness or from an extensive bound alone.

Always extend allowed wavefunctions by zero into forbidden configurations and
evaluate the ORIGINAL kinetic quadratic form. In labelled coordinates,

    q_T[psi]=C_a sum_{unordered configuration edges{X,Y}}|psi(X)-psi(Y)|².

The complete configuration graph has degree6N. In a finite open box use zero
extension outside it as well. Missing/forbidden hopping bonds retain their
original diagonal costs. Replacing6C_aN by the degree of the surviving graph
would instead define a reflecting operator, not(1). All arguments below use
this Dirichlet compression and the fixed threshold reference.

## 2. Fixed-a infinite-penalty resolvent limit

Fix a,R,N and work on either the full labelled or symmetric complete sector.
Write K=K_pair, M=||K|| and relative to P and F write

    K+Lambda Q = [[A,B],[B*,D+Lambda Q_F]],
    A=K_hard, Q_F>=I_F, ||B||<=M, D>=-M I_F.

For fixed nonreal z let eta=|Im z|, h=Lambda-M-|Re z|>0 and
L=D+Lambda Q_F-z. Its real part has lower bound h, hence ||L^(-1)||<=h^(-1).
Let delta=M²/h. The Schur complement is
S=A-z-B L^(-1) B*. If delta<=eta/2, the resolvent of self-adjoint A gives

    ||S^(-1)||<=2/eta,
    ||S^(-1)-(A-z)^(-1)||<=2M²/(h eta²).

The block inverse has upper off-diagonal -S^(-1)B L^(-1), lower off-diagonal
-L^(-1)B* S^(-1), and lower diagonal
L^(-1)+L^(-1)B* S^(-1)B L^(-1). Summing block norm bounds yields

    ||(K+Lambda Q-z)^(-1)-iota(A-z)^(-1)iota*||
      <= 2M²/(h eta²)+4M/(h eta)+1/h+2M²/(eta h²).      (2)

Here iota includes the allowed space in the ambient one. This tends to0 as
Lambda->infinity at fixed a,N,z. If F is zero there is no penalty and equality
holds already. If P is zero the whole inverse is the forbidden-block inverse,
with norm<=1/h and limiting operator0. Empty blocks need no artificial inverses.

This is an operator-norm limit to an embedded COMPRESSED resolvent. If F is
nonzero the limiting operator has a kernel, so it is not the resolvent of an
ordinary densely defined self-adjoint operator on the original ambient space.
It does not establish a full-space unitary limit: forbidden states can retain
rapidly oscillating phases. Constants in(2) depend on a,N,z; no uniform Fock,
spacing or interchangeable-limits claim follows. Infinite volume is permitted
at fixed N because K is bounded there; no finite-volume spectral shortcut is used.

## 3. Extend the two-body energy bound to every total momentum

The prior scattering derivation establishes the zero-total-momentum relative
operator on ell²(Z³): h_a(0)=2C_a(6I-A)+g_a|0><0|. Its only possible negative
spectrum is the contact bound state: the rank-one resolvent formula excludes
negative spectral points except scalar denominator zeros. The odd parity sector
is free/nonnegative; the even sector is the previously analyzed Bose fiber.

For distinguishable particles at total momentum P, Fourier transformation of one
particle coordinate followed by the gauge exp(iP dot r/2) gives relative hopping
coefficient2C_a cos(P_j/2) in direction j, with unchanged diagonal12C_a and the
same real contact potential g_a delta_0. The gauge may be chosen separately on
each fiber; no global periodic gauge assertion is needed. Its hopping magnitudes
are <=2C_a. For any relative vector f, each bond satisfies

    -2 Re(c conjugate(f(r))f(r+e_j))
      >= -2|c||f(r)||f(r+e_j)|
      >= -4C_a|f(r)||f(r+e_j)|.

The diagonal, contact potential and norm are unchanged by f->|f|. Thus

    <f,h_a(P)f> >= <|f|,h_a(0)|f|>.                     (3)

This compares bottoms at the SAME threshold diagonal, not after resetting each
fiber minimum. It covers unrestricted relative vectors, hence distinguishable
as well as symmetric two-particle states after direct integration in P.

For s<0 the previous monotonic rank-one denominator analysis gives no negative
spectrum at P=0, so h_pair,a>=0. For s>0 the unique negative eigenvalue tends
to -2kappa/s². More explicitly the prior negative-energy expansion, tested at
chi_+=2/s in z=-2kappa chi², places the unique root above -8kappa/s² for all
sufficiently small a. Therefore there is a threshold a_pair(s,kappa)>0 such that
for all a<a_pair and EVERY total momentum,

    h_pair,a>=-E_s,
    E_s=8kappa/s² if s>0, and0 if s<0.                  (4)

This eventual bound is not a claimed explicit formula for a_pair or uniform
pole convergence. It also applies to two selected particles with arbitrary
other coordinates as spectators, by tensoring the operator inequality.
Open-box restrictions inherit it by zero extension. This all-momentum extension
is proved here; it is not inferred merely from a scalar on-shell amplitude.

## 4. Exact matching partition on allowed configurations

Fix r0=R/8 and r1=R/4. Define theta(d)=pi/2 for d<=r0, theta(d)=0 for d>=r1,
and interpolate linearly between them. It is Lipschitz with L_theta=4pi/R;
smoothness is not needed for a discrete quadratic form. For every unordered
label pair e={i,j}, set theta_e(X)=theta(a|x_i-x_j|).

Let E_N be all unordered label pairs. For each subset J of E_N define

    chi_J(X)=product_{e in J}sin(theta_e(X))
             product_{e not in J}cos(theta_e(X)).

For fixed finite N this is a finite analytic family; it is never enumerated
computationally. The product identities give for every configuration

    sum_J chi_J(X)²=1,
    sum_J chi_J(X)chi_J(Y)=product_e cos(theta_e(X)-theta_e(Y)). (5)

On an allowed X the graph of active edges theta_e(X)>0 is a matching. If two
active edges shared particle i, both distances from i would be <r1, so the
third pair distance would be <2r1<R, producing a forbidden triple. A nonmatching
J therefore has chi_J(X)=0 on all allowed configurations.

Moreover, if x_i=x_j, theta_ij=pi/2 and its cosine is zero. Every nonzero chi_J
at that configuration must include {i,j}. Thus every contact term on a localized
vector chi_J psi is assigned to a selected disjoint pair. Contact terms between
unmatched particles or different selected pairs vanish pointwise on that
localized vector. All unselected pairs still appear in the cosine product;
omitting them would invalidate(5).

Individual labelled chi_J need not preserve bosonic symmetry. We work on the
complete labelled tensor space for localized estimates using(4), then sum them
and apply the result to the original symmetric vector. Multiplication and zero
extension preserve allowed support, even though unconstrained hopping does not.

## 5. One-changing-angle lemma and discrete IMS error

Take allowed configurations X,Y differing by one lattice hop of particle i;
its physical displacement is a. Only theta_ij can change. At each endpoint
there is at most one active partner for i by section4. Suppose distinct j,k
were active at opposite endpoints. At X, distances satisfy

    d(i,j)<r1, d(i,k)<r1+a, d(j,k)<2r1+a.

For a<R/8, all three are <R, contradicting allowedness of X. Therefore the
union of active partners across both endpoints contains at most one label.
Angles zero at both endpoints do not change; at most ONE angle changes.
Distance is1-Lipschitz under a particle displacement, so its change obeys
|Delta theta|<=L_theta a. Equation(5) then yields

    D(X,Y):=sum_J|chi_J(X)-chi_J(Y)|²
      =2[1-cos(Delta theta)]<=L_theta²a².               (6)

This includes activation/deactivation across either cutoff and has no constant
depending on N or the number of matchings.

For a finitely supported allowed complex vector psi, expand the squared edge
differences. Diagonal potentials localize exactly by(5), so the discrete IMS
identity for the full hard quadratic form is

    sum_J q_hard[chi_J psi]-q_hard[psi]
      = C_a sum_{unordered edges{X,Y}} D(X,Y)
                Re(conjugate(psi(X))psi(Y)).            (7)

At forbidden or outside-box endpoints psi is zero. Those edges have zero cross
term in(7), and their original diagonal energy localizes exactly; no deleted
boundary degree or estimate against a forbidden endpoint is required. At allowed
edges use(6) and |Re(conjugate(u)v)|<=(|u|²+|v|²)/2. Each configuration has at
most6N neighbors, giving

    |sum_J q_hard[chi_J psi]-q_hard[psi]|
      <=3 C_a N L_theta²a² ||psi||²
      =48pi² kappa N/R² ||psi||².                      (8)

The factor3 is half the maximal directed degree6N. This calculation uses ordered
particle coordinates with hopping coefficient C_a, not occupation-coordinate
ladder weights; their Hilbert representations are equivalent. Fixed-N boundedness
extends(7)–(8) from finite support to every vector of the allowed sector.

## 6. Cutoff-uniform extensive lower bound

For a matching J put phi_J=chi_J psi, extended by zero. By exact contact allocation,
its quadratic form equals the expectation in the complete labelled space of

    sum_{{i,j} in J}(t_i+t_j+g_a delta_{x_i,x_j})
       +sum_{unmatched i}t_i.

Each selected pair retains its TWO FULL one-particle kinetic operators; there
is no fractional kinetic sharing that would alter critical contact tuning.
The pair estimates(4) act on disjoint coordinate pairs and singles are
nonnegative. Therefore

    q_hard[phi_J]>=-E_s |J| ||phi_J||²
                 >=-(E_s N/2)||phi_J||².

Nonmatching components vanish. Sum over J, use(5) and(8), obtaining

    K_hard,a,R,N >= -B(R,s,kappa) N I,
    B(R,s,kappa)=kappa[4*1_{s>0}/s²+48pi²/R²],           (9)

for0<a<min(R/8,a_pair), uniformly in N and spatial volume. Empty sectors need
no spectral-bottom assertion; the operator inequality is vacuous there.
The same proof covers finite open boxes through zero extension with original
boundary costs. No periodic-wrap estimate is silently invoked.

This is a coarse but explicit cutoff-uniform stability-of-the-second-kind bound
for the specified HARD lattice candidate. It handles all allowed configurations,
including separated clusters and chains: only the smaller-r1 localization graph
is a matching, not the full distance<=R graph. It does not follow merely by
removing the old collapse witness. The elementary n_x<=2 implication alone would
only give -|g_a|N/2, which diverges as a shrinks and is not used to prove(9).

At fixed N the bound excludes an energy bottom tending to minus infinity as
a->0 in this reference. It does not supply a preferred particle number, physical
vacuum, thermodynamic equation of state or N-independent lower bound on
infinite-volume full Fock space. R remains a supplied physical length. The result
is not empirical validation or derivation of space, matter or gravity.

## 7. Continuum interacting operator: unresolved gate

Let Omega_(R,N) be the continuum allowed open region with every triple diameter
>R. The free Dirichlet Laplacian on this region is well-defined, but is NOT the
intended tuned pair-contact theory. Setting its boundary condition alone would
lose the prescribed scattering length even though it avoids triple clustering.

A formal pair-contact expression near r_i=r_j would involve a singular amplitude
A_ij and the combination (1/|r_i-r_j|-1/s). Writing that expression does not
define the trace spaces, a closed renormalized form or a self-adjoint operator.
Pair-contact surfaces can intersect for DISJOINT pairs, even though collisions
sharing a particle are excluded. They also approach the hard triple boundary.
The lattice matching argument alone does not supply the needed form-domain
compatibility at these configurations.

This tranche does NOT establish any of the following:
1. A densely defined closed semibounded continuum contact form on Omega_(R,N),
   or an equivalent specified self-adjoint realization with the given s and
   Dirichlet exclusion boundary.
2. Compatibility of simultaneous disjoint contact singularities and their
   singular coefficients with that boundary, including overlap consistency of
   local pair constructions.
3. Common-space lattice embeddings plus a liminf/recovery-sequence or resolvent
   convergence theorem retaining these singularities and this domain.
4. A resulting convergence theorem for interacting unitary dynamics.

A proposed future route is localization of closed two-body point-interaction
forms and a proof that they agree on overlaps, followed by form convergence.
No such construction is claimed here: existence, closedness, overlap compatibility
and recovery sequences are the unresolved work, not minor notation omitted
from the proof. Uniform stability(9), allowed-indicator convergence, and the
previous scalar scattering-amplitude limit do not imply these results.

Stop at this gate. No substitute free-pair model, numerical example, extra field
or alternative constraint is introduced to claim continuum completion. This is
not a no-go statement about the unresolved construction.

## 8. Source register, prior audit and preservation

Protected sources, read but not changed:
- `cubic_finite_range_three_body_2026-09-13.md:10-87,89-188`: exact triple geometry,
  bounded-sector setup and why fixed penalties fail; its multiplier limit is not
  a hard-domain or kinetic convergence proof.
- `cubic_two_body_scattering_2026-09-13.md:218-274`: pair tuning and zero-P pole
  classification; section3 above proves the all-P comparison separately.
- `cubic_three_body_stability_2026-09-13.md:70-93,175-230`: boundary kinetic costs
  and the fixed-N witness excluded by the new domain.
- `substrate_fermionization_2026-09-12.md:46-57,99-136`: compression/Schur technique;
  no ring hard-core or Jordan–Wigner claim is transplanted.
- `cubic_contact_continuum_2026-09-13.md:66-80`: formal contact matching is not
  construction of a three-dimensional contact Hamiltonian.
All source names above are relative to doc/derivations/.

The prior bounded-range final audit notification from af068f77871b55405 arrived
during planning: it cleared20saved passes, both CLI outputs and all671current
bindings against its final-hashes.json. That prior note still says pending; it
is protected and is not rewritten in this different task. The received clearance
is recorded externally here rather than silently altering historical evidence.

Current baseline671tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-body-exclusion-295txqgd/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored caches,
config/build products and .git are excluded. This note is the sole new repository
file. No scientific imports, tests, matrix construction or numerical evaluation
are part of this proof-only tranche. All old failures/regressions remain intact.

## 9. Independent review record and stopping point

Reviewer aed9f4117fb3af8b1 independently cleared sections1–6 with no mathematical
blockers, including the explicit Schur bound, all-momentum pair comparison,
piecewise-linear cutoff, contact allocation, one-changing-angle lemma, boundary
costs and exact48pi² localization constant. The clearance covers uniform lattice
stability in infinite volume and finite open boxes, not a periodic extension
or a continuum contact Hamiltonian. Only direct file reads were used.

A separate reviewer a05ae4f0a84732fa5 found no continuum overclaim in section7.
They confirmed the contact domain, simultaneous disjoint singularities, contact
behavior at the exclusion boundary and full operator/dynamics convergence remain
substantive unresolved gates. This review does not clear a continuum construction.
No new theorem or scientific computation was performed by that scope review.

The originally submitted proof-draft-v1.md and its hash are preserved externally.
Only review-status/provenance text has changed after mathematical submission;
no proof constant or hypothesis was changed to obtain clearance. Independent
auditor a334e6a531eac666a confirmed all671baseline files unchanged, exactly one
addition and672total scoped files, with current hashes matching final-hashes.json.
Sections1–7 are identical to the submitted draft, and the reviewed note matched
pre-audit-note.md byte-for-byte. The scope text accurately preserves the unresolved
continuum gate and distinguishes prior audit provenance from new execution.
This was a saved-document/hash audit, not renewed mathematical certification or
proof of absence of unrecorded execution. The audited final-hashes.json is retained;
closing-hashes.json binds the final audit-status edits.

The result is a proved cutoff-uniform extensive lower bound for one newly
supplied hard lattice candidate, with unchanged two-body scattering. The work
stops at the unresolved continuum interacting-domain/convergence gate, as planned.
No new model, simulation, publication or TOE claim is initiated.
