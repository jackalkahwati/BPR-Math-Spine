# Four-particle continuum candidate with compatible matching contacts

2026-09-13. Status: the specified four-particle Friedrichs construction independently
cleared; independent saved-document/hash audit completed with no blockers.
This defines a particular Friedrichs candidate if the proof clears. It does
not identify a maximal domain or prove a four-particle lattice limit.

## 1. Ambient matching operators and intrinsic initial domain

Retain supplied R,kappa>0 and finite s!=0. Let Omega=Omega_R^(4) exclude every
three-label subset of physical diameter<=R. Let Sigma_e be the coincidence
surface for pair e. Let E_s=2kappa*1_{s>0}/s², and let Mset contain the empty
matching, six single matchings and three double matchings of four labels.

For P in Mset define on complete ordered L²(R^12)

    A_P=sum_{e={i,j} in P}(h_(s,e)-(kappa/2)Delta_(c_e))
            +sum_{unmatched i}(-kappa Delta_(x_i)).      (1)

Every pair retains its full two-particle kinetic operator. The established
relative point interactions and spectral tensor sums give self-adjoint A_P
with lower bound -|P|E_s. These are ambient matching operators; no naive free-
tangential charge space is imposed on a double contact.

Use the relative operator core C_rel proved in the double-contact note:
C_c^infinity(R³ minus{0})+span{b}, where
b=rho(G_mu+c_mu), G_mu=exp(-mu|r|)/(4pi|r|), c_mu=(mu-1/s)/(4pi),
mu>max(0,1/s) and rho smooth radial compact, equal1 near0. Allow all such
admissible choices. Take finite algebraic products of relative core factors
and smooth compact functions of all pair centers and unmatched coordinates;
call their span V_P. Each expanded product belongs to D(A_P).

A smooth multiplier eta is P-admissible when its support is a compact subset
of Omega minus the union of Sigma_e for e not in P, and for each e in P it
is independent of r_e in some uniform tube |r_e|<epsilon_e, with all other
P-adapted coordinates held fixed. Define

    D0=span{eta v: P in Mset, v in V_P, eta P-admissible}.             (2)

The definition uses no matching partition. Ordinary smooth compact tests away
from all contact surfaces belong to D0: use the empty matching and product
bumps equal1 around their support. Thus D0 is dense in L²(Omega). It also
contains nonzero allowed double-contact generators from the preceding local
lemma, using a flat cutoff equal1 around their separated-center support.

Define S u as the L² free differential action T=-kappa sum_i Delta_(x_i)
OFF the contact surfaces. Contact delta distributions are not retained in this
action. The tensor multiplier argument below establishes that the action is
indeed L². Equality of alternative generator descriptions implies equality of
the differential action off the measure-zero surfaces, so S is unambiguous.

## 2. Restricted tensor multiplier compatibility

This lemma concerns the declared generator class, not every ambient operator
vector under arbitrary multipliers. Expand one generator as
v=product_{e in P}w_e(r_e) q(z), where each w_e is a regular relative core
factor or a singular b_e, and q is smooth compact in remaining coordinates.
Suppose F is smooth compact, normal-flat near contacts in C subset P and zero
in a neighborhood of contacts in P minus C. Then

    Fv belongs to D(A_C).                              (3)

For a retained singular factor b_e, let B_e(y)=q(z)product_{f!=e}w_f(r_f).
Normal flatness gives the decomposition in its relative variable

    F(r_e,y)b_e(r_e)B_e(y)
      =b_e(r_e)F(0,y)B_e(y)
         +b_e(r_e)[F(r_e,y)-F(0,y)]B_e(y).

The first term satisfies the Hilbert-valued contact boundary condition, and the
second vanishes near r_e=0 and belongs to H² in r_e with L² spectator values.
Derivatives of b_e occur only away from its own singularity; the other factors
need only be L² there, even if they contain another contact singularity. The
same argument for each retained pair gives membership in each relative contact
operator domain. For dropped pairs, the product vanishes in their collision
tubes, making its relative second derivatives L² with zero charge. Center and
unmatched-coordinate derivatives act on smooth factors. The shifted positive
commuting tensor factors identify their domain intersection with D(A_C), proving(3).
This establishes an L² action without pretending the double-contact vector
has globally finite bare gradient energy.

If M contains C and supp(Fv) avoids Sigma_e for every e in M minus C, the
free complementary coordinates have H² regularity and zero trace at the added
surfaces. The corresponding relative contact domains then admit zero charge,
so Fv also belongs to D(A_M). The added pairs use only coordinates unmatched
in C; no singular retained pair is split by this step.

## 3. Symmetry across matchings uses their common contacts

Let u and v be generators of channels P and Q. Put C=P intersect Q and
K=supp u intersect supp v. Each channel avoids all its unselected surfaces,
so K is a compact subset of

    U_C=Omega minus union_{e not in C}Sigma_e.

Choose a real smooth compact zeta in U_C, equal1 near K, normal-flat near all
contacts in C. Such a cutoff exists: start with a plateau having positive
support and plateau margins, and in the mutually disjoint relative coordinates
of C compose it with smooth maps sending r_e to0 for |r_e|<=epsilon and equal
to r_e for |r_e|>=2epsilon. The maps commute and move coordinates uniformly
by O(epsilon); choosing epsilon smaller than the margins preserves support
and the plateau while giving normal flatness.

The multiplier lemma applied to zeta times each original admissible multiplier
gives zeta u,zeta v in D(A_C): dropped contacts are removed but shared contacts
remain. Expansion of the commutator and support locality gives

    <u,Sv>=<zeta u,A_C(zeta v)>.

Every cutoff derivative contribution paired with u is supported in the common
support where zeta is constant. The commutator terms are L² by the tensor
argument, including at a disjoint retained contact; no singular derivative is
used in its own uncut normal direction. The same identity with u,v exchanged
and self-adjointness of A_C proves <u,Sv>=<Su,v>. Finite linear expansion proves
symmetry on D0.

For example channels{12} and{12,34} must be compared through A_{12}, NOT the
free operator: their overlap can retain contact12. This is the essential new
compatibility issue absent from the N=3 contact-free cross-channel argument.

## 4. Global matching localization on the initial domain

Use the smooth theta from the continuum construction: pi/2 at distances<=R/8,
zero at distances>=R/4, with |theta'|<=L=8pi/R. For ALL64subsets J of the six
pair edges define the product chi_J of selected sine and unselected cosine
factors. Globally sum_J chi_J²=1. On the allowed open Omega, active near-pair
edges form a matching; consequently only the ten matching components survive.

A further local fact controls the operator domains. If e={i,j} has |r_e|<R/8,
any active angle on another edge incident to i or j would put the associated
triple below R by the triangle inequality. Such angles are identically zero
in that tube within Omega. Therefore, there,
- chi_M=0 if e is not in M;
- chi_M is independent of r_e if e is in M, although it may vary in a DISJOINT
  pair variable.

For a source generator eta v_P, set C=P intersect M. The smooth compact multiplier
F=eta chi_M is normal-flat at retained C contacts, vanishes near dropped P minus M
contacts, and avoids the target M minus P contacts. The statements hold globally
for F because eta has compact support inside Omega; outside that support it is
zero with a positive boundary margin. Lemma(3) and its added-contact extension
therefore prove

    chi_M u belongs to D(A_M), every u in D0.            (4)

This is actual operator-domain membership, not an informal form assignment.

## 5. Singular IMS and an explicit lower bound

For u in D0, the off-contact expression and tensor compatibility give the L²
identity

    A_M(chi_Mu)=chi_M Su-2kappa grad_X chi_M dot grad_Xu
                         -kappa(Delta_X chi_M)u.

Although u need not be H¹, the first-derivative contraction is L². In a source
matching's coordinates the relative contributions are
2 grad_(r_e)chi_M dot grad_(r_e)u. Their coefficients vanish near retained
source contacts, and the entire localizer vanishes near dropped contacts.
A derivative of a singular factor thus occurs only away from its own singularity.
An undifferentiated disjoint singular factor remains L², so its presence does
not invalidate the tensor derivative bound. Center/unmatched derivatives have
the smooth-factor control already used in lemma(3).

Summing the operator identities and the partition derivatives yields

    <u,Su>=sum_{M in Mset}<chi_Mu,A_M chi_Mu>
                  -integral_Omega W_loc|u|²,
    W_loc=kappa sum_M|grad_X chi_M|².                    (5)

All ambient functions here are zero extended. Compact interior support creates
no hard-boundary jump, and no divergent unrenormalized gradient integral has
been substituted for a contact energy.

The full product partition derivative identity is
sum_J|grad chi_J|²=sum_e|grad theta_e|². Nonmatching functions and derivatives
vanish on the open allowed region. At most two angles vary there, each pair
distance having total squared particle-coordinate gradient2. Hence

    0<=W_loc<=4kappa L²=256pi²kappa/R².

The matching operators are bounded below by-|M|E_s and |M|<=2. Equation(5) gives

    <u,Su>>=-B_4||u||²,
    B_4=4kappa*1_{s>0}/s²+256pi²kappa/R².               (6)

These constants follow from the specified SMOOTH cutoff, not the smaller
piecewise-linear lattice constant. No numerical optimization is performed.

## 6. Specified global N=4 Friedrichs candidate

The initial S is densely defined, symmetric and semibounded. Its Friedrichs
extension H^D_4 is self-adjoint, with closed form the completion of
q0[u]=<u,Su> on D0 under [q0[u]+(B_4+1)||u||²]^(1/2). This is a form norm,
not a claim that D0 is an operator graph core for the extension.

The hard-boundary prescription is this closure of the declared compact-interior
singular class. It contains the explicit double-contact initial vectors in its
operator domain because the extension extends S, with their assigned action.
It is therefore not a free Dirichlet replacement. The set of all channels and
admissible multipliers is permutation invariant, as is the off-contact action.
The closed form and operator commute with finite particle permutations and
have a bosonic reducing restriction.

The operator definition did not use the auxiliary matching partition. Every
admissible partition giving(5) reproduces the same intrinsic q0 and hence the
same closure; no partition-induced potential is introduced. This specifies one
continuum candidate, not uniqueness among all self-adjoint contact-boundary
extensions or selection by the hard lattice.

No maximal boundary/intersection trace characterization, equality with an
independently defined global local-energy domain, boundary-density theorem,
global lattice liminf/recovery or convergence theorem is proved here. The
local one-matching recovery result remains restricted to its earlier class.
Global bosonic construction here follows from the newly proved channel symmetry,
not from naively symmetrizing the earlier isolated local recovery. No arbitrary-N,
thermodynamic, empirical or TOE conclusion follows.

## 7. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_four_particle_double_contact_2026-09-13.md`: tensor domains, operator
  cores, mixed-charge warning and local matching recovery.
- `cubic_three_particle_continuum_2026-09-13.md`: relative contact operator,
  intrinsic/Friedrichs construction pattern and smooth partition normalization.
- `cubic_three_body_exclusion_2026-09-13.md`: hard geometry and matching argument.
The cross-matching tensor multiplier/symmetry argument above is a separate new
proposed proof; no earlier note is claimed to have established it.

Baseline681tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-four-particle-continuum-_n1oypq5/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 8. Independent mathematical review

Reviewer a307845f5a56bceb7 cleared the construction with no mathematical blocker,
using the established relative/tensor inputs. They checked the restricted
multiplier lemma, free-complement H² regularity, added-contact zero traces,
admissible cutoffs, symmetry through shared matching contacts, matching-localizer
domain transfer, L² singular IMS identity, lower bound and bosonic Friedrichs
restriction. No required repair or changed assumption was identified.

The clearance is limited to this specified continuum candidate. Boundary density,
maximal contact-domain characterization, uniqueness among all extensions and
global lattice convergence remain unproved. The reviewer used direct reads only;
no execution or numerical evaluation was performed. Original proof-draft-v1.md
and its hash are preserved, with no mathematical constant or physical hypothesis
changed. Independent auditor aaf5220d5d5046f5c confirmed all681baseline files
unchanged, exactly one note added and682current files matching final-hashes.json.
Mathematical sections1–6 are unchanged from the submitted draft; the note matched
pre-audit-note.md. This is saved-document/hash clearance, not mathematical
recertification or evidence of absence of unrecorded execution. The audited
final-hashes.json is retained; closing-hashes.json binds final audit-status edits.
