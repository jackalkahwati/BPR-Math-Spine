# Three-particle continuum contact candidate: construction before convergence

2026-09-13. Status: sections1–6 independently cleared for the specified N=3
Friedrichs candidate; separate scope review confirms no overclaim;
final independent saved-document/preservation audit cleared.
Scope: one specified N=3 Friedrichs realization. Its full operator-domain trace
characterization and identification as the hard lattice limit are not asserted.

## 1. Geometry and kinetic normalization

Supply R,kappa>0 and finite real s!=0, retaining the prior physical parameters.
Let Omega_R={X=(x1,x2,x3) in R^9: max_{i<j}|xi-xj|>R}. This excludes one close
triple, not every close pair. Let Sigma_e={xi=xj} for the three label pairs e.
In Omega_R distinct Sigma_e do not intersect; their common intersection would
be a forbidden triple collision. For a coincident pair at c and spectator y,
allowedness requires |c-y|>R. Pair surfaces still approach the exclusion boundary.
No N>=4 or simultaneous-disjoint-contact construction is attempted here.

First work in L²(Omega_R), then restrict permutation-invariant constructions to
its bosonic subspace. For e={i,j}, put r=xi-xj, c=(xi+xj)/2, y=xk. The absolute
Jacobian is1 and the differential expression becomes

    T=-2kappa Delta_r-(kappa/2)Delta_c-kappa Delta_y.     (1)

This is the continuum threshold reference, not an added chemical potential.
The hard lattice's original kinetic diagonal is not modified. Construction of
a continuum operator is distinct from proving any limit of that lattice.

## 2. Relative point interaction from a closed form

Use the inner product antilinear in its first argument. For mu>0 let
G_mu(r)=exp(-mu|r|)/(4pi|r|). It lies in L² but not H¹ and satisfies
(-Delta+mu²)G_mu=delta_0 distributionally. Its Fourier transform, with the usual
unitary convention, is (2pi)^(-3/2)(|p|²+mu²)^(-1).

Fix mu>max(0,1/s). On vectors u=phi+qG_mu, phi in H¹(R³), q in C, define

    t_s[u]+mu²||u||²
      = ||grad phi||²+mu²||phi||²+c_mu|q|²,
    c_mu=(mu-1/s)/(4pi)>0.                              (2)

The decomposition is unique since G_mu is not H¹. The right side is a Hilbert
norm on H¹ direct_sum C, controls ||u||², and the map (phi,q)->u is continuous
and injective. A sequence Cauchy in this norm converges in H¹ direct_sum C and
therefore in L² to its represented vector. This proves closedness of the shifted
form, not merely algebraic positivity. H¹ is contained in the form domain and
is L²-dense. The representation theorem gives a self-adjoint operator t_s.

The representation does not depend on mu. To check it, for another nu>0,
G_mu-G_nu belongs to H² and

    (G_mu-G_nu)(0)=(nu-mu)/(4pi),
    <G_mu,G_nu>=1/[4pi(mu+nu)].

Use phi_nu=phi_mu+q(G_mu-G_nu) to represent the same u. The identity
(-Delta+nu²)(G_mu-G_nu)=(nu²-mu²)G_mu in L² shows that expanding the two shifted
norms in(2) changes their difference by precisely (nu²-mu²)||u||². This can first
be checked for smooth phi and extended by H¹ continuity. Subtracting the shift
therefore gives the same t_s, including when the displayed charge coefficient
for one positive representation parameter is nonpositive. The domain is unchanged.

Its operator domain and action can be derived directly, rather than assumed:

    u=phi+qG_mu, phi in H²,
    phi(0)=c_mu q,
    (t_s+mu²)u=(-Delta+mu²)phi.                          (3)

Indeed regular H¹ test vectors in the representation identity force the last
L² equality and phi in H²; the test vector G_mu then gives phi(0)=c_mu q, using
<G_mu,(-Delta+mu²)phi>=phi(0). Conversely these conditions satisfy the identity
against every form test vector. Sobolev embedding makes phi continuous. Thus
u=q/(4pi)(1/|r|-1/s)+o(1) near the origin. Form-domain vectors need not satisfy
this operator-domain boundary relation.

Solving(3) gives the negative-parameter resolvent

    (t_s+mu²)^(-1)f
      =R_mu f+G_mu <G_mu,f>/c_mu,
    R_mu=(-Delta+mu²)^(-1),                             (4)

whenever c_mu!=0. The pole at mu=1/s exists only for s>0 and is the eigenvector
G_(1/s), with eigenvalue -1/s². Equations(2) at arbitrary sufficiently small
admissible shifts show t_s>=0 for s<0 and t_s>=-1/s² for s>0 by letting the shift
approach its threshold. Hence h_s=2kappa t_s has lower bound

    h_s>=-E_s, E_s=2kappa*1_{s>0}/s².                   (5)

The singular coefficient normalization and (3) retain the same scattering length
as the prior scalar amplitude convention. mu is auxiliary, not a new scale.

## 3. Pair plus center and spectator: actual tensor domain

For e define the full-space operator A_e=h_s tensor I+I tensor H_tan, with
H_tan=-(kappa/2)Delta_c-kappa Delta_y. This is a self-adjoint tensor sum bounded
below by-E_s. Its closed form domain is the intersection of the two shifted
square-root domains, completed in the sum form norm, not just their algebraic
tensor product. Finite tensor sums from form cores are a form core: spectral
cutoffs of the two commuting operators reduce to bounded energy rectangles,
where finite tensor sums are dense, and removing those cutoffs converges in
the sum form norm.

For precision, take lambda>E_s and Fourier transform the six tangential variables
with unitary measure. Write epsilon(p,k)=kappa|p|²/2+kappa|k|² and
m(p,k)=sqrt((lambda+epsilon)/(2kappa)). The same closed domain is given by the
measurable unique fiber decompositions

    uhat(r,p,k)=phihat(r,p,k)+qhat(p,k)G_m(r)

for which the integral

    integral [2kappa||grad_r phihat||²
              +(lambda+epsilon)||phihat||²
              +(2kappa/(4pi))(m-1/s)|qhat|²] dp dk       (6)

is finite. It equals a_e[u]+lambda||u||². The positive charge weight is comparable
to (1+|p|²+|k|²)^(1/2), with constants depending on lambda,s,kappa. Thus this
fiber charge belongs to H^(1/2)(R^6). The regular part has the integral regularity
shown in(6). This assertion uses momentum-dependent m; it is not a fixed-mu
ansatz with unconstrained spectator functions. Direct integration of the closed
positive forms(2) proves (6), equivalently the spectral tensor construction.

Smooth compact tangential charges and the compact relative singular generators
used below lie in the operator domain of A_e: their relative regular parts are
H² satisfying(3), and tangential derivatives through second order are L². We
use only this verified class for the initial intrinsic operator, not a guessed
maximal boundary-domain description.

## 4. An intrinsic initial compact singular operator domain

Below, C_s is the initial domain of S and a form core for its Friedrichs form
closure by construction. Any use of 'core' for this generator class does not
assert that it is a graph/operator core for the Friedrichs extension; that
stronger claim is not proved.

The following specifies the candidate without any localization partition.
For pair e, take q_e(c,y) smooth and compactly supported in {|c-y|>R}. Its
support has a positive margin delta from |c-y|=R. Choose a smooth radial rho_e(r)
equal1 near0, with sufficiently small compact support that its tube over this
charge support lies compactly inside Omega_R and avoids the other two collision
surfaces. For instance a relative radius smaller than delta and than R/4 is
sufficient after further shrinking if needed. Use ANY mu>max(0,1/s) and define

    v_e=q_e(c,y) rho_e(r)[G_mu(r)+c_mu].                 (7)

It has singular expansion q_e/(4pi)(1/|r|-1/s)+O(|r|). Its fixed-mu regular
relative part is (rho_e-1)G_mu+c_mu rho_e, in H², with value c_mu at0. Thus(3)
shows it is an actual pair-contact operator vector, not just an L² singularity.

Let C_s be the finite linear span of ALL such generators for all three pairs,
all admissible charges/radii/cutoffs/mu, and C_c^infinity(Omega_R minus Sigma),
where Sigma is the union of the collision surfaces. Define S u as the L² function
obtained by applying T in(1) away from Sigma. This is NOT the distributional
Laplacian with an unremoved delta source. For(7), the only local1/r term arising
from this action is square-integrable; cutoff derivative terms are smooth away
from0, and tangential derivatives multiply L² relative functions. Hence Su is
L². If two generator descriptions give the same u almost everywhere, their
differential actions agree off Sigma and therefore as L² functions. The action
is well-defined.

The regular functions alone are L²-dense because Sigma has measure zero and
Omega_R is open. They are not the entire core: nonzero-charge vectors(7) are
explicitly included. Each vector in C_s has compact support away from the hard
boundary, and near a collision has the real contact relation (3). Integration
by parts outside small pair spheres shows S is symmetric: the contact boundary
pairing is a constant times the difference of singular-coefficient/regular-part
cross products, which vanishes when both regular parts equal -1/s times their
singular coefficients. Tangential integration has no boundary term because
charges have compact support. Distinct pair tubes do not meet at their singular
surfaces, so there is no multiple-contact boundary term. This also follows by
localizing near each collision into the symmetric full-space A_e and using
ordinary integration by parts on the regular complement.

Permutation of particle labels preserves C_s and the action; pair reversal
leaves its radial generators unchanged. Common translations and rotations also
preserve the set of all admitted generators. For bosons restrict to the symmetric
core obtained by the finite permutation average. It is dense in L²_sym(Omega_R).
There are nonzero symmetric singular generators, obtained by symmetrizing a
nonzero compact charge patch, so the bosonic core is not a free core.

## 5. Singular localization identity and semiboundedness

We now choose auxiliary cutoffs ONLY to estimate this already defined S.
Take r0=R/8,r1=R/4 and a smooth nonincreasing theta with values pi/2 below r0
and0 above r1, with |theta'|<=L=8pi/R. Such a cutoff follows by integrating a
nonnegative smooth function supported in(0,1), integral1 and supremum<=2,
rescaled to [r0,r1]. The derivative bound is deliberately larger than the average
slope; no smooth endpoint-flat cutoff with the prior smaller Lipschitz constant
is asserted.

For three label pairs form the same all-subset product partition as in the
lattice note. On Omega_R only four components survive: chi_0 for the empty
matching and chi_e for each of the three pairs. They are smooth there and
sum_alpha chi_alpha²=1. The active near-pair graph has at most one edge, because
two incident distances<R/4 would give triple diameter<R. Moreover near every
allowed collision, its chi_e is exactly1 and all other components are exactly0.
This last property is local at every collision, not just on a selected state.

For u in C_s extend chi_alpha u by zero outside Omega_R. Since u has positive
support distance from the hard boundary, this extension creates no boundary
jump. The empty component is a smooth compact regular function away from all
contacts. The pair component lies in D(A_e): near its collision it agrees
identically with the original contact germ, and every cutoff derivative is
supported away from any singularity. More explicitly, subtract the appropriate
generators(7); the remainder is smooth compactly supported away from contacts,
hence H² in all coordinates. This also proves multiplication preserves the
relevant local operator domains for the core; no general nonsmooth multiplier
claim is used.

Apply the operator commutator identity to these localized vectors. Because the
localizers are constant near collisions, its derivative terms are ordinary L²
functions away from singularities and no contact delta or new charge term is
introduced. Summing, using sum chi_alpha grad chi_alpha=0 and
sum chi_alpha Delta chi_alpha=-sum|grad chi_alpha|², gives the exact identity

    <u,Su> = a_0[chi_0 u]+sum_e a_e[chi_e u]
             -integral_{Omega_R} W_loc(X)|u(X)|² dX,
    W_loc=kappa sum_{alpha,i}|grad_i chi_alpha|².        (8)

All forms on the right are evaluated on the zero extensions. Equation(8) is
an operator-domain identity on the stated singular core, not an unrenormalized
integral of |grad u|², which would diverge. The same commutator argument works
for polarized cross terms, giving the corresponding sesquilinear identity.

Where a cutoff varies, at most one pair angle varies. Its two surviving values
are cos theta and sin theta, so their summed gradient squares are those of
theta itself. The gradients with respect to the two particle positions give
|grad_i d_ij|²+|grad_j d_ij|²=2. Therefore

    0<=W_loc<=2kappa L².

The free component is nonnegative; each pair component has lower bound(5).
The norm partition thus gives

    <u,Su> >= -B_3||u||²,
    B_3=E_s+2kappa L²
       <=2kappa*1_{s>0}/s²+128pi²kappa/R².              (9)

This bound concerns N=3 only and uses a different smooth auxiliary cutoff from
the discrete proof. It is not asserted to be the same numerical constant as
the prior all-N lattice bound.

## 6. A definite Friedrichs contact candidate

S is densely defined, symmetric and bounded below by(9). Its semibounded quadratic
form is closable, and its Friedrichs extension H^D_(R,s) is a self-adjoint
semibounded operator. Define the closed form q^D as the closure of <u,Su> on
C_s in the norm [<u,Su>+(B_3+1)||u||²]^(1/2). This is a form norm, not the graph
norm of S. The same construction on the symmetric core yields the bosonic
realization; permutation invariance makes it the corresponding reducing part
of the full realization. Invariance also follows at the form-closure level.

The hard-boundary prescription is EXACTLY this closure of compactly supported
admissible singular states away from the exclusion boundary in their interacting
energy norm. It is not an unproved maximal finite-local-energy preimage domain,
and no classical trace at the contact/boundary intersection is asserted.
Friedrichs extension is unique for this specified semibounded initial operator;
uniqueness among all self-adjoint contact-boundary extensions is not claimed.

The operator extends S, so the explicit nonzero-charge core vectors remain in
its operator domain and retain their assigned action and contact condition.
It cannot be the free Dirichlet Laplacian: these1/r singular vectors are not in
its H¹ form domain. All ordinary tests compactly supported away from contacts
belong to C_s, so symmetry of the extension implies that its action on any
operator-domain vector agrees with T distributionally in that regular interior.
The spectral theorem now gives well-defined unitary dynamics for THIS candidate.
This is not yet a convergence result for lattice dynamics.

No auxiliary partition defines S. Every partition with the stated properties
reproduces the same intrinsic pairing in(8) on the same core. Therefore using a
different partition to estimate it cannot change q^D or its extension, and there
is no partition-induced bulk interaction. Likewise C_s includes all admissible
mu and relative cutoffs, not a hidden fixed choice. For identical charges the
difference of two such germs has zero charge, is H² locally and has zero
regular trace at contact. It can be removed from a small contact tube in form
norm: near contact the difference is O(|r|), its first derivative is bounded,
and a cutoff at radius epsilon costs O(epsilon³) in gradient energy and vanishes
in L². Smooth approximation away from the tube handles the remainder. Thus
these representation changes do not introduce a physical parameter or a new
closed-form condition. One does not need to identify their difference as
literally C_c^infinity away from contact.

Equation(8) also gives a useful precise local-domain inclusion. Shift each ambient
form a_alpha by E_s+1 to make it positive. The sum of their shifted norms of
chi_alpha u equals <u,Su>+(E_s+1)||u||²+integral W_loc|u|², which is bounded
by a constant times the global shifted form norm, and conversely controls that
norm after a larger common shift. Closedness of the ambient forms therefore
extends each map u->chi_alpha u continuously from the core closure. Each local
pair component has the weighted form description(6). This supplies a form-domain
inclusion and closure-defined boundary behavior, not a maximal-domain equality
or a pointwise trace theorem for every completed-domain vector.

## 7. Exact remaining identification and convergence gates

A semibounded self-adjoint continuum CANDIDATE with nontrivial contact core and
a precise Dirichlet closure has been defined above, subject to independent review.
It is important to distinguish it from stronger unproved assertions.

### Full contact trace characterization

For all initial singular generators the scattering-length condition is explicit.
For arbitrary vectors in D(H^D_(R,s)), a complete characterization still requires
a local core/regularity theorem: prove that the chosen compact singular generators
form the full local pair-plus-spectator form core on each interior patch, localize
operator-domain vectors, and identify their singular coefficients/regular parts
in the appropriate tangential trace spaces. The weighted inclusion from section6
does not alone prove that theorem, nor supply a classical boundary trace at
Sigma_e intersect boundary(Omega_R). Boundary behavior there is specified by
closure, not by an unproved pointwise prescription. No maximal-domain or universal
extension identification is claimed.

### Lattice-to-continuum convergence

The original hard lattice has the already proved uniform lower bound, but this
note does NOT prove it converges to H^D_(R,s). The missing obligations are:
1. Full two-body embedded resolvent/form convergence, including center/spectator
   momentum, rather than the existing scalar on-shell convergence.
2. Explicit common-space embeddings for the allowed N=3 lattice sectors and
   treatment of cells crossing the hard boundary.
3. Liminf/control of the renormalized contact contribution and charge norms near
   contact endpoints at the exclusion boundary.
4. Recovery sequences for singular-core vectors, constructed using lattice
   Green structure instead of undefined sampling at1/r coincidence, and density
   to extend those sequences to the entire closed form domain.
5. A proof that the limit selects THIS Friedrichs boundary closure, rather than
   another allowable contact-boundary realization, plus localization compatibility.
6. The exact resulting resolvent/form convergence topology and a justified
   consequence for embedded unitary dynamics.

No compactness claim suppresses the unbounded center/spectator directions.
Allowed-indicator convergence, a shared lower bound, or the scalar scattering
limit do not discharge these obligations. No N>=4 assertion is made: simultaneous
disjoint contacts there require additional domain work. The work stops at these
explicit gates without replacing the model, adjusting s, or running toy checks.

## 8. Provenance and review status

Protected sources:
- `cubic_three_body_exclusion_2026-09-13.md:145-178,224-298`: lattice matching
  geometry and explicit continuum gaps, not a singular continuum IMS theorem.
- `cubic_two_body_scattering_2026-09-13.md:130-134,218-275`: relative kinetic
  normalization and scalar scattering/bound-state results, not a prior form domain.
- `cubic_contact_continuum_2026-09-13.md:66-80`: why raw diagonal contact evaluation
  is not a closed continuum interaction.
- `cubic_finite_range_three_body_2026-09-13.md:155-188`: bounded multiplier limit,
  explicitly not convergence of kinetic/contact dynamics.
Paths above are relative to doc/derivations/. Standard closed-form representation,
semibounded Friedrichs-extension and spectral tensor-sum theorems are used with
their hypotheses stated; no empirical conclusion or scientific novelty is claimed.

Baseline672tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-continuum-ranc5elf/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard, excluding ignored
caches/config/build products and .git. This note is the sole new repository file.
No scientific imports, tests, matrices or numerical physics execution are involved.
All previous files and failed/incomplete campaign evidence remain protected.

## 9. Independent review record

Reviewer a5c94488e7d266a23 cleared sections1–6 with no blocking construction or
closability error. The clearance covers the relative building block, spectator
weighted form, initial singular domain, symmetry, singular localization identity,
N=3 bound and specified Friedrichs realization. The original proof submission
is preserved externally. The review's terminology clarification was incorporated:
C_s is an initial operator domain/form core, not a proved graph core of the
Friedrichs extension. No mathematical constant or physical hypothesis changed.

Separate scope reviewer ad8acdf6ce7f1b820 found no substantive overclaim. They
confirmed that the full operator-domain contact traces, graph-core density,
maximal-domain equality, uniqueness among all contact-boundary extensions and
lattice convergence are not cleared. A stale closing review-status sentence was
corrected; this does not promote any unresolved theorem.

Both reviews used direct reads only and no scientific execution. Independent
auditor a3bae9016b4449c96 cleared the saved-document/hash audit: all672baseline
files unchanged, one new note,673total, and current inventory matching
final-hashes.json. The note matched pre-audit-note.md exactly; draft changes
were limited to status/review provenance and the initial-domain versus graph-core
clarification. No mathematical constant or physical hypothesis changed. This is
not mathematical recertification or evidence of absence of unrecorded execution.
The audited final-hashes.json is retained; closing-hashes.json binds the final
audit-status wording. The task stops at section7's explicit identification and
convergence gaps, preserving the separately constructed candidate.
No numerical checks, new interaction, publication or TOE claim are initiated.
