# Hard-supported recovery for the three-particle lattice contact model

2026-09-13. Status: hard-supported recovery independently cleared;
independent saved-document/hash audit cleared.
This bounded step supplies a candidate recovery upper bound. It does not by
itself announce a reviewed full-convergence or dynamics theorem.

## 1. Fixed setting and recovery target

Retain supplied R,kappa>0, finite s!=0, the original tuned pair coupling, and
hard N=3 exclusion with its original threshold kinetic diagonal. Let q_a be the
hard lattice form, q^D the specified continuum Friedrichs form, and C_s its
initial compact singular domain/form core. C_s is not assumed to be a graph
core of the Friedrichs extension.

Use the physical ordered-particle cell embedding I_a of the preceding liminf
note: amplitude a^(-9/2) on centered cells of side a, with forbidden nodes
extended by zero. The established inputs are physical-cell generalized strong-
resolvent convergence for each COMPLETE ambient free or one-pair-plus-spectator
operator, their common lower bounds, and the shifted hard lower-limit theorem.
No ambient recovery is assumed automatically to satisfy hard support.

We seek for every v in D(q^D) allowed lattice psi_a such that

    I_a psi_a ->v strongly,
    limsup_(a down to0) q_a[psi_a]<=q^D[v].              (1)

The proof starts on every finite vector in C_s, retains cross terms between
contact channels, and then diagonalizes in the form norm. No singular vector
is sampled at its1/r collision.

## 2. Exact finite-lattice cutoff identity

On the complete ordered three-particle lattice write
A_(alpha,a)=T_a+V_(alpha,a), where T_a=(kappa/a²)(18I-Adj).
For alpha=0 the potential is zero; for alpha=e it is the single original
contact g_a1_(Sigma_e). Let gamma be real smooth compactly supported in R^9,
and gamma_a its values at lattice nodes. Expanding the form directly gives

    a_(alpha,a)[gamma_a u]
      =Re<gamma_a²u,A_(alpha,a)u>
        +(kappa/a²)sum_{unordered edges{n,m}}
          (gamma(an)-gamma(am))² Re(conjugate(u(n))u(m)). (2)

The correction has a PLUS sign and need not itself be nonnegative. The diagonal
contact potential cancels in the difference exactly, so its diverging coefficient
does not enter an estimate. All edges are those of the complete graph, before
any zero support is imposed.

Suppose I_a u_a->u and I_a A_(alpha,a)u_a->F strongly in L². Put U_a=I_a u_a.
For one of the nine positive coordinate directions l, the corresponding edge
correction is

    kappa integral d_(l,a)(X)²
             Re(conjugate(U_a(X))U_a(X+a e_l)) dX,

where d_(l,a) is cellwise [gamma(a(n+e_l))-gamma(an)]/a. Smooth bounded second
derivatives give uniform convergence d_(l,a)->partial_l gamma. Also
||U_a(.+ae_l)-u||_2<=||U_a-u||_2+||u(.+ae_l)-u||_2->0.
Products therefore converge in L¹. Summing the finite nine directions gives

    a_(alpha,a)[gamma_a u_a]
       ->Re<gamma²u,F>+kappa integral|grad gamma|²|u|². (3)

In particular the absolute correction is bounded by
9kappa||grad gamma||_infinity²||u_a||², uniformly in a. This is a proved lattice
cutoff statement on graph-recovery vectors, not an assumed multiplier theorem
from ambient Mosco convergence alone.

## 3. One-channel hard recovery

Let v_alpha in D(A_alpha) be compactly supported with a positive distance from
the hard boundary. A pair-channel vector must avoid all other collision surfaces;
a free vector avoids every collision surface. These conditions hold for the
channel-grouped C_s vectors used below.

Choose lambda larger than the common ambient lower-bound magnitude and put

    f_alpha=(A_alpha+lambda)v_alpha,
    u_(alpha,a)=(A_(alpha,a)+lambda)^(-1)I_a*f_alpha.

The established physical-cell ambient resolvent limit gives I_a u_(alpha,a)->v_alpha.
Since the complete cell-average projections P_a=I_aI_a* tend strongly to I,

    I_a A_(alpha,a)u_(alpha,a)
       =P_a f_alpha-lambda I_a u_(alpha,a)->A_alpha v_alpha. (4)

Choose a fixed real smooth zeta_alpha compactly supported in Omega_R, equal1
on a neighborhood of supp v_alpha, and avoiding the unwanted collision surfaces.
Define w_(alpha,a)=zeta_(alpha,a)u_(alpha,a). Every nonzero node is hard-allowed,
so it is an exact admissible lattice vector. Its support has no unwanted pair
potential, and all original boundary bonds remain in the zero-extended kinetic
form. Consequently q_a[w_(alpha,a)]=a_(alpha,a)[w_(alpha,a)].

Equation(3) applies. Because zeta_alpha=1 near the target support, its gradient
term vanishes on v_alpha and the first term becomes the target energy. Thus

    I_a w_(alpha,a)->v_alpha,
    q_a[w_(alpha,a)]-><v_alpha,A_alpha v_alpha>.         (5)

Cutoffs flat normally near contact may be chosen, but no normal-flat operator-
domain invariance is required for(2) at finite a. The continuum target itself
is unchanged by zeta_alpha. No extra boundary energy is discarded.

## 4. Strong discrete gradients away from contact

Cross terms require more than individual energy convergence. Take a smooth
compact gamma supported away from the selected contact. For the channel-core
vectors here, gamma v_alpha is smooth and regular there. Contact vanishes on
gamma_a u_(alpha,a), so(3),(4) and ordinary integration by parts imply

    t_a[gamma_a u_(alpha,a)]
      ->kappa||grad(gamma v_alpha)||².                 (6)

The embedded vectors converge strongly to gamma v_alpha. Their forward difference
vectors D_(a,l)F=[F(.+ae_l)-F]/a have squared summed L² norm equal to t_a/kappa.
Discrete integration by parts against smooth compact tests identifies any weak
limit of those bounded difference vectors with grad(gamma v_alpha). Boundedness
also guarantees weak subsequential compactness in the Hilbert space; uniqueness
of the weak limit gives weak convergence of the full family. Equation(6) gives
convergence of norms, so

    (D_(a,l) I_a(gamma_a u_(alpha,a)))_(l=1..9)
      ->grad(gamma v_alpha) strongly in L²(R^9;C^9).   (7)

Only regions separated from contact are treated by ordinary gradient energy.
No global H¹ bound for a singular whole state is inferred.

## 5. Finite sums and all cross-channel interference

Take an arbitrary v in C_s and group its finite generators as
v=v_0+sum_e v_e. The regular sum v_0 is supported away from all contacts.
Each v_e is a finite sum of compact pair-e singular generators, belongs to
D(A_e), and is supported in a finite union of compact tubes avoiding the other
contact surfaces and hard boundary. Use ONE ambient resolvent recovery for each
grouped channel as in section3. All same-channel cross terms are already inside
that recovery, not treated by adding separate energy bounds.

Let F_alpha=supp zeta_alpha. For distinct channels alpha,beta the compact
intersection F_alpha intersect F_beta is separated from EVERY collision surface:
each pair cutoff avoids the other's surface, and the free cutoff avoids all.
Choose eta_(alpha,beta) smooth compact in Omega_R minus Sigma, equal1 near the
intersection. If the intersection is empty, the compact sets have a positive
separation and cross terms vanish for sufficiently small a.

For nonempty intersection and sufficiently small a, expansion into diagonal
and nearest-neighbor products gives the EXACT identity

    q_a(w_(alpha,a),w_(beta,a))
      =t_a(eta_a w_(alpha,a),eta_a w_(beta,a)).           (8)

Every nonzero same-node product lies in the intersection, where eta=1. Every
nonzero cross-edge product has one endpoint in each F, at distance a. Both
endpoints eventually lie in the neighborhood where eta=1: otherwise compactness
would give a subsequential common limit in the intersection outside that
neighborhood. No contact bilinear term survives since the supports share no
point on any contact surface. This argument retains the complete kinetic
diagonal; it does not delete edges crossing forbidden nodes.

Apply(7) with gamma_alpha=eta zeta_alpha and gamma_beta=eta zeta_beta, both
supported away from their contacts. Inner products of the strongly convergent
discrete gradients give

    q_a(w_(alpha,a),w_(beta,a))
      ->kappa integral conjugate(grad(eta v_alpha)) dot grad(eta v_beta)
      =q^D(v_alpha,v_beta).                             (9)

For the last equality, both core functions have their common overlap separated
from contacts; S acts there as the ordinary free differential operator. eta=1
on a neighborhood of all relevant products, so regular integration by parts
identifies the pairing and introduces no contact boundary term. The finite
sesquilinear expansion of psi_a=w_(0,a)+sum_e w_(e,a), together with(5),(9), now
proves

    I_a psi_a->v,       q_a[psi_a]->q^D[v],  v in C_s.   (10)

Every psi_a is hard-supported. This is recovery for a whole finite combination,
not merely for separate singular generators.

## 6. Bosons and form-domain diagonalization

If v is symmetric, apply the finite permutation average P_sym to psi_a. The
cell embedding intertwines permutations, preserving strong convergence to v.
Use the common positive shift sigma from the lower-limit note. Permutation
invariance makes P_sym reducing for b_a=q_a+sigma||.||², hence

    b_a[P_sym psi_a]<=b_a[psi_a].

Strong norm convergence implies the desired unshifted energy limsup bound.
The already cleared shifted lower limit supplies the reverse inequality on
this strongly convergent sequence if equality is needed. No symmetry of an
individual labelled channel was assumed.

For general v in D(q^D), choose v_m in C_s converging in the shifted form norm,
using the symmetric core when appropriate. Then v_m->v in L² and q^D[v_m]->q^D[v].
For each m use the preceding recovery and choose decreasing epsilon_m down to0
so that for0<a<epsilon_m,

    ||I_a psi_a^(m)-v_m||<1/m,
    q_a[psi_a^(m)]<=q^D[v_m]+1/m.

Choose m(a) tending to infinity sufficiently slowly that a<epsilon_(m(a)).
The diagonal sequence gives(1), preserving hard support for every a. It uses
only form-core density, never an assumption that C_s is a graph core. The
physical R, pair tuning and kinetic reference remain exactly the earlier ones.

## 7. Scope and next synthesis gate

If independently cleared, (1) supplies the hard-supported recovery half paired
with the separately reviewed shifted lower limit. A subsequent synthesis must
state precisely the generalized form convergence, varying-range projections,
resolvent consequence and any unitary-dynamics consequence. This note itself
makes no such reviewed synthesis claim. In particular it does not assert norm-
resolvent convergence, uniformity in particle number, arbitrary-N continuum
theory, wave-operator convergence or empirical validation.

The construction preserves a specified N=3 continuum Friedrichs candidate and
its hard boundary, not a newly fitted interaction. The prior rejection of
bounded-strength stabilizers and all earlier failures remain unchanged.

## 8. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_three_particle_lattice_liminf_2026-09-13.md`: physical-cell ambient
  convergence, common shifts and lower-limit theorem; ambient recovery alone
  did not supply hard support.
- `cubic_three_particle_continuum_2026-09-13.md`: actual ambient operator membership
  of the grouped singular generators, intrinsic action and Friedrichs form core.
- `cubic_three_particle_boundary_density_2026-09-13.md`: whole-state form-domain
  identification and closure semantics.
- `cubic_two_particle_operator_convergence_2026-09-13.md`: retained normalization
  and full two-particle generalized resolvent convergence.
The cutoff graph-recovery and off-contact cross-gradient arguments above are
new proposed estimates, not circular uses of the desired hard recovery.

Baseline678tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-lattice-recovery-urskgbrh/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 9. Independent mathematical review

Reviewer ac7c355b033d526eb cleared the hard-supported recovery upper half with
no repair required. They checked the exact cutoff sign/coefficient, graph
resolvent recovery, retained boundary diagonal, off-contact strong gradients,
both endpoints in cross-edge localization, all finite-sum interference terms,
positive-form symmetry averaging and form-core diagonalization. No unresolved
cross-energy gap was found.

This clearance does not include a full generalized-Mosco synthesis or its
resolvent/dynamics consequences. The reviewer used direct reads only. The
original submitted proof-draft-v1.md and hash remain preserved externally;
no mathematical constant or physical hypothesis changed. Independent auditor
a1ea3a5bf894ec935 confirmed all678baseline files unchanged, exactly one new note
and679current files matching final-hashes.json. The mathematical body is unchanged
from the submitted draft; current note matches pre-audit-note.md. This is
document/hash clearance, not mathematical recertification or proof of absence
of unrecorded execution. The audited final-hashes.json is retained; closing-hashes.json
binds the final audit-status edits.
