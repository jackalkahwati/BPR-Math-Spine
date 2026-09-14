# Four-particle whole-state boundary density with shared contacts

Draft initiated 2026-09-13; written 2026-09-14. Status: specified N=4 global
form-domain equality independently cleared; preservation audit cleared.
Filename retains the initiated-step date.
This concerns the specified global FORM domains, not lattice convergence or
maximal operator traces at double-contact intersections.

## 1. Domains and the already justified inclusion

Retain supplied R,kappa>0 and finite s!=0. Let Omega=Omega_R^(4) forbid every
triple of diameter<=R, and let P range over the ten matchings: empty, six single
pairs and three double pairs. The ambient operators A_P and the intrinsic
initial domain D0 are those in the four-particle continuum note. Write
E_s=2kappa*1_{s>0}/s², and choose lambda>2E_s with lambda>0. Distinguish the
completed ambient FORM space

    Vcal_P=D((A_P+lambda)^(1/2))

from the algebraic operator-generator class in that note. Its squared norm is
a_P[v]+lambda||v||². No mixed-charge continuity in the sum form norm is assumed.

Let chi_P be the fixed smooth matching partition, with radial angle pi/2 below
r0=R/8 and0 above r1=R/4. On physical allowed configurations only matching
components survive. Define, on common L²(R^12),

    V_max={U: U=0 a.e. on Omega^c, chi_P U in Vcal_P for all P},
    ||U||_max²=sum_P(a_P[chi_P U]+lambda||chi_P U||²).   (1)

This is global physical zero-extension finite energy, not merely locally finite
interior energy. Let V_min be the prescribed Friedrichs form closure of D0.

The lower bound A_P>=-|P|E_s and the norm partition give
||U||_max²>=(lambda-2E_s)||U||². If U_n is Cauchy in(1), its L² limit is physically
supported, while each localized limit is in the corresponding complete ambient
form space and equals chi_P U. Hence V_max is complete. Subtracting the bounded
IMS multiplier W_loc gives a closed semibounded form

    q_loc[U]=sum_P a_P[chi_P U]-integral W_loc|U|².

After adding a sufficiently large positive L² shift, its norm is equivalent to
(1). The established singular IMS identity agrees with this form on D0, so
V_min is its closed subspace obtained by closing D0. Thus V_min subset V_max.
The new target is the reverse density, retaining shared singularities.

## 2. Completed-form multiplier and matching-transfer estimates

### Same matching

Let f be real C_b², independent of every selected relative variable r_e in a
uniform tube about r_e=0 for e in P. Then multiplication is bounded on Vcal_P:

    ||fv||_(Vcal_P)<=C_lambda(||f||_infinity+||grad_X f||_infinity)
                                      ||v||_(Vcal_P). (2)

First prove this on the established algebraic tensor operator core K_P. Its
vectors are compactly supported. The restricted multiplier lemma from the
continuum construction applies after multiplication by an auxiliary compact
normal-flat plateau containing that support. It places fv in D(A_P). Rescale
f by a positive constant to f0 with ||f0||<=1/2, and put j=sqrt(1-f0²). Both
multipliers are normal-flat at the selected contacts. Operator IMS gives

    a_P[f0v]+a_P[jv]=a_P[v]
            +kappa integral(|grad_X f0|²+|grad_X j|²)|v|².

Relative derivatives of singular factors occur only away from their own contact;
an undifferentiated disjoint singular factor remains L². This is not a divergent
bare-gradient calculation. Add the positive lambda shift and drop the nonnegative
j term. The bound on grad j gives(2) after rescaling. Ambient form-core density
extends it to all Vcal_P. This also provides a bound proportional to the displayed
W^(1,infinity) size for a family of multipliers tending to zero; the second-
derivative hypothesis justifies the core identity, not the final norm constant.

### Dropping and adding compatible contacts

If C subset P and f additionally vanishes in uniform tubes around every
contact in P minus C, then f:Vcal_P->Vcal_C is bounded. On core vectors the
restricted operator lemma puts fv in both domains, with A_P(fv)=A_C(fv).
Their forms agree. Apply(2) to differences of core approximants and use
closedness in Vcal_C to extend this assertion. All contacts in C are retained.

For C subset M there is also the form-domain inclusion

    Vcal_C subset Vcal_M,     a_M[w]=a_C[w] on Vcal_C.   (3)

An added pair involves only coordinates unmatched in C. The free complementary
kinetic form supplies their first derivatives in L². In those relative
coordinates the free H¹ form lies inside the contact form with ZERO charge,
and its energy is exactly the free energy. Tensoring with the retained contact
factors and completing the positive form norms proves(3). It asserts no
operator-domain inclusion for arbitrary smooth zero-charge functions with
nonzero trace. That stronger assertion would be false.

Combining these facts, for C=P intersect M a globally source-normal-flat f
which vanishes near source contacts P minus M defines a bounded map

    f:Vcal_P -> Vcal_M.                                 (4)

The estimate depends on its W^(1,infinity) size, the common shifts and fixed
parameters, not a mixed-charge extraction bound. The tensor sum form, not a
free-tangential H^(1/2) prescription at double contact, is used throughout.

## 3. Ambient replacements for physical matching multipliers

Physical matching cutoffs need not be normal-flat on arbitrary ambient states.
We supply a replacement before applying(4). Choose a fixed small epsilon>0
with4epsilon<r0 and R-4epsilon>r1. For source matching P, let C_P be a smooth
collapse map in its disjoint relative coordinates: r_e is sent to0 for
|r_e|<=epsilon, left unchanged for |r_e|>=2epsilon, and contracted smoothly in
between without increasing its norm. Centers and unmatched coordinates remain
unchanged. These relative maps commute and have uniformly bounded derivatives.

For1<=t<=2 define

    f_(MP,t)(X)=chi_M(tX)chi_P(X),
    F_(MP,t)=f_(MP,t) composed with C_P.                 (5)

The chi functions here are the global all-subset product functions restricted
to matching indices. F is globally normal-flat at all source contacts and
vanishes near every source contact not in M, by the zero cosine at a collapsed
unselected pair.

On PHYSICAL Omega, collapse preserves the cutoff values in(5). If a source
pair is changed, its distance is<2epsilon. For any third particle, both incident
distances exceed R-2epsilon: otherwise the short pair and triangle inequality
would make every distance in that triple<=R. After both selected collapses,
all affected cross distances still exceed R-4epsilon>r1. Their angles remain0,
also after outward multiplication by t. Each changed selected distance remains
below r0 even after t<=2, and its angle remains pi/2. Thus

    F_(MP,t)(X)=chi_M(tX)chi_P(X), X in Omega.           (6)

The collapse map need not itself preserve Omega; preservation of these cutoff
values on physical inputs is the statement used. For fixed epsilon and cutoffs,

    ||F_(MP,t)||_(W1,infinity)<=C,
    ||F_(MP,t)-F_(MP,1)||_(W1,infinity)<=C|t-1|.         (7)

To see the uniform t-variation, each radial angle changes only at bounded
relative distance; differentiation in t and X therefore has uniformly bounded
coefficients, independent of unbounded common centers. Finite products and
composition with the fixed bounded-derivative collapse map preserve the bounds.
By(4), F_(MP,t) maps Vcal_P boundedly to Vcal_M, and their differences have
operator norm O(|t-1|). The same dropped-contact conditions hold for the difference.

## 4. Dilation in ambient tensor form spaces

Define the unitary full physical dilation D_tU(X)=t^(-6)U(X/t). On each Vcal_P
it is uniformly bounded for1<=t<=2 and strongly continuous at t=1.

Here is a tensor proof avoiding an incorrect free-tangential double-charge
space. On one relative factor with Hilbert-valued spectator coefficients, use
u=phi+G_mu Q and fixed-mu relative form norm. Under unitary relative dilation,
the transformed coefficient is t^(-1/2)Q, and the singular function uses G_(mu/t).
Returning to G_mu adds (G_(mu/t)-G_mu)t^(-1/2)Q to the regular H¹ part. Fourier
subtraction bounds that correction by C|t-1|||Q|| in the relative H¹ norm,
uniformly for t in[1,2]. Thus relative dilation is bounded and strongly continuous
in its shifted relative form space. The other coordinates are spectators for
this estimate; their unitary dilation does not change the Hilbert-valued norm.
Free center/unmatched-coordinate energies transform by the usual t^(-2) factor.
Each partial shifted relative form and free kinetic form is controlled by the
positive shifted matching tensor-sum form. Summing these finitely many estimates
gives uniform boundedness on Vcal_P; finite tensor-core approximation gives
strong continuity of the full dilation there.

More explicitly, with Q_e the single-relative charge map into L² of its
spectators, the fixed-s energy scaling identity is

    a_(P,s)[D_t v]=t^(-2)a_(P,s)[v]
      +(2kappa/(4pi s))(t^(-2)-t^(-1))
                              sum_{e in P}||Q_e v||².  (8)

Derive it first on finite tensors from the relative form formula and change
of variables, then extend by the bounded form estimates. Each Q_e is controlled
by its partial relative form and hence the matching form. No simultaneous
mixed-charge map is assumed continuous in that form norm. Equation(8) explains
why operator contact conditions need not be dilation invariant while fixed-s
form convergence holds. The physical scattering length is not changed.

## 5. Whole-state dilation in the fixed global comparison domain

For U in V_max put v_P=chi_PU. On physical support the partition gives
U=sum_P chi_P v_P. From(5)–(6), exactly

    chi_M D_t U=sum_P D_t(F_(MP,t)v_P).                 (9)

Every term belongs to Vcal_M. Physical support is preserved by outward dilation,
since every triple diameter scales by t>1. Subtracting the t=1 formula gives

    chi_M(D_tU-U)=sum_P[(D_t-I)F_(MP,1)v_P
                       +D_t(F_(MP,t)-F_(MP,1))v_P].

Ambient strong continuity treats the first terms, and(7) with matching transfer
treats the second. The matching family is finite, so

    D_tU in V_max, ||D_tU-U||_max ->0 as t down to1.     (10)

Unlike the N=3 shortcut, this estimate does NOT put every mismatch in global H¹.
A mismatch can retain one contact while changing a disjoint pair's cutoff;
its remaining matching form is kept explicitly in(9).

Every triple diameter is sqrt(2)-Lipschitz in the full R^12 metric. Therefore
for W=D_tU,

    dist(supp W,Omega^c)>=(t-1)R/sqrt(2).               (11)

This is an essential-support statement for the WHOLE state; regular and Green
components need not separately obey it. The approximation does not replace
the original physical exclusion radius by a new parameter.

## 6. Compactification and minimal-domain approximation

Write v_P=chi_P W for fixed t>1. For P nonempty choose expanding smooth cutoffs
b_(P,L) in pair centers and unmatched coordinates, independent of selected
relative variables. For the empty matching use a cutoff in all coordinates.
The same-matching estimate(2) is uniform in L. Convergence on compact tensor
cores and then density give b_(P,L)v_P->v_P in Vcal_P.

Define

    W_L=sum_P chi_P b_(P,L)v_P.

Each summand is compact: selected relative distances are bounded by the sine
factors in chi_P, and remaining coordinates by b_(P,L). It retains the physical
boundary margin(11). It also avoids unselected collision surfaces: their cosine
factor vanishes throughout the fixed r0 plateau. For a physically supported
V in Vcal_P, the assembly map V->chi_PV is bounded into V_max using(4)–(6) at
t=1 for each target M. Hence W_L->W in V_max. Each individual summand is also
in its OWN ambient Vcal_P: set H_P=chi_P composed with C_P. The same collapse
geometry shows H_P=chi_P on Omega, and H_P is globally source-normal-flat.
Thus chi_P b_(P,L)v_P=H_P b_(P,L)v_P belongs to Vcal_P by(2), without dividing
by a cutoff or inferring own-channel membership from V_max alone.

It remains to place each compact summand into V_min without erasing charges.
Let w in Vcal_P have compact support K inside Omega and away from every contact
not in P. Choose eta smooth compact in that same open set, equal1 near K and
normal-flat at ALL selected contacts. The previously proved plateau/collapse
construction provides it: take its displacement below the compact plateau and
support margins, using disjoint relative coordinates of P.

Take ambient tensor operator-core v_n->w in Vcal_P. Then eta v_n belongs
literally to the original D0. For every target matching M, chi_M eta is globally
normal-flat at source contacts and vanishes near P minus M: on support eta
this follows from physical matching geometry, and outside its compact interior
support the product is zero. Matching transfer(4) gives

    ||eta v_n-w||_max<=C_eta||v_n-w||_(Vcal_P)->0.       (12)

In particular no unsupported equality between maximal and minimal domains has
been used to obtain these approximants. Apply(12) to every compact summand of
W_L. Then W_L belongs to V_min. Since V_min is a closed subspace of V_max,
first W belongs to V_min and then, by(10), U belongs to V_min.

We have proved the candidate equality

    V_min=V_max,
    q^D[U]=sum_P a_P[chi_PU]-integral W_loc|U|².         (13)

Permutation averaging of approximants preserves D0 and the equivalent form
norm, giving the bosonic restriction. This concerns the GLOBAL physical
zero-extension comparison domain in(1), not a merely locally finite-energy class.

## 7. Scope and remaining convergence work

If independently cleared, (13) identifies the specified four-particle Friedrichs
FORM domain with the matching-localized finite-energy comparison domain. The
proof retains shared and double-contact singularities; it does not impose a
false free-tangential charge condition or assume mixed-charge form continuity.

It supplies no maximal OPERATOR-domain trace characterization, uniqueness among
all possible contact-boundary extensions, arbitrary-N theorem, or lattice limit.
Four-particle discrete lower-limit and hard recovery estimates still require
proof; cross-matching energy overlaps can retain a common contact and must not
be treated by contact-free gradient arguments. No empirical validation or TOE
completion follows from this domain theorem. The same R,s,kappa and hard
constraint are retained throughout.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_four_particle_continuum_2026-09-13.md`: restricted operator multiplier
  lemma, common-contact symmetry, intrinsic D0 and Friedrichs candidate.
- `cubic_four_particle_double_contact_2026-09-13.md`: ambient tensor domains and
  operator/form-core distinction at double contact.
- `cubic_three_particle_boundary_density_2026-09-13.md`: dilation strategy; its
  global-H¹ mismatch argument is NOT transplanted to shared-contact regions.
- `cubic_three_particle_interior_contact_2026-09-13.md`: form multiplier method
  and the correction excluding invalid operator-core regular functions.
The completed-form matching transfer is the new proposed ingredient here.

Baseline682tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-four-particle-boundary-density-_tq221mj/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 9. Independent mathematical review

Reviewer a108633dd1e7971e4 cleared the specified N=4 GLOBAL form-domain equality
with no mathematical blocker. They checked linear W^(1,infinity) multiplier
control, dropping/adding contacts at FORM level, collapse-map geometry including
two selected pairs, tensor fixed-s dilation, exact partition assembly and
noncircular compact approximation. The requested bookkeeping clarification was
added: each compact summand belongs to its own ambient matching form domain
using the globally normal-flat H_P replacement, not division by chi_P.

Clearance excludes maximal operator/intersection traces, uniqueness among all
extensions, arbitrary N, lattice convergence and empirical claims. Direct reads
only were used. Original proof-draft-v1.md and its hash remain preserved; physical
hypotheses and mathematical constants are unchanged. Independent auditor
ae3ced9fed0c2bfcf confirmed all682baseline files unchanged, exactly one added
note and683current files matching final-hashes.json. The note matched its
pre-audit snapshot; draft changes are review/status text and the explicit H_P
own-channel membership clarification only. This is preservation/scope clearance,
not mathematical recertification or evidence of absence of unrecorded execution.
The audited final-hashes.json is retained; closing-hashes.json binds final
audit-status edits.
