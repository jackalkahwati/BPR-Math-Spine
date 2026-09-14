# Global boundary density at each fixed finite particle number

2026-09-14. Status: global form-domain equality at each fixed finite N independently
cleared; independent saved-document/hash audit completed.
This is a fixed-finite-N FORM-domain theorem, not lattice convergence or a
thermodynamic/uniform-in-N approximation theorem.

## 1. Domains and the easy inclusion

Retain R,kappa>0, finite s!=0, E_s=2kappa*1_{s>0}/s². Fix an integer N>=0,
m_N=floor(N/2), lambda_N>m_N E_s with lambda_N>0, and delta_N=lambda_N-m_N E_s.
Omega_N excludes every triple of diameter<=R; for N<3 no exclusion is imposed.
The N=0 sector has the usual one-dimensional vacuum convention.

For each matching P retain the ambient finite tensor operator A_P from the
finite-particle continuum construction, lower bounded by-|P|E_s. Distinguish
its completed form space Vcal_P=D((A_P+lambda_N)^(1/2)) from its algebraic
operator-core V_P. The intrinsic initial domain D_(0,N) consists of finite
sums eta v, v in V_P, with eta compactly supported in Omega_N away from
unselected collisions and normal-flat at every selected relative collision.

Use the fixed smooth angle theta, pi/2 below R/8, zero above R/4, with
|theta'|<=L=8pi/R. Define the global product partition for ALL edge subsets J.
On Omega_N active edges form a matching, so only matching components survive
and sum_P chi_P²=1. Define

    V_max,N={U in L²(R^(3N)): U=0 a.e. on Omega_N^c,
                                     chi_PU in Vcal_P for every P},
    ||U||_max,N²=sum_P(a_P[chi_PU]+lambda_N||chi_PU||²). (1)

It is the GLOBAL zero-extended ambient-energy space, not interior-local finite
energy. Let V_min,N be the prescribed Friedrichs form closure of D_(0,N).
The target is equality of these two domains and the exact localized form.

The norm in(1) controls delta_N||U||². A Cauchy sequence therefore has a physically
supported L² limit, and each complete ambient localization has that limit's
chi_P component. Hence V_max,N is complete. The reviewed singular IMS identity
has bounded W_N<=B_N=2kappa m_N L² on Omega_N. Defining
q_loc,N=sum_P a_P[chi_PU]-integral W_N|U|² gives a closed semibounded form,
because

    q_loc,N[U]+(lambda_N+B_N)||U||²
      =||U||_max,N²+integral(B_N-W_N)|U|²

is equivalent to(1), with upper norm ratio at most1+B_N/delta_N. Agreement on
D_(0,N) shows V_min,N is the corresponding closed subspace of V_max,N. The
remaining proof is reverse density, not an assumed maximal-domain equality.

## 2. Completed-form matching transfer at arbitrary depth

Let f be real C_b², normal-flat in every selected relative variable of P in a
uniform tube. On the algebraic tensor operator core, the reviewed restricted
multiplier lemma applies after an auxiliary compact normal-flat plateau around
its support. The commutator differentiates each singular factor only away from
its own collision; any finite number of other undifferentiated singular factors
remain tensor-product L². This does not require bare gradient energy of the
whole singular vector or a simultaneous mixed-charge map.

For f nonzero rescale to f0 with ||f0||_infinity<=1/2 and j=sqrt(1-f0²).
The operator IMS identity gives

    ||f0 v||_VcalP²+||jv||_VcalP²
      =||v||_VcalP²+kappa integral(|grad f0|²+|grad j|²)|v|².

Dropping the positive j term and using delta_N||v||²<=||v||_VcalP² proves

    ||fv||_VcalP<=C_N(||f||_infinity+||grad f||_infinity)||v||_VcalP. (2)

The zero multiplier is immediate. The ambient tensor operator core is a form
core, so completion extends(2). Its final bound depends on W^(1,infinity), not
second derivatives; these only justify the initial identity. In particular
the multiplier norm tends to zero with its displayed W^(1,infinity) size.

If C subset P and f also vanishes in tubes about P minus C contacts, the
restricted operator lemma puts core products in both domains with identical
actions. Their form energies agree. Apply(2) to core differences and complete
in Vcal_C to obtain bounded contact dropping f:Vcal_P->Vcal_C. Retained
contacts are not discarded.

For C subset M there is also the form inclusion

    Vcal_C subset Vcal_M, a_M[w]=a_C[w] on Vcal_C.       (3)

Every added pair uses only labels unmatched in C. Free complementary kinetic
energy gives the relative H¹ form, which embeds with zero charge into the
contact form with the same energy. Add each disjoint compatible pair in turn,
tensor with all retained factors, and complete positive form norms to prove(3).
This is NOT an operator-domain inclusion for arbitrary smooth nonzero-trace
functions, which would be false.

Consequently f maps Vcal_P boundedly into Vcal_M when it is source-normal-flat
and vanishes near every contact in P minus M, by dropping to P intersect M
then adding. Constants may depend on N and the chosen shift but are finite.
No higher mixed-trace continuity is used.

## 3. Global ambient replacement of physical cutoff products

Fix epsilon>0 with4epsilon<R/8 and R-4epsilon>R/4. For source matching P,
collapse each disjoint r_e smoothly to0 below epsilon and leave it unchanged
above2epsilon, using radial contraction without increasing |r_e|. Leave pair
centers and unmatched coordinates fixed. Call this map C_P. Each particle
moves by at most epsilon and any pair distance by at most2epsilon, independently
of depth. The FULL displacement can be sqrt(N)epsilon; compact plateau choices
below must account for that finite-N factor.

For1<=t<=2 define

    F_(MP,t)=[chi_M(tX)chi_P(X)] composed with C_P.

This is globally source-normal-flat. If e in P is not in M, its collapsed
zero distance forces a zero cosine in the target product, so F vanishes in
that contact tube. Therefore the matching-transfer bound applies.

On physical Omega_N it agrees exactly with chi_M(tX)chi_P(X). To prove this,
if a selected pair changes then its initial length is<2epsilon. For ANY third
label, both incident distances exceed R-2epsilon, or the triple would have all
distances<=R. After all disjoint collapses those distances remain>R-4epsilon,
so their angles stay zero even after outward t scaling. Every changed selected
distance stays on the pi/2 plateau since t<=2 and4epsilon<R/8. Unaffected
distances stay unchanged. The collapse itself need not preserve Omega_N;
only these cutoff values on physical input points are used.

For fixed N, finite product differentiation gives

    sup_(P,M,1<=t<=2)||F_(MP,t)||_W1,infinity<=C_N,
    ||F_(MP,t)-F_(MP,1)||_W1,infinity<=C_N|t-1|.         (4)

Every radial t derivative contains a bounded relative distance on its transition
support. Spatial derivatives have the same property; unbounded common centers
do not enter. Composition with fixed bounded-derivative collapse maps preserves
the estimates. Finite channel counts may enlarge C_N; no uniform-in-N claim
is made. Combining with(2)–(3) gives an O(|t-1|) operator bound from Vcal_P
to Vcal_M for the difference in(4).

## 4. Fixed-s dilation on every finite matching form space

Let D_tU(X)=t^(-3N/2)U(X/t). On one relative coordinate with Hilbert-valued
spectators use u=phi+G_mu Q at fixed mu>max(0,1/s). Relative unitary dilation
transforms charge to t^(-1/2)Q and Green function to G_(mu/t). Returning to
G_mu adds t^(-1/2)(G_(mu/t)-G_mu)Q to the regular part. Fourier subtraction,
or integration of the H¹-bounded parameter derivative of G_nu on[mu/2,mu],
gives ||G_(mu/t)-G_mu||_H1<=C_mu|t-1|. Thus relative dilation is uniformly
bounded and strongly continuous in the shifted relative form norm with arbitrary
L²-valued spectators.

Spectator dilation is unitary in that partial norm; free kinetic energies
scale by t^(-2). Every partial shifted form is controlled by the positive
matching sum form, with fixed-N constants. Summing finitely many partial estimates
gives boundedness of D_t on Vcal_P. Strong continuity follows on finite tensors,
then throughout by form-core density:

    ||D_tv-v||_VcalP ->0 as t down to1.                 (5)

The fixed-s energy identity is

    a_(P,s)[D_tv]=t^(-2)a_(P,s)[v]
      +(2kappa/(4pi s))(t^(-2)-t^(-1))sum_(e in P)||Q_e v||².

Here Q_e is only single-relative charge into L² of all spectators, bounded by
that partial relative form; no multiple extracted-charge map is used. The
identity follows on tensors from the relative form and extends by the established
bounds. Dilation may change operator boundary asymptotics but is a valid
approximation in the ORIGINAL fixed-s form domain; physical s is not retuned.

## 5. Whole-state outward dilation

For U in V_max,N set v_P=chi_PU. The physical partition gives U=sum_P chi_Pv_P,
and the ambient replacements yield exactly

    chi_M D_tU=sum_P D_t(F_(MP,t)v_P).

Subtract the t=1 identity. Terms (D_t-I)F_(MP,1)v_P converge by(5); terms
D_t(F_(MP,t)-F_(MP,1))v_P converge by(4) and matching transfer. Finitely many
channels suffice, regardless of their growth with N. Outward dilation preserves
all-triple allowed support. Hence

    D_tU in V_max,N, ||D_tU-U||_max,N ->0.              (6)

For N>=3 every triple diameter is sqrt(2)-Lipschitz in the full metric, and
the dilated support has EVERY triple diameter>=tR. Its distance from the
forbidden union is at least(t-1)R/sqrt(2). This is a whole-state essential-support
margin. No separate support assertion about regular and Green parts is used.

## 6. Compactification and literal initial-domain approximation

Fix t>1, W=D_tU, v_P=chi_PW. Use expanding smooth cutoffs b_(P,L) in pair
centers and unmatched coordinates, independent of selected relative variables;
for the empty matching use all coordinates. Uniform multiplier bounds and
convergence on compact tensor cores give b_(P,L)v_P->v_P in Vcal_P. The
assembly map v->chi_Pv from physically supported Vcal_P into V_max,N is bounded
by F_(MP,1) in each target channel. Therefore

    W_L=sum_P chi_P b_(P,L)v_P ->W in V_max,N.

Each summand is compact: selected relative distances are bounded by sine factors
and the remaining coordinates by b_(P,L). It preserves the boundary margin and
avoids all unselected collision surfaces by the cosine plateau. To establish
its OWN channel membership, use H_P=chi_P composed with C_P. It equals chi_P
on physical support and is globally source-normal-flat, so
chi_P b_(P,L)v_P=H_P b_(P,L)v_P belongs to Vcal_P by(2). No cutoff division is used.

Now let w in Vcal_P have compact support K inside U_P. Choose a smooth plateau
zeta supported in U_P, equal1 near K. Flatten it using a source collapse with
radius epsilon_K satisfying sqrt(N)epsilon_K smaller than the actual support
and plateau margins. The resulting eta is compact in U_P, equal1 near K and
normal-flat at all selected contacts. Choose ambient tensor operator-core v_n
converging to w in Vcal_P. Then eta v_n belongs LITERALLY to D_(0,N).

For each target matching M, chi_M eta is globally source-normal-flat and vanishes
near P minus M. On support eta this follows from the physical all-triples
geometry and the flatness of eta; outside that compact interior support it is
zero. Matching transfer therefore gives

    ||eta v_n-w||_max,N²
      =sum_M||chi_M eta(v_n-w)||_VcalM²
      <=C_(N,eta)||v_n-w||_VcalP² ->0.                  (7)

This obtains minimal-domain approximants without using the desired equality.
Apply(7) to each compact summand of W_L. Then W_L belongs to V_min,N; closedness
first yields W in V_min,N and then, by(6), U in V_min,N. Thus for N>=3,

    V_min,N=V_max,N.                                   (8)

## 7. Small sectors, exact form and bosons

No exclusion-boundary dilation is required for N<3. At N=0 both spaces are
the one-dimensional vacuum space with zero energy. At N=1 the only ambient
channel is free H¹, whose compact smooth core proves density. At N=2 there
are empty and pair channels. Any ambient pair-form vector localizes by the
sine multiplier in that form and by the cosine multiplier in the free form
(contact dropping). Conversely their partition reconstruction is in the pair
form space by same-matching multiplication and zero-charge free-form inclusion.
Thus V_max,2 is the ambient pair form. The intrinsic domain contains its
compact ambient tensor operator core, proving the same minimal closure.

The finite-particle construction explicitly identifies N=3's initial domain
and action with the earlier prescription, and N=4 already has that prescription.
Consequently(8) agrees with the prior three- and four-particle forms, rather
than identifying an unrelated family of extensions.

The localized identity holds on D_(0,N). Passing through its now-proved dense
form closure gives, throughout V_max,N,

    q_N^D[U]=sum_P a_P[chi_PU]-integral W_N|U|²,          (9)

including polarization. Permutations preserve physical support, permute matching
channels and preserve D_(0,N) and its norm. Finite permutation averaging of
approximants proves(8)–(9) on the bosonic subspace too.

## 8. Dependence, scope and remaining work

All arguments fix N before taking approximation limits. The number of matchings
is sum_(k=0..floor(N/2)) N!/[(N-2k)!2^k k!]; its finiteness, not an N-independent
bound, is used. Transfer, dilation, assembly and approximation constants may
depend on N, lambda_N, delta_N^(-1) and auxiliary choices. Pair-distance collapse
bounds and the physical dilation margin are independent of depth, whereas
compact plateau margins use the sqrt(N) full displacement. No uniform rate for
an arbitrary form vector follows from strong dilation continuity.

The proposed theorem identifies GLOBAL zero-extended FORM domains at every
fixed finite N. It supplies no maximal operator trace description, higher
mixed-charge form map, uniqueness among all contact-boundary extensions,
lattice convergence, N->infinity limit or empirical validation. The next
separate gates are all-matching physical-cell convergence, lattice lower limit,
hard recovery and their precise fixed-N synthesis. Physical R,s,kappa and pair
tuning are unchanged.

## 9. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_finite_particle_continuum_2026-09-14.md`: arbitrary-finite-depth tensor
  construction, restricted operator lemma and intrinsic Friedrichs family.
- `cubic_four_particle_boundary_density_2026-09-13.md`: completed-form transfer
  and collapse/dilation strategy, explicitly generalized here with N-dependence.
- `cubic_four_particle_double_contact_2026-09-13.md`: tensor core and mixed-charge
  limitations that cannot be silently extrapolated.
- `cubic_three_particle_continuum_2026-09-13.md`: relative contact form normalization.

Baseline687tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-finite-particle-boundary-density-pvu3okpy/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 10. Independent mathematical review

Reviewer a500878c728b05594 cleared the specified global zero-extended FORM-domain
equality at each fixed finite N with no mathematical blocker or new finite-depth
obstruction. They checked completed-form transfer, contact dropping/adding,
collapse geometry and N-dependent constants, Hilbert-valued dilation, exact
assembly, own-channel compact membership, literal initial-domain approximation,
small sectors and bosonic restriction.

The essential hypotheses remain global ambient energies of zero extensions,
a positive shift margin and uniform source-normal-flat tubes. Clearance does
not include maximal operator traces, mixed-charge trace continuity, uniqueness
among all extensions, lattice convergence or uniform-in-N approximation. Direct
reads only were used. Original proof-draft-v1.md and its hash remain preserved;
no mathematical constant or physical hypothesis changed. Independent auditor
a2325e126e9b4b1d3 confirmed all687baseline files unchanged, exactly one added
note and688current files matching final-hashes.json. Mathematical sections1–8
are byte-identical to the draft; the current note matched pre-audit-note.md.
This is artifact clearance, not mathematical recertification or evidence of
absence of unrecorded execution. The audited final-hashes.json is retained;
closing-hashes.json binds final audit-status edits.
