# Four particles: one compact-interior double-contact matching

2026-09-13. Status: local single-matching double-contact lemma independently cleared;
independent saved-document/hash audit cleared.
This note treats ONLY matching12|34 and a compact interior recovery class.
No global four-particle continuum Hamiltonian or convergence theorem is claimed.

## 1. Geometry, normalization and ambient operator

Retain supplied R,kappa>0, finite s!=0 and the unchanged tuned lattice coupling
 g_a^(-1)=-a²W/kappa+a³/(8pi kappa s). The allowed ordered four-particle region
Omega_R^(4) consists of configurations in R^12 where EVERY three-label subset
has diameter>R. At simultaneous coincidences x1=x2=c and x3=x4=d, this condition
is exactly |c-d|>R.

For the fixed matching define r=x1-x2,c=(x1+x2)/2 and t=x3-x4,d=(x3+x4)/2.
The Jacobian has absolute value1 and the free kinetic operator is

    -2kappa Delta_r-(kappa/2)Delta_c
    -2kappa Delta_t-(kappa/2)Delta_d.

Let h_s be the already constructed relative contact operator, lower bounded
by-E_s with E_s=2kappa*1_{s>0}/s². Let H_s^(2)=h_s-(kappa/2)Delta_COM.
Define the COMPLETE ambient matching operator

    A_(12|34)=H_(s,12)^(2) tensor I+I tensor H_(s,34)^(2) (1)

on L²(R^12), initially ordered particles. Both pair COM kinetic energies are
included. The spectral tensor sum is self-adjoint and at least-2E_s. It is a
local comparison operator for a matching, not the hard global four-particle
theory with every possible pair contact.

Choose beta>E_s and define positive commuting B_j=H_(s,j)^(2)+beta. If nu_u
is their joint spectral measure, the exact form and operator domains are

    D(a_(12|34))={u:integral(xi+eta)dnu_u<infinity},
    a_(12|34)[u]+2beta||u||²=integral(xi+eta)dnu_u,
    D(A_(12|34))={u:integral(xi+eta)²dnu_u<infinity}.     (2)

Positivity shows the operator domain equals the intersection of the domains
of the two pair operators. This domain is NOT specified by assigning two
independent free-tangential H^(1/2) charges. One contact charge may itself
contain the other contact's singularity.

## 2. An operator core including genuine product singularities

Fix mu>max(0,1/s), c_mu=(mu-1/s)/(4pi), and compact smooth radial rho=1 near0.
Set b(r)=rho(r)(G_mu(r)+c_mu), G_mu=exp(-mu|r|)/(4pi|r|).
The regular part (rho-1)G_mu+c_mu rho lies in H² and has value c_mu at0.
Thus b belongs to D(h_s), has charge1 and the correct contact condition.

The relative class C_rel=C_c^infinity(R³ minus{0})+span{b} is an OPERATOR core
for h_s. To verify this stronger statement here, subtract q b from any operator
vector phi+qG_mu. The remainder is H² and vanishes at0. Such H² functions can
be approximated in H² by smooth compact functions vanishing at0: first use
ordinary H² approximation, then correct the value with a fixed bump, using
continuity of point evaluation on H²(R³). For a smooth function vanishing at0,
remove a radius-epsilon ball. Near0 the function is O(|r|), its first derivatives
are bounded, and the worst second-derivative cutoff error has L² norm
O(epsilon^(1/2)). All other terms tend to0. Smooth compact functions avoiding0
therefore approximate the remainder in H², hence in the h_s graph norm since
it has zero charge. The singular coefficient is retained, not approximated
by deleting its singularity.

Consequently C_pair=C_rel algebraic_tensor C_c^infinity(R³_COM) is an operator
core for H_s^(2), and C_(12|34)=C_pair algebraic_tensor C_pair is an operator
core for(1). A proof uses bounded rectangles of the joint positive spectral
measure in(2), finite tensor approximations there, then factor-core approximation
in graph norm before removing spectral cutoffs. It is also a form core. The
compact hard-supported subcollection below is only a LOCAL test class, not a
claimed core for an unconstructed global hard four-particle operator.

The class includes

    v(r,c,t,d)=q1(c)q2(d)b1(r)b2(t),                    (3)

with smooth compact q1,q2 and allowed choices of the relative generators. Its
leading term is q1q2/(16pi²|r||t|). This product singularity is locally L² in
the six normal variables, since each three-dimensional integral of1/|r|² is
finite near0. Membership in D(A_(12|34)) and its L² action follow from the
operator tensor sum: apply the full pair operator to one factor at a time.
The off-contact differential expression is the free kinetic one. Distributional
contact delta sources are not left in that L² action.

For finite core vectors, the first pair has a Hilbert-space-valued decomposition
v=Phi_12+G_mu(r)Q_12 with Phi_12 in H²(R³_r;L²(c,t,d)) and
Phi_12(0)=c_mu Q_12 in the latter Hilbert space. There is an analogous34relation.
For(3), Q_12=q1(c)q2(d)b2(t) and Q_34=q1(c)q2(d)b1(r). Extracting both charges
on finite tensors gives the same mixed coefficient in either order. These are
tensor-domain conditions, not a presumed pointwise value at the intersection.

A nonzero Q_12 of this form is not ordinarily in the free-tangential
H^(1/2)(c,t,d): G_mu(t) has Fourier decay(|p|²+mu²)^(-1), producing a logarithmic
divergence of the three-dimensional H^(1/2) norm. Its regular H² remainder
cannot remove this divergence. Thus transplanting the one-contact spectator
charge space would exclude actual double-contact operator vectors.

Mixed-charge extraction is continuous in the matching OPERATOR graph norm.
Single relative charge extraction is bounded in its shifted pair form norm.
The tensor product of these two bounded extraction maps is controlled by the
product spectral weight xi eta, which obeys xi eta<=(xi+eta)²/4. It therefore
extends from finite tensors to D(A_(12|34)), with values in L²(c,d).
Any finite tensor sum where each summand has zero charge in at least one pair
has zero mixed charge. Such sums cannot graph-approximate(3) with nonzero
mixed charge. This is ONLY an operator-graph obstruction: no nonapproximability
in the weaker SUM FORM norm is claimed. Mixed-charge continuity in that weaker
norm does not follow from(2), and is not needed for the recovery lemma.

## 3. Physical-cell ambient convergence for the matching

On the complete ordered four-particle lattice define

    A_a=(kappa/a²)(24I-Adj)
                +g_a(1_{n1=n2}+1_{n3=n4})
       =H_a^(2) tensor I+I tensor H_a^(2).              (4)

No unwanted cross-pair potentials are included in this AMBIENT comparison.
The physical-cell isometry J_a is exactly the tensor product of the two pair
cell isometries, with amplitude a^(-6) on centered twelve-dimensional cells.
Cells are in original ordered particle coordinates, not independent COM grids.

The previously proved physical-cell bridge transfers the full two-particle
Fourier-embedded strong-resolvent result to each pair cell embedding. This uses
the exact inverse transform on a dense class compact in BOTH COM and relative
Fourier momenta and strong convergence of the bridge and its adjoint; no
operator-norm closeness of embeddings on ultraviolet states is assumed.

There is a common eventual pair lower bound -beta for every beta>E_s. For s>0,
the reviewed zero-P negative pole tends to-E_s, hence lies above-beta for smalla.
For s<0 the zero-P operator is nonnegative. The all-P modulus comparison with
unchanged12kappa/a² diagonal bounds every total-momentum fiber below by the
zero-P one. Therefore H_a^(2)>=-beta and A_a>=-2beta eventually.

Positive shifts give embedded pair heat-semigroup convergence by continuous
resolvent functional calculus, including t=0 by the complete cell projection
limit. Exact heat factorization then yields matching heat convergence: on
simple tensors it is the product of the two pair convergences, and the common
bound exp(2beta t) extends it by density to all vectors. Integrating against
exp(-lambda t) for lambda>2beta gives

    J_a(A_a+lambda)^(-1)J_a* ->(A_(12|34)+lambda)^(-1)
       strongly.                                      (5)

Continuous resolvent transformation gives generalized strong-resolvent
convergence at nonreal parameters too. No norm-resolvent claim is made.
This is convergence of the ambient TWO-MATCHED-PAIRS operator only.

## 4. Hard-supported recovery within one matching

Let Sigma_unwanted=Sigma_13 union Sigma_14 union Sigma_23 union Sigma_24.
Take ANY v in D(A_(12|34)) whose support is a compact subset of
Omega_R^(4) minus Sigma_unwanted. The support may contain both selected collision
surfaces and their intersection. Choose lambda>2beta and define

    f=(A_(12|34)+lambda)v,
    u_a=(A_a+lambda)^(-1)J_a*f.

Equation(5) and strong convergence of complete cell projections imply

    J_au_a->v,
    J_a A_a u_a=J_aJ_a*f-lambda J_au_a->A_(12|34)v.      (6)

Choose real smooth zeta compactly supported in Omega_R^(4) minus Sigma_unwanted,
equal1 near supp v. Put w_a(n)=zeta(an)u_a(n). Every nonzero lattice node is
hard-allowed. For the hard four-particle lattice form with ALL SIX pair
potentials, the unwanted four contacts vanish on this support, so

    q_a^hard[w_a]=a_a[w_a].                            (7)

The original24kappa/a² kinetic diagonal and bonds to forbidden zero-valued nodes
are retained; no reflecting graph degree replaces them.

The exact finite-lattice cutoff identity is

    a_a[zeta_a u_a]=Re<zeta_a²u_a,A_a u_a>
      +(kappa/a²)sum_{unordered edges{n,m}}
         (zeta(an)-zeta(am))² Re(conjugate(u_a(n))u_a(m)). (8)

Every diagonal potential cancels, including both singularly tuned pair terms.
There are twelve positive coordinate directions. Their cellwise cutoff difference
quotients converge uniformly to the corresponding derivatives. Strong L²
convergence in(6), translation continuity and the product L¹ estimate therefore
give the correction limit kappa integral|grad zeta|²|v|². The operator term
converges by graph recovery(6). Because zeta=1 near supp v, the derivative term
vanishes on v. We obtain

    J_a w_a->v,
    q_a^hard[w_a]-><v,A_(12|34)v>.                      (9)

The target need not be globally H¹; no singular-gradient estimate was used.
Nor is an arbitrary cutoff asserted to preserve the continuum double-contact
operator domain: zeta leaves the target unchanged, and(8) is applied at finitea.
A finite sum within matching12|34 is formed BEFORE computing f and applying
the single resolvent, so every interference term within that matching is retained.

## 5. The class contains allowed nonzero double-contact states

Choose pair-center points c0,d0 separated by D0>R, smooth nonzero q1,q2 supported
in radius-epsilon balls there, and relative cutoffs with |r|,|t|<=rho, where
D0-2epsilon-rho>R. Every cross-pair distance on the resulting support is then
at least D0-2epsilon-rho. Each three-label set contains a cross-pair distance,
so its diameter is>R. All unwanted collision surfaces and the hard boundary
are a positive distance from the compact support. The product vector(3) is
therefore an explicit nonzero member of the recovery class with nonzero mixed
charge. This is a genuine simultaneous double-contact statement, not only a
single-contact example in a larger particle space.

## 6. Remaining global four-particle work

The statements above are confined to a single compact-interior matching and
its complete ambient tensor operator. They do NOT construct a global hard
four-particle continuum form/operator, prove boundary density, identify cross-
matching interference, or establish a global liminf/recovery/convergence theorem.
No maximal pointwise intersection trace is asserted. Permuting the local matching
introduces other matchings, so a global bosonic conclusion does not follow merely
by averaging this isolated local result. Operator-graph mixed-charge distinctions
must not be relabeled as form-norm nonapproximability.

The next justified proof question is compatibility of different matching channels
and a global four-particle singular domain. The physical exclusion rule, range,
scattering length and kinetic normalization remain unchanged. No empirical,
arbitrary-N, thermodynamic or TOE conclusion follows from this local lemma.

## 7. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_three_particle_continuum_2026-09-13.md`: normalized pair-contact operator,
  tensor forms and singular generators; not an existing double-contact domain.
- `cubic_two_particle_operator_convergence_2026-09-13.md`: full pair convergence.
- `cubic_three_particle_lattice_liminf_2026-09-13.md`: physical-cell transfer and
  tensor-semigroup method.
- `cubic_three_particle_lattice_recovery_2026-09-13.md`: exact cutoff identity
  and graph recovery, now used with twelve rather than nine coordinates.
- `cubic_three_particle_convergence_2026-09-13.md`: completed N=3 scope and the
  need for a separate N=4 double-contact step.

Baseline680tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-four-particle-double-contact-imucv491/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 8. Independent mathematical review

Reviewer aa19f00aa2c99e237 cleared the local lemma with no blocker or repair
required. They checked the relative operator core, positive tensor domains,
product singularities and mixed-charge graph-norm control, tangential charge
warning, physical-cell ambient convergence, arbitrary compact operator-vector
hard recovery, twelve-coordinate cutoff identity and nonempty allowed class.

Clearance remains restricted to the fixed matching12|34 and its specified
compact-interior recovery class. No global four-particle, bosonic, cross-matching
or form-norm nonapproximability conclusion is supplied. The reviewer used direct
reads only. Original proof-draft-v1.md and its hash are preserved; no mathematical
constant or physical hypothesis changed. Independent auditor a5cb5cae2c4661cb1
confirmed all680baseline files unchanged, exactly one added note and681current
files matching final-hashes.json. Mathematical sections1–6 are byte-identical
to the submitted draft; the note matched pre-audit-note.md. Scope remains the
single matching and compact-interior recovery class. This is artifact/preservation
clearance, not mathematical recertification or proof of absence of unrecorded
execution. The audited final-hashes.json is retained; closing-hashes.json binds
the final audit-status edits.
