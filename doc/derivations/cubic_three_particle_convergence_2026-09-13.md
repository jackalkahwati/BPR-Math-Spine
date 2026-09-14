# Three-particle hard-lattice convergence to the specified continuum candidate

2026-09-13. Status: the specified N=3 convergence synthesis independently cleared;
independent saved-document/hash audit cleared.
The lower-limit and recovery inputs were reviewed separately. This synthesis
must itself be reviewed before its operator/dynamics conclusions are promoted.

## 1. Fixed model, target and reviewed inputs

Fix supplied R,kappa>0 and finite real s!=0. Retain

    g_a^(-1)=-a²W/kappa+a³/(8pi kappa s)

for sufficiently small a. Let Omega={X in R^9:diam(X)>R},
Lambda_a={n in Z^9:an in Omega}, H_a=ell²(Lambda_a). K_a is the zero-extension
compression of sum_i(kappa/a²)(6I-Adj_i)+g_a sum_{i<j}1_{n_i=n_j}.
Its original18kappa/a² kinetic diagonal and forbidden-endpoint costs remain.
Work initially with ordered particle labels and then restrict to bosons.

On H=L²(R^9), let P=1_Omega, H_Omega=ran P, and iota:H_Omega->H be inclusion.
The target H^D is the SPECIFIED continuum Friedrichs realization, whose closed
form q^D is the closure of the intrinsic singular C_s form. C_s includes
compact pair-contact singularities with scattering length s, supported away
from the exclusion boundary. It is a form core, not a presumed graph core.

Use exactly the common shift from the reviewed lower-limit proof:

    L=8pi/R, E_lat=8kappa*1_{s>0}/s², M=18kappa L²,
    sigma=E_lat+M+eta, eta>0,
    b_a=q_a+sigma||.||², b=q^D+sigma||.||².

Both shifted forms are bounded below by eta times squared norm. The reviewed
inputs, under the physical-cell isometry defined below, are:

(i) I_a psi_a weakly->U implies b[U]<=liminf b_a[psi_a], with b=+infinity off
the zero-extended D(q^D). Domain membership uses V_max=D(q^D), not just interior
finite energy.

(ii) For every u in D(q^D), hard-supported psi_a exist with I_a psi_a strongly
converging to iota u and limsup q_a[psi_a]<=q^D[u]. Strong norm convergence and
(i) give b_a[psi_a]->b[u].

No unshifted weak lower-limit claim replaces(i). The auxiliary sigma is not a
new interaction or physical parameter.

## 2. The changing physical-space projections

For centered cells C_(a,n)=a(n+[-1/2,1/2)^9), I_a psi equals a^(-9/2)psi(n)
on an allowed cell and zero on forbidden-node cells. Let P_a=I_aI_a*.
Let Q_a be the COMPLETE cell-average projection and m_a the indicator of cells
whose center is allowed. Then

    P_a=m_a Q_a=Q_a m_a.                                (1)

On C_c(R^9), uniform continuity and compact support show Q_af->f in L²;
density and contraction give Q_a->I strongly on all H. Each particle's position
within a cell differs from its node by at most sqrt(3)a/2, so

    |diam(X)-diam(an)|<=sqrt(3)a.

Thus m_a(X)->1_Omega(X) except on {diam=R}. This set lies in the finite union
of pair-distance-R hypersurfaces, which have measure zero for R>0. Dominated
convergence gives (m_a-P)f->0 for each f in H. Combining with(1),

    ||P_af-Pf||<=||Q_af-f||+||(m_a-P)f|| ->0.            (2)

Finite-a allowed cells may cross the continuum boundary; no containment of
ran P_a in ran P is assumed. This is strong fixed-vector projection convergence,
not convergence in operator norm.

## 3. Generalized Mosco convergence on a common ambient space

Define closed extended-valued quadratic functionals on H by

    B_a(U)=b_a[I_a*U] if U in ran I_a, and+infinity otherwise;
    B(U)=b[u] if U=iota u with u in D(q^D), and+infinity otherwise.

Their finite domains are not dense in H: they lie in the closed physical
subspaces ran P_a and ran P. They must not be called densely defined ambient
operator forms.

For any weakly convergent U_a->U, finite liminf allows passage to an attaining
subsequence with finite B_a and use of input(i). Infinite liminf is trivial.
Input(ii) gives strong recovery for every finite-domain U, with converging
shifted energy. For B(U)=+infinity the extended recovery inequality has no
energy restriction; the constant sequence U_a=U suffices. Hence

    B_a -> B in Mosco sense on H.                       (3)

This is precisely the generalized form convergence for the changing allowed
cell subspaces, including their complementary kernels.

## 4. Embedded real resolvents from minimizers

Fix lambda>0 and f in H. Let

    U_a=I_a(K_a+sigma+lambda)^(-1)I_a*f,
    U=iota(H^D+sigma+lambda)^(-1)iota*f.

These uniquely minimize F_a(V)=B_a(V)+lambda||V||²-2Re<f,V> and its continuum
counterpart F. Positivity gives ||U_a||<=||f||/(eta+lambda). Every subsequence
has a weakly convergent subsubsequence; the Mosco lower limit and recovery of
the limiting minimizer give

    F(U_*)<=liminf F_a(U_a)<=limsup F_a(U_a)<=F(U).

Uniqueness forces U_*=U, hence full weak convergence. Euler identities give
F_a(U_a)=-Re<f,U_a>->F(U). If V_a strongly recovers U with F_a(V_a)->F(U),
quadratic expansion around the minimizer yields

    F_a(V_a)-F_a(U_a)
      =B_a(V_a-U_a)+lambda||V_a-U_a||²
      >=(eta+lambda)||V_a-U_a||².

Thus U_a->U strongly, proving

    I_a(K_a+sigma+lambda)^(-1)I_a*
      ->iota(H^D+sigma+lambda)^(-1)iota* strongly.       (4)

The expansion occurs within the common finite-a physical subspace, so subtraction
of extended-infinite form values is not used.

## 5. Nonreal resolvents and comparison completions

Fix lambda0>0 and denote the two positive embedded resolvents in(4) by R_a,R.
They converge strongly and their spectra lie in [0,(eta+lambda0)^(-1)]. For
nonreal z the continuous function

    h_z(r)=r/[1-(z+sigma+lambda0)r], h_z(0)=0

has no singularity on this interval. Uniform polynomial approximation and
strong convergence of uniformly bounded operator powers give h_z(R_a)->h_z(R)
strongly. Functional calculus on each physical subspace identifies this with

    I_a(K_a-z)^(-1)I_a* ->iota(H^D-z)^(-1)iota* strongly. (5)

These are embedded generalized resolvents. If the forbidden complement is
nonzero, the right side annihilates it and is not the resolvent of an ordinary
densely defined self-adjoint operator on the full ambient H.

For comparison only define tilde K_a as the transported K_a on ran P_a direct-
summed with0 on ran(I-P_a), and tilde H as H^D on ran P direct-summed with0 on
ran(I-P). They are ordinary self-adjoint operators on H. Their resolvents add
-(1/z)(I-P_a) and -(1/z)(I-P), respectively, to(5). By(2),

    tilde K_a -> tilde H in ordinary strong-resolvent sense.         (6)

These zero completions do not assign physical dynamics to forbidden configurations
or modify the lattice compression. They specify a mathematical comparison on
missing modes. No norm-resolvent conclusion follows from(5) or(6).

## 6. Compact-time dynamics with spectral tails controlled

Let A_a=tilde K_a, A=tilde H and fix f in H. Choose chi_m in C_c(R),0<=chi_m<=1,
equal1 on expanding intervals. Spectral calculus gives
||(I-chi_m(A))f||->0. At fixed m, strong-resolvent convergence implies
chi_m(A_a)f->chi_m(A)f, and consequently

    limsup_a||(I-chi_m(A_a))f||<=||(I-chi_m(A))f||.       (7)

For fixed m,T the functions exp(-itx)chi_m(x), |t|<=T, form a compact family
in the uniform norm of C_0(R). A finite-net argument upgrades strong C_0
functional calculus to convergence uniform in this family, for the fixed f.
Unitarity bounds both omitted tails by(7), uniformly in t. Removing m gives

    sup_|t|<=T ||exp(-it tilde K_a)f-exp(-it tilde H)f|| ->0.          (8)

This argument is needed because exp(-itx) does not vanish at infinity; C_0
functional calculus alone is not a sufficient explanation.

Subtract the exact complementary identity terms from
exp(-it tilde K_a)=I_a exp(-itK_a)I_a*+(I-P_a) and its limit counterpart.
Projection convergence(2) then gives

    sup_|t|<=T ||[I_a exp(-itK_a)I_a*
                   -iota exp(-itH^D)iota*]f|| ->0.                  (9)

At t=0 this is P_af->Pf, not ambient identity convergence of unitary groups.
The embedded physical families are not themselves unitary groups on all H.

The comparison operators share a lower bound ell=min(0,eta-sigma). On their
spectra exp(-tx) is bounded by exp(T max(0,-ell)) for0<=t<=T. The same compact-
spectral-cutoff proof therefore gives embedded heat-semigroup convergence
uniform on0<=t<=T, including t=0. This uses the common lower bound; it is not
an assertion about unbounded exponential functional calculus without control.

For moving admissible initial states I_a psi_a->iota u strongly, u in H_Omega,
contraction of embedded unitaries and(9) imply

    sup_|t|<=T ||I_a exp(-itK_a)psi_a-iota exp(-itH^D)u|| ->0.         (10)

No finite-energy restriction on u is needed. The spectral cutoffs are proof
tools, not truncations of the physical initial state. The analogous heat result
uses its common compact-time norm bound.

## 7. Bosons, phases and what boundary selection now means

All domains, lattice operators and physical-cell embeddings intertwine particle
permutations, so the proof restricts to bosons. Either use the ambient symmetric
H throughout, in which case the projection limit is P, or remain in ordered H,
in which case bosonic range projections converge to P P_sym. These conventions
must not be mixed.

K_a uses the declared threshold reference: at N=3 it adds18kappa/a² to the
unshifted hopping-plus-contact Hamiltonian. Dropping that reference introduces
an a-dependent scalar phase; convergence of the unreferenced family is not
claimed. The auxiliary sigma is a COMMON energy shift, whose exp(-it sigma)
phase cancels between physical spaces. Zero completions above were defined
for K_a,H^D themselves, not by conflating this shift with complementary dynamics.

Conditional on the separately reviewed input proofs and review of this synthesis,
(3)–(10) identify the hard N=3 lattice limit with the SPECIFIED Friedrichs
form/operator. Boundary selection is supported by the lower-limit membership
in the global domain V_max=D(q^D) and hard-supported recovery of every form
vector, not just matching interior contact conditions. A maximal operator-domain
trace characterization at the contact/boundary intersection is not needed for
this identification and is not proved here. Uniqueness among all imaginable
self-adjoint boundary extensions is not asserted either.

The convergence is strong, vectorwise and, where stated, uniform on compact
time intervals. No norm-resolvent/operator-norm convergence, rates, uniformity
over bounded sets of states, all-time convergence, unbounded-observable or
wave-operator convergence, arbitrary-N continuum dynamics, thermodynamic/full-
Fock limit or empirical validation follows. Supplied R,s,kappa and the hard
constraint are still assumptions, not derived spacetime or matter/gravity physics.

## 8. Next bounded hypothesis, not an automatic generalization

At N=4 two disjoint pairs can collide simultaneously while their centers remain
farther apart than R. A justified next lemma would construct the compact-interior
tensor-sum contact operator for those two pairs and its genuine product singularities
1/(|r12||r34|), then prove hard-supported recovery for that local class.
N=3 proofs have no such intersection. Full N=4 boundary density, global overlap
control and convergence require separate work; no arbitrary-N theorem is imported
by treating the present particle count as a parameter.

## 9. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_three_particle_lattice_liminf_2026-09-13.md`: shifted lower limit,
  physical-cell convention and common bounds.
- `cubic_three_particle_lattice_recovery_2026-09-13.md`: hard recovery of the
  singular form core and its full-domain diagonal extension.
- `cubic_three_particle_continuum_2026-09-13.md`: specified self-adjoint target
  and precise Friedrichs closure.
- `cubic_three_particle_boundary_density_2026-09-13.md`: global form-domain equality.
- `cubic_three_body_exclusion_2026-09-13.md`: hard constraint and original kinetic costs.
This synthesis is a new reviewed-proof target, not a retroactive expansion of
its input notes' earlier scope.

Baseline679tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-convergence-d3ogm5ur/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 10. Independent mathematical review

Reviewer aa4e7a41a8316d6b9 cleared the synthesis and its use of reviewed inputs
with no blocker. They checked changing physical projections, shifted extended-
valued Mosco convergence, coercive minimizer strong convergence, resolvent signs,
zero-completion kernels, fixed-vector spectral tails and compact-time dynamics,
heat bounds, bosonic reduction and threshold-reference bookkeeping.

Clearance includes identification of this specified hard N=3 lattice limit with
the specified Friedrichs form/operator through global lower-limit membership
and hard recovery of every form vector. It covers exactly the form, resolvent
and vectorwise compact-time consequences(3)–(10), not norm convergence, all-time,
arbitrary-N, unbounded-observable or empirical conclusions. No maximal operator-
domain trace theorem or uniqueness among all boundary extensions is supplied.

The reviewer used direct reads only. Original proof-draft-v1.md and its hash
remain preserved; no mathematical constant or physical assumption changed.
Independent auditor a0474a2d3e3032c64 confirmed all679baseline files unchanged,
exactly one new note and680current files matching final-hashes.json. Mathematical
sections1–8 are byte-identical to the submitted draft; changes are status/review
material only. The note matched pre-audit-note.md, and the scope agrees with
the mathematical review record. This is document/hash clearance, not mathematical
recertification or an assertion of absence of unrecorded execution. The audited
final-hashes.json is retained; closing-hashes.json binds final audit-status edits.
