# Four-particle hard-lattice convergence synthesis

2026-09-14. Status: exact N=4 convergence synthesis independently cleared;
independent saved-document/hash audit cleared.
The N=4 form-domain, lower-limit and recovery theorems are reviewed inputs.
This note checks their consequences without claiming an arbitrary-N theorem.

## 1. Identical model, domain and shift on both convergence halves

Fix supplied R,kappa>0 and finite s!=0, retaining the original tuned coupling
 g_a^(-1)=-a²W/kappa+a³/(8pi kappa s). For each three-label subset T of four
particles set d_T=max_{i,j in T}|xi-xj|, and define

    Omega=intersection_{|T|=3}{d_T>R} in R^12.

ALL four triples123,124,134,234 must pass; using only the diameter of all four
particles would be a different constraint. Let Lambda_a={n:an in Omega} and
H_a=ell²(Lambda_a). K_a is the zero-extension compression of
(kappa/a²)(24I-sum_i Adj_i)+g_a sum_{i<j}1_{n_i=n_j}. The full diagonal and
forbidden-endpoint costs remain. q_a^hard denotes its form.

On H=L²(R^12) let P=1_Omega and iota:ran P->H be inclusion. The target H_4^D
is the specified Friedrichs operator with form q_4^D, not an unspecified contact
extension. The boundary-density theorem identifies its zero-extended domain
with physical U for which every chi_P U belongs to the completed matching
ambient form space. This includes genuine double-contact vectors without
imposing free-tangential charge regularity.

Keep precisely the shift in the reviewed lower-limit and recovery notes:

    L=8pi/R, E_lat=8kappa*1_{s>0}/s², M=36kappa L²,
    C=2E_lat+M, sigma=C+eta, eta>0,
    b_a=q_a^hard+sigma||.||², b=q_4^D+sigma||.||².

Eventually b_a,b>=eta||.||². The two reviewed inputs concern this SAME model,
embedding and shift:
(i) I_a psi_a weakly->U implies b[U]<=liminf b_a[psi_a], with +infinity off
the specified physically supported form domain.
(ii) Every u in D(q_4^D) has exactly hard-supported recovery I_a psi_a->iota u,
b_a[psi_a]->b[u] and q_a^hard[psi_a]->q_4^D[u], symmetric for bosonic u.
No unshifted weak lower-limit inequality replaces(i).

## 2. Physical projections for four separate triple exclusions

For centered cells C_(a,n)=a(n+[-1/2,1/2)^12), define I_a psi=a^(-6)psi(n)
on allowed-node cells and zero otherwise. Its norm factor is correct because
cell volume is a^12. Let Q_a be the COMPLETE cell-average projection and m_a
the indicator of allowed-node cells. Then

    P_a=I_aI_a*=m_a Q_a=Q_a m_a.

Uniform continuity on compact continuous functions, density and contraction
prove Q_a->I strongly. Within each cell, every particle differs from its node
by at most sqrt(3)a/2. Each of the FOUR triple diameters therefore differs by
at most sqrt(3)a. Thus m_a->1_Omega pointwise away from the union of {d_T=R}.
Every such equality set is contained in finitely many pair-distance-R
hypersurfaces, null because R>0. Dominated convergence and the cell-average
bound give

    ||P_a f-Pf||<=||Q_af-f||+||(m_a-P)f|| ->0.           (1)

This is P_a->P, NOT identity, and no finite-a containment of allowed cells
inside Omega is assumed. The convergence is strong on fixed vectors only.

## 3. Extended-valued generalized Mosco convergence

Define on common H the closed positive quadratic functionals

    B_a(U)=b_a[I_a*U] for U in ran I_a, +infinity otherwise;
    B(U)=b[u] for U=iota u, u in D(q_4^D), +infinity otherwise.

Their finite domains are not dense in ambient H and must not be described as
densely defined ambient operator forms. A finite weak liminf reduces to a
subsequence with finite B_a and input(i); infinite liminf is immediate. Input(ii)
gives strong exact-energy recovery on the finite domain. If B(U)=+infinity,
the recovery inequality imposes no finite-energy requirement and the constant
ambient sequence suffices. Therefore

    B_a -> B in Mosco sense on H.                       (2)

This uses all the reviewed N=4 domain and recovery content, including shared-
contact interference, rather than merely replacing a particle-count symbol in
the N=3 theorem.

## 4. Resolvents from coercive minimization

For lambda>0 and f in H define

    U_a=I_a(K_a+sigma+lambda)^(-1)I_a*f,
    U=iota(H_4^D+sigma+lambda)^(-1)iota*f.

They minimize B_a(V)+lambda||V||²-2Re<f,V> and its limit. The lower bound gives
||U_a||<=||f||/(eta+lambda). Weak subsequential limits, Mosco liminf, and recovery
of the limiting minimizer force every weak cluster point to be U. The Euler
identity identifies the minimum as -Re<f,U_a>, hence minimum values converge.
For a strong recovery V_a of U, quadratic expansion within the same finite-a
physical subspace gives

    F_a(V_a)-F_a(U_a)
      =b_a[I_a*(V_a-U_a)]+lambda||V_a-U_a||²
      >=(eta+lambda)||V_a-U_a||².

Thus U_a->U strongly. Fixing lambda0>0, denote these embedded positive resolvents
by R_a,R. Their spectra lie in [0,(eta+lambda0)^(-1)]. For nonreal z,

    h_z(r)=r/[1-(z+sigma+lambda0)r], h_z(0)=0

is continuous there. Uniform polynomial approximation and strong convergence
of bounded powers give

    I_a(K_a-z)^(-1)I_a* -> iota(H_4^D-z)^(-1)iota*
       strongly, Im z!=0.                              (3)

The right side annihilates the forbidden complement and is an embedded generalized
resolvent, not an ordinary resolvent on the whole ambient Hilbert space.

## 5. Comparison completions and their common lower bound

For comparison only let tilde K_a be transported K_a on ran P_a direct-summed
with0 on ran(I-P_a), and tilde H_4^D be H_4^D on ran P direct-summed with0 on
its complement. These are ordinary self-adjoint ambient operators. Their
resolvents add respectively -z^(-1)(I-P_a) and -z^(-1)(I-P) to(3). Equation(1)
therefore implies ordinary strong-resolvent convergence of these completions.

The positivity of b_a,b gives K_a,H_4^D>=eta-sigma=-C. Since0>=-C, the FULL
comparison completions also share this lower bound. It is needed below for
heat evolution. The zero-complement choice is only a comparison device, not
physical dynamics assigned to forbidden configurations or a change to the hard
compression. No norm-resolvent convergence is inferred.

## 6. Compact-time unitary and heat convergence

Set A_a=tilde K_a, A=tilde H_4^D and fix f in H. Choose0<=chi_m<=1 in C_c(R),
equal1 on expanding intervals. Spectral calculus gives
||(I-chi_m(A))f||->0. For fixed m, strong-resolvent convergence gives
chi_m(A_a)f->chi_m(A)f, hence

    limsup_a||(I-chi_m(A_a))f||<=||(I-chi_m(A))f||.       (4)

The functions exp(-itx)chi_m(x), |t|<=T, form a compact family in uniform C_0
norm. Finite nets and strong C_0 functional calculus give convergence uniformly
in this family for the fixed f. Unitarity controls omitted tails by(4), uniformly
in t. Removing m proves compact-time strong unitary convergence for completions.
Subtract their exact complementary identity terms and use(1) to obtain

    sup_|t|<=T ||[I_a exp(-itK_a)I_a*
                  -iota exp(-itH_4^D)iota*]f|| ->0.    (5)

The spectral-tail argument is necessary: exp(-itx) itself does not vanish at
infinity, and cannot be handled by C_0 calculus without that control.

For heat evolution, the common lower bound -C bounds both full comparison
semigroups by exp(CT) for0<=t<=T. The compact family exp(-tx)chi_m(x) and the
same tail argument give

    sup_0<=t<=T ||[I_a exp(-tK_a)I_a*
                   -iota exp(-tH_4^D)iota*]f|| ->0.    (6)

Both formulas include t=0, where they say P_af->Pf. Embedded physical unitary
families are not unitary groups on the full ambient space.

If allowed initial states obey I_a psi_a->iota u strongly with u in ran P,
the uniform norm1 of embedded unitaries and(5) imply

    sup_|t|<=T ||I_a exp(-itK_a)psi_a-iota exp(-itH_4^D)u|| ->0.       (7)

The heat analogue follows with exp(CT). No finite-energy assumption on u is
needed. These are fixed-vector/strongly moving-state results, not uniform
operator-norm assertions over all initial vectors or all times.

## 7. Bosons, references and precise boundary selection

All four triple conditions, six pair contacts, kinetic terms and physical-cell
embeddings intertwine permutations. The continuum form and its reviewed recovery
are permutation invariant. On the symmetric ambient Hilbert space projection
limits are P restricted there; if kept on ordered H the bosonic projections
are P_a P_sym->P P_sym. These conventions are not mixed. Individual matching
localizations may remain ordered, not symmetric.

For the hard hopping-plus-contact operator H_a^hop without the threshold diagonal,

    K_a=H_a^hop+24kappa/a²,
    exp(-itH_a^hop)=exp(it24kappa/a²)exp(-itK_a).

The theorem concerns threshold-referenced K_a, not convergence after dropping
this divergent phase. The common auxiliary sigma multiplies both physical
unitaries by exp(-it sigma), and similarly for heat; it changes no physical
parameter. Zero completions were defined for K_a,H_4^D themselves: completing
shifted operators by zero is not the same as shifting the whole comparison space.

Conditional on the reviewed inputs and review of this synthesis, the hard N=4
lattice selects the SPECIFIED Friedrichs form/operator. The domain conclusion
comes from global lower-limit membership and recovery of every vector of that
same domain, including matching overlaps with shared contacts. It is not based
only on local scattering or the absence of the old collapse witness.

No maximal operator trace characterization or uniqueness among all conceivable
contact-boundary extensions is proved. No norm-resolvent or operator-norm limit,
rate, all-time, unbounded-observable, wave-operator, arbitrary-N, thermodynamic,
full-Fock or empirical conclusion follows. The hard rule and supplied R,s,kappa
remain assumptions; this is not derivation of spacetime, matter or gravity.

## 8. Next bounded generalization question

The useful next task is an arbitrary-FIXED-finite-N hypothesis audit, not an
endless sequence of isolated particle counts. A proof would have to establish
completed matching tensor forms at every finite depth, compatible contact
transfer and boundary approximation across arbitrary overlaps, all-matching
physical-cell convergence, finite-N IMS bounds and generator recovery/positive-
form assembly. N=4 tests only two simultaneous disjoint contacts.

There are finitely many matchings at each fixed N, but their number grows with N.
Uniformity across the channels at fixed N is not uniformity as N->infinity.
No such general theorem is asserted by this synthesis.

## 9. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_four_particle_lattice_liminf_2026-09-14.md`: exact N=4 lower half,
  all-matching convergence, shift and physical support.
- `cubic_four_particle_lattice_recovery_2026-09-14.md`: full form-domain hard
  recovery including noncircular positive-form cross assembly and bosons.
- `cubic_four_particle_boundary_density_2026-09-13.md`: global specified domain.
- `cubic_three_particle_convergence_2026-09-13.md`: abstract synthesis strategy;
  changed N=4 geometry, cells, shifts and inputs are checked explicitly above.

Baseline685tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-four-particle-convergence-2ibexxkl/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 10. Independent mathematical review

Reviewer a53bf7469903b9886 cleared the exact N=4 synthesis with no mathematical
blocker or required repair. They checked all-four-triple geometry and cell
normalization, common positive shift and target domain, extended Mosco and
coercive minimizer arguments, resolvent signs and comparison completions,
fixed-vector spectral tails, compact-time unitary/heat convergence, moving
states, bosonic projection conventions and threshold phase.

Clearance identifies this hard N=4 lattice limit with the specified Friedrichs
realization through global domain membership and recovery, not uniqueness among
all conceivable boundary extensions. It covers the stated generalized Mosco,
embedded strong-resolvent and compact-time strong-dynamics conclusions only,
not norm convergence, unbounded observables, arbitrary N or empirical validation.
Direct reads only were used. Original proof-draft-v1.md and its hash are preserved;
no mathematical constant or physical hypothesis changed. Independent auditor
ab7b6725819861cce confirmed all685baseline files unchanged, exactly one added
note and686current files matching final-hashes.json. Mathematical sections1–8
are byte-identical to the submitted draft; the note matched pre-audit-note.md.
This is artifact/scope clearance, not mathematical recertification or proof of
absence of unrecorded execution. The audited final-hashes.json is retained;
closing-hashes.json binds final audit-status edits.
