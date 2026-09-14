# Hard-supported recovery at every fixed finite particle number

2026-09-14. Status: hard-supported recovery at each fixed finite N independently
cleared; independent saved-document/hash audit completed.
This note does not itself promote full convergence or a Fock/thermodynamic limit.

## 1. Fixed sector and reviewed lower-half input

Fix integer N before a down to0, retaining supplied R,kappa>0, finite s!=0,
the original g_a^(-1)=-a²W/kappa+a³/(8pi kappa s), all-triple hard exclusion
and the original6Nkappa/a² kinetic diagonal. Write m_N=floor(N/2),
E_lat=8kappa*1_{s>0}/s², L=8pi/R and use precisely

    M_N=3kappa N(N-1)L², sigma_N=m_N E_lat+M_N+eta, eta>0,
    b_(N,a)=q_(N,a)^hard+sigma_N||.||²,
    b_N=q_N^D+sigma_N||.||².

Both positive forms are eventually >=eta||.||². The reviewed lower half is

    I_a u_a weakly->u implies b_N[u]<=liminf b_(N,a)[u_a],            (1)

including infinity off the specified physically zero-extended form domain.
The target is: for every u in D(q_N^D), exactly hard-supported psi_a exist with

    I_a psi_a->u strongly,
    b_(N,a)[psi_a]->b_N[u], q_(N,a)^hard[psi_a]->q_N^D[u].            (2)

Recovery can be symmetric for bosonic u. Constants and approximation scales
may depend on fixed N. No uniform-N convergence claim is intended.

## 2. Each intrinsic generator is an ambient operator vector

A generator g=eta_P v of the ACTUAL D_(0,N) has v in the algebraic matching
operator core V_P and eta_P admissible: compact in Omega_N away from every
unselected contact, normal-flat at every selected contact. The reviewed
arbitrary-finite-depth restricted multiplier lemma, used with C=P, proves

    g in D(A_P), A_Pg=S_Ng,
    q_N^D[g]=<g,S_Ng>=<g,A_Pg>.                         (3)

This is actual operator membership, not merely a form-energy assignment. The
support is compact in U_P=Omega_N minus unselected collision surfaces, so it
has positive hard-boundary and unwanted-contact margins. The lemma is applied
to finitely many tensor factors only, requiring no higher mixed-charge trace
or multiplication theorem for arbitrary completed operator domains.

## 3. Graph recovery on complete physical cells

Let J_a be the COMPLETE ordered physical-cell embedding in3Ndimensions,
amplitude a^(-3N/2), on cells a(n+[-1/2,1/2)^(3N)). Its complete range projection
Q_a=J_aJ_a* tends strongly to identity. On hard-supported vectors J_a agrees
with I_a.

The ambient matching A_(P,a) retains6Nkinetic diagonal and only the selected
pair contacts. For lambda>m_N E_lat the reviewed finite-factor tensor theorem
gives J_a(A_(P,a)+lambda)^(-1)J_a* ->(A_P+lambda)^(-1) strongly.
Set f=(A_P+lambda)g and

    h_a=(A_(P,a)+lambda)^(-1)J_a*f.

Then

    J_a h_a->g,
    J_a A_(P,a)h_a=Q_a f-lambda J_a h_a->A_Pg.           (4)

Thus convergence includes the operator action, not merely vector norm or
scattering amplitudes. No hard projection enters this ambient resolvent product.

## 4. Exact hard support and limiting energy

Choose fixed real zeta in C_c^infinity(U_P), equal1 near supp g, and put
w_a(n)=zeta(an)h_a(n). Every nonzero node is hard-allowed and avoids unselected
contacts, hence

    q_(N,a)^hard[w_a]=a_(P,a)[w_a].                      (5)

This keeps every original forbidden-endpoint kinetic bond and diagonal cost.
The complete-lattice cutoff identity is

    a_(P,a)[zeta_a h_a]=Re<zeta_a²h_a,A_(P,a)h_a>
      +(kappa/a²)sum_n sum_(l=1..3N)
         (zeta(a(n+e_l))-zeta(an))²
                          Re(conjugate(h_a(n))h_a(n+e_l)).          (6)

Every unordered edge is counted once by its positive coordinate direction;
there are3Ndirections. All diagonal potentials cancel exactly. The coefficient
is kappa/a², with a plus sign for the correction.

Let F_a=J_ah_a and let d_(l,a) be the cellwise difference quotient of zeta.
Uniformly d_(l,a)->partial_l zeta by bounded smooth derivatives. Physical-cell
normalization rewrites the correction as

    kappa sum_l integral d_(l,a)² Re(conjugate(F_a(X))F_a(X+ae_l))dX.

Strong L² convergence(4) and translation continuity give F_a(.+ae_l)->g in
L²; the products converge to |g|² in L¹. Thus the correction tends to
kappa integral|grad zeta|²|g|²=0. The first term converges by graph recovery to
Re<zeta²g,A_Pg>=<g,A_Pg>. Consequently

    I_aw_a->g, q_(N,a)^hard[w_a]->q_N^D[g],
    b_(N,a)[w_a]->b_N[g].                               (7)

The singular g is never sampled at collision. No global H¹ bound is used.
Nor is zeta assumed to preserve an arbitrary continuum contact operator domain:
it leaves this target unchanged, and(6) is a finite-lattice identity.

## 5. Positive-form cross assembly at arbitrary finite matching depth

Use sesquilinear forms conjugate-linear in the first argument. Suppose u_a,v_a
converge strongly under I_a to u,v and have exact limits of diagonal b energies.
Let z_a=b_(N,a)(u_a,v_a), c=b_N(u,v). Positivity bounds z_a by Cauchy-Schwarz.
Apply ONLY(1) to u_a+t v_a for each fixed t in C. Since diagonal energies converge,

    Re(t c)<=liminf_a Re(t z_a).

The choices t=1,-1 force real-part convergence; t=i,-i force imaginary-part
convergence because Re(i z)=-Im z. Therefore

    b_(N,a)(u_a,v_a)->b_N(u,v).                         (8)

No upper bound for a combined vector is presumed. The argument is a positive-
form consequence of exact separate energies and the independently proved
lower half, not a contact-free energy calculation.

For an arbitrary initial-domain u=sum_{j=1}^r c_j g_j, construct(7) for every
g_j on the SAME a-family and put psi_a=sum_j c_j w_(j,a). Hard support and strong
vector convergence are preserved. Equation(8) gives every entry of the finite
Gram matrix, hence b_(N,a)[psi_a]->b_N[u]. Strong norms allow subtraction of
the fixed sigma_N, yielding unshifted energy convergence.

This covers overlaps retaining any finite number of shared contacts, with no
higher mixed-charge regularity requirement. There are finitely many generators
in a fixed core vector and finitely many channels at fixed N; no convergence
uniform in their growth is used.

## 6. Bosonic recovery and form-domain completion

For symmetric u, apply P_sym,a=(1/N!)sum_permutations U_pi,a to the initial-domain
recovery. This finite average preserves hard support and intertwines with the
physical embedding. Permutation invariance makes it a reducing projection for
the positive b_(N,a), so its energy cannot exceed that of the original recovery.
The lower limit(1) applied to this strongly convergent projected sequence forces
the matching lower bound b_N[u]. Thus exact shifted energy is retained; strong
norms give exact unshifted energy too. No generator individually had to be symmetric.

For general u in D(q_N^D), the defining form-core property supplies u_m in D_(0,N)
with b_N[u_m-u]->0, choosing symmetric approximants when needed. Their norms
and energies converge. For each m use the preceding recovery and choose decreasing
delta_m down to0 such that for0<a<delta_m,

    ||I_a psi_a^(m)-u_m||<1/m,
    |b_(N,a)[psi_a^(m)]-b_N[u_m]|<1/m.

Select m(a)->infinity slowly enough that a<delta_(m(a)), and define the diagonal
family. It remains exactly hard-supported and symmetric if required, and gives(2).
Only form-core density is used, never a graph-core assertion for the Friedrichs
extension. The proof shift is removed ONLY on strongly convergent recoveries,
not from a general weak lower-limit statement.

## 7. Small sectors and exact remaining scope

At N=0 use identity recovery in C with zero energy and sigma_0=eta. At N=1
there is one empty matching and ordinary free physical-cell recovery, retaining
6kappa/a² diagonal. At N=2 no hard exclusion exists and the specified target is
exactly the established full pair operator, with12kappa/a² diagonal and contact
g_a, not2g_a. The R-dependent proof shift does not alter that physical operator.

If independently cleared, this is the complete upper half for every form vector
at each fixed finite N. A separate synthesis must prove convergence of allowed
cell projections to1_(Omega_N), not ambient identity, combine the same shifted
forms and embeddings into generalized form/resolvent convergence, and state
precise compact-time dynamics consequences. Sectorwise recovery alone is not
that synthesis.

Any future Fock direct-sum argument needs its own number-sector truncation and
uniform nonreal resolvent bound. It would not automatically imply unshifted
full-Fock lower boundedness, a physical chemical potential or preferred vacuum.
No joint N,a limit, recovery rate uniform in N, thermodynamic pressure or
empirical validation follows. Extensive stability does not make the N-dependent
cutoffs, shifts and diagonal selections uniform convergence estimates.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_finite_particle_lattice_liminf_2026-09-14.md`: exact shift, general
  ambient tensor convergence and reviewed weak lower half.
- `cubic_finite_particle_continuum_2026-09-14.md`: actual initial generators,
  many-contact restricted multiplier lemma and Friedrichs form-core definition.
- `cubic_finite_particle_boundary_density_2026-09-14.md`: exact global domain.
- `cubic_four_particle_lattice_recovery_2026-09-14.md`: abstract positive-form
  assembly, now applied with explicit arbitrary-finite-sector inputs.

Baseline689tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-finite-particle-lattice-recovery-3zp75wmb/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 9. Independent mathematical review

Reviewer a39dc33e8070071b6 cleared exact hard-supported recovery for every form
vector at each fixed finite N, including bosons, with no blocker or required
repair. They checked actual generator operator membership, finite-depth graph
recovery, hard cutoff and3Nedge normalization, translated-L² singular-target
control, noncircular positive-form cross assembly, symmetry and form-core
completion, and the small-sector conventions.

Clearance includes strong norm and shifted/unshifted energy convergence for
recoveries, but not the separate convergence/dynamics synthesis, Fock convergence
or any uniform-N limit. Direct reads only were used. Original proof-draft-v1.md
and its hash remain preserved; no mathematical constant or physical hypothesis
changed. Independent auditor a2f5a267947255118 confirmed all689baseline files
unchanged, exactly one new note and690current hashes matching final-hashes.json.
Mathematical sections1–7 are byte-identical to the submitted draft; current note
matched pre-audit-note.md. This is artifact/scope clearance, not mathematical
recertification or proof of absence of unrecorded execution. The audited
final-hashes.json is retained; closing-hashes.json binds final audit-status edits.
