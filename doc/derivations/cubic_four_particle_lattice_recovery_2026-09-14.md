# Four-particle hard recovery by positive-form assembly

2026-09-14. Status: hard-supported N=4 recovery upper half independently cleared;
independent saved-document/hash audit cleared.
No full convergence or dynamics synthesis is promoted by this note.

## 1. Fixed inputs and target

Retain R,kappa>0, finite s!=0, the original tuned g_a, hard exclusion of every
close triple, and the original24kappa/a² kinetic diagonal. Let q_a^hard be the
hard lattice form, q_4^D the specified continuum Friedrichs form, and I_a the
ordered twelve-dimensional physical-cell isometry. Use EXACTLY the common shift
of the reviewed lower-limit note:

    L=8pi/R, E_lat=8kappa*1_{s>0}/s², M=36kappa L²,
    sigma=2E_lat+M+eta, eta>0,
    b_a=q_a^hard+sigma||.||², b=q_4^D+sigma||.||².

Both positive forms are >=eta times squared norm for sufficiently small a.
The already reviewed lower limit says I_a psi_a weakly->u implies
b[u]<=liminf b_a[psi_a], including infinity outside the specified global form
domain. It is a shifted statement and is not replaced by unshifted weak lower
semicontinuity.

The target is: for every u in D(q_4^D), there exist exactly hard-supported psi_a
with

    I_a psi_a->u strongly, b_a[psi_a]->b[u],
    q_a^hard[psi_a]->q_4^D[u].                           (1)

For bosonic u the recovery may be chosen symmetric. This proof uses the actual
initial-domain generators D0 and form closure, not an operator-core assumption
for the Friedrichs extension.

## 2. Positive-form cross-assembly lemma

Suppose positive Hermitian forms b_a,b have the stated embedded lower-limit
property, with inner products/forms conjugate-linear in the first argument.
Let u_a,v_a strongly recover u,v in vector norm and exactly in diagonal energy:

    I_au_a->u, I_av_a->v,
    b_a[u_a]->b[u], b_a[v_a]->b[v].

Then

    b_a(u_a,v_a)->b(u,v).                               (2)

To prove it put z_a=b_a(u_a,v_a), c=b(u,v). Positive-form Cauchy-Schwarz bounds
z_a. For any fixed complex t, vector convergence gives
I_a(u_a+t v_a)->u+t v. Apply ONLY the lower-limit inequality:

    b[u+t v]<=liminf_a(b_a[u_a]+|t|²b_a[v_a]+2Re(t z_a)).

The diagonal terms converge, hence Re(t c)<=liminf Re(t z_a). Taking t=1,-1
forces convergence of the real parts, and t=i,-i forces convergence of imaginary
parts (Re(i z)=-Im z). This proves(2) without any upper bound for the combination.
Equivalently any subsequential limiting Gram-matrix difference is positive
semidefinite with zero diagonal, hence zero off-diagonal entries.

For finitely many vectors g_j with exact-energy recoveries w_(j,a) defined on
the SAME mesh family, (2) applies to each pair. The complete Gram matrices
converge; for fixed coefficients c_j,

    b_a[sum_j c_j w_(j,a)] -> b[sum_j c_j g_j].          (3)

This supplies all interference terms, including those on overlaps retaining a
shared contact. There is no presupposed recovery of the combined vector and
no contact-free or mixed-charge regularity assumption in this algebra lemma.

## 3. Exact recovery of one intrinsic-domain generator

Take a generator g=eta_P v of D0. By the established restricted tensor multiplier
lemma, it lies in the ACTUAL ambient operator domain D(A_P). Its compact support
is contained in U_P=Omega minus the union of all unselected collision surfaces.
Its ambient action equals the intrinsic off-contact action Sg, so

    q_4^D[g]=<g,Sg>=<g,A_Pg>.                           (4)

Let J_a be the COMPLETE ordered physical-cell isometry; its restriction to
hard-supported vectors is I_a. Ambient matching convergence is established for
all ten cases: four free singles; one full pair plus two free singles; two
disjoint full pairs. All have kinetic diagonal24kappa/a² and eventual lower
bound -2E_lat. For lambda>2E_lat,

    J_a(A_(P,a)+lambda)^(-1)J_a* ->(A_P+lambda)^(-1)
       strongly.

Put f=(A_P+lambda)g and h_a=(A_(P,a)+lambda)^(-1)J_a*f. Since complete cell
projections tend strongly to identity,

    J_ah_a->g,
    J_a A_(P,a)h_a=J_aJ_a*f-lambda J_ah_a->A_Pg.        (5)

Choose fixed real zeta in C_c^infinity(U_P), equal1 near supp g, and put
w_a(n)=zeta(an)h_a(n). Every nonzero node is hard-allowed, and all unselected
contact potentials vanish. Consequently

    q_a^hard[w_a]=a_(P,a)[w_a].                         (6)

This equality retains the complete diagonal and bonds to forbidden zero-valued
nodes. It is not a reflecting-boundary approximation.

The exact lattice identity, with unordered nearest-neighbor edges, is

    a_(P,a)[zeta_a h_a]=Re<zeta_a²h_a,A_(P,a)h_a>
      +(kappa/a²)sum_{edges{n,m}}(zeta(an)-zeta(am))²
                                  Re(conjugate(h_a(n))h_a(m)).       (7)

All diagonal potentials cancel exactly. There are12positive coordinate
directions; their cellwise cutoff difference quotients converge uniformly to
partial_l zeta. Strong L² convergence from(5), translation continuity, and
L¹ convergence of products show that the correction tends to
kappa integral|grad zeta|²|g|²=0. The operator term tends by(5) to
Re<zeta²g,A_Pg>=<g,A_Pg>. Thus

    I_aw_a->g, q_a^hard[w_a]->q_4^D[g], b_a[w_a]->b[g]. (8)

No global singular H¹ estimate is used. Nor is zeta assumed to preserve every
continuum contact operator domain: it leaves the target g unchanged, and(7)
is an exact finite-lattice identity. A compact cutoff may be chosen because
the target has positive hard-boundary and unwanted-contact support margins.

## 4. Every finite D0 vector, including shared matching overlaps

Write u=sum_{j=1}^m c_j g_j in the intrinsic initial domain. Construct(8) for
each generator on the same a-family, and set psi_a=sum_j c_j w_(j,a).
It is exactly hard-supported. Equation(3) gives b_a[psi_a]->b[u], and vector
convergence gives strong convergence of norms. Subtracting the common sigma
shift therefore yields q_a^hard[psi_a]->q_4^D[u].

For example channels{12} and{12,34} are included: their overlap need not be
contact-free. Their continuum cross form already exists by the reviewed
common-contact compatibility construction. Positivity and the independent
lower-limit theorem force the lattice cross energy to converge to that form.
No replacement by free H¹ energy on the overlap is made.

## 5. Bosonic recovery and full form-domain extension

For symmetric u in D0, apply the finite permutation average P_sym,a to psi_a.
Hard support is permutation invariant, and the embedding intertwines permutations.
The projected vectors still converge strongly to u. Symmetry makes P_sym,a a
reducing orthogonal projection for the positive form b_a, so

    b_a[P_sym,a psi_a]<=b_a[psi_a].

The lower-limit theorem supplies the reverse limiting bound b[u], hence the
projected shifted energies converge exactly. Strong norms allow subtraction
of sigma. Individual labelled generators need not be symmetric.

By construction D0 is a form core for q_4^D, with a defining shifted norm
equivalent to the b-norm. For u in D(q_4^D), choose u_m in D0 with b[u_m-u]->0,
using symmetric approximants for a bosonic target. Then u_m->u and b[u_m]->b[u].
For each m choose its exact-energy recovery and decreasing delta_m down to0
such that whenever0<a<delta_m,

    ||I_a psi_a^(m)-u_m||<1/m,
    |b_a[psi_a^(m)]-b[u_m]|<1/m.

Choose m(a)->infinity sufficiently slowly that a<delta_(m(a)). This diagonal
family remains hard-supported and symmetric when required, and gives(1).
The shift is removed ONLY along strongly convergent recovery sequences, never
from a general weak lower-limit inequality.

## 6. Scope and synthesis still to be reviewed

If independently cleared, this is the complete recovery upper half for the
specified N=4 Friedrichs form. Its new ingredient is positive-form assembly,
which uses the reviewed lower half and individual exact-energy generator
recoveries; it is not circular use of full Mosco convergence.

A separate synthesis must state the changing allowed-space projections,
generalized form convergence and precise resolvent/dynamics consequences.
No norm-convergence, arbitrary-N continuum limit, thermodynamic/full-Fock
construction or empirical result is asserted here. The physical exclusion,
range, scattering length, pair tuning and kinetic reference are unchanged.

## 7. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_four_particle_lattice_liminf_2026-09-14.md`: positive shift, lower half
  and all ten ambient physical-cell matching convergences.
- `cubic_four_particle_continuum_2026-09-13.md`: D0, restricted multiplier
  operator membership, common-contact symmetry and Friedrichs form core.
- `cubic_four_particle_double_contact_2026-09-13.md`: one matching's graph/cutoff
  recovery; the present argument covers all generators and their sums.
- `cubic_three_particle_lattice_recovery_2026-09-13.md`: exact cutoff identity;
  the N=3 contact-free cross-energy proof is not transplanted here.

Baseline684tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-four-particle-lattice-recovery-obowib2x/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 8. Independent mathematical review

Reviewer a49570c5d907ee8dd cleared the recovery upper half with no mathematical
blocker or required repair. They checked the noncircular positive-form cross-
assembly lemma including complex signs, actual generator operator membership,
all-ten-matching graph recovery, exact hard cutoff and boundary diagonal,
finite Gram assembly with shared contacts, positive-shift bosonic projection
and form-domain diagonalization.

Clearance covers equation(1) for every vector of the specified N=4 Friedrichs
form domain, including symmetric recovery. It does not certify a convergence
or dynamics synthesis. Direct reads only were used. Original proof-draft-v1.md
and its hash are preserved; no mathematical constant or physical hypothesis
changed. Independent auditor afaae1adc30dfd3a2 confirmed all684baseline files
unchanged, exactly one new note and685current files matching final-hashes.json.
Mathematical/scope sections1–6 are unchanged from the draft; the note matched
pre-audit-note.md. This is artifact clearance, not mathematical recertification
or evidence of absence of unrecorded execution. The audited final-hashes.json
is retained; closing-hashes.json binds final audit-status edits.
