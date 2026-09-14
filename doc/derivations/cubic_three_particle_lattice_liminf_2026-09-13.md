# Shifted lattice lower limit for the hard N=3 contact model

2026-09-13. Status: the shifted weak lower-limit theorem independently cleared;
independent saved-document/hash audit cleared with no blockers.
This note proves a candidate LOWER-LIMIT statement. Hard-supported singular
recovery and full lattice/dynamics convergence are not claimed.

## 1. Statement and spaces

Retain supplied R,kappa>0, finite s!=0, C_a=kappa/a² and the original tuned
pair coupling g_a^(-1)=-a²W/kappa+a³/(8pi kappa s). Let q_a be the hard N=3
lattice form, with wavefunctions extended by zero to forbidden configurations
and the ORIGINAL kinetic diagonal retained. Work on ordered particle coordinates
first and then restrict to bosons. Let q^D be the specified continuum Friedrichs
form on Omega_R={diam(X)>R}, identified with its zero-extended form on R^9.

Use centered physical cells

    C_(a,n)=a(n+[-1/2,1/2)^9),
    (I_a psi)(X)=a^(-9/2)psi(n) for X in C_(a,n).        (1)

On complete lattice sectors I_a is an isometry into physical L²(R^9). On hard
sectors the forbidden node values are zero. Its range projection for complete
sectors is the cell-average projection, tending strongly to I. Hard ranges are
not assumed to be the exact continuum allowed subspace at finite a.

Choose the SAME smooth theta and global all-subset partition as the continuum
construction, r0=R/8,r1=R/4, |theta'|<=L=8pi/R. Define

    E_lat=8kappa*1_{s>0}/s², M=18kappa L²,
    sigma=E_lat+M+eta, eta>0 fixed.                     (2)

These are auxiliary energy bounds/shifts, not new physical interactions.
For a_j down to0 and allowed lattice psi_j with I_(a_j)psi_j weakly converging
to U in L²(R^9), the candidate theorem is

    q^D[U]+sigma||U||²
      <=liminf_j(q_(a_j)[psi_j]+sigma||psi_j||²).        (3)

The left side is +infinity off the physically supported Friedrichs form domain.
This is a SHIFTED weak lower-limit statement: dropping the shift on a weakly
convergent sequence would require extra norm information and is not asserted.

## 2. Bridge from exact COM Fourier embedding to physical cells

Let E_a be the established two-particle Fourier embedding from the prior
operator-convergence note, followed by inverse continuum Fourier transform and
the Jacobian-one (c,r)<->(x1,x2) conversion. Let I_a^(2) be the six-dimensional
physical cell isometry, amplitude a^(-3), and T_a=I_a^(2)E_a*. Both are maps
into the SAME physical L²(R^6); ||T_a||<=1.

Choose the dense test class with FULL COM-relative Fourier transform
fhat in C_c^infinity(R_K³ x R_k³). Its support lies inside (B/a)x(B/a) for all
sufficiently small a; compactness in COM momentum alone is not being assumed
sufficient. Inverting the
EXACT gauged lattice transform gives

    (E_a*f)(n1,n2)=a³ f(an1,an2).                       (4)

Its phase is K dot a(n1+n2)/2+k dot a(n1-n2), retaining the half-integer COM
parity and the zone/gauge convention. The Fourier integral is supported away
from wrapping for small a on this fixed test class. It is not an independently
chosen COM sampling rule.

Consequently T_af is the piecewise constant physical sampling of the Schwartz
function f. The mean-value estimate over each cell, with integrable Schwartz
tails for the gradient, proves ||T_af-f||_2->0. Density and contraction extend
this to all L². Moreover

    ||T_a*f-f||²<=2||f||²-2 Re<f,T_af> ->0.             (5)

Thus both T_a and its adjoint converge strongly to I, though not asserted in
operator norm. Exact isometry identities give for each nonreal z

    I_a^(2)(H_a^(2)-z)^(-1)I_a^(2)*
      =T_a[E_a(H_a^(2)-z)^(-1)E_a*]T_a*.

The already proved Fourier-embedded strong-resolvent theorem and uniform
nonreal resolvent bound transfer it to physical cells. No closeness of embeddings
on arbitrary moving ultraviolet states is assumed. The convention H-z here is
the negative of the earlier z-H convention, with the same convergence content.

## 3. Complete ambient pair-plus-spectator form convergence

Let A_(e,a)=H_(e,a)^(2)+t_(k,a) on the complete ordered three-particle lattice.
Each pair retains TWO full one-particle kinetic operators. The separately
proved all-total-momentum estimate gives H_a^(2)>=-E_lat for sufficiently small
a, while t_(k,a)>=0. The free spectator Fourier symbol is
(2kappa/a²)sum_l(1-cos ak_l), tending to kappa|k|². Dominated convergence of
its resolvent multipliers, then the one-particle version of section2, gives
physical-cell generalized strong-resolvent convergence of the spectator.

To pass to the tensor sum, shift the pair by E_lat. For nonnegative operators,
embedded resolvent convergence implies embedded heat-semigroup convergence for
each t>0: apply continuous functional calculus to (H+E_lat+1)^(-1), using the
function x->exp[-t(x^(-1)-1)] on(0,1], continuous with value0 at0. Approximate
it uniformly by polynomials with zero constant term. Embedded powers multiply
correctly because the lattice embedding is an isometry. A real resolvent below
the uniform lower bound follows from the nonreal result by the resolvent identity
and bounded functional calculus (or continuation through a finite chain of
resolvent-identity neighborhoods in the common resolvent set).

The physical three-particle cell embedding factors exactly into pair and
spectator embeddings. The shifted tensor heat semigroups therefore converge
strongly on simple tensors and then all vectors by their contraction bounds.
Integrating against exp(-lambda t), lambda>0, proves generalized strong-resolvent
convergence of A_(e,a) to the specified ambient A_e=h_s+H_tan. The same argument
covers the complete ambient free operator A_(0,a). No bounded-total-momentum
or norm-uniform fiber hypothesis has been introduced.

## 4. Exact IMS rearrangement with a nonnegative edge remainder

Keep ALL eight global subset-product partition functions chi_J in R^9, even
though only the empty and three single-pair components survive on allowed
configurations. The global sum_J chi_J²=1 is essential at boundary nodes.
Sample them at lattice nodes a n. For each unordered configuration edge{n,m}, set

    D_a(n,m)=sum_J |chi_J(an)-chi_J(am)|²,
    w_a(n)=(kappa/(2a²))sum_{m adjacent n}D_a(n,m).       (6)

The full labelled configuration graph has18 neighbors. Exact discrete IMS and
Re(conjugate(u)v)=(|u|²+|v|²-|u-v|²)/2 give

    q_a[psi]=sum_(alpha=0,e) a_(alpha,a)[chi_(alpha,a)psi]
                    -sum_n w_a(n)|psi(n)|²+Rem_a[psi],
    Rem_a[psi]=(kappa/(2a²))sum_{unordered{n,m}}
                        D_a(n,m)|psi(n)-psi(m)|²>=0.   (7)

All vectors are globally zero extended. Forbidden-endpoint bonds remain in both
sums; their diagonal cost has not been discarded. On allowed configurations,
contact allocation to the selected matching is exact, so a_(alpha,a) really
is the ambient free or one-pair-plus-spectator form, not a weakened kinetic form.

Define the global continuum multiplier

    W_glob(X)=kappa sum_(J,l)|partial_l chi_J(X)|².

The global functions have bounded first/second derivatives: radial angles are
constant near pair coincidence and constant past r1. Taylor expansion of(6)
yields sup_n|w_a(n)-W_glob(an)|->0. W_glob has bounded gradient, so the cellwise
multiplier tilde(w_a) converges uniformly to W_glob on R^9 too.

For each scalar particle coordinate at most two pair angles depend on it. The
product-partition derivative identity gives
sum_J|partial_l chi_J|²=sum_e|partial_l theta_e|²<=2L². Along each directed
edge, integration of that derivative vector gives D_a<=2L²a². Hence

    0<=w_a<=18kappa L²=M.                              (8)

On Omega_R the nonmatching components vanish locally and W_glob=W_loc, the
continuum IMS correction. The estimates used for(8) are global and also cover
edges near forbidden configurations; the more economical allowed-edge constant
from the earlier stability proof is not required here.

Equation(7) avoids needing convergence of IMS expectations on weak sequences.
Its unfavorable term is a bounded multiplier and its leftover edge energy has
a favorable sign. No compactness assumption hides oscillatory states or mass
escaping to infinity.

## 5. Positive ambient forms and weak lower semicontinuity

For each surviving component define

    b_(alpha,a)[v]=a_(alpha,a)[v]+sum_n(sigma-w_a(n))|v(n)|²,
    b_alpha[v]=a_alpha[v]+integral(sigma-W_glob)|v|².    (9)

Equations(2),(8) and the ambient lower bounds imply b_(alpha,a)>=eta||v||²;
the same is true of b_alpha. The bounded multiplier convergence preserves the
ambient generalized strong-resolvent convergence from section3. Explicitly,
a resolvent Neumann expansion at a sufficiently negative spectral value is
uniformly convergent because the perturbations are uniformly bounded; each
embedded product converges strongly and the geometric series has a common
bound. Bounded functional calculus then transfers convergence to inverses of
the uniformly positive represented operators B_(alpha,a),B_alpha. All products
are between complete-sector isometries; no hard projection is used in this
ambient perturbation step.

For any I_a v_a weakly converging to v and any fixed f in L², positivity gives

    b_(alpha,a)[v_a]>=2 Re<I_a v_a,f>
                  -<f,I_a B_(alpha,a)^(-1)I_a*f>.

Pass to the limit and take the supremum over f. For a positive self-adjoint B,
that supremum is ||B^(1/2)v||², including +infinity outside its form domain
(by spectral truncation). Therefore

    b_alpha[v]<=liminf_a b_(alpha,a)[v_a].              (10)

This proof uses weak vector convergence and strong inverse convergence, not
convergence of energies from resolvents without justification.

For the AMBIENT forms only, the same reasoning supplies recovery: approximate
a form vector in form norm by v_m=B_alpha^(-1)f_m using spectral truncations,
and take v_(a,m)=B_(alpha,a)^(-1)I_a*f_m. For fixed m the embedded vectors and
energies converge by the inverse identity; a diagonal choice completes recovery.
This ambient recovery is not assumed to have hard allowed support.

## 6. Weak-limit support and final lower-limit inequality

Every particle in a centered physical cell differs from its node position by
at most sqrt(3)a/2, so each pair distance differs by at most sqrt(3)a. An
allowed-node cell wavefunction therefore vanishes on
{diam(X)<R-sqrt(3)a}. Testing a weak limit against functions on any fixed
{diam<R-delta} and exhausting delta>0 proves that U vanishes on {diam<R}.
The equality set {diam=R} is contained in finitely many distance-R hypersurfaces
of measure zero. Thus U is physically supported in closure(Omega_R).

Sampled smooth localizers converge uniformly on cells to their continuum values.
Weak convergence and its implied uniform L² norm bound yield

    I_a(chi_(alpha,a)psi_a) weakly -> chi_alpha U.        (11)

Combining(7) with the exact partition identity gives

    q_a[psi_a]+sigma||psi_a||²
      =sum_alpha b_(alpha,a)[chi_(alpha,a)psi_a]+Rem_a[psi_a]
      >=sum_alpha b_(alpha,a)[chi_(alpha,a)psi_a].       (12)

If the liminf is finite, restrict to a subsequence attaining it. Positivity
bounds each localized form along that subsequence. Apply(10),(11) and the finite
sum lower-limit inequality. Each chi_alpha U has finite ambient energy.
Together with physical support this places U in the GLOBAL V_max, not merely
an interior-local class. The independently proved boundary-density theorem gives
V_max=D(q^D). Finally W_glob=W_loc on Omega_R and the partition squares sum to1,
so

    liminf_a(q_a[psi_a]+sigma||psi_a||²)
      >=sum_alpha b_alpha[chi_alpha U]
      =q^D[U]+sigma||U||²,

proving(3). Infinite liminf is trivial; positivity rules out minus infinity.
Cell embeddings commute with particle permutations, so the argument restricts
to bosons. Localized components themselves are correctly treated on the full
ordered space rather than assumed individually symmetric.

## 7. Remaining recovery requirement

This proposed proof settles only the shifted hard-model lower limit. For full
convergence it remains to construct, for every original compact singular core
vector v in C_s, HARD-ALLOWED lattice psi_a such that

    I_a psi_a -> v strongly,
    limsup_a q_a[psi_a]<=q^D[v].                         (13)

The core contains genuine contact singularities. Sampling1/r at coincidence is
undefined, and ambient recovery from section5 does not guarantee physical hard
support. A lattice Green-vector construction or an independently justified
cutoff of ambient recovery is needed, including its contact and boundary energy
cost. Core form density can extend a proved(13), but cannot supply(13) itself.
No full generalized-Mosco, resolvent or dynamics convergence is claimed for the
hard three-particle model until this upper/recovery half is established.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_two_particle_operator_convergence_2026-09-13.md`: exact COM Fourier
  embedding and full two-particle strong-resolvent result.
- `cubic_three_body_exclusion_2026-09-13.md`: original hard diagonal, pair bound
  and contact allocation/discrete IMS structure.
- `cubic_three_particle_continuum_2026-09-13.md`: smooth partition, ambient forms
  and the specified continuum closure.
- `cubic_three_particle_boundary_density_2026-09-13.md`: global form-domain equality.
The physical-cell bridge and shifted lower-limit argument are new proof steps,
not claims that the old scalar scattering result already proved them.

Baseline677tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-particle-lattice-liminf-g8bw8wcb/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this proof note is added. No scientific imports,
tests, numerical evaluations or model changes are authorized.

## 9. Independent mathematical review

Reviewer a56e6842627239376 cleared the shifted weak lower-limit theorem with
no blocking gap. They checked the physical-cell bridge, ambient tensor and
bounded-perturbation convergence, exact global-partition IMS rearrangement,
positive-form variational lower limit, physical support and V_max identification.
The dense test class in section2 was made explicit as compactly supported in
BOTH continuum momentum variables, as requested; this clarifies the intended
class rather than claiming COM compactness alone suffices.

The original submitted proof-draft-v1.md and hash remain preserved. No physical
hypothesis, energy shift or mathematical constant changed. Clearance is limited
to the shifted liminf, not hard-supported recovery, full hard-model Mosco,
resolvent or dynamics convergence. Direct reads only were used. Independent
auditor a47620dd5279d5cc3 confirmed all677baseline files unchanged, exactly one
new note and678current files matching final-hashes.json. The note matched its
pre-audit snapshot; draft changes are status/review reporting and the explicit
full COM-relative Fourier test-class clarification only. Constants, shifts and
physical assumptions remain unchanged. This is document-integrity clearance,
not mathematical recertification or proof of absence of unrecorded execution.
The audited final-hashes.json is retained; closing-hashes.json binds the final
audit-status edits.
