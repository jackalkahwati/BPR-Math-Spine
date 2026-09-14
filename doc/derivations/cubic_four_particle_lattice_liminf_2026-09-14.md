# Four-particle shifted lattice lower limit

2026-09-14. Status: shifted N=4 lower-limit theorem independently cleared;
independent saved-document/hash audit cleared.
This proves a candidate lower-half statement only. Four-particle hard recovery
and the convergence synthesis are not claimed by this note.

## 1. Fixed model and statement

Retain supplied R,kappa>0, finite s!=0 and the original coupling
 g_a^(-1)=-a²W/kappa+a³/(8pi kappa s). Let Omega in R^12 require diameter>R
for EACH of the four three-label subsets. Let q_a^hard be the ordered hard
lattice form, defined by zero extension with original24kappa/a² kinetic diagonal
and all six pair contacts. Let q_4^D be the specified continuum Friedrichs form.

Use centered ordered physical cells C_(a,n)=a(n+[-1/2,1/2)^12), with
I_a psi=a^(-6)psi(n) on the corresponding cell, zero for forbidden nodes.
No independently discretized center coordinates are introduced.

For the same smooth continuum angle theta, r0=R/8,r1=R/4, choose

    L=8pi/R, E_lat=8kappa*1_{s>0}/s²,
    M=36kappa L², sigma=2E_lat+M+eta, eta>0.

The proposed assertion is: if a_j down to0 and I_(a_j)psi_j weakly->U in L²,
then

    q_4^D[U]+sigma||U||²
      <=liminf_j(q_(a_j)^hard[psi_j]+sigma||psi_j||²).   (1)

The left side is +infinity off the physical zero-extended Friedrichs form
domain. The shift cannot be dropped on weakly convergent sequences. All
statements use sufficiently small a for the inherited pair bound; no extra
physical parameter or counterterm is supplied.

## 2. Ten complete ambient matching operators

Let Mset consist of the empty matching, six single matchings and three double
matchings. Define on the COMPLETE ordered lattice

    A_(P,a)=(kappa/a²)(24I-Adj)+g_a sum_{e in P}1_(Sigma_e).

The continuum A_P is the previously constructed tensor sum of full pair-contact
operators for pairs in P and free single-particle operators for unmatched labels.
The three cases must be distinguished explicitly:
- Empty matching: four free single-particle factors.
- One pair: one full two-particle contact factor plus two free singles;
  their kinetic diagonals sum12+6+6=24 in units kappa/a².
- Two pairs: two disjoint full two-particle factors, kinetic diagonals12+12=24.

Every physical cell embedding factors into those indicated factors after a
fixed permutation of labels. The established pair physical-cell generalized
strong-resolvent theorem and the free one-particle symbol convergence give
convergence of each factor. Shift each pair by E_lat to make it nonnegative;
embedded heat convergence follows from continuous resolvent functional calculus.
At t=0 the complete cell projections converge strongly to identity. Exact heat
factorization gives convergence on simple tensors, and contraction bounds extend
it to all vectors. Laplace integration, undoing the pair shifts, gives for
lambda>2E_lat

    I_a(A_(P,a)+lambda)^(-1)I_a* ->(A_P+lambda)^(-1)
       strongly.                                      (2)

The same holds at nonreal parameters by continuous resolvent transformation.
This invokes the previously proved physical-cell bridge, not a new assumption
that distinct momentum embeddings are interchangeable on ultraviolet states.
All ten matching operators obey eventually

    A_(P,a)>=-|P|E_lat>=-2E_lat,
    A_P>=-|P|E_s>=-2E_lat,
    E_s=2kappa*1_{s>0}/s².                             (3)

These are finite N=4 tensor arguments. No arbitrary-N convergence or improper
free-tangential double-contact charge condition is used.

## 3. Global partition and exact discrete rearrangement

Let E be the six labelled pair edges. Define chi_J for ALL64subsets J by the
product of selected sin theta_e and unselected cos theta_e. Globally
sum_J chi_J²=1. On Omega active edges form a matching: two incident distances
<R/4 force their third side<R/2, violating exclusion. Thus only the ten
matching components survive on physical configurations and hard lattice vectors.

At coincidence of an unselected pair, the corresponding cos theta(0) factor
vanishes. Therefore the all-six-pair interaction evaluated on chi_(P,a)psi is
exactly the selected matching potential. Each localized form is the complete
ambient a_(P,a) with all four particles' full kinetic energy, not a diluted
coupling. Define for unordered configuration edges

    D_a(n,m)=sum_J|chi_J(an)-chi_J(am)|²,
    w_a(n)=(kappa/(2a²))sum_{m adjacent n}D_a(n,m).       (4)

There are24neighbors per full configuration node. Discrete IMS and
Re(conjugate(u)v)=(|u|²+|v|²-|u-v|²)/2 give the EXACT identity

    q_a^hard[psi]=sum_{P in Mset}a_(P,a)[chi_(P,a)psi]
                    -sum_n w_a(n)|psi(n)|²+Rem_a[psi],
    Rem_a[psi]=(kappa/(2a²))sum_{unordered{n,m}}
                    D_a(n,m)|psi(n)-psi(m)|²>=0.        (5)

All values are globally zero extended. Forbidden-endpoint bonds remain and
retain their diagonal costs. Keeping all64global components in(4) is important:
the neighboring endpoint may be forbidden, even if one endpoint is allowed.

## 4. Uniform control of the bounded correction

The product-partition derivative identity is

    sum_J|partial_l chi_J|²=sum_e|partial_l theta_e|².

A scalar coordinate affects at most THREE pair angles. Thus this is <=3L²
on the whole ambient configuration space, not only on allowed states. Integral
Cauchy-Schwarz along an edge gives D_a<=3L²a², and hence

    0<=w_a<= (kappa/(2a²))*24*3L²a²=36kappa L²=M.      (6)

Define W_glob=kappa sum_(J,l)|partial_l chi_J|². All first and second derivatives
of the chi_J are globally bounded: each radial angle is constant near coincidence
and outside a fixed transition annulus. Uniform Taylor expansion in all12positive
and12negative coordinate directions shows

    sup_n|w_a(n)-W_glob(an)|->0.

Indeed each squared difference is a² times the squared derivative plus uniform
O(a³); paired directions supply exactly the normalization defining W_glob.
Bounded second derivatives make W_glob globally Lipschitz. Its cellwise comparison
therefore also gives ||tilde(w_a)-W_glob||_infinity->0, and0<=W_glob<=M.
On the OPEN allowed Omega, every nonmatching component vanishes in a neighborhood
of each point, including its derivatives. Thus W_glob=W_loc there.

This rearrangement handles the negative IMS multiplier without assuming its
expectation converges on arbitrary weak sequences. The residual edge term in(5)
has a favorable sign, so no hidden compactness is required.

## 5. Positive ambient forms and variational liminf

For every matching put

    b_(P,a)[v]=a_(P,a)[v]+sum_n(sigma-w_a(n))|v(n)|²,
    b_P[v]=a_P[v]+integral(sigma-W_glob)|v|².

By(3),(6), both are >=eta||v||². Let B_(P,a),B_P be their represented positive
operators. The ambient convergence(2) and uniform multiplier convergence imply

    I_a B_(P,a)^(-1)I_a* -> B_P^(-1) strongly.           (7)

One direct justification uses a resolvent Neumann series at a sufficiently
large additional positive shift. The multiplication perturbations are uniformly
bounded, every embedded product converges strongly on the COMPLETE sector
ranges, and the series has a common geometric bound. Positivity by eta then
allows removal of that shift using the uniformly convergent inverse series
B^(-1)=sum_{k>=1}tau^(k-1)(B+tau)^(-k), tau>0. The norm ratio is at most
tau/(eta+tau)<1. No hard projection enters this ambient perturbation argument.

If I_a v_a weakly->v, for every ambient f positivity gives

    b_(P,a)[v_a]>=2Re<I_a v_a,f>
                         -<f,I_a B_(P,a)^(-1)I_a*f>.

Take lower limits using(7), then the supremum over f. The positive inverse
variational formula, verified by spectral truncation also outside the form
domain, yields

    b_P[v]<=liminf_a b_(P,a)[v_a].                      (8)

This uses the completed tensor-sum form space, not independent free-tangential
charges or mixed-charge continuity. It also retains the shifted norm under
weak convergence rather than subtracting possibly lost norm mass.

## 6. Physical support and specified-domain membership

Within a physical centered cell each particle moves from its node by at most
sqrt(3)a/2. Every pair distance and every triple diameter d_T therefore change
by at most sqrt(3)a. For EACH of the four triples T, allowed-node wavefunctions
vanish on {d_T<R-sqrt(3)a}. Testing weak convergence on {d_T<R-delta}, then
exhausting delta>0, shows U vanishes on the union of all {d_T<R}. Every
{d_T=R} lies in finitely many distance-R hypersurfaces and has measure zero.
Thus U=0 almost everywhere on Omega^c.

Uniform convergence of sampled smooth localizers on cells and weak boundedness
of the original sequence give

    I_a(chi_(P,a)psi_a) weakly ->chi_P U.                (9)

The partition and(5) imply

    q_a^hard[psi_a]+sigma||psi_a||²
       =sum_P b_(P,a)[chi_(P,a)psi_a]+Rem_a[psi_a]
       >=sum_P b_(P,a)[chi_(P,a)psi_a].                 (10)

If its liminf is finite, choose a subsequence attaining it. Each of the ten
nonnegative localized forms is bounded. Apply(8),(9) and the finite-sum liminf
inequality. All chi_P U have finite ambient form energy, so physical support
places U in the GLOBAL V_max of the reviewed N=4 boundary-density theorem.
That theorem gives V_max=D(q_4^D) and its exact localized form identity.
Using W_glob=W_loc on Omega and the norm partition, we conclude

    liminf_a(q_a^hard[psi_a]+sigma||psi_a||²)
       >=sum_P b_P[chi_P U]
       =q_4^D[U]+sigma||U||²,

which is(1). Infinite liminf is immediate, and(10) rules out minus infinity.
Physical cells commute with particle permutations, so the assertion restricts
to bosons. Localized vectors need not themselves be symmetric and have correctly
been treated on ambient ordered spaces.

## 7. Remaining recovery and cross-matching task

This lower-half proof uses localized energies, not an unproved assembly of
vectors with cancelling singularities. It preserves the genuine double-contact
tensor domains. It does not supply hard recovery for every vector of the
global N=4 initial domain D0.

The separate upper-half target is: for each u in D0, construct hard-allowed
psi_a with I_a psi_a->u and limsup q_a^hard[psi_a]<=q_4^D[u], then extend by
form density. Channelwise recovery must account for polarized cross energies.
For channels{12} and{12,34}, overlaps may retain contact12; comparing their
cross energy with a free H¹ expression would be incorrect. Their continuum
comparison is through the common-contact A_{12}, as already proved.

There is a possible alternative to a new explicit shared-contact derivative
estimate: once this shifted liminf and exact-energy recoveries of each generator
are independently established, positive-form polarization may force the cross
energies of compatible recovery families to converge. That assembly lemma and
its hypotheses must be proved separately, not assumed here. Neither hard
recovery, full Mosco/resolvent/dynamics convergence nor an arbitrary-N or
empirical conclusion is claimed by the present lower-limit statement.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_four_particle_continuum_2026-09-13.md`: specified matching operators,
  global Friedrichs candidate and shared-contact compatibility.
- `cubic_four_particle_boundary_density_2026-09-13.md`: global form-domain equality.
- `cubic_four_particle_double_contact_2026-09-13.md`: disjoint-pair tensor
  convergence and the limits of isolated matching recovery.
- `cubic_three_particle_lattice_liminf_2026-09-13.md`: positive IMS rearrangement
  and physical-cell bridge, rederived above with N=4 counts and matching cases.

Baseline683tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-four-particle-lattice-liminf-onhttsg1/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 9. Independent mathematical review

Reviewer af598ccd52651fc54 cleared the shifted N=4 lower-limit theorem with
no mathematical blocker or required repair. They checked all ten ambient matching
convergences and full kinetic diagonals, the64global partition and positive
unordered-edge remainder, the36kappa L² multiplier bound and uniform limit,
varying-range bounded perturbations, positive inverse variational liminf, all
four triple support conditions and the global Friedrichs-domain identification.

Clearance covers equation(1) only. Cross-matching recovery/assembly, the upper
half, full convergence and broader claims remain outside it. The reviewer used
direct reads only. Original proof-draft-v1.md and its hash are preserved;
no mathematical constant or physical hypothesis changed. Independent auditor
a9e6b595a606ee528 confirmed all683baseline files unchanged, exactly one added
note and684current files matching final-hashes.json. Mathematical sections1–7
are unchanged from the submitted draft; the note matched pre-audit-note.md.
This is artifact clearance, not mathematical recertification or proof of absence
of unrecorded execution. The audited final-hashes.json is retained; closing-hashes.json
binds the final audit-status edits.
