# Local quantum compactness from an extensive fractional estimate

2026-09-14. Status: local quantum-compactness estimate independently cleared;
independent saved-document/hash audit completed.
This addresses averaged local density-operator compactness, not thermodynamic
equilibrium, a physical constrained observable algebra, or phase selection.

## 1. Setting and theorem target

Retain supplied R,kappa>0, finite s!=0 and the same zero-temperature canonical
box sequence Q_j=(0,L_j)^3, N_j/L_j³->rho in(0,1/(8R³)). Choose normalized
bosonic finite-box form vectors psi_j with energies E(N_j,L_j)+o(L_j³), and
zero-extend them into the ambient symmetric coordinate/Fock representation.
The canonical energy theorem bounds their energy density. No ground-state
attainment, stationarity or homogeneous phase is assumed.

For bounded U, spatially average translated density operators over Q_j and take
the ambient local Fock reduction D_(j,U), as defined in the positional note.
This is a mathematical local representation of finite-volume states, not an
identification of canonical fields on the constrained physical spaces.

Fix0<alpha<1/2. Put eps_R=kappa/R², L_theta=8pi/R,
E_s=2kappa*1_{s>0}/s² and B=E_s/2+kappa L_theta². Define

    T_alpha[psi]=sum_i ||(R|D_(x_i)|)^alpha psi||².

The proposed uniform-in-N estimate for zero-extended box form vectors is

    eps_R T_alpha[psi]
      <=C_(alpha,R,s,kappa)(q_(N,L)[psi]+(B+eps_R)N||psi||²).        (1)

This will give finite-rank local Fock projections Pi_(U,m) increasing strongly
to identity, with uniform tails Tr[(I-Pi_(U,m))D_(j,U)] tending to zero. The
right side of(1) uses singular contact FORM energy, not a presumed finite bare
kinetic energy of psi.

## 2. Fractional regularity in one matching with depth-independent constants

The established relative form representation u=phi+G_mu Q, with fixed
mu>max(0,1/s), has phi in H¹ and positive shifted norm controlling phi and Q.
It extends with identical constants to arbitrary Hilbert-valued spectators.
The Fourier decay of G_mu is proportional to (|p|²+mu²)^(-1). Its squared
H^alpha high-frequency integral behaves as integral_1^infinity r^(2alpha-2)dr,
finite precisely for alpha<1/2. The regular-plus-Green decomposition therefore
bounds fractional relative energy by the shifted pair form. Comparing positive
shifts with h_s>=-E_s gives

    eps_R ||(R|D_r|)^alpha u||²
       <=C_alpha(h_s[u]+(E_s+eps_R)||u||²).             (2)

Constants depend on fixed parameters, not spectator dimension. Free centers
and unmatched particles obey the analogous Fourier bound by their positive
kinetic energy plus eps_R||u||². In a matched pair, p_i=p_r+p_c/2 and
p_j=-p_r+p_c/2; the elementary Fourier inequality for |p_i|^(2alpha)+|p_j|^(2alpha)
converts relative and center estimates into individual-particle estimates.

Sum POSITIVE constituent estimates, rather than compare entire tensor norms
with depth-dependent constants. For every matching M this gives

    eps_R T_alpha[v]
      <=C_alpha(a_M[v]+(E_s|M|+eps_R N)||v||²),          (3)

with one constant independent of N and the matching count. The pair/single
number factors are absorbed by eps_R N. No simultaneous mixed-charge map or
free-tangential H^(1/2) condition is used. Core approximation and closed positive
constituent forms extend(3) to the whole ambient matching form domain.

## 3. Full partition-vector reconstruction avoids channel-count losses

Use the full sine/cosine partition vector boldchi=(chi_J)_J over ALL edge
subsets, including nonmatchings outside the allowed region. Its norm is1
everywhere. Set v_J=chi_J psi; nonmatching v_J vanish on physically supported
psi. For X'=X+h e_i, where the three-dimensional displacement h moves particle i,
reconstruction gives

    psi(X')-psi(X)
      =boldchi(X') dot [boldv(X')-boldv(X)]
                  +[boldchi(X')-boldchi(X)] dot boldv(X).

Consequently

    |psi(X')-psi(X)|²
      <=2sum_J |v_J(X')-v_J(X)|²
          +2|psi(X)|²||boldchi(X')-boldchi(X)||².         (4)

The derivative bound for the second term is independent of N on its relevant
support. If psi(X)!=0 then the other N-1 particles retain their hard triple
exclusion while i moves along ANY path; intermediate full configurations need
not be allowed. At every location of i, at most TWO other particle occurrences
can be within R/4 of it. Three such others would themselves have diameter<R/2,
contradicting their unchanged triple constraint. The global product derivative
identity therefore yields along the straight path

    sum_J |grad_(x_i)chi_J|²
       =sum_(k!=i)|grad_(x_i)theta_ik|²<=2L_theta².

Unit-vector boundedness and path integration imply

    ||boldchi(X+h e_i)-boldchi(X)||²
       <=min(4,2L_theta²|h|²) whenever psi(X)!=0.        (5)

This remains true for N=1,2 and the zero case is vacuous. Integrate(4) against
the three-dimensional fractional difference kernel |h|^(-3-2alpha), and then
sum over i. The integral of the bound(5) is finite for0<alpha<1; rescaling uses
R L_theta=8pi. The standard Fourier/difference identity for fractional seminorms,
valid with nonnegative possibly infinite integrals by Tonelli, gives

    T_alpha[psi]<=2sum_M T_alpha[chi_M psi]+C_alpha N||psi||².        (6)

The right side is finite by(3) and the known form localization. Thus the argument
proves fractional regularity rather than assuming it. There is no exponential
matching-count factor: Cauchy-Schwarz was in the unit partition-vector space.

## 4. Extensive positive matching energy

Set

    S_N[psi]=sum_M(a_M[chi_M psi]
                       +(E_s|M|+eps_R N)||chi_M psi||²).

Each term is nonnegative and S_N>=eps_R N||psi||². The reviewed completed-form
IMS identity, restricted to the finite-box closure, gives

    S_N=q_(N,L)[psi]+<W_loc>+E_s sum_M|M|||chi_M psi||²
                                     +eps_R N||psi||².

Using W_loc<=kappa N L_theta² and |M|<=N/2 yields

    S_N<=q_(N,L)[psi]+(B+eps_R)N||psi||².               (7)

Combining(3),(6),(7), and absorbing the additional N term by S_N's positive
lower bound, proves(1). The vacuum N=0 has both sides zero. This is where
negative contact cancellation is controlled: constituents are shifted positively
before fractional regularity is extracted.

For the canonical sequence, q_j/L_j³ and N_j/L_j³ are bounded. Applying(1) gives

    sup_j (T_alpha[psi_j]+N_j)/L_j³<infinity,            (8)

after discarding finitely many inadmissible initial boxes or absorbing a finite
initial segment of form vectors. In fact a uniform energy-density upper bound,
not near-minimality itself, suffices for this estimate.

## 5. Spatially averaged sharp-window fractional control

Let U be a bounded Lipschitz window. Define an AUXILIARY one-particle form on
L²(U) by b_U[f]=||f||²+||(R|D|)^alpha tilde f||², using zero extension. This
is a compactness diagnostic, not a local physical Hamiltonian or a contact-form
assignment after sharp restriction.

For Hilbert-valued u(x), expand the difference of1_U(x-y)u(x) and integrate
over y in R³. The fractional seminorm then obeys

    integral_R3 [1_U(.-y)u]^2_(dotHalpha)dy
      <=2|U|[u]^2_(dotHalpha)
         +C_alpha||u||² integral_R3
                |U symmetric_difference(U+h)|/|h|^(3+2alpha)dh.     (9)

This follows from separating the wavefunction difference from the indicator
difference and using squared triangle inequality and Tonelli. For Lipschitz U,
the symmetric-difference volume is <=C_U|h| near0, so the radial integral there
is bounded by a constant times integral_0^1 r^(-2alpha)dr. It is finite for
alpha<1/2; at large h use2|U|. This accounts for sharp local cuts, including
cuts that split a contact pair. Ordinary H¹ control after that operation would
not be justified.

Apply(9) in every particle coordinate with all others as Hilbert-valued spectators.
The local ambient Fock reduction's one-particle density is the spatial compression
of the global one-particle density. By nonnegative quadratic-form approximation
(or finite spectral cutoffs first), the summed sharp-restriction expectation
therefore equals the expectation of dGamma(B_U) in the local reduction, where
B_U>=I is represented by b_U. Spatial averaging over Q_j can be bounded by
integration over all translations because the integrand is nonnegative. Equations
(8),(9) imply

    Tr[D_(j,U) dGamma(B_U)]
       <=C_(alpha,U,R)(T_alpha[psi_j]+N_j)/L_j³<=C_U.    (10)

This is a uniform local SPATIAL-mode bound, not merely a local number bound.

## 6. Finite-rank tails and trace-norm compactness

The form b_U is densely defined and closed: zero extension is isometric in L²,
and convergence in its shifted fractional norm preserves exterior zero support.
Its domain embeds compactly into L²(U). Indeed all zero extensions have bounded
support and fractional Fourier tails uniformly controlled; restricting a bounded
frequency projection to bounded spatial support gives compact approximations.
This is fractional compactness of a diagnostic form, not use of ordinary Rellich
on the singular global state.

B_U therefore has compact resolvent with eigenvalues beta_l>=1 tending to infinity.
On ambient Fock(L²(U)) define Pi_(U,m)=1_[0,m](dGamma(B_U)). These projections
are finite rank: total energy<=m permits at most floor(m) particles and only
finitely many one-particle modes, hence finitely many bosonic occupation patterns.
They increase strongly to identity, with the vacuum included. From(10), for m>0,

    sup_j Tr[(I-Pi_(U,m))D_(j,U)]<=C_U/m ->0.            (11)

Positive trace-one density matrices with this tail bound are trace-norm relatively
compact. If epsilon=Tr[(I-Pi)D], then
||D-Pi D Pi||_1<=epsilon+2sqrt(epsilon); finite-rank compressions form a bounded
precompact set. Packing's deterministic local number cap remains valid, but(10)
already provides the number control needed for this particular tail argument.

For any bounded measurable U, choose a containing bounded cube V. The ambient
factorization over V=U union(V minus U) identifies D_(j,U) as the partial trace
of D_(j,V). Trace-norm contraction transfers relative compactness to U, without
assuming a Lipschitz boundary there. Uniform finite-rank tightness follows from
trace-norm precompactness: approximate a finite net of density matrices by
finite-rank compressions, then enlarge their finite-dimensional spans. The
explicit C_U/m bound is asserted only for the diagnostic Lipschitz-window
construction above.

A diagonal subsequence along a countable exhausting family of cubes gives
compatible local density-operator limits, by continuity of partial trace.
This is local quantum compactness in an AMBIENT representation. Identifying
its physical constrained observable algebra and equilibrium dynamics remains
separate, as do minimization retention, stationarity and phase purity.

## 7. Scope and next state question

If independently cleared, the result establishes an N-uniform extensive
fractional one-particle estimate for every fixed0<alpha<1/2 and uniform local
finite-mode tightness of the specified spatially averaged near-minimizers.
Contact singularities and sharp spatial cuts are treated below their critical
H^(1/2) threshold rather than suppressed.

No thermodynamic ground-state condition, infinite-volume physical dynamics,
canonical CCR, extremality, correlation decay, stiffness, sound or relativity
is established. The auxiliary fractional operator B_U is not an added model
term. The original state sequence, averaging prescription, density and R,s,kappa
are unchanged. The next separate step would identify what state on a precisely
defined observable algebra these compatible local limits supply and which,
if any, equilibrium properties can be proved to pass to it.

## 8. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_density_retaining_positional_limit_2026-09-14.md`: spatial averaging,
  ambient local Fock diagnostic and missing finite-mode tail criterion.
- `cubic_canonical_energy_density_2026-09-14.md`: uniform energy-density bound.
- `cubic_finite_particle_continuum_2026-09-14.md`: extensive stability and matching
  constituent forms with Hilbert-valued relative contact factors.
- `cubic_finite_particle_boundary_density_2026-09-14.md`: completed-form localization.
- `cubic_fixed_density_phase_gate_2026-09-14.md`: prescribed finite-box closure.
- `cubic_three_particle_continuum_2026-09-13.md`: normalized contact Green function
  and relative closed form.

Baseline696tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-local-quantum-compactness-kt06m2lu/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical calculations, model edits or publication are authorized.

## 9. Independent mathematical review

Reviewer a93b8db3629a99f65 cleared the compactness theorem with no blocker or
required repair. They checked the spectator-independent fractional contact
estimate, positive constituent summation without an N-dependent energy multiplier,
full partition-vector reconstruction and forbidden-path derivative bound,
completed IMS control of contact cancellation, averaged sharp-window estimates
including split contacts, local partial-trace expectation and finite-rank Fock tails.

Clearance covers the stated N-uniform fractional estimate, averaged local
finite-mode tightness and compatible ambient local density-operator subsequential
limits. It does not establish equilibrium, retention of minimization, physical
constrained CCR, infinite-volume dynamics or phase properties. Direct analytical
reads only were used. The original proof-draft-v1.md and its hash are preserved;
no mathematical constant or physical assumption changed. Independent auditor
a4df603fcec2a1926 confirmed all696baseline files unchanged, exactly one added
note and697current files matching final-hashes.json. Mathematical sections1–7
are byte-identical to the submitted draft; the note matched pre-audit-note.md.
This is document/hash clearance, not mathematical recertification or proof of
absence of unrecorded execution. The audited final-hashes.json is retained;
closing-hashes.json binds final audit-status edits.
