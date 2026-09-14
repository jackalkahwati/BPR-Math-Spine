# Convergence on constrained particle-number sector direct sums

2026-09-14. Status: constrained sector-direct-sum theorem independently cleared;
independent saved-document/hash audit cleared.
This is a normalizable-vector direct-sum statement, not a thermodynamic limit
or an identification of canonical field operators or a physical vacuum.

## 1. Spaces and operator domains

Retain supplied R,kappa>0, finite s!=0 and the original tuned g_a. For each
finite N let H_(N,a) be the symmetric allowed lattice sector, H_N the symmetric
continuum L²(Omega_N) sector, and A_N the symmetric ambient L²(R^(3N)) space.
At N=0 all are C. Define square-summable Hilbert direct sums

    F_a=direct_sum_(N>=0) H_(N,a),
    F=direct_sum_(N>=0) H_N, A=direct_sum_(N>=0) A_N.

These include vectors with infinitely many nonzero components; they are not
restricted to algebraic finite-sector sums. Define isometries I_a=direct_sum I_(N,a)
and iota=direct_sum iota_N into A, retaining the exact centered physical cells
and amplitude a^(-3N/2) sectorwise. Let P_a=I_aI_a*, P=iota iota*.

Choose a common a_g>0 on which g_a is finite. Its defining expression contains
no N, so this choice is independent of particle number. Every fixed-sector
lattice operator exists there. Particular convergence thresholds may depend
on N; a finite head of sectors uses the minimum of finitely many thresholds,
not a positive infimum over every N.

Define K_a^F=direct_sum K_(N,a) and H^F=direct_sum H_N^D on their maximal
direct-sum graph domains: every component lies in its sector operator domain
and the sum of squared norms of operator images is finite. These operators
are self-adjoint without an N-independent lower bound. Indeed the component
resolvents at nonreal z have norm<=1/|Im z|; the identity
H_N(H_N-z)^(-1)f_N=f_N+z(H_N-z)^(-1)f_N shows their output is in the direct-sum
domain. Finite-sector truncations of domain vectors converge in graph norm.
No algebraic intersection with D(Nhat) is substituted for these domains.

## 2. Projections and generalized strong resolvents

Let Pi_<=M truncate A to sectors0..M. The reviewed fixed-sector projection
convergence implies convergence on every finite head. Since projections have
norm<=1, for f in A

    ||(P_a-P)f||<=||(P_a-P)Pi_<=M f||+2||(I-Pi_<=M)f||.

Take a down to0 at fixed M, then M to infinity. Thus P_a->P strongly, not
ambient identity. Square-summability supplies the tail bound, with no finite
expected-particle-number requirement.

For nonreal z define R_a(z)=I_a(K_a^F-z)^(-1)I_a* and
R(z)=iota(H^F-z)^(-1)iota*. These are direct sums of the reviewed embedded
sector resolvents, with norms<=1/|Im z| uniformly in a,N. Therefore

    ||(R_a-R)f||<=||(R_a-R)Pi_<=M f||
                          +(2/|Im z|)||(I-Pi_<=M)f||.

Finite-head convergence and removal of the tail prove

    I_a(K_a^F-z)^(-1)I_a* ->iota(H^F-z)^(-1)iota*
       strongly on A.                                 (1)

This is a GENERALIZED embedded resolvent, zero on the forbidden complement,
not an ordinary resolvent of an ambient densely defined operator. Optional
zero completion on those complements adds -z^(-1)(I-P_a) and -z^(-1)(I-P),
giving ordinary strong-resolvent convergence of the comparison completions.
Those zero actions do not assign physical dynamics to forbidden states.

## 3. Compact-time unitaries and moving direct-sum states

Set U_a(t)=I_a exp(-itK_a^F)I_a* and U(t)=iota exp(-itH^F)iota*.
Their norms are at most1 and their values at0 are P_a and P, not ambient identity.
For every fixed f and finite T,

    sup_|t|<=T||(U_a(t)-U(t))f||
      <=sup_|t|<=T||(U_a(t)-U(t))Pi_<=M f||
                          +2||(I-Pi_<=M)f||.

The finite-head squared error is bounded by a finite sum of the reviewed
sector compact-time errors, each tending to0. Removing the square-summable tail
gives

    sup_|t|<=T||[I_a exp(-itK_a^F)I_a*
                       -iota exp(-itH^F)iota*]f|| ->0. (2)

This needs no uniform rate in N, no finite energy, and no finite N expectation
for f. It remains vectorwise convergence, not uniformity over all moving states.

If I_a psi_a->iota u strongly, contraction yields

    sup_|t|<=T||I_a exp(-itK_a^F)psi_a-iota exp(-itH^F)u||
      <=||I_a psi_a-iota u||+sup_|t|<=T||(U_a(t)-U(t))iota u|| ->0.   (3)

The resolvent analogue uses the factor1/|Im z| for the initial error. Strong
convergence itself controls particle-number tails:
||(I-Pi_<=M)I_a psi_a||<=||I_a psi_a-iota u||+||(I-Pi_<=M)iota u||.
A normalized sequence concentrated only in N(a)->infinity does not meet this
strong-convergence hypothesis. A superposition with infinitely many nonzero
finite-N components can meet it and is included.

## 4. The reference phase is not scalar across sectors

Let H_a^(hop,F) be the maximal sector direct sum of hard hopping/contact operators
without the threshold diagonal. Sectorwise K_(N,a)=H_(N,a)^hop+6kappa N/a².
The bounded unitary groups therefore satisfy

    exp(-itH_a^(hop,F))
       =exp[it(6kappa/a²)Nhat] exp(-itK_a^F).            (4)

This identity holds on all direct-sum vectors without assuming D(Nhat).
Generator-sum notation must mean the maximal sectorwise spectral sum, not an
unproved equality of algebraic-sum domains.

The prefactor in(4) is NOT a global scalar for superpositions of particle number.
It changes N-versus-M relative phase by exp[it6kappa(N-M)/a²]. It cancels for
number-preserving observables but may matter for number-changing observables.
Number is conserved by these Hamiltonians; a number superselection rule has
not been derived. Thus(1)–(3) concern the THRESHOLD-REFERENCED direct-sum model,
not convergence of raw unreferenced evolution. Calling (4) a harmless common
phase for all direct-sum states would be incorrect.

## 5. A separately labeled auxiliary number shift

The reviewed extensive bounds are, on a common sufficiently small a-range,

    K_(N,a)>=-B_lat N,
    B_lat=kappa[4*1_{s>0}/s²+48pi²/R²],
    H_N^D>=-B_cont N,
    B_cont=kappa[1_{s>0}/s²+64pi²/R²].

The lattice range comes from the all-N hard-stability proof and is independent
of N; intersect it with a<a_g. Let b=max(B_lat,B_cont) and define maximal
sector sums A_a=direct_sum(K_(N,a)+bN), A=direct_sum(H_N^D+bN).
Both are nonnegative. Their domains are defined by squared sector-image sums;
no claim D(A)=D(H^F) intersect D(Nhat) is needed.

For each fixed sector its shifted heat convergence follows from the reviewed
sector theorem by multiplying by exp(-tbN). Globally these auxiliary heat
operators are contractions. The same finite-head/tail argument proves

    sup_0<=t<=T||[I_a exp(-tA_a)I_a*-iota exp(-tA)iota*]f|| ->0,      (5)

including strongly convergent moving initial vectors. This is an AUXILIARY
number-shifted theorem, not unshifted full-direct-sum heat convergence. Nonreal
resolvent convergence of A_a to A likewise follows by the uniform resolvent
bound and fixed-sector spectral-parameter shift z->z-bN.

For completeness, the nonnegative sector-sum forms have a generalized Mosco
limit. Here is an explicit sector argument that does not subtract a weakly lost
norm term. Write T_a=K_(N,a)+bN>=0 and T=H_N^D+bN>=0, with embeddings J_a,J
and physical projections P_a,P. The bounded embedded resolvents
B_a=J_a(T_a+1)^(-1)J_a* converge strongly to B=J(T+1)^(-1)J*. This follows
from the shifted nonreal resolvent theorem by nonnegative zero completion and
subtraction of the complementary resolvent; +1 avoids inversion at zero.
For fixed f, v_a=B_af->v=Bf and their operator images converge to Pf-Bf,
since J_a T_a J_a*v_a=P_af-B_af. Their form energies therefore converge as
< B_af,P_af-B_af > -> < Bf,Pf-Bf >.

For weakly convergent finite-energy u_a->u, strong convergence of projections
forces u=Pu. Nonnegativity gives
q_a[u_a]>=2Re<u_a,P_af-B_af>-q_a[B_af]. Take limits and the supremum over f.
The vectors Bf range over the physical operator domain; spectral truncation
shows that this supremum is q[u], including infinity outside its form domain.
Thus the weak lower inequality is proved directly, without subtracting norms
from a shifted liminf. The same B_af give exact-energy recovery of operator-
domain vectors; spectral truncation and diagonalization extend it to all form
vectors.

For the sums, retain finitely many NONNEGATIVE sector energies for liminf,
then increase the sector cutoff. Recovery truncates a finite-total-form-energy
target in norm and energy, recovers the finite head, and diagonalizes. This
proves auxiliary generalized Mosco convergence for A_a,A only, not an unshifted
globally semibounded form.

An extensive estimate -BN alone does not establish an N-independent lower
bound for K_a^F or H^F. It does not justify unshifted heat convergence for
arbitrary direct-sum vectors; if a spectrum is unbounded below its positive-time
heat exponential is not a bounded operator on the whole Hilbert space.

The added bNhat is a proof comparison, NOT a derived physical chemical potential.
The vacuum has zero auxiliary shifted energy, but nonnegativity alone does not
prove uniqueness of that ground state. Choosing b strictly above both bound
coefficients would impose positive energies on nonvacuum sectors; any resulting
vacuum selection would be introduced by that choice, not discovered in the
original model. No such physical adoption is made here.

## 6. What this does not turn into a field theory

The spaces here are constrained bosonic SECTOR DIRECT SUMS. They are not
automatically canonical free Fock spaces over one one-particle Hilbert space:
naive creation can produce a forbidden close triple. Any projected creation
maps, their commutators and domains require separate definitions and proofs.
Canonical commutation relations cannot be assumed just from the direct-sum
Hilbert notation.

These normalizable vectors are not thermodynamic states at infinite density.
Even an infinite expectation of particle number is not a thermodynamic limit.
No joint large-N/small-a estimate, equation of state, physical vacuum selection,
particle-production mechanism, relativistic completion, gravity or empirical
identification follows. The supplied R,s,kappa, hard constraint and cubic
regularization remain supplied assumptions. The statement is a continuum-limit
extension of a conditional nonrelativistic model, not a TOE or a derivation of
those physical ingredients.

## 7. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_finite_particle_convergence_2026-09-14.md`: fixed-sector projection,
  resolvent and compact-time dynamics, including their scope restrictions.
- `cubic_finite_particle_continuum_2026-09-14.md`: extensive continuum bound and
  specified sector family.
- `cubic_three_body_exclusion_2026-09-13.md`: all-N lattice extensive bound,
  common small-spacing threshold and original energy reference.
The tail arguments here are separate proofs, not an assertion that sectorwise
convergence is automatically uniform in N.

Baseline691tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-sector-direct-sum-convergence-0owvqsnl/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical evaluations or model edits are authorized.

## 8. Independent mathematical review

Reviewer aef84380c1812212b cleared the direct-sum theorem with no mathematical
blocker. They checked constrained Hilbert sums and maximal graph domains,
finite-head/tail convergence, strongly moving states, number-dependent reference
phases, and the common auxiliary number shift. Their explicit +1-resolvent
variational proof was incorporated to make the nonnegative-form Mosco corollary
complete without an invalid weak norm subtraction.

Clearance includes generalized strong resolvents and vectorwise compact-time
unitaries for the threshold-referenced sums, plus separately the auxiliary
nonnegative-form Mosco and heat result. It does not include unshifted full-sum
heat convergence, canonical commutation relations, thermodynamics, physical
vacuum selection or empirical identification. Direct reads only were used.
Original proof-draft-v1.md and its hash remain preserved; physical assumptions
and mathematical constants are unchanged. Independent auditor a1874fe5c64c2974b
confirmed all691baseline files unchanged, exactly one new note and692current
files matching final-hashes.json. Draft changes are status/review attribution
and the explicit +1-resolvent Mosco justification only; current note matched
pre-audit-note.md. This is saved-document/preservation clearance, not mathematical
recertification or proof of absence of unrecorded execution. The audited
final-hashes.json is retained; closing-hashes.json binds final audit-status edits.
