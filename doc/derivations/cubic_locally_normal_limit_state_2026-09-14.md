# A density-retaining locally normal ambient quantum limit state

2026-09-14. Status: locally normal ambient-state construction independently
cleared; independent saved-document/hash audit completed.
This constructs a state on a specified AMBIENT quasi-local algebra. Quantum
equilibrium, a physical constrained observable net and phase selection remain open.

## 1. Fixed sequence and precise algebra

Retain supplied R,kappa>0, finite s!=0 and0<rho<1/(8R³). Let Lambda_j=(0,L_j)^3,
L_j->infinity, N_j/L_j³->rho. Choose normalized bosonic finite-box form vectors
psi_j with q[psi_j]=E(N_j,L_j)+o(L_j³), as in the preceding energy-density and
compactness notes. Embed them by zero extension into ambient bosonic Fock space.
The translation average is unchanged.

For bounded measurable U, modulo Lebesgue-null sets, let

    F_U=F_s(L²(U)), A(U)=B(F_U).

For U subset V, canonical ambient Fock factorization F_V=F_U tensor F_(V minus U)
gives the compatible unital isometric inclusion A->A tensor I. Define A_ql
as the operator-norm closure of the union of bounded-region algebras under these
inclusions, concretely represented in the full ambient Fock space. This does
not declare arbitrary members to be physical constrained operations.

Let t_a f(x)=f(x-a), T_a=Gamma(t_a), and alpha_a(A)=T_a A T_a*. These maps
send A(U) to A(U+a), respect inclusions and extend isometrically to automorphisms
of A_ql. No point-norm continuity of translations on this algebra is assumed.

With D_j=|psi_j><psi_j| define the trace-class spatial average

    Dbar_j=(1/L_j³) integral_(Lambda_j) T_(-y)D_jT_(-y)* dy.           (1)

It exists as a Bochner integral: strong continuity of translations of each
Fock vector gives trace-norm continuity of its rank-one density operator.
It is positive of trace1. Its local reductions are the previously analyzed
D_(j,U), and

    omegabar_j(A)=(1/L_j³) integral_(Lambda_j)
                              <psi_j,alpha_y(A)psi_j>dy.

Neither a periodic box nor a new canonical ensemble has been introduced.

## 2. Compatible trace-norm limits define a state

Choose the nested cofinal cubes C_m=(-mR,mR)^3. The reviewed local quantum
compactness theorem supplies trace-norm precompactness of D_(j,C_m) for each m.
A diagonal subsequence therefore gives D_(j,C_m)->D_(C_m) in trace norm for
all m, with positive trace-one limits. Partial trace is trace-norm contractive,
so these limits are compatible on nested cubes.

For any bounded measurable U inside C_m, define D_U by partial trace of D_(C_m).
Compatibility makes it independent of the containing cube. Moreover
D_(j,U)->D_U in trace norm along the SAME subsequence. A new extraction for
every U is unnecessary: all bounded U lie inside some exhaustion cube.

For A in A(U), define omega_rho(A)=Tr(D_U A). The compatible reductions make
this well-defined on the union. Positivity, normalization and the bound
|omega_rho(A)|<=||A|| extend it uniquely to a state on A_ql. It is locally normal
by construction, with the stated density-operator representation in every U.
No energy-minimization property has been passed to this state in this step.

## 3. Translation and gauge invariance

For each fixed a in R³ and bounded local A, shifting the averaging variable gives

    |omegabar_j(alpha_a(A))-omegabar_j(A)|
       <=||A|| |(Lambda_j+a) symmetric_difference Lambda_j|/L_j³ ->0.

Both local operators lie in one sufficiently large exhaustion cube. Trace-norm
convergence on that cube controls expectations of ALL its bounded operators,
so omega_rho(alpha_a(A))=omega_rho(A). Since a and A were arbitrary, this is
invariance under every translation along the same chosen subsequence, not an
uncountable diagonal argument. Norm density and the isometry of alpha_a extend
invariance to all A_ql. Point-norm continuity is not needed.

Each finite psi_j has definite total particle number, so its density operator,
all translates, and their average are invariant under global gauge rotations.
Partial traces commute with the factorized gauge action. Thus every local
D_(j,U) commutes with exp(i theta N_U), and trace-norm passage gives the same
for D_U. The state is gauge invariant; restriction to the gauge-fixed ambient
subalgebra is available if desired. This is not a derivation of a physical
superselection rule or canonical constrained fields.

## 4. Hard support and uniform local number cutoff

Define Q_U on ambient local Fock space as the direct sum of multiplication by
the indicator that every triple of particle labels inside U has diameter>R.
For n<3 this condition is vacuous. Every translated finite-volume state obeys
the global hard constraint, so its local reduction satisfies

    D_(j,U)=Q_U D_(j,U)Q_U.

This can be checked in each sector by integrating the global position kernel:
a forbidden triple wholly in U is forbidden regardless of all exterior coordinates.
Convex averaging preserves the identity. Trace-norm convergence yields

    D_U=Q_U D_U Q_U, Tr[D_U(I-Q_U)]=0.                  (2)

Cover bounded U by m_U sets of diameter<R. Each has at most two occurrences,
so N_U<=K_U=2m_U on allowed configurations. Let P_U=1_[0,K_U](N_U). Then
Q_U<=P_U and both D_(j,U),D_U are supported in P_U. Hence the otherwise unbounded
N_U has bounded representative P_U N_U P_U on these states, with norm<=K_U.
Its expectations therefore pass through the trace-norm limit. This argument,
not trace-norm convergence alone, justifies retention of local number moments.

For bounded f supported in U, n(f)=dGamma(M_f) similarly has representative
P_U n(f)P_U of norm<=K_U||f||_infinity on the common support. The state notation
omega_rho(n(f)) below means its locally normal expectation, not membership of
this unbounded observable in the norm algebra.

## 5. Prescribed mean density and improved positional support

For f in C_c(R³), the earlier boundary-packing/Fubini argument gives

    omegabar_j(n(f))=(N_j/L_j³)integral f+o(1).

Bounded support representatives from section4 and trace-norm convergence imply

    omega_rho(n(f))=rho integral f.                    (3)

The locally normal reductions and local number caps define a locally finite
countably additive intensity measure. Its values on C_c identify it as rho dx.
Thus(3) extends to bounded measurable compactly supported f and
Tr(D_U N_U)=rho|U| for every bounded measurable U.

Local normality also improves the earlier purely positional compactness result.
A positive trace-class operator on local Fock space has a spectral decomposition
into square-integrable sector wavefunctions. Its positional distribution in each
sector is therefore absolutely continuous with respect to Lebesgue measure.
There are only finitely many occupied number sectors here by the local cap.
Coincidence surfaces x_i=x_j are null, and triple-diameter=R sets are contained
in finitely many pair-distance-R null hypersurfaces. Neither carries local
positional mass. Together with(2), local configurations have every triple
diameter strictly greater than R almost surely and no coincident occurrences.
A countable exhaustion covers all finite collections of occurrences.

These statements are compatible with contact singularities: L² contact
wavefunctions still assign zero probability to exact coincidences. The conclusion
does not remove singular charges or prove that their operator-domain conditions
are retained in D_U. It does rule out the double-atom and equality-boundary
concentration that the earlier vague positional topology alone could allow.

## 6. Why this state is not yet quantum equilibrium

The construction gives a translation- and gauge-invariant, locally normal state
on the specified ambient quasi-local algebra with mean density rho and local
hard support. It does NOT construct a physical constrained observable net.
An arbitrary A in B(F_U) may create a forbidden configuration. A fixed corner
Q_U B(F_U)Q_U is an algebra, but compression itself is not generally multiplicative,
and naive corner inclusions must account for hard constraints crossing the
boundary of U. No compatible physical net, canonical constrained CCR or
infinite-volume physical dynamics is supplied by(2).

Local trace-norm convergence preserves bounded local expectations. It does not
by itself preserve unbounded contact energy, singular-domain boundary conditions,
or an energy-minimizing property. The auxiliary fractional compactness operator
is not a local physical Hamiltonian. The limiting value e(rho) of finite-box
energy densities does not yet identify energy density of this state through
a local energy observable or variation principle.

A remaining equilibrium theorem needs a justified local energy/variation
framework or an infinite-volume physical dynamics, followed by proof that the
state satisfies a ground-state or appropriate canonical equilibrium condition.
Choosing shrinking absolute spectral windows in finite volume can control
finite-volume stationarity, but passing that control to a bulk algebra remains
a separate argument. No response kernel or sound mode follows yet.

Local normality is not global normality in a free-Fock representation. Translation
invariance from averaging is not extremality, phase purity or correlation decay.
Mean density does not assign that density to every component or exclude a vacuum
component. No uniform-fluid or molecular/droplet phase preference is established.
The existing R,s,kappa, hard constraint, ensemble and averaging prescription are
unchanged; no new physical assumption has been used to choose a phase.

## 7. Sources and preservation

Protected references, relative to doc/derivations/:
- `cubic_local_quantum_compactness_2026-09-14.md`: trace-norm precompact local
  reductions and compatible subsequences, with energy/state limits separated.
- `cubic_density_retaining_positional_limit_2026-09-14.md`: translation average,
  boundary packing, intensity identity and limitations of vague positional support.
- `cubic_canonical_energy_density_2026-09-14.md`: finite energy-infimum limit,
  not an existing equilibrium state theorem.
- `cubic_fixed_density_phase_gate_2026-09-14.md`: finite boxes and phase distinctions.

Baseline697tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-locally-normal-limit-state-036vr_d9/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical calculations, model changes or publication are authorized.

## 8. Independent mathematical review

Reviewer a87d01c56973ddbd6 cleared the ambient-state construction with no blocker
or required repair. They checked compatible Fock-factor algebra inclusions,
trace-class translation averaging and its sign, the cofinal cube diagonal,
all-region trace-norm consistency, translations, gauge invariance, hard local
support, bounded number representatives and exact intensity. They also confirmed
that local Born absolute continuity excludes the positional boundary concentration
allowed by the weaker earlier topology without eliminating contact charges.

Clearance constructs a genuine locally normal AMBIENT state only. It does not
establish equilibrium, minimization retention, a physical constrained net/CCR,
infinite-volume dynamics or extremality. Direct reads only were used. Original
proof-draft-v1.md and its hash are preserved; no mathematical constant or physical
assumption changed. Independent auditor abd6d5029d10dfef1 confirmed all697baseline
files unchanged, exactly one added note and698current files matching
final-hashes.json. Mathematical sections1–6 are byte-identical to the draft;
the note matched pre-audit-note.md. This is artifact clearance, not mathematical
recertification or proof of absence of unrecorded execution. The audited
final-hashes.json is retained; closing-hashes.json binds final audit-status edits.
