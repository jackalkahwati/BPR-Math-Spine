# Density-retaining spatially averaged positional limits

2026-09-14. Status: positional lemma independently reviewed with no mathematical
blocker; independent saved-document/hash audit completed.
This concerns a COMMUTATIVE positional probability law, not a locally normal
quantum equilibrium state or an identified homogeneous phase.

## 1. Finite-volume input and precise limiting object

Keep supplied R,kappa>0, finite signed s!=0 and0<rho<1/(8R³), with the prescribed
Dirichlet continuum boxes Q_j=(0,L_j)^3, L_j->infinity, N_j/L_j³->rho.
The canonical energy-density theorem guarantees eventually admissible sectors
and finite energy infima. Choose normalized bosonic form vectors psi_j with
q[psi_j]-E(N_j,L_j)=o(L_j³), for example using the established intrinsic-core
approximate minimizers. No ground eigenvector or attainment is assumed.

Push their Born probabilities |psi_j(X)|²dX forward to counting measures
eta_X=sum_i delta_(x_i), empty outside Q_j. Let P_j be these laws. Distinct
particle labels may share a position in the counting-measure description;
multiplicities must not be discarded. Each finite-volume law is supported on
configurations whose every triple has diameter>R.

Define translation by (tau_y eta)(A)=eta(A+y), and average laws over the original
box:

    Pbar_j=(1/L_j³) integral_(Q_j)(tau_y)_*P_j dy.        (1)

This is a spatial average of positional probabilities, not periodic boundary
conditions or a newly selected quantum ensemble. Each translated configuration
still contains exactly N_j particles, although its support moves relative to
the observation origin.

We prove subsequential weak convergence in the vague configuration topology
to a translation-invariant probability law P_rho with intensity rho dx. The
closed limiting exclusion is NO triple of diameter STRICTLY LESS than R.
Equality at R is not excluded by this compactness argument. Nor are double
atoms excluded. These distinctions are essential, not changes silently made
to the finite-volume Hamiltonian.

## 2. Configuration-space compactness from packing

Let X_R be the locally finite integer-valued Radon measures on R³, counting
multiplicities, with no three particle occurrences of diameter<R. For a compact
K cover it by finitely many sets of diameter<R. Every such set contains at
most two occurrences, so

    eta(K)<=M_R(K)<infinity, eta in X_R.                 (2)

These deterministic compact-local mass bounds imply vague relative compactness
of positive Radon measures. Limits remain integer-valued: on bounded regions,
uniformly bounded numbers of atoms admit convergent finite point lists after
subsequence extraction, allowing merging and loss through region boundaries.
A diagonal exhaustion identifies an integer-valued locally finite limiting
measure. Equivalently, relatively compact continuity sets have integer masses
and determine that same counting-measure limit.

The exclusion condition is closed. If a limit had three occurrences of diameter
<R, small neighborhoods of their finitely many locations would still have
combined diameter<R and force at least three occurrences in the approximating
measures by vague lower semicontinuity. For coincident occurrences use their
multiplicity in the neighborhood count. This would contradict the approximants'
condition. Thus X_R is compact in the vague topology. It is metrizable using
a countable compactly supported test-function family on the locally compact
second-countable space R³. All Pbar_j are supported in X_R, so their probability
laws admit a weakly convergent subsequence on this compact metric space.

Bounded continuous cylinder observables may be written
F(eta)=g(eta(f1),...,eta(fm)), fi in C_c(R³), g continuous bounded. The maps
eta->eta(f) are continuous and uniformly bounded on X_R by(2). This is a
specified positional observable topology, not a complete quantum algebra.

## 3. Translation invariance of subsequential limits

For fixed a in R³ and bounded continuous F,

    |Pbar_j(F composed with tau_a)-Pbar_j(F)|
       <=||F||_infinity |(Q_j+a) symmetric_difference Q_j|/L_j³.

The boundary-volume fraction tends to0. Translation is continuous in the vague
topology, since it translates each compact test function continuously on the
bounded region relevant to that test. Passing through weak convergence gives
P_rho(F composed with tau_a)=P_rho(F). Thus every subsequential law constructed
this way is translation invariant. No ergodicity, extremality or single-phase
homogeneity is inferred from this averaging operation.

## 4. Exact density retention without a moment-tail assumption

Let f in C_c(R³), with support inside[-M,M]^3. Fubini gives

    Pbar_j[eta(f)]=(1/L_j³) E_(P_j) sum_i integral_(Q_j) f(x_i-y)dy.

For particles farther than M from all box faces, the inner integral equals
integral f. Only the fixed-width boundary layer contributes an error. Partition
space with a fixed grid of side<R/sqrt(3). At most O_(R,M)((L_j+1)²) cells meet
the layer, and each has at most two particle occurrences. Therefore its count
is deterministically bounded by C_(R,M)(L_j+1)², and

    |Pbar_j[eta(f)]-(N_j/L_j³)integral f|
       <=C_(R,M)(L_j+1)²||f||_1/L_j³ ->0.              (3)

The bound works for signed or complex f by absolute values. Since eta(f) is
bounded continuous on X_R, its expectation passes to the weak limit. Hence

    E_(P_rho) eta(f)=rho integral f, f in C_c(R³).       (4)

This identifies the locally finite intensity measure as rho dx. In particular
any fixed bounded window with Lebesgue-null boundary has zero boundary particle
count almost surely, by the corresponding intensity identity. This does NOT
say that the random configuration has no interparticle distances equal R;
that is a different boundary in configuration space.

## 5. Exact limitations of this positional lemma

The selected psi_j may be approximate energy minimizers, but the proof above
never uses their energies. Consequently it establishes positional compactness
and mean density retention, not retention of the quantum minimizing property.
The limit need not be unique. It may be a mixture of phases, and its intensity
rho does not imply that every ergodic component has almost-sure density rho.
Uniform one-point density is obtained by translation averaging, not derived
as the property of a pure homogeneous ground phase.

Finite-volume Born laws can concentrate onto configuration boundaries in a weak
limit. Triple diameters may equal R, and two simple atoms may merge into a
double atom. The original strict allowed condition is not automatically closed
under this topology. No substitution of this weaker positional support condition
for the hard quantum form domain is made.

The positional marginal does not determine off-diagonal coherence, singular
contact charges, energy, quantum stationarity, or linear response. Calling
P_rho a quantum thermodynamic ground state would therefore exceed the lemma.
No acoustic or relativistic conclusion follows.

## 6. First additional requirement for a quantum-state route

The packing estimate controls particle number in each bounded region, but not
occupation of arbitrarily fine SPATIAL MODES within its infinite-dimensional
local Hilbert space. Thus it is not trace-norm compactness of quantum density
matrices.

One diagnostic can be defined without asserting canonical constrained physical
fields. Embed finite-volume vectors by zero extension into ordinary ambient
bosonic Fock space; this is a mathematical representation, not a claim that the
constrained sector sum is itself a canonical free field theory. Spatially average
the translated ambient density operators at finite j and take their reduced
density operators D_(j,U) in a fixed bounded physical region U using the ambient
Fock factorization. These are finite-j positive trace-one operators. Packing
bounds local particle number, but their finitely many number sectors still
contain infinitely many spatial modes.

A sufficient compactness condition is a family of finite-rank projections
Pi_(U,m) increasing strongly to identity on that ambient local Fock space, with

    lim_(m->infinity) sup_j Tr[(I-Pi_(U,m))D_(j,U)]=0.    (5)

Together with positivity and unit trace, this implies trace-norm relative
compactness: finite-rank compressions are compact, and the uniform tail controls
the trace norm of the discarded diagonal and off-diagonal blocks. Initial
finite sets of j can be absorbed into the approximation if only an eventual
uniform estimate is established. No assertion that(5) holds is made here.

The reviewed global contact-form bound is not a spatially local positive
kinetic/charge estimate for a thermodynamic sequence. Ordinary H¹ compactness
cannot be applied to singular states to establish(5). A local finite-mode
estimate accommodating contact singularities and energy cancellation is the
missing mathematical step for this local-normality route.

Separately, a physical observable algebra on the constrained spaces, its
inclusions/localization and its infinite-volume dynamics must be constructed.
Compressing ambient local operators does not automatically supply canonical
commutation relations or a closed dynamics. Local quantum normality alone
would not prove the state is an equilibrium ground state of such dynamics.

Energy errors o(L_j³) do not by themselves establish stationarity. Nonzero
spectral projections in arbitrarily small absolute windows above a finite-box
spectral infimum can be used without eigenvalue attainment, but passing their
finite-volume evolution to a bulk equilibrium claim still needs a limiting
observable algebra/dynamics and domain control. None is assumed or proved here.

The next question is therefore local quantum finite-mode compactness, not a
sound calculation on the positional law. The unchanged R,s,kappa and density
remain the supplied inputs. No new physical law or ensemble is adopted to
conceal the missing estimate.

## 7. Sources and preservation

Protected sources, relative to doc/derivations/:
- `cubic_canonical_energy_density_2026-09-14.md`: energy-infimum limit and its
  explicit exclusion of a thermodynamic state construction.
- `cubic_fixed_density_phase_gate_2026-09-14.md`: admissible finite boxes,
  packing bound, approximate minimizers and state/response distinctions.
- `cubic_finite_particle_continuum_2026-09-14.md`: actual singular form setting
  and the limits of its extensive lower bound.

Baseline695tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-density-retaining-positional-limit-t4jdn1ej/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored files
and .git are excluded. Only this note is added. No scientific imports, tests,
numerical calculations, model changes or publication are authorized. Independent
mathematical review and independent preservation audit have cleared their stated scopes.

## 8. Independent mathematical review

Reviewer a1f8f8748c2de8497 cleared the positional lemma with no blocker or required
repair. They checked compactness of the counting-measure space including
multiplicities, the relaxed closed exclusion, probability subsequences, translation
invariance and exact intensity retention from the boundary packing estimate.
They confirmed no energy estimate enters that proof and that positive intensity
need not exclude a vacuum component or impose the same density on every ergodic
component.

The ambient local Fock diagnostic and sufficient finite-rank tail criterion were
also checked as representation/compactness statements, not as established local
quantum normality. No minimizing-property retention, quantum equilibrium, phase
selection or response is cleared. Direct reads only were used. Original
proof-draft-v1.md and its hash remain preserved; physical hypotheses and
mathematical constants are unchanged. Independent auditor a45576c562ef2be9b
confirmed all695baseline files unchanged, exactly one added note and696current
files matching final-hashes.json. Mathematical sections1–6 are byte-identical
to the draft; current note matched pre-audit-note.md. This is artifact/scope
clearance, not mathematical recertification or proof of absence of unrecorded
execution. The audited final-hashes.json is retained; closing-hashes.json binds
final audit-status edits.
