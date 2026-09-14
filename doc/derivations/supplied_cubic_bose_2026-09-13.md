# A supplied cubic Bose model (2026-09-13)

Status: conditional mathematics and implementation independently reviewed;
20 focused tests and both fixed demos passed. Independent saved-evidence audit
completed with no blockers.
This is a conventional Bose-Hubbard baseline on assumed geometry, not new physics
or a completion of BPR. No original ring files or claims are changed.

## 1. Supplied assumptions

For an integer n >= 3, choose G_n = C_n square C_n square C_n. Sites x are
triples in {0,...,n-1}^3, ordered lexicographically. Edges join coordinates differing
by plus or minus one modulo n in exactly one component. Edges are unordered,
simple and counted once: C_3 is a triangle, not a doubled-edge graph. The two
neighbors in each component are distinct, so degree is six, M=n^3 and |E|=3M.
Graph distance is the sum of the three periodic coordinate distances.

Supply the tensor product of M unrestricted site Bose Fock spaces, normalized
occupation kets, continuous external time, hbar=1, C>0 and g>=0. On the algebraic
finite-occupation core define

    H = (g/2) sum_x n_x(n_x-1)
        - C sum_{unordered {x,y} in E} (a_x^dagger a_y + a_y^dagger a_x).

There is no local occupation cutoff or link/gauge field. Dimension, connectivity,
quantization, couplings, time, diagnostic populations and the continuum scaling
below are inputs. No map from the old ring to this spatial graph is asserted.
Local support of the terms is not a proof of a relativistic light cone or a
state-independent finite propagation bound for unrestricted Bose fields.

## 2. Well-defined dynamics and stability

Let F_N be the span of occupations nu with sum_x nu_x=N. Stars and bars gives
D_N=binomial(M+N-1,N). The occupation basis is lexicographic. A hopping term from
source y to target x has matrix element -C sqrt(nu_y(nu_x+1)); the destination
is nu-e_y+e_x. Reverse matrix elements coincide. Interaction is real diagonal.
Thus the complete H_N is a finite Hermitian matrix and preserves N.

On F = direct_sum_{N>=0} F_N define

    D(H) = {psi=(psi_N): sum_N ||H_N psi_N||^2 < infinity},
    (H psi)_N = H_N psi_N.

This operator is self-adjoint: testing the adjoint against vectors supported in
a single sector forces its component action to be H_N; membership in F of that
action imposes exactly the displayed domain. Conversely that domain satisfies
the adjoint identity by Cauchy-Schwarz. Finite sector truncations converge in
both vector norm and H graph norm, so the algebraic finite-occupation vectors
are a core. In particular this defines the closure of the stated core operator,
not an unspecified extension. The direct sum of exp(-itH_N) is a strongly
continuous unitary group. Dominated convergence proves continuity, sectorwise
unitarity proves preservation of norm, and the generator is H on D(H).

Every number spectral projection commutes with this group. Thus H and total
number strongly commute, even where their unbounded products are not defined.
This conservation does not select or prepare a population. Translation and
signed coordinate-permutation graph automorphisms permute occupation kets and
commute with every sector Hamiltonian; these are discrete spatial symmetries.

For any site vector u, the edge inequality
2|Re(conjugate(u_x)u_y)| <= |u_x|^2+|u_y|^2 gives -6I <= A <= 6I.
On N bosons the hopping is the restriction of sum_{i=1}^N (-CA)_i, hence at
least -6CN. Also sum_x nu_x^2 >= N^2/M on each occupation. Consequently

    H_N >= (g/2)(N^2/M-N) - 6CN.                         (1)

Every fixed sector is bounded below. For g=0, N particles in the normalized
uniform spatial mode have energy -6CN, so H on full Fock space is unbounded
below. This does not invalidate self-adjoint dynamics. For g>0, completing the
square in (1), even allowing real N, gives the finite-volume lower bound

    H >= -M(g/2+6C)^2/(2g).                              (2)

These are not uniform interacting continuum or thermodynamic limit claims.
The empty occupation vacuum has energy zero whereas the uniform N=1 vector
has energy -6C for every g. The empty vacuum therefore is not a ground state
of the unshifted model. No selected physical vacuum is supplied by this result.

## 3. Exact one-particle dynamics and comparison frame

Identify the site basis vector delta_x with the occupation e_x. In N=1 every
interaction entry vanishes, so H_1=-CA. Define for comparison only

    K_n = H_1 + 6CI = C(6I-A),
    exp(-itK_n) = exp(-i6Ct) exp(-itH_1).                 (3)

In lexicographic occupation order, the vector e_x is not in site lexicographic
order: for M sites its row is M-1-index(x). More generally use the explicit
lookup e_x -> basis.index(e_x), not an implicit assumption about ordering.
The scalar in (3) is a declared one-sector phase convention. Replacing the full
H by H+6C Nhat would change intersector energies and is not adopted here.
N=1 is a diagnostic sector, not the physical vacuum.

## 4. Controlled nonrelativistic continuum bridge

Supply ell,kappa>0 and the torus T_ell^3 with Lebesgue inner product. Its
normalized Fourier modes are f_m(r)=ell^(-3/2) exp(i q_m dot r),
q_m=2pi m/ell. The standard self-adjoint generator h=-kappa Delta has Fourier
energies lambda_m=kappa|q_m|^2 (domain sum_m lambda_m^2 |c_m|^2 finite).
Fix integer J>=0, the band B_J={m: |m_j|<=J}, and require 2J<n. Set

    a=ell/n,    C_n=kappa/a^2.

This is a supplied sequence, not calibration or a fitted scale. Define I_n on
the finite continuum band by f_m -> chi_m, where
chi_m(x)=n^(-3/2) exp(2pi i m dot x/n) in the site inner product sum_x conjugate(u_x)v_x.
Equivalently I_n f(x)=a^(3/2) f(ax) on this band. Apply the occupation-row
permutation from section 3 if using an occupation matrix.

Finite geometric sums give <chi_m,chi_l>=delta_ml: since each difference is
strictly smaller than n, congruence modulo n means equality. Thus I_n is an
isometry. It is not an asserted isometry on all continuum modes. Translations
by neighboring sites show A chi_m=2 sum_j cos(2pi m_j/n) chi_m. Both the lattice
image band and continuum band are exactly invariant and

    epsilon_n(m)=4kappa/a^2 sum_j sin^2(pi m_j/n).         (4)

For real z, 0<=z^2-sin^2 z<=z^4/3. To see the upper bound, for z>=0 write
z-sin z=integral_0^z (1-cos u)du <= z^3/6, using 1-cos u<=u^2/2.
The factors z-sin z and z+sin z are nonnegative, the latter at most 2z;
use evenness for z<0. The lower bound follows from |sin z|<=|z|.
Apply this with z=a q_m,j/2 to obtain

    0 <= lambda_m-epsilon_n(m)
      <= (kappa a^2/12) sum_j q_m,j^4 =: b_n(m).         (5)

For real lambda,epsilon,
|exp(-it epsilon)-exp(-it lambda)| <= min(2,|t||lambda-epsilon|)
by integrating the phase derivative and by the triangle inequality. Orthogonality
of chi_m makes the norm of the difference of the two band maps the maximum
of these scalar phase differences. Hence for every unit f in the fixed band,
|t|<=T,

    ||exp(-itK_n) I_n f - I_n exp(-ith) f||
      <= min(2,T max_{m in B_J} b_n(m)).                 (6)

For fixed ell,kappa,J,T this is O(a^2) as n tends to infinity. Exact invariance
is zero leakage, not zero generator mismatch at finite a. This sufficient
bound is not a converse test of generator closeness. This is a finite-time,
fixed-band, one-particle nonrelativistic Schrödinger limit only. It does not
control arbitrary ultraviolet states, all times uniformly, or interacting
many-body convergence. Expansion at fixed q gives

    epsilon_n = kappa |q|^2 - kappa a^2 sum_j q_j^4/12 + O(a^4).

The leading symbol is isotropic; the lattice correction is anisotropic. Neither
continuous rotations at finite spacing nor Lorentz symmetry follows.

## 5. Fixed finite witness targets

The executable witness is only n=3,C=1,N in {0,1,2},g in {0,1}, using existing
NumPy and normalized dense sector matrices of dimensions 1,27,378. These are
complete sectors, not independently truncated site spaces. No eigensolver,
exponential, SVD, larger grid, search, cache or arbitrary time/state API.

At N=2 there are 27 doublons and binomial(27,2)=351 split states. Interaction
is g on each doublon and zero elsewhere. Neighboring doublon d=2e_x and split
s=e_x+e_y have H_sd=-sqrt(2). Let D=diag(sum_x nu_x(nu_x-1)/2) and T=H at g=0.
For x=(0,0,0), y=(1,0,0),

    [D,T]_sd = (D_ss-D_dd)T_sd = +sqrt(2).              (7)

At g=1 this is a fixed nonzero interaction/hopping commutator, not a finite-time
interacting propagation test. At g=0 the corresponding [gD,T] is zero.
Each n=3 coordinate mode has Laplacian energy 0 or 3, so the 27 shifted energies
are 0,3,6,9 with multiplicities 1,6,12,8.

The continuum illustration fixes ell=3,kappa=1,J=1,a=1 and times 0,0.1. It
checks all 27 mode inequalities (5),(6), not numerical asymptotic convergence.
Absolute comparison allowance 1e-11 is fixed before evaluation. It is heuristic
floating-point software tolerance, not a certified roundoff envelope. Preserve
raw residuals and signed bound excesses without clipping to zero. Nonfinite
computed outputs must raise ValueError identifying numerical failure.

### Frozen executable contract (before implementation/evaluation)

Independent reviewer af5a6f9333821d9ac found no mathematical blockers in the
submitted proof (saved externally as proof-draft-v1.md). They checked domain/core,
counting, stability, phase, isometry, sine inequality, evolution bound and finite
targets without scientific execution. This is agent review, not human approval
or empirical validation.

Public `cubic_sector(N,g)` returns `(basis,H)`, where basis is a fresh list of
immutable occupation tuples in ascending lexicographic order and H is a fresh
owned float64 NumPy matrix. Only built-in int N in {0,1,2} and g in {0,1} are
accepted; all other values raise ValueError before enumeration/allocation.
`MAX_DIMENSION=378` is checked live before enumeration and each dense allocation
block, including Fourier/report intermediates; a cap rejection raises ValueError.
Private seams: `_admit(dimension)`, `_occupations(sites,number)`, `_graph()` returning
`(sites,edges)` as tuples, `_fourier_diagnostics(basis,H)` for N=1, and
`_finite(value)` recursively validating a JSON-native result. Tests may patch these
seams for early rejection and explicit nonfinite failure; no public arbitrary
graph, state, time or Fourier-band interface is introduced.

`demonstration_report()` builds exactly six sector cases in order N=0,1,2 with
g=0,1 within each N. It performs Fourier diagnostics only on (N,g)=(1,0).
No sector is rebuilt within a report; large matrices are released after summaries.
Report top-level fields (exact set):

- `schema_version`: integer 1.
- `status`: `supplied_geometry_nonrelativistic_demonstrator`.
- `empirical_validation`: false; `derived_dimension`: false.
- `controls`: {`n`:3,`C`:1,`populations`:[0,1,2],`couplings`:[0,1],
  `ell`:3,`kappa`:1,`J`:1,`times`:[0,0.1],`absolute_allowance`:1e-11}.
- `graph`: {`sites`:27,`edges`:81,`degrees`:[6 repeated 27 times]}.
- `sectors`: six dictionaries with exact fields `N`, `g`, `dimension`,
  `hermiticity_residual` (max entry modulus of H-H.T), `interaction_trace`
  (trace H), `doublons` (count of basis occupations containing 2),
  `commutator_sd` (null except N=2; there `(H_ss-H_dd)*H_sd`, for the fixed
  sites in (7)). This equals [gD,T]_sd, with no matrix-square allocation.
- `fourier`: {`orthogonality_residual`, `eigenvector_residual`, `modes`}.
  Construct normalized characters in occupation-row order. Residuals are max
  entry modulus of F^dagger F-I and K F-F diag(epsilon), respectively.
  `modes` has 27 entries in lexicographic m in {-1,0,1}^3, with fields `m`
  (three integers), `lattice_energy` (formula (4)), `continuum_energy`,
  `generator_error` (lambda-epsilon, signed), `generator_bound` (b_n),
  `generator_lower_excess` (negative generator_error), `generator_upper_excess`
  (generator_error minus b_n), `phases`. Each phases list is ordered t=0,0.1
  with fields `time`, `error` (raw complex phase difference modulus), `bound`
  (min(2,abs(t)*b_n)), `signed_excess` (error minus bound).
- `limitations`: fresh list of exactly these strings: `Three-dimensional geometry
  is supplied, not derived.`, `The continuum theorem is nonrelativistic and
  one-particle only.`, `The empty vacuum is not the ground state of the unshifted
  model.`, `No physical fermions, matter, gravity, calibration or empirical
  distinction is established.`, `Finite floating checks are not certified
  roundoff bounds or empirical validation.`

All report numbers are built-in finite JSON numbers, lists/dictionaries are detached
between calls, and computed nonfinite entries raise `ValueError` containing
`numerical failure`. Structural errors propagate, not unavailable-success reports.
The report never converts numerical inequalities to physical validation flags.

The stdout-only CLI has `main(argv=None)`, default text and `--json` only, with
argparse rejecting extra scientific controls. Successful calls return 0. JSON
uses allow_nan=False. Text prints the title `Supplied cubic Bose demonstrator`,
counts, raw Fourier residuals, and every limitation. It imports the sibling
module without executing bpr/__init__.py. Tests mock report calls at CLI boundaries;
the actual text and JSON demos run once each separately after the test suite.

Independent tests must use pairwise coordinate-distance adjacency, unordered
particle-position enumeration and normalized first-quantized symmetric-pair
matrix elements as oracles, not production builders. Freeze 1e-11 absolute,
zero relative tolerance throughout. No scientific runs have occurred at this
contract freeze; failures stop evaluation without automatic retry.

## 6. Provenance and remaining obligations

Source patterns, not inherited proofs or changed APIs:

- `bpr/substrate_fermionization.py:125-187`: complete occupation enumeration and
  normalized ladder amplitudes; its ring constructor/caps are not reused.
- `bpr/substrate_vacuum_selection.py:165-180`: admission before allocation.
- `doc/derivations/substrate_longwave_limit_2026-09-12.md:39-45`: Taylor/phase
  comparison technique; the present hypotheses and proof are established above.
- `scripts/demo_substrate_locality_dimension.py:14-18`: isolated sibling import.
- `doc/derivations/foundation_compatibility_2026-09-13.md:138-207`: leakage,
  target-generator mismatch and comparison conventions are distinct obligations.
- `doc/derivations/foundation_decision_2026-09-13.md:33-42`: common-foundation
  requirements remain open. This separate demonstrator does not supersede them.

A supplied cubic graph avoids asking a bounded-range, bounded-fiber ring map to
reproduce growing cubic balls; it does not derive that graph. No geometry
selection, physical fermions, anomaly-complete matter, metric dynamics, physical
scale calibration or distinguishing prediction is constructed. This model is
conventional Bose-Hubbard physics and has no asserted empirical validation.

Protected inventory: 655 pre-existing tracked and nonignored untracked files,
from `git ls-files -z --cached --others --exclude-standard`. Ignored caches,
configuration, build products and .git are excluded from this preservation claim.
Baseline saved outside the repository in
`/Volumes/T9 Backup/bpr-verification/supplied-cubic-bose-kun60qxn/baseline.json`.
Previous incomplete regressions/failures are not resumed or reclassified.

## 7. Verification record (pre-execution)

The independently authored test file has 20 methods. Author a113a3b9db729029e
read the frozen contract and plan, not the implementation, and did not execute
it. Static reviewer ac6b4c6eaa4c08e8a checked source, tests, CLI and runner. They
found no mathematical/production implementation blocker, but identified recursive
installation of allocation mocks in a cap test. Before any evaluation, that
wrapper was restricted to installing guards at the outermost 27-site call.
Original tests are preserved externally; no target or tolerance was changed.
The same independent reviewer verified the repair statically and found no further
blocker in the reviewed scope. This does not assert test success.

Execution is restricted to one new exact-pattern unittest discovery (120s),
then one text and one JSON demo (30s each), sequential fresh processes/directories.
The runner and children use the same explicit CommandLineTools Python; saved
sys.version identifies the runner runtime, not a separate child runtime probe.
Six numerical thread settings are 1, warnings are errors, bytecode is disabled,
and PYTHONPATH is removed. All four pre-run files will be saved byte-for-byte
with hashes. No old tests, imports-as-probes or supplemental scientific runs.
The runner saves outcomes/logs/inventories, kills timed-out child process groups,
and fails on residual groups, interruptions observed during wait, or file changes.
Abrupt uncatchable termination of the runner is not comprehensively recorded or
cleaned up by this harness; a missing completion record never earns success.
### Actual bounded outcomes

External artifacts: `supplied-cubic-bose-kun60qxn/first-bounded-run/` under the
verification directory above. The first and only attempt completed:

| Process | Outcome | Whole child elapsed time |
| --- | --- | --- |
| Exact new unittest discovery | 20 methods passed; unittest time 1.044s | 2.202712208s |
| Default text demo | Exit 0 | 0.510745959s |
| Strict JSON demo | Exit 0 | 0.387185750s |

All three returned zero without timeout, observed interruption or residual process
group. Runner runtime was Python 3.9.6 using the same absolute interpreter path
for all children. All 659 bound file hashes matched before and after each child,
including all 655 protected pre-existing files. No rerun or supplemental science
was performed. Pre-run snapshots and frozen hashes preserve the evaluated inputs;
post-run changes are restricted to this note's status/results.

The text report has exactly six controls, dimensions 1/27/378, zero Hermiticity
residuals, g=1 two-particle interaction trace 27 and commutator
1.4142135623730951. Its raw Fourier orthogonality residual is
6.661341070768738e-16 and eigenvector residual 9.930136612989092e-16.
These are finite implementation diagnostics, not precision certification or
physical evidence. Independent reviewer ac6b4c6eaa4c08e8a audited saved logs,
strict finite JSON, launch/environment/outcome records, byte snapshots and hashes
without scientific imports or reruns. They confirmed all stated results, 655
unchanged protected files, exactly four additions, frozen module/CLI/tests, and
an exact match to final-hashes.json; no blockers were found. Their clearance
concerns saved evidence, not independently observed execution history or a new
live-process audit. The audit-status edits to this note are subsequently bound
by closing-hashes.json; the audited final-hashes.json is retained unchanged.

### Conclusion

The supplied cubic Bose model has well-defined conditional dynamics and an
analytically controlled fixed-band one-particle Schrödinger limit. Independent
finite software checks agree with the specified graph, complete sectors,
interaction and Fourier bounds. Three-dimensional geometry is still assumed;
no dimension-selection mechanism, relativistic matter/gravity completion or
empirical distinguishing prediction has been obtained. No next model search,
old regression restart or publication is authorized by completion of this task.
