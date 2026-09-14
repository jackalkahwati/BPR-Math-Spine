# Two-particle contact interaction continuum gate

2026-09-13. Status: conditional mathematics independently reviewed with no blockers;
executable contract frozen below; 20 tests and both CLI modes passed;
final independent saved-evidence audit cleared with no blockers.
This is a conditional mathematical analysis of the supplied cubic Bose model,
not a new Hamiltonian, empirical validation or a theory of everything.

## 1. Model, spaces and phase

Retain the assumptions of `supplied_cubic_bose_2026-09-13.md`: the simple cubic
torus G_n=C_n square C_n square C_n, n>=3, M=n^3 sites, unrestricted Bose sites,
external time, hbar=1, C_n>0 and g_n>=0. Geometry and couplings are supplied.
The complete N=2 sector has dimension M(M+1)/2. Its normalized occupation ket
with both particles at x maps to delta_x tensor delta_x. For x!=y the ket maps
to (delta_x tensor delta_y+delta_y tensor delta_x)/sqrt(2). These kets form an
orthonormal basis of Sym² ell²(G_n); no local truncation has been imposed.

Let A be graph adjacency and K_n=C_n(6I-A). On the symmetric space let

    D psi(x,y) = 1_{x=y} psi(x,y),
    L_n = K_n tensor I + I tensor K_n,
    L_{n,g} = L_n + g_n D = H_{n,2} + 12 C_n I.          (1)

D is an orthogonal projection. The interaction g_n sum_x n_x(n_x-1)/2 is exactly
g_n D: it equals g_n on doublons and zero on separated pairs. Normalized Bose
hopping is the symmetric restriction of -C_n(A tensor I+I tensor A), establishing
(1) including normalization. These are finite Hermitian matrices, hence their
unitary dynamics are defined without unbounded-domain manipulations.

The propagator compared below is exp(-itL_{n,g})=exp(-i12C_n t)exp(-itH_{n,2}).
This declared N=2 phase is not addition of 6C_n Nhat to the full Hamiltonian.
No population preparation, vacuum selection or change to old model APIs follows.

## 2. Two-particle Fourier band and contact matrix

Supply fixed ell,kappa>0 and integer J>=0. Let a=ell/n, C_n=kappa/a² and
B_J={p in Z³: |p_j|<=J}; write K=(2J+1)^3 and require 2J<n. Normalized lattice
characters are chi_p(x)=M^(-1/2) exp(2pi i p dot x/n). Normalized continuum
characters on T_ell³ are f_p(r)=ell^(-3/2) exp(2pi i p dot r/ell).

The previous one-particle isometry I_n:f_p->chi_p lifts by tensor product and
restriction to an isometry V_n=I_n^(2) on E_J=Sym² span{f_p:p in B_J}. Its image
has dimension K(K+1)/2. Let P=V_n V_n^dagger on the complete symmetric lattice
space and Q=I-P. This P is the full two-particle band, not the uniform ray.
Free L_n preserves the image. Interacting L_{n,g} need not do so.

In ordered tensor Fourier coordinates, finite character orthogonality gives

    <chi_p tensor chi_q, g_n D (chi_r tensor chi_s)>
      = (g_n/M) 1_{p+q congruent r+s modulo n}.          (2)

Indeed the diagonal-site sum is g_n/M² times a character sum of M terms. An
orthonormal symmetric ket indexed by an unordered pair p,q is
(chi_p tensor chi_q+chi_q tensor chi_p)/sqrt(2(1+delta_pq)). Thus its contact
matrix element against the corresponding r,s ket is (2) times

    2 / sqrt((1+delta_pq)(1+delta_rs)).                  (3)

Each coordinate of p+q-r-s lies in [-4J,4J]. Therefore n>4J suffices to replace
all congruences in (2) by integer equalities. The weaker condition 2J<n suffices
for the isometry, but not this replacement. At n=3,J=1, totals 1+1 and -1+0
are distinct but congruent in a coordinate. Other coordinates may be zero.
This does not violate the one-particle isometry.

For supplied lambda>=0 define, only on smooth finite Fourier polynomials, the
sesquilinear expression

    v_lambda(F,G) = lambda integral_{T_ell³} conjugate(F(r,r)) G(r,r) dr.

Its ordered matrix elements are lambda/ell³ times the indicator of exact total
momentum equality, with the same symmetric factors (3). Hence

    g_n a³=lambda                                               (4)

matches these finite-band matrix elements when n>4J. This does not define a
self-adjoint three-dimensional delta Hamiltonian or prove a closed interaction
form on a continuum energy domain. Evaluation on the diagonal is not a bounded
multiplication operator on all L²(T_ell^6). Equation (4) is bare finite-band
matching, not scattering-length matching, renormalization or a physical fit.

## 3. Sharp band contact norm

For any normalized two-particle band state psi and each fixed x, evaluation in
the second coordinate gives

    |psi(x,x)|² <= (K/M) sum_y |psi(x,y)|².

To prove it, expand the second coordinate in K orthonormal characters. At x their
squared evaluation amplitudes sum to K/M; Cauchy-Schwarz gives the inequality.
Sum in x to obtain ||D psi||²<=K/M. This also holds in the symmetric subspace.
The normalized symmetric vector

    psi_* = K^(-1/2) sum_{p in B_J} chi_p tensor chi_{-p}

attains equality: its K ordered terms are orthogonal, inverse momenta remain in
the band, and its diagonal value is sqrt(K)/M at every site. Thus

    ||DP||² = K/M,       ||g_n DP||² = g_n² K/M.         (5)

An independent momentum-fiber proof groups coefficients c_pq by p+q modulo n:

    ||D psi||² = (1/M) sum_s |sum_{p+q=s mod n} c_pq|².

Every fiber has at most K ordered pairs, since each p determines at most one
admitted q when 2J<n. The zero-total fiber has exactly K. Cauchy-Schwarz on each
fiber proves the upper bound and equal coefficients in that fiber attain it.
At n=3,J=1, K=M and the norm is1, as an entire-space projection must have.

## 4. Positive fixed-initial-band free continuum theorem

For each finite n Duhamel gives, with U_g(t)=exp(-itL_{n,g}) and U_0(t)=exp(-itL_n),

    (U_g(t)-U_0(t))P
      = -i integral_0^t U_g(t-s) g_n D U_0(s)P ds.

Only U_0 preserves the band. Using (5), unitarity and the independent endpoint
bound2, for |t|<=T,

    ||(U_g(t)-U_0(t))P|| <= min(2,T g_n sqrt(K/M)).       (6)

Let h=-kappa Delta on the continuum torus and h_0^(2)=h tensor I+I tensor h on
the symmetric space. On E_J these operators restrict to finite diagonal ones.
Define q_p=2pi p/ell and b_n(p)=(kappa a²/12) sum_j q_p,j^4. The prior proof gives
0<=kappa|q_p|²-epsilon_n(p)<=b_n(p). For a pair the errors add, so the free
V_n-intertwining error is at most min(2,2T max_{p in B_J} b_n(p)). Combine this
with (6) by the triangle inequality and separately bound the difference of the
two endpoint isometries by2. Uniformly for unit F in E_J and |t|<=T,

    ||U_g(t)V_n F - V_n exp(-it h_0^(2))F||
      <= min(2,T[g_n sqrt(K/M)+2 max_p b_n(p)]).         (7)

No interacting-band invariance, convergence of unbounded generators or exchange
of unbounded time limits is used. Since M=(ell/a)^3, for bounded g_n the bound
is O(a^(3/2))+O(a²), at fixed ell,kappa,J,T. More generally g_n=o(sqrt(M)) makes
it vanish. This proves FREE two-particle continuum dynamics for these supplied
scalings and initial bands. The estimate is sufficient and its rate may be
nonsharp. It says nothing here about growing populations, arbitrary ultraviolet
preparations, all-time convergence, or unbounded-observable convergence.

## 5. Uniform-state leakage and the limitation of matching

Take u=chi_0 tensor chi_0, a unit symmetric vector in the band. Its position
amplitude is1/M. Since D retains precisely M equal-position entries,

    <u,g_n D u> = g_n/M,
    ||g_n D u||² = g_n²/M.                              (8)

In the complete ordered Fourier basis g_n D u has amplitude g_n/M for exactly
the M ordered inverse pairs (p,-p), and zero otherwise. Exactly K of these
pairs have both modes in B_J. Although ordered coordinates are redundant for
describing a symmetric vector, their tensor basis is orthonormal, so their
squared coefficients correctly give its norm. Projection into the symmetric
band agrees with the tensor-band projection on this symmetric vector. Hence

    ||P g_n D u||² = g_n² K/M²,
    ||Q g_n D u||² = g_n²(M-K)/M².                      (9)

In normalized unordered coordinates, each non-self-inverse pair has amplitude
sqrt(2)g_n/M and each self-inverse pair has amplitude g_n/M. Counting both
ordered partners with weight1 or the unordered pair with weight2 gives identical
answers. For the odd executable n, only the zero mode is self-inverse; retained
unordered pairs number (K+1)/2. The general ordered proof also covers even n
allowed by 2J<n. Leakage outside the uniform RAY would instead involve M-1,
not M-K, and is not the diagnostic in (9).

At n=3,J=1 the band fills the one-particle space: Q=0 and (9) is zero. This
finite saturation control is not evidence of continuum decoupling. Under (4),

    ||Q g_n D u||² = lambda²(M-K)/ell^6.                 (10)

For fixed positive lambda this grows without bound as n grows. Since free L_n
preserves the band, Q L_{n,g} u=Q g_n D u. For any self-adjoint target h_J on E_J,
Q(L_{n,g}V_n-V_n h_J) applied to the continuum uniform vector equals this same
leakage. Thus no choice of band-valued target cancels the perpendicular residual.
This obstructs a vanishing generator-residual route on this fixed embedding.
It is NOT a lower bound on finite-time propagator error: (6) and the prior
residual bound are sufficient, not converse implications. No exclusion of other
embeddings, energy-dependent estimates or renormalized limits follows. Matching
(2) alone has not established interacting continuum physics.

## 6. Frozen scalar controls and planned implementation

Only n=3,5,7, J=1, ell=1 and labels zero/fixed/formal-contact with g=0,1,M.
These nine cases are mathematical scalar/counting controls, not larger
Hamiltonian simulations. ell=1 differs deliberately from the prior ell=3 demo;
no physical calibration or change to that frozen demo occurs. Exact Fraction
arithmetic will report reduced numerator/positive-denominator pairs, never
approximate roots or pi. Norm-squared fields are not norms or evolution errors.

| n | M | g=M mean | total norm² | projected norm² | leakage norm² |
| --- | --- | --- | --- | --- | --- |
| 3 | 27 | 1 | 27 | 27 | 0 |
| 5 | 125 | 1 | 125 | 27 | 98 |
| 7 | 343 | 1 | 343 | 27 | 316 |

At g=1 leakage norm² is0,98/15625,316/117649 respectively; g=0 gives zeros.
One-coordinate total-momentum equality collisions among {-1,0,1}^4 number19;
modular collisions number27,19,19 for n=3,5,7. Factorization cubes these counts
for the three-dimensional ordered contact selection pattern. Maximum single
coordinate pair-total fiber size is3, so the cubic maximum is27, consistent
with (5). These are analytical integer targets pending independent review.

### Executable contract frozen before implementation

Reviewer a3cc41694d8a19924 independently cleared sections1–6 and their exact
targets, including even-n and J=0 analytic edge cases, without execution. The
reviewed submission is preserved externally as proof-draft-v1.md. This is scoped
agent mathematical review, not empirical validation or software clearance.

Public functions: `demonstration_report()` and `main(argv=None)`. Script is
stdlib-only, importable by file path without bpr initialization. CLI accepts
only default text or `--json`, returns0 on success, argparse rejects extra
controls before report generation. JSON uses allow_nan=False. Default text
starts `Cubic contact continuum: exact finite identities`, prints all nine
cases with rational mean/total/projected/leakage values and every limitation.
No file writes, cache, scientific imports or numerical propagation.

Private seams (frozen for independent tests):
- `_admit_case(n,g)`: only built-in int n in {3,5,7} and g in {0,1,n**3};
  reject bool, subclasses, floats, coercions or off-grid values with ValueError.
  Check live `MAX_MOMENTA=343` against n³ and `MAX_ALIAS_QUADRUPLES=81`
  against81 before any enumeration. Return None on acceptance.
- `_counts(n)`: admission at g=0 before enumeration; return exactly
  `inverse_pairs`, `retained_pairs`, `excluded_pairs`, `self_inverse_pairs`,
  `retained_unordered_pairs`, `excluded_unordered_pairs`,
  `integer_collisions_1d`, `modular_collisions_1d`, `modular_collisions_3d`,
  `max_fiber_3d`. All are ordinary integers. Count at most343 modular inverse
  pairs and81 one-coordinate quadruples; never full-space M² pairs.
- `_ordered_contact_element(n,g,p,q,r,s)`: admission first, then require each
  momentum an exact built-in tuple of three built-in ints in {-1,0,1};
  ValueError otherwise. Return Fraction from (2). No enumeration needed.
- `_rational(value)`: accept only built-in int or exact Fraction; otherwise
  ValueError. Return exactly {`numerator`:int,`denominator`:int}, reduced with
  positive denominator, including zero and integral rationals.
- `_case_report(n,g,label)`: enforce n/g admission and the literal coupling-label
  mapping zero->0, fixed->1, formal-contact->n³; otherwise ValueError. Return
  a fresh case dictionary as below.

Exact report top-level fields: `schema_version` (1), `status`
(`conditional_contact_scaling_diagnostics`), `empirical_validation` (false),
`controls` ({`n`:[3,5,7],`J`:1,`ell`:1,`coupling_labels`:[`zero`,`fixed`,
`formal-contact`]}), `cases`, `limitations`. The nine cases are n-major,
label-minor in the specified order. Each has exactly:

- integer `n`, `M`, `J`, `K`, `g` and string `coupling_label`;
- booleans `band_saturated`, `no_alias_condition_met`;
- dictionary `counts` with exactly the ten integer fields from `_counts`;
- canonical rational `expectation`, `contact_norm_squared`,
  `projected_norm_squared`, `leakage_norm_squared`,
  `band_contact_norm_squared` (K/M, for DP, not gDP),
  `interaction_duhamel_coefficient_squared` (g²K/M).

All containers must be detached between calls, with only JSON-native output
and no floats. The exact limitations list, in order:

1. `Geometry, quantization and coupling scalings are supplied assumptions.`
2. `The proved continuum comparison is free, fixed-band and two-particle only.`
3. `Formal contact matching does not define an interacting continuum Hamiltonian.`
4. `Generator leakage is not a lower bound on finite-time evolution error.`
5. `Exact finite identities are not empirical validation or matter/gravity completion.`

No numerical tolerance is used: every executable comparison is exact. Tests
may use at most729 ordered position pairs at n=3,343 inverse-momentum pairs
and81 one-coordinate quadruples; no matrices. Independent test author derives
oracles from this contract before seeing implementation. No NumPy, old tests,
full two-particle matrix, delta solver or scale search.

## 7. Sources, preservation and status

- `doc/derivations/supplied_cubic_bose_2026-09-13.md:79-147`: protected model,
  phase, Fourier isometry and one-particle free-error proof.
- `tests/test_supplied_cubic_bose.py:62-79,274-294`: protected normalized pair
  convention and equal-site interaction oracle; reasoning reused, no imports.
- `doc/derivations/foundation_compatibility_2026-09-13.md:138-207`: finite
  leakage/mismatch identities, Duhamel and explicit nonconverse limitation.
- `scripts/check_foundation_compatibility.py`, `tests/test_foundation_compatibility.py`:
  isolated stdlib script/importlib testing patterns, not modified.

Fresh baseline: 659 tracked/nonignored files captured using git ls-files with
--cached --others --exclude-standard. Ignored caches/config/build products/.git
are excluded. Saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-contact-continuum-mdvpujr_/baseline.json`.
Only this note, `scripts/check_cubic_contact_continuum.py` and
`tests/test_cubic_contact_continuum.py` may be added. All old files remain
protected, including prior failures and incomplete regressions.

Independent mathematical review cleared the conditional proof and targets. The
script and 20 independent unittest methods have been authored. Test author
 a2b99904a54067b93 used the frozen contract without reading the implementation.
Static reviewer af317c78c0fa0d7f5 identified one CLI issue before execution:
argparse accepted abbreviated --json flags. The parser now disables abbreviation,
and the existing rejection test includes --j, --js and --jso with assertions
that report generation never starts. Original source/tests are preserved
externally. No arithmetic target, formula or control changed. The reviewer
cleared the narrow repair by direct reads; bounded execution completed below.

Procedural deviation: during static review a skill unexpectedly spawned another
review agent, a57a8b130cc1cd2bb, contrary to the no-subdelegation instruction.
The reviewer reported it stopped after one read of this note, with no imports,
execution, probes, edits or completed review. Its output was not used. This
exception is recorded rather than describing that review as delegation-free.

The planned single run uses exact new-test discovery (120s), then text and JSON
(30s each), sequentially with saved logs, byte snapshots and hashes. Parent and
children use the same explicit CommandLineTools interpreter; recorded sys.version
identifies the parent, not an independent child probe. Bytecode is disabled,
PYTHONPATH removed, warnings are errors and six thread variables are1. The runner
fails on observed timeout/interruption, residual child group or changed inventory;
uncatchable abrupt runner termination is not comprehensively handled, and missing
completion never earns success. No old tests or scientific probes are authorized.
No empirical validation, interacting continuum Hamiltonian, geometry selection,
physical matter/gravity completion or novelty is asserted.

## 8. Actual bounded verification

Evidence directory: `cubic-contact-continuum-mdvpujr_/first-bounded-run/` under
`/Volumes/T9 Backup/bpr-verification/`. First and only attempt:

| Child | Saved result | Whole child seconds |
| --- | --- | --- |
| Exact new unittest discovery | 20 methods passed, unittest time0.710s | 1.219915792 |
| Default text | exit0 | 0.193251125 |
| Strict JSON | exit0 | 0.191863542 |

All three saved outcomes report no timeout, observed interruption or residual
process group. All662file hashes matched before and after each child, including
all659protected prior files. Source/tests and all three pre-run byte snapshots
are bound by frozen-hashes.json. No automatic rerun or supplemental science.
Only this note's status and actual results are updated after evaluation.

The saved text gives fixed-coupling leakage0,98/15625,316/117649 and formal-contact
leakage0,98,316, in exact agreement with the frozen targets. These are squared
norm diagnostics, not measured propagation errors. All assertions used exact
arithmetic, not adjustable floating tolerance. Independent reviewer
 af317c78c0fa0d7f5 cleared the saved-evidence audit with no blockers: all20passes,
54canonical rational fields, bothCLIoutputs, prescribed commands/environment,
all659protected hashes and exactly3additions were confirmed. Current script and
tests remain frozen, and the full662file inventory matched final-hashes.json.
The reviewer used direct reads and stdlib artifact parsing/hashing only, without
scientific imports, reruns, skills or agents. This is saved-evidence clearance,
not independent observation of live execution. The audited final-hashes.json
is retained; closing-hashes.json binds the final audit-status edits to this note.

### Conclusion

For the supplied two-particle model, fixed nonnegative interaction strength
has a proved free continuum limit on each fixed initial Fourier band and finite
time interval. The stronger bare scaling that matches formal contact matrix
elements produces nonvanishing generator leakage from that band. This limits
the fixed-band residual argument; it does not prove that all interacting limits
are impossible. Geometry, quantization and scaling are inputs. An interacting
continuum construction, physical matter/gravity and empirical distinction remain
open. This gate does not authorize another model, old regression or publication.
