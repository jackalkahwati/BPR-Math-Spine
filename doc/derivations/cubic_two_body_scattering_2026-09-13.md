# Cubic two-body scattering and continuum tuning

2026-09-13. Status: conditional mathematics independently cleared; exact witness
contract frozen below; 20 exact tests and both CLI modes passed;
final independent saved-evidence audit cleared with no blockers.
This is a conditional scattering analysis, not an adopted modification of BPR,
empirical validation, or a stable theory of matter and gravity.

## 1. Setting, normalization and order of limits

Retain the supplied cubic nearest-neighbor hopping with C_a=kappa/a²,
a,kappa>0, and onsite interaction g_a. Nonnegative g_a is the original route;
negative g_a is considered only as an explicitly separate two-body tuning.
The complete two-boson occupation space is identified with symmetric ordered
position wavefunctions: a doublon at x maps to delta_x tensor delta_x, and a
split pair to (delta_x tensor delta_y+delta_y tensor delta_x)/sqrt(2).
Consequently the interaction is g_a times the equal-position projector, not2g_a.
The comparison energy subtracts the band minimum: H_2+12C_a I. This is a declared
fixed-N=2 phase reference, not a change in full-Fock population energetics.

On a periodic M-site torus the zero-total-momentum embedding is
psi(x,y)=M^(-1/2)phi(x-y), phi(-r)=phi(r). Summing over x,y proves it is an
isometry from even relative functions; contact becomes g_a|0><0|. Moving either
particle changes r by a neighbor displacement, so the shifted free relative
operator is 2C_a(6I-A), where A is the simple cubic adjacency.

For infinite volume take ordered coordinates (r,y)=(x-y,y), a bijection of Z^6,
and Fourier transform y with measure dP/(2pi)^3. This is a unitary direct-integral
decomposition. Exchange acts on the fiber as phi_P(r)=exp(iP dot r)phi_P(-r)
for the convention sum_y exp(-iP dot y)psi(y+r,y). At P=0 this is ordinary evenness.
The explicitly defined P=0 fiber is meaningful although a single sharp P is a
measure-zero label, not a normalizable vector of the full infinite-volume space.
On its even ell²(Z³) space, delta_0 has norm1 and

    h0,a=2C_a(6I-A),    ha=h0,a+g_a|0><0|.               (1)

At fixed a and finite real g_a these operators are bounded self-adjoint. Evenness
commutes with both. Since delta_0 is even, its resolvent matrix element is the
same as in the unrestricted relative ell² space. No extra parity factor occurs.

Limits here differ from the earlier fixed-side-length torus result: first take
box size to infinity at fixed a and Im z>0, then take outgoing boundary Im z down
to0, then take a down to0 at fixed physical energy. For the local free resolvent,
finite-volume Fourier sums converge to the integral for Im z>0 by continuity
of the integrand. The rank-one formula below transfers this convergence to its
perturbation, whose denominator is nonzero there. A finite-box threshold sum
has a zero-mode singularity and is never substituted for the infinite-volume
threshold limit. No exchange of these three limits is assumed.

## 2. Rank-one resolvent and continuum normalization

Fourier measure is dp/(2pi)^3 on B=[-pi,pi]^3. Dimensionless relative momentum p
has energy kappa omega(p)/a², with

    omega(p)=4 sum_j(1-cos p_j),     0<=omega<=24.
    F(zeta)=integral_B dp/(2pi)^3 [zeta-omega(p)]^(-1),
    G_a(z)=<0|(z-h0,a)^(-1)|0>=(a²/kappa)F(a²z/kappa).

For R0=(z-h0,a)^(-1), R=(z-ha)^(-1), solve R=R0+R0 g_a|0><0|R to obtain

    tau_a(z)=g_a/[1-g_a G_a(z)],
    R=R0+R0|0>tau_a(z)<0|R0.                            (2)

Both R0 factors are evaluated at z; the second is not an adjoint. When g_a=0,
tau=0 directly. For Im z>0, Im G_a(z)=-(Im z)||R0(z)delta_0||²<0. Thus for real
nonzero g_a the denominator cannot vanish: G_a cannot equal real1/g_a. This is
also consistent with self-adjointness. Real poles require separate exclusions.

The physical momentum variable is k=p/a. Changing Fourier measure gives a³
d³k/(2pi)^3. Equivalently, on the continuum band-limited space with momentum support B/a,
the position-space sampling isometry carries amplitudes by a^(3/2). This is
not a sampling isometry on unrestricted continuum L². Thus the continuum-normalized scalar contact coefficient is

    T_a(z)=a³ tau_a(z).                                 (3)

An exact lattice energy shell uses p satisfying kappa omega(p)/a²=E. At fixed
E=2kappa k² and any fixed direction, sufficiently small a admits such momenta
with p/a tending to that direction times k. This follows from omega(p)=2|p|²+
O(|p|^4) and its positive radial derivative near0. Setting p=a k at finite a
would not lie exactly on that shell. Since tau is independent of incident and
outgoing momentum, this distinction does not alter (3), but matters to its
interpretation. Lattice outgoing waves are not exactly spherical at finite a.

## 3. Threshold constant and scattering length

Define, without numerical evaluation,

    W=integral_B dp/(2pi)^3 /omega(p).                   (4)

The only zero of omega on the torus is p=0. Near zero omega=2|p|²+O(|p|^4),
so1/omega is locally integrable in3D; away from zero it is bounded. Its positive
integrand on a set of positive measure gives0<W<infinity. Dominated convergence
from negative energies gives F(0-)=-W and

    G_a(0)=-a²W/kappa.                                  (5)

This notation denotes a scalar threshold limit, not a bounded inverse of h0,a.

To justify its length normalization, let r in Z³ and use the zero-energy Green
kernel

    G_a(0;r)=-(a²/kappa) integral_B exp(ip dot r)/omega(p) dp/(2pi)^3.

Its leading large-|r| term is -a²/(8pi kappa|r|). One direct justification is to
choose a smooth radial cutoff chi equal1 near0 and supported inside B. Subtract
chi(p)/(2|p|²) from1/omega(p). The remainder is bounded near0 with weak first
derivatives O(1/|p|), hence is W^(1,1) on the torus. Integration by parts and
the Riemann-Lebesgue lemma imply its Fourier coefficients are o(1/|r|): select
a coordinate with |r_j|>=|r|/sqrt(3), and use the finite set of derivative
transforms. The cutoff singular part has leading Fourier coefficient1/(8pi|r|).
Indeed radial integration gives (1/(4pi²|r|)) integral_0^infinity chi(t)
sin(|r|t)/t dt. Explicitly, subtract exp(-t): that last integral is
arctan(|r|)+integral_0^infinity [(chi(t)-exp(-t))/t]sin(|r|t)dt.
The bracket is L¹ (bounded near0 and integrable at infinity), so its transform
vanishes and the limit is pi/2. This proves the asserted
leading term without claiming isotropy of subleading lattice corrections.

For a nonresonant threshold coupling, the even zero-energy scattering solution
with incident constant1 solves psi=1+G_a(0;r)g_a psi(0), hence g_a psi(0)=tau_a(0).
In physical distance R=a|r| it obeys

    psi(R)=1-s_a/R+o(1/R),
    s_a=a³ tau_a(0)/(8pi kappa)=T_a(0)/(8pi kappa).       (6)

The solution is generalized, not square-summable; substituting the integrable
kernel verifies the discrete equation. A threshold resonance, where the scalar
denominator vanishes, is excluded from (6). No identical-boson cross section is
inferred; exchange changes scattering-state conventions, not the coupling in(1).

Units: kappa is energy times length², g_a energy, T_a energy times length³,
s_a length, W dimensionless. The continuum free relative generator is
-2kappa Delta. Its outgoing Green kernel, solving (E+2kappa Delta)G=delta,
is -exp(ikR)/(8pi kappa R), for E=2kappa k². This fixes the limiting amplitude
convention f=-T/(8pi kappa), not an exact finite-lattice spherical-wave formula.

## 4. Direct real outgoing boundary estimate

The following proof controls the boundary before taking the spacing limit.
Choose a small fixed delta>0 such that {omega<delta} lies in a coordinate
neighborhood of0. Let y_j=2sqrt(2)sin(p_j/2). Exactly,

    omega(p)=|y|²,
    dp=(1/(2sqrt(2))) product_j(1-y_j²/8)^(-1/2) dy.

Spherical integration for t=|y|² gives the local spectral density

    rho(t)=c sqrt(t)+t^(3/2)b(t),
    c=1/(8sqrt(2)pi²),                                 (7)

where b is smooth on [0,delta] after shrinking delta if necessary. To see the
regularity, the analytic even Jacobian has a convergent power series in the
squared coordinates; its spherical average is analytic in t. The coefficient
is (2pi)^(-3)(1/(2sqrt(2)))(4pi/2)=c.

For0<e<delta/2, taking Im zeta down to0 in the local integral yields

    F(e+i0)+W
      = PV integral_0^delta rho(t)[1/(e-t)+1/t]dt
        -i pi rho(e) + far(e),                          (8)

where the contribution far(e) from omega>=delta is real O(e), by bounding
e/[omega(e-omega)] directly in the original p integral. No smooth global
density assumption at other critical energies is needed. Formula(8) follows
by splitting a smooth density near t=e into its value there plus its difference:
the constant part integrates explicitly, the difference has an integrable
limit, and the imaginary Lorentzian kernel integrates to -pi rho(e).

For the leading c sqrt(t) term, substitute t=v² to find its real contribution

    c sqrt(e) log[(sqrt(delta)+sqrt(e))/(sqrt(delta)-sqrt(e))]=O(e).

For the remainder put h(t)=sqrt(t)b(t). Its real contribution is

    e PV integral_0^delta h(t)/(e-t)dt.

Subtract h(e). Since h is uniformly1/2-Hölder, the resulting difference integrand
is bounded in absolute value by a constant times |t-e|^(-1/2), with uniformly
bounded integral. The constant term is h(e)log[e/(delta-e)], which is bounded
because h(e)=O(sqrt(e)). Thus the remainder is O(e). The imaginary part is
-pi c sqrt(e)+O(e^(3/2)). We have proved the actual boundary expansion

    F(e+i0)=-W-i pi c sqrt(e)+O(e).                      (9)

For each fixed k>0 take E=2kappa k² and e=2a²k². Equations(7)–(9) give

    G_a(E+i0)/a³=-W/(kappa a)-ik/(8pi kappa)+O(a k²).    (10)

Constants here may depend on fixed kappa and the chosen threshold neighborhood.
Only sufficiently small a is used, so E lies in its low-energy lattice regime.
This is a pointwise fixed-energy statement, not a claim uniform over all energies,
all resonances or all times. It was not inferred by analytic continuation of
an off-axis asymptotic expansion.

## 5. Every nonnegative onsite scaling has vanishing scattering

At threshold, for finite g_a>=0,

    T_a(0)=a³g_a/[1+g_a a²W/kappa],
    0<=T_a(0)<=kappa a/W,
    0<=s_a<=a/(8pi W) ->0.                              (11)

This includes arbitrary positive sequences g_a, not just fixed coupling. g_a=0
is evaluated directly; the hard-repulsion limit saturates the upper bound, but
infinity is not an admitted finite Hamiltonian input here.

At fixed E=2kappa k²>0 and g_a>0, (10) gives

    Re(T_a(E+i0)^(-1))=1/(a³g_a)+W/(kappa a)+O(a k²).

The remainder is independent of g_a. For sufficiently small a at fixed k, this
is at least W/(2kappa a), so |T_a(E+i0)|<=2kappa a/W, uniformly in all positive
couplings. With the zero case included, the continuum-normalized amplitude
vanishes for every nonnegative bare scaling. This is a result about this
single-channel onsite cubic model, not every repulsive finite-range interaction
or every possible modification of BPR. It is stronger than the prior fixed-band
residual estimate and is proved here from scattering, not from its nonconverse.

## 6. Conditional tuning with nonzero scattering length

Supply a finite real s!=0 and choose, only as a two-body comparison family,

    1/g_a=-a²W/kappa+a³/(8pi kappa s).                   (12)

For s>0 require a<8pi W s; for s<0 every positive a has a negative right side.
Thus for sufficiently small a the coupling is finite and negative for either
sign of s. This leaves the protected nonnegative-coupling model. The length s
is supplied, not derived from BPR or fitted to data.

At threshold (12) gives s_a=s exactly. At fixed positive energy, (10) yields

    T_a(E+i0)^(-1)=(s^(-1)+ik)/(8pi kappa)+O(a k²).

Since k>0 and finite s!=0 the limiting denominator is nonzero. Therefore

    T_a(E+i0) -> 8pi kappa/(s^(-1)+ik),
    f(k)=-1/(s^(-1)+ik).                                (13)

The signs satisfy f(0)=-s and Im f=k|f|², equivalently
Im T=-k|T|²/(8pi kappa). These are algebraic normalization checks on the limiting
amplitude, not experimental evidence. This is convergence of a continuum-normalized
scalar on-shell contact amplitude. No strong/norm resolvent convergence, wave
operator convergence, interacting time evolution or many-body continuum theorem
has been established by it. Finite-a energy-shell anisotropy is not denied.

## 7. Negative-energy pole check and full-Fock instability

For u>0 small, a separate real integral gives

    F(-u)+W=pi c sqrt(u)+O(u).                          (14)

The leading integral is c u integral_0^delta t^(-1/2)/(u+t)dt
=2c sqrt(u) arctan(sqrt(delta/u))=pi c sqrt(u)+O(u).
The density remainder is bounded by a constant times
u integral_0^delta sqrt(t)/(u+t)dt=O(u), and the far part is O(u).
Consequently at z=-2kappa chi², fixed chi>0,

    G_a(z)/a³=-W/(kappa a)+chi/(8pi kappa)+O(a chi²).

On z<0, G_a is continuous and strictly decreasing: its derivative is minus the
positive integral of1/(z-energy)². Its range is (-a²W/kappa,0). At fixed a,
negative poles solve G_a(z)=1/g_a. With (12), for s>0 and sufficiently small a
the right side lies strictly in that interval, yielding exactly one pole. For
s<0 it lies below the interval, so none exists. A rank-one perturbation has no
other negative eigenstate: its eigenvector must be proportional to
(z-h0,a)^(-1)delta_0, and a state with zero contact amplitude would be a negative
free eigenvector, which is impossible. The resulting state is even.

To locate the pole for s>0, evaluate its denominator at fixed
chi_-<1/s<chi_+, both positive. After division by a³ it tends to
(s^(-1)-chi)/(8pi kappa), with opposite signs at these endpoints. Monotonicity
traps the unique root between them for small a. Taking chi_-,chi_+ arbitrarily
close to1/s proves its energy tends -2kappa/s². This is root squeezing from the
negative-energy integral, not analytic continuation of(13). The pole is excluded
from resolvent statements. Infinite s, above-band poles and uniform pole
convergence are not treated.

Crucially, this attractive tuning is not yet a stable unrestricted Bose theory.
At any fixed a with g_a<0, the normalized state of N particles all at one site
has original hopping expectation0 and interaction g_a N(N-1)/2. Its expectation
tends to minus infinity as N grows. Adding a term linear in N, including the
comparison energy reference, cannot cure the negative quadratic growth. Thus
full-Fock lower boundedness fails even though each two-particle fiber is bounded
self-adjoint. No hard-core cutoff, three-body stabilizer or new field is adopted.
The original positive-coupling stability proof cannot be transferred to (12).

## 8. Frozen finite algebra witness; review gates

The proof is the scattering deliverable. A separate stdlib Fraction checker
will verify only rank-one algebra on the abstract fixture A=[[0,1],[1,0]],
e=(1,0), P=e e^T. This is not a BPR Hamiltonian, a density-of-states approximation,
or a numerical measurement of W.

Freeze six regular cases z=-2,2 and g=-1,0,1, with
G_*(z)=z/(z²-1), tau_*=g/(1-gG_*), and source resolvent G_*/(1-gG_*).
The independent test oracle must invert zI-A-gP from its matrix entries and
check the full resolvent identity, not reuse the production scalar formula.
One separate known pole is (z,g)=(2,3/2): the free inverse exists but the
interacting determinant is0. Label it explicitly; no infinite or invented finite
amplitude. Private free-pole controls z=±1 reject. Matrices never exceed2x2.
No quadrature, floating tolerance, complex framework, physical fitting or old
scientific rerun is authorized. These real off-spectrum cases do not test the
optical identity nontrivially; its imaginary parts would reduce to0=0.

### Frozen executable contract

Independent reviewer a5ced126c5cb9ab66 cleared the conditional mathematics with
no blockers using direct reads only. Before code, their two explanatory
clarifications were incorporated: sampling is restricted to B/a band-limited
functions, and the cutoff Dirichlet integral is justified explicitly. Original
proof submission remains preserved externally. No physical validation follows.

Public API: `demonstration_report()` and `main(argv=None)`. No scientific package
imports. CLI supports default text and exact `--json` with allow_abbrev=False;
argparse rejects other scientific controls/abbreviations before report generation.
Successful main returns0. JSON uses allow_nan=False. Text starts
`Cubic two-body scattering: finite resolvent algebra`, prints six regular cases,
labels the separate expected pole, and prints every limitation. No files/cache.

Private seams, fixed before independent tests:
- `MAX_DIMENSION=2`; `_admit_dimension()` raises ValueError if the live cap is
  below2. Call before any owned2x2 matrix/list construction, and at entry to
  case/report helpers so even scalar representations honor admission.
- `_rational(value)` accepts exact built-in int or exact Fraction only; ValueError
  otherwise. Output exactly {`numerator`:int,`denominator`:int}, reduced and
  denominator positive, including zero and integral rationals.
- `_free_resolvent(z)` accepts exact built-in int in {-2,-1,1,2}. Off-grid/type
  errors raise ValueError; ±1 raises ValueError mentioning `free pole`. At±2
  return Fraction z/(z²-1). No user-supplied frequencies beyond these controls.
- `_case_report(z,g)` accepts exact built-in z in {-2,2}; g is a built-in int in
  {-1,0,1}, or exact Fraction(3,2) only with z=2. All other inputs raise
  ValueError before arithmetic/construction, including bool and subclasses.
  Returns the case schema below; the only permitted pole is explicitly labeled.

Report top-level exact keys: `schema_version`:1,
`status`:`abstract_rank_one_resolvent_diagnostics`, `empirical_validation`:false,
`fixture`:{`A`:[[0,1],[1,0]],`e`:[1,0],`dimension`:2}, `regular_cases`,
`pole_case`, `limitations`.

Six regular_cases are z-major with z=-2,2 and g=-1,0,1 within each. All seven
case dictionaries (including pole_case) have exactly these fields:
- ordinary integer `z`, canonical rational `g`;
- `status`: `regular` or `expected_interacting_pole`;
- canonical rational `free_source`=G, `denominator`=1-gG,
  `interacting_determinant`=z(z-g)-1;
- canonical rational `tau`=g/(1-gG) and `interacting_source`=G/(1-gG) for
  regular cases; both null for the expected pole.
Production uses scalar expressions only, not a reusable matrix inverse engine.
All containers are detached between calls, all output is native JSON with no
floats, and only structural programmer errors may propagate outside admission.

Literal regular targets, ordered as the report:
(z,g,determinant,tau,interacting_source) =
(-2,-1,1,-3,-2), (-2,0,3,0,-2/3), (-2,1,5,3/5,-2/5),
(2,-1,5,-3/5,2/5), (2,0,3,0,2/3), (2,1,1,3,2).
At (2,3/2), G=2/3, denominator=0 and determinant=0. The free poles at±1
must never be relabeled interacting poles, which depend on g.

The exact ordered limitations strings are:
1. `This fixture checks finite rank-one algebra, not a cubic lattice approximation.`
2. `The scattering limit rests on the reviewed analytic boundary proof, not these tests.`
3. `Nonnegative onsite couplings have vanishing continuum scattering in the stated model.`
4. `The attractive two-body tuning is not an adopted stable many-body model.`
5. `No empirical validation, matter/gravity completion or TOE claim is made.`

Independent tests use literal matrices and adjugate inversion, not production
scalar helpers, for the full resolvent identity and both inverse residuals.
Every comparison is exact Fraction equality. At most2x2 matrices, no scientific
imports, complex arithmetic, continuum evaluation or hidden subprocess campaign.
Tests mock CLI generation; the actual text/JSON modes run separately once each.

If the real boundary estimate or normalization fails review, stop with the
established partial result; no executable witness substitutes for the missing proof.

## 9. Provenance and preservation

Protected references (reasoning reused, APIs not modified):
- `doc/derivations/cubic_contact_continuum_2026-09-13.md:11-80,159-180`: symmetric
  normalization, contact and phase, limitations of bare matching/leakage.
- `doc/derivations/supplied_cubic_bose_2026-09-13.md:60-77,94-147`: original
  stability assumptions, Fourier measure and dispersion.
- `tests/test_supplied_cubic_bose.py:73-79,274-294`: normalized position-pair oracle.
- `doc/derivations/substrate_collective_dynamics_2026-09-12.md:19-35`: finite
  resolvent signs; no inherited infinite-volume boundary theorem.
- `scripts/check_cubic_contact_continuum.py` and its tests: standalone exact
  rational report/strict CLI patterns, not a scattering solver.

The present infinite-lattice, real-boundary and tuning arguments are new derivations
within this work, not claims of scientific novelty. No external constant or
experimental input has been fitted. Local exploration found no existing scattering
construction to import; ring doublon-hole spectral compression is not this problem.

Baseline of662tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-two-body-scattering-9phycpvf/baseline.json`.
Inventory uses git ls-files --cached --others --exclude-standard; ignored caches,
config, build products and .git are outside the preservation claim. Only this
note and the new checker/test may be added. All old failures and incomplete
regressions remain unchanged.

## 10. Software review and execution record

Test author a0ee5e95a5468f93a independently wrote20unittest methods from the
frozen contract without inspecting the implementation. Static reviewer
ac2b83589d08a6976 cleared the checker, tests and external runner with no blockers,
using direct reads only. The independent matrix oracle uses explicit2x2
adjugate inversion, not the production rank-one formula. No source/test repair
was needed before execution. Initial files are preserved externally.

The single planned run is exact new-test discovery120s, then text30s and JSON30s,
sequentially in fresh directories. Pre-run byte snapshots and hashes bind all
three additions. Parent and children use the same explicit CommandLineTools
Python path; saved sys.version describes the parent rather than an independent
child probe. Bytecode is disabled, PYTHONPATH removed, warnings are errors and
six thread variables are1. Timeout, observed interruption, residual process group
or changed inventory stops the runner. Uncatchable abrupt runner termination is
not comprehensively handled; missing completion cannot earn success. No old
suites, numerical integration, scattering curves or supplemental science.
### Actual bounded execution

Artifacts are in `cubic-two-body-scattering-9phycpvf/first-bounded-run/` under
the external verification directory. First and only attempt:

| Process | Saved outcome | Whole child seconds |
| --- | --- | --- |
| New exact-pattern unittest discovery | 20 methods passed; unittest time0.072s | 0.343271375 |
| Default text CLI | exit0 | 0.132152125 |
| Strict JSON CLI | exit0 | 0.129563292 |

All outcomes report no timeout, observed interruption or residual process group.
All665bound file hashes matched before and after each child, including all662
protected prior files. The six regular cases match the frozen rational targets;
the separate expected pole prints undefined amplitudes, not finite replacements.
These checks verify only the abstract finite algebra. No W evaluation or numerical
scattering evidence was produced. Script and tests remain frozen after evaluation;
only this note's status and actual results change. Independent reviewer
ac2b83589d08a6976 cleared the saved-evidence audit with no blockers:20passes,
strict rational JSON, both CLI outputs, prescribed commands/environment,
all662protected files and exactly3additions were confirmed. Every saved post-child
inventory matches the pre-run inventory; the current665file inventory matched
final-hashes.json. Script/tests remain byte-identical to initial and pre-run
snapshots. The reviewer used direct reads and stdlib artifact hashing/parsing,
without scientific imports, tests or reruns. This is saved-evidence clearance,
not independent observation of live execution. Retain the audited final-hashes.json;
closing-hashes.json binds these final audit-status edits.

## 11. Scoped conclusion

Within the stated infinite-lattice, zero-total-momentum, single-channel onsite
model, every nonnegative bare-coupling scaling has vanishing continuum-normalized
scattering at fixed physical energy. A precisely supplied attractive counterterm
instead yields the finite nonzero scalar amplitude in(13), with the proved
negative-energy pole behavior. This closes the conditional two-body scattering
question, not the stability or full-continuum-operator problem. The tuning makes
unrestricted full-Fock energy unbounded below, so it is not an adopted stable
BPR model. Geometry, quantization and the target length remain supplied inputs;
matter/gravity completion and empirical distinction remain open. No next model,
old regression or publication starts automatically.
