# Onsite three-body repulsion: stability and its continuum limitation

2026-09-13. Status: conditional mathematics independently cleared with no blockers;
executable contract frozen below; 20 exact tests and both CLI modes passed;
final independent saved-evidence audit cleared with no blockers.
This is a separate conditional candidate, not a modification of any protected
BPR Hamiltonian/API or a claim of physical validation.

## 1. Candidate, energy reference and claims to distinguish

On the cubic lattice supply a,kappa>0, C_a=kappa/a² and the attractive pair
coupling g_a from the prior scattering analysis. Add a supplied finite w_a>0:

    P_j=sum_x binomial(n_x,j),
    H_a=-C_a sum_{edges{x,y}}(a_x†a_y+a_y†a_x)+g_a P_2+w_a P_3,
    K_a=H_a+6C_a Nhat.                                  (1)

There is no local occupancy cutoff. The cubic graph, quantization, pair tuning
and new onsite repulsion are inputs. The coefficient w_a may grow arbitrarily
fast as a decreases, but is finite at each fixed a. K_a is an explicitly
chosen comparison with the sum of free single-particle thresholds, not a silent
change to intersector population energetics of H_a.

We distinguish fixed-a finite-volume full-Fock lower boundedness, fixed-a
volume-independent extensive lower bounds, and bounds uniform as a decreases.
The last can fail even if the first two hold. A diverging lower estimate cannot
prove collapse; section5 instead constructs a negative variational upper bound.

## 2. Sectors, domains and exact two-particle preservation

Let F_N be the complete normalized Bose N-particle sector. On an infinite
cubic lattice, hopping is a sum of N bounded one-particle adjacency operators,
restricted to symmetric tensors; its norm is at most6C_aN. For any occupation,
0<=P_j<=binomial(N,j): each same-site j-tuple is one of the N-particle j-tuples.
Thus all interactions in(1) are bounded on F_N, even though the infinite-volume
sector itself is infinite-dimensional. Each H_{a,N},K_{a,N} is bounded
self-adjoint. In a finite volume the sectors are finite-dimensional instead.

Define the full operators by their self-adjoint sector direct sums, for example

    D(H_a)={psi=(psi_N): sum_N ||H_{a,N}psi_N||²<infinity}.

Testing the adjoint sectorwise gives the same domain; truncation in N converges
in graph norm. Finite-number-support vectors form a core. On the infinite
lattice, finite occupation-support approximants within each fixed sector also
converge in graph norm because that sector operator is bounded. Total number
spectral projections commute with the dynamics. No claim of semiboundedness of
the infinite-volume full direct sum follows from self-adjointness alone.

For N=0,1,2, P_3=0 as an operator on the COMPLETE sector. The new Hamiltonian
restriction is identical to the old pair-only restriction for the same g_a,C_a.
All previously derived two-body scattering lengths, amplitudes and pole results
are exactly unchanged. Number conservation prevents virtual transitions to other
N sectors here. This is not merely vanishing expectation on a chosen state.
The prior two-body result remains a scalar scattering-amplitude limit, not a
many-body continuum-operator theorem.

## 3. Fixed-cutoff stability

Write u=|g_a|>0, w=w_a>0 and

    v(n)=-u n(n-1)/2+w n(n-1)(n-2)/6.

For n>=1 put t=n-1. Completing the square gives

    v(n)/n = (w/6)[t-(3u+w)/(2w)]² - B_a,
    B_a=(3u+w)²/(24w),
    v(n)+B_a n = n[2w(n-1)-(3u+w)]²/(24w)>=0.           (2)

For n=0, v(0)=0 and the last identity also equals0; no division by n is used.
On the infinite cubic lattice and periodic cubic boxes, shifted hopping is
C_a sum_edges(a_x-a_y)†(a_x-a_y)>=0. On an open induced finite box there are
additional nonnegative missing-neighbor diagonal terms when the shift6C_aNhat
is used. Therefore, sectorwise as bounded-operator inequalities,

    K_{a,N}>=-B_a N I,
    H_{a,N}>=-(6C_a+B_a)N I.                            (3)

These extend as quadratic-form comparisons wherever both sides are defined;
in particular on finite-number-support vectors. B_a is independent of volume
at fixed couplings, but not asserted uniform in a. These statements are not
bounds independent of N on full Fock space.

For a finite volume Lambda, use hopping>=-6C_aNhat directly:

    H_a>=sum_{x in Lambda}[v(n_x)-6C_a n_x]
       >= |Lambda| min_{n in N_0}[v(n)-6C_a n].          (4)

The scalar cubic has positive leading coefficient w/6, tends to plus infinity,
and hence has a finite minimum on nonnegative integers. This proves genuine
finite-volume full-Fock lower boundedness at fixed a. It is independent of
any truncation of n. It is not derived from(3) alone, does not give a bound
uniform in a, and is not asserted as an infinite-volume full-Fock bound.

## 4. A sufficient analytic bound on the inherited tuning

The protected scattering construction supplies finite s!=0 and

    1/g_a=-a²W/kappa+a³/(8pi kappa s),
    g_a/C_a=[-W+a/(8pi s)]^(-1),                        (5)
    W=integral_{[-pi,pi]^3} dp/(2pi)^3 /omega(p),
    omega(p)=4 sum_j(1-cos p_j),    0<W<infinity.

We need only a coarse bound, not a numerical value of W. Concavity of sin on
[0,pi/2] gives sin(t)>=2t/pi there. Hence for |p|<=pi,
1-cos p=2sin²(p/2)>=2p²/pi². If r=||p||_infinity, then
omega(p)>=8r²/pi². Cubic shells have enclosed volume(2r)^3 and derivative24r²,
so the integrable upper bound yields

    W <= [pi²/(8(2pi)^3)] integral_0^pi r^(-2)24r² dr
      = 3/8.                                           (6)

The single point r=0 has measure zero; its integrable singularity is handled
by the shell integral. Combining0<W<=3/8 with(5),

    g_a/C_a -> -1/W <= -8/3 < -2.

Thus g_a/C_a<=-2 for every sufficiently small a, for either sign of the fixed
finite s. This is a strict eventual margin. There is no quadrature, fit or
claim that the upper bound3/8 is W's actual value.

## 5. Trial states invisible to all onsite triple repulsion

### Local state and exact moments

For a fixed x>0 define the normalized local vector

    phi=(|0>+sqrt(x)|1>+x|2>/sqrt(2))/sqrt(Z),
    Z=1+x+x²/2.

Use only x=1/4 henceforth. The probabilities of occupations0,1,2 are
32/41,8/41,1/41. In the ordinary normalized occupation inner product,

    m=<n>=10/41,
    alpha=<a>=20/41,
    p2=<binomial(n,2)>=1/41,
    p3=<binomial(n,3)>=0,
    m-alpha²=10/1681.                                  (7)

For example alpha=sqrt(p0 p1)+sqrt(2p1 p2)=(16+4)/41.
This is a variational vector in the unrestricted Bose space, not a change of
operator domain or an imposed hard-core constraint. Hopping can leave its
occupancy<=2 support; a Rayleigh expectation does not require invariance.

### Boundary-correct finite block

Take Phi_b as the tensor product of phi on a b-sided cubic block and vacuum
on all other sites. The finite-support product is a normalized Fock vector
with number support0..2b³. Its block has M=b³ sites and E=3b²(b-1) unordered
internal edges, with6b² edges leading to the exterior. Every internal hopping
edge contributes -2C_a alpha², and every crossing edge has zero hopping
expectation because the exterior vacuum has <a>=0. The threshold shift still
counts every active particle. The triple penalty annihilates Phi_b exactly,
for any w_a. Therefore

    <Phi_b,K_a Phi_b>/C_a
      =6b³m-6b²(b-1)alpha²+(g_a/C_a)b³p2.              (8)

Equivalently the kinetic edge form contributes2(m-alpha²) per internal edge
and m per crossing edge, giving the same expression. This retains boundary
costs instead of replacing the finite block by an infinite uniform product.

At the reference dimensionless pair ratio-2, (7)–(8) give

    E_ref(b)=[-22b³+2400b²]/1681.                       (9)

Use the single predetermined block b=128:

    E_ref(128)=-6,815,744/1681=:-D<0.                   (10)

For boundary-positive controls, E_ref(1)=2378/1681 and E_ref(2)=9424/1681.
No block size is optimized. b=128 is only a scalar in the formula: no such
graph, tensor vector, occupation list or sector is constructed computationally.

### One fixed number sector, fixed for every cutoff

Define on the infinite lattice the reference dimensionless number-conserving
operator K_ref=T_shift-2P_2, where T_shift is the hopping-plus6Nhat operator
at C=1. Decompose Phi_128 into its finitely many orthogonal number components.
Let q_N be their squared norms and psi_N their normalized nonzero components.
Because K_ref preserves number,

    -D=sum_{N:q_N>0} q_N <psi_N,K_ref psi_N>,
    sum_N q_N=1,    0<=N<=2(128)^3.

At least one component has expectation no greater than-D. N=0 has energy0,
so one may select a SINGLE N_* with1<=N_*<=4,194,304 and its psi_* such that

    <psi_*,K_ref psi_*><=-D.                            (11)

This is an existence argument, not a computed particle number or a ground-state
construction. Select the component once from the reference operator, not anew
as a changes. Number projection preserves the property of no triple occupancy,
so P_3 psi_*=0. The nonnegative operator P_2 has finite expectation since N_*
is finite. For sufficiently small a as in section4,

    <psi_*,K_a psi_*>/C_a
      = <psi_*,K_ref psi_*>
        +(g_a/C_a+2)<psi_*,P_2 psi_*>
        +(w_a/C_a)<psi_*,P_3 psi_*>
      <= -D.                                           (12)

This holds for EVERY finite positive w_a, no matter how fast it grows as a->0.
The same normalized lattice-coordinate vector, in the same fixed number sector,
thus gives

    inf spectrum(K_{a,N_*}) <= -D kappa/a² -> -infinity. (13)

It is a variational upper bound, not a failure of a lower estimate. In particular
there can be no lower bound independent of a at this fixed N_*, nor a bound
K_{a,N}>=-B N with B independent of a valid for all N. The conclusion concerns
the explicitly threshold-referenced energy K, not merely the unshifted free
energy which already carries a divergent -6C_aN reference.

The finite-support occupation pattern also embeds into increasingly fine grids
in a fixed physical box: choose it away from the boundary/periodic seam once
there are enough sites. Its physical support diameter is O(128a), tending to0.
No uniform finite-density hypothesis is imposed. The upper bound remains valid
there with the same local hopping expectations and number. No large volume
or particle number is being taken to infinity along the spacing sequence.

### What the obstruction does not assert

We do not determine N_* or the smallest unstable population, claim N_*=3,
compute an eigenvalue or a time-to-collapse, or exclude metastable low-energy
dynamics. This does not prove nonexistence of every possible continuum operator
limit or rule out finite-physical-range interactions or other three-body terms.
It refutes cutoff-uniform lower boundedness for the particular onsite stabilizer
in(1) while keeping the prior pair tuning. The trial's occupancy support is not
an invariant subspace and is not an adopted occupancy truncation.

## 6. Finite verification scope and source register

Planned scalar controls: fourteen local cases n=0..6 with g=-1,w=1,2; the one
local state(7); block labels b=1,2,128 in(9). Exact arithmetic checks binomial
counts, square completion and finite-block energy algebra. Such finite tests
cannot prove universal-n inequalities or the number-sector existence argument.
Those require the reviewed derivation above. There is no numerical W evaluation,
large graph, Hamiltonian matrix, number projection, spectrum or continuum run.
### Frozen executable contract

Reviewer a097b2e1e7e20bc99 cleared the conditional mathematics and literal targets
using read-only review. The submitted proof is preserved as proof-draft-v1.md
externally. This clears neither software nor empirical physics.

Only public `demonstration_report()` and `main(argv=None)`. Stdlib Fraction only,
no scientific imports or file writes. CLI supports default text and exact
`--json` (allow_abbrev=False), parses before generation and returns0 on success.
JSON uses allow_nan=False; text begins `Cubic three-body stability: exact scalar checks`,
prints all14onsite cases, three block energies, and every limitation.

Private seams:
- `_admit()` checks live MAX_OCCUPATION=6, MAX_BLOCK_SIDE=128 and
  MAX_LOCAL_ENTRIES=3 against the fixed requirements6,128,3; if any is smaller,
  raise ValueError before arithmetic/construction. These caps admit scalar
  labels, not enumeration of block sites or number sectors.
- `_rational(value)` accepts only exact built-in int or exact Fraction;
  otherwise ValueError. Return reduced {numerator:int,denominator:int} with
  positive denominator, including zero/integral values.
- `_onsite_case(n,w)` calls admission first, requires built-in int n in0..6
  and w in{1,2}, otherwise ValueError. g=-1 is fixed. Return exactly integer
  fields n,w,pair_count,triple_count and canonical rational fields energy,
  bound_coefficient,square_remainder. Here energy=-binom(n,2)+w binom(n,3),
  bound_coefficient=(3+w)²/(24w), square_remainder=energy+n*bound_coefficient,
  including n=0 without division by n.
- `_local_moments()` calls admission and returns exactly canonical rational
  x=1/4,normalization=41/32,mean_number=10/41,annihilation=20/41,
  pair_mean=1/41,triple_mean=0,kinetic_difference=10/1681, plus probabilities
  as a list of canonical rational32/41,8/41,1/41. No other keys.
- `_block_case(b)` calls admission first, requires built-in int b in{1,2,128}.
  Return exactly ordinary integers b,sites=b³,internal_edges=3b²(b-1),
  crossing_edges=6b²,max_number_support=2b³, plus canonical rational fields
  mean_number=sites*10/41,shifted_kinetic=6sites*10/41-2internal_edges*(20/41)²,
  pair_energy=-2sites/41,triple_energy=0,reference_energy=sum of these energies.
  Evaluate these as scalars; never construct a graph or loop to b³ or2b³.

Report exact top-level keys: schema_version=1,
status=`conditional_onsite_three_body_stability_diagnostics`,
empirical_validation=false, onsite_cases, local_state, block_cases, limitations.
Onsite order is w=1,2 with n=0..6 within each; block order1,2,128. Local_state
is the local moment dictionary. Every container is detached between calls.
All numbers are native JSON integers or canonical rational records, no floats.

Exact ordered limitations:
1. `The onsite three-body penalty is a supplied assumption and leaves N<=2 unchanged.`
2. `Fixed-spacing finite-volume boundedness is not uniform continuum stability.`
3. `The continuum obstruction is proved by a trial state with zero triple occupancy.`
4. `The fixed particle number exists analytically; no large block or sector is constructed.`
5. `Exact scalar checks are not empirical validation or a theory of everything.`

Freeze literal onsite energy lists for n0..6: w1 gives[0,0,-1,-2,-2,0,5],
w2 gives[0,0,-1,-1,2,10,25]. Bound coefficients are2/3 and25/48. Block energies
are2378/1681,9424/1681,-6815744/1681. No tolerance or optimized control.
Independent tests use at most6labelled occupants (20triples),3probability entries,
8small-block sites and28unordered site pairs. b128 uses factored scalar counts.
Use integer perfect-square checks for the two ladder-weighted probability products,
not floating roots or a modified inner product. Tests mock CLI generation; real
text and JSON run separately once. Proof of all-n stability and fixed-N extraction
is analytic, not inferred from the finite test count.

Protected sources (reasoning reused, files unchanged):
- `doc/derivations/cubic_two_body_scattering_2026-09-13.md:218-284`: pair tuning,
  supplied scattering length and distinction between two-body and full-Fock claims.
- `doc/derivations/supplied_cubic_bose_2026-09-13.md:33-77`: complete sectors,
  direct-sum construction and graph hopping bound.
- `doc/derivations/substrate_extensive_stability_2026-09-13.md:13-63`: stability
  distinctions and variational reasoning; its positive-pair bound is not imported.
- `scripts/check_cubic_two_body_scattering.py` and its tests: exact canonical
  fractions, isolated report/CLI and independent algebra oracles, not new physics.

Baseline665tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-three-body-stability-ez7pnb0y/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard, excluding ignored
caches/config/build products and .git. Only this note plus the new stability
checker and tests may be added. No old failures/regressions are resumed or
reclassified.

## 7. Software review and bounded verification

Independent mathematical reviewer a097b2e1e7e20bc99 cleared the proof and literal
targets. Test author a75dded78c4d893bf wrote20methods from the frozen contract
without inspecting production. Static reviewer af316787b6e7d3e10 cleared the
checker, independent oracles and runner with no blockers, using reads only.
No implementation or test repair was needed before execution.

The single planned run is new exact-pattern unittest discovery120s, text30s and
JSON30s, sequentially in fresh directories. All three pre-run files are preserved
byte-for-byte with hashes. Parent and children use the same explicit CommandLineTools
Python path; recorded sys.version identifies the parent, not a separate child
runtime probe. Bytecode is disabled, PYTHONPATH removed, warnings are errors and
six thread variables are1. The runner stops on failure, timeout, observed
interruption, residual child group or changed inventory. Abrupt uncatchable
termination may prevent cleanup or snapshots; missing completion earns no success.
No old suites, large-block computation or supplemental science is authorized.
### Actual bounded execution

Evidence is saved at `cubic-three-body-stability-ez7pnb0y/first-bounded-run/`
under the external verification directory. First and only attempt:

| Process | Saved outcome | Whole child seconds |
| --- | --- | --- |
| New exact-pattern unittest discovery | 20 methods passed; unittest time0.044s | 0.345292083 |
| Default text CLI | exit0 | 0.133415958 |
| Strict JSON CLI | exit0 | 0.131417875 |

No saved outcome reports timeout, observed interruption or residual process group.
All668bound file hashes matched before and after each child, including665protected
prior files. Script and tests remain frozen; only this note's status/results change.
The text retains both positive block controls and the negative b128control,
with exactly zero triple energy in all three. Its b1energy58/41 is the canonical
reduction of2378/1681. This verifies exact scalar arithmetic, not success of the
stabilizing candidate or numerical simulation of a large block. No rerun or
supplemental science occurred. Independent reviewer af316787b6e7d3e10 cleared
the saved-evidence audit with no blockers:20passes, bothCLIoutputs, canonical
rational fields and all14onsite/local/three-block records matched; all665protected
files and exactly3additions were confirmed. Script/tests match initial and pre-run
snapshots; proof and frozen contract are unchanged. The complete668file inventory
matched final-hashes.json at audit time. This is saved-artifact clearance, not
live-process observation or empirical validation; no tests, checker or runner
were rerun. The audited final-hashes.json is retained, and closing-hashes.json
binds these final audit-status edits.

## 8. Candidate decision

The added onsite three-body repulsion preserves two-body scattering exactly and
restores fixed-spacing finite-volume full-Fock lower boundedness. It nevertheless
fails to give a lower bound uniform in the continuum limit under the specified
pair tuning. The proved counterexample has zero triple occupancy, so arbitrarily
large finite w_a cannot remove it. This is a failure of this proposed adjustment,
not a software-test failure or a general no-go for all stabilizing interactions.
No alternative model, changed range, physical calibration or publication is
started by completion of this gate. BPR is not thereby a theory of everything.
