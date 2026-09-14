# Fixed-range, bounded-strength three-particle repulsion

2026-09-13. Status: conditional mathematics independently cleared with no blockers;
executable contract frozen below; 20 exact tests and both CLI modes passed;
final independent saved-evidence audit pending.
This is a separate conditional candidate, not an adopted stable BPR model.

## 1. Definition and counting conventions

Supply fixed physical R>0 and finite Lambda>0, independent of lattice spacing a.
Retain the prior attractive pair tuning, C_a=kappa/a², fixed kappa>0 and finite
supplied scattering length s!=0. On the complete symmetric N-particle position
space define the real diagonal operator

    Q_(a,R)(x_1,...,x_N)
      = sum_{i<j<k} 1{max(a|x_i-x_j|,a|x_i-x_k|,a|x_j-x_k|)<=R},
    K_new,a = K_pair,a + Lambda Q_(a,R).                (1)

The norm in(1) is ordinary Euclidean distance on the infinite cubic lattice,
or an open box without periodic wrapping. Indices are distinct particle labels;
their positions may coincide. The cutoff is inclusive and all three pair
distances must pass. Closeness to one anchor alone does not suffice.

K_pair,a is the previously tuned pair Hamiltonian in the explicit threshold
reference H_pair,a+6C_a Nhat. The new term replaces the previous candidate's
onsite triple penalty in this analysis; penalties are not silently accumulated.
R,Lambda and the geometry are new supplied assumptions, not predicted scales
or measured fits. Infinite barriers and cutoff-dependent Lambda_a are not the
candidate defined here.

Every unordered label triple contributes either0 or1. Thus Q is permutation
symmetric, number preserving, and on every complete N sector

    0<=Q_(a,R)<=binomial(N,3) I.                        (2)

For N<=2 it is identically zero. Hence the two-body operators, their scattering
length and the previously reviewed scalar scattering-amplitude/pole statements
are unchanged, with all prior scope limitations retained. Number conservation
prevents virtual transitions to other N sectors in this model.

For occupations n_x at a finite collection of occupied sites, a disjoint
classification by number of distinct positions gives

    Q = sum_x binomial(n_x,3)
        + sum_{x!=y, a|x-y|<=R} binomial(n_x,2)n_y
        + sum_{x<y<z, a diameter(x,y,z)<=R} n_x n_y n_z. (3)

The middle sum is ordered: x carries the two labels and y the single label.
The last sum is unordered using any fixed site order. The three cases partition
all unordered label triples, proving(3) without an extra symmetry factor.
In particular Q>=P3=sum_x binomial(n_x,3), since coincident triples always pass.
Discarding repeated positions would change the operator and invalidate this bound.

## 2. Fixed-spacing dynamics and stability

On the infinite lattice at fixed N, pair interaction is bounded by
|g_a|binomial(N,2), hopping by6C_aN in norm, and the new interaction by
Lambda binomial(N,3). Thus the sector Hamiltonian is bounded self-adjoint.
Finite-volume sectors are finite matrices. Full Fock operators are defined by
self-adjoint direct sums, with for example

    D(K_new,a)={psi=(psi_N): sum_N ||K_new,a,N psi_N||²<infinity}.

The sectorwise adjoint has exactly this domain. Number truncations approximate
in graph norm, giving a finite-number-support core; at each fixed N, finite
occupation-support approximants also work because the sector operator is bounded.
Self-adjointness alone makes no infinite-volume full-Fock semiboundedness claim.

For sufficiently small a the retained g_a is negative; set u=|g_a|. Since
Q>=P3, comparison with the reviewed onsite polynomial
v(n)=-u binomial(n,2)+Lambda binomial(n,3) gives

    v(n)+B_a n
      = n[2Lambda(n-1)-(3u+Lambda)]²/(24Lambda)>=0,
    B_a=(3u+Lambda)²/(24Lambda).                         (4)

The formula includes n=0 without division by n. Shifted hopping is nonnegative,
so sectorwise K_new,a>=-B_a Nhat. On open boxes the6C_aNhat reference includes
nonnegative missing-neighbor diagonal contributions, so the comparison remains
valid. For a finite volume Gamma, unshifted hopping>=-6C_aNhat yields

    H_new,a >= |Gamma| min_{n>=0 integer}[v(n)-6C_a n].  (5)

The scalar polynomial has positive cubic leading coefficient, hence its minimum
is finite. This proves fixed-a finite-volume full-Fock lower boundedness, not
just a fixed-N statement. Neither(4) nor(5) supplies a cutoff-uniform bound or
an infinite-volume full-Fock lower bound.

## 3. Reuse of the fixed-number collapse witness

The protected onsite-stability derivation supplies a normalized psi_* supported
in a128-sided block in lattice coordinates and in ONE fixed number sector,
with1<=N_*<=4,194,304, such that for all sufficiently small a

    <psi_*,K_pair,a psi_*> <= -D kappa/a²,
    D=6,815,744/1681.                                   (6)

The same vector and N_* are selected once at reference dimensionless pair
ratio-2. It has no triple occupancy, so the prior onsite penalty made zero
contribution; its pair-only expectation is precisely the one bounded in(6).
This is a previously reviewed variational result, not a sector computed here.
Its transfer to small a uses the retained g_a/C_a<=-2 eventually.

A useful sharpening follows without identifying N_*: the reference operator
T_shift-2P2 has T_shift>=0 and P2<=binomial(N,2). For N<=2 it is at least-2.
But the selected component has reference expectation<=-D<-2. Thus the selected
sector must satisfy

    3<=N_*<=4,194,304.                                  (7)

This does not say N_*=3 or compute a smallest unstable population.

For EVERY a, (2) bounds the new repulsive expectation by
Lambda binomial(N_*,3). Combining with(6), only for sufficiently small a,

    inf spectrum(K_new,a,N_*)
      <= <psi_*,K_new,a psi_*>
      <= -D kappa/a²+Lambda binomial(N_*,3)
      -> -infinity.                                    (8)

The second term is a finite constant, however large, since N_* and Lambda are
fixed. The conclusion concerns threshold-referenced energy, not the unrelated
divergence of the free unshifted energy minimum.

In fact, the new potential does not miss this cluster. Every pair of lattice
sites in its support is separated by at most sqrt(3)*127. Whenever
sqrt(3)*127a<=R, every unordered triple in each supported configuration passes
the range test, giving the exact action

    Q_(a,R) psi_* = binomial(N_*,3) psi_*.               (9)

Number projection in the old construction does not enlarge support. Thus(9)
is an operator statement on this vector, not merely a bound on an average.
The bounded-strength term detects all triples but cannot offset the negative
energy diverging as a^(-2). The same occupation pattern embeds away from the
boundary in any sufficiently fine fixed physical box; its physical diameter
shrinks to0. No number or large volume is sent to infinity along this sequence.

Equation(8) is a negative VARIATIONAL UPPER BOUND, not a failed lower estimate.
It rules out a cutoff-uniform lower bound at this particular fixed finite N_*,
and therefore any cutoff-uniform extensive bound valid for all N. It does not
rule out metastable dynamics, a restricted low-energy theory, or every notion
of continuum operator convergence.

For comparison only, suppose a different family Lambda_a were claimed to have
a uniform lower bound -L at N_*. Since a lower spectral bound must also hold
on psi_*, (6) and(9) would require

    Lambda_a binomial(N_*,3) >= D kappa/a²-L.            (10)

The divisor is positive by(7). This is necessary, not sufficient, for that
hypothetical claim. A divergent strength or hard-core constraint is not adopted,
proved stable or evaluated here. Fixed range by itself does not address strength.

## 4. Limited continuum meaning of the potential

For fixed N on symmetric L²((R³)^N), define Q_R by the same triple rule with
physical coordinates r_i. It is bounded measurable real multiplication with
0<=Q_R<=binomial(N,3). Consequently

    H_free,R=-kappa sum_i Delta_i+Lambda Q_R

is self-adjoint on the symmetric free H² domain by the bounded self-adjoint
perturbation theorem, and nonnegative. This is a FREE-PAIR comparison fact,
not the many-body continuum Hamiltonian of the tuned attractive pair model.
The old scattering proof does not supply that missing operator construction.

One limited convergence statement is available on this common L² space. Define
M_a(r_1,...,r_N)=Lambda Q_(a,R)(floor(r_1/a),...,floor(r_N/a)), componentwise.
Physical rounding errors are at most sqrt(3)a for each particle. Every pair
distance therefore converges to its unrounded value. Away from the finite union
of sets |r_i-r_j|=R, every threshold comparison eventually agrees, hence
M_a(r)->Lambda Q_R(r) pointwise almost everywhere. Each such distance-R set has
Lebesgue measure zero: use relative and remaining coordinates and the zero-volume
sphere of radius R, then Fubini on bounded boxes. R>0 is fixed.

All M_a and Lambda Q_R have absolute value at most Lambda binomial(N,3).
For any fixed L² vector f, dominated convergence applied to
|(M_a-Lambda Q_R)f|² proves

    ||(M_a-Lambda Q_R)f||_2 ->0.                         (11)

Permutation symmetry preserves the bosonic subspace. This is strong convergence
of multiplication operators on a common continuum space, not an unstated lattice
sampling isometry. It does not establish operator-norm convergence of discontinuous
cutoffs, kinetic-operator convergence, or convergence of the full tuned interacting
resolvents or propagators. The nonnegative free-pair comparison cannot replace
the unstable tuned candidate in(8) and be called its stable continuum limit.

## 5. Frozen geometry controls and planned checker

Use R=Lambda=1 as mathematical illustration units, not fitted physical values.
With O=(0,0,0), X=(1,0,0), Y=(0,1,0), freeze exactly13report cases:

| Label | Positions | a | Q |
| --- | --- | --- | --- |
| empty | () | 1 | 0 |
| single | O | 1 | 0 |
| coincident_pair | O,O | 1 | 0 |
| separated_pair | O,2X | 1 | 0 |
| coincident_triple | O,O,O | 1 | 1 |
| doubled_near | O,O,X | 1/2 | 1 |
| doubled_boundary | O,O,X | 1 | 1 |
| doubled_far | O,O,2X | 1 | 0 |
| distinct_near | O,X,Y | 1/2 | 1 |
| diagonal_far | O,X,Y | 1 | 0 |
| anchor_trap | -X,O,X | 1 | 0 |
| six_near | O,O,O,X,X,Y | 1/2 | 20 |
| split_clusters | O,O,O,3X,3X,3X | 1 | 2 |

These demonstrate counting semantics, including coincident positions, inclusive
boundary equality, Euclidean diagonal rejection and the all-three-pairs rule.
The old block and N_* remain entirely analytic; they are never allocated.
No numerical continuum dynamics, spectrum, W integral or scattering run occurs.
The approved13cases are retained; a later design suggestion to expand them is
not adopted. Independent reviewer a631ebacbed17ae48 cleared the mathematics and
all13counts by direct reads only; the submitted draft is preserved externally.
No software or physical clearance is inferred from this review.

### Frozen executable contract

Public functions are `demonstration_report()` and `main(argv=None)` only.
Stdlib Fraction arithmetic, no scientific imports or file writes. CLI supports
default text and exact `--json`, with allow_abbrev=False; parse before generating
report, return0 on success, JSON allow_nan=False. Text starts
`Finite-range three-body repulsion: exact geometry checks`, prints all13IDs and
counts with exact energies, then every limitation. No graph or sector allocation.

Private seams:
- `MAX_PARTICLES=6`, `MAX_PAIRS=15`, `MAX_TRIPLES=20` are live workload caps.
  `_admit(positions,spacing)` requires an exact built-in tuple of at most6
  position tuples and an exact Fraction equal1 or1/2. Each coordinate tuple
  has exactly3 built-in int components in[-4,4]. Reject bool, subclasses,
  floats, list containers or unsupported values with ValueError. Check outer
  type/length and both fixed and live combinatorial caps before iteration,
  then coordinate types/shapes before distance arithmetic. Return None.
  Raising caps does not admit more than6labels. Zero-particle cases obey the
  natural zero pair/triple counts, not an artificial minimum workload.
- `_squared_distance(left,right,spacing)` is a private arithmetic seam for
  already admitted points; returns exact Fraction physical squared distance.
- `_count_triples(positions,spacing)` admits first, computes each unordered pair
  distance once (at most15), counts labelled triples with all three squared
  distances<=1 (at most20), and returns a built-in int. No position deduplication.
- `_rational(value)` accepts built-in int or exact Fraction only, otherwise
  ValueError. Output exactly {numerator:int,denominator:int}, reduced with
  positive denominator, including zero and integral values.
- `_fixtures()` returns the fixed13label/positions/spacing triples in table order,
  with exact tuples and Fraction spacing. `_case_report(label,positions,spacing)`
  first admits geometry, then requires a built-in string label from the table.
  The private case helper need not require geometry identical to the labelled
  fixture; bounded permutation/translation tests may reuse a valid label.

Exact top-level report keys: schema_version=1,
status=`conditional_fixed_range_three_body_diagnostics`, empirical_validation=false,
range and strength (canonical rationals1), cases, limitations. There are exactly
13cases in table order. Each dictionary has exactly:
- string id, positions as detached lists of3ordinary ints, canonical rational
  spacing, integer particle_count;
- integer triple_count, onsite_triple_count=sum_x binomial(n_x,3),
  triple_upper_bound=binomial(N,3);
- canonical rational repulsion_energy=triple_count, since Lambda=1.

Every container is fresh/detached between calls; no floats or nonnative JSON.
The exact ordered limitations list is:
1. `Range and strength are supplied fixed physical parameters, not fitted predictions.`
2. `Triples count particle labels, including coincident positions, using all three distances.`
3. `Two-particle scattering is unchanged, but bounded strength does not cure continuum collapse.`
4. `Multiplier convergence is not convergence of the tuned many-body dynamics.`
5. `Exact geometry checks are not empirical validation or a theory of everything.`

Independent tests group positions into occupations and use(3), with independent
squared-distance arithmetic and literal table targets, rather than production
label-triple enumeration as their oracle. One fixed reversal/translation may be
checked per fixture; no factorial sweep. Caps6labels,15pair distances,20triples
are per configuration; no prior large block is constructed. Tests mock CLI
report generation; actual text/JSON modes run separately once each.

## 6. Sources, preservation and status

Protected references:
- `doc/derivations/cubic_three_body_stability_2026-09-13.md:29-93,175-230`: complete
  sectors, onsite bound, fixed-number witness and its prior scope exclusions.
- `doc/derivations/cubic_two_body_scattering_2026-09-13.md:218-284`: retained pair
  tuning, supplied scattering length and two-body-only scattering scope.
- `doc/derivations/supplied_cubic_bose_2026-09-13.md:33-77`: direct sums and hopping.
- `scripts/check_cubic_three_body_stability.py` and its tests: exactFraction
  serialization, independent bounded oracles and strict isolated CLI patterns.

The transferred witness in(8), finite-range counting and multiplier statement
are proved here with explicit new hypotheses, not claimed as prior finite-range
results. No scientific novelty, empirical validation, matter/gravity completion
or TOE claim follows. No old code or claim is modified.

Baseline668tracked/nonignored files is saved externally at
`/Volumes/T9 Backup/bpr-verification/cubic-finite-range-three-body-m1oy57gf/baseline.json`.
Inventory is git ls-files --cached --others --exclude-standard; ignored caches,
config/build products and .git are outside the preservation claim. Only this
note and the new finite-range checker/test may be added.

## 7. Software review and bounded verification

Independent mathematical reviewer a631ebacbed17ae48 cleared the conditional
proof and all13geometry targets. Test author a3d395499ef63807f wrote20methods
from the frozen contract without inspecting production. Static reviewer
af068f77871b55405 cleared checker, independent occupation oracles and runner
with no blockers using direct reads only. No implementation/test repair was
needed; a stale closing review-status sentence was corrected before execution.

The single planned run is new exact-pattern unittest discovery120s, text30s and
JSON30s, sequentially in fresh directories. Child wait budgets exclude possible
cleanup overhead. All three pre-run files are preserved byte-for-byte with hashes.
Parent and children use the same explicit CommandLineTools Python path; saved
sys.version identifies the parent, not an independent child runtime probe.
Bytecode is disabled, PYTHONPATH removed, warnings are errors and six thread
variables are1. The runner stops on failure, timeout, observed interruption,
residual process group or changed inventory. Abrupt uncatchable parent termination
may prevent cleanup or snapshots; missing completion cannot earn success.
No old suites, large-block computation or supplemental science is authorized.
### Actual bounded execution

Artifacts are saved in `cubic-finite-range-three-body-m1oy57gf/first-bounded-run/`
under the external verification directory. First and only attempt:

| Process | Saved result | Whole child seconds |
| --- | --- | --- |
| New exact-pattern unittest discovery | 20 methods passed; unittest time1.467s | 9.058520166 |
| Default text CLI | exit0 | 0.646952083 |
| Strict JSON CLI | exit0 | 1.803125125 |

No saved outcome reports timeout, observed interruption or residual process group.
All671bound file hashes match before and after each child, including668protected
prior files. Script/tests remain frozen; only this note's status and results are
updated afterward. The thirteen triple counts match0,0,0,0,1,1,1,0,1,0,0,20,2;
near distinct-site triples are detected, boundary equality is included, and the
overlong diagonal/anchor traps are rejected. Exact arithmetic checks validate
counting semantics, not successful stabilization. No old suite, large block or
continuum simulation ran. Final independent saved-evidence audit remains pending.

## 8. Candidate decision

Fixed finite physical range corrects the onsite penalty's failure to detect
nearby particles on different sites, and leaves two-particle scattering exactly
unchanged. But at any fixed finite strength it cannot offset the existing
fixed-number negative energy diverging as a^(-2). It therefore fails the stated
cutoff-uniform stability requirement. The potential's limited strong multiplier
convergence does not supply stable tuned many-body continuum dynamics. This is
a rejection of this bounded-strength candidate, not a software-test failure or
a no-go for all finite-range/singular interactions. No next model, growing
strength, hard-core constraint, physical calibration or publication is started.
