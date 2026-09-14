# Deductive compatibility of the current foundation assumptions

2026-09-13. **Independent mathematical review and bounded software verification complete.** This note and its [structural record](foundation_compatibility_2026-09-13.json) add no physical construction, numerical evaluation or empirical validation. Reused results retain their explicit source review provenance; that review does not transfer automatically to the new deductions. No novelty or minimality claim is made.

## 1. Scope, sources and logical bookkeeping

The question is which *specified conjunctions* of assumptions and targets can hold, not which theory is uniquely correct. A failed conjunction is an `excluded_under_premises` result only in its declared domain. A missing operator dictionary, state, limit or action is an open construction obligation, not the negation of its possible existence.

The unchanged microscopic reference is the unrestricted Bose ring on `ell²(N_0)^(tensor L)`, `L >= 3`, with

`H = (g/2) sum_x n_x(n_x-1) - C sum_x (a†_(x+1) a_x + a†_x a_(x+1))`,

where `C > 0`, `g >= 0`, and indices are cyclic. For infinite-space arguments H means its self-adjoint direct sum of complete fixed-total-number blocks. Its finite occupation span D is a core. Finite-sector calculations below are algebra, not a replacement of the unrestricted site factors by occupation cutoffs.

### Source register

The JSON source IDs refer to the following repository files and textual locators. Locators express provenance and are not machine-checked proof references.

| ID | Source and locator | What is reused |
|---|---|---|
| `s_local` | [substrate_local_symmetry_2026-09-13.md](substrate_local_symmetry_2026-09-13.md), Theorem; Domain-safe proof; Self-adjoint realization; Counterexamples fix the scope | Independently derived and adversarially reviewed bounded interval theorem, including the weak-domain formulation and exceptions |
| `s_graph` | [substrate_locality_dimension_2026-09-13.md](substrate_locality_dimension_2026-09-13.md), Theorem and proof; What the result does not establish | Independently reviewed graph-ball inequality, actual-ball and finite-boundary qualifications, nonsufficiency examples |
| `s_projection` | [substrate_constraint_algebra_2026-09-12.md](substrate_constraint_algebra_2026-09-12.md), Model and scope; Reviewed identities and predetermined witnesses | Independently reviewed compression multiplication identity for exact isometries |
| `s_encoding` | [substrate_gauge_encoding_2026-09-13.md](substrate_gauge_encoding_2026-09-13.md), Independently cross-checked mathematical results; Observed bounded result | Reviewed Hamiltonian defect, leakage versus fixed-target mismatch, phase convention and Duhamel bounds in the specified existing encoding |
| `s_anomaly` | [chiral_parent_anomaly_2026-09-12.md](chiral_parent_anomaly_2026-09-12.md), Frozen convention and scope; Derived formulas for independent verification; Status | Independently reviewed nonzero irreducible local gravitational anomaly of the supplied lone parent |
| `s_observation` | [foundation_observation_gate_2026-09-13.md](foundation_observation_gate_2026-09-13.md), Conventional alternative and indistinguishability; Decision and next prerequisite | Independently reviewed identical-distribution nondiscrimination, not experimental evidence |
| `s_claims` | [foundation_claims_2026-09-13.json](foundation_claims_2026-09-13.json), claims `family_count`, `gravity_sources`, `gravity_normalization`, `sampled_numerics` | Input/target and provenance distinctions, preserved limitations |
| `s_decision` | [foundation_decision_2026-09-13.md](foundation_decision_2026-09-13.md), Decision; Dependency decisions; Optional proposal-development direction; Completion requirements | Scoped common-foundation stop condition and explicit missing constructions, not a universal impossibility theorem |
| `s_campaign` | [foundation_prerequisites_2026-09-13.md](foundation_prerequisites_2026-09-13.md), First focused verification evidence; Selected-regression interruption | Latest execution record; earlier preregistration headers are not current execution status |
| `s_note` | This file, sections 2 through 8 | Independently reviewed deductions and conditional conflict sets; open obligations and structural contract |

The prior record's `depends_on` entries are contextual dependencies. They are not copied as logical implication edges. The new `arguments` record only conditional mathematical implications with stated domains. Targets are desiderata, not assertions that constructions already exist. Where a conflict argument includes a target, its conclusion is that the **whole specified set has no simultaneous realization**, not that the target alone is false.

The retained three-family value is a supplied empirical input, not a topological prediction or an exact universal family-count theorem; the source record explicitly limits the LEP interpretation. A viable claimed physical model would also need a relativistic regime for observables over independently specified domains and uncertainties where evidence requires it. This is a conditional physical target here, not an assumption of fundamental Lorentz invariance. No dataset, quantitative relativistic tolerance or observation map is supplied by this note, and neither target is used as an unstated premise in the proofs.

## 2. Proof A: exact integer spatial encoding costs

**Status: deductions independently reviewed.** The starting inequality and its geometric proof are reused from `s_graph`.

Let G be a nonempty unweighted graph and `f: V(G) -> V(C_L)` a site assignment, with `L >= 3`. Each edge has target graph distance at most `K >= 0`; put `k = floor(K)`, an integer. Each target site has at most `m >= 1` preimages, with m an integer. Let `r >= 0` be an integer and let `n = |B_G(v,r)| >= 1` be the size of the **actual** source ball. These assumptions ensure it is finite. Edge distances are integers, so each is at most k. The triangle inequality puts the image in the target radius-kr ball, whose size is `min(L, 2kr+1)`. Therefore

`n <= m min(L, 2kr+1)`.                                                   (A1)

### A.1 Exact rounding and boundary cases

Write `b = min(L, 2kr+1)`, a positive integer. For integer m, (A1) is equivalent to

`m >= ceil(n / b)`.                                                       (A2)

For `r >= 1`, distribute the minimum and then use integrality:

`n <= m min(L,2kr+1)`

`iff n <= mL and n <= m(2kr+1)`

`iff n <= mL and ceil(n/m) <= 2kr+1`

`iff n <= mL and k >= max(0, ceil((ceil(n/m)-1)/(2r)))`.                   (A3)

The maximum retains the declared nonnegative domain of k; for n positive it is harmless. Conversely these last two inequalities recover both terms of (A1), proving equivalence **to the counting inequality**, not to map existence. No real-valued relaxation replaces either ceiling.

At `r = 0`, (A1) is `n <= m`, independent of k. For an ordinary graph ball n is exactly 1, so this is automatic. Division by r is not used. At `k = 0`, (A1) is also `n <= m` for every r. On a connected source graph, all vertices have the same image and the whole graph has at most m sites. At cycle saturation `2kr+1 >= L`, equivalently `kr >= floor(L/2)`, the target ball is the entire cycle and (A1) reduces to `n <= mL`; the capacity condition in (A3) cannot be dropped by increasing k.

Finite boxes and boundary-truncated balls use their actual n. An interior infinite-lattice formula can be substituted only when the ball fits within the specified interior margin. A single fixed finite box supplies no cubic-growth family at arbitrarily large radius.

### A.2 Quantified family tradeoff

Consider a family indexed by j with actual radii `r_j >= 1` tending to infinity, graphs, maps, and parameters `n_j,m_j,k_j,L_j` satisfying (A1). Suppose **for every member** `n_j >= c r_j^3` for one constant `c > 0` independent of j. Then each member must satisfy

`m_j (2k_j + 1/r_j) >= c r_j^2`,     `m_j L_j >= c r_j^3`.                (A4)

The first follows by using `min(L_j,2k_j r_j+1) <= 2k_j r_j+1` and dividing by r_j; the second is global capacity on that ball. Equivalently,

`m_j k_j >= (c/2) r_j^2 - m_j/(2r_j)`.

If `m_j <= M` uniformly for a fixed positive integer M, (A3) gives the integer necessary bound

`k_j >= max(0, ceil((ceil(c r_j^3/M)-1)/(2r_j)))`,

and in particular

`k_j >= (c/(2M)) r_j^2 - 1/(2r_j)`,     `L_j >= ceil(c r_j^3/M)`.          (A5)

Thus effective integer range must grow at least quadratically; K is no smaller than k. If instead `k_j = k_0 > 0` is fixed, then

`m_j >= ceil(c r_j^3/(2k_0 r_j+1)) >= c r_j^2/(2k_0+1)`                  (A6)

for `r_j >= 1`. A fixed upper bound `k_j <= k_0` gives the same necessary lower bounds. Zero range is stronger:

`k_j = 0 implies m_j >= ceil(c r_j^3)`.                                  (A7)

All these bounds coexist with capacity `m_j L_j >= c r_j^3`. For fixed L, that capacity alone requires cubic multiplicity, even when the range bound suggests only quadratic growth. No sufficiency assertion follows in any case.

In particular, uniform finite `m_j <= M` and `k_j <= k_0` contradict the cubic family: dividing `c r_j^3 <= M(2k_0 r_j+1)` by `r_j^3` gives `c <= 2Mk_0/r_j^2 + M/r_j^3`, whose right side tends to zero. This excludes the conjunction of graph-site maps, uniform bounds and this actual-ball family, irrespective of the chosen `L_j`.

### A.3 Counting is not construction

Keep the reviewed triangle-to-square counterexample. With source `C_3`, target `C_4`, `m=k=1`, r=0 has count 1 and every r>=1 has source count 3, which is within the target bound (3 at r=1, 4 thereafter). All ball bounds pass. Yet multiplicity one requires an injective assignment, and every source pair is an edge. Three pairwise adjacent distinct vertices do not exist in `C_4`. No such edge-preserving map exists.

These are graph-site encoding costs. A bound on preimage **sites** is not an information-capacity bound on an infinite-dimensional Bose site. Abstract Hilbert-space packing need not preserve locality, energy or an observable dictionary; neither direction of that distinction constructs the desired physical map.

## 3. Proof B: strongly commuting interval-supported generators

**Status: extension independently reviewed.** Use the original H, D and unrestricted tensor factors in section 1. Let S be a nonempty cyclic contiguous interval with `|S| <= L-2`. Let `G = g_S tensor I_(S^c)` be self-adjoint, with the usual self-adjoint tensor realization. Assume, explicitly,

`exp(itG) exp(-isH) = exp(-isH) exp(itG)` for every real s,t.              (B1)

No formal unbounded commutator is substituted for (B1).

For each t, functional calculus gives `U(t)=exp(itG)=exp(itg_S) tensor I`, a bounded, interval-supported unitary of norm 1. Fix t. For any `psi in Dom(H)`, (B1) shows that the orbit `s -> exp(-isH)U(t)psi` has a strong derivative at zero, because it equals `U(t)exp(-isH)psi`. The generator domain criterion therefore gives `U(t)psi in Dom(H)` and `HU(t)psi=U(t)Hpsi`. Consequently, for `phi,psi in D`,

`<H phi,U(t)psi> = <phi,HU(t)psi> = <phi,U(t)H psi>`.

Thus every U(t) satisfies precisely the bounded theorem's weak conservation hypothesis. That independently reviewed theorem (`s_local`, Theorem and Domain-safe proof) yields

`U(t) = c(t) I`,     `|c(t)|=1`.                                         (B2)

### B.1 Continuous character proof

Choose a unit vector psi. Then `c(t)=<psi,U(t)psi>` is continuous by strong continuity; `c(0)=1` and the group law gives `c(t+u)=c(t)c(u)`. For completeness, continuity permits an interval `(-delta,delta)` on which the unique continuous argument theta obeys `c(t)=exp(i theta(t))` and `|theta(t)| < pi/4`, with `theta(0)=0`. If x,y,x+y are in this interval, their group-law difference `theta(x+y)-theta(x)-theta(y)` is an integer multiple of `2pi` with absolute value less than `3pi/4`, hence zero. Theta is locally additive and odd.

Set `a=delta/2`. Successive partial sums of a/n remain inside the interval, so `theta(a/n)=theta(a)/n`. For every rational p/n with `|pa/n|<delta`, successive positive or negative partial sums similarly give `theta(pa/n)=(p/n)theta(a)`. Rational density and continuity imply `theta(t)=alpha t` throughout the interval, with real `alpha=theta(a)/a`. For arbitrary real t choose an integer n with `|t/n|<delta`. Then

`c(t)=c(t/n)^n=exp(i alpha t)`.

The strongly continuous groups generated by G and by `alpha I` coincide for all real t. Uniqueness of the self-adjoint generator gives

`G = alpha I`, including equality of domains.                            (B3)

An initially possibly unbounded G is therefore forced to be this bounded scalar operator under the full hypotheses.

### B.2 Scope of the obstruction

If the desired microscopic Gauss generator must be nonscalar on the full unrestricted Hilbert space, these hypotheses are incompatible. Scalar phase groups do not furnish that nonscalar action. This does not rule out redundancy acting trivially on physical states, constrained physical subspaces, approximate conservation, low-energy limits, extended or dressed support. Their relation to physical gauge redundancy needs a different dictionary.

The existing support exceptions remain: full-support functions of total number; the one-exterior-site reflection; disconnected opposite-site SWAP on the four-site ring. Zero hopping also lies outside `C>0`. None can silently be included in the short contiguous interval theorem. Strong commutation was assumed; it was not inferred from an unspecified equation `[G,H]=0` on a common set.

## 4. Proof C: exact compression, dynamics and finite Duhamel identity

**Status: reused multiplication identity reviewed at `s_projection`; the general synthesis below is independently reviewed.** For a Bose application the ambient space is one finite **complete** number sector or a finite direct sum of complete sectors of the original ring. The proof is elementary finite-dimensional operator algebra and also holds abstractly in finite Hilbert spaces. The abstract counterexamples below are not additional ring models or proposed physical encodings. No infinite/unbounded extension is asserted.

Let the nonzero code space E and ambient space F be finite-dimensional, `V:E -> F` an exact isometry, `V†V=I_E`, `P=VV†`, `Q=I_F-P`. For ambient operators A,B write `A_c=V†AV`, `B_c=V†BV`. All norms in this section are operator norms. Let H be self-adjoint and set `H_c=V†HV`, `R=QHV`. A supplied code target h is self-adjoint and uses the same time and phase convention as H. It is not selected by fitting H_c.

### C.1 Products and commutators

Insert `I=P+Q` between A and B:

`V†ABV = V†APBV + V†AQBV = A_c B_c + V†AQBV`.

Hence the reused multiplication identity is

`V†ABV - A_c B_c = V†AQBV`.                                             (C1)

With `A=B=H`, self-adjointness and `Q²=Q` give

`V†H²V - H_c² = V†HQHV = R†R`.                                          (C2)

In particular zero Hamiltonian multiplication defect is equivalent to `R=0`, since `||Rx||²=<x,R†Rx>` for every code vector x. Subtract the two ordered versions of (C1) to obtain

`V†[H,A]V = [H_c,A_c] + V†H Q A V - V†A Q H V`

`             = [H_c,A_c] + R† Q A V - V†A Q R`.                         (C3)

Here `R†Q=R†` and `QR=R`; A need not be self-adjoint. The signs are fixed by the convention `[H,A]=HA-AH`. Compression is not generally multiplicative, and a compressed commutator alone discards these terms.

### C.2 Leakage versus target mismatch

Let `D_h=H_c-h` and `F_h=HV-Vh` (the subscript avoids confusing this residual with the ambient space F). Decompose HV into its orthogonal P and Q parts:

`F_h = R + V D_h`,

`F_h† F_h = R†R + D_h²`.                                                (C4)

Indeed `V†R=0`, `R†V=0`, and D_h is self-adjoint, so both cross terms vanish. No commutation between H_c and h is required. Leakage and compressed mismatch occupy orthogonal ambient subspaces and cannot cancel in F_h. In particular `F_h=0` iff `R=0` and `H_c=h`.

### C.3 Finite Duhamel identity, bound and exact criterion

For fixed real t define `W_t(s)=exp(-i(t-s)H) V exp(-is h)`. In finite dimensions differentiation is unrestricted, and

`dW_t(s)/ds = i exp(-i(t-s)H) (HV-Vh) exp(-is h)`.

Integrating from 0 to t and using the endpoints yields the exact identity

`exp(-itH)V - V exp(-ith)`

`    = -i integral_0^t exp(-i(t-s)H) F_h exp(-is h) ds`.                 (C5)

It holds for negative t with the oriented integral as written. Both exponentials are unitary; taking norms bounds the integral by `|t| ||F_h||`. Independently each endpoint operator is an isometry of norm 1, so the triangle inequality gives

`||exp(-itH)V - V exp(-ith)|| <= min(2, |t| ||F_h||)`.                    (C6)

If F_h=0, (C5) proves exact intertwining for every real t. Conversely, exact intertwining for all t can be differentiated at t=0, giving `-iHV=-iVh`. Together with (C4),

`[for all real t, exp(-itH)V = V exp(-ith)]`

`iff F_h=0 iff [R=0 and H_c=h]`.                                        (C7)

Thus any **specified** finite candidate with `R != 0` or `H_c != h` fails this exact all-time target. This is not an inference that every future candidate has either defect. A zero bound at t=0 or equality at isolated times is not the all-time property. A small compressed commutator is not a bound on F_h. The residual-based bound is sufficient, not necessary, for approximate evolution control: small finite-time evolution error need not imply a small generator residual. Phase shifts have to be declared consistently; (C7) is exact vector intertwining, not equivalence only up to a phase or after forgetting observables.

### C.4 Explicit abstract algebra counterexamples, not new ring models

These matrices are elementary logical counterexamples to invalid implications. They are not Hamiltonians proposed for BPR, not modified microscopic ring dynamics, and not numerical controls evaluated in this work.

1. **Exact compressed algebra without invariance.** Take ambient `C²`, code `C`, `V z=(z,0)`, `H=[[0,1],[1,0]]`, and `h=0`. The ambient scalar algebra `{a I_2}` compresses exactly and multiplicatively to `{a I_1}`, its whole target scalar algebra. Every compressed observable commutes with `H_c=0`. Nevertheless `R z=(0,z)`, `||R||=1`, and `V†H²V-H_c²=1`. Explicitly `exp(-itH)V z=(cos(t)z,-i sin(t)z)`, not Vz for all t. Even `[H_c,H_c]=0` misses the nonzero square defect. Correct compressed algebra and commutators do not certify invariance.
2. **Invariance without the supplied dynamics, not merely a phase discrepancy.** Take ambient `C³`, code `C²`, `V(z_1,z_2)=(z_1,z_2,0)`, `H=0_3`, and supplied `h=diag(0,1)`. Embed any code matrix A as `diag(A,0)` in the ambient space. This supported matrix algebra compresses exactly to all of `M_2(C)` and respects products and commutators. The ambient scalar identity can also be included without changing that fact. Here `R=0` and `H_c=0_2`, but `F_h=-Vh != 0`. Actual evolution is V, whereas target evolution is `V diag(1,exp(-it))`; a superposition of the two code basis vectors develops a target relative phase absent from the actual evolution. On the diagonal observable subalgebra both generators even have zero commutators, yet the full target dynamics differs. Neither algebra closure nor invariance fixes a supplied target generator.

A physical encoding would require its own state-independent operator dictionary and dynamical/error analysis. No abstract example above supplies one. Infinite sectors, unbounded observables, approximate isometries or limiting effective generators need separate hypotheses and domain or error estimates; the finite identities are not a license to omit them.

## 5. Compatibility table: conflicts versus missing construction

Every conflict below is scoped to all of its named premises. No premise set is certified minimal. Removing a premise only removes this argument; it is not a witness that the remaining assumptions are jointly satisfiable.

| Set and record IDs | Explicit conjunction or conditional input | Conclusion, provenance and status |
|---|---|---|
| Local generator: `p_ring`, `p_local_bridge`, `p_local_nontrivial`; result `p_local_scalar` | Original unrestricted self-adjoint ring with C>0; self-adjoint interval G with nonempty contiguous support of size at most L-2; strong commutation for every s,t; desired G nonscalar on full Fock space | `excluded_under_premises`: section 3 forces scalar G. Extension/conflict independently reviewed; bounded source theorem independently reviewed |
| Spatial map family: `p_graph_maps`, `p_graph_uniform`, `p_graph_cubic`; result `p_integer_costs` | For every family member an actual graph-site map with the stated edge/fiber bounds; radii tending to infinity with one positive cubic-growth constant; uniform finite multiplicity and range bounds | `excluded_under_premises`: section 2 contradicts cubic versus linear ball capacity. Integer tradeoff/conflict independently reviewed; source counting theorem independently reviewed |
| Exact finite target: `p_finite_setup`; result `p_encoded_dynamics` | A specified exact finite isometry V, self-adjoint H,h in matching conventions, and a demonstrated nonzero R or H_c-h, together with a demand for all-time intertwining | `excluded_under_premises` **if the defect premise is established for that candidate**: section 4. No new candidate or defect is supplied here. The old occupation-orbit failure remains specific to `s_encoding` |
| Supplied chiral parent: `p_parent_content`, `p_parent_target`; result `p_parent_anomaly` | One complex six-dimensional Weyl Spin(10) 16 of chirality s=+1 or -1, no anomaly-cancelling completion, the cited anomaly convention; target is vanishing full local anomaly polynomial | `excluded_under_premises`: irreducible p2 coefficient is -s/90, nonzero for either chirality. Known source result reviewed; this record's conflict synthesis independently reviewed. Restricted reduced cancellation does not cancel the parent polynomial |
| Distinguishing test: `p_equal_distributions`, `p_observation_target`; result `p_nondiscrimination` | Paired descriptions with identical joint observation distributions for every allowed design, including shared nuisance/noise conventions; requested statistical test with different rejection probabilities for some allowed design | `excluded_under_premises`: for any measurable rejection rule `0<=phi<=1`, its expectations under equal probability measures are equal. Source nondiscrimination reviewed; conflict synthesis independently reviewed. Additional frequencies/precision within the same equality premise do not help |
| Common physical construction: `p_common_obligation`, `p_relativistic_target` | Physical fermion/gauge dictionary, state/limit, metric dynamics, gravitational normalization and calibrated observation map have not been supplied as a single common construction in the reviewed sources | **Missing construction**, not `excluded_under_premises` for all possible models. `s_decision` and `s_claims` record why existing source identities, internal mode assignments and supplied actions do not complete it |

For the anomaly row, vanishing of the local anomaly polynomial is the specified consistency target, not a proof of complete quantum consistency. Cancellation, if separately proposed, would still require checking the full content, background and global issues. No cancellation fields, inflow or geometry are silently added here.

For the observation row, the ordinary Bose alternative has the same H, Hilbert space, state/preparation, operators and observation map in `s_observation`. Equal distributions give `integral phi dP = integral phi dQ` for every such rule. This is a known mathematical restriction, not a new acquisition or a claim that all future physical extensions have identical predictions. Unknown calibration and missing data are additional obligations, not its proof.

## 6. Conditional routes and construction obligations

Only three named routes are recorded. None is an adopted model or a demonstrated common solution. Other alternatives remain unassessed; the set is not exhaustive or ranked.

- **Relax uniform graph costs** (`r_graph_costs`): changes `p_graph_uniform`, not the physical demand for actual cubic neighborhoods. This removes the uniform-bound premise from `a_graph_conflict`. Obligation `p_graph_obligation`: supply the actual graph-site/observable map, its fibers and range as functions of scale, a state and limit, the physical locality/energy interpretation, dynamics and controlled errors. Meet both tradeoff and capacity; passing them is insufficient for a map. No new three-dimensional graph is adopted.
- **Constrained or approximate dynamics** (`r_effective_dynamics`): changes `p_local_bridge` by replacing full unrestricted-space strong conservation with an explicitly specified constrained, low-energy or approximate condition. It affects `a_local_scalar` and `a_local_conflict`; it does not invalidate their original statements. Obligation `p_effective_obligation`: give constraints, physical Hilbert space, state/limit, support and observable dictionary, the target generator, an exact intertwiner or a controlled error on a stated time/energy regime, and measurement meaning. Where the exact finite hypotheses still hold, section 4 remains an obligation, not something evaded by a label.
- **Extended or dressed support** (`r_dressed_support`): changes `p_local_bridge` by dropping the short contiguous interval hypothesis. It affects the same two local arguments, not the graph or anomaly restrictions. Obligation `p_dressed_obligation`: specify actual supported/dressed operators and their algebra, physical states and limiting regime, induced locality, target dynamics and quantitative error control. Existing long/disconnected-support exceptions alone supply no local gauge theory.

The cross-cutting `p_common_obligation` additionally requires physical fermion spin/statistics and anomaly consistency, a common dynamical action and metric/source dictionary rather than external-source Ward identities alone, independently accounted scales and normalizations, a justified relativistic regime where needed, and a genuinely distinguishing calibrated observation protocol. These are requirements for a proposed claim, not conclusions that a construction satisfying them exists. No route is credited with solving another row by omission.

## 7. Frozen structural record and checker API contract

This section is the exact interface for the checker author. The checker and tests are separate deliverables, **not implemented or executed as part of this two-document authoring task**.

### API and command line

- `load_record(path) -> JSON dict`: reject duplicate object keys at every nesting depth and all nonfinite constants with `ValueError`, including a syntactically finite numeric literal whose float conversion overflows to infinity. The JSON root must be a built-in dict. Malformed JSON is a `ValueError` (including the standard JSON decoding subclass); file I/O failures may remain I/O errors at this loading boundary and must be handled by the CLI.
- `validate_record(record, root) -> None` on success or `ValueError` on every validation failure. Validation errors must be deterministic; avoid unordered set rendering or nondeterministic traversal when constructing diagnostics. Structural type/path/reference errors must not escape as `TypeError`, `KeyError` or other incidental exceptions. The validator does not mutate the record or import scientific modules.
- `main(argv=None) -> exit int`. The default script root is exactly `Path(__file__).resolve().parents[1]`. Accept one optional positional record path; its default is `root / 'doc/derivations/foundation_compatibility_2026-09-13.json'`. Root and the default record are independent of the caller's working directory; an explicitly supplied relative record path follows normal CLI working-directory resolution. No `bpr` imports.
- CLI validation, I/O and JSON errors print to **stdout** and return exit 1. Explicit exception: argparse's normal unknown-argument errors may use stderr and exit 2. Success prints exactly `structurally valid record; mathematical and physical validity not checked.` followed by one newline, and returns 0. No scientific computation, writes or proof-status promotion occurs.

### Exact schema

All input mappings and lists must be exact built-in `dict` and `list` objects, not subclasses or alternative containers. Field sets are exact, with no extras. String fields must be strings and nonempty after a trim check; a trim check is not permission to silently rewrite values or IDs. Reference lists contain no duplicate IDs. IDs are globally unique across **all four** source, proposition, argument and route sections, not merely within each section. `sources`, `propositions` and `arguments` are nonempty lists; `routes` is a list allowed to be empty. Every reference resolves in its designated section.

| Object | Exact fields and constraints |
|---|---|
| Root | `schema_version`, `validation_scope`, `physics_validated`, `empirical_validation`, `sources`, `propositions`, `arguments`, `routes`. `schema_version` is exact built-in int 1, never bool. `validation_scope` is `record_structure_only`. Both flags are exact bool false |
| Source | `id`, `path`, `locator`, all nonempty strings. Path is POSIX repository-relative: reject absolute paths, every `..` component and any backslash. Resolve against root; require an existing regular file whose resolved path remains within resolved root. A symlink is allowed only when its resolved destination is contained. Directories, missing files, traversal and symlink escapes fail |
| Proposition | `id`, `kind`, `statement`, `domain`, `quantifiers`, `sources`, `status`. Except the `sources` list these are nonempty strings. `kind` is one of `microscopic_assumption`, `empirical_input`, `physical_target`, `bridge_assumption`, `conditional_result`, `diagnostic`, `obligation`. `sources` is a nonempty duplicate-free list of source IDs. `status` is one of `assumed`, `supplied_input`, `reviewed_result`, `open` |
| Argument | `id`, `kind`, `premises`, `conclusion`, `proof`, `review_status`, `minimality`. `kind` is `deduction` or `claimed_incompatibility`. `premises` is a nonempty duplicate-free list of proposition IDs. `conclusion` is a proposition ID. `proof` is a built-in dict with exactly `source` (a source ID) and `locator` (a nonempty string). `review_status` is `unreviewed` or `independently_reviewed`. `minimality` is exactly `not_claimed` |
| Route | `id`, `changed_premises`, `affected_arguments`, `obligations`, `assessment`, `explanation`. The first three fields after `id` are nonempty duplicate-free reference lists: respectively proposition IDs, argument IDs, proposition IDs. Every `obligations` referent must have kind `obligation`. `assessment` is `conditional_escape_obligations` or `unassessed`; `explanation` is a nonempty string |

For every argument, form directed edges from each premise proposition to its conclusion proposition. The resulting graph must be acyclic, including rejection of self-cycles. This is structural dependency checking, not verification that the edge is a valid implication. List order is documentary, not a required topological order.

There is **no automatic kind/status semantic proof validation**. The checker does not authenticate a `reviewed_result` or `independently_reviewed` label; this note is responsible for explaining their provenance. The three reused source results retain their review provenance. The seven new result propositions and corresponding arguments now carry reviewed labels following the separate mathematical review recorded in section 8. No target, obligation or route is promoted. `open` for a target or obligation means not supplied/proved, not false.

Source locators are not machine checked. A false or unsupported mathematical assertion can pass structural validation if its fields and references are well formed. No solver, expression parser, status authentication, source-entailment test, physical-compatibility score or empirical check is included. Successful validation must never be reported as mathematical or physical validity.

## 8. Review and verification record

### Protected baseline and authoring boundary

The coordinator supplied an external baseline of **651 pre-existing tracked and nonignored untracked files**, enumerated with `git ls-files --cached --others --exclude-standard`. Ignored caches, build products and local configuration are outside this inventory; it is not a full-filesystem preservation audit. The baseline is at

`/Volumes/T9 Backup/bpr-verification/deductive-compatibility-lnodzv9r/pre-existing-hashes.json`.

The coordinator verified all four planned destinations absent before authoring. The documentation author created this note and its JSON record without execution; the coordinator implemented the checker and a separate author supplied its tests. The completed tranche adds only these four files. No old source, test, instruction, ledger, demo, benchmark or regression artifact was edited. All 651 pre-existing bindings matched before execution and through each post-run snapshot. The four additions were also unchanged throughout execution (655 bindings total). This note's final execution-status update occurs after that frozen run; the reviewed JSON, checker and tests remain unchanged.

### Preserved earlier execution status

The current authoritative campaign record is `s_campaign`, especially Selected-regression interruption, not its older planning headers. Earlier separate focused runs passed **617 tests** in total: provenance 16, locality 366, symmetry 235. Eight named algebra-check groups passed separately. These results do not belong to the present note or count as proof checks for its extensions.

The later selected regression verified **1,742 completed passed nodes in groups 001 through 030**. The harness stopped group 031 for reported low memory; logs alone do not establish that cause. `test_quantum_report_frozen_counts_and_strict_json[5-0.7]` reached setup only and has no completed-node credit. Outcome records show signal-15 cancellation and process-group quiescence; a later process snapshot found no surviving verification Python processes. The 30 completed groups recorded no assertion failures or skips.

All 64 saved integrity snapshots matched and all 651 bound files matched at the independent interruption audit. Group 031's posthash and the run's final hash/completion records are absent. **Full regression and isolated demos remain incomplete. Neither new scientific suite nor demo was reached in that regression.** Collection of 5,572 nodes is not execution. No old regression, demo or supplemental probe is restarted here. The previous four tiny-noise nonzero/sign failures and older one-ULP repeatability failure remain preserved and unresolved by this work.

### Completed review and bounded execution

1. Mathematical gate completed: independent read-only reviewer `a553220594d07e392` found no confirmed mathematical or premise-to-conclusion defect in the three proof targets and all ten JSON arguments. Review checked rounding, growth quantifiers, strong commutation and domains, the character proof, Duhamel signs/negative times, counterexamples, exclusions and conditional routes. Seven new result/argument pairs were promoted; targets, obligations, routes and empirical flags were not. This review did not authenticate historical execution artifacts. Initial unreviewed drafts are preserved externally as `draft-v1-*`.
2. Checker and independently authored focused tests passed separate static review by `af3d872aa6d314aba`; the external bounded runner also cleared that review before execution. The independent test author found an exact-string-type rejection inconsistent with the frozen contract, which requires exact built-in types for dict/list/int/bool but only strings for text fields. Before any execution, `_string` was repaired to accept string instances, including ordinary subclasses; the acceptance test was retained. The original source is preserved as `draft-v1-check_foundation_compatibility.py` and the repair as `pre-execution-repairs.json` externally. Tests cover malformed types, duplicate keys/IDs/references, nonfinite constants and overflow-to-infinity, dangling references, cycles, path/symlink escapes, enum/flag failures and a deliberately unsupported but structurally valid mathematical assertion that **must pass**. Exactly three CLI child invocations executed in the suite, each with a ten-second timeout.
3. The first bounded execution passed **31 new unittest methods in 0.521s** and **16 existing provenance methods in 0.036s**, followed by a successful standalone checker invocation. Subtest cases are not counted as extra test methods. All three processes returned zero, with no timeout or residual process group; child elapsed times were 0.655100s, 0.131969s and 0.077201s respectively. Unittest's normal verbose reports are on stderr, ending in `OK`; suite stdout is empty. Standalone CLI stderr is empty and stdout is exactly `structurally valid record; mathematical and physical validity not checked.` plus newline. No retries occurred.
4. Execution used `/Library/Developer/CommandLineTools/usr/bin/python3`, actual Python **3.9.6**, warnings as errors, bytecode disabled, no PYTHONPATH, and all six numerical thread variables at one. Separate fresh working directories and exact unittest-discovery patterns avoided pytest discovery/scientific imports. Budgets were 120/120/30 seconds. The three in-suite CLI subprocesses also verified empty-working-directory behavior and stdout/error conventions. No Python 3.8 runtime claim is made.
5. External evidence under `/Volumes/T9 Backup/bpr-verification/deductive-compatibility-lnodzv9r/` includes `mathematical-review.json`, `static-review.json`, `runner-review.json`, initial drafts, the repaired pre-execution snapshot (`pre-static-v2-*`), and `first-bounded-run/` with environment, exact launch commands, stdout/stderr, per-child outcomes, pre/post hashes and completion. All 655 bindings matched throughout the run. JSON SHA256: `0c6c22d803dcb272e9a77b59265a6865c62ca4b046e460034959407f7e46ed28`; checker: `5289dcc0f651615666e307bcc6cdfdf6e9303f97f4c6d3b3245750f5bc962d13`; tests: `555107a103eea34f2369cab2d8c975072fe58d7b46a8dcf6b1940578f9db9ff4`. The run's note hash was `e1268618feb44250a9a1f3faab10b43c6ccb58258e6e4d9cade9aa6253e67bda`; only its execution-status prose was updated afterward. The final inventory is recorded externally, not self-hashed inside this note. These are bookkeeping/software checks, not new proof or empirical evidence.

## 9. Plain-English conclusion

The bounded deductions rule out specified combinations: uniformly cheap graph-site encodings of growing cubic neighborhoods; nonscalar, strongly conserved short-interval generators on the original unrestricted ring; and exact fixed-target evolution for any specified finite code that demonstrably leaks or has the wrong compression. The supplied lone chiral parent's anomaly and identical-distribution nondiscrimination remain known restrictions, not new discoveries.

What remains to be supplied is an explicit common operator and dynamics dictionary, physical states and controlled limits, matter and gravity consistency, scale/normalization accounting, and a distinguishable calibrated prediction. No concrete common construction is justified by this note. That missing construction is an honest unresolved outcome, not a universal impossibility proof or a physical solution claim.
