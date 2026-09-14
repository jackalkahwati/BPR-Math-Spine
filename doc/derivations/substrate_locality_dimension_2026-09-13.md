# Conditional locality and dimension: derivation and proposed contract

2026-09-13. Foundation prerequisite package 3. Independent derivation and independent adversarial mathematics review agree on the counting theorem below. Independent API/mathematical review found no blocking changes, including exact input/intermediate domains, box preallocation cap and all fixed demo cases. The contract below is **frozen before implementation**. No controls have executed; reviewer-computed expectations are analytic review, not execution results.

## Theorem and proof

Let `G` be a nonempty unweighted graph, `C_L` the simple cycle with `L>=3`, and `f:V(G)->V(C_L)` a vertex assignment. Assume each source edge maps to target graph distance at most `K=p/q>=0`, and each target vertex has at most `m>=1` preimages. Set `k=floor(K)`. For every source vertex `v` and integer `r>=0`,

`|B_G(v,r)| <= m * min(L, 2*k*r+1)`.

Target distances are integers, so every edge image has length at most `k`. Triangle inequality puts the image of a source radius-r ball within target radius `k*r`. The latter ball has exactly `min(L,2*k*r+1)` vertices. Counting at most m preimages per target vertex proves the result, including target wraparound. Connectivity is not required for this ball statement; for a connected G and `K<1`, all vertices map to one cell and `|V(G)|<=m`. Global capacity `|V(G)|<=mL` also holds independently of edge dilation.

For the nearest-neighbor integer lattice in dimension d,

`V_d(r) = sum_(j=0..min(d,r)) 2^j binomial(d,j) binomial(r,j)`.

Choose j nonzero coordinates, their signs and positive magnitudes with sum at most r. For d=1,2,3 this gives `2r+1`, `2r²+2r+1` and `(4r³+6r²+8r+3)/3` respectively. For open boxes this formula applies only if the entire ball lies inside the box; at other centers/radii count the actual finite box. A source torus has these counts under the sufficient condition `2r+1<=min(side lengths)`; this module does not expose a source-torus API.

A family of actual source balls with radii tending to infinity and size at least `c*r³` for fixed `c>0` cannot satisfy this inequality with uniformly bounded K,m, whatever the target L. The same argument excludes any uniformly superlinear growth. For d>=2 a nonminimal explicit witness radius is `r*=m*(k+1)`: since `V_d>=V_2`,

`V_d(r*) - m*(2*k*r*+1) >= 2*r*(r*-m*k+1)+1-m > 0`.

It requires a source box with interior margin at least r*. This algebraic witness is not a scan or claim about the existing microscopic ring's physically realized dimension.

## What the result does not establish

Bounded graph fibers are an explicit encoding assumption, not a consequence of unrestricted Bose on-site Hilbert spaces. Abstract separable Hilbert-space isomorphisms can pack modes without preserving energy, locality or observables. This theorem excludes only the specified graph-site maps. It proves neither a universal dimensional no-go nor Lorentz symmetry or time emergence.

Counting is necessary, not sufficient for a map. For example, source C3, target C4, k=m=1 satisfies every ball-capacity bound but cannot embed injectively with edges preserved: C4 has no triangle. Likewise a 2x2 open square cannot inject edge-preservingly into C5 even though its ball counts fit. Passing a counting test must never be labeled existence or physical compatibility.

## Frozen minimal implementation contract

Files: `bpr/substrate_locality_dimension.py`, matching `tests/test_substrate_locality_dimension.py`, stdout-only `scripts/demo_substrate_locality_dimension.py`. No NumPy, eigensolver, floating point, dense occupation construction or new physical geometry is needed. Python3.8 grammar. Reports use detached finite JSON-native values.

Pre-execution CLI clarification (inherited campaign convention): absolute script invocation from an empty working directory with no PYTHONPATH prints human-readable text by default; `--json` prints exactly one strict JSON serialization of `demonstration_report()`. Successful invocations exit0 with empty stderr and no files written. Argparse rejects unknown arguments. Text must identify conditional scope and no physical-dimension/empirical derivation; exact prose is not frozen. This clarifies the omitted flag/default detail without changing mathematical controls.

All integers must be **built-in int, excluding bool**. Reject floats, fractions supplied as objects, strings, NumPy scalar types and custom conversion objects without coercion. Validation errors raise `ValueError`. Scalar numeric input ceiling is `MAX_INTEGER=10**6`. Dimension is 1..3; r is 0..MAX_INTEGER; L is 3..MAX_INTEGER; K_num is 0..MAX_INTEGER; K_den and multiplicity are 1..MAX_INTEGER. Intermediate/output integers are exact and are not restricted to the input ceiling. This is an algorithm/resource domain, not a physical bound.

Public functions:

- `lattice_ball_count(d, r) -> int`: exact V_d formula.
- `ring_ball_count(L, r) -> int`: exact `min(L,2*r+1)`.
- `box_ball_count(shape, center, r) -> int`: shape and center must be built-in list/tuple of equal length d=1..3, each side 1..MAX_INTEGER, center integer in 0..side-1. Before enumeration, reject product(shape)>`MAX_BOX_SITES=4096`. Count points satisfying actual Manhattan distance <=r; no infinite-lattice substitution outside the interior regime. No graph allocation or BFS needed in production. Product checking must stop once the cap is exceeded, before Cartesian enumeration.
- `growth_report(d, r, L, K_num=1, K_den=1, multiplicity=1) -> dict`: use the infinite lattice neighborhood explicitly as the source. Keys `source_kind="integer_lattice_ball"`, `dimension`, `radius`, `target_sites`, `dilation={numerator,denominator,effective_integer}`, `multiplicity`, `source_count`, `target_ball_count`, `capacity`, `signed_excess`, `status`, `scope`. Count target ball by exact internal radius `k*r`, which may exceed MAX_INTEGER; do not feed it through a public input validator that would wrongly reject an intermediate. `status="excluded_by_ball_count"` iff signed_excess>0, else `"not_excluded_by_ball_count"`. Equality is not exclusion. Scope explicitly includes `map_existence_established=false`, `physical_dimension_derived=false`, `empirical_validation=false`, and bounded-fiber assumption.
- `demonstration_report() -> dict`: fixed controls below only. Top-level `module`, `growth_cases`, `finite_box_controls`, `limitations`. No arbitrary graph/map search, polynomial fitting, optimized witness search, datasets, caching or files.

No public API evaluates whether a physical operator map exists. Minimal scalar helpers suffice for diagnostics; the reviewed infinite-family theorem remains in this derivation, not a numeric rank label.

## Fixed demonstration and independent tests

Freeze growth cases in order: d=1,2,3 outer, r=0,1,2,3 inner, target L=31,K_num=K_den=m=1. Then append rational/edge controls `(d,r,L,K_num,K_den,m)`:

`(3,2,31,3,2,1)`, `(3,1,31,1,2,1)`, `(1,3,4,1,1,1)`, `(1,2,9,0,1,5)`.

These 16 cases retain excluded and nonexcluded outcomes without favorable replacement. The last case is exact equality capacity. Every report is a counting diagnostic only.

Finite-box records have `shape`, `center`, `radius`, `count`. Fixed shapes/centers `([3,3,3],[1,1,1])`, `([1,1,5],[0,0,2])`, `([2,2],[0,0])`; for each use r=0,1,2,3, hence 12 records. Exact first two count sequences are `(1,7,19,27)` and `(1,3,5,5)`. Keep the boundary-truncated cubic count distinct from infinite `(1,7,25,63)`.

Independent tests must derive expected lattice counts from bounded BFS/enumeration, not reuse the production formula as sole oracle. Validate cycle counts on explicit small odd/even graphs including saturation; actual box BFS handles boundaries and thin boxes. Include graph C3->C4 or square->C5 as necessary-not-sufficient control without scanning embeddings. Check all 16/12 retained demo slots and strict JSON. Public scalar extrema may be checked algebraically without allocating graphs. Oversize box validation must precede enumeration; monkeypatch the allocation/iteration seam if necessary. Test rational K flooring per edge, K0, r0, equality, global capacity limitation, invalid shapes/types, detached reports and input nonmutation.

This module uses exact integers; no heuristic numerical tolerance, precision escalation, platform float skip or error proxy is appropriate. Finite controls check code; they do not replace proof.

## Verification and inherited boundaries

Separate implementation/test authors, then independent static review before any imports or controls. Freeze source/demo/test hashes and a bounded execution manifest. Run focused tests, isolated text/strictJSON demos, eight algebra groups, Python3.8 grammar checks on separately reported actual runtime, and inherited selected 35 suites plus new suites in serialized fresh processes. Protect baseline files except authorized provenance changes. The previous tiny-noise supplemental failure and old one-ULP repeatability limitation remain unchanged. No previous single-attempt probe is retried. Publication is not authorized.
