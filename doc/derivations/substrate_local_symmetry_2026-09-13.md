# Exact bounded local symmetry of the Bose ring

2026-09-13. Foundation prerequisite package 2. An independent derivation and a separate adversarial mathematical review agree on the theorem below. The first focused implementation run passed 235 tests with unchanged bound hashes; full selected regression and isolated demos remain pending. The separately reviewed API/control section below was frozen before implementation. This is not a broader physical no-go theorem.

## Theorem

Let L>=3 and let the site Hilbert space be the unrestricted tensor product of L copies of `ell²(N_0)`. Let D be the finite linear span of occupation vectors, and use inner products conjugate-linear in the first slot. The Hamiltonian is

`H = (g/2) sum_x n_x(n_x-1) - C sum_x (a†_(x+1) a_x + a†_x a_(x+1))`,

with C>0,g>=0, all sites uniform and labels modulo L. Let S be a nonempty cyclic contiguous interval with `|S|<=L-2`. Suppose `B=b tensor I_(S^c)` is bounded, not necessarily Hermitian, and

`<H phi, B psi> = <phi, B H psi>` for all `phi,psi in D`.

Then `B=cI` for a complex scalar c. This is a classification of bounded interval-supported operators commuting with the original Hamiltonian on unrestricted Fock space. It is stronger than the earlier fixed-number occupation-diagonal statement, which alone did not prove it.

## Domain-safe proof

Choose an endpoint s of S and its outside neighbor t. Since the complement is a contiguous interval of at least two sites, t has exactly one neighbor in S. Let Omega be the exterior vacuum and e_t its one-particle state at t. For finite local vectors u,v, insert `phi=u tensor e_t`, `psi=v tensor Omega` in the weak equality. Only the s,t crossing bond changes the exterior vacuum to e_t. Other internal, exterior and boundary contributions vanish by exterior particle-number or occupation orthogonality. Since C is nonzero,

`<a_s† u, b v> = <u, b a_s v>`.

Independently reverse the exterior occupations to obtain

`<a_s u, b v> = <u, b a_s† v>`.

No expression applies an unbounded ladder operator to b v. The second equation is derived separately, not by assuming b is Hermitian.

Split the local space into endpoint occupation and remaining factors. Its bounded operator blocks `b_mn` satisfy

`sqrt(m+1) b_(m+1,n) = sqrt(n) b_(m,n-1)`,

`sqrt(m) b_(m-1,n) = sqrt(n+1) b_(m,n+1)`.

Negative indices mean zero. These identities first hold between finite remaining-factor vectors and extend by boundedness of the blocks. Taking n=0 in the first and m=0 in the second kills all nonzero first-column/first-row blocks. For m,n>=1, the first recurrence reduces both indices by one. Thus `b_mn=delta_mn*b_00`, and `b=I_s tensor b_00`.

B now has a smaller interval support but obeys the same weak relation with the **original L-site Hamiltonian**. Repeat the boundary argument until the support is empty. Never replace the original ring with a smaller ring during this induction.

The proof does not use the sign of g, but the adopted physical model retains g>=0. It includes singleton support, not just intervals of length two or more.

### Self-adjoint realization

H is the self-adjoint direct sum of its finite complete fixed-total-number blocks; D is graph-dense. Boundedness makes both weak-form terms well defined without requiring B(D) subset D. The adjoint-domain criterion implies B psi belongs to Dom(H) and H B psi=B H psi for psi in D. Closedness extends this to Dom(H). This is a consequence, not a hidden initial core-invariance assumption. The boundary proof itself needs only the stated weak form.

A bounded B need not have a bounded commutator with H in general. Finite diagnostics must not report a finite unrestricted commutator norm merely because a cutoff matrix norm is finite.

## Counterexamples fix the scope

- **Zero hopping:** with C=0 any bounded nonconstant onsite number function commutes. For g>0 the equal onsite energies of occupations 0 and 1 also permit bounded mixing within that degenerate pair. This is outside C>0.
- **Full support:** bounded nonconstant functions of total number commute, including total parity. The unrestricted global commutant is not scalar.
- **One exterior site:** proper contiguity alone is insufficient. For exterior vertex t, reflection `x -> 2t-x mod L` fixes that site and acts nontrivially only on the other L-1 sites. Its bounded unitary commutes with uniform H for all g. Boundary extraction in this case isolates a sum of neighbor ladders, not either one separately.
- **Disconnected opposite sites of L=4:** put `a_+=(a_0+a_2)/sqrt(2)`, `a_-=(a_0-a_2)/sqrt(2)` and `N_-=a_-†a_-`. At g=0 every bounded f(N_-) commutes with H because the hopping couples only the bright mode to exterior sites. Moreover `exp(i*pi*N_-)=SWAP_(0,2)` and this operator commutes for every uniform g, as a ring automorphism. Generic f(N_-) is not conserved at g>0: the interaction contains `g/4*(a_+†² a_-²+a_-†² a_+²)`. Within this number-function family, conservation requires f(n+2)=f(n), yielding identity/parity combinations.

For the dark-vacuum projector P, the normalized two-particle witness is

`<0_+,2_-| [H,P] |2_+,0_-> = g/2`.

These exceptions are retained controls, not reasons to broaden the theorem after testing.

## Finite diagnostics are not the theorem

Use `P_<=K = direct_sum_(N=0..K) P_N` if a direct-sum control is needed. Its dimension is `binomial(L+K,K)`, checked against cap512 before any allocation. This is a sum of complete number sectors, not a local tensor occupation cutoff. Since P commutes with H,

`P[H,B]P = [H_<=K, PBP]`

in the weak matrix-element sense, including bounded B that changes total number. A fixed single-number sector misses those cross-number blocks.

A nonzero matrix element certifies nonconservation of the specified operator. A vanishing finite compression never proves conservation of an arbitrary full-Fock operator: the onsite projector onto occupation K+1 is invisible below the cutoff but not conserved under nonzero hopping.

A minimal off-diagonal control is the local bounded flip `X=|0><1|+|1><0|`. With s adjacent to exterior t,

`<0_s,1_t|[H,X_s]|0_s,0_t> = -C`,

with all other sites vacuum, independently of g. This requires total-number 0 and 1 together. At least two particles are required to see the interacting dark-projector counterexample.

Implementation should reuse complete-sector builders and rectangular ladder maps, not solve a full Liouville-space commutant or numerically infer an infinite theorem from a tolerance rank. Exact occupation matrix-entry formulas provide an independent test oracle. Any finite norm is labeled as a finite-compression diagnostic, with raw residuals retained.

## Physical conclusion

The theorem excludes nontrivial exact microscopic conserved bounded operators on the specified short interval supports. Under an application that identifies local gauge transformations with such operators commuting on the full microscopic Hilbert space, it obstructs that route.

It does **not** exclude symmetries acting only in a constrained physical subspace, low-energy projections, approximate dynamics, dressed or nonlocal operators, or all emergent gauge descriptions. Those need their own explicit construction and controlled errors. The result supplies neither physical fermions nor a Standard Model gauge sector.

## Frozen implementation contract

Independent numerical/API review cleared this contract after five pre-execution clarifications covering error translation, live caps, ownership counts, structural validation and witness comparisons. At contract freeze, no theorem-control values had been generated. No previous module is changed and no new Hamiltonian terms are introduced. The contract below was frozen before separate implementation and test authoring; subsequent execution status is recorded above and in the campaign ledger.

New files: `bpr/substrate_local_symmetry.py`, `tests/test_substrate_local_symmetry.py`, and `scripts/demo_substrate_local_symmetry.py`. Reuse `substrate_vacuum_selection.all_number_model` and `capped_binomial`, and `substrate_charged_response.local_annihilation_map`; do not edit them. No eigensolver, matrix exponential, SVD/rank, Liouville-space solver or arbitrary operator callback.

### Public APIs

- `interval_support_report(L,start,length)`: built-in int only (exclude bool/coercions); L3..64, start0..L-1, length1..L. Invalid values/types raise ValueError before work. Ordered support is `[(start+j)%L for j in range(length)]`. Return `L,start,length,support,complement_size,status,peel_steps,scope`. Status `theorem_applies` iff length<=L-2, else `outside_theorem_one_exterior` or `outside_theorem_full_support`. For applicable intervals, peel the listed start endpoint successively in the ORIGINAL ring: steps `{removed_site,exterior_neighbor,remaining_support}` where exterior neighbor is `(removed_site-1)%L`. For inapplicable intervals steps is empty. Scope explicitly says this checks support hypotheses, not conservation of an arbitrary operator; conditional on boundedness, weak conservation and C>0, and not excluding low-energy gauge emergence. It does not return a floating rank.
- `case_report(L,g)`: built-in ints only; L in {3,4,5}, g in {0,1}; C=1 and total-number cutoff K=2 fixed. No arbitrary numerical parameters. Report schema below. Input ValueError is a programmer error, not unavailable science.
- `demonstration_report()`: six case slots L3,4,5 outer and g0,1 inner; one case construction each, no retries. Also list all interval hypotheses start0 and lengths1..L for each L3,4,5 (12 records), plus three wrap cases `(3,2,1),(4,3,2),(5,4,3)`. Top-level `module,cases,interval_controls,limitations`. If an inherited arithmetic failure prevents a case, retain its slot as `numerical_unavailable` with reason and report null. No alternate case replacement. A case with computed diagnostic discrepancies is returned and retained, not caught as unavailability.

Each case builds complete sectors N0,1,2 once, concatenated by increasing N and lexicographic occupation within each. Sum dimension is `binomial(L+2,2)` (10,15,21). Check direct-sum dimension and all sector dimensions against `MAX_DIMENSION=512` BEFORE calling a builder or allocating. Local-annihilation maps at site0 for N1 and N2 are each obtained once and embedded in the direct sum. Thus each fully available case has three model builds and two rectangular ladder calls; a fully available six-case demo has exactly18/12, zero eigensolves, no retries. A failed case stops without dummy calls, so unavailable demos may have fewer calls. Pass the current MAX_DIMENSION explicitly to every capped_binomial preflight. Recheck the live cap immediately before each inherited builder/map call and each owned dense allocation, not merely once at entry. Do not rely on inherited constants/default caps; initially insufficient cap permits zero builder/map calls.

Private `_owned_case(L,g)` may own arrays and operators and be used by case/demo orchestration; no arrays or mutable shared caches returned publicly. Define local `NumericalUnavailable`. Public validation and cap checks stay outside exception translation. At the two inherited call boundaries only, translate (a) ValueError with exact message `numerically unresolved arithmetic or singular solve` and an arithmetic cause of FloatingPointError, OverflowError or numpy.linalg.LinAlgError; (b) exact messages of form `numerically unresolved <label>: <reason>` with labels `C`, `g`, `hopping scale`, `interaction scale`, `hopping matrix`, `interaction diagonal`, `annihilation map real component`, `annihilation map imaginary component` and reasons `nonfinite`, `subnormal`, `underflow`. This explicit allowlist follows the called inherited paths. Do not catch arbitrary ValueError/TypeError, cap failures, structural/missing data errors, or programmer bugs as unavailability. Owned numerical operations translate FloatingPointError/OverflowError only; explicit nonfinite checks raise NumericalUnavailable. No linear solve is called. Demo catches only NumericalUnavailable.

Validate inherited model L/N/C/g metadata, complete lexicographic occupation bases and expected Hamiltonian shape, and each ladder's expected rectangular shape before embedding. Missing or structurally invalid returned data is an integration error (ValueError), not numerical unavailability. Finiteness checks cover both components of all input arrays and all derived commutators, scales, allowances, norms and witness quantities. Finite diagnostic disagreements remain available and retain raw residuals.

`case_report` returns its schema on construction success and raises NumericalUnavailable on recognized numerical failure. Every demo case slot has keys `L,g,status,reason,report`; success status is `available_diagnostic`, reason null and report the case dict; failure status is `numerical_unavailable`, reason nonempty and report null. All six ordered slots remain present; construction success is not comparison success.

### Fixed operator controls

For every owned complete direct sum:

1. Identity.
2. Bounded local flip at site0: lower only occupations n0=1 via the rectangular annihilation maps, then add its adjoint. Its upper boundary is the exact compression, not a replacement onsite truncated CCR.
3. Local vacuum projector n0=0.
4. Global total-number parity, labeled globally conserved and scalar within each fixed sector.
5. Ring reflection fixing exterior vertex1: site permutation x->2-x modulo L. Label one-exterior/symmetry counterexample; actual moved support may be smaller when reflection has additional fixed vertices.
6. For L4 only, opposite-site SWAP0,2 and the dark-vacuum projector on modes0,2.

The dark projector preserves exterior occupations and local total n=n0+n2. In the normalized site basis, entries for fixed n are `sqrt(binomial(n,n0)*binomial(n,m0))/2**n` (n<=2); all other entries zero. This is the full bounded dark-vacuum projector compressed to the declared sectors, not projection onto global vacuum. SWAP is an exact occupation permutation.

Build `[H,A]=H@A-A@H` in binary64; retain all real/imaginary residuals without clipping. Never call the matrix norm an unrestricted commutator norm. Expected conservation labels come from exact arguments: all identity/parity/reflection/SWAP, plus dark projector at g0. Flip/vacuum projector are nonconserved; dark projector at g1 nonconserved. Finite zero residual alone never supplies an exact label.

Predetermined scalar witnesses (all unused sites vacuum):
- flip: row one particle at site1, column vacuum; expected -1;
- local vacuum projector: row one at site1, column one at site0; expected +1;
- L4 dark projector: normalized row dark two-particle vector, column bright two-particle vector, expected g/2. With basis occupations |2_0>,|1_0,1_2>,|2_2>, bright coefficients `(1/2,1/sqrt(2),1/2)` and dark `(1/2,-1/sqrt(2),1/2)`.

No norm-sign inference substitutes for these witness values. Independent tests include non-Hermitian local matrix units for both boundary recurrence orientations using finite occupation-entry oracles; this need not expand production report operators.

### Numerical policy and report schema

With dimension D, float64 epsilon u=2^-52, define entrywise pre-cancellation scale `S=abs(H)@abs(A)+abs(A)@abs(H)` and comparison allowance `T=256*u*D*S`. For exact conserved controls compare absolute real and imaginary residuals separately to T. This is a frozen heuristic forward-arithmetic diagnostic, not a rigorous certificate of libm/source construction. No result-relative floor, `max(1,S)`, clipping or tolerance fitting. Here bounded fixed inputs avoid extreme underflow/overflow; nonfinite assembly yields unavailability.

For scalar witness vectors v,w, let `delta=v†[H,A]w-expected` and `allowance=256*u*D*sum_(ij)|v_i|S_ij|w_j|`. Require BOTH `abs(delta.real)<=allowance` AND `abs(delta.imag)<=allowance`, conjunctively with any separate sign requirement. Use the same componentwise absolute-error convention for conserved matrix entries. Add no expectation-relative floor. Predetermined nonzero witnesses separately require nonzero real part with the expected sign. At g0 the dark expected value is zero and no nonzero requirement applies. Source vector and coefficient rounding remain part of the heuristic status, not exact arithmetic proof.

`case_report` keys: `L,g,C,cutoff,dimension,basis_order,status,operators,scope`. `status=available_diagnostic` means constructed, not all comparisons passed. Each operator record: `name,support_class,exact_conservation,commutator,finite_frobenius_norm,conservation_diagnostic,witness`. Encode matrices as `{shape,real,imag}` lists and scalar complex values as `{real,imag}`. `exact_conservation` is boolean from the algebraic control, not solver inference. `conservation_diagnostic` is `within_heuristic_allowance`/`outside_heuristic_allowance` for conserved controls and `not_applicable_nonconserved` for other controls. Retain full commutator matrix even when it disagrees. `witness` null except three designated controls; otherwise `{value,expected,allowance,status,nonzero_sign_required,nonzero_sign_satisfied}`. Witness status is `within_heuristic_allowance` iff both component differences pass AND any required nonzero/sign passes, else `outside_heuristic_allowance`; absence of sign requirement reports satisfied=true. Scope includes `finite_compression_only=true,numerical_error_certified=false,empirical_validation=false,infinite_theorem_from_numerics=false`.

CLI: absolute invocation in empty working directory/noPYTHONPATH, default human-readable text, --json exactly one strict JSON demonstration; successful exits0, empty stderr, no writes. Text explains conditional theorem and finite diagnostic scope; no exact wording freeze.

### Pre-execution interface clarification

Before either new file executed, implementation/test authors requested names and mock seams omitted from the reviewed schema. These clarify interoperability, not the equations or controls. Operator names in order are `identity`, `local_flip`, `local_vacuum_projector`, `total_number_parity`, `reflection`, then L4-only `opposite_site_swap`, `dark_vacuum_projector`. Their `support_class` values are respectively `empty_support`, `singleton`, `singleton`, `full_support`, `one_exterior_counterexample`, `disconnected_opposite_sites`, `disconnected_opposite_sites`. `basis_order` is a list of occupation lists. Module identifier is `conditional-substrate-local-symmetry-v1`. Interval scope is a dict including `checks_support_hypotheses_only=true`, a `conditional_on` list and `low_energy_gauge_emergence_excluded=false`; explanatory prose is not frozen.

Imported `all_number_model`, `capped_binomial` and `local_annihilation_map` are module-level patch targets. Inherited models are attribute-bearing `FixedNumberModel` instances, not dictionaries. Private `_owned_case(L,g)` returns `{L,g,dimension,basis,H,operators}`; each private operator contains `{name,support_class,exact_conservation,matrix,witness}`, with witness null or `{row,column,expected}`. `_case_report(owned)` produces the detached public schema. `case_report` validates then builds and reports. Private seams `_preflight(L)`, `_zeros(L,shape)` and `_witness_report(L,commutator,scale,row,column,expected)` support independent cap and comparison tests. The CLI entry is `main(argv=None)`. These are private fixed-control seams, not a new public arbitrary-operator API. Independent boundary-recurrence tests use complete K2 occupation-entry oracles and the existing private `_case_report` seam. The separate invisible-projector counterexample uses a complete N=3 occupation oracle only; no different cutoff is passed to production reporting. This descriptive correction was applied after the first focused run: its wording recommendation arrived after launch, so the bound document remained unchanged during execution. The original document is preserved with the initial quartet; no tests, controls or equations changed.

### Tests and verification

Separate test authors derive occupation-matrix and permutation/projector controls independently rather than reuse production helper output. Fixed six cases are enough for physics; share owned report fixtures, do not repeat real demo in multiple tests. Mock orchestration/CLI where possible, and run actual demos in isolated verification. Check missing/invalid data, cap before allocations, zero model/eigh retry, all retained six/15 slots, cross-number flip, positive-g swap versus dark projector, and nonzero/sign witnesses. All previous baselines stay protected.

Only after independent contract and static implementation review: focused tests, isolated demos, eight algebra groups, Python3.8 grammar with actual runtime distinct, selected inherited-plus-new suites in serialized bounded processes. Preserve all failures and versions; no rerun of historical supplemental sign probe. This numerical work checks the new implementation and fixed controls, not all possible local operators.
