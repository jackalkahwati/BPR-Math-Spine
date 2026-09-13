# Fixed occupation-orbit gauge encoding

2026-09-13. TOE connection campaign module1. **Mathematics and API frozen before implementation. Independent derivation and adversarial review completed; no scientific controls have been evaluated at this freeze.**

## Purpose and scope

Test a single explicitly chosen occupation-space encoding against the supplied dihedral link algebra and the actual complete Bose dynamics. This is a diagnostic encoding, not an inherited microscopic gauge dictionary. A successful regular-representation algebra is not evidence for independently local endpoint transformations or vertex Gauss constraints.

The model remains

\[
H=gD-C\sum_x(T_x+T_x^\dagger),\qquad
D=\sum_x n_x(n_x-1)/2,\qquad
T_x=a^\dagger_{x+1}a_x.
\]

Use complete fixed-number lexicographic occupation bases; dimension is \(\binom{L+N-1}{N}\), checked against512 before allocation. No occupation cutoff, new interaction or energy fit is allowed.

## Fixed control table

- L=N=3,4,5.
- C=1; g/C=0,0.7,40.
- Dimensionless times Ct=0,0.01,0.1.
- Seed occupation \((N-1,1,0,\ldots)\).
- Group labels \((k,s)\), s=+1 then−1, k=0,...,L−1 within each sign.
- Spatial action x maps to k+sx modulo L.
- Group multiplication \((k,s)(l,t)=(k+sl,st)\), with modular first coordinate.
- W has columns \(|g\cdot\mathrm{seed}\rangle\) in that label ordering.
- Right endpoint convention \(R_g|h\rangle=|hg^{-1}\rangle\).
- Only L5 compares against the existing `gauge_heat_kernel.central_laplacian(5)`, with fixed dimensionless single-link normalization lambda1. L3/4 are orbit/algebra controls; the older gauge API is not expanded.

These are nine physical model cases with three time records each, not a seed/parameter search. Algebraic fixtures and invalid-input tests will be separately frozen before implementation.

## Energy and interpretation conventions

Report the raw Hamiltonian/compression separately from the comparison using the known orbit onsite scalar

\[
E_*=g(N-1)(N-2)/2.
\]

The centered generator comparison is \((W^\dagger HW-E_*I)/C\) versus the supplied central D5 generator. The full target Hamiltonian for phase-consistent evolution is \(E_*I+C\Delta_5\). This fixes a comparison convention, not a physical vacuum-energy subtraction. No best-fit offset, spectral alignment or optimized scale is permitted.

Transported endpoint actions and coordinate projectors are operators supported on \(P=WW^\dagger\); their algebraic identity is P, not the ambient identity. Full spatial permutations and these supported operators must remain distinct. Numerical microscopic density commutators can witness support in the complete sector; they do not establish a site-tensor-factor gauge construction.

## Required mathematical gate

Two independent roles are deriving and adversarially checking:

1. Trivial orbit stabilizer, regular group action and coordinate/endpoint algebra.
2. Actual compression, including the L3 exception and the proposed scalar L4/5 compression.
3. Complement leakage and exact occupation-hop witnesses.
4. Compression and fixed-target Duhamel bounds with the scalar phase treated consistently.
5. The supplied D5 generator spectrum and unfitted mismatch.
6. Microscopic symmetry/support distinctions and the limits of any inferred gauge interpretation.

Independent reports and resolved corrections will be recorded before control execution. An expected negative dynamical result is not a test failure and will not trigger a new seed or Hamiltonian.

## Independently cross-checked mathematical results

The separate adversarial reviewer derived the following exact candidate oracles by occupation hopping, without evaluating controls. Let s0=(0,−1), a=(1,−1), B=(I−P)HW, and J6 denote the all-ones6×6 matrix.

| L=N | Hc | B†B/C² | ||B||/C |
|---|---|---|---|
| 3 | gI−C(R_s0+2R_a) | 3(I+R_s0)+2J6 | 3sqrt(2) |
| 4 | E*I | 14I+8R_s0+6R_a | 2sqrt(7) |
| 5 | E*I | 18I+9R_s0 | 3sqrt(3) |

The L3 coefficient2 is the hopping matrix element from (2,1,0) to (1,2,0); omission would miss Bose enhancement. Its centered spectrum is ±3C once each and ±sqrt(3)C twice each. Its right-reflection commutator has norm2sqrt(3)C. B is independent of g because the onsite term maps every orbit column to E* times that column.

Every column is a normalized occupation ket, with no orbit-average normalization. For L4/5 the seed-column squared leakage is C²(4N−2). The L4 sign-character vector is dark to B, whereas L5 has B†B≥9C²I. Non-invariance of the full code therefore must not be reported as leakage of every state for all sizes.

The independent derivation author separately confirmed the full compression and leakage Gram formulas. The Gram spectra divided by C² are L3:0(3),6(2),18(1); L4:0,4(2),12,16,24(2),28; L5:9(5),27(5). Thus the minimum eigenvalues are0,0,9C² and the seed-column norms are Csqrt(5),Csqrt(14),Csqrt(18). The sign-character vector is also dark at L3 and is an exact eigenvector of H with energy g+3C. The orbit projector must not be confused with the old hard-core projector: every orbit state lies outside the hard-core subspace.

The Hamiltonian multiplication defect supplies an additional independent identity, `W†H²W−Hc²=B†B`. It can be checked directly without requiring the inherited helper's narrower component domain. Exact coordinate matrix units are `F_a L_(ab^-1) F_b=|a><b|`; their closure proves the encoded matrix algebra, not physical emergence.

The supplied D5 central generator has eigenvalues0,2,(7−sqrt(5))/2,(7+sqrt(5))/2 with multiplicities1,1,4,4. Write b=(7+sqrt(5))/2. Since the L5 centered compression is zero, its target mismatch has norm b. For K=E*I+C Delta5,

`F=HW−WK=B−CW Delta5`,
`F†F=B†B+C² Delta5²`,
`||F||=C sqrt(27+b²)`.

The code vector `v_(k,s)=cos(4πk/5)/sqrt(5)` is a proposed independent maximal witness. A scalar offset cannot remove the target's nonzero spectral splittings; no offset optimization is performed.

For compression and target respectively, Duhamel gives upper bounds `min(2,|t| ||B||)` and `min(2,|t| ||F||)`. A proposed additional L5 lower bound is

`max(0,2 sin(|Ct|b/2)−27(Ct)²/2)`

on `|Ct|b≤π`. It follows from the centered compression-return bound `||W†exp(−it(H−E*I))W−I||≤t²||B||²/2`. The adversarial reviewer confirmed the twofold integral proof: with centered P H P=0, `Q U(t)W=−i∫ exp(−i(t−s)QHQ) B W†U(s)W ds`, so its norm is at most |t| ||B||. Integrating `d(W†U(t)W)/dt=−iB†Q U(t)W` gives the displayed return bound with no ||QHQ|| factor. The independent derivation author confirmed the same bound through the separate identity `A''(t)=−B†exp(−it(H−E*I))B` for A(t)=W†exp(−it(H−E*I))W. Hence ||A''||≤||B||², A(0)=I and A'(0)=0. Both proofs exclude any extra g-dependent term. The lower bound is adopted on its stated interval, containing all frozen times. The analytical lower expressions at0.01 and0.1 are strictly positive; this is not a reported numerical run.

Full microscopic spatial permutations U_g obey HU_g=U_gH and U_gW=WL_g. Transported partial actions W A_g W† are different operators. For L4/5, their full microscopic commutator norm is ||B|| even though the compressed commutator vanishes; the identity action transports to P and already witnesses this distinction. For transported right reflection, `[n_x,W R_s0 W†]` has norm1 for every site. This nonzero density witness excludes support on a proper site subset; vanishing density commutators would not prove locality. In particular P commutes with every density while imposing a global occupation pattern.

## Numerical/API gate

Frozen contract after adversarial review, before implementation:

- Public `case_report(L,g)` accepts only built-in/NumPy integers L3,4,5 and built-in/NumPy finite real scalars g exactly0,0.7,40; rejects bool, strings and custom conversion objects before coercion. C=1,N=L are fixed. No parameter snapping, scale fitting or caller-supplied model/cache. Public `demonstration_report()` returns nine cases in L-major, g-minor order.
- Module constants: `MAX_DIMENSION=512`, `SIZES=(3,4,5)`, `COUPLINGS=(0.0,0.7,40.0)`, `TIMES=(0.0,0.01,0.1)`, `ATOL=RTOL=2e-10`. Alias `NumericalUnavailable` from the inherited current-response module.
- Private `_orbit_data(L)` returns owned basis/labels/W and label left/right actions for independent fixtures. Private `_screened_eigensystem(H)` validates a copied square finite numeric matrix, cap512, Hermiticity, and eigenpair/orthogonality screens; returns owned eigenvalues/vectors plus diagnostic residuals. Exact return keys will be settled in the final API freeze.
- Eigenpair/orthogonality tolerance tau=256 eps_float64 times dimension; Frobenius orthogonality residual≤tau and eigenpair residual≤tau max(1,||H||F). These are heuristic diagnostics only. No unique-ground or gap gate; arbitrary degeneracy is allowed. Do not symmetrize an invalid input or clip computed spectrum/norms.
- Scientific matrix norms are operator2-norms; Frobenius norms occur only in explicit numerical screens. Scalar/matrix oracle comparisons use each real/imaginary component against `ATOL+RTOL*abs(reference_component)`, not a maximum-channel or Frobenius acceptance criterion. These are regression checks, not roundoff certificates. Eigensystem acceptance does not certify2e-10 propagator accuracy: sensitivity depends on time, residual and orthogonality. Bounds retain raw computed errors, exact-model bounds and regression tolerance separately; at zero time a positive roundoff residual may be regression-consistent with the zero analytic bound without satisfying the literal observed inequality.
- Invalid public/helper inputs raise TypeError/ValueError; numerical failures use NumericalUnavailable. Reject nonfinite/complex/bool inputs before membership checks. Membership compares exact stored numeric values to frozen binary64 constants, using integer ratios for floating scalars rather than NumPy mixed-dtype equality: float32(0.7) must not be silently admitted by weak scalar promotion, and a higher-precision value must not be rounded into the set. No generic custom numeric conversion protocols. Eigensystem outputs require expected shapes, finite real eigenvalues, finite eigenvectors and residual screens, not a positivity or degenerate-eigenvector convention.
- Distinguish full-space target error from compressed-code propagation error. Leakage norm is an amplitude/generator norm, not a probability. Analytic leakage and target residual formulas remain available if a numerical SVD/eigh diagnostic fails. Demonstration aggregation preserves all nine ordered cases even when one is numerically unavailable; no case replacement or silent omission.
- Compute unitary propagation in the common centered frame H−E*I. Compression uses Hc−E*I; the D5 target uses C Delta5. This removes the same known phase from both sides and is not a fitted centering. At t=0 retain the numerically evaluated residual; compare it to the exact zero reference without forcing a zero output.
- Retain successfully computed algebra/generator/leakage results if an eigensystem or a later time evaluation fails. Failed dynamics fields become null with `numerical_unavailable` and a reason; exact analytic envelopes remain. Unsupported target L3/4 is `not_applicable`, never zero mismatch. A failed numerical construction is not evidence for a physical obstruction. Pre-eigensystem observation failures can be isolated by section; keep independent sections and analytic references. A failed ambient eigensystem blocks all numerical dynamics; a failed compression/target eigensystem blocks only that comparison, not the other. Its dependent full/projected errors are null; independently computed ambient leakage_amplitude may remain available. Private matrix helper object-dtype support is not required, but bool/string/custom conversion leaves must not be coerced into accepted numeric matrices. A single-time propagation or norm failure blocks only its dependent record. A Gram-minimum eigvalsh failure nulls that minimum but retains the Gram and independent norm if computed, with its comparison inconclusive unless another verified mismatch takes precedence. Section-level unavailability is for early assembly failures, not discarding already successful independent observations. Catch only expected numerical failures, not programming errors or invalid public input.
- Reports are detached JSON-native trees, complex matrices represented by `{shape,real,imag}` and unavailable values by null. No NaN/Infinity, mutable caller arrays or caller caches are exposed. Distinguish exact analytic obstruction statements, calculated diagnostics and numerical availability.
- Demo has text/default and `--json` modes, stdout only, no filesystem artifacts or import-time computation. Tests and demonstration use the same fixed control table; extra fixtures test mathematical identities/input handling, not alternative candidate searches.

No ground-state uniqueness requirement belongs in a full-unitary calculation. The report/helper schema below is frozen after independent review. The permitted L3 nullable analytical ambient right-reflection norm is selected for this implementation; its observed norm still receives an independent test oracle. Review-derived value3sqrt(2)C can be tested independently but is not required as an exported analytic reference. The independent derivation author subsequently confirmed it using the same representation-block decomposition: standard blocks combine squared couplings12C² and6C², while the trivial block has squared norm18C². This additional confirmation does not change the frozen nullable exported field.

### Report and helper schema

`_orbit_data(L)` returns a fresh dict with keys `L`, `N`, `basis`, `labels`, `seed`, `W`, `left`, `right`. Basis/labels/seed are immutable tuples; left/right are tuples of real matrices in label order. W and action arrays are detached per call. No H or model eigensystem is built here. Enumerate the complete basis with `substrate_fermionization._occupations` after the explicit dimension guard (not by diagonalizing or building a second model). Build spatial actions in a separate private helper to avoid allocating every ambient matrix at once.

`_screened_eigensystem(H)` returns a dict `values`, `vectors`, `orthogonality_residual`, `eigenpair_residual`, `tolerance`, `scale`; arrays are detached and owned. Validate finite numeric non-bool square input (1≤dimension≤512) and Hermiticity `||H−H†||F≤tau*max(1,||H||F)`, rejecting material non-Hermiticity rather than symmetrizing. Bounded validation copies are allowed; reject shape/size/type before further dense work. Accepted small asymmetry remains a disclosed heuristic input screen, not an exact Hermitian certification. Diagonalize H/scale with `scale=max(1,||H||F)` and rescale eigenvalues, then check residuals against the original H. No eigenvector phase convention is imposed. Invalid input fails before substantial dense allocation/diagonalization (bounded validation copies are permitted); unsuccessful or invalid numerical eigenoutputs raise NumericalUnavailable. A valid zero matrix is supported. No normal/subnormal restrictions are added beyond finite representability and arithmetic checks; this helper is not an arbitrary-precision solver.

`_unitary_columns(system,t,columns)` computes V exp(−itE) V† columns from an owned screened eigensystem, returning a finite owned complex array. Its use allows tests to inject a failure at one time without changing other records. No full ambient matrix exponential is required. `_operator_norm(matrix)` returns the finite spectral norm, raising NumericalUnavailable on numerical failure; tests may instrument it. These private helpers are fixture seams, not public scientific APIs.

`case_report` top-level keys:

- `model_id`, `L`, `N`, `C`, `g`, `dimension`, `code_dimension`, `status`, `reason`, `conventions`, `scope`.
- `orbit`: `seed`, `labels`, `orbit_dimension`, `isometry_residual`, `left_action_residual`, `right_action_residual`, `endpoint_commutator_residual`, `coordinate_covariance_residual`, `transport_identity_residual`, `comparison`. All orbit residuals compare with exact zero using the fixed regression tolerance, and participate in top-level aggregation. Algebra residuals may use entrywise maximum absolute differences of real permutation/projector matrices; do not label these as scientific operator norms. Check all label products and coordinate covariance in the small code; use exact one-hot W structure and representative full-space checks rather than cubic ambient work for every product.
- `compression`: `onsite_energy`, `matrix`, `centered_matrix`, `analytic_matrix`, `analytic_centered_eigenvalues`, `right_reflection_commutator_norm`, `analytic_right_reflection_commutator_norm`, `comparison`.
- `leakage`: `gram`, `analytic_gram`, `norm`, `analytic_norm`, `seed_column_norm`, `analytic_seed_column_norm`, `minimum_gram_eigenvalue`, `analytic_minimum_gram_eigenvalue`, `invariance_status`, `all_states_leak`, `comparison`. `all_states_leak` is an analytic boolean from the exact Gram spectrum, not a numerical rank threshold.
- `symmetry`: `spatial_intertwining_residual`, `spatial_hamiltonian_commutator_norm`, `partial_right_reflection_commutator_norm`, `analytic_partial_right_reflection_commutator_norm`, `density_right_reflection_commutator_norms`, `analytic_density_right_reflection_commutator_norms`, `density_multiplication_defect_residual`, `comparison`. Generator spatial rotation/reflection suffice for group symmetry checks. Compute norms by equivalent rectangular/code blocks where justified, retaining an independent dense test oracle. The analytic partial-right-reflection commutator norm is required for L4/5; for L3 it may be null with its comparison explicitly omitted, while the observed norm and independent dense-versus-block test remain. Do not silently substitute the L4/5 formula at L3.
- `target`: `status`, `reason`, `generator`, `analytic_eigenvalues`, `generator_mismatch_norm`, `analytic_generator_mismatch_norm`, `full_residual_norm`, `analytic_full_residual_norm`, `maximal_witness`, `witness_residual`, `comparison`. L3/4 status `not_applicable`; other target fields null with explicit unsupported-comparison reason. L5 status `analytic_obstruction` independently of diagnostic availability.
- `eigensystems`: `ambient`, `compression`, `target`, each diagnostic metadata only, with `status`, `reason`, `orthogonality_residual`, `eigenpair_residual`, `tolerance`, `scale`. No arrays in this section; target not_applicable outsideL5.
- `times`: three records with `t`, `tau`, `compression`, `target`. Each dynamics record has `status`, `reason`, `full_space_error`, `projected_error`, `leakage_amplitude`, `analytic_upper_bound`, `analytic_lower_bound`, `upper_bound_comparison`, `lower_bound_comparison`. The full-space compression/target errors use their respective unitary columns; projected error is ||W†U(t)W−Ucomparison(t)||, and leakage amplitude is ||(I−P)U(t)W||. No probability field. Compression lower bound0; target lower bound only the proved L5 formula. Analytic lower/upper envelopes and their comparisons apply specifically to `full_space_error`; `projected_error` and `leakage_amplitude` are separately retained diagnostics, not silently tested against a target lower bound. Analytic bounds remain on unavailable records. Target dynamics not_applicable outsideL5.

Complex matrices use `{shape,real,imag}`; real matrices/vectors are JSON nested lists. Do not emit the large ambient H/W in reports. Analytic small matrices/spectra/Gram norms are separate references, never substitutes for failed observations. Group orbit labels serialize as nested lists.

Successful eigensystem/dynamics status is `available_heuristic`; failed numerical status is `numerical_unavailable`. A dynamics record with available numbers but failed bound comparison uses `diagnostic_mismatch`. Demo path is `scripts/demo_substrate_gauge_encoding.py`. Summary keys are `case_status_counts` (mapping statuses to counts) and `analytic_d5_obstruction_count` (reference-claim count, not a numerical success count).

`density_multiplication_defect_residual` is the maximum absolute component across all site pairs of `W†n_x n_yW−(W†n_xW)(W†n_yW)`, compared to zero. Leakage comparison also includes the independently assembled Hamiltonian defect `(HW)†(HW)−Hc²` against its analytic Gram, without needing a full H² allocation. These clarifications resolve author questions before execution and do not change the mathematical control grid.

Each `comparison` is `{status, max_absolute_error, reason}` with status `consistent`, `mismatch`, or `inconclusive`. Compression/target section comparisons include their observed eigenvalue spectra against the analytic references. Failure of the corresponding eigensystem therefore makes that section's aggregate inconclusive (unless another mismatch dominates), while preserving its available matrix/norm observations; it does not leave the whole section consistent merely because pre-eigensystem checks passed. Section comparison combines its declared analytic oracle checks using mismatch before inconclusive before consistent; max_absolute_error is a diagnostic summary only, while acceptance remains componentwise. Bound comparisons apply the fixed reference-based tolerance to the relevant one-sided inequality. Equality checks and one-sided bound checks must not be conflated.

Top-level `status` is `available_heuristic` if all applicable observations/comparisons are available and regression-consistent, `diagnostic_mismatch` if any applicable comparison mismatches, otherwise `numerical_unavailable`. A mathematical obstruction remains a separate analytic claim, not a failing numerical status. `reason` describes the first mismatching component when any mismatch exists, otherwise the first unavailable component, consistent with status precedence. Bad public inputs raise rather than return a report. An internally failed model/orbit construction preserves metadata, exact references and all three time placeholders as feasible, instead of skipping a case.

`demonstration_report` keys: `model_id`, `status`, `scope`, `controls` (sizes,couplings,times,C), `cases` (allnine), `summary` with case-status counts and D5 analytic-obstruction count explicitly labeled as reference claims, not observed successes. Status precedence matches the cases. `scope` includes `diagnostic_encoding=True`, `local_gauge_emergence=False`, `numerical_error_certified=False`, `empirical_validation=False`, `empirical_status='empirical_test_unavailable'`. `conventions` states spatial action, inverse right endpoint, complete occupation basis, scalar-phase centering, fixed target lambda, and norm meanings.

Budget: one canonical full model per case, one ambient eigensystem and one compression eigensystem per case, plus one10×10 target eigensystem in each L5 case (21 propagation eigendecompositions on a fully successful demonstration). Also allow one code-sized Hermitian Gram eigenvalue calculation per case for the observed minimum_gram_eigenvalue (9 eigvalsh calls). This signed minimum must not be replaced by a Gram singular value or a thin rectangular B spectrum that omits null directions. Reuse propagation eigensystems for all times. No eigensolve for analytic references. Do not serialize large ambient operator families or form Kronecker/Liouville matrices. Internal arrays remain within512×512 and code rank≤10; group operators should be streamed. The exact eigensolve count is a successful-path work/ownership test, not a reason to suppress necessary numerical validation.

## Execution record

- Local baseline verified: `2319b1e54d97f775f32ebd57d98d42b97d106e73` on `science/substrate-prediction-contract`.
- Created local branch `science/substrate-gauge-encoding`.
- Only unrelated AGENTS.md and CLAUDE.md were untracked at baseline; neither is edited/staged.
- New campaign ledger created.
- Independent derivation and separate adversarial mathematics/API review confirmed compression, full leakage Grams/spectra, support witnesses, D5 generator/target residual and the g-independent lower bound. No controls were executed by either reviewer.
- Pre-execution API corrections: distinguish21 propagation eigendecompositions from9 additional signed Gram eigenvalue diagnostics; explicitly compare orbit residuals; align mismatch/reason precedence; define the Hermiticity screen and bounded validation; use exact stored-value scalar membership; apply envelopes only to full-space errors. These were contract corrections before implementation, not numerical failures or tuned acceptance.
- Mathematics/API frozen. Separate implementation and independent test authors dispatched with exclusive source/demo and test ownership. They were not authorized to execute controls/tests during authoring.
- Implementation readiness: source741lines SHA256 `a964d0b6f2a6ad241d16548f3fe12c4b213f18140df2d0ac8c73329143eb703f`; demo66lines SHA256 `bc9c1ba2e52ebcea8148d99c45596164fe09533819567f925ba5ac0c0c5dc878`. Parent independently verified these hashes. Author reports AST/syntax-only checks, no implementation import or numerical execution. Separate static code review started; independent tests still finalizing at this point.
- Independent test author ready:1192lines,43test functions, SHA256 `94484125c1b4728ff1b54076bee86bab27738ff132c6ee2b78b051afa5f20315`. Tests were authored without reading the new source/demo and without numerical execution. Parent verified all three source/demo/test hashes and Python3.8 AST grammar; the separate reviewer is statically checking implementation and tests before the first execution.
- Separate static reviewer read source/demo and the independent test file. No verified source mathematical/implementation blocker was found. One pre-execution test expectation defect was identified at initial test lines846–847: failure of compression/target eigensystem makes the associated spectral comparison inconclusive, not consistent. Available matrices and unrelated sections remain preserved. Test author independently confirmed and made the minimal expectation correction, additionally asserting retained generator matrices. No tolerances changed. Corrected test1197lines SHA256 `85a971c8c5b3d2617a1baa7731c48a315eda0a8c61263fba6031cc2beb8608d3`; parent verified this and unchanged source/demo hashes. Initial hash remains recorded. This repair preceded every scientific execution.
- First focused warnings-as-errors run used `/private/tmp/bpr-gauge-encoding-verification.py focused`:146passed,1failed,1skipped in2.41s (runner2.8037s), actual Python3.9.6, all three hashes unchanged. Failure `test_screen_signed_tiny_spectrum_is_not_clipped` at test line756: `_screened_eigensystem` calls `_frobenius`, where division at source line125 overflows for a tiny signed scalar. Independent author/reviewer diagnosis requested before edits; no conclusion about fixture versus implementation is assumed yet. Skip reason will be recorded separately. No failed result is counted as passing. Original stdout/stderr log is retained at `/private/tmp/claude-501/-Volumes-T9-Backup-Code-BPR-Math-Spine/0e4b0068-afa4-4ae7-8a4c-0507d30ad6d7/tasks/butqlwkqb.output`.
- Independent author and reviewer each reproduced the failure using isolated NumPy arithmetic only. Confirmed production defect: complex-array division by a subnormal real scale overflows internally even when the normalized result is representable; separate real/imaginary divisions preserve the smallest subnormal. The same normalization defect affects `_operator_norm`. Authorized minimal componentwise-scaling repair in those two helpers and independent operator-norm regression fixtures, with no clipping, tolerance change or domain restriction. The original signed-eigenvalue test remains unchanged. The single skip is `test_extended_precision_cannot_round_into_allowed_coupling`: this Mac's `longdouble` and binary64 both have52 mantissa bits, so the wider-precision fixture is unavailable.
- Minimal repair ready: source742lines SHA256 `c32b603806885d0b9149d2d6dd9a363599cd1d2e90ab59257b3cd3b0189bcbff`; demo unchanged. Independent test author added real and pure-imaginary singleton smallest-subnormal operator-norm fixtures requiring finite, positive, exact results; original signed-eigenvalue fixture and all tolerances unchanged. Test1209lines SHA256 `12b48245d1f29c370414267b786caa3cb9acb02581024650d0e5abf88c2c968e`. Author checks were syntax-only; targeted execution follows this freeze.
- Post-repair targeted warnings-as-errors selection `tiny or subnormal`:3passed,147deselected in0.86s. Full focused rerun:149passed,1platform-dependent skip in2.12s (runner2.4018s), Python3.9.6. Python3.8 grammar passed separately; source/demo/test hashes unchanged across both executions. Logs: `bfwxp31p0.output` and `bhtw9xuz6.output` in the same task-output directory as the first failure. The passing repair does not replace the original failing record.
- Verification runners prepared outside the repository and Python3.8 grammar checked without executing them. The first focused run is the first scientific execution; no earlier smoke/demo run preceded it. No commit, push or PR creation has occurred for this module.

## Regression verification

The frozen29 inherited suites followed by `substrate_gauge_encoding` all completed successfully in fresh sequential processes:3635passed,1platform-dependent skip,30 exit-zero suites,1550.6501s total. Warnings were errors, numerical thread limits were1, and bytecode generation was disabled. Actual runtime Python3.9.6; Python3.8 grammar passed separately. Source/demo/test hashes were unchanged. The exact ordered selection is in the campaign ledger; per-suite stdout, counts, durations and outcomes are retained in `b61zbwex2.output` in the task-output directory above. This is fresh-process coverage, not a claim of full single-process combined coverage.

Independent numerical reviewer ran the prepared bounded probe script once after the sequential run completed, with warnings-as-errors and all numerical thread limits1. Four frozen cases `(3,.7),(4,.7),(5,.7),(5,40)` at all3 times matched independently assembled Bose matrices and dense SciPy `expm`, a distinct algorithm from the test author's `expm_multiply`; maximum dynamics discrepancy7.494005416219807e−15. Rank-deficient thin-complement QR, comparison-eigensystem failure isolation, independently failed-field retention, complex componentwise tolerance and one-sided zero-bound checks passed. Source hash assertions passed before/after. Script `/private/tmp/module1_independent_review_probes.py`, SHA256 `07c53275efc18580677f600b192e7504e16f49d4a773827a4d164ccede824cf3`; output `/private/tmp/module1_independent_review_probes.output`. Reviewer found no remaining verified blocking defect. No retry or edits occurred; agreement is heuristic, not numerical certification.

The one frozen bounded same-process prefix passed1093 tests, with2543 deselected, in24.99s. It collected all30 suites but executed only the ordered prefix through `tests/test_substrate_nonlinear_response.py::test_strict_frozen_json_arithmetic_and_analytic_status_separate`; the new suite is collected, not executed in this prefix. Source/demo/test hashes stayed unchanged. Log `b0nc3dl3u.output`; collection/selection artifacts `/var/folders/ks/wwvm7v0n1tzdy3_gq88152tr0000gn/T/bpr-gauge-prefix-vqp96kh7`. This diagnostic ran once, not until green. Its passing result neither resolves nor erases the earlier independently reproduced one-ULP failure, and is not full single-process coverage.

Pass counts in the ledger's order:110,73,104,52,154,122,6,64,89,53,79,78,112,119,96,113,97,79,75,71,105,88,137,229,393,155,167,289,177,149. The single skip belongs to the new suite and is explained above.

## Observed bounded result

Isolated text and strictJSON demos passed from separate empty temporary directories, with stdout only, no stderr or artifacts. The JSON contains all9 cases and all3 times per case:9 `available_heuristic`, no diagnostic mismatch or numerical unavailability, and3 independently analytic D5 obstructions. The8 algebra-check groups passed. Python3.8 grammar and source/demo/test hash stability passed; actual runtime was Python3.9.6. Isolated runner elapsed8.5855s; retained log `bgsll9zcz.output` in the task-output directory above.

For L5 the observed full-space target errors are:

| g/C | Ct=.01 | Ct=.1 |
|---|---:|---:|
| 0 | 0.069489152679 | 0.667714260498 |
| .7 | 0.069488704666 | 0.667409133615 |
| 40 | 0.067088307135 | 0.462759061112 |

At these two times the exact-model lower bounds are respectively0.0448262364431 and0.322710770354, with upper bounds0.0695170755435 and0.695170755435. They are independent of g; the observations remain floating numerical diagnostics. Nonzero t0 roundoff is retained with an explicit literal-zero-bound violation reason despite regression consistency. No error certificate is inferred from passing tolerances.

The fixed code realizes the label algebra but is noninvariant in all three sizes. L5 additionally has a positive leakage Gram minimum9C², so no encoded state has zero instantaneous leakage. Its compressed centered Hamiltonian is zero rather than the stipulated nonzero D5 generator. This is a structural obstruction for the preregistered encoding, not a general impossibility theorem for emergent gauge physics. No local gauge field, Gauss constraint or empirical comparison is supplied.

## Publication

Scientific commit `86e5a193eb6f5784312b9c786e71294090e83a9f` was pushed by ordinary branch publication. [PR42](https://github.com/jackalkahwati/BPR-Math-Spine/pull/42) was created with base `science/substrate-prediction-contract` and head `science/substrate-gauge-encoding`; GitHub and the remote ref both verified that scientific head. PR state OPEN, mergedAt null. No merge was attempted. This publication-record update is a subsequent documentation-only commit; its eventual head is available from git/PR history rather than a self-referential hash. Unrelated AGENTS.md and CLAUDE.md remain untracked and unstaged.

The inherited same-process nonlinear-response repeatability limitation remains recorded in the campaign ledger; no older implementation/test changes are authorized by this module.
