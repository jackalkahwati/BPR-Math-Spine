# Sampled joint response reconstruction

2026-09-13. Module6, the final module of the approved connection campaign. Branch `science/substrate-sampled-response` starts from verified PR46 documentation head `1a01e7d2e068719e73d3cb92794c85425a9c9b7d`. PR46 is OPEN/not merged. Independent derivation, adversarial mathematical/schema review and numerical-policy review are complete. The implementation contract below is frozen before separate source/test authoring. No new scientific controls have executed.

## Approved scope

Study finite-record reconstruction of the existing two-source retarded response, distinguishing finite-window, trapezoid and declared sample-error contributions. Preserve the transposed negative-frequency numerator for complex cross responses, existing dimensionless ratio conventions and the difference between exact-model error bounds and floating numerical diagnostics.

Preserve the unchanged complete Bose Hamiltonian, all occupation states and inherited allocation caps. No additional interaction, bath, metric, fitted scale, physical calibration or empirical data is supplied. Exponential weighting is a transform regulator, not dissipation. No adaptive sampling until a desired result appears.

Approved physical controls: L=N3,4,5; C1; g0,0.7,40; both inherited energy partitions; m1. Frequencies y=eta*Delta1 for eta0.5,1,2. Paired schedules CT4 with128 intervals and CT8 with256 intervals, at most257 samples per trace. Reuse an owned system per physical case. A fixed synthetic reference L3,g0.7,symmetric has row gains diag(2,-3), column gains diag(5,7), and sample-error levels0 and1e-8. The deterministic perturbation, precise normalization, APIs, availability gates, tolerances and solver budgets must be frozen after independent review before execution.

Ratios require a denominator uncertainty set excluding zero. Loose or zero-containing bounds remain inconclusive. Model-known spectral envelopes do not establish experimental bandwidth knowledge, source-contact reconstruction or empirical validation; retain empirical_test_unavailable.

## Preliminary independent-review agreements (not yet a frozen API)

Both reviewers independently confirm the proposed retarded/Laplace signs and tail/quadrature bounds under per-state Hermitian positive-semidefinite Gram hypotheses. Arbitrary inherited kernel inputs do not by themselves establish those hypotheses; the new fixture should own transitions and construct Gram weights rather than trust a caller's cached matrices.

Use tau=Ct, Theta=CT, delta=Delta/C, nu=y/C=eta*delta1 and source degrees d=(0,1). Normalize the time trace by C^(-d_a-d_b), and the transform by C^(1-d_a-d_b). Both approved schedules have the same normalized step1/32: the second doubles the window, not the sampling resolution. Its prescribed quadrature envelope doubles; its tail decreases. Do not require monotone total error or ratio availability.

Provisional observation convention: errors are declared AFTER gains in normalized time-response units. Apply G=[[10,14],[-15,-21]] to completed entrywise traces/kernels and abs(G) to model tail/quadrature budgets. Do not multiply Gram weights by G and then transpose them in the negative-frequency term: that would replace G_ab by G_ba and change the response. Gained matrices need not be reciprocal or positive semidefinite. Observation error has no extra gain factor under this postgain convention.

The selected deterministic perturbation for final review is epsilon*(-1)^(j+a+b) for zero-based indices and the common integer sample index j. It is real with absolute value epsilon, within the stipulated complex-modulus error disk; the two schedules share their sample prefix. Epsilon0 keeps an explicit noiseless record. Constant gains cancel the ratio Q but are not inferred from observations.

For general Hermitian sources, the time response and imaginary-axis transform are entrywise real but may be nonsymmetric; off-diagonal response at time zero need not vanish. A complex two-level fixture exposes this: W=[[1,i],[-i,1]] with gap delta gives R12=2cos(delta*tau), R21=-R12, R11=-2sin(delta*tau), and K12(i*nu)=2nu/(nu²+delta²). Q=-nu²/delta², so no generic Q-in-[0,1], reciprocity or transform-PSD assumption is valid. Raw imaginary numerical residuals must not be silently discarded. These are pre-execution mathematical checks, not numerical observations.

The zero-time sample is the continuous right limit R(0+)=-i<[Oa,Ob]>, not half that value. A Heaviside-at-zero convention is irrelevant to the exact integral but would introduce an unbudgeted first-order endpoint error into this trapezoid rule. The complex fixture must retain R12(0+)=2.

The declared epsilon bounds the ideal synthetic perturbation, not necessarily the difference of stored floating samples after addition. Floating assembly is explicitly uncertified and receives separate diagnostics/proxies; do not assert an exact stored-noise bound or enlarge the declared observation budget to hide rounding. For even interval count n and q=exp(-nu*h), independent scalar oracles are the alternating shift epsilon*(-1)^(a+b)*(h/2)*(1-q^n)*(1-q)/(1+q), and the worst-case observation envelope epsilon*(h/2)*(1-q^n)*(1+q)/(1-q). These do not replace the separated error budgets.

## Consolidated mathematical conclusions

For Hermitian sources and complete positive-gap spectral transitions, write W_n,ab=x_n+i*v_n. Then R_ab(t)=2*sum_n[v_n*cos(Delta_n*t)-x_n*sin(Delta_n*t)], and chi_ab(i*y)=2*sum_n[(y*v_n-Delta_n*x_n)/(y²+Delta_n²)]. Per-state Gram structure gives sum_n|W_n,ab| <= sqrt(Wtot_aa*Wtot_bb). Integrating the uniform response envelope beyond T gives the approved tail bound. For F(t)=exp(-y*t)R_ab(t), |F''| <=2*sum_n|W_n,ab|*(y²+Delta_n²) <=2*sum_n|W_n,ab|*(y+Delta_n)²; use the approved looser expression in the trapezoid bound rather than tune it after execution.

This is the connected commutator response only. Independent diagonal or cross source contacts can change full source response without changing any of these samples. Contacts are not reconstructed and must not be inserted as a first-sample impulse.

For estimated kernel entries k_ab with ideal-record error radii u_ab, define N=k01*k10, D=k00*k11, E_N=|k01|u10+|k10|u01+u01*u10 and E_D=|k00|u11+|k11|u00+u00*u11. Only if |D|>E_D, define Q=N/D and E_Q=(E_N+|Q|E_D)/(|D|-E_D). Equality is inconclusive. This sufficient product-disk test can fail even when individual diagonal disks exclude zero; it does not assert the true denominator vanishes. Do not substitute an absolute-square ratio, clip negative radii or fold floating proxies into the ideal observation disks.

The inherited joint-system builder already supplies both partitions, permitting9 owned Hamiltonian solves across18 partition cases,108 schedule/frequency records and reuse of the one synthetic reference. The final attempt budget, failure-tree schema and synthetic record count must be explicit in the API freeze. No repeated full reports, finite differences, source-norm SVDs per sample, or fresh dense exponentials are planned. API/schema drafting is now underway; scientific controls remain unexecuted.

### Ratio arithmetic policy selected for API review

Targeted static inspection found reusable exact rational complex products/quotients and `substrate_prediction_contract._denominator_resolved(a2,b2,ua,ub)`, an exact radical-free product-disk gate for supplied squared moduli and nonnegative radii. The inherited Decimal modulus/radius evaluation is not outward-rounded, and no directed modulus/square-root helper exists in the reviewed modules. Accordingly, distinguish the exact-input denominator decision from the floating, non-enclosed evaluation of the conditional radius. Report numerical_error_certified=false and explicit non-enclosed radius semantics; do not label the displayed radius a certified numerical enclosure. No new directed-rounding subsystem is introduced. A failed radius conversion must preserve independently available kernel/budget information and exact gate state. This policy does not convert computed spectral or sample-assembly errors into observation bounds.

### Independent draft review checkpoint

The external draft `module6/api-contract-draft.md` has now been independently reviewed against the derivation. Signs, normalization, gain placement, analytic envelopes and nine-system ownership were confirmed by static reasoning only. Four required pre-freeze corrections were identified: distinguish the exact rational quotient from its rounded displayed center; explicitly reject invariance of the absolute-per-state quadrature envelope under degenerate rotations; replace the draft's larger suggested sparse times with the already selected degree32 nodes below; and close all helper/report keys and test tolerances. These are pre-execution contract corrections, not numerical results or post-failure tuning.

The selected ratio convention keeps the conditional radius about the exact supplied-component quotient. Its displayed center is a rounded approximation, and the radius evaluation is non-enclosed. The emitted value/radius pair is therefore not a certified containing disk; no center-displacement or directed-rounding subsystem is added. Independently available quotient/gate information must survive a radius-conversion failure.

An earlier independent report overclaimed degenerate-rotation budget invariance. Only the summed Gram, response, finite/infinite references, total diagonal weights and tail envelope are invariant. The absolute-per-state cross-weight sum, quadrature envelope and associated arithmetic proxy need not be. For an exactly degenerate block, transition rows (1,1),(1,-1) have absolute cross-weight sum2, while an orthogonal rotation to (sqrt(2),0),(0,sqrt(2)) gives0 with the same total Gram2I. The final tests must compare response invariance while checking each basis's own envelope separately.

The revised exact schema passed independent mathematical review after narrowing private gains to real-only matrices, consistent with their serialization. It fixes exact keys, closed statuses/reasons, dependency-specific nulls, and separate owned-case/joint-system/eigensolver attempt counters. Inconclusive denominator disks are valid scientific outcomes, not computational failures. An independently proposed cancellation-safe tolerance table also passed conceptual/algebraic review, explicitly as heuristic comparison allowances rather than numerical certificates. The sparse action fixture now covers both partitions of the same owned L3,g0.7,C1 system at the unchanged three nodes; this adds no eigensolve. Final numerical-policy transcription and permanent contract freeze are still pending. No new production/demo/test files or scientific execution are authorized by this checkpoint. All485 protected inherited files were rechecked with no mismatches.

## Independent oracle design (pre-execution)

Retain real one-pole, complex Pauli, unequal-gap complex (gaps1,3; transition rows(1,1+i),(2,-i)), exactly degenerate rotation, exact zero-column and tiny-nonzero fixtures. Rotate only exactly equal-gap rows; response invariance does not require equality of the absolute-per-state quadrature envelope. Signed gains on the complex Pauli fixture expose incorrect gained-Gram transposition.

For lambda=nu +/- i*delta, exact finite-window scalar integral is F(lambda)=(1-exp(-lambda*Theta))/lambda; the exact composite-trapezoid scalar is J(lambda)=(h/2)*(1-exp(-lambda*Theta))*(1+exp(-lambda*h))/(1-exp(-lambda*h)). Independently assemble both Lehmann terms from these scalars. Stable near-cancellation evaluation must not change the formula or scheduled sampling.

A bounded physical oracle uses independently assembled occupation/source matrices and connected vectors v_a, with A=(H-E0)/C. Compare R_ab(tau)=2*Im<v_a|exp(-i*A*tau)|v_b> to fixed degree32 Taylor action at tau0,2^-10,2^-8, with remainder2*||v_a||*||v_b||*exp(x)*x^33/33!, x=tau*((g/C)*binomial(N,2)+4N). This avoids extra full dense exponentials or eigensolves. These are independent exact-operator relations with separate floating comparison tolerances to be frozen; they do not certify computed eigenvectors.

The free-control oracle has normalized total weight (N/L)*[[1,alpha],[alpha,alpha²+beta²]], alpha=-(1+cos(k)), beta0 for symmetric or sin(k)/2 for improved, active gap4*sin²(k/2). An inactive tiny transition is not removed by this analytic oracle; numerical residuals remain raw.

## Protection and verification provenance

New verification storage is `/Volumes/T9 Backup/bpr-verification/module6/`, outside the repository on T9, with separate artifacts and temporary directories. The inherited snapshot protects485 tracked files under bpr/scripts/tests/doc, excluding only the current campaign ledger. Snapshot `inherited-hashes.json` SHA256 `86a4b664a30707906e41002086c1f92a23ac87cf294681807f4e49e53a6571be`. README may receive the approved new-module link. AGENTS.md and CLAUDE.md remain untouched and unstaged.

Normalized inherited scheduling data is stored in `inherited-plan.json`, SHA256 `d7f4b62754b7868dc480a8ae313a2901f65aac5ff829d51c6a0ddd83ff730755`:34-suite order,4780 ordered node metadata records and114 disjoint exhaustive groups. Preparation verified the original1271-file full-run artifact fingerprint before/after normalization as `adc8fdbd35410cb4a10870a36368cbf7d8f6984fe7bbcff34baf863ad03cbbc7`. Independent preparation review verified all34 suites,4780 unique metadata records,114 ordered groups, portable evidence references, original artifact fingerprint and485 protected hashes. Future execution credit is zero globally and per group; new Module6 counts remain null until fresh collection. No new runner, collection or numerical execution occurred. This clears inherited scheduling data only, not Module6 mathematics or implementation.

Module5's complete fresh-process coverage was4772 passes/8 platform skips across4780 nodes in114 groups. It supplies no execution credit for Module6. The separately failed same-process prefix remains1092 passes/1 failure, reproducing the older adjacent-binary64 discarded_correction discrepancy. Do not silently repair protected older code, retry that bounded check until green, or claim that fresh-process success resolves it. Each new verification attempt must preserve exact ordered selections, outputs, hashes and failures.

No automatic seventh module is authorized. A finite synthetic reconstruction protocol is not empirical validation or completion of a theory of everything.

## Frozen implementation contract

2026-09-13. Coordinator freeze under the already approved campaign plan. The following reviewed API candidate and numerical policy are adopted for implementation. Earlier preliminary/pending language records historical preparation, not current gate state. Within the API material, section10 and revisions2/3 supersede its exploratory schema; the numerical policy supplements and supersedes earlier unspecified tolerance/conversion choices. Its fixed both-partition Taylor fixture takes precedence. Draft/approval-pending labels inside the preserved candidate are historical. No scientific execution has occurred.

Source candidate SHA256: `8ad47ecae436c8aaa18cc830cf148f07de4f794982eb27e37cbe72d04541e200`. Numerical policy SHA256: `ec4331c0025040dc6cb8acca8836671544907a6a9b0b45b0c06775b0bd4aa68f`. Independent mathematical/schema reviewers: a175aaec00a8c6502 and afe4299f80a492a5a. Numerical reviewer: a97d0d50bbff26014, with independent algebraic review of tolerance scales.

### Adopted API candidate (preserved revision provenance)

# Module6 sampled response: draft API and numerical contract

Read-only mathematical review synthesis for coordinator approval. This is a DRAFT, not an executed result or a frozen production contract. No numerical code, imports, tests or controls were executed to prepare it. Approval and a permanent repository derivation freeze must precede execution. Existing computational modules/tests remain unchanged.

## 1. Scope and fixed controls

Keep the complete unit-filled Bose Hamiltonian and existing real standing density/energy sources. No new Hamiltonian, calibration, physical clock, empirical observation, adaptive sampling, fitted spectrum, or seventh module.

Public scientific controls: L=N in (3,4,5), C=1, g/C in (0,.7,40), m=1, partitions symmetric and improved, eta in (.5,1,2). Frequencies use the actual ground gap Delta1, not the first visible source pole or a grouped excitation. Schedules (Theta=CT, intervals) are (4,128) and (8,256). Both have dimensionless step s=1/32. The second schedule is a longer window, NOT finer quadrature. At most257 samples per trace.

There are9 Hamiltonians,18 partition cases and108 physical acquisition records (9*2*2*3). A fully available demonstration calls joint.joint_system exactly9 times and performs exactly9 np.linalg.eigh solves. Each owned system supplies both partitions. The L3,g.7,C1 symmetric reference is reused for12 synthetic records (2 error levels*2 schedules*3 frequencies); no tenth eigensolve. Standalone synthetic_report builds that reference once. No source/Gram eigvalsh, SVD or additional dense exponentials are needed in report production. Independent fixture solvers are separately bounded tests, not hidden report work.

Failure slots count toward the same108+12 prescribed records. Failed cases are never dropped, substituted or retried until passing.

## 2. Definitions, units and endpoint

Use hbar=1 model units, d=(0,1), tau=Ct, delta_n=Delta_n/C, nu=y/C=eta delta1, Theta=CT. Define Obar_a=O_a/C^d_a, ubar_na=<n|Obar_a|0>, Wbar_n,ab=conj(ubar_na)ubar_nb.

Rbar_ab(tau)=C^(-d_a-d_b) R_ab(tau/C)
             =-i sum_n[Wbar_n,ab exp(-i delta_n tau)-Wbar_n,ba exp(i delta_n tau)].

Kbar_ab(i nu)=C^(1-d_a-d_b) chi_ab(i y)
             =integral_0^infinity exp(-nu tau) Rbar_ab(tau) d tau.

Use the CONTINUOUS RIGHT LIMIT at tau=0, Rbar(0+)=-i<[Obar_a,Obar_b]>. Never replace it by a half-theta endpoint; trapezoid endpoint weights already supply the half weight. A half-valued response would introduce an unbudgeted O(s) error for a noncommuting cross source.

The infinite reference is sum_n[Wbar_ab/(i nu-delta_n)-Wbar_ba/(i nu+delta_n)]. The finite reference is

-i sum_n[Wbar_ab (1-exp(-(nu+i delta_n)Theta))/(nu+i delta_n)
       -Wbar_ba (1-exp(-(nu-i delta_n)Theta))/(nu-i delta_n)].

Use actual individual gaps, not cluster centers. Hermitian sources, including complex matrices, give entrywise real Rbar and Kbar(i nu), but a complex-source fixture can be nonsymmetric. Real zero-flux ring sources additionally give reciprocity and -Kbar(i nu) PSD. Neither PSD nor reciprocity applies to arbitrary signed unequal row/column gains. Preserve raw floating imaginary residuals; do not clip them or Q.

The target is the connected commutator kernel of the inherited linear source family, not a contact-completed response. A separate instantaneous source contact cannot be reconstructed from the regular time trace. The inherited diagonal scalar contact changes a full response/Hessian without changing these samples; an arbitrary mixed scalar contact can likewise change cross entries. Neither contact is inferred or added here. The nonzero regular R(0+) of a noncommuting source is not itself a delta-function contact.

## 3. Analytic envelopes and their prerequisites

For exact Hermitian sources and their exact positive semidefinite spectral Gram measure, with all excited gaps positive and nu>0, set T_ab=sum_n |Wbar_n,ab|, Wtot_aa=sum_n Wbar_n,aa. Then

E_tail_ab=2 exp(-nu Theta) sqrt(Wtot_aa Wtot_bb)/nu.
E_quad_ab=(s^2 Theta/12) 2 sum_n |Wbar_n,ab| (nu+delta_n)^2.

The trapezoid acts on f_ab(tau)=exp(-nu tau)Rbar_ab(tau), not on an unweighted R followed by a global regulator. The stated quadrature envelope is conservative: differentiating f twice yields a bound no larger than 2 sum |W_ab|(nu+delta)^2. Do not replace sums of absolute per-state weights by the absolute total cross weight; that would fail under cancellation.

For nonnegative declared complex-modulus sample radii e_jab, define omega_0=omega_N=s/2, other omega_j=s, and

E_obs_ab=sum_j omega_j exp(-nu tau_j) e_jab.
E_total=E_tail+E_quad+E_obs.

The tail and quadrature budgets require model-known complete spectral information. These are not experimental bandwidth estimates. Floating evaluation of exact-model formulas is not an outward-rounded enclosure; eigenpair/source-assembly errors remain unbounded by these formulas. Every availability label below is conditional, never a machine certificate.

Use transitions as the provenance for PSD and positive diagonal weights. joint._spectral_inputs only validates shapes/positive gaps, NOT Hermiticity or PSD. Do not use it as a bound validator for arbitrary Gram matrices. For private fixture inputs, construct all W from the supplied transitions, rather than accept arbitrary indefinite Gram matrices. Rounded Gram eigenvalues, if inspected in tests, must never be clipped or used to certify the exact measure. A negative computed diagonal/budget is a numerical failure, not max(value,0). Exact zero transitions can give exact zero budgets; underflow of a nonzero budget cannot.

The longer schedule decreases E_tail but doubles this E_quad envelope at fixed s. E_obs with a constant positive radius increases toward a limit. No assertion that E_total decreases or that any ratio becomes available is allowed.

## 4. Synthetic records and fixed rounding diagnostic

Dr=diag(2,-3), Dc=diag(5,7), G_ab=Dr_aa Dc_bb=[[10,14],[-15,-21]]. Constant gains act AFTER forming the response: M_ab(tau)=G_ab Rbar_ab(tau). The negative-frequency term is also multiplied by G_ab, not by G_ba. Never set V=G elementwise W and feed V to a Hermitian Gram kernel: unequal signed gains break that convention.

Declare epsilon in (0,1e-8) AFTER gains in normalized observed time-response units. Ideal synthetic observations are

Mideal_jab=G_ab Rbar_ab(tau_j)+epsilon*(-1)^(j+a+b), a,b=0,1.

Use common integer j for the two nested grids, so the shorter ideal record is the first129 points of the longer one. The perturbation is deterministic, real, bounded in complex modulus by epsilon, independent of frequency, and fixed before evaluation. It is not random noise or a measured error estimate. Tail and quadrature budgets multiply by |G_ab|. Observation budgets use epsilon directly, without another gain factor. Save/reuse the noiseless owned trace and retain both error-level report records.

The stated epsilon bounds the IDEAL perturbation. Binary64 formation of stored samples can add rounding beyond epsilon; do not assert |stored-observed minus stored-noiseless|<=epsilon exactly. Keep assembly rounding separate and uncertified.

Freeze u=eps_float64, K=excited_count, M=sample_count, B_ab=sum_n(|Wbar_n,ab|+|Wbar_n,ba|), delta_max=max(delta), and A=sum_j omega_j exp(-nu tau_j). Use the fixed dimensionless heuristic arithmetic-comparison proxy

P_ab=256*u*(K+M+1)*(1+Theta*(nu+delta_max))*A*(|G_ab| B_ab+epsilon).

Physical records use G_ab=1 and epsilon=0. This depends only on prescribed operands and counts, never measured discrepancies. It is deliberately conservative but NOT proved to bound libm, eigenpair or source-assembly errors. Do not add it to E_total under an analytic label, and NEVER include P in the analytic ratio radius. No adaptive adjustment after a mismatch.

For a record with trapezoid estimate Mhat, compare raw complex-modulus residuals using the same P:

window: |Mfinite-Minfinite| versus gained E_tail+P;
quadrature: |Mhat-Mfinite| versus gained E_quad+E_obs+P;
total: |Mhat-Minfinite| versus E_total+P.

Labels are within_diagnostic_envelope/outside_diagnostic_envelope/inconclusive, not certified pass/fail. Record residuals and analytic limits separately. Epsilon0 controls exercise pure quadrature; perturbed controls include E_obs. Comparisons never tune P or override a ratio gate.

If P alone is unrepresentable or otherwise unavailable, preserve available estimates, references and analytic budgets; diagnostic comparisons become inconclusive. An exact zero P is allowed only from exact zero operands, not underflow. No directed-rounding infrastructure is required by this draft.

## 5. Ratio and denominator disk semantics

For a computed center X with conditional nonnegative entry radii U=E_total, define

N=X01 X10, D=X00 X11,
EN=|X01|U10+|X10|U01+U01 U10,
ED=|X00|U11+|X11|U00+U00 U11.

If D=0 in exact supplied-component arithmetic: status zero_denominator, meaning the CENTER product is zero, not proof that the unknown true denominator vanishes. If |D|<=ED: status unresolved_denominator. Equality is unresolved. This conservative product disk can contain zero even when a sharper entrywise analysis could resolve the denominator; no fallback sharpening is adopted.

Only if |D|>ED, Qhat=N/D and

UQ=(EN+|Qhat| ED)/(|D|-ED).

In this theorem Qhat means the EXACT rational quotient q*=N/D of the supplied binary64 center components. The mathematical disk centered at q* with exact formula radius UQ conditionally encloses the ideal infinite-record Q, provided the input radii actually bound the entry errors. The emitted floating center q_out and radius are non-enclosed displays; the theorem does NOT assert an enclosing disk centered at q_out. Quotient rounding is not silently covered by UQ. Include center_evaluation:'rounded_non_enclosed' alongside error_radius_evaluation:'non_enclosed'. Floating evaluation and the model-to-array map remain uncertified. Report conditional_available, never certified or available_heuristic borrowed from the older prediction contract. A zero numerator is valid and may have positive UQ. No additional cross-witness or PSD gate is necessary for defining Q.

Q=X01 X10/(X00 X11), not |X01|^2/(X00 X11) and not a real projection. Constant nonzero G cancels Q for the unperturbed gained kernel, including the finite trapezoid estimate; it does not eliminate finite-window/quadrature bias or perturbation. No empirical gain estimation is performed. Do not apply Q in [0,1] to gain-skewed raw PSD diagnostics or arbitrary complex Hermitian-source fixtures.

Use exact rational arithmetic on supplied finite real/imaginary components for complex products, D=0, division and the radical-free strict product-disk gate. Existing prediction._denominator_resolved is an acceptable pure arithmetic seam after validation. Existing prediction._ratio_proxy may supply the scalar Decimal evaluation, but the new helper owns its observation-bound interpretation and statuses. Irrational moduli/radius are evaluated approximately with bounded high-precision Decimal arithmetic and checked final conversion; Fraction does not make square roots exact and inherited Decimal calculations are NOT directed enclosures. Include error_radius_evaluation:'non_enclosed' and exact_denominator_gate:true on ratio records. The exact gate applies to supplied centers/radii, not certified original-model errors. Existing prediction._joint_ratio/_entry_proxy status or sensitivity semantics are NOT reusable as observation bounds. A nonrepresentable nonzero quotient/radius or exhausted precision gives numerical_unavailable, not zero/infinity. Reuse prediction._computed/_complex_result for checked nonzero-subnormal outputs; no directed-rounding implementation expansion.

## 6. Public API

New module bpr/substrate_sampled_response.py. No edits to old computational modules or tests.

case_report(L, g, C=1.0) -> detached JSON-native case record
synthetic_report() -> detached JSON-native synthetic record
demonstration_report() -> detached JSON-native demonstration record

Case report owns one system, both partitions and the fixed two schedules/three frequencies. There is no public caller-supplied model, system, source, trace, cache, frequency, schedule, gain or observed dataset. C is bounded as below to support unit fixtures, not another scientific scaling sweep.

case_report keys:
- L,N,g,C,m,dimension; dimension null if unavailable before construction.
- status: available_conditional/partially_unavailable/numerical_unavailable; reason string or null.
- ground_gap, normalized_ground_gap, normalized_resolution: finite scalars or null.
- partitions: ordered list symmetric, improved. Each has partition,status,reason,normalized_total_weight,source_metadata,records.
- source_metadata: rho/h each normalized_operator_frobenius_norm and normalized_transition_norm; raw values/null, not a proof of source resolution.
- records: exactly6 acquisition records, schedule outer (4/128 then8/256), eta inner (.5,1,2).
- scope.

An acquisition record has fixed keys:
- partition,schedule:{theta,intervals,sample_count,step},eta,nu,error_level,gains.
- status,reason.
- finite_reference,infinite_reference,estimate: complex2x2 encodings or null.
- errors:{tail,quadrature,observation,total}: real2x2 encodings or null, plus error_status/error_reason.
- arithmetic_proxy,proxy_status,proxy_reason.
- diagnostics:{window,quadrature,total}: each {status,residual,analytic_limit,proxy,reason}; residual and analytic_limit real2x2 encodings/null.
- ratio:{status,value,radius,reason,denominator_margin_status,conditional:true,numerical_error_certified:false}.
- numerical_error_certified:false.

Real2x2 encoding is nested float lists. Complex2x2 encoding is {shape:[2,2],real:[[...]],imag:[[...]]}. Complex scalar is {real,imag}. No NaN/Infinity. Unavailable numerical fields are null. All status records use reason=null when available and a descriptive reason when unavailable. Ratios contain no hidden heuristic witness flag.

synthetic_report keys reference:{L:3,N:3,C:1,g:.7,m:1,partition:'symmetric'},readout_gains:[2,-3],source_gains:[5,7],error_levels:[0,1e-8],perturbation_definition,status,reason,records,scope. records are exactly12, error level outer, schedule middle, eta inner. Do not return all dense operators or all sample arrays publicly.

demonstration_report keys module,physical_cases (exactly9 case records in L outer/g inner order),synthetic,counts:{hamiltonian_slots:9,partition_slots:18,physical_records:108,synthetic_records:12,max_samples:257},limitations,scope. Scope includes empirical_status:'empirical_test_unavailable',empirical_validation:false,numerical_error_certified:false,analytic_bounds:'exact-model conditional formulas; floating evaluation not enclosed',assembly_proxy:'heuristic, excludes eigensystem certification',contacts:'not reconstructed',clock:'model C units only'. Counts are prescribed slots, not claimed successful computations.

## 7. Private fixture seams and ownership

Required fixture helpers (names/signatures to freeze before test authoring):

_time_trace(gaps, transitions, times) -> owned complex array shape(M,2,2).
Inputs are dimensionless excited-only gaps (K,), normalized transitions (K,2), and times (M,). It constructs Gram entries from transitions and returns the right-limit endpoint. No eigensolver. Use bounded modal sums rather than dense propagators.

_acquire(gaps, transitions, nu, theta, intervals, gains, error_level) -> owned internal acquisition data including sample times, noiseless samples, ideal-perturbation samples, estimate, finite/infinite references, separate budgets/proxy and ratio. This is a private finite-array fixture boundary, not arbitrary public science input. Internal orchestration may supply an already-owned trace through a separate nonexported function to avoid recomputing traces per nu. No caller cache is trusted as provenance.

_ratio_disk(kernel, entry_error) -> the ratio record above. Finite shapes(2,2), entry_error strictly real/nonnegative. It performs no source, PSD or cross-witness gating and no eigensolver. Input radii are stipulated conditional bounds, never prediction-contract proxies.

_owned_case(L,g,C) -> internal owned system and report-building data. It calls joint.joint_system once. Extract excited-only gaps as system['gaps'][1:]/C; the exported full gaps vector contains ground0, while transitions are already excited-only. Normalize transition columns by [1,C] and actual source matrices with d=(0,1). Use joint._kernel as a reference arithmetic seam and prediction._normalize_matrix(...,extra_degree=-1) for Kbar, not joint report's differently defined source-coordinate 'dimensionless' Hessian/kernel labels. Do not call joint.joint_report or prediction.case_report to obtain data; they add work/status assumptions. Reuse the reference owned case in demo synthetic orchestration.

All fixture arrays must be finite real/complex numeric arrays/lists/tuples with validated leaves and rectangular shapes, bounded before allocation. Reject bool, strings, object/custom conversion, ragged arrays, complex storage in real-only inputs, and longdouble/complex extended precision wider than float64/complex128. Builtin/NumPy integer inputs are accepted only when exactly representable as binary64 after checked conversion. No caller array aliases, mutation or global cache; reports are fresh JSON trees. Existing joint exports are detached; owning the returned object does not mean trusting arbitrary public dictionaries.

Dimension limits:1<=K<=511,1<=M<=257, total input component cap512^2. gaps strictly positive; transitions shape(K,2); times nonnegative strictly increasing, with singleton time allowed for endpoint fixture. nu/theta strictly positive finite real, intervals builtin/NumPy integer1..256 excluding bool, gains shape(2,2) finite real-only numeric; reject complex storage even when all imaginary components are zero. Error level finite real>=0. Private zero gains are valid stress fixtures and lead to denominator unavailability where appropriate; public gains are fixed nonzero. Time controls and production coefficient arithmetic use binary64 only, except exact rational/high-precision scalar guards. No extended scientific domain through helper knobs.

Public L integer3..5, C finite real in[.5,2], g finite real>=0 with g/C<=40, m fixed1. Reject bool/complex/string/custom scalars. Preserve inherited 0<g/C<2^-40 as numerical_unavailable, not g=0 and not invalid input. Check binomial(2L-1,L) against this module's512 cap before constructing a model. Do not allocate a basis before validating parameters/cap.

## 8. Failures and diagnostic acceptance

Programmer/malformed-domain inputs raise ValueError. Valid-domain unrepresentable arithmetic, failed owned eigensystem screen, unresolved ground gap, negative computed radius/diagonal weight, overflow, nonzero underflow or insufficient bounded scalar precision are numerical unavailability. Catch inherited NumericalUnavailable before broader ValueError because it is a subclass. Do not broadly convert all exceptions into unavailable outcomes.

Public wrappers retain all required descendants after a numerical construction failure, filling metadata/schedule/eta/error-level fields and null dependent values with reasons. Failure before a normalized gap exists leaves nu null, not zero. Independently known schedule metadata and empirical limitations remain present. A failed synthetic reference leaves all12 synthetic records, not an absent synthetic block.

Once references or estimates exist, preserve them if only budgets, proxy or ratio fail. A failed ratio never erases an available trace estimate. A failed proxy makes comparisons inconclusive but need not prevent a conditional ratio based on available analytic E_total. An unavailable analytic budget prevents dependent ratio inference even if its raw center is finite. A synthetic assembly failure affects that error-level record, not the noiseless reference. Case/top-level aggregate statuses must retain partial failures instead of calling the entire case available.

Exact structural zeros require an analytic identity or exactly zero private stimulus. Numerical smallness, negative roundoff, apparent rank deficiency, small imaginary residue, or an unresolved ratio does not prove a structural zero. No negative-radius clipping, favorable interval choice, adaptive quadrature, extra frequency or new model-gap bound after observing results.

## 9. Independent fixtures to freeze before execution

1. Complex two-level source: H=diag(0,delta), O1=sigma_x, O2=sigma_y, transitions=(1,i). W12=i, R11=R22=-2sin(delta tau), R12=2cos(delta tau), R21=-2cos(delta tau), including R12(0)=2. K11=K22=-2delta/(nu^2+delta^2), K12=2nu/(nu^2+delta^2), K21=-K12, Q=-nu^2/delta^2. It catches sign, transpose, endpoint and false PSD assumptions.

2. Exact cancellation with unequal gaps: transitions(1,1) at delta1 and (1,-1) at delta2. Total cross weight is zero but its time response and finite/imaginary-frequency response generally are not. Per-state absolute weights must remain in E_quad. Add exactly degenerate states and a unitary rotation to check response/total-Gram invariance; never rotate unequal-gap states together. The sum of absolute per-state cross weights, and hence this quadrature envelope, need NOT be invariant under rotations inside a degenerate eigenspace. Do not assert that stronger false invariance.

3. Free symmetric m1 rank-one identity and improved free Q=alpha^2/(alpha^2+beta^2), normalized alpha=-(1+cos k), normalized beta=sin(k)/2. Compare closed scalar finite-window/trapezoid sums rather than assuming all free partitions give Q=1. A separate dark-channel private fixture must preserve zero-denominator status without numerical clipping.

4. Signed gains [[10,14],[-15,-21]] on the complex two-level fixture. Verify gain multiplication follows the completed response, not transformed Gram transposition; verify Q unchanged noiselessly. No observed PSD/reciprocity assertion.

5. Scalar modal quadrature: compare direct trapezoid against independent finite geometric sums for exp(-(nu plus/minus i delta)j s), including endpoint weights, nearly cancelling numerator, and short theta. No helper-under-test as sole oracle.

6. Alternating perturbation closed form: for even N, q=exp(-nu s), the ideal observation-only trapezoid shift is epsilon*(-1)^(a+b)*(s/2)*(1-q^N)*(1-q)/(1+q). The constant-radius worst-case E_obs is epsilon*(s/2)*(1-q^N)*(1+q)/(1-q). Test using an independent high-precision scalar oracle; evaluate near q=1 stably. These characterize cancellation, not replacement of the approved worst-case budget. The two prescribed N are even.

7. Denominator disk: exact center zero; equality |D|=ED; just above/below equality; zero numerator; valid subnormal components; overflowing/underflowing intermediate D with finite Q; nonrepresentable final Q/radius; and a conservative product disk unresolved even when each entry disk separately excludes zero. Keep equality exact in supplied-component arithmetic.

8. Array/API validation: booleans hidden among numeric leaves, extended precision, ragged/oversized arrays, negative/complex errors, zero/negative gaps, NaN/Infinity, zero frequency, invalid interval count, out-of-range C/g, positive tiny-g unavailable, cap-before-allocation, readonly/mutation detachment and independent repeated reports.

9. Instrumentation: fully available demo exactly9 joint calls/eigh calls, both partitions per owned system, no synthetic extra solve, no dense time exponentials/SVD, maximum257 samples, exactly108+12 record slots. Inject base/reference/kernel/budget/proxy failures and check retained slots and dependency-specific nulls without running a larger control grid.

10. Independent Hamiltonian-action short-time Taylor checks on BOTH symmetric and improved partitions of the single already-owned L3,g.7,C1 reference use degree32 and tau=(0,2^-10,2^-8), acting on normalized connected source-ground vectors v_a with (H-E0)/C. Freeze x=tau*((g/C)*binomial(N,2)+4N). The absolute response remainder is at most 2||v_a|| ||v_b|| exp(x) x^33/33!. This follows from ||(H-E0)/C||<= (g/C)*binomial(N,2)+4N and the exponential-series norm remainder. At x=0 the analytic remainder is exactly0. Add only the separately frozen arithmetic-comparison tolerance for floating oracle comparison; do not relabel it part of the Taylor theorem. Do not use a fresh full dense exponential at every sample, a larger node, or another eigensolver sweep.

11. Contact non-identifiability fixture: algebraically add a fixed diagonal scalar quadratic source contact, then a fixed symmetric cross contact to the full response target; leave all connected time samples unchanged. This is a diagnostic distinction, not adoption of a new physical action. No new dense solve.

Final approval must adopt the exact schema supplement below and the independently reviewed numerical tolerance/conversion policy before separate implementation and independent tests. None of the above constitutes empirical validation or a numerical result.

## 10. Implementable schema supplement, revision 1

This supplement records coordinator-requested corrections to the original draft, before implementation or numerical execution. The original review provenance and equations above remain; where an API/schema statement conflicts, THIS supplement supersedes it. No observed numerical outcome informed the revision. Mathematical review additionally confirms that per-state-absolute quadrature envelopes and their arithmetic proxies need not be invariant under a degenerate-state rotation, although response/total Gram are invariant.

### 10.1 Types and representation

Public L must have type(L) is int; public g,C must have type(value) in (int,float). Thus bool, numeric subclasses, NumPy scalars, strings and conversion objects are rejected publicly. Builtin integers must fit finite binary64 and convert exactly: Fraction(float(value))==Fraction(value); failure is ValueError before an owned-system call. Builtin floats must be finite. After exact conversion, use inherited bounded C/g domain and compute g/C domain comparisons without overflowing products. Private fixtures retain numeric ndarray/list/tuple containers with separately checked builtin and NumPy numeric leaves, excluding bool/object/extended precision and rejecting inexact integer conversion. Check own stricter types BEFORE inherited prediction._array_input; its broader acceptance is not this module's contract.

Every listed JSON key is required, and each listed tree has exactly that key set: no additional machine-readable keys are permitted without a preregistered schema revision. Reasons consist of reason (closed code or null) and detail (descriptive string or null), not arbitrary status strings. Public arrays use the encodings specified above. Internal arrays are owned complex128 or float64 arrays, never encoded JSON.

Closed reason codes:
base_system_unavailable, tiny_g_unavailable, ground_gap_unresolved, normalized_data_unavailable, trace_unavailable, sample_assembly_unavailable, reference_unavailable, estimate_unavailable, envelope_unavailable, arithmetic_proxy_unavailable, denominator_center_zero, denominator_disk_contains_zero, ratio_arithmetic_unavailable, dependency_unavailable, nonfinite_arithmetic, nonzero_underflow, negative_computed_bound, precision_exhausted.

Malformed caller inputs raise ValueError rather than a reason-coded report. NumericalUnavailable caught only around owned construction/arithmetic is represented by its relevant stage reason and original diagnostic detail. A reason does not alter prescribed metadata. Deepest cause is preserved in detail when a dependency record uses dependency_unavailable.

### 10.2 Ratio exact tree and precedence

_ratio_disk(kernel, entry_error) returns exactly:
{status,value,radius,reason,detail,denominator_margin_status,conditional,numerical_error_certified,exact_denominator_gate,center_evaluation,center_convention,error_radius_evaluation}.

Constants: conditional=true; numerical_error_certified=false; center_evaluation='rounded_non_enclosed'; center_convention='exact_quotient_radius_rounded_display'; error_radius_evaluation='non_enclosed'. The serialized value/radius pair is NOT a certified containing disk. No center-displacement radius expansion or additional modulus pipeline is adopted. value is complex-scalar display or null; radius is nonnegative float or null.

status enum: conditional_available, zero_denominator, unresolved_denominator, numerical_unavailable, dependency_unavailable.

denominator_margin_status enum: center_zero, contains_zero, excludes_zero, not_evaluated. exact_denominator_gate is true IF the exact supplied-component center/product gate completed, otherwise false. Do not reset a completed gate if later quotient/radius conversion fails.

Precedence: absent dependency -> dependency_unavailable/not_evaluated, reason dependency_unavailable; exact D=0 -> zero_denominator/center_zero, reason denominator_center_zero; |D|<=ED -> unresolved_denominator/contains_zero, reason denominator_disk_contains_zero; strict gate passes -> excludes_zero. Then quotient and radius evaluation are attempted separately in order. If quotient converts, preserve value even when radius fails; if a quotient component cannot convert, value=null and radius=null, but retain excludes_zero and exact_denominator_gate=true. A radius failure sets radius=null. Either arithmetic failure gives numerical_unavailable with reason ratio_arithmetic_unavailable and specific detail. Available result has conditional_available, value/radius finite, reason/detail null. No cross-witness flag or structural-dark override is inferred from numerical data.

zero_denominator and unresolved_denominator are valid INCONCLUSIVE SCIENTIFIC outcomes, not computational case failures. A zero numerator with an excluded denominator remains conditional_available. The exact gate and the conditional theorem remain distinct from the non-enclosed floating display.

### 10.3 Acquisition exact JSON tree

Each record has exactly:
{partition,schedule,eta,nu,error_level,gains,status,reason,detail,finite_reference,infinite_reference,estimate,errors,error_status,error_reason,error_detail,arithmetic_proxy,proxy_status,proxy_reason,proxy_detail,diagnostics,ratio,numerical_error_certified}.

schedule exactly {theta,intervals,sample_count,step}; gains is the real2x2 G matrix. physical records error_level=0,gains=[[1,1],[1,1]]. nu may be null only when its required gap/arithmetic is unavailable. eta remains the prescribed value. partition is symmetric/improved.

status enum: available_conditional, partially_unavailable, numerical_unavailable. It refers to computation, NOT successful ratio conditioning or diagnostic agreement. available_conditional means the requested estimates/references/budgets/proxy/diagnostics completed and ratio is conditional_available/zero_denominator/unresolved_denominator. partially_unavailable means some required numerical fields completed but a computational dependent failed. numerical_unavailable means no estimate or reference completed. Ratios unresolved and outside_diagnostic_envelope comparisons do not change computational status.

errors exactly {tail,quadrature,observation,total}; each value real2x2 or null. Preserve each successfully completed component if a later budget fails. error_status enum available_conditional/partially_unavailable/numerical_unavailable; all components completed -> available_conditional, some -> partially_unavailable, none -> numerical_unavailable. error_reason/error_detail null iff all complete; otherwise envelope_unavailable plus diagnostic detail. total is null unless all three components and their checked sum succeed. Negative computed entries invalidate that budget without clipping. Per-sample scalar epsilon is the only declared observation-error input seam; do not add arbitrary error arrays.

proxy_status enum available_heuristic/numerical_unavailable/dependency_unavailable. arithmetic_proxy is real2x2 or null. A failed proxy alone leaves errors/ratio intact. proxy_reason is arithmetic_proxy_unavailable or dependency_unavailable as appropriate, and proxy_detail explains the cause.

diagnostics exactly {window,quadrature,total}. Each diagnostic exactly {status,entry_status,residual,analytic_limit,proxy,reason,detail}. status enum within_diagnostic_envelope/outside_diagnostic_envelope/inconclusive. entry_status is a2x2 nested string array with the same enum. residual is real2x2 or null; analytic_limit is real2x2 or null; proxy is real2x2 or null. If all required arrays exist, classify entries by residual<=analytic_limit+P and aggregate any outside -> outside_diagnostic_envelope, else within_diagnostic_envelope. If any required whole array is unavailable, all entry statuses are inconclusive and aggregate inconclusive, with reason dependency_unavailable. This first implementation does not need partial per-entry arithmetic recovery inside a2x2 array. Retain a completed residual even if its limit or proxy is unavailable. Known outside comparison is never reclassified as numerical unavailability; it remains a retained verification discrepancy.

Top-level reason/detail null iff available_conditional. If multiple computational failures occur, select first in the fixed stage order base, normalized data, trace, synthetic assembly, reference, estimate, envelope, proxy, ratio; preserve other failures in their dedicated records/details. status aggregation never substitutes a misleading success for missing fields. The errors/ref/estimate computations should run independently where their inputs remain available; this stage priority is reporting precedence, not a mandate to abandon subsequent independent work.

### 10.4 Case/synthetic/demo exact trees

Case exactly {L,N,g,C,m,dimension,status,reason,detail,ground_gap,normalized_ground_gap,normalized_resolution,partitions,scope}. No separate case-level eigenvectors/operator dumps.

Partition exactly {partition,status,reason,detail,normalized_total_weight,source_metadata,records}.
source_metadata exactly {rho:{normalized_operator_frobenius_norm,normalized_transition_norm},h:{normalized_operator_frobenius_norm,normalized_transition_norm}} with finite scalars/null. normalized_total_weight is complex2x2 or null; a small/zero norm does not silently invalidate subsequent computations. records exactly6 in prescribed order. Partition status uses acquisition computation status: all records available and metadata complete -> available_conditional; no completed numerical metadata or acquisition estimate/reference -> numerical_unavailable; otherwise partially_unavailable. Case uses the same rule across partitions plus base metadata. Reason/detail choose first failed required stage/partition/record in fixed order. Unresolved ratios are not failures.

Synthetic exactly {reference,readout_gains,source_gains,error_levels,perturbation_definition,status,reason,detail,records,scope}. reference exactly {L,N,C,g,m,partition}; other values/order as above. perturbation_definition is the fixed string 'ideal postgain delta[j,a,b]=epsilon*(-1)^(j+a+b); common j across nested schedules'. records exactly12. Status/reason/detail aggregate computation as for a case.

Demo exactly {module,physical_cases,synthetic,counts,limitations,scope}. module='substrate_sampled_response'; counts exactly the five prescribed slot-count keys above. limitations is a list of fixed explanatory strings, never an inferred empirical result. No overall demo status is needed: preserve all child statuses.

Every scope object exactly {empirical_status,empirical_validation,numerical_error_certified,analytic_bounds,assembly_proxy,contacts,clock} with fixed values stated above. Physical/synthetic record numerical_error_certified=false.

### 10.5 Internal private exact seams

_time_trace(gaps,transitions,times) has the signature/array return defined above. No errors/gains or report dict are added to this pure helper. Raises ValueError for malformed fixture inputs and NumericalUnavailable for valid-input numerical evaluation failure.

_acquire(gaps,transitions,nu,theta,intervals,gains,error_level) returns exactly:
{times,noiseless_samples,samples,sample_status,sample_reason,sample_detail,record}.

times: owned float64 shape(N+1) or null if grid construction fails. noiseless_samples: gained owned complex128 shape(N+1,2,2) or null. samples: floating realization of ideal gained-plus-perturbation samples, same shape or null. sample_status enum available/numerical_unavailable; sample_reason null or trace_unavailable/sample_assembly_unavailable; sample_detail string/null. record is the exact acquisition JSON tree above EXCEPT orchestration-only partition and eta are null in this private helper. nu and schedule are supplied; no gap-derived eta is inferred. Private helper still requires theta>0,nu>0,intervals1..256 and the prescribed array caps, but it need not restrict fixture schedules to the two scientific schedules. Gains may be zero and need not factor into rows/columns in this finite-array arithmetic fixture; no gain-invariance claim is inferred from arbitrary inputs.

_acquire owns and validates its inputs and does no model construction/eigensolve. References, envelopes and trace evaluation are independent computations; a time-trace failure need not erase available finite/infinite references or model-known budgets. Numerical stage failures return reason-coded partial internal data/record rather than dropping the acquisition. Caller-input ValueError still raises.

Private orchestration may use an additional nonexported _acquire_owned with owned data to compute the maximum257-point trace once per partition, then slice its129-prefix and reuse it across all three nu and both synthetic epsilon levels. Do not pass a caller-exported report/array as a trusted cache. A shared maximum-trace failure is retained for its affected schedules/frequencies; do not retry a shorter trace merely to obtain a passing slot. Frequency-only arithmetic failure does not invalidate the owned trace or other frequencies. The scientific requirement is at most one maximum trace per partition per owned case; helper direct calls remain independent fixtures.

_owned_case(L,g,C) returns exactly {system,normalized,report}.
system is the internally owned detached joint_system result or null.
normalized is null on whole normalization failure, otherwise exactly {gaps,ground_gap,resolution,partitions}, with excited-only owned float64 gaps, scalar ground_gap/resolution, and partitions exactly {symmetric:DATA,improved:DATA}. DATA exactly {sources,transitions,total_weight,maximum_times,maximum_trace,trace_status,trace_reason,trace_detail}; arrays sources(2,d,d),transitions(K,2),total_weight(2,2),maximum_times(257),maximum_trace(257,2,2). maximum_trace may be null with numerical_unavailable trace status; maximum_times may be null only if grid construction failed. sources/total_weight are in normalized units. system retains H/ground needed by independently authored action fixtures, not by public report consumers. Normalize source-partition data once; if partition normalization fails, that DATA is null and its public partition records remain explicit failures while the other partition may proceed. normalized.gaps failure makes normalized=null. report is the exact public case tree including all failures.

_owned_case validates public parameters before the owned call; malformed input raises ValueError. It catches NumericalUnavailable from its single joint_system attempt and returns system=null,normalized=null plus the fully populated unavailable case report. It does not catch arbitrary ValueError from valid-but-unexpected implementation paths. Demo calls it at most9 times, once per prescribed Hamiltonian, with no retry. Standalone case_report returns only its detached report. Standalone synthetic_report calls it once for the reference; demo synthetic assembly uses the already-owned reference object, including failure, and never calls it again.

### 10.6 Inherited arithmetic boundary clarification

The private fixture domain permits representable binary64 subnormals. joint._kernel/_grams and current._normal have a stricter inherited nonzero-subnormal policy, so they are NOT valid generic fixture arithmetic backends. The new infinite-reference evaluation owns the short two-term exact-rational kernel contraction and uses prediction._computed/_complex_result for checked output. Construct Gram contributions from supplied transitions with exact component products; do not first materialize underflowing products merely to call a checked converter. Exact denominator/kernel algebra may succeed even when an intermediate binary64 Gram would not. A required final nonrepresentable output remains numerical_unavailable. No old helper/module is changed.

For ordinary physical data within its documented input domain, joint._kernel may additionally be reused as a reference consistency check; it is not a required extra calculation per fixture/record and cannot certify the new numerical result. Inherited refusal of valid new-domain fixture values must never leak out as malformed-input ValueError. Use the owned small formula rather than expand the old validator.

Before delegating to joint.joint_system, evaluate public positive g/C<2^-40 with exact rational arithmetic on validated original inputs. _owned_case returns a fully populated numerical_unavailable case with reason tiny_g_unavailable,system=null,normalized=null and zero joint calls for that case. This precedes inherited _component, whose subnormal rejection could otherwise obscure the intended legitimate tiny-g status. g=0 is a valid separate branch, never obtained by snapping or conversion underflow.

### 10.7 Coordinator clarification, revision 2

This second pre-execution revision narrows private gains to real-only finite shape(2,2), including arbitrary signed or zero real gains. Reject complex dtype/storage even if imaginary components are all zero. No complex-gain API is introduced. The output gains field is therefore always real2x2. The pure ratio helper continues accepting complex kernel entries as separately specified.

All schema key lists are exact key sets, not minimum required subsets. The counts object reports prescribed SLOTS only; it must never imply successful computations or observed solver counts. Instrument _owned_case entry, joint.joint_system entry and np.linalg.eigh entry separately. On a fully available demonstration each count is9. Under injected/legitimate failures, _owned_case is still called once for each of the9 prescribed Hamiltonian slots, while joint/eigh counts can be lower depending on failure stage; neither may exceed9. A failed call is an attempt and is never retried. Synthetic reuse incurs zero additional calls at all three levels. A monkeypatched early _owned_case failure, before the function can construct its wrapper, must be caught as NumericalUnavailable by the fixed demo slot wrapper and converted to the same prescribed unavailable descendants; unexpected ValueError remains uncaught. Standalone synthetic builds its reference once, independent of any previous public call.

### 10.8 Sparse-oracle clarification, revision 3

Before execution, the coordinator selected both symmetric and improved partitions for the sparse Hamiltonian-action oracle on the same already-owned L3,g.7,C1 reference. Independent derivation/numerical reviewers reviewed this clarification. Degree32, times(0,2^-10,2^-8), and the remainder formula are unchanged. This adds no Hamiltonian, eigensystem, scientific grid point, or adaptive choice.

### 10.9 Remaining external freeze dependency

All scientific controls, equations, counts, sparse-action degree/nodes/remainder, API keys/statuses, ownership and failure ordering are selected above. Numerical comparison tolerances and exact overflow/underflow/conversion policy are being independently reviewed by the separate numerical reviewer; incorporate that single reviewed policy verbatim before freezing the permanent derivation. Do not invent or tune a tolerance in this draft, and do not execute until that gate and coordinator approval are complete. Checked final conversion cannot recover underflow already lost in intermediate float operations; implementation must follow the reviewed safe-evaluation policy, not merely wrap a damaged result in _computed.


### Adopted numerical policy

# Module6 numerical policy selected before implementation

2026-09-13. Coordinator adoption of the independent static numerical review by agent a97d0d50bbff26014. The tolerance envelopes were independently checked algebraically by afe4299f80a492a5a. No scientific imports, experiments or tests informed these choices. This is a heuristic floating-comparison policy, not numerical certification. It supplements API draft revision3; the permanent repository freeze must precede implementation.

## Fixed independent comparison table

u=2^-52 is machine epsilon, q0=2^-1074. Tol(F,S)=256*u*F*S+8*F*q0. Represent q0 as Fraction(1,2**1074). Lift finite binary64 operands exactly and compare without first rounding the complete tolerance to binary64. Complex residual comparisons may use exact squared component residual versus squared allowance. Moduli used in scale construction are approximate, not directed enclosures.

K is excited-transition count, M=intervals+1, B_n,ab=|W_n,ab|+|W_n,ba|, B_ab=sum_n B_n,ab. A_h=sum_j omega_j exp(-nu*tau_j), with trapezoid weights including the step. A_c=(1-exp(-nu*Theta))/nu, evaluated with -expm1(-nu*Theta)/nu. G=1 for ungained entries. All scales use pre-cancellation operand magnitudes, not residuals or final-answer magnitudes.

| Comparison | F | S |
|---|---|---|
| Modal/Pauli time at tau | (K+1)*(1+delta_max*tau) | abs(G_ab)*B_ab |
| Finite-window reference | (K+1)*(1+Theta*(nu+delta_max)) | abs(G_ab)*B_ab*A_c |
| Infinite reference | K+1 | abs(G_ab)*sum_n B_n,ab/sqrt(nu^2+delta_n^2) |
| Trapezoid/geometric oracle | (K+M+1)*(1+Theta*(nu+delta_max)) | A_h*(abs(G_ab)*B_ab+epsilon) |
| Isolated alternating noise | (M+1)*(1+nu*Theta) | epsilon*A_h |
| Observation scalar envelope | (M+1)*(1+nu*Theta) | epsilon*A_h |

Two independently rounded results use the sum of their allowances, including degenerate-basis responses (each basis uses its own B). Noisy-minus-noiseless on a nonzero background uses the sum of the two FULL estimate allowances, not a noise-only allowance. Require an isolated zero-background noise oracle to prevent swallowed-noise vacuity. Independent transcendental scalar oracles use fixed100 decimal digits, not precision tuned after discrepancies, and never call the production finite-factor/summation helper as their oracle. High precision remains non-enclosed.

No max(1,S), max(1,abs(result)), universal absolute tolerance or fitted residual scale. Exact zero-column/gain/error/endpoint/rational-ratio fixtures require their prescribed exact outcomes separately. Designated analytically nonzero tiny fixtures also require nonzero output and prescribed sign/component. Exact rational fixtures compare against the independently implemented exact mathematical expression followed by checked conversion. Exact cancellation of approximate transcendental operands is a numerical zero, not a structural theorem.

## Sparse physical action oracle

Use both partitions of the ONE already-owned L=N=3,g=.7,C=1 reference. Degree32, tau=(0,2^-10,2^-8). Independently assemble connected normalized source-ground vectors v_a and A=(H-E0)/C. x=tau*((g/C)*binomial(N,2)+4N), S_ab=2*norm(v_a)*norm(v_b). The theorem remainder is S_ab*exp(x)*x^33/33!, exactly0 at tau0. Keep it separate from the arithmetic allowance Tol(F,S_ab) with F=(33*(d+1)+K+1)*(1+x)*exp(x), d=sector dimension. Compare raw complex residual against remainder plus allowance; neither certifies ground/source/eigensystem/libm arithmetic.

Invocation limits: two physical _time_trace calls with the three times, one per partition; six time-indexed2x2 comparisons; at most256 Hamiltonian-action matrix-vector products (2 partitions*2 vectors*2 nonzero times*32 actions). No additional model/joint/eigensolver/dense exponential/SVD. No new case or node.

## Safe arithmetic and conversion

Exact supplied-component lifting for Gram products/sums, rational kernel contractions/denominators, gains, normalization, positive-factor products/sums where intermediate loss can occur, ratio products/division/gate, and public tiny-g classification. Checked conversion rejects nonfinite results, exact nonzero components rounded to zero, and required final nonrepresentable results. Representable subnormals are accepted. A converter after an already underflowed float product is insufficient. Exact lifting does not make sin/cos/exp/expm1 exact.

Each REQUIRED phase product is first exact-lifted, then checked finite/nonzero-preserving and abs(phase)<=2**20. Otherwise report numerical unavailability using the existing stage reason and phase-range detail, not malformed-input ValueError. No manual reduction, zero substitution or retry. This fixed engineering range is not a libm theorem. Approved controls satisfy delta<=420,Theta<=8 by the operator bound, without evaluating controls.

For required exp(-a), a>=0: a0 gives exactly1. Positive a requires nonzero-preserving conversion; returned exponential0 is numerical unavailability, never structural zero. Rounded exponential1 for tiny positive a is allowed, but differences from1 must use expm1. -expm1(-a) is known positive for positive a and cannot be accepted as0. An exact zero stimulus may bypass irrelevant factors, never a merely small stimulus.

Stable finite-window numerator for a=nu*Theta,b=delta*Theta:
1-exp(-a-i*b) = -expm1(-a)+exp(-a)*(1-cos(b)) + i*exp(-a)*sin(b).
Use 1-cos(b)=sin(b)^2/(1+cos(b)) for abs(b)<=1 and 2*sin(b/2)^2 otherwise. Exact-lift returned transcendental operands before products/divisions, including squares. Use analogous stable geometric denominator at the step. Real observation/noise factors use expm1 for 1-q and 1-q^N. Required exponential underflow can invalidate finite/weighted stages but must retain independently available rational infinite reference and other data.

## Ratio

Keep center_convention='exact_quotient_radius_rounded_display', center_evaluation='rounded_non_enclosed', error_radius_evaluation='non_enclosed', numerical_error_certified=false. Radius refers to exact supplied-component q*, not certified containing disk about displayed rounded center. No center-displacement expansion. Radical-free exact gate retains strict sign conditions before squaring; equality unresolved. Reuse inherited approximate radius precision80 doubling through10240 then unavailable; not directed rounding. Preserve completed gate and converted quotient on later radius failure. No proxy/test allowance in ratio radius.

## Validation and attempts

Private containers: BASE ndarray, BUILTIN list/tuple only; reject subclasses/custom conversion before coercion. Validate leaves/shapes before bounded allocation. Numeric scalar leaf domain follows draft, excluding bool, object and precision wider than float64/complex128; integer conversion exact. Private intervals builtin int or NumPy integer excluding Python/NumPy bool,1..256. Public L builtin int only; public g,C builtin int/float only.

Demo exactly9 owned-case slot calls, at most9 joint-system and9 eigh attempts, exactly9 each when fully available. Synthetic reuse no extra call; standalone synthetic one owned attempt. At most one257-sample maximum trace per partition per owned case. Retain108 physical plus12 synthetic slots through failures. Private modal/Pauli/geometric/noise/ratio fixtures have zero model/eigensolver calls. No test-node count asserted before authoring/collection.

## Production proxy unchanged

P_ab=256*u*(K+M+1)*(1+Theta*(nu+delta_max))*A_h*(abs(G_ab)*B_ab+epsilon).
The q0 term belongs ONLY to independent comparison tolerances, not P, analytic budgets or ratio radius. Nonrepresentable nonzero P makes diagnostics inconclusive, preserving estimates/references/budgets/ratio. P is heuristic and does not certify eigensystem or libm.

## Review disposition

Ready to incorporate into permanent preregistration and assign separate implementation/test authors. This is static contractual clearance, not execution evidence. All prior protected computational files remain unchanged. No seventh module, new physics, empirical validation or complete TOE claim.

## Pre-execution clarification A: unrepresentable private grid step

During independent authoring, the production author identified a valid private theta equal to the smallest positive binary64 value with intervals>1: its positive rational step cannot be represented in binary64. Independent mathematical/schema reviewer afe4299f80a492a5a approved this narrow clarification before any execution. Original freeze SHA256 was `ef22910d2fb42200dd213fa283d9db88dfbb472ea149fd9228ceb2a0300df108`. No equation, control or tolerance changes.

`schedule.step=null` only when the required positive theta/intervals fails checked binary64 conversion; never serialize it as zero. Retain theta, intervals and sample_count=intervals+1. Set times and both sample arrays null, sample_status=numerical_unavailable, sample_reason=trace_unavailable, and the grid-conversion cause in sample_detail. The sampled estimate is unavailable. Independently attempt/preserve finite/infinite references and each analytic budget wherever its mathematical inputs remain available; exact rational step arithmetic does not imply construction of a binary64 grid. All prescribed public schedules retain step=1/32 even on base-system failure.

The numerical-policy transcription was independently audited against the original review: fixed100-digit independent transcendental test oracles remain unchanged. An intervening reviewer mention of80-digit independent oracles was withdrawn; inherited production norm/radius precision remains separately specified. Exact grid endpoints/order and overflow-safe diagnostic-limit comparison clarify the existing safe-arithmetic rule. No scientific controls have run.
