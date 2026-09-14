# Sampled response: verification and retained acceptance failure

2026-09-13. Companion to the [frozen derivation and implementation contract](substrate_sampled_response_2026-09-13.md) and [six-module campaign ledger](toe_connection_campaign_2026-09-13.md). The contract is left byte-for-byte unchanged so its execution binding remains verifiable.

## Disposition

**The calculation and scheduled verification have run, but Module 6 acceptance is not established.** A specifically adopted supplemental tiny-noise fixture failed four nonzero/sign assertions. Its passing error-size comparisons do not waive that separate requirement. After being informed of this failure, the user explicitly requested commit, push and merge to main. That request authorizes repository integration with the failure retained; it does not satisfy or waive the failed scientific criterion. The original publication hold is superseded by this request. Independent final regression audit verified 4,945 passes and 10 platform skips across all 4,955 selected nodes in 125 fresh-process groups, with no failures/errors or computational skips. This does not satisfy the separate failed supplemental criterion.

No fixture substitution, tolerance adjustment, assertion removal, production repair or supplemental retry has followed that failure. No seventh module is authorized. This is not an all-gates-passed report.

## What the calculation does

For the existing two-source connected response, the module reconstructs an exponentially weighted frequency response from a finite sampled record. It keeps three exact-model error formulas separate: omitted late-time response, trapezoid quadrature, and declared sample errors. Their floating evaluations are not certified enclosures. The arithmetic proxy is heuristic and is not added to the analytic ratio radius.

The nine prescribed small Bose systems produce 108 physical acquisition records; the reused synthetic reference produces 12 more. In the saved standalone demonstration:

- All 120 acquisition computations report `available_conditional`.
- All 360 window/quadrature/total diagnostics report `within_diagnostic_envelope`.
- 72 ratios report `conditional_available`; 48 report `unresolved_denominator`.

These are report statuses, not additional test passes or experiments. An unresolved denominator is a legitimate inconclusive result. The conditional ratio radius refers to the exact quotient of supplied binary64 entries, while its displayed center is rounded; the emitted pair is not a certified containing disk.

Both acquisition schedules have the same step, 1/32 in normalized time. The longer schedule is a longer observation window, not finer quadrature; no monotonic improvement of total error or ratio availability is claimed. Source contacts, physical calibration, experimental data and empirical validation remain absent.

## Authorship and preserved pre-execution repairs

Independent mathematics, adversarial schema review and numerical-policy review preceded the permanent freeze. Production/demo and the initial tests had separate authors; the initial test author did not read the new implementation. The final tests include explicitly disclosed post-authorship source-seam inspection and additions, and must not be described as wholly blind.

The first settled source, SHA256 `10aecb5b71a37b3841554ceae308cb9c621ad795a5e4a976768c7c65552f992e`, dropped otherwise known frequency metadata when partition normalization failed. Static review found this before execution. The narrow repair retains checked `eta * normalized_ground_gap` where independently available, without adding a model, trace, eigensolver call or retry. It preserves the actual partition failure. A transient earlier source version was not preserved; no exact diff to that transient version is claimed.

The preserved blind tests, SHA256 `8556aebc6544ac5d8abc5ce76f03e605fba51a46dd9e89ce119504dfe9e549ea`, were reviewed and corrected to avoid sparse retries following maximum-trace unavailability. Post-authorship additions check known-frequency failure slots with mocks, the free returned quotient against its actual estimate, and six synthetic noise pairs against a fixed 100-digit oracle with both full estimate allowances. Paired comparisons reuse existing acquisitions; the failure-slot regression stubs the joint system and performs no actual physical solve. Original versions, findings and exact repaired hashes remain preserved outside the repository.

The private-grid clarification was independently reviewed before execution: an unrepresentable positive private step is null with numerical-unavailability metadata, not a dropped sample or invalid-input reinterpretation. Independent transcendental oracles remain fixed at 100 decimal digits; the inherited approximate ratio/norm precision is separate.

## Verification results

| Gate | Audited result | Meaning and limitation |
|---|---|---|
| Collection only | 4,955 unique nodes, 35 suites | Zero test-execution credit |
| Module 6 focused | 173 passes, 2 platform skips, no failures/errors or computational skips; 175 nodes in 11 fresh processes | Physical ownership/budget assertions and both-partition sparse action checks passed; raw solver counters are not separately persisted |
| Standalone text and JSON | Both exited 0; empty stderr and isolated working directories | Absolute unwrapped scripts; strict JSON, exact schemas and status/dependency checks audited |
| Algebra | All eight named groups passed | Separate from pytest totals |
| Sole bounded same-process prefix | 1,093 passes, no skips/failures/errors; 3,862 deselected | No Module 6 test executed; historical nonlinear-response repeatability limitation remains |
| Supplemental saved-data audit and fixed fixture | Exit 1; four nonzero/sign failures; exactly one private acquisition and zero guard trips | Failed criterion remains unsatisfied; no retry |
| Full fresh-process regression | 4,945 passes, 10 platform skips, no failures/errors or computational skips; 4,955 unique nodes across 35 suites in 125 groups | Independently audited complete grouped coverage, not uninterrupted same-process coverage or resolution of the failed supplemental criterion |

The full audit reconciled 14,865 setup/call/teardown reports and 4,955 ordered finished-node records. Inherited coverage contributed 4,772 passes and eight platform skips; Module 6 contributed 173 passes and two platform skips. All 125 groups exited 0 with empty stderr. The full run took 2,109.459960250 seconds; it needed no interrupted-run recovery. Its ten platform skips concern unavailable wider floating precision/exponent ranges: gauge encoding one, quantum propagation two, source integrability four, extensive stability one, and sampled response two. Exact nodes and reasons are preserved in the independent audit.

The two focused skips are `longdouble` and `clongdouble` storage-rejection tests: this platform has no wider precision dtype to reject. They receive no pass credit. Focused, prefix and full coverage overlap and must not be summed into a larger independent test total.

Actual execution used Python 3.9.6 on macOS. Python 3.8 AST checks establish grammar compatibility, not execution under Python 3.8. Numerical work was serialized with six numerical thread limits set to one and bytecode disabled. Recorded pytest and isolated commands use warnings as errors. The supplemental launch did not explicitly request warnings as errors; no such claim is made for it.

The bounded prefix's exclusive launch marker is retained. Its passing result neither explains nor repairs the historical one-ULP `discarded_correction` discrepancy at `tests/test_substrate_nonlinear_response.py::test_strict_frozen_json_arithmetic_and_analytic_status_separate`. Module 5's 1,092-pass/1-failure prefix remains unchanged. The prefix plugin records starts and failures rather than complete phase/finish events; its audit reconciled summaries, progress, exact selection and outcomes instead.

## The failed tiny-noise criterion

The preregistered supplemental stimulus has zero physical background, gaps `[1]`, `nu=2^-10`, `theta=2^-20`, two intervals, gains `[[10,14],[-15,-21]]`, and alternating sample perturbations of magnitude `epsilon=1e-8`. Both the permanent policy and supplemental plan require nonzero output with prescribed signs for this designated fixture.

Writing `h=2^-21` and `x=nu*h=2^-31`, its ideal alternating trapezoid is

`J_ab = (-1)^(a+b) * epsilon*h/2 * (1 - exp(-x))^2`.

It is positive on the diagonal, negative off diagonal, and approximately `epsilon*2^-84` in magnitude, about `5.169878826e-34`. This ideal value is representable; it is not a structural zero.

Independent source and mathematical reviewers established a sufficient cancellation mechanism. Correctly rounded binary64 exponentials at these arguments have values `1-x` and `1-2*x`: their quadratic corrections are below the relevant half-spacing. Exact subsequent summation then gives `1 - 2*(1-x) + (1-2*x) = 0`. Exact arithmetic cannot recover information lost before lifting the transcendental operands. This is not final-conversion underflow.

The production estimator sums individually evaluated exponential node weights against stored samples. The independent closed-form noise oracle uses stable `expm1` difference factors. No generic checked-arithmetic defect is established by the saved zero and source review. However, the specifically adopted sign-preservation requirement was not met. The actual exponential operands were not captured in the failed artifact; the mechanism above is analytically sufficient and consistent with the saved result, not a runtime trace of those operands.

The frozen comparison allowance, approximately `2.168404346e-27`, is much larger than the ideal residual. Thus zero can pass the error-size comparison while failing the separate sign condition. This does not make the frozen sign condition optional after execution.

The supplemental report contains 29,690 events:

| Event category | Count |
|---|---:|
| `exact_pass` | 23,580 |
| `definitive_failure` | 4 |
| `diagnostic_agreement` | 4,719 |
| `heuristic_within` | 36 |
| `raw_discrepancy` | 1,295 |
| `unavailable` | 56 |

The four failures are `fixture.nonzero_sign[a,b]` for all four entries. These categories are not pytest totals. Raw discrepancies have no invented acceptance tolerance and are neither promoted to confirmed defects nor relabeled passes. The 56 unavailable events concern missing saved metadata/scales, not extra physical calls. There were no `audit_incomplete` events. All original output and the exit-1 completion record remain intact.

## Bindings and evidence locations

All external paths below are relative to `/Volumes/T9 Backup/bpr-verification/module6/`.

| Bound item | SHA256 |
|---|---|
| Executed production | `f3ce767ced18659d830cfefe86dcd0ad91273578bb350c992da0cc1600198a2c` |
| Executed demo script | `b50c0ef4f7d96d7d77d450054e43c4be51de993b3e4cf5780606b23fbc164914` |
| Executed tests | `7cff9bd98f74e6c5e6efaf9997bbaeb86166a9960eebe6ae152aafe37b9b9bf4` |
| Frozen contract | `e6ea4a8bf3c24963c13d2c4f021fc17ec8cf5566de1d967b67dbe5a6a6d0fe10` |
| Approved execution manifest | `59fdcf233d79bb1886b4e5dcbd1ceb177823774e21ff309f6338929eeab1d5f7` |
| Protected 485-file snapshot | `86a4b664a30707906e41002086c1f92a23ac87cf294681807f4e49e53a6571be` |
| Saved isolated demo JSON | `f2dc413aa0650ac6dbaff8e2e4cc4e831f834c6e160ac9e635a6c5fed7863039` |
| Executed supplemental probe | `80041a70af560ad325c95125a9847ce9b93e0169a3855f628b2230feb43fc5cf` |

| Original artifact directory under `artifacts/` | Files | Tree fingerprint |
|---|---:|---|
| `bpr-sampled-response-collect-only-7szw5sdb` | 20 | `ee144f8e0a03bd6e1754b1fd35ee3df4c657ef0e7a3b4efc7ea5de0425e7132e` |
| `bpr-sampled-response-focused-ys51pi73` | 165 | `1c5e0e56988694654e717a8e1f73fa8e9421b721511db591d9d6e9409cd9f3bb` |
| `bpr-sampled-response-isolated-317ai3mz` | 27 | `c63f467ac5267eac4c856ce8d36139704cd3cfd014d0ac839f4dfaa3d30b499b` |
| `bpr-sampled-response-prefix-n2ghmwto` | 26 | `a459f70639baf67b7c46256b6da61d8a404806099d658e50172f8879ed042e0b` |
| `bpr-sampled-response-probes-86dsixhs` | 4 | `254ee31a4c41cca5e597829796077b3abfeab934d8ebfe1b58f9c168b5bc1afc` |
| `bpr-sampled-response-full-vau5ztiv` | 1,647 | `7ac4d9787d6fc0ed1ba0f139d7042d7023204cd6064f7d2b32890ae25f3ade82` |

Fingerprint method: sorted regular-file relative POSIX paths in `pathlib.Path` ordering; reject symlinks; feed each UTF-8 path, NUL, and raw 32-byte SHA256 of file contents into SHA256. Empty directories and external reports are excluded.

Focused, isolated, prefix and full audits verified respectively 25, eight, four and 253 protection snapshots, each containing 492 bound files including all 485 protected inherited files. The full audit reconfirmed that all five earlier artifact trees, prior audit reports and the exclusive prefix marker were unchanged. Its external report is `full-regression-independent-evidence-audit-v1.json`, SHA256 `9b445e86e7ea3c14082bca14dd8eef14ef11378eaf7de92601413aeaadfebd8b`. The probe itself records pre/post hashes for its five supplied inputs, not a complete probe-time protected-file snapshot. Later independent checks verified all 485 still matched; they do not substitute for a missing contemporaneous probe snapshot.

External review records include `static-review-findings-v1.md`, `focused-isolated-independent-evidence-audit-v1.json`, `prefix-independent-evidence-audit-v1.json`, `supplemental-first-failure-independent-evidence-audit-v1.json`, and coordinator record `supplemental-first-failure-v1.md`. Older pending statements in these records retain their historical meaning; later audits supersede status, not failed evidence. Independent reviewers' reports are not human acceptance or publication approval.
