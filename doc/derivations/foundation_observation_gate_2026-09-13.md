# Observation gate: can this response distinguish BPR?

2026-09-13. Work package 5 of the [foundation prerequisite campaign](foundation_prerequisites_2026-09-13.md). This document performs an identifiability argument, not an experiment or new numerical evaluation. Independent read-only adversarial review found no confirmed defects in the scoped argument, after comparison with the prediction and sampled-response contracts. That review does not certify measurements.

## Candidate and operational requirements

Use one existing candidate only: the dimensionless joint density/energy response ratio

`Q(y) = chi_rho,h(iy) chi_h,rho(iy) / [chi_rho,rho(iy) chi_h,h(iy)]`,

at `y = eta * Delta1`, with the same ground gap measured on the same clock. The gap is `E1-E0` counting multiplicity, not a substituted first visible source pole. Denominator resolution is a prerequisite. This is not absolute-square coherence and need not lie in `[0,1]` for arbitrary complex source systems.

The [existing prediction contract](substrate_prediction_contract_2026-09-13.md) defines the actual Bose Hamiltonian, source operators, unit-filled small-ring domain and symmetric energy partition. `bpr/substrate_prediction_contract.py:case_report`, `scaling_report`, and `bpr/substrate_joint_source_kernel.py:joint_system` generate model predictions; they do not accept acquired measurements.

An operational experiment would need all of the following, none supplied merely by evaluating these APIs:

- A prepared ring with independently measured population, hopping and interaction parameters, a justified ground-state approximation and controlled preparation error.
- A density probe and an implementable perturbation/readout matching the prescribed local energy operator, including its interaction and bond partition. Naming a probe “energy” is not enough.
- Both source-to-readout directions and both diagonal responses, with a fixed retarded convention and calibrated linear-response regime. Separate connected response from any source contacts.
- A shared time calibration and an independently measured actual ground gap. Convert raw time and energy units explicitly; setting `C=1` in software does not calibrate a physical clock.
- Finite time sampling and imaginary-frequency reconstruction with uncertainty from missing late-time response, sampling, readout, preparation and model mismatch. Existing exact-model truncation formulas and heuristic arithmetic proxies do not certify all these experimental errors.
- Independently constrained backgrounds, channel mixing and pair-specific gains. Frequency-wise row/column factorization must be tested or justified rather than inferred from a favorable ratio.

No hardware, laboratory, real dataset or calibration record is selected here.

## Multiplicative cancellation and its limits

For nonzero factorized readout/source gains,

`M_ab(y) = r_a(y) c_b(y) chi_ab(iy)`,

all gains cancel in `M01*M10/(M00*M11)` wherever the denominator is resolved. Signed and frequency-dependent gains are allowed pointwise. This cancellation does not cover additive backgrounds, crosstalk or independent pair gains. It also does not determine the model parameters uniquely.

If an arbitrary independent additive background `B_ab(y)` is allowed at every measured channel/frequency, every candidate kernel fits any record by choosing

`B_ab(y) = M_ab(y) - r_a(y)c_b(y)chi_ab(iy)`.

Thus the unconstrained-background observation model has no response discrimination. This is an exact nonidentifiability statement for that nuisance class, not a claim that real backgrounds cannot be measured or constrained. No fitted background is introduced in this campaign.

## Conventional alternative and indistinguishability

Compare two descriptions with **the same** Bose Hamiltonian, Hilbert space, state/preparation, density and energy operators, parameter domain and observation/noise map:

1. The current BPR substrate response calculation.
2. Conventional Bose–Hubbard ring dynamics with the same conventions.

Their time evolutions and response kernels are identical by definition, hence so is Q. This requires matching ring/bond conventions explicitly; a name alone does not establish equivalence. With identical observation distributions, any statistical test has the same rejection probability under the paired descriptions. More precise data or additional frequencies cannot distinguish labels attached to identical predictions.

This conclusion is conditional on the stated common model and observation map. It does not exclude a future BPR-specific consequence derived from additional independently justified physics. It means the present candidate tests this Bose response model, not the unique physical existence of a BPR substrate or a theory of everything. An intentionally sign-flipped synthetic record is a software negative control, not a competing physical theory.

## Decision and next prerequisite

Disposition following independent argument review:

- `discrimination_status: no_distinguishing_prediction`
- `calibration_status: calibration_missing`
- `data_status: data_missing`
- `empirical_validation: false`
- `new_measurements_acquired: false`

The first blocker is a distinct prediction, not insufficient precision or a missing data loader. No acquisition or fitting pipeline should be built for a BPR-discrimination claim until that blocker changes.

If a future revised model supplies a distinct consequence, a separate experiment proposal must preregister the competing distributions, allowed nuisance families, independent calibration procedure, uncertainty model, resource requirements and decision rule before inspecting held-out outcomes. Calibration/development and held-out observations must be separated prospectively. There is no justified numerical rejection threshold to invent for the present identical-model comparison.

`bpr/research/evidence.py` can represent a sourced literature measurement and review status. A populated provenance record is not proof of independent acquisition, extraction correctness or calibration. Literature summaries, the existing synthetic held-out example and simulation datasets remain distinct evidence categories.

## Preserved numerical debt

The sampled-response module's four failed tiny-noise sign assertions remain failed. No repeat of that single-attempt supplemental probe is part of this work. Even a separately approved future stable-arithmetic revision would not break the physical-model equivalence established above.
