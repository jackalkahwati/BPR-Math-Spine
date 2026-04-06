# BPR Parameter Uniqueness Audit — Final Memo

**Date:** 2026-04-06
**Purpose:** Determine whether p = 104729 and z = 6 are structurally unique, or whether nearby alternatives perform comparably.
**Approach:** Skeptical research audit, not a sales demo.

---

## What Was Tested

We swept across 51 primes near 104729 (25 below, 25 above) at fixed z = 6, then ablated z over [4, 5, 6, 7, 8] at fixed p = 104729, then ran a joint grid of 9 primes × 5 z values (45 total runs). For each run we collected every observable the codebase supports and computed a composite score.

## How It Was Tested

For each (p, z) pair we called the BPR Python library directly, collected predicted values for six observables, and computed a composite score defined as the RMS of fractional errors across the **independently varying** observables. The score definition was fixed before looking at results.

**Composite score = sqrt( mean( (|predicted - expected| / expected)^2 ) )**

Only observables that actually change when p or z changes were included in the score. Four others were collected but excluded with documented reasons (see below).

## What Files and Functions Matter Most

| File | Key function | Observable |
|------|-------------|-----------|
| `bpr/alpha_derivation.py` | `inverse_alpha_from_substrate(p, z)` | 1/α at q²=0 |
| `bpr/gauge_unification.py` | `electroweak_scale_GeV(p, z)` | Higgs VEV |
| `bpr/gauge_unification.py` | `GaugeCouplingRunning(p).weinberg_angle_at_MZ` | sin²θ_W (see caveat) |
| `bpr/charged_leptons.py` | `ChargedLeptonSpectrum()` | lepton masses (see caveat) |
| `bpr/cross_predictions.py` | `dark_energy_from_impedance(p)` | Ω_Λ (see caveat) |
| `bpr/constants.py` | `P_DEFAULT = 104729, Z_DEFAULT = 6` | parameter definitions |

## Main Results

### Baseline (p = 104729, z = 6)

| Observable | Predicted | Experimental | Error |
|-----------|-----------|-------------|-------|
| 1/α(0) | 137.0316 | 137.0360 | 32 ppm |
| 1/α(M_Z) | 127.9476 | 127.9520 | 34 ppm |
| v_EW (GeV) | 243.49 | 246.22 | 1.1% |
| sin²θ_W | 0.23122 | 0.23122 | 0.00% |

### Prime Sweep (z = 6 fixed)

| Result | Value |
|--------|-------|
| Baseline rank | #26 out of 51 primes |
| Primes scoring BETTER than 104729 | 25 out of 25 primes above it |
| Primes scoring WORSE than 104729 | 25 out of 25 primes below it |
| Score trend | Strictly monotone increasing as p decreases |
| Best prime in sweep | p = 105019 (score 0.00708 vs baseline 0.00784) |

The score varies smoothly and monotonically. There is no local minimum at p = 104729.

### Z Ablation (p = 104729 fixed)

| z | Score |
|---|-------|
| 4 | 0.0979 |
| 5 | 0.0529 |
| **6** | **0.0078** |
| 7 | 0.0372 |
| 8 | 0.0822 |

z = 6 is the clear winner here. However, this is entirely explained by the formula structure (see "Does z = 6 look unique?" below).

---

## Does the Evidence Support Uniqueness of p = 104729?

**No.**

Every one of the 25 primes tested above 104729 scores better. The landscape is a featureless slope, not a sharp minimum. p = 104729 ranks 26th out of 51 tested primes. The reason is mathematical: the core formula is `1/α = [ln(p)]² + z/2 + γ − 1/(2π)`. The term `[ln(p)]²` changes by about 0.0022 per prime spacing near 104729. The baseline residual is −0.0044 (underpredicting by 32 ppm). Increasing p by roughly 20 units would eliminate most of the error. Primes at p ≈ 104749 would give a closer match.

The framework has two free parameters (p, z) and at most two genuinely parameter-dependent observables (1/α and v_EW). A model with 2 free parameters and 2 targets is exactly determined — zero degrees of freedom, no predictive surplus.

## Does the Evidence Support Uniqueness of z = 6?

**Weakly yes, but for the wrong reason.**

z = 6 clearly outperforms z = 4, 5, 7, 8. But this is almost entirely because `z/2 = 3` is the integer that brings `[ln(p)]² + z/2` closest to 137.036 given the already-chosen p = 104729. The formula is `1/α = [ln(p)]² + z/2 + γ − 1/(2π)`, and z enters only as `z/2`. Changing z by ±1 shifts the prediction by ±0.5, while the experimental tolerance is about ±0.005. So z must be exactly 6 given p = 104729, or the prediction fails badly. This looks like z being selected to compensate for whatever value p contributes, not like an independent physical constraint.

If p were chosen differently, the optimal z would shift accordingly. For example, a p that gives `[ln(p)]² ≈ 133.1` would prefer z = 7.

## What Are the Biggest Caveats?

**1. Only two observables change with p and z.**

Four of the six collected observables are excluded from the score because they do not actually depend on the substrate parameters:

- **sin²θ_W**: The `GaugeCouplingRunning.weinberg_angle_at_MZ` method constructs threshold corrections to force gauge unification, then runs back down from those corrected values. By construction it recovers the hardcoded input coupling — the experimental value 0.23122 — for every value of p. It returns 0.23122 for p = 3, p = 104729, or p = 999983. This is a mathematical identity, not a prediction.

- **Lepton masses**: `ChargedLeptonSpectrum` takes no p or z arguments. The mass predictions come from fixed angular momentum mode numbers (l = 1, 14, 59) anchored to the tau mass as the single experimental input. These would be identical under any parameter choice.

- **Ω_Λ (dark energy)**: The `dark_energy_from_impedance` function uses `p_cosmo = R_Hubble / L_Planck ≈ 2.7 × 10^61` as its mode count, not the substrate p = 104729. For every value of the p argument, it computes Ω_Λ ≈ 10^−104, which is wrong by 100 orders of magnitude. The value 0.685 cited in the documentation comes from `OMEGA_LAMBDA = 0.685` hardcoded in constants.py.

- **1/α(M_Z)**: This equals `1/α(0) − 9.084` by definition. It contains no additional information.

**2. The formula is a 2-parameter fit to (at most) 2 numbers.**

With `1/α = [ln(p)]² + z/2 + γ − 1/(2π)` and `v_EW = Λ_QCD × p^(1/3) × (ln(p) + z − 2)`, the system has exactly as many equations as parameters. There is no overdetermination and no independent test of the framework.

**3. The l_modes for leptons appear reverse-engineered.**

The values l = 1, 14, 59 appear to have been chosen to match the known lepton mass hierarchy, not derived from first principles of the BPR substrate. They are not functions of p or z. This makes the lepton mass "predictions" post-hoc fits with hidden parameters, not genuine forward predictions.

**4. The sweep covers a narrow range.**

The 51-prime sweep spans ±0.3% in p. A wider sweep might reveal a local minimum further from 104729, or might confirm that the monotone slope continues indefinitely.

## What Next Experiment Would Most Decisively Confirm or Refute Uniqueness?

The strongest possible test: **derive the proton-to-electron mass ratio m_p/m_e ≈ 1836.15 from (p, z) without using it as input, then check whether the error landscape has a minimum at p = 104729, z = 6.**

This ratio is dimensionless, precisely measured, and independent of the fine-structure constant. If the BPR formula for this ratio is `f(p, z)` and shows a sharp minimum at exactly (104729, 6) with a substantially different functional form than the 1/α formula, that would be genuinely compelling evidence.

If instead the formula is another smooth function of `ln(p)` and `z`, the same analysis above applies: it would be fitting 3 numbers with 2 parameters, which is underdetermined.

A secondary test: extend the prime sweep out to ±500 primes, plot the score landscape, and look for any non-monotone feature. The current data shows a strict monotone slope with no hint of structure.

---

## Files Produced

```
analysis/
  sweep.py              — sweep harness (run to regenerate data)
  analyze.py            — analysis and plotting script
  memo.md               — this document
  summary.txt           — machine-readable sensitivity summary
  results/
    prime_sweep.csv     — 51 primes, z=6 fixed
    z_ablation.csv      — p=104729 fixed, z in [4,5,6,7,8]
    joint_sweep.csv     — 9 primes × 5 z values
    sweep_results.csv   — all runs, ranked by score
    sweep_results.json  — same, structured JSON
  plots/
    score_vs_prime.png       — score landscape over prime sweep
    score_vs_z.png           — score and observables vs z
    heatmap_joint.png        — 2D score heatmap (prime × z)
    alpha_error_vs_prime.png — signed 1/α error vs prime
```
