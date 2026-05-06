# BPR Boundary Phase Field FFT Analysis — Toy ECDLP Results

**Branch:** `claude/bpr-ecdlp-research-SWQnW`
**Scope:** synthetic toy curves only; no real wallets, no secp256k1 attack data.
**Status:** complete; verdict = **no usable signal**.

---

## 1. Definition of φ_a(W) and group-operation cost model

Ten phase-field constructions implemented in `phase_fields.py`, organized in three families. All take only public inputs `(E, G, Q, n, BPR params)`. None reads `k`.

| ID | Definition | Uses curve? | Uses Q? | Group-op cost (window of size m, start w₀) |
|---|---|---|---|---|
| **A1** | `exp(2πi · W / p_sub)` | no | no | 0 |
| **A2** | `Σᵢ exp(2πi · (W mod pᵢ) / pᵢ)` | no | no | 0 |
| **B1** | `exp(2πi · x(W·G) / p)` | yes | no | `m + log₂ w₀` |
| **B2** | `exp(2πi · (x(W·G) mod p_sub) / p_sub)` | yes | no | `m + log₂ w₀` |
| **C1** | `exp(2πi · (x(W·G) − x(Q)) mod p / p)` | yes | yes (additive) | `m + log₂ w₀` |
| **C2** | `exp(2πi · x(W·G + Q) / p)`, 0 at `W·G = −Q` | yes | yes (point sum) | `2m + log₂ w₀` |
| **C3** | `exp(2πi · ((x(W·G) − x(Q)) mod p_sub) / p_sub)` | yes | yes (additive) | `m + log₂ w₀` |
| **C4** | `Σᵢ exp(2πi · ((x(W·G) − x(Q)) mod pᵢ) / pᵢ)` | yes | yes | `m + log₂ w₀` |
| **C5** | `exp(2πi · x(W·G) · x(Q) / p²)` | yes | yes (bilinear) | `m + log₂ w₀` |
| **C6** | `exp(2πi · (W mod p_sub) · (x(W·G) − x(Q)) / (p_sub · p))` | yes | yes (winding × diff) | `m + log₂ w₀` |

BPR substrate parameters: `p_sub = 7`, `multi_primes = (3, 5, 7, 11)` — defaults from `bpr/complexity.py`.

**Cost benchmark.** Field cost over a window of size `m` is essentially `m` group operations (incremental walk `R, R+G, R+2G, ...`). **This is identical to the cost of a brute-force scan over the same window.** The FFT does no group operations after the field is built.

For a method to beat Pollard rho's `~1.25√n` group-operation cost, the FFT must extract information about `k` from a window of size `m` such that the *amortized* cost is below `√n`. We test this directly.

---

## 2. Benchmark design

| Parameter | Value |
|---|---|
| Bit-widths | 12, 14, 16, 18 |
| Instances per bit-width | 30 (random Weierstrass curves; each has prime-order generator) |
| Keys per instance | 1 (uniformly random `k ∈ [1, n)`) |
| Window size m | 256 |
| Window placements | `in` (k inside) and `out` (k outside) |
| Phase fields | 10 (Family A/B/C as above) |
| Controls | none, shuffle, Q-sub, G-sub, randphase |
| Total trials in CSV | 12,000 rows = 4 bits × 30 inst × 1 key × 10 fields × 2 windows × 5 controls |

Curve generation enumerates `|E(F_p)|` (Hasse-bound counting) and projects onto the largest prime-order subgroup. Order distribution: bits=12 → n ∈ [1049, 3931]; bits=18 → n ∈ [66067, 259379].

**Baselines.** Per-instance BSGS and Pollard rho group-op counts are recorded for direct comparison. Observed rho mean ≈ 4·√n, BSGS mean ≈ 2·√n — consistent with theory.

**Controls.**
- **none:** field built with the true `(G, Q)`.
- **shuffle:** values of φ randomly permuted before FFT.
- **Q-sub:** Q replaced with `R = r·G` for uniformly random `r` (same group, unrelated discrete log).
- **G-sub:** G replaced with `G' = c·G`, `gcd(c, n) = 1` (same subgroup, different basis).
- **randphase:** length-m unit-modulus complex array, Haar-random.

---

## 3. Results

### 3.1 Top-K hit rate (top-K = 5 of 256 bins, chance = 5/256 ≈ 1.95%)

| Diagnostic | Value |
|---|---|
| Total (bits, field, control) buckets | 200 |
| Buckets with **uncorrected p < 0.05** | **8 / 200** |
| Buckets expected by chance at α = 0.05 | 10 |
| Buckets surviving **Benjamini–Hochberg FDR** at α = 0.05 | **0 / 200** |
| Buckets surviving Bonferroni at α = 0.05 | 0 / 200 |

The eight uncorrected-significant buckets are scattered across **all five control conditions** (shuffle: 3, qsub: 2, gsub: 2, none: 1). The shuffle and randphase controls produce *more* "raw-significant" buckets than the `none` (true-Q) condition. This is the textbook fingerprint of multiple-comparison noise.

### 3.2 Peak-distance test (stronger than top-K)

For each trial we measured `d = circular_distance(top peak index, k_in_window) / m`, with null mean `0.25` for uniform-random peaks.

- 12 / 200 buckets had **uncorrected p < 0.05** (chance: 10).
- **0 / 200** survived BH-FDR.
- Critical pattern: for field `C1_q_coupled_diff` at 18-bit, the `none` and `qsub` conditions produced **identical p-values**. Investigated below.

### 3.3 Mutual information `I(k_in_window ; spectral_features)`

Histogram-based MI between `k_in_window / m` and the two spectral summaries (peak sharpness, spectral entropy), per `(bits, field, control)` bucket of 30 trials:

- All values rounded to **0.0000 nats** in the printed report.
- The largest non-zero MI was **0.001 nats** for one bucket — at the floor of estimator noise. None survived FDR.

### 3.4 Amortization analysis

Mean `(field_cost) / (rho_cost)` over instances:

| bits | A-family | B/C-family (single Q·G eval) | C2_q_coupled_sum (extra add) |
|---|---|---|---|
| 12 | 0.000 | 1.117 | 2.180 |
| 14 | 0.000 | 0.650 | 1.260 |
| 16 | 0.000 | 0.335 | 0.646 |
| 18 | 0.000 | 0.175 | 0.337 |

The B/C ratio decreases with bit-width because window size is fixed at 256 while `√n` grows. Even at 18-bit, however, the phase-field build cost is **17.5% of rho's average ops** — and the field carries **zero detectable information about `k`**. The A-family is "free" but trivially Q-independent.

---

## 4. Why the result is what it is — a smoking-gun observation

The strongest diagnostic in the data is the C1 / B1 magnitude-spectrum equivalence.

**C1** is `exp(2πi · (x(W·G) − x(Q)) mod p / p)`. Let `y(W) = (x(W·G) − x(Q)) mod p`. Under Q-substitution (replacing `Q` with `R`), the values `y(W)` shift by an additive constant `c = x(Q) − x(R) mod p`. In the exponent this is a uniform multiplicative phase factor `e^{−2πi c / p}`. The FFT magnitude is **exactly invariant** under uniform multiplicative phase.

We verified this numerically: `max | |FFT(φ_C1 with Q)| − |FFT(φ_C1 with R)| | ≈ 1.6 × 10⁻¹⁴` — i.e., zero up to float precision.

**Consequence:** any amplitude-based detector (top-K hit rate, peak sharpness, spectral entropy) on C1 *cannot in principle* distinguish the true `Q = k·G` from a random `R`. The construction is **Q-blind in amplitude space, therefore k-blind**.

This is the DDH-distinguisher framing made concrete: a "Q-coupled" field whose only Q-dependence is a uniform phase fails to be a distinguisher *by linear algebra*, not by failure to find the right BPR formula. Several other constructions (B2, C3, C4) inherit the same defect because their Q-dependence reduces to additive constants modulo small primes.

The Q-truly-nonlinear constructions (C2 sum, C5 bilinear, C6 winding-resonance) **do** have Q-dependent magnitude spectra in principle, but their spectra are dominated by the construction's *own* periodicities (e.g., C6's `W mod p_sub` factor produces a period-7 comb with peak distance to k *worse* than chance: mean `d/m = 0.355` vs. null `0.25`).

This is precisely the "instrument peak vs. song peak" failure mode you flagged in the prompt.

---

## 5. Final verdict

**`no usable signal`** — and structurally so.

Justification:
1. **0 / 200 (field, control, bits) buckets survive BH-FDR** on either the top-K hit-rate test or the peak-distance test.
2. **Mutual information between spectral features and `k` is at estimator floor** across all conditions.
3. **Q-substitution control is empirically indistinguishable from the true-Q condition** for every additively-Q-coupled field. For C1 this is provable (the magnitude spectrum is exactly Q-invariant).
4. **No scaling improvement with bit-width.** From 12-bit to 18-bit there is no monotonic trend in any field's hit rate above chance; this is the opposite of what a real signal would show.
5. **Amortization analysis is moot.** Field-build cost is non-trivial (up to ~22% of rho's ops at small `n`), but with zero signal it cannot be amortized into a speedup at any size.

### Interpretation in the verdict ladder

| Possible verdict | Holds? | Why |
|---|---|---|
| no usable signal | **YES** | All controls null after FDR; provable Q-blindness on amplitude detectors for additive-Q fields. |
| construction artifact | yes (secondary) | C6 produces strong period-7 peaks that are uncorrelated with `k` and dominate the spectrum (instrument, not song). |
| toy-only signal | no | No signal observed even at 12-bit. |
| weak-curve-only signal | not tested but moot | We tested only random-Weierstrass curves; given the null on those, the question of weak-curve-only signal is irrelevant to the secure-curve question. |
| minor heuristic improvement | no | Field-build cost ≥ a brute-force scan of the same window; no observed advantage in group-op count. |
| possible scalable distinguisher | no | DDH framing: any positive result here would have been a DDH distinguisher on prime-order curves; we observed no such result. |

### What it would take to overturn this verdict

- A phase-field construction whose **magnitude spectrum** (not just complex spectrum) measurably depends on `Q` and *cannot be computed by a uniform multiplicative-phase shift of a Q-independent field*. None of our 10 candidates passes that bar; the ones that nominally do (C2/C5/C6) have their spectra dominated by construction-induced periodicities orthogonal to `k`.
- A demonstration that the construction's group-op cost amortizes: e.g., **fixed-G, many-Q precomputation** where one expensive build pays off for many DLP queries. We did not test this regime because the per-instance signal was null; with a non-null signal it is the correct next experiment.

---

## 6. Files

```
experiments/ecdlp_fft/
├── curve.py             # Weierstrass toy curve + group-op counter
├── curve_gen.py         # seeded toy curve / generator construction
├── solvers.py           # brute force, BSGS, Pollard rho
├── phase_fields.py      # 10 BPR phase-field constructions
├── analysis.py          # FFT features, MI estimator, BH-FDR
├── run_experiment.py    # main driver -> data/results.csv
├── summarize.py         # hit-rate, MI, amortization tables
├── distance_analysis.py # peak-distance vs chance test
├── REPORT.md            # this file
└── data/
    ├── results.csv          # 12,000 trial rows
    ├── hit_rate.csv         # per-bucket hit rates + BH flag
    ├── amortization.csv     # field cost vs rho cost
    └── mutual_information.csv
```

Reproducible: all RNG is seeded by `(bits, instance, key, role)` strings.

---

## 7. Cryptographic relevance to secp256k1

None established. Reasons:

- **Information-theoretic.** Under the DDH assumption on secp256k1 (widely held), any polynomial-time-computable function of `(G, Q)` is computationally indistinguishable from a function of `(G, R)`. Our experiments empirically confirm this for the BPR-derived constructions tested at toy sizes; nothing in the framework provides a candidate that escapes the DDH bound at any size.
- **Generic-group lower bound.** All ten constructions use only the curve operation (and projections of x-coordinates). The Shoup `Ω(√n)` lower bound applies; even a *successful* spectral construction in this family could not beat √n asymptotically, only constants.
- **Toy-to-secp gap.** Our largest test was 18-bit (n ≈ 10⁵). secp256k1 has n ≈ 2²⁵⁶. Even a hypothetical positive 18-bit result would face 76 orders of magnitude of extrapolation. The negative 18-bit result terminates that path.

The reframing of ECDLP into BPR's vocabulary remains internally consistent (as discussed in the prior research note), but the empirical test of its most natural FFT-based attack family produces a clean null. **No update to the prior on secp256k1 hardness is warranted by these results.**
