# BPR ↔ ECDLP — Extended (Non-FFT-Magnitude) Transform Suite

**Branch:** `claude/bpr-ecdlp-research-SWQnW`
**Scope:** synthetic toy curves only; no real wallets, no secp256k1 attack data.
**Builds on:** `REPORT.md` (FFT-magnitude analysis, verdict = no signal).
**Status:** complete; verdict = **no usable signal across all transform families**.

This report extends the FFT-magnitude analysis with phase-sensitive
spectra, localized spectra, modular character sums, graph topology, and
two transform-agnostic distinguishers (MMD and held-out classifier
AUC).

---

## 1. Transforms tested

39 scalar features per phase-field per control, grouped into 9 families
implemented in `transforms.py`.

| Prefix | Family | Features | Phase preserved? | Uses Q? | Cost beyond φ build |
|---|---|---|---|---|---|
| `ph_` | Phase spectrum | mean, circ-var, slope, coherence, top peak arg | yes | via φ | O(m log m) |
| `ac_` | Autocorrelation | ACF at lags 1, 2, 4, 8, 16; max-abs | yes | via φ | O(m log m) |
| `bs_` | Bispectrum | max, mean over 32×32 low-freq cell | yes (3rd order) | via φ | O(K²) with K=32 |
| `st_` | Sliding STFT | window-energy mean / std / cv | yes | via φ | O(m·win) |
| `wv_` | Wavelet (Ricker) | max, entropy, top scale | partial (\|φ\|) | via \|φ\| | O(m·n_scales) |
| `mse_` | Multiscale entropy | s = 1, 2, 4, 8 | partial (\|φ\|) | via \|φ\| | O(m) |
| `ch_` | Multiplicative char | Legendre-symbol walk: mean, max, end, runs | – | direct (xs) | O(m·log p) |
| `ad_` | Additive char sums | normalized \|S_a\| and arg, a ∈ {1,2,3,5,7} | – | direct (xs) | O(m) |
| `gl_` | Graph Laplacian | k-NN spectral gap, top eig | – | direct (xs) | O(m²) |

`xs` = integer x-coordinate sequence of `[w₀·G, …, (w₀+m−1)·G]`. Features
in `ch_`, `ad_`, `gl_` are computed directly on `xs`, independent of the
particular φ construction.

NTT was implemented but is `O(m²)` in pure Python; replaced by `bs_` and
`ad_` which provide equivalent finite-field structure tests at a tractable
cost.

Persistent homology / Betti curves: skipped. The orbit `{i·G}` is a
discrete cycle; its persistent topology is trivially `(β₀=1, β₁=1)` and
carries no `k`-dependent information by construction.

---

## 2. Benchmark design

Same as the FFT-magnitude experiment:
- Bit-widths 12, 14, 16, 18; 30 seeded random Weierstrass curves per
  bit-width; 1 random key per instance; window size m = 256.
- Five controls: `none`, `qsub`, `gsub`, `shuffle`, `randphase`.
- Per (instance × field × control) we compute all 39 features.
- Total rows in `data/extended_features.csv`: **6,000**.

Wallclock: ~2m12s for the full sweep on one CPU.

### Statistical pipeline

Two complementary distinguisher tests, both with controls:

**(a) Marginal per-feature t-test.** For each `(bits, field, feature)`,
Welch's two-sample test of feature distribution under `control='none'`
vs `control='qsub'`. Total tests: 4 bits × 10 fields × 39 features =
**1,560 marginal tests**. BH-FDR applied across all 1,560.

**(b) Joint distinguisher.** For each `(bits, field)`, stack all 39
features as a feature vector, then test whether the distributions
`{features under real Q}` and `{features under Q-substitution R}` are
distinguishable, via:

  1. **MMD²** with Gaussian RBF (median-heuristic σ) + 200-permutation
     p-value.
  2. **Group-aware K-fold logistic-regression AUC** with 50-permutation
     label-shuffling p-value. Group = instance index (each instance is a
     `(real, qsub)` pair; the group split keeps both members of a pair
     in the same fold; otherwise the classifier is biased toward AUC ≈ 0
     by paired-instance leakage — a methodological hazard we hit and
     fixed during this work, see §6).

BH-FDR applied across the 40 joint tests, separately for MMD and AUC.

---

## 3. Results

### 3.1 Marginal tests (1,560 features)

| Diagnostic | Value |
|---|---|
| Total tests | 1,560 |
| Raw `p < 0.05` | **15** (chance: ≈78) |
| BH-FDR significant @ α=0.05 | **0** |

The raw count is *below* the chance expectation, confirming no
meta-signal across the full feature suite. Per-family BH-significant
counts are all 0:

| Family | BH sig / total |
|---|---|
| ac (autocorrelation) | 0 / 240 |
| ad (additive char) | 0 / 400 |
| bs (bispectrum) | 0 / 80 |
| ch (Legendre) | 0 / 160 |
| gl (graph Laplacian) | 0 / 80 |
| mse (multiscale entropy) | 0 / 160 |
| ph (phase spectrum) | 0 / 200 |
| st (STFT) | 0 / 120 |
| wv (wavelet) | 0 / 120 |

The 10 smallest raw p-values cluster on `C4_q_coupled_multiprime`
(multi-prime Q-coupled BPR resonance) at multiple bit-widths — see §4
for an honest discussion of whether this represents anything real.

### 3.2 Joint distinguishers (40 buckets)

| Test | Raw p<0.05 | BH-FDR @ α=0.05 | Notes |
|---|---|---|---|
| MMD² + permutation | 1 / 40 | **1 / 40** | (18-bit, C4) only |
| Group-fold AUC + perm | 2 / 40 | **0 / 40** | not corroborated by MMD pattern |

Detail of the four AUC-strongest buckets:

| bits | field | MMD² | p_mmd | AUC | p_auc |
|---|---|---|---|---|---|
| 18 | C4_q_coupled_multiprime | 1.36e-01 | **0.000** ✓ BH | 0.694 | 0.020 |
| 16 | C1_q_coupled_diff | 4.1e-08 | 1.000 | 0.650 | 0.020 |
| 14 | C4_q_coupled_multiprime | 1.93e-02 | 0.590 | 0.689 | 0.080 |
| 14 | C1_q_coupled_diff | 3.7e-08 | 1.000 | 0.644 | 0.100 |

**Critical sanity check passed:** for all A and B family fields (which
do not use Q), MMD² ≡ 0 and AUC ≡ 0.500 to floating-point precision,
across all bit-widths. After fixing the group-aware CV bug (see §6) the
distinguisher correctly says these constructions are Q-blind.

### 3.3 Are the C4 / C1 hits real?

The single BH-significant MMD bucket is `(18-bit, C4_q_coupled_multiprime)`.
Things to check:

| Diagnostic | Observation | Interpretation |
|---|---|---|
| Scaling 12 → 18 bit | C4 raw p_mmd: 0.900, 0.590, 0.695, 0.000 | non-monotonic; consistent with noise + a single lucky bit |
| Marginal corroboration | C4 best raw p across features: 0.0001 (mse_s1), 0.005 (bispectrum); 0 BH-significant | no feature survives FDR |
| Classifier corroboration at same bucket | AUC=0.694, p_auc=0.020 | passes raw, fails BH-AUC |
| Cross-correlation with C1 | C1 has tiny MMD (~1e-8) at all bits; classifier AUC 0.6+ at 14, 16 bits | reflects that x(W·G) varies with Q-substitution by tiny floating-point modular wrap-around; orthogonal to k |
| What C4 field does | sums `exp(2πi·((x(W·G) − x(Q)) mod pᵢ) / pᵢ)` over small primes pᵢ ∈ {3,5,7,11} | the modular projections do introduce a non-trivial Q-dependence in higher moments; the spectrum however is dominated by the period-pᵢ comb structure |

**Verdict on the C4 bucket:** the construction does have a measurable
higher-order statistical dependence on `Q` — that is not surprising,
because unlike additive C1 the modular reductions break the uniform-phase
shift symmetry. But the dependence is (i) at FDR significance only at
one bit-width out of four, (ii) not corroborated by classifier AUC under
multiple-comparison correction, (iii) carries no apparent information
about `k` itself: the marginal tests against `k_in_window` (rerun from
the FFT-magnitude analysis) show no significant correlation. This is
"the construction is not perfectly Q-symmetric" — interesting from a
DDH-distinguisher framing but a long way from a `k`-recovering attack.

---

## 4. Per-transform-family verdicts

| Family | Verdict | Why |
|---|---|---|
| Phase spectrum (`ph_`) | **no signal** | 0 / 200 marginal BH; phase slope and circ-var of A/B fields exactly Q-blind by symmetry; C-fields show no significant deviation. |
| Autocorrelation (`ac_`) | **no signal** | 0 / 240 marginal BH; ACFs of all fields collapse onto curve-walk autocorrelation, not k-correlation. |
| Bispectrum (`bs_`) | **no signal** | 0 / 80 marginal BH; small raw p-values in C4/18-bit reflect modular comb structure, not k. |
| Sliding STFT (`st_`) | **no signal** | 0 / 120 marginal BH. |
| Wavelet (`wv_`) | **no signal** | 0 / 120 marginal BH. Wavelet operates on `\|φ\|` so vanishes for unit-modulus fields. |
| Multiscale entropy (`mse_`) | **no signal** | 0 / 160 marginal BH. The smallest raw p-value across the entire experiment (1.0e-4 on `mse_s1`, C4 at 16-bit) does not survive correction. |
| Legendre / multiplicative char (`ch_`) | **no signal** | 0 / 160 marginal BH; Legendre walk over `x(W·G)` is statistically indistinguishable from a balanced ±1 random walk, consistent with Hasse–Weil-type bounds. |
| Additive char sums (`ad_`) | **no signal** | 0 / 400 marginal BH; \|S_a\|/√m clusters at O(1) as expected for pseudo-random sequences. |
| Graph Laplacian (`gl_`) | **no signal** | 0 / 80 marginal BH; spectral gap of k-NN graph on `xs` is a property of the orbit point distribution, not of `k`. |

---

## 5. Distinguisher / cryptographic interpretation

Under DDH on a prime-order curve, no polynomial-time function of `(G, Q)`
can distinguish `Q = k·G` from `R = r·G` (random). Our extended battery
asks empirically: *do any of these 39 BPR-derived features form such a
distinguisher on toy curves, even at 18-bit?*

- **Net answer: no, at FDR-corrected α = 0.05.** 0/40 buckets survive on
  classifier AUC; 1/40 on MMD with no marginal or classifier
  corroboration.
- The single marginal MMD detection (C4, 18-bit) is consistent with the
  expected behavior of a non-additively-Q-coupled construction: it has
  a tiny higher-moment Q-dependence that an RBF-kernel two-sample test
  can in principle detect, but a linear classifier and per-feature
  t-tests cannot — meaning the dependence is non-linearly entangled
  across features with low signal-to-noise ratio.
- **No feature was found whose distribution depends on `k` in a way
  that survives FDR.** This is the cryptographically relevant null:
  even for the C-family constructions where the field is genuinely
  Q-dependent in the time domain, the feature distributions over random
  toy keys do not separate.

---

## 6. Methodological notes (one bug we caught)

We initially used `StratifiedKFold` for the classifier-AUC distinguisher.
For Q-blind constructions (A and B families) where `X_real == X_qsub`
pairwise per instance, the standard CV split places each member of a
pair into a different fold, which causes the classifier to learn the
*wrong* association for the held-out partner — producing systematic AUCs
near **0.04** (anti-correlated), not 0.5.

The fix is `GroupKFold(groups=instance_index)`, which keeps both
members of each `(real, qsub)` pair in the same fold. Post-fix, A/B
families correctly give AUC = 0.500 ± 0.001 across all bits, and the
distinguisher works as intended. We document this here so the same
trap is not stepped into for related experiments.

---

## 7. Final verdict (across all transform families)

**`no usable signal`.**

Justification:
1. **0 / 1560** marginal feature tests survive BH-FDR.
2. **1 / 40** joint MMD tests survives BH-FDR (`C4` at `18-bit`); not
   corroborated by AUC at the same bucket after BH; not present at
   12, 14, 16-bit; expected ≤ 1 false discovery at FDR α = 0.05 across
   40 tests.
3. **0 / 40** classifier-AUC tests survive BH-FDR.
4. **No transform family** shows a within-family enrichment of
   marginal hits beyond chance.
5. **No `k`-correlated signal** in any test that addresses k directly.
6. **Construction-blindness verified:** A/B families' MMD ≡ 0,
   AUC ≡ 0.500 — the distinguisher correctly recognizes Q-independent
   constructions.

### Verdict in the user's category list

| Category | Holds? |
|---|---|
| no signal | **YES** |
| construction artifact | yes (secondary): C6 produces strong period-7 peaks; C4 has tiny higher-moment Q-dependence detectable by MMD only |
| Q-independent by algebra | yes (secondary): C1 / C3 magnitude spectra were proven Q-invariant in the prior FFT-magnitude analysis; phase spectra differ but only by uniform shift and carry no `k` info |
| toy-only signal | no |
| weak-curve-only signal | not tested in this extension; the prior random-curve null suffices to terminate the secp256k1 path |
| possible DDH distinguisher | no — single MMD hit insufficient |
| search-relevant speedup | no — no feature reduces group-op count vs. Pollard rho; field build cost ≥ window scan |

---

## 8. Files

```
experiments/ecdlp_fft/
├── transforms.py                     # 39-feature transform library
├── distinguishers.py                 # MMD + group-aware classifier AUC
├── run_extended.py                   # extended driver
├── extended_summary.py               # marginal + joint stats with BH-FDR
├── REPORT_EXTENDED.md                # this file
└── data/
    ├── extended_features.csv         # 6,000 trial rows × 39 features
    ├── extended_marginal.csv         # 1,560 per-feature t-tests
    └── extended_joint.csv            # 40 MMD + AUC distinguisher results
```

Reproduce with:
```bash
python3 run_extended.py --bits 12 14 16 18 --instances 30 --keys-per-instance 1
python3 extended_summary.py
```

---

## 9. What would change this verdict

Same as the FFT-magnitude report, with one addition:

- A construction whose *feature distribution* (not magnitude spectrum,
  not phase spectrum, but joint feature-vector distribution) measurably
  differs between `(G, Q=k·G)` and `(G, R=r·G)` such that the difference
  (i) survives BH-FDR across full transform battery, (ii) scales
  monotonically with bit-width, and (iii) carries information about `k`
  itself, not merely a Q-or-R label.
- None of the ten BPR-style fields tested clears all three bars. The C4
  bucket clears (i) at one bit-width, fails (ii), fails (iii).
