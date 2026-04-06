# BPR Viable Prime Corridor — Full Characterization
## Memo: Every Stable Candidate at z = 6

**Date:** 2026-04-06  
**Branch:** `claude/audit-bpr-parameters-6fSrQ`  
**Data:** `analysis/results/corridor/` (580 primes fully enumerated)  
**Plots:** `analysis/plots/corridor_*.png` (4 plots)

---

## 1. The Viable p Family

**580 primes** lie in the continuous 1%-threshold corridor at z = 6:

```
p ∈ [103,935 – 110,634]   width = 6,699 units   (6.4% of baseline)
```

Every prime in this range scores below 1.00% on the core RMS metric (√mean of squared fractional errors for 1/α and v_EW). No prime outside this range achieves sub-1%.

### Score distribution

| Zone | Score range | Count | Fraction |
|---|---|---|---|
| Elite | 0.00–0.35% | 99 | 17% |
| Good | 0.35–0.50% | 139 | 24% |
| Acceptable | 0.50–0.70% | 142 | 24% |
| Near-threshold | 0.70–0.85% | 104 | 18% |
| Fragile | 0.85–1.00% | 96 | 17% |

| Percentile | Score |
|---|---|
| 10th (best 10%) | 0.3257% |
| 25th | 0.3950% |
| 50th (median) | 0.5725% |
| 75th | 0.7847% |
| 90th (worst 10%) | 0.9180% |

The baseline p = 104,729 scores 0.7843%, placing it at the **75th percentile** of viable primes — meaning three quarters of the corridor performs better.

---

## 2. The Four Special Points

The corridor is not a flat family. It has clear internal structure shaped by the competing pulls of two core observables.

### α-optimal: p = 104,743

- 1/α error: **9.71 ppm** (minimum in corridor)
- v_EW error: 1.104%
- Core score: 0.781%
- This prime minimises the `ln(p)² + z/2 + γ − 1/2π` expression. It is the prime that the BPR `1/α` formula most precisely predicts.

### v_EW-optimal: p = 107,713

- 1/α error: 4,713 ppm
- v_EW error: **0.0014%** (essentially exact — 0.003 GeV off from 246.22)
- Core score: 0.333%
- This is the prime where `Λ_QCD × p^{1/3} × (ln(p) + z − 2)` equals the electroweak scale most closely. Its v_EW error is **800× smaller** than the baseline's.

### RMS-optimal: p = 107,251

- 1/α error: 3,986 ppm
- v_EW error: 0.169%
- Core score: **0.306%** (lowest in corridor)
- Neither α-optimal nor v_EW-optimal individually, but best balanced. Scores 2.5× better than the baseline overall.

### Robustness-optimal: p = 107,243

- Core score: 0.306%
- Robustness (mean score in ±200p neighbourhood): **0.31%** (minimum)
- Same cluster as the RMS-optimal; the neighbourhood around p ≈ 107,250 is consistently good.

### Summary table

| Role | p | 1/α ppm | v_EW % | Core score | Rank/580 |
|---|---|---|---|---|---|
| α-optimal | 104,743 | **9.71** | 1.104% | 0.781% | 433 |
| v_EW-optimal | 107,713 | 4,713 | **0.001%** | 0.333% | 13 |
| RMS-optimal | 107,251 | 3,986 | 0.169% | **0.306%** | 1 |
| Robust-optimal | 107,243 | 3,974 | 0.172% | 0.306% | 2 |
| **Baseline** | **104,729** | **32.3** | **1.109%** | **0.784%** | **435** |

---

## 3. Is p = 104,729 Special Within the Corridor?

**Yes — but specifically for one reason and for one observable only.**

### What makes it special

p = 104,729 has a 1/α error of **32.3 ppm**, making it the 4th-closest prime to the analytic α-minimum at p = 104,743. Within the 580 viable primes, only 3 others match `1/α` more closely. Its classification is "alpha-optimal", and it belongs to a cluster of ≈15 primes near the lower corridor edge that all share this property.

The baseline's 1/α match is genuinely exceptional in absolute terms: 32.3 ppm is a 1-in-3000 fractional error. The issue is that this performance is bought by proximity to the analytic zero of the `ln(p)²` term, not by anything structurally special about 104,729 specifically.

### What does not make it special

1. **Core RMS score:** 0.784% — 435th out of 580 (bottom 25%). Three-quarters of the corridor outperforms it on the combined metric.

2. **v_EW error:** 1.109% — **766× larger than the minimum** (0.001% at p=107,713). The baseline `v_EW` prediction is 243.5 GeV, 2.7 GeV below experiment.

3. **Robustness:** 75th percentile (0.0079%). Robustness is mediocre because the surrounding region (near corridor edge) degrades faster than the interior.

4. **Distance from corridor edges:** p = 104,729 is only 11.9% into the corridor from the lower edge. This positions it in a zone where score sensitivity is higher and the nearby performance degrades more rapidly than at the corridor centre.

### Conclusion on specialness

p = 104,729 is the **`1/α`-special prime** — the prime the framework most precisely predicts for fine-structure constant. That is a legitimate and nontrivial fact. However:

- It is not the most physically complete prediction (v_EW disagrees at 1.1%)  
- It is not the most robust (85th percentile would be a more stable choice)  
- It is not the globally best (p ≈ 107,251 scores 2.5× better)  
- It is not isolated — 14 other primes share the "alpha-optimal" classification

---

## 4. The Most Balanced Prime

**p = 107,251** (also p = 107,243–107,279 form a dense cluster):

- α error: 3,986–4,031 ppm — within acceptable range for theoretical physics (3 d.p.)
- v_EW error: 0.14–0.19% — better than experiment's own precision on the Higgs VEV
- Core score: 0.306–0.307%
- Robustness: top 10th percentile (consistently good neighbours)
- Classification: RMS-optimal + robust-interior + robust

This cluster of ≈20 primes around p ≈ 107,250 is the corridor's **sweet spot**: both observables are simultaneously well-matched, the neighbourhood is stable, and the score is 2.5× better than the baseline.

---

## 5. The Most Robust Prime

**p = 107,243**: lowest mean-neighbourhood score (0.0031%). The ±200-unit window around this prime has the most consistently low core scores in the corridor.

**Contrast with baseline robustness:**  
The baseline's ±200 window mean is 0.0079% — 2.5× less robust. This is a direct consequence of proximity to the lower corridor edge, where scores rise steeply as p decreases.

---

## 6. Corridor Structure: Flat Family or Structured Basin?

**Structured basin with asymmetric walls, not a flat family.**

Several lines of evidence establish this:

### Curvature

All curvatures are positive (range: 5.8×10⁻¹¹ to 2.6×10⁻⁹ per p²). The corridor is everywhere concave-up — a bowl, not a plateau. The curvature is smallest in the middle (flatter), highest near the edges (steeper walls). This means:

- The corridor has a genuine minimum region around p ≈ 107,000–108,000
- The walls become increasingly steep below p ≈ 105,000 and above p ≈ 110,000

### Asymmetric wall steepness

| Side | Score change per 1,000p |
|---|---|
| Lower edge (near p=104,000) | +0.26% per 1,000p (steeper) |
| Upper edge (near p=110,000) | +0.15% per 1,000p (gentler) |

The lower wall is ~1.7× steeper than the upper. This is because the `ln(p)²` term accelerates faster in the downward direction (where `1/α` overshoots) than the `v_EW` term does near the upper boundary.

### Two competing attractors

The corridor's shape is determined by the pull of two analytic optima:
- **α-attractor at p ≈ 104,749** (lower edge) — `ln(p)²` term minimised
- **v_EW-attractor at p ≈ 107,709** (interior, ~55% across) — dimensional hierarchy minimised

These sit 2,960 units apart. The RMS minimum at p ≈ 107,250 is the Pareto compromise point between them. The corridor boundary is set by where each observable individually reaches 1% error, not by the RMS score.

### Classification landscape

| Region | p range | Primes | Dominant character |
|---|---|---|---|
| Lower edge (fragile) | 103,935–104,270 | ~37 | α-optimal but fragile |
| Lower interior | 104,270–105,500 | ~130 | mid-range, α-leaning |
| Central interior | 105,500–108,500 | ~200 | RMS-optimal, robust |
| Upper interior | 108,500–110,100 | ~150 | v_EW-leaning |
| Upper edge (fragile) | 110,100–110,634 | ~26 | near v_EW threshold |

---

## 7. Physical Interpretation

### What the corridor reflects

The corridor is the set of primes where two independently motivated formulas — `ln(p)²` (electromagnetic fine structure) and `Λ_QCD × p^{1/3} × (ln(p) + z − 2)` (electroweak hierarchy) — are simultaneously within 1% of experiment. The corridor's width (~6,700 units) represents the fundamental **degeneracy** of the BPR framework: at current precision, all 580 primes are observationally equivalent.

### Why the corridor has the structure it does

The shape is not arbitrary. The two formulas have different p-dependencies:
- `1/α` depends on `ln(p)²` — grows slowly, with a unique zero near p ≈ 104,749
- `v_EW` depends on `p^{1/3} × ln(p)` — grows faster, zero near p ≈ 107,709

The corridor exists precisely because these two optima are close enough (within ~3,000 units) that a common region satisfies both. If the analytic optima were more than ~8,000 units apart, the corridor would vanish entirely.

### What it means for the choice of p = 104,729

The historically chosen baseline minimises the `1/α` error almost exactly. This was likely the observable used to select p in the first place — and in the narrow sense of `1/α` agreement, it is the best choice. But it carries a 1.1% v_EW error that 433 other primes in the corridor do not.

### The honest single-sentence description

> BPR predicts a coordination number of z = 6 uniquely and unambiguously, and a prime in the corridor p ∈ [103,935–110,634] — most likely near p ≈ 107,250 if both electroweak observables are given equal weight, or near p ≈ 104,743 if only fine-structure is weighted.

---

## 8. Summary Table: Five Key Primes

| Prime | Role | 1/α ppm | v_EW err | Core score | Rank | Special property |
|---|---|---|---|---|---|---|
| 104,743 | α-optimal | **9.7** | 1.104% | 0.781% | 433 | Nearest to α-minimum |
| **104,729** | **Baseline** | **32.3** | **1.109%** | **0.784%** | **435** | **Historically chosen** |
| 107,243 | Robust-optimal | 3,974 | 0.172% | 0.306% | 2 | Stable neighbourhood |
| 107,251 | RMS-optimal | 3,986 | 0.169% | **0.306%** | **1** | Best balanced |
| 107,713 | v_EW-optimal | 4,713 | **0.001%** | 0.333% | 13 | Nearest to EW-minimum |

---

## 9. Recommended Derivation Target

The existence of two competing attractors (α and v_EW) at different primes is itself a falsifiable prediction: **if the BPR framework is correct, its derivation of the v_EW formula should pick out the same prime as its derivation of 1/α**. Currently they disagree by 2,960 units, which is the corridor's fundamental unresolved tension.

A derivation that shows both formulas have a simultaneous extremum at the same prime — or a third independent observable that selects one prime from the 580 — would be the minimal result needed to make the choice of p meaningful rather than a free parameter within the corridor.
