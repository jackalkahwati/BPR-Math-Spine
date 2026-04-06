# BPR Solution Space Map
## Memo: Full Landscape Analysis of (p, z) Parameter Space

**Date:** 2026-04-06  
**Branch:** `claude/audit-bpr-parameters-6fSrQ`  
**Files:** `analysis/solution_space.py`, `analysis/solution_space_plots.py`  
**Data:** `analysis/results/solution_space/`  
**Plots:** `analysis/plots/solution_space_*.png` (7 plots)

---

## 1. Parameter Space Explored

**Total points computed:** 3,938 (358 primes × 11 z values)

**Primes:** Multi-tier logarithmic sampling  
- Dense window: p ∈ [104,229, 105,229] (±500 around baseline, ~200 primes)  
- Medium window: p ∈ [99,729, 109,729] (±5,000, ~100 primes)  
- Wide range: p ∈ [30,000, 300,000] (log-spaced, ~300 primes)

**Coordination number:** z ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

**Baseline reference:** p = 104,729, z = 6

---

## 2. Observable Provenance

Each observable is classified before scoring. This classification is not post-hoc; it was set in `analysis/solution_space.py` before the sweep ran.

| Observable | Status | Formula | Included in Core Score |
|---|---|---|---|
| `inv_alpha_0` | **core-derived** | `ln(p)² + z/2 + γ − 1/(2π)` | YES |
| `v_EW_GeV` | **bridge formula** | `Λ_QCD × p^{1/3} × (ln(p) + z − 2)` | YES |
| `n_s` | **ansatz** | `1 − 2/N, N = p^{1/3} × 4/3` | NO |
| `δ_CP` | **ansatz** | `π/2 − 1/√(z+1)` | NO |
| `sin²θ_W` | **circular** | Returns hardcoded 0.23122 for all (p,z) | NO |
| `Ω_Λ` | **broken** | Uses Hubble-radius mode count, returns ~10⁻¹⁰⁴ | NO |
| `1/α(M_Z)` | **downstream** | `inv_alpha_0 − 9.084` (trivially derived) | NO |

**Core score** = RMS fractional error of {`inv_alpha_0`, `v_EW_GeV`}.  
**All-in score** = RMS of all four varying observables (ansatz observables included but labeled).

---

## 3. Core-Only Landscape

### 3.1 Where (104729, 6) Sits

| Metric | Value |
|---|---|
| Core score | 0.007843 (0.78%) |
| Rank | 49th / 3,938 (top **1.2%**) |
| `1/α` error | 32.3 ppm (0.0032%) |
| `v_EW` error | 1.109% |
| `n_s` error (ansatz) | 0.34% — 0.78σ from Planck 2018 |
| `δ_CP` error (ansatz) | 0.003 rad — 0.07σ from PDG |
| Local score gradient | −2.66 per 10⁶ units of p |

The baseline is a **top-tier but not globally optimal** point. It excels at `1/α` (32.3 ppm) while carrying a 1.1% `v_EW` error.

### 3.2 The Genuine Viable Corridor

The continuous 1%-threshold corridor at z=6 is:

```
p ∈ [103,935, 110,634]   (width ≈ 6,700 units, 6.4% of baseline)
```

**96 sampled primes** fall inside this corridor, all at z=6.  
No prime at any other z achieves sub-1% core score.

This is the **solution family**: not a point, but a 6,700-unit strip at z=6.

### 3.3 The Two Core Observables Pull in Opposite Directions

Both observables are minimized at different primes within z=6:

| Observable | p-optimal | Error at optimum |
|---|---|---|
| `1/α` | **104,749** | 9.7 ppm (1.1% v_EW) |
| `v_EW` | **107,709** | 0.00003% (4707 ppm 1/α) |
| RMS minimum | **107,250** | 0.306% (compromise) |

The baseline p=104,729 is essentially the `1/α`-optimal prime — it sits 20 units below the analytic `1/α` minimum. It sacrifices v_EW precision (1.1%) to achieve near-perfect `1/α` (32.3 ppm).

**Interpretation:** p=104,729 is not the globally best core point. It is the prime that most precisely matches `1/α`, at the cost of a 1.1% electroweak scale error. The framework does not explain why `1/α` precision should be prioritized over v_EW precision.

---

## 4. Full Phenomenology Landscape (Ansatz Layer)

### 4.1 z Dependence (Core Observables)

z=6 is the **only** integer coordination number with any sub-1% solutions in the sampled prime range.

| z | Best Core Score | Sub-1% Points | Sub-5% Points | δ_CP σ-error |
|---|---|---|---|---|
| 2 | 17.7% | 0 | 0 | 4.50σ |
| 3 | 13.2% | 0 | 0 | 2.78σ |
| 4 | 8.6% | 0 | 0 | 1.61σ |
| 5 | 4.1% | 0 | 8 | 0.74σ |
| **6** | **0.31%** | **96** | **157** | **0.07σ** |
| 7 | 2.3% | 0 | 156 | 0.47σ |
| 8 | 6.7% | 0 | 0 | 0.92σ |
| 9–12 | 5–18% | 0 | 0 | 1.3–2.2σ |

z=6 is **discontinuously better** than all neighbors. This is a genuine structural preference, not a smooth minimum. The framework's `z/2` contribution to `1/α` makes z=6 analytically optimal near p≈104,749, while `v_EW` also prefers z≈6.17. Both core observables independently point to the same z.

### 4.2 The Ansatz Observables (Separate Layer)

**n_s at baseline:** 0.9682 (exp: 0.9649 ± 0.0042) — 0.78σ.  
- The `n_s`-optimal prime at z=6 is p≈78,046 — far from the core corridor.
- Within the viable corridor [103,935–110,634], n_s error ranges from 0.62σ to 0.93σ — all acceptable, but none optimal.
- n_s adds no discriminating power within the viable corridor (too flat).

**δ_CP at baseline:** 1.1928 rad (exp: 1.196 ± 0.045) — 0.07σ.  
- Depends only on z, so all primes in the corridor give the same δ_CP.
- z=6 is genuinely the best integer z (0.07σ), with z=7 giving 0.47σ and z=5 giving 0.74σ.
- **Caveat (from derivation audit):** The formula `π/2 − 1/√(z+1)` was reverse-engineered — the fallback value in the source code is exactly 1.196 rad (PDG). The `+1` offset that selects z=6 is not derived. This undermines the evidential value of the δ_CP match.

---

## 5. Pareto-Optimal Regions

**Four-objective Pareto front** (minimizing `inv_alpha_err`, `v_EW_err`, `n_s_err`, `δ_CP_err`): **246 points** out of 3,938.

The baseline is **Pareto-optimal** — confirming it cannot be improved on all four objectives simultaneously. However, this is a relatively weak constraint (246/3938 = 6.2% of points are Pareto-optimal in 4D).

The **2-objective core Pareto front** (just `1/α` and `v_EW`) defines the fundamental tradeoff curve: any point moving lower on the front gains v_EW precision at the cost of `1/α` precision. The entire viable corridor lies near the knee of this curve.

---

## 6. Solution Families

### Family 1 (only viable family at ≤1%): z=6 corridor

```
p ∈ [103,935, 110,634] at z = 6
96 sampled prime representatives
Width: 6,700 units (6.4% of baseline)
Contains baseline: YES (104,729 is near the lower edge)
Best point: p=107,323 (or continuously p≈107,250)
```

**Structure of the family:**  
The corridor is shaped by the competing pulls of `1/α` (prefers p≈104,749) and `v_EW` (prefers p≈107,709). The lower bound (~103,935) is where `1/α` error grows to 1% of the total. The upper bound (~110,634) is where `v_EW` error swings back past 1%.

### Family 2 (z=5 and z=7 shoulder, 1–5% range):

```
z=5: p ∈ [105,971, 109,721]  (8 pts, 1.6–4.1% core)
z=7: p ∈ [99,761, 109,097]  (156 pts, 2.3–5.0% core)
```

These families share no overlap with the z=6 sub-1% family. At z≠6, the analytic `1/α` minimum moves to a different prime range, preventing simultaneous good fits on both core observables.

---

## 7. Key Questions Answered

| Question | Answer |
|---|---|
| **What regions of (p,z) are viable?** | One primary family: z=6, p∈[103,935–110,634]. Two shoulder families at z=5/7 above 1.6%. No other z achieves sub-5% at any p in the range studied. |
| **Are there multiple disconnected viable regions?** | No sub-1% disconnected families. At z=6, the corridor is continuous. |
| **Is 104,729 inside a viable basin?** | Yes — it is near the lower edge of the z=6 corridor (1.2nd percentile overall). |
| **Is z=6 part of a wider low-z family, or genuinely distinct?** | **Genuinely distinct.** No other z achieves sub-1% core score. The z=5/7 shoulders are qualitatively worse by a factor of 5–15×. |
| **Which observable constrains p most?** | `1/α` constrains p very tightly (varies as ~8000 ppm/1000 units) but its optimal p (104,749) is 2960 units below the v_EW optimum (107,709). Together they create a corridor, not a point. |
| **Which observable constrains z most?** | Both `1/α` and `v_EW` jointly constrain z=6. `1/α` prefers z≈6.01 analytically; `v_EW` prefers z≈6.17. They converge on z=6. |
| **Are there multiple solution families?** | No, not at <1% precision. There is one family: z=6, p≈104k–110k. |
| **Does the framework look rigid, underdetermined, or family-valued?** | **Family-valued at the z level; corridor-valued at the p level.** z is pinned sharply to 6. p is constrained to a ~6,700-unit corridor. Within that corridor, the framework cannot further discriminate (it would need a third independent observable). |

---

## 8. Where the Baseline Sits

```
p = 104,729  is the 1/α-optimal prime.
p = 107,250  is the RMS-optimal prime.
p ∈ [103,935, 110,634] is the viable corridor.
```

The baseline is not the best core-score point — it lies 2,521 units below the RMS minimum. It is the prime that was historically chosen, and its uniqueness claim was based on `1/α` match alone. The `v_EW` formula (added later, also as a bridge formula) actually prefers a different prime.

**Structural interpretation:** p=104,729 achieves a near-exact `1/α` match (32.3 ppm) because the formula `ln(p)² + z/2 + ...` happens to land near 137.036 there. There are dozens of other primes in the corridor that achieve sub-1000 ppm `1/α` error and substantially better `v_EW` alignment.

---

## 9. Most Promising Alternative Regions

| Rank | p | z | Core Score | `1/α` error | `v_EW` error | Notes |
|---|---|---|---|---|---|---|
| 1 | 107,323 | 6 | 0.31% | 4100 ppm | 0.14% | Best RMS tradeoff |
| 2 | 106,801 | 6 | 0.33% | 3276 ppm | 0.34% | Near-best |
| 3 | 107,941 | 6 | 0.36% | 5071 ppm | 0.09% | v_EW optimal nearby prime |
| 4 | 106,367 | 6 | 0.40% | 2588 ppm | 0.50% | — |
| **26** | **104,729** | **6** | **0.78%** | **32 ppm** | **1.11%** | **Baseline** |

The baseline ranks 26th in the top-49 list by core score — well within the corridor, but not at the minimum.

---

## 10. Recommended Next Steps

### Derivation targets (to rescue uniqueness)

1. **Derive the v_EW formula non-perturbatively.** The formula `Λ_QCD × p^{1/3} × (ln(p) + z − 2)` was motivated heuristically. If it can be derived with higher precision from first principles, the `v_EW` constraint will become sharper, potentially narrowing the p corridor significantly.

2. **Find a third independent observable that varies with p.** Currently the framework has two core observables and two free parameters — zero degrees of freedom. A third independent observable would over-constrain the system and either falsify or strengthen the framework. Candidates:
   - **Proton/electron mass ratio:** `m_p/m_e ≈ 1836` — could potentially be derived from boundary modes
   - **Strong coupling `α_s(M_Z)`:** currently not computed from (p,z) in the repo
   - **Neutrino mass scale:** if the framework extends to neutrino physics

3. **Derive `N = p^{1/3} × (1 + 1/d)` from the slow-roll equations.** This would convert the `n_s` formula from ansatz to core-derived, adding a genuine independent constraint on p. The p-optimal for `n_s` (≈78,046) is far from the corridor, so this derivation would also test whether the framework is internally consistent.

4. **Derive `δ_CP = π/2 − 1/√(z+1)` from the boundary geometry.** Compute the boundary overlap integral explicitly on an S² boundary with z=6 coordination. Specifically justify why the denominator is `√(z+1)` and not `√(z)`. If the derivation succeeds, the δ_CP constraint would sharpen the z selection significantly.

### Structural questions

5. **Is there a theoretical reason to prefer the `1/α`-optimal prime (104,729) over the RMS-optimal prime (107,250)?** The 2,521-unit gap between them needs justification if the framework is to make a uniqueness claim.

6. **What is the correct scoring weight between `1/α` and `v_EW`?** Currently they are equally weighted in the RMS. If `1/α` is more precisely measured (σ_exp = 2.1×10⁻⁸ vs σ_v_EW ≈ 0.02 GeV), a sigma-weighted score shifts the optimal prime significantly toward p≈104,749.

---

## 11. Summary

| Claim | Status |
|---|---|
| "z=6 is uniquely preferred" | **SUPPORTED** — only z=6 achieves sub-1% core score; the preference is sharp and not gradual |
| "p=104,729 is uniquely preferred" | **NOT SUPPORTED** — 95 other primes at z=6 achieve sub-1% core score; 104,729 is the `1/α`-optimal prime but not the globally best |
| "The framework is tightly constrained" | **PARTLY** — z is tightly constrained; p is corridor-constrained to ≈6.4% window |
| "No alternative parameters work as well" | **FALSE** — p≈107,250 has a 2.5× better core score than p=104,729 |
| "The framework has zero free parameters" | **MISLEADING** — it has 2 free parameters and 2 genuine observables = 0 d.o.f., but the degeneracy within the corridor shows the system is not over-constrained |

The most honest current description:

> BPR has a **genuine structural preference for z=6** from the joint intersection of its two core formulas. Within z=6, it has a **~6,700-unit viable corridor** centered around p≈107,000. The historically chosen baseline p=104,729 sits near the lower edge of this corridor and is the prime that most closely matches `1/α`, but is not the globally best fit. Reaching a uniqueness claim for p would require either a third independent observable or a derivation that explains why `1/α` precision should outweigh v_EW precision.
