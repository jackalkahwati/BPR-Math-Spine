# BPR Independence Audit — Memo

**Date:** 2026-04-06
**Question:** Find one observable that is (a) claimed derived from (p, z), (b) not downstream of the alpha formula, (c) not hardcoded or circular, then sweep it.

---

## Step 1 — Observable Classification

We examined every observable in the codebase and classified each one.

### Disqualified (not genuinely p/z-dependent)

| Observable | Why Disqualified |
|-----------|-----------------|
| `1/α(M_Z)` | = `1/α(0) − 9.084`, trivially downstream |
| `sin²θ_W` | `GaugeCouplingRunning` constructs threshold corrections to force unification, then recovers the hardcoded input 0.23122. Returns exactly 0.23122 for p = 3 or p = 999,983. Mathematical identity. |
| Lepton masses (`m_e`, `m_μ`) | `ChargedLeptonSpectrum` takes no p or z arguments. Constant. |
| `Ω_Λ` | `dark_energy_from_impedance` uses cosmological `p_cosmo = R_H/L_P ≈ 2.7×10^61`, ignoring the substrate p argument. Returns ~10^−104 for all p. |
| `\|V_cb\|` | `sqrt(m_s_exp/m_b_exp) / sqrt(ln(p) + z/3)` — uses PDG quark masses as direct inputs. Not derived. |
| `s13 = 0.150` in PMNSMatrix | Hardcoded as a literal constant. |
| `r` (tensor-to-scalar) | `12/N²` where N = p^(1/3)×4/3 — trivially downstream of n_s. |
| `BlackHoleEntropy` | `ln(p)` appears but cancels exactly; result is the Bekenstein-Hawking formula, independent of p. |

### Two Surviving Independent Candidates

**Candidate 1: `n_s` — Inflationary spectral index**

```
N   = p^(1/3) × (1 + 1/d)      d=3 (spatial dimensions, not a fit parameter)
    = p^(1/3) × 4/3

n_s = 1 − 2/N  =  1 − 3/(2 × p^(1/3))
```

- Depends on **p only** — uses `p^(1/3)`, which is mathematically unrelated to `[ln(p)]²`
- No z dependence at all
- No hardcoded experimental targets in the formula
- Experimental: `n_s = 0.9649 ± 0.0042` (Planck 2018)

The function `p^(1/3)` and the function `[ln(p)]²` cannot be expressed as functions of each other for arbitrary p, so n_s is **not algebraically downstream of 1/α**.

**Candidate 2: `δ_CP` — CKM CP-violating phase**

```
δ_CP = π/2 − 1/√(z+1)
```

- Depends on **z only** — uses `1/√(z+1)`, which is unrelated to `z/2`
- No p dependence
- No hardcoded experimental targets
- Experimental: `δ_CP = 1.196 ± 0.045 rad` (PDG 2024)

---

## Step 2 — Sweep Results

### n_s sweep: 51 primes near 104729, d=3 fixed

| Observation | Value |
|-------------|-------|
| n_s at p=104729 | 0.96818 |
| n_s experimental | 0.9649 ± 0.0042 |
| Signed error at baseline | +0.00328 (+0.78σ) |
| n_s trend over prime sweep | **Monotonically increasing** with p |
| Has local minimum at p=104729? | **No** |
| Prime minimizing n_s error in sweep | p = 104479 (leftmost) |
| Globally optimal prime for n_s alone | p ≈ 78,046 (26,682 below baseline) |

The n_s prediction changes negligibly over the ±500-unit sweep. Between the extreme primes p = 104,479 and p = 105,019, n_s varies by only 0.00005 — less than 1/100th of the experimental uncertainty. This observable provides almost no discriminating power within the local neighborhood.

### δ_CP sweep: z in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

| z | δ_CP (rad) | Error |
|---|-----------|-------|
| 4 | 1.124 | 1.61σ |
| 5 | 1.163 | 0.74σ |
| **6** | **1.193** | **0.07σ** |
| 7 | 1.217 | 0.47σ |
| 8 | 1.238 | 0.92σ |

z = 6 gives the closest match (0.07σ). The minimum IS at z = 6.

---

## Step 3 — Combined 3-Observable Sweep

We built a composite score from three genuinely independent observables:

1. **1/α** = `[ln(p)]²` + z/2 + γ − 1/(2π)
2. **n_s** = `1 − 3/(2 × p^(1/3))`
3. **δ_CP** = `π/2 − 1/√(z+1)`

These three use different functional forms: `[ln(p)]²`, `p^(1/3)`, and `1/√(z+1)`.

**Result:** When all three observables are included, a genuine local minimum appears in the score landscape — but at p ≈ 104,623, **not at p = 104,729**.

| p | Composite score | α err | n_s err | δ_CP err |
|---|----------------|-------|---------|----------|
| 104,623 (best) | **0.002485** | 203 ppm | 0.338% | 0.265% |
| 104,729 (baseline) | 0.002487 | 32 ppm | 0.340% | 0.265% |
| 104,743 (1/α best) | 0.002488 | 10 ppm | 0.340% | 0.265% |

The baseline ranks **#24 out of 51 tested primes** in the combined score. The spread across all 51 primes is only 1.1% (0.002485 to 0.002513) — a very shallow plateau.

Analytically, the true combined minimum (1/α + n_s, continuous p) is at p ≈ 104,600 — 129 units below 104,729. This is because:
- 1/α currently underestimates by 32 ppm → prefers slightly larger p
- n_s currently overestimates by 0.78σ → prefers significantly smaller p (~78,046)
- The balance between these two opposing gradients lands near p ≈ 104,600

---

## Step 4 — Does the Error Landscape Have a Local Minimum at (104729, 6)?

**For p (using n_s as independent test):**

| Question | Answer |
|----------|--------|
| Does a local minimum exist in the combined landscape? | Yes — at p ≈ 104,623 |
| Is the minimum at p = 104,729? | **No** |
| Is the minimum sharp? | **No** — score varies by only 1.1% across 51 primes |
| Does n_s alone prefer p = 104,729? | **No** — n_s prefers p ≈ 78,046 |
| Do 1/α and n_s agree on the optimal prime? | **No** — they differ by ~26,700 units |

**For z (using δ_CP as independent test):**

| Question | Answer |
|----------|--------|
| Does a local minimum exist in the δ_CP landscape? | Yes |
| Is the minimum at z = 6? | **Yes** (within integer z values tested) |
| Is the minimum sharp? | Moderately — drops from 4.5σ at z=2 to 0.07σ at z=6, then rises to 2.2σ at z=12 |
| Is z = 6 uniquely better than z = 5 and z = 7? | Yes — 0.07σ vs 0.74σ and 0.47σ |

---

## Is This Observable Independent?

**n_s: Genuinely independent.** The formula `1 − 3/(2 × p^(1/3))` uses `p^(1/3)`, which is not algebraically derivable from `[ln(p)]²` without knowing p independently. Given only 1/α (which encodes `[ln(p)]²`), you cannot recover `p^(1/3)`. These are different invertible functions of p, and knowing one does not determine the other.

**δ_CP: Genuinely independent.** The formula `π/2 − 1/√(z+1)` uses `1/√(z+1)`, which is not algebraically derivable from `z/2`. The 1/α formula encodes z through `z/2`; δ_CP encodes z through `1/√(z+1)`. These are different monotone functions of z, and one does not determine the other.

---

## Does the Evidence Support Uniqueness?

### p = 104,729:

**No.** The combined minimum is at p ≈ 104,623, not 104,729. The two independent observables (1/α and n_s) pull in opposite directions: 1/α prefers larger p while n_s prefers p ≈ 78,046. The balance point is not at 104,729. The landscape is shallow — 51 primes span only a 1.1% range in score. Within this neighborhood, ~10 primes score better than 104,729 when n_s is included.

Critically: **n_s provides almost no discriminating power near p = 104,729**. The quantity p^(1/3) varies by only 0.06% across the prime sweep, compared to [ln(p)]² varying by 0.6%. So n_s moves too slowly to pin down p precisely. The n_s error is stuck at ~0.78σ for all primes in the neighborhood.

### z = 6:

**Weakly yes, for δ_CP.** The δ_CP formula `π/2 − 1/√(z+1)` genuinely prefers z = 6 among integer values. The error at z = 6 is 0.07σ, significantly better than z = 5 (0.74σ) or z = 7 (0.47σ). The minimum is not sharp (δ_CP is a smooth function of z with no discontinuity), but within the integer values {4, 5, 6, 7, 8}, z = 6 gives the best match.

**However**, this only demonstrates that `π/2 − 1/√(z+1)` passes through the experimental δ_CP near integer z = 6. The key question is whether this formula was **derived from the framework** or **chosen to match the data**. The formula is given with no independent derivation of the coefficient 1/√(z+1) other than "boundary mode overlap phase."

---

## What Specifically Breaks the Uniqueness Claim

1. **n_s and 1/α prefer different primes.** The n_s-optimal prime is p ≈ 78,046. The 1/α-optimal prime is p ≈ 104,749. These are 26,700 units apart. A framework that uniquely selects a prime should have both observables agree. They don't.

2. **The combined minimum misses 104,729 by 106 units.** Even taking the most favorable view — combining both observables with equal weight — the optimal prime is p ≈ 104,623, not 104,729.

3. **n_s provides almost no discriminating power near 104,729.** The function p^(1/3) changes so slowly relative to [ln(p)]² that n_s adds almost no constraint on p locally. It contributes ~0.338% error for every prime in the sweep, varying by only 0.006%. It does not sharpen the minimum.

4. **δ_CP is consistent with z = 6 but not derived.** The formula `π/2 − 1/√(z+1)` uses integer arithmetic (z+1 = 7) in a way that looks ad hoc. The formula might have been constructed by asking "which simple expression of z gives the right δ_CP?" — which is post-hoc fitting. No independent derivation of the `1/√(z+1)` structure from boundary geometry is provided in the code.

5. **The entire n_s prediction is within 1σ for p = 78,046 through p = 999,983.** Because n_s changes so slowly with p, essentially any large prime gives an n_s prediction within 1σ of Planck. This makes n_s a non-discriminating test of uniqueness.

---

## Summary Table

| Observable | Independent? | p = 104729 minimizes? | z = 6 minimizes? | Discriminating? |
|-----------|-------------|----------------------|-----------------|----------------|
| 1/α | Core formula | No (trend → ∞) | Weakly yes | Yes |
| n_s | **Yes** (p^1/3 form) | No (prefers p≈78046) | N/A | **No** (too slow) |
| δ_CP | **Yes** (1/√(z+1) form) | N/A | **Yes** (0.07σ) | Moderate |
| v_EW | Partial (ln(p) overlap) | No (slope) | Weakly yes | Yes |
| sin²θ_W | No (circular) | — | — | — |
| Ω_Λ | No (broken code) | — | — | — |
| Lepton masses | No (constant) | — | — | — |

---

## Conclusion

The strongest genuinely independent observable is the **inflationary spectral index n_s**, which depends on `p^(1/3)` (a different functional form from `[ln(p)]²`). However, n_s is nearly flat across all primes near 104,729 — it provides no meaningful constraint on p in this neighborhood.

The combined minimum of {1/α + n_s} is at p ≈ 104,623, not p = 104,729. The difference is small but real, and the minimum is extremely shallow.

The **CKM CP-phase δ_CP** is the cleanest independent test of z. It genuinely prefers z = 6 among integer values tested. However, this preference may reflect post-hoc formula selection rather than physical derivation.

**Overall verdict on uniqueness:** p = 104,729 is not selected by any independent observable. z = 6 is supported by δ_CP, but the formula's derivation is questionable. No pair of observables simultaneously points to (104,729, 6) as a sharp optimum.

---

## What Would Decisively Confirm or Refute Uniqueness

The decisive test: find a third observable O₃ that:
1. Uses a functional form other than `[ln(p)]²`, `p^(1/3)`, and `1/√(z+1)`
2. Depends on **both** p and z jointly (not just one)
3. Is precisely measured experimentally (< 1% uncertainty)
4. Has NOT been used to construct any formula in the framework

The proton-to-electron mass ratio `m_p/m_e ≈ 1836.15` is the best candidate. If BPR can derive it from (p, z) with a formula that independently selects the same optimum as 1/α and n_s, that would be compelling evidence. If the derivation again shows a flat landscape or a different optimum, uniqueness is refuted.
