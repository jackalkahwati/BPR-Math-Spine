# BPR Third Observable Analysis
## Memo: What Can Break the 580-Prime Degeneracy?

**Date:** 2026-04-06  
**Branch:** `claude/audit-bpr-parameters-6fSrQ`

---

## 1. The Core Question

The viable corridor contains 580 primes at z=6. Two of them are structurally special:
- **p ≈ 104,749** — the α-attractor (minimises `1/α` error)
- **p ≈ 107,709** — the v_EW-attractor (minimises electroweak scale error)

These are not the same prime. The gap is 2,960 units. For the framework to predict a unique p, it needs either a third observable to create a three-way intersection, or a shared primitive that forces both existing attractors to coincide.

This memo assesses every candidate path.

---

## 2. Why the Gap Is Structural, Not Numerical

The two attractor equations are:

```
1/α zero:   ln(p)² + z/2 + γ − 1/(2π) = 137.036
            → p* = exp(√(137.036 − 3.418)) ≈ 104,749

v_EW zero:  0.332 × p^{1/3} × (ln(p) + z − 2) = 246.22 GeV
            → p* ≈ 107,709  [solved numerically]
```

These are **transcendental equations in algebraically independent functional forms**. `ln(p)²` and `p^{1/3} × ln(p)` have no algebraic relationship over the reals for large p. Their zeros cannot coincide unless one of the equations changes form, or unless the two experimental targets (137.036 and 246.22 GeV) happen to encode the same prime — which they currently do not.

### The gap encodes experimental tension

At p = 104,749 (α-optimum), BPR predicts v_EW = **243.51 GeV** vs. experimental 246.22 GeV — a gap of −2.71 GeV (−1.10%).

At p = 107,709 (v_EW optimum), BPR predicts 1/α = **137.681** vs. experimental 137.036 — a gap of +0.645 (+4,707 ppm).

The 2,960-unit attractor gap is a direct encoding of a 1.1% tension between the experimental values of `1/α` and `v_EW` as seen through their respective BPR functional forms. It is not noise. It cannot be tuned away without changing a formula.

---

## 3. The Three Candidate Third Observables — Assessed

### 3.1 α_s(M_Z) via GUT Running

**Status: Blocked. Must fix GUT coupling formula first.**

BPR provides `α_GUT = π / (p^{1/3} × z)` and `M_GUT = M_Pl / p^{1/4}`.  
Standard one-loop QCD running from these to M_Z gives:

| p | 1/α_GUT | M_GUT | α_s(M_Z) BPR | α_s exp | ratio |
|---|---|---|---|---|---|
| 104,729 | 90.0 | 6.79×10¹⁷ GeV | 0.0203 | 0.1179 | 0.17× |
| 107,251 | 90.7 | 6.75×10¹⁷ GeV | 0.0200 | 0.1179 | 0.17× |

**BPR predicts α_s(M_Z) ≈ 0.020, off by a factor of 6.**

Root cause: BPR's `1/α_GUT ≈ 90`, but reproducing the correct α_s(M_Z) requires `1/α_GUT ≈ 49` (given BPR's M_GUT scale). Standard GUT unification gives `1/α_GUT ≈ 24-25`, with a different M_GUT ≈ 2×10¹⁶ GeV.

The discrepancy has two components: BPR's GUT scale is ~40× too high, and its GUT coupling is ~3.6× too weak. These partially cancel in the running, but not enough.

**Verdict:** α_s cannot serve as a third observable until the GUT-sector formulas are corrected. This is actually a pre-existing problem in the framework that the corridor analysis surfaces — the GUT coupling formula `α_GUT = π/(p^{1/3} × z)` is not self-consistent with the low-energy 1/α formula.

### 3.2 Top Quark Mass m_t

**Status: Downstream of v_EW unless Yukawa is independently derived.**

In the SM, `m_t = y_t × v_EW / √2` with `y_t ≈ 0.994`.

If BPR sets `y_t = 1` (simplest ansatz), then `m_t = v_EW / √2`:

| p | v_EW | m_t predicted | m_t exp | error |
|---|---|---|---|---|
| 104,729 | 243.49 GeV | 172.17 GeV | 172.76 GeV | 0.34% |
| 107,251 | 245.80 GeV | 173.81 GeV | 172.76 GeV | 0.61% |
| 107,713 | 246.22 GeV | 174.10 GeV | 172.76 GeV | 0.78% |

The p-optimum for `m_t` (under y_t=1) falls at p ≈ **105,630** — not coinciding with either existing attractor, but much closer to the α-attractor than the v_EW-attractor. However, this is entirely downstream of v_EW — no new p-information enters.

`m_t` becomes an independent third observable **only if** the top Yukawa coupling `y_t` is derived from `(p, z)` independently of `v_EW`. That requires a new formula for the Yukawa sector.

**Verdict:** m_t as an independent observable requires deriving the top Yukawa coupling, which is a substantially harder derivation than any currently in the codebase.

### 3.3 Proton/Electron Mass Ratio m_p/m_e

**Status: No formula exists. Requires new derivation from first principles.**

`m_p/m_e = 1836.15267...` measured to 6×10⁻¹⁰ — extraordinary precision.

No BPR formula for this exists. A derivation would need to calculate both the proton mass (a QCD composite bound state) and the electron mass (from the leptonic Yukawa sector) from the substrate prime p. The functional form is unknown in advance, which is what makes it simultaneously the most powerful and most difficult target.

The corridor variation in any `p`-dependent function is 1–3% across [103,935–110,634], so once derived, even a formula with modest theoretical motivation would have sufficient discriminating power given the extraordinary experimental precision.

**Verdict:** Long-term target. Not assessable until a derivation exists.

---

## 4. Four Paths to Uniqueness

Any path to predicting a unique substrate prime must do one of:

**(A) Change the v_EW functional form**  
Replace `p^{1/3}` with `exp(√(ln(p)² − const))` or similar, so the v_EW attractor moves to p ≈ 104,749. Requires deriving why the electroweak hierarchy formula has a different p-structure than previously assumed. Must come from a calculation, not a fit.

**(B) Add a p^{1/3} correction to the 1/α formula**  
A small `p^{1/3}/p` or `1/(p^{2/3} ln p)` boundary correction would shift the α-attractor toward p ≈ 107,709 without changing the dominant `ln(p)²` structure. Must be motivated by the underlying boundary phase dynamics, not by the fact that it closes the gap.

**(C) Find a third independent observable**  
The observable must have:
- A functional form independent of both `ln(p)²` and `p^{1/3} ln(p)`
- No reliance on hardcoded experimental inputs
- Experimental precision better than ~1% (to resolve the 2.4% gap between attractors)

Best candidates requiring derivation: `m_p/m_e`, top Yukawa, a derived `α_s` after fixing the GUT coupling.

**(D) Show both formulas are outputs of one primitive**  
Requires finding `Q(p)` such that `F(Q(p)) = 1/α_0` and `G(Q(p)) = v_EW` have their zero-error conditions satisfied at the same p. The constraint is that `F` and `G`, evaluated at a single value of `Q`, must simultaneously match experiment. No such `Q` is currently known to exist.

---

## 5. The Most Actionable Path

The α_s failure reveals a pre-existing inconsistency in BPR's GUT sector: `1/α_GUT ≈ 90` is the bare lattice coupling, but the standard GUT prediction at M_GUT ≈ 2×10¹⁶ GeV is `1/α_GUT ≈ 24-25`, and at BPR's higher scale M_GUT ≈ 6.8×10¹⁷ GeV it would be still smaller.

**The actionable order is:**

1. Derive a self-consistent GUT sector: show `M_GUT` and `α_GUT` are jointly determined by `(p, z)` and reproduce the known relationship between them. This closes the internal inconsistency and may shift one or both attractors.

2. From a corrected GUT coupling, derive `α_s(M_Z)` via standard RG running. If the corrected `1/α_GUT ≈ 49` (the value needed to reproduce α_s), then α_s becomes a genuine third observable.

3. Check whether a self-consistent GUT sector changes the location of either attractor. It may automatically close part of the 2,960-unit gap.

---

## 6. Summary

| Path | Status | Blocker | Payoff if unblocked |
|---|---|---|---|
| α_s(M_Z) via RG | **Blocked** | GUT coupling off by 3.6× | High — would add independent p constraint |
| m_t with derived y_t | **Blocked** | Yukawa not derived | Medium — attractor near α-side |
| m_p/m_e | **No formula** | Requires new QCD derivation | Highest — extraordinary precision |
| Fix GUT sector first | **Actionable** | Internal inconsistency | Unlocks α_s path |
| Third observable (generic) | **Open** | Needs independent derivation | Depends on functional form |

**The most concise characterisation of the current state:**

> The framework has a genuine structural preference for z=6 and a real constraint on p. The p-constraint currently takes the form of a 6,700-unit corridor rather than a unique prime, because the two core formulas map to two different functional forms of p and their error zeros land at different primes. The gap between those zeros is exactly 2,960 units and encodes a 1.1% tension between the experimental values of 1/α and v_EW as seen through the BPR functional forms. Closing the gap requires deriving both formulas from the same mathematical primitive — or adding a third independently derived observable that creates a well-defined three-way intersection.
