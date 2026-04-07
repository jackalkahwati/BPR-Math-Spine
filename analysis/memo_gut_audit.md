# BPR GUT Sector Audit
## Memo: α_GUT and M_GUT — Derivation, Consistency, and Impact

**Date:** 2026-04-07  
**Branch:** `claude/audit-bpr-parameters-6fSrQ`  
**Files:** `analysis/gut_audit.py`  
**Data:** `analysis/results/gut_audit/`

---

## 1. Formulas Audited

Two GUT-sector objects appear in `bpr/alpha_derivation.py` (root commit `61fc581`):

```python
def alpha_gut_from_lattice(p, z):
    n_modes = p ** (1.0 / 3.0)
    return np.pi / (n_modes * z)        # α_GUT = π / (p^{1/3} · z)

def gut_scale_GeV(p):
    return _M_PL_GEV / p ** 0.25        # M_GUT = M_Pl / p^{1/4}
```

For the baseline `p=104729, z=6`:
- `M_GUT = 1.22×10¹⁹ / 17.97 = 6.79×10¹⁷ GeV`
- `α_GUT = π / (47.14 × 6) = 0.01109`  →  `1/α_GUT = 90.0`

---

## 2. Provenance — Both ANSATZ

Both formulas appear in the root commit with no derivation preceding them.

**`M_GUT = M_Pl / p^{1/4}`**

The docstring gives the formula in one line with no calculation. There is no physical argument shown for why the GUT scale should be the Planck scale divided by `p^{1/4}`. In a standard GUT, M_GUT is the energy where the three SM gauge couplings meet — determined by the RG flow, not by a substrate parameter.

**`α_GUT = π / (p^{1/3} · z)`**

The derivation sketch in the docstring is: define `N_eff = p^{1/3} · z / (2π)`, then set `α_GUT = 1/(2 N_eff)`. The N_eff formula counts boundary modes but is asserted without a path-integral calculation. The code itself contains an explicit acknowledgment:

```
NOTE: The effective α_GUT *at* M_GUT (after boundary-mode threshold
corrections between M_Pl and M_GUT) differs from this bare value.
See derive_couplings_top_down for the full chain.
```

And the `alpha_em_from_gut_running` function explicitly states:

```
"Bare GUT coupling without threshold corrections. Full calculation requires
S² cohomology charges for boundary mode representations (η_i values)."
```

**Verdict: Both formulas are ansatz with a placeholder note that the real derivation is absent.**

---

## 3. Internal Consistency — Three Compounding Failures

Running the BPR GUT sector through standard one-loop SM gauge coupling equations reveals a tripartite inconsistency. With `M_GUT = 6.79×10¹⁷ GeV` and `1/α_GUT = 90`:

| Observable | BPR prediction | Experimental | Error |
|---|---|---|---|
| `α_s(M_Z)` | 0.0203 | 0.1179 | **82.8%** |
| `sin²θ_W(M_Z)` | 0.274 | 0.231 | **18.5%** |
| `1/α_em(M_Z)` | 261.4 | 127.952 | **104%** |

No single choice of `1/α_GUT` can simultaneously reproduce all three SM coupling constants with BPR's M_GUT. The fundamental reason is that **BPR's M_GUT is not a coupling-unification scale in the SM**:

- At BPR's M_GUT = 6.79×10¹⁷ GeV, SM running predicts:  
  `1/α₁ = 35.2`, `1/α₂ = 48.0`, `1/α₃ = 49.2` — spread of **14.0 units**
- Full unification requires spread → 0
- Actual SM α₂/α₃ near-unification occurs at M ≈ 9.6×10¹⁶ GeV — **7× below** BPR's scale

### The coupling ratio

BPR's bare `1/α_GUT = 90.0` vs. the SM QCD coupling at the same scale `1/α₃ = 49.2`:

```
Ratio = 90.0 / 49.2 = 1.83
```

BPR's GUT coupling is **1.83× weaker** than what QCD actually predicts at that energy scale. The claimed coupling is not a physical GUT coupling; it is a lattice counting formula (`π/N_eff`) that produces the wrong value by a factor of ~2.

---

## 4. Impact on the Corridor Attractors

**The GUT sector problems do NOT shift the corridor attractors.**

The two attractors at p ≈ 104,749 (α-attractor) and p ≈ 107,709 (v_EW-attractor) are determined entirely by the **low-energy formulas**:

- `1/α₀ = ln(p)² + z/2 + γ − 1/(2π)` — these are the M_Z-scale and zero-momentum results
- `v_EW = Λ_QCD × p^{1/3} × (ln(p) + z − 2)`

Neither of these uses `α_GUT` or `M_GUT` as an input. The GUT sector objects are computed as downstream outputs, not as inputs to the core formulas. A corrected `α_GUT` formula would not move either attractor unless it also led to a corrected form of the `1/α₀` formula.

---

## 5. Number-Theoretic Findings

The most structurally significant finding from the full GUT audit is not about the GUT sector — it is about **why p = 104,729 was chosen**.

```
p = 104,729  is the 10,000th prime  [π(104729) = 10,000]
p = 104,743  is the 10,001st prime  [α-optimal, 9.7 ppm error]
```

The baseline is not the prime that most precisely matches `1/α`. The prime with the best `1/α` match is 104,743 — one step later in the prime sequence. The baseline was chosen as a memorable round number in the prime counting function. The choice of the 10,000th prime is analogous to choosing `N = 100` or `N = 1000` — it is a human-readable landmark, not a physical optimum.

| Prime | Rank | `1/α` error | `v_EW` error | Core score | Role |
|---|---|---|---|---|---|
| 104,723 | 9,999th | 41.9 ppm | 1.117% | 0.790% | — |
| **104,729** | **10,000th** | **32.3 ppm** | **1.109%** | **0.784%** | **Baseline** |
| 104,743 | 10,001st | 9.7 ppm | 1.104% | 0.781% | α-optimal |

The baseline is 23 units below the analytic `1/α` minimum (~104,749) and one prime before the best discrete match (104,743). It is the nearest landmark prime to the α-minimum, not the α-minimum itself.

This is a definitive answer to one of the four audit questions: **p = 104,729 is the 10,000th prime. That is why it was chosen.**

---

## 6. Summary Table: All Audit Findings

| Formula / claim | Status | Key evidence |
|---|---|---|
| `M_GUT = M_Pl/p^{1/4}` | ANSATZ | Root commit; not a unification scale in SM |
| `α_GUT = π/(p^{1/3}·z)` | ANSATZ | Root commit; wrong by 1.83× vs SM at same scale |
| GUT coupling self-consistency | FAILS | No single `1/α_GUT` reproduces all three SM couplings |
| `α_s(M_Z)` from BPR | 82.8% off | 0.0203 vs 0.1179; wrong by factor 5.8× |
| `sin²θ_W` from top-down | 18.5% off | 0.274 vs 0.231 |
| Choice of p = 104,729 | 10,000th prime | Landmark in prime counting function, not physical optimum |
| GUT impact on attractors | NONE | Attractors depend only on low-energy formulas |

---

## 7. Conclusion

The BPR GUT sector is an incomplete scaffold, not a derivation. Both formulas were introduced in the root commit as ansatz. The docstring explicitly notes that the full calculation "requires S² cohomology charges for boundary mode representations" — a derivation that does not exist in the codebase. The numerical predictions for `α_s(M_Z)` and `sin²θ_W` are 83% and 19% off respectively, and no single GUT coupling value can fix all three SM observables simultaneously with BPR's M_GUT scale.

The GUT sector failure does **not** affect the main corridor results — the z=6 preference and the 580-prime corridor are determined by low-energy formulas that are independent of the GUT sector. However, it does mean that:

1. `α_s(M_Z)` cannot serve as a third independent observable without first reconstructing the GUT sector from scratch
2. The BPR framework's high-energy sector is currently unconstrained by experiment
3. The framework's internal consistency is broken at the GUT level

And independently: the baseline p = 104,729 was chosen because it is the 10,000th prime — a memorable landmark, not the physical optimum for any observable.
