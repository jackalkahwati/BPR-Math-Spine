# BPR Derivation Audit — Memo

**Date:** 2026-04-06
**Question:** Were n_s and δ_CP derived from prior principles, or are they phenomenological ansätze inserted to match known values?

---

## What Was Examined

- Full git history of `bpr/cosmology.py` and `bpr/qcd_flavor.py`
- Root commit content and message
- All documentation: `doc/bpr_textbook/main.tex`, `doc/derivations/DERIVATION_ROADMAP.md`, `doc/derivations/PREDICTION_AUDIT_10_TO_100.md`
- All tests referencing `InflationaryParameters` and `CKMMatrix`
- Nearby functional form analysis: which alternative formulas fit equally well or better

---

## Evidence Timeline

Both files were created in the repository's **root commit** (SHA `61fc581`), dated 2026-02-16:

```
commit 61fc581
"Add response to editorial letter"
Co-authored-by: Cursor <cursoragent@cursor.com>
```

There is no commit predating this one. Both formulas appear with their current content from the very first entry in the repository. **No earlier derivation, iterative refinement, or intermediate state exists in version control.** They were inserted whole.

The commit message is significant: "response to editorial letter" is the stage at which a paper is revised after peer review rejection or major revision requests. Formulas introduced at this stage are at high risk of post-hoc fitting — they address referees who asked "can you derive X?" by inserting a formula that matches X.

---

## n_s = 1 − 2/N, N = p^(1/3) × (1 + 1/d)

### What the code says

```python
@property
def n_efolds(self) -> float:
    """Number of e-folds: N = p^{1/3} × (1 + 1/d)."""
    return self.p ** (1.0 / 3.0) * (1.0 + 1.0 / self.d)

@property
def spectral_index(self) -> float:
    """Scalar spectral index n_s = 1 - 2/N (Starobinsky)."""
    return 1.0 - 2.0 / self.n_efolds
```

### Breakdown into two components

**Component A: `n_s = 1 − 2/N` and `r = 12/N²`**

These are the standard Starobinsky R² inflation predictions, well-known since Starobinsky (1980). They are NOT BPR-specific — they follow from the Starobinsky potential `V ∝ (1 − e^(−√(2/3)φ/M_Pl))²` which applies to any inflaton with this potential shape. BPR adopts the formulas wholesale; no re-derivation is provided.

**Component B: `N = p^(1/3) × (1 + 1/d)`**

This is the BPR-specific claim. The sole justification appears in the textbook (line 791):

> "Cosmic inflation in BPR is not driven by an ad-hoc inflaton field but by the information pressure of the primordial boundary field undergoing a Class D boundary phase transition: N_efolds = p^{1/3} × (1 + 1/d) ≈ 63 (required: 50-60)"

No calculation is shown. The formula is asserted.

### What the derivation would need to establish

To be first-principles, BPR would need to show: starting from the dynamics of the boundary phase field on Z_p, solve for when the slow-roll conditions `ε < 1` and `|η| < 1` are satisfied, and show that the integral `N = ∫ dφ / √(2ε)` evaluates to `p^(1/3) × (1 + 1/d)`. This calculation does not exist in the repository.

### Internal inconsistency

The textbook says the required range is "50-60" e-folds, but the formula gives N ≈ 63. The code predicts n_s = 0.968, while Planck measures 0.9649 ± 0.0042 — a 0.78σ discrepancy. N ≈ 63 is acceptable cosmologically, but it is not "required" by the theory — the textbook inadvertently acknowledges this by noting the "required" range does not include the predicted value.

### Alternative functional forms

| N formula | N value | n_s | σ from Planck |
|-----------|---------|-----|--------------|
| `p^{1/3} × 4/3` (BPR) | 62.85 | 0.96818 | 0.78σ |
| `p^{0.35}` | 57.15 | 0.96500 | **0.02σ** |
| `p^{0.3249}` (optimal) | 53.09 | 0.96234 | 0.63σ |
| `ln(p)² / 2` | 66.81 | 0.97006 | 1.23σ |

The formula `N = p^{0.35}` (exponent 0.35 instead of 1/3, no 4/3 factor) gives n_s within 0.02σ — far better than BPR's formula. The exponent 1/3 is not uniquely selected by the data.

### Test quality

The only test is:
```python
assert 0.95 < infl.spectral_index < 0.98
```

This range (0.95–0.98) accommodates every Starobinsky-like model with N between 40 and 100. It does not test the specific formula for N.

### Verdict on n_s

**ANSATZ.** The Starobinsky formulas (`n_s = 1 − 2/N`, `r = 12/N²`) are legitimately borrowed. The BPR-specific claim — that `N = p^(1/3) × (1 + 1/d)` — is asserted without derivation. The formula fits within 1σ, but so do many alternatives. The formula was inserted as a complete claim in the root commit with no derivation document or calculation.

---

## δ_CP = π/2 − 1/√(z+1)

### What the code says

```python
# δ_CP: DERIVED — δ = π/2 − 1/√(z+1) from boundary geometry
delta = (
    np.pi / 2.0 - 1.0 / np.sqrt(self.z + 1.0)
    if self.p is not None
    else 1.196  # fallback
)
```

### Smoking gun: the fallback value

When `p is None`, the code falls back to `1.196` — the exact PDG experimental value for δ_CP. This is the strongest available evidence of post-hoc construction. A formula genuinely derived from first principles would either:
- Compute the answer from z regardless of whether p is available (δ_CP doesn't use p anyway), or
- Raise an error

Instead it defaults to the measurement. This means at some point during development, the code used `1.196` directly as the answer, and the formula `π/2 − 1/√(z+1)` was subsequently added to reproduce it.

### The "derivation" in documentation

The DERIVATION_ROADMAP.md states:

> `δ_CP = π/2 − 1/√(z+1)`: CP phase from boundary coordination: the 1/√(z+1) term **encodes the boundary mode overlap phase**.

The word "encodes" is not a derivation. No overlap integral is computed. The docstring in the code says: "The 1/√(z+1) correction encodes the boundary mode overlap phase." This restates the formula in words without showing any calculation.

### The +1 offset is the sole discriminant

The entire "uniqueness" of z=6 from this formula rests on the `+1` in the denominator:

| Formula | Selects z= | Error at that z |
|---------|-----------|----------------|
| `π/2 − 1/√(z−1)` | z=5 | 0.07σ |
| `π/2 − 1/√(z)`   | z=7 | 0.07σ |
| **`π/2 − 1/√(z+1)` [BPR]** | **z=6** | **0.07σ** |
| `π/2 − 1/√(z+2)` | z=5 | 0.07σ |

Every form `π/2 − 1/√(z+k)` for integer k selects a different z as uniquely best. All of them give 0.07σ at their preferred z. The assertion that z=6 is physically preferred requires independently deriving the `+1` offset. No such derivation exists.

### Relationship to coordinate geometry

One might attempt to argue: `z+1 = 7` represents the "6 neighbors plus the central site" in a 6-connected lattice. But:
1. This argument is nowhere stated in the code or documentation
2. The S² boundary has z = 6 nearest neighbors, but the full first coordination shell has 6+1 = 7 points only in specific lattice conventions
3. If this were the derivation, it would need to show why the CP phase integral produces exactly `1/√(z+1)` from first principles

### Listed as "recent derivation upgrade"

The DERIVATION_ROADMAP.md lists δ_CP as a recent upgrade (v0.9.2 → v0.9.3):
> `δ_CP`: π/2 − 1/√(z+1): 0.25% off

The phrase "recent derivation upgrade" implies it **replaced** an earlier formula or status. Since no prior git history exists before the root commit, we cannot confirm what the prior formula was. But the fact that it is called an "upgrade" strongly suggests the formula was constructed to improve an earlier, poorer fit.

### Test quality

No test in the test suite checks the value of δ_CP. The `TestCKMMatrix` class tests:
- Unitarity
- Cabibbo angle within (10°, 16°) — a 6° window
- Jarlskog invariant is nonzero
- Wolfenstein λ within (0.20, 0.25)

None of these test the CP phase.

### Verdict on δ_CP

**ANSATZ.** The formula `π/2 − 1/√(z+1)` is a phenomenological choice made to match the experimental δ_CP = 1.196 rad at z = 6. The fallback value `1.196` in the code is a direct fingerprint of reverse engineering. The documentation describes but does not derive the formula. The `+1` offset, which is the only structural feature that selects z=6 over z=7 or z=5, has no derivation.

---

## Does the Surviving Support for z = 6 Remain Meaningful?

From the independence audit, δ_CP was the strongest evidence that z=6 is genuinely preferred. After this derivation audit, that evidence collapses.

The argument was:
> "δ_CP = π/2 − 1/√(z+1) genuinely and independently prefers z=6 (0.07σ), and it uses a functional form (1/√(z+1)) unrelated to z/2 in 1/α."

This is technically correct as stated, but misses the key question: *why is the formula `π/2 − 1/√(z+1)` rather than `π/2 − 1/√(z)` or `π/2 − 1/√(z+2)`?* The formula's functional form is independent of z/2, but the specific `+1` offset is selected precisely because it produces the correct result at z=6. Without an independent derivation of that offset, the formula is a disguised fit.

**The support for z=6 from δ_CP is not meaningful** because the formula was constructed to fit the data at z=6, not derived from a calculation that independently predicted z=6.

The support for z=6 from 1/α remains, but it is not independent: z=6 is needed to make `z/2 = 3` contribute the right amount to `1/α = [ln(p)]² + z/2 + γ − 1/(2π)`. This makes z=6 the best integer given the chosen p, but it does not constitute independent evidence.

---

## Summary Table

| Property | n_s | δ_CP |
|---------|-----|------|
| Formula introduced | Root commit "editorial response" | Root commit "editorial response" |
| Independent derivation exists? | **No** | **No** |
| Description in docs | "N fixed by boundary transition" | "overlap phase between boundary modes" |
| Level of documentation | One-sentence assertion | One-sentence description |
| Borrowed from established theory? | Yes (Starobinsky `n_s = 1-2/N`) | No |
| BPR-specific part derived? | **No** (N = p^{1/3}×4/3 asserted) | **No** |
| Fallback reveals target value? | No | **Yes — fallback is 1.196 (PDG exact)** |
| Alternative equally-good form exists? | Yes (`p^{0.35}` gives 0.02σ) | Yes (`π/2-1/√z` gives 0.07σ at z=7) |
| Specific discriminating feature derived? | No (4/3 factor unjustified) | **No (+1 offset unjustified)** |
| Tests verify specific value? | No (loose range 0.95-0.98) | No (not tested at all) |
| Verdict | **ANSATZ** | **ANSATZ** |

---

## What Exact Derivation Would Rescue the Uniqueness Claim?

### For δ_CP

The minimal rescue would require:

1. Starting from the quark-sector boundary overlap integral `V_{ij} = ∫_{∂M} ψ*_{up,i}(x) ψ_{down,j}(x) dS` on the S² boundary with z=6 coordination
2. Computing the phase of this integral explicitly from the geometry of the coordination lattice
3. Showing that this phase evaluates to `π/2 − 1/√(z+1)` — specifically explaining why the denominator is `√(z+1)` and not `√(z)`, `z`, or any other function
4. The derivation must contain no parameters tuned to match the output

Without step 3, the formula remains a convenient fit. The entire claim rests on an unexamined `+1`.

### For n_s

The minimal rescue would require:

1. Writing down the equation of motion for the boundary phase field φ during the Class D transition
2. Computing the slow-roll parameters ε and η from this equation of motion in terms of p and d
3. Integrating to show `N = ∫dφ/√(2ε) = p^(1/3) × (1 + 1/d)` — specifically justifying why the factor is exactly `(1 + 1/d) = 4/3` for d=3 and not, say, `d/(d-1)` or `ln(d+1)`
4. Connecting this to the Starobinsky potential explicitly (why does the boundary phase potential take exactly the Starobinsky form?)

Without steps 2-3, the formula is a dressed-up version of "we chose N = 63 because it gives the right n_s."

### What would close the audit skeptically

For both formulas, the following single document would suffice: a calculation from the boundary action that produces the formula with no adjustable parameters, published before the experimental value of n_s or δ_CP was known (or at minimum, derived independently before fitting). Without a prior prediction, even a correct derivation cannot distinguish "derived" from "reverse-engineered."

---

## Files Produced

```
analysis/memo_derivation_audit.md    ← this document
```

(Uses results from `analysis/independence_audit.py` and direct code/git inspection.)
