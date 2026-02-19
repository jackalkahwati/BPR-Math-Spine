# BPR: Limitations and Falsification

> **Purpose:** Position BPR as a testable framework with known limitations.
> **For:** Physicists evaluating whether to engage with the theory.

---

## 1. Known Limitations

### OPEN (Not Yet Derived)

| Item | Status | Notes |
|------|--------|-------|
| Planck length ℓ_P | OPEN | Used as input (CODATA); not derived from substrate |
| Electroweak hierarchy M_Pl/v | OPEN | √(pN) ≈ 3×10⁴ vs observed 5×10¹⁶; flagged as unsolved |
| GUT scale | Tension | BPR ~6.8×10¹⁷ GeV vs standard ~2×10¹⁶ GeV (30× off) |

### FRAMEWORK (Formula from BPR, Some Inputs from Experiment)

| Item | Status | Notes |
|------|--------|-------|
| Tc(Nb), Tc(MgB₂) | FRAMEWORK | BCS formula; N(0)V from experiment, not derived |
| n_s, r (inflation) | FRAMEWORK | Starobinsky potential assumed; N_efolds from p |
| θ₁₂, θ₂₃ (PMNS) | FRAMEWORK | Mass-hierarchy correction; some fit to data |
| MOND a₀ | FRAMEWORK | ~13% off; boundary formula uses R_Hubble |

### CONSISTENT (BPR Matches, but SM/GR Also Predict)

| Item | Status | Notes |
|------|--------|-------|
| v_GW = c | CONSISTENT | GR prediction; BPR consistent |
| Tsirelson 2√2 | CONSISTENT | QM prediction; BPR derives from boundary |
| Proton mass | CONSISTENT | QCD; BPR confinement formula matches |
| Proton lifetime bound | CONSISTENT | SM/GUT; BPR satisfies |

### CONJECTURAL (Not Yet Testable)

| Item | Status | Notes |
|------|--------|-------|
| Oscillatory memory decay | CONJECTURAL | P1.1; no direct test |
| Casimir fine-structure wiggles | CONJECTURAL | P1.5; below current precision |
| W_crit(C60) | CONJECTURAL | Implies C60 quantum; not yet measured |

### Input Anchors

BPR uses **4 anchor masses** when v_EW is not derived: m_t, m_b, m_d, m_τ. With v_EW from boundary, m_t and m_b are derived; m_d and m_τ remain as normalization anchors in some sectors. See `VALIDATION_STATUS.md` for full audit.

---

## 2. Top 5 Falsifiable Predictions

| # | Prediction | Experiment | Timeline | FALSIFICATION |
|---|------------|------------|----------|---------------|
| 1 | **Normal neutrino ordering** | JUNO | 2027 | Inverted ordering → BPR boundary topology wrong |
| 2 | **No 0νββ (Dirac neutrinos)** | LEGEND, nEXO | 2025–2030 | Observation of 0νββ → BPR p≡1 mod 4 orientability wrong |
| 3 | **Casimir deviation δ ~ 1.37** | Delft/STM, phonon-MEMS | 1–3 yr | Null at 10⁻⁹ → BPR phonon channel ruled out |
| 4 | **LIV \|δc/c\| ~ 3.4×10⁻²¹** | CTA, GRBs | 2026+ | Null below 10⁻²¹ → BPR substrate discreteness wrong |
| 5 | **Born rule κ ~ 10⁻⁵** | Many-photon Sorkin | 2–5 yr | Born rule holds to 10⁻⁷ → BPR microstate counting wrong |

**Code references:** See [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md).

---

## 3. BPR-Unique vs Shared Predictions

| Prediction | BPR | SM/GR | Unique? |
|------------|-----|-------|---------|
| m_s/m_d = 20.0 | ✓ | No explanation | **BPR-unique** |
| Neutrino normal ordering | ✓ | Both orderings allowed | **BPR-unique** |
| Dirac (no 0νββ) | ✓ | Majorana allowed | **BPR-unique** |
| Casimir exponent δ = 1.37 | ✓ | Standard has no correction | **BPR-unique** |
| Born rule κ ~ 10⁻⁵ | ✓ | Exact in QM | **BPR-unique** |
| LIV \|δc/c\| ~ 10⁻²¹ | ✓ | Zero in GR | **BPR-unique** |
| v_GW = c | ✓ | ✓ | Shared |
| Tsirelson 2√2 | ✓ | ✓ | Shared |
| Proton τ > 10³⁴ yr | ✓ | ✓ | Shared |
| Higgs m_H ~ 125 GeV | ✓ | ✓ (from measurement) | Partially shared |

---

## 4. How to Read the 115 CONFIRM Verdicts

The experiments document lists 115 CONFIRM verdicts. Interpretation:

- **~40** are BPR-unique comparisons (quark masses, neutrino angles, CKM, η, etc.).
- **~30** are shared with SM/GR (v_GW, Tsirelson, proton bound, etc.) — consistency checks.
- **~45** are split/individual tests of the same underlying prediction (e.g., each quark mass as separate row).

A conservative count of *distinct* BPR-unique confirmations: **~25–30**. The remainder establish that BPR is not in conflict with known physics.

---

## 5. Summary for Reviewers

**BPR is a testable framework** with:

1. **Explicit falsification criteria** — 5 near-term tests (see above).
2. **Known limitations** — OPEN items documented; no claim to derive Planck length or hierarchy.
3. **Honest classification** — DERIVED vs FRAMEWORK vs CONSISTENT vs CONJECTURAL in `VALIDATION_STATUS.md`.
4. **All code public** — 488 tests; reproducible benchmarks.

**Recommended first step:** Run `pytest -q` and `python scripts/benchmark_predictions.py` to verify reproducibility.

---

*Last updated Feb 2026. See [doc/experiments/](experiments/) for papers.*
