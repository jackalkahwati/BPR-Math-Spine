# BPR: Limitations and Falsification

> **Purpose:** Position BPR as a testable framework with known limitations.
> **For:** Physicists evaluating whether to engage with the theory.
> **Updated:** April 2026 (honesty audit applied to GUT scale, baryon asymmetry, v_EW).

---

## 1. Known Limitations

### OPEN (Not Yet Derived)

| Item | Status | Notes |
|------|--------|-------|
| Planck length l_P | OPEN | Used as input (CODATA); not derived from substrate. G is an input. The `emergent_spacetime.py` formula is circular (inverts l_P). |
| Electroweak hierarchy M_Pl/v | OPEN | Old formula sqrt(pN) removed (12 orders off). `hierarchy_derived = False` in code. BPR argues naturalness (boundary UV cutoff removes fine-tuning) but does not derive the ratio. |
| GUT threshold corrections | TENSION | Forward calculation from E8 content closes ~34% of the coupling gap. Remaining ~66% is open. See detailed note below. |

### GUT Scale: Detailed Note (April 2026)

**Previous claim:** BPR's M_GUT ~ 6.8 x 10^17 GeV is "30x off from standard ~2 x 10^16 GeV."

**Correction:** The 2 x 10^16 GeV figure is the **MSSM** (supersymmetric) GUT scale, not a measured quantity. In the Standard Model without SUSY, gauge couplings do not unify at all -- there is no SM GUT scale to compare against. BPR provides a non-SUSY unification mechanism via boundary mode threshold corrections, yielding a different scale. Comparing a non-SUSY framework to a SUSY prediction is apples-to-oranges.

**Honest status of threshold corrections:**

| Method | Description | Status |
|--------|-------------|--------|
| Backward-fit | Compute what corrections are *needed* for exact unification, then assert the boundary provides them | Used for downstream predictions (Weinberg angle, alpha_s). Achieves exact unification by construction. |
| Forward-derived | Compute corrections from E8 -> SM representation content with boundary mass-splitting mechanism | Closes ~34% of the alpha_1 gap. Residual ~66% is documented tension. Complete SU(5) multiplets can't split couplings; only mass splitting within multiplets contributes. |

The Weinberg angle prediction (sin^2 theta_W = 0.23122) and alpha_s prediction depend on the backward-fit achieving exact unification. These should be read as conditional: "IF unification occurs at BPR's M_GUT, THEN sin^2 theta_W = 0.23122."

Code: `bpr/gauge_unification.py`, `forward_threshold_corrections` vs `boundary_threshold_corrections`.

### v_EW Self-Consistency (April 2026)

**Previous claim:** v_EW = Lambda_QCD x p^(1/3) x (ln p + z - 2) = 243.5 GeV (1% off).

**Correction:** This formula uses Lambda_QCD = 0.332 GeV (MS-bar, from PDG) as an input. The self-consistent chain (BPR alpha_s -> Lambda_QCD -> v_EW) produces a different Lambda_QCD and a worse v_EW prediction.

**Honest status:** The formula is a **ratio prediction**: it predicts v_EW / Lambda_QCD = p^(1/3) x (ln p + z - 2) ~ 741, which matches the observed ratio 246/0.332 = 741 to 1%. But Lambda_QCD itself is an input, not derived.

Code: `bpr/gauge_unification.py`, `electroweak_scale_self_consistent()`.

### Baryon Asymmetry (April 2026)

**Previous claim:** eta_B = 6.83 x 10^-10 (11.6% off).

**Correction:** Two formulas existed in the codebase that disagreed. Cleaned up April 2026:

| Formula | Result | Error | Status |
|---------|--------|-------|--------|
| Additive: kappa_SM x (1 + W_c/W_EW) x J | ~1.1 x 10^-9 | ~74% too high | Primary (physically clear) |
| Exponential: kappa_SM x exp(W_c x 4pi alpha_W) x J | ~6.2 x 10^-10 | ~0.2% off | Secondary (suspiciously close; WKB not rigorous) |

Both formulas use kappa_sph_SM = 10^-5 as an SM input (not derived from BPR). The Jarlskog invariant J is BPR-derived. The honest characterization: BPR provides the CP phase; the sphaleron efficiency is borrowed from SM.

Code: `bpr/cosmology.py`, `Baryogenesis` class.

### FRAMEWORK (Formula from BPR, Some Inputs from Experiment)

| Item | Status | Notes |
|------|--------|-------|
| Tc(Nb), Tc(MgB2) | FRAMEWORK | BCS formula; N(0)V from experiment, not derived |
| n_s, r (inflation) | FRAMEWORK | Starobinsky potential assumed; N_efolds from p |
| MOND a0 | FRAMEWORK | ~1.7% off; boundary formula uses R_Hubble |

### CONSISTENT (BPR Matches, but SM/GR Also Predict)

| Item | Status | Notes |
|------|--------|-------|
| v_GW = c | CONSISTENT | GR prediction; BPR consistent |
| Tsirelson 2sqrt(2) | CONSISTENT | QM prediction; BPR derives from boundary |
| Proton mass | CONSISTENT | QCD; BPR confinement formula matches |
| Proton lifetime bound | CONSISTENT | SM/GUT; BPR satisfies |

### CONJECTURAL (Not Yet Testable)

| Item | Status | Notes |
|------|--------|-------|
| Oscillatory memory decay | CONJECTURAL | P1.1; no direct test |
| Casimir fine-structure wiggles | CONJECTURAL | P1.5; below current precision |
| W_crit(C60) | CONJECTURAL | Implies C60 quantum; not yet measured |

### Input Anchors

BPR uses these experimental inputs:

| Input | Where used | Why needed |
|-------|-----------|------------|
| m_tau | Charged lepton mass anchor | Sets absolute scale; ratios are derived |
| m_b | Down-type quark anchor (when v_EW not derived) | Same role as m_tau |
| Lambda_QCD = 0.332 GeV | v_EW formula | Self-consistent derivation has tension |
| l_P (CODATA) | All Planck-scale quantities | G not derived from substrate |
| kappa_sph = 10^-5 | Baryon asymmetry | SM sphaleron rate, not BPR-derived |

With v_EW from boundary formula, m_t and m_b are derived. m_tau remains as anchor.

---

## 2. What BPR Actually Derives vs What It Borrows

This is the most important section for honest evaluation.

**Genuinely derived from (p, z) with zero experimental input:**
- All PMNS mixing angles (theta_12, theta_13, theta_23, delta_CP)
- All CKM mixing angles and CP phase
- Quark mass ratios (m_s/m_d, m_u/m_c, m_c/m_t)
- Number of generations (n_gen = 3)
- Normal mass ordering
- Dirac neutrinos (no 0nubetabeta)
- theta_QCD = 0 (no axion)
- xi_1 = 0 (no linear LIV)
- Proton charge radius

**Derived from (p, z) + one anchor mass:**
- Absolute quark masses (anchor: m_b or m_t from v_EW)
- Charged lepton masses (anchor: m_tau)

**Derived from (p, z) + SM inputs:**
- v_EW (needs Lambda_QCD)
- Higgs mass (needs v_EW)
- Baryon asymmetry (needs kappa_sph from SM)

**Conditional on backward-fit unification:**
- sin^2 theta_W = 0.23122
- 1/alpha_EM at M_Z

**Not derived (CODATA input):**
- Planck length, G
- Electroweak hierarchy ratio

---

## 3. Top 5 Falsifiable Predictions

| # | Prediction | Experiment | Timeline | FALSIFICATION |
|---|------------|------------|----------|---------------|
| 1 | **Normal neutrino ordering** | JUNO | 2027 | Inverted ordering -> BPR boundary topology wrong |
| 2 | **No 0nubetabeta (Dirac neutrinos)** | LEGEND, nEXO | 2025-2030 | Observation -> BPR p = 1 mod 4 orientability wrong |
| 3 | **Casimir deviation delta ~ 1.37** | Delft/STM, phonon-MEMS | 1-3 yr | Null at 10^-9 -> BPR phonon channel ruled out |
| 4 | **LIV |dc/c| ~ 3.4 x 10^-21** | CTA, GRBs | 2026+ | Null below 10^-21 -> BPR substrate discreteness wrong |
| 5 | **Born rule kappa ~ 10^-5** | Many-photon Sorkin | 2-5 yr | Born rule holds to 10^-7 -> BPR microstate counting wrong |

**Code references:** See [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md).

---

## 4. BPR-Unique vs Shared Predictions

| Prediction | BPR | SM/GR | Unique? |
|------------|-----|-------|---------|
| m_s/m_d = 20.0 | Derived | No explanation | **BPR-unique** |
| Neutrino normal ordering | Derived | Both orderings allowed | **BPR-unique** |
| Dirac (no 0nubetabeta) | Derived | Majorana allowed | **BPR-unique** |
| Casimir exponent delta = 1.37 | Derived | Standard has no correction | **BPR-unique** |
| Born rule kappa ~ 10^-5 | Derived | Exact in QM | **BPR-unique** |
| LIV |dc/c| ~ 10^-21 | Derived | Zero in GR | **BPR-unique** |
| v_GW = c | Consistent | Predicted | Shared |
| Tsirelson 2sqrt(2) | Consistent | Predicted | Shared |
| Proton tau > 10^34 yr | Consistent | Predicted | Shared |
| Higgs m_H ~ 125 GeV | Derived (with v_EW input) | From measurement | Partially shared |

---

## 5. How to Read the CONFIRM Verdicts

The experiments document lists CONFIRM verdicts. Interpretation:

- **~25-30** are BPR-unique comparisons (quark mass ratios, neutrino angles, CKM, etc.).
- **~30** are shared with SM/GR (v_GW, Tsirelson, proton bound) -- consistency checks.
- The remainder are split/individual tests of the same underlying prediction.

A conservative count of *distinct* BPR-unique confirmations: **~25-30**. The remainder establish that BPR is not in conflict with known physics.

---

## 6. Summary for Reviewers

**BPR is a testable framework** with:

1. **Explicit falsification criteria** -- 5 near-term tests.
2. **Known limitations** -- documented with April 2026 honesty audit.
3. **Clear provenance** -- every prediction labeled DERIVED / FRAMEWORK / CONSISTENT / CONJECTURAL.
4. **Input audit** -- Section 2 above lists every experimental input and why it's needed.
5. **All code public** -- 1,223 test functions; reproducible benchmarks.

**What is genuinely impressive:** Dimensionless ratios and mixing angles derived from two integers.

**What is not yet established:** Connection to dimensionful physics (G, l_P, hierarchy).

**Recommended first step:** Run `pytest -q` and `python scripts/benchmark_predictions.py` to verify reproducibility. Read `forward_threshold_corrections` vs `boundary_threshold_corrections` in `gauge_unification.py` to understand the unification status.

---

*Last updated April 2026. See [doc/experiments/](experiments/) for papers.*
