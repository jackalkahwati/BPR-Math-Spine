# Doubly-Charmed Baryons — BPR Calibration

> **Status:** SM-embedding confirmation; NOT BPR-discriminating against
> lattice QCD.
> **Scope:** Records BPR's predictions for the doubly-charmed baryon
> family (Ξcc⁺⁺, Ξcc⁺, Ωcc⁺) against LHCb measurements and lattice QCD,
> with explicit accounting of what BPR contributes versus what is
> inherited from the Standard Model.
> **Companion modules:** [`bpr/qcd_flavor.py`](../../bpr/qcd_flavor.py)
> §12.2 (QuarkMassSpectrum), §12.7 (doubly_charmed_isospin_splitting).

---

## 1. Headline finding

For the SU(3)-flavor mass splitting Ωcc⁺ − Ξcc⁺, BPR predicts (using
substrate-derived m_s and m_d, central HQET κ = 1.18 from lattice
studies):

```
Δm_BPR   = κ · (m_s − m_d) = 1.18 · 89.15 MeV = 105.20 MeV
Δm_LHCb  = 3727 − 3622 = 105.45 ± 5.02 MeV
Residual = −0.25 MeV  (−0.05σ)
```

**Status: NON-DISCRIMINATING.** The prediction is within both
experimental uncertainty and the lattice QCD prediction range
(85–130 MeV). It is consistent with both BPR and lattice QCD. The
LHCb discovery confirms the Standard Model embedding to ~0.05σ at
the central κ value but does not select between BPR and lattice QCD.

---

## 2. The discoveries

| Particle | Quark content | Mass (MeV/c²) | Source |
|----------|--------------|---------------|--------|
| Ξcc⁺⁺ | ccu | 3621.55 ± 0.40 | LHCb 2017 ([arXiv:1707.01621](https://arxiv.org/abs/1707.01621)) |
| Ξcc⁺ | ccd | ~3622 (isospin partner) | LHCb 2026 ([outreach](https://lhcb-outreach.web.cern.ch/2026/03/17/observation-of-the-doubly-charmed-heavy-proton-%CE%BEcc/)) |
| Ωcc⁺ | ccs | ~3727 ± 5 | LHCb 2026 Beauty conference ([outreach](https://lhcb-outreach.web.cern.ch/2026/06/03/observation-of-the-doubly-charmed-baryon-%CF%89cc/)) |

The Ωcc⁺ completes the SU(3)-flavor triplet of doubly-charmed spin-½
baryons predicted by Gell-Mann's quark model (1964 extension of the
Eightfold Way to include charm) and refined since charm was discovered
in 1974.

---

## 3. What BPR contributes vs what is inherited

### DERIVED from BPR substrate (J, p, z, l-modes)

`bpr/qcd_flavor.py:QuarkMassSpectrum(v_EW_GeV=246.0, p=104761)` yields
current quark masses (MS-bar at 2 GeV) from boundary mode spectra:

```
m_u =     2.172 MeV  (PDG 2.16, 0.6% off)
m_d =     4.733 MeV  (PDG 4.67, 1.4% off)
m_s =    93.883 MeV  (PDG 93.4, 0.5% off)
m_c =  1251.0   MeV  (PDG 1270, 1.5% off)
m_b =  4194.8   MeV  (PDG 4180, 0.4% off)
m_t = 173948    MeV  (PDG 172760, 0.7% off)
```

Specifically for the SU(3) splitting:
- m_s − m_d = 89.15 MeV (PDG ~88.7)
- m_s/m_d = 19.84 (PDG 19-22)

These are derived from (z=6, n_gen=3, W_c=√3, p=104761, v_EW=246 GeV)
with no per-quark fitting — the down-type spectrum is anchored to m_b
via boundary coordination, and m_b itself is derived from m_t via
Higgs-doublet isospin structure plus boundary phase-space suppression
(see `QuarkMassSpectrum._m_b_MeV`).

### INHERITED from Standard Model / QCD (NOT BPR-derived)

```
κ_HQET = 1.18 (central value from lattice QCD)
```

The heavy-quark effective theory coefficient κ that maps light-quark
current mass to baryon mass contribution. LO HQET gives κ = 1; with
1/m_Q corrections and constituent dressing, lattice studies (Brown
et al. 2014 [arXiv:1409.0497](https://arxiv.org/abs/1409.0497); Mathur
et al. 2018 [arXiv:1807.00174](https://arxiv.org/abs/1807.00174)) favor
κ ∈ [1.0, 1.30] with central values around 1.15–1.20.

κ is QCD dynamical physics. BPR's current implementation does not
derive it from substrate. This is the gap between "BPR predicts the
splitting" and "BPR predicts the splitting in a way that discriminates
against lattice QCD."

---

## 4. Sensitivity to κ

| κ      | Δm_BPR [MeV] | Residual vs LHCb [MeV] | Status |
|--------|--------------|-------------------------|--------|
| 1.00 (LO HQET) | 89.15 | −16.30 | Within lattice range, outside measurement |
| 1.10 |  98.06 |  −7.39 | Non-discriminating |
| 1.15 | 102.52 |  −2.93 | Non-discriminating |
| 1.18 | 105.20 |  −0.25 | Non-discriminating (lattice central value) |
| 1.20 | 106.98 |  +1.53 | Non-discriminating |
| 1.25 | 111.44 |  +5.99 | Non-discriminating |
| 1.30 | 115.89 | +10.44 | Within lattice range, outside measurement |

The LHCb measurement is consistent with κ ∈ [1.10, 1.25] at the BPR-
derived m_s − m_d. The κ value that best fits LHCb (≈1.18) is the
same as the lattice QCD central value — not surprising, since both
frameworks use the same HQET machinery.

---

## 5. Evidence pipeline classification

**Status:** SM-EMBEDDING-CONFIRMED, NON-DISCRIMINATING.

**What this discovery confirms:**
- BPR's quark mass derivation reproduces SM/PDG values to ~1% (already
  in the validation status)
- The SU(3)-flavor multiplet structure inherited from those quark
  masses correctly predicts the existence and approximate mass of the
  Ωcc⁺ via Gell-Mann's quark-model machinery
- The framework's "fewer fitted constants" claim cashes out for this
  particular SU(3) splitting: BPR's m_s and m_d are derived, not fitted

**What this discovery does NOT establish:**
- BPR-unique prediction power over lattice QCD on this observable
- A discriminating test of BPR against the Standard Model
- Validation of any BPR claim beyond the SM embedding it already had

**What this discovery does NOT refute:**
- The framework's prediction is consistent with the LHCb measurement
- No tension is opened

The right entry for this discovery in `VALIDATION_STATUS.md` or the
benchmark scorecard is "CONFIRMED inherited prediction" with the
explicit note that the κ coefficient was taken from lattice QCD, not
derived from substrate.

---

## 6. First-pass derivation of κ from BPR substrate

**Status:** PRINCIPLED-BUT-NOT-RIGOROUS. Implemented in
`bpr/qcd_flavor.py:derive_kappa_HQET_from_substrate()`. None of the
four candidate ansätze below is uniquely selected by BPR's existing
boundary structure; a rigorous derivation requires building doubly-
heavy baryon spectroscopy from the boundary action (real research
project, not a one-function task).

### Four candidate ansätze using BPR's existing structure

| Ansatz | Formula | κ value |
|--------|---------|---------|
| LO HQET | 1 | 1.0000 |
| Mode-ratio | 1 + (l_s − l_d) / l_c | 1.1250 |
| Mass-ratio with color factor | 1 + N_c · (m_s + m_d) / (2 m_c) | 1.1182 |
| CKM-analog | 1 + (m_s + m_d) / (m_c + m_b) · √(ln(p) + z/3) | 1.0667 |

All four use BPR substrate-derived inputs (l-modes, quark masses,
N_c = z/2, p = 104761, z = 6); the *choice* of ansatz is what's
heuristic. Each is motivated by an existing BPR derivation pattern
(the V_cb formula's mass-ratio × boundary-coordination structure,
the m_b derivation's boundary phase-space correction, etc.).

### Splitting predictions and residuals against LHCb

| Ansatz | Δm_BPR [MeV] | Residual [MeV] | Residual [σ] |
|--------|--------------|-----------------|---------------|
| LO HQET | 89.15 | −16.30 | **−3.25** (tension) |
| Mode-ratio | 100.29 | −5.16 | −1.03 |
| Mass-ratio | 99.69 | −5.76 | −1.15 |
| CKM-analog | 95.09 | −10.36 | −2.06 |

**LHCb measured: 105.45 ± 5.02 MeV** (Ωcc⁺ at 3727 − Ξcc⁺⁺ at 3621.55)

### What this attempt actually shows

1. **BPR's natural LO HQET gives κ = 1, which is in 3.25σ tension with
   the LHCb measurement.** This is itself informative: a strict LO-HQET
   reading of the framework is *discriminated against* by the discovery.
   The framework requires some correction to κ to be consistent with
   experiment.
2. **Two structurally-distinct BPR-internal corrections (mode-ratio and
   mass-ratio) yield nearly identical κ ≈ 1.12.** The agreement between
   the two derivations is interesting — they encode different physical
   intuitions (boundary-mode geometry vs. mass-ratio 1/m_Q expansion)
   but converge on the same value to within 0.5%. This may indicate
   the framework genuinely points toward κ ≈ 1.12 from substrate, even
   though the rigorous derivation is not yet done.
3. **BPR's principled-derivation range κ ∈ [1.07, 1.13] is slightly
   *lower* than the lattice QCD central value κ = 1.18.** The
   corresponding Δm prediction is ~99-100 MeV vs lattice ~105 MeV vs
   LHCb 105.45 ± 5.02 MeV. All three are mutually consistent at the
   current experimental precision (~5 MeV).
4. **If LHCb improves precision to ~1 MeV on this splitting, the
   measurement could distinguish between BPR's substrate-derived κ
   (~1.12) and the lattice QCD value (1.18).** That's the testable
   prediction that emerges from this attempt: **BPR predicts the Ωcc⁺
   − Ξcc⁺ splitting is ~100 MeV, not ~105 MeV.** Currently the
   measurement is consistent with both; future precision could
   discriminate.

### Verdict on the derivation attempt

This is a first-pass exercise that:

- **Refutes the strict LO-HQET reading** of BPR (3.25σ tension)
- **Yields a substrate-derived κ ≈ 1.12** under two independent
  principled ansätze with structurally different motivations
- **Predicts Δm(Ωcc⁺ − Ξcc⁺) ≈ 99 − 100 MeV** as the BPR-derived
  value (vs lattice 105 MeV, LHCb 105.45 ± 5.02 MeV)
- **Identifies a testable discriminating prediction** for future
  higher-precision LHCb measurements

The attempt is *not* a rigorous derivation. The mode-ratio and
mass-ratio ansätze were chosen by analogy with existing BPR
patterns; the framework's boundary action does not currently
uniquely select among them. A rigorous version requires extending
the framework's hadron-mass machinery into doubly-heavy baryon
spectroscopy.

But the consistency of the two non-trivial ansätze at κ ≈ 1.12 is
suggestive enough that the BPR-discriminating prediction "Δm ≈ 99-100
MeV" is worth recording as a target for future LHCb measurements.

### Related research targets (for full rigor)

- Build substrate-level doubly-heavy baryon spectroscopy from the
  boundary action: derive M(QQq) = 2M_Q + λ_QQ + κ·m_q with all
  three constants from substrate, then compare with the family of
  LHCb measurements
- Magnetic moments of Ωcc⁺ and Ξcc⁺ from BPR's boundary spin-flavor
  structure (lattice predictions exist; future experiments may
  measure)
- Hyperfine splitting Ξcc⁺(½) − Ξcc⁺(3/2) (spin-3/2 family is the
  next LHCb target per the discovery announcement)
- Decay rates Ωcc⁺ → Ξc⁺K⁻π⁺ from BPR's CKM structure (lifetime not
  yet measured)

---

## 7. References

- Gell-Mann, M. (1964). "A schematic model of baryons and mesons,"
  *Physics Letters* 8, 214–215. (Quark model with SU(3) flavor;
  predicts Ωcc as natural extension once charm added.)
- Gell-Mann & Ne'eman (1961). "The Eightfold Way." (Original SU(3)
  classification.)
- LHCb 2017. "Observation of the doubly-charmed baryon Ξcc⁺⁺,"
  *Phys. Rev. Lett.* 119, 112001 ([arXiv:1707.01621](https://arxiv.org/abs/1707.01621)).
- LHCb 2026 (March). "Observation of the doubly-charmed heavy proton
  Ξcc⁺" ([outreach](https://lhcb-outreach.web.cern.ch/2026/03/17/observation-of-the-doubly-charmed-heavy-proton-%CE%BEcc/)).
- LHCb 2026 (June, Beauty conference). "Observation of the doubly-
  charmed baryon Ωcc⁺" ([outreach](https://lhcb-outreach.web.cern.ch/2026/06/03/observation-of-the-doubly-charmed-baryon-%CF%89cc/)).
- Brown, Z. S. et al. (2014). "Charmed baryon spectroscopy from
  lattice QCD," ([arXiv:1409.0497](https://arxiv.org/abs/1409.0497)).
- Mathur, N. et al. (2018). "Lattice QCD study of doubly-charmed
  strange baryons," ([arXiv:1807.00174](https://arxiv.org/abs/1807.00174)).
- `bpr/qcd_flavor.py` — QuarkMassSpectrum, doubly_charmed_isospin_splitting,
  doubly_charmed_splitting_kappa_scan
- `doc/experiments/CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md` — parallel
  calibration of Eq (5) against psi literature
