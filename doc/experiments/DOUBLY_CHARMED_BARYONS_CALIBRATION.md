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

## 6. What would make this BPR-discriminating

The clean target is **deriving κ from boundary-mode dynamics**.

A BPR-discriminating extension of `bpr/qcd_flavor.py` would have to:
1. Build a substrate-level model of the cc-diquark binding in the
   light-quark field
2. Derive κ from the boundary phase-field coupling between the
   diquark and the light quark
3. Show that the derived κ either agrees with or differs from the
   lattice QCD value in a way that the LHCb measurement can test

If the derived κ matches lattice (~1.18), the framework demonstrates
deeper inheritance — substrate predicts the same QCD dynamics as
lattice. If it differs by enough that LHCb can distinguish (say
κ_BPR ≠ κ_lattice by > 5%), the framework makes a discriminating
prediction against lattice QCD.

Either outcome is informative. Currently neither is done.

Related research targets:
- Magnetic moments of Ωcc⁺ and Ξcc⁺ from BPR's boundary spin-flavor
  structure (lattice predictions exist; LHCb will not measure soon
  but future experiments might)
- Hyperfine splitting Ξcc⁺(½) − Ξcc⁺(3/2) (the spin-3/2 doubly-charmed
  baryon family is the next LHCb target per the video transcript)
- Decay rates Ωcc⁺ → Ξc⁺K⁻π⁺ (or whatever final state) from BPR's
  CKM structure (Ωcc⁺ lifetime not yet measured at LHCb)

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
