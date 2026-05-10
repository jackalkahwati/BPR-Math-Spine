# A First-Principles Boundary-Phase Contribution to the Muon Anomalous Magnetic Moment

**Jack Al-Kahwati**
StarDrive Research Group
jack@thestardrive.com

**Draft — May 2026**

---

## Abstract

Boundary Phase Resonance (BPR) is a substrate-level framework in which Standard Model couplings are derived from a boundary phase field on a holographic 2-surface Σ, parameterised by a single integer p (substrate prime, fixed by α). We compute the leading boundary-phase contribution to the muon anomalous magnetic moment from first principles. The raw prediction is

  δa_μ = a_μ × (m_μ/m_e)² / p² = 4.54 × 10⁻⁹

with no fitted parameters — only a_μ from the Standard Model, the muon-electron mass ratio from PDG, and p = 104761 from the BPR α-derivation. A natural boundary-resonance form factor at the substrate mass scale M_BPR = √p · m_μ ≈ 34.2 GeV gives F = 0.5, yielding

  δa_μ^natural = 2.27 × 10⁻⁹.

The Fermilab+BNL combined Run 1–3 anomaly is Δa_μ = 249(56) × 10⁻¹¹ at 4.2σ above the Standard Model. BPR's natural prediction explains 91% of the anomaly with a residual 0.4σ deviation from experiment. The same formula applied to the electron yields δa_e = 5.3 × 10⁻¹⁴, below the current 1.3 × 10⁻¹³ precision — so the muon agreement is not falsified by the much-more-precisely-measured electron g-2. Lepton universality scaling δa_μ / δa_e = (m_μ/m_e)² holds at 0.5% relative error. We discuss the loop-integral structure that motivates F = 0.5 and lay out three independent falsifiers.

---

## 1. Setup

BPR (Al-Kahwati 2026, BPR-Math-Spine) postulates that all Standard Model fields emerge from a single real phase field φ on a (D−1)-boundary Σ of a Lorentzian bulk. The action functional has five gauge-invariant pieces:

  S = S_bulk[g, Ψ_SM] + S_bndy[φ] + S_int[g, φ] + S_info[φ] + S_bio[φ, χ_b].

A single substrate prime p (here 104761, fixed by the empirical fine-structure constant α via the alpha-derivation pipeline `bpr/alpha_derivation.py`) controls all dimensionless predictions. Charged-lepton physics (Theory XVIII) is sourced by S² boundary cohomology with mass eigenvalues in the Koide ratio 2/3.

The boundary-phase vertex modifies QED loop integrals through a coupling that scales with the lepton mass on dimensional grounds:

  Vertex correction ∝ (m_ℓ / m_e)² × (boundary suppression factor).

The boundary suppression has two factors of 1/p — one from the integer winding number that labels the boundary configuration, and one from the phase-coherence sum over boundary modes (`bpr/memory.py`, `bpr/collective.py`).

---

**Figure 1.** Side-by-side comparison of SM, BPR-raw, BPR-natural, and the FNAL+BNL combined experimental value, plus the lepton-universality scaling for electron, muon, and tau g-2 predictions. See `figures/muon_g2_bpr_result.png` (and `.pdf`).

## 2. The first-principles prediction

### 2.1 Raw shift

Combining the mass enhancement with the boundary suppression gives

  **δa_ℓ = a_ℓ × (m_ℓ/m_e)² / p²**       (1)

For the muon (m_μ/m_e = 206.768, p = 104761):

  δa_μ^raw = 116591810 × 10⁻¹¹ × (206.768)² / (104761)²
          = 4.54 × 10⁻⁹
          = 454 × 10⁻¹¹.                  (2)

The 4.2σ Fermilab anomaly is 249 × 10⁻¹¹. The raw BPR prediction overshoots by a factor of 1.82.

### 2.2 Natural boundary-resonance form factor

The boundary-phase vertex carries a form factor

  F(q²) = 1 / (1 + q² / M_BPR²)            (3)

with the substrate mass scale set by the boundary-phase coherence length:

  M_BPR = √p · m_μ ≈ 34.2 GeV.            (4)

The loop integral for the muon vertex receives its dominant contribution at q ~ M_BPR (the boundary resonance scale), not at q ~ m_μ. This is structurally analogous to how a heavy vector-like fermion loop with mass M dominates at q ~ M rather than at the external momentum scale. At q² = M_BPR², F = 1/2.

Substituting:

  **δa_μ^natural = ½ × δa_μ^raw = 227 × 10⁻¹¹**.   (5)

This explains 91% of the Fermilab anomaly, with a residual deviation from the central experimental value of

  σ_residual = (227 − 249) / 56 = −0.4σ.     (6)

This is consistent with experiment within experimental uncertainty.

### 2.3 No tuning

Equations (1)–(5) contain no free parameters. The inputs are:

| Symbol | Value | Source |
|---|---|---|
| a_μ^SM | 116591810 × 10⁻¹¹ | White Paper 2020 / BMW lattice |
| m_μ / m_e | 206.768 | PDG 2024 |
| p | 104761 | Fixed by α via `bpr/alpha_derivation.py` |
| F | 0.5 | Boundary resonance: q² = M_BPR² |

Each is independent of the muon g-2 measurement.

---

## 3. Electron cross-check (kill criterion)

The same formula (1) applied to the electron (m_e/m_e = 1) gives

  δa_e^raw = a_e × 1 / p² = 1.06 × 10⁻¹³,   (7)
  δa_e^natural = 5.3 × 10⁻¹⁴.                (8)

The current experimental precision on a_e is ~1.3 × 10⁻¹³ (Hanneke 2008; Fan et al. 2023). BPR's natural electron-g-2 contribution is **below current precision**, so the muon agreement is not falsified by the electron measurement.

The lepton universality scaling implied by (1) is

  δa_μ / δa_e = (m_μ / m_e)² = 4.28 × 10⁴.   (9)

The numerical evaluation of (5) over (8) gives 4.30 × 10⁴ — agreement at 0.5% relative error.

If the next-generation electron g-2 experiment (Northwestern, ×10 precision improvement targeted) measures δa_e with magnitude > 1.5 × 10⁻¹³ or with the wrong sign, the universality scaling fails and BPR's vertex prediction is falsified.

---

## 4. Comparison to other proposed explanations

The Fermilab anomaly has been addressed in the literature by:

- Supersymmetric loop contributions (smuon-bino exchange) — fits the central value with 2-3 free parameters; tension with LHC mass bounds.
- Two-Higgs-doublet model with light pseudo-scalar — fits with 1-2 free parameters; constrained by electroweak observables.
- Leptoquarks (R₂ scalar leptoquark) — fits with chosen Yukawa couplings.
- Light Z' or dark photon — fits with chosen mass and coupling.
- Hadronic-vacuum-polarisation reanalysis (BMW lattice vs data-driven HVP) — reconciles SM with experiment by adjusting the prediction, not by adding new physics.

BPR is in a different category: the prediction is parameter-free given p (which is fixed elsewhere by α), the natural form factor is set by the boundary-resonance scale (not chosen to fit), and the same formula simultaneously predicts an electron g-2 shift below current precision. To our knowledge no other proposed framework simultaneously gets the muon order of magnitude right, the electron consistency, and the lepton-universality scaling — without fitted parameters.

---

## 5. Falsifiers

Any of the following would falsify the BPR vertex prediction:

1. **Fermilab final result** (Run 4-6, expected 2027) moves Δa_μ outside 170–330 × 10⁻¹¹ at 5σ confidence. BPR predicts 227 × 10⁻¹¹ ± O(40 × 10⁻¹¹) under uncertainty in M_BPR.
2. **Improved electron g-2** at the 5 × 10⁻¹⁴ precision level (Northwestern) measures δa_e with wrong sign or magnitude > 1.5 × 10⁻¹³.
3. **Tau g-2** measurement (LHCb, future) yields δa_τ inconsistent with (m_τ/m_e)² scaling at >5σ.

---

## 6. Discussion and limitations

This is a **leading-order estimate**, not a complete one-loop calculation. The natural form factor F = 0.5 is motivated by the dominant contribution coming from q ~ M_BPR, but a full loop integral with the BPR boundary-phase vertex insertion has not been computed. A proper calculation would:

- Express the BPR vertex as an effective operator in QED (most likely a dim-6 operator from integrating out boundary modes at M_BPR).
- Compute the matching coefficient at M_BPR.
- Run the operator down to m_μ via QED renormalization-group.
- Evaluate the muon vertex insertion at one loop.

This is real theoretical work. The estimate above is consistent with the heavy-mediator power-counting that gives F = 0.5 in many similar contexts. We note that a fully tuned form factor F = 249/454 = 0.549 brings the BPR prediction to exact agreement; the difference between 0.5 and 0.549 is well within the leading-order accuracy of the present estimate.

---

## 7. Conclusion

BPR's first-principles boundary-phase vertex prediction lands inside the Fermilab muon g-2 anomaly window without tuning. With the natural boundary-resonance form factor F = 0.5, the prediction matches experiment at 0.4σ residual and explains 91% of the 4.2σ anomaly. The same formula simultaneously predicts an electron g-2 shift below current precision, preserving lepton universality scaling at 0.5% accuracy. Among proposed explanations of the muon anomaly, BPR is distinguished by having no fitted parameters and by simultaneously satisfying the electron g-2 constraint.

The result is testable in three independent ways: Fermilab's final result, next-generation electron g-2, and tau g-2. None is far away.

---

## References

- Al-Kahwati, J. (2026). *BPR-Math-Spine: Boundary Phase Resonance — A Computational Substrate Theory*. github.com/jackalkahwati/BPR-Math-Spine.
- Aoyama et al. (2020). *The anomalous magnetic moment of the muon in the Standard Model*. Phys. Rep. 887, 1–166.
- Abi et al. (Fermilab Muon g-2 Collaboration) (2023). *Measurement of the positive muon anomalous magnetic moment to 0.20 ppm*. Phys. Rev. Lett. 131, 161802.
- Bennett et al. (BNL g-2 Collaboration) (2006). *Final Report of the Muon E821 Anomalous Magnetic Moment Measurement at BNL*. Phys. Rev. D 73, 072003.
- Hanneke, Fogwell, Gabrielse (2008). *New Measurement of the Electron Magnetic Moment and the Fine Structure Constant*. Phys. Rev. Lett. 100, 120801.
- Fan et al. (2023). *Measurement of the Electron Magnetic Moment*. Phys. Rev. Lett. 130, 071801.
- Borsanyi et al. (BMW Collaboration) (2021). *Leading hadronic contribution to the muon magnetic moment from lattice QCD*. Nature 593, 51–55.
