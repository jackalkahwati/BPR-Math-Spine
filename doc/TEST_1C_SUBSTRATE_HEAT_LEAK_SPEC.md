# Test 1C: Substrate Thermal Heat Leak

**Status:** Specification — Pre-Experiment  
**Version:** 0.1 (2026-02-09)  
**Depends on:** `bpr/substrate_heat_flow.py`, BPR Axiom 2 (Hamiltonian substrate dynamics)  
**Complements:** Test 1B (calorimetric boundary-state cycle)

---

## 1  Objective

Determine whether a superconducting boundary device at cryogenic temperature
receives an anomalous, steady-state heat load from a hypothesised hot substrate
(at effective temperature T\_eff), through the BPR boundary coupling channel.

This is a **qualitatively different** claim from Test 1B (closed-cycle work):

| Test 1B | Test 1C |
|---|---|
| Cyclic process on conservative F(W,T,B) | Steady-state heat transfer from external reservoir |
| Measures ΔE\_excess per cycle | Measures continuous Q̇ into DUT |
| Requires vortex state cycling | Device sits in calorimeter; no drive |
| Result: ΔF\_bpr = 0 over a closed cycle | Result: Q̇ ≠ 0 if substrate reservoir is real |

---

## 2  Hypothesis

> The RPST substrate has an effective temperature
> T\_eff = J / (k\_B ln p) set by the coupling J and prime modulus p.
> Physical matter at T\_cold ≪ T\_eff is out of thermal equilibrium with
> the substrate.  The BPR boundary coupling provides a thermal link from
> substrate to device, producing a measurable steady-state heat current:
>
>     Q̇ = ∫₀^∞ (dω / 2π) ℏω T(ω) [n(ω, T\_eff) − n(ω, T\_cold)]
>
> where T(ω) depends on the coupling strength, boundary geometry, and
> spectral overlap between substrate modes and device modes.

**Critical caveat:** T\_eff as defined above is a *conjecture* derived from
coarse-graining entropy.  It requires KMS-state verification.  This test
treats T\_eff as a hypothesis parameter to be bounded or falsified.

---

## 3  Primary Metric

```
Q̇_excess  =  Q̇_measured(DUT)  −  Q̇_background(same package, inert boundary)
```

All measurements at constant cryostat temperature T\_cold, stable to ±1 mK.

**Units:** Watts (pW – nW range expected).

---

## 4  Acceptance Criteria

### 4.1  Detection

- Q̇\_excess > 5σ above measurement noise floor, sustained over ≥ 72 hours.
- Absolute value consistent with at least one transfer-law prediction
  (Landauer, ohmic, or SB proxy) to within 3 orders of magnitude.

### 4.2  Attribution (rules out mundane heat)

- Q̇\_excess does NOT track cryostat wall temperature variations.
- Q̇\_excess DOES scale with boundary area (halve area → predictable reduction).
- Q̇\_excess DOES change when boundary material is swapped (SC → normal metal).
- Q̇\_excess survives all five null configurations (§7).

### 4.3  Spectral (if bolometric readout available)

- Spectral content of Q̇\_excess is structured (not broadband 1/f drift).
- Spectral peaks, if present, correlate with device resonance frequencies.
- Spectrum changes when device resonance is tuned.

---

## 5  Predicted Signal Range

From `bpr/substrate_heat_flow.py` with default parameters
(J = 1 eV, p = 10⁵, λ\_eff = 6×10⁻⁷, A = 1 cm², T\_cold = 4 K):

| Backend     | T\_eff [K] | Q̇ [W]     | Q̇/A [W/m²] | Status        |
|-------------|-----------|-----------|------------|---------------|
| Landauer    | 1008      | see run   | see run    | Physics-based |
| Ohmic–Kubo  | 1008      | ~pW       | ~nW/m²     | Conservative  |
| SB proxy    | 1008      | ~μW       | ~mW/m²     | Upper bound   |

**Backend spread:** up to 6 orders of magnitude.  The correct answer depends
on T(ω), which is experimentally unknown.

**Instrument noise floors:**
- TES bolometer: NEP ~ 10⁻¹⁵ W
- SQUID-based: NEP ~ 10⁻¹⁸ W

---

## 6  Apparatus

### 6.1  Device Under Test (DUT)

- Superconducting thin-film resonator (Nb or NbN, T\_c ≈ 9–16 K)
- Film area: 1 cm² (baseline), 0.5 cm² and 2 cm² for scaling test
- Film thickness: 100 nm
- Mounted on dielectric substrate (sapphire or diamond for thermal isolation)
- High-Q design (Q > 10⁶ at 4 K) to maximise λ\_eff

### 6.2  Calorimeter

- Adiabatic dewar with radiation shields at 40 K and 4 K stages
- DUT thermally anchored to a weak link (calibrated thermal conductance G)
- Temperature readout: SQUID-coupled NTD-Ge thermistor (ΔT resolution ≤ 1 μK)
- Calibration heater on DUT stage (known P\_cal for absolute calibration)

### 6.3  Shielding and Controls

- Magnetic shielding: μ-metal + superconducting lead can (< 1 μT residual)
- RF shielding: Faraday enclosure (> 80 dB attenuation, DC – 40 GHz)
- Vibration isolation: pneumatic platform + cryostat suspension
- Orientation test: rotate cryostat 90° and 180° to check for mundane
  radiative asymmetry

---

## 7  Null Configurations (Kill Switches)

Each null isolates a specific mechanism.  The signal must survive ALL nulls.

| # | Name | Change | Expected Q̇\_excess | Kills |
|---|------|--------|---------------------|-------|
| 1 | No coupling | Set λ\_eff → 0 (theory; experimentally: amorphous boundary film) | Zero | Instrumental artifact |
| 2 | Inert spectrum | Shift device resonance far from substrate thermal band | Zero | Spectral coincidence |
| 3 | Normal metal | Replace SC film with Cu of same geometry | Zero (or << baseline) | Generic film heating |
| 4 | Q-suppressed | Add lossy element to resonator (Q → 10²) | Reduced by ~Q factor | Q-dependent parasitic |
| 5 | Area scaling | Halve or double boundary area | Proportional change | Fixed systematic offset |

**Additional controls:**
- **Orientation invariance:** Signal must not change with cryostat rotation.
- **Temporal stability:** Signal must be stable over > 72 h (rules out outgassing, adsorption).
- **T\_cold scan:** Vary T\_cold from 0.1 K to 8 K.  Q̇ should follow
  n(T\_eff) − n(T\_cold) dependence, NOT track cryostat parasitic curves.

---

## 8  Error Budget

| Source | Magnitude | Mitigation |
|--------|-----------|------------|
| Cryostat radiative leak | ~pW for well-shielded system | Radiation baffles; null-1 subtraction |
| Electrical lead heat leak | ~nW per wire pair at 4 K | Superconducting leads; count and bound |
| RF pickup | Variable | Faraday cage; power-off baseline |
| Outgassing / adsorption | Decays over hours | 72-h stability requirement |
| Calibration heater error | ±1% | NIST-traceable resistance standard |
| Thermometry error | ±1 μK | Cross-calibrated against known transition |
| Vibration heating | ~fW for good isolation | Vibration monitor; correlation analysis |
| Magnetic field drift | Sub-pW for SC shield | Fluxgate outside shield; SQUID inside |

---

## 9  Data Products

1. **Q̇\_excess vs. time** — primary trace, ≥ 72 h, 1 Hz sampling.
2. **Q̇\_excess vs. T\_cold** — scan 0.1 K → 8 K, compare to model curves.
3. **Q̇\_excess vs. area** — three DUT sizes.
4. **Q̇\_excess vs. material** — SC film, normal-metal film, bare substrate.
5. **Spectral density of Q̇** — if bolometric readout resolves it.
6. **Null configuration suite** — all five nulls, each ≥ 24 h.
7. **Orientation test** — 0°, 90°, 180° cryostat rotation.

---

## 10  Decision Logic

```
IF Q̇_excess > 5σ AND stable > 72 h:
    IF survives all 5 nulls AND area scales:
        IF spectral structure matches device resonance:
            → STRONG DETECTION (proceed to Test 2: scaling / multi-device)
        ELSE:
            → CANDIDATE DETECTION (seek independent replication)
    ELSE:
        → SYSTEMATIC ARTIFACT (identify and publish null result)
ELSE:
    → NULL RESULT
    → Publish upper bound on Q̇ as function of (λ_eff, T_eff)
    → Constrain T_eff × λ_eff product for BPR parameter space
```

---

## 11  Relationship to BPR Parameter Space

A null result bounds the product T\_eff × λ\_eff:

- If no signal at 10⁻¹⁸ W sensitivity:
  - Ohmic model:  α × k\_B × ΔT × λ\_eff × ω\_c < 10⁻¹⁸ W
  - For T\_eff = 1000 K:  λ\_eff < 10⁻¹² (6 orders below current estimate)
  - For λ\_eff = 6×10⁻⁷:  T\_eff < 0.002 K (substrate colder than experiment)

Either bound is informative.  A null kills the "hot substrate" hypothesis
for the tested parameter range and constrains BPR coupling models.

---

## 12  Schedule and Cost Estimate

| Phase | Duration | Cost (USD) |
|-------|----------|------------|
| DUT fabrication (3 sizes, 2 materials) | 3 months | $50k |
| Calorimeter integration and calibration | 2 months | $30k (assumes existing cryostat) |
| Baseline runs (parasitic characterisation) | 1 month | $10k |
| Signal runs (5 nulls × 24 h + baseline × 72 h) | 2 months | $20k |
| Data analysis and reporting | 1 month | $10k |
| **Total** | **9 months** | **~$120k** |

---

## 13  DOE Framing Paragraph

> StarDrive proposes to measure steady-state thermal transport into a
> superconducting boundary device at cryogenic temperature under
> precision calorimetric conditions.  The hypothesis under test is
> whether the device-substrate boundary coupling channel admits an
> anomalous heat load correlated with boundary coherence, area, and
> material properties, while being invariant under orientation and
> environmental temperature changes.  Five null configurations isolate
> the candidate signal from known parasitic heat sources.  Independent
> of outcome, the measurement advances cryogenic calorimetry methodology,
> superconducting resonator thermal management, and boundary-loss
> characterisation relevant to DOE priorities in quantum information
> systems, advanced sensors, and superconducting RF cavities.

---

## 14  References

- `bpr/substrate_heat_flow.py` — computational model for this test.
- `bpr/rpst/boundary_energy.py` lines 282–307 — T\_eff derivation.
- `bpr/decoherence.py` — Γ\_dec expression used to derive Γ\_ex.
- `bpr/direct_coupling/stacked_enhancement.py` — λ\_eff derivation.
- `doc/TEST_1B_CALORIMETRIC_SPEC.md` — companion cycling test.
- Pendry (1983), J. Phys. A 16, 2161 — quantum limits on heat flow.
- Schwab et al. (2000), Nature 404, 974 — quantised thermal conductance.
