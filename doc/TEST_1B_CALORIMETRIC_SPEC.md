# Test 1B: Calorimetric Boundary-State Cycle

**Internal Specification — StarDrive Research Group**

| Field | Value |
|-------|-------|
| Document ID | SD-TEST-1B-001 |
| Version | 0.1 (Draft) |
| Date | February 2026 |
| Author | Jack Al-Kahwati |
| Classification | Internal / Pre-Publication |
| Contact | jack@thestardrive.com |

---

## 1. Objective

Determine whether controlled topological boundary-state transitions in a superconducting thin-film resonator produce a **measurable, repeatable, state-correlated excess energy term** under strict calorimetric accounting.

This test gates all downstream claims about BPR energy transduction. A null result closes the energy pathway. A positive result, if it survives all null configurations, opens the anomalous energy channel for engineering characterization.

---

## 2. Hypothesis

**H₁ (Alternative):** Cycling a superconducting thin film through vortex creation/annihilation (winding number transitions) produces a cycle-averaged energy excess ΔE\_excess > 0 that:
- is correlated with the topological state variable W (winding number),
- is not attributable to drive energy, thermal bath exchange, magnetic field work, or readout back-action,
- scales with film area, winding number, and cycle frequency as predicted by the BPR coupling model.

**H₀ (Null):** All measured energy is fully accounted for by standard superconductor thermodynamics (BCS condensation, vortex self-energy, flux-flow dissipation, thermal transport). ΔE\_excess = 0 within measurement uncertainty.

---

## 3. Primary Metric

```
ΔE_excess = ∫_cycle (P_out − P_in,drive − P_in,thermal − P_in,field) dt
```

| Symbol | Definition | Units |
|--------|-----------|-------|
| P\_out | Total power leaving the device (heat + electrical + mechanical) | W |
| P\_in,drive | Electrical power supplied to thermal and magnetic drives | W |
| P\_in,thermal | Net heat flow from thermal bath into the device | W |
| P\_in,field | Magnetic field energy flow (coil work) | W |
| ΔE\_excess | Residual energy per cycle after accounting | J |

---

## 4. Acceptance Criteria

### 4.1 Detection Threshold

| Criterion | Requirement |
|-----------|-------------|
| Statistical significance | ΔE\_excess > 5σ above zero, sustained over ≥100 consecutive cycles |
| Temporal stability | Signal must not decay by >10% over 72 hours of continuous cycling |
| Absolute floor | \|ΔE\_excess\| > 10⁻¹⁵ J per cycle (above thermal noise floor at 4 K) |

### 4.2 Scaling Verification

If ΔE\_excess > 0, it must follow at least two of the following predicted scalings:

| Control knob | Predicted scaling | Tolerance |
|-------------|-------------------|-----------|
| Winding number W | ΔE ∝ W² | Exponent 2.0 ± 0.3 |
| Film area A | ΔE ∝ A | Exponent 1.0 ± 0.2 |
| Cycle frequency f | P\_anomaly ∝ f (for f < f\_relax) | Linear within 20% |
| Temperature T\_low | ΔE increases as T\_low decreases below T\_c | Monotonic |

### 4.3 Null Survival

ΔE\_excess must be **absent or statistically insignificant** in ALL five null configurations (Section 7). If any null shows the same signal, the result is attributed to that parasitic channel.

---

## 5. Apparatus

### 5.1 Device Under Test (DUT)

| Component | Specification |
|-----------|--------------|
| Substrate | Single-crystal diamond, 5 mm × 5 mm × 0.3 mm |
| SC film | Niobium, 100 nm thickness, e-beam deposited |
| Boundary engineering | Periodic surface grating (pitch 1–10 μm, depth 20 nm) |
| Resonator | MEMS cantilever or membrane, Q > 10⁵ at 4 K |
| Thermometry | Ruthenium oxide on-chip (±1 mK at 4 K) |
| Field coil | Superconducting solenoid, 0–50 mT, rise time < 1 ms |

### 5.2 Calorimetric Enclosure

| Component | Specification |
|-----------|--------------|
| Inner shield | Gold-plated copper, thermal link to mixing chamber |
| Outer shield | Superconducting lead, magnetic shielding > 60 dB |
| Thermal isolation | Kevlar suspension, < 10⁻⁸ W/K to bath |
| Heat-flow sensor | Differential thermocouple array, resolution < 10⁻¹² W |
| Calibration heater | On-chip resistive heater, known to < 0.1% |

### 5.3 Drive and Readout

| Component | Specification |
|-----------|--------------|
| Temperature cycling | Pulsed laser heating + cryostat feedback, ΔT/Δt ~ 1 K/ms |
| Field cycling | DAC-driven current source, jitter < 10⁻⁵ |
| Mechanical readout | Fiber-optic interferometer, displacement noise < 10⁻¹⁵ m/√Hz |
| Electrical readout | SQUID magnetometer for flux monitoring |
| Data acquisition | 24-bit ADC, 1 MS/s, synchronous with drive waveforms |

---

## 6. Cycle Protocol

Each cycle consists of five phases, executed at the target repetition rate (default 100 Hz for Nb):

| Phase | Duration | Start State | End State | Drive |
|-------|----------|-------------|-----------|-------|
| 1. Cool | 2 ms | (W=0, T=12 K, B=0) | (W=0, T=4 K, B=0) | Cryostat |
| 2. Vortex entry | 1 ms | (W=0, T=4 K, B=0) | (W=W\_t, T=4 K, B=10 mT) | Solenoid ramp |
| 3. Hold / measure | 2 ms | (W=W\_t, T=4 K, B=10 mT) | same | Readout active |
| 4. Vortex exit | 1 ms | (W=W\_t, T=4 K, B=10 mT) | (W=0, T=4 K, B=0) | Solenoid ramp down |
| 5. Heat | 4 ms | (W=0, T=4 K, B=0) | (W=0, T=12 K, B=0) | Heater pulse |

Total cycle time: 10 ms (100 Hz). Adjustable for scaling tests.

---

## 7. Null Configurations (Kill Switches)

Each null eliminates one specific mechanism. The primary measurement must **survive all five** to count as a positive result.

### NULL-1: No Superconductivity

| Parameter | Change |
|-----------|--------|
| Film | Replace Nb with Cu (same geometry, same thickness) |
| Cycle | Identical thermal and field cycling |
| Expected result | ΔE\_excess = 0 (no condensation, no vortices, no winding) |
| Eliminates | All SC-specific mechanisms |

### NULL-2: No Vortices

| Parameter | Change |
|-----------|--------|
| Film | Same Nb film |
| Cycle | Same thermal cycling, but B = 0 throughout |
| Expected result | ΔE\_excess = 0 (W = 0 at all times) |
| Eliminates | Vortex creation/annihilation; tests whether W ≠ 0 is required |

### NULL-3: No Phase Transition

| Parameter | Change |
|-----------|--------|
| Film | Same Nb film |
| Cycle | Stay at T = 4 K throughout; cycle B only |
| Expected result | ΔE\_excess reduced or absent |
| Eliminates | T\_c crossing; tests whether condensation transition is required |

### NULL-4: Readout Back-Action

| Parameter | Change |
|-----------|--------|
| Film | Same Nb film |
| Cycle | Identical, but replace readout with matched dummy load |
| Expected result | ΔE\_excess unchanged if readout is passive |
| Eliminates | Measurement-injected energy |

### NULL-5: Reciprocity

| Parameter | Change |
|-----------|--------|
| Film | Same Nb film |
| Cycle | Invert cycle order (heat first, cool second) and reverse field polarity |
| Expected result | ΔE\_excess has same magnitude, tracks W not protocol timing |
| Eliminates | Protocol-specific artifacts, asymmetric drive coupling |

---

## 8. Error Budget

| Source | Estimated magnitude | Mitigation |
|--------|-------------------|------------|
| Calibration heater accuracy | < 0.1% of Q\_drive | NIST-traceable resistor |
| Thermal link drift | < 10⁻¹² W over 1 hour | Continuous monitoring, regression |
| Magnetic field hysteresis | < 1% of field energy | Degauss protocol between runs |
| Flux-flow dissipation uncertainty | ~10% of vortex energy | Compare with flux-flow resistivity measurement |
| Eddy current heating in shields | < 10⁻¹⁰ W at 100 Hz | Low-conductivity shield material |
| Readout laser heating | < 10⁻¹² W | Calibrate with laser on/off |
| RF pickup / ground loops | < 10⁻¹⁴ W | Shielded enclosure, battery-powered readout |
| Thermal acoustic noise (4 K) | kT ~ 5.5 × 10⁻²³ J per mode | Average over 10⁶ cycles |
| Systematic offset | Unknown | Null configurations bound this |

**Total systematic uncertainty budget: < 10⁻¹⁴ J per cycle** (goal).

---

## 9. Data Products

| Product | Format | Cadence |
|---------|--------|---------|
| Raw time-series | HDF5 (all channels, full bandwidth) | Continuous |
| Per-cycle energy balance | CSV (one row per cycle) | Per cycle |
| Null comparison table | CSV (primary vs each null) | Per run |
| Scaling fits | JSON (exponents, uncertainties, chi²) | Per parameter sweep |
| Go/no-go summary | PDF (1-page decision document) | Per campaign |

---

## 10. Decision Logic

```
IF (ΔE_excess > 5σ)
  AND (stable over 72 hours)
  AND (NULL-1 through NULL-4 show no signal)
  AND (NULL-5 shows same signal tracking W)
  AND (scaling matches ≥2 of 4 predicted laws)
THEN → ANOMALY DETECTED
  → Proceed to engineering characterization
  → External replication request

ELSE IF (ΔE_excess > 3σ but fails one or more nulls)
THEN → INCONCLUSIVE
  → Identify parasitic channel
  → Redesign to eliminate
  → Repeat

ELSE → NULL RESULT
  → BPR boundary-state energy channel is closed
  → Publish null result with bounds
  → Platform retains value for SC resonator / pulsed-power applications
```

---

## 11. Schedule (Minimum Viable)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Design & procurement | 3 months | Detailed drawings, parts ordered |
| Fabrication & assembly | 4 months | DUT + calorimeter integrated |
| Calibration | 2 months | Error budget validated |
| Primary measurement | 2 months | ΔE\_excess with error bars |
| Null campaign | 3 months | All 5 nulls completed |
| Analysis & write-up | 2 months | Go/no-go report |
| **Total** | **16 months** | |

---

## 12. Cost Estimate (ROM)

| Category | Estimate |
|----------|----------|
| Cryostat + mixing chamber | $150K (existing lab infrastructure assumed) |
| MEMS device fabrication (10 units) | $80K |
| SC solenoid + drive electronics | $40K |
| Calorimetry sensors + calibration | $60K |
| Data acquisition + computing | $30K |
| Labor (2 FTE × 16 months) | $400K |
| Contingency (20%) | $150K |
| **Total** | **~$910K** |

---

## 13. DOE Alignment Statement

> StarDrive is developing a pulsed-field superconducting boundary platform
> to test whether controlled topological state transitions introduce an
> anomalous, state-correlated energy term under strict calorimetric
> accounting.  The program's first milestone is a closed energy balance
> measurement with multiple null configurations that bound any anomaly.
> Independent of outcome, the platform advances high-Q superconducting
> resonators, pulsed power delivery, and precision thermal metrology
> relevant to DOE priorities in advanced materials, sensors, and
> grid-resilient pulsed energy systems.

---

## 14. Code Reference

The energy balance model and scaling predictions are implemented in:

```python
from bpr.energy_balance import BoundaryStateCycle, run_full_analysis

# Default Nb-on-diamond configuration
config = BoundaryStateCycle.default_diamond_mems()

# Full energy balance
balance = config.compute_energy_balance()
print(balance.summary())

# Scaling predictions (falsifiable curves)
scalings = config.compute_scaling_predictions()

# Null configurations
from bpr.energy_balance import generate_null_configurations
nulls = generate_null_configurations(config)
for null in nulls:
    null_balance = null.cycle.compute_energy_balance()
    print(f"{null.name}: ΔE = {null_balance.delta_E_excess:.4e} J")

# Complete report
print(run_full_analysis())
```

---

*Document generated as part of BPR-Math-Spine Test 1B specification.*
*Contact: jack@thestardrive.com*
