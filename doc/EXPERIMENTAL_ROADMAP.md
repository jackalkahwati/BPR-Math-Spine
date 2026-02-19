# BPR Experimental Roadmap

> **Version:** 0.7.0 (Jan 2026)
> **Purpose:** Concrete experimental tests that can confirm or falsify BPR predictions.
> **Policy:** Every test lists the BPR prediction, the experiment, the sensitivity needed,
> and what outcome would **falsify** the theory.

---

## Priority 1: Near-Term Tests (Existing Technology, 1-3 Years)

### Test 1: Casimir Force Phonon-Channel Deviation

**The flagship BPR experiment.**

| | |
|---|---|
| **BPR predicts** | In the phonon-collective channel, boundary coupling is lambda ~ 10^{-8}, producing a measurable deviation dF/F in the Casimir force between phonon-active surfaces at sub-micron separations. |
| **Specific signal** | The Casimir force across a superconducting phase transition has a BPR component with a functional form distinct from standard Lifshitz theory. The deviation is maximized when boundary mode density is highest (near Tc). |
| **Who can test this** | Delft University (2024): on-chip superconducting Casimir platform with subatomic displacement resolution via STM. They already observe superconductivity-dependent Casimir shifts. |
| **Sensitivity needed** | ~10^{-8} fractional force deviation. Current MEMS platforms reach piconewton resolution (borderline). |
| **FALSIFICATION** | If the Casimir force matches standard Lifshitz theory to 10^{-9} across a phase transition with no anomalous structure, BPR's phonon channel is ruled out. |

**Code reference:**
```python
from bpr.casimir import sweep_radius
sweep_radius(r_min=0.2e-6, r_max=5e-6, n=40, out='data/casimir.csv')
```

---

### Test 2: Neutrino Mass Ordering (JUNO)

**Binary yes/no test already underway.**

| | |
|---|---|
| **BPR predicts** | Normal ordering (m1 < m2 < m3). Derived from p mod 4 = 1 for p = 104729 (orientable boundary). |
| **Experiment** | JUNO (Jiangmen Underground Neutrino Observatory), China. Reactor antineutrino oscillations at 53 km baseline. Data collection started 2024. |
| **Sensitivity** | ~3 sigma determination of mass ordering after 6 years of running. |
| **FALSIFICATION** | If JUNO determines **inverted ordering**, BPR's boundary topology prediction is wrong. |
| **Timeline** | First results on ordering possible by **2027**. |

**Code reference:**
```python
from bpr.neutrino import neutrino_nature
print(neutrino_nature(104729))  # → "Dirac" (p ≡ 1 mod 4)
```

---

### Test 3: Lorentz Invariance Violation (CTA / GRBs)

**BPR is just below current bounds — next-gen could see it or kill it.**

| | |
|---|---|
| **BPR predicts** | Fractional speed variation \|dc/c\| ~ 3.4 x 10^{-21} from substrate discreteness. Quadratic LIV parameter xi_2 = 1/p ~ 10^{-5}. Linear LIV xi_1 = 0 (protected by boundary CPT). |
| **Current bound** | Fermi-LAT: \|dc/c\| < 6 x 10^{-21}. LHAASO GRB 221009A: E_{QG,2} > 6 x 10^{-8} E_Pl. **BPR is just barely below these bounds.** |
| **Experiment** | Cherenkov Telescope Array (CTA), coming online ~2026-2027. Multi-TeV photon timing from blazars and GRBs with ~10x better resolution than Fermi. |
| **What to look for** | Energy-dependent photon arrival time: dt proportional to E^2 x distance. BPR predicts this is nonzero at a specific level. |
| **FALSIFICATION** | If CTA pushes the bound below 10^{-21} with no signal, BPR's substrate discreteness prediction is ruled out. |
| **This is the most exciting test** because BPR predicts a value at the edge of detectability. |

**Code reference:**
```python
from bpr.quantum_gravity_pheno import ModifiedDispersion
md = ModifiedDispersion(p=104729)
print(f"xi_1 = {md.xi_1}")        # 0.0 (CPT protected)
print(f"xi_2 = {md.xi_2:.2e}")    # 9.55e-06
print(f"GRB delay (1 TeV, 1 Gpc) = {md.grb_time_delay(1000, 1000):.2e} s")
```

---

### Test 4: Neutrinoless Double Beta Decay (LEGEND / nEXO)

**Tests Dirac vs Majorana nature.**

| | |
|---|---|
| **BPR predicts** | Dirac neutrinos (p = 1 mod 4 gives orientable boundary, forbidding Majorana mass term). Therefore 0-nu-beta-beta decay does NOT occur. |
| **Experiments** | LEGEND-200 (running, Gran Sasso), LEGEND-1000 (planned), nEXO. Half-life sensitivity up to ~10^{28} years. |
| **FALSIFICATION** | If 0-nu-beta-beta decay is observed at any rate, BPR's orientability prediction for p = 104729 is falsified. |
| **Timeline** | LEGEND-200 results in 2025-2026. LEGEND-1000 by ~2030. |

---

## Priority 2: Medium-Term Tests (2-5 Years)

### Test 5: Born Rule Violation at 10^{-5}

**BPR predicts a specific deviation amplitude.**

| | |
|---|---|
| **BPR predicts** | P(alpha) = \|psi_alpha\|^2 x (1 + O(1/p)), with deviation ~10^{-5}. This shows up as a nonzero Sorkin parameter kappa_3 in triple-slit interference. |
| **Current bound** | Best measurement: kappa_3 < 2 x 10^{-3} (single-particle). BPR prediction is 100x below this. |
| **Path forward** | Many-particle interference (Phys. Rev. Research, 2020) is exponentially more sensitive. A 10-photon coincidence experiment could reach 10^{-5}. Superconducting qubit measurements (2025 proposals) may also reach this precision. |
| **FALSIFICATION** | If Born rule holds to 10^{-7}, the BPR microstate counting model is wrong. |

**Code reference:**
```python
from bpr.quantum_foundations import BornRule
br = BornRule(p=104729)
print(f"Born rule accuracy: {br.accuracy}")      # 0.999990
print(f"Deviation amplitude: {br.deviation:.2e}") # 9.55e-06
```

---

### Test 6: Decoherence Rate Scaling (Molecule Interferometry)

**Tests the impedance-mismatch decoherence mechanism.**

| | |
|---|---|
| **BPR predicts** | Decoherence rate scales as Gamma proportional to DeltaZ^2 (quadratic in impedance mismatch), with sub-linear temperature dependence below ~1 K. A critical winding W_crit marks the quantum-classical boundary. |
| **Experiments** | OTIMA (Vienna): matter-wave interferometry with molecules up to 10^4 amu. MAQRO (space mission proposal): decoherence in microgravity. |
| **What to look for** | Plot decoherence rate vs. molecule mass/size. BPR predicts a sharper quantum-classical transition than standard decoherence theory, at a mass scale set by W_crit. |
| **FALSIFICATION** | If decoherence rates scale linearly with DeltaZ (not quadratically), the impedance mechanism is wrong. |

**Code reference:**
```python
from bpr.decoherence import DecoherenceRate
dr = DecoherenceRate(T=0.1, delta_Z=10.0)
print(f"Gamma = {dr.gamma:.4e} Hz")
```

---

### Test 7: GUP from Mechanical Oscillators

**Tests generalized uncertainty principle beta = 1/p.**

| | |
|---|---|
| **BPR predicts** | Modified commutator [x,p] = i*hbar*(1 + beta * p^2 / M_Pl^2 c^2) with beta = 1/p ~ 10^{-5}. Minimum measurable length: l_Pl / sqrt(p) ~ 5 x 10^{-38} m. |
| **Current bound** | Quartz BAW resonators: beta_0 < 4 x 10^4 (still 9 orders above BPR). |
| **Path forward** | Next-generation optomechanical oscillators and atom interferometers are improving steadily. The beta ~ 10^{-5} regime may be reachable in 5-10 years with quantum-limited position measurement. |
| **FALSIFICATION** | If beta < 10^{-6} is established, BPR's substrate discreteness scale is wrong. |

---

## Priority 3: Analysis of Existing Data (No New Experiment Needed)

### Test 8: MOND Acceleration Scale from SPARC Galaxies

**Pure data analysis with existing rotation curves.**

| | |
|---|---|
| **BPR predicts** | a_0 = c * H_0 / (2*pi) = 1.04 x 10^{-10} m/s^2. |
| **Observed** | a_0 = 1.2 x 10^{-10} m/s^2 (Milgrom fit). Discrepancy: ~15%. |
| **Data** | SPARC database: 175 galaxies with high-quality HI/Ha rotation curves (Lelli, McGaugh, Schombert 2016). Publicly available. |
| **What to do** | Fit the Radial Acceleration Relation (RAR) from SPARC with a_0 as a free parameter. Check if 1.04 x 10^{-10} is within the allowed range. |
| **Also** | BPR predicts a_0 = c*H_0/(2*pi) evolves with cosmic time. JWST high-redshift galaxy kinematics could test a_0(z). |
| **FALSIFICATION** | If precision RAR fits exclude a_0 < 1.05 x 10^{-10}, BPR's cosmological boundary formula needs revision. |

**Code reference:**
```python
from bpr.impedance import MONDInterpolation
mond = MONDInterpolation(H0_km_s_Mpc=67.4)
print(f"a0 = {mond.a0:.4e} m/s^2")  # 1.0422e-10
```

---

### Test 9: Dark Matter Self-Interaction (Cluster Lensing)

**BPR predicts sigma/m ~ 0.02 cm^2/g.**

| | |
|---|---|
| **BPR predicts** | sigma/m ~ 0.019 cm^2/g from winding-mode scattering. |
| **Current bound** | sigma/m < 0.6 cm^2/g (cluster lensing, 95% CL). BPR is 30x below. |
| **Path forward** | Rubin Observatory (LSST): hundreds of merging clusters over the next decade, tightening bounds by ~10x. If they reach sigma/m < 0.01 cm^2/g, BPR is constrained. |
| **Timeline** | Rubin first light 2025, science operations through 2035. |

---

### Test 10: Neutrino Mixing Angles (Already Measured)

**BPR's strongest existing validation.**

| Angle | BPR | PDG 2024 | Status |
|-------|-----|----------|--------|
| theta_13 | 8.63 deg | 8.54 +/- 0.15 deg | **Within 0.6 sigma** |
| theta_12 | 33.65 deg | 33.41 +/- 0.8 deg | **Within 0.3 sigma** |
| theta_23 | 47.6 deg | ~49 +/- 1.3 deg | **Within 1.1 sigma** |

These are the cleanest existing comparisons between BPR and data.

---

## Summary: What Kills BPR

| Outcome | Implication |
|---------|-------------|
| Casimir phonon deviation NOT seen at 10^{-9} | Phonon channel coupling wrong |
| JUNO finds **inverted** mass ordering | Boundary topology prediction fails |
| 0-nu-beta-beta decay **observed** | Dirac neutrino prediction falsified |
| CTA pushes LIV bound below 10^{-22} | Substrate discreteness scale wrong |
| Born rule holds to 10^{-7} | Microstate counting model wrong |
| If ALL of the above come back null | BPR is falsified as a physical theory |

## Summary: What Validates BPR

| Outcome | Implication |
|---------|-------------|
| Anomalous Casimir shift near Tc at predicted level | Direct evidence for boundary coupling |
| Normal ordering confirmed by JUNO | Consistent (but not unique to BPR) |
| No 0-nu-beta-beta after LEGEND-1000 | Consistent with Dirac prediction |
| CTA sees energy-dependent photon delay at ~10^{-21} | Evidence for substrate discreteness |
| Born rule violated at ~10^{-5} in multi-photon experiment | Strong evidence for substrate microstructure |
| Decoherence shows quadratic DeltaZ scaling | Evidence for impedance mechanism |

**No single positive result proves BPR.** But a pattern of confirmations across
independent experiments would build a strong cumulative case.

---

## How to Run Predictions for Specific Experiments

```python
from bpr.first_principles import SubstrateDerivedTheories

sdt = SubstrateDerivedTheories.from_substrate(p=104729, N=10000, J_eV=1.0)
preds = sdt.predictions()

# Print all predictions
for key, value in sorted(preds.items()):
    print(f"{key}: {value}")

# Or generate CSV for analysis
import csv
with open('data/predictions.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['key', 'value'])
    for k, v in sorted(preds.items()):
        w.writerow([k, v])
```

---

*Document generated as part of BPR-Math-Spine v0.7.0 validation audit.*
*Contact: jack@thestardrive.com*
