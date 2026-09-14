# BPR Flavor Sector — Headline Predictions

> **Status:** This is the framework's strongest content for outside-physicist
> evaluation. Repackages BPR's phenomenological flavor-sector formulas and candidate predictions
> (quark mass ratios, CKM, CP phase, doubly-heavy baryons) in the same
> registered-pre-diction style that Asymptotic Safety used for the Higgs mass.
> The speculative sectors (phason propulsion, consciousness coupling, moral
> phenomenology) are intentionally NOT in this doc — they belong elsewhere
> and should not influence assessment of the flavor-sector content.

## 1. What this doc claims

BPR records specific candidate predictions in the Standard Model flavor
sector. Their numerical formulas are retained, but their physical mode
assignments are conjectural. They use substrate quantities (J, p, z),
assumed boundary labels, empirical n_gen=3, normalization inputs, and
inherited quark-model relations. They are not established parameter-free
predictions from a derived fermionic Hamiltonian. The historical registration
and measurement-status labels below are retained, not independently
reverified in this audit. Code-locking establishes reproducibility, not
independent timestamp verification or empirical success.

These formulas can be assessed separately from BPR's broader interpretive
claims. Whether a comparison distinguishes BPR from conventional models
requires an explicit alternative model, independently constrained inputs
and uncertainty analysis; numerical specificity alone does not establish it.
The [negative-findings registry](CLOSED_AND_DEPRECATED.md) and
[current foundation campaign](derivations/foundation_prerequisites_2026-09-13.md)
supersede stronger historical derivation and validation claims below.

## 2. The framework, briefly

The framework's flavor-sector machinery:

- Substrate parameters: J (energy anchor), p = 104761 (structural prime), z = 6 (coordination), n_gen = 3 (empirical input, not derived from topology).
- Conjectural flavor-mode labels evaluated from (z, n_gen): l_u = 1, l_c = z(z−2) = 24, l_t = (z²−1)(z+n_gen+2−N_c) + n_gen = 283; l_d = 1, l_s = z−2 = 4, l_b = z(z−1) = 30.
- Quark masses from boundary mode squared, anchored to v_EW for up-type (m_t = v_EW/√2) and to m_b via boundary coordination (factor 2 + 1/(3 ln p)) for down-type.
- CKM angles from quark mass ratios using Gatto-Sartori-Tonin (Cabibbo) and Fritzsch-style relations (V_cb, V_ub), with a boundary-coordination suppression √(ln p + z/3).
- CP phase from boundary geometry: δ_CP = π/2 − 1/√(z+1).
- Doubly-heavy baryons: exact spin algebra + κ * (m_s − m_d) for SU(3) splittings + 3/2 × (1 + 2/z) for hyperfine; κ from mode-ratio ansatz at z = 6.

The numerical formulas and anchors are retained, including the m_τ anchor.
The physical mode assignments and normalization prescriptions remain
assumptions, so this is not a demonstrated parameter-free chain from the
substrate. In the PMNS sector, θ₂₃ coefficient 1.35 and θ₁₂ coefficient 3.5
are fitted, not derived. θ₁₃ additionally depends on the chosen Gaussian
localization and mode-identification assumptions in `bpr/neutrino.py`.

## 3. Post-dictions (matches against measured values)

| Observable | BPR value | Quoted empirical / literature reference | Residual | Source |
|---|---|---|---|---|
| m_u | 2.172 MeV | 2.16 ± 0.05 | 0.6% | `QuarkMassSpectrum` |
| m_d | 4.733 MeV | 4.67 ± 0.07 | 1.4% | `QuarkMassSpectrum` |
| m_s | 93.88 MeV | 93.4 ± 0.8 | 0.5% | `QuarkMassSpectrum` |
| m_c | 1251 MeV | 1270 ± 20 | 1.5% | `QuarkMassSpectrum` |
| m_b | 4195 MeV | 4180 ± 30 | 0.4% | `QuarkMassSpectrum` |
| m_t | 173.95 GeV | 172.76 ± 0.30 | 0.7% | `QuarkMassSpectrum` |
| sin θ_C (Cabibbo) | 0.225 | 0.225 | 0% | GST relation |
| V_cb | 0.0406 | 0.0405 ± 0.001 | 0.2% | Fritzsch + boundary |
| V_ub | 0.00367 | 0.00382 ± 0.020 | 4% | mass ratio |
| δ_CP (CKM) | 1.193 rad | 1.196 ± 0.04 | 0.25% | boundary geometry |
| sin² θ_C (predicted from m_d/m_s) | 0.0504 | 0.0506 | 0.4% | GST + BPR masses |
| Ωcc⁺ − Ξcc⁺ splitting | 105.2 MeV | 105.45 ± 5.02 | 0.05σ | κ × (m_s − m_d) |

These are historical numerical comparisons with the quoted reference
values. Their values and residuals are retained without a fresh source
audit. Agreement with inherited empirical relations or a literature target
does not independently validate the substrate, establish the physical mode
assignments, or demonstrate fewer independently calibrated parameters.

## 4. Registered pre-dictions (BEFORE measurement)

The following historical candidate targets and thresholds are retained
without regrading. Their code tests and dates do not alone establish
independent preregistration, uniqueness against alternatives or empirical success.

### Spin-3/2 doubly-charmed family (LHCb's announced next target)

| Pre-diction | Registered | Lattice band | Falsifier | Test |
|---|---|---|---|---|
| Ξcc* − Ξcc hyperfine | **85.9 MeV** | 70 – 100 | outside 64–90 MeV refutes coordination-shell | `test_hyperfine_registered_prediction_value` |
| Ωcc* − Ωcc hyperfine | **94.3 MeV** | similar to Ξcc band | analogous | `test_omega_cc_hyperfine_registered` |

### Doubly-bottom and triply-charmed (LHCb pipeline, not yet observed)

| Pre-diction | Registered | Lattice band | Status | Test |
|---|---|---|---|---|
| Ξbb* − Ξbb hyperfine | **28.3 MeV** | 22 – 45 | not yet measured | `test_xi_bb_hyperfine_registered` |
| Ωbb* − Ωbb hyperfine | **33.3 MeV** | 28 – 50 | not yet measured | `test_omega_bb_hyperfine_registered` |
| Ωbb − Ξbb (SU(3) split) | **100.3 MeV** | 85 – 120 | not yet measured | `test_omega_bb_xi_bb_su3_splitting` |
| Ωccc⁺⁺ (spin-3/2) mass | **4900 MeV** | 4760 – 5050 | not yet observed | `test_omega_ccc_within_lattice_band` |

### Casimir physics (precision-Casimir target)

| Pre-diction | Registered | Standard QED | Falsifier | Reference |
|---|---|---|---|---|
| δ (Casimir deviation exponent) | **2** (derived, Postulate 0c) | no anomalous exponent | δ ≈ 1.37 or null at 10⁻⁹ falsifies | `bpr/casimir.py` |

### Other registered pre-dictions

- **Σm_ν (neutrino sum)** ≈ 0.06 eV in the conditional mass ansatz of `bpr/neutrino.py`. Cosmological analyses constrain the sum subject to their model assumptions; beta-decay experiments constrain a different effective-mass observable.
- **0νββ decay = null** under the model's p ≡ 1 mod 4 → Dirac prescription. The cited LEGEND-200 null result is compatible with this prescription, but does not establish Dirac neutrinos or validate the proposed mechanism.
- **Strong CP θ_QCD = 0** without axion — testable by neutron EDM bounds tightening

## 5. Why this matters for comparative ranking

In the comparative scoreboard against other frameworks (Asymptotic Safety,
holographic emergence, LQG, causal sets, etc.):

- The Tier 1 frameworks earned their position with **specific sharp pre-
  dictions** (Asymptotic Safety's Higgs mass, AdS/CFT's η/s, causal sets'
  Λ magnitude). The qualitative move that distinguishes them from "looks
  consistent with known physics" frameworks is registering values before
  measurement.

- BPR's flavor-sector pre-dictions above are in that same class: sharp
  values registered before measurement, with defined falsifiers and locked
  tests. If the Ξcc* hyperfine lands at 85.9 ± 2 MeV, BPR's flavor sector
  joins Asymptotic Safety in the "successful pre-diction" category. If it
  lands far from 86, the framework loses credibility — which is exactly
  what a falsifiable framework should do.

- The other sectors of BPR (phason propulsion, consciousness coupling, etc.)
  do not have this property and SHOULD NOT be presented as part of this
  evaluation. They are speculative extensions whose evidence requirements
  are separate.

## 6. The asymptotic-safety analogy explicit

| Aspect | Asymptotic Safety | BPR flavor sector |
|---|---|---|
| Substrate constraint | UV completeness of SM under exact RG | Boundary mode integers from (z, n_gen) + p from α |
| Single sharp pre-diction at LHC | m_H ≈ 126 GeV | Ξcc* − Ξcc = 85.9 MeV |
| Pre-diction registered before measurement | 2009 (Shaposhnikov-Wetterich) | 2026 (this doc, in code tests) |
| Status today | Confirmed by 2012 measurement (125.1) | LHCb spin-3/2 family forthcoming |
| Framework scope | Quantum gravity + SM Higgs | Boundary substrate + SM flavor sector |
| Speculative extensions outside this scope | None presented as evidence | Phason / consciousness / etc. — SEPARATE |

This is the framework's strongest positioning. Outside physicists evaluating
BPR should be directed to this doc first.

## 7. What this doc does NOT include

Explicitly excluded so the flavor-sector evaluation is clean:

- Phason sector and propulsion claims (`bpr/phason_sector.py`)
- Consciousness coupling Eq (5) (`bpr/information.py`)
- Moral phenomenology / privation overlay (`doc/MORAL_PHENOMENOLOGY_PRIVATION.md`)
- Exorcism and re-coherence device material
- Coercive audio calibration
- All the Tier-3 viral-content engagement (Buga, Integratron, pyramids, etc.)

These are real parts of the broader project but they should not be presented
together with the flavor-sector content for outside-physicist evaluation.
Combining them weakens the flavor sector's credibility by association. If
those sectors are eventually vindicated, they earn their place separately.

## 8. References

- `bpr/qcd_flavor.py` — QuarkMassSpectrum, CKMMatrix, doubly_charmed and
  doubly_bottom predictions
- `bpr/neutrino.py` — neutrino mass sum, mixing angles, CP phase
- `bpr/charged_leptons.py` — lepton mass hierarchy from boundary modes
- `bpr/casimir.py` — δ = 2 from Postulate 0c
- `doc/experiments/DOUBLY_CHARMED_BARYONS_CALIBRATION.md` — Ωcc detailed calibration
- `doc/BPR_MAINSTREAM_MAPPING.md` — where BPR sits in the larger conversation
- Tests: `tests/test_doubly_charmed.py` (13 tests including all registered pre-dictions)

External references:
- Shaposhnikov & Wetterich (2009), "Asymptotic safety of gravity and the
  Higgs boson mass," arXiv:0912.0208
- Brown, Z. S. et al. (2014), arXiv:1409.0497 (lattice doubly-heavy baryons)
- Mathur, N. et al. (2018), arXiv:1807.00174 (lattice doubly-charmed)
- Padmanath, M. (2019), various (doubly-bottom lattice)
- Gatto, Sartori, Tonin (1968) — Cabibbo angle from quark masses
- Fritzsch (1977) — CKM mass-hierarchy structure
