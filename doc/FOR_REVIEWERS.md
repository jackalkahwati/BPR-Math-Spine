# BPR: For Physicists Reviewing This Framework

> **One-page summary for reviewers.**
> **Contact:** jack@thestardrive.com

## What BPR is, accurately positioned

**BPR is a holographic emergent-spacetime framework on a discrete Z_p substrate, in the same research family as Loop Quantum Gravity, causal sets, and tensor-network approaches to emergence.** It distinguishes itself by attempting to derive Standard Model flavor-sector parameters from three substrate quantities (J, p = 104761, z = 6), giving CKM elements and quark mass ratios at ~1% accuracy and providing a specific Casimir falsifier (δ = 2 from Postulate 0c).

The framework's strongest content is its **flavor-sector pre-dictions** — registered numerical values for not-yet-measured doubly-heavy baryon properties that LHCb will test in the next few years. This is the Asymptotic-Safety-style positioning the framework deserves to be evaluated on.

For full structural mapping into mainstream concepts: [BPR_MAINSTREAM_MAPPING.md](BPR_MAINSTREAM_MAPPING.md).

## Flavor sector — the headline content

Full details: **[BPR_FLAVOR_SECTOR.md](BPR_FLAVOR_SECTOR.md)** ← read this first if you are evaluating BPR as a flavor-physics framework.

### Registered pre-dictions (BEFORE measurement)

| Pre-diction | Registered value | Test | Status |
|---|---|---|---|
| Ξcc* − Ξcc hyperfine | **85.9 MeV** | `test_hyperfine_registered_prediction_value` | LHCb spin-3/2 family upcoming |
| Ωcc* − Ωcc hyperfine | **94.3 MeV** | `test_omega_cc_hyperfine_registered` | LHCb upcoming |
| Ξbb* − Ξbb hyperfine | **28.3 MeV** | `test_xi_bb_hyperfine_registered` | LHCb pipeline |
| Ωbb* − Ωbb hyperfine | **33.3 MeV** | `test_omega_bb_hyperfine_registered` | LHCb pipeline |
| Ωbb − Ξbb SU(3) split | **100.3 MeV** | `test_omega_bb_xi_bb_su3_splitting` | LHCb pipeline |
| Ωccc⁺⁺ (spin-3/2) mass | **4900 MeV** | `test_omega_ccc_within_lattice_band` | not yet observed |
| Casimir δ exponent | **2** (Postulate 0c) | `bpr/casimir.py` | precision-Casimir target |

### Post-dictions (matches known values, demonstrating consistency)

CKM elements V_cb (0.2% off PDG), V_ub (4% off), δ_CP (0.25% off), Cabibbo angle (exact from m_d/m_s), and all six quark masses to ≤ 1.5%. Substrate-derived using boundary mode integers from (z = 6, n_gen = 3). Full details in `bpr/qcd_flavor.py`.

## Other testable claims

| Test | BPR Prediction | Timeline | Falsification |
|------|-----------------|----------|----------------|
| Neutrino mass ordering | Normal | JUNO ~2027 | Inverted → falsifies |
| 0νββ decay | No signal (Dirac, p ≡ 1 mod 4) | LEGEND-200+ | Signal → falsifies |
| Casimir δ exponent | 2 (derived, Postulate 0c) | Delft/STM 1–3 yr | δ ≈ 1.37 or null at 10⁻⁹ → falsifies |
| Lorentz invariance | \|δc/c\| ≤ 3.4×10⁻²¹ | CTA 2026+ | Stronger violation → falsifies |
| Born rule | small modification ~ 10⁻⁵ | Many-photon | Born exact to 10⁻⁷ → would constrain BPR |

## What this framework does NOT claim

For credibility, the framework's distinctive content (above) should be evaluated independently of the speculative extensions documented elsewhere in the repo. Specifically:

- **Phason sector / topological propulsion** (`bpr/phason_sector.py`) is a separate research target gated on the δ = 2 Casimir measurement. Honest verdict for engineering: UNDECIDABLE at 31 orders below current sensitivity. See `doc/RECOHERENCE_DEVICE_ROADMAP.md` for the scoping.
- **Eq (5) consciousness coupling** (`bpr/information.py`) has 8 free parameters and an empirical bound (χ_max · ε ≲ 10⁻³ behavioral, ≲ 10⁻⁵ at QRNG). Calibrated against the psi literature in `doc/experiments/CONSCIOUSNESS_EMPIRICAL_CALIBRATION.md`. Not part of the flavor-sector evaluation.
- **Moral phenomenology overlays** (privation framework, exorcism mapping, coercive audio) are interpretive overlays on the framework's identity / permission / salience objects. Documented honestly but should not influence flavor-sector evaluation.

These sectors exist in the repo for the framework author's own program. They should not be cited as part of BPR's flavor-physics or substrate-physics credentials.

## Parameter honesty

The framework is explicit about which numbers are derived from substrate vs which are fitted phenomenologically. The June 2026 audit (see commit history) flagged:

- δ_Casimir = 2 (derived from Postulate 0c, NOT the earlier fitted 1.37)
- l_μ = √210 (derived from z(z²−1), NOT the earlier l_μ = 14)
- θ_23 coefficient 1.35 and θ_12 coefficient 3.5 in PMNS: FITTED, not derived
- ln(p)/(ln(p)+1) finite-boundary correction in M_Pl/v_EW: FITTED
- 1/(4 ln p) interband factor in MgB₂ Tc: FITTED

The fitted coefficients are flagged in the relevant docstrings. The framework's "two-integer" parameter-honesty claim is explicitly bounded to the dimensionless sector with these caveats. See `doc/LIMITATIONS_AND_FALSIFICATION.md`.

## Quick verification

```bash
git clone https://github.com/jackalkahwati/BPR-Math-Spine.git && cd BPR-Math-Spine
pip install -r requirements-docker.txt  # or mamba env create -f environment.yml
pytest -q                                # 1471 tests pass on main (June 2026)
python -c "from bpr.qcd_flavor import doubly_charmed_hyperfine_splitting; print(doubly_charmed_hyperfine_splitting()['registered_prediction_MeV'])"  # → 85.93
```

## Key documents (in evaluation order)

| Document | Purpose |
|----------|---------|
| **[BPR_FLAVOR_SECTOR.md](BPR_FLAVOR_SECTOR.md)** | **Read first** — headline pre-dictions in Asymptotic-Safety style |
| [BPR_MAINSTREAM_MAPPING.md](BPR_MAINSTREAM_MAPPING.md) | Where BPR sits among LQG, AdS/CFT, causal sets, etc. |
| [LIMITATIONS_AND_FALSIFICATION.md](LIMITATIONS_AND_FALSIFICATION.md) | Parameter accounting; BPR-unique vs shared content |
| [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md) | Tests with falsification criteria |
| [VALIDATION_STATUS.md](../VALIDATION_STATUS.md) | Classification of all predictions |
| [BENCHMARK_SCORECARD.md](BENCHMARK_SCORECARD.md) | Quantitative comparisons vs PDG/Planck/CODATA |
| [experiments/DOUBLY_CHARMED_BARYONS_CALIBRATION.md](experiments/DOUBLY_CHARMED_BARYONS_CALIBRATION.md) | Calibration against the LHCb Ωcc⁺ discovery |

## Bottom line

BPR is a **testable framework with sharp registered pre-dictions in the SM flavor sector**, in the same epistemic position as Asymptotic Safety circa 2009 (before the Higgs measurement vindicated its mass prediction). Whether BPR's flavor sector ascends to the same status depends on LHCb's upcoming doubly-heavy baryon measurements landing near or far from the registered values above.

Evaluating BPR on its flavor-sector content alone — separately from the speculative extensions — gives a fair test of the framework's discriminating empirical claims.
