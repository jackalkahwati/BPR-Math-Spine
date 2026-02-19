# BPR: For Physicists Reviewing This Framework

> **One-page summary for reviewers.**  
> **Contact:** jack@thestardrive.com

---

## What Is BPR?

BPR (Boundary Phase Resonance) derives physical observables from a discrete phase field on a lattice boundary. From three substrate parameters (J, p, N) plus four fermion anchor masses, it produces 50 benchmarked predictions, all within 2σ of PDG/Planck/CODATA.

**Core hypothesis:** Observables correspond to stabilized phase configurations on constrained boundaries. Coarse-graining a ℤ_p lattice Hamiltonian yields a boundary action; coupling to bulk geometry produces predictions across particle physics, cosmology, and quantum foundations.

---

## Top 5 Falsification Tests

| Test | BPR Prediction | Timeline | Falsification |
|------|-----------------|----------|----------------|
| 1. Neutrino mass ordering | Normal | JUNO ~2027 | Inverted → falsifies |
| 2. 0νββ decay | No signal (Dirac) | LEGEND 2025+ | Signal → falsifies |
| 3. Casimir deviation | δ = 1.37 ± 0.05 at 10⁻⁸ | Delft/STM 1–3 yr | Null at 10⁻⁹ → falsifies |
| 4. Lorentz invariance | \|δc/c\| = 3.4×10⁻²¹ | CTA 2026+ | Null below 10⁻²¹ → falsifies |
| 5. Born rule | κ ~ 10⁻⁵ deviation | Many-photon 2–5 yr | Born exact to 10⁻⁷ → falsifies |

**Details:** [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md), [LIMITATIONS_AND_FALSIFICATION.md](LIMITATIONS_AND_FALSIFICATION.md)

---

## Known Limitations

- **OPEN:** Planck length (input), electroweak hierarchy (not derived).
- **FRAMEWORK:** Superconducting Tc (N(0)V from experiment), some PMNS angles.
- **CONSISTENT:** Many predictions (v_GW=c, Tsirelson) also from SM/GR — consistency checks, not unique.
- **BPR-unique:** m_s/m_d = 20.0, neutrino ordering, Dirac nature, Casimir exponent, Born κ, LIV.

---

## Quick Verification

```bash
git clone https://github.com/jackalkahwati/BPR-Math-Spine.git && cd BPR-Math-Spine
mamba env create -f environment.yml && conda activate bpr
pytest -q                                    # 488 tests
python scripts/benchmark_predictions.py      # 51 predictions vs experiment
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [LIMITATIONS_AND_FALSIFICATION.md](LIMITATIONS_AND_FALSIFICATION.md) | Limitations; BPR-unique vs shared |
| [EXPERIMENTAL_ROADMAP.md](EXPERIMENTAL_ROADMAP.md) | 10 tests with falsification criteria |
| [VALIDATION_STATUS.md](../VALIDATION_STATUS.md) | Honest classification of all 205 predictions |
| [BENCHMARK_SCORECARD.md](BENCHMARK_SCORECARD.md) | 51 predictions vs PDG/Planck/CODATA |
| [experiments/papers.md](experiments/papers.md) | 250+ papers; 115 CONFIRM across 21 theories |

---

## Bottom Line

BPR is a **testable framework** with explicit falsification criteria. It is not claimed to be a complete theory; OPEN items are documented. The framework is reproducible and ready for independent verification.
