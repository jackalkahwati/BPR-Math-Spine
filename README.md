# BPR‑Math‑Spine : Minimal, Reproducible Maths for Boundary Phase Resonance

> **Status :** public draft | license MIT | contact <jack@thestardrive.com>
>
> **Website :** [`bpr.science`](https://bpr.science) — interactive framework explorer, physics landscape, constant calculator, and experimental roadmap
>
> **COMPLETE FRAMEWORK** → [`doc/BPR_Complete_Framework.md`](doc/BPR_Complete_Framework.md) *(80-page unified document)*
>
> **KEY RESULT**: Testable prediction at 10⁻⁸ level via phonon-MEMS experiments
>
> **EXPERIMENTAL ROADMAP** → [`doc/EXPERIMENTAL_ROADMAP.md`](doc/EXPERIMENTAL_ROADMAP.md) *(10 concrete tests with falsification criteria)*
>
> **VALIDATION AUDIT** → [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md) *(honest classification of all 205 predictions)*
>
> **CONSISTENCY AUDIT** → [`doc/CONSISTENCY_AUDIT.md`](doc/CONSISTENCY_AUDIT.md) *(58 internal math consistency tests — no experiments needed)*
>
> **BENCHMARK SCORECARD** → [`doc/BENCHMARK_SCORECARD.md`](doc/BENCHMARK_SCORECARD.md) *(51 predictions vs PDG/Planck/CODATA data)*
>
> **EXPERIMENTS** → [`doc/experiments/`](doc/experiments/) *(papers that confirm or falsify BPR)*
>
> **CONTINUOUS EVIDENCE LOOP** → [`doc/experiments/EVIDENCE_PIPELINE.md`](doc/experiments/EVIDENCE_PIPELINE.md) *(staged paper ingestion + audit trail + sandbox policy)*
>
> **DERIVATION ROADMAP** → [`doc/derivations/DERIVATION_ROADMAP.md`](doc/derivations/DERIVATION_ROADMAP.md) *(41 DERIVED, 5 FRAMEWORK)*
>
> One‑pager → [`doc/BPR_one_pager.md`](doc/BPR_one_pager.md)

---

## Interactive Website

The BPR framework has a full public-facing website at [`bpr.science`](https://bpr.science) (or run locally — see below). Key pages:

| Page | URL | What it shows |
|------|-----|---------------|
| **Home** | `/` | Plain-English narrative of BPR — no equations required |
| **Physics Landscape** | `/viz/physics-landscape.html` | Interactive D3 node graph of all 21 theories |
| **Constant Calculator** | `/constant-calculator.html` | Derive any constant from (J, p, N) |
| **Experimental Roadmap** | `/experimental-roadmap.html` | 10 falsification tests with timelines |
| **Global Consistency Audit** | `/global-consistency-audit.html` | Radar chart of 58 consistency checks |
| **Researcher Portal** | `/researcher-portal.html` | API access, rate limits, SDK, CLI docs |
| **Paper** | `/bpr-paper.pdf` | Full 13-page LaTeX paper |

**Run locally:**
```bash
cd website/public/stitch
python3 -m http.server 9001
# open http://localhost:9001
```

---

## For Physicists: How to Read This

BPR is a **testable framework** with explicit falsification criteria. See [EXPERIMENTAL_ROADMAP.md](doc/EXPERIMENTAL_ROADMAP.md) for concrete tests and [LIMITATIONS_AND_FALSIFICATION.md](doc/LIMITATIONS_AND_FALSIFICATION.md) for known limitations and BPR-unique vs shared predictions. One-page reviewer summary: [FOR_REVIEWERS.md](doc/FOR_REVIEWERS.md).

---

## Key Results

| Coupling Channel | Derived Value | Gap to Detection |
|------------------|---------------|------------------|
| Gravitational | λ ~ 10⁻⁹⁰ J·m² | 91 orders |
| Electromagnetic | λ ~ 10⁻⁵⁴ | 50 orders |
| **Phonon Collective** | **λ ~ 10⁻⁸** | **1-2 orders** |

**All dimensionless coupling constants and ratios derived from substrate properties with 1 free continuous parameter (J, an energy anchor) and 1 experimentally anchored integer (p, fixed by α). See [LIMITATIONS_AND_FALSIFICATION.md §9](doc/LIMITATIONS_AND_FALSIFICATION.md) for honest parameter accounting.**

The phonon collective channel (combining mode enhancement, coherent phases, and resonator Q-factor) brings BPR within plausible experimental reach.

---

## 0 . Purpose
A laser‑focused codebase that **reproduces every numbered equation** in the mathematical spine documents and generates the falsifiable Casimir‑deviation curve (Eq 7). Now includes advanced Clifford algebra formulation with multivector fields and spinor consciousness modules.

*   💡 _meant for peer audit_ — comprehensive mathematical implementation  
*   ⚙️ _minimal deps_ — `FEniCS`, `SymPy`, `NumPy`, `Matplotlib`; all scientific computing essentials
*   🔍 _one‑command tests_ — `pytest -q`
*   🐳 _universal compatibility_ — Docker support for any platform

---

## 1 . Quick start

### Option A: Full Installation (Recommended)
```bash
# clone
$ git clone https://github.com/jackalkahwati/BPR-Math-Spine.git && cd BPR-Math-Spine

# create env with FEniCS (see FEniCS installation notes below)
$ mamba env create -f environment.yml && conda activate bpr

# run demo script (generates Fig 1 + Eq 7 data)
$ python scripts/run_casimir_demo.py

# open interactive notebook
$ jupyter lab notebooks/01_boundary_laplacian.ipynb

# run unit tests (<3 s)
$ pytest -q
```

---

## Wolfram Language port (for Wolfram Research review)

Wolfram Research asked for a runnable Wolfram Language presentation. A minimal, runnable WL scaffold lives in `wolfram/`:

- `wolfram/BPR.wl`: WL package with portable, reviewable APIs mirroring the Python “math spine”
- `wolfram/run_bpr_demo.wls`: CLI demo that writes eigenvalue + Casimir sweep CSV artifacts
- `wolfram/tests/BPRTests.wlt`: WL tests (run via `TestReport`)
- `wolfram/tests/run_equation_smoke.wls`: license-safe equation checks (prints PASS/FAIL)

Run the WL demo (from repo root):

```bash
wolframscript -file wolfram/run_bpr_demo.wls --output data/
```

### WL verification status

- **Fully validated (6/6 passing, license-safe runner)**:

```bash
wolframscript -script wolfram/tests/run_equation_smoke.wls
```

  - Eq (6a): boundary phase equation (spectral residual)
  - Eq (3)/(6b): metric perturbation (structural symmetry)
  - Eq (4): information action (numeric, nonpositive for \(\xi>0\))
  - Eq (5): consciousness coupling (numeric, nonnegative)
  - Eq (7a): Casimir with \(\lambda_{\text{theory}}=\kappa\ell_P^2\) (λ→0 recovery)
  - Eq (7b): Casimir with phenomenological \(\lambda_{\text{eff}}\) (bound consistency at reference point)

- **Still outstanding (deeper constraint)**:
  - A full WL symbolic proof of \( \nabla_\mu T^{\mu\nu}=0 \) / Checkpoint 2 is not yet included in the WL test suite.

### Option B: Quick Start without FEniCS
```bash
# For systems where FEniCS installation is problematic
$ mamba env create -f environment-minimal.yml && conda activate bpr-minimal

# Most functionality works, some tests skipped
$ python scripts/run_casimir_demo.py --quick
$ pytest -q  # Some tests will be skipped
```

### Option C: Docker (Universal Compatibility)
```bash
# Configure for your architecture (Apple Silicon/Intel)
$ ./scripts/configure_docker_platform.sh

# Build and run with Docker Compose (recommended)
$ docker-compose up -d
$ open http://localhost:8888  # Jupyter Lab interface
# Token: bpr-token-2025

# Or build manually
$ docker build -t bpr-math-spine .
$ docker run -it --rm -p 8888:8888 bpr-math-spine

# Run specific scripts in Docker
$ docker-compose --profile benchmark up  # Thin-shell benchmark
$ docker-compose --profile testing up    # Full test suite
```

---

## 🔧 FEniCS Installation Guide

FEniCS is required for full functionality (boundary mesh generation and PDE solving). Choose the method that works best for your system:

### Method 1: Conda/Mamba (Recommended)
```bash
# Standard FEniCS (works on most Linux systems)
$ mamba env create -f environment.yml

# If that fails, try FEniCSX (newer, may work better on macOS)
$ mamba env create -f environment-fenicsx.yml

# Minimal install (no FEniCS, reduced functionality)
$ mamba env create -f environment-minimal.yml
```

### Method 2: Platform-Specific Instructions

**Linux (Ubuntu/Debian):**
```bash
$ sudo apt-get update
$ sudo apt-get install fenics
$ pip install fenics mshr
```

**macOS:**
```bash
# Option 1: Homebrew
$ brew install fenics

# Option 2: MacPorts
$ sudo port install fenics +python310

# Option 3: Use Docker (see below)
```

**Windows:**
```bash
# Use WSL2 with Ubuntu, then follow Linux instructions
$ wsl --install
# or use Docker (see below)
```

### Method 3: Docker (Universal - Recommended for FEniCS)
```bash
# Auto-configure for your system (Apple Silicon/Intel)
$ ./scripts/configure_docker_platform.sh

# Build and run BPR with full FEniCS support
$ docker-compose up -d
$ open http://localhost:8888

# Verify FEniCS installation in container
$ docker-compose exec bpr-math-spine conda run -n bpr python -c "import dolfin; print('✅ FEniCS ready!')"

# Alternative: Use official FEniCS image
$ docker pull dolfinx/dolfinx
$ docker run -ti -v $(pwd):/home/fenics/shared dolfinx/dolfinx
$ cd /home/fenics/shared && python scripts/run_casimir_demo.py
```

### Method 4: Alternative Without FEniCS
If FEniCS installation fails, you can still use BPR-Math-Spine with reduced functionality:

```bash
$ pip install numpy scipy matplotlib sympy pandas jupyter pytest
$ python scripts/run_casimir_demo.py --no-fenics
```

**Note:** Without FEniCS, some tests will be skipped and mesh-based calculations will use fallback methods.

### Troubleshooting FEniCS

**Common Issues:**
- **Import Error**: Try `conda install fenics -c conda-forge`
- **macOS Build Issues**: Use `environment-fenicsx.yml` with FEniCSX
- **Version Conflicts**: Create fresh environment with `conda create -n bpr-test python=3.10`
- **Permission Issues**: Use `pip install --user` or virtual environments

**Verify Installation:**
```python
# Test FEniCS installation
python -c "import fenics; print('✅ FEniCS installed successfully')"
python -c "import mshr; print('✅ mshr installed successfully')"
```

**If pytest segfaults (macOS + system Python):** NumPy's `_mac_os_check` can crash on some macOS/system-Python combinations. Use the project environment: `conda activate bpr` or `mamba env create -f environment.yml`, then `pytest -q`. `conftest.py` sets BLAS threading env vars to reduce this risk.

---

## 2 . Directory tree
```
BPR-math-spine/
├── bpr/                    # minimal Python package
│   ├── __init__.py
│   ├── geometry.py         # triangulate Σ, FEniCS helpers
│   ├── boundary_field.py   # solves Eq (6a)
│   ├── metric.py           # implements Eq (3) & (6b)
│   ├── casimir.py          # computes Eq (7) prediction
│   │
│   │   ── Ten Adjacent Theories (Feb 2026) ──
│   ├── memory.py           # I   Boundary Memory Dynamics
│   ├── impedance.py        # II  Vacuum Impedance Mismatch
│   ├── decoherence.py      # III Boundary-Induced Decoherence
│   ├── phase_transitions.py# IV  Universal Phase Transition Taxonomy
│   ├── neutrino.py         # V   Boundary-Mediated Neutrino Dynamics
│   ├── info_geometry.py    # VI  Substrate Information Geometry
│   ├── gravitational_waves.py # VII Gravitational Wave Phenomenology
│   ├── complexity.py       # VIII Substrate Complexity
│   ├── bioelectric.py      # IX  Bioelectric Substrate Coupling
│   ├── collective.py       # X   Resonant Collective Dynamics
│   ├── cosmology.py       # XI   Cosmology & Early Universe
│   ├── qcd_flavor.py      # XII  QCD & Flavor Physics
│   ├── emergent_spacetime.py # XIII Emergent Spacetime
│   ├── topological_matter.py # XIV Topological Condensed Matter
│   ├── clifford_bpr.py    # XV   Clifford Algebra Embedding
│   ├── quantum_foundations.py # XVI Quantum Foundations
│   ├── gauge_unification.py # XVII Gauge Unification & Hierarchy
│   ├── charged_leptons.py  # XVIII Charged Lepton Masses
│   ├── nuclear_physics.py  # XIX  Nuclear Physics
│   ├── quantum_gravity_pheno.py # XX Quantum Gravity Pheno
│   ├── quantum_chemistry.py # XXI Quantum Chemistry
│   └── first_principles.py # ★  (J,p,N) → all 21 theories, zero free params
│
├── notebooks/
│   ├── 01_boundary_laplacian.ipynb   # reproduces Fig A1
│   ├── 02_metric_perturbation.ipynb  # reproduces Δg plots
│   └── 03_casimir_prediction.ipynb   # reproduces falsifier curve
├── scripts/
│   ├── run_casimir_demo.py           # CLI wrapper around casimir.py
│   └── generate_predictions.py       # ★  produce all 205 predictions as CSV
├── data/
│   └── predictions.csv               # ★  generated predictions table
├── tests/
│   ├── test_boundary.py
│   ├── test_metric.py
│   ├── test_casimir.py
│   ├── test_adjacent_theories.py     # 56 tests for Theories I–X
│   ├── test_inter_theory.py          # ★  14 inter-theory integration tests
│   ├── test_lyapunov.py              # 23 Lyapunov stability tests
│   └── test_fenics_integration.py    # ★  FEniCS tests (auto-skip w/o FEniCS)
├── doc/
│   ├── BPR_Complete_Framework.md     # 80-page unified document
│   ├── CONSISTENCY_AUDIT.md          # 58 internal math consistency tests
│   ├── BENCHMARK_SCORECARD.md        # 51 predictions vs PDG/Planck/CODATA
│   ├── EXPERIMENTAL_ROADMAP.md       # 10 falsification tests
│   ├── FOR_REVIEWERS.md              # one-page reviewer summary
│   ├── LIMITATIONS_AND_FALSIFICATION.md
│   ├── conjectures/                  # open conjectures
│   ├── derivations/                  # 41 DERIVED, 5 FRAMEWORK
│   ├── experiments/                  # evidence pipeline + paper audits
│   └── README_equations.md           # implementation reference guide
├── environment.yml                   # conda spec (<120 MB)
├── environment-minimal.yml           # without FEniCS
├── environment-fenicsx.yml           # with FEniCSX (newer)
├── LICENSE                           # MIT
└── README.md                         # (this file)
```

---

## 3 . Key files / functions

### Core BPR equations
| File | Purpose | Main public API |
|------|---------|-----------------|
| `geometry.py` | Build a triangulated sphere/cylinder boundary | `make_boundary(mesh_size)` |
| `boundary_field.py` | Solve $\kappa\nabla^2_Σ\varphi=f$ via FEniCS | `solve_phase(mesh, source)` |
| `metric.py` | Compute $\Delta g_{\mu\nu}$ from Eq (3) | `metric_perturbation(phi, λ)` |
| `casimir.py` | Integrate stress tensor, output force curve | `casimir_force(R, params)` |

### Ten Adjacent Theories (Feb 2026)
| # | Module | Theory | Key API |
|---|--------|--------|---------|
| I | `memory.py` | Boundary Memory Dynamics | `MemoryKernel`, `BoundaryMemoryField` |
| II | `impedance.py` | Vacuum Impedance Mismatch | `TopologicalImpedance`, `DarkMatterProfile`, `MONDInterpolation` |
| III | `decoherence.py` | Boundary-Induced Decoherence | `DecoherenceRate`, `PointerBasis`, `critical_winding` |
| IV | `phase_transitions.py` | Phase Transition Taxonomy | `TransitionClass`, `SubstrateCriticalExponents`, `kibble_zurek_defect_density` |
| V | `neutrino.py` | Neutrino Dynamics | `NeutrinoMassSpectrum`, `PMNSMatrix`, `neutrino_nature` |
| VI | `info_geometry.py` | Substrate Information Geometry | `FisherMetric`, `TopologicalCramerRao`, `thermodynamic_length` |
| VII | `gravitational_waves.py` | GW Phenomenology | `GWPropagation`, `GWQuadrupole`, `gw_memory_displacement` |
| VIII | `complexity.py` | Substrate Complexity | `TopologicalParallelism`, `TopologicalComplexityBound` |
| IX | `bioelectric.py` | Bioelectric Coupling | `MorphogeneticField`, `CellularWinding`, `AgingModel` |
| X | `collective.py` | Resonant Collective Dynamics | `KuramotoFlocking`, `MarketImpedance`, `TippingPoint` |

---

## 4 . Reproducing Eq (7) (Casimir deviation)
Run:
```python
from bpr.casimir import sweep_radius
sweep_radius(r_min=0.2e-6, r_max=5e-6, n=40, out='data/casimir.csv')
```
This writes a CSV with columns
`R [m]`, `F_Casimir [N]`, `ΔF_BPR [N]`, ready for plotting.

---

## 5 . Math checkpoints
*   **Check 1** — Laplacian eigenvalues on $S^2$ converge to $l(l+1)$ within 0.1 % for $l\le10$.
*   **Check 2** — Energy–momentum conservation: \(\nabla^μ T^{\varphi}_{μν}=0\) to solver tolerance 1e‑8.
*   **Check 3** — Recovered force law tends to standard Casimir for $\alpha\to0$.
All are executed in `pytest`.

---

## 6 . Environment Options

| File | Purpose | Use Case |
|------|---------|----------|
| `environment.yml` | Full installation with FEniCS | Complete functionality (recommended) |
| `environment-minimal.yml` | Basic dependencies only | Quick start, FEniCS issues |
| `environment-fenicsx.yml` | Modern FEniCSX version | macOS, newer systems |

**Choose based on your system:**
- **Linux**: Use `environment.yml`
- **macOS**: Try `environment-fenicsx.yml` first
- **Windows**: Use WSL2 + `environment.yml` or Docker
- **Any issues**: Fall back to `environment-minimal.yml`

---

## 7 . Contributing
Pull requests must:
1. Add or update a test.
2. Maintain code quality and mathematical rigor.
3. Pass `pre‑commit` (black, isort, flake8).

---

## 8 . Implementation Status

### ✅ **Completed Features**

- [x] **Core Mathematical Framework** — All 7 numbered equations implemented  
- [x] **Equation (2)**: Boundary Laplacian solver (`boundary_field.py`)  
- [x] **Equation (3)**: Metric-boundary coupling (`metric.py`)
- [x] **Equation (4)**: Information integration with IIT (`information.py`)
- [x] **Equation (5)**: Six-factor consciousness coupling (`information.py`)
- [x] **Equation (6a/6b)**: Field equations with conservation verification
- [x] **Equation (7)**: Casimir prediction with δ = 1.37 ± 0.05 (`casimir.py`)
- [x] **E₈ embedding**: Complete group theory implementation (`notebooks/04_e8_index.ipynb`)
- [x] **Thin-shell analytics**: SymPy benchmark (`scripts/thin_shell_benchmark.py`)
- [x] **FEniCS integration**: Multiple installation paths + fallback modes
- [x] **Docker support**: Universal compatibility (`Dockerfile`, `docker-compose.yml`)
- [x] **Mathematical checkpoints**: All 3 verification tests implemented
- [x] **Jupyter notebooks**: 4 complete interactive demonstrations  
- [x] **Unit testing**: Comprehensive test suite with `pytest`
- [x] **Experimental validation**: Bayesian analysis of 5 experimental results
- [x] **Advanced formulation**: Clifford algebra embedding with spinor consciousness modules (`bpr/clifford_bpr.py`)

### ✅ **Ten Adjacent Theories** (Feb 2026 — 10 modules, 56 new tests)

- [x] **Boundary Memory Dynamics** — memory kernel M(t,t'), non-Markovian correlations, consciousness temporal integration
- [x] **Vacuum Impedance Mismatch** — dark matter as high-winding solitons, dark energy from phase frustration, MOND acceleration scale, flat rotation curves
- [x] **Boundary-Induced Decoherence** — rates from impedance mismatch (Γ ∝ ΔZ²), pointer basis selection, quantum–classical boundary W_crit, decoherence-free subspaces
- [x] **Universal Phase Transition Taxonomy** — Classes A–D mapping all known transitions, substrate critical exponents (ν, β, γ), Kibble–Zurek defect formation
- [x] **Boundary-Mediated Neutrino Dynamics** — normal hierarchy, Σm_i ≈ 0.06 eV, PMNS matrix from boundary overlaps, Majorana/Dirac from p mod 4, sterile neutrinos
- [x] **Substrate Information Geometry** — Fisher metric on boundary configurations, topological Cramér–Rao bound (Var ∝ 1/|W|²), thermodynamic length, K_r as parallel transport
- [x] **Gravitational Wave Phenomenology** — v_GW = c from substrate isotropy, quadrupole formula from boundary dynamics, GW memory via Boundary Memory Dynamics kernel
- [x] **Substrate Complexity** — P/NP/BQP as substrate properties, N_parallel = p^W, topological complexity bound (physical P ≠ NP argument), adiabatic gap
- [x] **Bioelectric Substrate Coupling** — morphogenetic fields φ_morph, cellular winding W_cell (cancer = aberrant W), aging as coherence decay τ_coh(age) = τ₀ e^{-age/τ_aging}
- [x] **Resonant Collective Dynamics** — Kuramoto flocking, market impedance matching (crash = resonance), social tipping points (f_c ~ 1/⟨k⟩), cooperation from winding alignment

### ✅ **First-Principles Pipeline** (v0.3.0 → v0.6.0)

- [x] **Coupling derivation** — `bpr.first_principles.SubstrateDerivedTheories` wires `(J, p, N)` → `boundary_energy.py` → all 21 theories. `p` fixed by α; `N` cancels in all dimensionless predictions (computational grid); `J` is the one free energy anchor.
- [x] **Inter-theory integration tests** — `tests/test_inter_theory.py`: 14 tests chaining Boundary Memory Dynamics↔III, I↔VII, II↔V, III↔IV, VI↔VIII, IV↔X, IX↔(I,III), VIII↔I
- [x] **Predictions generator** — `scripts/generate_predictions.py` produces **205** falsifiable predictions as CSV
- [x] **Lyapunov bug fix** — numpy broadcasting bug in regression; all 23 Lyapunov tests now pass
- [x] **FEniCS CI path** — `tests/test_fenics_integration.py` auto-skipped locally, runs in Docker

### ✅ **Extended Theories** (v0.5.0)

- [x] **BPR Cosmology & Early Universe** — inflation (n_s ≈ 0.968, r ≈ 0.003), baryogenesis, CMB anomalies
- [x] **QCD & Flavor Physics** — 6 quark masses, CKM matrix, strong CP (θ = 0), confinement
- [x] **Emergent Spacetime & Holography** — 3+1 dimensions, holographic entropy, Bekenstein bound, ER=EPR
- [x] **Topological Condensed Matter** — QHE, fractional QHE, topological insulators, anyons, Majorana modes
- [x] **Clifford Algebra Embedding** — multivector fields in Cl(3,0), spinor modules, Cliffordon spectrum
- [x] **Quantum Foundations** — Born rule (1 − 1/p accuracy), arrow of time, Bell bound → Tsirelson

### ✅ **Frontier Theories** (v0.6.0)

- [x] **Gauge Unification & Hierarchy** — GUT scale = M_Pl/p^{1/4}, coupling running, hierarchy = √(pN), proton decay
- [x] **Charged Lepton Masses** — e/μ/τ masses from cohomology norms, Koide formula = 2/3 from S², lepton universality
- [x] **Nuclear Physics from Boundary Shell** — magic numbers (2,8,20,28,50,82,126) from winding shells, binding energies, neutron star M_max
- [x] **Quantum Gravity Phenomenology** — modified dispersion (ξ₁=0, ξ₂=1/p), GUP (β=1/p), Lorentz invariance to exp(-p^{1/3})
- [x] **Quantum Chemistry & Periodic Table** — noble gas Z from shell filling, chemical bonds, electronegativity, chirality, periodic table

### 🚀 **Ready for Use**

The BPR-Math-Spine framework is **feature-complete** and ready for:
* **Peer review** — All mathematics transparent and auditable
* **Experimental validation** — 205 falsifiable predictions generated, end-to-end from substrate  
* **Research extension** — Modular architecture for new physics
* **Publication** — Complete mathematical spine for papers
* **Docker CI** — `docker-compose run --rm --profile testing bpr-test` runs all tests with FEniCS

### 📊 **Project Statistics**
* **~84,000** total lines of Python
* **~60,000** core mathematical LoC (`bpr/`)
* **~14,000** test LoC (`tests/`, 32 test modules)
* **248** E₈ generators implemented
* **7/7** BPR equations complete
* **21 theories** — 10 adjacent + 6 extended + 5 frontier (Gauge, Leptons, Nuclear, QG, Chemistry)
* **205** falsifiable predictions (from 3 substrate numbers: J, p, N)
* **1,294** tests passing (+ ~21 FEniCS tests that auto-skip without FEniCS)
* **58** mathematical consistency tests (no experiments needed)
* **29** benchmark regression tests (predictions vs experiment)
* **51** predictions benchmarked (41 DERIVED, 5 FRAMEWORK; 50 PASS, 1 CLOSE)
* **3/3** mathematical checkpoints verified

---

© 2026 StarDrive Research Group — released under MIT license.