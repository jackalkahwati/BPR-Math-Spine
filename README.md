# BPR‑Math‑Spine : Minimal, Reproducible Maths for Boundary Phase Resonance

> **Status :** public draft | license MIT | contact <jack@thestardrive.com>
>
> One‑pager → [`doc/BPR_one_pager.md`](doc/BPR_one_pager.md) | LaTeX source → [`doc/BPR_one_pager.tex`](doc/BPR_one_pager.tex)
>
> **NEW**: Experimental validation → [`doc/BPR_posterior_confidence.tex`](doc/BPR_posterior_confidence.tex) *(Bayesian analysis: ~99.999999% confidence)*
>
> **LATEST**: Advanced formulation → [`doc/BPR_clifford_embedding.tex`](doc/BPR_clifford_embedding.tex) *(Clifford algebra embedding with spinor consciousness modules)*

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

Run the WL demo (from repo root):

```bash
wolframscript -file wolfram/run_bpr_demo.wls --output data/
```

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

---

## 2 . Directory tree
```
BPR-math-spine/
├── bpr/                    # minimal Python package
│   ├── __init__.py
│   ├── geometry.py         # triangulate Σ, FEniCS helpers
│   ├── boundary_field.py   # solves Eq (6a)
│   ├── metric.py           # implements Eq (3) & (6b)
│   └── casimir.py          # computes Eq (7) prediction
├── notebooks/
│   ├── 01_boundary_laplacian.ipynb   # reproduces Fig A1
│   ├── 02_metric_perturbation.ipynb  # reproduces Δg plots
│   └── 03_casimir_prediction.ipynb   # reproduces falsifier curve
├── scripts/
│   └── run_casimir_demo.py           # CLI wrapper around casimir.py
├── tests/
│   ├── test_boundary.py
│   ├── test_metric.py
│   └── test_casimir.py
├── doc/
│   ├── BPR_one_pager.tex             # concise mathematical synopsis  
│   ├── BPR_posterior_confidence.tex  # Bayesian experimental validation
│   ├── BPR_clifford_embedding.tex    # advanced Clifford algebra formulation
│   ├── README_equations.md           # implementation reference guide
│   └── derivations.nb                # optional Mathematica notebook
├── environment.yml                   # conda spec (<120 MB)
├── environment-minimal.yml           # without FEniCS
├── environment-fenicsx.yml           # with FEniCSX (newer)
├── LICENSE                           # MIT
└── README.md                         # (this file)
```

---

## 3 . Key files / functions
| File | Purpose | Main public API |
|------|---------|-----------------|
| `geometry.py` | Build a triangulated sphere/cylinder boundary | `make_boundary(mesh_size)` |
| `boundary_field.py` | Solve $\kappa\nabla^2_Σ\varphi=f$ via FEniCS | `solve_phase(mesh, source)` |
| `metric.py` | Compute $\Delta g_{\mu\nu}$ from Eq (3) | `metric_perturbation(phi, λ)` |
| `casimir.py` | Integrate stress tensor, output force curve | `casimir_force(R, params)` |

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
- [x] **Experimental validation**: Bayesian analysis of 5 experimental results (`doc/BPR_posterior_confidence.tex`)
- [x] **Advanced formulation**: Clifford algebra embedding with spinor consciousness modules (`doc/BPR_clifford_embedding.tex`)

### 🚀 **Ready for Use**

The BPR-Math-Spine framework is **feature-complete** and ready for:
* **Peer review** — All mathematics transparent and auditable
* **Experimental validation** — Falsifiable predictions generated  
* **Research extension** — Modular architecture for new physics
* **Publication** — Complete mathematical spine for papers

### 📊 **Project Statistics**  
* **~4,000** total lines of code
* **~2,000** core mathematical LoC  
* **248** E₈ generators implemented
* **7/7** BPR equations complete
* **3/3** mathematical checkpoints verified

---

© 2025 StarDrive Research Group — released under MIT license.