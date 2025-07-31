# BPR‑Math‑Spine : Minimal, Reproducible Maths for Boundary Phase Resonance

> **Status :** public draft | license MIT | contact <jack@thestardrive.com>
>
> One‑pager PDF → [`doc/BPR_one_pager.pdf`](doc/BPR_one_pager.pdf)

---

## 0 . Purpose
A laser‑focused codebase that **reproduces every numbered equation** in the one‑page synopsis and generates the falsifiable Casimir‑deviation curve (Eq 7).  Nothing else.

*   💡 _meant for peer audit_ — less than ~500 LoC
*   ⚙️ _minimal deps_ — `FEniCS`, `SymPy`, `NumPy`, `Matplotlib`; Wolfram notebooks optional
*   🔍 _one‑command tests_ — `pytest -q`

---

## 1 . Quick start

### Option A: Full Installation (Recommended)
```bash
# clone
$ git clone https://github.com/StarDrive/BPR-math-spine.git && cd BPR-math-spine

# create env with FEniCS (see FEniCS installation notes below)
$ mamba env create -f environment.yml && conda activate bpr

# run demo script (generates Fig 1 + Eq 7 data)
$ python scripts/run_casimir_demo.py

# open interactive notebook
$ jupyter lab notebooks/01_boundary_laplacian.ipynb

# run unit tests (<3 s)
$ pytest -q
```

### Option B: Quick Start without FEniCS
```bash
# For systems where FEniCS installation is problematic
$ mamba env create -f environment-minimal.yml && conda activate bpr-minimal

# Most functionality works, some tests skipped
$ python scripts/run_casimir_demo.py --quick
$ pytest -q  # Some tests will be skipped
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

### Method 3: Docker (Universal)
```bash
# Pull FEniCS Docker image
$ docker pull dolfinx/dolfinx

# Run BPR-Math-Spine in container
$ docker run -ti -v $(pwd):/home/fenics/shared dolfinx/dolfinx
$ cd /home/fenics/shared
$ python scripts/run_casimir_demo.py
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
│   ├── BPR_one_pager.pdf             # typeset one‑pager
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
2. Keep core package size <500 LoC.
3. Pass `pre‑commit` (black, isort, flake8).

---

## 8 . Roadmap
- [ ]  Thin‑shell analytic benchmark (symbolic SymPy)
- [ ]  E$_8$ index‑theorem stub notebook (`notebooks/04_e8_index.ipynb`)
- [ ]  Information integration implementation (Equation 4)
- [ ]  Consciousness coupling implementation (Equation 5)
- [ ]  GitHub Actions CI (Ubuntu + Mac)
- [x]  FEniCS installation documentation and alternatives
- [x]  Docker support for universal compatibility
- [x]  Fractal scaling δ = 1.37 ± 0.05 (Equation 7)

---

© 2025 StarDrive Research Group — released under MIT license.